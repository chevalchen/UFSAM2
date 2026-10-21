"""EXP-004: deterministic support context and a bounded trainable residual.

This module is opt-in. It does not alter the accepted SANSA forward method.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def local_centers(tokens, k=4, iterations=10):
    """Tokens must be row-major [N,C]. argmax/argmin ties choose first index."""
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError('Empty support foreground or invalid token shape')
    if not torch.isfinite(tokens).all():
        raise ValueError('Non-finite foreground features')
    x = F.normalize(tokens, dim=-1)
    indices = [0]
    distance = 1 - x @ x[0]
    for _ in range(1, min(k, len(x))):
        # Do not select an already selected token when all distances tie at zero.
        distance[indices] = -float('inf')
        idx = int(distance.argmax())
        indices.append(idx)
        distance = torch.minimum(distance, 1 - x @ x[idx])
    centers = x[[indices[i % len(indices)] for i in range(k)]].clone()
    for _ in range(iterations):
        labels = (x @ centers.T).argmax(dim=1)
        centers = torch.stack([F.normalize(x[labels == j].mean(0), dim=0)
                               if (labels == j).any() else centers[j] for j in range(k)])
    return centers


@torch.no_grad()
def support_prototypes(features, masks, *, global_mean=False):
    """[S,C,H,W] raw final encoder maps; masks are [S,H0,W0]."""
    if features.ndim != 4 or masks.ndim != 3 or len(features) != len(masks):
        raise ValueError('Invalid support feature/mask shapes')
    foreground = F.interpolate(masks[:, None].float(), features.shape[-2:], mode='nearest')[:, 0] > 0.5
    centers = []
    for feat, mask in zip(features, foreground):
        tokens = feat.permute(1, 2, 0)[mask]
        if len(tokens) == 0:
            raise ValueError('Empty support foreground at encoder resolution')
        if global_mean:
            center = F.normalize(tokens.mean(0), dim=0)
            centers.append(center[None].repeat(4, 1))
        else:
            centers.append(local_centers(tokens))
    return torch.cat(centers)


def prototype_context(query, prototypes):
    """Cosine affinity at temperature 1; reconstruct unit prototype vectors."""
    b, c, h, w = query.shape
    q = query.flatten(2).transpose(1, 2)
    affinity = F.normalize(q, dim=-1) @ prototypes.T
    return (affinity.softmax(-1) @ prototypes).transpose(1, 2).reshape(b, c, h, w)


def row_context(query, prototypes, row):
    if row not in ('C1', 'C2', 'C3', 'C4'):
        raise ValueError(f'Unsupported residual row: {row}')
    context = prototype_context(query, prototypes)  # Also executed for C2.
    if row == 'C2':
        return query
    if row == 'C4':
        return torch.roll(context, (query.shape[-2] // 2, query.shape[-1] // 2), (-2, -1))
    return context


class LocalPrototypeResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = [nn.Conv2d(channels * 3, 32, 1), nn.GroupNorm(8, 32), nn.GELU()]
        for _ in range(2):
            layers.extend([nn.Conv2d(32, 32, 3, padding=1, groups=32),
                           nn.Conv2d(32, 32, 1), nn.GroupNorm(8, 32), nn.GELU()])
        self.body = nn.Sequential(*layers)
        self.output = nn.Conv2d(32, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, memory, query, context, *, enabled=True):
        if not enabled:
            return memory
        return memory + 0.1 * torch.tanh(self.output(self.body(torch.cat([memory, query, context], 1))))


@dataclass
class EpisodeFeatures:
    memory: torch.Tensor
    query: torch.Tensor
    support: torch.Tensor
    support_masks: torch.Tensor
    high_resolution: list
    original_size: tuple


class FrozenSANSARepair(nn.Module):
    """Freeze all SANSA weights; preserve decoder input gradients to the residual."""
    def __init__(self, sansa):
        super().__init__()
        self.sansa = sansa.requires_grad_(False).eval()
        self.residual = LocalPrototypeResidual(sansa.sam.hidden_dim).to(sansa.device)

    def train(self, mode=True):
        super().train(mode)
        self.sansa.eval()
        return self

    @torch.no_grad()
    def prepare(self, samples, support_masks):
        from util.promptable_utils import rescale_prompt
        if samples.ndim != 5 or samples.shape[:2] != (1, 6) or len(support_masks) != 5:
            raise ValueError('EXP-004 requires batch=1, exactly 5 support + 1 query')
        model = self.sansa
        flat, _, _, sizes = model._preprocess_visual_features(samples, model.sam.image_size)
        bb = model._forward_backbone(flat, sizes)
        bank = {}
        for i in range(5):
            prompt = rescale_prompt(support_masks[i:i+1, None], 'mask', sizes[i], model.sam.image_size)
            decoded = model.sam._use_mask_as_output(bb, prompt, i)
            bank[i] = model._compute_memory_bank_dict(decoded, bb, i)
        feats = bb.get_current_feats(5)
        memory = model.sam._prepare_memory_conditioned_features(
            frame_idx=5, current_vision_feats=feats[-1:],
            current_vision_pos_embeds=bb.get_current_pos_embeds(5)[-1:],
            feat_sizes=bb.feat_sizes[-1:], num_frames=6, memory_bank=bank)
        return EpisodeFeatures(memory, bb.get_current_feats_x16(5),
                               torch.cat([bb.get_current_feats_x16(i) for i in range(5)]),
                               support_masks, bb.get_high_res_features(feats), sizes[5])

    def context(self, cache, row):
        prototypes = support_prototypes(cache.support, cache.support_masks, global_mean=row == 'C1')
        return row_context(cache.query, prototypes, row)

    def decode(self, cache, row='C0', *, context=None):
        if row == 'C0':
            feature = cache.memory
        else:
            if context is None:
                context = self.context(cache, row)
            feature = self.residual(cache.memory, cache.query, context)
        decoded = self.sansa.sam._forward_sam_heads(
            backbone_features=feature, high_res_features=cache.high_resolution, multimask_output=True)
        # Same interpolation as accepted SANSA: selected low-resolution logits.
        return F.interpolate(decoded.masks, cache.original_size, mode='bilinear', align_corners=False)[0, 0]
