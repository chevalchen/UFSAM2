"""EXP-006 candidate generation. Query ground truth is deliberately absent here."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class QueryCache:
    feature: torch.Tensor
    high_resolution: list
    query: torch.Tensor
    support_prototype: torch.Tensor
    size: tuple


class GradientPrompting:
    def __init__(self, sansa, config):
        self.sansa = sansa.requires_grad_(False).eval()
        self.config = config
        self.calls = 0

    @torch.no_grad()
    def prepare(self, images, support_masks):
        from util.promptable_utils import rescale_prompt
        n = len(support_masks)
        assert images.shape[:2] == (1, n + 1)
        s = self.sansa
        flat, _, _, sizes = s._preprocess_visual_features(images, s.sam.image_size)
        bb = s._forward_backbone(flat, sizes)
        bank = {}  # Local state: no query prediction is written into support memory.
        for i in range(n):
            prompt = rescale_prompt(support_masks[i:i+1, None], 'mask', sizes[i], s.sam.image_size)
            decoded = s.sam._use_mask_as_output(bb, prompt, i)
            bank[i] = s._compute_memory_bank_dict(decoded, bb, i)
        feats = bb.get_current_feats(n)
        feature = s.sam._prepare_memory_conditioned_features(
            frame_idx=n, current_vision_feats=feats[-1:],
            current_vision_pos_embeds=bb.get_current_pos_embeds(n)[-1:],
            feat_sizes=bb.feat_sizes[-1:], num_frames=n+1, memory_bank=bank)
        support = torch.cat([bb.get_current_feats_x16(i) for i in range(n)])
        masks = F.interpolate(support_masks[:, None], support.shape[-2:], mode='area')
        # Each support contributes equally; small supports do not vanish by nearest resize.
        proto = ((support * masks).sum((2, 3)) / masks.sum((2, 3)).clamp_min(1e-6)).mean(0)
        return QueryCache(feature.detach(), bb.get_high_res_features(feats),
                          bb.get_current_feats_x16(n), proto, sizes[n])

    def decode(self, cache, points=None, feature=None):
        # Capture auxiliary masks without editing the official SAM2/SANSA implementation.
        captured = []
        handle = self.sansa.sam.sam_mask_decoder.register_forward_hook(
            lambda module, inputs, output: captured.append(output[0]))
        try:
            out = self.sansa.sam._forward_sam_heads(
                backbone_features=cache.feature if feature is None else feature,
                high_res_features=cache.high_resolution, point_inputs=points,
                multimask_output=True)
            out.low_res_multimasks = captured[0]
            self.calls += 1
            return out
        finally:
            handle.remove()

    @staticmethod
    def detach(out):
        from models.sansa.model_utils import DecoderOutput
        result = DecoderOutput(low_res_masks=out.low_res_masks.detach())
        result.low_res_multimasks = out.low_res_multimasks.detach()
        return result

    @staticmethod
    def quality(out):
        z = out.low_res_masks
        area_u = (z > -.05).float().sum()
        stability = torch.where(area_u > 0, (z > .05).float().sum() / area_u.clamp_min(1), z.new_tensor(1))
        disagreement = (out.low_res_multimasks.sigmoid() - z.sigmoid()).abs().mean()
        return float((stability - .25 * disagreement).clamp(0, 1))

    def domains(self, out):
        p = out.low_res_masks.detach().sigmoid()
        fg = (p > self.config['positive_threshold']).float()
        k = self.config['erode_kernel']
        interior = 1 - F.max_pool2d(1-fg, k, stride=1, padding=k//2)
        if interior.any():
            fg = interior
        positive = p * fg
        outside = p < self.config['negative_threshold']
        negative = out.low_res_multimasks.detach().sigmoid().amax(1, keepdim=True) * outside
        return positive[0, 0], negative[0, 0]

    @staticmethod
    def cells(points, shape, image_size=1024):
        if points is None:
            return []
        h, w = shape
        xy = points['point_coords'][0].detach().cpu().tolist()
        return [(min(h-1, int(y*h/image_size)), min(w-1, int(x*w/image_size))) for x, y in xy]

    def pack(self, cells, labels, like):
        if not cells or 1 not in labels:
            return None
        h, w = like.shape[-2:]
        size = self.sansa.sam.image_size
        return dict(point_coords=like.new_tensor([[(x+.5)*size/w, (y+.5)*size/h] for y, x in cells])[None],
                    point_labels=torch.tensor([labels], dtype=torch.int32, device=like.device))

    def points(self, out, previous=None, response=None, shuffle_seed=None):
        """Replace both points; the same candidate domains and exclusion radius serve all rows."""
        scores = self.domains(out)
        domains = (scores[0] >= self.config['positive_threshold'], scores[1] >= 1e-6)
        old = self.cells(previous, scores[0].shape, self.sansa.sam.image_size)
        yy, xx = torch.meshgrid(torch.arange(scores[0].shape[0], device=scores[0].device),
                                torch.arange(scores[0].shape[1], device=scores[0].device), indexing='ij')
        allowed = torch.ones_like(domains[0])
        radius = self.config['dedup_radius_lowres']
        for y, x in old:
            allowed &= (yy-y)**2 + (xx-x)**2 > radius**2
        chosen, labels = [], []
        for i, (score, domain) in enumerate(zip(scores, domains)):
            domain = domain & allowed
            if chosen:
                y, x = chosen[0]
                domain &= (yy-y)**2 + (xx-x)**2 > radius**2
            if not domain.any():
                if i == 0:
                    return None
                continue
            rank = score if response is None else response
            rank = rank.clone()
            if shuffle_seed is not None:
                indices = domain.flatten().nonzero().flatten()
                generator = torch.Generator(device=rank.device).manual_seed(shuffle_seed+i)
                rank.flatten()[indices] = rank.flatten()[indices[torch.randperm(len(indices), generator=generator, device=rank.device)]]
            index = int(rank.masked_fill(~domain, -torch.inf).flatten().argmax())
            chosen.append(divmod(index, score.shape[1])); labels.append(1-i)
        return self.pack(chosen, labels, scores[0])

    def anchors(self, logits):
        """One confident extremum per coarse cell, then at most 32 cells per sign."""
        z = logits.detach()[0, 0]
        g = self.config['anchor_grid']
        h, w = z.shape
        assert h % g == 0 and w % g == 0
        # reshape makes spatial strata, independent of the output area of each sign.
        blocks = z.reshape(g, h//g, g, w//g).permute(0, 2, 1, 3).reshape(g*g, -1)
        result = []
        for sign in (1, -1):
            values, offset = (sign*blocks).max(1)
            threshold = torch.log(z.new_tensor(.9/.1))
            selected = (values > threshold).nonzero().flatten()
            k = self.config['anchors_per_domain']
            # Stable, spatially spread subsampling rather than only largest object cells.
            if len(selected) > k:
                selected = selected[torch.linspace(0, len(selected)-1, k, device=z.device).long()]
            oy, ox = offset[selected] // (w//g), offset[selected] % (w//g)
            y = selected // g * (h//g) + oy
            x = selected % g * (w//g) + ox
            result.append((y, x))
        return result

    def gradient(self, temporary, out):
        anchors = self.anchors(out.low_res_masks)
        counts = [len(y) for y, x in anchors]
        if min(counts) == 0:
            return None, dict(anchor_counts=counts, gradient_valid=False)
        a, b = anchors
        z = out.low_res_masks[0, 0]
        objective = z[a].mean() - z[b].mean()
        grad, = torch.autograd.grad(objective, temporary)
        valid = bool(torch.isfinite(grad).all() and torch.count_nonzero(grad))
        if not valid:
            raise RuntimeError('Nonfinite or zero decoder feature gradient')
        response = F.interpolate(grad.detach().norm(dim=1, keepdim=True), z.shape,
                                 mode='bilinear', align_corners=False)[0, 0]
        return response, dict(anchor_counts=counts, gradient_valid=True, objective=float(objective.detach()),
                              gradient_l2=float(grad.norm()), gradient_abs_max=float(grad.abs().max()))

    @torch.no_grad()
    def semantic_points(self, cache, out, previous):
        """Forward-only support semantic search: deliberately allows positives beyond M1."""
        sim = F.cosine_similarity(cache.query, cache.support_prototype[None, :, None, None], dim=1)[None]
        sim = F.interpolate(sim, out.low_res_masks.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        p = out.low_res_masks[0, 0].sigmoid()
        # Positive maximum across full image; negative low-similarity location within prediction.
        yy, xx = torch.meshgrid(torch.arange(p.shape[0], device=p.device), torch.arange(p.shape[1], device=p.device), indexing='ij')
        allowed = torch.ones_like(p, dtype=torch.bool)
        r = self.config['dedup_radius_lowres']
        for y, x in self.cells(previous, p.shape, self.sansa.sam.image_size):
            allowed &= (yy-y)**2+(xx-x)**2 > r*r
        idx = int(sim.masked_fill(~allowed, -torch.inf).flatten().argmax())
        pos = divmod(idx, p.shape[1]); y, x = pos
        domain = allowed & (p > .5) & ((yy-y)**2+(xx-x)**2 > r*r)
        cells, labels = [pos], [1]
        if domain.any():
            idx = int((-sim).masked_fill(~domain, -torch.inf).flatten().argmax())
            cells.append(divmod(idx, p.shape[1])); labels.append(0)
        return self.pack(cells, labels, p)
