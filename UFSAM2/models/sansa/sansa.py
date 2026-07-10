import os
from typing import Any, Dict, List, Tuple

import py3_wget
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
import torch.nn.functional as F

from models.sam2.modeling.sam2_utils import preprocess
from models.sam2.modeling.sam2_base import SAM2Base 
from models.sansa.model_utils import BackboneOutput, DecoderOutput
from models.sansa.boundary_refine import BoundaryRefinementModule
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt


class SANSA(nn.Module):
    def __init__(
        self,
        sam: SAM2Base,
        device: torch.device,
        hflip_tta: bool = False,
        boundary_refine: bool = False,
        memory_to_point_prompt: bool = False,
        mtp_trigger_threshold: float = 0.92,
        mtp_accept_margin: float = 0.0,
        mtp_num_positive_points: int = 1,
        mtp_num_negative_points: int = 1,
        mtp_pos_threshold: float = 0.65,
        mtp_neg_threshold: float = 0.35,
    ):
        super().__init__()
        self.sam = sam
        self.device = device
        self.hflip_tta = hflip_tta
        self.brm = BoundaryRefinementModule() if boundary_refine else None
        self.memory_to_point_prompt = memory_to_point_prompt
        self.mtp_trigger_threshold = mtp_trigger_threshold
        self.mtp_accept_margin = mtp_accept_margin
        self.mtp_num_positive_points = mtp_num_positive_points
        self.mtp_num_negative_points = mtp_num_negative_points
        self.mtp_pos_threshold = mtp_pos_threshold
        self.mtp_neg_threshold = mtp_neg_threshold
        self.reset_memory_to_point_stats()

    def reset_memory_to_point_stats(self) -> None:
        self.memory_to_point_stats = {"triggered": 0, "accepted": 0}

    def forward(
        self,
        samples: torch.Tensor,
        prompt_dict: List[Dict[str, Any]],
        return_traces: bool = False,
    ) -> Dict[str, Any]:
        """
        Run SANSA.
        Args:
            samples: [B, T, C, H, W].
            targets: list (len B) of dicts with:
                - 'is_support': list[bool] of len T
                - 'masks': Tensor [T, H, W]

        Returns:
            {"pred_masks": Tensor [B*T, H', W']}
        """

        samples, B, T, orig_size = self._preprocess_visual_features(samples, self.sam.image_size)
        backbone_output: BackboneOutput = self._forward_backbone(samples, orig_size)
        flipped_backbone_output = None
        if self.hflip_tta:
            flipped_backbone_output = self._forward_backbone(torch.flip(samples, dims=[-1]), orig_size)
        outputs = {"masks": []}
        traces = []
        support_traces = []

        n_shots = prompt_dict['shots']
        for b in range(B):
            self.memory_bank = {}
            for idx in range(T):
                absolute_idx = b * T + idx

                if idx < n_shots:
                    frame_prompt = prompt_dict[b][idx]['prompt']
                    prompt_type = prompt_dict[b][idx]['prompt_type']
                    frame_prompt = rescale_prompt(frame_prompt, prompt_type, orig_size[b], self.sam.image_size)
                    if prompt_type == 'mask':
                        decoder_out: DecoderOutput = self.sam._use_mask_as_output(backbone_output, frame_prompt, absolute_idx)
                    else:
                        decoder_out: DecoderOutput = self._compute_decoder_out_no_mem(backbone_output, absolute_idx, prompt_input=frame_prompt)
                    if return_traces:
                        support_traces.append(self._build_support_trace(decoder_out, b, idx, absolute_idx))
                        
                else:
                    decoder_out: DecoderOutput = self._compute_decoder_out_w_mem(backbone_output, absolute_idx, idx, self.memory_bank)
                    if self.hflip_tta:
                        assert flipped_backbone_output is not None
                        decoder_out = self._apply_hflip_tta(decoder_out, flipped_backbone_output, absolute_idx, idx, self.memory_bank)
                    decoder_out = self._apply_boundary_refine(decoder_out)
                    if return_traces:
                        traces.append(self._build_uncertainty_trace(decoder_out, b, idx, absolute_idx))

                # update memory bank
                mem_entry = self._compute_memory_bank_dict(decoder_out, backbone_output, absolute_idx)
                self.memory_bank[idx] = mem_entry
                outputs["masks"].append(decoder_out.masks[0])

        masks = torch.cat(outputs["masks"])
        masks = F.interpolate(masks[None], size=orig_size[0], mode='bilinear', align_corners=False)[0]
        result = {"pred_masks": masks}
        if return_traces:
            result["traces"] = traces
            result["support_traces"] = support_traces
        return result

    def _preprocess_visual_features(
        self, samples: torch.Tensor, image_size: int
    ) -> Tuple[torch.Tensor, int, int, List[Tuple[int, int]]]:
        """
        Flatten [B,T,C,H,W] -> [B*T,C,H,W], store original sizes, and apply SAM2 preprocess.

        Args:
            samples:   Tensor [B, T, C, H, W].
            image_size: target side for SAM2 preprocessing.

        Returns:
            (samples_bt, B, T, orig_sizes)
        """

        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        return samples, B, T, orig_size

    def _compute_decoder_out_no_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        prompt_input: Dict[str, torch.Tensor] | None,
    ) -> DecoderOutput:
        """
        Decode a frame without memory: used for reference frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            prompt:       "mask" | "point" | "scribble" | "box".
            prompt_input: inputs for point/scribble/box (ignored for "mask").

        Returns:
            DecoderOutput.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)

        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(1, 256, 64, 64)
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            point_inputs=prompt_input,
            high_res_features=high_res_features,
        )
        return decoder_out

    def _apply_hflip_tta(
        self,
        decoder_out: DecoderOutput,
        flipped_backbone_out: BackboneOutput,
        idx: int,
        memory_idx: int,
        memory_bank: Dict[int, Dict[str, torch.Tensor]],
    ) -> DecoderOutput:
        """
        Re-decode a horizontally flipped query frame with the same support memory,
        flip logits back, and average in logit space.
        """
        flipped_out = self._compute_decoder_out_w_mem(flipped_backbone_out, idx, memory_idx, memory_bank)
        decoder_out.low_res_masks = 0.5 * (
            decoder_out.low_res_masks + torch.flip(flipped_out.low_res_masks, dims=[-1])
        )
        decoder_out.masks = decoder_out.low_res_masks
        if decoder_out.high_res_masks is not None and flipped_out.high_res_masks is not None:
            decoder_out.high_res_masks = 0.5 * (
                decoder_out.high_res_masks + torch.flip(flipped_out.high_res_masks, dims=[-1])
            )
        return decoder_out

    def _apply_boundary_refine(self, decoder_out: DecoderOutput) -> DecoderOutput:
        if self.brm is None:
            return decoder_out
        brm_feat = getattr(decoder_out, "_brm_feat", None)
        if brm_feat is None:
            raise RuntimeError("BRM is enabled, but decoder output has no high-resolution BRM feature.")
        corrected = self.brm(decoder_out.low_res_masks, brm_feat)
        decoder_out.low_res_masks = corrected
        decoder_out.high_res_masks = F.interpolate(
            corrected.detach(),
            size=(self.sam.image_size, self.sam.image_size),
            mode="bilinear",
            align_corners=False,
        )
        decoder_out.masks = decoder_out.low_res_masks
        return decoder_out

    def _compute_decoder_out_w_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        memory_idx: int,
        memory_bank: Dict[int, Dict[str, torch.Tensor]],
    ) -> DecoderOutput:
        """
        Decode a frame with memory: used for target frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            memory_idx:   temporal index t (0-based).
            memory_bank:  dict of memory entries from previous frames.

        Returns:
            DecoderOutput
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)

        # take only the highest res feature map
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)
        
        pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx+1,
            memory_bank=memory_bank
        )

        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            high_res_features=high_res_features,
            multimask_output=True if memory_idx > 0 else False
        )
        decoder_out.memory_summary = pix_feat_with_mem.mean(dim=(-2, -1))
        decoder_out._brm_feat = high_res_features[0]
        if self.memory_to_point_prompt:
            decoder_out = self._apply_memory_to_point_prompt(
                decoder_out,
                pix_feat_with_mem,
                high_res_features,
            )
        return decoder_out

    def _apply_memory_to_point_prompt(
        self,
        decoder_out: DecoderOutput,
        pix_feat_with_mem: torch.Tensor,
        high_res_features: List[torch.Tensor],
    ) -> DecoderOutput:
        """
        Convert an uncertain memory-only query mask into sparse point prompts and
        run one extra SAM head pass. This is intentionally conservative: it accepts
        the second pass only when the unsupervised quality score does not regress.
        """
        quality_before = self._memory_to_point_quality(decoder_out)
        decoder_out.memory_to_point_quality_before = quality_before.detach()
        decoder_out.memory_to_point_quality_after = quality_before.detach()
        decoder_out.memory_to_point_triggered = False
        decoder_out.memory_to_point_accepted = False
        decoder_out.memory_to_point_coords = None
        decoder_out.memory_to_point_labels = None

        if quality_before.item() >= self.mtp_trigger_threshold:
            return decoder_out

        self.memory_to_point_stats["triggered"] += 1
        decoder_out.memory_to_point_triggered = True
        point_inputs = self._build_memory_to_point_inputs(decoder_out)
        if point_inputs is None:
            return decoder_out

        refined_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            point_inputs=point_inputs,
            high_res_features=high_res_features,
            multimask_output=True,
        )
        refined_out.memory_summary = decoder_out.memory_summary
        refined_out._brm_feat = getattr(decoder_out, "_brm_feat", None)

        quality_after = self._memory_to_point_quality(refined_out)
        accept = quality_after.item() >= quality_before.item() + self.mtp_accept_margin
        target_out = refined_out if accept else decoder_out
        if accept:
            self.memory_to_point_stats["accepted"] += 1

        target_out.memory_to_point_quality_before = quality_before.detach()
        target_out.memory_to_point_quality_after = quality_after.detach()
        target_out.memory_to_point_triggered = True
        target_out.memory_to_point_accepted = accept
        target_out.memory_to_point_coords = point_inputs["point_coords"].detach()
        target_out.memory_to_point_labels = point_inputs["point_labels"].detach()
        return target_out

    def _memory_to_point_quality(self, decoder_out: DecoderOutput) -> torch.Tensor:
        stability = self._logit_stability_score(decoder_out.low_res_masks).mean()
        disagreement = self._multimask_disagreement(decoder_out)
        return (stability - 0.25 * disagreement).clamp(0.0, 1.0)

    @staticmethod
    def _logit_stability_score(mask_logits: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
        flat_logits = mask_logits.flatten(-2)
        area_i = torch.sum(flat_logits > delta, dim=-1).float()
        area_u = torch.sum(flat_logits > -delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u.clamp_min(1.0), torch.ones_like(area_i))

    @staticmethod
    def _multimask_disagreement(decoder_out: DecoderOutput) -> torch.Tensor:
        if decoder_out.low_res_multimasks is None or decoder_out.low_res_multimasks.size(1) <= 1:
            return decoder_out.low_res_masks.new_tensor(0.0)
        selected = decoder_out.low_res_masks.sigmoid()
        candidates = decoder_out.low_res_multimasks.sigmoid()
        return torch.mean(torch.abs(candidates - selected))

    def _build_memory_to_point_inputs(self, decoder_out: DecoderOutput) -> Dict[str, torch.Tensor] | None:
        if decoder_out.low_res_masks is None:
            return None

        prob = decoder_out.low_res_masks.detach().sigmoid()
        _, _, mask_h, mask_w = prob.shape
        points: list[tuple[int, int]] = []
        labels: list[int] = []

        positive_score = self._positive_point_score(prob)
        for y, x in self._topk_spatial_points(
            positive_score,
            self.mtp_num_positive_points,
            min_score=self.mtp_pos_threshold,
        ):
            points.append((y, x))
            labels.append(1)

        if not points:
            return None

        negative_score = self._negative_point_score(decoder_out, prob)
        for y, x in self._topk_spatial_points(
            negative_score,
            self.mtp_num_negative_points,
            min_score=1e-6,
        ):
            points.append((y, x))
            labels.append(0)

        coords = prob.new_tensor(
            [
                [
                    (x + 0.5) * self.sam.image_size / mask_w,
                    (y + 0.5) * self.sam.image_size / mask_h,
                ]
                for y, x in points
            ]
        )
        point_labels = torch.tensor(labels, device=prob.device, dtype=torch.int32)
        return {
            "point_coords": coords.unsqueeze(0),
            "point_labels": point_labels.unsqueeze(0),
        }

    def _positive_point_score(self, prob: torch.Tensor) -> torch.Tensor:
        foreground = (prob > self.mtp_pos_threshold).float()
        erode_kernel = 9
        if min(prob.shape[-2:]) >= erode_kernel:
            inverse = 1.0 - foreground
            interior = 1.0 - F.max_pool2d(
                inverse,
                kernel_size=erode_kernel,
                stride=1,
                padding=erode_kernel // 2,
            )
            if interior.sum() > 0:
                foreground = interior
        return prob * foreground

    def _negative_point_score(self, decoder_out: DecoderOutput, selected_prob: torch.Tensor) -> torch.Tensor:
        outside_selected = (selected_prob < self.mtp_neg_threshold).float()
        if decoder_out.low_res_multimasks is not None and decoder_out.low_res_multimasks.size(1) > 1:
            candidate_prob = decoder_out.low_res_multimasks.detach().sigmoid().max(dim=1, keepdim=True).values
            return candidate_prob * outside_selected
        return (1.0 - selected_prob) * outside_selected

    @staticmethod
    def _topk_spatial_points(
        score: torch.Tensor,
        num_points: int,
        min_score: float,
    ) -> list[tuple[int, int]]:
        if num_points <= 0:
            return []
        score_2d = score[0, 0]
        flat_score = score_2d.flatten()
        k = min(num_points, flat_score.numel())
        values, indices = torch.topk(flat_score, k=k)
        points = []
        width = score_2d.shape[-1]
        for value, index in zip(values, indices):
            if value.item() < min_score:
                continue
            y = int(index.item() // width)
            x = int(index.item() % width)
            points.append((y, x))
        return points

    def _build_uncertainty_trace(
        self, decoder_out: DecoderOutput, batch_idx: int, frame_idx: int, absolute_idx: int
    ) -> Dict[str, Any]:
        return {
            "batch_idx": batch_idx,
            "frame_idx": frame_idx,
            "absolute_idx": absolute_idx,
            "sam_score": self._detach_cpu(decoder_out.ious),
            "query_iou_token": self._detach_cpu(decoder_out.iou_token),
            "query_mask_tokens": self._detach_cpu(decoder_out.mask_tokens),
            "query_mask_token": self._detach_cpu(decoder_out.selected_mask_token),
            "query_obj_ptr": self._detach_cpu(decoder_out.obj_ptr),
            "query_memory_summary": self._detach_cpu(decoder_out.memory_summary),
            "memory_to_point_triggered": getattr(decoder_out, "memory_to_point_triggered", False),
            "memory_to_point_accepted": getattr(decoder_out, "memory_to_point_accepted", False),
            "memory_to_point_quality_before": self._detach_cpu(getattr(decoder_out, "memory_to_point_quality_before", None)),
            "memory_to_point_quality_after": self._detach_cpu(getattr(decoder_out, "memory_to_point_quality_after", None)),
            "memory_to_point_coords": self._detach_cpu(getattr(decoder_out, "memory_to_point_coords", None)),
            "memory_to_point_labels": self._detach_cpu(getattr(decoder_out, "memory_to_point_labels", None)),
        }

    def _build_support_trace(
        self, decoder_out: DecoderOutput, batch_idx: int, frame_idx: int, absolute_idx: int
    ) -> Dict[str, Any]:
        memory_summary = decoder_out.memory_summary
        if memory_summary is None and decoder_out.pix_feat_with_mem is not None:
            memory_summary = decoder_out.pix_feat_with_mem.mean(dim=(-2, -1))
        return {
            "batch_idx": batch_idx,
            "frame_idx": frame_idx,
            "absolute_idx": absolute_idx,
            "support_sam_score": self._detach_cpu(decoder_out.ious),
            "support_iou_token": self._detach_cpu(decoder_out.iou_token),
            "support_mask_tokens": self._detach_cpu(decoder_out.mask_tokens),
            "support_mask_token": self._detach_cpu(decoder_out.selected_mask_token),
            "support_obj_ptr": self._detach_cpu(decoder_out.obj_ptr),
            "support_memory_summary": self._detach_cpu(memory_summary),
        }

    @staticmethod
    def _detach_cpu(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return x.detach().cpu()

    def _compute_memory_bank_dict(
        self, decoder_out: DecoderOutput, backbone_out: BackboneOutput, idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Encode current prediction into memory for later frames.

        Args:
            decoder_out: decoder output with high_res/low_res masks.
            backbone_out:  backbone features.
            idx: absolute idx.

        Returns:
            Memory entry dict.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        feat_sizes = backbone_out.feat_sizes

        mem_feats, mem_pos = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        return {
            "maskmem_features": mem_feats,
            "maskmem_pos_enc": mem_pos,
            "pred_masks": decoder_out.low_res_masks,
            "obj_ptr": decoder_out.obj_ptr,
        }

    def _forward_backbone(
        self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]
    ) -> BackboneOutput:
        """
        Run SAM2 image encoder and prepare backbone features for decoding.

        Args:
            samples:  Tensor [B*T, C, H, W] after preprocessing.
            orig_size:   list of original frame sizes.

        Returns:
            BackboneOutput.
        """
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)

        # discard lowest resolution
        feats, pos = feats[:-1], pos[:-1]

        feats[0] = self.sam.sam_mask_decoder.conv_s0(feats[0])
        feats[1] = self.sam.sam_mask_decoder.conv_s1(feats[1])

        bb = {
            "vision_features": feats[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": feats,
        }
        vision_feats, vision_pos, sizes = self.sam._prepare_backbone_features(bb)
        return BackboneOutput(orig_size, vision_feats, vision_pos, sizes)


def build_sansa(
    sam2_version: str = 'large',
    adaptformer_stages: List[int] = [2, 3],
    channel_factor: float = 0.3,
    device: str = 'cuda',
    hflip_tta: bool = False,
    boundary_refine: bool = False,
    memory_to_point_prompt: bool = False,
    mtp_trigger_threshold: float = 0.92,
    mtp_accept_margin: float = 0.0,
    mtp_num_positive_points: int = 1,
    mtp_num_negative_points: int = 1,
    mtp_pos_threshold: float = 0.65,
    mtp_neg_threshold: float = 0.35,
) -> SANSA:
    assert sam2_version in SAM2_PATHS_CONFIG.keys(), f'wrong argument sam2_version: {sam2_version}'
    
    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        print(f"Downloading SAM2-{sam2_version}")
        py3_wget.download_file(SAM2_WEIGHTS_URL[sam2_version], sam2_weights)

    with initialize(version_base=None, config_path=".", job_name="test_app"):
        cfg = compose(config_name=sam2_config, overrides=[
            f"++model.image_encoder.trunk.adaptformer_stages={adaptformer_stages}",
            f"++model.image_encoder.trunk.adapt_dim={channel_factor}",
        ])

        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = False
        cfg.model.pred_obj_scores_mlp = False
        cfg.model.fixed_no_obj_ptr = False
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)["model"]
    sam.load_state_dict(state_dict, strict=False)
    model = SANSA(
        sam=sam,
        device=torch.device(device),
        hflip_tta=hflip_tta,
        boundary_refine=boundary_refine,
        memory_to_point_prompt=memory_to_point_prompt,
        mtp_trigger_threshold=mtp_trigger_threshold,
        mtp_accept_margin=mtp_accept_margin,
        mtp_num_positive_points=mtp_num_positive_points,
        mtp_num_negative_points=mtp_num_negative_points,
        mtp_pos_threshold=mtp_pos_threshold,
        mtp_neg_threshold=mtp_neg_threshold,
    )

    # freeze everything except adapters and optional BRM
    for name, p in model.named_parameters():
        p.requires_grad = ("adapter" in name or "brm" in name)

    return model
