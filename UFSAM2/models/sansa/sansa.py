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
from models.sansa.post_memory_calibration import PostMemoryFeatureCalibrator
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt


class SANSA(nn.Module):
    def __init__(
        self,
        sam: SAM2Base,
        device: torch.device,
        post_memory_calibration: bool = False,
        pmc_mode: str = "gated",
        pmc_projection_dim: int = 64,
        pmc_hidden_dim: int = 128,
        pmc_residual_scale: float = 0.1,
        pmc_spatial_threshold: float = 0.0,
        pmc_episode_threshold: float = 0.0,
        pmc_gate_temperature: float = 0.25,
    ):
        super().__init__()
        self.sam = sam
        self.device = device
        self.post_memory_calibrator = (
            PostMemoryFeatureCalibrator(
                feature_dim=getattr(sam, "hidden_dim", 256),
                projection_dim=pmc_projection_dim,
                hidden_dim=pmc_hidden_dim,
                residual_scale=pmc_residual_scale,
                mode=pmc_mode,
                spatial_threshold=pmc_spatial_threshold,
                episode_threshold=pmc_episode_threshold,
                gate_temperature=pmc_gate_temperature,
            )
            if post_memory_calibration
            else None
        )
        self.reset_post_memory_calibration_stats()

    def reset_post_memory_calibration_stats(self) -> None:
        self.post_memory_calibration_stats = {
            "eligible": 0,
            "triggered": 0,
            "spatial_area_sum": 0.0,
            "predicted_gain_sum": 0.0,
        }

    def set_post_memory_calibration_train_stage(self, stage: str | None) -> None:
        if self.post_memory_calibrator is None:
            raise RuntimeError("Post-memory calibration is not enabled.")
        self.post_memory_calibrator.set_train_stage(stage)

    def forward(
        self,
        samples: torch.Tensor,
        prompt_dict: List[Dict[str, Any]],
        return_calibration_data: bool = False,
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
        outputs = {"masks": []}
        calibration_records = []

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
                        
                else:
                    decoder_out: DecoderOutput = self._compute_decoder_out_w_mem(backbone_output, absolute_idx, idx, self.memory_bank)
                    if return_calibration_data and self.post_memory_calibrator is not None:
                        calibration_records.append(
                            self._build_post_memory_calibration_record(
                                decoder_out,
                                b,
                                idx,
                                absolute_idx,
                            )
                        )

                # update memory bank
                mem_entry = self._compute_memory_bank_dict(decoder_out, backbone_output, absolute_idx)
                self.memory_bank[idx] = mem_entry
                outputs["masks"].append(decoder_out.masks[0])

        masks = torch.cat(outputs["masks"])
        masks = F.interpolate(masks[None], size=orig_size[0], mode='bilinear', align_corners=False)[0]
        result = {"pred_masks": masks}
        if return_calibration_data:
            result["post_memory_calibration"] = calibration_records
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
        if self.post_memory_calibrator is not None:
            decoder_out = self._apply_post_memory_calibration(
                decoder_out,
                backbone_out.get_current_feats_x16(idx),
                pix_feat_with_mem,
                high_res_features,
                multimask_output=memory_idx > 0,
            )
        return decoder_out

    def _apply_post_memory_calibration(
        self,
        baseline_out: DecoderOutput,
        query_feature: torch.Tensor,
        pix_feat_with_mem: torch.Tensor,
        high_res_features: List[torch.Tensor],
        multimask_output: bool,
    ) -> DecoderOutput:
        calibrator = self.post_memory_calibrator
        if calibrator is None:
            return baseline_out

        calibration = calibrator(query_feature, pix_feat_with_mem, baseline_out)
        target_out = baseline_out
        if calibration.applied:
            target_out = self.sam._forward_sam_heads(
                backbone_features=calibration.calibrated_feature,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        target_out.memory_summary = calibration.calibrated_feature.mean(dim=(-2, -1))
        target_out.baseline_low_res_masks = baseline_out.low_res_masks
        target_out.baseline_high_res_masks = baseline_out.high_res_masks
        target_out.baseline_ious = baseline_out.ious
        target_out.predicted_delta_iou = calibration.predicted_delta_iou
        target_out.calibration_applied = calibration.episode_gate
        target_out.spatial_benefit = calibration.spatial_benefit
        target_out.spatial_gate = calibration.spatial_gate
        target_out.calibration_residual = calibration.residual

        stats = self.post_memory_calibration_stats
        stats["eligible"] += int(calibration.episode_gate.numel())
        stats["triggered"] += int(calibration.episode_gate.detach().sum().item())
        stats["spatial_area_sum"] += float(calibration.spatial_gate.detach().mean().item())
        stats["predicted_gain_sum"] += float(
            calibration.predicted_delta_iou.detach().mean().item()
        )
        return target_out

    @staticmethod
    def _build_post_memory_calibration_record(
        decoder_out: DecoderOutput,
        batch_idx: int,
        frame_idx: int,
        absolute_idx: int,
    ) -> Dict[str, Any]:
        residual = decoder_out.calibration_residual
        residual_norm = None
        if residual is not None:
            residual_norm = residual.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return {
            "batch_idx": batch_idx,
            "frame_idx": frame_idx,
            "absolute_idx": absolute_idx,
            "baseline_low_res_masks": decoder_out.baseline_low_res_masks,
            "calibrated_low_res_masks": decoder_out.low_res_masks,
            "baseline_ious": decoder_out.baseline_ious,
            "calibrated_ious": decoder_out.ious,
            "predicted_delta_iou": decoder_out.predicted_delta_iou,
            "episode_gate": decoder_out.calibration_applied,
            "spatial_benefit": decoder_out.spatial_benefit,
            "spatial_gate": decoder_out.spatial_gate,
            "residual_norm": residual_norm,
            "applied": bool(
                decoder_out.calibration_applied.detach().bool().any().item()
            ),
        }

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
    post_memory_calibration: bool = False,
    pmc_mode: str = "gated",
    pmc_projection_dim: int = 64,
    pmc_hidden_dim: int = 128,
    pmc_residual_scale: float = 0.1,
    pmc_spatial_threshold: float = 0.0,
    pmc_episode_threshold: float = 0.0,
    pmc_gate_temperature: float = 0.25,
    pmc_train_stage: str | None = None,
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
        post_memory_calibration=post_memory_calibration,
        pmc_mode=pmc_mode,
        pmc_projection_dim=pmc_projection_dim,
        pmc_hidden_dim=pmc_hidden_dim,
        pmc_residual_scale=pmc_residual_scale,
        pmc_spatial_threshold=pmc_spatial_threshold,
        pmc_episode_threshold=pmc_episode_threshold,
        pmc_gate_temperature=pmc_gate_temperature,
    )

    # freeze everything except adapters
    for name, p in model.named_parameters():
        p.requires_grad = ("adapter" in name)
    if model.post_memory_calibrator is not None:
        if pmc_train_stage is not None:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.set_post_memory_calibration_train_stage(pmc_train_stage)

    return model
