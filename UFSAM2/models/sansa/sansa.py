from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
import torch.nn.functional as F

from models.sam2.modeling.sam2_utils import preprocess
from models.sam2.modeling.sam2_base import SAM2Base 
from models.sansa.mask_post_correction import BoundaryAwareResidualLogitRefiner
from models.sansa.model_utils import BackboneOutput, DecoderOutput
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt


class SANSA(nn.Module):
    def __init__(
        self,
        sam: SAM2Base,
        device: torch.device,
        mask_post_correction: bool = False,
    ):
        super().__init__()
        self.sam = sam
        self.device = device
        self.mask_post_refiner = (
            BoundaryAwareResidualLogitRefiner()
            if mask_post_correction
            else None
        )

    def forward(
        self,
        samples: torch.Tensor,
        prompt_dict: List[Dict[str, Any]],
        return_mask_post_correction_data: bool = False,
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
        if return_mask_post_correction_data and self.mask_post_refiner is None:
            raise RuntimeError(
                "Mask post-correction data requires mask_post_correction=True."
            )
        outputs = {
            "masks": [],
            "baseline_masks": [],
            "support_fractions": [],
        }

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
                    baseline_mask = decoder_out.masks[0]
                    decoder_out, support_fraction = self._apply_mask_post_correction(
                        decoder_out
                    )
                    outputs["support_fractions"].append(support_fraction)

                if idx < n_shots:
                    baseline_mask = decoder_out.masks[0]

                # update memory bank
                mem_entry = self._compute_memory_bank_dict(decoder_out, backbone_output, absolute_idx)
                self.memory_bank[idx] = mem_entry
                outputs["masks"].append(decoder_out.masks[0])
                outputs["baseline_masks"].append(baseline_mask)

        masks = torch.cat(outputs["masks"])
        masks = F.interpolate(masks[None], size=orig_size[0], mode='bilinear', align_corners=False)[0]
        result = {"pred_masks": masks}
        if return_mask_post_correction_data:
            baseline_masks = torch.cat(outputs["baseline_masks"])
            result["mask_post_baseline_pred_masks"] = F.interpolate(
                baseline_masks[None],
                size=orig_size[0],
                mode="bilinear",
                align_corners=False,
            )[0]
            result["mask_post_support_fraction"] = torch.stack(
                outputs["support_fractions"]
            )
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
        decoder_out._mask_post_feature = high_res_features[0]
        return decoder_out

    def _apply_mask_post_correction(
        self,
        decoder_out: DecoderOutput,
    ) -> Tuple[DecoderOutput, torch.Tensor]:
        if self.mask_post_refiner is None:
            return decoder_out, decoder_out.low_res_masks.new_zeros(())
        decoder_feature = getattr(decoder_out, "_mask_post_feature", None)
        if decoder_feature is None:
            raise RuntimeError(
                "Mask post-correction is enabled, but the decoder output has "
                "no highest-resolution feature."
            )
        correction = self.mask_post_refiner(
            decoder_out.low_res_masks,
            decoder_feature,
        )
        decoder_out.low_res_masks = correction.logits
        decoder_out.high_res_masks = F.interpolate(
            correction.logits.detach(),
            size=(self.sam.image_size, self.sam.image_size),
            mode="bilinear",
            align_corners=False,
        )
        decoder_out.masks = decoder_out.low_res_masks
        return decoder_out, correction.support_mask.float().mean()

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
    mask_post_correction: bool = False,
) -> SANSA:
    assert sam2_version in SAM2_PATHS_CONFIG.keys(), f'wrong argument sam2_version: {sam2_version}'
    
    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        import py3_wget

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
        mask_post_correction=mask_post_correction,
    )

    # Preserve SANSA's adapter-training default while exposing the optional
    # candidate module. EXP-002's dedicated trainer freezes the adapters.
    for name, p in model.named_parameters():
        p.requires_grad = (
            "adapter" in name
            or "mask_post_refiner" in name
        )

    return model
