"""Frozen EXP-002 mask post-correction operator definition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MaskPostCorrectionOutput:
    """Outputs needed for training and matched operator-headroom evaluation."""

    logits: Tensor
    support_mask: Tensor
    residual_offset: Tensor


class BoundaryAwareResidualLogitRefiner(nn.Module):
    """Three-layer residual refiner frozen by the EXP-002 Stage A contract."""

    def __init__(
        self,
        feature_channels: int = 32,
        hidden_channels: int = 16,
        probability_low: float = 0.4,
        probability_high: float = 0.6,
    ) -> None:
        super().__init__()
        if feature_channels != 32:
            raise ValueError("EXP-002 v1 requires feature_channels=32.")
        if hidden_channels != 16:
            raise ValueError("EXP-002 v1 requires hidden_channels=16.")
        if (probability_low, probability_high) != (0.4, 0.6):
            raise ValueError("EXP-002 v1 requires the frozen probability interval (0.4, 0.6).")

        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.probability_low = probability_low
        self.probability_high = probability_high
        self.refine = nn.Sequential(
            nn.Conv2d(feature_channels + 1, hidden_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, logits: Tensor, decoder_feature: Tensor) -> MaskPostCorrectionOutput:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                "Mask logits must have shape [B,1,H,W], "
                f"received {tuple(logits.shape)}."
            )
        if decoder_feature.ndim != 4 or decoder_feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"Decoder feature must have shape [B,{self.feature_channels},H,W], "
                f"received {tuple(decoder_feature.shape)}."
            )
        if logits.shape[0] != decoder_feature.shape[0] or logits.shape[-2:] != decoder_feature.shape[-2:]:
            raise ValueError(
                "Mask logits and decoder feature must share batch and spatial dimensions: "
                f"{tuple(logits.shape)} vs {tuple(decoder_feature.shape)}."
            )

        with torch.no_grad():
            probability = torch.sigmoid(logits)
            support_mask = (
                (probability > self.probability_low)
                & (probability < self.probability_high)
            ).to(logits.dtype)

        residual_offset = self.refine(torch.cat((logits, decoder_feature), dim=1))
        refined_logits = logits + support_mask * residual_offset
        return MaskPostCorrectionOutput(
            logits=refined_logits,
            support_mask=support_mask,
            residual_offset=residual_offset,
        )


def mask_post_refiner_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Return only the EXP-002 refiner tensors, detached on CPU."""
    state = {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
        if key.startswith("mask_post_refiner.")
    }
    if not state:
        raise RuntimeError("Model contains no mask_post_refiner tensors.")
    return state


def load_mask_post_refiner_state(
    module: nn.Module,
    checkpoint_state: dict[str, Tensor],
) -> list[str]:
    """Strictly validate and load a refiner-only checkpoint state."""
    expected = sorted(
        key for key in module.state_dict() if key.startswith("mask_post_refiner.")
    )
    provided = sorted(checkpoint_state)
    if provided != expected:
        missing = sorted(set(expected) - set(provided))
        unexpected = sorted(set(provided) - set(expected))
        raise RuntimeError(
            "Mask-refiner checkpoint key mismatch. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )
    shape_mismatches = [
        (key, tuple(checkpoint_state[key].shape), tuple(module.state_dict()[key].shape))
        for key in expected
        if tuple(checkpoint_state[key].shape) != tuple(module.state_dict()[key].shape)
    ]
    if shape_mismatches:
        raise RuntimeError(f"Mask-refiner checkpoint shape mismatch: {shape_mismatches}.")
    missing_keys, unexpected_keys = module.load_state_dict(checkpoint_state, strict=False)
    loaded = set(expected)
    unexpected_loaded = sorted(key for key in unexpected_keys if key in loaded)
    missing_loaded = sorted(key for key in missing_keys if key in loaded)
    if unexpected_loaded or missing_loaded:
        raise RuntimeError(
            "Mask-refiner runtime load mismatch. "
            f"Missing: {missing_loaded}; unexpected: {unexpected_loaded}."
        )
    return expected
