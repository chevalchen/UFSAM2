from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.sansa.model_utils import DecoderOutput


def load_post_memory_calibrator_checkpoint(
    calibrator: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("post_memory_calibrator", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported AV-PMC checkpoint format: {checkpoint_path}")
    for prefix in ("module.post_memory_calibrator.", "post_memory_calibrator."):
        filtered = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if filtered:
            state = filtered
            break
    calibrator.load_state_dict(state, strict=strict)
    return checkpoint


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DepthwisePointwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


@dataclass
class PostMemoryCalibrationOutput:
    calibrated_feature: Tensor
    predicted_delta_iou: Tensor
    episode_gate: Tensor
    spatial_benefit: Tensor
    spatial_gate: Tensor
    residual: Tensor

    @property
    def applied(self) -> bool:
        return bool(self.episode_gate.detach().bool().any().item())


class PostMemoryFeatureCalibrator(nn.Module):
    """Selective post-memory repair for SANSA query features.

    The module predicts a bounded residual from pre/post-memory disagreement.
    A dense benefit head limits the residual spatially, while a signed global
    action-value head decides whether a second decoder pass is worthwhile.
    """

    VALID_MODES = {"operator", "spatial", "gated"}

    def __init__(
        self,
        feature_dim: int = 256,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        residual_scale: float = 0.1,
        mode: str = "gated",
        spatial_threshold: float = 0.0,
        episode_threshold: float = 0.0,
        gate_temperature: float = 0.25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown post-memory calibration mode: {mode}")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.residual_scale = float(residual_scale)
        self.mode = mode
        self.spatial_threshold = float(spatial_threshold)
        self.episode_threshold = float(episode_threshold)
        self.gate_temperature = float(gate_temperature)

        self.query_projection = nn.Sequential(
            nn.Conv2d(feature_dim, projection_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(projection_dim), projection_dim),
            nn.GELU(),
        )
        self.memory_projection = nn.Sequential(
            nn.Conv2d(feature_dim, projection_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(projection_dim), projection_dim),
            nn.GELU(),
        )

        evidence_dim = 4 * projection_dim + 1
        self.evidence_stem = nn.Sequential(
            nn.Conv2d(evidence_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.evidence_blocks = nn.Sequential(
            DepthwisePointwiseBlock(hidden_dim),
            DepthwisePointwiseBlock(hidden_dim),
        )
        self.residual_head = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)

        self.spatial_head = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim // 2), hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

        global_dim = 2 * hidden_dim + 5 * feature_dim + 8
        self.action_value_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.spatial_head[-1].weight)
        nn.init.zeros_(self.spatial_head[-1].bias)
        nn.init.zeros_(self.action_value_head[-1].weight)
        nn.init.zeros_(self.action_value_head[-1].bias)

    def set_mode(self, mode: str) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown post-memory calibration mode: {mode}")
        self.mode = mode

    def set_train_stage(self, stage: Optional[str]) -> None:
        """Freeze parameters for the operator -> spatial -> gain training funnel."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        if stage is None:
            return
        if stage == "operator":
            modules = (
                self.query_projection,
                self.memory_projection,
                self.evidence_stem,
                self.evidence_blocks,
                self.residual_head,
            )
        elif stage == "spatial":
            modules = (self.spatial_head,)
        elif stage == "gain":
            modules = (self.action_value_head,)
        else:
            raise ValueError(f"Unknown post-memory calibration train stage: {stage}")
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        query_feature: Tensor,
        memory_feature: Tensor,
        baseline_output: DecoderOutput,
    ) -> PostMemoryCalibrationOutput:
        if query_feature.shape != memory_feature.shape:
            raise ValueError(
                "query_feature and memory_feature must have the same shape, got "
                f"{tuple(query_feature.shape)} and {tuple(memory_feature.shape)}"
            )
        if memory_feature.ndim != 4 or memory_feature.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected BCHW features with C={self.feature_dim}, got {tuple(memory_feature.shape)}"
            )

        evidence = self._feature_evidence(query_feature, memory_feature)
        mask_entropy, multimask_disagreement = self._decoder_maps(
            baseline_output,
            evidence.shape[-2:],
        )
        spatial_benefit = self.spatial_head(
            torch.cat((evidence, mask_entropy, multimask_disagreement), dim=1)
        )
        predicted_delta_iou = self._predict_delta_iou(
            evidence,
            baseline_output,
            mask_entropy,
            multimask_disagreement,
        )

        residual = self.residual_scale * torch.tanh(self.residual_head(evidence))
        if self.mode == "operator":
            episode_gate = torch.ones_like(predicted_delta_iou, dtype=torch.bool)
            spatial_gate = torch.ones_like(spatial_benefit)
        else:
            spatial_gate = torch.sigmoid(
                (spatial_benefit - self.spatial_threshold) / self.gate_temperature
            )
            if self.mode == "spatial":
                episode_gate = torch.ones_like(predicted_delta_iou, dtype=torch.bool)
            else:
                episode_gate = predicted_delta_iou > self.episode_threshold

        calibrated_feature = memory_feature + (
            episode_gate.to(memory_feature.dtype).view(-1, 1, 1, 1)
            * spatial_gate
            * residual
        )
        return PostMemoryCalibrationOutput(
            calibrated_feature=calibrated_feature,
            predicted_delta_iou=predicted_delta_iou,
            episode_gate=episode_gate,
            spatial_benefit=spatial_benefit,
            spatial_gate=spatial_gate,
            residual=residual,
        )

    def _feature_evidence(self, query_feature: Tensor, memory_feature: Tensor) -> Tensor:
        query = self.query_projection(query_feature)
        memory = self.memory_projection(memory_feature)
        cosine = F.cosine_similarity(query, memory, dim=1, eps=1e-6).unsqueeze(1)
        evidence = torch.cat(
            (query, memory, torch.abs(memory - query), memory * query, cosine),
            dim=1,
        )
        return self.evidence_blocks(self.evidence_stem(evidence))

    def _decoder_maps(
        self,
        baseline_output: DecoderOutput,
        output_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        logits = baseline_output.low_res_masks
        if logits is None:
            raise ValueError("baseline_output.low_res_masks is required for post-memory calibration")
        probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())

        candidates = baseline_output.low_res_multimasks
        if candidates is None or candidates.size(1) <= 1:
            disagreement = torch.zeros_like(entropy)
        else:
            disagreement = candidates.sigmoid().std(dim=1, keepdim=True, unbiased=False)

        entropy = F.interpolate(entropy, size=output_size, mode="bilinear", align_corners=False)
        disagreement = F.interpolate(disagreement, size=output_size, mode="bilinear", align_corners=False)
        return entropy, disagreement

    def _predict_delta_iou(
        self,
        evidence: Tensor,
        baseline_output: DecoderOutput,
        entropy: Tensor,
        disagreement: Tensor,
    ) -> Tensor:
        batch_size = evidence.size(0)
        device = evidence.device
        dtype = evidence.dtype

        pooled_mean = evidence.mean(dim=(-2, -1))
        pooled_max = evidence.amax(dim=(-2, -1))
        iou_token = self._vector_or_zeros(
            baseline_output.iou_token, batch_size, self.feature_dim, device, dtype
        )

        mask_tokens = baseline_output.mask_tokens
        if mask_tokens is None:
            mask_mean = torch.zeros(batch_size, self.feature_dim, device=device, dtype=dtype)
            mask_std = torch.zeros_like(mask_mean)
        else:
            mask_tokens = mask_tokens.to(device=device, dtype=dtype)
            mask_mean = mask_tokens.mean(dim=1)
            mask_std = mask_tokens.std(dim=1, unbiased=False)

        obj_ptr = self._vector_or_zeros(
            baseline_output.obj_ptr, batch_size, self.feature_dim, device, dtype
        )
        memory_summary = self._vector_or_zeros(
            baseline_output.memory_summary, batch_size, self.feature_dim, device, dtype
        )

        iou_stats = self._iou_statistics(
            baseline_output.ious, batch_size, device, dtype
        )
        scalar_stats = torch.cat(
            (
                iou_stats,
                entropy.mean(dim=(-2, -1)),
                entropy.amax(dim=(-2, -1)),
                disagreement.mean(dim=(-2, -1)),
                disagreement.amax(dim=(-2, -1)),
                baseline_output.low_res_masks.sigmoid().mean(dim=(-2, -1)),
            ),
            dim=1,
        )
        global_feature = torch.cat(
            (
                pooled_mean,
                pooled_max,
                iou_token,
                mask_mean,
                mask_std,
                obj_ptr,
                memory_summary,
                scalar_stats,
            ),
            dim=1,
        )
        return self.action_value_head(global_feature)

    @staticmethod
    def _vector_or_zeros(
        value: Optional[Tensor],
        batch_size: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if value is None:
            return torch.zeros(batch_size, width, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if value.ndim > 2:
            value = value.flatten(1)
        if value.shape != (batch_size, width):
            raise ValueError(f"Expected vector shape {(batch_size, width)}, got {tuple(value.shape)}")
        return value

    @staticmethod
    def _iou_statistics(
        ious: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if ious is None:
            return torch.zeros(batch_size, 3, device=device, dtype=dtype)
        ious = ious.to(device=device, dtype=dtype).flatten(1)
        mean = ious.mean(dim=1, keepdim=True)
        maximum = ious.max(dim=1, keepdim=True).values
        if ious.size(1) > 1:
            top_two = torch.topk(ious, k=2, dim=1).values
            margin = top_two[:, :1] - top_two[:, 1:2]
        else:
            margin = torch.zeros_like(maximum)
        return torch.cat((mean, maximum, margin), dim=1)
