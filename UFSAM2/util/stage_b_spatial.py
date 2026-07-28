"""Deterministic utilities for the EXP-001 Stage B spatial-policy harness."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def validate_area_budget(area_budget: float) -> float:
    budget = float(area_budget)
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"area_budget must be in (0, 1], got {area_budget!r}")
    return budget


def selected_cell_count(area_budget: float, height: int, width: int) -> int:
    budget = validate_area_budget(area_budget)
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}")
    return max(1, round(budget * height * width))


def hard_top_area_gate(
    score: Tensor,
    area_budget: float,
    *,
    largest: bool = True,
) -> Tensor:
    """Return an exact-cardinality gate with stable flattened-index tie breaks."""
    if score.ndim != 4 or score.size(1) != 1:
        raise ValueError(f"Expected score shaped [B,1,H,W], got {tuple(score.shape)}")
    height, width = score.shape[-2:]
    count = selected_cell_count(area_budget, height, width)
    if count == height * width:
        return torch.ones_like(score)

    flat_score = score.flatten(start_dim=1)
    order = torch.argsort(
        flat_score,
        dim=1,
        descending=largest,
        stable=True,
    )
    chosen = order[:, :count]
    flat_gate = torch.zeros_like(flat_score)
    flat_gate.scatter_(1, chosen, 1.0)
    return flat_gate.view_as(score)


def deterministic_random_score(
    reference: Tensor,
    *,
    episode_id: str,
    seed: int,
) -> Tensor:
    """Create an episode-keyed random score without changing global RNG state."""
    if reference.ndim != 4 or reference.size(1) != 1:
        raise ValueError(
            f"Expected reference shaped [B,1,H,W], got {tuple(reference.shape)}"
        )
    if reference.size(0) != 1:
        raise ValueError("Stage B deterministic random routing requires batch size 1.")
    key = f"EXP-001|Stage-B|{episode_id}|{int(seed)}".encode("utf-8")
    rng_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = random.Random(rng_seed)
    values = [rng.random() for _ in range(reference.numel())]
    return torch.tensor(
        values,
        dtype=reference.dtype,
        device=reference.device,
    ).view_as(reference)


def choose_calibration_budget(
    budget_to_miou: Mapping[float, float],
    *,
    tolerance_points: float = 0.2,
) -> float:
    """Choose the smallest budget within tolerance of the best calibration mIoU."""
    if not budget_to_miou:
        raise ValueError("At least one calibration budget result is required.")
    if tolerance_points < 0:
        raise ValueError("tolerance_points must be non-negative.")
    normalized = {
        validate_area_budget(float(budget)): float(miou)
        for budget, miou in budget_to_miou.items()
    }
    if not all(math.isfinite(value) for value in normalized.values()):
        raise ValueError("Calibration mIoU values must be finite.")
    best = max(normalized.values())
    eligible = [
        budget
        for budget, value in normalized.items()
        if best - value <= tolerance_points + 1e-12
    ]
    return min(eligible)


def paired_bootstrap_interval(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    resamples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap the mean paired episode effect with a percentile interval."""
    treatment_array = np.asarray(treatment, dtype=np.float64)
    control_array = np.asarray(control, dtype=np.float64)
    if treatment_array.ndim != 1 or control_array.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be one-dimensional.")
    if treatment_array.size == 0 or treatment_array.shape != control_array.shape:
        raise ValueError("Paired bootstrap inputs must be non-empty and shape-matched.")
    if not np.isfinite(treatment_array).all() or not np.isfinite(control_array).all():
        raise ValueError("Paired bootstrap inputs must be finite.")
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    differences = treatment_array - control_array
    rng = np.random.default_rng(seed)
    sample_size = differences.size
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    chunk_size = max(1, min(resamples, 1024))
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(
            0,
            sample_size,
            size=(stop - start, sample_size),
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return {
        "episodes": int(sample_size),
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "point_mean": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
    }


def dense_average_precision(
    scores: Iterable[np.ndarray],
    labels: Iterable[np.ndarray],
) -> float:
    """Compute exact binary average precision from dense score/label chunks."""
    score_chunks = [np.asarray(chunk, dtype=np.float64).reshape(-1) for chunk in scores]
    label_chunks = [np.asarray(chunk, dtype=np.bool_).reshape(-1) for chunk in labels]
    if not score_chunks or len(score_chunks) != len(label_chunks):
        raise ValueError("Dense AP requires non-empty, paired score and label chunks.")
    if any(score.shape != label.shape for score, label in zip(score_chunks, label_chunks)):
        raise ValueError("Every dense score chunk must match its label chunk.")
    score = np.concatenate(score_chunks)
    label = np.concatenate(label_chunks)
    positives = int(label.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="stable")
    ranked_label = label[order].astype(np.float64)
    true_positives = np.cumsum(ranked_label)
    precision = true_positives / np.arange(1, ranked_label.size + 1)
    return float((precision * ranked_label).sum() / positives)
