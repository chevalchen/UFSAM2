"""Matched EXP-002 Stage A metrics and frozen oracle-row construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class BinaryCounts:
    background_intersection: float
    foreground_intersection: float
    background_union: float
    foreground_union: float


@dataclass(frozen=True)
class EpisodeMaskRecord:
    episode_id: str
    class_id: int
    b0: BinaryCounts
    b1: BinaryCounts
    support_fraction: float


def binary_counts(prediction: Tensor, target: Tensor) -> BinaryCounts:
    """Return foreground/background intersection and union on valid pixels."""
    prediction = prediction.detach().to(torch.bool).reshape(-1)
    target_flat = target.detach().reshape(-1)
    valid = target_flat != 255
    target_bool = target_flat.to(torch.bool)
    prediction = prediction[valid]
    target_bool = target_bool[valid]
    foreground_intersection = (prediction & target_bool).sum().item()
    foreground_union = (prediction | target_bool).sum().item()
    background_prediction = ~prediction
    background_target = ~target_bool
    background_intersection = (
        background_prediction & background_target
    ).sum().item()
    background_union = (background_prediction | background_target).sum().item()
    return BinaryCounts(
        background_intersection=float(background_intersection),
        foreground_intersection=float(foreground_intersection),
        background_union=float(background_union),
        foreground_union=float(foreground_union),
    )


def episode_foreground_iou(counts: BinaryCounts) -> float:
    return 100.0 * counts.foreground_intersection / max(counts.foreground_union, 1.0)


def _aggregate(
    records: Sequence[EpisodeMaskRecord],
    treatment_indices: set[int],
) -> dict[str, float]:
    class_counts: dict[int, list[float]] = {}
    episode_ious: list[float] = []
    total_bg_intersection = 0.0
    total_fg_intersection = 0.0
    total_bg_union = 0.0
    total_fg_union = 0.0
    support_fractions: list[float] = []
    for index, record in enumerate(records):
        use_treatment = index in treatment_indices
        counts = record.b1 if use_treatment else record.b0
        bucket = class_counts.setdefault(record.class_id, [0.0, 0.0])
        bucket[0] += counts.foreground_intersection
        bucket[1] += counts.foreground_union
        total_bg_intersection += counts.background_intersection
        total_fg_intersection += counts.foreground_intersection
        total_bg_union += counts.background_union
        total_fg_union += counts.foreground_union
        episode_ious.append(episode_foreground_iou(counts))
        support_fractions.append(record.support_fraction if use_treatment else 0.0)

    class_ious = [
        100.0 * intersection / max(union, 1.0)
        for intersection, union in class_counts.values()
    ]
    fb_iou = 50.0 * (
        total_bg_intersection / max(total_bg_union, 1.0)
        + total_fg_intersection / max(total_fg_union, 1.0)
    )
    decile_count = max(1, int(math.ceil(0.1 * len(episode_ious))))
    return {
        "miou": float(np.mean(class_ious)),
        "fb_iou": float(fb_iou),
        "mean_episode_iou": float(np.mean(episode_ious)),
        "worst_decile_episode_iou": float(
            np.mean(np.sort(np.asarray(episode_ious))[:decile_count])
        ),
        "intervention_rate": float(len(treatment_indices) / len(records)),
        "mean_intrinsic_activated_area": float(np.mean(support_fractions)),
    }


def paired_bootstrap(
    effects: Sequence[float],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float | list[float]]:
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Paired bootstrap requires a non-empty 1-D effect vector.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "resamples": int(resamples),
        "seed": int(seed),
    }


def _oracle_top_indices(
    records: Sequence[EpisodeMaskRecord],
    rate: float,
) -> set[int]:
    count = max(1, int(round(rate * len(records))))
    ranked = sorted(
        range(len(records)),
        key=lambda index: (
            -(
                episode_foreground_iou(records[index].b1)
                - episode_foreground_iou(records[index].b0)
            ),
            records[index].episode_id,
        ),
    )
    return set(ranked[:count])


def frozen_row_indices(
    records: Sequence[EpisodeMaskRecord],
) -> dict[str, set[int]]:
    if not records:
        raise ValueError("Matched evaluation requires at least one episode.")
    episode_ids = [record.episode_id for record in records]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Matched evaluation requires unique episode IDs.")
    positive = {
        index
        for index, record in enumerate(records)
        if episode_foreground_iou(record.b1) - episode_foreground_iou(record.b0) > 0.0
    }
    return {
        "B0": set(),
        "B1": set(range(len(records))),
        "B8_r25": _oracle_top_indices(records, 0.25),
        "B8_r50": _oracle_top_indices(records, 0.50),
        "B8_r75": _oracle_top_indices(records, 0.75),
        "B8_positive": positive,
    }


def evaluate_frozen_rows(
    records: Sequence[EpisodeMaskRecord],
    *,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Evaluate all frozen rows and emit the automatic, non-adjudicating gate."""
    row_indices = frozen_row_indices(records)
    b0_episode_ious = np.asarray(
        [episode_foreground_iou(record.b0) for record in records],
        dtype=np.float64,
    )
    rows: dict[str, dict[str, object]] = {}
    for row_name, treatment_indices in row_indices.items():
        aggregate = _aggregate(records, treatment_indices)
        treatment_episode_ious = np.asarray(
            [
                episode_foreground_iou(
                    record.b1 if index in treatment_indices else record.b0
                )
                for index, record in enumerate(records)
            ],
            dtype=np.float64,
        )
        bootstrap = paired_bootstrap(
            treatment_episode_ious - b0_episode_ious,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        rows[row_name] = {
            **aggregate,
            "paired_vs_b0": bootstrap,
            "selected_episode_ids": [
                records[index].episode_id for index in sorted(treatment_indices)
            ],
        }

    b0 = rows["B0"]
    effects: dict[str, dict[str, object]] = {}
    qualifying_rows: list[str] = []
    for row_name in ("B1", "B8_r25", "B8_r50", "B8_r75", "B8_positive"):
        row = rows[row_name]
        aggregate_effect = float(row["miou"]) - float(b0["miou"])
        fb_drop = float(b0["fb_iou"]) - float(row["fb_iou"])
        worst_decile_drop = (
            float(b0["worst_decile_episode_iou"])
            - float(row["worst_decile_episode_iou"])
        )
        ci_lower = float(row["paired_vs_b0"]["ci95"][0])  # type: ignore[index]
        guardrails_pass = fb_drop <= 0.2 and worst_decile_drop <= 0.5
        passes_headroom = aggregate_effect >= 0.5 or ci_lower > 0.0
        if passes_headroom and guardrails_pass:
            qualifying_rows.append(row_name)
        effects[row_name] = {
            "aggregate_miou_effect": aggregate_effect,
            "fb_iou_drop": fb_drop,
            "worst_decile_episode_iou_drop": worst_decile_drop,
            "guardrails_pass": guardrails_pass,
            "passes_headroom": passes_headroom,
        }

    b1_deltas = [
        episode_foreground_iou(record.b1) - episode_foreground_iou(record.b0)
        for record in records
    ]
    positive = sum(delta > 0.0 for delta in b1_deltas)
    negative = sum(delta < 0.0 for delta in b1_deltas)
    tied = len(b1_deltas) - positive - negative
    return {
        "schema_version": 1,
        "episode_count": len(records),
        "unique_episode_ids": len({record.episode_id for record in records}),
        "rows": rows,
        "effects": effects,
        "b1_episode_outcomes": {
            "positive": positive,
            "negative": negative,
            "tied": tied,
            "positive_rate": positive / len(records),
            "negative_rate": negative / len(records),
        },
        "automatic_stage_suggestion": (
            "PASS_TO_STAGE_B" if qualifying_rows else "DROP_OPERATOR"
        ),
        "qualifying_rows": qualifying_rows,
        "note": "Automatic suggestion only; the host Idea/Decision session adjudicates.",
    }


def record_to_json(record: EpisodeMaskRecord) -> dict[str, object]:
    return asdict(record)


def record_from_json(value: Mapping[str, object]) -> EpisodeMaskRecord:
    return EpisodeMaskRecord(
        episode_id=str(value["episode_id"]),
        class_id=int(value["class_id"]),
        b0=BinaryCounts(**value["b0"]),  # type: ignore[arg-type]
        b1=BinaryCounts(**value["b1"]),  # type: ignore[arg-type]
        support_fraction=float(value["support_fraction"]),
    )
