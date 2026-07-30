"""Pure frozen-row construction and metrics for EXP-003 Stage A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
import random
from typing import Any, Mapping, Sequence


PRIMITIVE_ROWS = ("B0", "D0", "D1", "D2", "D3", "D4")
RANDOM_CONTROL_SEEDS = (0, 1, 2, 3, 4)


def compact_support_entries(
    entries: Sequence[Any],
    *,
    drop_slot: int | None,
) -> dict[int, Any]:
    """Return a compact SANSA memory bank without copying entry values."""
    if len(entries) != 5:
        raise ValueError(f"EXP-003 requires five support entries, got {len(entries)}.")
    if drop_slot is not None and drop_slot not in range(5):
        raise ValueError(f"drop_slot must be None or 0..4, got {drop_slot!r}.")
    kept = [
        entry
        for slot, entry in enumerate(entries)
        if slot != drop_slot
    ]
    return {compact_slot: entry for compact_slot, entry in enumerate(kept)}


def random_drop_slot(episode_id: str, seed: int) -> int:
    """Frozen SHA256 selection for B1R_s0...s4."""
    payload = f"{episode_id}:{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % 5


def stable_argmin(values: Sequence[float]) -> int:
    if len(values) != 5:
        raise ValueError("EXP-003 similarity selection requires five scores.")
    return min(range(5), key=lambda index: (float(values[index]), index))


def stable_argmax(values: Sequence[float]) -> int:
    if len(values) != 5:
        raise ValueError("EXP-003 similarity selection requires five scores.")
    return min(range(5), key=lambda index: (-float(values[index]), index))


@dataclass(frozen=True)
class BinaryCounts:
    background_intersection: int
    foreground_intersection: int
    background_union: int
    foreground_union: int

    def foreground_iou_fraction(self) -> Fraction:
        if self.foreground_union == 0:
            return Fraction(1, 1)
        return Fraction(self.foreground_intersection, self.foreground_union)

    def foreground_iou_points(self) -> float:
        return 100.0 * float(self.foreground_iou_fraction())


@dataclass(frozen=True)
class PrimitiveEpisodeRecord:
    episode_id: str
    class_id: int
    primitive_counts: Mapping[str, BinaryCounts]
    similarity_scores: tuple[float, float, float, float, float]
    primitive_fingerprints: Mapping[str, str]
    six_decode_latency_ms: float
    peak_device_memory_bytes: int


def _validate_record(record: PrimitiveEpisodeRecord) -> None:
    if tuple(record.primitive_counts) != PRIMITIVE_ROWS:
        raise ValueError(
            "Primitive counts must be insertion-ordered exactly as B0,D0,D1,D2,D3,D4."
        )
    if set(record.primitive_fingerprints) != set(PRIMITIVE_ROWS):
        raise ValueError("Every primitive prediction requires a fingerprint.")
    if len(record.similarity_scores) != 5:
        raise ValueError("Every episode requires five similarity scores.")


def selected_rows(record: PrimitiveEpisodeRecord) -> dict[str, str]:
    """Return the primitive row selected by every frozen derived row."""
    _validate_record(record)
    selections = {"B0": "B0"}
    for seed in RANDOM_CONTROL_SEEDS:
        slot = random_drop_slot(record.episode_id, seed)
        selections[f"B1R_s{seed}"] = f"D{slot}"

    similarities = record.similarity_scores
    selections["B2S"] = f"D{stable_argmin(similarities)}"
    selections["B7I"] = f"D{stable_argmax(similarities)}"

    drop_fractions = [
        record.primitive_counts[f"D{slot}"].foreground_iou_fraction()
        for slot in range(5)
    ]
    forced_slot = min(
        range(5),
        key=lambda slot: (-drop_fractions[slot], slot),
    )
    selections["B8_forced"] = f"D{forced_slot}"
    b0_fraction = record.primitive_counts["B0"].foreground_iou_fraction()
    selections["B8_positive"] = (
        f"D{forced_slot}"
        if drop_fractions[forced_slot] > b0_fraction
        else "B0"
    )
    return selections


def _aggregate_counts(
    records: Sequence[PrimitiveEpisodeRecord],
    primitive_by_episode: Sequence[str],
) -> dict[str, float]:
    class_counts: dict[int, list[int]] = {}
    episode_ious: list[float] = []
    totals = [0, 0, 0, 0]
    for record, primitive in zip(records, primitive_by_episode):
        counts = record.primitive_counts[primitive]
        bucket = class_counts.setdefault(record.class_id, [0, 0])
        bucket[0] += counts.foreground_intersection
        bucket[1] += counts.foreground_union
        totals[0] += counts.background_intersection
        totals[1] += counts.foreground_intersection
        totals[2] += counts.background_union
        totals[3] += counts.foreground_union
        episode_ious.append(counts.foreground_iou_points())

    class_ious = [
        100.0 * intersection / max(union, 1)
        for intersection, union in class_counts.values()
    ]
    decile_count = max(1, int(math.ceil(0.1 * len(episode_ious))))
    return {
        "miou": sum(class_ious) / len(class_ious),
        "fb_iou": 50.0 * (
            totals[0] / max(totals[2], 1)
            + totals[1] / max(totals[3], 1)
        ),
        "mean_episode_iou": sum(episode_ious) / len(episode_ious),
        "worst_decile_episode_iou": (
            sum(sorted(episode_ious)[:decile_count]) / decile_count
        ),
    }


def paired_bootstrap(
    effects: Sequence[float],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    if not effects:
        raise ValueError("Paired bootstrap requires at least one effect.")
    size = len(effects)
    try:
        import numpy as np

        values = np.asarray(effects, dtype=np.float64)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, size, size=(resamples, size))
        sample_means = values[indices].mean(axis=1)
        ci95 = [
            float(np.percentile(sample_means, 2.5)),
            float(np.percentile(sample_means, 97.5)),
        ]
    except ImportError:
        rng = random.Random(seed)
        sorted_means = sorted(
            sum(effects[rng.randrange(size)] for _ in range(size)) / size
            for _ in range(resamples)
        )

        def percentile(probability: float) -> float:
            position = probability * (len(sorted_means) - 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                return sorted_means[lower]
            weight = position - lower
            return (
                sorted_means[lower] * (1.0 - weight)
                + sorted_means[upper] * weight
            )

        ci95 = [percentile(0.025), percentile(0.975)]

    return {
        "mean": sum(effects) / size,
        "ci95": ci95,
        "resamples": resamples,
        "seed": seed,
    }


def evaluate_frozen_rows(
    records: Sequence[PrimitiveEpisodeRecord],
    *,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("EXP-003 matched evaluation requires records.")
    episode_ids = [record.episode_id for record in records]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("EXP-003 requires unique episode identities.")
    selections = [selected_rows(record) for record in records]
    row_names = (
        "B0",
        "B1R_s0",
        "B1R_s1",
        "B1R_s2",
        "B1R_s3",
        "B1R_s4",
        "B2S",
        "B7I",
        "B8_forced",
        "B8_positive",
    )
    b0_ious = [
        record.primitive_counts["B0"].foreground_iou_points()
        for record in records
    ]
    rows: dict[str, dict[str, Any]] = {}
    row_episode_ious: dict[str, list[float]] = {}
    for row_name in row_names:
        primitives = [
            selection[row_name] for selection in selections
        ]
        episode_ious = [
            record.primitive_counts[primitive].foreground_iou_points()
            for record, primitive in zip(records, primitives)
        ]
        row_episode_ious[row_name] = episode_ious
        rows[row_name] = {
            **_aggregate_counts(records, primitives),
            "paired_vs_b0": paired_bootstrap(
                [
                    treatment - baseline
                    for treatment, baseline in zip(episode_ious, b0_ious)
                ],
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
            "selected_primitive_counts": {
                primitive: primitives.count(primitive)
                for primitive in PRIMITIVE_ROWS
                if primitive in primitives
            },
        }

    random_rows = [f"B1R_s{seed}" for seed in RANDOM_CONTROL_SEEDS]
    rows["B1R_mean"] = {
        metric: sum(float(rows[name][metric]) for name in random_rows) / 5.0
        for metric in (
            "miou",
            "fb_iou",
            "mean_episode_iou",
            "worst_decile_episode_iou",
        )
    }
    mean_random_episode_ious = [
        sum(row_episode_ious[name][index] for name in random_rows) / 5.0
        for index in range(len(records))
    ]
    rows["B1R_mean"]["paired_vs_b0"] = paired_bootstrap(
        [
            treatment - baseline
            for treatment, baseline in zip(mean_random_episode_ious, b0_ious)
        ],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    effects = {}
    for row_name in (*row_names[1:], "B1R_mean"):
        episode_ious = (
            mean_random_episode_ious
            if row_name == "B1R_mean"
            else row_episode_ious[row_name]
        )
        deltas = [
            treatment - baseline
            for treatment, baseline in zip(episode_ious, b0_ious)
        ]
        effects[row_name] = {
            "aggregate_miou_effect": rows[row_name]["miou"] - rows["B0"]["miou"],
            "paired_episode_effect": rows[row_name]["paired_vs_b0"],
            "positive_episodes": sum(delta > 0.0 for delta in deltas),
            "negative_episodes": sum(delta < 0.0 for delta in deltas),
            "tied_episodes": sum(delta == 0.0 for delta in deltas),
        }

    b2s_fb_drop = rows["B0"]["fb_iou"] - rows["B2S"]["fb_iou"]
    b2s_worst_drop = (
        rows["B0"]["worst_decile_episode_iou"]
        - rows["B2S"]["worst_decile_episode_iou"]
    )
    b2s_vs_random_effects = [
        b2s - random_mean
        for b2s, random_mean in zip(
            row_episode_ious["B2S"],
            mean_random_episode_ious,
        )
    ]
    b2s_vs_random = {
        "aggregate_miou_effect": (
            rows["B2S"]["miou"] - rows["B1R_mean"]["miou"]
        ),
        "paired_episode_effect": paired_bootstrap(
            b2s_vs_random_effects,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }
    primary = effects["B8_positive"]
    guardrails_pass = b2s_fb_drop <= 0.2 and b2s_worst_drop <= 0.5
    stage_a_gate = (
        primary["aggregate_miou_effect"] >= 0.5
        and primary["paired_episode_effect"]["ci95"][0] > 0.0
        and guardrails_pass
    )
    deterministic_candidate = (
        stage_a_gate
        and effects["B2S"]["aggregate_miou_effect"] >= 0.5
        and effects["B2S"]["paired_episode_effect"]["ci95"][0] > 0.0
        and guardrails_pass
    )
    similarity_oracle_agreement = sum(
        selection["B2S"] == selection["B8_forced"]
        for selection in selections
    ) / len(records)
    return {
        "schema_version": 1,
        "episode_count": len(records),
        "unique_episode_ids": len(set(episode_ids)),
        "rows": rows,
        "effects": effects,
        "deployable_supporting_effects": {
            "B2S_minus_B0": effects["B2S"],
            "B2S_minus_B1R_mean": b2s_vs_random,
        },
        "supporting": {
            "oracle_intervention_rate": sum(
                selection["B8_positive"] != "B0"
                for selection in selections
            ) / len(records),
            "similarity_oracle_top1_agreement": similarity_oracle_agreement,
            "mean_six_decode_latency_ms": sum(
                record.six_decode_latency_ms for record in records
            ) / len(records),
            "peak_device_memory_bytes": max(
                record.peak_device_memory_bytes for record in records
            ),
        },
        "guardrails": {
            "b2s_fb_iou_drop_points_vs_b0": b2s_fb_drop,
            "b2s_worst_decile_episode_iou_drop_points_vs_b0": b2s_worst_drop,
            "b2s_fb_iou_pass": b2s_fb_drop <= 0.2,
            "b2s_worst_decile_pass": b2s_worst_drop <= 0.5,
            "all_pass": guardrails_pass,
        },
        "automatic_stage_suggestion": (
            "PASS_TO_STAGE_B" if stage_a_gate else "DROP"
        ),
        "deterministic_candidate": deterministic_candidate,
        "note": "Automatic contract check only; the host Idea/Decision session adjudicates.",
    }


def record_to_json(record: PrimitiveEpisodeRecord) -> dict[str, Any]:
    return asdict(record)


def record_from_json(value: Mapping[str, Any]) -> PrimitiveEpisodeRecord:
    primitive_counts = {
        name: BinaryCounts(**value["primitive_counts"][name])
        for name in PRIMITIVE_ROWS
    }
    return PrimitiveEpisodeRecord(
        episode_id=str(value["episode_id"]),
        class_id=int(value["class_id"]),
        primitive_counts=primitive_counts,
        similarity_scores=tuple(float(v) for v in value["similarity_scores"]),
        primitive_fingerprints=dict(value["primitive_fingerprints"]),
        six_decode_latency_ms=float(value["six_decode_latency_ms"]),
        peak_device_memory_bytes=int(value["peak_device_memory_bytes"]),
    )
