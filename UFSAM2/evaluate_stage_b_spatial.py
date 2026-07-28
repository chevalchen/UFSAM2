"""Run the frozen EXP-001 Stage B matched-area spatial-policy evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import os
from os.path import join
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models.sansa.post_memory_calibration import (
    load_post_memory_calibrator_checkpoint,
)
from models.sansa.sansa import build_sansa
from train_post_memory_calibration import _as_binary_mask, _load_base_checkpoint
from util.commons import make_deterministic, setup_logging
from util.episode_manifest import (
    EpisodeReplayDataset,
    file_sha256,
    load_episode_manifest,
)
from util.metrics import Evaluator
from util.promptable_utils import build_prompt_dict
from util.stage_b_spatial import (
    choose_calibration_budget,
    dense_average_precision,
    deterministic_random_score,
    hard_top_area_gate,
    paired_bootstrap_interval,
    selected_cell_count,
    validate_area_budget,
)


def _checkpoint_payload_and_state(path: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported AV-PMC checkpoint payload: {path}")
    state = payload.get("post_memory_calibrator", payload.get("model", payload))
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported AV-PMC state dictionary: {path}")
    for prefix in ("module.post_memory_calibrator.", "post_memory_calibrator."):
        filtered = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if filtered:
            state = filtered
            break
    return payload, state


def _validate_stage_b_checkpoints(
    operator_path: str,
    benefit_path: str,
    risk_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    operator_sha = file_sha256(operator_path)
    operator_payload, operator_state = _checkpoint_payload_and_state(operator_path)
    benefit_payload, benefit_state = _checkpoint_payload_and_state(benefit_path)
    risk_payload, risk_state = _checkpoint_payload_and_state(risk_path)

    for label, payload, expected_target in (
        ("benefit", benefit_payload, "benefit"),
        ("risk", risk_payload, "risk"),
    ):
        if payload.get("stage") != "spatial":
            raise RuntimeError(f"{label} checkpoint is not a Stage B spatial checkpoint.")
        if payload.get("spatial_target") != expected_target:
            raise RuntimeError(
                f"{label} checkpoint target is {payload.get('spatial_target')!r}, "
                f"expected {expected_target!r}."
            )
        if payload.get("preceding_pmc_checkpoint_sha256") != operator_sha:
            raise RuntimeError(
                f"{label} checkpoint does not bind the frozen Stage A operator."
            )

    expected_keys = set(operator_state)
    if set(benefit_state) != expected_keys or set(risk_state) != expected_keys:
        raise RuntimeError("Stage A/B calibrator checkpoint keys do not match exactly.")
    for key in sorted(expected_keys):
        if key.startswith("spatial_head."):
            continue
        if not torch.equal(operator_state[key], benefit_state[key]):
            raise RuntimeError(f"Benefit checkpoint changed frozen tensor {key!r}.")
        if not torch.equal(operator_state[key], risk_state[key]):
            raise RuntimeError(f"Risk checkpoint changed frozen tensor {key!r}.")

    risk_spatial_state = {
        key[len("spatial_head."):]: value
        for key, value in risk_state.items()
        if key.startswith("spatial_head.")
    }
    if not risk_spatial_state:
        raise RuntimeError("Risk checkpoint contains no spatial-head tensors.")
    return benefit_payload, risk_payload, risk_spatial_state


def _episode_id(batch: dict[str, Any], index: int) -> str:
    value = batch.get("episode_id")
    if isinstance(value, (list, tuple)):
        value = value[0]
    if value is None:
        raise RuntimeError(f"Episode {index} has no stable episode_id.")
    return str(value)


def _query_logits_to_mask(
    logits: torch.Tensor,
    *,
    image_size: tuple[int, int],
    threshold: float,
) -> torch.Tensor:
    mask = F.interpolate(
        logits,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return (mask.sigmoid() > threshold)[:, 0]


class _MetricAccumulator:
    def __init__(self, *, nclass: int, class_ids: Iterable[int]) -> None:
        self.nclass = int(nclass)
        self.class_ids = np.asarray(list(class_ids), dtype=np.int64)
        self.intersection = np.zeros((2, self.nclass), dtype=np.float64)
        self.union = np.zeros((2, self.nclass), dtype=np.float64)
        self.episode_iou: list[float] = []
        self.episode_intersection: list[list[float]] = []
        self.episode_union: list[list[float]] = []
        self.class_id: list[int] = []
        self.area: list[float] = []

    def update(
        self,
        mask: torch.Tensor,
        batch: dict[str, Any],
        *,
        device: torch.device,
        class_id: int,
        area: float,
    ) -> float:
        area_inter, area_union = Evaluator.classify_prediction(
            mask.float(),
            batch,
            device=device,
        )
        inter = area_inter[:, 0].detach().cpu().numpy().astype(np.float64)
        union = area_union[:, 0].detach().cpu().numpy().astype(np.float64)
        self.intersection[:, class_id] += inter
        self.union[:, class_id] += union
        foreground_iou = float(inter[1] / max(union[1], 1.0) * 100.0)
        self.episode_iou.append(foreground_iou)
        self.episode_intersection.append(inter.tolist())
        self.episode_union.append(union.tolist())
        self.class_id.append(int(class_id))
        self.area.append(float(area))
        return foreground_iou

    def summary(self, baseline_episode_iou: list[float] | None = None) -> dict[str, Any]:
        selected_intersection = self.intersection[:, self.class_ids]
        selected_union = self.union[:, self.class_ids]
        class_iou = selected_intersection / np.maximum(selected_union, 1.0)
        miou = float(class_iou[1].mean() * 100.0)
        fb_iou = float(
            np.mean(
                selected_intersection.sum(axis=1)
                / np.maximum(selected_union.sum(axis=1), 1.0)
            )
            * 100.0
        )
        episode_values = np.asarray(self.episode_iou, dtype=np.float64)
        worst_count = max(1, int(np.ceil(0.1 * episode_values.size)))
        result: dict[str, Any] = {
            "miou": miou,
            "fb_iou": fb_iou,
            "episode_mean_iou": float(episode_values.mean()),
            "worst_decile_episode_iou": float(
                np.partition(episode_values, worst_count - 1)[:worst_count].mean()
            ),
            "activated_area_mean": float(np.mean(self.area)),
            "episodes": int(episode_values.size),
        }
        if baseline_episode_iou is not None:
            baseline = np.asarray(baseline_episode_iou, dtype=np.float64)
            if baseline.shape != episode_values.shape:
                raise RuntimeError("Baseline and treatment episode arrays are not matched.")
            result["negative_repair_rate"] = float(
                np.mean(episode_values < baseline)
            )
            result["tied_repair_rate"] = float(
                np.mean(episode_values == baseline)
            )
        return result


def _build_partition_dataset(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    partition: str,
):
    base_dataset = build_dataset(
        args.dataset_file,
        image_set=manifest["source_split"],
        args=args,
    )
    return EpisodeReplayDataset(
        base_dataset,
        manifest,
        partition,
        data_root=args.data_root,
        expected_dataset=args.dataset_file,
        expected_fold=args.fold,
        expected_shots=args.shots,
        expected_source_split=manifest["source_split"],
    )


def _row_name(policy: str, budget: float, seed: int | None = None) -> str:
    suffix = f"q{round(budget * 100):03d}"
    if seed is None:
        return f"{policy}_{suffix}"
    return f"{policy}_seed{seed}_{suffix}"


def _decode_with_gate(
    model: nn.Module,
    record: dict[str, Any],
    gate: torch.Tensor,
) -> torch.Tensor:
    memory_feature = record["memory_feature"]
    residual = record["residual"]
    high_res_features = record["high_res_features"]
    if memory_feature is None or residual is None or high_res_features is None:
        raise RuntimeError("Stage B decoder bundle is incomplete.")
    calibrated = memory_feature + gate * residual
    decoder_out = model.sam._forward_sam_heads(
        backbone_features=calibrated,
        high_res_features=high_res_features,
        multimask_output=bool(record["multimask_output"]),
    )
    if decoder_out.low_res_masks is None:
        raise RuntimeError("Stage B second decoder returned no selected-mask logits.")
    return decoder_out.low_res_masks


def _evaluate_partition(
    model: nn.Module,
    risk_head: nn.Module,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    partition: str,
    budgets: list[float],
    random_seeds: list[int],
    include_controls: bool,
) -> dict[str, Any]:
    dataset = _build_partition_dataset(args, manifest, partition)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)

    row_names = ["B0", "B1"]
    for budget in budgets:
        row_names.append(_row_name("B2", budget))
        if include_controls:
            row_names.extend(
                [
                    *[
                        _row_name("B3", budget, seed)
                        for seed in random_seeds
                    ],
                    _row_name("B4E", budget),
                    _row_name("B4R", budget),
                    _row_name("B7S", budget),
                    _row_name("B8S", budget),
                ]
            )
    accumulators = {
        name: _MetricAccumulator(nclass=dataset.nclass, class_ids=dataset.class_ids)
        for name in row_names
    }
    episodes: list[dict[str, Any]] = []
    dense_scores: list[np.ndarray] = []
    dense_labels: list[np.ndarray] = []
    seen_episode_ids: set[str] = set()
    full_area_sanity = True

    model.eval()
    risk_head.eval()
    progress = tqdm(loader, desc=f"Stage B {partition}", dynamic_ncols=True)
    for index, batch in enumerate(progress):
        episode_id = _episode_id(batch, index)
        if episode_id in seen_episode_ids:
            raise RuntimeError(f"Duplicate {partition} episode_id: {episode_id}")
        seen_episode_ids.add(episode_id)
        class_id = int(batch["class_id"].reshape(-1)[0].item())
        support_imgs = batch["support_imgs"]
        query_img = batch["query_img"]
        images = torch.cat([support_imgs[0], query_img]).unsqueeze(0).to(device)
        image_size = tuple(images.shape[-2:])
        prompt_dict = build_prompt_dict(
            batch["support_masks"],
            args.prompt,
            n_shots=args.shots,
            train_mode=False,
            device=model.device,
        )

        with torch.no_grad():
            outputs = model(
                images,
                prompt_dict,
                return_calibration_data=True,
            )
        records = outputs.get("post_memory_calibration", [])
        if len(records) != 1:
            raise RuntimeError("Stage B requires exactly one query record per episode.")
        record = records[0]
        baseline_logits = record["baseline_low_res_masks"]
        always_on_logits = record["calibrated_low_res_masks"]
        benefit_score = record["spatial_benefit"]
        entropy_score = record["mask_entropy"]
        spatial_head_input = record["spatial_head_input"]
        if any(
            value is None
            for value in (
                baseline_logits,
                always_on_logits,
                benefit_score,
                entropy_score,
                spatial_head_input,
            )
        ):
            raise RuntimeError("Stage B record is missing a required tensor.")

        with torch.no_grad():
            risk_score = risk_head(spatial_head_input)
        target = _as_binary_mask(
            batch["query_mask"],
            always_on_logits.shape[-2:],
            device,
        )
        baseline_loss = F.binary_cross_entropy_with_logits(
            baseline_logits.detach(),
            target,
            reduction="none",
        )
        always_on_loss = F.binary_cross_entropy_with_logits(
            always_on_logits.detach(),
            target,
            reduction="none",
        )
        dense_teacher = (baseline_loss - always_on_loss).clamp(-1.0, 1.0)
        dense_teacher = F.interpolate(
            dense_teacher,
            size=benefit_score.shape[-2:],
            mode="area",
        )
        dense_scores.append(benefit_score.detach().cpu().numpy())
        dense_labels.append((dense_teacher > 0).detach().cpu().numpy())

        baseline_mask = _query_logits_to_mask(
            baseline_logits,
            image_size=image_size,
            threshold=args.threshold,
        )
        always_on_mask = _query_logits_to_mask(
            always_on_logits,
            image_size=image_size,
            threshold=args.threshold,
        )
        episode_rows: dict[str, float] = {}
        baseline_iou = accumulators["B0"].update(
            baseline_mask,
            batch,
            device=device,
            class_id=class_id,
            area=0.0,
        )
        episode_rows["B0"] = baseline_iou
        episode_rows["B1"] = accumulators["B1"].update(
            always_on_mask,
            batch,
            device=device,
            class_id=class_id,
            area=1.0,
        )

        random_scores = {
            seed: deterministic_random_score(
                benefit_score,
                episode_id=episode_id,
                seed=seed,
            )
            for seed in random_seeds
        }
        for budget in budgets:
            expected_count = selected_cell_count(
                budget,
                benefit_score.shape[-2],
                benefit_score.shape[-1],
            )
            policies: list[tuple[str, torch.Tensor]] = [
                (
                    _row_name("B2", budget),
                    hard_top_area_gate(benefit_score, budget),
                )
            ]
            if include_controls:
                policies.extend(
                    [
                        *[
                            (
                                _row_name("B3", budget, seed),
                                hard_top_area_gate(random_scores[seed], budget),
                            )
                            for seed in random_seeds
                        ],
                        (
                            _row_name("B4E", budget),
                            hard_top_area_gate(entropy_score, budget),
                        ),
                        (
                            _row_name("B4R", budget),
                            hard_top_area_gate(risk_score, budget),
                        ),
                        (
                            _row_name("B7S", budget),
                            hard_top_area_gate(
                                benefit_score,
                                budget,
                                largest=False,
                            ),
                        ),
                        (
                            _row_name("B8S", budget),
                            hard_top_area_gate(dense_teacher, budget),
                        ),
                    ]
                )

            for row_name, gate in policies:
                actual_count = int(gate.detach().sum().item())
                if actual_count != expected_count:
                    raise RuntimeError(
                        f"{row_name} selected {actual_count} cells, expected "
                        f"{expected_count} for episode {episode_id}."
                    )
                actual_area = float(gate.detach().mean().item())
                if expected_count == benefit_score.shape[-2] * benefit_score.shape[-1]:
                    full_area_sanity = full_area_sanity and bool(
                        torch.equal(gate, torch.ones_like(gate))
                    )
                    policy_mask = always_on_mask
                else:
                    with torch.no_grad():
                        policy_logits = _decode_with_gate(model, record, gate)
                    policy_mask = _query_logits_to_mask(
                        policy_logits,
                        image_size=image_size,
                        threshold=args.threshold,
                    )
                episode_rows[row_name] = accumulators[row_name].update(
                    policy_mask,
                    batch,
                    device=device,
                    class_id=class_id,
                    area=actual_area,
                )

        episodes.append(
            {
                "episode_idx": index,
                "episode_id": episode_id,
                "class_id": class_id,
                "rows": episode_rows,
            }
        )

    baseline_episode_iou = accumulators["B0"].episode_iou
    summaries = {
        name: accumulator.summary(
            None if name == "B0" else baseline_episode_iou
        )
        for name, accumulator in accumulators.items()
    }
    return {
        "partition": partition,
        "episode_count": len(episodes),
        "unique_episode_ids": len(seen_episode_ids),
        "rows": summaries,
        "episodes": episodes,
        "dense_positive_benefit_auprc": dense_average_precision(
            dense_scores,
            dense_labels,
        ),
        "full_area_sanity": bool(full_area_sanity),
    }


def _episode_values(partition: dict[str, Any], row: str) -> list[float]:
    return [float(episode["rows"][row]) for episode in partition["episodes"]]


def _validation_decision(
    validation: dict[str, Any],
    *,
    selected_budget: float,
    random_seeds: list[int],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    q_suffix = f"q{round(selected_budget * 100):03d}"
    b2_row = f"B2_{q_suffix}"
    b4e_row = f"B4E_{q_suffix}"
    b4r_row = f"B4R_{q_suffix}"
    b2_values = np.asarray(_episode_values(validation, b2_row))
    b0_values = np.asarray(_episode_values(validation, "B0"))
    b3_values = np.mean(
        np.asarray(
            [
                _episode_values(validation, f"B3_seed{seed}_{q_suffix}")
                for seed in random_seeds
            ],
            dtype=np.float64,
        ),
        axis=0,
    )
    b4e_values = np.asarray(_episode_values(validation, b4e_row))
    b4r_values = np.asarray(_episode_values(validation, b4r_row))

    intervals = {
        "B2_minus_B0": paired_bootstrap_interval(
            b2_values,
            b0_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "B2_minus_mean_B3": paired_bootstrap_interval(
            b2_values,
            b3_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "B2_minus_B4E": paired_bootstrap_interval(
            b2_values,
            b4e_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "B2_minus_B4R": paired_bootstrap_interval(
            b2_values,
            b4r_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }
    rows = validation["rows"]
    b2_summary = rows[b2_row]
    b0_summary = rows["B0"]
    mean_b3_miou = float(
        np.mean(
            [
                rows[f"B3_seed{seed}_{q_suffix}"]["miou"]
                for seed in random_seeds
            ]
        )
    )
    point_effects = {
        "B2_minus_B0_miou": b2_summary["miou"] - b0_summary["miou"],
        "B2_minus_mean_B3_miou": b2_summary["miou"] - mean_b3_miou,
        "B2_minus_B4E_miou": b2_summary["miou"] - rows[b4e_row]["miou"],
        "B2_minus_B4R_miou": b2_summary["miou"] - rows[b4r_row]["miou"],
        "B2_minus_B1_miou": b2_summary["miou"] - rows["B1"]["miou"],
        "B8S_minus_B2_miou": (
            rows[f"B8S_{q_suffix}"]["miou"] - b2_summary["miou"]
        ),
    }
    guardrails = {
        "full_area_sanity": bool(validation["full_area_sanity"]),
        "fb_iou": b2_summary["fb_iou"] >= b0_summary["fb_iou"] - 0.2,
        "worst_decile": (
            b2_summary["worst_decile_episode_iou"]
            >= b0_summary["worst_decile_episode_iou"] - 0.5
        ),
    }
    required_interval_keys = (
        "B2_minus_mean_B3",
        "B2_minus_B4E",
        "B2_minus_B4R",
    )
    pass_threshold = (
        point_effects["B2_minus_B0_miou"] >= 0.5
        or intervals["B2_minus_B0"]["lower"] > 0.0
    )
    all_specificity_intervals_positive = all(
        intervals[key]["lower"] > 0.0 for key in required_interval_keys
    )
    point_keys = (
        "B2_minus_B0_miou",
        "B2_minus_mean_B3_miou",
        "B2_minus_B4E_miou",
        "B2_minus_B4R_miou",
    )
    any_non_positive_point = any(point_effects[key] <= 0.0 for key in point_keys)
    if (
        selected_budget < 1.0
        and pass_threshold
        and all_specificity_intervals_positive
        and all(guardrails.values())
    ):
        decision = "PASS_TO_STAGE_C"
    elif (
        selected_budget == 1.0
        or any_non_positive_point
    ):
        decision = "DROP_AND_STOP_EXP_001"
    elif not all(guardrails.values()):
        decision = "INCONCLUSIVE_GUARDRAIL_RERUN_REQUIRED"
    else:
        decision = "INCONCLUSIVE"
    return {
        "selected_budget": selected_budget,
        "decision": decision,
        "point_effects": point_effects,
        "paired_episode_bootstrap": intervals,
        "guardrails": guardrails,
    }


def main(args: argparse.Namespace) -> None:
    if not args.resume:
        raise ValueError("A verified SANSA --resume checkpoint is required.")
    for label, path in (
        ("operator", args.operator_checkpoint),
        ("benefit", args.benefit_checkpoint),
        ("risk", args.risk_checkpoint),
        ("episode manifest", args.episode_manifest),
    ):
        if not path:
            raise ValueError(f"--{label.replace(' ', '_')}_checkpoint/path is required.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    budgets = sorted({validate_area_budget(value) for value in args.pmc_area_budgets})
    if budgets != [0.1, 0.25, 0.5, 1.0]:
        raise ValueError(
            "EXP-001 Stage B requires the frozen area budgets 0.10, 0.25, 0.50, 1.00."
        )
    random_seeds = list(dict.fromkeys(int(seed) for seed in args.pmc_random_seeds))
    if random_seeds != [0, 1, 2]:
        raise ValueError("EXP-001 Stage B requires random control seeds 0, 1, 2.")

    output_path = join(args.output_dir, args.stage_b_metrics_file)
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite Stage B metrics: {output_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)

    benefit_payload, risk_payload, risk_spatial_state = (
        _validate_stage_b_checkpoints(
            args.operator_checkpoint,
            args.benefit_checkpoint,
            args.risk_checkpoint,
        )
    )
    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
        post_memory_calibration=True,
        pmc_mode="operator",
        pmc_projection_dim=args.pmc_projection_dim,
        pmc_hidden_dim=args.pmc_hidden_dim,
        pmc_residual_scale=args.pmc_residual_scale,
        pmc_spatial_area_budget=None,
    )
    _load_base_checkpoint(model, args.resume)
    load_post_memory_calibrator_checkpoint(
        model.post_memory_calibrator,
        args.benefit_checkpoint,
        strict=True,
    )
    model.post_memory_calibrator.set_mode("operator")
    model.to(torch.device(args.device))
    model.eval()

    risk_head = copy.deepcopy(model.post_memory_calibrator.spatial_head)
    risk_head.load_state_dict(risk_spatial_state, strict=True)
    risk_head.to(torch.device(args.device))
    risk_head.eval()

    manifest = load_episode_manifest(args.episode_manifest)
    calibration = _evaluate_partition(
        model,
        risk_head,
        args,
        manifest,
        partition="calibration",
        budgets=budgets,
        random_seeds=random_seeds,
        include_controls=False,
    )
    calibration_curve = {
        budget: calibration["rows"][_row_name("B2", budget)]["miou"]
        for budget in budgets
    }
    selected_budget = choose_calibration_budget(
        calibration_curve,
        tolerance_points=args.pmc_budget_tolerance,
    )
    validation = _evaluate_partition(
        model,
        risk_head,
        args,
        manifest,
        partition="validation",
        budgets=budgets,
        random_seeds=random_seeds,
        include_controls=True,
    )
    decision = _validation_decision(
        validation,
        selected_budget=selected_budget,
        random_seeds=random_seeds,
        bootstrap_resamples=args.pmc_bootstrap_resamples,
        bootstrap_seed=args.pmc_bootstrap_seed,
    )

    payload = {
        "schema_version": 1,
        "evidence_label": "FORMAL_ONLY_WHEN_BOUND_TO_CLEAN_RUN_MANIFEST",
        "experiment_id": "EXP-001",
        "stage": "Stage B",
        "dataset": args.dataset_file,
        "fold": args.fold,
        "shots": args.shots,
        "training_seed": args.seed,
        "frozen_area_budgets": budgets,
        "random_control_seeds": random_seeds,
        "budget_tolerance_points": args.pmc_budget_tolerance,
        "provenance": {
            "base_checkpoint": args.resume,
            "base_checkpoint_sha256": file_sha256(args.resume),
            "operator_checkpoint": args.operator_checkpoint,
            "operator_checkpoint_sha256": file_sha256(args.operator_checkpoint),
            "benefit_checkpoint": args.benefit_checkpoint,
            "benefit_checkpoint_sha256": file_sha256(args.benefit_checkpoint),
            "benefit_checkpoint_epoch": benefit_payload.get("epoch"),
            "risk_checkpoint": args.risk_checkpoint,
            "risk_checkpoint_sha256": file_sha256(args.risk_checkpoint),
            "risk_checkpoint_epoch": risk_payload.get("epoch"),
            "episode_manifest": args.episode_manifest,
            "episode_manifest_sha256": file_sha256(args.episode_manifest),
        },
        "calibration": calibration,
        "calibration_budget_curve": {
            str(budget): value for budget, value in calibration_curve.items()
        },
        "validation": validation,
        "stage_decision": decision,
    }
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_path, output_path)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"Wrote frozen Stage B matched-area evaluation to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluate EXP-001 Stage B matched-area spatial policies",
        parents=[opts.get_args_parser()],
    )
    parser.add_argument("--operator_checkpoint", required=True)
    parser.add_argument("--benefit_checkpoint", required=True)
    parser.add_argument("--risk_checkpoint", required=True)
    parser.add_argument(
        "--pmc_area_budgets",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--pmc_random_seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument("--pmc_budget_tolerance", type=float, default=0.2)
    parser.add_argument("--pmc_bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--pmc_bootstrap_seed", type=int, default=0)
    parser.add_argument(
        "--stage_b_metrics_file",
        default="stage_b_spatial_matched_metrics.json",
    )
    parsed_args = parser.parse_args()
    parsed_args.output_dir = join(parsed_args.output_dir, parsed_args.name_exp)
    main(parsed_args)
