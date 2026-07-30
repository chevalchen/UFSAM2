"""Frozen cached B0/D0...D4 evaluator for EXP-003 Stage A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from util.commons import make_deterministic, setup_logging
from util.episode_manifest import file_sha256
from util.exp003_leave_one_out import (
    BinaryCounts,
    PRIMITIVE_ROWS,
    PrimitiveEpisodeRecord,
    evaluate_frozen_rows,
    record_to_json,
)
from util.exp003_runtime import (
    build_frozen_model,
    load_resolved_contract,
    replay_validation_partition,
    resolve_repository_path,
)
from util.promptable_utils import build_prompt_dict


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _binary_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> BinaryCounts:
    prediction = prediction.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    target = target.detach().to(device="cpu").reshape(-1)
    valid = target != 255
    prediction = prediction[valid]
    target = target[valid].to(torch.bool)
    foreground_intersection = int((prediction & target).sum().item())
    foreground_union = int((prediction | target).sum().item())
    background_prediction = ~prediction
    background_target = ~target
    background_intersection = int(
        (background_prediction & background_target).sum().item()
    )
    background_union = int(
        (background_prediction | background_target).sum().item()
    )
    return BinaryCounts(
        background_intersection=background_intersection,
        foreground_intersection=foreground_intersection,
        background_union=background_union,
        foreground_union=foreground_union,
    )


def _prediction_fingerprint(
    episode_id: str,
    row_name: str,
    prediction: torch.Tensor,
) -> str:
    prediction = (
        prediction.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    )
    digest = hashlib.sha256()
    digest.update(episode_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(row_name.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(prediction.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(prediction.numpy().tobytes())
    return digest.hexdigest()


def _reference_primitive_predictions(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, torch.Tensor]:
    predictions = {}
    for row_name in PRIMITIVE_ROWS:
        drop_slot = None if row_name == "B0" else int(row_name[1:])
        kept_slots = [
            slot for slot in range(5)
            if slot != drop_slot
        ]
        support_imgs = batch["support_imgs"][:, kept_slots]
        support_masks = batch["support_masks"][:, kept_slots]
        samples = torch.cat(
            (support_imgs, batch["query_img"].unsqueeze(1)),
            dim=1,
        ).to(device)
        prompt_dict = build_prompt_dict(
            support_masks,
            "mask",
            n_shots=len(kept_slots),
            train_mode=False,
            device=model.device,
        )
        outputs = model(samples, prompt_dict)
        predictions[row_name] = (
            outputs["pred_masks"][-1].sigmoid() > threshold
        ).cpu()
    return predictions


def _verify_reference_equivalence(
    model: torch.nn.Module,
    batch: dict[str, Any],
    cached_logits: dict[str, torch.Tensor],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    references = _reference_primitive_predictions(
        model,
        batch,
        device=device,
        threshold=threshold,
    )
    rows = {}
    for row_name in PRIMITIVE_ROWS:
        cached_prediction = (
            cached_logits[row_name].sigmoid() > threshold
        ).cpu()
        exact = torch.equal(cached_prediction, references[row_name])
        rows[row_name] = {"binary_mask_exact": exact}
        if not exact:
            raise RuntimeError(
                f"Cached {row_name} is not mask-equivalent to its full reference path."
            )
    return {"passed": True, "rows": rows}


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    max_episodes: int | None,
    verify_reference_episodes: int,
) -> tuple[
    list[PrimitiveEpisodeRecord],
    dict[str, str],
    list[dict[str, Any]],
]:
    records = []
    global_digests = {
        row_name: hashlib.sha256() for row_name in PRIMITIVE_ROWS
    }
    reference_checks = []
    model.eval()
    for batch in loader:
        episode_value = batch["episode_id"]
        episode_id = (
            str(episode_value[0])
            if isinstance(episode_value, (list, tuple))
            else str(episode_value)
        )
        support_imgs = batch["support_imgs"]
        support_masks = batch["support_masks"]
        query_img = batch["query_img"]
        query_mask = batch["query_mask"][0]
        samples = torch.cat(
            (support_imgs, query_img.unsqueeze(1)),
            dim=1,
        ).to(device)
        prompt_dict = build_prompt_dict(
            support_masks,
            "mask",
            n_shots=5,
            train_mode=False,
            device=model.device,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model.forward_leave_one_out(samples, prompt_dict)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = 1000.0 * (time.perf_counter() - started)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        cached_logits = outputs["primitive_masks"]
        if len(reference_checks) < verify_reference_episodes:
            with torch.no_grad():
                check = _verify_reference_equivalence(
                    model,
                    batch,
                    cached_logits,
                    device=device,
                    threshold=threshold,
                )
            check["episode_id"] = episode_id
            reference_checks.append(check)

        primitive_counts = {}
        primitive_fingerprints = {}
        for row_name in PRIMITIVE_ROWS:
            prediction = (
                cached_logits[row_name].sigmoid() > threshold
            ).cpu()
            primitive_counts[row_name] = _binary_counts(
                prediction,
                query_mask,
            )
            fingerprint = _prediction_fingerprint(
                episode_id,
                row_name,
                prediction,
            )
            primitive_fingerprints[row_name] = fingerprint
            global_digests[row_name].update(episode_id.encode("utf-8"))
            global_digests[row_name].update(b"\0")
            global_digests[row_name].update(fingerprint.encode("ascii"))
            global_digests[row_name].update(b"\n")

        records.append(
            PrimitiveEpisodeRecord(
                episode_id=episode_id,
                class_id=int(batch["class_id"].reshape(-1)[0].item()),
                primitive_counts=primitive_counts,
                similarity_scores=tuple(
                    float(value)
                    for value in outputs["similarity_scores"].detach().cpu()
                ),
                primitive_fingerprints=primitive_fingerprints,
                six_decode_latency_ms=latency_ms,
                peak_device_memory_bytes=peak_memory,
            )
        )
        if max_episodes is not None and len(records) >= max_episodes:
            break
    return (
        records,
        {
            row_name: digest.hexdigest()
            for row_name, digest in global_digests.items()
        },
        reference_checks,
    )


def main(args: argparse.Namespace) -> None:
    resolved, contract, resolved_path, original_path = load_resolved_contract(
        args.contract
    )
    execution_config_path = resolve_repository_path(args.execution_config)
    execution_config = json.loads(
        execution_config_path.read_text(encoding="utf-8")
    )
    if execution_config.get("experiment_id") != "EXP-003":
        raise ValueError("Execution config is not for EXP-003.")
    if execution_config.get("execution_ready") is not True:
        raise ValueError("Execution config is not implementation-ready.")
    if float(args.threshold) != 0.5:
        raise ValueError("EXP-003 freezes the mask probability threshold at 0.5.")
    if not args.diagnostic_only and args.max_episodes is not None:
        raise ValueError("--max-episodes is DIAGNOSTIC_ONLY.")
    if not args.diagnostic_only and args.verify_reference_episodes:
        raise ValueError(
            "Full reference-path checks are a DIAGNOSTIC_ONLY smoke gate; "
            "formal compute is frozen at six cached query decodes per episode."
        )

    run_dir = Path(args.output_dir) / args.name_exp
    if run_dir.exists():
        unexpected = [
            path.name for path in run_dir.iterdir()
            if path.name != "run_manifest.json"
        ]
        if unexpected:
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory {run_dir}: "
                f"{unexpected}"
            )
    setup_logging(str(run_dir), console="info", rank=0)
    make_deterministic(int(contract["dataset"]["seed"]))
    device = torch.device(args.device)

    model, adapter_keys = build_frozen_model(
        contract,
        adapter_checkpoint=args.adapter_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
    )
    dataset = replay_validation_partition(
        resolved,
        contract,
        data_root=args.data_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    records, primitive_fingerprints, reference_checks = _evaluate(
        model,
        loader,
        device=device,
        threshold=float(args.threshold),
        max_episodes=args.max_episodes,
        verify_reference_episodes=args.verify_reference_episodes,
    )
    expected_count = int(
        resolved["episode_manifest"]["partition_counts"]["validation"]
    )
    unique_episode_ids = len({record.episode_id for record in records})
    if not args.diagnostic_only and (
        len(records) != expected_count
        or unique_episode_ids != expected_count
    ):
        raise RuntimeError(
            "Formal validation identity mismatch: "
            f"records={len(records)}, unique={unique_episode_ids}, "
            f"expected={expected_count}."
        )

    bootstrap = contract["evaluation"]["bootstrap"]
    metrics = evaluate_frozen_rows(
        records,
        bootstrap_resamples=int(bootstrap["resamples"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    evidence_label = "DIAGNOSTIC_ONLY" if args.diagnostic_only else "FORMAL"
    metrics.update(
        {
            "experiment_id": "EXP-003",
            "stage": "operator_headroom",
            "evidence_label": evidence_label,
            "git_sha": _git_sha(),
            "resolved_contract_sha256": file_sha256(resolved_path),
            "research_contract_sha256": file_sha256(original_path),
            "execution_config_sha256": file_sha256(execution_config_path),
            "episode_manifest_sha256": file_sha256(
                resolve_repository_path(
                    resolved["episode_manifest"]["path"]
                )
            ),
            "adapter_sha256": file_sha256(args.adapter_checkpoint),
            "sam2_checkpoint_sha256": file_sha256(args.sam2_checkpoint),
            "loaded_adapter_tensors": len(adapter_keys),
            "threshold": float(args.threshold),
            "primitive_prediction_fingerprints": primitive_fingerprints,
            "exact_replay": {
                "manifest_identity_verified": True,
                "expected_validation_episodes": expected_count,
                "observed_unique_episode_ids": unique_episode_ids,
                "b0_prediction_fingerprint": primitive_fingerprints["B0"],
            },
            "cached_reference_checks": reference_checks,
        }
    )

    records_path = run_dir / "stage_a_leave_one_out_per_episode.json"
    records_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "EXP-003",
                "stage": "operator_headroom",
                "evidence_label": evidence_label,
                "records": [record_to_json(record) for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics["per_episode_path"] = str(records_path)
    metrics["per_episode_sha256"] = file_sha256(records_path)
    metrics_path = run_dir / "stage_a_leave_one_out_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "metrics_path": str(metrics_path),
                "metrics_sha256": file_sha256(metrics_path),
                "automatic_stage_suggestion": metrics[
                    "automatic_stage_suggestion"
                ],
                "evidence_label": evidence_label,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluate frozen EXP-003 cached leave-one-out support memories"
    )
    parser.add_argument(
        "--contract",
        default=(
            "experiments/EXP-003/configs/"
            "stage_a_leave_one_out_contract_v2.json"
        ),
    )
    parser.add_argument(
        "--execution-config",
        default=(
            "experiments/EXP-003/configs/"
            "stage_a_leave_one_out_execution.json"
        ),
    )
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name-exp", required=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--verify-reference-episodes", type=int, default=0)
    main(parser.parse_args())
