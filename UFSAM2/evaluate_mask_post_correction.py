"""Locked matched B0/B1/B8 evaluation for EXP-002 Stage A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch
from torch.utils.data import DataLoader

from models.sansa.mask_post_correction import load_mask_post_refiner_state
from util.commons import make_deterministic, setup_logging
from util.episode_manifest import file_sha256
from util.mask_post_correction_metrics import (
    EpisodeMaskRecord,
    binary_counts,
    evaluate_frozen_rows,
    record_to_json,
)
from util.mask_post_correction_runtime import (
    build_model_with_official_adapter,
    load_stage_a_contract,
    replay_partition_from_contract,
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


def _checkpoint_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("Refiner checkpoint must contain a model state mapping.")
    return checkpoint["model"]


def _hash_prediction(
    digest: Any,
    episode_id: str,
    prediction: torch.Tensor,
) -> None:
    digest.update(episode_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        prediction.detach().to(torch.uint8).cpu().contiguous().numpy().tobytes()
    )


def _evaluate_once(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    shots: int,
    prompt: str,
    threshold: float,
    max_episodes: int | None,
) -> tuple[list[EpisodeMaskRecord], str, str]:
    records: list[EpisodeMaskRecord] = []
    b0_digest = hashlib.sha256()
    b1_digest = hashlib.sha256()
    model.eval()
    for batch in loader:
        support_imgs = batch["support_imgs"]
        support_masks = batch["support_masks"]
        query_img = batch["query_img"]
        query_mask = batch["query_mask"]
        samples = torch.cat(
            (support_imgs[:, :shots], query_img.unsqueeze(1)),
            dim=1,
        ).to(device)
        prompt_dict = build_prompt_dict(
            support_masks[:, :shots],
            prompt,
            n_shots=shots,
            train_mode=False,
            device=model.device,
        )
        with torch.no_grad():
            outputs = model(
                samples,
                prompt_dict,
                return_mask_post_correction_data=True,
            )
        b0_prediction = (
            outputs["mask_post_baseline_pred_masks"][-1].sigmoid() > threshold
        )
        b1_prediction = outputs["pred_masks"][-1].sigmoid() > threshold
        target = query_mask[0]
        episode_id_value = batch["episode_id"]
        episode_id = (
            str(episode_id_value[0])
            if isinstance(episode_id_value, (list, tuple))
            else str(episode_id_value)
        )
        class_id = int(batch["class_id"].reshape(-1)[0].item())
        record = EpisodeMaskRecord(
            episode_id=episode_id,
            class_id=class_id,
            b0=binary_counts(b0_prediction, target),
            b1=binary_counts(b1_prediction, target),
            support_fraction=float(
                outputs["mask_post_support_fraction"].reshape(-1)[0].cpu()
            ),
        )
        records.append(record)
        _hash_prediction(b0_digest, episode_id, b0_prediction)
        _hash_prediction(b1_digest, episode_id, b1_prediction)
        if max_episodes is not None and len(records) >= max_episodes:
            break
    return records, b0_digest.hexdigest(), b1_digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    contract = load_stage_a_contract(args.contract)
    if float(args.threshold) != 0.5:
        raise ValueError("EXP-002 Stage A freezes the SANSA mask threshold at 0.5.")
    if not args.diagnostic_only and args.max_episodes is not None:
        raise ValueError("--max_episodes requires --diagnostic_only.")
    if not args.diagnostic_only and not args.verify_exact_replay:
        raise ValueError("Formal evaluation requires --verify_exact_replay.")

    run_dir = Path(args.output_dir) / args.name_exp
    if run_dir.exists():
        unexpected = [
            path.name for path in run_dir.iterdir() if path.name != "run_manifest.json"
        ]
        if unexpected:
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory {run_dir}: {unexpected}"
            )
    setup_logging(str(run_dir), console="info", rank=0)
    make_deterministic(int(contract["dataset"]["seed"]))

    model, adapter_keys = build_model_with_official_adapter(
        contract,
        adapter_path=args.adapter_checkpoint,
        sam2_checkpoint_path=args.sam2_checkpoint,
        device=args.device,
    )
    checkpoint = torch.load(
        args.refiner_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    contract_path = resolve_repository_path(args.contract)
    expected_contract_hash = file_sha256(contract_path)
    checkpoint_contract_hash = checkpoint.get("contract_sha256")
    if checkpoint_contract_hash != expected_contract_hash:
        raise RuntimeError(
            "Refiner checkpoint contract mismatch: "
            f"expected {expected_contract_hash}, found {checkpoint_contract_hash}."
        )
    loaded_refiner_keys = load_mask_post_refiner_state(
        model,
        _checkpoint_model_state(checkpoint),
    )
    dataset = replay_partition_from_contract(
        contract,
        "validation",
        data_root=args.data_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device)
    first_records, first_b0_hash, first_b1_hash = _evaluate_once(
        model,
        loader,
        device=device,
        shots=int(contract["dataset"]["shots"]),
        prompt=str(contract["dataset"]["prompt"]),
        threshold=float(args.threshold),
        max_episodes=args.max_episodes,
    )
    exact_replay = {
        "required": bool(args.verify_exact_replay),
        "passed": None,
        "b0_fingerprint": first_b0_hash,
        "b1_fingerprint": first_b1_hash,
    }
    if args.verify_exact_replay:
        second_records, second_b0_hash, second_b1_hash = _evaluate_once(
            model,
            loader,
            device=device,
            shots=int(contract["dataset"]["shots"]),
            prompt=str(contract["dataset"]["prompt"]),
            threshold=float(args.threshold),
            max_episodes=args.max_episodes,
        )
        if (
            first_records != second_records
            or first_b0_hash != second_b0_hash
            or first_b1_hash != second_b1_hash
        ):
            raise RuntimeError(
                "Exact replay failed: repeated matched predictions differ."
            )
        exact_replay["passed"] = True

    expected_count = int(
        contract["dataset"]["episode_manifest"]["partition_counts"]["validation"]
    )
    if not args.diagnostic_only and len(first_records) != expected_count:
        raise RuntimeError(
            f"Formal validation count mismatch: {len(first_records)} vs {expected_count}."
        )

    metrics = evaluate_frozen_rows(
        first_records,
        bootstrap_resamples=int(
            contract["evaluation"]["bootstrap"]["resamples"]
        ),
        bootstrap_seed=int(contract["evaluation"]["bootstrap"]["seed"]),
    )
    metrics.update(
        {
            "experiment_id": "EXP-002",
            "stage": "operator_headroom",
            "evidence_label": (
                "DIAGNOSTIC_ONLY" if args.diagnostic_only else "FORMAL"
            ),
            "git_sha": _git_sha(),
            "contract_sha256": expected_contract_hash,
            "episode_manifest_sha256": file_sha256(
                resolve_repository_path(
                    contract["dataset"]["episode_manifest"]["path"]
                )
            ),
            "adapter_sha256": file_sha256(
                Path(
                    args.adapter_checkpoint
                    or contract["baseline"]["official_adapter"]["path"]
                )
            ),
            "refiner_checkpoint_sha256": file_sha256(args.refiner_checkpoint),
            "loaded_adapter_tensors": len(adapter_keys),
            "loaded_refiner_tensors": len(loaded_refiner_keys),
            "threshold": float(args.threshold),
            "exact_replay": exact_replay,
        }
    )
    per_episode_path = run_dir / "stage_a_mask_refiner_per_episode.json"
    per_episode_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "EXP-002",
                "stage": "operator_headroom",
                "evidence_label": metrics["evidence_label"],
                "records": [record_to_json(record) for record in first_records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics["per_episode_path"] = str(per_episode_path)
    metrics["per_episode_sha256"] = file_sha256(per_episode_path)
    metrics_path = run_dir / "stage_a_mask_refiner_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "metrics_path": str(metrics_path),
                "metrics_sha256": file_sha256(metrics_path),
                "automatic_stage_suggestion": metrics["automatic_stage_suggestion"],
                "evidence_label": metrics["evidence_label"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate EXP-002 Stage A matched headroom")
    parser.add_argument(
        "--contract",
        default="experiments/EXP-002/configs/stage_a_mask_refiner_contract.json",
    )
    parser.add_argument("--adapter_checkpoint", default=None)
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--refiner_checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name_exp", required=True)
    parser.add_argument("--verify_exact_replay", action="store_true")
    parser.add_argument("--diagnostic_only", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    main(parser.parse_args())
