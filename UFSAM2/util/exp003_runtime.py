"""Runtime binding for the frozen EXP-003 Stage A implementation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from datasets import build_dataset
from models.sansa.sansa import SANSA, build_sansa
from util.checkpoint_validation import validate_sansa_base_state
from util.episode_manifest import (
    EpisodeReplayDataset,
    file_sha256,
    load_episode_manifest,
)


EXPERIMENT_ID = "EXP-003"
STAGE_ID = "operator_headroom"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def load_resolved_contract(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    resolved_path = resolve_repository_path(path)
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if resolved.get("schema_version") != 2:
        raise ValueError("EXP-003 requires resolved contract schema_version=2.")
    if resolved.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Resolved contract is not for EXP-003.")
    if resolved.get("stage", {}).get("id") != STAGE_ID:
        raise ValueError("Resolved contract is not for Stage A operator headroom.")
    if resolved.get("implementation_ready") is not True:
        raise ValueError("Resolved contract is not implementation-ready.")
    if resolved.get("execution_ready") is not False:
        raise ValueError(
            "The research contract must retain execution_ready=false; "
            "execution authority comes from the separately signed config."
        )
    original_path = resolve_repository_path(resolved["supersedes"]["path"])
    original_hash = file_sha256(original_path)
    if original_hash != resolved["supersedes"]["sha256"]:
        raise RuntimeError(
            "Superseded research contract SHA256 mismatch: "
            f"expected {resolved['supersedes']['sha256']}, found {original_hash}."
        )
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Superseded contract is not for EXP-003.")
    return resolved, original, resolved_path, original_path


def dataset_args(
    contract: Mapping[str, Any],
    *,
    data_root: str | None,
) -> SimpleNamespace:
    dataset = contract["dataset"]
    baseline = contract["baseline"]
    return SimpleNamespace(
        data_root=data_root or dataset["data_root"],
        fold=int(dataset["fold"]),
        shots=int(dataset["shots"]),
        prompt=str(dataset["prompt"]),
        sam2_version=str(baseline["sam2_version"]),
        adaptformer_stages=list(baseline["adaptformer_stages"]),
        channel_factor=float(baseline["channel_factor"]),
    )


def replay_validation_partition(
    resolved: Mapping[str, Any],
    original: Mapping[str, Any],
    *,
    data_root: str | None,
) -> EpisodeReplayDataset:
    manifest_config = resolved["episode_manifest"]
    manifest_path = resolve_repository_path(manifest_config["path"])
    actual_hash = file_sha256(manifest_path)
    if actual_hash != manifest_config["sha256"]:
        raise RuntimeError(
            "Episode manifest SHA256 mismatch: "
            f"expected {manifest_config['sha256']}, found {actual_hash}."
        )
    manifest = load_episode_manifest(manifest_path)
    args = dataset_args(original, data_root=data_root)
    dataset_config = original["dataset"]
    dataset = build_dataset(
        str(dataset_config["name"]),
        image_set=str(dataset_config["source_split"]),
        args=args,
    )
    return EpisodeReplayDataset(
        dataset,
        manifest,
        "validation",
        data_root=args.data_root,
        expected_dataset=str(dataset_config["name"]),
        expected_fold=int(dataset_config["fold"]),
        expected_shots=int(dataset_config["shots"]),
        expected_source_split=str(dataset_config["source_split"]),
    )


def _checkpoint_state(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Official adapter checkpoint must be a mapping.")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise RuntimeError("Official adapter model state must be a mapping.")
    return state


def build_frozen_model(
    contract: Mapping[str, Any],
    *,
    adapter_checkpoint: str,
    sam2_checkpoint: str,
    device: str,
) -> tuple[SANSA, list[str]]:
    baseline = contract["baseline"]
    adapter_expected = baseline["official_adapter"]
    adapter_path = Path(adapter_checkpoint)
    adapter_hash = file_sha256(adapter_path)
    if adapter_hash != adapter_expected["sha256"]:
        raise RuntimeError(
            "Official adapter SHA256 mismatch: "
            f"expected {adapter_expected['sha256']}, found {adapter_hash}."
        )
    sam2_expected = baseline["sam2_large_base_weights"]
    sam2_path = Path(sam2_checkpoint)
    sam2_hash = file_sha256(sam2_path)
    if sam2_hash != sam2_expected["sha256"]:
        raise RuntimeError(
            "SAM2-Large SHA256 mismatch: "
            f"expected {sam2_expected['sha256']}, found {sam2_hash}."
        )
    model = build_sansa(
        sam2_version=str(baseline["sam2_version"]),
        adaptformer_stages=list(baseline["adaptformer_stages"]),
        channel_factor=float(baseline["channel_factor"]),
        device=device,
        sam2_checkpoint=str(sam2_path),
    )
    checkpoint = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=False,
    )
    loadable, adapter_keys = validate_sansa_base_state(
        model.state_dict(),
        _checkpoint_state(checkpoint),
    )
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    missing_adapters = sorted(set(adapter_keys) & set(missing))
    unexpected = [
        key for key in unexpected
        if not key.endswith(("total_ops", "total_params"))
    ]
    if missing_adapters or unexpected:
        raise RuntimeError(
            "Official adapter load mismatch: "
            f"missing adapters={missing_adapters}; unexpected={unexpected}."
        )
    model.to(torch.device(device))
    model.eval()
    return model, adapter_keys
