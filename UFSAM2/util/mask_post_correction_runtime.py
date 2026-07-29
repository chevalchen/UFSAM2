"""Shared runtime helpers for the frozen EXP-002 Stage A implementation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from datasets import build_dataset
from models.sansa.sansa import SANSA, build_sansa
from util.checkpoint_validation import validate_sansa_base_state
from util.episode_manifest import EpisodeReplayDataset, file_sha256, load_episode_manifest


EXPERIMENT_ID = "EXP-002"
STAGE_ID = "operator_headroom"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root() / candidate


def load_stage_a_contract(path: str | Path) -> dict[str, Any]:
    contract_path = resolve_repository_path(path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(
            f"Expected {EXPERIMENT_ID} contract, found {contract.get('experiment_id')!r}."
        )
    if contract.get("stage", {}).get("id") != STAGE_ID:
        raise ValueError(
            f"Expected stage {STAGE_ID!r}, found {contract.get('stage')!r}."
        )
    if contract.get("execution_ready") is not False:
        raise ValueError(
            "The research contract must retain execution_ready=false. "
            "Formal authority comes from a separate frozen execution config."
        )
    return contract


def dataset_args_from_contract(
    contract: Mapping[str, Any],
    *,
    data_root: str | None = None,
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


def replay_partition_from_contract(
    contract: Mapping[str, Any],
    partition: str,
    *,
    data_root: str | None = None,
) -> EpisodeReplayDataset:
    args = dataset_args_from_contract(contract, data_root=data_root)
    dataset_config = contract["dataset"]
    manifest_config = dataset_config["episode_manifest"]
    manifest_path = resolve_repository_path(manifest_config["path"])
    actual_hash = file_sha256(manifest_path)
    if actual_hash != manifest_config["sha256"]:
        raise RuntimeError(
            "Episode manifest SHA256 mismatch: "
            f"expected {manifest_config['sha256']}, found {actual_hash}."
        )
    manifest = load_episode_manifest(manifest_path)
    dataset = build_dataset(
        str(dataset_config["name"]),
        image_set=str(dataset_config["source_split"]),
        args=args,
    )
    return EpisodeReplayDataset(
        dataset,
        manifest,
        partition,
        data_root=args.data_root,
        expected_dataset=str(dataset_config["name"]),
        expected_fold=int(dataset_config["fold"]),
        expected_shots=int(dataset_config["shots"]),
        expected_source_split=str(dataset_config["source_split"]),
    )


def _checkpoint_model_state(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint must be a mapping.")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint model state must be a mapping.")
    return state


def build_model_with_official_adapter(
    contract: Mapping[str, Any],
    *,
    adapter_path: str | None,
    device: str,
) -> tuple[SANSA, list[str]]:
    args = dataset_args_from_contract(contract)
    model = build_sansa(
        sam2_version=args.sam2_version,
        adaptformer_stages=args.adaptformer_stages,
        channel_factor=args.channel_factor,
        device=device,
        mask_post_correction=True,
    )
    expected_adapter = contract["baseline"]["official_adapter"]
    resolved_adapter_path = Path(adapter_path or expected_adapter["path"])
    actual_hash = file_sha256(resolved_adapter_path)
    if actual_hash != expected_adapter["sha256"]:
        raise RuntimeError(
            "Official adapter SHA256 mismatch: "
            f"expected {expected_adapter['sha256']}, found {actual_hash}."
        )
    checkpoint = torch.load(
        resolved_adapter_path,
        map_location="cpu",
        weights_only=False,
    )
    loadable, adapter_keys = validate_sansa_base_state(
        model.state_dict(),
        _checkpoint_model_state(checkpoint),
    )
    missing_keys, unexpected_keys = model.load_state_dict(loadable, strict=False)
    unexpected_keys = [
        key
        for key in unexpected_keys
        if not key.endswith(("total_ops", "total_params"))
    ]
    missing_adapters = sorted(set(adapter_keys) & set(missing_keys))
    if missing_adapters or unexpected_keys:
        raise RuntimeError(
            "Official adapter runtime load mismatch. "
            f"Missing adapters: {missing_adapters}; unexpected: {unexpected_keys}."
        )
    model.to(torch.device(device))
    return model, adapter_keys


def freeze_except_mask_post_refiner(model: SANSA) -> list[torch.nn.Parameter]:
    if model.mask_post_refiner is None:
        raise RuntimeError("EXP-002 requires mask_post_correction=True.")
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("mask_post_refiner.")
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("No mask-post-refiner parameters were made trainable.")
    return trainable
