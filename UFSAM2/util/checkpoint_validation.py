"""Strict candidate-independent validation for SANSA adapter checkpoints."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_module_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for raw_key, value in state.items():
        key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
        if key in normalized:
            raise RuntimeError(f"Duplicate checkpoint key after prefix removal: {key}")
        normalized[key] = value
    return normalized


def validate_sansa_base_state(
    model_state: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    state = normalize_module_prefix(checkpoint_state)
    ignored = {
        key for key in state if key.endswith(("total_ops", "total_params"))
    }
    provided = sorted(key for key in state if "adapter" in key.lower())
    expected = sorted(key for key in model_state if "adapter" in key.lower())
    non_adapter = sorted(
        key for key in state if key not in provided and key not in ignored
    )
    unexpected = sorted(
        key for key in state if key not in model_state and key not in ignored
    )
    mismatches = sorted(
        (key, tuple(value.shape), tuple(model_state[key].shape))
        for key, value in state.items()
        if key in model_state
        and tuple(value.shape) != tuple(model_state[key].shape)
    )
    missing = sorted(set(expected) - set(provided))
    if not expected:
        raise RuntimeError("Configured SANSA contains no adapter parameters.")
    if not provided:
        raise RuntimeError("Checkpoint contains no SANSA adapter parameters.")
    if missing or non_adapter or unexpected or mismatches:
        raise RuntimeError(
            "SANSA adapter-only checkpoint mismatch. "
            f"missing={missing[:16]}; non_adapter={non_adapter[:16]}; "
            f"unexpected={unexpected[:16]}; shape_mismatches={mismatches[:16]}"
        )
    return {key: state[key] for key in provided}, expected
