"""Pure validation helpers for candidate-independent SANSA base checkpoints."""

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
    """Return the loadable state and exact configured adapter-key contract."""
    state = normalize_module_prefix(checkpoint_state)
    ignored_profiler_keys = {
        key for key in state if key.endswith(("total_ops", "total_params"))
    }
    provided_adapter_keys = sorted(
        key for key in state if "adapter" in key.lower()
    )
    non_adapter_checkpoint_keys = sorted(
        key
        for key in state
        if key not in provided_adapter_keys and key not in ignored_profiler_keys
    )
    unexpected = sorted(
        key for key in state if key not in model_state and key not in ignored_profiler_keys
    )
    shape_mismatches = sorted(
        (
            key,
            tuple(value.shape),
            tuple(model_state[key].shape),
        )
        for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    )
    expected_adapter_keys = sorted(
        key for key in model_state if "adapter" in key.lower()
    )
    missing_adapter_keys = sorted(set(expected_adapter_keys) - set(provided_adapter_keys))

    if not expected_adapter_keys:
        raise RuntimeError(
            "The configured SANSA model contains no adapter parameters. "
            "Check --adaptformer_stages before loading the paper checkpoint."
        )
    if not provided_adapter_keys:
        raise RuntimeError("Checkpoint contains no SANSA adapter parameters.")
    if (
        missing_adapter_keys
        or non_adapter_checkpoint_keys
        or unexpected
        or shape_mismatches
    ):
        raise RuntimeError(
            "SANSA adapter-only checkpoint mismatch. "
            f"Missing configured adapter keys: {missing_adapter_keys[:16]}; "
            f"non-adapter keys: {non_adapter_checkpoint_keys[:16]}; "
            f"unexpected keys: {unexpected[:16]}; "
            f"shape mismatches: {shape_mismatches[:16]}"
        )

    loadable_state = {
        key: state[key] for key in provided_adapter_keys
    }
    return loadable_state, expected_adapter_keys
