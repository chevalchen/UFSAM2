"""Identity-checked replay of tracked few-shot episode manifests."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterator, Mapping


PARTITIONS = ("train", "calibration", "validation")


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_identity(
    value: Any,
    data_root: str | os.PathLike[str] | None,
) -> str:
    identity = os.fspath(value)
    if data_root and os.path.isabs(identity):
        try:
            identity = os.path.relpath(identity, os.fspath(data_root))
        except ValueError:
            pass
    return identity.replace("\\", "/")


def _scalar_int(value: Any) -> int:
    if hasattr(value, "numel") and callable(value.numel):
        if value.numel() != 1:
            raise ValueError("Expected scalar class_id.")
        value = value.item()
    elif hasattr(value, "item") and callable(value.item):
        value = value.item()
    return int(value)


def extract_episode_identity(
    sample: Mapping[str, Any],
    data_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    missing = [
        key for key in ("query_name", "support_names", "class_id")
        if key not in sample
    ]
    if missing:
        raise KeyError(f"Episode replay identity fields are missing: {missing}")
    support_names = sample["support_names"]
    if isinstance(support_names, (str, os.PathLike)):
        support_names = [support_names]
    return {
        "class_id": _scalar_int(sample["class_id"]),
        "query_id": _normalize_identity(sample["query_name"], data_root),
        "support_ids": [
            _normalize_identity(value, data_root) for value in support_names
        ],
    }


def validate_episode_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_dataset: str | None = None,
    expected_fold: int | None = None,
    expected_shots: int | None = None,
    expected_source_split: str | None = None,
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Only episode manifest schema_version=1 is supported.")
    for key, expected in (
        ("dataset", expected_dataset),
        ("fold", expected_fold),
        ("shots", expected_shots),
        ("source_split", expected_source_split),
    ):
        if expected is not None and manifest.get(key) != expected:
            raise ValueError(
                f"Episode manifest {key} mismatch: "
                f"expected {expected!r}, found {manifest.get(key)!r}."
            )
    partitions = manifest.get("partitions")
    counts = manifest.get("partition_counts")
    if not isinstance(partitions, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("Episode manifest requires partitions and partition_counts.")
    shots = int(manifest["shots"])
    seen_episode_ids: set[str] = set()
    images_by_partition: dict[str, set[str]] = {}
    for partition in PARTITIONS:
        entries = partitions.get(partition)
        if not isinstance(entries, list):
            raise ValueError(f"Manifest partition {partition!r} must be a list.")
        if len(entries) != int(counts.get(partition, -1)):
            raise ValueError(f"Manifest partition count mismatch for {partition}.")
        images: set[str] = set()
        for entry in entries:
            required = {
                "episode_id",
                "dataset_index",
                "episode_seed",
                "class_id",
                "query_id",
                "support_ids",
            }
            missing = required - set(entry)
            if missing:
                raise ValueError(
                    f"Manifest entry in {partition} is missing {sorted(missing)}."
                )
            episode_id = str(entry["episode_id"])
            if episode_id in seen_episode_ids:
                raise ValueError(f"Duplicate episode_id: {episode_id}")
            seen_episode_ids.add(episode_id)
            support_ids = [str(value) for value in entry["support_ids"]]
            if len(support_ids) != shots:
                raise ValueError(
                    f"Episode {episode_id} has {len(support_ids)} supports."
                )
            if len(set(support_ids)) != shots:
                raise ValueError(f"Episode {episode_id} has duplicate supports.")
            query_id = str(entry["query_id"])
            if query_id in support_ids:
                raise ValueError(f"Episode {episode_id} reuses its query.")
            images.add(query_id)
            images.update(support_ids)
        images_by_partition[partition] = images
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1:]:
            overlap = images_by_partition[left] & images_by_partition[right]
            if overlap:
                raise ValueError(
                    f"Episode partitions {left} and {right} share identities."
                )


def load_episode_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_episode_manifest(manifest)
    return manifest


@contextmanager
def seeded_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_module = None
    numpy_state = None
    torch_module = None
    torch_state = None
    try:
        try:
            import numpy as numpy_module  # type: ignore[no-redef]
            numpy_state = numpy_module.random.get_state()
            numpy_module.random.seed(seed % (2**32))
        except ImportError:
            numpy_module = None
        try:
            import torch as torch_module  # type: ignore[no-redef]
            torch_state = torch_module.random.get_rng_state()
            torch_module.manual_seed(seed)
        except ImportError:
            torch_module = None
        random.seed(seed)
        yield
    finally:
        random.setstate(python_state)
        if numpy_module is not None and numpy_state is not None:
            numpy_module.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.random.set_rng_state(torch_state)


class EpisodeReplayDataset:
    """Replay one frozen manifest partition and reject identity drift."""

    def __init__(
        self,
        dataset: Any,
        manifest: Mapping[str, Any],
        partition: str,
        *,
        data_root: str | os.PathLike[str] | None = None,
        expected_dataset: str | None = None,
        expected_fold: int | None = None,
        expected_shots: int | None = None,
        expected_source_split: str | None = None,
    ) -> None:
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown episode partition: {partition}")
        validate_episode_manifest(
            manifest,
            expected_dataset=expected_dataset,
            expected_fold=expected_fold,
            expected_shots=expected_shots,
            expected_source_split=expected_source_split,
        )
        self.dataset = dataset
        self.entries = list(manifest["partitions"][partition])
        self.partition = partition
        self.data_root = data_root
        for attribute in ("class_ids", "nclass", "benchmark"):
            if hasattr(dataset, attribute):
                setattr(self, attribute, getattr(dataset, attribute))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        with seeded_rng(int(entry["episode_seed"])):
            sample = self.dataset[int(entry["dataset_index"])]
        actual = extract_episode_identity(sample, self.data_root)
        expected = {
            "class_id": int(entry["class_id"]),
            "query_id": str(entry["query_id"]),
            "support_ids": [str(value) for value in entry["support_ids"]],
        }
        if actual != expected:
            raise RuntimeError(
                "Episode replay identity drift detected: "
                f"episode_id={entry['episode_id']}; "
                f"expected={expected}; actual={actual}"
            )
        replayed = dict(sample)
        replayed["episode_id"] = str(entry["episode_id"])
        replayed["episode_partition"] = self.partition
        return replayed
