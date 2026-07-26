"""Deterministic, identity-checked episode manifests for formal FSS runs."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
PARTITIONS = ("train", "calibration", "validation")
GENERATION_ORDER = ("validation", "calibration", "train")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_seed(seed: int, partition: str, attempt: int, namespace: str) -> int:
    payload = f"{seed}:{partition}:{attempt}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _normalize_identity(value: Any, data_root: str | os.PathLike[str] | None) -> str:
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
            raise ValueError(f"Expected scalar class_id, got shape {tuple(value.shape)}")
        value = value.item()
    elif hasattr(value, "item") and callable(value.item):
        value = value.item()
    return int(value)


def extract_episode_identity(
    sample: Mapping[str, Any],
    data_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    missing = [
        key
        for key in ("query_name", "support_names", "class_id")
        if key not in sample
    ]
    if missing:
        raise KeyError(
            "Episode replay requires identity fields in every dataset sample. "
            f"Missing: {missing}"
        )
    support_names = sample["support_names"]
    if isinstance(support_names, (str, os.PathLike)):
        support_names = [support_names]
    identity = {
        "class_id": _scalar_int(sample["class_id"]),
        "query_id": _normalize_identity(sample["query_name"], data_root),
        "support_ids": [
            _normalize_identity(value, data_root) for value in support_names
        ],
    }
    if not identity["support_ids"]:
        raise ValueError("An episode must contain at least one support identity.")
    return identity


@contextmanager
def seeded_rng(seed: int) -> Iterator[None]:
    """Temporarily seed Python, NumPy, and Torch without leaking RNG state."""
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


def sample_seeded_episode(dataset: Any, dataset_index: int, episode_seed: int) -> Any:
    with seeded_rng(episode_seed):
        return dataset[dataset_index]


def _episode_id(
    dataset_name: str,
    fold: int,
    shots: int,
    source_split: str,
    identity: Mapping[str, Any],
) -> str:
    payload = {
        "dataset": dataset_name,
        "fold": fold,
        "shots": shots,
        "source_split": source_split,
        **identity,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def _partition_images(entries: Sequence[Mapping[str, Any]]) -> set[str]:
    images: set[str] = set()
    for entry in entries:
        images.add(str(entry["query_id"]))
        images.update(str(value) for value in entry["support_ids"])
    return images


def validate_episode_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_dataset: str | None = None,
    expected_fold: int | None = None,
    expected_shots: int | None = None,
    expected_source_split: str | None = None,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported episode manifest schema: {manifest.get('schema_version')}"
        )
    for key, expected in (
        ("dataset", expected_dataset),
        ("fold", expected_fold),
        ("shots", expected_shots),
        ("source_split", expected_source_split),
    ):
        if expected is not None and manifest.get(key) != expected:
            raise ValueError(
                f"Episode manifest {key} mismatch: expected {expected!r}, "
                f"found {manifest.get(key)!r}"
            )

    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("Episode manifest has no partitions mapping.")
    seen_episode_ids: set[str] = set()
    partition_images: dict[str, set[str]] = {}
    declared_counts = manifest.get("partition_counts")
    if not isinstance(declared_counts, Mapping):
        raise ValueError("Episode manifest has no partition_counts mapping.")
    shots = int(manifest.get("shots", 0))
    if shots <= 0:
        raise ValueError(f"Episode manifest shots must be positive, got {shots}.")
    for partition in PARTITIONS:
        entries = partitions.get(partition)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Episode manifest partition {partition!r} is empty.")
        if int(declared_counts.get(partition, -1)) != len(entries):
            raise ValueError(
                f"Episode manifest count mismatch for {partition!r}: "
                f"declared {declared_counts.get(partition)!r}, found {len(entries)}."
            )
        for entry in entries:
            required = {
                "episode_id",
                "dataset_index",
                "episode_seed",
                "class_id",
                "query_id",
                "support_ids",
            }
            missing = sorted(required - set(entry))
            if missing:
                raise ValueError(
                    f"Episode manifest entry in {partition!r} is missing {missing}."
                )
            episode_id = str(entry["episode_id"])
            if episode_id in seen_episode_ids:
                raise ValueError(f"Duplicate episode_id in manifest: {episode_id}")
            seen_episode_ids.add(episode_id)
            support_ids = [str(value) for value in entry["support_ids"]]
            if len(support_ids) != shots:
                raise ValueError(
                    f"Episode {episode_id} has {len(support_ids)} supports, expected {shots}."
                )
            if len(set(support_ids)) != len(support_ids):
                raise ValueError(f"Episode {episode_id} contains duplicate supports.")
            if str(entry["query_id"]) in set(support_ids):
                raise ValueError(f"Episode {episode_id} reuses its query as support.")
        partition_images[partition] = _partition_images(entries)

    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            overlap = partition_images[left] & partition_images[right]
            if overlap:
                examples = sorted(overlap)[:8]
                raise ValueError(
                    f"Episode partitions {left!r} and {right!r} share image identities: "
                    f"{examples}"
                )


def generate_episode_manifest(
    dataset: Any,
    *,
    dataset_name: str,
    source_split: str,
    fold: int,
    shots: int,
    counts: Mapping[str, int],
    seed: int,
    data_root: str | os.PathLike[str] | None = None,
    source_git_sha: str | None = None,
    max_attempts_per_episode: int = 500,
) -> dict[str, Any]:
    """Materialize disjoint train/calibration/validation episode identities."""
    if len(dataset) <= 0:
        raise ValueError("Cannot generate an episode manifest from an empty dataset.")
    normalized_counts = {name: int(counts.get(name, 0)) for name in PARTITIONS}
    if any(value <= 0 for value in normalized_counts.values()):
        raise ValueError(f"All manifest partition counts must be positive: {normalized_counts}")

    partitions: dict[str, list[dict[str, Any]]] = {name: [] for name in PARTITIONS}
    reserved_images: set[str] = set()
    for partition in GENERATION_ORDER:
        entries = partitions[partition]
        partition_images: set[str] = set()
        partition_episode_ids: set[str] = set()
        attempt = 0
        maximum_attempts = normalized_counts[partition] * max_attempts_per_episode
        while len(entries) < normalized_counts[partition]:
            if attempt >= maximum_attempts:
                raise RuntimeError(
                    f"Could not generate {normalized_counts[partition]} disjoint "
                    f"{partition} episodes after {maximum_attempts} attempts. "
                    "Reduce counts or use a dataset-specific identity split."
                )
            episode_seed = _stable_seed(seed, partition, attempt, "episode")
            dataset_index = _stable_seed(seed, partition, attempt, "index") % len(dataset)
            sample = sample_seeded_episode(dataset, dataset_index, episode_seed)
            identity = extract_episode_identity(sample, data_root)
            if (
                identity["query_id"] in set(identity["support_ids"])
                or len(set(identity["support_ids"])) != len(identity["support_ids"])
                or len(identity["support_ids"]) != shots
            ):
                attempt += 1
                continue
            image_ids = {identity["query_id"], *identity["support_ids"]}
            episode_id = _episode_id(
                dataset_name,
                fold,
                shots,
                source_split,
                identity,
            )
            attempt += 1
            if image_ids & reserved_images or episode_id in partition_episode_ids:
                continue
            entry = {
                "episode_id": episode_id,
                "dataset_index": int(dataset_index),
                "episode_seed": int(episode_seed),
                **identity,
            }
            entries.append(entry)
            partition_episode_ids.add(episode_id)
            partition_images.update(image_ids)
        reserved_images.update(partition_images)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "source_split": source_split,
        "fold": int(fold),
        "shots": int(shots),
        "generation_seed": int(seed),
        "source_git_sha": source_git_sha,
        "partition_counts": normalized_counts,
        "partitions": partitions,
    }
    validate_episode_manifest(
        manifest,
        expected_dataset=dataset_name,
        expected_fold=fold,
        expected_shots=shots,
        expected_source_split=source_split,
    )
    return manifest


def write_episode_manifest(
    manifest: Mapping[str, Any],
    output_path: str | os.PathLike[str],
) -> None:
    validate_episode_manifest(manifest)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_episode_manifest(
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_episode_manifest(manifest)
    return manifest


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeReplayDataset:
    """Replay one manifest partition and reject any identity drift."""

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
            raise ValueError(f"Unknown episode manifest partition: {partition}")
        validate_episode_manifest(
            manifest,
            expected_dataset=expected_dataset,
            expected_fold=expected_fold,
            expected_shots=expected_shots,
            expected_source_split=expected_source_split,
        )
        self.dataset = dataset
        self.manifest = dict(manifest)
        self.partition = partition
        self.entries = list(manifest["partitions"][partition])
        self.data_root = data_root
        for attribute in ("class_ids", "nclass", "benchmark"):
            if hasattr(dataset, attribute):
                setattr(self, attribute, getattr(dataset, attribute))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample = sample_seeded_episode(
            self.dataset,
            int(entry["dataset_index"]),
            int(entry["episode_seed"]),
        )
        if not isinstance(sample, Mapping):
            raise TypeError("Episode datasets must return a mapping.")
        actual = extract_episode_identity(sample, self.data_root)
        expected = {
            "class_id": int(entry["class_id"]),
            "query_id": str(entry["query_id"]),
            "support_ids": [str(value) for value in entry["support_ids"]],
        }
        if actual != expected:
            raise RuntimeError(
                "Episode replay identity drift detected. "
                f"episode_id={entry['episode_id']}; expected={expected}; actual={actual}"
            )
        replayed = dict(sample)
        replayed["episode_id"] = str(entry["episode_id"])
        replayed["episode_partition"] = self.partition
        return replayed
