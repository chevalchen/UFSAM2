import copy
import json
from pathlib import Path
import random
import unittest

from util.episode_manifest import (
    EpisodeReplayDataset,
    file_sha256,
    load_episode_manifest,
    validate_episode_manifest,
)


class FakeDataset:
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return {
            "query_name": f"query-{random.randrange(100000)}.jpg",
            "support_names": [
                f"support-{random.randrange(100000)}.jpg"
                for _ in range(2)
            ],
            "class_id": random.randrange(4),
            "payload": random.random(),
        }


def fake_manifest(dataset):
    partitions = {}
    for partition_index, partition in enumerate(
        ("train", "calibration", "validation")
    ):
        seed = 100 + partition_index
        state = random.getstate()
        random.seed(seed)
        sample = dataset[0]
        random.setstate(state)
        partitions[partition] = [
            {
                "episode_id": f"episode-{partition}",
                "dataset_index": 0,
                "episode_seed": seed,
                "class_id": sample["class_id"],
                "query_id": sample["query_name"],
                "support_ids": sample["support_names"],
            }
        ]
    return {
        "schema_version": 1,
        "dataset": "fake",
        "source_split": "train",
        "fold": 0,
        "shots": 2,
        "generation_seed": 0,
        "partition_counts": {
            "train": 1,
            "calibration": 1,
            "validation": 1,
        },
        "partitions": partitions,
    }


class Exp003ManifestTest(unittest.TestCase):
    def test_tracked_manifest_exact_hash_and_structure(self):
        repository_root = Path(__file__).resolve().parents[2]
        path = (
            repository_root
            / "experiments"
            / "EXP-003"
            / "episodes"
            / "coco_fold0_5shot_train1200_cal600_val600_seed0.json"
        )
        self.assertEqual(
            file_sha256(path),
            "2ff8181ad80bbe11b4da4eef48d90ceb90aa3a02af06f70bb051dda18ab1a8b6",
        )
        manifest = load_episode_manifest(path)
        self.assertEqual(
            {name: len(entries) for name, entries in manifest["partitions"].items()},
            {"train": 1200, "calibration": 600, "validation": 600},
        )

    def test_replay_is_exact_and_restores_python_rng(self):
        dataset = FakeDataset()
        manifest = fake_manifest(dataset)
        replay = EpisodeReplayDataset(
            dataset,
            manifest,
            "validation",
            expected_dataset="fake",
            expected_fold=0,
            expected_shots=2,
            expected_source_split="train",
        )
        rng_state = random.getstate()
        first = replay[0]
        self.assertEqual(random.getstate(), rng_state)
        second = replay[0]
        self.assertEqual(first, second)
        self.assertEqual(first["episode_id"], "episode-validation")

    def test_replay_rejects_identity_drift(self):
        dataset = FakeDataset()
        manifest = fake_manifest(dataset)
        tampered = copy.deepcopy(manifest)
        tampered["partitions"]["validation"][0]["query_id"] = "wrong.jpg"
        replay = EpisodeReplayDataset(dataset, tampered, "validation")
        with self.assertRaisesRegex(RuntimeError, "identity drift"):
            replay[0]

    def test_validation_rejects_cross_partition_overlap(self):
        dataset = FakeDataset()
        manifest = fake_manifest(dataset)
        manifest["partitions"]["train"][0]["query_id"] = (
            manifest["partitions"]["validation"][0]["query_id"]
        )
        with self.assertRaisesRegex(ValueError, "share identities"):
            validate_episode_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
