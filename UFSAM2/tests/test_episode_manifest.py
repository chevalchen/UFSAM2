import copy
import random
import tempfile
import unittest

from util.episode_manifest import (
    EpisodeReplayDataset,
    file_sha256,
    generate_episode_manifest,
    load_episode_manifest,
    validate_episode_manifest,
    write_episode_manifest,
)


class FakeEpisodeDataset:
    class_ids = list(range(8))
    nclass = 8
    benchmark = "fake"

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        query = f"images/query-{index}-{random.randrange(1_000_000)}.jpg"
        supports = [
            f"images/support-{random.randrange(1_000_000)}.jpg"
            for _ in range(2)
        ]
        return {
            "query_name": query,
            "support_names": supports,
            "class_id": random.randrange(8),
            "payload": random.random(),
        }


class EpisodeManifestTest(unittest.TestCase):
    def setUp(self):
        self.dataset = FakeEpisodeDataset()
        self.manifest = generate_episode_manifest(
            self.dataset,
            dataset_name="fake",
            source_split="train",
            fold=0,
            shots=2,
            counts={"train": 8, "calibration": 4, "validation": 4},
            seed=17,
            data_root=None,
            source_git_sha="deadbeef",
        )

    def test_generation_is_deterministic_and_disjoint(self):
        repeated = generate_episode_manifest(
            self.dataset,
            dataset_name="fake",
            source_split="train",
            fold=0,
            shots=2,
            counts={"train": 8, "calibration": 4, "validation": 4},
            seed=17,
            data_root=None,
            source_git_sha="deadbeef",
        )
        self.assertEqual(self.manifest, repeated)
        validate_episode_manifest(self.manifest)

    def test_replay_restores_exact_identity_and_rng_state(self):
        replay = EpisodeReplayDataset(
            self.dataset,
            self.manifest,
            "validation",
            expected_dataset="fake",
            expected_fold=0,
            expected_shots=2,
            expected_source_split="train",
        )
        state = random.getstate()
        first = replay[0]
        self.assertEqual(state, random.getstate())
        second = replay[0]
        self.assertEqual(first, second)
        self.assertEqual(first["episode_partition"], "validation")
        self.assertEqual(first["episode_id"], self.manifest["partitions"]["validation"][0]["episode_id"])

    def test_replay_rejects_identity_drift(self):
        tampered = copy.deepcopy(self.manifest)
        tampered["partitions"]["validation"][0]["query_id"] = "wrong.jpg"
        replay = EpisodeReplayDataset(self.dataset, tampered, "validation")
        with self.assertRaisesRegex(RuntimeError, "identity drift"):
            replay[0]

    def test_json_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/episodes.json"
            write_episode_manifest(self.manifest, path)
            self.assertEqual(load_episode_manifest(path), self.manifest)
            self.assertEqual(len(file_sha256(path)), 64)

    def test_validation_rejects_cross_partition_overlap(self):
        tampered = copy.deepcopy(self.manifest)
        tampered["partitions"]["train"][0]["query_id"] = tampered["partitions"]["validation"][0]["query_id"]
        with self.assertRaisesRegex(ValueError, "share image identities"):
            validate_episode_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
