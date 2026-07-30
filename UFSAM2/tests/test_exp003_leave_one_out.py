import hashlib
import unittest

from util.exp003_leave_one_out import (
    BinaryCounts,
    PRIMITIVE_ROWS,
    PrimitiveEpisodeRecord,
    compact_support_entries,
    evaluate_frozen_rows,
    random_drop_slot,
    selected_rows,
    stable_argmax,
    stable_argmin,
)


def counts(foreground_intersection, foreground_union, pixels=100):
    foreground_prediction = foreground_intersection
    background_intersection = pixels - foreground_union
    background_union = pixels - foreground_prediction
    return BinaryCounts(
        background_intersection=background_intersection,
        foreground_intersection=foreground_intersection,
        background_union=background_union,
        foreground_union=foreground_union,
    )


def record(index=0, *, b0=(50, 100), drops=None, similarities=None):
    drops = drops or [(60, 100), (55, 100), (50, 100), (45, 100), (40, 100)]
    primitive_counts = {"B0": counts(*b0)}
    primitive_counts.update(
        {
            f"D{slot}": counts(*value)
            for slot, value in enumerate(drops)
        }
    )
    return PrimitiveEpisodeRecord(
        episode_id=f"episode-{index}",
        class_id=index % 2,
        primitive_counts=primitive_counts,
        similarity_scores=tuple(
            similarities or (0.1, 0.2, 0.3, 0.4, 0.5)
        ),
        primitive_fingerprints={
            name: hashlib.sha256(f"{index}:{name}".encode()).hexdigest()
            for name in PRIMITIVE_ROWS
        },
        six_decode_latency_ms=10.0 + index,
        peak_device_memory_bytes=1000 + index,
    )


class Exp003LeaveOneOutTest(unittest.TestCase):
    def test_compact_memory_bank_preserves_order_and_object_identity(self):
        entries = [object() for _ in range(5)]
        full = compact_support_entries(entries, drop_slot=None)
        self.assertEqual(list(full), [0, 1, 2, 3, 4])
        self.assertTrue(all(full[index] is entries[index] for index in range(5)))

        dropped = compact_support_entries(entries, drop_slot=2)
        self.assertEqual(list(dropped), [0, 1, 2, 3])
        self.assertIs(dropped[0], entries[0])
        self.assertIs(dropped[1], entries[1])
        self.assertIs(dropped[2], entries[3])
        self.assertIs(dropped[3], entries[4])

    def test_random_control_uses_frozen_sha256_serialization(self):
        episode_id = "67d0c7f599c2dcd102b61316"
        for seed in range(5):
            expected = int.from_bytes(
                hashlib.sha256(f"{episode_id}:{seed}".encode("utf-8")).digest(),
                "big",
            ) % 5
            self.assertEqual(random_drop_slot(episode_id, seed), expected)

    def test_similarity_and_oracle_ties_choose_lowest_slot(self):
        self.assertEqual(stable_argmin((0.1, 0.1, 0.2, 0.3, 0.4)), 0)
        self.assertEqual(stable_argmax((0.4, 0.3, 0.4, 0.2, 0.1)), 0)
        tied = record(
            drops=[(60, 100), (60, 100), (50, 100), (40, 100), (30, 100)],
            similarities=(0.1, 0.1, 0.2, 0.3, 0.4),
        )
        rows = selected_rows(tied)
        self.assertEqual(rows["B2S"], "D0")
        self.assertEqual(rows["B8_forced"], "D0")
        self.assertEqual(rows["B8_positive"], "D0")

    def test_safe_oracle_requires_strict_improvement(self):
        no_gain = record(
            b0=(60, 100),
            drops=[(60, 100), (50, 100), (40, 100), (30, 100), (20, 100)],
        )
        rows = selected_rows(no_gain)
        self.assertEqual(rows["B8_forced"], "D0")
        self.assertEqual(rows["B8_positive"], "B0")

    def test_metrics_emit_all_frozen_rows_and_joint_stage_gate(self):
        records = [record(index) for index in range(10)]
        metrics = evaluate_frozen_rows(
            records,
            bootstrap_resamples=200,
            bootstrap_seed=0,
        )
        expected_rows = {
            "B0",
            "B1R_s0",
            "B1R_s1",
            "B1R_s2",
            "B1R_s3",
            "B1R_s4",
            "B1R_mean",
            "B2S",
            "B7I",
            "B8_forced",
            "B8_positive",
        }
        self.assertEqual(set(metrics["rows"]), expected_rows)
        self.assertAlmostEqual(
            metrics["effects"]["B8_positive"]["aggregate_miou_effect"],
            10.0,
        )
        self.assertGreater(
            metrics["effects"]["B8_positive"]["paired_episode_effect"]["ci95"][0],
            0.0,
        )
        self.assertTrue(metrics["guardrails"]["all_pass"])
        self.assertEqual(
            metrics["automatic_stage_suggestion"],
            "PASS_TO_STAGE_B",
        )
        self.assertIn(
            "B2S_minus_B1R_mean",
            metrics["deployable_supporting_effects"],
        )


if __name__ == "__main__":
    unittest.main()
