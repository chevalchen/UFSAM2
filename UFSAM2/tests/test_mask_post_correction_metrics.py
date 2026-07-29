import unittest

from util.mask_post_correction_metrics import (
    BinaryCounts,
    EpisodeMaskRecord,
    evaluate_frozen_rows,
    frozen_row_indices,
    paired_bootstrap,
    record_from_json,
    record_to_json,
)


def counts(iou_percent):
    return BinaryCounts(
        background_intersection=90.0,
        foreground_intersection=float(iou_percent),
        background_union=100.0,
        foreground_union=100.0,
    )


class MaskPostCorrectionMetricsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            EpisodeMaskRecord("e0", 0, counts(40), counts(70), 0.10),
            EpisodeMaskRecord("e1", 0, counts(50), counts(60), 0.20),
            EpisodeMaskRecord("e2", 1, counts(60), counts(50), 0.30),
            EpisodeMaskRecord("e3", 1, counts(70), counts(70), 0.40),
        ]

    def test_frozen_oracle_rows_use_stable_true_gain_order(self):
        rows = frozen_row_indices(self.records)
        self.assertEqual(rows["B8_r25"], {0})
        self.assertEqual(rows["B8_r50"], {0, 1})
        self.assertEqual(rows["B8_positive"], {0, 1})
        self.assertEqual(rows["B1"], {0, 1, 2, 3})

    def test_evaluation_reports_all_rows_and_non_adjudicating_gate(self):
        result = evaluate_frozen_rows(
            self.records,
            bootstrap_resamples=200,
            bootstrap_seed=3,
        )
        self.assertEqual(result["episode_count"], 4)
        self.assertEqual(result["unique_episode_ids"], 4)
        self.assertEqual(
            set(result["rows"]),
            {"B0", "B1", "B8_r25", "B8_r50", "B8_r75", "B8_positive"},
        )
        self.assertEqual(result["automatic_stage_suggestion"], "PASS_TO_STAGE_B")
        self.assertIn("Automatic suggestion only", result["note"])

    def test_bootstrap_is_deterministic(self):
        first = paired_bootstrap([1.0, -1.0, 2.0], resamples=100, seed=7)
        second = paired_bootstrap([1.0, -1.0, 2.0], resamples=100, seed=7)
        self.assertEqual(first, second)

    def test_record_json_round_trip(self):
        record = self.records[0]
        self.assertEqual(record_from_json(record_to_json(record)), record)


if __name__ == "__main__":
    unittest.main()
