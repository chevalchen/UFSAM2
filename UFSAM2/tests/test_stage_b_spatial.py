import hashlib
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch

from util.stage_b_spatial import (
    choose_calibration_budget,
    dense_average_precision,
    deterministic_random_score,
    hard_top_area_gate,
    paired_bootstrap_interval,
    selected_cell_count,
)


class StageBSpatialUtilityTest(unittest.TestCase):
    def test_execution_config_binds_frozen_contract_and_rows(self):
        repo = Path(__file__).resolve().parents[2]
        config_path = (
            repo
            / "experiments"
            / "EXP-001"
            / "configs"
            / "stage_b_spatial_execution.json"
        )
        contract_path = (
            repo
            / "experiments"
            / "EXP-001"
            / "configs"
            / "stage_b_spatial_contract.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        self.assertEqual(
            config["research_contract"]["sha256"],
            contract_sha,
        )
        self.assertEqual(
            config["evaluation"]["area_budgets"],
            [0.1, 0.25, 0.5, 1.0],
        )
        self.assertEqual(
            config["evaluation"]["matched_rows"],
            ["B0", "B1", "B2", "B3", "B4E", "B4R", "B7S", "B8S"],
        )
        runs = {run["run_id"]: run for run in config["runs"]}
        self.assertEqual(
            runs["stage_b_b2_benefit_seed0"]["argv"][
                runs["stage_b_b2_benefit_seed0"]["argv"].index(
                    "--pmc_spatial_target"
                )
                + 1
            ],
            "benefit",
        )
        self.assertEqual(
            runs["stage_b_b4r_risk_seed0"]["argv"][
                runs["stage_b_b4r_risk_seed0"]["argv"].index(
                    "--pmc_spatial_target"
                )
                + 1
            ],
            "risk",
        )

    def test_hard_gate_has_exact_area_and_stable_ties(self):
        score = torch.zeros(1, 1, 2, 4)
        gate = hard_top_area_gate(score, 0.25)
        self.assertEqual(int(gate.sum().item()), 2)
        self.assertTrue(
            torch.equal(
                gate.flatten(),
                torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )

    def test_hard_gate_selects_largest_or_smallest(self):
        score = torch.tensor([[[[3.0, 1.0], [4.0, 2.0]]]])
        largest = hard_top_area_gate(score, 0.5)
        smallest = hard_top_area_gate(score, 0.5, largest=False)
        self.assertTrue(
            torch.equal(largest.flatten(), torch.tensor([1.0, 0.0, 1.0, 0.0]))
        )
        self.assertTrue(
            torch.equal(smallest.flatten(), torch.tensor([0.0, 1.0, 0.0, 1.0]))
        )

    def test_full_area_is_exactly_all_ones(self):
        score = torch.randn(2, 1, 3, 5)
        gate = hard_top_area_gate(score, 1.0)
        self.assertTrue(torch.equal(gate, torch.ones_like(score)))
        self.assertEqual(selected_cell_count(1.0, 3, 5), 15)

    def test_random_score_is_episode_keyed_and_rng_neutral(self):
        reference = torch.zeros(1, 1, 4, 4)
        python_state = random.getstate()
        first = deterministic_random_score(
            reference,
            episode_id="episode-1",
            seed=0,
        )
        self.assertEqual(python_state, random.getstate())
        replay = deterministic_random_score(
            reference,
            episode_id="episode-1",
            seed=0,
        )
        other = deterministic_random_score(
            reference,
            episode_id="episode-1",
            seed=1,
        )
        self.assertTrue(torch.equal(first, replay))
        self.assertFalse(torch.equal(first, other))

    def test_budget_selection_uses_smallest_near_best(self):
        selected = choose_calibration_budget(
            {
                0.1: 71.85,
                0.25: 72.00,
                0.5: 72.04,
                1.0: 72.03,
            },
            tolerance_points=0.2,
        )
        self.assertEqual(selected, 0.1)

    def test_paired_bootstrap_is_deterministic(self):
        treatment = [2.0, 3.0, 4.0, 5.0]
        control = [1.0, 2.0, 3.0, 4.0]
        first = paired_bootstrap_interval(
            treatment,
            control,
            resamples=500,
            seed=7,
        )
        second = paired_bootstrap_interval(
            treatment,
            control,
            resamples=500,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["point_mean"], 1.0)
        self.assertGreater(first["lower"], 0.0)

    def test_dense_average_precision(self):
        score_chunks = [np.asarray([0.9, 0.8]), np.asarray([0.2, 0.1])]
        label_chunks = [np.asarray([1, 0]), np.asarray([1, 0])]
        self.assertAlmostEqual(
            dense_average_precision(score_chunks, label_chunks),
            (1.0 + 2.0 / 3.0) / 2.0,
        )


if __name__ == "__main__":
    unittest.main()
