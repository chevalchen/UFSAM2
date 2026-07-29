import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "EXP-002"
    / "configs"
    / "stage_a_mask_refiner_contract.json"
)
CONTRACT_SHA256 = "d4c2b5b99b6f312cb0c8cc8345cd55bc6940416b3162598ab653a2f77e7040b2"
EXECUTION_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "EXP-002"
    / "configs"
    / "stage_a_mask_refiner_execution.json"
)


class Exp002ContractBindingTest(unittest.TestCase):
    def test_frozen_contract_hash_and_core_fields(self):
        content = CONTRACT_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), CONTRACT_SHA256)
        contract = json.loads(content)
        self.assertEqual(contract["experiment_id"], "EXP-002")
        self.assertEqual(contract["stage"]["id"], "operator_headroom")
        self.assertFalse(contract["execution_ready"])
        self.assertEqual(
            contract["baseline"]["accepted_parent_git_sha"],
            "000d4507f16028871c58a4f17a59a038beb073d2",
        )
        self.assertEqual(contract["training"]["epochs"], 10)
        self.assertEqual(contract["training"]["max_optimizer_steps"], 12000)
        self.assertEqual(
            contract["training"]["formal_checkpoint"],
            "final optimizer step at epoch 10",
        )
        self.assertEqual(
            [row["row"] for row in contract["causal_rows"]],
            ["B0", "B1", "B8_r25", "B8_r50", "B8_r75", "B8_positive"],
        )
        self.assertTrue(all(contract["disabled_interventions"].values()))

    def test_execution_input_binds_contract_and_frozen_paths(self):
        execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
        self.assertTrue(execution["execution_ready"])
        self.assertEqual(execution["experiment_id"], "EXP-002")
        self.assertEqual(
            execution["research_contract"]["sha256"],
            CONTRACT_SHA256,
        )
        self.assertEqual(
            execution["training"]["formal_checkpoint"],
            "mask_post_refiner_epoch10_final.pth",
        )
        self.assertTrue(execution["evaluation"]["verify_exact_replay"])
        self.assertEqual(
            execution["evaluation"]["rows"],
            ["B0", "B1", "B8_r25", "B8_r50", "B8_r75", "B8_positive"],
        )
        self.assertTrue(all(execution["disabled_interventions"].values()))


if __name__ == "__main__":
    unittest.main()
