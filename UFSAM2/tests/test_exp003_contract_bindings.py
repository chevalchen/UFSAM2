import json
from pathlib import Path
import unittest

from util.episode_manifest import file_sha256


class Exp003ContractBindingsTest(unittest.TestCase):
    def test_execution_packet_binds_exact_frozen_inputs(self):
        root = Path(__file__).resolve().parents[2]
        config_dir = root / "experiments" / "EXP-003" / "configs"
        resolved_path = config_dir / "stage_a_leave_one_out_contract_v2.json"
        original_path = config_dir / "stage_a_leave_one_out_contract.json"
        execution_path = config_dir / "stage_a_leave_one_out_execution.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        self.assertEqual(
            file_sha256(resolved_path),
            "462f410e46fae80ccc4f54d4b70b0fdc6f1fc272abefd40c9302629b57696a2d",
        )
        self.assertEqual(
            file_sha256(original_path),
            "fb244a41919c1c279b8ecd8e0ca473f57968a4172bf701357145e6af9f969cdf",
        )
        self.assertEqual(
            execution["research_contract"]["sha256"],
            file_sha256(resolved_path),
        )
        self.assertEqual(
            execution["research_contract"]["inherited_contract_sha256"],
            file_sha256(original_path),
        )
        self.assertTrue(execution["execution_ready"])
        self.assertFalse(execution["formal_execution_authorized"])
        self.assertEqual(
            execution["formal_git_sha_binding"]["status"],
            "PENDING_HOST_IDEA_DECISION_SIGNATURE",
        )


if __name__ == "__main__":
    unittest.main()
