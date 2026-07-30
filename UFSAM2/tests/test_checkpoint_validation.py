import unittest

from util.checkpoint_validation import validate_sansa_base_state


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


class CheckpointValidationTest(unittest.TestCase):
    def setUp(self):
        self.model = {
            "sam.weight": FakeTensor(4, 4),
            "sam.adapter.down.weight": FakeTensor(2, 4),
            "sam.adapter.up.weight": FakeTensor(4, 2),
        }

    def test_accepts_exact_adapter_only_state(self):
        state = {
            "module.sam.adapter.down.weight": FakeTensor(2, 4),
            "module.sam.adapter.up.weight": FakeTensor(4, 2),
        }
        loadable, expected = validate_sansa_base_state(self.model, state)
        self.assertEqual(sorted(loadable), expected)

    def test_rejects_missing_or_contaminated_state(self):
        with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
            validate_sansa_base_state(
                self.model,
                {
                    "sam.adapter.down.weight": FakeTensor(2, 4),
                    "candidate.weight": FakeTensor(1),
                },
            )


if __name__ == "__main__":
    unittest.main()
