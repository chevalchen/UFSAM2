import unittest

from util.checkpoint_validation import validate_sansa_base_state


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


class CheckpointValidationTest(unittest.TestCase):
    def setUp(self):
        self.model_state = {
            "sam.encoder.weight": FakeTensor(4, 4),
            "sam.encoder.adapter.down.weight": FakeTensor(2, 4),
            "sam.encoder.adapter.up.weight": FakeTensor(4, 2),
            "mask_post_refiner.refine.0.weight": FakeTensor(16, 33, 3, 3),
        }

    def test_accepts_adapter_only_checkpoint_and_module_prefix(self):
        checkpoint = {
            "module.sam.encoder.adapter.down.weight": FakeTensor(2, 4),
            "module.sam.encoder.adapter.up.weight": FakeTensor(4, 2),
        }
        loadable, adapter_keys = validate_sansa_base_state(
            self.model_state,
            checkpoint,
        )
        self.assertEqual(
            sorted(loadable),
            [
                "sam.encoder.adapter.down.weight",
                "sam.encoder.adapter.up.weight",
            ],
        )
        self.assertEqual(sorted(loadable), adapter_keys)

    def test_rejects_missing_adapter_key(self):
        checkpoint = {
            "sam.encoder.adapter.down.weight": FakeTensor(2, 4),
        }
        with self.assertRaisesRegex(RuntimeError, "Missing configured adapter keys"):
            validate_sansa_base_state(self.model_state, checkpoint)

    def test_rejects_shape_mismatch(self):
        checkpoint = {
            "sam.encoder.adapter.down.weight": FakeTensor(3, 4),
            "sam.encoder.adapter.up.weight": FakeTensor(4, 2),
        }
        with self.assertRaisesRegex(RuntimeError, "shape mismatches"):
            validate_sansa_base_state(self.model_state, checkpoint)

    def test_rejects_candidate_module_contamination(self):
        checkpoint = {
            "sam.encoder.adapter.down.weight": FakeTensor(2, 4),
            "sam.encoder.adapter.up.weight": FakeTensor(4, 2),
            "mask_post_refiner.refine.0.weight": FakeTensor(16, 33, 3, 3),
        }
        with self.assertRaisesRegex(RuntimeError, "non-adapter keys"):
            validate_sansa_base_state(self.model_state, checkpoint)


if __name__ == "__main__":
    unittest.main()
