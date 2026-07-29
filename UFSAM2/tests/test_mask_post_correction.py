import unittest

import torch

from models.sansa.mask_post_correction import (
    BoundaryAwareResidualLogitRefiner,
    load_mask_post_refiner_state,
    mask_post_refiner_state_dict,
)
from models.sansa.model_utils import DecoderOutput
from models.sansa.sansa import SANSA


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_post_refiner = BoundaryAwareResidualLogitRefiner()
        self.other = torch.nn.Conv2d(1, 1, 1)


class FakeSam:
    image_size = 8


class MaskPostCorrectionTest(unittest.TestCase):
    def test_zero_initialized_operator_is_exact_identity(self):
        module = BoundaryAwareResidualLogitRefiner()
        logits = torch.randn(2, 1, 8, 8)
        feature = torch.randn(2, 32, 8, 8)
        output = module(logits, feature)
        self.assertTrue(torch.equal(output.logits, logits))
        self.assertEqual(output.support_mask.requires_grad, False)

    def test_residual_is_confined_to_frozen_probability_support(self):
        module = BoundaryAwareResidualLogitRefiner()
        torch.nn.init.zeros_(module.refine[-1].weight)
        torch.nn.init.ones_(module.refine[-1].bias)
        logits = torch.tensor([[[[0.0, 10.0, -10.0]]]])
        feature = torch.zeros(1, 32, 1, 3)
        output = module(logits, feature)
        expected_support = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        self.assertTrue(torch.equal(output.support_mask, expected_support))
        self.assertTrue(
            torch.equal(output.logits - logits, expected_support)
        )

    def test_frozen_shape_and_hyperparameter_contract_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "feature_channels=32"):
            BoundaryAwareResidualLogitRefiner(feature_channels=16)
        with self.assertRaisesRegex(ValueError, "hidden_channels=16"):
            BoundaryAwareResidualLogitRefiner(hidden_channels=32)
        module = BoundaryAwareResidualLogitRefiner()
        with self.assertRaisesRegex(ValueError, "Decoder feature"):
            module(torch.zeros(1, 1, 4, 4), torch.zeros(1, 16, 4, 4))

    def test_refiner_checkpoint_helpers_are_strict(self):
        source = Wrapper()
        state = mask_post_refiner_state_dict(source)
        target = Wrapper()
        loaded = load_mask_post_refiner_state(target, state)
        self.assertEqual(sorted(state), loaded)
        contaminated = dict(state)
        contaminated["other.weight"] = source.other.weight.detach()
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_mask_post_refiner_state(target, contaminated)

    def test_sansa_integration_uses_query_decoder_feature(self):
        model = SANSA(
            FakeSam(),
            torch.device("cpu"),
            mask_post_correction=True,
        )
        torch.nn.init.zeros_(model.mask_post_refiner.refine[-1].weight)
        torch.nn.init.ones_(model.mask_post_refiner.refine[-1].bias)
        decoder_out = DecoderOutput(
            low_res_masks=torch.zeros(1, 1, 4, 4),
            high_res_masks=torch.zeros(1, 1, 8, 8),
        )
        decoder_out._mask_post_feature = torch.zeros(1, 32, 4, 4)
        corrected, support_fraction = model._apply_mask_post_correction(decoder_out)
        self.assertTrue(torch.equal(corrected.low_res_masks, torch.ones(1, 1, 4, 4)))
        self.assertEqual(tuple(corrected.high_res_masks.shape), (1, 1, 8, 8))
        self.assertEqual(float(support_fraction), 1.0)


if __name__ == "__main__":
    unittest.main()
