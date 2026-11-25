import unittest

import torch

from models.sansa.granularity import ReferenceGranularity, interior_prompt, nesting_level


class GranularityTest(unittest.TestCase):
    def setUp(self):
        self.masks = torch.zeros(3, 12, 12, dtype=torch.bool)
        for i, (start, stop) in enumerate([(5, 7), (3, 9), (1, 11)]):
            self.masks[i, start:stop, start:stop] = True
        self.logits = self.masks[None].float() * 2 - 1
        self.quality = torch.tensor([[0.80, 0.81, 0.82]])

    def test_reference_changes_choice_at_fixed_quality(self):
        for mode in ("area", "nesting", "both"):
            selector = ReferenceGranularity(mode)
            for index in (0, 2):
                ref = selector.describe(self.masks[index:index+1, None], self.logits)
                self.assertEqual(selector(self.logits, self.quality, [ref]).item(), index)
            self.assertFalse(selector.state_dict())

    def test_unknown_hierarchy_and_empty_anchor_preserve_native_choice(self):
        selector = ReferenceGranularity("nesting")
        ref = selector.describe(self.masks[:1, None])
        self.assertEqual(selector(self.logits, self.quality, [ref]).item(), 2)
        self.assertEqual(selector(-self.logits.abs(), self.quality, [ref]).item(), 2)
        self.assertIsNone(interior_prompt(torch.zeros_like(self.masks[:1, None])))

    def test_unrelated_region_is_not_a_different_granularity(self):
        masks = torch.zeros_like(self.masks)
        masks[0, :3, :3] = True
        masks[1, 5:11, 5:11] = True
        masks[2] = masks[1]
        self.assertTrue(nesting_level(masks, masks).isnan().all())
        selector = ReferenceGranularity("area")
        ref = selector.describe(masks[:1, None])
        self.assertEqual(selector(masks[None].float()*2-1, self.quality, [ref]).item(), 2)

    def test_visual_prompt_lands_inside_nonconvex_target(self):
        mask = self.masks[2:3, None] & ~self.masks[1:2, None]
        x, y = interior_prompt(mask)["point_coords"].long().flatten()
        self.assertTrue(mask[0, 0, y, x])
