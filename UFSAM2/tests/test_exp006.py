"""Coordinate and causal-interface checks for the gradient point generator."""
import inspect
import unittest
from types import SimpleNamespace
import torch
from models.sansa.gradient_prompting import GradientPrompting


class PromptGeometryTests(unittest.TestCase):
    def setUp(self):
        self.module = GradientPrompting.__new__(GradientPrompting)
        self.module.sansa = SimpleNamespace(sam=SimpleNamespace(image_size=1024))
        self.module.config = dict(positive_threshold=.65, negative_threshold=.35,
                                  erode_kernel=9, dedup_radius_lowres=8,
                                  anchor_grid=16, anchors_per_domain=32)

    def test_xy_order_and_non_square_cell_centers(self):
        plane = torch.zeros(128, 256)
        points = self.module.pack([(17, 93), (104, 212)], [1, 0], plane)
        self.assertEqual(points['point_coords'][0, 0].tolist(), [374., 140.])
        self.assertEqual(self.module.cells(points, plane.shape), [(17, 93), (104, 212)])
        # When displayed at 640 x 640, the same physical position remains (x,y).
        self.assertEqual(self.module.cells(points, (640, 640))[0], (87, 233))

    def test_gradient_targets_only_temporary_feature(self):
        feature = torch.randn(1, 2, 16, 16, requires_grad=True)
        # Opposite signs in independent spatial strata prevent cancellation.
        logits = torch.cat([10+feature[:, :1, :, :8], -10+feature[:, :1, :, 8:]], dim=-1)
        response, stats = self.module.gradient(feature, SimpleNamespace(low_res_masks=logits))
        self.assertTrue(stats['gradient_valid'])
        self.assertGreater(stats['gradient_l2'], 0)
        self.assertTrue(torch.isfinite(response).all())
        self.assertIsNone(feature.grad)  # autograd.grad does not accumulate state.

    def test_no_query_gt_in_generation_interface(self):
        for name in ('prepare', 'decode', 'points', 'gradient', 'semantic_points'):
            self.assertNotIn('target', inspect.signature(getattr(GradientPrompting, name)).parameters)


if __name__ == '__main__':
    unittest.main()
