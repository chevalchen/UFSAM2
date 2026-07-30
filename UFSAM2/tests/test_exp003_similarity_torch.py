import unittest

try:
    import torch
    from models.sansa.sansa import SANSA
except ImportError:
    torch = None
    SANSA = None


@unittest.skipIf(torch is None, "Torch runtime is server-only in this host checkout.")
class Exp003SimilarityTorchTest(unittest.TestCase):
    def test_masked_support_pool_and_query_global_pool(self):
        class FakeBackbone:
            def __init__(self):
                self.features = [
                    torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
                    torch.tensor([[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
                    torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
                    torch.tensor([[[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
                    torch.ones(1, 2, 2, 2),
                    torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
                ]

            def get_current_feats_x16(self, index):
                return self.features[index]

        masks = torch.ones(5, 1, 2, 2)
        scores = SANSA._support_query_similarity(
            object(),
            FakeBackbone(),
            masks,
            query_idx=5,
        )
        self.assertEqual(tuple(scores.shape), (5,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertAlmostEqual(float(scores[-1]), 2 ** -0.5, places=6)


if __name__ == "__main__":
    unittest.main()
