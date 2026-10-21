import unittest

import torch
from evaluate_exp003_leave_one_out import configure_evaluation_precision


class Exp003PrecisionTest(unittest.TestCase):
    def test_overrides_inherited_tf32_for_both_cuda_backends(self):
        matmul = torch.backends.cuda.matmul.allow_tf32
        cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            configure_evaluation_precision()
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
