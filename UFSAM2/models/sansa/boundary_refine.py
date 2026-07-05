import torch
import torch.nn as nn
from torch import Tensor


class BoundaryRefinementModule(nn.Module):
    """
    Lightweight residual correction for uncertain boundary pixels.

    The branch refines SAM2 low-resolution mask logits with the highest-resolution
    decoder feature. Only pixels with sigmoid(logit) in (0.4, 0.6) receive a
    residual, so confident foreground/background regions stay unchanged.
    """

    def __init__(self, feat_channels: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(feat_channels + 1, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, logits: Tensor, feat: Tensor) -> Tensor:
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            boundary_mask = ((prob > 0.4) & (prob < 0.6)).float()

        x = torch.cat([logits, feat], dim=1)
        offset = self.refine(x)
        return logits + boundary_mask * offset
