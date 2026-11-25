"""Reference-conditioned reranking of a single frame's SAM multimask output."""

import torch
from torch import nn
import torch.nn.functional as F


def interior_prompt(mask):
    """Pick a foreground pixel with high local foreground density, near the centroid on ties."""
    foreground = mask[0, 0] > 0
    points = foreground.nonzero()
    if not len(points):
        return None
    coords = points.float()
    density = F.avg_pool2d(mask.float(), 9, stride=1, padding=4)[0, 0]
    values = density[points[:, 0], points[:, 1]]
    distances = (coords - coords.mean(0)).square().sum(1)
    center = coords[distances.masked_fill(values < values.max(), float("inf")).argmin()]
    return {"point_coords": center.flip(0).reshape(1, 1, 2),
            "point_labels": torch.ones(1, 1, dtype=torch.int32, device=mask.device)}


def nesting_level(masks, family):
    """Outer=1, inner=0. NaN denotes lack of a distinct nested proposal."""
    rows = masks.flatten(1).float()
    relatives = family.flatten(1).float()
    area = rows.sum(1)[:, None]
    other = relatives.sum(1)[None, :]
    smaller = torch.minimum(area, other)
    larger = torch.maximum(area, other)
    comparable = ((rows @ relatives.T >= 0.9 * smaller) &
                  (smaller > 0) & (smaller <= 0.8 * larger))
    children = (comparable & (other < area)).sum(1)
    return children / comparable.sum(1)


class ReferenceGranularity(nn.Module):
    """Stateless inference module. No new checkpoint keys or target labels."""

    def __init__(self, mode):
        super().__init__()
        if mode not in ("area", "nesting", "both"):
            raise ValueError(f"Unknown granularity mode: {mode}")
        self.mode = mode

    def describe(self, mask, proposals=None):
        area = (mask > 0).float().mean().sqrt()
        level = area.new_full((), float("nan"))
        if proposals is not None:
            family = proposals[0] > 0
            target = F.interpolate(mask.float(), size=family.shape[-2:], mode="area") > 0.5
            level = nesting_level(target[0], family)[0]
        return torch.stack((area, level))

    def forward(self, logits, quality, references):
        # SANSA decodes one frame at a time. Keep the native winner on ties.
        masks = logits[0] > 0
        native = quality[0].argmax()
        anchor = masks[native]
        area = masks.sum((-2, -1))
        intersection = (masks & anchor).sum((-2, -1))
        eligible = (area > 0) & (intersection >= 0.9 * torch.minimum(area, area[native]))
        eligible = eligible & (area[native] > 0)
        target = torch.stack(references).nanmedian(dim=0).values
        scores = quality[0].float()
        if self.mode in ("area", "both"):
            occupancy = masks.float().mean((-2, -1)).sqrt()
            scores = scores - 0.1 * (occupancy - target[0]).abs()
        if self.mode in ("nesting", "both"):
            level = nesting_level(masks, masks)
            error = (level - target[1]).abs()
            # Compare penalties relative to the native candidate only where both are known.
            scores = scores - 0.1 * (error - error[native]).nan_to_num()
        best = scores.masked_fill(~eligible, -torch.inf).argmax()
        return torch.where(eligible[best] & (scores[best] > scores[native]), best, native).view(1)
