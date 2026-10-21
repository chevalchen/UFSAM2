"""Pixel-count metrics and preregistered three-seed paired screen."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def segmentation_loss(logits, target):
    valid = target != 255
    if not valid.any() or not torch.isfinite(logits).all():
        raise ValueError('Empty target or non-finite logits')
    z, y = logits[valid], target[valid].float()
    if not ((y == 0) | (y == 1)).all():
        raise ValueError('Non-binary training target')
    p = z.sigmoid()
    return F.binary_cross_entropy_with_logits(z, y) + 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)


def pixel_counts(logits, target):
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite prediction')
    valid = target != 255
    pred, gt = logits.sigmoid() > 0.5, target == 1
    return [int(((pred == c) & (gt == c) & valid).sum()) for c in (0, 1)] + [
        int((((pred == c) | (gt == c)) & valid).sum()) for c in (0, 1)]


def episode_iou(counts):
    return counts[1] / max(counts[3], 1)


def aggregate(records):
    if not records:
        raise ValueError('Empty evaluation')
    by_class = {}
    total = np.zeros(4, dtype=np.float64)
    fg = []
    for e in records:
        counts = np.asarray(e['counts'], dtype=np.float64)
        if counts.shape != (4,) or not np.isfinite(counts).all() or (counts < 0).any() or (counts[:2] > counts[2:]).any():
            raise ValueError('Invalid pixel counts')
        by_class.setdefault(e['class_id'], np.zeros(4, dtype=np.float64))
        by_class[e['class_id']] += counts
        total += counts
        fg.append(episode_iou(counts))
    return dict(class_miou=100 * float(np.mean([episode_iou(v) for v in by_class.values()])),
                fb_iou=50 * float(np.sum(total[:2] / np.maximum(total[2:], 1))),
                worst_decile=100 * float(np.mean(sorted(fg)[:math.ceil(len(fg) * 0.1)])),
                mean_episode_fg_iou=100 * float(np.mean(fg)))


def bootstrap(effect, iterations=10000):
    effect = np.asarray(effect, dtype=np.float64)
    rng = np.random.default_rng(0)
    means = np.empty(iterations)
    for start in range(0, iterations, 100):
        n = min(100, iterations - start)
        means[start:start+n] = effect[rng.integers(0, len(effect), size=(n, len(effect)))].mean(1)
    return (100 * np.percentile(means, [2.5, 97.5])).tolist()


def adjudicate(rows):
    """rows maps C0 and C{1,2,3,4,8}/seed to identically ordered counts."""
    required = {'C0'} | {f'{r}/{s}' for r in ('C1', 'C2', 'C3', 'C4', 'C8') for s in range(3)}
    if set(rows) != required:
        raise ValueError('Missing or unexpected causal rows')
    identity = [(e['episode_id'], e['class_id']) for e in rows['C0']]
    if len(identity) != 600 or len(set(x[0] for x in identity)) != 600:
        raise ValueError('Validation must contain 600 distinct episodes')
    for records in rows.values():
        if [(e['episode_id'], e['class_id']) for e in records] != identity:
            raise ValueError('Unmatched paired episodes')
    summary = {key: aggregate(records) for key, records in rows.items()}
    comparisons, passes = {}, []
    for control in ('C1', 'C2', 'C0'):
        effects, gains, guards = [], [], []
        for seed in range(3):
            a, b = f'C3/{seed}', f'{control}/{seed}' if control != 'C0' else 'C0'
            effects.append([episode_iou(x['counts']) - episode_iou(y['counts']) for x, y in zip(rows[a], rows[b])])
            gains.append(summary[a]['class_miou'] - summary[b]['class_miou'])
            guards.append(dict(seed=seed, fb_delta_pp=summary[a]['fb_iou'] - summary[b]['fb_iou'],
                               worst_delta_pp=summary[a]['worst_decile'] - summary[b]['worst_decile']))
        ci = bootstrap(np.mean(effects, axis=0))
        passed = np.mean(gains) >= 0.5 and ci[0] > 0 and all(g['fb_delta_pp'] >= -0.2 and g['worst_delta_pp'] >= -0.5 for g in guards)
        passes.append(bool(passed))
        comparisons['C3-' + control] = dict(mean_class_miou_delta_pp=float(np.mean(gains)),
             seed_delta_pp=gains, seed_delta_sd_pp=float(np.std(gains, ddof=1)),
             episode_effect_ci95_pp=ci, guards=guards, passed=bool(passed))
    seed_summary = {r: {metric: dict(mean=float(np.mean([summary[f'{r}/{s}'][metric] for s in range(3)])),
                                         sd=float(np.std([summary[f'{r}/{s}'][metric] for s in range(3)], ddof=1)))
                        for metric in summary['C0']} for r in ('C1', 'C2', 'C3', 'C4', 'C8')}
    return dict(decision='NEXT_DESIGN_ELIGIBLE_NOT_KEEP' if all(passes) else 'DROP',
                rows=summary, seed_summary=seed_summary, comparisons=comparisons,
                ci_scope='episode sampling only; neither aggregate-class-mIoU nor training-seed CI',
                oracle_label='C8 uses query GT, diagnostic only')
