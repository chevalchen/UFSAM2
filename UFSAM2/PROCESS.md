# UFSAM2 Process

This file is the compact engineering/process log for branch `exp/uncertainty-guided-fss`.

## Goal

Project: **Uncertainty-Guided SAM2 Few-Shot Segmentation**.

The final claim must be an official FSS table:

- `1-shot / 5-shot`
- `fold0-3 / mean`
- official `mIoU / FB-IoU`
- `Ours` better than the SANSA baseline

Support-selection IoU, calibration, AUROC, and risk curves are supporting evidence only. They cannot replace official FSS metrics.

## Current Experiment Lines

| Line | Dataset / weight | Role | Status |
| --- | --- | --- | --- |
| Generalist part segmentation | Pascal-Part / PACO-Part, `adapter_generalist.pth`, `channel_factor=0.8` | Module A auxiliary evidence | Mostly closed |
| Strict FSS | COCO-20i fold adapters, FSS-1000, possibly Pascal-5i | Main paper table, especially 5-shot Module B / A+B | Active |

## Code State

Official `inference_fss.py` now supports:

| Feature | Flag | Notes |
| --- | --- | --- |
| pure hflip TTA | `--hflip_tta` | averages normal/flipped query logits |
| UQ-gated hflip | `--uq_hflip_tta --uq_head_ckpt ... --uq_gate_threshold ...` | runs hflip only for low expected-IoU episodes |
| BRM | `--boundary_refine` | requires a checkpoint with `brm.*` weights |
| Module B weighted logits | `--support_agg weighted_logits --support_uq_head_ckpt ...` | first official-path support aggregation prototype |
| smoke cap | `--max_eval_episodes` | use only for small-loop validation |

Trace/uncertainty tools:

- `collect_uncertainty_cache.py`: collects query/support decoder traces and true IoU.
- `train_uncertainty_head.py`: trains expected-IoU heads; `tokens_match` includes support-query matching features.
- `evaluate_support_selection.py`: diagnostic only; do not use it as the main result.

Current limitations:

- `--support_agg` is not yet combined with `--uq_hflip_tta`.
- First Module B weighted-logits prototype is not ready for full COCO runs because COCO smoke hurts mIoU.

## Environment

Run from:

```bash
cd /data6/chensq/UFSAM2/UFSAM2
conda activate sam2coco
export MPLCONFIGDIR=/tmp/matplotlib
```

Key paths:

- Data root: `/data6/chensq/datasets`
- Generalist adapter: `pretrain/adapter_generalist.pth`
- FSS fold0 adapter: `pretrain/adapter_fss_fold0.pth`
- COCO-20i adapters: `pretrain/coco-20i-4/adapter_coco_fold{0..3}.pth`
- Pascal/PACO mixed support head: `output/uncertainty_head_mixed_tokens_match_fss_pascal_paco/uncertainty_head.pt`

GPU note: if using `CUDA_VISIBLE_DEVICES=1`, the visible device is still addressed as `--device cuda`.

## Module A Results

### Pascal-Part Fold0, 1-shot

| Method | Threshold | Triggered | mIoU | FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| SANSA baseline | - | - | 36.29 | 64.26 |
| SANSA + hflip | - | 2500/2500 | 37.21 | 65.16 |
| SANSA + UQ-gated hflip | 0.3 | 1017/2500 | 37.49 | 65.24 |
| SANSA + UQ-gated hflip | 0.4 | 1235/2500 | 37.49 | 65.31 |

Read: UQ-gated hflip beats pure hflip on fold0. Threshold `0.3` is the efficient default.

### Pascal-Part 4-fold

UQ-gated hflip, threshold `0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.49 | 65.24 |
| 1 | 959 | 112 | 65.27 | 72.39 |
| 2 | 2500 | 908 | 38.49 | 65.63 |
| 3 | 2500 | 378 | 56.65 | 76.06 |
| mean | - | - | 49.48 | 69.83 |

BRM always + UQ-gated hflip, threshold `0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.29 | 64.70 |
| 1 | 959 | 112 | 66.00 | 72.83 |
| 2 | 2500 | 908 | 38.64 | 65.53 |
| 3 | 2500 | 378 | 56.83 | 76.09 |
| mean | - | - | 49.69 | 69.79 |

Read:

- Trigger rate: `2415/8459 = 28.6%`.
- BRM+UQ vs UQ-only: `+0.21` mIoU, `-0.04` FB-IoU.
- BRM+UQ is effectively tied with prior BRM + unconditional hflip (`49.61 / 69.82`) while avoiding most hflip runs.

### PACO-Part 4-fold

BRM always + UQ-gated hflip, threshold `0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 991 | 41.59 | 67.84 |
| 1 | 2500 | 856 | 45.49 | 66.83 |
| 2 | 2500 | 767 | 46.42 | 66.20 |
| 3 | 2500 | 846 | 41.36 | 64.47 |
| mean | - | - | 43.72 | 66.34 |

Read:

- Trigger rate: `3460/10000 = 34.6%`.
- Prior PACO BRM + unconditional hflip from original logs: `43.65 / 66.44`.
- Current BRM+UQ is essentially tied: `+0.07` mIoU, `-0.10` FB-IoU.
- This closes the Module A generalist part-segmentation line for now.

## Module B Results

First official-path prototype: `--support_agg weighted_logits`.

| Dataset | Fold | Shots | Episodes | Method | mIoU | FB-IoU | Extra |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Pascal-Part | 0 | 5 | 50 | SANSA all-support baseline | 45.84 | 66.22 | - |
| Pascal-Part | 0 | 5 | 50 | UQ-weighted logits | 47.43 | 70.61 | fallback 0/50, mean score 0.415 |
| COCO-20i | 0 | 5 | 50 | SANSA all-support baseline | 59.83 | 79.73 | - |
| COCO-20i | 0 | 5 | 50 | UQ-weighted logits | 58.47 | 80.55 | fallback 0/50, mean score 0.540 |

Read:

- Pascal-Part smoke is positive: `+1.59` mIoU and `+4.39` FB-IoU.
- COCO-20i smoke is not a main-metric win: `-1.36` mIoU and `+0.82` FB-IoU.
- Do not run full COCO weighted logits yet.
- Next: stricter fallback/adaptive-k or a COCO-specific support reliability head.

## Diagnostic Summary

Keep these as motivation only:

| Area | Current read |
| --- | --- |
| FSS-1000 | near ceiling; useful sanity check, weak main evidence |
| Pascal-Part support selection | token head beats random/SAM-score and roughly ties all-supports |
| PACO-Part support selection | mixed/tokens-match heads improve diagnostics but top1 gains are not robust |
| Memory propagation | official SANSA does not use query-to-query propagation; branch is paused |

## Current Priorities

1. Fix Module B before running full COCO.
   - Run fallback sweep on COCO fold0 5-shot smoke.
   - Try `--support_fallback_margin 0.10`, `0.20`, and `--support_fallback_min_score 0.60`.
   - If mIoU still falls below baseline, train/use a COCO-specific support reliability head.

2. Build the strict FSS main table.
   - First: COCO-20i fold0 5-shot SANSA vs improved Module B.
   - Then: COCO-20i fold0-3.
   - Add FSS-1000 sanity check.

3. Combine A+B only after Module B is stable.
   - Current `--support_agg` and `--uq_hflip_tta` are intentionally not combined yet.

4. Stop expanding Module A generalist runs.
   - Pascal-Part/PACO-Part already support the auxiliary Module A story.

## Canonical Commands

COCO-20i fold0 5-shot baseline smoke:

```bash
python inference_fss.py \
  --dataset_file coco --prompt mask --shots 5 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.3 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/coco-20i-4/adapter_coco_fold0.pth \
  --name_exp eval_coco_f0_5shot_baseline_smoke \
  --max_eval_episodes 50
```

COCO-20i fold0 5-shot Module B smoke with stricter fallback:

```bash
python inference_fss.py \
  --dataset_file coco --prompt mask --shots 5 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.3 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/coco-20i-4/adapter_coco_fold0.pth \
  --name_exp eval_coco_f0_5shot_uq_weighted_logits_margin020_smoke \
  --support_agg weighted_logits \
  --support_uq_head_ckpt output/uncertainty_head_mixed_tokens_match_fss_pascal_paco/uncertainty_head.pt \
  --support_fallback_margin 0.20 \
  --max_eval_episodes 50
```

## Guardrails

- Main results must come from official `inference_fss.py` metrics.
- Do not use support-selection avg IoU as the main table.
- Do not sell calibration/AUROC as the final result.
- Do not use Pascal-Part/PACO-Part generalist as the strict FSS main claim.
- Do not commit checkpoints, caches, output folders, or datasets.
- Update `PROCESS.md` or `EXPERIMENT.md` after each stage-level conclusion and commit the docs.
