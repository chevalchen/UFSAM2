# UFSAM2 Process

> [!WARNING]
> **HISTORICAL RECORD — NOT CURRENT AUTHORITY**
> This file preserves the pre-baseline-restart engineering history for hypothesis generation and debugging only. It must not be used as the current experiment contract, status, baseline, result, or decision authority. See [`EXPERIMENT_LEDGER.md`](../EXPERIMENT_LEDGER.md) for the current index and [`experiments/`](../experiments/) for authoritative per-experiment records.

This is the compact engineering/process log for branch `exp/uncertainty-guided-fss`.

> **Primary audience:** Agents maintaining the engineering and experiment workflow.

## Goal

Project: **Uncertainty-Guided SAM2 Few-Shot Segmentation**.

The final claim must be an official FSS table:

- `1-shot / 5-shot`
- `fold0-3 / mean`
- official `mIoU / FB-IoU`
- `Ours` better than the SANSA baseline

Support-selection IoU, calibration, AUROC, and risk curves are supporting evidence only. They cannot replace official FSS metrics.

## Experiment Organization

Use a two-level structure for every result:

1. **Strict FSS**
   - Datasets: COCO-20i, FSS-1000, possibly Pascal-5i.
   - Weights: fold-specific FSS adapters, e.g. `pretrain/coco-20i-4/adapter_coco_fold{0..3}.pth`.
   - Role: main paper table.
2. **Generalist In-context**
   - Datasets: Pascal-Part and PACO-Part.
   - Weight: `pretrain/adapter_generalist.pth`, `channel_factor=0.8`.
   - Role: auxiliary SANSA-aligned generalist evidence.

Inside each setting, organize rows as:

- **Baseline**
- **Module A**: uncertainty-guided ambiguity refinement.
- **Module B**: uncertainty-guided support reliability and aggregation.
- **Module C**: uncertainty-triggered query self-prompting.
- **A+B**: final combination.

Parameter variants stay under the relevant module, e.g. `uq_gate_threshold=0.3` under Module A and `support_fallback_margin=0.20` under Module B.

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

## Code State

Official `inference_fss.py` supports:

| Feature | Flag | Notes |
| --- | --- | --- |
| pure hflip TTA | `--hflip_tta` | averages normal/flipped query logits |
| UQ-gated hflip | `--uq_hflip_tta --uq_head_ckpt ... --uq_gate_threshold ...` | runs hflip only for low expected-IoU episodes |
| BRM | `--boundary_refine` | requires a checkpoint with `brm.*` weights |
| Module B weighted logits | `--support_agg weighted_logits --support_uq_head_ckpt ...` | first official-path support aggregation prototype |
| Module C memory-to-point | `--memory_to_point_prompt --mtp_trigger_threshold ...` | reruns uncertain query masks with automatic point prompts |
| smoke cap | `--max_eval_episodes` | use only for small-loop validation |

Current limitations:

- `--support_agg` is not yet combined with `--uq_hflip_tta`.
- `weighted_logits + support_fallback_margin=0.20` completed COCO-20i 5-shot full 4-fold evaluation, but it is below the SANSA official 5-shot mIoU baseline. Treat it as an ablation, not the main result.

## Strict FSS

### Baseline

Use SANSA paper / official numbers for baseline alignment whenever available.

| Dataset | Shot | SANSA mean mIoU | Notes |
| --- | ---: | ---: | --- |
| COCO-20i | 1-shot | 60.2 | SANSA paper / official |
| COCO-20i | 5-shot | 64.3 | SANSA paper / official |
| FSS-1000 | 1-shot | 91.4 | SANSA paper / official |
| FSS-1000 | 5-shot | 92.1 | SANSA paper / official |

### Module A

No finalized strict-FSS Module A result yet. Current Module A evidence is mainly from the Generalist In-context part datasets.

### Module B

Current official-path prototype: `--support_agg weighted_logits`.

COCO-20i 5-shot, `support_fallback_margin=0.20`:

| Fold | Episodes | mIoU | FB-IoU | Fallback | Mean score | Mean max score | Mean margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1000 | 63.45 | 78.15 | 458/1000 | 0.534 | 0.656 | 0.280 |
| 1 | 1000 | 65.36 | 80.49 | 459/1000 | 0.557 | 0.680 | 0.273 |
| 2 | 1000 | 66.18 | 82.11 | 583/1000 | 0.589 | 0.688 | 0.228 |
| 3 | 1000 | 59.69 | 78.88 | 464/1000 | 0.574 | 0.692 | 0.271 |
| mean | - | 63.67 | 79.91 | 1964/4000 | 0.564 | 0.679 | 0.263 |

Baseline alignment:

| Method | COCO-20i 5-shot mean mIoU | Delta vs SANSA |
| --- | ---: | ---: |
| SANSA official / paper baseline | 64.30 | - |
| UQ-weighted logits + fallback margin 0.20 | 63.67 | -0.63 |

Read: this validates the official-path implementation, but it is not a main result. Replace it with adaptive-k, a COCO-specific support reliability head, or a more conservative reliability gate.

### Module C

Current official-path prototype: uncertainty-triggered Memory-to-Point Self-Prompting (`--memory_to_point_prompt`).

Mechanism:

- run normal memory-conditioned SANSA query decoding;
- compute an unsupervised query quality score from logit stability and multimask disagreement;
- if the score is below `--mtp_trigger_threshold`, sample automatic positive/negative points from the query prediction;
- rerun the SAM2 head with the same memory-conditioned feature and the generated point prompts;
- accept the second pass only when its quality score does not regress.

COCO-20i 5-shot, fold-specific adapters:

| Setting | F0 mIoU | F1 mIoU | F2 mIoU | F3 mIoU | Mean mIoU | Mean FB-IoU | Triggered | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP `t085` | 64.65 | 67.49 | 65.50 | 60.29 | 64.48 | 80.28 | 34/4000 | 30/34 |
| MTP `t090` | 64.66 | 67.44 | 65.58 | 60.17 | 64.46 | 80.31 | 79/4000 | 74/79 |
| MTP `t092` | 64.52 | 67.43 | 65.58 | 60.28 | 64.45 | 80.29 | 121/4000 | 113/121 |

Fold0 threshold stress:

| Setting | Fold0 mIoU | Fold0 FB-IoU | Triggered | Accepted |
| --- | ---: | ---: | ---: | ---: |
| MTP `t095` | 64.26 | 79.25 | 68/1000 | 52/68 |

Baseline alignment:

| Method | COCO-20i 5-shot mean mIoU | Delta vs SANSA |
| --- | ---: | ---: |
| SANSA official / paper baseline | 64.30 | - |
| MTP `t085` | 64.48 | +0.18 |
| MTP `t090` | 64.46 | +0.16 |
| MTP `t092` | 64.45 | +0.15 |

Read:

- MTP is a low-trigger precision intervention: `t085` triggers only `34/4000 = 0.85%` episodes and gives the best mean mIoU.
- More aggressive triggering does not help; fold0 `t095` drops below the official 5-shot baseline.
- Current best strict-FSS candidate is MTP `t085`, but the gain is small. A local no-MTP 4-fold baseline and ablations such as `--mtp_num_negative_points 0` are still needed before making a strong claim.

### A+B

No finalized result yet. Combine only after Module B is stable enough not to underperform the SANSA baseline.

## Generalist In-context

### Baseline

Setting: `adapter_generalist.pth`, `channel_factor=0.8`.

The following rows are 1-shot mIoU, reported as `fold0 / fold1 / fold2 / fold3 / mean`.

| Dataset | Shot | Weight | F0 | F1 | F2 | F3 | Mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Pascal-Part | 1-shot | `adapter_generalist.pth` | 36.29 | 65.16 | 38.20 | 56.77 | 49.105 |
| PACO-Part | 1-shot | `adapter_generalist.pth` | 40.27 | 44.20 | 46.09 | 41.24 | 42.95 |

These rows are mIoU only; they do not include FB-IoU.

### Module A

Module A includes pure hflip, UQ-gated hflip, and BRM. BRM is a trainable boundary-aware mask refinement branch, not TTA.

Pascal-Part 1-shot, UQ-gated hflip, `uq_gate_threshold=0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.49 | 65.24 |
| 1 | 959 | 112 | 65.27 | 72.39 |
| 2 | 2500 | 908 | 38.49 | 65.63 |
| 3 | 2500 | 378 | 56.65 | 76.06 |
| mean | - | - | 49.48 | 69.83 |

Pascal-Part 1-shot, BRM always + UQ-gated hflip, `uq_gate_threshold=0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.29 | 64.70 |
| 1 | 959 | 112 | 66.00 | 72.83 |
| 2 | 2500 | 908 | 38.64 | 65.53 |
| 3 | 2500 | 378 | 56.83 | 76.09 |
| mean | - | - | 49.69 | 69.79 |

PACO-Part 1-shot, BRM always + UQ-gated hflip, `uq_gate_threshold=0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 991 | 41.59 | 67.84 |
| 1 | 2500 | 856 | 45.49 | 66.83 |
| 2 | 2500 | 767 | 46.42 | 66.20 |
| 3 | 2500 | 846 | 41.36 | 64.47 |
| mean | - | - | 43.72 | 66.34 |

Read:

- Pascal-Part BRM+UQ vs UQ-only: `+0.21` mIoU, `-0.04` FB-IoU.
- Pascal-Part BRM+UQ is essentially tied with prior BRM + unconditional hflip (`49.61 / 69.82`) while running hflip on only `28.6%` of episodes.
- PACO-Part BRM+UQ is essentially tied with prior BRM + unconditional hflip (`43.65 / 66.44`) while running hflip on `34.6%` of episodes.
- The safe claim is uncertainty-controlled refinement efficiency with comparable or slightly better mIoU, not a large accuracy gain.

### Module B

Pascal-Part 5-shot smoke:

| Dataset | Fold | Shots | Episodes | Method | mIoU | FB-IoU | Extra |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Pascal-Part | 0 | 5 | 50 | SANSA all-support baseline | 45.84 | 66.22 | local smoke |
| Pascal-Part | 0 | 5 | 50 | UQ-weighted logits | 47.43 | 70.61 | fallback 0/50, mean score 0.415 |

Read: positive smoke result (`+1.59` mIoU, `+4.39` FB-IoU), but it needs full 4-fold matched evaluation before becoming an auxiliary result.

### A+B

No finalized result yet. Do not mix A+B into the story until Module B has a reliable full result.

## Priorities

1. Strict FSS: replace the current Module B candidate; it is below SANSA COCO-20i 5-shot mIoU.
2. Strict FSS: every row must include SANSA official baseline and delta.
3. Generalist In-context: Module A is mostly closed; only expand if needed for the paper story.
4. Generalist In-context: if continuing Module B, run matched full 4-fold 5-shot baseline and Module B.
5. A+B: combine only after Module B is stable.

## Guardrails

- Main results must come from official `inference_fss.py` metrics.
- Do not use support-selection avg IoU as the main table.
- Do not sell calibration/AUROC as the final result.
- Do not use Pascal-Part/PACO-Part generalist as the strict FSS main claim.
- Do not commit checkpoints, caches, output folders, or datasets.
- Update `PROCESS.md` or `EXPERIMENT.md` after each stage-level conclusion and commit the docs.

## AV-PMC Post-Memory Calibration (implemented, no result yet)

The code now contains a default-off, selective post-memory feature repair. It must be trained in three ordered stages while SANSA/adapters remain frozen:

1. `operator`: always-on zero-initialized feature residual;
2. `spatial`: signed dense repair-benefit prediction using the frozen operator;
3. `gain`: signed episode delta-IoU prediction using the frozen operator and spatial gate.

Example sequence (replace paths and dataset/fold arguments with the locked protocol):

```powershell
python train_post_memory_calibration.py --resume <sansa.pth> --pmc_train_stage operator --batch_size 1 --name_exp pmc_operator
python train_post_memory_calibration.py --resume <sansa.pth> --pmc_checkpoint <pmc_operator.pth> --pmc_train_stage spatial --batch_size 1 --name_exp pmc_spatial
python train_post_memory_calibration.py --resume <sansa.pth> --pmc_checkpoint <pmc_spatial.pth> --pmc_train_stage gain --batch_size 1 --name_exp pmc_gain
```

Standalone evaluation:

```powershell
python inference_fss.py --resume <sansa.pth> --post_memory_calibration --pmc_checkpoint <pmc_gain.pth> --pmc_mode gated --dataset_file coco --shots 5 --fold 0 --name_exp pmc_eval
```

For attribution, evaluate `pmc_mode=operator`, `pmc_mode=spatial`, and `pmc_mode=gated` on identical episodes. Do not combine AV-PMC with hflip, BRM, MTP, or support aggregation until the standalone result is positive. The implementation status does not imply experimental effectiveness.
