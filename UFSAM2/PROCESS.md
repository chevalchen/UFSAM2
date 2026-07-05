# UFSAM2 Process

## Scope

- Base model: SANSA copied under `UFSAM2/`.
- Current setting: mask-only few-shot segmentation.
- Support masks are dataset GT masks.
- Core route: freeze SANSA, collect query-side decoder traces, train a lightweight expected-IoU / failure-risk head, and validate it by intervention gain.
- Do not move to joint training, semantic negative prompting, or real memory gating until the frozen uncertainty evidence is clean.

## Constraints

- Prefer intervention gain over standalone correlation/AUROC.
- Use class-disjoint, cross-seed, or cross-dataset checks before treating a result as general.
- Keep experiment conclusions in this file.
- Commit stage-level code/docs only; never commit checkpoints, caches, datasets, or generated output.

## Code State

- `collect_uncertainty_cache.py`: collects per-episode metadata, true IoU, SAM score, query tokens, object pointer, memory summary, and area stats.
  - Now also stores support-side trace aggregates: support IoU token, selected mask token, object pointer, memory summary.
- `train_uncertainty_head.py`: trains expected-IoU MLP heads from cached traces.
  - Main feature set: `query_iou_token + query_mask_token + query_obj_ptr`.
  - `tokens_match` adds support-side tokens plus query-support absdiff/product/cosine matching features.
  - Supports multiple `--cache_path` inputs for mixed/unified training.
  - Supports `--split_by dataset_class` and `--heldout_dataset`.
  - Writes `test_by_dataset` and `sam_score_test_by_dataset`.
- `evaluate_support_selection.py`: 5-shot intervention. Runs each support independently as 1-shot, then compares random, SAM-score, token-head, all-supports, and oracle.
  - Optional `--official_metrics` also accumulates SANSA-style class mIoU / FB-IoU for each support-selection strategy.
- `evaluate_memory_propagation_risk.py`: side-branch probe for test-time sequential pseudo-query memory.
- `inference_fss.py`: official FSS evaluation path now supports Module-A no-training variants.
  - `--hflip_tta` averages normal and horizontally flipped query logits in logit space while reusing the same support memory.
  - `--uq_hflip_tta --uq_head_ckpt ... --uq_gate_threshold ...` first estimates query expected IoU from decoder traces, then triggers hflip only for low-confidence episodes.
  - `--max_eval_episodes` is available for smoke/small-loop validation; omit it for full official mIoU / FB-IoU tables.
- SANSA/SAM2 trace hooks:
  - `mask_decoder.py`: saves `last_iou_token_out`, `last_mask_tokens_out`.
  - `model_utils.py`: extends `DecoderOutput`.
  - `sam2_base.py`: fills token/output fields.
  - `sansa.py`: supports `return_traces=True` for query frames and `support_traces` for support frames.
  - `sansa.py`: supports optional query-frame hflip TTA for official inference without changing support prompts or output format.

## Environment

Run from:

```bash
cd /data6/chensq/UFSAM2/UFSAM2
```

Common settings:

```bash
MPLCONFIGDIR=/tmp/matplotlib
CUDA_VISIBLE_DEVICES=1
```

Inside Python, `CUDA_VISIBLE_DEVICES=1` remaps that physical GPU to logical `cuda:0`, so use `--device cuda`, not `--device cuda:1`.

Key paths:

- Conda env: `sam2coco`
- Data root: `/data6/chensq/datasets`
- Weights: `pretrain/`
- Generalist adapter: `pretrain/adapter_generalist.pth`
- FSS adapter: `pretrain/adapter_fss_fold0.pth`

## Reusable Commands

FSS cache:

```bash
MPLCONFIGDIR=/tmp/matplotlib python collect_uncertainty_cache.py \
  --dataset_file fss --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.3 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_fss_fold0.pth \
  --cache_path output/fss_fold0_1shot_mask_uncertainty.pt
```

Pascal-Part/PACO-Part use the generalist adapter with `channel_factor=0.8`:

```bash
MPLCONFIGDIR=/tmp/matplotlib python collect_uncertainty_cache.py \
  --dataset_file pascal_part \
  --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --cache_path output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt
```

Single-dataset class-disjoint head:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt \
  --output_dir output/uncertainty_head_pascal_part_fold0_1shot_class_split \
  --device cuda --epochs 200 --batch_size 128 \
  --feature_set tokens --split_by class_id
```

Mixed/unified head:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path \
    output/fss_fold0_1shot_mask_uncertainty.pt \
    output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt \
    output/paco_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt \
  --output_dir output/uncertainty_head_mixed_fss_pascal_paco_1shot_dataset_class \
  --device cuda --epochs 200 --batch_size 128 \
  --feature_set tokens --split_by dataset_class
```

Support-query matching head, after recollecting caches with support-side traces:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path \
    output/fss_fold0_1shot_mask_uncertainty_match.pt \
    output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08_match.pt \
    output/paco_part_fold0_1shot_mask_uncertainty_generalist_cf08_match.pt \
  --output_dir output/uncertainty_head_mixed_tokens_match_fss_pascal_paco \
  --device cuda --epochs 200 --batch_size 128 \
  --feature_set tokens_match --split_by dataset_class
```

Support selection:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib python evaluate_support_selection.py \
  --seed 0 \
  --dataset_file paco_part \
  --prompt mask --shots 5 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --head_ckpt output/uncertainty_head_mixed_fss_pascal_paco_1shot_dataset_class/uncertainty_head.pt \
  --max_episodes 200 \
  --output_path output/support_selection_paco_part_fold0_200_mixed_head_seed0.json
```

Leave-one-dataset-out example:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path \
    output/fss_fold0_1shot_mask_uncertainty.pt \
    output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt \
    output/paco_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt \
  --output_dir output/uncertainty_head_lodo_paco_1shot_dataset_class \
  --device cuda --epochs 200 --batch_size 128 \
  --feature_set tokens --split_by dataset_class \
  --heldout_dataset paco_part
```

## Results

### 2026-07-05 Module-A Official FSS Path

Implemented the first Module-A official-evaluation path:

| item | status |
| --- | --- |
| SANSA baseline path | unchanged default in `inference_fss.py` |
| hflip TTA | implemented as `--hflip_tta` |
| UQ-gated hflip | implemented as `--uq_hflip_tta` with an expected-IoU head checkpoint |
| official metrics | still accumulated through `AverageMeter` / `Evaluator.classify_prediction` |
| smoke controls | `--max_eval_episodes` added for small-loop checks only |

Verification completed in the `sam2coco` environment:

- `conda run -n sam2coco python -m py_compile opts.py inference_fss.py models/sansa/sansa.py`
- `conda run -n sam2coco python inference_fss.py --help`

GPU evaluation was not run in this session because CUDA was not visible (`torch.cuda.is_available() == False`, `cuda_count == 0`; `nvidia-smi` could not communicate with the driver). Next run on a GPU-visible session should start with Pascal-Part fold0 1-shot/5-shot small loops:

```bash
MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES=1 python inference_fss.py \
  --dataset_file pascal_part --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_pascal_part_f0_1shot_baseline_smoke \
  --max_eval_episodes 50

MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES=1 python inference_fss.py \
  --dataset_file pascal_part --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_pascal_part_f0_1shot_hflip_smoke \
  --hflip_tta --max_eval_episodes 50

MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES=1 python inference_fss.py \
  --dataset_file pascal_part --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_pascal_part_f0_1shot_uq_hflip_smoke \
  --uq_hflip_tta \
  --uq_head_ckpt output/uncertainty_head_pascal_part_fold0_1shot_class_split/uncertainty_head.pt \
  --uq_gate_threshold 0.5 \
  --max_eval_episodes 50
```

Repeat the same three commands with `--shots 5` and matching experiment names for the 5-shot small loop before launching full fold0 runs without `--max_eval_episodes`.

Pascal-Part fold0 1-shot smoke results reported from GPU runs (`--max_eval_episodes 50`):

| setting | threshold | triggered | mIoU | FB-IoU | delta vs baseline | delta vs hflip | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SANSA baseline | - | 0/50 | 30.78 | 62.07 | - | -0.94 / -0.95 | no TTA |
| pure hflip TTA | - | 50/50 | 31.72 | 63.02 | +0.94 / +0.95 | - | best current smoke result |
| UQ-gated hflip | 0.3 | 24/50 | 31.62 | 63.28 | +0.84 / +1.21 | -0.10 / +0.26 | efficient gate: half the flips, near-hflip mIoU, better FB-IoU |
| UQ-gated hflip | 0.4 | 32/50 | 31.99 | 63.45 | +1.21 / +1.38 | +0.27 / +0.43 | best smoke result; beats pure hflip on both metrics |
| UQ-gated hflip | 0.5 | 35/50 | 31.74 | 63.19 | +0.96 / +1.12 | +0.02 / +0.17 | roughly ties pure hflip with fewer flips |
| UQ-gated hflip | 0.6 | 38/50 | 31.99 | 63.23 | +1.21 / +1.16 | +0.27 / +0.21 | high mIoU, slightly lower FB-IoU than `t=0.4` |
| UQ-gated hflip | 0.7 | 46/50 | 31.64 | 62.93 | +0.86 / +0.86 | -0.08 / -0.09 | very permissive gate; close to pure hflip but not better |

Immediate read: pure hflip clearly improves the 50-episode smoke split. UQ-gated hflip is now positive in the intended sense: `t=0.4` beats pure hflip on both mIoU and FB-IoU while skipping 18/50 flips, and `t=0.3` is the efficient backup with 24/50 flips and better FB-IoU than pure hflip. For full Pascal-Part fold0, run baseline, pure hflip, UQ-gated hflip `t=0.4`, and optionally `t=0.3` as an efficiency ablation.

Pascal-Part fold0 1-shot full results reported from GPU runs (`2500` episodes):

| setting | threshold | triggered | mIoU | FB-IoU | delta vs baseline | delta vs hflip | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SANSA baseline | - | 0/2500 | 36.29 | 64.26 | - | -0.92 / -0.90 | matches previous full baseline scale |
| pure hflip TTA | - | 2500/2500 | 37.21 | 65.16 | +0.92 / +0.90 | - | clean no-training TTA gain |
| UQ-gated hflip | 0.4 | 1235/2500 | 37.49 | 65.31 | +1.20 / +1.05 | +0.28 / +0.15 | positive Module-A result: better than pure hflip with about half the flips |

Full-fold read: UQ-gated hflip at `t=0.4` validates the Module-A idea on Pascal-Part fold0 1-shot. It improves over baseline and pure hflip while triggering hflip for only 49.4% of episodes. This is stronger than the smoke result because it holds on the full official fold0 evaluation path. Next checks: run `t=0.3` full as an efficiency ablation, then repeat the baseline / hflip / UQ-gated hflip comparison for Pascal-Part fold0 5-shot.

### FSS-1000

FSS-1000 is a sanity/calibration dataset, not the final proof: SANSA is near ceiling.

| cache | episodes | mean IoU | IoU<0.5 | IoU<0.7 |
| --- | ---: | ---: | ---: | ---: |
| 1-shot | 2400 | 0.9122 | 2.71% | 6.29% |
| 5-shot all-supports | 2400 | 0.9205 | 1.96% | 4.87% |

Class-disjoint expected-IoU head:

| setting | MAE | Pearson | Spearman | AUROC<0.5 | AUROC<0.7 | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token head | 0.0592 | 0.4322 | 0.7094 | 0.7486 | 0.8091 | 0.0240 |
| SAM `sam_score` | 0.0696 | 0.3135 | 0.7075 | 0.6437 | 0.7918 | 0.0675 |

Full 2400-episode support selection:

| setting | mIoU | fail<0.5 | risk<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.9138 | 2.67% | 5.75% |
| SAM-score | 0.9170 | 2.58% | 5.67% |
| token | 0.9211 | 2.13% | 4.83% |
| all-supports | 0.9205 | 1.96% | 4.87% |
| oracle | 0.9353 | 1.13% | 2.83% |

Read: token beats random/SAM-score and ties all-supports globally. In high-spread episodes, token advantage over SAM-score grows, but FSS is too close to ceiling to be the main evidence.

### Pascal-Part

Use strict object+part class-disjoint setup and `adapter_generalist.pth` with `--adaptformer_stages 2 3 --channel_factor 0.8`.

1-shot cache:

- episodes: `2500`
- mean IoU: `0.4047`
- `IoU<0.5`: `57.12%`
- `IoU<0.7`: `76.44%`
- mean SAM score: `0.5410`

Class-disjoint expected-IoU head:

| setting | MAE | Pearson | Spearman | AUROC<0.5 | AUROC<0.7 | ECE | pred mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token head | 0.1503 | 0.7221 | 0.6921 | 0.8875 | 0.8473 | 0.0514 | 0.3246 |
| SAM `sam_score` | 0.1914 | 0.6916 | 0.6422 | 0.9005 | 0.8931 | 0.1330 | 0.4567 |

Support selection, full 2500 episodes, cross-seed:

| seed | random | SAM-score | token | all-supports | oracle | token-SAM | token-all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3905 | 0.4282 | 0.4643 | 0.4645 | 0.5636 | +0.0361 | -0.0002 |
| 1 | 0.3930 | 0.4232 | 0.4626 | 0.4591 | 0.5535 | +0.0394 | +0.0034 |
| 2 | 0.3977 | 0.4308 | 0.4592 | 0.4609 | 0.5563 | +0.0284 | -0.0018 |
| mean | 0.3938 | 0.4274 | 0.4620 | 0.4615 | 0.5578 | +0.0346 | +0.0005 |

Risk across seeds:

| setting | fail<0.5 | risk<0.7 |
| --- | ---: | ---: |
| random | 58-59% | 78-79% |
| SAM-score | 52-54% | 73-74% |
| token | 48-49% | 70-71% |
| all-supports | about 50% | about 72% |

Read: Pascal-Part is the strongest current positive result. Token support selection consistently beats random and SAM-score, ties all-supports in mIoU, and has lower failure/risk than all-supports. Oracle remains much higher, so support reliability is not solved.

Module A official evaluation, Pascal-Part 1-shot generalist, UQ-gated hflip:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.49 | 65.24 |
| 1 | 959 | 112 | 65.27 | 72.39 |
| 2 | 2500 | 908 | 38.49 | 65.63 |
| 3 | 2500 | 378 | 56.65 | 76.06 |
| mean | - | - | 49.48 | 69.83 |

Read:

- Threshold `0.3` gives a complete 4-fold generalist part-segmentation result.
- Total hflip trigger count is `2415/8459` episodes, or `28.6%`.
- This is a good Module A result: uncertainty preserves most hflip/refinement benefit while avoiding unconditional hflip on most episodes.
- Caveat: the uncertainty head is trained on Pascal-Part fold0 1-shot class split and used cross-fold here. This is fine for the current fast loop; if the result becomes central, rerun with fold-specific or leave-one-fold-out heads.

BRM integration:

- Current branch now supports `--boundary_refine` in `inference_fss.py`.
- The intended next Module A variant is BRM always + UQ-gated hflip.
- Use `/data6/chensq/SANSA_M/UncSANSA/output/train_brm_stage2/checkpoint0002.pth` for the first Pascal-Part run.
- The checkpoint is about 101MB because it stores the adapter, BRM, optimizer, lr scheduler, and args. The BRM itself is only 6 tensors, about 7K trainable parameters.
- Interpretation: BRM is a trainable boundary-aware mask refinement branch; hflip is the test-time consistency refinement; UQ controls whether to trigger hflip.

BRM always + UQ-gated hflip, Pascal-Part 1-shot generalist:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.29 | 64.70 |
| 1 | 959 | 112 | 66.00 | 72.83 |
| 2 | 2500 | 908 | 38.64 | 65.53 |
| 3 | 2500 | 378 | 56.83 | 76.09 |
| mean | - | - | 49.69 | 69.79 |

Read:

- Total hflip trigger count is unchanged from UQ-gated hflip-only: `2415/8459`, or `28.6%`.
- Compared with UQ-gated hflip-only (`49.48 / 69.83`), BRM always + UQ-gated hflip gives `+0.21` mIoU and `-0.04` FB-IoU.
- Compared with prior BRM + unconditional hflip from `TTA_SUM.md` (`49.61 / 69.82`), the UQ-gated version is effectively tied while avoiding unconditional hflip.
- Current read: keep BRM as compatible trainable mask refinement, but do not oversell it as a large independent gain. The stronger Module A claim is uncertainty-controlled refinement efficiency with slight mIoU improvement.

### PACO-Part

PACO-Part is the current generalization stress test: long-tail, fine-grained, and support-quality variance is stronger. Current sampler is stochastic and ignores `idx`; use fixed seeds. Add fixed episode lists if PACO becomes a final table.

1-shot cache:

- episodes: `2500`
- mean IoU: `0.4329`
- `IoU<0.5`: `55.84%`
- `IoU<0.7`: `74.60%`
- mean SAM score: `0.5501`

PACO-only class-disjoint head:

| setting | MAE | Pearson | Spearman | AUROC<0.5 | AUROC<0.7 | ECE | pred mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token head | 0.2837 | 0.2196 | 0.2046 | 0.5990 | 0.6483 | 0.1331 | 0.3864 |
| SAM `sam_score` | 0.2673 | 0.3747 | 0.3183 | 0.6586 | 0.7529 | 0.1476 | 0.5703 |

PACO-only support selection, 2 seeds x 200 episodes:

| setting | mIoU | fail<0.5 | risk<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.4520 | 54.3% | 71.8% |
| SAM-score | 0.4830 | 49.3% | 65.3% |
| token | 0.4858 | 50.0% | 67.8% |
| all-supports | 0.4948 | 48.5% | 66.5% |
| oracle | 0.5834 | 36.8% | 58.0% |

Read: PACO-only token head has weak signal but no useful intervention gain. It is below all-supports and not meaningfully better than SAM-score.

Mixed FSS+Pascal+PACO head:

| test set | scorer | MAE | Pearson | Spearman | AUROC<0.5 | AUROC<0.7 | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | mixed token | 0.1630 | 0.7476 | 0.7904 | 0.8673 | 0.9174 | 0.0578 |
| overall | SAM `sam_score` | 0.1520 | 0.7938 | 0.8374 | 0.9116 | 0.9449 | 0.0959 |
| PACO | mixed token | 0.2430 | 0.4099 | 0.3722 | 0.6968 | 0.7724 | 0.1046 |
| PACO | SAM `sam_score` | 0.2275 | 0.5938 | 0.5378 | 0.8551 | 0.8868 | 0.1469 |

Mixed `tokens_match` head:

| test set | scorer | MAE | Pearson | Spearman | AUROC<0.5 | AUROC<0.7 | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | `tokens_match` | 0.1550 | 0.7646 | 0.8075 | 0.8835 | 0.9227 | 0.0530 |
| overall | mixed token | 0.1630 | 0.7476 | 0.7904 | 0.8673 | 0.9174 | 0.0578 |
| Pascal-Part | `tokens_match` | 0.1356 | 0.5916 | 0.5760 | 0.7745 | 0.8420 | 0.0506 |
| Pascal-Part | mixed token | 0.1488 | 0.5873 | 0.4923 | 0.7408 | 0.7559 | 0.0864 |
| PACO-Part | `tokens_match` | 0.2281 | 0.4756 | 0.4446 | 0.7494 | 0.7878 | 0.0975 |
| PACO-Part | mixed token | 0.2430 | 0.4099 | 0.3722 | 0.6968 | 0.7724 | 0.1046 |

Matching-head read:

- Support-query matching features improve the mixed token head on overall metrics, Pascal-Part, and PACO-Part.
- Pascal-Part ranking improves most strongly: AUROC<0.7 rises from `0.7559` to `0.8420`.
- PACO-Part improves but still trails SAM-score in correlation and AUROC; intervention remains the deciding check.
- Next check: rerun PACO support selection with the `tokens_match` head for seeds 0/1.

Mixed-head PACO support selection, 2 seeds x 200 episodes:

| setting | PACO-only token | mixed token |
| --- | ---: | ---: |
| random | 0.4520 | 0.4520 |
| SAM-score | 0.4830 | 0.4830 |
| token | 0.4858 | 0.4949 |
| all-supports | 0.4948 | 0.4948 |
| oracle | 0.5834 | 0.5834 |
| token fail<0.5 | 50.0% | 48.0% |
| token risk<0.7 | 67.8% | 65.8% |

Mixed-head differences:

- mixed token - PACO-only token: `+0.0091`
- mixed token - SAM-score: `+0.0119 ± 0.0077` SE
- mixed token - all-supports: `+0.0001 ± 0.0079` SE
- oracle - mixed token: `+0.0885 ± 0.0081` SE

Read: mixed/unified training fixes part of PACO-only weakness and reaches all-supports, but does not robustly beat all-supports. This points toward adding support-query matching features rather than relying only on query-side tokens.

`tokens_match` PACO support-selection, 2 seeds x 200 episodes:

| setting | mIoU | fail<0.5 | risk<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.4520 | 54.3% | 71.8% |
| SAM-score | 0.4830 | 49.3% | 65.3% |
| mixed token | 0.4949 | 48.0% | 65.8% |
| `tokens_match` | 0.4945 | 47.3% | 65.8% |
| all-supports | 0.4948 | 48.5% | 66.5% |
| oracle | 0.5834 | 36.8% | 58.0% |

Per-seed `tokens_match` summaries:

| seed | random | SAM-score | mixed token | `tokens_match` | all-supports | oracle | match-SAM | match-all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4540 | 0.4867 | 0.5068 | 0.5098 | 0.5029 | 0.5918 | +0.0230 | +0.0069 |
| 1 | 0.4500 | 0.4794 | 0.4830 | 0.4792 | 0.4867 | 0.5750 | -0.0001 | -0.0075 |

Matching intervention read:

- `tokens_match` improves calibration/ranking metrics, but the support-selection gain does not survive seed 1.
- Across two seeds, `tokens_match` is effectively tied with mixed query-only and all-supports: match-all is about `-0.0003` mIoU.
- Risk is mildly better than all-supports (`65.8%` vs `66.5%` for IoU<0.7), but not better than SAM-score.
- Current read: support-query matching features are useful for expected-IoU prediction, but this version is not a robust PACO intervention win.

### Memory-Propagation Risk

Important correction:

- SANSA standard FSS inference does not use query-to-query sequential propagation.
- Support/reference frames are encoded into memory; each target/query is segmented independently.
- Therefore this branch is a test-time sequential memory extension, not SANSA baseline behavior.

Pascal-Part 100-chain debug:

| query | independent mIoU | sequential mIoU | mean harm | harm>0.05 |
| --- | ---: | ---: | ---: | ---: |
| q1 | 0.4057 | 0.4057 | 0.0000 | 0.0% |
| q2 | 0.3765 | 0.3589 | +0.0176 | 18.0% |
| q3 | 0.4104 | 0.4116 | -0.0012 | 18.0% |
| q2/q3 | 0.3935 | 0.3853 | +0.0082 | 18.0% |

Read: pseudo-query memory can hurt downstream queries, but current expected-IoU head is weak for predicting next-step harm. Pause this branch unless explicitly resuming memory gating.

## Current Summary

- FSS-1000: sanity check passed; token head predicts risk and support selection beats SAM-score, but dataset is near ceiling.
- Pascal-Part: main positive evidence. Token support selection consistently beats random/SAM-score and reduces risk relative to all-supports.
- PACO-Part: current query-only token head does not generalize strongly enough. Mixed training helps and reaches all-supports, but does not robustly beat it.
- Memory risk: real side signal, but not central to SANSA baseline and not ready for main claims.

## Next Directions

1. Run the support-query matching head.
   - First implementation is now in place: support-side traces plus `tokens_match`.
   - Recollect 1-shot FSS/Pascal/PACO caches so the new support fields exist.
   - Train mixed `tokens_match` head and rerun PACO 200-episode support selection for seeds 0/1.
   - Success criterion: improve over mixed query-only head and move beyond all-supports, not just SAM-score.

2. Keep mixed/unified and leave-one-dataset-out checks.
   - Mixed head showed PACO improvement, so it is a stronger default than PACO-only.
   - LODO is needed to separate true error prediction from dataset-distribution recognition.

3. Make PACO evaluation more reproducible if it becomes a final table.
   - Add fixed episode-list support or save sampled episode identities.
   - Run more seeds or full 2500-episode intervention only after the feature set is stronger.

4. Do not expand to joint training/memory gating yet.
   - Frozen SANSA + post-hoc uncertainty remains the cleanest current story.

5. Add a standard FSS benchmark for Module B.
   - Use COCO-20i 4-fold with SANSA per-fold pretrained adapters under `pretrain/coco-20i-4/`.
   - Report both official-style mIoU/FB-IoU and support-selection avg IoU.
   - This makes support reliability evidence more authoritative than Pascal-Part/PACO-only diagnostics.
