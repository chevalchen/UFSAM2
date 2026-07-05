# EXPERIMENT: Uncertainty-Guided SAM2 Few-Shot Segmentation

本文件用于当前分支 `exp/uncertainty-guided-fss` 的实验交接。核心目标是把 uncertainty 从诊断信号变成能提升 official FSS mIoU / FB-IoU 的 SANSA 改进模块。

## 1. Main Goal

课题：

> Uncertainty-Guided SAM2 Few-Shot Segmentation

最终结果必须是标准 FSS 表，而不是 support-selection 诊断表：

- `1-shot / 5-shot`
- `fold0 / fold1 / fold2 / fold3 / mean`
- official-style `mIoU / FB-IoU`
- 主结论：`Ours` 超过 `SANSA baseline`

support-selection avg IoU、calibration、AUROC、risk curve 只作为辅助分析，不能当主结果。

## 2. Current Experimental Position

当前采用双线实验设置。

| Line | Dataset / weight | Role | Main-claim status |
| --- | --- | --- | --- |
| Generalist part segmentation | Pascal-Part / PACO-Part, `pretrain/adapter_generalist.pth`, `channel_factor=0.8` | Module A proof, part ambiguity, boundary / small-part analysis | 辅助表 / SANSA Table 2 风格 |
| Strict FSS | COCO-20i fold weights, FSS-1000, possibly Pascal-5i | 最终标准 FSS 主表，尤其 5-shot Module B / A+B | 主结论 |

结论：

- 不需要推倒当前 generalist Pascal-Part 结果。它非常适合证明 Module A：uncertainty gate 对边界模糊、部件断裂、part ambiguity 的作用。
- 但最终 A+B 论文主结论不能只停在 Pascal-Part generalist；必须接到 strict FSS official evaluation。
- 一周内优先顺序：先把 Pascal-Part UQ-gated hflip 4 folds 跑完整，再决定 BRM 组合；随后尽快实现 Module B，并迁移到 strict FSS 5-shot。

## 3. Method Story

### Module A: Uncertainty-Guided Ambiguity Refinement

目标问题：

- 边界模糊；
- 部件边缘不准；
- 小部件断裂；
- query prediction 高风险时，SANSA 不知道是否值得额外 refinement。

当前动作：

- `SANSA`: no TTA；
- `SANSA + hflip`: 纯净 hflip TTA；
- `SANSA + UQ-gated hflip`: 只在 uncertainty head 预测低 expected IoU 时触发 hflip；
- `SANSA + BRM`: trainable boundary refinement branch；
- `SANSA + UQ-gated hflip + BRM`: Module A 完整候选。

注意：

- hflip / BRM 本身不是 uncertainty；
- uncertainty 的角色是 gate：决定是否触发额外 refinement；
- BRM 是 trainable branch，不能写成 frozen uncertainty baseline。

### Module B: Uncertainty-Guided Support Reliability And Aggregation

目标问题：

- 5-shot 中 support 质量不均；
- SANSA 默认把多个 support 等价写入 memory；
- 低质量 support 可能污染语义或边界。

最终不能只做 top1 support selection。Module B 应做成：

| Strategy | Role |
| --- | --- |
| UQ-top1 | ablation only |
| UQ-topk | 主表候选 |
| UQ-adaptive-k | 主表候选 |
| UQ-weighted logits | 主表候选 |
| UQ-fallback to all supports | 稳定性保护 |

推荐主说法：

> We estimate support-query reliability from SAM2 decoder traces and use it to adaptively select, aggregate, or fall back among support examples.

预期：

- 1-shot：Module B 基本 no-op，主要看 Module A；
- 5-shot：Module B / A+B 必须超过或至少稳定持平 all-support SANSA。

## 4. Current Official Results

### Pascal-Part Fold0, 1-shot, Generalist Weight

配置：

- dataset: `pascal_part`
- fold: `0`
- shots: `1`
- weight: `pretrain/adapter_generalist.pth`
- adapter: `--adaptformer_stages 2 3 --channel_factor 0.8`
- official `inference_fss.py`

Full fold0 results:

| Method | Threshold | Triggered | mIoU | FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| SANSA baseline | - | - | 36.29 | 64.26 |
| SANSA + hflip | - | 2500/2500 | 37.21 | 65.16 |
| SANSA + UQ-gated hflip | 0.3 | 1017/2500 | 37.49 | 65.24 |
| SANSA + UQ-gated hflip | 0.4 | 1235/2500 | 37.49 | 65.31 |

Conclusion:

- hflip itself is a strong no-training Module A baseline: `+0.92 mIoU` over SANSA on fold0.
- UQ-gated hflip improves over pure hflip: `+0.28 mIoU` at both threshold `0.3` and `0.4`.
- Threshold `0.3` is currently preferred for efficiency: same mIoU as `0.4`, fewer triggered episodes, slightly lower FB-IoU.
- Use threshold `0.3` for Pascal-Part fold1-3 unless later evidence changes it.

Smoke threshold sweep, 50 episodes:

| Method | Threshold | Triggered | mIoU | FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| baseline | - | - | 30.78 | 62.07 |
| hflip | - | 50/50 | 31.72 | 63.02 |
| UQ-gated hflip | 0.3 | 24/50 | 31.62 | 63.28 |
| UQ-gated hflip | 0.4 | 32/50 | 31.99 | 63.45 |
| UQ-gated hflip | 0.5 | 35/50 | 31.74 | 63.19 |
| UQ-gated hflip | 0.6 | 38/50 | 31.99 | 63.23 |
| UQ-gated hflip | 0.7 | 46/50 | 31.64 | 62.93 |

Full 4-fold UQ-gated hflip at threshold `0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.49 | 65.24 |
| 1 | 959 | 112 | 65.27 | 72.39 |
| 2 | 2500 | 908 | 38.49 | 65.63 |
| 3 | 2500 | 378 | 56.65 | 76.06 |
| mean | - | - | 49.48 | 69.83 |

Total trigger rate: `2415/8459 = 28.6%`.

Conclusion:

- The 4-fold result is a clean positive Module A signal: UQ-gated hflip reaches `49.48 / 69.83` while triggering hflip on only 28.6% of episodes.
- This should be treated as generalist part-segmentation evidence, not the final strict FSS main claim.
- The gate uses the Pascal-Part fold0 1-shot uncertainty head cross-fold; this is acceptable for the first full run, but a paper-critical version should consider fold-specific or leave-one-fold-out heads.

Full 4-fold BRM always + UQ-gated hflip at threshold `0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.29 | 64.70 |
| 1 | 959 | 112 | 66.00 | 72.83 |
| 2 | 2500 | 908 | 38.64 | 65.53 |
| 3 | 2500 | 378 | 56.83 | 76.09 |
| mean | - | - | 49.69 | 69.79 |

Total trigger rate: `2415/8459 = 28.6%`.

Conclusion:

- BRM always + UQ-gated hflip is slightly better than UQ-gated hflip-only in mIoU: `49.69` vs `49.48`, but FB-IoU is essentially tied/slightly lower: `69.79` vs `69.83`.
- Compared with prior BRM + unconditional hflip (`49.61 / 69.82` from `TTA_SUM.md`), BRM + UQ-gated hflip is effectively tied while using hflip on only 28.6% of episodes.
- Best reading: BRM is a compatible trainable mask-refinement base, while UQ is useful as the inference-time controller for expensive consistency refinement. The gain is modest, so do not oversell BRM+UQ as a large accuracy jump.

PACO-Part full 4-fold BRM always + UQ-gated hflip at threshold `0.3`:

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 991 | 41.59 | 67.84 |
| 1 | 2500 | 856 | 45.49 | 66.83 |
| 2 | 2500 | 767 | 46.42 | 66.20 |
| 3 | 2500 | 846 | 41.36 | 64.47 |
| mean | - | - | 43.72 | 66.34 |

Total trigger rate: `3460/10000 = 34.6%`.

PACO-Part read:

- Compared with prior PACO BRM + unconditional hflip (`43.65 / 66.44` from `TTA_SUM.md`), BRM + UQ-gated hflip is effectively tied: `+0.07` mIoU and `-0.10` FB-IoU.
- Compared with prior PACO generalist + hflip (`43.22 / 66.29`), the BRM + UQ-gated version is better in mIoU and essentially tied in FB-IoU.
- This confirms the Module A efficiency story beyond Pascal-Part: the gate uses hflip on only about one third of PACO episodes while preserving the BRM+hflip performance level.
- The PACO UQ head is trained on PACO-Part fold0 and used cross-fold; keep that caveat in any auxiliary table.

### Prior BRM / TTA Evidence From `TTA_SUM.md`

Pascal-Part, BRM checkpoint:

| Setting | mIoU | FB-IoU |
| --- | ---: | ---: |
| BRM, fold0 no flip | 36.13 | 63.85 |
| BRM + hflip, fold0 | 37.10 | 64.77 |
| BRM, 4-fold avg | 49.30 | 69.49 |
| BRM + hflip, 4-fold avg | 49.61 | 69.82 |

PACO-Part:

| Setting | mIoU | FB-IoU |
| --- | ---: | ---: |
| plain generalist, fold0 | 40.27 | 67.17 |
| generalist + hflip, fold0 | 40.70 | 67.52 |
| BRM + hflip, fold0 | 41.24 | 67.74 |
| generalist + hflip, 4-fold avg | 43.22 | 66.29 |
| BRM + hflip, 4-fold avg | 43.65 | 66.44 |

Use these as evidence that refinement helps, but current branch still needs a clean Module A story inside official evaluation.

## 5. Current Diagnostic Evidence

These results support the uncertainty story, but are not final main-table metrics.

### FSS-1000, 5-shot Support Selection

| Strategy | avg IoU |
| --- | ---: |
| random | 0.9138 |
| SAM-score | 0.9170 |
| token | 0.9211 |
| all-supports | 0.9205 |
| oracle | 0.9353 |

Conclusion: token uncertainty has signal, but FSS-1000 is near ceiling and weak as the main improvement dataset.

### Pascal-Part, 5-shot Support Selection

3 seeds, full 2500 episodes:

| Strategy | avg IoU |
| --- | ---: |
| random | 0.3938 |
| SAM-score | 0.4274 |
| token | 0.4620 |
| all-supports | 0.4615 |
| oracle | 0.5578 |

Conclusion: uncertainty support reliability clearly beats random / SAM-score and is close to all-supports; oracle gap remains large, so adaptive aggregation has room.

### PACO-Part, 5-shot Support Selection

2 seeds x 200 episodes:

| Strategy | avg IoU | fail<0.5 | risk<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.4520 | 54.3% | 71.8% |
| SAM-score | 0.4830 | 49.3% | 65.3% |
| mixed token | 0.4949 | 48.0% | 65.8% |
| tokens_match | 0.4945 | 47.3% | 65.8% |
| all-supports | 0.4948 | 48.5% | 66.5% |
| oracle | 0.5834 | 36.8% | 58.0% |

Conclusion: top1 support selection alone is not stable enough; Module B should use adaptive-k / weighted logits / fallback.

## 6. Main Tables To Build

### Table A: Strict FSS Main Table

This is the final claim table.

| Method | 1-shot F0 | F1 | F2 | F3 | Mean | 5-shot F0 | F1 | F2 | F3 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SANSA | | | | | | | | | | |
| SANSA + Module A | | | | | | | | | | |
| SANSA + Module B | - | - | - | - | - | | | | | |
| Ours A+B | | | | | | | | | | |

Dataset priority:

1. COCO-20i: strongest strict FSS benchmark available locally through fold weights.
2. FSS-1000: sanity / near-ceiling check.
3. Pascal-5i: add only if protocol and weights are clean enough within time.

### Table B: Generalist Part Segmentation / Module A Table

This supports the SANSA Table 2 style story and the part ambiguity story.

| Method | Pascal-Part F0 | F1 | F2 | F3 | Mean | PACO-Part Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SANSA | 36.29 | | | | | |
| SANSA + hflip | 37.21 | | | | | |
| SANSA + UQ-gated hflip | 37.49 | 65.27 | 38.49 | 56.65 | 49.48 | |
| SANSA + BRM | | | | | | |
| SANSA + BRM + UQ-gated hflip | 37.29 | 66.00 | 38.64 | 56.83 | 49.69 | 43.72 |

For current Pascal-Part UQ-gated hflip folds, label the gate as:

> UQ head trained on Pascal-Part fold0 1-shot class split, used as a cross-fold gate.

This is acceptable as a first complete run, but if results become central to the paper, fold-specific or leave-one-fold-out UQ heads should be considered.

## 7. Current Priorities

### Priority 1: Finish Pascal-Part UQ-Gated Hflip 4 Folds

Done:

- fold0-3, threshold `0.3`: mean `49.48 / 69.83`, triggered `2415/8459`.

Purpose:

- complete Module A no-training gate evidence;
- decide whether UQ-gated hflip is consistently better than pure hflip;
- choose whether to spend time porting / rerunning BRM in this branch.

### Priority 2: Combine With BRM

BRM should be treated as a trainable refinement branch.

Options:

Current branch supports `--boundary_refine`, so evaluate:

- SANSA + BRM;
- SANSA + BRM + hflip;
- SANSA + BRM + UQ-gated hflip.

Recommended first combination:

> BRM always + UQ-gated hflip.

Reason:

- BRM is the trainable boundary-aware mask refinement branch.
- UQ controls the additional hflip consistency refinement.
- This avoids incorrectly describing BRM as pure TTA.

Checkpoint note:

- Use `/data6/chensq/SANSA_M/UncSANSA/output/train_brm_stage2/checkpoint0002.pth` for the first Pascal-Part BRM pass.
- The file is large because it is a full training checkpoint containing adapter weights, BRM weights, optimizer, scheduler, and args. The BRM itself is tiny: 6 tensors, about 7K trainable parameters.

### Priority 3: Implement Module B In Official Inference

Do this protocol-agnostically so it works on both generalist and strict FSS.

Minimum viable strategies:

- `adaptive-k`: select reliable supports, fall back to all supports when confidence / margin is weak;
- or `weighted logits`: run supports individually, fuse logits by uncertainty-derived reliability.

First debug on Pascal-Part fold0 5-shot, then immediately move to COCO-20i fold0 5-shot.

Current smoke result, official `inference_fss.py` path:

| Dataset | Fold | Shots | Episodes | Method | mIoU | FB-IoU | Extra |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Pascal-Part | 0 | 5 | 50 | SANSA all-support baseline | 45.84 | 66.22 | - |
| Pascal-Part | 0 | 5 | 50 | UQ-weighted logits | 47.43 | 70.61 | fallback 0/50, mean support score 0.415 |
| COCO-20i | 0 | 5 | 50 | UQ-weighted logits | 58.47 | 80.55 | fallback 0/50, mean support score 0.540 |

Read:

- Pascal-Part smoke is a strong first Module B signal: `+1.59` mIoU and `+4.39` FB-IoU over the 50-episode all-support baseline.
- COCO-20i smoke confirms the strict-FSS path runs, but needs the matched 50-episode SANSA baseline before judging improvement.
- Fallback did not trigger at margin `0.03`; keep an eye on whether this remains true on full folds.

### Priority 4: Strict FSS Main-Line Runs

Once Module B works:

1. COCO-20i fold0 5-shot SANSA vs Module B vs A+B;
2. COCO-20i fold0 1-shot SANSA vs Module A;
3. expand to fold0-3;
4. FSS-1000 sanity.

## 8. Canonical Commands

Run directory:

```bash
cd /data6/chensq/UFSAM2/UFSAM2
conda activate sam2coco
export MPLCONFIGDIR=/tmp/matplotlib
```

Pascal-Part generalist baseline:

```bash
python inference_fss.py \
  --dataset_file pascal_part --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_pascal_part_f0_1shot_baseline_full
```

Pascal-Part hflip:

```bash
python inference_fss.py \
  --dataset_file pascal_part --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_pascal_part_f0_1shot_hflip_full \
  --hflip_tta
```

Pascal-Part UQ-gated hflip:

```bash
python inference_fss.py \
  --dataset_file pascal_part --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_pascal_part_f0_1shot_uq_hflip_t03_full \
  --uq_hflip_tta \
  --uq_head_ckpt output/uncertainty_head_pascal_part_fold0_1shot_class_split/uncertainty_head.pt \
  --uq_gate_threshold 0.3
```

Pascal-Part BRM always + UQ-gated hflip:

```bash
python inference_fss.py \
  --dataset_file pascal_part --prompt mask --shots 1 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.8 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume /data6/chensq/SANSA_M/UncSANSA/output/train_brm_stage2/checkpoint0002.pth \
  --name_exp eval_pascal_part_f0_1shot_brm_uq_hflip_t03_full \
  --boundary_refine \
  --uq_hflip_tta \
  --uq_head_ckpt output/uncertainty_head_pascal_part_fold0_1shot_class_split/uncertainty_head.pt \
  --uq_gate_threshold 0.3
```

COCO-20i strict FSS skeleton:

```bash
python inference_fss.py \
  --dataset_file coco --prompt mask --shots 5 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.3 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/coco-20i-4/adapter_coco_fold0.pth \
  --name_exp eval_coco_f0_5shot_sansa
```

Module B first runnable path:

```bash
python inference_fss.py \
  --dataset_file coco --prompt mask --shots 5 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.3 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/coco-20i-4/adapter_coco_fold0.pth \
  --name_exp eval_coco_f0_5shot_uq_weighted_logits \
  --support_agg weighted_logits \
  --support_uq_head_ckpt output/uncertainty_head_mixed_tokens_match_fss_pascal_paco/uncertainty_head.pt
```

Current Module B implementation status:

- `--support_agg weighted_logits` runs official `inference_fss.py` evaluation.
- For each 5-shot episode, it decodes each support independently, scores support-query reliability with an expected-IoU head, and fuses query logits with reliability weights.
- If support scores are nearly tied, it falls back to original all-support SANSA logits.
- This is the first paper-facing Module B path; top1 support selection remains diagnostic only.
- `--support_agg` is not yet combined with `--uq_hflip_tta`; evaluate Module B alone first, then wire A+B deliberately.

## 9. Do Not Drift

- 主表必须来自 `inference_fss.py` official evaluation。
- 不要用 support-selection avg IoU 替代 mIoU / FB-IoU。
- 不要只证明 uncertainty head 有 calibration；必须让它改变预测并提升 official metric。
- 不要把 Pascal-Part generalist 写成最终 strict FSS 主表。
- 不要提交 checkpoints、cache、output、datasets。
- 阶段性实验结论写回 `PROCESS.md` 或本文件，并 git commit。

One-line reminder:

> Diagnostic uncertainty is only evidence; official FSS improvement is the result.
