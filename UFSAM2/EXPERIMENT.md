# EXPERIMENT: Uncertainty-Guided SAM2 Few-Shot Segmentation

当前分支：`exp/uncertainty-guided-fss`

核心目标：把 uncertainty 从诊断信号变成能提升 official FSS `mIoU / FB-IoU` 的 SANSA 改进模块。最终主表必须是标准 FSS 表，而不是 support-selection 诊断表。

## 1. 主线设定

最终论文表格需要：

- `1-shot / 5-shot`
- `fold0 / fold1 / fold2 / fold3 / mean`
- official `mIoU / FB-IoU`
- 主结论：`Ours` 超过 `SANSA baseline`

当前采用两条实验线：

| 实验线 | 数据 / 权重 | 作用 | 主表地位 |
| --- | --- | --- | --- |
| Generalist part segmentation | Pascal-Part / PACO-Part, `adapter_generalist.pth`, `channel_factor=0.8` | Module A 辅助证明：part ambiguity、边界、小部件 | 辅助表 |
| Strict FSS | COCO-20i fold weights, FSS-1000, possibly Pascal-5i | 最终标准 FSS 主表，尤其 5-shot Module B / A+B | 主结论 |

重要约束：

- support-selection avg IoU、calibration、AUROC、risk curve 只能作为辅助分析。
- 最终结果必须来自 `inference_fss.py` official evaluation。
- Pascal-Part / PACO-Part generalist 不能替代 strict FSS 主表。

## 2. 方法故事

### Module A: Uncertainty-Guided Ambiguity Refinement

目标：处理边界模糊、部件边缘不准、小部件断裂和高风险 query prediction。

当前实现：

- `--hflip_tta`: 纯 hflip TTA，作为 no-training refinement baseline。
- `--uq_hflip_tta`: 先用 expected-IoU head 判断风险，只在低置信 episode 触发 hflip。
- `--boundary_refine`: 打开 BRM，BRM 是 trainable boundary refinement branch。

推荐表述：

> BRM is a trainable boundary-aware mask refinement branch, hflip is a test-time consistency refinement, and uncertainty controls when to trigger the extra refinement.

不要把 BRM 写成 frozen uncertainty baseline，也不要把 BRM 说成 TTA。

### Module B: Uncertainty-Guided Support Reliability And Aggregation

目标：解决 5-shot 中 support 质量不均的问题。SANSA 默认把多个 support 等价写入 memory，低质量 support 可能污染语义或边界。

当前实现：

- `--support_agg weighted_logits`
- 每个 support 单独 forward，得到 query logits。
- 用 support-query expected-IoU score 估计 reliability。
- 按 reliability 融合 logits。
- 当 support 分数过于接近或过低时可 fallback 到 all-support SANSA。

当前结论：裸 weighted logits 在 Pascal-Part smoke 正向，但 COCO-20i smoke 的 mIoU 负向；加入更保守的 fallback 后，COCO-20i fold0 smoke 转为正向。目前 `support_fallback_margin=0.20` 已完成 COCO-20i full fold0-2，fold3 待补。

## 3. 当前代码状态

已接入 official `inference_fss.py`：

| 功能 | 参数 | 状态 |
| --- | --- | --- |
| pure hflip | `--hflip_tta` | 已实现 |
| UQ-gated hflip | `--uq_hflip_tta --uq_head_ckpt ... --uq_gate_threshold ...` | 已实现 |
| BRM | `--boundary_refine` | 已实现，需使用带 `brm.*` 的 checkpoint |
| Module B weighted logits | `--support_agg weighted_logits --support_uq_head_ckpt ...` | 已实现第一版 |
| smoke episode cap | `--max_eval_episodes` | 已实现 |

当前限制：

- `--support_agg` 暂未和 `--uq_hflip_tta` 合并。
- 裸 weighted logits 不适合直接跑 full COCO；带 fallback 的 `margin=0.20` 版本是当前 Module B 主候选。

## 4. Official Results

### 4.1 Pascal-Part Fold0 1-shot

设置：generalist adapter，`channel_factor=0.8`，official `inference_fss.py`。

| Method | Threshold | Triggered | mIoU | FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| SANSA baseline | - | - | 36.29 | 64.26 |
| SANSA + hflip | - | 2500/2500 | 37.21 | 65.16 |
| SANSA + UQ-gated hflip | 0.3 | 1017/2500 | 37.49 | 65.24 |
| SANSA + UQ-gated hflip | 0.4 | 1235/2500 | 37.49 | 65.31 |

读法：UQ-gated hflip 在 fold0 上超过 pure hflip；`t=0.3` 和 `t=0.4` mIoU 相同，`t=0.3` 更省。

### 4.2 Pascal-Part 4-fold Module A

UQ-gated hflip，threshold `0.3`：

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.49 | 65.24 |
| 1 | 959 | 112 | 65.27 | 72.39 |
| 2 | 2500 | 908 | 38.49 | 65.63 |
| 3 | 2500 | 378 | 56.65 | 76.06 |
| mean | - | - | 49.48 | 69.83 |

BRM always + UQ-gated hflip，threshold `0.3`：

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.29 | 64.70 |
| 1 | 959 | 112 | 66.00 | 72.83 |
| 2 | 2500 | 908 | 38.64 | 65.53 |
| 3 | 2500 | 378 | 56.83 | 76.09 |
| mean | - | - | 49.69 | 69.79 |

读法：

- UQ-gated hflip trigger rate: `2415/8459 = 28.6%`。
- BRM+UQ 相比 UQ-only：`+0.21` mIoU，`-0.04` FB-IoU。
- 相比旧 BRM+unconditional hflip (`49.61 / 69.82`)，BRM+UQ 基本持平，但只在 28.6% episodes 上跑 hflip。
- Module A 的主说法应是“uncertainty-controlled refinement efficiency with comparable/slightly better mIoU”，不要夸成大幅 accuracy gain。

### 4.3 PACO-Part 4-fold Module A

BRM always + UQ-gated hflip，threshold `0.3`：

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 991 | 41.59 | 67.84 |
| 1 | 2500 | 856 | 45.49 | 66.83 |
| 2 | 2500 | 767 | 46.42 | 66.20 |
| 3 | 2500 | 846 | 41.36 | 64.47 |
| mean | - | - | 43.72 | 66.34 |

读法：

- Trigger rate: `3460/10000 = 34.6%`。
- 旧 PACO BRM + unconditional hflip 原始 log: `43.65 / 66.44`。
- 当前 BRM+UQ: `43.72 / 66.34`，mIoU `+0.07`，FB-IoU `-0.10`。
- PACO 结论应写成“基本持平，省掉约三分之二 hflip”，而不是全面更好。

### 4.4 Module B Official-Path Smoke

| Dataset | Fold | Shots | Episodes | Method | mIoU | FB-IoU | Extra |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Pascal-Part | 0 | 5 | 50 | SANSA all-support baseline | 45.84 | 66.22 | - |
| Pascal-Part | 0 | 5 | 50 | UQ-weighted logits | 47.43 | 70.61 | fallback 0/50, mean score 0.415 |
| COCO-20i | 0 | 5 | 50 | SANSA all-support baseline | 59.83 | 79.73 | - |
| COCO-20i | 0 | 5 | 50 | UQ-weighted logits, margin 0.03 | 58.47 | 80.55 | fallback 0/50, mean score 0.540 |
| COCO-20i | 0 | 5 | 50 | UQ-weighted logits, margin 0.10 | 60.43 | 81.37 | fallback 12/50, mean margin 0.249 |
| COCO-20i | 0 | 5 | 50 | UQ-weighted logits, margin 0.20 | 60.89 | 81.90 | fallback 29/50, mean margin 0.249 |
| COCO-20i | 0 | 5 | 50 | UQ-weighted logits, min score 0.60 | 57.79 | 78.97 | fallback 16/50 |
| COCO-20i | 0 | 5 | 50 | UQ-weighted logits, margin 0.10 + min score 0.60 | 59.85 | 79.91 | fallback 25/50 |

读法：

- Pascal-Part 5-shot smoke 正向：`+1.59` mIoU，`+4.39` FB-IoU。
- COCO-20i 裸 weighted logits 主指标负向：`-1.36` mIoU，`+0.82` FB-IoU。
- 加入 fallback 后，`margin=0.20` 最好：相对 baseline `+1.06` mIoU，`+2.17` FB-IoU。
- `min_score=0.60` 不适合作为当前规则，会显著伤害 mIoU。
- 当前已进入 COCO full evaluation。

### 4.5 COCO-20i 5-shot Module B Full

设置：COCO-20i fold adapters，`shots=5`，`--support_agg weighted_logits --support_fallback_margin 0.20`，official `inference_fss.py`。

| Fold | Episodes | mIoU | FB-IoU | Fallback | Mean score | Mean max score | Mean margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1000 | 63.45 | 78.15 | 458/1000 | 0.534 | 0.656 | 0.280 |
| 1 | 1000 | 65.36 | 80.49 | 459/1000 | 0.557 | 0.680 | 0.273 |
| 2 | 1000 | 66.18 | 82.11 | 583/1000 | 0.589 | 0.688 | 0.228 |
| 0-2 mean | - | 65.00 | 80.25 | 1500/3000 | 0.560 | 0.675 | 0.260 |
| 3 | pending | | | | | | |

读法：

- Module B full evaluation 已完成 fold0-2，三折均值 `65.00 / 80.25`。
- Fallback rate 为 `1500/3000 = 50.0%`，说明当前方法不是裸替换 SANSA，而是约一半 episode 回退到 all-support baseline。
- fold3 跑完后再形成正式 4-fold mean，并与 SANSA paper / official baseline 表对比。

## 5. 诊断结果摘要

这些结果只支持方法动机，不作为主表：

| Dataset | 结论 |
| --- | --- |
| FSS-1000 | near ceiling；token head 有风险预测信号，但主提升空间小 |
| Pascal-Part support selection | token consistently beats random/SAM-score and roughly ties all-supports; oracle gap large |
| PACO-Part support selection | mixed/tokens-match improves diagnostics, but top1 intervention 不稳定 |
| memory propagation risk | SANSA official baseline 不使用 query-to-query sequential propagation；该分支暂停 |

关键判断：Module B 不能停在 top1 support selection，必须落到 official FSS aggregation。

## 6. 目标表格

### Strict FSS Main Table

这是最终主结论表。

| Method | 1-shot F0 | F1 | F2 | F3 | Mean | 5-shot F0 | F1 | F2 | F3 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SANSA | | | | | | | | | | |
| SANSA + Module A | | | | | | | | | | |
| SANSA + Module B | - | - | - | - | - | | | | | |
| Ours A+B | | | | | | | | | | |

优先级：

1. COCO-20i 5-shot fold0：先找出不伤 mIoU 的 Module B。
2. COCO-20i 4 folds：主表。
3. FSS-1000：sanity / near-ceiling check。
4. Pascal-5i：仅在 protocol 和权重足够干净时补。

### Generalist Part Segmentation Auxiliary Table

| Method | Pascal-Part F0 | F1 | F2 | F3 | Mean | PACO-Part Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SANSA | 36.29 | | | | | |
| SANSA + hflip | 37.21 | | | | | |
| SANSA + UQ-gated hflip | 37.49 | 65.27 | 38.49 | 56.65 | 49.48 | |
| SANSA + BRM + UQ-gated hflip | 37.29 | 66.00 | 38.64 | 56.83 | 49.69 | 43.72 |

注意：当前 Pascal/PACO UQ head 是 fold0 trained、cross-fold used。辅助表可以接受；若作为强 claim，需要 fold-specific 或 leave-one-fold-out head。

## 7. 当前下一步

Module A:

- Pascal-Part / PACO-Part generalist line 已足够收口。
- 不继续调 PACO threshold，除非 Module B 长期卡住。

Module B:

1. 补完 COCO fold3 5-shot full `--support_fallback_margin 0.20`。
2. 用 SANSA paper / official table 作为 matched baseline，对齐 4-fold mean。
3. 如果 4-fold mean 正向，再考虑 A+B 或 1-shot Module A strict FSS。
4. 如果 fold3 拉低明显，再实现更保守的 `adaptive-k + fallback` 或训练 COCO-specific support reliability head。

## 8. 常用命令骨架

运行目录：

```bash
cd /data6/chensq/UFSAM2/UFSAM2
conda activate sam2coco
export MPLCONFIGDIR=/tmp/matplotlib
```

COCO-20i 5-shot baseline smoke：

```bash
python inference_fss.py \
  --dataset_file coco --prompt mask --shots 5 --fold 0 \
  --sam2_version large --adaptformer_stages 2 3 --channel_factor 0.3 \
  --device cuda --data_root /data6/chensq/datasets \
  --resume pretrain/coco-20i-4/adapter_coco_fold0.pth \
  --name_exp eval_coco_f0_5shot_baseline_smoke \
  --max_eval_episodes 50
```

COCO-20i 5-shot Module B smoke：

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

## 9. 不要偏航

- 主表必须是 official FSS `mIoU / FB-IoU`。
- 不要把 support-selection avg IoU 当主结果。
- 不要把 calibration/AUROC 当主结果。
- 不要把 Pascal-Part generalist 写成最终 strict FSS。
- 不要提交 checkpoints、cache、output、datasets。
- 阶段性结论更新 `PROCESS.md` 或本文件并 commit。

一句话：

> Diagnostic uncertainty is only evidence; official FSS improvement is the result.
