# EXPERIMENT: Uncertainty-Guided SAM2 Few-Shot Segmentation

> [!WARNING]
> **HISTORICAL / 非当前权威记录**
> 本文件仅保留 baseline restart 前的实验历史，用于假设生成和调试。不得将其作为当前 experiment contract、状态、baseline、结果或裁决的 authority。当前实验索引见 [`EXPERIMENT_LEDGER.md`](../EXPERIMENT_LEDGER.md)，单实验权威记录见 [`experiments/`](../experiments/)。

> **面向对象：** 人类研究者审阅；正文使用中文，必要的 academic/technical terms 保留英文。

当前分支：`exp/uncertainty-guided-fss`

核心目标：把 uncertainty 从诊断信号变成能提升 official FSS `mIoU / FB-IoU` 的 SANSA 改进模块。最终主表必须是标准 FSS 表，而不是 support-selection 诊断表。

## 1. 实验组织

后续所有实验按两级组织：

1. **Strict FSS**
   - 数据：COCO-20i、FSS-1000、可能补 Pascal-5i。
   - 权重：fold-specific FSS adapters，例如 `pretrain/coco-20i-4/adapter_coco_fold{0..3}.pth`。
   - 作用：论文主表；必须对齐 SANSA paper / official baseline。
2. **Generalist In-context**
   - 数据：Pascal-Part、PACO-Part。
   - 权重：`pretrain/adapter_generalist.pth`，`channel_factor=0.8`。
   - 作用：和 SANSA generalist setting 对齐，主要支撑 part ambiguity / boundary / support reliability 的辅助结论。

每个设置内部再分：

- **Baseline**：SANSA 原始设置或 paper/README 可直接引用的 official 数字。
- **Module A**：Uncertainty-Guided Ambiguity Refinement，包括 hflip、UQ-gated hflip、BRM。
- **Module B**：Uncertainty-Guided Support Reliability and Aggregation，包括 weighted logits、adaptive-k、fallback。
- **Module C**：Uncertainty-Triggered Query Self-Prompting，包括 memory-to-point automatic prompts。
- **A+B**：Module A 和 Module B 的最终组合。

参数变体必须挂在对应模块下面，例如 `uq_gate_threshold=0.3` 属于 Module A，`support_fallback_margin=0.20` 属于 Module B。

## 2. Strict FSS

### 2.1 Baseline

Strict FSS baseline 优先使用 SANSA paper / official table，因为最终主表要和 SANSA 对齐。

| Dataset | Shot | SANSA mean mIoU | Notes |
| --- | ---: | ---: | --- |
| COCO-20i | 1-shot | 60.2 | SANSA paper / official |
| COCO-20i | 5-shot | 64.3 | SANSA paper / official |
| FSS-1000 | 1-shot | 91.4 | SANSA paper / official |
| FSS-1000 | 5-shot | 92.1 | SANSA paper / official |

主表格式必须包含 `fold0 / fold1 / fold2 / fold3 / mean`，并在可获得时同时报告 `mIoU / FB-IoU`。

### 2.2 Module A

Strict FSS 的 Module A 尚未形成正式主结果。当前 Module A 证据主要来自 Generalist In-context 的 Pascal-Part / PACO-Part。

可尝试方向：

- COCO-20i 1-shot：`hflip` vs `UQ-gated hflip`。
- COCO-20i 5-shot：在 Module B 稳定后再考虑 A+B。

### 2.3 Module B

当前 official-path prototype：`--support_agg weighted_logits`，使用 support-query expected-IoU score 做 support reliability，加权融合每个 support 的 query logits，并用 fallback 控制风险。

#### COCO-20i 5-shot smoke

| Fold | Episodes | Method | mIoU | FB-IoU | Extra |
| ---: | ---: | --- | ---: | ---: | --- |
| 0 | 50 | SANSA all-support baseline | 59.83 | 79.73 | local smoke |
| 0 | 50 | UQ-weighted logits, margin 0.03 | 58.47 | 80.55 | fallback 0/50 |
| 0 | 50 | UQ-weighted logits, margin 0.10 | 60.43 | 81.37 | fallback 12/50 |
| 0 | 50 | UQ-weighted logits, margin 0.20 | 60.89 | 81.90 | fallback 29/50 |
| 0 | 50 | UQ-weighted logits, min score 0.60 | 57.79 | 78.97 | fallback 16/50 |
| 0 | 50 | UQ-weighted logits, margin 0.10 + min score 0.60 | 59.85 | 79.91 | fallback 25/50 |

读法：`margin=0.20` 在 smoke 上最好，但 smoke 不能替代 full fold evaluation。

#### COCO-20i 5-shot full, margin 0.20

设置：`--support_agg weighted_logits --support_fallback_margin 0.20`。

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

当前判断：

- 这条 Module B 已经跑通 official path，但不能作为主结果。
- 它低于 SANSA 5-shot official baseline `64.3`，只能作为 ablation / implementation proof。
- 下一步应优先换成更强的 adaptive-k、COCO-specific reliability head，或更保守的 gating 策略。

### 2.4 Module C

当前 official-path prototype：Memory-to-Point Self-Prompting，开关为 `--memory_to_point_prompt`。

机制：

- 先运行正常的 memory-conditioned SANSA query decode；
- 用 logit stability 和 multimask disagreement 构造无监督 query quality score；
- 当 score 低于 `--mtp_trigger_threshold` 时，从 query mask 内部采样自动正点，从候选分歧/背景区域采样自动负点；
- 用同一个 memory-conditioned feature 加自动 point prompt 再跑一次 SAM2 head；
- 只有 second pass 的 quality score 不退化时才接受。

#### COCO-20i 5-shot full, Memory-to-Point

设置：fold-specific COCO adapters，`--memory_to_point_prompt`，默认 `1` 个正点 + `1` 个负点。

| Setting | F0 mIoU | F1 mIoU | F2 mIoU | F3 mIoU | Mean mIoU | Mean FB-IoU | Triggered | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP `t085` | 64.65 | 67.49 | 65.50 | 60.29 | 64.48 | 80.28 | 34/4000 | 30/34 |
| MTP `t090` | 64.66 | 67.44 | 65.58 | 60.17 | 64.46 | 80.31 | 79/4000 | 74/79 |
| MTP `t092` | 64.52 | 67.43 | 65.58 | 60.28 | 64.45 | 80.29 | 121/4000 | 113/121 |

Fold0 额外阈值压力测试：

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

当前判断：

- MTP 是低触发率的精准干预：`t085` 只触发 `34/4000 = 0.85%` episodes，但 mean mIoU 最高。
- 阈值越激进不一定越好；fold0 的 `t095` 触发过多，mIoU 掉到 `64.26`。
- 当前 strict-FSS 主线候选应优先放 MTP `t085`，但增益只有 `+0.18`，还需要本地 no-MTP 4-fold baseline 和 `--mtp_num_negative_points 0` 等 ablation 来稳住结论。

### 2.5 A+B

尚未形成正式组合结果。

当前代码中 `--support_agg` 暂未和 `--uq_hflip_tta` 合并；即使合并，也应等 Module B 本身不低于 SANSA 后再作为主线推进。

## 3. Generalist In-context

### 3.1 Baseline

设置：`adapter_generalist.pth`，`channel_factor=0.8`。

以下为 1-shot mIoU，按 `fold0 / fold1 / fold2 / fold3 / mean` 记录：

| Dataset | Shot | Weight | F0 | F1 | F2 | F3 | Mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Pascal-Part | 1-shot | `adapter_generalist.pth` | 36.29 | 65.16 | 38.20 | 56.77 | 49.105 |
| PACO-Part | 1-shot | `adapter_generalist.pth` | 40.27 | 44.20 | 46.09 | 41.24 | 42.95 |

注意：上表是 mIoU fold row，不包含 FB-IoU；不要把其中的 fold 数字当成 FB-IoU。

### 3.2 Module A

Module A 包含：

- pure hflip：`--hflip_tta`
- UQ-gated hflip：`--uq_hflip_tta --uq_gate_threshold ...`
- BRM：`--boundary_refine`，BRM 是 trainable boundary-aware mask refinement branch，不是 TTA。

推荐表述：

> BRM is a trainable boundary-aware mask refinement branch, hflip is a test-time consistency refinement, and uncertainty controls when to trigger the extra refinement.

#### Pascal-Part 1-shot, fold0

| Method | Threshold | Triggered | mIoU | FB-IoU |
| --- | ---: | ---: | ---: | ---: |
| SANSA baseline | - | - | 36.29 | 64.26 |
| SANSA + hflip | - | 2500/2500 | 37.21 | 65.16 |
| SANSA + UQ-gated hflip | 0.3 | 1017/2500 | 37.49 | 65.24 |
| SANSA + UQ-gated hflip | 0.4 | 1235/2500 | 37.49 | 65.31 |

读法：UQ-gated hflip 在 fold0 上超过 pure hflip；`t=0.3` 和 `t=0.4` mIoU 相同，`t=0.3` 更省。

#### Pascal-Part 1-shot, 4 folds

UQ-gated hflip，`uq_gate_threshold=0.3`：

| Fold | Episodes | Triggered | mIoU | FB-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2500 | 1017 | 37.49 | 65.24 |
| 1 | 959 | 112 | 65.27 | 72.39 |
| 2 | 2500 | 908 | 38.49 | 65.63 |
| 3 | 2500 | 378 | 56.65 | 76.06 |
| mean | - | - | 49.48 | 69.83 |

BRM always + UQ-gated hflip，`uq_gate_threshold=0.3`：

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

#### PACO-Part 1-shot, 4 folds

BRM always + UQ-gated hflip，`uq_gate_threshold=0.3`：

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

### 3.3 Module B

Generalist In-context 的 Module B 目前只有 Pascal-Part 5-shot smoke 正向，正式 4-fold 结果仍待补。

| Dataset | Fold | Shots | Episodes | Method | mIoU | FB-IoU | Extra |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Pascal-Part | 0 | 5 | 50 | SANSA all-support baseline | 45.84 | 66.22 | local smoke |
| Pascal-Part | 0 | 5 | 50 | UQ-weighted logits | 47.43 | 70.61 | fallback 0/50, mean score 0.415 |

当前判断：

- Pascal-Part 5-shot smoke 正向：`+1.59` mIoU，`+4.39` FB-IoU。
- 这可以支持 Module B 在 part generalist setting 下有潜力，但不能替代 strict FSS 主表。
- 若要写成正式 auxiliary result，需要跑 Pascal-Part / PACO-Part matched full 4-fold 5-shot baseline 和 Module B。

### 3.4 A+B

尚未形成正式结果。Generalist A+B 应在 Module B full 4-fold 正向后再跑，避免把不同设置和不同参数混成一个主张。

## 4. 代码状态

`inference_fss.py` 已支持：

| 功能 | 参数 | 状态 |
| --- | --- | --- |
| pure hflip | `--hflip_tta` | 已实现 |
| UQ-gated hflip | `--uq_hflip_tta --uq_head_ckpt ... --uq_gate_threshold ...` | 已实现 |
| BRM | `--boundary_refine` | 已实现，需使用带 `brm.*` 的 checkpoint |
| Module B weighted logits | `--support_agg weighted_logits --support_uq_head_ckpt ...` | 已实现第一版 |
| smoke episode cap | `--max_eval_episodes` | 已实现 |

当前限制：

- `--support_agg` 暂未和 `--uq_hflip_tta` 合并。
- 当前 `weighted_logits + margin 0.20` 低于 SANSA COCO-20i 5-shot official baseline，不能作为主方法定稿。

## 5. 下一步

Strict FSS：

1. 不把当前 `weighted_logits + margin 0.20` 作为主结果；它低于 SANSA official 5-shot mIoU `64.3`。
2. 优先改进 Module B：`adaptive-k + fallback`、COCO-specific support reliability head，或更保守的 reliability gate。
3. 每个 strict FSS 结果都必须显式列出 SANSA official baseline 与 delta。

Generalist In-context：

1. Module A 的 Pascal-Part / PACO-Part 1-shot 结果已经足够作为辅助故事。
2. 若继续推进 Generalist Module B，优先补 Pascal-Part / PACO-Part 5-shot full 4-fold matched baseline 和 Module B。
3. A+B 等 Module B 正向后再组合。

## 6. 不要偏航

- 主表必须是 official FSS `mIoU / FB-IoU`。
- 不要把 support-selection avg IoU 当主结果。
- 不要把 calibration/AUROC 当主结果。
- 不要把 Pascal-Part / PACO-Part generalist 写成最终 strict FSS。
- 不要提交 checkpoints、cache、output、datasets。
- 阶段性结论更新 `PROCESS.md` 或本文件并 commit。

一句话：

> Diagnostic uncertainty is only evidence; official FSS improvement is the result.
