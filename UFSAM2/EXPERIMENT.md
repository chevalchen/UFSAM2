# EXPERIMENT: Uncertainty-Guided SANSA

这个文件是给新窗口 / 后续实验交接用的。读完这里后，应能理解当前课题目标、主实验表、A/B 模块、已有证据、下一步应该实现和跑什么。

## 0. 核心目标

课题方向：

> 基于 SAM2 的小样本分割，在 SANSA baseline 上引入 UncertainSAM 风格的不确定性估计，解决 SANSA 在 support 可靠性、边界模糊、部件分割能力不足上的问题。

最终结果必须回到标准 FSS 评测表：

- `1-shot / 5-shot`
- `fold0 / fold1 / fold2 / fold3 / mean`
- official-style `mIoU / FB-IoU`
- 主要结论应是 `Ours` 超过 `SANSA baseline`

重要：support-selection avg IoU、calibration、AUROC、risk curve 都是过程证明和 ablation，不是最终主表。

## 1. 当前故事

SANSA 的强项：

- 把 FSS 转成 SAM2 pseudo-video semantic propagation；
- support image + GT mask 写入 memory；
- query image 通过 memory-conditioned decoder 得到 mask；
- mask-only 设置下 support prompt 已经很强。

SANSA 的问题：

- 不知道哪个 support 更可靠；
- 不知道 query prediction 什么时候高风险；
- 对边界模糊、小部件、part ambiguity 缺少自适应处理；
- 5-shot 时默认把多个 support 等价使用，低质量 support 可能污染语义。

UncertainSAM 给出的启发：

- SAM2 decoder token 中包含 uncertainty 信息；
- 可以 post-hoc 读取 `iou_token` / `mask_token`；
- 用轻量 MLP 预测 expected IoU / uncertainty；
- 不需要一开始 joint training。

本课题要做：

```text
SANSA / SAM2
    |
decoder tokens + memory / support traces
    |
uncertainty layer
    |
+-------------------------------+
| Module A                      |
| uncertainty-gated refinement  |
+-------------------------------+
| Module B                      |
| uncertainty-guided support    |
| reliability / aggregation     |
+-------------------------------+
    |
final FSS mask
```

## 2. 两个模块定义

### Module A: Uncertainty-Guided Ambiguity Refinement

目标问题：

- 边界模糊；
- 部件边缘不准；
- 小部件断裂；
- query 预测风险高但 SANSA 不知道是否需要额外处理。

候选动作：

- hflip TTA；
- BRM boundary refinement；
- uncertainty-gated hflip；
- uncertainty-gated BRM / hflip。

最终要进入主表的形式：

| Method | 训练额外模块 | 用 uncertainty | 预期作用 |
| --- | --- | --- | --- |
| SANSA | no | no | baseline |
| SANSA + hflip | no | no | no-training TTA |
| SANSA + BRM | yes | no | boundary refinement |
| SANSA + UQ-gated hflip | no | yes | 高风险样本才触发 TTA |
| SANSA + UQ-gated BRM/hflip | yes | yes | Module A 完整版 |

预期效果：

- hflip：小幅稳定收益，约 `+0.2 ~ +0.5 mIoU`；
- BRM+hflip：比 hflip 更强，但需要额外 checkpoint；
- UQ-gated hflip：目标不是一定超过 unconditional hflip，而是接近其收益并降低额外计算；
- UQ-gated BRM/hflip：中期报告可作为 “uncertainty-guided refinement” 主模块。

已有旁证来自 `TTA_SUM.md`：

PACO-Part fold0：

| setting | mIoU | FB-IoU |
| --- | ---: | ---: |
| plain generalist | 40.27 | 67.17 |
| generalist + hflip | 40.70 | 67.52 |
| BRM + hflip | 41.24 | 67.74 |

PACO-Part 4-fold average：

| setting | mIoU | FB-IoU |
| --- | ---: | ---: |
| generalist + hflip | 43.22 | 66.29 |
| BRM + hflip | 43.65 | 66.44 |

注意：

- hflip/BRM 本身不是 uncertainty；
- 应讲成：uncertainty 作为 gate，决定是否触发 refinement；
- BRM 是 trainable branch，不要和 frozen uncertainty baseline 混在一起。

### Module B: Uncertainty-Guided Support Reliability And Aggregation

目标问题：

- 5-shot 中不同 support 质量差异大；
- SANSA 默认多 support 等价进入 memory；
- 在 part segmentation 中，错误 support 可能带来错误部件语义或边界。

已有实现基础：

- `collect_uncertainty_cache.py` 已收集 query/support trace；
- `train_uncertainty_head.py` 已支持 `tokens` 和 `tokens_match`；
- `evaluate_support_selection.py` 已支持 support-selection 诊断；
- `evaluate_support_selection.py --official_metrics` 可输出 SANSA-style mIoU / FB-IoU。

不要只停在 top1 support selection。模块 B 最终应升级为 support aggregation：

| 子策略 | 说明 | 是否最终主表候选 |
| --- | --- | --- |
| UQ-top1 | 选 expected IoU 最高的 support | ablation |
| UQ-topk | 选 top-k reliable supports 后重新跑 SANSA | yes |
| UQ-adaptive-k | 根据 top1 分数和 top1-top2 margin 动态选 k | yes |
| UQ-weighted logits | 每个 support 单独预测，按 reliability 加权融合 logits | yes |
| UQ-fallback | 不确定时退回 all-supports | yes |

推荐最终说法：

> We do not treat all support examples as equally reliable. Instead, we estimate support-query reliability from SAM2 decoder traces and use it to adaptively select, aggregate, or fall back among supports.

预期效果：

- 1-shot：Module B 基本 no-op，不应作为 1-shot 主要提升来源；
- 5-shot：Module B 应超过或至少稳定持平 all-supports，并降低 fail/risk；
- 如果 UQ-top1 不稳，则应使用 adaptive-k / weighted logits / fallback。

## 3. 最终主表应该长什么样

### Table 1: Strict Few-Shot Segmentation Setting

主表必须使用 official-style evaluation，不要用 support-selection avg IoU 顶替。

模板：

| Method | 1-shot F0 | F1 | F2 | F3 | Mean | 5-shot F0 | F1 | F2 | F3 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SANSA | | | | | | | | | | |
| SANSA + hflip | | | | | | | | | | |
| SANSA + BRM | | | | | | | | | | |
| SANSA + Module A | | | | | | | | | | |
| SANSA + Module B | - | - | - | - | - | | | | | |
| Ours A+B | | | | | | | | | | |

数据集优先级：

1. Pascal-Part：自构造 object+part class-disjoint FSS，最贴合部件分割故事。
2. PACO-Part：长尾、细粒度、support 质量差异强，适合压力测试。
3. FSS-1000：sanity check，接近天花板，不作为主提升证据。
4. COCO-20i：更权威标准 FSS benchmark，如果时间允许应加入。

### Table 2: Against Generalist In-Context Models

用于对齐 SANSA paper 的 Table 2 风格。

模板：

| Method | Type | Pascal-Part | PACO-Part | FSS-1000 |
| --- | --- | ---: | ---: | ---: |
| SAM2 + mask prompt | generalist | | | |
| PerSAM / Matcher / SegGPT 等 | in-context | | | |
| SANSA | FSS baseline | | | |
| Ours | uncertainty-guided SANSA | | | |

中期报告如果没有复现其它模型，不要硬填。可以写：

- SANSA baseline：已复现；
- Ours A / B / A+B：正在补；
- 其它模型：后续按 SANSA paper reported numbers 或 matched protocol 补充。

## 4. 过程证明放在哪里

以下实验是 auxiliary analysis，不是主表：

### 4.1 Expected-IoU / Failure-Risk Calibration

作用：

- 证明 decoder tokens 有 uncertainty 信号；
- 证明 UncertainSAM 思路能迁移到 SANSA；
- 指标：MAE、Pearson、Spearman、AUROC、ECE、risk-coverage。

### 4.2 Support Selection Diagnostic

作用：

- 证明 uncertainty 能识别 support reliability；
- 比较 random / SAM-score / token / all-supports / oracle；
- 解释 Module B 为什么合理。

注意：

- 这里的 `avg IoU` 是 episode-average binary IoU；
- 不能和 official SANSA fold mIoU 直接比较；
- 最终主表必须用 official mIoU / FB-IoU。

### 4.3 Risk / Coverage / Qualitative Boundary Analysis

作用：

- 证明高 uncertainty 样本确实更容易失败；
- 证明 Module A gate 不是任意阈值；
- 展示边界模糊、小部件断裂、part confusion 的可视化。

## 5. 当前已有结果摘要

### FSS-1000

SANSA 接近天花板。

5-shot support-selection avg IoU：

| strategy | avg IoU |
| --- | ---: |
| random | 0.9138 |
| SAM-score | 0.9170 |
| token | 0.9211 |
| all-supports | 0.9205 |
| oracle | 0.9353 |

结论：

- token uncertainty 有信号；
- 但 FSS-1000 不适合证明主要提升。

### Pascal-Part

当前最强 positive evidence。

5-shot support-selection avg IoU，3 seeds full 2500：

| strategy | avg IoU |
| --- | ---: |
| random | 0.3938 |
| SAM-score | 0.4274 |
| token | 0.4620 |
| all-supports | 0.4615 |
| oracle | 0.5578 |

结论：

- uncertainty support reliability 明显超过 random / SAM-score；
- 与 all-supports 持平，risk 更低；
- oracle 仍很高，说明 adaptive aggregation 还有空间。

### PACO-Part

压力测试，目前不是最终胜利。

2 seeds x 200 episodes：

| strategy | avg IoU | fail<0.5 | risk<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.4520 | 54.3% | 71.8% |
| SAM-score | 0.4830 | 49.3% | 65.3% |
| mixed token | 0.4949 | 48.0% | 65.8% |
| tokens_match | 0.4945 | 47.3% | 65.8% |
| all-supports | 0.4948 | 48.5% | 66.5% |
| oracle | 0.5834 | 36.8% | 58.0% |

结论：

- mixed / tokens_match 提升 calibration；
- support-selection intervention 不稳定；
- 需要 adaptive-k / weighted logits / fallback，而不是只做 top1。

## 6. 实验协议和权重

运行目录：

```bash
cd /data6/chensq/UFSAM2/UFSAM2
```

环境：

```bash
conda activate sam2coco
export MPLCONFIGDIR=/tmp/matplotlib
export CUDA_VISIBLE_DEVICES=1
```

注意：

- 设置 `CUDA_VISIBLE_DEVICES=1` 后，程序内部用 `--device cuda`，不要写 `cuda:1`；
- 数据路径：`/data6/chensq/datasets`；
- 权重路径：`pretrain/`。

已知权重：

| 用途 | 权重 | adapter config |
| --- | --- | --- |
| FSS-1000 | `pretrain/adapter_fss_fold0.pth` | stages `2 3`, channel `0.3` |
| Pascal-Part / PACO-Part | `pretrain/adapter_generalist.pth` | stages `2 3`, channel `0.8` |
| SANSA universal | `pretrain/adapter_sansa_universal.pth` | 待核对 |
| COCO-20i fold0-3 | `pretrain/coco-20i-4/adapter_coco_fold{0..3}.pth` | likely stages `2 3`, channel `0.3` |

## 7. 下一步优先级

### Setting Decision: Generalist vs Strict FSS

结论：

- Pascal-Part / PACO-Part with `adapter_generalist.pth` 适合继续作为 Module A 和 part ambiguity 的主要验证场景。
  - 理由：SANSA paper 的 generalist in-context setting 本身包含 Pascal-Part / PACO-Part part segmentation；这里最贴合边界模糊、部件断裂、part ambiguity。
  - 当前 Pascal-Part fold0 1-shot full 已经证明 UQ-gated hflip 比 pure hflip 更好，同时只触发约一半 episodes。
- 但最终 A+B 主结果不能只停在 generalist part setting。
  - 论文主目标仍是标准 strict FSS 表：`1-shot / 5-shot / fold0-3 / mean / official mIoU-FB-IoU`。
  - SANSA paper 的 strict FSS Table 1 明确包含 `1-shot` 和 `5-shot`，对应 COCO-20i / LVIS-92i / FSS-1000。
  - Pascal-Part / PACO-Part 在 SANSA paper 中更偏 generalist in-context / one-shot part segmentation evidence，适合作为 Table 2 或 auxiliary part-seg evidence。
- 因此不需要推倒重来，但需要双线推进：
  1. **Generalist part line**：继续用 Pascal-Part fold0 完成 Module A、Module B、A+B 的快速闭环和 qualitative/risk 分析。
  2. **Strict FSS main line**：把已经实现的 A/B 策略接到 strict FSS 权重和 official evaluation 上，优先 COCO-20i fold0 5-shot，再扩 fold0-3；FSS-1000 作为 sanity。

当前定位：

| Experiment line | Dataset / weight | Role | Can support final main claim? |
| --- | --- | --- | --- |
| Generalist part | Pascal-Part / PACO-Part, `adapter_generalist.pth`, `channel_factor=0.8` | Module A proof, part ambiguity, qualitative/risk | no, auxiliary / Table 2 style |
| Strict FSS | COCO-20i fold weights, `channel_factor=0.3` | main standard FSS table, 1-shot/5-shot/folds | yes |
| FSS-1000 | `adapter_fss_fold0.pth`, `channel_factor=0.3` | sanity / near-ceiling check | weak as main evidence |

Practical next step:

- Do not abandon the current Pascal-Part generalist results; they are useful and already positive for Module A.
- Do not make the final A+B claim only on Pascal-Part generalist.
- Implement Module B aggregation in a protocol-agnostic way inside the official inference path, then run it first on Pascal-Part fold0 5-shot for debugging and immediately on strict FSS COCO-20i fold0 5-shot for the main-table direction.

### Step 1: 标准主表评测脚本接入 Module A / B

当前 `inference_fss.py` 是 official evaluation 入口。

需要做：

- 增加 `--hflip_tta`；
- 增加 Module A 的 UQ-gated hflip 开关；
- 如要用 BRM，单独加 `--boundary_refine` 和 checkpoint；
- 增加 Module B 的 support aggregation 策略；
- 输出 official mIoU / FB-IoU。

验收标准：

- `SANSA baseline` 和新脚本原始结果一致；
- 开关关闭时完全不改变 baseline；
- 1-shot / 5-shot 都能跑。

### Step 2: Pascal-Part fold0 小闭环

先跑最小闭环，不要一开始跑全量 4 folds。

要跑：

| Setting | shot | fold | 目的 |
| --- | ---: | ---: | --- |
| SANSA baseline | 1, 5 | 0 | 对齐 baseline |
| SANSA + hflip | 1, 5 | 0 | Module A no-training |
| SANSA + UQ-gated hflip | 1, 5 | 0 | 验证 gate |
| SANSA + Module B adaptive/weighted | 5 | 0 | 验证 support aggregation |
| Ours A+B | 1, 5 | 0 | 主方法雏形 |

预期：

- 1-shot：主要看 Module A；
- 5-shot：Module A+B 应超过 SANSA baseline；
- 如果 Module B top1 不稳，直接切 adaptive-k / weighted logits。

### Step 3: Pascal-Part 4 folds

fold0 小闭环有效后，再跑 fold0-3。

报告：

- 1-shot F0/F1/F2/F3/Mean；
- 5-shot F0/F1/F2/F3/Mean；
- mIoU 为主，FB-IoU 可放附表。

### Step 4: PACO-Part 4 folds

作为压力测试。

预期：

- PACO 不一定大幅提升；
- 只要 A+B 比 SANSA 稳定正收益，且 risk / qualitative 更好，就能支撑故事；
- 如果不稳，报告为 long-tail limitation，并强调 adaptive policy。

### Step 5: FSS-1000 sanity

只需要 1-shot / 5-shot mean，不必强调大提升。

目的：

- 证明方法不会破坏 near-ceiling 数据集；
- 如果小幅提升，可以作为补充；
- 如果持平，也可以接受。

### Step 6: COCO-20i 标准 benchmark

如果时间允许，加入 COCO-20i 4-fold。

优点：

- 更权威；
- 与 SANSA paper 严格 FSS setting 更接近；
- 本地已有 per-fold adapter。

最小命令骨架见第 8 节。

## 8. 常用命令骨架

### 8.1 SANSA official baseline

Pascal-Part / PACO-Part 用 generalist：

```bash
MPLCONFIGDIR=/tmp/matplotlib python inference_fss.py \
  --dataset_file pascal_part \
  --prompt mask \
  --shots 5 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --name_exp eval_pascal_part_f0_5shot_sansa
```

FSS-1000：

```bash
MPLCONFIGDIR=/tmp/matplotlib python inference_fss.py \
  --dataset_file fss \
  --prompt mask \
  --shots 5 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.3 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_fss_fold0.pth \
  --name_exp eval_fss_5shot_sansa
```

COCO-20i fold0：

```bash
MPLCONFIGDIR=/tmp/matplotlib python inference_fss.py \
  --dataset_file coco \
  --prompt mask \
  --shots 5 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.3 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/coco-20i-4/adapter_coco_fold0.pth \
  --name_exp eval_coco_fold0_5shot_sansa
```

### 8.2 Collect uncertainty cache

```bash
MPLCONFIGDIR=/tmp/matplotlib python collect_uncertainty_cache.py \
  --dataset_file pascal_part \
  --prompt mask \
  --shots 1 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --cache_path output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08_match.pt
```

### 8.3 Train uncertainty head

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path \
    output/fss_fold0_1shot_mask_uncertainty_match.pt \
    output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08_match.pt \
    output/paco_part_fold0_1shot_mask_uncertainty_generalist_cf08_match.pt \
  --output_dir output/uncertainty_head_mixed_tokens_match_fss_pascal_paco \
  --device cuda \
  --epochs 200 \
  --batch_size 128 \
  --feature_set tokens_match \
  --split_by dataset_class
```

### 8.4 Support-selection diagnostic with official metrics

这个不是最终主表，但可用于 Module B ablation。

```bash
MPLCONFIGDIR=/tmp/matplotlib python evaluate_support_selection.py \
  --seed 0 \
  --dataset_file pascal_part \
  --prompt mask \
  --shots 5 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --head_ckpt output/uncertainty_head_mixed_tokens_match_fss_pascal_paco/uncertainty_head.pt \
  --official_metrics \
  --output_path output/support_selection_pascal_part_fold0_5shot_seed0_official.json
```

## 9. 不要偏离的约束

- 主表必须是 official FSS evaluation，不能只报 support-selection avg IoU。
- 不要把 PACO support-selection 的小差异说成稳定提升。
- 不要把 BRM 混成 frozen uncertainty baseline，它是 trainable refinement branch。
- 不要马上 joint training，先做 frozen SANSA + post-hoc uncertainty 的可解释改进。
- 每次阶段性实验结论更新 `PROCESS.md` 或本文件。
- 完成一个阶段性任务后 git commit。

## 10. 当前最该做的实现

优先实现顺序：

1. 在 `inference_fss.py` / SANSA forward 中加入 `--hflip_tta`，复现 TTA_SUM 的 hflip。
2. 加 `--uq_head_ckpt` 和 `--uq_gate_threshold`，实现 uncertainty-gated hflip。
3. 在 5-shot 下实现 Module B 的 `adaptive-k` 或 `weighted logits`，不要只做 top1。
4. 输出一张 Pascal-Part fold0 的 official table：
   - SANSA；
   - hflip；
   - UQ-gated hflip；
   - Module B；
   - Ours A+B。
5. fold0 有收益后，再跑 4-fold 主表。

一句话提醒：

> Support-selection 证明 uncertainty 有用；official FSS table 才证明方法有效。

## 11. 新窗口启动清单

另一个窗口开始时按这个顺序做：

1. 读本文件，先不要继续跑 support-selection 当主结果。
2. 读 `TTA_SUM.md`，只提取 hflip / BRM 对 Module A 的证据。
3. 读 `PROCESS.md`，只提取 uncertainty cache / head / support diagnostic 的阶段结论。
4. 打开 `inference_fss.py`，以 official evaluation 为主入口实现新方法。
5. 第一阶段只做 Pascal-Part fold0：
   - baseline 对齐；
   - hflip；
   - UQ-gated hflip；
   - 5-shot Module B aggregation；
   - A+B。
6. 结果写回本文件或 `PROCESS.md`，再决定是否跑 4 folds。

若只能记住一句话：

> 现在的目标不是继续证明 uncertainty head 有信号，而是把 uncertainty 变成能提升 official FSS mIoU 的 SANSA 改进模块。
