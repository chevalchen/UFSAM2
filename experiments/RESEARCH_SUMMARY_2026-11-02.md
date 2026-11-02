# SANSA / SAM2 小样本分割阶段实验总结

整理日期：2026-11-02。覆盖 EXP-000 至 EXP-005。

本文是截至上述日期的阶段快照，用于组会、研究回顾和后续方案讨论。单实验的协议、结果和裁决以链接的 EXP 文档为准；后续新结果应记录在相应实验中，不覆盖本快照。

## 1. 当前研究结论

研究目标是在 SANSA/SAM2 的小样本分割流程中，利用 support 语义、可靠性或不确定性信号改善最终分割。已完成的路线包括 post-memory 特征校准、mask 后校正、support memory 筛选、局部原型语义修复，以及概率图 support 特征增强。

**截至目前，尚未形成一个满足预设标准、可以作为最终方法接受的新增模块，也没有证据支持将这些模块组合后宣称稳定增益。**

已有两个值得保留的正向观察：EXP-001 的固定特征修复算子具有平均收益；EXP-003 的 support 删除动作具有较明显的 oracle 选择空间。但可部署的收益预测与选择机制均未得到验证。EXP-004 和 EXP-005 的同容量、错位或打乱对照进一步表明，新增网络的微小收益不能直接归因于所声称的语义对应关系或不确定性机制。

## 2. 基线、证据和比较边界

- 基础模型为官方 SANSA/SAM2，使用 SAM2-Large；EXP-005 在三个数据集统一使用官方 COCO fold-0 adapter。
- [EXP-000](EXP-000.md) 接受论文与官方表格作为 absolute reference，没有单独重跑整套论文 benchmark。各候选实验以自身相同代码、权重和 episode 下的原 SANSA 输出作为 matched B0。
- EXP-001/002 为 COCO fold-0 1-shot 筛选，EXP-003/004/005 为 5-shot。各轮 episode 清单并不相同；不同轮次的绝对 mIoU 不构成纵向进步证据。
- 本文“提升/下降”均为指标百分点。Class-mIoU 与平均 episode 前景 IoU 的权重不同；配对 bootstrap CI 对应后者，不能作为 class-mIoU CI。
- Oracle 使用 query GT 选择动作，只表示所测试动作集合内的诊断上限，不是可部署结果。
- 历史 MTP、BRM、support aggregation 等结果未在当前匹配协议下重验的部分，仅作历史线索。中期报告中的模拟数值不纳入本总结。

## 3. 实验总览

| 实验 | 核心问题 | 关键结果（class-mIoU 点） | 当前处理 |
|---|---|---|---|
| [EXP-001](EXP-001.md)：AV-PMC | 能否预测 action-specific 修复收益并选择空间区域？ | 固定全区域修复 +0.873；学习式选择 +0.076，低于 entropy/risk controls 0.251/0.260 | DROP；未进入 Stage C |
| [EXP-002](EXP-002.md)：Mask 后校正 | 轻量局部修复是否有足够收益空间？ | Always-on −0.0169；最佳 oracle +0.0348 | Stage A 按原规则通过，但实际空间很小；Stage B 暂缓，NOT_KEEP |
| [EXP-003](EXP-003.md)：Support memory 筛选 | 删掉一个不合适的 support 是否有用？ | Oracle +1.6330；最低相似度删除 −0.1919，worst-decile −1.1860 | DROP；未训练 selector |
| [EXP-004](EXP-004.md)：局部原型 residual | 局部语义是否优于全局语义和通用 residual？ | 局部原型对 B0 +0.1290、对全局原型 +0.0018；空间错位几乎不改变结果 | DROP |
| [EXP-005](EXP-005.md)：概率 Support 图 | 节点不确定性是否提供额外收益？ | COCO/PASCAL-Part/PACO-Part 对 B0 +0.0119/+0.0587/+0.2923；与均匀、打乱权重几乎一致 | DROP 当前配置；不延长或解冻 |

## 4. 各轮实验的收获与局限

### EXP-001：修复本身有收益，选择修复区域仍然困难

在冻结 post-memory repair operator 后，对比 learned benefit、随机区域、mask entropy 和通用风险引导。B0 为 71.560，固定全区域修复为 72.433；按 calibration 选定 10% 区域后，learned benefit 仅为 71.636，entropy/risk 为 71.888/71.896。

这支持“某些特征干预能够改善平均分割”，但不支持“预测 action-specific benefit 能优于通用风险信号”。对应的必要配对 CI 未通过，Stage C 未启动。该算子的单轮筛选结果也不足以作为多 seed、跨数据集最终方法证据。

### EXP-002：先看算子空间，再决定是否训练路由器

固定轻量 mask refiner 完成 12000 步训练。B0 为 71.5602，always-on 为 71.5433；最佳 GT oracle 为 71.5950。Oracle CI 为正，按原冻结规则可以晋级，但最大收益仅 0.0348 点。

研究处理因此是保留 Stage A 合同通过的事实，同时暂缓 Stage B。即使选择器足够准确，该动作集合也缺乏实用收益空间；不能靠更复杂的 uncertainty router 弥补算子能力不足。

### EXP-003：Support 影响具有异质性，简单相似度不能可靠选择

600 个 5-shot episodes 上，B0 为 74.6563；用 GT 选择是否删除某张 support 的 oracle 为 76.2893。Oracle 配对 episode effect CI 为 [+0.9427, +1.7776] 点，表明该动作集合具有选择空间。

但删除与 query 相似度最低的 support，mIoU 为 74.4644，最差十分位 IoU 从 19.5261 降至 18.3401，超过原定下降 0.5 点的护栏。其删除选择与 forced oracle 的 top-1 agreement 仅 19.67%。因此停止当前筛选路线；不能把 DROP 解读为不存在 support 选择空间。

### EXP-004：局部原型没有表现出额外的语义对应价值

冻结 SANSA，在 memory attention 之后加入局部原型条件 residual。全局原型、generic residual、局部原型三行各训练三个 seed，共九个 run，每 run 2400 步。

| 设置 | class-mIoU，三 seed mean ± SD |
|---|---:|
| 原 SANSA，单次 | 76.173825 |
| 全局原型 | 76.301050 ± 0.068526 |
| Generic residual | 76.261473 ± 0.057129 |
| 局部原型 | 76.302814 ± 0.068138 |
| 局部 context 空间错位 | 76.302824 ± 0.068186 |

局部原型相对全局原型仅 +0.001764 点，三个必要比较的 CI 均跨零。空间错位几乎不影响结果，不支持局部空间对应关系带来额外收益；但不能仅据此断言网络完全没有使用 context。当前模块 DROP。

### EXP-005：训练和梯度链路有效，不确定性信号未形成有效区分

借鉴概率图与可靠性机制，在 support encoder features 进入 memory encoder 前插入图网络。该实验并非复现 UNICL-SAM：没有引入额外 DINO 编码器、硬聚类或 query 二次解码。

冻结 SANSA，仅训练 49,984 个新增参数。CPU 测试、三数据集两步 GPU pilot 验证了零初始化输出等价、分割损失通过五个 memories 回传、第二步各新增模块获得有效梯度，以及冻结参数不变。六个正式 run 均完成 2400 步，累计 14400 updates；只按 COCO calibration 选择 checkpoint，六个均选中 step 2400。

| 数据集（固定 episodes） | B0 | G0 均匀可靠性 | G1 节点可靠性 | G2 打乱可靠性 | G1−B0 |
|---|---:|---:|---:|---:|---:|
| COCO base（600） | 78.1084 | 78.1203 ± 0.0059 | 78.1203 ± 0.0058 | 78.1203 ± 0.0058 | +0.0119 |
| PASCAL-Part（1000） | 30.2871 | 30.3458 ± 0.0044 | 30.3458 ± 0.0043 | 30.3458 ± 0.0043 | +0.0587 |
| PACO-Part（1000） | 27.2167 | 27.5090 ± 0.0102 | 27.5089 ± 0.0099 | 27.5089 ± 0.0099 | +0.2923 |

训练行报告三 seed mean ± sample SD。G2 复用 G1 checkpoint，只打乱每张 support 内可靠性对应关系。

G1−B0 的配对 episode 前景 IoU 95% CI 分别为 [−0.3241, +0.1141]、[−0.0451, +0.3643]、[−0.0890, +0.3685] 点，均跨零。G1−G0 和 G1−G2 同样未通过。预设要求至少两个数据集相对 B0 和 G0 均提升 0.5 点且对应 CI 下界为正，本轮通过数据集数为 0。

G1 的节点可靠性均值约 0.65–0.67，但单 support 内标准差平均仅约 0.0001–0.0003。修整幅度非零，说明新增模块确实改变了特征；然而局部可靠性近乎均匀，当前配置没有证明不确定性对应关系有用。PASCAL-Part Boundary IoU 为 17.4091→17.3594，PACO-Part 为 15.3432→15.3963，三个数据集的 worst-decile 均下降。

PASCAL-Part 覆盖 56 类，PACO-Part 覆盖 194 类，均包含原四个评估分组且保持 object crop。仅 COCO base 训练；part 不训练、不调参。PACO 排除了与新增模块 COCO train/calibration 重叠的原始图片。COCO novel 保留未用；两个 part 结果属于固定清单的迁移筛选，不代表完整 benchmark 复现，也不代表原 adapter 未见过相关语义。

2026-11-02 已复核结果文件 hash、六个锁定完成记录、24 个 checkpoint hash、14400 条顺序训练记录的有限值、checkpoint 选择及失败标记。完整逐 seed 指标、配对 CI、成本和 artifact hashes 见 [EXP-005 正式结果](EXP-005.md#正式结果与复核2026-11-02)。

## 5. 目前可以支持什么结论

1. **算子收益与选择能力是两个问题。** EXP-001/003 有干预空间，但可观测信号未能可靠地把空间转化为实际收益；EXP-002 则首先受限于算子空间。
2. **额外容量不能作为机制成立的证据。** EXP-004 的错位对照、EXP-005 的均匀和打乱对照，均未表现出所声称机制的独立贡献。
3. **新增模块训练已经执行，效果不足不能简单归因于“没有训练”。** EXP-004/005 均完成多 seed 训练；是否需要联合微调 SANSA 仍是未验证问题。
4. **当前尚不能宣称 uncertainty calibration 或可靠性预测有效。** EXP-005 的方差是模型输出的分布参数，并未被证明对应错误概率或干预收益。
5. **现有负结果有明确适用边界。** 它们约束当前架构、数据、插入位置、目标与预算，不能推出所有 support-aware 或 uncertainty 方法无效。

## 6. 下一阶段建议（PROPOSED，尚未执行）

优先对已有 EXP-005 checkpoint 做机制诊断：比较节点输入、聚合后表示、均值、方差和 residual 的节点差异；测试可靠性与实际错误、遮蔽或移除 support 节点后收益的关系。需要区分“信号本身没有区分能力”和“信号有用但当前作用方式无法影响分割”。这些都只是待检验解释。

如确认信号缺乏区分能力，应先考虑明确的可靠性/干预收益监督及相应无不确定性对照；如确认作用位置限制了收益，再设计有限范围的联合微调比较。联合微调需要同时给同预算无不确定性模型相同训练机会，才能解释收益来源。

不以当前结果自动延长训练、解冻 SANSA 或合并多个失败模块。后续改变架构、损失、预算或数据，应单独登记实验；已观察的 validation 应视为开发证据，不能重复用作独立确认。

## 7. 复现资料入口

- [实验总账](../EXPERIMENT_LEDGER.md)：实验状态和权威记录索引。
- [研究规范](../UNCERTAINTY_SAM2_FSS_RESEARCH_SPEC.md)：目标、证据标签与比较边界；流程已按 owner 要求简化。
- EXP-001/002 原始结果：`/data6/chensq/UFSAM2_runs/EXP-001/stage_a`、`/data6/chensq/UFSAM2_runs/EXP-002/stage_a`；各阶段精确路径以相应 EXP 文档为准。
- EXP-003：有效版本为 `stage_a_leave_one_out_seed0_20261021_v3`，执行 SHA `334497b64880f02c2399e29a7c222683d02efb48`；早期失败和 superseded 版本不混入结论。
- EXP-004：`/data6/chensq/UFSAM2_runs/EXP-004/formal_20261023_v1`，执行 SHA `6ffc9f0e4be4ffa3f1920b75bd1e20716283b280`。
- EXP-005：`/data6/chensq/UFSAM2_runs/EXP-005/run_20261023_v1`，执行 SHA `256662e0241863fe749e9fef8bf3dcad93723bd0`；完整指标为 `validation/results.json`，逐 episode 原始计数为 `validation/*_paired.jsonl`。

本快照不改写既有冻结配置和 episode 清单，不纳入历史或模拟数值作为实测结论。
