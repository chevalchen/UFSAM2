# 实验台账

> **面向对象：** 人类研究者审阅；正文使用中文，必要的 academic/technical terms 保留英文。  
> **流程依据：** `.agents/skills/manage-ufs-research/SKILL.md`  
> **研究依据：** `UNCERTAINTY_SAM2_FSS_RESEARCH_SPEC.md`  
> **历史记录：** `UFSAM2/EXPERIMENT.md` 与 `UFSAM2/PROCESS.md`。在 certified baseline
> 上完成复现前，重启前的结果一律视为 `HISTORICAL`。

## 当前状态

- Baseline policy：`SANSA_PAPER_REFERENCE_ACCEPTED`；研究负责人确认 COCO-20i、FSS-1000、PASCAL-Part、PACO-Part generalist 与 strict-FSS 论文结果可直接作为 absolute reference，不再要求完整重跑论文 baseline 表。
- Local control 状态：`DEFAULT-OFF_EQUIVALENCE_ASSUMED_BY_OWNER`；研究负责人决定不单独验证 MTP、BRM、support aggregation、hflip 与 AV-PMC 关闭时的 baseline equivalence。EXP-001 只需保存同一 episodes 上第一次 baseline decode 的 `M0/B0`，用于 paired causal comparison。
- 预定 accepted branch：`main`（尚未创建）
- 权威 SANSA source：upstream provenance 为 nested-repository commit `f562b5642940e08a57e3da2608b804b99af3763b`，加研究负责人批准的 visualization/mask-prompt preprocessing fix；已逐文件核对 `101/101` 个 donor files，并导入 UF-SAM 顶层仓库 commit `000d450`。
- EXP-001 formal implementation/control SHA：`TBD`
- EXP-001 主机训练前基础设施 SHA：`e89a189`，Python 3.8 运行时兼容补丁 SHA：`d477b29`；前者修复 SANSA adapter-only checkpoint load contract，并实现 identity-checked episode manifest/replay、互斥 base-train/base-calibration/base-validation partitions、held-out validation 与 best-checkpoint selection，后者确保这些新增入口可在 SANSA 常用的 Python 3.8 环境解析。它们仍需服务器 data/checkpoint smoke，不能直接视为 formal implementation SHA。
- 当前实验分支：`exp/EXP-001-av-pmc`；AV-PMC minimal integration 已完成本地静态检查，但尚未构成 formal run evidence。
- 已接受模块：无
- 首个候选实验：`EXP-001 — AV-PMC`
- Legacy implementation snapshot：`e6203d1`

## EXP-000 — SANSA Reference Baseline 与 Local Control 约定

- 状态：`DECIDED`
- 证据标签：论文表格为 `LITERATURE`；研究负责人此前的复现判断为 `HISTORICAL`；EXP-001 内新生成的 matched `B0` 才可随对应 run 作为 causal control。
- 协议决定：接受 SANSA paper/official tables 作为所有数据集的 absolute baseline，不要求为 EXP-000 重新执行 COCO-20i、FSS-1000、PASCAL-Part、PACO-Part 或完整 strict-FSS baseline 表。该决定由研究负责人于 2026-07-22 明确作出。
- 目标：记录 baseline policy。研究负责人接受论文表格为 absolute baseline，并假设所有历史候选模块与 AV-PMC 关闭时 baseline 一致；不再执行独立 default-off equivalence experiment。正式因果结论仍使用 EXP-001 同一 run/SHA、checkpoint、episodes 上保存的 `M0/B0`，而不是仅用论文 aggregate number 做差。
- 权威源码及版本：SANSA upstream commit `f562b5642940e08a57e3da2608b804b99af3763b`（本地 nested repository `SANSA/`，remote `https://github.com/chevalchen/SANSA.git`）加已批准的本地修复；UF-SAM 顶层 authority import commit 为 `000d450`，导入前已核对 donor working tree 的 `101/101` 个 tracked files 与 Git index blob 完全一致。论文来源：`Paper/SANSA1.pdf`，arXiv `2505.21795v2`，2025-11-15；`Paper/SANSA2.pdf` 为 appendix；upstream `SANSA/README.md` 提供 official commands/tables。
- 本地修复分类：`inference_fss.py` 与 `util/visualization.py` 的改动属于 visualization-only；`util/commons.py::resize_mask` 的 `>0` 到 `>0.5` 改动会作用于 mask-only benchmark 的 support-mask resize，因此分类为经研究负责人批准的 shared mask-prompt preprocessing fix，而非 AV-PMC 变量。该修复必须独立 commit，并在 `B0-B8`、所有 datasets/folds 与训练/评估路径中保持完全一致。
- Checkpoint 与 SHA256：正式 EXP-001 使用的 SANSA paper-released checkpoint 必须记录 SHA256，当前为 `TBD`；无需重新训练或用它重建整张论文表。
- Tracked config 与 hash：不在 EXP-000 单独生成；由 EXP-001 在 `FROZEN` 前记录。
- Episode manifest 与 hash：不在 EXP-000 单独生成；由 EXP-001 记录 matched `B0-B8` manifest。
- Datasets / folds / seeds：沿用 SANSA paper protocol；具体设置由 EXP-001 冻结。
- Environment fingerprint：不单独生成；由 EXP-001 formal run manifest 记录。
- 主指标与容差：论文 aggregate mIoU/FB-IoU 用于 absolute table comparison；AV-PMC 因果效应以同 episode 的 `B6-B0`、`B6-B3/B4/B5` paired delta 和 CI 为准。不再设置或执行独立 default-off output-equivalence tolerance。
- 护栏：`B0/M0` 关闭全部候选功能；使用 clean Git SHA；run directory 不可覆盖；保留 per-episode baseline 与 treatment 输出。不得把论文 aggregate number 当成 paired sample，也不得因免除完整 baseline 重跑而省略 EXP-001 内部 matched `B0`。
- Result run IDs：`N/A — no standalone EXP-000 run`
- 决策：`ACCEPTED_BY_OWNER_ASSUMPTION`；full-paper baseline reproduction 与 standalone default-off equivalence check 均免除。若后续 matched `B0` 与论文表格出现重大冲突，再回到 Idea/Decision 阶段审查该假设。

## EXP-001 — AV-PMC Action-Value-Gated Post-Memory Feature Calibration

- 状态：`DRAFT`；EXP-000 已按研究负责人假设结束。补齐 EXP-001 的 formal SHA、checkpoint/config/episode hashes、folds 与 seeds 后即可转为 `FROZEN`；不要求重跑完整 SANSA paper baseline 表或 standalone default-off equivalence test。
- 证据标签：`PROPOSED`；当前没有 observed result。
- 设计依据：`POST_MEMORY_FEATURE_CALIBRATION_DESIGN.md`。若账本与该设计文档冲突，以本条冻结后的 experiment contract 和研究规范为准；观察结果后不得回写原合同。
- 假设：对于同一个 frozen post-memory feature repair operator，预测 action-specific signed gain 的 spatial/episode routing，在 matched intervention rate、activated area 与 second-decoder compute 下，能够比 random routing、entropy/current-risk routing 和 SAM2 confidence routing 更有效地提升 held-out final segmentation。
- Baseline run/SHA：absolute reference 使用 SANSA paper/official tables；因果 baseline 使用 EXP-001 相同 formal SHA、checkpoint 与 episode manifest 下保存的第一次 baseline decode `M0/B0`。formal SHA 当前为 `TBD`。现有 `e6203d1` 仅作 `HISTORICAL` implementation reference，不自动成为 baseline 或 EXP-001 implementation SHA。
- 唯一变更因素：在完全相同的 SANSA baseline、frozen repair operator、checkpoint、episodes 与 inference budget 下，只改变 AV-PMC trigger/selection policy。因果矩阵为 `B0` baseline、`B1` always-on、`B2` spatial-only、`B3` matched random、`B4` entropy/risk、`B5` expected-IoU risk、`B6` predicted action value、`B7` shuffled/inverted negative control、`B8` oracle positive gain。
- 明确非目标：不联合 hflip、BRM、MTP、support aggregation 或其他 repair；不 fine-tune SAM2/SANSA/AdaptFormer；不加入高分辨率 boundary branch、query pseudo-memory、automatic points、Monte Carlo/ensemble；不把 uncertainty calibration/AUROC 当成主结果；不在 novel/test episodes 上调 architecture、threshold 或 budget。
- 固定实现边界：SANSA/SAM2/adapters 保持冻结；AV-PMC 插在 Memory Attention 之后、mask decoder 之前；baseline decode 后最多复用 cached features 再运行一次 mask decoder；repair operator、spatial benefit head 与 episode action-value head 按 Stage A/B/C 顺序训练并逐阶段冻结。
- 当前实现状态（2026-07-23，`DIAGNOSTIC_ONLY`）：已在 `exp/EXP-001-av-pmc` 上以 `000d450` 为 authority-import parent 完成 minimal integration，包括 decoder evidence 暴露、零初始化 bounded residual、spatial/action-value heads、Stage A/B/C 分阶段冻结训练入口，以及同一 episode 内 `B0`、treatment 与 `B8 oracle` 的 paired metrics 输出。MTP、BRM、support aggregation 与 hflip 均未接入活动路径；旧候选脚本已从本实验分支移除但保留于 Git 历史。`B2-B7` 的完整 matched-budget policy harness 尚未实现，因此当前代码只能用于 Stage A diagnostic，不能用于完整 EXP-001 结论。
- 本地验证状态（2026-07-26）：checkpoint contract 与 episode manifest/replay 共 9 个 tests 已在 Python 3.8.20 + PyTorch 2.4.1 CPU 环境通过，11 个相关 Python 文件通过 AST syntax check。测试覆盖 adapter-only 接受、adapter 缺失/shape mismatch/AV-PMC contamination 拒绝、manifest deterministic generation/JSON round-trip、RNG restoration、identity drift 与跨 partition image overlap 拒绝。本机现有环境缺少 SANSA 依赖 `py3_wget`，且没有正式 dataset/checkpoint，因此完整 model import 与真实 tensor/data/checkpoint smoke 仍必须在服务器完成；当前没有 observed result。
- 固定 episode 与 validation contract：正式训练入口必须提供 `--episode_manifest`，随机未追踪 episodes 会被拒绝。manifest 同时包含互斥的 `train`、`calibration`、`validation` partitions；训练仅使用 `train`，每个 epoch 仅用 `validation` 计算 matched `B0/treatment/B8` 并按 treatment mIoU 保存 best checkpoint。`calibration` 保留给后续 threshold/budget selection，不参与 best-checkpoint 选择。manifest 的实际 dataset-bound 内容与 SHA256 仍待服务器生成并回传 Git 后冻结。
- Config / checkpoint / episode hashes：`TBD`，必须引用 tracked resolved config、EXP-000 checkpoint SHA256、base-train/base-calibration/base-validation episode manifests 及其 hashes；formal run 不得使用空 checkpoint path。
- Datasets / folds / seeds：开发与选择只使用 strict-FSS base classes 上相互独立的 train/calibration/validation episodes；先进行一个 held-out base-validation fold 的 diagnostic smoke，再按冻结顺序评估 PASCAL-5i 1-shot/5-shot 和 COCO-20i 1-shot/5-shot。具体 folds、episode manifests 和不少于多个 trainable-module seeds：`TBD before FROZEN`。PASCAL-Part/PACO-Part 仅作为单独标注的 secondary part track。
- 主指标与最小有意义效应：primary metric 为 paired episode `mIoU`；由于 absolute baseline 采用论文表格且不重估 baseline training-seed SD，formal KEEP 要求 `B6-B0 >= 0.5 mIoU point`，且 paired episode-bootstrap 95% CI 下界高于 0；同时报告多个 AV-PMC training seeds。FB-IoU 为质量护栏；part track 另报 Boundary IoU/F-score。论文数值仅用于最终 absolute comparison，不用于 paired CI。
- 护栏与 compute budget：所有比较使用同一 episode list；匹配 episode activation rate、activated spatial area、second-decoder calls、training exposure、trainable parameter count，并报告 intervention rate、FLOPs、latency、negative-repair rate 与 worst-decile episode IoU。每个 episode 最多一次额外 decoder pass；formal outputs 写入 repository 外不可覆盖目录，并绑定 clean detached SHA 与 run manifest。
- 预注册 bounded search space：repair topology 固定为两个 depthwise-pointwise blocks；projection width `{32, 64}`，bounded residual scale `{0.05, 0.10, 0.20}`。只在 base-validation 上选择并冻结 operator，之后不得在 gating evaluation 中更换。Spatial area budgets `{0.10, 0.25, 0.50, 1.00}`；episode intervention-rate budgets `{0.25, 0.50, 0.75, 1.00}`；threshold 仅在 base calibration 上选择。
- 阶段停止规则：Stage A 先比较 `B0/B1/B8`。若 always-on repair 与 oracle positive-gain routing 均无有意义 headroom，停止 EXP-001，不训练 spatial/action-value heads。Stage B/C 仅在前一阶段通过后进行；operator 或 target 定义发生实质变化时停止本实验并以新 Experiment ID `REVISE`。
- KEEP：`B6` 达到预注册主效应和 positive paired CI；在 matched budget 下同时优于 `B3`、`B4` 与 `B5`；guardrails 与成本可接受；并在第二个 dataset family 上保持效果或明确收窄 claim。Efficiency claim 另要求 `B6` 距离 `B1/B2` 不超过 `0.2` mIoU point，同时减少至少 `30%` second-decoder calls。
- DROP：`B1` 与 `B8` 均无有意义 headroom；或 `B6` 在 matched budget 下不能优于 random；或固定 episodes/paired analysis 后增益消失。不得将 EXP-001 feature code 合入 accepted branch。
- REVISE：oracle 显示 headroom，但需要改变 repair operator、target、主要 architecture、数据协议、threshold policy 或 primary metric；保留本记录并创建新 Experiment ID。
- INCONCLUSIVE：缺少 EXP-001 frozen hashes、clean formal SHA、run manifest、matched `B0-B8` controls 或足够 training seeds；或者效果只存在于 diagnostic smoke。此时模块不得进入 accepted branch。
- Result run IDs 与 paired summary：`TBD`
- 决策：`PENDING`
- Artifact URI/path 与 SHA256：`TBD`

## 记录模板

### EXP-NNN — 简短名称

- 状态：`DRAFT | FROZEN | RUNNING | EVALUATED | DECIDED`
- 证据标签：`PROPOSED | HISTORICAL | VERIFIED | ORACLE`
- 假设：
- Baseline run/SHA：
- 唯一变更因素：
- 明确非目标：
- Config / checkpoint / episode hashes：
- Datasets / folds / seeds：
- 主指标与最小有意义效应：
- 护栏与 compute budget：
- Search space 与停止规则：
- Result run IDs 与 paired summary：
- 决策：`KEEP | DROP | REVISE | INCONCLUSIVE`
- 理由：
- Artifact URI/path 与 SHA256：
