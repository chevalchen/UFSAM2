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
- EXP-001 formal implementation/control SHA：Stage A 为 `7ab637758698ddb19fdacf6c17df1f09446bec26`；Stage B/C 为 `TBD`。
- EXP-001 主机训练前基础设施 SHA：`e89a189`，Python 3.8 运行时兼容补丁 SHA：`d477b29`；前者修复 SANSA adapter-only checkpoint load contract，并实现 identity-checked episode manifest/replay、互斥 base-train/base-calibration/base-validation partitions、held-out validation 与 best-checkpoint selection，后者确保这些新增入口可在 SANSA 常用的 Python 3.8 环境解析。服务器 data/checkpoint smoke 已通过，Stage A formal inputs 已绑定到上列 formal SHA；Stage B/C 仍未冻结。
- 当前实验分支：`exp/EXP-001-av-pmc`；AV-PMC minimal integration 已完成本地静态检查，但尚未构成 formal run evidence。
- 已接受模块：无
- 首个候选实验：`EXP-001 — AV-PMC`
- Legacy implementation snapshot：`e6203d1`
- 执行节奏约束（non-scientific）：研究负责人通常在每日 `12:00` 与 `21:00`（Asia/Shanghai）启动或检查服务器任务。服务器 task packet 原则上安排为约 `6–9` 小时；较短的独立 runs 可由同一 launcher 顺序执行，但必须保留各自独立的 run ID、config、artifact directory 与 metrics。该时间偏好不得用于截断 frozen training exposure、减少 matched controls，或改变已冻结的 episodes、seeds、指标与停止规则。
- 30 天 portfolio 目标：依次完成 AV-PMC、mask post-correction、support-memory fusion 三个模块的主干 screening。每个模块至少取得 fixed held-out episodes 上的 operator headroom 结果；存在 headroom 时，再完成一个 training seed 的 learned-selection 对 random/generic-risk screening。三模块主干完成前，优先暂停 large qualitative analysis、part-track 扩展、完整 latency sweep 与非关键 ablations。one-seed screening 只能用于排序后续工作或触发既定停止规则，不能单独满足最终 `KEEP`。

## EXP-000 — SANSA Reference Baseline 与 Local Control 约定

- 状态：`DECIDED`
- 证据标签：论文表格为 `LITERATURE`；研究负责人此前的复现判断为 `HISTORICAL`；EXP-001 内新生成的 matched `B0` 才可随对应 run 作为 causal control。
- 协议决定：接受 SANSA paper/official tables 作为所有数据集的 absolute baseline，不要求为 EXP-000 重新执行 COCO-20i、FSS-1000、PASCAL-Part、PACO-Part 或完整 strict-FSS baseline 表。该决定由研究负责人于 2026-07-22 明确作出。
- 目标：记录 baseline policy。研究负责人接受论文表格为 absolute baseline，并假设所有历史候选模块与 AV-PMC 关闭时 baseline 一致；不再执行独立 default-off equivalence experiment。正式因果结论仍使用 EXP-001 同一 run/SHA、checkpoint、episodes 上保存的 `M0/B0`，而不是仅用论文 aggregate number 做差。
- 权威源码及版本：SANSA upstream commit `f562b5642940e08a57e3da2608b804b99af3763b`（本地 nested repository `SANSA/`，remote `https://github.com/chevalchen/SANSA.git`）加已批准的本地修复；UF-SAM 顶层 authority import commit 为 `000d450`，导入前已核对 donor working tree 的 `101/101` 个 tracked files 与 Git index blob 完全一致。论文来源：`Paper/SANSA1.pdf`，arXiv `2505.21795v2`，2025-11-15；`Paper/SANSA2.pdf` 为 appendix；upstream `SANSA/README.md` 提供 official commands/tables。
- 本地修复分类：`inference_fss.py` 与 `util/visualization.py` 的改动属于 visualization-only；`util/commons.py::resize_mask` 的 `>0` 到 `>0.5` 改动会作用于 mask-only benchmark 的 support-mask resize，因此分类为经研究负责人批准的 shared mask-prompt preprocessing fix，而非 AV-PMC 变量。该修复必须独立 commit，并在 `B0-B8`、所有 datasets/folds 与训练/评估路径中保持完全一致。
- Checkpoint 与 SHA256：服务器 smoke 使用的 SANSA paper-released COCO-20i fold-0 adapter checkpoint SHA256 为 `b02b96f30ee558c37ef4fc5f889e90772dafb117e42b3fcf0945000220bc57e5`；SAM2-Large base weights SHA256 为 `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`。formal checkpoint 仍须随 dataset/fold 在 `FROZEN` 前确认并记录对应 official hash；无需重新训练或用它重建整张论文表。
- Tracked config 与 hash：不在 EXP-000 单独生成；由 EXP-001 在 `FROZEN` 前记录。
- Episode manifest 与 hash：不在 EXP-000 单独生成；由 EXP-001 记录 matched `B0-B8` manifest。
- Datasets / folds / seeds：沿用 SANSA paper protocol；具体设置由 EXP-001 冻结。
- Environment fingerprint：不单独生成；由 EXP-001 formal run manifest 记录。
- 主指标与容差：论文 aggregate mIoU/FB-IoU 用于 absolute table comparison；AV-PMC 因果效应以同 episode 的 `B6-B0`、`B6-B3/B4/B5` paired delta 和 CI 为准。不再设置或执行独立 default-off output-equivalence tolerance。
- 护栏：`B0/M0` 关闭全部候选功能；使用 clean Git SHA；run directory 不可覆盖；保留 per-episode baseline 与 treatment 输出。不得把论文 aggregate number 当成 paired sample，也不得因免除完整 baseline 重跑而省略 EXP-001 内部 matched `B0`。
- Result run IDs：`N/A — no standalone EXP-000 run`
- 决策：`ACCEPTED_BY_OWNER_ASSUMPTION`；full-paper baseline reproduction 与 standalone default-off equivalence check 均免除。若后续 matched `B0` 与论文表格出现重大冲突，再回到 Idea/Decision 阶段审查该假设。

## EXP-001 — AV-PMC Action-Value-Gated Post-Memory Feature Calibration

- 状态：EXP-001 总体为 `DRAFT`，其中 `Stage A: FROZEN`；Stage B/C 尚未冻结。Stage A 只允许执行 frozen operator grid 与 matched `B0/B1/B8` headroom study；不要求重跑完整 SANSA paper baseline 表或 standalone default-off equivalence test。
- 证据标签：`PROPOSED`；当前没有 observed result。
- 设计依据：`POST_MEMORY_FEATURE_CALIBRATION_DESIGN.md`。若账本与该设计文档冲突，以本条冻结后的 experiment contract 和研究规范为准；观察结果后不得回写原合同。
- 假设：对于同一个 frozen post-memory feature repair operator，预测 action-specific signed gain 的 spatial/episode routing，在 matched intervention rate、activated area 与 second-decoder compute 下，能够比 random routing、entropy/current-risk routing 和 SAM2 confidence routing 更有效地提升 held-out final segmentation。
- Baseline run/SHA：absolute reference 使用 SANSA paper/official tables；因果 baseline 使用 EXP-001 相同 formal SHA、checkpoint 与 episode manifest 下保存的第一次 baseline decode `M0/B0`。formal SHA 当前为 `TBD`。现有 `e6203d1` 仅作 `HISTORICAL` implementation reference，不自动成为 baseline 或 EXP-001 implementation SHA。
- 唯一变更因素：在完全相同的 SANSA baseline、frozen repair operator、checkpoint、episodes 与 inference budget 下，只改变 AV-PMC trigger/selection policy。因果矩阵为 `B0` baseline、`B1` always-on、`B2` spatial-only、`B3` matched random、`B4` entropy/risk、`B5` expected-IoU risk、`B6` predicted action value、`B7` shuffled/inverted negative control、`B8` oracle positive gain。
- 明确非目标：不联合 hflip、BRM、MTP、support aggregation 或其他 repair；不 fine-tune SAM2/SANSA/AdaptFormer；不加入高分辨率 boundary branch、query pseudo-memory、automatic points、Monte Carlo/ensemble；不把 uncertainty calibration/AUROC 当成主结果；不在 novel/test episodes 上调 architecture、threshold 或 budget。
- 固定实现边界：SANSA/SAM2/adapters 保持冻结；AV-PMC 插在 Memory Attention 之后、mask decoder 之前；baseline decode 后最多复用 cached features 再运行一次 mask decoder；repair operator、spatial benefit head 与 episode action-value head 按 Stage A/B/C 顺序训练并逐阶段冻结。
- 当前实现状态（2026-07-23，`DIAGNOSTIC_ONLY`）：已在 `exp/EXP-001-av-pmc` 上以 `000d450` 为 authority-import parent 完成 minimal integration，包括 decoder evidence 暴露、零初始化 bounded residual、spatial/action-value heads、Stage A/B/C 分阶段冻结训练入口，以及同一 episode 内 `B0`、treatment 与 `B8 oracle` 的 paired metrics 输出。MTP、BRM、support aggregation 与 hflip 均未接入活动路径；旧候选脚本已从本实验分支移除但保留于 Git 历史。`B2-B7` 的完整 matched-budget policy harness 尚未实现，因此当前代码只能用于 Stage A diagnostic，不能用于完整 EXP-001 结论。
- 本地验证状态（2026-07-26）：checkpoint contract 与 episode manifest/replay 共 9 个 tests 已在 Python 3.8.20 + PyTorch 2.4.1 CPU 环境通过，11 个相关 Python 文件通过 AST syntax check。测试覆盖 adapter-only 接受、adapter 缺失/shape mismatch/AV-PMC contamination 拒绝、manifest deterministic generation/JSON round-trip、RNG restoration、identity drift 与跨 partition image overlap 拒绝。本机没有正式 dataset/checkpoint，因此本地结果仅用于 host-side preflight。
- 服务器 preflight 状态（2026-07-27，`DIAGNOSTIC_ONLY`）：在 clean detached SHA `4f5222d81d0ca053d65771d1aacf7e192facc2fa` 上完成 COCO-20i fold 0、1-shot smoke。环境为 Python 3.10.19、PyTorch 2.5.1+cu121、CUDA 12.1、2×RTX 3090；官方 adapter checkpoint 的 `240/240` tensors 严格加载。9 个 repository tests、exact episode replay、forward/backward、一次 optimizer step、held-out validation、paired `B0/treatment/B8` 输出、epoch/best checkpoint saving 均通过。单 episode manifest hash 为 `e74f63c4b3968ddf18d97a0ea2d9a1a2c74677acc1c0bdebf7d571720de6b45e`，只用于执行链路验证，不作为 formal episode input、observed gain 或 scientific evidence。`detectron2` 对 `iopath` 的版本约束不一致未影响当前路径，记录为 non-blocking environment note。
- Stage A frozen episodes（2026-07-27）：tracked manifest 为 `experiments/EXP-001/episodes/coco_fold0_1shot_train1200_cal600_val600_seed0.json`，Git-blob/file SHA256 为 `c0abf8545a07800e4f56cb4f83afd3738553dda87d1223855d46891156a23de9`，包含 COCO-20i fold 0、1-shot、seed 0 的 train/calibration/validation `1200/600/600` episodes。三个 partitions 之间 query/support identity overlap 为 0，均覆盖全部 60 个 base classes；每类 episode 数范围依次为 train `8–35`、calibration `3–20`、validation `5–19`。
- Stage A frozen config（2026-07-27）：tracked config 为 `experiments/EXP-001/configs/stage_a_operator_grid.json`，Git-blob/file SHA256 为 `05d4f7836f8952e86e0c138e6ca1ad428a3ccdeaf259da3ec2643389a4d5eb5d`。每 config 固定 `10 epochs / 12000 optimizer steps`，每 epoch 使用同一 1200 train episodes 并在同一 600 validation episodes 上选择 best checkpoint；六个 run IDs 对应 projection width `{32,64}` × residual scale `{0.05,0.10,0.20}`，其余 training/execution fields 完全共享。30-step `DIAGNOSTIC_ONLY` probe 估计每 config 约 `2.6–3.0 h`，两张独立 GPU 各串行三个 configs 约 `7.8–9.0 h`；该 timing 不属于 scientific evidence。
- 会话与交接状态：Stage A 已从 Idea/Research Decision 会话移交新的 Experiment 会话。Experiment 会话必须先读取 skill、research spec、本账本、AV-PMC design 和 tracked config，在 clean detached worktree checkout exact formal-input SHA `7ab637758698ddb19fdacf6c17df1f09446bec26`，逐 run capture manifest 后仅执行 Stage A 六配置 `B0/B1/B8`。运行中不得修改 tracked files、设计、episodes、exposure、seeds 或 controls；任何失败必须回传并由主机产生新 commit/run ID。
- 固定 episode 与 validation contract：正式训练入口必须提供 `--episode_manifest`，随机未追踪 episodes 会被拒绝。manifest 同时包含互斥的 `train`、`calibration`、`validation` partitions；训练仅使用 `train`，每个 epoch 仅用 `validation` 计算 matched `B0/treatment/B8` 并按 treatment mIoU 保存 best checkpoint。`calibration` 保留给后续 threshold/budget selection，不参与 best-checkpoint 选择。Stage A dataset-bound manifest 的 exact bytes、tracked path 与 hash 已按上述记录冻结；后续 stages 不得用未追踪 manifest 替代。
- Config / checkpoint / episode hashes：Stage A config 与 episode hashes 见上述 frozen records；official COCO-20i fold-0 adapter SHA256 为 `b02b96f30ee558c37ef4fc5f889e90772dafb117e42b3fcf0945000220bc57e5`，SAM2-Large base weights SHA256 为 `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`。formal run 不得使用空 checkpoint path。Stage B/C inputs 仍为 `TBD`。
- Datasets / folds / seeds：Stage A 冻结为 strict-FSS COCO-20i fold 0、1-shot、operator-training seed 0，并使用上述互斥 train/calibration/validation episodes。该 seed 只用于 bounded architecture screen，不单独满足最终多 seed `KEEP`。Stage B/C 与后续 PASCAL-5i/COCO-20i 1-shot/5-shot confirmation 仍须在进入对应阶段前冻结；PASCAL-Part/PACO-Part 仅作为单独标注的 secondary part track。
- 主指标与最小有意义效应：primary metric 为 paired episode `mIoU`；由于 absolute baseline 采用论文表格且不重估 baseline training-seed SD，formal KEEP 要求 `B6-B0 >= 0.5 mIoU point`，且 paired episode-bootstrap 95% CI 下界高于 0；同时报告多个 AV-PMC training seeds。FB-IoU 为质量护栏；part track 另报 Boundary IoU/F-score。论文数值仅用于最终 absolute comparison，不用于 paired CI。
- 护栏与 compute budget：所有比较使用同一 episode list；匹配 episode activation rate、activated spatial area、second-decoder calls、training exposure、trainable parameter count，并报告 intervention rate、FLOPs、latency、negative-repair rate 与 worst-decile episode IoU。每个 episode 最多一次额外 decoder pass；formal outputs 写入 repository 外不可覆盖目录，并绑定 clean detached SHA 与 run manifest。
- 预注册 bounded search space：repair topology 固定为两个 depthwise-pointwise blocks；projection width `{32, 64}`，bounded residual scale `{0.05, 0.10, 0.20}`。只在 base-validation 上选择并冻结 operator，之后不得在 gating evaluation 中更换。Spatial area budgets `{0.10, 0.25, 0.50, 1.00}`；episode intervention-rate budgets `{0.25, 0.50, 0.75, 1.00}`；threshold 仅在 base calibration 上选择。
- 阶段停止规则：Stage A 先比较 `B0/B1/B8`。若 always-on repair 与 oracle positive-gain routing 均无有意义 headroom，停止 EXP-001，不训练 spatial/action-value heads。Stage B/C 仅在前一阶段通过后进行；operator 或 target 定义发生实质变化时停止本实验并以新 Experiment ID `REVISE`。
- 执行分包：EXP-001 的 architecture-search configs、training seeds 与 causal rows 均作为独立、不可覆盖的 runs。两张 GPU 可并行运行两个独立 config，但不使用 data parallel 改变单 run 的 optimization contract。每个 `12:00/21:00` 窗口优先启动预计约 `6–9` 小时完成的 task packet；实际 training steps、episodes、seeds 与 validation frequency 必须预先冻结，并在 intended comparison rows 中保持一致。较短 runs 可由一个 launcher 串联，但不得合并 provenance 或 metrics。
- 快速主干规则：先完成 Stage A 的 `B0/B1/B8`；失败立即停止并转向下一个候选模块，通过才进入 Stage B/C。三个模块完成主干 screening 前，EXP-001 暂不扩展 part track、额外 qualitative figures 或非必要 ablations。最终 `KEEP` 仍须补齐多 seed、matched `B3/B4/B5` controls、第二数据族与统计区间。
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
