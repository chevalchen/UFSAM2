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
- EXP-001 formal implementation/control SHA：Stage A 为 `7ab637758698ddb19fdacf6c17df1f09446bec26`；Stage B 为 `d2e841baa5810b3f76ea12d17081bb0a36490c81`；Stage C 为 `TBD`。
- EXP-001 主机训练前基础设施 SHA：`e89a189`，Python 3.8 运行时兼容补丁 SHA：`d477b29`；前者修复 SANSA adapter-only checkpoint load contract，并实现 identity-checked episode manifest/replay、互斥 base-train/base-calibration/base-validation partitions、held-out validation 与 best-checkpoint selection，后者确保这些新增入口可在 SANSA 常用的 Python 3.8 环境解析。服务器 data/checkpoint smoke 已通过，Stage A formal inputs 已绑定到上列 formal SHA；Stage B research contract、implementation 与 formal execution inputs 已冻结；Stage C 未冻结。
- 当前实验分支：`exp/EXP-001-av-pmc`；AV-PMC Stage A 已形成 formal run evidence；Stage B 已完成主机实现和冻结，等待 Experiment 会话执行服务器 preflight/formal run；Stage C 尚未冻结。
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

- 状态：EXP-001 总体为 `DRAFT`；`Stage A: EVALUATED / PASS_TO_STAGE_B`，已选 operator 冻结；`Stage B: FROZEN / READY_FOR_EXPERIMENT_SESSION`；Stage C 尚未冻结。Stage A 的通过不是 EXP-001 的 `KEEP`，也不证明 action-value gating 主张。
- 证据标签：Stage A operator/headroom 为 `VERIFIED`；完整 AV-PMC 主张仍为 `PROPOSED`。
- 设计依据：`POST_MEMORY_FEATURE_CALIBRATION_DESIGN.md`。若账本与该设计文档冲突，以本条冻结后的 experiment contract 和研究规范为准；观察结果后不得回写原合同。
- 假设：对于同一个 frozen post-memory feature repair operator，预测 action-specific signed gain 的 spatial/episode routing，在 matched intervention rate、activated area 与 second-decoder compute 下，能够比 random routing、entropy/current-risk routing 和 SAM2 confidence routing 更有效地提升 held-out final segmentation。
- Baseline run/SHA：absolute reference 使用 SANSA paper/official tables；因果 baseline 使用 EXP-001 相同 formal SHA、checkpoint 与 episode manifest 下保存的第一次 baseline decode `M0/B0`。Stage A formal SHA 为 `7ab637758698ddb19fdacf6c17df1f09446bec26`；Stage B formal SHA 为 `d2e841baa5810b3f76ea12d17081bb0a36490c81`；Stage C formal SHA 仍为 `TBD`。现有 `e6203d1` 仅作 `HISTORICAL` implementation reference，不自动成为 baseline 或 EXP-001 implementation SHA。
- 唯一变更因素：在完全相同的 SANSA baseline、frozen repair operator、checkpoint、episodes 与 inference budget 下，只改变 AV-PMC trigger/selection policy。因果矩阵为 `B0` baseline、`B1` always-on、`B2` spatial-only、`B3` matched random、`B4` entropy/risk、`B5` expected-IoU risk、`B6` predicted action value、`B7` shuffled/inverted negative control、`B8` oracle positive gain。
- 明确非目标：不联合 hflip、BRM、MTP、support aggregation 或其他 repair；不 fine-tune SAM2/SANSA/AdaptFormer；不加入高分辨率 boundary branch、query pseudo-memory、automatic points、Monte Carlo/ensemble；不把 uncertainty calibration/AUROC 当成主结果；不在 novel/test episodes 上调 architecture、threshold 或 budget。
- 固定实现边界：SANSA/SAM2/adapters 保持冻结；AV-PMC 插在 Memory Attention 之后、mask decoder 之前；baseline decode 后最多复用 cached features 再运行一次 mask decoder；repair operator、spatial benefit head 与 episode action-value head 按 Stage A/B/C 顺序训练并逐阶段冻结。
- 当前实现状态：已在 `exp/EXP-001-av-pmc` 上以 `000d450` 为 authority-import parent 完成 staged AV-PMC integration。Stage A 已于 2026-07-28 正式完成；Stage B implementation SHA `d2e841baa5810b3f76ea12d17081bb0a36490c81` 已补齐 deterministic hard top-area、B2/B3/B4E/B4R/B7S/B8S cached-feature matched decoding、calibration budget selection、per-episode outputs、dense AUPRC、negative-repair/worst-decile guardrails 与 paired bootstrap。MTP、BRM、support aggregation、hflip 和 episode action-value gate 均未接入 Stage B 活动路径。Stage C 所需 episode-level `B5/B6` routing harness 仍未实现，因此当前代码不能用于完整 EXP-001 `KEEP` 结论。
- 本地验证状态（2026-07-26）：checkpoint contract 与 episode manifest/replay 共 9 个 tests 已在 Python 3.8.20 + PyTorch 2.4.1 CPU 环境通过，11 个相关 Python 文件通过 AST syntax check。测试覆盖 adapter-only 接受、adapter 缺失/shape mismatch/AV-PMC contamination 拒绝、manifest deterministic generation/JSON round-trip、RNG restoration、identity drift 与跨 partition image overlap 拒绝。本机没有正式 dataset/checkpoint，因此本地结果仅用于 host-side preflight。
- 服务器 preflight 状态（2026-07-27，`DIAGNOSTIC_ONLY`）：在 clean detached SHA `4f5222d81d0ca053d65771d1aacf7e192facc2fa` 上完成 COCO-20i fold 0、1-shot smoke。环境为 Python 3.10.19、PyTorch 2.5.1+cu121、CUDA 12.1、2×RTX 3090；官方 adapter checkpoint 的 `240/240` tensors 严格加载。9 个 repository tests、exact episode replay、forward/backward、一次 optimizer step、held-out validation、paired `B0/treatment/B8` 输出、epoch/best checkpoint saving 均通过。单 episode manifest hash 为 `e74f63c4b3968ddf18d97a0ea2d9a1a2c74677acc1c0bdebf7d571720de6b45e`，只用于执行链路验证，不作为 formal episode input、observed gain 或 scientific evidence。`detectron2` 对 `iopath` 的版本约束不一致未影响当前路径，记录为 non-blocking environment note。
- Stage A frozen episodes（2026-07-27）：tracked manifest 为 `experiments/EXP-001/episodes/coco_fold0_1shot_train1200_cal600_val600_seed0.json`，Git-blob/file SHA256 为 `c0abf8545a07800e4f56cb4f83afd3738553dda87d1223855d46891156a23de9`，包含 COCO-20i fold 0、1-shot、seed 0 的 train/calibration/validation `1200/600/600` episodes。三个 partitions 之间 query/support identity overlap 为 0，均覆盖全部 60 个 base classes；每类 episode 数范围依次为 train `8–35`、calibration `3–20`、validation `5–19`。
- Stage A frozen config（2026-07-27）：tracked config 为 `experiments/EXP-001/configs/stage_a_operator_grid.json`，Git-blob/file SHA256 为 `05d4f7836f8952e86e0c138e6ca1ad428a3ccdeaf259da3ec2643389a4d5eb5d`。每 config 固定 `10 epochs / 12000 optimizer steps`，每 epoch 使用同一 1200 train episodes 并在同一 600 validation episodes 上选择 best checkpoint；六个 run IDs 对应 projection width `{32,64}` × residual scale `{0.05,0.10,0.20}`，其余 training/execution fields 完全共享。30-step `DIAGNOSTIC_ONLY` probe 估计每 config 约 `2.6–3.0 h`，两张独立 GPU 各串行三个 configs 约 `7.8–9.0 h`；该 timing 不属于 scientific evidence。
- Stage A formal execution（2026-07-28）：服务器在 clean detached execution SHA `7ab637758698ddb19fdacf6c17df1f09446bec26` 执行，documentation SHA 为 `dab7a70beebd41874aeca9863ac705c38bdccd05`，artifact root 为 `/data6/chensq/UFSAM2_runs/EXP-001/stage_a`。六组均成功退出，matched `B0` 在六组中完全一致；每个 best metrics 文件均含 600 个唯一 validation episode IDs。两张 RTX 3090 的总体墙钟时间约 `7:54`；运行后、全部进程退出且 artifacts 验证完成之后的一次 `nvidia-smi` probe 无法连接驱动，不影响本次结果有效性。

| Stage A run ID | Best epoch | B0 mIoU / FB-IoU | B1 mIoU / FB-IoU | B8 mIoU / FB-IoU | B1−B0 | B8−B1 | 改善 / 受损 / 持平 episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stage_a_p32_s005_seed0` | 5 | 71.560 / 86.086 | 72.247 / 86.494 | 72.600 / 86.676 | +0.687 | +0.353 | 290 / 300 / 10 |
| `stage_a_p32_s010_seed0` | 10 | 71.560 / 86.086 | 72.203 / 86.663 | 73.274 / 87.180 | +0.643 | +1.071 | 260 / 333 / 7 |
| `stage_a_p32_s020_seed0` | 6 | 71.560 / 86.086 | 72.433 / 86.737 | 73.767 / 87.403 | **+0.873** | **+1.334** | 247 / 346 / 7 |
| `stage_a_p64_s005_seed0` | 2 | 71.560 / 86.086 | 72.162 / 86.470 | 72.310 / 86.526 | +0.602 | +0.148 | 287 / 301 / 12 |
| `stage_a_p64_s010_seed0` | 2 | 71.560 / 86.086 | 72.068 / 86.448 | 72.480 / 86.608 | +0.508 | +0.412 | 263 / 327 / 10 |
| `stage_a_p64_s020_seed0` | 2 | 71.560 / 86.086 | 72.141 / 86.471 | 72.640 / 86.664 | +0.580 | +0.500 | 248 / 343 / 9 |

- Stage A 阶段裁决：`PASS_TO_STAGE_B`。六组 always-on repair 均超过 matched `B0`；`stage_a_p32_s020_seed0` 同时取得最高 B1（+0.873 mIoU）、最高 B8（相对 B0 +2.207 mIoU）和最大 `B8−B1` headroom（+1.334 mIoU），因此冻结为后续 Stage B/C 唯一 operator。其 validation episodes 中改善/受损/持平为 `247/346/7`：均值改善与大量 episode 受损并存，支持继续验证 selective gating，但不能作为 gating 已有效的证据。
- 冻结 operator artifact：`/data6/chensq/UFSAM2_runs/EXP-001/stage_a/stage_a_p32_s020_seed0/pmc_operator_best.pth`，SHA256 `c7a5860a694d047cd22b2070732ace877d128c3a780b8af0c25eebe1c2307819`；对应 `run_manifest.json` SHA256 `d9c78051c1949cfe2a70bbf44bea2295c5c962d6720ffd030911f913c5494a1d`；best-epoch metrics `pmc_operator_validation_epoch6.json` SHA256 `2993ce937241b1a11f6f704280e16447b1e8d1a1222a148c7ab77ddf0694748e`。后续 Stage B/C 不得重新选择或训练 operator；若改变 operator 定义或选择结果，必须按停止规则 `REVISE` 为新 Experiment ID。

| Stage A run | Run manifest SHA256 | Best checkpoint SHA256 | Best-epoch metrics SHA256 |
|---|---|---|---|
| `p32_s005` | `3537b9b774bc0c090fe7dcaf6d8e50e57860cbed63ef31f60717bb0f61f73f9e` | `18aa7d55aa203b6c71f8a41e6f09a136b8b37430736fad3c5ef3a2743bcc7ae8` | `0b5f34659695f29c0898dc30324f0254c5f43a0b7b3000ef49c17970a7a892b7` |
| `p32_s010` | `c363bbf4b5da82e8242998bcb24c6ea37b1108c5df9c02516653586cdada781c` | `d2fe9764d98cb906cb4ac7667e5b168ee5923bc622bfac530d1d872d1da49ff1` | `3531ae5452c2bfccb1ac3408622b5d406b124f6f66939f27850c31f6e7c26e7f` |
| `p32_s020` | `d9c78051c1949cfe2a70bbf44bea2295c5c962d6720ffd030911f913c5494a1d` | `c7a5860a694d047cd22b2070732ace877d128c3a780b8af0c25eebe1c2307819` | `2993ce937241b1a11f6f704280e16447b1e8d1a1222a148c7ab77ddf0694748e` |
| `p64_s005` | `124510e969f616269502b0faac04f48bddd388475daeceb8367844a953d12702` | `604673f315abfd91252fd35d6759a1323b4e51497239c036798c3a42f3a6005c` | `3c54685a3ee20199b0f61e35b2c2c969f063c1ada1e9efb1b054248cc9be7d9d` |
| `p64_s010` | `ae8c24768ac417cfcd16dab6494081facb8204600d2ad51e3154ffe4483341f3` | `b7f48db76cddbd35b1f6516796bb16cce0e31339057fd9a199f457ac8ee249fd` | `de7fff7e9d788dd61fd33275857dd7759792cd9624891514841847a79a4d0a4a` |
| `p64_s020` | `4dc809aa7a6b2f71a1c9b2ee0df095423dd4c0e7e370e81b91e15dbda949dc1a` | `4eb7aa8fbc4996633339421c9a67d98677cfd13f06f9d496d409cf1b62a8ecc5` | `6cea9e999a6c7e55b74c3500522825cf9277e45ea86ce9d8bd7d55f6f0ad22bc` |

- Stage B research contract（2026-07-28）：tracked agent-facing contract 为 `experiments/EXP-001/configs/stage_b_spatial_contract.json`，file SHA256 为 `f5dffaa777b6612da00d423f379ffb6b42a97d6ea0a26f26a842ac530ab791e0`。该文件只冻结科学合同并永久保留当时的 `execution_ready=false`，不得回写观察后状态；当前正式执行权限来自下列独立 frozen execution config 与 formal SHA。
- Stage B frozen execution input（2026-07-28）：tracked runnable config 为 `experiments/EXP-001/configs/stage_b_spatial_execution.json`，exact file/Git-blob SHA256 为 `489800da3fa70f76d241679e0dc6d083986fdfc717515e9eb808a3dfdf60f8b7`，formal implementation SHA 为 `d2e841baa5810b3f76ea12d17081bb0a36490c81`。config 固定两个并行 training runs：GPU 0 的 `stage_b_b2_benefit_seed0` 与 GPU 1 的 `stage_b_b4r_risk_seed0`；两者完成后执行 `stage_b_spatial_matched_eval_seed0`，在 calibration 选择 `q*` 并一次性输出全部 validation matched rows、per-episode metrics、10000-resample paired bootstrap 与自动阶段建议。formal run 必须在该 SHA 的 clean detached worktree 执行；不得从当前 branch tip 直接运行。
- Stage B host validation（2026-07-28，`DIAGNOSTIC_ONLY`）：`D:\anaconda3\envs\isat_env` 的 Python 3.8.20 + PyTorch 2.4.1 CPU 环境完成 17 个 repository tests，全部通过；覆盖 hard gate exact cardinality/stable ties、full-area identity、episode-keyed random replay/RNG neutrality、calibration selector、paired bootstrap determinism、dense AP 与 execution-config/contract binding。仓库 61 个 Python 文件通过 AST parse，两份 Stage B JSON 通过 parse；trainer/evaluator CLI parser 在仅为主机缺失的 `py3_wget/einops` 提供不执行代码的 import stubs 后成功加载。主机无真实 dataset、SAM2 weights 和 Stage A operator artifact，因此 GPU tensor path、checkpoint load 与一阶 smoke 仍必须由 Experiment 会话在服务器标记为 `DIAGNOSTIC_ONLY` 后验证。
- Stage B 单一假设：固定 Stage A `p32_s020` operator 后，learned signed action-specific dense benefit score 在相同 activated area 与 second-decoder compute 下，能比 equal-area random、mask entropy 和同架构 generic current-error risk 更有效地改善 held-out segmentation。Stage B 只验证 spatial actionability；episode action-value gate 保持关闭。
- Stage B teacher target：在同一 train episode 上生成 baseline logits `M0` 与 frozen operator 全空间开启的 `M_all`，以逐像素 `BCEWithLogits(M0,Y) - BCEWithLogits(M_all,Y)` 构造 signed target，截断到 `[-1,1]` 后用 area interpolation 下采样至 feature grid；`u* > 0` 为正收益标签。`M0/M_all`、operator、SANSA、SAM2 与 adapters 全部 detach，只训练 spatial head。优化目标固定为 `SmoothL1(score,u*) + 0.25 × BCEWithLogits(score/0.25, 1[u*>0])`。该 target 标记为 `ORACLE_TRAINING_TARGET`，只是 always-on repair 的 dense benefit proxy，不得表述为单 feature-cell 的精确 causal effect。
- Stage B training 与 budget：使用原 manifest 的 train/calibration/validation `1200/600/600` partitions、COCO-20i fold 0、1-shot、training seed 0；spatial head 固定训练 `10 epochs / 12000 steps`，其余 optimizer fields 与 Stage A 一致。Best checkpoint 按 validation `B2@25% area` mIoU 选择，完全相同时取更早 epoch。Locked evaluation 只用 deterministic hard top-area gate，area grid 固定为 `{0.10,0.25,0.50,1.00}`，每 episode 选择 `K=max(1,round(qHW))` 个 feature cells，分数相同时按 flattened spatial index 稳定打破。最终 `q*` 只在 calibration 上选择：在 B2 calibration mIoU 距 grid 内最高值不超过 `0.2` point 的 budgets 中取最小者；validation 报告完整曲线，但只以 `q*` 行作阶段裁决。
- Stage B frozen controls：`B0` no repair；`B1` frozen operator all-area；`B2` learned benefit top-area；`B3` episode-keyed random equal-area，固定 random seeds `{0,1,2}`；`B4E` baseline mask entropy equal-area；`B4R` 同 spatial-head architecture/evidence/initialization policy/training exposure 的 generic current-error risk，唯一变化为 teacher `clamp(BCEWithLogits(M0,Y),0,1)`；`B7S` inverted learned score；`B8S` 以 `u*` 排序的 equal-area oracle-proxy。所有选择行必须共享 operator、episode IDs、面积、decoder calls 与 evaluation path；`q=1.0` 时所有 spatial rows 必须精确复现 `B1` masks 和 aggregate metrics。
- Stage B primary metrics：primary selection effect 为 `B2 - mean(B3 seeds)` paired episode mIoU；primary specificity effect 为 `B2-B4R`，并报告 `B2-B4E`。在 600 个 validation episode identities 上做 paired percentile bootstrap，固定 `10000` resamples、seed 0、two-sided 95% CI；B3 先对同 episode 的三个 random seeds 求均值再 bootstrap。必须同时报告 `B2-B0`、`B2-B1`、`B8S-B2`、FB-IoU、negative-repair episode rate、worst-decile episode IoU、实际 activated area 与 dense positive-benefit AUPRC。
- Stage B `PASS_TO_STAGE_C`：必须同时满足 `q* < 1.0`；`B2-B0 >= +0.5 mIoU point` 或其 paired 95% CI 下界大于 0；`B2-mean(B3)`、`B2-B4E`、`B2-B4R` 的 paired 95% CI 下界均大于 0；area/provenance/full-area sanity 全部通过；且 B2 的 FB-IoU 不比 B0 低超过 `0.2` point、worst-decile episode IoU 不比 B0 低超过 `0.5` point。通过只授权设计和实现 Stage C，不等于 EXP-001 `KEEP`。
- Stage B `DROP_AND_STOP_EXP_001`：若 `q*=1.0`，或 B2 相对 B0、mean(B3)、B4E、B4R 任一 point effect 非正，或可复现的 preregistered guardrail failure，则停止本 EXP-001，不训练 Stage C。若所有 point effects 为正但必要 CI 包含 0 且 `B2-B0 < +0.5`，记为 `INCONCLUSIVE`；若继续需要改变 operator、teacher、head architecture、budget grid、data protocol、primary metric 或 threshold，则保持本合同并以新 Experiment ID `REVISE`。
- 会话与交接状态：Stage B 已达到移交 Experiment 会话的条件。新 Experiment 会话必须先读取 skill、research spec、本账本、AV-PMC design、Stage B research contract 与 frozen execution config；在 clean detached worktree checkout exact SHA `d2e841baa5810b3f76ea12d17081bb0a36490c81`，先做不改变 tracked files 的 server preflight/smoke，验证两个 Stage B targets、frozen operator load、25% hard gate、checkpoint provenance 和 matched evaluator。smoke 通过后，才按 config 并行执行两项正式训练，再执行 matched evaluation；不得启动 Stage C、修改 frozen inputs 或覆盖 artifact directories。任何代码/环境修复必须返回主机产生新 commit、formal SHA 与 run IDs。
- 固定 episode 与 validation contract：正式训练入口必须提供 `--episode_manifest`，随机未追踪 episodes 会被拒绝。manifest 同时包含互斥的 `train`、`calibration`、`validation` partitions；训练仅使用 `train`，每个 epoch 仅用 `validation` 计算 matched `B0/treatment/B8` 并按 treatment mIoU 保存 best checkpoint。`calibration` 保留给后续 threshold/budget selection，不参与 best-checkpoint 选择。Stage A dataset-bound manifest 的 exact bytes、tracked path 与 hash 已按上述记录冻结；后续 stages 不得用未追踪 manifest 替代。
- Config / checkpoint / episode hashes：Stage A config 与 episode hashes 见上述 frozen records；Stage B research contract、execution config、formal SHA 与 frozen operator hash 见 Stage B records。official COCO-20i fold-0 adapter SHA256 为 `b02b96f30ee558c37ef4fc5f889e90772dafb117e42b3fcf0945000220bc57e5`，SAM2-Large base weights SHA256 为 `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`。formal run 不得使用空 checkpoint path。全部 Stage C inputs 仍为 `TBD`。
- Datasets / folds / seeds：Stage A 与 Stage B 均冻结为 strict-FSS COCO-20i fold 0、1-shot、training seed 0，并使用上述互斥 train/calibration/validation episodes；Stage B random control seeds 固定为 `{0,1,2}`。one-seed screening 只用于主干阶段裁决，不单独满足最终多 seed `KEEP`。Stage C 与后续 PASCAL-5i/COCO-20i 1-shot/5-shot confirmation 仍须在进入对应阶段前冻结；PASCAL-Part/PACO-Part 仅作为单独标注的 secondary part track。
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
- Result run IDs 与 paired summary：Stage A 六个 run IDs、matched 指标及 hashes 见上表；所选 `stage_a_p32_s020_seed0` 的 `B1−B0 = +0.873 mIoU`，`B8−B0 = +2.207 mIoU`，`B8−B1 = +1.334 mIoU`。
- 决策：EXP-001 总体仍为 `PENDING`；Stage A 阶段决策为 `PASS_TO_STAGE_B`。当前结果只认证 operator headroom 和 unconditional repair 的平均改善，不认证 spatial/action-value gating、matched-budget superiority 或最终 `KEEP`。
- Artifact URI/path 与 SHA256：artifact root 为 `/data6/chensq/UFSAM2_runs/EXP-001/stage_a`；六组 formal provenance 见上表。Stage B 的唯一输入 operator 为 `stage_a_p32_s020_seed0/pmc_operator_best.pth`，SHA256 `c7a5860a694d047cd22b2070732ace877d128c3a780b8af0c25eebe1c2307819`。

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
