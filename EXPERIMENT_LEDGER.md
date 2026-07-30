# 实验总账索引

> **职责：** 本文件只保存实验索引与组合状态，不保存假设、冻结合同、变量、指标、运行结果、artifact/hash 明细或决策历史。
> **单实验权威来源：** `experiments/EXP-xxx.md`
> **跨实验研究规范：** `UNCERTAINTY_SAM2_FSS_RESEARCH_SPEC.md`
> **流程门禁：** `.agents/skills/manage-ufs-research/SKILL.md`

## 实验索引

| Experiment ID | 名称 | 当前状态 | 最终结论 | 权威文档 | 关键 Git SHA |
|---|---|---|---|---|---|
| `EXP-000` | SANSA Reference Baseline 与 Local Control 约定 | `DECIDED` | `ACCEPTED_BY_OWNER_ASSUMPTION`：接受论文 absolute reference，免除 standalone baseline/equivalence 重跑；候选实验仍须保留 matched `B0`。 | [`experiments/EXP-000.md`](experiments/EXP-000.md) | Upstream `f562b5642940e08a57e3da2608b804b99af3763b`; authority import `000d450` |
| `EXP-001` | AV-PMC Action-Value-Gated Post-Memory Feature Calibration | `DECIDED / DROP` | Frozen unconditional operator 有 headroom，但 learned spatial routing 未优于 generic controls；触发 Stage B hard-stop，Stage C 未启动，feature code 不合入 accepted branch。 | [`experiments/EXP-001.md`](experiments/EXP-001.md) | Stage A `7ab637758698ddb19fdacf6c17df1f09446bec26`; Stage B `d2e841baa5810b3f76ea12d17081bb0a36490c81`; Stage C `N/A` |
| `EXP-002` | Repair-Benefit-Routed Mask Post-Correction | `FROZEN / READY_FOR_EXPERIMENT_SESSION_V2` | `PENDING`：v1 smoke 暴露并已修复两个 implementation/runtime blocker；formal training/evaluation 尚未执行。 | [`experiments/EXP-002.md`](experiments/EXP-002.md) | Accepted parent `000d450`; current formal SHA `80607302a71fde8ab9e26f0558c4605a67c770b4`; superseded v1 `81773c2` |
