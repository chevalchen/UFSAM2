# EXP-NNN — 实验名称

> **Experiment ID：** `EXP-NNN`
> **状态：** `DRAFT | FROZEN | RUNNING | EVALUATED | DECIDED`
> **最终裁决：** `PENDING | KEEP | DROP | REVISE | INCONCLUSIVE`
> **证据标签：** `PROPOSED | HISTORICAL | VERIFIED | ORACLE`
> **权威性：** 本文件是 EXP-NNN 的唯一人类可审阅权威记录。冻结后的原始合同不得因观察结果而改写；后续事实追加到执行、结果和决策历史。
> **全局依据：** `UNCERTAINTY_SAM2_FSS_RESEARCH_SPEC.md`
> **全局索引：** `EXPERIMENT_LEDGER.md`

## 1. 摘要

- 名称：
- 当前状态：
- 最终结论：
- 实验分支：
- Baseline / accepted parent SHA：
- Formal implementation SHA：
- 对应设计文档：

## 2. 单一假设

> 在固定……条件下，仅改变……，能够……

## 3. 冻结合同

### 3.1 Baseline 与唯一变更因素

- Baseline run / SHA：
- 唯一 primary change：
- 明确非目标：
- 默认关闭与兼容性约束：

### 3.2 数据与冻结输入

- Dataset / track：
- Folds / shots：
- Train / calibration / validation partitions：
- Seeds：
- Episode manifest 路径与 SHA256：
- Tracked config 路径与 SHA256：
- Checkpoint 路径与 SHA256：
- Environment 要求：

JSON config 与 manifest 使用英文，并作为机器执行的冻结输入。观察结果后不得回写冻结内容；如需改变 hypothesis、primary implementation、data protocol、metric、threshold 或 stop rule，保留本记录并创建新 Experiment ID。

### 3.3 变量与因果对照

| Row | 变量/处理 | 目的 |
|---|---|---|
| `B0` |  | Baseline |
| `B1` |  | Treatment |

### 3.4 指标、最小效应与护栏

- Primary metric：
- Minimum meaningful effect：
- Statistical test / CI：
- Guardrails：
- Compute budget：
- Matched comparison 要求：

### 3.5 Search space 与停止规则

- Bounded search space：
- Configuration selection rule：
- `KEEP`：
- `DROP`：
- `REVISE`：
- `INCONCLUSIVE`：

## 4. 执行记录

### YYYY-MM-DD — 事件名称

- 证据性质：`DIAGNOSTIC_ONLY | FORMAL`
- Git SHA：
- Run ID：
- Exact command / launcher：
- Environment fingerprint：
- Artifact root：
- Manifest / config / checkpoint hashes：
- 执行结果：

## 5. 结果

### 5.1 Primary result

| Comparison | Point effect | Paired effect | 95% CI |
|---|---:|---:|---:|
|  |  |  |  |

### 5.2 Supporting diagnostics

-

### 5.3 Artifact 与 hash

| Artifact | Path / URI | SHA256 |
|---|---|---|
|  |  |  |

## 6. 裁决

- 当前状态：
- 最终裁决：
- 结论：
- 可支持的 claim：
- 不可支持的 claim：
- 代码合入决定：
- 后续动作：

## 7. 决策历史

只追加，不删除或覆写已经发生的决定。

| 日期 | 状态变化 | 决定与依据 | Git SHA / Run ID |
|---|---|---|---|
| YYYY-MM-DD | `DRAFT → FROZEN` |  |  |
