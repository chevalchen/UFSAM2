---
name: manage-ufs-research
description: Govern evidence, baselines, experiment branches, and local-to-server Git handoffs for UFSAM2/SANSA deep-learning research. Use when planning, implementing, porting, running, comparing, accepting, rejecting, or documenting experiments; reusing legacy MTP, BRM, support aggregation, or AV-PMC work; establishing or changing the baseline; or coordinating local development with server-only formal execution.
---

# Manage UFSAM2 Research

Use this skill only as a process gate. Treat `UNCERTAINTY_SAM2_FSS_RESEARCH_SPEC.md`
as the authority for cross-experiment research goals, evidence levels, baseline policy, and
Go/No-Go principles. Use the following non-overlapping record authorities:

- `EXPERIMENT_LEDGER.md` is the global index only. For each experiment, it may contain only the
  Experiment ID, name, current status, final conclusion, authoritative document path, and key Git
  SHAs.
- `experiments/EXP-<id>.md` is the sole human-review authority for that experiment's hypothesis,
  frozen contract, variables, metrics, decision thresholds, execution history, results, artifact
  hashes, and decision history.
- Tracked JSON configs and manifests are English machine-facing frozen inputs. Never rewrite an
  observed experiment's frozen input to reflect later state; record later state in its EXP document.

Do not duplicate mutable experiment details in the ledger, research spec, design documents, or
multiple EXP documents. Update the ledger and the corresponding EXP document together when an
experiment's indexed status, conclusion, or key SHA changes.

## Language

Match language to audience: use English for agent-facing files and Chinese for human-review
documents. Keep established academic or technical terms in English when translation reduces precision.

## Start Work

1. Locate the Git root. Read the research spec, experiment ledger, the current experiment's
   authoritative `experiments/EXP-<id>.md`, and any relevant design document.
2. Inspect `git status --short --branch`, the current SHA, and recent commits.
3. Classify the task as exploration, formal experimentation, or result adjudication. Never present
   exploratory output as evidence.
4. Return to the Idea/Decision stage if goals, metrics, or protocols must change. Do not silently
   change them while coding.

## Handle Legacy Results

- Apply the evidence labels defined by the research spec. Initially label all pre-restart numbers
  and code as `HISTORICAL`.
- Preserve the old HEAD with a remote archive branch or tag before creating the new mainline.
  Keep its history, but do not inherit its code automatically. `2bee18f` is an extended pre-MTP
  snapshot, not clean upstream SANSA.
- Use `HISTORICAL` only to generate hypotheses, diagnose failures, and find implementation clues.
  Never use it for scientific claims, final threshold selection, baseline values, or automatic
  module promotion.
- Separate neutral infrastructure from research modules. Port traces, multimask outputs, or
  second-decoder interfaces only after proving default-off equivalence. Re-evaluate MTP, BRM,
  support aggregation, AV-PMC, and other effect-bearing modules from a new Experiment ID.
- Prefer a minimal reimplementation. Cherry-pick legacy commits only after auditing dependencies;
  never import a hidden module combination wholesale.

## Certify the Baseline First

1. Make `EXP-000` the first experiment on the new mainline.
2. Freeze authoritative source provenance, checkpoint, data protocol, episode manifest, seeds,
   environment, and metric implementation.
3. Disable every candidate module. Reproduce the local baseline on the server and retain
   per-episode outputs.
4. Prove output or metric equivalence for neutral, default-off instrumentation.
5. Before certification, describe differences only as smoke or diagnostic results, never gains.

Keep `main`, or an explicitly named `accepted` branch, limited to the certified baseline, minimal
`KEEP` changes, and all decision records. Create each candidate as `exp/EXP-<id>-<slug>` from that
SHA. Prefer control/treatment comparisons on the same code SHA with one configuration factor changed.
Never claim a single-module causal gain from nonequivalent code snapshots. Do not combine modules
before each has passed its standalone gate.

## Freeze the Experiment Contract

For a new experiment, create its authoritative document from `experiments/TEMPLATE.md` and add one
index row to `EXPERIMENT_LEDGER.md`.

Before coding or formal execution, freeze the following in the experiment's authoritative
`experiments/EXP-<id>.md`:

- Experiment ID, single hypothesis, and baseline run/SHA;
- single primary change and explicit non-goals;
- tracked config, checkpoint, episode manifest, folds, and seeds;
- primary metric, minimum meaningful effect, guardrails, and compute budget;
- search space, stop rule, and `KEEP/DROP/REVISE/INCONCLUSIVE` criteria.

Do not rewrite the original contract after observing results. Create a new Experiment ID when the
hypothesis, threshold, data, or primary implementation changes. Lock the final configuration before
evaluation. Treat post-final tuning as development evidence rather than independent confirmation.
Keep the corresponding ledger row index-only; it must link to the EXP document rather than repeat
the contract.

## Hand Off Between Local and Server

- Use remote Git as the only code handoff between local and server. Do not copy working directories
  or use stash to transfer experiment state.
- Treat the server as the only execution environment for formal metrics. Use local results only for
  static checks and explicitly labeled smoke tests.
- Allow only one machine to edit an experiment branch at a time. Before switching machines, commit
  and push; on the other machine, fetch and update by fast-forward only.
- Allow clearly labeled WIP commits on experiment branches. Bind every formal result to an exact SHA,
  never only to a branch name.
- Run the exact SHA in a clean detached worktree on the server. Never edit, pull, switch branches, or
  hot-fix the running worktree.
- The server execution session is execution-only: save immutable external artifacts and return a
  structured report containing the run ID, exact Git SHA, config/manifest/checkpoint hashes,
  artifact URIs and hashes, metric summary, and any execution errors. Do not edit research documents
  or create commits from the detached execution worktree.
- The host Idea/Decision session is the sole adjudicator and writer of experiment state. It updates
  the authoritative EXP document and ledger index together, then commits and pushes the decision
  record.
- If a server-side Git report is unavoidable, wait until every run process has exited, create a
  separate writable report branch/worktree, and commit only an `UNADJUDICATED_EXECUTION_REPORT`.
  That report may contain execution facts but must not change the frozen contract, experiment
  status, final conclusion, or `EXPERIMENT_LEDGER.md`.
- Give every fix a new commit and run ID. Never overwrite or reinterpret an old run.

Before formal execution, run:

```text
python .agents/skills/manage-ufs-research/scripts/capture_run_manifest.py \
  --experiment EXP-<id> --config <tracked-config> --episodes <tracked-episodes> \
  --checkpoint <checkpoint> --seed 0 --fold 0 --command "<exact command>" \
  --output <external-run-dir>/manifest.json
```

Require the script to reject dirty worktrees and record the Git SHA, tracked-input hashes,
checkpoint hash, command, and environment fingerprint. Write formal outputs to immutable directories
outside the repository. Keep datasets, checkpoints, raw logs, W&B directories, and large per-sample
artifacts out of Git. Commit only compact summaries, necessary paired metrics, external artifact
paths, and SHA256 values.

## Adjudicate Results

- `KEEP`: require the preregistered effect threshold, a positive paired CI, acceptable guardrails
  and cost, and marginal value over the current accepted stack. Merge only the minimal code and add
  a regression test.
- `DROP`: do not merge feature code. Preserve the negative result, exact SHA, config, and artifact
  reference in the experiment's authoritative EXP document, and update only the indexed status,
  final conclusion, and key SHA in the ledger.
- `REVISE`: preserve the old record and create a new Experiment ID.
- `INCONCLUSIVE`: keep the change out of the mainline until stronger evidence exists.

Report multiple training seeds for trainable modules and use matched episodes for every paired
analysis. If baseline certification, a frozen contract, a clean SHA, a run manifest, or a matched
control is missing, stop describing the work as formal evidence; continue only as explicitly labeled
diagnostic work.
