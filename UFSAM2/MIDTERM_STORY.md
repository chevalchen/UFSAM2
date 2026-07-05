# Midterm Story: Uncertainty-Guided SAM2 Few-Shot Segmentation

## One-Sentence Goal

Use uncertainty estimation as a lightweight decision layer on top of SANSA/SAM2, so the model can know when to trust a support example, when a prediction is risky, and when boundary/part ambiguity needs extra inference or refinement.

## Problem Framing

SANSA is a strong SAM2-based few-shot segmentation baseline. It converts few-shot segmentation into pseudo-video semantic propagation:

- support image + GT mask are written into the SAM2 memory bank;
- query image is decoded with memory-conditioned SAM2 features;
- mask-only prompting gives a strong support signal.

But SANSA is mostly action-only, not self-aware. It does not explicitly answer:

- Is this support reliable for the current query?
- Is the current query prediction likely to fail?
- Is the error caused by semantic mismatch, part ambiguity, or boundary ambiguity?
- Should we use a single support, all supports, extra TTA, or a refinement path?

This gives the central thesis:

> In SAM2-based few-shot segmentation, uncertainty should not only be a score after prediction. It should become a pre-action controller for support selection and ambiguity-aware inference.

## Link To UncertainSAM

UncertainSAM shows that SAM/SAM2 decoder tokens contain useful uncertainty information. Its key engineering idea is post-hoc and lightweight:

- patch SAM2 to expose decoder tokens;
- read `iou_token` and `mask_token`;
- train small MLP predictors for expected IoU / uncertainty / gap scores;
- keep the original SAM2 segmentation model frozen.

Our adaptation to SANSA keeps this spirit but changes the target:

- UncertainSAM estimates uncertainty for class-agnostic SAM predictions;
- UFSAM2 estimates uncertainty for few-shot semantic propagation in SANSA;
- the uncertainty score is used to guide few-shot decisions, especially support reliability and later boundary/TTA/refinement.

## Proposed Method Story

### Module A: Uncertainty-Aware Support Reliability

Input:

- query decoder `iou_token`;
- selected query `mask_token`;
- query `obj_ptr`;
- optional memory summary;
- support-side tokens and support-query matching features.

Output:

- expected IoU;
- failure risk;
- support reliability score.

Action:

- in 5-shot inference, run each support independently as a 1-shot candidate;
- rank supports by predicted expected IoU;
- choose the most reliable support or fall back to all supports when uncertainty is high.

This is the current main validated module.

### Module B: Uncertainty-Guided Ambiguity Refinement

Target SANSA failure modes:

- fuzzy object/part boundaries;
- small or thin parts;
- support-query semantic mismatch;
- ambiguous part masks.

Candidate actions:

- horizontal-flip TTA only for risky predictions;
- boundary refinement module for uncertain boundary regions;
- later: uncertainty-gated local refinement instead of unconditional mask editing.

Current status:

- TTA summary gives preliminary evidence that hflip and BRM+hflip can improve SANSA-side inference;
- this branch should be kept separate from the frozen uncertainty baseline because BRM introduces a trainable refinement checkpoint.

### A+B Combined View

The intended midterm architecture can be presented as:

```text
Support set + query image
        |
      SANSA / SAM2
        |
  decoder tokens + memory features
        |
  Uncertainty layer
        |
  +-------------------------+
  |                         |
Module A                 Module B
support reliability      ambiguity-aware TTA/refinement
  |                         |
selected / weighted       refine only when uncertain
support decision
        |
final mask
```

For the midterm report, the clean claim is:

- Module A already has positive intervention evidence on Pascal-Part.
- Module B is supported by completed TTA/BRM observations and should be the next engineering branch.
- A+B combined experiments are the immediate next step, not yet the final claim.

## Current Evidence

### 1. SANSA Baseline And Trace Collection

Completed:

- reproduced SANSA mask-only baseline;
- added decoder trace hooks;
- collected uncertainty caches;
- trained post-hoc MLP heads while freezing SANSA.

Collected signals:

- true IoU;
- SAM score;
- query `iou_token`;
- query selected `mask_token`;
- query `obj_ptr`;
- memory summary;
- support-side token summaries;
- support-query matching features.

### 2. FSS-1000 Sanity Check

FSS-1000 is near ceiling, so it should not be the main proof.

5-shot support-selection average episode IoU:

| strategy | avg IoU |
| --- | ---: |
| random support | 0.9138 |
| SAM-score support | 0.9170 |
| token support | 0.9211 |
| all supports | 0.9205 |
| oracle support | 0.9353 |

Read:

- token uncertainty has signal;
- but the dataset is too easy to show a large improvement.

### 3. Pascal-Part: Main Positive Evidence

Pascal-Part object+part class-disjoint setup is the strongest current evidence because it stresses fine-grained semantic and part-level ambiguity.

5-shot support-selection average episode IoU, full 2500 episodes, 3 seeds:

| strategy | avg IoU |
| --- | ---: |
| random support | 0.3938 |
| SAM-score support | 0.4274 |
| token support | 0.4620 |
| all supports | 0.4615 |
| oracle support | 0.5578 |

Risk summary:

- token support selection consistently beats random and SAM-score;
- token selection approximately ties all-supports in average IoU;
- token selection has lower failure/risk than all-supports;
- oracle remains much higher, so support reliability is still not solved.

This is the best current proof that uncertainty can guide a real SANSA decision.

### 4. PACO-Part: Stress Test, Not A Final Win Yet

PACO-Part is harder: long-tail, fine-grained parts, stronger support quality variation.

2 seeds x 200 episodes, support-selection average episode IoU:

| strategy | avg IoU | fail<0.5 | risk<0.7 |
| --- | ---: | ---: | ---: |
| random support | 0.4520 | 54.3% | 71.8% |
| SAM-score support | 0.4830 | 49.3% | 65.3% |
| mixed token | 0.4949 | 48.0% | 65.8% |
| tokens_match | 0.4945 | 47.3% | 65.8% |
| all supports | 0.4948 | 48.5% | 66.5% |
| oracle support | 0.5834 | 36.8% | 58.0% |

Read:

- mixed/unified training improves over PACO-only token heads;
- support-query matching improves expected-IoU prediction;
- but the support-selection gain does not robustly exceed all-supports;
- PACO should be presented as a stress test and motivation for pairwise ranking / fallback policies.

### 5. TTA / BRM Evidence For Module B

From `TTA_SUM.md`:

PACO-Part fold 0:

| setting | mIoU | FB-IoU | note |
| --- | ---: | ---: | --- |
| plain generalist | 40.27 | 67.17 | baseline |
| generalist + hflip | 40.70 | 67.52 | no-training TTA |
| BRM + hflip | 41.24 | 67.74 | lightweight trainable refinement |

PACO-Part 4-fold average:

| setting | mIoU | FB-IoU |
| --- | ---: | ---: |
| generalist + hflip | 43.22 | 66.29 |
| BRM + hflip | 43.65 | 66.44 |

Read:

- hflip is a clean no-training add-on with small gains;
- BRM+hflip is stronger but changes the training/checkpoint story;
- uncertainty should be used as a gate for these actions, not as unconditional mask editing.

## Important Metric Clarification

Do not mix these two metrics in the report:

1. `support-selection avg IoU`
   - average binary IoU over sampled episodes;
   - used to compare random / SAM-score / token / all-supports / oracle under the same episodes;
   - values are in `[0, 1]`.

2. official SANSA fold mIoU
   - class-level accumulated mIoU from the SANSA evaluation pipeline;
   - values are usually reported as percentages, e.g. `40.27`;
   - used for baseline, hflip, and BRM/TTA comparisons.

The support-selection tables prove decision quality under controlled episodes. They should not be directly compared with official PACO fold mIoU.

## Suggested Midterm Report Structure

### Slide 1: Title

Uncertainty-Guided SAM2 Few-Shot Segmentation

### Slide 2: Background

- SAM2 is strong but class-agnostic.
- SANSA adapts SAM2 to few-shot segmentation through memory-conditioned propagation.
- In mask-only FSS, support masks are strong prompts, but support reliability and part ambiguity remain difficult.

### Slide 3: Problem

SANSA lacks explicit uncertainty awareness:

- it does not know which support is reliable;
- it does not know when semantic propagation is risky;
- it does not know when boundaries or parts are ambiguous.

### Slide 4: Inspiration From UncertainSAM

- decoder tokens contain uncertainty information;
- small post-hoc MLPs can estimate expected IoU / uncertainty;
- no need to retrain the full SAM2 backbone.

### Slide 5: Proposed Framework

Uncertainty layer on top of frozen SANSA:

- Module A: support reliability selection;
- Module B: ambiguity-aware TTA/refinement;
- shared signal: decoder-token uncertainty.

### Slide 6: Implementation

- expose SAM2/SANSA decoder tokens;
- collect trace cache;
- train expected-IoU head;
- evaluate by intervention, not only correlation.

### Slide 7: FSS-1000 Sanity Check

Show that the token head works but the dataset is near ceiling.

### Slide 8: Pascal-Part Main Evidence

Show the 3-seed support-selection table.

Main sentence:

> On fine-grained part segmentation, uncertainty-guided support selection improves over both random support and SAM-score selection, and reaches all-supports performance with lower risk.

### Slide 9: PACO-Part Stress Test

Show that PACO is harder:

- mixed/matching heads improve prediction;
- intervention gain is not stable yet;
- motivates pairwise ranking and uncertainty-gated fallback.

### Slide 10: Boundary / TTA Direction

Show hflip and BRM+hflip summary from `TTA_SUM.md`.

Main sentence:

> Boundary ambiguity is a second uncertainty-driven action point: instead of modifying every prediction, we should trigger extra inference/refinement only when uncertainty is high.

### Slide 11: Current Limitations

- support-selection avg IoU is not official fold mIoU;
- PACO still lacks a robust improvement;
- current head is regression-style, not optimized for ranking;
- BRM is trainable, so it should be reported separately from frozen uncertainty.

### Slide 12: Next Plan

1. Pairwise/listwise support ranking head.
2. Hybrid policy: choose top-1 support only when confidence margin is high, otherwise use all supports.
3. Uncertainty-gated hflip TTA.
4. Boundary refinement branch with uncertainty gate.

## Minimal Next Experiments For A Strong Midterm

### Experiment A1: Pairwise Support Ranking

Current expected-IoU regression is useful but not directly optimized for choosing the best support.

Add:

- pairwise labels inside each 5-shot episode;
- train support A > support B when true IoU(A) > true IoU(B);
- report oracle match, top-1 avg IoU, and risk.

Success criterion:

- beat SAM-score and all-supports on Pascal-Part;
- on PACO, improve over mixed token and reduce the gap to oracle.

### Experiment A2: Hybrid Support Policy

Policy:

- if top1 score is high and top1-top2 margin is large: use top1 support;
- otherwise: use all supports.

Baselines:

- all-supports;
- token top1;
- SAM-score top1;
- random fallback;
- oracle fallback.

This is likely stronger than always choosing exactly one support.

### Experiment B1: Uncertainty-Gated HFlip

Policy:

- run normal SANSA for all episodes;
- run hflip TTA only when predicted risk is high;
- compare against unconditional hflip and random-gated hflip.

Report:

- official fold mIoU / FB-IoU;
- cost ratio;
- risk bucket analysis.

This turns hflip from generic TTA into uncertainty-guided TTA.

### Experiment B2: Boundary Refinement With Uncertainty Gate

Keep BRM as a separate trainable branch:

- baseline SANSA;
- hflip only;
- BRM only;
- BRM + hflip;
- uncertainty-gated BRM/hflip.

Report qualitative masks for boundaries and small parts.

## Safe Claims For The Midterm

Use these claims:

- We reproduced SANSA and established a mask-only few-shot baseline.
- We adapted the UncertainSAM post-hoc token idea to SANSA by exposing SAM2 decoder traces.
- Decoder-token uncertainty predicts failure risk in few-shot semantic propagation.
- On Pascal-Part, uncertainty-guided support selection gives a real intervention gain over random and SAM-score support choice.
- PACO-Part shows that naive expected-IoU regression is not enough; robust long-tail part segmentation needs pairwise ranking and fallback policies.
- TTA/BRM results suggest boundary ambiguity is another suitable target for uncertainty-guided actions.

Avoid overclaiming:

- Do not say PACO is already solved.
- Do not compare support-selection avg IoU directly with official fold mIoU.
- Do not merge BRM into the frozen uncertainty baseline without naming it as a separate trainable refinement module.

