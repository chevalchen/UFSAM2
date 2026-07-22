# Actionable Uncertainty for SAM2 Few-Shot Segmentation

> **Document type:** Research ideas and experimental decision guide  
> **Primary audience:** Agents designing or implementing the next experiments  
> **Backbone assumption:** Start from SANSA/SAM2, but change the backbone if evidence shows that it blocks the idea  
> **Primary objective:** Improve final segmentation quality  
> **Status:** Proposed research plan; no result attributed to the proposed method has been verified

## 1. Outcome and Current Decision

The project is result-oriented. Uncertainty estimation is not an end in itself. A proposed uncertainty signal or intervention is useful only if it improves final segmentation, prevents harmful repairs, or preserves segmentation quality with meaningfully less computation.

The current decisions are:

1. Use SANSA as the initial controlled baseline.
2. Give priority to the three modules proposed in the midterm report:
   - query-conditioned support-memory fusion;
   - post-memory feature calibration;
   - mask post-correction.
3. Treat those modules as priority hypotheses, not as a mandatory three-module stack.
4. Allow other interventions to challenge, complement, or replace them when final validation metrics justify the change.
5. Keep intermediate uncertainty analysis minimal. Measure only what is needed to explain whether uncertainty improves intervention selection.
6. Never use the simulated midterm results to rank modules or support a scientific claim.

Two different priority orders are useful:

- **Scientific potential:** post-memory feature calibration > mask post-correction > support-memory fusion.
- **Fast experimental screening:** cheap mask post-correction > post-memory feature calibration > support-memory fusion on 5-shot episodes.

The first ordering reflects potential contribution and SAM2 specificity. The second ordering reflects how quickly an experiment can answer whether a repair operator has useful headroom.

The central research question is:

> Which uncertainty signal improves which fixed repair operator, under which few-shot protocol and intervention budget?

## 2. Evidence Status and Provenance

Every result, claim, table, or figure used by this project should carry one of the following meanings.

| Label | Meaning | Permitted use |
|---|---|---|
| VERIFIED | Reproducible result backed by logs, a checkpoint, configuration, and evaluation output | Scientific evidence |
| HISTORICAL | Result recorded before the current baseline was certified and not yet reproduced under the frozen protocol | Hypothesis generation and debugging only; no scientific claim or final model selection |
| LITERATURE | Result or mechanism reported by a cited external paper | Motivation or comparison, with attribution |
| PROPOSED | Our current hypothesis or planned method | Design discussion only |
| SIMULATED | Invented value or qualitative mock-up used to illustrate a report format | No scientific use |
| ORACLE | Uses query ground truth to measure headroom or construct training targets | Upper bound or training-only analysis |

Unless explicitly labeled otherwise, all method choices and decision rules in Sections 3-16 are PROPOSED. The cited external-work summaries are LITERATURE. Mathematical definitions organize the hypotheses; they are not evidence that the corresponding signal is learnable or useful.

At the baseline restart, every result inherited from `UFSAM2/EXPERIMENT.md`, `UFSAM2/PROCESS.md`, or the legacy experiment branch is HISTORICAL unless a new matched run explicitly promotes it to VERIFIED. Historical results may guide reproduction priorities and prevent repeated mistakes, but they must not determine final thresholds or support a scientific claim.

### 2.1 Mandatory warning about the midterm report

The midterm report describes conceptual modules. Its experimental values and qualitative examples are simulated.

In particular, values such as 50.3, 44.1, 75.4, and 79.1, the 49.1 to 50.3 ablation progression, and all “Ours” qualitative panels are not observed results. They must not be:

- copied into a paper as experimental evidence;
- used to choose a module;
- used as an expected improvement target;
- described as completed experiments;
- used to claim that the three modules are effective or complementary.

Any future result table should contain TBD until a real run is available.

## 3. Research Thesis: Risk Is Not Repair Value

The most useful idea inherited from UncertainSAM is not generic entropy. It is the counterfactual question: how much will a particular intervention improve the current result?

For the initial mask \(M_0\), a quality head may predict

\[
\hat q(z) \approx
\mathbb{E}[\operatorname{IoU}(M_0,Y)\mid z],
\]

with current-mask risk

\[
\hat R(z)=1-\hat q(z).
\]

This answers whether the current mask is likely to be poor. It does not answer whether a specific repair can fix it.

For a fixed intervention \(a\), define its action value as

\[
V_a(z)=
\mathbb{E}\left[
\operatorname{IoU}(M_a,Y)-
\operatorname{IoU}(M_0,Y)
\mid z
\right].
\]

The decision rule is

\[
a^*=\arg\max_a [\hat V_a(z)-\lambda C_a],
\]

and no intervention is applied when every predicted net value is non-positive. Here \(C_a\) is the additional computation or latency of action \(a\).

This distinction is essential:

- a low-IoU mask can be impossible for a small boundary refiner to repair;
- a medium-quality mask can have an easy, high-value boundary correction;
- a difficult episode can have no beneficial available action;
- an intervention can have negative gain even on an uncertain sample.

The main claim should therefore be about **actionable uncertainty or predicted repair benefit**, not about producing an attractive uncertainty heatmap.

## 4. Protocol Facts That Constrain the Ideas

### 4.1 SANSA/SAM2 episode structure

SANSA converts annotated support images and masks into SAM2 memory and predicts the query from that support-conditioned memory. When the support mask is ground truth, support uncertainty mainly concerns representativeness, support-query compatibility, and harmful memory propagation rather than support-mask prediction confidence.

The standard public evaluation contains one independently evaluated query after the support set. Consequently, writing the query prediction into memory after producing its mask cannot improve that already-produced mask.

A query pseudo-memory proposal is meaningful only when the method explicitly introduces:

- a second pass on the same query;
- multiple sequential queries; or
- a streaming evaluation protocol.

Self-refinement or streaming must be reported as an additional, explicit protocol. It must not silently replace standard independent-query FSS.

### 4.2 Claim tracks

| Track | Main datasets | Shot setting | Permitted claim |
|---|---|---|---|
| Strict FSS | PASCAL-5i and COCO-20i | 1-shot and 5-shot | Base-class training to untouched novel-class evaluation |
| Generalist part segmentation | PASCAL-Part and PACO-Part | Follow the SANSA/in-context protocol, normally reported separately | Generalist or in-context part segmentation |

Strict FSS and generalist part segmentation must not be merged into one generalization claim.

For strict FSS, use base-class data for training and create disjoint base calibration and base validation episodes. Novel-query labels are final-evaluation-only.

If PACO is used for training, PACO-Part is in-domain and must be labeled as such. A no-PACO-training evaluation is strongly preferred when claiming transfer to unseen part data.

## 5. Priority Module Portfolio

The three midterm modules are retained as the first portfolio, with important corrections.

### 5.1 M1: Query-Conditioned Support-Memory Contribution

#### Report idea

For each support memory, pool support and query features, predict a scalar uncertainty, convert it into a reliability weight, and reweight the support memories before Memory Attention.

#### What is worth keeping

- Reliability is conditioned on the current query, rather than being an intrinsic score for a support image.
- It targets a real multi-shot problem: some supports can be outliers, occluded, poorly matched, or redundant.
- It acts before incorrect support information is propagated through the query.

#### Required correction

A support's single-shot error is not the same as its contribution to a multi-support set. A support can be weak alone but complementary when combined with other views.

Use a leave-one-support-out or sampled-subset target:

\[
v_j=
\operatorname{IoU}(f(S),Y)-
\operatorname{IoU}(f(S\setminus\{j\}),Y).
\]

A positive \(v_j\) means support \(j\) is helpful in the current set; a negative value means it is harmful. The inference head predicts this query-conditioned contribution without access to the query label.

Uniform normalization also forces the model to trust at least one support when every support is poor. If this becomes a practical problem, compare against a safe baseline path or a reliability floor rather than merely sharpening a softmax.

#### Critical 1-shot fact

With one support and normalized support weights, this module is an identity operation. It cannot explain an improvement on a 1-shot benchmark.

If the midterm PASCAL-Part experiment is 1-shot, the simulated M1 ablation change is therefore structurally inconsistent with the stated normalized-weight design, not merely unverified.

Therefore:

- report M1 as N/A for 1-shot when using the original normalized design;
- evaluate it primarily on PASCAL-5i and COCO-20i 5-shot;
- do not attribute a 1-shot PASCAL-Part or PACO-Part gain to M1;
- add an absolute memory-versus-safe-path gate only if a separate experiment justifies the extra overlap with M2.

#### Fixed operator and controls

The fixed operator is per-support memory contribution weighting. Compare:

- uniform support fusion;
- deterministic support-query similarity weighting;
- learned report-style reliability weighting;
- learned action-value or marginal-contribution weighting;
- random matched perturbation of the support weights;
- oracle leave-one-out contribution weighting.

#### Go/no-go rule

Keep M1 only if it improves 5-shot final metrics beyond uniform and deterministic similarity weighting. If it helps only under synthetic support corruption, present it as a robustness extension rather than the main source of clean-benchmark gain.

### 5.2 M2: Memory-Induced Spatial Risk and Feature Calibration

#### Report idea

After Memory Attention, predict a spatial uncertainty map from the memory-enhanced query feature. Use that map to gate a residual correction generated from the memory-enhanced feature and the original query feature.

#### What is worth keeping

- The insertion point is highly relevant to SAM2: it directly observes the result of support-to-query memory propagation.
- The original query feature provides a useful anchor when memory conditioning is locally harmful.
- Spatial correction can affect the current query in both 1-shot and multi-shot settings.
- Feature-level repair can recover semantic regions before the final mask is discretized, so it is potentially stronger than logit-only boundary cleanup.

#### Required correction

If the spatial head sees only the memory-enhanced feature and is supervised by final mask error, it is an ordinary error-attention map. That error can originate from the decoder, boundary ambiguity, or label noise; it is not necessarily memory-matching uncertainty.

The diagnostic input should compare at least:

- memory-enhanced query feature \(F_A\);
- original query feature \(F_t\);
- their difference or disagreement;
- support-query semantic affinity.

Memory-attention statistics or support-subset disagreement may be added only if they improve final decisions.

Once a fixed feature repair operator is available, supervise the gate with its actual dense or episode-level benefit:

\[
G_{\mathrm{M2}}(p)=
\ell(M_0(p),Y(p))-
\ell(M_{\mathrm{M2}}(p),Y(p)).
\]

This predicts where the particular feature correction helps, rather than asking a generic error map to stand in for uncertainty.

Keep dense and episode-level targets distinct:

- use \(G_{\mathrm{M2}}(p)\) to train or evaluate the spatial gate;
- use \(V_{\mathrm{M2}}=\operatorname{IoU}(M_{\mathrm{M2}},Y)-\operatorname{IoU}(M_0,Y)\) for episode-level routing.

Pixelwise loss reduction is not numerically equivalent to delta-IoU. If one score is derived from the other, that aggregation must be validated rather than assumed.

Two interpretable repair forms are:

\[
F_B=F_A+g(p)\Delta F,
\]

or a rollback mixture

\[
F_B=(1-g(p))F_A+g(p)F_{\mathrm{safe}}.
\]

The safe branch may use the original query anchor, a support-only alternative, or a lightweight calibrated feature. It must remain a valid segmentation path; a support-free query is not automatically meaningful in FSS.

#### Fixed operator and controls

Freeze one residual or rollback operator and vary only the spatial gate:

- always-on;
- random equal-area;
- mask-entropy equal-area;
- generic pixel-error risk;
- predicted M2 repair-benefit map;
- shuffled or inverted benefit map;
- oracle positive-gain map.

#### Go/no-go rule

First test whether the always-on or oracle-gated operator has headroom. If neither improves segmentation, uncertainty cannot rescue the operator and M2 should be redesigned or dropped.

Promote M2 to the scientific core only if its learned gate improves final segmentation over random and generic-risk gates at matched activated area and compute.

### 5.3 M3: Repair-Benefit-Routed Mask Correction

#### Report idea

Predict initial-mask IoU from decoder tokens, convert it to global risk, and route high-risk samples to a three-layer residual logit refiner. Fuse the M2 spatial map with mask entropy to control where the residual is applied.

#### What is worth keeping

- IoU token, mask token, candidate tokens, and object pointer are sensible quality and routing features.
- The module directly optimizes the final prediction and is fast to screen.
- A local refiner is relevant to broken boundaries, holes, background adhesion, and fine part shapes.
- Conditional computation can avoid unnecessary or harmful refinement.

#### Required correction

Predicted mask quality and predicted repair benefit must be separate:

\[
V_{\mathrm{refine}}=
\operatorname{IoU}(M_{\mathrm{refined}},Y)-
\operatorname{IoU}(M_{\mathrm{initial}},Y).
\]

Route with \(\hat V_{\mathrm{refine}}\), not only with \(1-\hat q\). A very poor mask can be unrepairable by a local CNN, while a moderately good mask can have a high-value boundary correction.

Mask entropy is also insufficient by itself. Confident false positives and false negatives can have low entropy. Candidate-mask disagreement, decoder-token error features, support-semantic disagreement, or a learned dense gain map should be compared as alternatives.

The report-faithful three-layer logit residual is a useful cheap baseline, but its claim should be limited to local or boundary correction. If the goal is to recover large missing regions or incorrect semantics, compare it with:

- a feature-conditioned mask residual;
- support-aware candidate reranking;
- a second decoder pass using the first mask as a dense prompt.

M3 must first have a standalone uncertainty path. Only afterwards should it be tested with the M2 map. Otherwise M2 and M3 cannot be attributed independently.

#### Fixed operator and controls

For one frozen mask refiner, compare:

- always-on refinement;
- random matched episode and area routing;
- entropy routing;
- SAM2 predicted-IoU routing;
- generic learned failure-risk routing;
- action-value routing;
- shuffled or inverted value routing;
- oracle error routing;
- oracle positive-gain routing.

#### Go/no-go rule

Keep the simple post-refiner if it improves mIoU or part boundary quality. Claim uncertainty-driven routing only if action-value routing beats random and generic-risk routing at matched compute.

If the simple refiner has too little oracle headroom, replace the repair operator with a stronger decoder second pass before investing further in its uncertainty head.

## 6. Eligible Challenger Ideas

The following ideas remain available. They should enter as challengers, not automatically become additional main modules.

| Challenger | Why it may help | Entry condition |
|---|---|---|
| Support-aware SAM2 candidate reranking | Low-cost use of existing multimask ambiguity; potentially strong for part granularity | Screen alongside M3 |
| Selective second decoder pass | Stronger than logit-only cleanup while reusing image and memory features | Enter if M3 has little headroom |
| Reliability-aware query pseudo-memory self-refinement | SAM2-specific second-pass memory intervention; can use uncertainty for writing and stopping | Enter after the standard one-query system is understood |
| FSSAM-style IMR or SCMA | Strong deterministic memory repair baseline and possible fixed operator | Use as a non-uncertainty comparator |
| Automatic positive/negative correction points | Can target false-negative and false-positive regions | Later challenger after dense gain is reliable |
| Encoder sampling or repair | May capture model uncertainty but is distant from final errors and can be expensive | Diagnostic or late-stage ablation only |

An alternative may replace a report module if it clearly Pareto-dominates it on final segmentation versus computation. Priority does not grant immunity from negative results.

## 7. Result-Oriented Experimental Funnel

Do not run a large uncertainty study before establishing repair headroom.

| Stage | Main experiment | Decision |
|---|---|---|
| 0. Baseline | Reproduce SANSA on fixed episodes and record variance | Stop if the baseline is not stable |
| 1. Operator headroom | Baseline, identical always-on repair, and oracle positive-gain routing at several budgets | Drop or redesign an operator with no useful headroom |
| 2. Gating proof | Freeze the repair and compare random, entropy, SAM2 confidence, generic risk, action value, and oracle triggers | Decide whether uncertainty adds value beyond the module |
| 3. Cross-benchmark check | Test survivors on every validation track covered by the intended claim | Drop dataset-specific artifacts or narrow the claim |
| 4. Combination and challengers | Combine only surviving modules; allow stronger challenger operators | Select the smallest best-performing system |
| 5. Locked final evaluation | Freeze design, thresholds, and budgets, then evaluate untouched novel/test episodes | Make only claims supported by the locked results |

### 7.1 Efficient screening order

Use held-out base validation episodes rather than novel test folds for design decisions.

1. Screen the cheap M3 refiner on PASCAL-5i held-out base-validation episodes and a held-out part-validation split.
2. Screen M2 on the same whole-object and part validation settings.
3. Screen M1 on PASCAL-5i 5-shot.
4. Confirm survivors on COCO-20i base validation and a second held-out part-validation setting.
5. Run multi-seed full benchmarks only for surviving designs.

This order saves computation while keeping both whole-object and fine-grained failure modes visible. Named benchmark test episodes from PASCAL-Part and PACO-Part must remain locked until the design is frozen.

### 7.2 Practical early go threshold

A default early promotion target is an improvement larger than

\[
\delta_{\mathrm{mIoU}}=
\max(0.5\text{ point},2\sigma_{\mathrm{baseline}}),
\]

or a paired episode-bootstrap 95% confidence interval above zero. Treat this as a practical screening rule, not a universal statistical law.

For part boundary metrics, use a slightly stricter practical target, such as 0.75 point or a positive paired confidence interval.

Here \(\sigma_{\mathrm{baseline}}\) means the standard deviation across independently trained baseline seeds evaluated on the same fixed episode list. Episode-bootstrap uncertainty should be reported separately and must not be substituted for seed variance in this rule.

## 8. Core Causal Trigger Matrix

For each repair operator, keep the operator, parameters, training exposure, and inference path fixed. Change only the trigger.

| Run | Repair | Trigger | What it establishes |
|---|---|---|---|
| Baseline | None | None | Reference result |
| Always-on | Fixed operator | All eligible episodes or pixels | Whether the repair module itself helps |
| Random budget | Same operator | Random, matched rate and area | Whether selection is better than chance |
| Entropy | Same operator | Mask entropy | Generic predictive-uncertainty baseline |
| SAM2 confidence | Same operator | Predicted IoU or candidate margin | Native-confidence baseline |
| Generic risk | Same operator | Predicted current-mask failure | Risk-versus-value comparison |
| Proposed | Same operator | Predicted action-specific gain | Main actionable-uncertainty test |
| Negative control | Same operator | Shuffled or inverted gain | Checks that score ordering matters |
| Oracle error | Same operator | Ground-truth error | Error-localization upper bound |
| Oracle gain | Same operator | True positive repair gain | Best possible selection for this operator |

Match:

- episode activation rate;
- activated pixel area for spatial gates;
- number of second passes or prompts;
- average FLOPs and measured latency;
- trainable parameter count;
- training data and number of optimization steps.

Separate three effects:

\[
G_{\mathrm{module}}=
\mathrm{AlwaysOn}-\mathrm{Baseline},
\]

\[
G_{\mathrm{selection}}=
\mathrm{ValueGated}-\mathrm{RandomBudget},
\]

\[
G_{\mathrm{specificity}}=
\mathrm{ValueGated}-\mathrm{GenericRiskOrEntropy}.
\]

The uncertainty-driven claim requires positive selection and specificity effects. Always-on improvement alone proves only that the repair module is useful.

A routed method is also useful when it is within 0.2 mIoU point of always-on while reducing intervention compute by at least 30%. These values are working engineering targets and should be reported as full accuracy-compute curves rather than as a single chosen threshold.

## 9. Minimal Intermediate Validation

Final segmentation metrics control decisions. Intermediate analysis should be limited to the following:

1. **Gain ordering:** observed repair gain by predicted-value quartile.
2. **Harm avoidance:** fraction of selected interventions with negative actual gain.
3. **Spatial actionability:** one dense metric, preferably AUPRC for positive repair-gain pixels, when a spatial gate is claimed.
4. **One perturbation sanity check:** controlled support corruption or an outlier support should increase M1/M2 risk and make selective intervention more useful.

Do not build a large calibration study unless failure detection or calibrated risk becomes a separate claimed contribution. A visually plausible heatmap is never a go criterion.

## 10. Metrics and Statistical Reporting

### 10.1 Primary metrics

- mIoU: primary decision metric.
- FB-IoU: checks foreground/background balance.
- Boundary IoU or boundary F-score: required for PASCAL-Part and PACO-Part.

### 10.2 Efficiency and robustness

- intervention rate;
- activated area for spatial repair;
- FLOPs and latency;
- worst-decile episode IoU as a secondary robustness measure.

### 10.3 Uncertainty-specific evidence

- predicted-versus-observed action gain;
- gain by score quartile;
- negative-repair rate.

Use identical episode lists for paired comparisons and report paired episode-bootstrap confidence intervals. Use multiple training seeds for the final selected system. Early smoke tests can use one seed, but they are not final evidence.

Tune router thresholds and budgets only on the corresponding held-out base validation split, then freeze them before novel/test evaluation. A strict-FSS fold may use its own base-only calibration because its training classes differ. A generalist claim across part datasets should use one frozen router setting; if dataset-specific calibration is used, label the result accordingly.

## 11. Module Combination Plan

Do not begin with all three modules.

The minimum attribution sequence is:

1. baseline;
2. baseline + M1 on 5-shot only;
3. baseline + M2;
4. baseline + M3 standalone;
5. baseline + M2 + M3;
6. baseline + M1 + the best 5-shot module;
7. all surviving modules.

For M3, explicitly compare:

- standalone M3 uncertainty;
- M3 using the M2 map;
- M3 using both signals through a learned benefit head.

This prevents a forced dependency from hiding which module actually contributes.

Initially allow at most one expensive repair action per episode. Multi-action chains should be studied only after individual actions have positive, calibrated counterfactual value.

## 12. Decision Rules and Honest Outcomes

### 12.1 Promote a module when

- its repair operator has real or oracle headroom;
- it improves final segmentation on held-out validation episodes;
- its action-value gate beats random and generic-risk gates at matched budget;
- the effect survives a second dataset family or the claim is explicitly narrowed;
- it adds value beyond already retained modules.

### 12.2 Reinterpret rather than overclaim when

- always-on repair helps but value gating does not: keep it as an ordinary repair module;
- risk predicts low IoU but not repair gain: keep it only for failure detection;
- M1 helps only 5-shot: present it as a multi-shot extension;
- a refiner improves boundary metrics but not semantic IoU: claim boundary refinement only;
- a module helps only under support corruption: claim robustness, not clean-benchmark SOTA.

### 12.3 Drop or replace a module when

- oracle routing shows negligible headroom;
- learned routing does not beat random at matched budget;
- gains disappear under fixed episodes or paired analysis;
- the module duplicates a later-stage signal without adding final performance;
- a challenger gives a better mIoU-compute trade-off.

Negative results are informative. The final method should be the smallest system with defensible final gains, not the system with the largest number of uncertainty heads.

## 13. Locked Evaluation Matrix

| Dataset | Protocol | Shots | Main role | Required reporting note |
|---|---|---:|---|---|
| PASCAL-5i | Strict base-to-novel FSS | 1 and 5 | Whole-object transfer and fast development | Per-fold and mean results |
| COCO-20i | Strict base-to-novel FSS | 1 and 5 | Larger and harder whole-object validation | Per-fold and mean results |
| PASCAL-Part | Generalist/in-context part segmentation | Protocol-specific | Fine-grained transfer and boundaries | Keep separate from strict FSS |
| PACO-Part | Generalist/in-context part segmentation | Protocol-specific | Large-vocabulary part segmentation | Label in-domain if PACO is used for training |

Whole-object claims should be supported by both PASCAL-5i and COCO-20i. Part claims should report both PASCAL-Part and PACO-Part with boundary metrics. If a claim is supported by only one family, state that scope directly.

## 14. Compact Literature Boundary

| Work | Relevant lesson | Boundary for this project |
|---|---|---|
| UncertainSAM, ICML 2025 | Decoder-token prediction of error and intervention gap on SAM2 | No SAM2 few-shot memory intervention or autonomous dense repair |
| SANSA, NeurIPS 2025 | Strong SAM2/AdaptFormer baseline with support memory and part benchmarks | No explicit actionable uncertainty |
| FSSAM, ICML 2025 | Pseudo-query memory, iterative memory refinement, and support-calibrated memory attention | Useful fixed repair operators; refinement is not selected by learned action value |
| UNICL-SAM, CVPR 2025 | Spatial uncertainty can guide in-context feature refinement | No SAM2 memory-specific counterfactual harm/value study |
| SAM2Long, ICCV 2025 | Memory confidence and multiple paths can reduce video propagation errors | Video tracking protocol differs from independent-query FSS |
| PR-MaGIC, 2026 preprint | Prompt/decoder refinement can be a strong repair action | Repair operator rather than the uncertainty contribution |

The intended novelty is not “the first uncertainty method for segmentation.” A defensible claim would be:

> In SAM2 few-shot segmentation, model stage-specific repair benefit and use it to select memory-feature and mask-level interventions, then show that the learned selection is better than the identical unconditional, random, and generic-risk-driven repair at matched budget.

## 15. Minimum Convincing Experiment Package

The smallest convincing package is:

1. stable SANSA baselines;
2. one fixed feature-level or mask-level repair with measurable headroom;
3. a current-mask risk head and a separate action-value head;
4. always-on, random, entropy, SAM2-confidence, generic-risk, proposed-value, and oracle triggers;
5. strict PASCAL-5i and COCO-20i results;
6. PASCAL-Part and PACO-Part analysis when part claims are made;
7. mIoU, FB-IoU, part boundary quality, and compute;
8. paired confidence intervals and gain-by-value analysis.

This package is stronger than five interventions whose gains cannot be separated from extra capacity or repeated inference.

## 16. Handoff Rules for Future Coding Agents

An implementation agent should not implement this entire document at once. Each coding task should name:

- the fixed repair operator being tested;
- the uncertainty or value signal being compared;
- the baseline and fair controls;
- the applicable datasets and shot settings;
- the primary final metric;
- the go/no-go rule.

Before the first run for that task, freeze and record:

- the exact SANSA baseline and episode protocol;
- the calibration, validation, and final-evaluation episode lists;
- one concrete repair operator and, for M2, one valid safe-path definition;
- whether the target is dense repair gain or episode delta-IoU;
- the budget grid and the metric implementation.

Every returned result should include:

- evidence label;
- exact protocol and data split;
- run configuration and seed;
- whether the result is 1-shot or 5-shot;
- operator and trigger;
- intervention rate and compute;
- mIoU, FB-IoU, and the relevant boundary metric;
- paired comparison with baseline and matched random/generic-risk controls;
- any oracle use, clearly isolated.

No coding agent should treat the midterm tables or figures as expected outputs. The implementation sequence must follow experimental evidence, even when that evidence rejects one of the priority ideas.

## 17. References

- T. Kaiser, T. Norrenbrock, and B. Rosenhahn. [UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model](https://proceedings.mlr.press/v267/kaiser25a.html). ICML 2025. [Code](https://github.com/GreenAutoML4FAS/UncertainSAM).
- C. Cuttano et al. [SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](https://arxiv.org/html/2505.21795). NeurIPS 2025 Spotlight. [Code](https://github.com/ClaudiaCuttano/SANSA).
- Q. Xu et al. [Unlocking the Power of SAM 2 for Few-Shot Segmentation](https://proceedings.mlr.press/v267/xu25an.html). ICML 2025. [Code](https://github.com/Sam1224/FSSAM).
- D. Sheng et al. [UNICL-SAM: Uncertainty-Driven In-Context Segmentation with Part Prototype Discovery](https://openaccess.thecvf.com/content/CVPR2025/html/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.html). CVPR 2025.
- S. Ding et al. [SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html). ICCV 2025.
- M. Lee et al. [PR-MaGIC: Prompt Refinement Via Mask Decoder Gradient Flow for In-Context Segmentation](https://arxiv.org/abs/2604.12113). 2026 preprint.
