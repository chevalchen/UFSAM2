# AV-PMC: Action-Value-Gated Post-Memory Feature Calibration

> **Status:** MINIMALLY IMPLEMENTED FOR DIAGNOSTIC USE. No result in this document is an observed experimental result.  
> **Purpose:** A decision-complete research and implementation handoff for one post-memory calibration module.  
> **Primary audience:** Research and coding agents implementing or evaluating AV-PMC.  
> **Primary objective:** Improve final few-shot segmentation, not uncertainty metrics in isolation.  
> **Starting point:** SANSA/SAM2, the intervention-value principle of UncertainSAM, and the current UF-SAM codebase.

The first minimal implementation lives on `exp/EXP-001-av-pmc`, based on the
authoritative SANSA import commit `000d450`. It provides the staged AV-PMC
module and matched `B0`/treatment/`B8` output needed for Stage A. The complete
matched-budget `B2-B7` policy harness is intentionally not yet implemented, so
this implementation is not a frozen formal experiment by itself.

## 1. Final Design Decision

Implement **AV-PMC**, a selective two-pass module:

1. SANSA produces the normal memory-conditioned query feature and baseline mask.
2. A lightweight head uses pre/post-memory feature disagreement and baseline decoder evidence to predict the signed gain of applying one fixed feature repair.
3. Only episodes with positive predicted net gain enter the repair path.
4. A dense benefit map limits the repair to spatial locations where that same repair is predicted to help.
5. The mask decoder is rerun once with the calibrated feature; the encoder and memory attention are not rerun.

The core claim is not “we estimate uncertainty.” It is:

> The predicted counterfactual benefit of a fixed post-memory repair selects better interventions than random, entropy, SAM2 confidence, or generic current-mask risk at matched compute and activated area.

This design retains the midterm idea of spatial post-memory calibration, but corrects its main weakness: a generic error or entropy map is not evidence that the proposed repair will improve the mask.

## 2. Why a Two-Pass Design Is Required

The current SANSA path makes two useful evidence sources available at different times:

- Before decoding: original query feature \(F_q\) and memory-conditioned feature \(F_m\).
- After a baseline decode: IoU token, all mask tokens, predicted IoUs, multimask disagreement, mask entropy, object pointer, and memory summary.

UncertainSAM-style decoder-token evidence is therefore unavailable to a purely pre-decoder, single-pass calibrator. AV-PMC first performs the unchanged baseline decode, predicts intervention value, and reuses the cached features for one selective second decode.

This is compatible with the current memory-to-point prototype, which already demonstrates that the mask decoder can be called a second time without rerunning the image encoder or memory attention.

## 3. Exact Location in the Current Source

The source repository root is:

`D:\CODE\UFSAM2`

The live insertion point, relative to that root, is:

`UFSAM2\models\sansa\sansa.py`

inside `SANSA._compute_decoder_out_w_mem`, between:

1. `self.sam._prepare_memory_conditioned_features(...)`, which returns `pix_feat_with_mem`; and
2. `self.sam._forward_sam_heads(...)`, which performs the baseline decode.

Use:

- \(F_m\): `pix_feat_with_mem`, shape `[B, 256, H, W]`;
- \(F_q\): `backbone_out.get_current_feats_x16(idx)`, also `[B, 256, H, W]`.

The current runtime commonly uses internal `B=1` and `H=W=64`, but the new module must use `feat_sizes[-1]` and remain shape-agnostic. Do not copy the existing hard-coded `[1, 256, 64, 64]` assumption.

The calibrated tensor must preserve the channel count, spatial resolution, dtype, device, and positional-encoding contract expected by `_forward_sam_heads`. Positional encodings remain separate and must not be added again by AV-PMC.

## 4. Module Topology

```text
raw query feature F_q -----------------------------+
                                                    |
support memory -> Memory Attention -> F_m ----------+-> feature evidence D
                                      |                         |
                                      +-> baseline decoder -> M0, tokens, candidates
                                                                  |
                                                    episode gain head v_hat
                                                                  |
                                             v_hat <= threshold --+--> return M0
                                                                  |
                                             v_hat > threshold
                                                                  v
                                      dense benefit map G + fixed residual R
                                                                  |
                                      F_cal = F_m + G * R --------+
                                                                  |
                                                     second decoder -> M1
```

There are three trainable components, but they form one intervention module:

1. **Feature Repair Operator** \(R_\theta\): produces a bounded residual correction.
2. **Spatial Benefit Head** \(U_\phi\): predicts where this repair is useful.
3. **Episode Action-Value Head** \(V_\psi\): predicts whether paying for the repair is useful.

The repair operator and the two selectors must be trained and evaluated separately enough to attribute their contributions.

## 5. Feature Evidence

Project the pre- and post-memory features into a small shared width \(d\), initially `d=64`:

\[
Q=P(\operatorname{Norm}(F_q)), \qquad
M=P(\operatorname{Norm}(F_m)).
\]

Construct memory-effect evidence:

\[
D=[Q, M, |M-Q|, M\odot Q, \cos(M,Q)].
\]

The absolute difference and product expose where memory attention changed or contradicted the original query representation. This is more source-specific than using \(F_m\) alone.

Recommended implementation:

- separate `1x1` projections for \(F_q\) and \(F_m\);
- GroupNorm or LayerNorm2d rather than batch-dependent normalization;
- two lightweight depthwise `3x3` plus pointwise `1x1` blocks;
- no attention block in version 1;
- no high-resolution FPN feature in version 1, so M2 remains a semantic feature repair rather than another boundary refiner.

## 6. Fixed Feature Repair Operator

The operator predicts a residual with the original 256 channels:

\[
R_\theta(D) \in \mathbb{R}^{B\times256\times H\times W}.
\]

Use a zero-initialized final `1x1` convolution and a bounded residual:

\[
\bar R_\theta = s\tanh(R_\theta),
\]

where \(s\) is a small fixed or learned bounded scale. The initial module must be an exact identity:

\[
F_{\mathrm{cal}}=F_m \quad \text{at initialization}.
\]

For a spatial gate \(G\in[0,1]^{B\times1\times H\times W}\) and episode gate \(a\in\{0,1\}\):

\[
F_{\mathrm{cal}}=F_m+a\,G\odot\bar R_\theta(D).
\]

Do not make a full rollback to \(F_q\) the default. A support-free query feature is an anchor, not automatically a valid few-shot semantic prediction path. A deterministic rollback residual

\[
R=-(F_m-F_q)
\]

is useful only as an ablation against the learned residual.

## 7. Spatial Benefit Head

After the baseline decode, form two cheap decoder maps and resize them to the feature grid:

- selected-mask Bernoulli entropy;
- disagreement among the three multimask candidate logits.

The spatial head receives these maps together with \(D\) and outputs a **signed repair-benefit score**, not a sigmoid uncertainty probability:

\[
\hat u(p)=U_\phi(D,E_{\mathrm{mask}},D_{\mathrm{multi}}).
\]

Convert it to a soft training gate:

\[
G(p)=\sigma\left(\frac{\hat u(p)-\tau_p}{T}\right).
\]

Use a hard threshold or a fixed top-area budget only for locked evaluation. Select \(\tau_p\) on base-class calibration episodes.

### Dense supervision

First train the repair operator in always-on mode and freeze it. For every training episode, decode:

- \(M_0\) from \(F_m\);
- \(M_{\mathrm{all}}\) from \(F_m+\bar R_\theta\).

Define a practical dense benefit teacher at mask resolution:

\[
u^*(p)=\ell(M_0(p),Y(p))-\ell(M_{\mathrm{all}}(p),Y(p)),
\]

then area-resize it to the post-memory feature grid. Positive values mean that the fixed repair reduced local segmentation loss.

This is a dense benefit proxy for a globally applied repair, not an exact causal contribution of one feature cell. Validate that approximation with a small sampled patch-toggle analysis; do not pay for exhaustive per-cell interventions.

Train the head with signed Smooth-L1 regression plus a small positive-benefit classification term. The final decision is still made by segmentation metrics, not dense AUPRC.

## 8. Episode Action-Value Head

The episode head predicts the exact counterfactual gain of the complete frozen local calibrator:

\[
\Delta_i=\operatorname{IoU}(M_{1,i},Y_i)-\operatorname{IoU}(M_{0,i},Y_i),
\]

where \(M_1\) is produced using the learned spatial gate and the frozen repair operator.

Recommended baseline evidence:

- IoU token;
- mean and standard deviation of all four raw mask tokens;
- object pointer and memory summary;
- global mean/max summaries of \(|F_m-F_q|\) and cosine conflict;
- predicted-IoU mean, maximum, and margin;
- mask entropy and multimask-disagreement statistics;
- predicted foreground area.

Use all four raw mask tokens rather than assuming that `selected_mask_token` always maps to the chosen multimask candidate. The current SAM2 configuration can make that assumption false.

Use a normalized MLP with a **linear signed output**. Do not reuse the current sigmoid `IoUHead` output layer. The current head estimates absolute baseline IoU; it cannot represent a negative intervention gain.

A useful initial loss is:

\[
\mathcal L_V=operatorname{SmoothL1}(\hat\Delta,\Delta)+
\lambda_{rank}\mathcal L_{pairwise-rank}.
\]

At inference:

\[
a=\mathbb 1[\hat\Delta-\lambda C_{2nd}>\tau_e],
\]

where \(C_{2nd}\) is the measured cost of the dense calibrator and second decoder. In the first implementation, absorb the fixed cost into a validation-selected \(\tau_e\).

## 9. Training Schedule

Keep SANSA, SAM2, and existing adapters frozen for the first attribution experiment.

### Stage A: Establish repair headroom

1. Disable hflip TTA, BRM, support-logit weighting, and memory-to-point prompting.
2. Train only \(R_\theta\), with `a=1` and `G=1`.
3. Use the project's segmentation objective, preferably BCE/focal plus Dice in the same mask space used by SANSA.
4. Keep the best checkpoint by base-validation mIoU, not training loss.
5. Stop if both always-on repair and oracle positive-gain episode routing have negligible headroom.

### Stage B: Learn spatial actionability

1. Freeze \(R_\theta\).
2. Generate \(M_0\), \(M_{\mathrm{all}}\), and the signed dense benefit teacher online or in a provenance-complete cache.
3. Train \(U_\phi\).
4. Select the spatial threshold/area budget on base calibration episodes.
5. Freeze \(R_\theta\) and \(U_\phi\).

### Stage C: Learn episode routing

1. Produce the complete calibrated result \(M_1\) for every base-training episode.
2. Store signed \(\Delta\)IoU together with baseline decoder evidence.
3. Train \(V_\psi\) with a linear output.
4. Select \(\tau_e\) using the final mIoU-versus-compute Pareto curve on base calibration episodes.
5. Freeze all thresholds before novel-class or named benchmark evaluation.

### Optional Stage D

Only after the frozen attribution experiment is positive, try a low-learning-rate joint fine-tune of the repair operator and spatial head. Retrain the gain head afterwards because the intervention distribution has changed. Jointly training SANSA adapters from the start is not the default experiment.

## 10. Data Splits and Cache Provenance

For each strict-FSS fold:

- train every new head only on base classes;
- split by query image/identity, not by cached record;
- keep base-train, base-calibration, and base-validation groups disjoint;
- use novel classes only after architecture, thresholds, and budgets are frozen.

Every cache must record:

- dataset, fold, shot count, episode/query identity, and seed;
- SANSA checkpoint path and hash;
- calibrator checkpoint path and hash;
- SAM2 size/configuration and channel factor;
- enabled intervention flags;
- baseline and calibrated IoUs;
- the exact feature schema and target definition.

Do not allow an empty SANSA checkpoint path for evidence collection. Do not mix 1-shot and 5-shot records unless shot count is an explicit model input and the mixture is a deliberate experiment.

Train the dense head online if storing dense features would dominate disk usage. The episode head needs only tokens and pooled statistics and can use a compact cache.

## 11. Required Causal Comparisons

Use the identical frozen repair operator in every row.

| ID | Episode trigger | Spatial trigger | Purpose |
|---|---|---|---|
| B0 | None | None | Unchanged SANSA baseline |
| B1 | Always | All pixels | Operator headroom |
| B2 | Always | Learned dense benefit | Value of spatial selection alone |
| B3 | Random matched rate | Random equal-area | Chance control |
| B4 | Entropy/risk matched rate | Entropy equal-area | Generic uncertainty control |
| B5 | Matched expected-IoU risk | Same dense gate as B2 | Current-risk versus action-value routing |
| B6 | Predicted action value | Learned dense benefit | Full AV-PMC |
| B7 | Shuffled/inverted value | Shuffled/inverted map | Negative control |
| B8 | Oracle positive gain | Oracle positive-benefit area | Operator upper bound |

Match episode intervention rate, activated spatial area, number of second decoder calls, and training exposure wherever the comparison is intended to test selection quality.

For B5, train a head with the same evidence, architecture, split, and optimization budget as the action-value head, but target baseline IoU instead of delta-IoU. The existing project `IoUHead` may be reported as an extra historical baseline, but it is not a clean risk-versus-value comparison because its feature schema and training distribution may differ. Repeat random routing with multiple random seeds rather than reporting one random draw.

Do not combine AV-PMC with hflip, BRM, support weighting, or memory-to-point prompting until B6 is independently positive. Afterwards, compare AV-PMC directly with memory-to-point because both pay for a second decoder call but change different inputs.

## 12. Minimal Evidence That “Uncertainty Exists”

Do not build a large standalone uncertainty study. Report only:

1. Distribution of true \(\Delta\)IoU, including positive-repair and harmful-repair rates.
2. Observed \(\Delta\)IoU by predicted-value quartile.
3. Full-system negative-repair rate.
4. One dense positive-benefit AUPRC or sampled patch-toggle agreement.

Actionable uncertainty exists for this module only when:

- the fixed operator has heterogeneous effects across episodes or regions;
- oracle selection is materially better than random selection;
- the learned ordering transfers to held-out episodes;
- learned selection improves final segmentation or preserves it with less computation.

High entropy alone is not sufficient evidence.

## 13. Evaluation Order

### Primary strict-FSS track

1. Development smoke tests on one held-out base-validation fold.
2. PASCAL-5i 1-shot and 5-shot for fast full-fold screening.
3. COCO-20i 1-shot and 5-shot for the main hard-benchmark confirmation.
4. Report per-fold and mean mIoU and FB-IoU.

### Secondary part track

1. PASCAL-Part under the existing generalist/in-context protocol.
2. PACO-Part under the same clearly labeled protocol.
3. Report mIoU, FB-IoU, and boundary IoU/F-score.

Keep strict FSS and generalist part results in separate tables and claims. If PACO data is used during training, label PACO-Part as in-domain.

## 14. Go/No-Go Rules

### Keep the repair operator when

- always-on repair improves the held-out base-validation result; or
- oracle routing reveals a useful positive-gain subset large enough to exploit.

### Claim uncertainty-driven calibration when

- B6 beats B3 and B4 at matched intervention rate and area;
- B6 beats the existing expected-IoU risk gate B5, showing that intervention value adds information beyond current-mask quality; and
- the final gain has a paired confidence interval above zero or exceeds the working screen threshold of `max(0.5 mIoU point, 2 x baseline seed SD)`.

### Claim efficiency when

- B6 remains within 0.2 mIoU point of B1/B2 while reducing second decoder calls by at least 30%.

### Redesign or drop AV-PMC when

- the always-on and oracle runs both show negligible headroom;
- the learned gate does not beat random at a matched budget;
- gain appears only in smoke subsets or only after combining another repair;
- the result depends on tuning thresholds on novel/test episodes.

## 15. Implementation Contract for a Coding Agent

The first implementation must satisfy all of the following:

- one default-off AV-PMC flag; disabled mode reproduces baseline output exactly;
- insertion in SANSA, not inside generic `SAM2Base`;
- no hard-coded feature resolution or batch size;
- zero-initialized final residual projection;
- separate signed outputs for spatial benefit and episode gain;
- reuse of cached `pix_feat_with_mem`, prompt inputs, and high-resolution features for the second decoder;
- baseline and calibrated decoder traces kept separately;
- no silent use of `selected_mask_token` as the winning candidate token;
- new parameter names included in the trainability filter, which currently enables only names containing `adapter` or `brm`;
- checkpoint/config hashes written into every cache and result file;
- explicit incompatibility checks for the initial standalone experiment;
- matched episode IDs for every ablation;
- no use of simulated midterm values as targets or expected results.

Suggested returned diagnostics are:

- `baseline_decoder_output`;
- `calibrated_decoder_output` when executed;
- `predicted_delta_iou`;
- `episode_gate`;
- `spatial_benefit_map` and activated-area ratio;
- residual norm;
- measured extra latency.

## 16. What Not to Add in Version 1

- Monte Carlo sampling or ensembles;
- calibration inside every SAM2 stage;
- high-resolution boundary refinement inside AV-PMC;
- query pseudo-memory writing;
- automatic point generation;
- joint hflip, BRM, MTP, and support-memory weighting;
- a full SAM2 or AdaptFormer fine-tune;
- exhaustive per-cell causal decoding.

These may be challenger experiments later. Version 1 must first answer whether post-memory repair benefit is predictable and useful for final segmentation.

## 17. Recommended First Experiment

Run a small but causally interpretable headroom study before building the gain heads:

1. Unchanged local SANSA baseline on a fixed held-out base-validation episode list.
2. Zero-initialized always-on repair operator, with SANSA frozen.
3. Baseline-versus-repair paired \(\Delta\)IoU distribution.
4. Oracle positive-gain episode selection at several budgets.

If this experiment shows no useful headroom, redesign the residual operator before implementing uncertainty routing. If it shows heterogeneous positive and negative gains, proceed to the spatial and episode value heads.

For the first bounded architecture search, keep two depthwise-pointwise blocks fixed, try projection widths `{32, 64}` and residual scales `{0.05, 0.10, 0.20}`, and select by paired base-validation mIoU. After fixing the operator, evaluate spatial area budgets `{0.10, 0.25, 0.50, 1.00}` and episode intervention-rate budgets `{0.25, 0.50, 0.75, 1.00}`. Do not jointly search operator capacity, spatial threshold, and episode threshold on final benchmark episodes.
