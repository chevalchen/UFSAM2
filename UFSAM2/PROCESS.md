# UFSAM2 Process

## Current Scope

- Base model: SANSA copied under `UFSAM2/`.
- Current setting: mask-only few-shot segmentation.
- Support masks are dataset GT masks.
- First goal: prove frozen SANSA query-side tokens can predict true query IoU / failure risk.
- Do not change memory gating, support weighting, semantic negative prompting, or joint training yet.

## Operating Constraints

- Keep the center of gravity on mask-only frozen-SANSA uncertainty until Stage 1 and Stage 3 evidence is solid.
- Every experiment conclusion that changes direction or confidence must be recorded in this `PROCESS.md`.
- Prefer intervention gain over standalone correlation/AUROC when deciding whether a component matters.
- Add new mechanisms only after a minimal baseline proves the need; avoid joint training, memory gating, or semantic negative prompts for now.
- Use class-disjoint or cross-dataset checks before treating an uncertainty result as general.
- After each stage-level milestone, commit only the relevant code and documentation; never commit checkpoints, datasets, caches, or generated outputs.

## Research Plan

1. Stage 1: FSS-1000 mask-only expected IoU calibration.
   - Freeze SANSA.
   - Collect query decoder traces.
   - Train a lightweight MLP on token features to regress true IoU.
   - Metrics: Pearson, Spearman, AUROC for IoU thresholds, risk-coverage, ECE.

2. Stage 2: Pascal-Part class-disjoint mask-only.
   - Construct class as `object+part`.
   - Use class-disjoint folds.
   - Validate expected IoU, semantic ambiguity, and support reliability.

3. Stage 3: 5-shot support selection.
   - Run each support independently.
   - Compare all supports, random support, oracle best support, uncertainty-selected support.

4. Stage 4: PACO-Part generalization.
   - Validate expected-IoU calibration and support selection on a harder long-tail part dataset.
   - Treat memory-propagation risk as a paused side branch, not the main PACO objective.

## Code Changes So Far

- `models/sam2/modeling/sam/mask_decoder.py`
  - Save decoder internals:
    - `last_iou_token_out`
    - `last_mask_tokens_out`

- `models/sansa/model_utils.py`
  - Extended `DecoderOutput` with:
    - `ious`
    - `iou_token`
    - `mask_tokens`
    - `selected_mask_token`
    - `memory_summary`
    - `object_score_logits`

- `models/sam2/modeling/sam2_base.py`
  - Fill the new `DecoderOutput` fields in `_forward_sam_heads`.
  - Fixed the mask-as-output path so `out` is defined even when object pointers are disabled.

- `models/sansa/sansa.py`
  - Added `return_traces=True`.
  - Trace is recorded only for query frames, i.e. `idx >= n_shots`.
  - Query trace contains:
    - `sam_score`
    - `query_iou_token`
    - `query_mask_tokens`
    - `query_mask_token`
    - `query_obj_ptr`
    - `query_memory_summary`

- `collect_uncertainty_cache.py`
  - New script for collecting per-episode uncertainty cache.
  - Saves metadata, true IoU, SAM score, query tokens, object pointer, memory summary, and area statistics.

- `train_uncertainty_head.py`
  - New script for Stage 1 expected-IoU MLP training.
  - Default feature set is `query_iou_token + query_mask_token + query_obj_ptr`.
  - Reports MAE, RMSE, Pearson, Spearman, AUROC for IoU risk thresholds, ECE, and risk-coverage.

- `evaluate_support_selection.py`
  - New script for minimal Stage 3 intervention.
  - For each 5-shot episode, runs each support independently as a 1-shot episode.
  - Compares random support, SAM-score selected support, token-head selected support, oracle best support, and all-support SANSA.

## Environment

- Conda env: `sam2coco`
- Data root: `/data6/chensq/datasets`
- Pretrain root: `UFSAM2/pretrain`
- Run commands from:

```bash
cd /data6/chensq/UFSAM2/UFSAM2
```

Use `MPLCONFIGDIR=/tmp/matplotlib` to avoid Matplotlib cache warnings.

If using a specific physical GPU:

```bash
CUDA_VISIBLE_DEVICES=1
```

Inside Python this remaps the selected physical GPU to logical `cuda:0`, so use `--device cuda`, not `--device cuda:1`.

## FSS-1000 Sanity Checks

FSS-1000 is now treated as a sanity/calibration dataset, not the final proof: SANSA is near ceiling, failure cases are rare, and intervention gains are necessarily small.

Main 1-shot cache command:

```bash
MPLCONFIGDIR=/tmp/matplotlib conda run -n sam2coco python collect_uncertainty_cache.py \
  --dataset_file fss \
  --prompt mask \
  --shots 1 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.3 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_fss_fold0.pth \
  --cache_path output/fss_fold0_1shot_mask_uncertainty.pt
```

Use the same command with `--shots 5` and `--cache_path output/fss_fold0_5shot_mask_uncertainty.pt` for the 5-shot all-support cache.

Cache distributions:

| cache | episodes | mean IoU | IoU<0.5 | IoU<0.7 |
| --- | ---: | ---: | ---: | ---: |
| 1-shot | 2400 | 0.9122 | 2.71% | 6.29% |
| 5-shot all-supports | 2400 | 0.9205 | 1.96% | 4.87% |

Main class-disjoint token-head command:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path output/fss_fold0_1shot_mask_uncertainty.pt \
  --output_dir output/uncertainty_head_fss_1shot_class_split \
  --device cuda \
  --epochs 200 \
  --batch_size 128 \
  --feature_set tokens \
  --split_by class_id
```

Feature set: `query_iou_token + query_mask_token + query_obj_ptr`. Class split: train `1680/168` episodes/classes, val `360/36`, test `360/36`.

Class-disjoint result:

| setting | test MAE | test RMSE | test Pearson | test Spearman | test AUROC IoU<0.5 | test AUROC IoU<0.7 | test ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tokens`, class split | 0.0592 | 0.1317 | 0.4322 | 0.7094 | 0.7486 | 0.8091 | 0.0240 |
| SAM `sam_score`, same class split | 0.0696 | 0.1525 | 0.3135 | 0.7075 | 0.6437 | 0.7918 | 0.0675 |

Class-disjoint read:

- The token head loses much of the random-split severe-failure AUROC, so random split was optimistic.
- It still beats `sam_score` on the same held-out classes, especially calibration and `IoU<0.5`.
- `memory_summary` did not help in FSS ablations; keep `tokens` as the main intervention head.
- FSS evidence is useful but not decisive because `IoU<0.7` gain is modest and the dataset is near ceiling.

Main full support-selection command:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib python evaluate_support_selection.py \
  --dataset_file fss \
  --prompt mask \
  --shots 5 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.3 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_fss_fold0.pth \
  --head_ckpt output/uncertainty_head_fss_1shot_class_split/uncertainty_head.pt \
  --output_path output/support_selection_full2400_class_head.json
```

Full 2400 support-selection result with class-disjoint token head:

| setting | mIoU | fail IoU<0.5 | risk IoU<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.9138 | 2.67% | 5.75% |
| SAM-score selected | 0.9170 | 2.58% | 5.67% |
| token selected | 0.9211 | 2.13% | 4.83% |
| oracle best | 0.9353 | 1.13% | 2.83% |
| all supports | 0.9205 | 1.96% | 4.87% |

Paired differences over 2400 episodes:

- token - random: `+0.0074 ± 0.0016` SE
- token - SAM-score: `+0.0042 ± 0.0013` SE
- token - all-supports: `+0.0006 ± 0.0014` SE
- oracle - token: `+0.0141 ± 0.0013` SE

Full-run selection diagnostics:

- token oracle match: `30.8%`
- SAM-score oracle match: `27.9%`
- token beats SAM-score: `33.8%`
- token - SAM-score has many exact ties because multiple supports often produce nearly identical masks.

High-spread subsets:

| support spread threshold | episodes | random | SAM-score | token | all supports | oracle | token - SAM | token - all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `>0.02` | 554 | 0.8172 | 0.8312 | 0.8486 | 0.8454 | 0.9025 | +0.0175 | +0.0032 |
| `>0.05` | 353 | 0.7733 | 0.7941 | 0.8219 | 0.8172 | 0.8988 | +0.0278 | +0.0047 |
| `>0.10` | 269 | 0.7495 | 0.7745 | 0.8096 | 0.8030 | 0.9025 | +0.0351 | +0.0066 |
| `>0.20` | 194 | 0.7333 | 0.7567 | 0.8052 | 0.8001 | 0.9178 | +0.0486 | +0.0051 |

Full Stage 3 read:

- Token support selection has a stable positive intervention gain over random and SAM-score selection on all 2400 FSS episodes.
- Token selected and all-supports are effectively tied globally; token is only `+0.0006` over all-supports.
- The intended behavior appears clearly in high-spread episodes, where support quality actually matters.
- FSS-1000 is close to ceiling, so this is a positive sanity check rather than the final proof; Pascal-Part/PACO-Part should be more diagnostic.
- Oracle remains meaningfully above token selection, so support reliability prediction still has headroom.

## Pascal-Part Class-Disjoint Check

Important configuration note:

- For Pascal-Part mask-only part FSS, use `pretrain/adapter_generalist.pth`.
- The generalist/universal adapters expect `--adaptformer_stages 2 3 --channel_factor 0.8`.
- The earlier Pascal-Part cache with `--channel_factor 0.3` produced many checkpoint shape-mismatch skips and should not be used as a main result.

Correct 1-shot Pascal-Part cache:

```bash
MPLCONFIGDIR=/tmp/matplotlib python collect_uncertainty_cache.py \
  --dataset_file pascal_part \
  --prompt mask \
  --shots 1 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --cache_path output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt
```

Correct cache distribution:

- Number of records: `2500`
- Mean true IoU: `0.4047`
- Min / max true IoU: `0.0 / 0.9739`
- `IoU < 0.5`: `57.12%`
- `IoU < 0.7`: `76.44%`
- Mean SAM score: `0.5410`

Class-disjoint token-head command:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path output/pascal_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt \
  --output_dir output/uncertainty_head_pascal_part_fold0_1shot_class_split \
  --device cpu \
  --epochs 200 \
  --batch_size 128 \
  --feature_set tokens \
  --split_by class_id
```

For this split:

- train: `1693` episodes, `21` classes
- val: `323` episodes, `4` classes
- test: `484` episodes, `6` classes

Class-disjoint result:

| setting | test MAE | test RMSE | test Pearson | test Spearman | test AUROC IoU<0.5 | test AUROC IoU<0.7 | test ECE | pred mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tokens`, class split | 0.1503 | 0.2094 | 0.7221 | 0.6921 | 0.8875 | 0.8473 | 0.0514 | 0.3246 |
| SAM `sam_score`, same split | 0.1914 | 0.2527 | 0.6916 | 0.6422 | 0.9005 | 0.8931 | 0.1330 | 0.4567 |

Pascal-Part read:

- Pascal-Part is much harder than FSS-1000 and is therefore a better diagnostic setting for uncertainty.
- Token head is substantially better calibrated than `sam_score`: lower MAE/RMSE/ECE and predicted mean close to test true mean `0.3276`.
- `sam_score` is overconfident on this split: predicted mean `0.4567` against true mean `0.3276`.
- `sam_score` has slightly stronger AUROC for the two hard thresholds, so failure ranking and expected-IoU calibration should be reported separately.
- Next necessary intervention check is Pascal-Part 5-shot support selection with the same generalist `cf0.8` setup.

The 200-episode pilot already showed the same trend as the full run, so the full 2500-episode result below is the main Pascal-Part intervention result.

Pascal-Part 5-shot support-selection full command:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib python evaluate_support_selection.py \
  --dataset_file pascal_part \
  --prompt mask \
  --shots 5 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --head_ckpt output/uncertainty_head_pascal_part_fold0_1shot_class_split/uncertainty_head.pt \
  --output_path output/support_selection_pascal_part_fold0_full_generalist_cf08.json
```

Pascal-Part support-selection full 2500-episode result:

| setting | mIoU | fail IoU<0.5 | risk IoU<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.3905 | 59.28% | 79.12% |
| SAM-score selected | 0.4282 | 53.44% | 74.32% |
| token selected | 0.4643 | 48.76% | 70.00% |
| oracle best | 0.5636 | 36.20% | 60.68% |
| all supports | 0.4645 | 49.84% | 72.32% |

Paired differences over 2500 episodes:

- token - random: `+0.0738 ± 0.0047` SE
- token - SAM-score: `+0.0361 ± 0.0037` SE
- token - all-supports: `-0.0002 ± 0.0033` SE
- oracle - token: `+0.0993 ± 0.0032` SE

Selection diagnostics:

- token oracle match: `31.44%`
- SAM-score oracle match: `23.76%`
- token beats SAM-score: `35.12%`
- token and SAM-score choose the same support in `1002/2500` episodes.
- token beats all-supports in `47.56%` of episodes and loses in `48.72%`; globally they are effectively tied.

High-spread subsets:

| support spread threshold | episodes | random | SAM-score | token | all supports | oracle | token - SAM | token - all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `>0.05` | 2257 | 0.4007 | 0.4425 | 0.4823 | 0.4822 | 0.5911 | +0.0398 | +0.0001 |
| `>0.10` | 2020 | 0.3988 | 0.4453 | 0.4892 | 0.4889 | 0.6071 | +0.0438 | +0.0003 |
| `>0.20` | 1581 | 0.3987 | 0.4561 | 0.5090 | 0.5069 | 0.6420 | +0.0529 | +0.0021 |
| `>0.30` | 1266 | 0.4114 | 0.4823 | 0.5423 | 0.5372 | 0.6823 | +0.0599 | +0.0051 |

Full Pascal-Part Stage 3 read:

- Token support selection has a stable intervention gain over random and SAM-score on all 2500 Pascal-Part episodes.
- Token selected and all-supports are effectively tied globally, but token has lower severe-failure and `IoU<0.7` risk than all-supports.
- In high-spread episodes, token selection gradually beats all-supports, suggesting the uncertainty head is useful specifically when support quality varies.
- The oracle gap remains large, so the current token head is not a solved support-reliability estimator.
- This is stronger Stage 3 evidence than FSS-1000 because Pascal-Part is lower-IoU, part-level, and not ceiling-limited.

Cross-seed check, full 2500-episode support selection:

| seed | random | SAM-score | token | all supports | oracle | token - random | token - SAM | token - all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3905 | 0.4282 | 0.4643 | 0.4645 | 0.5636 | +0.0738 | +0.0361 | -0.0002 |
| 1 | 0.3930 | 0.4232 | 0.4626 | 0.4591 | 0.5535 | +0.0695 | +0.0394 | +0.0034 |
| 2 | 0.3977 | 0.4308 | 0.4592 | 0.4609 | 0.5563 | +0.0614 | +0.0284 | -0.0018 |
| mean ± sd | 0.3938 ± 0.0037 | 0.4274 ± 0.0039 | 0.4620 ± 0.0026 | 0.4615 ± 0.0028 | 0.5578 ± 0.0052 | +0.0682 ± 0.0063 | +0.0346 ± 0.0057 | +0.0005 ± 0.0027 |

Cross-seed risk check:

| setting | fail IoU<0.5, seeds 0/1/2 | risk IoU<0.7, seeds 0/1/2 |
| --- | ---: | ---: |
| random | 59.28 / 58.88 / 57.88% | 79.12 / 78.44 / 78.80% |
| SAM-score selected | 53.44 / 53.88 / 52.56% | 74.32 / 73.56 / 73.48% |
| token selected | 48.76 / 48.56 / 48.56% | 70.00 / 70.28 / 70.52% |
| all supports | 49.84 / 49.76 / 49.48% | 72.32 / 71.92 / 72.28% |

Cross-seed read:

- The Pascal-Part support-selection conclusion is not an episode-seed artifact.
- Token selection is consistently better than random and SAM-score in all three seeds.
- Token and all-supports remain statistically tied globally across seeds.
- Token consistently has lower failure/risk rates than all-supports, even when mean mIoU is tied.
- The high-spread `>0.30` subset keeps the intended trend: token beats SAM-score in all seeds and is near/all-supports or above all-supports depending on the sampled episodes.

## Memory-Propagation Risk Probe

Important correction:

- SANSA standard FSS inference does not use query-to-query sequential propagation.
- Reference/support frames are encoded into memory without undergoing Memory Attention; this keeps target prediction invariant to support order.
- At inference, each target/query is segmented independently given the annotated references.
- Therefore this probe is a test-time sequential memory extension, not a SANSA baseline behavior.

New script:

- `evaluate_memory_propagation_risk.py`
- Builds a chain `support -> q1 -> q2 -> q3`.
- Compares:
  - independent: each query is run as `support -> q_t`
  - sequential: queries are run as `support -> q1 -> q2 -> q3`, so pseudo-query predictions are written to memory.
- Defines harm as `independent_iou - sequential_iou`; positive harm means pseudo-query memory hurt that query.
- First version supports class-indexed datasets such as Pascal-Part. PACO-Part needs a dedicated class-conditioned sampler because its current dataset class ignores `idx`.

Smoke test passed with CPU/tiny for one Pascal-Part chain:

```bash
MPLCONFIGDIR=/tmp/matplotlib conda run -n sam2coco python evaluate_memory_propagation_risk.py \
  --dataset_file pascal_part \
  --prompt mask \
  --shots 1 \
  --fold 0 \
  --sam2_version tiny \
  --adaptformer_stages 2 3 \
  --channel_factor 0.3 \
  --device cpu \
  --data_root /data6/chensq/datasets \
  --chain_length 2 \
  --max_chains 1 \
  --output_path /tmp/memory_risk_smoke.json
```

Recommended Pascal-Part debug command:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib python evaluate_memory_propagation_risk.py \
  --dataset_file pascal_part \
  --prompt mask \
  --shots 1 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --head_ckpt output/uncertainty_head_pascal_part_fold0_1shot_class_split/uncertainty_head.pt \
  --chain_length 3 \
  --max_chains 100 \
  --output_path output/memory_risk_pascal_part_fold0_100_generalist_cf08.json
```

Pascal-Part 100-chain debug result:

| query | independent mIoU | sequential mIoU | mean harm | harm>0.01 | harm>0.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q1 | 0.4057 | 0.4057 | 0.0000 | 0.0% | 0.0% |
| q2 | 0.3765 | 0.3589 | +0.0176 | 31.0% | 18.0% |
| q3 | 0.4104 | 0.4116 | -0.0012 | 37.0% | 18.0% |
| downstream q2/q3 | 0.3935 | 0.3853 | +0.0082 | 34.0% | 18.0% |

Additional debug observations:

- q1 is effectively identical because both paths read only support memory.
- q2 is hurt on average, which confirms pseudo-query memory can contaminate downstream prediction.
- q3 mean recovers, but the distribution is wide: downstream harm ranges roughly from `-0.648` to `+0.648`.
- The current Pascal expected-IoU head is weak for predicting next-step harm: `prev_token_score_vs_next_harm_pearson = -0.102`.
- A rough offline gating proxy can improve over always sequential by about `+0.01` downstream mIoU, but a real gating script is needed because independent-vs-sequential replay is not equivalent to selectively dropping only one pseudo-query memory.

Debug read:

- Sequential memory extension has a real long-tail risk signal, but not a simple monotonic failure mode.
- The current expected-IoU head is not yet a strong propagation-risk head.
- Marker: pause this branch here. If resumed, implement explicit memory gating during the sequential forward pass and compare: always sequential, support-only independent, random gating, score-threshold gating, and oracle gating.
- Main line continues with harder part-setting generalization, especially PACO-Part expected-IoU calibration and support selection.

## PACO-Part Main Line

Current purpose:

- Test whether Pascal-Part uncertainty/support-selection conclusions generalize to a harder part setting.
- PACO-Part is long-tail and more diverse than Pascal-Part, so it should be more diagnostic than FSS-1000.
- Use `pretrain/adapter_generalist.pth` with `--adaptformer_stages 2 3 --channel_factor 0.8`.
- Current PACO dataset sampler is stochastic and ignores `idx`; use fixed seeds for reproducibility. A fixed episode-list sampler can be added later if PACO becomes a final table.

PACO dataset smoke checks:

- Data exists under `/data6/chensq/datasets/PACO-Part`.
- Fold 0 validation currently exposes `2500` episodes and `79` sampled part classes.
- `collect_uncertainty_cache.py` smoke passed for 1 CPU/tiny episode.

Recommended 1-shot PACO cache:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib python collect_uncertainty_cache.py \
  --dataset_file paco_part \
  --prompt mask \
  --shots 1 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --cache_path output/paco_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt
```

PACO 1-shot cache result:

- Number of records: `2500`
- Mean true IoU: `0.4329`
- Min / max true IoU: `0.0 / 0.9904`
- `IoU < 0.5`: `55.84%`
- `IoU < 0.7`: `74.60%`
- Mean SAM score: `0.5501`

Train PACO class-split uncertainty head:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path output/paco_part_fold0_1shot_mask_uncertainty_generalist_cf08.pt \
  --output_dir output/uncertainty_head_paco_part_fold0_1shot_class_split \
  --device cpu \
  --epochs 200 \
  --batch_size 128 \
  --feature_set tokens \
  --split_by class_id
```

PACO class-disjoint split:

- train: `1709` episodes, `55` classes
- val: `355` episodes, `11` classes
- test: `436` episodes, `13` classes

PACO class-disjoint result:

| setting | test MAE | test RMSE | test Pearson | test Spearman | test AUROC IoU<0.5 | test AUROC IoU<0.7 | test ECE | pred mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tokens`, class split | 0.2837 | 0.3632 | 0.2196 | 0.2046 | 0.5990 | 0.6483 | 0.1331 | 0.3864 |
| SAM `sam_score`, same split | 0.2673 | 0.3540 | 0.3747 | 0.3183 | 0.6586 | 0.7529 | 0.1476 | 0.5703 |

PACO class-split read:

- Unlike Pascal-Part, the token head does not beat `sam_score` on held-out PACO classes.
- `sam_score` is overconfident, but it ranks PACO failures better than the current token head.
- Token ECE is slightly better, but correlation/AUROC are weak; this suggests the current expected-IoU head is not yet robust to PACO's long-tail part distribution.
- PACO should therefore be treated as a failure/generalization stress test for the current uncertainty representation.
- Intervention remains the deciding metric: run support selection before changing the head.

Support-selection debug after the head is trained:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib python evaluate_support_selection.py \
  --dataset_file paco_part \
  --prompt mask \
  --shots 5 \
  --fold 0 \
  --sam2_version large \
  --adaptformer_stages 2 3 \
  --channel_factor 0.8 \
  --device cuda \
  --data_root /data6/chensq/datasets \
  --resume pretrain/adapter_generalist.pth \
  --head_ckpt output/uncertainty_head_paco_part_fold0_1shot_class_split/uncertainty_head.pt \
  --max_episodes 200 \
  --output_path output/support_selection_paco_part_fold0_200_generalist_cf08.json
```

Repeat with `--seed 1` and `--output_path output/support_selection_paco_part_fold0_200_generalist_cf08_seed1.json`.

PACO support-selection cross-seed result, 2 seeds x 200 episodes:

| setting | mIoU | fail IoU<0.5 | risk IoU<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.4520 | 54.3% | 71.8% |
| SAM-score selected | 0.4830 | 49.3% | 65.3% |
| token selected | 0.4858 | 50.0% | 67.8% |
| all supports | 0.4948 | 48.5% | 66.5% |
| oracle best | 0.5834 | 36.8% | 58.0% |

Per-seed support-selection summaries:

| seed | random | SAM-score | token | all supports | oracle | token - SAM | token - all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4540 | 0.4867 | 0.4921 | 0.5029 | 0.5918 | +0.0054 | -0.0108 |
| 1 | 0.4500 | 0.4794 | 0.4794 | 0.4867 | 0.5750 | +0.0001 | -0.0073 |

Mean cross-seed differences:

- token - random: `+0.0338`
- token - SAM-score: `+0.0027`
- token - all-supports: `-0.0090`
- oracle - token: `+0.0976`
- token oracle match: `35.5%`
- SAM-score oracle match: `32.5%`
- token beats SAM-score: `27.8%`

High-spread subsets:

| support spread threshold | episodes | random | SAM-score | token | all supports | oracle | token - SAM | token - all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `>0.05` | 171 | 0.4536 | 0.4920 | 0.4976 | 0.5111 | 0.6137 | +0.0056 | -0.0136 |
| `>0.10` | 154 | 0.4311 | 0.4746 | 0.4800 | 0.4929 | 0.6050 | +0.0054 | -0.0129 |
| `>0.20` | 126 | 0.4373 | 0.4894 | 0.4940 | 0.5083 | 0.6379 | +0.0047 | -0.0142 |
| `>0.30` | 95 | 0.4339 | 0.5172 | 0.5178 | 0.5354 | 0.6796 | +0.0006 | -0.0176 |

PACO support-selection read:

- Token selection improves over random, but the advantage over SAM-score is too small to count as a useful intervention gain.
- Token is consistently below all-supports in mean mIoU and has worse `IoU<0.7` risk than both SAM-score and all-supports.
- High-spread subsets do not reveal a Pascal-like token advantage; token and SAM-score are effectively tied there.
- This matches the PACO calibration result: the current token head is weak under PACO's long-tail part distribution.
- Treat PACO support selection as a negative/weak-generalization result for the current expected-IoU head; do not use the tiny token-over-SAM difference as evidence.

## Notes

- For FSS-1000, `--fold` is currently metadata only; the dataset split is controlled by `datasets/fss.py`.
- Keep `--adaptformer_stages` and `--channel_factor` identical to the reproduced SANSA baseline.
- `sam_score` is the mask decoder's estimated IoU score, not the true IoU.
- The cache script currently stores the last query trace. This matches the current SANSA FSS input layout: support frames followed by one query frame.
