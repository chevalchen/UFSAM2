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

4. Stage 4: PACO-Part propagation risk.
   - Test support -> query_1 -> query_2 -> query_3 pseudo-video memory updates.
   - Compare default memory, no pseudo-query memory, random gating, uncertainty gating, oracle gating.

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

## Verified Smoke Tests

CPU tiny smoke test succeeded:

```bash
MPLCONFIGDIR=/tmp/matplotlib conda run -n sam2coco python collect_uncertainty_cache.py \
  --dataset_file fss \
  --prompt mask \
  --shots 1 \
  --sam2_version tiny \
  --device cpu \
  --data_root /data6/chensq/datasets \
  --max_episodes 1 \
  --cache_path /tmp/ufs2_smoke_uncertainty.pt \
  --output_dir /tmp/ufs2_smoke \
  --name_exp cache_smoke
```

GPU large debug run succeeded for 10 FSS episodes:

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
  --max_episodes 10 \
  --cache_path output/debug_fss_10.pt
```

The debug cache contains 10 records with fields:

- `dataset`
- `fold`
- `shot`
- `episode_idx`
- `class_id`
- `category`
- `query_name`
- `support_names`
- `true_iou`
- `sam_score`
- `query_iou_token`
- `query_mask_token`
- `query_mask_tokens`
- `query_obj_ptr`
- `query_memory_summary`
- `pred_area`
- `gt_area`
- `support_area_mean`
- `support_area_std`

Expected tensor shapes:

- `query_iou_token`: `(256,)`
- `query_mask_token`: `(256,)`
- `query_mask_tokens`: `(4, 256)`
- `query_obj_ptr`: `(256,)`
- `query_memory_summary`: `(256,)`

## Full FSS Cache Commands

1-shot FSS:

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

5-shot FSS:

```bash
MPLCONFIGDIR=/tmp/matplotlib conda run -n sam2coco python collect_uncertainty_cache.py \
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
  --cache_path output/fss_fold0_5shot_mask_uncertainty.pt
```

Check cache:

```bash
conda run -n sam2coco python -c "import torch; r=torch.load('output/debug_fss_10.pt', map_location='cpu', weights_only=False); print(len(r)); print(r[0].keys())"
```

Full 1-shot FSS cache check result:

- Number of records: `2400`
- Mean true IoU: `0.9122`
- Min / max true IoU: `0.0 / 0.9952`
- `IoU < 0.5`: `2.71%`
- `IoU < 0.7`: `6.29%`

## Train Stage 1 Uncertainty Head

Default token features:

```text
query_iou_token + query_mask_token + query_obj_ptr
```

Train on the full FSS 1-shot cache:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path output/fss_fold0_1shot_mask_uncertainty.pt \
  --output_dir output/uncertainty_head_fss_1shot \
  --device cuda \
  --epochs 200 \
  --batch_size 128 \
  --feature_set tokens
```

Class-disjoint split, grouped by `class_id`:

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

For the current FSS cache this split gives:

- train: `1680` episodes, `168` classes
- val: `360` episodes, `36` classes
- test: `360` episodes, `36` classes

Class-disjoint result:

| setting | test MAE | test RMSE | test Pearson | test Spearman | test AUROC IoU<0.5 | test AUROC IoU<0.7 | test ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tokens`, class split | 0.0592 | 0.1317 | 0.4322 | 0.7094 | 0.7486 | 0.8091 | 0.0240 |
| SAM `sam_score`, same class split | 0.0696 | 0.1525 | 0.3135 | 0.7075 | 0.6437 | 0.7918 | 0.0675 |

Class-disjoint read:

- The token head loses much of the random-split severe-failure AUROC, so random split was optimistic.
- It still beats `sam_score` on the same held-out classes, especially calibration and `IoU<0.5`.
- `IoU<0.7` gain is modest; this should be treated as sanity-check evidence, not yet a final intervention result.

## Stage 3 Support Selection

Minimal intervention script:

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
  --max_episodes 20 \
  --output_path output/support_selection_debug20.json
```

This produces summary metrics:

- `miou_random`
- `miou_sam_score`
- `miou_token`
- `miou_oracle`
- `miou_all_supports`
- `token_oracle_match`
- `sam_score_oracle_match`
- `token_beats_sam_score`

Debug20 result:

| setting | mIoU |
| --- | ---: |
| random | 0.9450 |
| SAM-score selected | 0.9432 |
| token selected | 0.9448 |
| oracle best | 0.9473 |
| all supports | 0.9443 |

Debug20 read:

- This is a smoke test only; all methods are within about `0.004` mIoU.
- Token selected beats SAM-score selected by `0.0016` average, but does not beat random on this tiny sample.
- Need a larger support-selection run before making any intervention claim.

Support-selection 200-episode result:

| setting | mIoU | fail IoU<0.5 | risk IoU<0.7 |
| --- | ---: | ---: | ---: |
| random | 0.8972 | 4.5% | 7.0% |
| SAM-score selected | 0.9009 | 3.5% | 8.0% |
| token selected | 0.9115 | 3.0% | 6.0% |
| oracle best | 0.9324 | 1.0% | 2.0% |
| all supports | 0.9091 | 2.5% | 7.0% |

Paired differences over 200 episodes:

- token - random: `+0.0143 ± 0.0065` SE
- token - SAM-score: `+0.0107 ± 0.0062` SE
- token - all-supports: `+0.0025 ± 0.0087` SE
- oracle - token: `+0.0209 ± 0.0073` SE

Selection diagnostics:

- token oracle match: `33.0%`
- SAM-score oracle match: `28.5%`
- token beats SAM-score: `33.5%`
- token and SAM-score choose different support in `127/200` episodes.

High-spread subset, where best support minus worst support IoU is greater than `0.05`:

- subset size: `31/200`
- random: `0.6929`
- SAM-score selected: `0.7176`
- token selected: `0.7849`
- all supports: `0.7684`
- oracle best: `0.9004`

Stage 3 read:

- Token selection shows a real positive intervention signal at 200 episodes.
- The effect is modest over all episodes because most FSS-1000 support choices are already similar; only `15.5%` of episodes have support spread greater than `0.05`.
- In high-spread episodes, token selection is meaningfully better than SAM-score and all-supports, which matches the intended use case.
- Token does not close the oracle gap; support reliability remains partially unsolved.
- Next check should be either a larger run or a harder dataset/part setting where support quality variance is larger.

5-shot all-support cache result:

- Number of records: `2400`
- Mean true IoU: `0.9205`
- Min / max true IoU: `0.0 / 0.9949`
- `IoU < 0.5`: `1.96%`
- `IoU < 0.7`: `4.87%`
- Compared with 1-shot cache, mean IoU is `+0.0083`, `IoU<0.5` drops by `0.75` points, and `IoU<0.7` drops by `1.42` points.

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

Optional memory-summary feature ablation:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train_uncertainty_head.py \
  --cache_path output/fss_fold0_1shot_mask_uncertainty.pt \
  --output_dir output/uncertainty_head_fss_1shot_tokens_mem \
  --device cuda \
  --epochs 200 \
  --batch_size 128 \
  --feature_set tokens_mem
```

First `tokens` run result on `output/fss_fold0_1shot_mask_uncertainty.pt`:

| split | MAE | RMSE | Pearson | Spearman | AUROC IoU<0.5 | AUROC IoU<0.7 | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.0062 | 0.0109 | 0.9972 | 0.9804 | 0.9996 | 0.9997 | 0.0014 |
| val | 0.0623 | 0.1242 | 0.4856 | 0.6344 | 0.8244 | 0.8359 | 0.0265 |
| test | 0.0501 | 0.1064 | 0.5863 | 0.7164 | 0.9237 | 0.8565 | 0.0204 |

Interpretation:

- Query tokens contain useful expected-IoU / failure-risk signal.
- The train-vs-val gap is large, so regularization and split protocol need attention.
- `IoU<0.7` is more stable than `IoU<0.5` on FSS because failure samples are rare.
- Next useful ablations: `tokens_mem`, linear/ridge baseline, smaller hidden dim, stronger dropout, and class/category-disjoint split if possible.

Follow-up ablations:

| setting | split | MAE | RMSE | Pearson | Spearman | AUROC IoU<0.5 | AUROC IoU<0.7 | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM `sam_score` | full | 0.0603 | 0.1342 | 0.4423 | 0.7457 | 0.7986 | 0.8293 | 0.0566 |
| `tokens` | test | 0.0501 | 0.1064 | 0.5863 | 0.7164 | 0.9237 | 0.8565 | 0.0204 |
| `tokens_mem` | test | 0.0505 | 0.1110 | 0.5199 | 0.7144 | 0.8597 | 0.8065 | 0.0238 |
| `tokens`, hidden 64, dropout 0.3, wd 1e-3 | test | 0.0452 | 0.0994 | 0.6397 | 0.7060 | 0.8572 | 0.8201 | 0.0256 |

Current read:

- `memory_summary` does not help on FSS-1000 Stage 1; it likely adds dataset-specific or noisy context.
- The smaller regularized head improves regression error and Pearson, but hurts low-IoU risk AUROC.
- For intervention-style ranking, keep the original `tokens` head as the main baseline for now.
- For calibrated expected IoU, the smaller regularized head is a reasonable secondary baseline.
- SAM's own `sam_score` has strong rank correlation, but it is poorly calibrated and much weaker for `IoU<0.5` failure detection than the token head.

## Notes

- For FSS-1000, `--fold` is currently metadata only; the dataset split is controlled by `datasets/fss.py`.
- Keep `--adaptformer_stages` and `--channel_factor` identical to the reproduced SANSA baseline.
- `sam_score` is the mask decoder's estimated IoU score, not the true IoU.
- The cache script currently stores the last query trace. This matches the current SANSA FSS input layout: support frames followed by one query frame.
