#!/usr/bin/env bash
# One paired dataset/fold/shot job, using the existing EXP-008 server environment.
set -euo pipefail
if (( $# != 4 )); then
  echo "Usage: bash scripts/run_exp009.sh {paco_part|pascal_part|fss|partimagenet} FOLD {1|5} NEW_ABSOLUTE_OUTPUT_DIR" >&2
  exit 2
fi
dataset=$1
fold=$2
shots=$3
out=$4
[[ "$dataset" =~ ^(paco_part|pascal_part|fss|partimagenet)$ && "$fold" =~ ^[0-3]$ && "$shots" =~ ^(1|5)$ && "$out" = /* ]]
[[ "$dataset" != fss || "$fold" == 0 ]]
root=$(git rev-parse --show-toplevel)
base=${SAM2_CHECKPOINT:-/data6/chensq/SANSA/pretrain/sam2_hiera_small.pt}
adapter=${ADAPTER:-/data6/chensq/UFSAM2/exp008_artifacts/train_v1/checkpoint0004.pth}
data=${DATA_ROOT:-/data6/chensq/datasets}
cd "$root/UFSAM2"
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
git rev-parse HEAD
printf '%s\n' "$dataset fold=$fold shots=$shots data=$data"
printf '%s  %s\n' 95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3 "$base" \
  b3229c45a3d8304cfd059ecfd693dacee148f0dedafd0724c2c585f188da37b9 "$adapter" | sha256sum --check
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
extra=()
if [[ "$dataset" == partimagenet ]]; then
  [[ "$fold" == 0 ]]
  sha256sum "${PART_ANNOTATIONS:?Specify the held-out part annotation JSON}"
  extra=(--part_annotations "$PART_ANNOTATIONS" --part_images "${PART_IMAGES:?Specify the image root}")
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for mode in off area nesting both; do
  python inference_fss.py --sam2_version small --sam2_checkpoint "$base" --resume "$adapter" \
    --adaptformer_stages 2 3 --channel_factor 0.8 --dataset_file "$dataset" --data_root "$data" \
    --fold "$fold" --shots "$shots" --prompt mask --seed 0 --num_workers 0 --threshold 0.5 \
    --granularity "$mode" --audit_candidates --output_dir "$out" --name_exp "$mode" "${extra[@]}" > "$out/$mode.log" 2>&1
done
python - "$out" <<'PY'
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
paired = None
control = None
summary = {}
for mode in ("off", "area", "nesting", "both"):
    text = (out / f"{mode}.log").read_text()
    records = [json.loads(line.split("candidate_audit ", 1)[1])
               for line in text.splitlines() if "candidate_audit " in line]
    assert records, f"No episode audit for {mode}"
    keys = ("episode", "class", "query", "supports", "native_index", "intersections", "unions",
            "candidate_sha256")
    for index, row in enumerate(records):
        assert row["episode"] == index
        assert all(row[key] not in ("None", "[]", "") for key in ("query", "supports"))
        assert len(row["candidate_sha256"]) == len(row["intersections"]) == len(row["unions"])
        assert 0 <= row["selected_index"] < len(row["candidate_sha256"])
    signature = [tuple(row[key] for key in keys) for row in records]
    if paired is not None and signature != paired:
        index = next((i for i, (a, b) in enumerate(zip(signature, paired)) if a != b),
                     min(len(signature), len(paired)))
        raise RuntimeError(f"Episode or candidate mismatch: {mode}, episode {index}")
    paired = signature
    if control is None:
        assert all(row["selected_index"] == row["native_index"] for row in records)
        control = records
    switched = improved = worsened = unchanged = 0
    for before, after in zip(control, records):
        if before["selected_index"] == after["selected_index"]:
            assert (before["selected_intersection"], before["selected_union"]) == (
                after["selected_intersection"], after["selected_union"])
            continue
        switched += 1
        # Compare actual evaluated foreground IoU exactly, with empty unions treated as zero.
        old_inter, old_union = before["selected_intersection"], max(1, before["selected_union"])
        new_inter, new_union = after["selected_intersection"], max(1, after["selected_union"])
        delta = new_inter * old_union - old_inter * new_union
        improved += delta > 0
        worsened += delta < 0
        unchanged += delta == 0
    metrics = re.findall(r"Fold \d+ mIoU:\s*([\d.]+)\s+FB-IoU:\s*([\d.]+)", text)
    if len(metrics) != 1:
        raise RuntimeError(f"Missing or ambiguous final metrics: {mode}")
    summary[mode] = dict(episodes=len(records), mIoU=float(metrics[0][0]), fbIoU=float(metrics[0][1]),
                         paired=True, switched=switched, improved=improved, worsened=worsened,
                         unchanged=unchanged)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(summary)
PY
