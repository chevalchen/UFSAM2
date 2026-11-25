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
summary = {}
for mode in ("off", "area", "nesting", "both"):
    text = (out / f"{mode}.log").read_text()
    records = [json.loads(line.split("candidate_audit ", 1)[1])
               for line in text.splitlines() if "candidate_audit " in line]
    assert records, f"No episode audit for {mode}"
    # Identical IDs and candidate intersections/unions prove a paired selection comparison.
    signature = [tuple(row[key] for key in
                       ("episode", "class", "query", "supports", "native_index", "intersections", "unions"))
                 for row in records]
    if paired is not None and signature != paired:
        raise RuntimeError(f"Episode or candidate mismatch: {mode}")
    paired = signature
    metrics = re.findall(r"Fold \d+ mIoU:\s*([\d.]+)\s+FB-IoU:\s*([\d.]+)", text)
    if len(metrics) != 1:
        raise RuntimeError(f"Missing or ambiguous final metrics: {mode}")
    summary[mode] = dict(episodes=len(records), mIoU=float(metrics[0][0]), fbIoU=float(metrics[0][1]))
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(summary)
PY
