#!/usr/bin/env bash
# Full hold-out head-to-head on CPU (~25 min on 4 cores).
#
# Probes once on the select split, then compares every layer-selection strategy
# and the SVD in-place baseline on the untouched test split. Emits the run
# labels that scripts/make_phase3c_report.py expects, so the Phase 3C table
# regenerates straight after.
#
# Usage: ./scripts/run_headtohead.sh [--weights PATH] [--device cpu|cuda]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WEIGHTS="yolo11n-obb.pt"
DEVICE="cpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

H=datasets/dota128_holdout
SENS=results/summaries/structural_layer_sensitivity_holdout_select.json

[[ -d "$H" ]] || python scripts/create_dota128_holdout.py --seed 42

echo "### baseline calibration (test split)"
python scripts/evaluate_calibration.py --weights "$WEIGHTS" \
  --data-yaml "$H/dota128_holdout_test.yaml" --compression baseline \
  --run-label holdout_test_baseline --device "$DEVICE"

echo "### baseline latency reference"
python scripts/benchmark_inference.py --weights "$WEIGHTS" \
  --data-yaml "$H/dota128_holdout_test.yaml" --batch-size 1 --device "$DEVICE" \
  --run-label holdout_test_baseline > /dev/null

echo "### structural probe (select split)"
python scripts/compress_model.py --method structural-probe --rank-ratio 0.84 \
  --weights "$WEIGHTS" --data-yaml "$H/dota128_holdout_select.yaml" \
  --target-layers backbone neck --run-label holdout_select --device "$DEVICE"

# label | strategy | top-k | extra compress flags
ARMS=(
  "holdout_top3_sens|sensitivity|3|"
  "holdout_top5_sens|sensitivity|5|"
  "holdout_top3_pareto|pareto-gain|3|"
  "holdout_top5_pareto|pareto-gain|5|"
  "holdout_top8_pareto|pareto-gain|8|"
  "holdout_top5_qaoa|quantum-qaoa|5|--quantum-backend classical"
  "holdout_top5_qaoa_sim|quantum-qaoa|5|--quantum-backend qaoa"
)

for arm in "${ARMS[@]}"; do
  IFS='|' read -r label strategy k extra <<< "$arm"
  echo "### $label"
  # shellcheck disable=SC2086
  python scripts/compress_model.py --method structural-low-rank --rank-ratio 0.84 \
    --weights "$WEIGHTS" --selection-strategy "$strategy" --max-replaced-layers "$k" \
    --sensitivity-file "$SENS" \
    --bn-data-yaml "$H/dota128_holdout_train.yaml" \
    --eval-data-yaml "$H/dota128_holdout_test.yaml" \
    --bn-recalibration-batches 20 --run-label "$label" --device "$DEVICE" $extra

  python scripts/evaluate_calibration.py \
    --weights "runs/compressed/structural_low_rank/structural_low_rank_r0.840_${label}.pt" \
    --data-yaml "$H/dota128_holdout_test.yaml" --compression structural-low-rank \
    --rank-ratio 0.84 --run-label "${label}_test" --device "$DEVICE"
done

echo "### SVD in-place r=0.84 (reference)"
python scripts/compress_model.py --method low-rank --rank-ratio 0.84 \
  --weights "$WEIGHTS" --eval-data-yaml "$H/dota128_holdout_test.yaml" \
  --run-label holdout_low_rank_r084 --device "$DEVICE"
python scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.840/low_rank_r0.840.pt \
  --data-yaml "$H/dota128_holdout_test.yaml" --compression low-rank \
  --rank-ratio 0.84 --run-label holdout_low_rank_r084_test --device "$DEVICE"

echo "### quality gates"
python - <<'PY'
import glob, json, subprocess

def latest(pattern, label):
    hits = [p for p in glob.glob(pattern) if json.load(open(p)).get("run_label") == label]
    if not hits:
        raise SystemExit(f"no summary with run_label={label}")
    return max(hits, key=lambda p: json.load(open(p)).get("run_id", ""))

baseline = latest("results/summaries/calibration_*.json", "holdout_test_baseline")
for label in [
    "holdout_top3_sens", "holdout_top5_sens", "holdout_top3_pareto",
    "holdout_top5_pareto", "holdout_top8_pareto", "holdout_top5_qaoa",
    "holdout_top5_qaoa_sim",
]:
    subprocess.run([
        "python", "scripts/check_reliability_gate.py",
        "--baseline", baseline,
        "--candidate", latest("results/summaries/calibration_*.json", f"{label}_test"),
        "--compression-summary", latest("results/summaries/structural-low-rank_*.json", label),
    ], check=True)
PY

python scripts/make_phase3c_report.py
echo "### done — see results/reports/qgeocompress_phase3c_comparison.md"
