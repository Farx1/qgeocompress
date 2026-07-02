#!/usr/bin/env bash
# CPU hold-out experiments: SVD + quantum-qaoa + Phase 3C report
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
BEST="runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt"
HOLDOUT="datasets/dota128_holdout"
SENS="results/summaries/structural_layer_sensitivity_holdout_select.json"
BASELINE="results/summaries/calibration_baseline_holdout_test_baseline_3cebaa50.json"
LOG="results/summaries/cpu_holdout_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds) CPU hold-out batch start ==="

echo "=== 1/6 SVD low-rank compress ==="
"$PY" scripts/compress_model.py \
  --method low-rank --rank-ratio 0.84 --weights "$BEST" \
  --eval-data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --run-label holdout_low_rank_r084 --device cpu

echo "=== 2/6 SVD calibration (test) ==="
"$PY" scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.840/low_rank_r0.840.pt \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --compression low-rank --rank-ratio 0.84 \
  --run-label holdout_low_rank_r084_test --device cpu

echo "=== 3/6 quantum-qaoa compress (classical QUBO solver, n=28 layers) ==="
"$PY" scripts/compress_model.py \
  --method structural-low-rank --rank-ratio 0.84 --weights "$BEST" \
  --selection-strategy quantum-qaoa --quantum-backend classical \
  --max-replaced-layers 5 --sensitivity-file "$SENS" \
  --bn-data-yaml "$HOLDOUT/dota128_holdout_train.yaml" \
  --eval-data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --bn-recalibration-batches 20 \
  --run-label holdout_top5_qaoa --device cpu

echo "=== 4/6 quantum-qaoa calibration (test) ==="
COMPRESSED=$(ls -t runs/compressed/structural_low_rank/structural_low_rank_r0.840_holdout_top5_qaoa*.pt 2>/dev/null | head -1)
if [[ -z "$COMPRESSED" ]]; then
  echo "ERROR: compressed weights not found" >&2
  exit 1
fi
"$PY" scripts/evaluate_calibration.py \
  --weights "$COMPRESSED" \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --compression structural-low-rank --rank-ratio 0.84 \
  --run-label holdout_top5_qaoa_test --device cpu

echo "=== 5/6 reliability gate (quantum-qaoa) ==="
CAND=$(ls -t results/summaries/calibration_structural_r0.840_holdout_top5_qaoa_test_*.json 2>/dev/null | head -1)
SUM=$(ls -t results/summaries/structural-low-rank_*.json 2>/dev/null | while read -r f; do
  if grep -q '"run_label": "holdout_top5_qaoa"' "$f" 2>/dev/null; then echo "$f"; break; fi
done)
if [[ -n "$CAND" && -n "$SUM" ]]; then
  "$PY" scripts/check_reliability_gate.py \
    --baseline "$BASELINE" --candidate "$CAND" --compression-summary "$SUM"
else
  echo "WARN: skip gate — candidate=$CAND summary=$SUM"
fi

echo "=== 6/6 Phase 3C report ==="
"$PY" scripts/make_phase3c_report.py

echo "=== $(date -Iseconds) CPU hold-out batch DONE ==="
