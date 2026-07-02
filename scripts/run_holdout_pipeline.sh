#!/usr/bin/env bash
# Q-GEOCompress hold-out post-training pipeline (local E2E)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SKIP_TRAIN=0
SKIP_PROBE=0
DRY_RUN=0
GPU=0
MAX_LAYERS=5
SELECTION="pareto-gain"
EPOCHS=10
BEST="${BEST:-runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt}"
HOLDOUT="${HOLDOUT:-datasets/dota128_holdout}"
SENS="${SENS:-results/summaries/structural_layer_sensitivity_holdout_select.json}"
DEVICE="cpu"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Hold-out pipeline: split → baseline eval → probe → compress → calibrate → gate → Phase 3C report

Options:
  --skip-train     Use existing baseline checkpoint (default BEST path)
  --skip-probe     Reuse existing sensitivity JSON at \$SENS
  --gpu            Pass --device cuda to Python scripts
  --dry-run        Print commands without executing
  --max-layers N   Max replaced layers (default: 5)
  --epochs N       Training epochs if not skipping train (default: 10)
  -h, --help       Show this help

Environment:
  BEST, HOLDOUT, SENS, PYTHON
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-probe) SKIP_PROBE=1; shift ;;
    --gpu) GPU=1; DEVICE="cuda"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --max-layers) MAX_LAYERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

echo "=== Q-GEOCompress hold-out pipeline ==="
echo "DEVICE=$DEVICE  BEST=$BEST  HOLDOUT=$HOLDOUT"

run "$PYTHON" scripts/create_dota128_holdout.py --seed 42

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  run "$PYTHON" scripts/train_baseline.py \
    --data-yaml "$HOLDOUT/dota128_holdout_select.yaml" \
    --epochs "$EPOCHS" --project runs/obb/runs --name baseline_holdout \
    --device "$DEVICE"
  BEST="runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt"
fi

run "$PYTHON" scripts/evaluate_calibration.py \
  --weights "$BEST" \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --compression baseline --run-label holdout_test_baseline \
  --device "$DEVICE"

if [[ "$SKIP_PROBE" -eq 0 ]]; then
  run "$PYTHON" scripts/compress_model.py \
    --method structural-probe --rank-ratio 0.84 --weights "$BEST" \
    --data-yaml "$HOLDOUT/dota128_holdout_select.yaml" \
    --run-label holdout_select --device "$DEVICE"
  SENS="results/summaries/structural_layer_sensitivity_holdout_select.json"
fi

RUN_LABEL="holdout_top${MAX_LAYERS}_pareto"
if [[ "$SELECTION" == "sensitivity" ]]; then
  RUN_LABEL="holdout_top${MAX_LAYERS}_sens"
fi
run "$PYTHON" scripts/compress_model.py \
  --method structural-low-rank --rank-ratio 0.84 --weights "$BEST" \
  --selection-strategy "$SELECTION" --max-replaced-layers "$MAX_LAYERS" \
  --sensitivity-file "$SENS" \
  --bn-data-yaml "$HOLDOUT/dota128_holdout_train.yaml" \
  --eval-data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --run-label "$RUN_LABEL" --device "$DEVICE"

# Find latest compressed weights (user may run calibration manually if path differs)
COMPRESSED=$(ls -t runs/compressed/structural_low_rank/structural_low_rank_r0.840_${RUN_LABEL}*.pt 2>/dev/null | head -1 || true)
if [[ -z "$COMPRESSED" ]]; then
  COMPRESSED=$(ls -t runs/compressed/structural_low_rank/*.pt 2>/dev/null | head -1 || true)
fi

if [[ -n "$COMPRESSED" ]]; then
  run "$PYTHON" scripts/evaluate_calibration.py \
    --weights "$COMPRESSED" \
    --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
    --compression structural-low-rank --rank-ratio 0.84 \
    --run-label "${RUN_LABEL}_test" --device "$DEVICE"
fi

BASELINE_JSON=$(ls -t results/summaries/calibration_baseline_holdout_test_baseline_*.json 2>/dev/null | head -1 || true)
CANDIDATE_JSON=$(ls -t results/summaries/calibration_structural_r0.840_${RUN_LABEL}_test_*.json 2>/dev/null | head -1 || true)
SUMMARY_JSON=$(ls -t results/summaries/structural-low-rank_*.json 2>/dev/null | head -1 || true)

if [[ -n "$BASELINE_JSON" && -n "$CANDIDATE_JSON" && -n "$SUMMARY_JSON" ]]; then
  run "$PYTHON" scripts/check_reliability_gate.py \
    --baseline "$BASELINE_JSON" \
    --candidate "$CANDIDATE_JSON" \
    --compression-summary "$SUMMARY_JSON"
fi

run "$PYTHON" scripts/make_phase3c_report.py

echo "=== Pipeline complete ==="
