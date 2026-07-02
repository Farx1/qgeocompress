# GPU Runbook — Phase 3B/3C hold-out experiments

Copy-paste commands for a CUDA machine. Assumes repo root and an activated venv with `pip install -e ".[dev]"`.

**Baseline checkpoint (hold-out):**

```bash
BEST="runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt"
HOLDOUT="datasets/dota128_holdout"
```

If the hold-out split or baseline is missing, run the full local pipeline first (CPU-friendly with `--skip-train` if weights exist):

```bash
./scripts/run_holdout_pipeline.sh --skip-train --skip-probe --dry-run   # preview
./scripts/run_holdout_pipeline.sh --skip-train --skip-probe             # CPU smoke
./scripts/run_holdout_pipeline.sh --gpu                                 # full GPU E2E
```

---

## Tonight's GPU checklist

| Step | Command | Expected artifact |
| ---- | ------- | ----------------- |
| Pareto top-5 + BN=20 | See §1 | `results/summaries/structural-low-rank_*.json` |
| Calibration (test) | See §2 | `results/summaries/calibration_structural_*_bn20_*.json` |
| Quality gate | See §3 | `results/summaries/reliability_gate_*.json` |
| Latency compare (CUDA) | See §4 | `results/summaries/benchmark_*.json` |
| Full E2E | `./scripts/run_holdout_pipeline.sh --gpu` | All of the above + Phase 3C report |
| Regenerate report | `python scripts/make_phase3c_report.py` | `results/reports/qgeocompress_phase3c_comparison.md` |

---

## 1. Structural compression — pareto top-5, BN recalibration = 20

Default CLI now sets `--bn-recalibration-batches 20` for `structural-low-rank` (override with `--bn-recalibration-batches 0` if needed).

```bash
python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --selection-strategy pareto-gain \
  --max-replaced-layers 5 \
  --sensitivity-file results/summaries/structural_layer_sensitivity_holdout_select.json \
  --bn-data-yaml "$HOLDOUT/dota128_holdout_train.yaml" \
  --eval-data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --run-label holdout_top5_pareto_bn20 \
  --device cuda
```

Note the new `structural-low-rank_*.json` path in the log — use its `output_weights` below.

---

## 2. Calibration on hold-out test split

```bash
COMPRESSED="runs/compressed/structural_low_rank/structural_low_rank_r0.840_holdout_top5_pareto_bn20_pareto.pt"

python scripts/evaluate_calibration.py \
  --weights "$COMPRESSED" \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --compression structural-low-rank \
  --rank-ratio 0.84 \
  --run-label holdout_top5_pareto_bn20_test \
  --device cuda
```

---

## 3. Reliability gate

After compression + calibration JSONs exist, pass the matching paths (filenames include content hashes):

```bash
python scripts/check_reliability_gate.py \
  --baseline results/summaries/calibration_baseline_holdout_test_baseline_3cebaa50.json \
  --candidate results/summaries/calibration_structural_r0.840_holdout_top5_pareto_bn20_test_XXXXXXXX.json \
  --compression-summary results/summaries/structural-low-rank_XXXXXXXX.json
```

Replace `XXXXXXXX` with the hash suffix from your new run files.

---

## 4. Latency benchmark — baseline vs pareto (CUDA, no-fuse for structural)

```bash
python scripts/benchmark_inference.py \
  --weights "$BEST" \
  --dataset dota128 \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --batch-size 1 4 8 \
  --device cuda

python scripts/benchmark_inference.py \
  --weights "$COMPRESSED" \
  --dataset dota128 \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --batch-size 1 4 8 \
  --device cuda
```

Structural checkpoints use `predict_no_fuse` internally — no manual flag required.

---

## 5. Export (if gate accepts)

```bash
python scripts/export_model.py \
  --weights "$COMPRESSED" \
  --output-dir runs/export/holdout_top5_pareto_bn20 \
  --quality-gate results/summaries/reliability_gate_calibration_structural_r0.840_holdout_top5_pareto_bn20_test_XXXXXXXX.json
```

Structural models: `.pt` copy only; ONNX skipped with documented reason.

---

## 6. Regenerate Phase 3C comparison report

```bash
python scripts/make_phase3c_report.py
```

Outputs:

- `results/reports/qgeocompress_phase3c_comparison.md`
- `results/summaries/phase3c_comparison_table.csv`

---

## Optional: SVD in-place hold-out baseline (Phase 3C row)

```bash
python scripts/compress_model.py \
  --method low-rank --rank-ratio 0.84 --weights "$BEST" --device cuda

python scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.840/low_rank_r0.840.pt \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --compression low-rank --rank-ratio 0.84 \
  --run-label holdout_r0.84_test --device cuda

python scripts/make_phase3c_report.py
```
