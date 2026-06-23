# Q-GEOCompress

> Reliability-preserving quantum-inspired compression for deployable geospatial AI — reduce inference cost (latency, VRAM, model size) without breaking detection performance or operational trust.

---

## Overview

- Built to benchmark **GeoAI model compression** on aerial/satellite detection (DOTA OBB) with deployment-oriented metrics.
- Combines **classical compression** (FP16, INT8, pruning) with **quantum-inspired low-rank factorization**.
- Measures not only **mAP** but also **calibration (ECE)**, **confident errors**, and **robustness under corruptions**.

---

## Tech Stack

- **Languages**: Python 3.11+
- **Frameworks**: PyTorch, Ultralytics YOLO OBB
- **Data**: DOTA128 (MVP), DOTA, xView (extensions)
- **Tools**: ONNX, pytest, ruff

---

## Features

- **Baseline training** — YOLO-OBB on DOTA128 / DOTA with reproducible configs
- **Compression pipeline** — FP16, INT8 PTQ, magnitude pruning, low-rank factorization
- **System benchmarks** — latency (p50/p95), throughput, VRAM, model size
- **Trust metrics** — ECE, temperature scaling, confident-error rate, selective prediction
- **Robustness** — synthetic corruptions (cloud, blur, noise, JPEG, brightness)

---

## Project Structure

```text
qgeocompress/
├── configs/          # dataset, model, experiment YAML configs
├── src/qgeocompress/ # core library
├── scripts/          # thin CLI wrappers
├── tests/            # unit tests
├── notebooks/        # analysis notebooks (optional)
└── results/          # JSON summaries, figures, reports
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Farx1/qgeocompress.git
cd qgeocompress
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Prepare dataset (DOTA128 auto-downloads via Ultralytics)

```bash
python scripts/prepare_dataset.py --dataset dota128
```

### 3. Train baseline (MVP: 5 epochs on DOTA128)

```bash
python scripts/train_baseline.py \
  --dataset dota128 \
  --model yolo_obb_small \
  --epochs 5 \
  --imgsz 640
```

### 4. Benchmark inference

```bash
python scripts/benchmark_inference.py \
  --weights runs/obb/runs/baseline/train/weights/best.pt \
  --dataset dota128 \
  --batch-size 1 4 8
```

### 5. Compress and compare

Always use the **baseline** checkpoint explicitly (never `find runs ... best.pt`):

```bash
BEST="runs/obb/runs/baseline/train/weights/best.pt"

python scripts/compress_model.py --method fp16 --weights "$BEST"
python scripts/compress_model.py --method low-rank --rank-ratio 0.85 --weights "$BEST"
```

Rank sensitivity sweep (no fine-tune) around the stability boundary:

```bash
BEST="runs/obb/runs/baseline/train/weights/best.pt"
for R in 0.88 0.86 0.84 0.82 0.80; do
  python scripts/compress_model.py --method low-rank --rank-ratio "$R" --weights "$BEST"
done
python scripts/make_report.py
```

Phase 2A report outputs:
- `results/summaries/comparison_table.csv` — baseline + low-rank sweep only (fine-tune excluded by default)
- `results/figures/rank_ratio_vs_map50.png` — rank sensitivity curve
- `results/summaries/phase2a_summary.md` — auto-generated findings

To include archived fine-tune runs: `python scripts/make_report.py --include-unsupported-finetune`

### 7. Phase 2B — GT-matched calibration

Evaluate reliability (ECE, CER, selective prediction) with real GT matching:

```bash
BEST="runs/obb/runs/baseline/train/weights/best.pt"

python scripts/evaluate_calibration.py \
  --weights "$BEST" --dataset dota128 --compression baseline --iou-threshold 0.5

python scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.880/low_rank_r0.880.pt \
  --dataset dota128 --compression low-rank --rank-ratio 0.88 --iou-threshold 0.5

python scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.840/low_rank_r0.840.pt \
  --dataset dota128 --compression low-rank --rank-ratio 0.84 --iou-threshold 0.5

python scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.820/low_rank_r0.820.pt \
  --dataset dota128 --compression low-rank --rank-ratio 0.82 --iou-threshold 0.5

python scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.800/low_rank_r0.800.pt \
  --dataset dota128 --compression low-rank --rank-ratio 0.80 --iou-threshold 0.5

python scripts/make_calibration_report.py
```

Outputs: `results/summaries/calibration_*.json`, `calibration_comparison_table.csv`, `calibration_summary.md`, reliability and coverage-risk figures.

### 8. Phase 2 scientific report

After completing Phase 2A and 2B:

```bash
python scripts/make_report.py
python scripts/make_calibration_report.py
```

Read the consolidated report:

```text
results/reports/qgeocompress_phase2_report.md
results/reports/qgeocompress_phase2_internal_note.md
```

### 9. Phase 3 — Structural low-rank (deployable compression)

Plan: [`docs/PHASE3_PLAN.md`](docs/PHASE3_PLAN.md)

Phase 3A (full replace) confirmed naive structural compression is too aggressive at r=0.84. Phase 3B adds **sensitivity-guided selective** replacement (QCompress-inspired).

```bash
source .venv/bin/activate
BEST="runs/obb/runs/baseline/train/weights/best.pt"

# Step 1 — probe each candidate layer (slow: ~1 val per layer)
python scripts/compress_model.py \
  --method structural-probe \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --target-layers backbone neck

# Step 2 — selective structural compression (requires probe JSON)
python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --target-layers backbone neck \
  --selection-strategy sensitivity \
  --max-replaced-layers 3 \
  --bn-recalibration-batches 20

python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --target-layers backbone neck \
  --selection-strategy sensitivity \
  --max-replaced-layers 5 \
  --bn-recalibration-batches 20
```

Probe output: `results/summaries/structural_layer_sensitivity.json` (per-layer `compressibility_score`, `map50_drop`, `local_output_error`).

Quick debug probe (first N layers only): add `--probe-max-layers 3`.

### 6. Calibrate and report

```bash
python scripts/calibrate_model.py --weights runs/compressed/low_rank.pt
python scripts/make_report.py --results-dir results/summaries
```

---

## MVP Experiments

| ID | Method              | Goal                          |
|----|---------------------|-------------------------------|
| E0 | Baseline FP32       | Reference mAP / latency / ECE |
| E1 | FP16                | Fast baseline compression     |
| E2 | INT8 PTQ            | Aggressive size reduction     |
| E3 | Low-rank (r=0.5)    | Quantum-inspired trade-off    |
| E4 | Low-rank (r=0.25)   | Stronger compression          |

Target output: a **latency vs mAP** scatter plot with point size = VRAM and color = ECE.

---

## License

MIT
