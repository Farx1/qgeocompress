# Q-GEOCompress

> Reliability-preserving, quantum-inspired compression for deployable geospatial AI — reduce inference cost (latency, VRAM, model size) without breaking detection performance or operational trust.

A post-training optimization stack for **YOLO11n-OBB** on **DOTA** aerial detection: compress checkpoints under **quality gates** (mAP, ECE, CER@0.8), not blind size reduction.

---

## Overview

- Benchmark **GeoAI model compression** on oriented bounding-box detection with deployment-oriented metrics.
- Combine **classical methods** (FP16, INT8, pruning) with **low-rank / structural factorization** inspired by quantum circuit compression ideas.
- Measure **mAP** alongside **calibration (ECE)**, **confident error rate (CER@0.8)**, latency, and robustness.
- Integrate as a **Valohai post-training DAG**: baseline eval → layer probe → selective compression → inference compare → quality gate → export or reject.

**Current status (Phase 3B hold-out):** selective structural compression with **pareto-gain** layer selection preserves mAP and reliability while cutting ~**5–6%** of parameters on a strict 80/24/24 train/select/test split. Naive full-layer replacement remains non-viable.

Reports: [`results/reports/qgeocompress_phase3b_holdout.md`](results/reports/qgeocompress_phase3b_holdout.md) · [`results/reports/qgeocompress_phase3b_summary.md`](results/reports/qgeocompress_phase3b_summary.md)

---

## Tech Stack

| Layer | Tools |
| ----- | ----- |
| **Languages** | Python 3.11+ |
| **Models** | PyTorch, Ultralytics YOLO11n-OBB |
| **Data** | DOTA128 (MVP), DOTA / xView (extensions) |
| **Metrics** | GT-matched calibration, ECE, CER, selective prediction |
| **MLOps** | Valohai (`valohai.yaml`), Docker |
| **Quality** | pytest (~95 tests), ruff |

---

## Features

- **Baseline training** — reproducible YOLO-OBB training with explicit checkpoint paths
- **Compression methods** — FP16, INT8 PTQ, magnitude pruning, in-place low-rank (SVD), structural low-rank
- **Layer sensitivity probe** — per-layer mAP impact before selective replacement
- **Selection strategies** — `sensitivity` (compressibility score) or **`pareto-gain`** (param savings under mAP drop cap)
- **Hold-out protocol** — strict train / select / test splits (no test leakage for probe or BN)
- **Reliability gate** — accept/reject compressed models vs baseline thresholds
- **Valohai DAG** — six-step post-training pipeline with versioned artifacts
- **System benchmarks** — latency, throughput, VRAM, model size

---

## How It Works

```text
trained checkpoint
  → baseline eval (test)          # mAP, ECE, CER reference
  → structural probe (select)     # layer sensitivity JSON
  → selective compression         # top-k layers, BN recalibration (train)
  → inference compare (test)      # baseline vs compressed
  → quality gate                  # accept / research-only / reject
  → export if accepted            # .pt (+ ONNX when supported)
```

**Quality gate rules (defaults):**

| Check | Threshold |
| ----- | --------- |
| mAP50 drop vs baseline | ≤ 0.10 |
| CER@0.8 delta vs baseline | ≤ +0.02 |
| Parameter reduction (deployable) | ≥ 2% |

**Gate statuses:** `deployable_compression_candidate` · `methodologically_valid` · `rejected`  
**Valohai aliases:** `accepted_for_export` · `accepted_for_research` · `rejected`

---

## Project Structure

```text
qgeocompress/
├── configs/                 # dataset, model, experiment YAML
├── src/qgeocompress/
│   ├── compression/         # low-rank, structural, probe, pruning
│   ├── evaluation/          # calibration, GT matching, reliability gate
│   └── valohai/             # pipeline step logic
├── scripts/                 # CLI entry points
│   └── valohai/             # Valohai step wrappers
├── tests/
├── valohai.yaml             # Valohai steps + pipeline definition
├── Dockerfile               # production runtime image
└── results/
    ├── summaries/           # JSON run artifacts
    ├── figures/
    └── reports/
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

### 2. Prepare dataset

DOTA128 downloads automatically via Ultralytics on first use:

```bash
python scripts/prepare_dataset.py --dataset dota128
```

### 3. Train baseline

```bash
python scripts/train_baseline.py \
  --dataset dota128 \
  --model yolo_obb_small \
  --epochs 5 \
  --imgsz 640
```

Always pin the baseline checkpoint explicitly:

```bash
BEST="runs/obb/runs/baseline/train/weights/best.pt"
```

### 4. Compress and benchmark

```bash
python scripts/compress_model.py --method fp16 --weights "$BEST"
python scripts/compress_model.py --method low-rank --rank-ratio 0.84 --weights "$BEST"

python scripts/benchmark_inference.py \
  --weights "$BEST" \
  --dataset dota128 \
  --batch-size 1 4 8
```

Rank sensitivity sweep (Phase 2A):

```bash
for R in 0.88 0.86 0.84 0.82 0.80; do
  python scripts/compress_model.py --method low-rank --rank-ratio "$R" --weights "$BEST"
done
python scripts/make_report.py
```

### 5. Calibration and reliability (Phase 2B)

GT-matched ECE, CER, and selective prediction:

```bash
python scripts/evaluate_calibration.py \
  --weights "$BEST" \
  --dataset dota128 \
  --compression baseline

python scripts/evaluate_calibration.py \
  --weights runs/compressed/low_rank_r0.840/low_rank_r0.840.pt \
  --dataset dota128 \
  --compression low-rank \
  --rank-ratio 0.84

python scripts/make_calibration_report.py
```

Phase 2 reports: `results/reports/qgeocompress_phase2_report.md`

### 6. Structural compression (Phase 3B)

Phase 3A showed **full structural replacement at r=0.84 breaks mAP**. Phase 3B adds **selective**, sensitivity-guided replacement.

```bash
# Step 1 — probe (~1 validation pass per candidate layer)
python scripts/compress_model.py \
  --method structural-probe \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --target-layers backbone neck

# Step 2 — selective compression (recommended: pareto-gain)
python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --selection-strategy pareto-gain \
  --max-replaced-layers 5 \
  --sensitivity-file results/summaries/structural_layer_sensitivity.json \
  --bn-recalibration-batches 20
```

Probe output: `results/summaries/structural_layer_sensitivity.json`

### 7. Hold-out validation (recommended)

Strict split: **80 train / 24 select / 24 test** — never use test for probe or BN recalibration.

```bash
python scripts/create_dota128_holdout.py --seed 42

python scripts/train_baseline.py \
  --data-yaml datasets/dota128_holdout/dota128_holdout_select.yaml \
  --epochs 10 --project runs/obb/runs --name baseline_holdout

BEST="runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt"
HOLDOUT=datasets/dota128_holdout

# Baseline on test
python scripts/evaluate_calibration.py \
  --weights "$BEST" \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --compression baseline \
  --run-label holdout_test_baseline

# Probe on select only
python scripts/compress_model.py \
  --method structural-probe \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --data-yaml "$HOLDOUT/dota128_holdout_select.yaml" \
  --run-label holdout_select

# Compress with pareto-gain
python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --selection-strategy pareto-gain \
  --max-replaced-layers 5 \
  --sensitivity-file results/summaries/structural_layer_sensitivity_holdout_select.json \
  --bn-data-yaml "$HOLDOUT/dota128_holdout_train.yaml" \
  --eval-data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --bn-recalibration-batches 20 \
  --run-label holdout_top5_pareto

# Quality gate
python scripts/check_reliability_gate.py \
  --baseline results/summaries/calibration_baseline_holdout_test_baseline_3cebaa50.json \
  --candidate results/summaries/calibration_structural_r0.840_holdout_top5_pareto_test_f215351d.json \
  --compression-summary results/summaries/structural-low-rank_11845535.json
```

**Hold-out results (test split):**

| Method | mAP50 | CER@0.8 | Params | Gate |
| ------ | ----: | ------: | -----: | ---- |
| Baseline | 0.929 | 0.004 | ref | — |
| Top-5 sensitivity | 0.879 | 0.005 | −0.21% | methodologically_valid |
| Top-5 pareto-gain | **0.933** | 0.005 | **−5.63%** | **deployable** |
| Full structural r=0.84 | 0.266 | — | −7.34% | rejected |

### 8. Valohai post-training DAG

Production target: orchestrated pipeline with versioned artifacts, not a one-off script.

Config: [`valohai.yaml`](valohai.yaml) · Wrappers: [`scripts/valohai/`](scripts/valohai/)

```bash
# Cloud (Valohai CLI + linked project)
vh lint
vh pipeline run qgc-post-training-compression --adhoc

# Local smoke test
export VALOHAI_OUTPUTS_DIR=./valohai_outputs
python scripts/valohai/baseline_eval_step.py \
  --model "$BEST" --test-yaml "$HOLDOUT/dota128_holdout_test.yaml"
# … probe → compress → inference_compare → quality_gate → export_if_accepted
```

**Docker image for production runs:**

```bash
docker build -t qgeocompress:latest .
# Set image: your-registry/qgeocompress:latest in valohai.yaml
```

Pipeline steps: `qgc-baseline-eval` → `qgc-structural-probe` → `qgc-compress-selective` → `qgc-inference-compare` → `qgc-quality-gate` → `qgc-export-if-accepted`

Outputs land in `/valohai/outputs/` (or `VALOHAI_OUTPUTS_DIR`). Each step emits JSON metrics to stdout for Valohai experiment tracking.

> **Note:** Structural checkpoints require **no-fuse** inference. ONNX export is attempted only for standard models; structural export remains experimental.

---

## Key Results Summary

| Phase | Finding |
| ----- | ------- |
| **2A** | Low-rank viable frontier around r ≈ 0.82–0.84 |
| **2B** | Baseline mAP50 ≈ 0.954, CER@0.8 ≈ 0.022 on DOTA128 |
| **3A** | Full structural r=0.84 → mAP collapse (~0.06 in-sample) |
| **3B in-sample** | Selective top-5 sensitivity → mAP 0.888, −4.75% params |
| **3B hold-out** | Pareto-gain top-5 → mAP 0.933, −5.63% params, CER stable |

---

## Possible Improvements

- Scale validation to **full DOTA** or larger hold-out splits
- Fix activation-based `local_output_error` in the structural probe
- Prove **latency / VRAM** gains on GPU for pareto-gain candidates
- ONNX / TensorRT export path for structural layers (or collapsed fallback)
- Parallel Valohai branch: top-3 / top-5 / top-8 → select-best candidate
- CI: pytest + `vh lint` on pull requests

---

## About

**Jules Barth** — M2 Data & AI Engineering, ESILV (Paris). Focus on LLMs, agentic AI, privacy-preserving ML, and quantum computing.

- Portfolio: [julesbarth-myportfolio.fr](https://julesbarth-myportfolio.fr)
- LinkedIn: [linkedin.com/in/jules-barth](https://www.linkedin.com/in/jules-barth)
- GitHub: [github.com/Farx1](https://github.com/Farx1)

---

## License

MIT
