# Q-GEOCompress

> **Work in progress** — active research & engineering project (M2 internship portfolio).  
> Not production-ready. APIs, results, and the Valohai DAG may change between commits.

Reliability-preserving, quantum-inspired compression for deployable geospatial AI — reduce inference cost (latency, VRAM, model size) without breaking detection performance or operational trust.

A post-training optimization stack for **YOLO11n-OBB** on **DOTA** aerial detection: compress checkpoints under **quality gates** (mAP, ECE, CER@0.8), not blind size reduction.

---

## Project Status

| Area | Status | Notes |
| ---- | ------ | ----- |
| **Core library** | Stable enough to run | ~95 pytest tests; fresh clone verified |
| **Phase 2 (low-rank + calibration)** | Done | Rank frontier + GT-matched ECE/CER |
| **Phase 3B (structural selective)** | Validated on hold-out | Pareto-gain ~5–6% params, mAP preserved |
| **Valohai DAG** | MVP implemented | Not yet battle-tested on Valohai cloud |
| **ONNX / TensorRT export** | Experimental | Structural models need `no-fuse`; export blocked |
| **Full DOTA scale** | Not started | MVP runs on DOTA128 (128 images) |
| **Production deployment** | Out of scope (for now) | Research → MLOps brick, not a shipped product |

**Latest validated result (Phase 3B hold-out, test split):** pareto-gain top-5 → mAP50 **0.933**, CER@0.8 **0.005**, **−5.63%** params vs baseline. Naive full structural replacement still **rejected** (mAP50 ≈ 0.27).

Reports: [`results/reports/qgeocompress_phase3b_holdout.md`](results/reports/qgeocompress_phase3b_holdout.md) · [`results/reports/qgeocompress_phase3b_summary.md`](results/reports/qgeocompress_phase3b_summary.md) · Plan: [`docs/PHASE3_PLAN.md`](docs/PHASE3_PLAN.md)

---

## Overview

**What this project is**

- A **research codebase** to study GeoAI model compression with **deployment-oriented metrics** (not mAP-only).
- A **post-training pipeline** prototype: train elsewhere → evaluate → probe layers → compress selectively → gate → export or reject.
- An exploration of **quantum-inspired low-rank / structural factorization** applied to YOLO-OBB on aerial imagery (DOTA).

**What this project is not (yet)**

- A drop-in replacement for Ultralytics export or TensorRT serving.
- A guaranteed compression recipe for any checkpoint without per-model probing.
- Validated at full DOTA scale or on edge hardware in production.

**Design principles**

1. **Never compress blindly** — layer sensitivity probe before replacement.
2. **Never trust in-sample alone** — strict train / select / test splits for method choice vs final proof.
3. **Reliability matters** — ECE and CER@0.8 gate acceptance alongside mAP50.
4. **Negative results are first-class** — Phase 3A (full structural) failure is documented and informs Phase 3B.

---

## Research Roadmap

```text
Phase 2A  Low-rank rank sensitivity (SVD in-place)     [done]
Phase 2B  GT-matched calibration (ECE, CER, selective)  [done]
Phase 3A  Full structural low-rank replace            [failed — mAP collapse]
Phase 3B  Selective structural + sensitivity probe    [done — in-sample]
Phase 3B' Hold-out validation (80/24/24)               [done]
Phase 3B'' Pareto-gain layer selection                 [done — deployable gate on hold-out]
Phase 4   Valohai DAG + export path                     [in progress — MVP coded]
Phase 5   Scale (full DOTA), latency proof, ONNX/TRT    [planned]
```

---

## Tech Stack

| Layer | Tools |
| ----- | ----- |
| **Languages** | Python 3.11+ |
| **Models** | PyTorch, Ultralytics YOLO11n-OBB |
| **Data** | DOTA128 (MVP, bundled in repo), DOTA / xView (extensions) |
| **Metrics** | GT-matched calibration, ECE, CER@0.8, selective prediction |
| **MLOps** | Valohai (`valohai.yaml`), Docker |
| **Quality** | pytest (95 tests), ruff |

---

## Features

### Implemented

- **Baseline training** — reproducible YOLO-OBB with explicit checkpoint paths
- **Compression methods** — FP16, INT8 PTQ, magnitude pruning, in-place low-rank (SVD), structural low-rank
- **Layer sensitivity probe** — per-layer mAP impact on a **select** split (never test)
- **Selection strategies** — `sensitivity` (compressibility score) or **`pareto-gain`** (absolute param savings under mAP drop cap)
- **Hold-out protocol** — `scripts/create_dota128_holdout.py` → 80 train / 24 select / 24 test
- **Reliability gate** — `scripts/check_reliability_gate.py` accept / research-only / reject
- **Valohai DAG** — six steps in `valohai.yaml` + wrappers in `scripts/valohai/`
- **System benchmarks** — latency, throughput, VRAM, model size (standard checkpoints only)

### Experimental / incomplete

- **Structural inference** — requires `no-fuse` validation path (`obb_validate.py`)
- **ONNX export** — skipped for `StructuralLowRankConv2d` checkpoints
- **Activation-aware probe** — `local_output_error` often `null` (hooks not wired)
- **Valohai cloud runs** — config present; end-to-end cloud execution not documented here
- **GPU latency gains** — param reduction proven; significant speedup not yet demonstrated

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

**Split discipline (critical for valid results)**

| Split | Size (hold-out) | Used for |
| ----- | --------------- | -------- |
| **train** | 80 | Baseline training, BN recalibration |
| **select** | 24 | Structural probe, layer ranking only |
| **test** | 24 | Final mAP, ECE, CER — never for probe or BN |

**Quality gate rules (defaults)**

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
├── datasets/dota128/        # MVP dataset (128 images, tracked in git)
├── docs/                    # phase plans and internal notes
├── src/qgeocompress/
│   ├── compression/         # low-rank, structural, probe, pruning
│   ├── data/                # prepare_dota, corruptions (source module)
│   ├── evaluation/          # calibration, GT matching, reliability gate
│   └── valohai/             # pipeline step logic
├── scripts/                 # CLI entry points
│   └── valohai/             # Valohai step wrappers
├── tests/                   # unit + integration tests (95)
├── valohai.yaml             # Valohai steps + pipeline definition
├── Dockerfile               # production runtime image (WIP)
└── results/
    ├── summaries/           # JSON run artifacts (selected commits)
    ├── figures/
    └── reports/             # phase reports (markdown)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- ~2 GB disk for DOTA128 + PyTorch/Ultralytics
- GPU optional (CPU works for tests and small runs)

### 1. Clone, install, verify

```bash
git clone https://github.com/Farx1/qgeocompress.git
cd qgeocompress
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Sanity check — should collect and pass all tests
pytest -q
```

DOTA128 images and labels are **included in the repository** under `datasets/dota128/`. No external download is required for the MVP test suite.

### 2. Prepare dataset (optional)

Re-run if you need to refresh Ultralytics caches:

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

Always pin the baseline checkpoint explicitly (do not rely on `find runs ... best.pt`):

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

Phase 3A showed **full structural replacement at r=0.84 breaks mAP**. Phase 3B adds **selective**, probe-guided replacement.

```bash
# Step 1 — probe (~1 validation pass per candidate layer; slow)
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

### 7. Hold-out validation (recommended for credible results)

```bash
python scripts/create_dota128_holdout.py --seed 42

python scripts/train_baseline.py \
  --data-yaml datasets/dota128_holdout/dota128_holdout_select.yaml \
  --epochs 10 --project runs/obb/runs --name baseline_holdout

BEST="runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt"
HOLDOUT=datasets/dota128_holdout

python scripts/evaluate_calibration.py \
  --weights "$BEST" \
  --data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --compression baseline \
  --run-label holdout_test_baseline

python scripts/compress_model.py \
  --method structural-probe --rank-ratio 0.84 --weights "$BEST" \
  --data-yaml "$HOLDOUT/dota128_holdout_select.yaml" \
  --run-label holdout_select

python scripts/compress_model.py \
  --method structural-low-rank --rank-ratio 0.84 --weights "$BEST" \
  --selection-strategy pareto-gain --max-replaced-layers 5 \
  --sensitivity-file results/summaries/structural_layer_sensitivity_holdout_select.json \
  --bn-data-yaml "$HOLDOUT/dota128_holdout_train.yaml" \
  --eval-data-yaml "$HOLDOUT/dota128_holdout_test.yaml" \
  --bn-recalibration-batches 20 \
  --run-label holdout_top5_pareto

python scripts/check_reliability_gate.py \
  --baseline results/summaries/calibration_baseline_holdout_test_baseline_3cebaa50.json \
  --candidate results/summaries/calibration_structural_r0.840_holdout_top5_pareto_test_f215351d.json \
  --compression-summary results/summaries/structural-low-rank_11845535.json
```

**Hold-out results (test split, WIP — small n=24):**

| Method | mAP50 | CER@0.8 | Params | Gate |
| ------ | ----: | ------: | -----: | ---- |
| Baseline | 0.929 | 0.004 | ref | — |
| Top-5 sensitivity | 0.879 | 0.005 | −0.21% | methodologically_valid |
| Top-5 pareto-gain | **0.933** | 0.005 | **−5.63%** | deployable |
| Full structural r=0.84 | 0.266 | — | −7.34% | rejected |

> Results on 24 test images are **directional**, not publication-grade. Treat as proof-of-method, not final performance claims.

### 8. Valohai post-training DAG (WIP)

Target: orchestrated MLOps pipeline with versioned artifacts.

Config: [`valohai.yaml`](valohai.yaml) · Wrappers: [`scripts/valohai/`](scripts/valohai/)

```bash
vh lint
vh pipeline run qgc-post-training-compression --adhoc

# Local smoke test
export VALOHAI_OUTPUTS_DIR=./valohai_outputs
python scripts/valohai/baseline_eval_step.py \
  --model "$BEST" --test-yaml "$HOLDOUT/dota128_holdout_test.yaml"
# … probe → compress → inference_compare → quality_gate → export_if_accepted
```

```bash
docker build -t qgeocompress:latest .
```

Pipeline: `qgc-baseline-eval` → `qgc-structural-probe` → `qgc-compress-selective` → `qgc-inference-compare` → `qgc-quality-gate` → `qgc-export-if-accepted`

> Structural checkpoints require **no-fuse** inference. ONNX export is attempted only for standard models.

---

## Key Results Summary

| Phase | Finding | Confidence |
| ----- | ------- | ---------- |
| **2A** | Low-rank viable frontier around r ≈ 0.82–0.84 | In-sample DOTA128 |
| **2B** | Baseline mAP50 ≈ 0.954, CER@0.8 ≈ 0.022 | In-sample (train=val caveat) |
| **3A** | Full structural r=0.84 → mAP collapse | Reproduced |
| **3B in-sample** | Selective top-5 → mAP 0.888, −4.75% params | In-sample only |
| **3B hold-out** | Pareto top-5 → mAP 0.933, −5.63% params | Strict split (n=24 test) |

---

## Known Limitations

- **Small dataset** — DOTA128 (128 images); hold-out test = 24 images only.
- **In-sample Phase 2** — early results used `train == val`; hold-out protocol fixes this for Phase 3B+.
- **Structural serving** — no production ONNX/TensorRT path yet.
- **Probe signal** — `local_output_error` not fully implemented; selection relies mainly on single-layer mAP drop.
- **Ultralytics paths** — nested `runs/obb/runs/obb/runs/...` from default project settings.
- **No CI/CD** — tests run locally; GitHub Actions not configured.

---

## Roadmap (what comes next)

1. **Push + CI** — GitHub Actions: `pytest` on every PR
2. **Valohai cloud** — first end-to-end pipeline run with custom Docker image
3. **Latency proof** — GPU benchmark on pareto-gain candidates vs baseline
4. **Export path** — collapsed structural fallback or documented no-fuse TorchScript
5. **Scale** — full DOTA or larger hold-out for publication-grade numbers
6. **Parallel candidates** — Valohai branch top-3 / top-5 / top-8 → select-best

Contributions and feedback welcome while the project is active.

---

## About

**Jules Barth** — M2 Data & AI Engineering, ESILV (Paris). Focus on LLMs, agentic AI, privacy-preserving ML, and quantum computing.

- Portfolio: [julesbarth-myportfolio.fr](https://julesbarth-myportfolio.fr)
- LinkedIn: [linkedin.com/in/jules-barth](https://www.linkedin.com/in/jules-barth)
- GitHub: [github.com/Farx1](https://github.com/Farx1)

---

## License

MIT
