# Q-GEOCompress

[![CI](https://github.com/Farx1/qgeocompress/actions/workflows/ci.yml/badge.svg)](https://github.com/Farx1/qgeocompress/actions/workflows/ci.yml)

> **Work in progress** an active research & engineering project.  
> Not production-ready. APIs, results, and the Valohai DAG may change between commits.

Reliability-preserving compression for deployable geospatial AI — reduce inference cost (latency, VRAM, model size) without breaking detection performance or operational trust.

A post-training optimization stack for **YOLO11n-OBB** on **DOTA** aerial detection: compress checkpoints under **quality gates** (mAP, ECE, CER@0.8), not blind size reduction.

**Two distinct techniques, named honestly:**

- **Quantum** — the *which-layers-to-compress* decision is encoded as a **QUBO** and solved with **QAOA** (a real variational quantum algorithm) on a CPU state-vector simulator. See [`docs/QUANTUM.md`](docs/QUANTUM.md).
- **Quantum-inspired** — the *how-to-compress* step uses **SVD / structural low-rank factorization**, classical linear algebra rooted in quantum many-body methods (no qubits).

---

## Project Status

| Area | Status | Notes |
| ---- | ------ | ----- |
| **Core library** | Stable enough to run | 119 unit tests + 5 real-model tests; lint + both suites in CI |
| **Phase 2 (low-rank + calibration)** | Done | Rank frontier + GT-matched ECE/CER |
| **Phase 3B (structural selective)** | Validated on hold-out | Pareto-gain ~5–6% params, mAP preserved |
| **Quantum selection (QAOA)** | Implemented | QUBO + QAOA on simulator; classical exact reference |
| **Head-to-head comparison** | Done on DOTA128 | 7 arms, one command; latency measured and found flat |
| **Valohai DAG** | MVP implemented | Not yet battle-tested on Valohai cloud |
| **ONNX / TensorRT export** | Multi-format CLI | Baseline ONNX/TS; structural via collapsed fallback or `.pt` no-fuse |
| **Full DOTA scale** | Not started | MVP runs on DOTA128 (128 images) |
| **Production deployment** | Out of scope (for now) | Research → MLOps brick, not a shipped product |

**Latest validated result — compression frontier (80-image test split, paired bootstrap).**
The project's own quality gate accepts a candidate when mAP50 drops ≤ 0.10, CER@0.8 rises ≤ 0.02
and parameters fall ≥ 2%. Largest gate-passing compression, per pipeline:

| Pipeline | Params Δ | mAP50 | Δ vs baseline (95% CI) | CER@0.8 | Gate |
| -------- | -------: | ----: | ---------------------- | ------: | ---- |
| Baseline `yolo11n-obb.pt` | ref | 0.8943 | reference | 0.0182 | — |
| Shipped (plain SVD, r=0.9) | −0.34% | 0.8091 | −0.085 [−0.172, −0.028] | 0.0191 | pass |
| Shipped (plain SVD, r=0.84) | −2.29% | 0.7848 | −0.110 [−0.195, −0.053] | 0.0196 | **fail** |
| **Data-aware + transfer-weighted budget** | **−20.67%** | **0.8007** | **−0.094 [−0.170, −0.032]** | 0.0180 | **pass** |

**61× more compression at the same gate.** Two other ways to read the same frontier:

- At −16.2% parameters the new pipeline scores **0.8143**, higher than *every* old-method arm — including
  the one that compresses essentially nothing (−0.02% → 0.8014).
- At the aggressive end, −32.5% parameters scores 0.761 against 0.715 for the old method at −8.5%:
  3.8× the compression **and** +0.045 mAP50.

Full frontier: [`results/reports/qgeocompress_frontier_frontier.md`](results/reports/qgeocompress_frontier_frontier.md).

**No latency gain is established**: a back-to-back rerun gives 98.8 ± 4.3 ms baseline vs 96.0 ± 1.7 ms compressed.

Reproduce on CPU:

```bash
python scripts/frontier_study.py --preset frontier   # ~8 min — the table above
python scripts/frontier_study.py --preset hypotheses  # ~15 min — which idea earns its place
./scripts/run_headtohead.sh                          # ~25 min — layer-selection strategies
```

Reports: [`results/reports/qgeocompress_phase3c_comparison.md`](results/reports/qgeocompress_phase3c_comparison.md) · [`results/reports/qgeocompress_phase3b_holdout.md`](results/reports/qgeocompress_phase3b_holdout.md) · [`results/reports/qgeocompress_phase3b_summary.md`](results/reports/qgeocompress_phase3b_summary.md) · Plan: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) · Phase 3: [`docs/PHASE3_PLAN.md`](docs/PHASE3_PLAN.md) · GPU: [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md)

---

## Overview

**What this project is**

- A **research codebase** to study GeoAI model compression with **deployment-oriented metrics** (not mAP-only).
- A **post-training pipeline** prototype: train elsewhere → evaluate → probe layers → compress selectively → gate → export or reject.
- An exploration of **quantum-inspired low-rank / structural factorization** (SVD) plus **quantum QAOA** layer selection applied to YOLO-OBB on aerial imagery (DOTA).

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
Phase Q   Quantum QAOA layer selection (QUBO)          [done — simulator + classical ref]
Phase 4   Valohai DAG + export path                     [done — MVP + multi-format]
Phase 3C  Head-to-head, all strategies + CPU latency   [done — no speedup found]
Phase 5   Scale (full DOTA), GPU latency, TensorRT      [planned — user GPU]
```

---

## Tech Stack

| Layer | Tools |
| ----- | ----- |
| **Languages** | Python 3.11+ |
| **Models** | PyTorch, Ultralytics YOLO11n-OBB |
| **Data** | DOTA128 (MVP, bundled in repo), DOTA / xView (extensions) |
| **Metrics** | GT-matched calibration, ECE, CER@0.8, selective prediction |
| **Quantum** | PennyLane (QAOA on `default.qubit` simulator) |
| **MLOps** | Valohai (`valohai.yaml`), Docker |
| **Quality** | pytest (119 unit + 5 real-model), ruff (enforced in CI), GitHub Actions CI |

---

## Features

### Implemented

- **Baseline training** — reproducible YOLO-OBB with explicit checkpoint paths
- **Compression methods** — FP16, INT8 PTQ, magnitude pruning, in-place low-rank (SVD), structural low-rank
- **Layer sensitivity probe** — per-layer mAP impact on a **select** split (never test)
- **Selection strategies** — `sensitivity` (compressibility score), **`pareto-gain`** (absolute param savings under mAP drop cap), or **`quantum-qaoa`** (QUBO solved by QAOA on a simulator)
- **Hold-out protocol** — `scripts/create_dota128_holdout.py` → 80 train / 24 select / 24 test
- **Reliability gate** — `scripts/check_reliability_gate.py` accept / research-only / reject
- **Valohai DAG** — six steps in `valohai.yaml` + wrappers in `scripts/valohai/`
- **System benchmarks** — latency, throughput, VRAM, model size (structural via no-fuse `predict`)
- **Phase 3C report** — `scripts/make_phase3c_report.py` joins calibration + compression + gate JSONs
- **Export tooling** — `scripts/export_model.py` (`--format onnx|torchscript|pt|all`, collapsed structural fallback, JSON manifest)
- **Local E2E pipeline** — `scripts/run_holdout_pipeline.sh` (`--skip-train`, `--skip-probe`, `--gpu`, `--dry-run`)
- **CI** — GitHub Actions runs `ruff check`, the mocked suite and the real-model suite on Python 3.11
- **Activation-aware probe** — `local_output_error` measured per layer via forward pre-hooks (28/28 candidates on the current probe)
- **Data-aware factorization** — activation-weighted SVD, Lagrangian rank allocation under a parameter budget, sensitivity weighting (`compression/data_aware.py`)
- **Bootstrapped AP** — `evaluation/ap.py` gives mAP50 with paired confidence intervals, the only way to compare arms on 80 images
- **Head-to-head runner** — `scripts/run_headtohead.sh` regenerates every arm and the Phase 3C table from scratch

### What made the difference (and what did not)

Four hypotheses, each isolated on the same split
([`results/reports/qgeocompress_frontier_hypotheses.md`](results/reports/qgeocompress_frontier_hypotheses.md)):

| Hypothesis | Verdict | Evidence |
| ---------- | ------- | -------- |
| **Rank ratio was the binding constraint** | Confirmed | At r=0.84 the factorization saves `1 − r/c_out − r/(c_in·k²)` per layer, so compressing *all 28* candidates caps at **7.34%** of the model. The shipped −5.6% was near that ceiling, not near a selection optimum. |
| **1×1 convs must be included** | Confirmed | They hold **39.4%** of the weights and were excluded by `kernel_size != (3,3)`. Candidate coverage went 41.1% → 80.5%. |
| **Data-aware SVD beats plain SVD** | Confirmed, large | Minimizing `‖(W−Ŵ)X‖` instead of `‖W−Ŵ‖_F`: **33–48% lower output error at identical rank**; at −17.6% params, mAP50 **0.554 vs 0.320**. |
| **Budget allocation beats a global ratio** | Confirmed, large | Lagrangian allocation over per-layer spectra: at −16.9% params it scores 0.783 where a uniform ratio at −17.6% scores 0.320. |
| **Weighting the budget by layer sensitivity** | Confirmed — but only in the right units | Raw sensitivity is worth ~+0.01 mAP50, near the noise floor. The allocation prices layers in relative tail energy, so the weight must be a **transfer coefficient** `w_l / ε_l(r₀)` — network error per unit of that layer's own error. At a −31% budget: **0.767 vs 0.719 (raw) vs 0.700 (unweighted)**. |
| **Drift-corrected sequential refit** | Partial | At a fixed ratio r=0.5 (−17.6% params): plain 0.320 → data-aware 0.554 → sequential **0.755**. But combined with budget allocation it *loses* (0.675 vs 0.717 at −32.6%), so it is not in the recommended path. Needs ≥16 probe images: deep layers see ~400 patch positions each against `d` up to 2304, and at 3 images the refit is underdetermined. |
| **Energy-threshold ranks (τ)** | Rejected | τ=0.95 gives −44.7% params at mAP50 **0.057**. A per-layer energy threshold ignores both the parameter cost of a rank and how much the layer matters downstream. |
| **Per-layer factorization scheme** | Real per layer, no end-to-end gain | Three matricizations of the same kernel cost `r(c_in k²+c_out)`, `r(c_in+c_out k²)` and `r·k(c_in+c_out)`. At a fixed 0.20 output error per layer, one scheme everywhere reaches 19.9 / 24.3 / 26.8% reduction; **best-per-layer reaches 34.5%**. End to end it only ties the single-scheme pipeline (−20.8% → 0.800 vs −20.7% → 0.801) and loses past that, so per-layer efficiency is not the binding constraint — error composition across 61 layers is. |
| **Compressing the detection head** | Rejected | The head holds 18.6% of the weights but only saves parameters at ranks that destroy it: mAP50 0.15–0.18 at every budget from −20% to −46%, even with a no-op option available to the allocator. |
| **Feature distillation recovery** | Rejected here | Label-free distillation from the uncompressed teacher moved mAP50 by +0.019 / −0.066 / +0.044 across three budgets — inside the noise. Implemented and documented in `compression/distillation.py`, not part of the recommended path. |

### Experimental / incomplete

- **Structural inference** — requires `no-fuse` path (`obb_validate.py`, `system_metrics.py`); use collapsed export for ONNX/TorchScript
- **TensorRT** — `export_tensorrt.py` stub; requires NVIDIA GPU + `tensorrt` extra (not run in CI)
- **Valohai cloud runs** — config present; end-to-end cloud execution not documented here
- **Data-aware path is study-only** — `scripts/frontier_study.py` produces and scores the compressed model, but `scripts/compress_model.py` and the Valohai DAG still run the old uniform-ratio path. Wiring the winning configuration into the gated CLI is the next task.
- **Latency gains** — parameter reduction proven; no CPU speedup measurable at 2.7M params, CUDA still to run

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
├── tests/                   # 119 mocked unit tests + test_real_pipeline.py (real checkpoint, `-m slow`)
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

#### Quantum layer selection (QAOA)

Instead of the greedy `pareto-gain` heuristic, select layers by solving the
selection **QUBO** with **QAOA** on a simulator (`pip install -e ".[quantum]"`):

```bash
python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --selection-strategy quantum-qaoa \
  --quantum-backend qaoa \
  --max-replaced-layers 5 \
  --sensitivity-file results/summaries/structural_layer_sensitivity.json
```

`--quantum-backend`: `auto` (QAOA if PennyLane installed, else classical) · `qaoa` (require PennyLane) · `classical` (exact brute-force / greedy reference). Full formulation and honest scope in [`docs/QUANTUM.md`](docs/QUANTUM.md).

### 7. Hold-out validation (recommended for credible results)

One command runs the split, the baseline, the probe, every selection arm, the gates
and the Phase 3C report — and picks the gate JSONs up by run label instead of by
hard-coded hash:

```bash
./scripts/run_headtohead.sh                          # ~25 min on 4 CPU cores
./scripts/run_headtohead.sh --device cuda            # same run on GPU
./scripts/run_headtohead.sh --weights path/to/my.pt  # your own checkpoint
```

> **Do not fine-tune the baseline on the 80-image hold-out train split.** The
> previously documented `train_baseline.py --epochs 10` step produces a **0.469**
> mAP50 model (vs **0.927** for the pretrained checkpoint) — 80 images is too few.
> The runner compresses the pretrained checkpoint instead; see Known Limitations
> for what that costs in interpretation.

**Hold-out head-to-head (test split, n=24, all regenerated by `./scripts/run_headtohead.sh`):**

| Method | mAP50 | CER@0.8 | ECE | Params Δ | Latency ms | Gate |
| ------ | ----: | ------: | --: | -------: | ---------: | ---- |
| Baseline (`yolo11n-obb.pt`) | **0.927** | 0.0044 | 0.0421 | ref | 101.0 | reference |
| Top-3 sensitivity | 0.888 | 0.0049 | 0.0553 | −4.58% | 90.6 | deployable |
| Top-5 sensitivity | 0.875 | 0.0000 | 0.0637 | −5.04% | 85.0 | deployable |
| Top-3 pareto-gain | 0.910 | 0.0045 | 0.0420 | −5.09% | 86.6 | deployable |
| Top-5 pareto-gain | 0.909 | 0.0050 | 0.0473 | **−5.63%** | 90.1 | deployable |
| Top-8 pareto-gain | 0.894 | 0.0098 | 0.0480 | −5.96% | 90.1 | deployable |
| Top-5 quantum-qaoa (classical QUBO) | 0.875 | 0.0000 | 0.0637 | −5.04% | 99.6 | deployable |
| Top-5 quantum-qaoa (QAOA simulator) | **0.910** | 0.0045 | 0.0440 | −5.19% | 93.1 | deployable |
| SVD in-place r=0.84 | 0.891 | 0.0067 | 0.0369 | — | — | — |

What the run actually shows:

- **Pareto-gain and the QAOA simulator tie at the top** (mAP50 0.909–0.910). QAOA saves slightly fewer parameters than top-5 pareto-gain (−5.19% vs −5.63%) for the same accuracy.
- **The classical exact QUBO picks the same five layers as the sensitivity heuristic** and lands at 0.875 — so on this instance the QAOA *approximation* selected a better subset than the exact minimizer of the same objective. That says the QUBO objective is imperfectly aligned with test mAP, not that QAOA is stronger.
- **More layers is not better**: top-8 pareto-gain buys 0.33 extra points of parameter reduction and costs 0.015 mAP50.
- **No latency gain is established.** The per-arm latencies above come from separate runs at different times and are not comparable — the two arms with an *identical* layer set land 17% apart. Benchmarked back to back (5 repeats each), baseline is 98.8 ± 4.3 ms and top-5 pareto-gain is 96.0 ± 1.7 ms. −5.6% of parameters in 5 of 28 layers does not move wall time on a 2.7M-parameter model.

> 24 test images: **directional, not publication-grade**. Proof-of-method, not final performance claims.

### 9. Phase 3C comparison report

Aggregate hold-out calibration, compression, and gate JSONs:

```bash
python scripts/make_phase3c_report.py
```

Outputs: `results/reports/qgeocompress_phase3c_comparison.md`, `results/summaries/phase3c_comparison_table.csv`

### 10. One-command hold-out pipeline

```bash
chmod +x scripts/run_holdout_pipeline.sh
./scripts/run_holdout_pipeline.sh --skip-train --skip-probe   # reuse existing artifacts
./scripts/run_holdout_pipeline.sh --gpu                       # full GPU run (see GPU_RUNBOOK)
```

### 11. Export accepted model

```bash
python scripts/export_model.py \
  --weights "$COMPRESSED" \
  --format all \
  --device cpu \
  --output-dir runs/export/candidate
```

Produces `export_manifest.json` with per-format status. Structural checkpoints: `.pt` (no-fuse) + optional `export_model_collapsed.pt` for Ultralytics-compatible ONNX/TorchScript attempts. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) export matrix.

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
| **3B hold-out** | Pareto top-5 → mAP 0.909, −5.63% params, gate deployable | Strict split (n=24 test), current code |
| **Phase Q hold-out** | QAOA simulator top-5 → mAP 0.910, −5.19% params, gate deployable | QUBO + PennyLane simulator (16-variable prefilter) |
| **Head-to-head** | 7 arms under identical conditions; no CPU latency gain measurable | One command, regenerable |

> Rows 2A–3B in-sample predate the GT-matching fix and the working activation probe; they are kept as history, not as current measurements. Only the two hold-out rows and the head-to-head row come from the current code.

---

## Known Limitations

- **Small dataset** — DOTA128 (128 images). The frontier study uses a 24 train / 24 select / 80 test split (`datasets/dota128_power`) because a pretrained baseline is never trained on our splits, so a larger test set is legitimate and 24 images could not separate any two arms.
- **Two mAP50 numbers, on purpose.** The frontier tables use this repo's own AP (`evaluation/ap.py`, baseline 0.894) rather than Ultralytics' (0.927), because only ours can be bootstrapped. Per-class AP on a small split punishes rare classes, so absolute values run lower. Every arm is scored identically, and all comparisons are paired — but do not mix the two scales.
- **The accuracy deltas are not tight.** Even a 0.02%-parameter arm shows Δ ≈ −0.09 with a CI of roughly ±0.07. The *ordering* of methods is robust (the data-aware pipeline dominates at every parameter level by far more than the CI); individual Δ values are not.
- **The baseline is the pretrained checkpoint, not a fine-tune.** Fine-tuning `yolo11n-obb.pt` for 10 epochs on the 80-image hold-out train split (the command in step 7) yields **mAP50 0.469** on the test split, far below the pretrained model's 0.927 — 80 images is not enough to fine-tune on without wrecking it. The head-to-head therefore compresses the pretrained checkpoint. DOTA128 is a subset of the DOTA data that checkpoint was trained on, so **baseline mAP50 is optimistic**; the compression deltas are still measured under identical conditions and remain valid as relative results.
- **Calibration numbers depend on the Ultralytics version.** GT matching keyed predictions on `result.path`, which Ultralytics 8.4 renames to `image0`, `image1`, … for a list source. Every prediction missed its ground truth (TP 0, CER@0.8 1.0) until this was fixed. Summaries produced before the fix are in `results/archive/pre_ultralytics_8.4/` and are not reproducible by the current code.
- **In-sample Phase 2** — early results used `train == val`; hold-out protocol fixes this for Phase 3B+.
- **Structural serving** — direct ONNX/TorchScript blocked; use collapsed fallback or no-fuse `.pt` (see export matrix in `docs/PROJECT_PLAN.md`)
- **Ultralytics paths** — nested `runs/obb/runs/obb/runs/...` from default project settings.
- **Quantum scope** — QAOA runs on a **simulator**, capped at 16 variables (20 qubits was OOM-killed at ~11 GB; measured cost table in [`docs/QUANTUM.md`](docs/QUANTUM.md)). Beyond 16 candidates the solver keeps the highest parameter-gain layers and records `qaoa_prefiltered_from`. No real QPU, no quantum speedup claimed.
- **CI** — GitHub Actions runs `ruff check`, the mocked suite, and the real-model suite on push/PR; **no GPU** jobs in CI.

---

## Roadmap (what comes next)

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full end-to-end plan.

### Done — head-to-head at DOTA128 scale

Every selection strategy now runs under identical conditions from one command
(`./scripts/run_headtohead.sh`, ~25 min CPU), with latency columns filled. See the
comparison table above and [`results/reports/qgeocompress_phase3c_comparison.md`](results/reports/qgeocompress_phase3c_comparison.md).

Answers so far, at this scale:

- *Does quantum layer selection beat classical heuristics?* It ties the best classical heuristic (0.910 vs 0.910) and beats the **exact** minimizer of the same QUBO (0.875) — which points at the objective, not the solver.
- *Does compression speed up inference?* Not measurably here. Back-to-back, 98.8 ± 4.3 ms vs 96.0 ± 1.7 ms. Parameter reduction is real; wall-time reduction is not, at this model size on CPU.

### Next step — production scale

DOTA128 results are **directional, not publication-grade** (24 test images, and a baseline whose training data included them). The next step is the same pipeline on full DOTA with a properly trained baseline, plus CUDA latency:

| What to compare | Methods |
| --------------- | ------- |
| Layer selection | pareto-gain · sensitivity · **quantum-qaoa (simulator)** · classical QUBO |
| Compression | structural selective · SVD in-place · baseline |
| Metrics | mAP50 · CER@0.8 · ECE · **real latency (GPU)** · params · model size · export path |

Until that run, treat DOTA128 numbers as **portfolio evidence**, not final performance claims.

**Non-GPU (next):**

1. **Quantum annealing** — submit the same selection QUBO to D-Wave (Ocean SDK)
2. **Warm-start QAOA** — seed angles from greedy/pareto; hard-constraint (slack) QUBO
3. **Tensor-Train / MPS** — quantum-inspired weight factorization (method #2)
4. **Valohai cloud** — first end-to-end pipeline run with custom Docker image
5. **Parallel candidates** — Valohai branch top-3 / top-5 / top-8 → select-best

**GPU (user-local):**

6. **BN=20 + latency proof** — pareto CUDA benchmark vs baseline (`docs/GPU_RUNBOOK.md`)
7. **Scale** — full DOTA or larger hold-out for publication-grade numbers
8. **TensorRT** — engine build on GPU host with `tensorrt` extra

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
