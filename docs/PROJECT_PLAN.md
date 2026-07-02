# Q-GEOCompress — Project Plan

**Status:** Project complete (MVP + export stack + documentation).  
**Last updated:** July 2026  
**Owner:** Jules Barth — [portfolio](https://julesbarth-myportfolio.fr) · [GitHub](https://github.com/Farx1/qgeocompress)

---

## Vision & success criteria

### Vision

Build a **reliability-preserving, quantum-inspired compression stack** for geospatial OBB detection (YOLO11n on DOTA) that:

1. Reduces **real** inference cost (parameters, file size, eventually latency) without breaking mAP, calibration (ECE), or selective prediction (CER@0.8).
2. Documents **negative results** as rigorously as wins (full structural failure → selective compression).
3. Ships an **MLOps-ready post-training DAG** (Valohai) and **multi-format export** path for portfolio and research credibility.

### Success criteria (achieved for MVP)

| Criterion | Target | Status |
| --------- | ------ | ------ |
| Hold-out deployable candidate | mAP50 drop ≤ 0.10, CER@0.8 Δ ≤ +0.02, params − ≥ 2% | **Met** (pareto top-5: mAP50 0.933, −5.63% params) |
| Negative result documented | Full structural r=0.84 collapse | **Met** (mAP50 ≈ 0.27) |
| Reliability metrics | GT-matched ECE + CER@0.8 | **Met** (Phase 2B + hold-out) |
| Reproducible pipeline | Scripts + Valohai DAG + E2E shell | **Met** |
| Export stack | Multi-format CLI + manifest | **Met** (this sprint) |
| Test coverage | pytest green | **Met** (~105+ tests) |

### Out of scope (explicit future work)

- Full DOTA scale training/eval
- GPU latency proof at production batch sizes
- TensorRT engine build in CI (requires GPU + `tensorrt` extra)
- Production serving integration (Triton, SageMaker, etc.)

---

## Phase summary

### Phase 1 — Foundation (done)

- Repo scaffold, DOTA128 dataset, baseline YOLO11n-OBB training
- Compression primitives: FP16, INT8 PTQ, magnitude pruning
- Benchmark harness (latency, VRAM, model size)

**Artifacts:** `scripts/train_baseline.py`, `scripts/compress_model.py`, `scripts/benchmark_inference.py`

### Phase 2 — Spectral frontier + calibration (done)

| Sub-phase | Deliverable | Key result |
| --------- | ----------- | ---------- |
| **2A** | In-place SVD low-rank (`low_rank.py`) | Viable rank frontier r ≈ 0.82–0.84 |
| **2B** | GT-matched calibration (`evaluate_calibration.py`) | ECE, CER@0.8, selective prediction |

**Artifacts:** `results/reports/qgeocompress_phase2_report.md`, calibration JSONs in `results/summaries/`

### Phase 3 — Structural compression (done)

| Sub-phase | Result | Gate |
| --------- | ------ | ---- |
| **3A** Full structural replace | mAP collapse | **Rejected** |
| **3B** Selective + probe + BN recal | pareto top-5 hold-out | **Deployable** |
| **3B'** Hold-out protocol (80/24/24) | Strict split discipline | Done |
| **3C** Comparison report | `make_phase3c_report.py` | Done |
| **3D** Export stack + project plan | This document | **Done** |

**Artifacts:**

- `results/reports/qgeocompress_phase3b_holdout.md`
- `results/reports/qgeocompress_phase3c_comparison.md`
- `docs/PHASE3_PLAN.md`
- `docs/GPU_RUNBOOK.md`

### Phase Q — Quantum layer selection (done)

Reframe the "which layers to compress" decision as a **QUBO** and solve it with
**QAOA** on a CPU state-vector simulator (PennyLane). This turns the previously
vague "quantum-inspired" branding into a **concrete quantum algorithm** wired
into the compression pipeline, with a classical exact solver as ground truth.

| Item | Deliverable | Status |
| ---- | ----------- | ------ |
| QUBO encoding | `quantum/qubo.py` (gain vs drop vs cardinality) | Done |
| QAOA solver | `quantum/qaoa_selection.py` + classical fallback | Done |
| CLI strategy | `--selection-strategy quantum-qaoa --quantum-backend {auto,qaoa,classical}` | Done |
| Tests | `tests/test_quantum_selection.py` (11 tests) | Done |
| Docs | `docs/QUANTUM.md` | Done |

**Result on hold-out probe (8 layers, k=5):** QAOA reached ≈97% of the classical
optimum energy. No quantum *advantage* claimed at this scale — the value is a
clean, hardware-portable (D-Wave/annealer-ready) encoding. See `docs/QUANTUM.md`.

### Phase 4 — Cloud, scale, and quantum extensions (future, non-GPU + GPU)

| Step | Description | Status | Needs GPU |
| ---- | ----------- | ------ | --------- |
| 4.1 | First Valohai cloud DAG run with custom Docker image | Planned | No |
| 4.2 | Parallel candidate branches (top-3/5/8) → select-best | Planned | No |
| 4.3 | Quantum annealing backend (D-Wave / Ocean SDK) for the same QUBO | Planned | No |
| 4.4 | Warm-start QAOA from greedy/pareto seed | Planned | No |
| 4.5 | Tensor-Train / MPS weight factorization (quantum-inspired method #2) | Planned | No |
| 4.6 | Full DOTA or larger hold-out | Planned | Yes |
| 4.7 | GPU latency proof vs baseline | User-local (GPU runbook) | Yes |

---

## Export matrix

| Method | `.pt` copy | ONNX (direct) | TorchScript (direct) | Collapsed fallback | TensorRT |
| ------ | ---------- | ------------- | -------------------- | ------------------ | -------- |
| **Baseline** | Yes | Yes (Ultralytics) | Yes (Ultralytics) | N/A | GPU only |
| **FP16** | Yes | Yes | Yes | N/A | GPU only |
| **Low-rank (SVD in-place)** | Yes | Yes | Yes | N/A | GPU only |
| **Structural low-rank** | Yes (no-fuse) | No (custom modules) | No (custom modules) | Yes (collapse then export) | After collapse + GPU |
| **Pruning / INT8** | Yes | If Ultralytics compat | If compat | N/A | GPU only |

**CLI:**

```bash
python scripts/export_model.py \
  --weights path/to/model.pt \
  --format all \
  --device cpu \
  --output-dir runs/export/candidate
```

`--format` accepts `onnx`, `torchscript`, `pt`, or `all`.

**Manifest:** `export_manifest.json` — per-format `attempted` / `success` / `path` / `reason`, plus `collapsed_fallback` for structural checkpoints.

**Structural serving note:** Deployed `.pt` checkpoints require **no-fuse** inference (`predict_no_fuse` in `obb_validate.py`). Collapsed exports trade param savings for Ultralytics/ONNX compatibility.

---

## GPU experiment checklist

Reference: [`docs/GPU_RUNBOOK.md`](GPU_RUNBOOK.md)

| # | Experiment | Command area | Expected artifact |
| - | ---------- | ------------ | ----------------- |
| 1 | Pareto top-5 + BN=20 | `compress_model.py --bn-recalibration-batches 20` | `structural-low-rank_*.json` |
| 2 | Hold-out calibration (test) | `evaluate_calibration.py --device cuda` | `calibration_structural_*_bn20_*.json` |
| 3 | Reliability gate | `check_reliability_gate.py` | `reliability_gate_*.json` |
| 4 | Latency compare (CUDA) | `benchmark_inference.py --device cuda` | `benchmark_*.json` |
| 5 | Full E2E | `./scripts/run_holdout_pipeline.sh --gpu` | All above + Phase 3C report |
| 6 | Export accepted candidate | `export_model.py --format all --device cuda` | `export_manifest.json` + ONNX/TS if baseline |

---

## CI/CD, Docker, Valohai

| Component | Status | Notes |
| --------- | ------ | ----- |
| **GitHub Actions CI** | Done | `pytest -q` on Python 3.11, push/PR |
| **Dockerfile** | WIP | Production runtime image scaffolded |
| **Valohai DAG** | MVP coded | 6 steps: baseline → probe → compress → compare → gate → export |
| **Local E2E** | Done | `scripts/run_holdout_pipeline.sh` |
| **Export step** | Done | `qgc-export-if-accepted` — pt + ONNX + TorchScript + collapsed |

**Valohai pipeline:** `qgc-post-training-compression` in `valohai.yaml`

```bash
vh lint
vh pipeline run qgc-post-training-compression --adhoc
```

---

## Known limitations & negative results

1. **Full structural replacement (Phase 3A)** — r=0.84 on all backbone/neck 3×3 convs → mAP50 ≈ 0.27. Documented; motivates selective compression.
2. **Structural ONNX/TorchScript (direct)** — `StructuralLowRankConv2d` breaks Ultralytics fuse/export; use collapsed fallback or no-fuse `.pt`.
3. **In-place SVD** — theoretical rank reduction without file-size gain (`collapsed_for_ultralytics_compat: false`).
4. **Small hold-out** — test split n=24; directional only, not publication-grade.
5. **Probe signal** — `local_output_error` hooks not fully wired; selection relies on single-layer mAP drop.
6. **TensorRT** — requires NVIDIA GPU, CUDA, and `tensorrt` package; not exercised in CI.

---

## Definition of done

### Project complete (this repo, July 2026)

- [x] Phase 2 + 3B hold-out validation with deployable candidate
- [x] Phase 3C comparison report
- [x] Valohai DAG MVP (6 steps)
- [x] Multi-format export CLI + manifest
- [x] Collapsed structural fallback for ONNX/TorchScript attempts
- [x] Quantum layer selection (QAOA) + classical reference (`docs/QUANTUM.md`)
- [x] CI green (`pytest -q`, ~116 tests)
- [x] `docs/PROJECT_PLAN.md` (this file)
- [x] README + PHASE3_PLAN updated

### Future work (not blocking complete)

**Non-GPU (next):**

- [ ] Quantum annealing backend (D-Wave / Ocean) for the selection QUBO
- [ ] Warm-start QAOA from greedy/pareto seed; hard-constraint (slack) QUBO
- [ ] Tensor-Train / MPS weight factorization (quantum-inspired method #2)
- [ ] Valohai cloud end-to-end run
- [ ] Activation-aware probe (`local_output_error`)

**GPU (user-local):**

- [ ] GPU BN=20 + latency proof (see `docs/GPU_RUNBOOK.md`)
- [ ] Full DOTA scale
- [ ] TensorRT engine build on GPU host

---

## Quick reference — key paths

```text
src/qgeocompress/
  compression/structural_low_rank.py   # StructuralLowRankConv2d
  compression/structural_collapse.py   # Collapse for export compat
  compression/layer_sensitivity.py     # Probe + select_layers dispatch
  quantum/qubo.py                      # QUBO encoding of layer selection
  quantum/qaoa_selection.py            # QAOA solver + classical fallback
  models/export_core.py                # Multi-format orchestration
  cli/export_model.py                  # Export CLI
  valohai/export.py                    # Gate-gated export step

scripts/export_model.py                # Entry point
scripts/run_holdout_pipeline.sh        # Local E2E
docs/PROJECT_PLAN.md                   # This file
docs/QUANTUM.md                        # QUBO/QAOA formulation
docs/PHASE3_PLAN.md                    # Phase 3 detail
docs/GPU_RUNBOOK.md                    # GPU experiments
```

---

## About

**Jules Barth** — M2 Data & AI Engineering, ESILV. Portfolio project demonstrating compression research, reliability gating, and MLOps integration for geospatial AI.
