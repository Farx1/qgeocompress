# Q-GEOCompress — Phase 3 Plan

**Goal:** Deployable structural low-rank compression (real param / file-size / latency reduction).  
**Prerequisite:** Phase 2A–2B complete (spectral frontier at r ≈ 0.82–0.84, calibration at r = 0.80 cliff).

---

## Phase 2 vs Phase 3

| | Phase 2 (done) | Phase 3 (this plan) |
|---|----------------|---------------------|
| Method | SVD in-place | Structural factorization |
| Architecture | Unchanged Conv2d | `Conv(k×k) → Conv(in→r, k×k) + Conv(r→out, 1×1)` |
| Param count | Same (~5.68 MB) | **Reduced** |
| Purpose | Rank sensitivity + calibration map | **Deployable compression** |
| Fine-tune | Excluded (Ultralytics API) | Deferred (custom loop or train-compatible modules) |

---

## Sub-phases

### Phase 3A — Minimal structural pipeline (current sprint)

**Scope:** backbone/neck 3×3 `.conv` layers only, head OBB excluded.

| Step | Task | Success criterion |
|------|------|-------------------|
| 3A.1 | `StructuralLowRankConv2d` + SVD init | Forward shape preserved |
| 3A.2 | `apply_structural_low_rank()` | Params decrease measurably |
| 3A.3 | CLI `--method structural-low-rank` | JSON + `model.save()` works |
| 3A.4 | Eval mAP + size on DOTA128 | structural r=0.84: mAP50 > 0.75 |
| 3A.5 | Unit tests | pytest green |

**First command to run:**

```bash
BEST="runs/obb/runs/baseline/train/weights/best.pt"
python scripts/compress_model.py \
  --method structural-low-rank \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --target-layers backbone neck
```

### Phase 3B — Sensitivity-guided structural compression (current)

**Motivation:** Phase 3A full replace at r=0.84 collapsed mAP (0.058). QCompress principles → selective, activation-aware replacement.

| Step | Task | Success criterion |
|------|------|-------------------|
| 3B.1 | `layer_sensitivity.py` + `structural-probe` | JSON per-layer scores |
| 3B.2 | `--selection-strategy sensitivity` + `--max-replaced-layers N` | Only top-N layers replaced |
| 3B.3 | BN recalibration (`--bn-recalibration-batches`) | Stabilize running stats |
| 3B.4 | Eval selective top-3 / top-5 at r=0.84 | mAP50 > 0.75 (target > 0.80) |

**Probe (one val per candidate layer):**

```bash
BEST="runs/obb/runs/baseline/train/weights/best.pt"
python scripts/compress_model.py \
  --method structural-probe \
  --rank-ratio 0.84 \
  --weights "$BEST" \
  --target-layers backbone neck
```

Output: `results/summaries/structural_layer_sensitivity.json`

**Selective structural compression:**

```bash
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

### Phase 3C — Comparison matrix + calibration

Generate the unified hold-out comparison from existing JSON summaries:

```bash
python scripts/make_phase3c_report.py
```

Outputs: `results/reports/qgeocompress_phase3c_comparison.md`, `results/summaries/phase3c_comparison_table.csv`

| Model | mAP50 | params Δ | CER@0.8 | ECE | Gate |
|-------|-------|----------|---------|-----|------|
| baseline (hold-out test) | 0.929 | 0% | 0.004 | 0.041 | reference |
| structural top-5 pareto-gain | 0.933 | −5.63% | 0.005 | 0.042 | deployable_compression_candidate |
| structural top-5 sensitivity | 0.879 | −0.21% | 0.005 | 0.041 | methodologically_valid |
| structural full r=0.84 | 0.266 | −7.34% | pending cal | — | rejected |
| SVD in-place r=0.84 (hold-out) | pending | — | — | — | pending |
| structural top-5 pareto BN=20 | pending | — | — | — | pending (GPU) |

Full E2E pipeline (local):

```bash
./scripts/run_holdout_pipeline.sh --skip-train --skip-probe
```

GPU re-runs (BN=20, latency): see [`docs/GPU_RUNBOOK.md`](GPU_RUNBOOK.md).

```bash
python scripts/evaluate_calibration.py --weights ... --compression structural-low-rank --rank-ratio 0.84
python scripts/make_phase3c_report.py
```

### Phase 3D — Hardening (later)

- Optional `k×1 + 1×k` variant for separable approx
- GPU FP16 / TensorRT export of structural model
- Partial layer replacement (wide layers first) if full replace hurts mAP
- Custom fine-tune loop (if mAP gap vs SVD in-place is large)

---

## Technical risks

| Risk | Mitigation |
|------|------------|
| Ultralytics `fuse()` breaks on custom modules | Validate with `model.val()` + `model.save()` / reload; use `obb_validate` if needed |
| mAP collapse like early SVD r=0.5 | Start at r=0.84; compare to Phase 2 frontier |
| File size not shrinking | Count real params; ensure checkpoint saves new modules |
| Calibration drift | Reuse Phase 2B pipeline unchanged |

---

## Success criteria

**Minimal (3A):**

```text
structural r=0.84 → mAP50 > 0.75, real_param_reduction_pct > 0, model_size_mb < baseline
```

**Strong (3B):**

```text
structural r=0.84 → mAP50 > 0.82, CER@0.8 near baseline, measurable size/latency gain
```

---

## Artifact layout

```text
runs/compressed/structural_low_rank_r0.840/
results/summaries/structural-low-rank_*.json
results/summaries/calibration_structural_*.json   # Phase 3B
docs/PHASE3_PLAN.md                               # this file
```

---

## Status

- [x] Phase 3 plan documented
- [x] Phase 3A implementation (`structural_low_rank.py`, CLI, tests)
- [ ] Phase 3A eval r=0.84 on DOTA128
- [x] Phase 3B hold-out validation (pareto-gain deployable)
- [x] Phase 3C comparison report (`make_phase3c_report.py`)
- [ ] Phase 3C GPU re-run (BN=20, latency proof)
