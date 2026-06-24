# Q-GEOCompress — Phase 3B Hold-out Validation (DOTA128)

**Protocol:** 80 train / 24 select / 24 test (seed=42), strict separation.

| Split | Role | Used for |
| ----- | ---- | -------- |
| train (80) | learning | baseline training, BN recalibration |
| select (24) | method choice | structural-probe, layer ranking |
| test (24) | final proof | mAP50, ECE, CER@0.8 only |

Baseline: retrained on train only (`runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt`).

## Results on **test split only**

### sensitivity (Phase 3B initial)

| Méthode | mAP50 | CER@0.8 | ECE | Params | Gate |
| ------- | ----: | ------: | ---: | -----: | ---- |
| baseline hold-out | **0.929** | **0.004** | 0.041 | ref | safe |
| structural top-3 sens | 0.889 | 0.005 | 0.045 | −0.15% | methodologically_valid |
| structural top-5 sens | 0.879 | 0.005 | 0.043 | −0.21% | methodologically_valid |
| structural full r=0.84 | 0.266 | — | — | −7.34% | rejected |

### pareto-gain (Phase 3B stack upgrade)

| Méthode | mAP50 | CER@0.8 | ECE | Params | Gate |
| ------- | ----: | ------: | ---: | -----: | ---- |
| structural top-3 pareto | **0.932** | 0.004 | 0.040 | **−5.09%** | **deployable_compression_candidate** |
| structural top-5 pareto | **0.933** | 0.005 | 0.042 | **−5.63%** | **deployable_compression_candidate** |
| structural top-8 pareto | **0.925** | 0.005 | 0.039 | **−5.96%** | **deployable_compression_candidate** |

**pareto-gain** filters `map50_drop ≤ 0.03` on the select split, then ranks by `param_gain_abs` (absolute parameter savings). It targets large backbone/neck convs (`model.5.conv`, `model.7.conv`, …) instead of small neck blocks chosen by compressibility score alone.

## Comparison vs in-sample Phase 3B

| Setting | baseline mAP50 | top-5 mAP50 | full mAP50 |
| ------- | ---------------: | ----------: | ---------: |
| in-sample (train=val) | 0.954 | 0.888 | 0.058 |
| **hold-out test** | **0.929** | **0.879** | **0.266** |

## Conclusion

> Sur un split hold-out strict, **la compression structurelle sélective conserve un avantage net** par rapport au remplacement global. Avec **sensitivity**, top-3/top-5 restent viables en performance mais la réduction de params est négligeable (~0.2 %). Avec **pareto-gain**, top-3/5/8 passent le quality gate complet (`deployable_compression_candidate`) : mAP50 ≥ baseline, CER@0.8 stable, **~5–6 % de params en moins**. Le remplacement global reste non viable (mAP50 = 0.266).

**Critère performance (mAP50 ≥ baseline − 0.10, CER stable) :** sensitivity top-3/5 et pareto top-3/5/8 **passent**.

**Critère compression (≥ 2 % params) :** sensitivity **échoue** ; pareto-gain **passe**.

Export ONNX/TensorRT reste **expérimental** (no-fuse requis pour `StructuralLowRankConv2d`).

## Artifacts

- Split manifest: `datasets/dota128_holdout/holdout_manifest.json`
- Probe (select): `results/summaries/structural_layer_sensitivity_holdout_select.json`
- Baseline cal (test): `calibration_baseline_holdout_test_baseline_3cebaa50.json`
- Top-5 cal sens (test): `calibration_structural_r0.840_holdout_top5_sens_test_e751441c.json`
- Top-3 cal sens (test): `calibration_structural_r0.840_holdout_top3_sens_test_f634248b.json`
- Top-5 pareto cal (test): `calibration_structural_r0.840_holdout_top5_pareto_test_f215351d.json`
- Top-5 pareto gate: `reliability_gate_calibration_structural_r0.840_holdout_top5_pareto_test_f215351d.json`

## Reproduce

```bash
python scripts/create_dota128_holdout.py --seed 42

python scripts/train_baseline.py \
  --data-yaml datasets/dota128_holdout/dota128_holdout_select.yaml \
  --epochs 10 --project runs/obb/runs --name baseline_holdout

BEST="runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt"

python scripts/evaluate_calibration.py \
  --weights "$BEST" \
  --data-yaml datasets/dota128_holdout/dota128_holdout_test.yaml \
  --compression baseline --run-label holdout_test_baseline

python scripts/compress_model.py \
  --method structural-probe --rank-ratio 0.84 --weights "$BEST" \
  --data-yaml datasets/dota128_holdout/dota128_holdout_select.yaml \
  --run-label holdout_select

# top-5 pareto-gain (recommended stack path)
python scripts/compress_model.py \
  --method structural-low-rank --rank-ratio 0.84 --weights "$BEST" \
  --selection-strategy pareto-gain --max-replaced-layers 5 \
  --sensitivity-file results/summaries/structural_layer_sensitivity_holdout_select.json \
  --bn-data-yaml datasets/dota128_holdout/dota128_holdout_train.yaml \
  --eval-data-yaml datasets/dota128_holdout/dota128_holdout_test.yaml \
  --bn-recalibration-batches 20 \
  --run-label holdout_top5_pareto

python scripts/check_reliability_gate.py \
  --baseline results/summaries/calibration_baseline_holdout_test_baseline_3cebaa50.json \
  --candidate results/summaries/calibration_structural_r0.840_holdout_top5_pareto_test_*.json \
  --compression-summary results/summaries/structural-low-rank_*.json
```
