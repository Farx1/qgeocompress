# Q-GEOCompress — Phase 3B Hold-out Validation (DOTA128)

**Protocol:** 80 train / 24 select / 24 test (seed=42), strict separation.

| Split | Role | Used for |
| ----- | ---- | -------- |
| train (80) | learning | baseline training, BN recalibration |
| select (24) | method choice | structural-probe, layer ranking |
| test (24) | final proof | mAP50, ECE, CER@0.8 only |

Baseline: retrained on train only (`runs/obb/runs/obb/runs/baseline_holdout/weights/best.pt`).

## Results on **test split only**

| Méthode | mAP50 | CER@0.8 | ECE | Params | Statut |
| ------- | ----: | ------: | ---: | -----: | ------ |
| baseline hold-out | **0.929** | **0.004** | 0.041 | ref | safe |
| structural top-3 sens | **0.889** | 0.005 | 0.045 | −0.15% | **viable** |
| structural top-5 sens | **0.879** | 0.005 | 0.043 | −0.21% | **viable** |
| structural full r=0.84 | **0.266** | — | — | −7.34% | broken |

## Comparison vs in-sample Phase 3B

| Setting | baseline mAP50 | top-5 mAP50 | full mAP50 |
| ------- | ---------------: | ----------: | ---------: |
| in-sample (train=val) | 0.954 | 0.888 | 0.058 |
| **hold-out test** | **0.929** | **0.879** | **0.266** |

## Conclusion

> Sur un split hold-out strict, **la compression structurelle sélective conserve un avantage net** par rapport au remplacement global. Top-3/top-5 restent viables (mAP50 ≥ baseline − 0.10, CER@0.8 stable). Le remplacement global reste non viable (mAP50 = 0.266).

**Critère de succès (mAP50 ≥ baseline − 0.10, CER stable) :** top-3 et top-5 **passent**.

Note: param reduction on hold-out top-3/5 is small (~0.2%) because probe ranked smaller neck layers on this split; the scientific signal is **generalization of selective vs full**, not max compression.

## Artifacts

- Split manifest: `datasets/dota128_holdout/holdout_manifest.json`
- Probe (select): `results/summaries/structural_layer_sensitivity_holdout_select.json`
- Baseline cal (test): `calibration_baseline_holdout_test_baseline_3cebaa50.json`
- Top-5 cal (test): `calibration_structural_r0.840_holdout_top5_sens_test_e751441c.json`
- Top-3 cal (test): `calibration_structural_r0.840_holdout_top3_sens_test_f634248b.json`

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

# top-5 / top-3 / full — see Phase 3B plan for full flags
```
