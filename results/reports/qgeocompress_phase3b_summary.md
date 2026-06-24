# Q-GEOCompress — Phase 3B Summary

**Sensitivity-guided structural low-rank** (DOTA128, YOLO11n-OBB, r=0.84)

## Comparison table

| Méthode | mAP50 | Params | Taille | ECE | CER@0.8 | Statut |
| ------- | ----: | -----: | -----: | ---: | ------: | ------ |
| baseline | 0.954 | 2.66M | 5.65 MB | 0.036 | 0.022 | safe |
| SVD in-place r=0.84 | 0.841 | ~2.66M | ~5.68 MB | 0.030 | 0.026 | viable |
| structural full r=0.84 | 0.058 | 2.47M (−7.3%) | 5.35 MB | — | — | broken |
| **structural top-3 sens** | **0.888** | **−1.8%** | **5.55 MB** | **0.034** | **0.020** | **safe** |
| **structural top-5 sens** | **0.888** | **−4.75%** | **5.46 MB** | **0.035** | **0.021** | **safe** |

## Key findings

1. **Naïve structural replacement collapses the model** (mAP50 0.058) despite real param reduction.
2. **Single-layer probes + compressibility score** identify layers tolerating factorization (e.g. `model.7.conv`, drop ≈ 0).
3. **Selective top-5** delivers the best deployable trade-off today: **−4.75% params**, mAP50 **0.888**, CER@0.8 **≤ baseline**.
4. **Top-3 vs top-5**: identical mAP50; top-3 has marginally lower CER@0.8 (0.020 vs 0.021) and ECE; top-5 adds ~3% extra param savings with no measurable reliability penalty.

## Calibration artifacts

| Run | JSON |
| --- | --- |
| top-3 | `results/summaries/calibration_structural_r0.840_top3_sens_67cde22b.json` |
| top-5 | `results/summaries/calibration_structural_r0.840_top5_sens_cdd13834.json` |
| layer probe | `results/summaries/structural_layer_sensitivity.json` |

## Conclusion

> La compression structurelle sélective guidée par sensibilité permet une **réduction réelle des paramètres** tout en conservant une **performance élevée** et un **risque d'erreurs confiantes contenu** (CER@0.8 proche ou meilleur que baseline).

**Meilleur checkpoint actuel :** `runs/compressed/structural_low_rank/structural_low_rank_r0.840_top5_sens.pt`
