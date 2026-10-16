# Model card: compressed YOLO11n-OBB (data-aware low-rank, 16% budget)

## Model

- **Base model:** Ultralytics `yolo11n-obb.pt` (YOLO11 nano, oriented bounding boxes, pretrained
  by Ultralytics on DOTAv1, 15 classes).
- **Compression:** post-training, no gradient training. Every eligible conv outside the detection
  head (3×3 and 1×1, dense, ≥ 8 channels) is a candidate. Each replaced conv becomes a rank-`r`
  `k×k` conv followed by a `1×1` conv, factored by activation-weighted SVD
  (`compression/data_aware.py`, derivation in [`docs/THEORY.md`](docs/THEORY.md) §3). Per-layer
  ranks come from a Lagrangian allocation under a 16% parameter budget, weighted by transfer
  coefficients; BatchNorm statistics are then recalibrated on the 24-image train split.
- **Result:** −20.67% parameters (budget arm "16%"; the allocation overshoots the target).
- **No weights are distributed.** The model is rebuilt by
  `python scripts/frontier_study.py --preset frontier` (CPU, ~8 min).

## Intended use

Research on compression that keeps detection quality and confidence reliability, on aerial
imagery. Not intended for production, safety-critical, surveillance or targeting use, and not
validated beyond the 80-image DOTA128 test split.

## Data

[`datasets/dota128_power`](datasets/dota128_power/DATASET_CARD.md): 24 train (BN recalibration),
24 select (activation statistics, layer sensitivity), 80 test (scoring), seed 42, from Ultralytics
DOTA128.

## Metrics

80 test images, 1434 boxes. mAP50 is this repo's own AP (`evaluation/ap.py`), which runs lower than
the Ultralytics figure; deltas are paired bootstraps over images (95% CI). ECE: 15 bins; CER@0.8:
share of detections with confidence ≥ 0.8 that are wrong (definitions in `docs/THEORY.md` §4).

| Model | Params Δ | mAP50 | Δ vs baseline (95% CI) | ECE | CER@0.8 | Gate |
| ----- | -------: | ----: | ---------------------- | --: | ------: | ---- |
| Baseline `yolo11n-obb.pt` | 0 | 0.8943 | reference | 0.0437 | 0.0182 | — |
| Baseline + BN recalibration only | 0 | 0.8173 | −0.0770 [−0.1508, −0.0218] | 0.0422 | 0.0165 | methodologically valid |
| **This model** | **−20.67%** | **0.8007** | −0.0937 [−0.1678, −0.0305] | 0.0458 | 0.0180 | deployable candidate |

Gate: `evaluate_reliability_gate` defaults (mAP50 drop ≤ 0.10, CER@0.8 rise ≤ 0.02, params ≥ 2%).
Source: `results/summaries/frontier_frontier.json` produced by the command above.

## Limitations

- Most of the mAP50 drop comes from BatchNorm recalibration on 24 images, not from the
  factorization (second row). Against that control the factorization costs about 0.017 mAP50.
- 80 test images: confidence intervals are wide; the gate compares point estimates.
- No latency gain is measured for this model. On CPU, back-to-back runs of a −5.6% model showed none.
- Classes are unbalanced in the test split (ship and small-vehicle are 66% of boxes); several
  classes have ≤ 2 boxes.
- The test tiles come from DOTAv1, the pretraining dataset of the base model; overlap is possible
  and was not checked.
- ECE uses binning and is a lower bound on the true calibration error.

## License

- Repository code: MIT ([LICENSE](LICENSE)).
- Base weights: Ultralytics YOLO11, AGPL-3.0; weights derived from them carry the same terms
  (Ultralytics also sells a commercial license).
- Data: DOTA128 is redistributed by Ultralytics under GPL-3.0; DOTA images are for academic,
  non-commercial use only.
