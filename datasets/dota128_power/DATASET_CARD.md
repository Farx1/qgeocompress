# Dataset card: `dota128_power`

A fixed train / select / test split of the Ultralytics DOTA128 sample, used by the frontier study
(`scripts/frontier_study.py`) and the README's headline table.

## Provenance

- **Images and labels:** `datasets/dota128/` — Ultralytics DOTA128, 128 image tiles of the split
  DOTAv1 set, oriented-box labels in YOLO-OBB format, 15 classes
  (see `datasets/dota128/README.md`).
- **Upstream dataset:** [DOTA](https://captain-whu.github.io/DOTA/dataset.html) (aerial images,
  15 categories). Its terms: academic use only, commercial use prohibited; Google Earth imagery
  must also respect Google Earth's terms of use.
- **This directory adds no image:** files under `images/` and `labels/` are relative symlinks into
  `datasets/dota128/` (Windows checkouts without symlink support get small text files instead; run
  the command below with `--copy` to get real files).

## How the split is made

```bash
python scripts/create_dota128_holdout.py --source datasets/dota128 --out datasets/dota128_power \
  --train 24 --select 24 --test 80 --seed 42
```

The sorted image list is shuffled with `random.Random(42)` and cut in that order.
`holdout_manifest.json` lists every file per split; rerunning the command reproduces it exactly.

| Split | Images | OBB boxes | Used for |
| ----- | -----: | --------: | -------- |
| train | 24 | 346 | BatchNorm recalibration after compression (`*_train.yaml`, `*_bn.yaml`) |
| select | 24 | 279 | Activation statistics (Gram matrices), layer sensitivity, method choice |
| test | 80 | 1434 | Final scoring only (paired bootstrap against the baseline) |

Splits are disjoint. The baseline `yolo11n-obb.pt` is pretrained by Ultralytics and never trained on
these splits, which is why the test split can be the largest one.

Test-split boxes per class: plane 79, ship 689, storage-tank 134, baseball-diamond 4,
tennis-court 2, basketball-court 0, ground-track-field 7, harbor 64, bridge 25,
large-vehicle 153, small-vehicle 265, helicopter 1, roundabout 8, soccer-ball-field 2,
swimming-pool 1.

## Limitations

- 128 tiles in total: mAP50 on 80 images has a wide confidence interval; only paired comparisons
  separate arms.
- Two classes dominate the test split (ship + small-vehicle = 66% of boxes); several classes have
  ≤ 2 boxes and basketball-court has none, so per-class AP is noisy and rare classes weigh heavily
  in the repo's own mAP50.
- `yolo11n-obb.pt` is pretrained by Ultralytics on DOTAv1 and DOTA128 tiles come from DOTAv1, so the
  test images may overlap the pretraining data (not checked).
  Absolute mAP50 may be optimistic; relative comparisons between arms are what the split supports.

## License

Not covered by the repository's code license. DOTA128 is redistributed from Ultralytics under
GPL-3.0 (`datasets/dota128/LICENSE`); the underlying DOTA images are for academic, non-commercial use.
