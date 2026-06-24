from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from qgeocompress.utils.config import project_root, save_json


def _image_extensions() -> set[str]:
    return {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _collect_images(source: Path) -> list[Path]:
    root = source / "images" / "train"
    if not root.is_dir():
        raise FileNotFoundError(f"Expected images at {root}")
    exts = _image_extensions()
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found under {root}")
    return images


def _label_for(image: Path, source: Path) -> Path:
    rel = image.relative_to(source / "images" / "train")
    label = source / "labels" / "train" / rel.with_suffix(".txt")
    if not label.exists():
        raise FileNotFoundError(f"Missing label for {image}: {label}")
    return label


def _link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        import shutil

        shutil.copy2(src, dst)


def _write_yaml(path: Path, dataset_root: Path, train_dir: str, val_dir: str) -> None:
    content = {
        "path": str(dataset_root.resolve()),
        "train": train_dir,
        "val": val_dir,
        "names": {
            0: "plane",
            1: "ship",
            2: "storage-tank",
            3: "baseball-diamond",
            4: "tennis-court",
            5: "basketball-court",
            6: "ground-track-field",
            7: "harbor",
            8: "bridge",
            9: "large-vehicle",
            10: "small-vehicle",
            11: "helicopter",
            12: "roundabout",
            13: "soccer-ball-field",
            14: "swimming-pool",
        },
    }
    path.write_text(yaml.dump(content, sort_keys=False))


def create_dota128_holdout(
    source: Path,
    out: Path,
    train_n: int = 80,
    select_n: int = 24,
    test_n: int = 24,
    seed: int = 42,
    use_symlinks: bool = True,
) -> dict:
    """Create train/select/test hold-out split with symlinks."""
    source = source.resolve()
    out = out.resolve()
    images = _collect_images(source)
    total = train_n + select_n + test_n
    if len(images) < total:
        raise ValueError(f"Need {total} images, found {len(images)}")

    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)
    train_imgs = shuffled[:train_n]
    select_imgs = shuffled[train_n : train_n + select_n]
    test_imgs = shuffled[train_n + select_n : total]

    splits = {"train": train_imgs, "select": select_imgs, "test": test_imgs}
    for split_name, split_imgs in splits.items():
        for img in split_imgs:
            rel = img.name
            _link_or_copy(img, out / "images" / split_name / rel, use_symlinks)
            lbl = _label_for(img, source)
            _link_or_copy(lbl, out / "labels" / split_name / lbl.name, use_symlinks)

    _write_yaml(out / "dota128_holdout_train.yaml", out, "images/train", "images/select")
    _write_yaml(out / "dota128_holdout_select.yaml", out, "images/train", "images/select")
    _write_yaml(out / "dota128_holdout_test.yaml", out, "images/train", "images/test")
    _write_yaml(out / "dota128_holdout_train.yaml", out, "images/train", "images/train")
    _write_yaml(out / "dota128_holdout_bn.yaml", out, "images/train", "images/train")

    manifest = {
        "source": str(source),
        "out": str(out),
        "seed": seed,
        "train_n": train_n,
        "select_n": select_n,
        "test_n": test_n,
        "train_images": [p.name for p in train_imgs],
        "select_images": [p.name for p in select_imgs],
        "test_images": [p.name for p in test_imgs],
        "yaml_files": {
            "train": str(out / "dota128_holdout_train.yaml"),
            "select": str(out / "dota128_holdout_select.yaml"),
            "test": str(out / "dota128_holdout_test.yaml"),
            "train_bn": str(out / "dota128_holdout_train.yaml"),
            "bn": str(out / "dota128_holdout_bn.yaml"),
        },
    }
    save_json(manifest, out / "holdout_manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create DOTA128 train/select/test hold-out split")
    parser.add_argument("--source", type=Path, default=project_root() / "datasets" / "dota128")
    parser.add_argument("--out", type=Path, default=project_root() / "datasets" / "dota128_holdout")
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--select", type=int, default=24)
    parser.add_argument("--test", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinks")
    args = parser.parse_args(argv)

    manifest = create_dota128_holdout(
        source=args.source,
        out=args.out,
        train_n=args.train,
        select_n=args.select,
        test_n=args.test,
        seed=args.seed,
        use_symlinks=not args.copy,
    )
    print(f"Hold-out split created at {manifest['out']}")
    print(f"  train={manifest['train_n']} select={manifest['select_n']} test={manifest['test_n']}")


if __name__ == "__main__":
    main()
