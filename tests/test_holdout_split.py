import pytest
from pathlib import Path

from qgeocompress.cli.create_dota128_holdout import create_dota128_holdout
from qgeocompress.data.prepare_dota import resolve_data_yaml
from ultralytics.data.utils import check_det_dataset


@pytest.fixture
def tiny_dota(tmp_path):
    source = tmp_path / "dota128"
    for split in ("train",):
        for kind in ("images", "labels"):
            (source / kind / split).mkdir(parents=True)
    img = source / "images" / "train" / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    (source / "labels" / "train" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    for i in range(2, 129):
        name = f"img{i}.jpg"
        (source / "images" / "train" / name).write_bytes(b"\xff\xd8\xff\xd9")
        (source / "labels" / "train" / name.replace(".jpg", ".txt")).write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
        )
    return source


def test_create_holdout_split_counts(tiny_dota, tmp_path):
    out = tmp_path / "holdout"
    manifest = create_dota128_holdout(
        source=tiny_dota,
        out=out,
        train_n=80,
        select_n=24,
        test_n=24,
        seed=42,
        use_symlinks=True,
    )
    assert manifest["train_n"] == 80
    assert len(list((out / "images" / "train").iterdir())) == 80
    assert len(list((out / "images" / "select").iterdir())) == 24
    assert len(list((out / "images" / "test").iterdir())) == 24
    assert (out / "dota128_holdout_test.yaml").exists()


def test_holdout_yaml_splits_disjoint(tiny_dota, tmp_path):
    out = tmp_path / "holdout"
    create_dota128_holdout(tiny_dota, out, seed=7)
    select_info = check_det_dataset(str(out / "dota128_holdout_select.yaml"))
    test_info = check_det_dataset(str(out / "dota128_holdout_test.yaml"))
    select_names = {p.name for p in Path(select_info["val"]).iterdir()}
    test_names = {p.name for p in Path(test_info["val"]).iterdir()}
    assert select_names.isdisjoint(test_names)


def test_resolve_data_yaml_path(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nval: images/test\nnames:\n  0: plane\n")
    resolved = resolve_data_yaml(data_yaml=yaml_path)
    assert resolved == str(yaml_path.resolve())
