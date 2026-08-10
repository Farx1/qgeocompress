import random

from qgeocompress.utils.config import load_dataset_config, load_experiment_config, make_run_id
from qgeocompress.utils.seed import set_seed


def test_config_loading():
    cfg = load_dataset_config("dota128")
    assert cfg["name"] == "dota128"
    exp = load_experiment_config("baseline")
    assert exp["compression"] == "none"


def test_run_id_stable():
    cfg = {"a": 1, "b": 2}
    assert make_run_id("test", cfg) == make_run_id("test", cfg)


def test_seed_reproducibility():
    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b
