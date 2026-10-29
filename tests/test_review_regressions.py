"""Targeted regression checks; run with PYTHONPATH=src python tests/test_review_regressions.py.

Uses real project modules, stdlib unittest and PyYAML. No model inference.
Fixture images are placeholder bytes: the split function never decodes them.
"""
import tempfile
import unittest
from pathlib import Path

from qgeocompress.cli.create_dota128_holdout import create_dota128_holdout
from qgeocompress.evaluation.reliability_gate import GateThresholds, evaluate_reliability_gate


class GateRegressions(unittest.TestCase):
    def gate(self, **kwargs):
        candidate = {"map50": 0.89, "confident_error_rate_08": 0.02, "status": "ok"}
        candidate.update(kwargs)
        return evaluate_reliability_gate(
            {"map50": 0.9, "confident_error_rate_08": 0.02},
            candidate, {"real_param_reduction_pct": 3.0},
        )

    def test_valid_candidate_is_accepted(self):
        self.assertTrue(self.gate()["accepted_for_export"])

    def test_nan_map_is_rejected(self):
        self.assertFalse(self.gate(map50=float("nan"))["accepted_for_export"])

    def test_infinite_map_is_rejected(self):
        self.assertFalse(self.gate(map50=float("inf"))["accepted_for_export"])

    def test_negative_cer_is_rejected(self):
        self.assertFalse(self.gate(confident_error_rate_08=-0.1)["accepted_for_export"])

    def test_explicit_zero_reduction_is_preserved(self):
        result = evaluate_reliability_gate(
            {"map50": 0.9, "confident_error_rate_08": 0.02},
            {"map50": 0.89, "confident_error_rate_08": 0.02, "status": "ok",
             "real_param_reduction_pct": 3.0},
            {"real_param_reduction_pct": 0.0},
        )
        self.assertFalse(result["accepted_for_export"])
        self.assertEqual(result["real_param_reduction_pct"], 0.0)

    def test_drop_above_threshold_is_not_rounded_into_acceptance(self):
        self.assertFalse(self.gate(map50=0.7999996)["accepted_for_export"])

    def test_exact_threshold_survives_float_arithmetic(self):
        result = evaluate_reliability_gate(
            {"map50": 0.93, "confident_error_rate_08": 0.02},
            {"map50": 0.83, "confident_error_rate_08": 0.02, "status": "ok"},
            {"real_param_reduction_pct": 3.0},
        )
        self.assertTrue(result["accepted_for_export"])

    def test_infinite_threshold_is_invalid(self):
        with self.assertRaises(ValueError):
            evaluate_reliability_gate({}, {}, thresholds=GateThresholds(max_map50_drop=float("inf")))


class SplitRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.out = self.root / "out"
        for i in range(8):
            self.add_image(f"img{i}.jpg")

    def add_image(self, relative):
        image = self.source / "images/train" / relative
        label = self.source / "labels/train" / Path(relative).with_suffix(".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture")
        label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

    def create(self, **kwargs):
        opts = dict(train_n=4, select_n=2, test_n=2, seed=42, use_symlinks=False)
        opts.update(kwargs)
        return create_dota128_holdout(self.source, self.out, **opts)

    def test_second_run_cannot_mix_splits(self):
        self.create(seed=42)
        try:
            self.create(seed=7)
        except FileExistsError:
            pass  # Explicit refusal is a valid safe contract.
        sets = [{p.name for p in (self.out / "images" / split).rglob("*.jpg")}
                for split in ("train", "select", "test")]
        self.assertEqual([len(s) for s in sets], [4, 2, 2])
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))

    def test_nested_names_do_not_overwrite(self):
        for name in ("a/same.jpg", "b/same.jpg"):
            self.add_image(name)
        # Distinct nested input files must retain their identity in the output.
        self.create(train_n=10, select_n=0, test_n=0)
        images = list((self.out / "images/train").rglob("*.jpg"))
        self.assertEqual(len(images), 10)
        self.assertTrue((self.out / "images/train/a/same.jpg").is_file())

    def test_negative_count_is_rejected(self):
        with self.assertRaises(ValueError):
            self.create(train_n=-1, select_n=2, test_n=2)

    def test_missing_labels_are_detected_before_output_is_written(self):
        for label in (self.source / "labels/train").glob("*.txt"):
            label.unlink()
        with self.assertRaises(FileNotFoundError):
            self.create()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
