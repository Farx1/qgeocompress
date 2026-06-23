import numpy as np

from qgeocompress.calibration.ece import confident_error_rate, expected_calibration_error


def test_ece_perfect_calibration():
    conf = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    correct = np.array([0, 0, 1, 1, 1])
    ece = expected_calibration_error(conf, correct, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_ece_overconfident():
    conf = np.array([0.95, 0.95, 0.95, 0.95])
    correct = np.array([0, 0, 1, 1])
    ece = expected_calibration_error(conf, correct, n_bins=5)
    assert ece > 0.1


def test_confident_error_rate():
    conf = np.array([0.95, 0.92, 0.5, 0.3])
    correct = np.array([0, 1, 1, 0])
    cer = confident_error_rate(conf, correct, threshold=0.9)
    assert cer == 0.5
