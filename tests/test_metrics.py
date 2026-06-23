import numpy as np

from qgeocompress.data.corruptions import apply_corruption


def test_corruption_preserves_shape():
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    for corruption in ("blur", "noise", "jpeg", "brightness", "cloud"):
        out = apply_corruption(img, corruption, severity=3)
        assert out.shape == img.shape


def test_corruption_labels_unchanged_conceptually():
    # Corruptions only touch pixels; this test documents the contract.
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = apply_corruption(img, "contrast", severity=2)
    assert out.dtype == np.uint8
