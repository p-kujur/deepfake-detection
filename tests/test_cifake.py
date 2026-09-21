"""Unit tests for CIFAKE label mapping and train augs (no network)."""

from __future__ import annotations

from PIL import Image

from deepfake_detection.data.cifake import _normalize_label
from deepfake_detection.data.transforms import TrainAugment, jpeg_compress


def test_normalize_label_named_fake_real():
    # dragonintelligence / yanbax: index 0 = FAKE/fake, 1 = REAL/real
    assert _normalize_label(0, ["FAKE", "REAL"]) == 1
    assert _normalize_label(1, ["FAKE", "REAL"]) == 0
    assert _normalize_label(0, ["fake", "real"]) == 1
    assert _normalize_label(1, ["fake", "real"]) == 0


def test_normalize_label_strings():
    assert _normalize_label("fake") == 1
    assert _normalize_label("REAL") == 0


def test_jpeg_compress_roundtrip():
    img = Image.new("RGB", (32, 32), color=(10, 20, 30))
    out = jpeg_compress(img, quality=50)
    assert out.size == (32, 32)
    assert out.mode == "RGB"


def test_train_augment_runs():
    img = Image.new("RGB", (32, 32), color=(100, 50, 0))
    aug = TrainAugment(jpeg_prob=1.0, blur_prob=1.0, blur_sigma_range=(0.5, 1.0))
    out = aug(img)
    assert out.mode == "RGB"
