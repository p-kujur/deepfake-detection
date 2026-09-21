"""M3 smoke: fusion math + folder dataset resolution."""

from __future__ import annotations

import numpy as np

from deepfake_detection.eval.cross_gen import fuse_probs


def test_fuse_probs_mean():
    a = np.array([0.0, 1.0, 0.5])
    b = np.array([1.0, 0.0, 0.5])
    out = fuse_probs(a, b, weight_a=0.5)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5])


def test_fuse_probs_weight():
    a = np.array([1.0])
    b = np.array([0.0])
    out = fuse_probs(a, b, weight_a=0.8)
    np.testing.assert_allclose(out, [0.8])
