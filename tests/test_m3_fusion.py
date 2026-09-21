"""M3/M3c smoke: fusion math (mean + logit) + dual-probe helpers."""

from __future__ import annotations

import numpy as np

from deepfake_detection.eval.cross_gen import fuse_logits, fuse_probs
from deepfake_detection.eval.dual_probe import confidence_router, soft_confidence_mix


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


def test_fuse_logits_mean():
    a = np.array([0.0, 2.0])
    b = np.array([0.0, -2.0])
    out = fuse_logits(a, b, weight_a=0.5)
    np.testing.assert_allclose(out, [0.5, 0.5], atol=1e-6)


def test_confidence_router_picks_stronger():
    # |a|=10 > |b|=1 → pick a → near 1
    a = np.array([10.0])
    b = np.array([-1.0])
    out = confidence_router(a, b)
    assert out[0] > 0.99


def test_soft_confidence_mix_shape():
    a = np.array([2.0, -2.0])
    b = np.array([0.1, 0.1])
    out = soft_confidence_mix(a, b)
    assert out.shape == (2,)
    assert np.all((out >= 0) & (out <= 1))
