"""Optional end-to-end smoke: skips if probe weights or heavy deps missing."""

from __future__ import annotations

from pathlib import Path

import pytest

WEIGHTS = Path("weights/unifd_clip_vit_l14_linear.pth")
FIXTURE = Path("tests/fixtures/smoke_gray.png")


def _has_torch_and_clip() -> bool:
    try:
        import torch  # noqa: F401
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="probe weights not present")
@pytest.mark.skipif(not _has_torch_and_clip(), reason="torch/open_clip not installed")
def test_detect_smoke_with_weights():
    from deepfake_detection.models.unifd import detect_image

    if not FIXTURE.is_file():
        pytest.skip("fixtures missing; run scripts/make_smoke_fixtures.py")
    result = detect_image(FIXTURE, probe_weights=WEIGHTS, device="cpu")
    d = result.to_dict()
    assert 0.0 <= d["p_fake"] <= 1.0
    assert d["label"] in ("real", "fake")
    assert d["model_id"]
    assert d["version"]


def test_cli_help():
    from deepfake_detection.infer.detect import build_parser

    p = build_parser()
    help_text = p.format_help()
    assert "IMAGE" in help_text.upper() or "image" in help_text
