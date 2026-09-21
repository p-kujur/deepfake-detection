"""M1 smoke gate: labeled 20-image real/fake pack with UniFD probe weights.

Requires:
  - weights/unifd_clip_vit_l14_linear.pth (see scripts/download_weights.py)
  - torch + open_clip
  - tests/fixtures/smoke/{real,fake}/*.png (10 each) or smoke/manifest.json

Gate: ≥18/20 correct at threshold 0.5 (docs/plan.md P1b / Smoke correctness).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

WEIGHTS = Path("weights/unifd_clip_vit_l14_linear.pth")
SMOKE_DIR = Path("tests/fixtures/smoke")
MANIFEST = SMOKE_DIR / "manifest.json"
MIN_CORRECT = 18
N_EXPECTED = 20


def _has_torch_and_clip() -> bool:
    try:
        import torch  # noqa: F401
        import open_clip  # noqa: F401

        return True
    except Exception:
        return False


def _load_smoke_items() -> list[tuple[Path, str]]:
    """Return (path, label) pairs from manifest or real/fake directories."""
    items: list[tuple[Path, str]] = []
    if MANIFEST.is_file():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for row in data:
            p = Path(row["path"])
            label = str(row["label"]).lower().strip()
            if label not in ("real", "fake"):
                raise AssertionError(f"bad label in manifest: {row}")
            items.append((p, label))
        return items

    for label in ("real", "fake"):
        d = SMOKE_DIR / label
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            items.append((p, label))
    return items


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="probe weights not present")
@pytest.mark.skipif(not _has_torch_and_clip(), reason="torch/open_clip not installed")
def test_smoke_pack_accuracy_gate():
    from deepfake_detection.models.unifd import detect_image, load_detector

    items = _load_smoke_items()
    if len(items) < N_EXPECTED:
        pytest.skip(
            f"labeled smoke pack incomplete ({len(items)}/{N_EXPECTED}); "
            "see tests/fixtures/README.md"
        )

    # Load once; reuse for all images
    model, _ = load_detector(probe_weights=WEIGHTS, device="cpu", require_weights=True)

    correct = 0
    details: list[str] = []
    for path, label in items[:N_EXPECTED]:
        assert path.is_file(), f"missing fixture: {path}"
        result = detect_image(path, model=model, device="cpu", threshold=0.5)
        ok = result.label == label
        correct += int(ok)
        details.append(
            f"{path.name}: true={label} pred={result.label} p_fake={result.p_fake:.4f}"
        )

    msg = f"smoke accuracy {correct}/{N_EXPECTED} (need ≥{MIN_CORRECT})\n" + "\n".join(
        details
    )
    assert correct >= MIN_CORRECT, msg


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="probe weights not present")
@pytest.mark.skipif(not _has_torch_and_clip(), reason="torch/open_clip not installed")
def test_detect_smoke_with_weights_schema():
    """Single-image schema smoke (any smoke real image)."""
    from deepfake_detection.models.unifd import detect_image

    items = _load_smoke_items()
    if not items:
        legacy = Path("tests/fixtures/smoke_gray.png")
        if not legacy.is_file():
            pytest.skip("no smoke fixtures")
        path = legacy
    else:
        path = items[0][0]

    result = detect_image(path, probe_weights=WEIGHTS, device="cpu")
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


def test_load_probe_weights_fc_format():
    """HF UniFD fc_weights.pth is Linear weight [1,768] + bias [1]."""
    if not WEIGHTS.is_file():
        pytest.skip("probe weights not present")
    if not _has_torch_and_clip():
        pytest.skip("torch/open_clip not installed")

    import torch
    from deepfake_detection.models.unifd import UniFDClipLinear

    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    assert "weight" in state and "bias" in state
    assert tuple(state["weight"].shape) == (1, 768)
    assert tuple(state["bias"].shape) == (1,)

    model = UniFDClipLinear(backbone="ViT-L-14", pretrained="openai", backend="open_clip")
    model.load_probe_weights(WEIGHTS)
    assert torch.allclose(model.head.weight, state["weight"])
    assert torch.allclose(model.head.bias, state["bias"])
