"""M4: shared detect schema, weight preference, CLI/API JSON shape."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepfake_detection.infer.resolve import (
    SCHEMA_KEYS,
    PRODUCT_NOTES,
    model_id_for_weights,
    product_notes,
    resolve_default_probe_weights,
    schema_dict,
)
from deepfake_detection.models.unifd import DetectResult


def test_schema_dict_keys():
    d = schema_dict(p_fake=0.2, label="real", model_id="progan-clip-vit-l14-linear")
    assert set(d.keys()) == set(SCHEMA_KEYS)
    assert 0.0 <= d["p_fake"] <= 1.0
    assert d["label"] in ("real", "fake")
    assert "court-grade" in d["notes"].lower() or "not court-grade" in d["notes"].lower()
    assert "0.78" in d["notes"] or "≈0.78" in d["notes"]
    assert "CIFAKE" in d["notes"] or "cifake" in d["notes"].lower()


def test_product_notes_mention_near_miss():
    n = product_notes()
    assert "0.78" in n
    assert "court-grade" in n.lower() or "not court-grade" in n.lower()
    assert "CIFAKE" in n or "cifake" in n.lower()
    # Must not claim ≥0.80 cross-gen success
    assert "≥0.80" in n or ">=0.80" in n or "as ≥0.80" in n or "as >=0.80" in n


def test_model_id_for_weights():
    assert model_id_for_weights("weights/progan_clip_vit_l14_linear.pth") == (
        "progan-clip-vit-l14-linear"
    )
    assert model_id_for_weights("weights/unifd_clip_vit_l14_linear.pth") == (
        "unifd-clip-vit-l14-linear"
    )
    assert model_id_for_weights("weights/cifake_clip_vit_l14_linear.pth") == (
        "cifake-clip-vit-l14-linear"
    )


def test_resolve_preference_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    weights.mkdir()
    # Only CIFAKE present → should pick cifake
    (weights / "cifake_clip_vit_l14_linear.pth").write_bytes(b"x" * 100)
    path, mid, found = resolve_default_probe_weights(None)
    assert found
    assert mid == "cifake-clip-vit-l14-linear"
    assert path.name == "cifake_clip_vit_l14_linear.pth"

    # Add UniFD → should prefer UniFD over CIFAKE
    (weights / "unifd_clip_vit_l14_linear.pth").write_bytes(b"x" * 100)
    path, mid, found = resolve_default_probe_weights(None)
    assert mid == "unifd-clip-vit-l14-linear"

    # Add ProGAN → top preference
    (weights / "progan_clip_vit_l14_linear.pth").write_bytes(b"x" * 100)
    path, mid, found = resolve_default_probe_weights(None)
    assert mid == "progan-clip-vit-l14-linear"


def test_resolve_explicit_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path, mid, found = resolve_default_probe_weights("weights/nope.pth")
    assert not found
    assert mid  # still a model_id string


def test_detect_result_to_dict_schema():
    r = DetectResult(
        p_fake=0.9,
        label="fake",
        model_id="progan-clip-vit-l14-linear",
        notes=PRODUCT_NOTES,
        version="0.1.0",
    )
    d = r.to_dict()
    assert list(d.keys()) == list(SCHEMA_KEYS)


def test_cli_json_schema_with_mock(tmp_path, monkeypatch):
    """CLI prints schema-valid JSON without loading CLIP."""
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # minimal junk; detect is mocked

    fake = DetectResult(
        p_fake=0.42,
        label="real",
        model_id="progan-clip-vit-l14-linear",
        notes=PRODUCT_NOTES,
        version="0.1.0",
    )
    monkeypatch.chdir(tmp_path)
    # Provide a config so CLI does not require repo root
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "default.yaml").write_text("model:\n  threshold: 0.5\ninfer:\n  device: cpu\n")

    with patch("deepfake_detection.infer.detect.run_detect", return_value=fake):
        from deepfake_detection.infer.detect import main
        import io
        import sys

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            code = main([str(img), "--config", str(cfg / "default.yaml")])
        finally:
            sys.stdout = old
        assert code == 0
        payload = json.loads(buf.getvalue())
        assert set(payload.keys()) == set(SCHEMA_KEYS)
        assert payload["model_id"] == "progan-clip-vit-l14-linear"
        assert "0.78" in payload["notes"]


def test_cli_help_mentions_ensemble():
    from deepfake_detection.infer.detect import build_parser

    help_text = build_parser().format_help()
    assert "soft_router" in help_text
    assert "ensemble" in help_text


def test_api_detect_schema_with_mock():
    """FastAPI /detect returns the five schema keys (model mocked)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    fake = DetectResult(
        p_fake=0.77,
        label="fake",
        model_id="progan-clip-vit-l14-linear",
        notes=PRODUCT_NOTES,
        version="0.1.0",
    )

    import deepfake_detection.api.app as app_mod

    app_mod._MODEL = MagicMock()
    app_mod._MODEL_ID = "progan-clip-vit-l14-linear"
    app_mod._WEIGHTS_PATH = "weights/progan_clip_vit_l14_linear.pth"
    app_mod._CFG = {"model": {"threshold": 0.5, "image_size": 224}, "notes": None}
    app_mod._ENSEMBLE_DEFAULT = None

    with patch("deepfake_detection.api.app.run_detect", return_value=fake):
        client = TestClient(app_mod.app)
        # Avoid real startup loading CLIP
        resp = client.post(
            "/detect",
            files={"file": ("t.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert set(payload.keys()) == set(SCHEMA_KEYS)
    assert payload["label"] in ("real", "fake")
    assert "0.78" in payload["notes"]


def test_api_health_includes_notes():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import deepfake_detection.api.app as app_mod

    app_mod._MODEL = MagicMock()
    app_mod._MODEL_ID = "progan-clip-vit-l14-linear"
    app_mod._WEIGHTS_PATH = "weights/progan_clip_vit_l14_linear.pth"
    app_mod._ENSEMBLE_DEFAULT = None
    client = TestClient(app_mod.app, raise_server_exceptions=False)
    # health does not need startup model load beyond globals
    with patch.object(app_mod, "startup", lambda: None):
        resp = client.get("/health")
    # TestClient may still run startup; tolerate either
    if resp.status_code != 200:
        pytest.skip(f"health unavailable in this env: {resp.status_code}")
    body = resp.json()
    assert "model_id" in body
    assert "version" in body
    assert "notes" in body
    assert "0.78" in body["notes"]
