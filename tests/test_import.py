"""Minimal import / package smoke tests (no weights required)."""

from __future__ import annotations


def test_package_version():
    import deepfake_detection as dd

    assert dd.__version__
    assert dd.__model_id__


def test_metrics_writer(tmp_path, monkeypatch):
    from deepfake_detection.metrics import MetricsRecord, write_metrics

    out = tmp_path / "runs"
    rec = MetricsRecord(
        dataset="unit",
        split="test",
        model_id="unifd-clip-vit-l14-linear",
        acc=0.9,
        ap=0.95,
        auc=0.96,
        n=10,
        threshold=0.5,
        latency_p50_ms=12.0,
        latency_p95_ms=40.0,
        aug={"jpeg": False, "resize": False},
        config_path=None,
    )
    path = write_metrics(rec, run_id="test-run", output_dir=out)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert '"dataset": "unit"' in text
    assert '"git_commit"' in text
    assert '"config_hash"' in text


def test_detect_result_schema():
    from deepfake_detection.models.unifd import DetectResult

    r = DetectResult(
        p_fake=0.12,
        label="real",
        model_id="unifd-clip-vit-l14-linear",
        notes="test",
        version="0.1.0",
    )
    d = r.to_dict()
    assert set(d.keys()) == {"p_fake", "label", "model_id", "notes", "version"}
