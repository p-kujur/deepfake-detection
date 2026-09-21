"""FastAPI app: POST /detect (multipart image) → shared JSON schema."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from deepfake_detection import __version__
from deepfake_detection.infer.pipeline import run_detect
from deepfake_detection.infer.resolve import (
    SCHEMA_KEYS,
    product_notes,
    resolve_default_probe_weights,
)
from deepfake_detection.models.unifd import load_detector

app = FastAPI(
    title="Deepfake Detection API",
    version=__version__,
    description=(
        "AIGC still-image detector (CLIP ViT-L/14 + linear probe). "
        "Scores are experimental — not court-grade. M3 Hemg cross-gen AUC ≈0.78 near-miss."
    ),
)

_MODEL = None
_DEVICE = None
_CFG: dict[str, Any] = {}
_MODEL_ID: str = "unloaded"
_WEIGHTS_PATH: str | None = None
_ENSEMBLE_DEFAULT: str | None = None


def _load_cfg() -> dict[str, Any]:
    path = Path("configs/default.yaml")
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _cfg_probe_explicit(model_cfg: dict[str, Any]) -> str | None:
    cfg_w = model_cfg.get("probe_weights")
    if cfg_w in (None, "", "auto"):
        return None
    return str(cfg_w)


@app.on_event("startup")
def startup() -> None:
    global _MODEL, _DEVICE, _CFG, _MODEL_ID, _WEIGHTS_PATH, _ENSEMBLE_DEFAULT
    _CFG = _load_cfg()
    model_cfg = _CFG.get("model", {})
    infer_cfg = _CFG.get("infer", {})
    explicit = os.environ.get("DFD_PROBE_WEIGHTS") or _cfg_probe_explicit(model_cfg)
    path, mid, found = resolve_default_probe_weights(explicit)
    ens = infer_cfg.get("ensemble") or os.environ.get("DFD_ENSEMBLE") or None
    if ens in ("", "none", "off", "null"):
        ens = None
    _ENSEMBLE_DEFAULT = ens
    _MODEL_ID = mid if found else f"{mid}-untrained"
    _WEIGHTS_PATH = str(path) if found else None
    try:
        _MODEL, _DEVICE = load_detector(
            probe_weights=path if found else None,
            backend=model_cfg.get("backend", "open_clip"),
            backbone=model_cfg.get("backbone", "ViT-L-14"),
            pretrained=model_cfg.get("pretrained", "openai"),
            device=infer_cfg.get("device", "auto"),
            require_weights=False,
        )
        if found:
            _MODEL_ID = mid
    except Exception as e:
        _MODEL = None
        _DEVICE = None
        app.state.startup_error = str(e)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "model_id": _MODEL_ID,
        "weights": _WEIGHTS_PATH,
        "ensemble_default": _ENSEMBLE_DEFAULT,
        "version": __version__,
        "notes": product_notes(),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    ensemble: str | None = Form(default=None),
) -> JSONResponse:
    """Return ``{p_fake, label, model_id, notes, version}``."""
    mode = ensemble if ensemble is not None else _ENSEMBLE_DEFAULT
    if mode in ("", "none", "off"):
        mode = None

    if _MODEL is None and mode != "soft_router":
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check startup logs / weights / dependencies.",
        )

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    model_cfg = _CFG.get("model", {})
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            if mode == "soft_router":
                result = run_detect(
                    tmp_path,
                    ensemble="soft_router",
                    probe_weights=_cfg_probe_explicit(model_cfg),
                    cifake_weights=model_cfg.get("cifake_weights"),
                    threshold=float(model_cfg.get("threshold", 0.5)),
                    image_size=int(model_cfg.get("image_size", 224)),
                    notes=_CFG.get("notes"),
                )
            else:
                result = run_detect(
                    tmp_path,
                    ensemble=None,
                    probe_weights=_WEIGHTS_PATH,
                    model=_MODEL,
                    threshold=float(model_cfg.get("threshold", 0.5)),
                    image_size=int(model_cfg.get("image_size", 224)),
                    notes=_CFG.get("notes"),
                    model_id=_MODEL_ID,
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    payload = result.to_dict()
    missing = [k for k in SCHEMA_KEYS if k not in payload]
    if missing:
        raise HTTPException(status_code=500, detail=f"schema missing keys: {missing}")
    return JSONResponse(payload)


def main() -> None:
    import uvicorn

    cfg = _load_cfg()
    api = cfg.get("api", {})
    uvicorn.run(
        "deepfake_detection.api.app:app",
        host=api.get("host", "0.0.0.0"),
        port=int(api.get("port", 8000)),
        reload=False,
    )


if __name__ == "__main__":
    main()
