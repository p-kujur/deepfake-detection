"""FastAPI app: POST /detect (multipart image)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from deepfake_detection import __model_id__, __version__
from deepfake_detection.models.unifd import detect_image, load_detector

app = FastAPI(
    title="Deepfake Detection API",
    version=__version__,
    description="AIGC still-image detector (UniFD-style CLIP ViT-L/14 + linear probe).",
)

_MODEL = None
_DEVICE = None
_CFG: dict[str, Any] = {}


def _load_cfg() -> dict[str, Any]:
    path = Path("configs/default.yaml")
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@app.on_event("startup")
def startup() -> None:
    global _MODEL, _DEVICE, _CFG
    _CFG = _load_cfg()
    model_cfg = _CFG.get("model", {})
    infer_cfg = _CFG.get("infer", {})
    try:
        _MODEL, _DEVICE = load_detector(
            probe_weights=model_cfg.get("probe_weights"),
            backend=model_cfg.get("backend", "open_clip"),
            backbone=model_cfg.get("backbone", "ViT-L-14"),
            pretrained=model_cfg.get("pretrained", "openai"),
            device=infer_cfg.get("device", "auto"),
            require_weights=False,
        )
    except Exception as e:
        # Keep API up; /detect will surface errors
        _MODEL = None
        _DEVICE = None
        app.state.startup_error = str(e)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "model_id": __model_id__,
        "version": __version__,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    if _MODEL is None:
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
            result = detect_image(
                tmp_path,
                model=_MODEL,
                threshold=float(model_cfg.get("threshold", 0.5)),
                image_size=int(model_cfg.get("image_size", 224)),
                notes=_CFG.get("notes"),
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(result.to_dict())


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
