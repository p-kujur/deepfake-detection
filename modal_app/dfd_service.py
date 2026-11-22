"""FastAPI service for the deepfake detector (no Modal dependency).

Used by modal_app/app.py on Modal, and runnable locally with uvicorn:

    PYTHONPATH=src:modal_app uvicorn --factory dfd_service:create_local_app --port 8000

Endpoints:
    GET  /health
    POST /detect  multipart: file=<image>, ensemble=none|soft_router (optional)
         -> {p_fake, label, model_id, notes, version}
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
THRESHOLD = 0.5
ENSEMBLES = ("none", "soft_router")


def _weights_dir() -> Path:
    env = os.environ.get("DFD_WEIGHTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "weights"


class Detector:
    """Frozen CLIP ViT-L/14 + ProGAN linear probe; CIFAKE head enables soft_router."""

    def __init__(self, weights_dir: str | Path | None = None, device: str = "cpu") -> None:
        from deepfake_detection.infer.resolve import model_id_for_weights
        from deepfake_detection.models.unifd import load_detector

        wd = Path(weights_dir) if weights_dir else _weights_dir()
        self.progan = wd / "progan_clip_vit_l14_linear.pth"
        self.cifake = wd / "cifake_clip_vit_l14_linear.pth"
        found = self.progan.is_file()
        self.model, _ = load_detector(
            probe_weights=self.progan if found else None,
            backend="open_clip",
            device=device,
            require_weights=False,
        )
        mid = model_id_for_weights(self.progan)
        self.model_id = mid if found else f"{mid}-untrained"
        self.has_router = found and self.cifake.is_file()

    def predict(self, image_path: str | Path, ensemble: str | None = None) -> dict[str, Any]:
        from deepfake_detection.infer.pipeline import run_detect

        mode = None if ensemble in (None, "", "none", "off") else ensemble
        if mode == "soft_router":
            return run_detect(
                image_path,
                ensemble="soft_router",
                probe_weights=self.progan,
                cifake_weights=self.cifake,
                model=self.model,  # reuse loaded CLIP; only the CIFAKE head is added
                threshold=THRESHOLD,
            ).to_dict()
        return run_detect(
            image_path,
            ensemble=None,
            probe_weights=self.progan,
            model=self.model,
            threshold=THRESHOLD,
            model_id=self.model_id,
        ).to_dict()


def create_app(detector: Detector) -> FastAPI:
    from deepfake_detection import __version__
    from deepfake_detection.infer.resolve import product_notes

    api = FastAPI(
        title="Deepfake Detection API",
        version=__version__,
        description=(
            "AIGC still-image detector (CLIP ViT-L/14 + linear probe). " + product_notes()
        ),
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": detector.model_id,
            "ensembles": list(ENSEMBLES) if detector.has_router else ["none"],
            "version": __version__,
        }

    @api.post("/detect")
    async def detect(
        file: UploadFile = File(...),
        ensemble: str = Form("none"),
    ) -> dict[str, Any]:
        ensemble = (ensemble or "none").strip()
        if ensemble not in ENSEMBLES:
            raise HTTPException(400, f"ensemble must be one of {ENSEMBLES}")
        if ensemble == "soft_router" and not detector.has_router:
            raise HTTPException(400, "soft_router needs ProGAN + CIFAKE weights")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image too large (max 10 MB)")
        if not data:
            raise HTTPException(400, "Empty file")
        suffix = Path(file.filename or "upload").suffix or ".img"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            return await run_in_threadpool(detector.predict, path, ensemble)
        except HTTPException:
            raise
        except Exception as e:  # unreadable image etc.
            raise HTTPException(400, "Could not read that file as an image") from e
        finally:
            os.unlink(path)

    return api


def create_local_app():
    """uvicorn --factory entry point for local testing."""
    return create_app(Detector())
