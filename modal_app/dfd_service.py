"""FastAPI service for the deepfake detector (no Modal dependency).

Used by modal_app/app.py on Modal, and runnable locally with uvicorn:

    PYTHONPATH=src:modal_app uvicorn --factory dfd_service:create_local_app --port 8000

Endpoints:
    GET  /health
    POST /detect  multipart: file=<image>, ensemble=none|progan|soft_router (optional)
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
ENSEMBLES = ("none", "progan", "soft_router")


def _weights_dir() -> Path:
    env = os.environ.get("DFD_WEIGHTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "weights"


class Detector:
    """Frozen CLIP ViT-L/14 loaded once + linear heads applied to the same features.

    ensemble=none        -> default head (modern M5 probe if present, else ProGAN)
    ensemble=progan      -> legacy ProGAN-only probe (M3b)
    ensemble=soft_router -> ProGAN + CIFAKE dual probe (M3c)
    """

    def __init__(self, weights_dir: str | Path | None = None, device: str = "cpu") -> None:
        from deepfake_detection.infer.pipeline import _load_linear_head
        from deepfake_detection.infer.resolve import model_id_for_weights
        from deepfake_detection.models.unifd import load_detector

        wd = Path(weights_dir) if weights_dir else _weights_dir()
        self.modern = wd / "modern_clip_vit_l14_linear.pth"
        self.progan = wd / "progan_clip_vit_l14_linear.pth"
        self.cifake = wd / "cifake_clip_vit_l14_linear.pth"
        self.model, self.device = load_detector(
            probe_weights=None, backend="open_clip", device=device, require_weights=False
        )
        self.heads: dict[str, Any] = {}
        for key, path in (("modern", self.modern), ("progan", self.progan), ("cifake", self.cifake)):
            if path.is_file():
                self.heads[key] = _load_linear_head(path, embed_dim=int(self.model.embed_dim)).to(self.device)
        self.default_key = "modern" if "modern" in self.heads else "progan"
        default_path = self.modern if self.default_key == "modern" else self.progan
        mid = model_id_for_weights(default_path)
        self.model_id = mid if self.default_key in self.heads else f"{mid}-untrained"
        self.has_progan = "progan" in self.heads
        self.has_router = self.has_progan and "cifake" in self.heads

    def ensembles(self) -> list[str]:
        out = ["none"]
        if self.has_progan and self.default_key != "progan":
            out.append("progan")
        if self.has_router:
            out.append("soft_router")
        return out

    def predict(self, image_path: str | Path, ensemble: str | None = None) -> dict[str, Any]:
        import numpy as np
        import torch

        from deepfake_detection import __version__
        from deepfake_detection.data.transforms import build_clip_preprocess, load_image, tensor_batch
        from deepfake_detection.eval.dual_probe import soft_confidence_mix
        from deepfake_detection.infer.resolve import ENSEMBLE_MODEL_ID, model_id_for_weights, product_notes

        mode = "none" if ensemble in (None, "", "none", "off") else ensemble
        img = load_image(image_path)
        batch = tensor_batch(img, build_clip_preprocess(224), device=str(self.device))
        with torch.inference_mode():
            feats = self.model._encode_image(batch)
            if feats.ndim > 2:
                feats = feats.flatten(1)
            feats = feats.float()

            def logit(key: str) -> float:
                return float(self.heads[key](feats).squeeze().cpu())

            if mode == "soft_router":
                p = float(np.asarray(soft_confidence_mix(np.array([logit("progan")]),
                                                         np.array([logit("cifake")]))).reshape(-1)[0])
                mid = ENSEMBLE_MODEL_ID
            elif mode == "progan":
                p = float(torch.sigmoid(torch.tensor(logit("progan"))))
                mid = model_id_for_weights(self.progan)
            elif self.default_key in self.heads:
                p = float(torch.sigmoid(torch.tensor(logit(self.default_key))))
                mid = self.model_id
            else:  # no weights at all: untrained head (kept for parity with v1 behaviour)
                p = float(torch.sigmoid(self.model.head(feats)).squeeze().cpu())
                mid = self.model_id
        return {
            "p_fake": round(p, 6),
            "label": "fake" if p >= THRESHOLD else "real",
            "model_id": mid,
            "notes": product_notes(),
            "version": __version__,
        }


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
            "ensembles": detector.ensembles(),
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
        if ensemble not in detector.ensembles():
            raise HTTPException(400, f"{ensemble} is not available on this server")
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
