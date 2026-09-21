"""Shared single-image inference for CLI / FastAPI / Gradio (M4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepfake_detection import __version__
from deepfake_detection.data.transforms import build_clip_preprocess, load_image, tensor_batch
from deepfake_detection.eval.dual_probe import soft_confidence_mix
from deepfake_detection.infer.resolve import (
    ENSEMBLE_MODEL_ID,
    product_notes,
    resolve_default_probe_weights,
    resolve_ensemble_weights,
)
from deepfake_detection.models.unifd import DetectResult, UniFDClipLinear, load_detector


def _final_notes(extra: str | None) -> str:
    """Always include product caveats; avoid duplicating config notes."""
    base = product_notes()
    if not extra or not str(extra).strip():
        return base
    ex = str(extra).strip()
    if "0.78" in ex and ("court-grade" in ex.lower() or "not court-grade" in ex.lower()):
        return ex  # config already carries required caveats
    if ex in base:
        return base
    return product_notes(ex)



def _load_linear_head(path: Path, embed_dim: int = 768) -> torch.nn.Linear:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    head = torch.nn.Linear(embed_dim, 1)
    # Accept head.weight / weight keys
    if isinstance(blob, dict):
        w = blob.get("weight", blob.get("head.weight"))
        b = blob.get("bias", blob.get("head.bias"))
        if w is None or b is None:
            raise RuntimeError(f"Cannot read Linear weight/bias from {path}")
        head.weight.data.copy_(w)
        head.bias.data.copy_(b)
    else:
        raise RuntimeError(f"Unexpected checkpoint type at {path}")
    head.eval()
    return head


def detect_single(
    image_path: str | Path,
    *,
    probe_weights: str | Path | None = None,
    backend: str = "open_clip",
    device: str = "auto",
    threshold: float = 0.5,
    image_size: int = 224,
    notes: str | None = None,
    model: UniFDClipLinear | None = None,
    model_id: str | None = None,
    require_weights: bool = False,
) -> DetectResult:
    """Single-probe detection with auto weight resolution."""
    path, mid, found = resolve_default_probe_weights(probe_weights)
    if require_weights and not found:
        raise FileNotFoundError(
            f"Required probe weights missing: {path}. "
            "Place ProGAN/UniFD/CIFAKE heads under weights/ or pass --weights."
        )
    if model is None:
        model, _ = load_detector(
            probe_weights=path if found else None,
            backend=backend,
            device=device,
            require_weights=require_weights and found,
        )
        if found and path is not None:
            # ensure loaded (load_detector already does if path exists)
            pass
    use_id = model_id or (mid if found else f"{mid}-untrained")
    from deepfake_detection.models.unifd import detect_image

    result = detect_image(
        image_path,
        model=model,
        threshold=threshold,
        image_size=image_size,
        notes=_final_notes(notes),
        model_id=use_id,
    )
    return result


def detect_soft_router(
    image_path: str | Path,
    *,
    progan_weights: str | Path | None = None,
    cifake_weights: str | Path | None = None,
    backend: str = "open_clip",
    device: str = "auto",
    threshold: float = 0.5,
    image_size: int = 224,
    notes: str | None = None,
    model: UniFDClipLinear | None = None,
) -> DetectResult:
    """Dual-probe soft_router (softmax over |logit| → mix probs)."""
    pa, pb = resolve_ensemble_weights(progan_weights, cifake_weights)
    if model is None:
        model, dev = load_detector(
            probe_weights=pa, backend=backend, device=device, require_weights=True
        )
    else:
        dev = next(model.parameters()).device

    head_b = _load_linear_head(pb, embed_dim=int(model.embed_dim)).to(dev)

    preprocess = build_clip_preprocess(image_size)
    img = load_image(image_path)
    batch = tensor_batch(img, preprocess, device=str(dev))

    with torch.inference_mode():
        feats = model._encode_image(batch)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        feats = feats.float()
        la = model.head(feats).squeeze(-1).detach().cpu().numpy()
        lb = head_b(feats).squeeze(-1).detach().cpu().numpy()
        # soft_confidence_mix expects arrays; scalar → 0-d
        p = float(np.asarray(soft_confidence_mix(la, lb)).reshape(-1)[0])

    label = "fake" if p >= threshold else "real"
    return DetectResult(
        p_fake=round(p, 6),
        label=label,
        model_id=ENSEMBLE_MODEL_ID,
        notes=_final_notes(notes),
        version=__version__,
    )


def run_detect(
    image_path: str | Path,
    *,
    ensemble: str | None = None,
    probe_weights: str | Path | None = None,
    cifake_weights: str | Path | None = None,
    **kwargs: Any,
) -> DetectResult:
    """Dispatch single vs soft_router ensemble."""
    if ensemble in (None, "", "none", "off"):
        return detect_single(image_path, probe_weights=probe_weights, **kwargs)
    if ensemble == "soft_router":
        allowed = {"backend", "device", "threshold", "image_size", "notes", "model"}
        sr_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return detect_soft_router(
            image_path,
            progan_weights=probe_weights,
            cifake_weights=cifake_weights,
            **sr_kwargs,
        )
    raise ValueError(f"Unknown ensemble mode: {ensemble!r} (supported: soft_router)")
