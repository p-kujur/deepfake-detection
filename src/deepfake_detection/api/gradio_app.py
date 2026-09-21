"""Gradio demo: same JSON schema as CLI / FastAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deepfake_detection.infer.pipeline import run_detect
from deepfake_detection.infer.resolve import product_notes, resolve_default_probe_weights
from deepfake_detection.models.unifd import load_detector


def _cfg() -> dict[str, Any]:
    p = Path("configs/default.yaml")
    if p.is_file():
        with p.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_demo():
    import gradio as gr

    cfg = _cfg()
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("infer", {})
    cfg_w = model_cfg.get("probe_weights")
    if cfg_w in (None, "", "auto"):
        cfg_w = None
    path, mid, found = resolve_default_probe_weights(cfg_w)
    model, _ = load_detector(
        probe_weights=path if found else None,
        backend=model_cfg.get("backend", "open_clip"),
        device=infer_cfg.get("device", "auto"),
        require_weights=False,
    )
    threshold = float(model_cfg.get("threshold", 0.5))
    model_id = mid if found else f"{mid}-untrained"
    default_ensemble = infer_cfg.get("ensemble") or "none"
    if default_ensemble in ("", None):
        default_ensemble = "none"

    def predict(image_path: str, ensemble: str) -> dict[str, Any]:
        if not image_path:
            return {"error": "no image"}
        mode = None if ensemble in (None, "", "none", "off") else ensemble
        if mode == "soft_router":
            return run_detect(
                image_path,
                ensemble="soft_router",
                probe_weights=None if cfg_w in (None, "", "auto") else cfg_w,
                cifake_weights=model_cfg.get("cifake_weights"),
                threshold=threshold,
                notes=cfg.get("notes"),
            ).to_dict()
        return run_detect(
            image_path,
            ensemble=None,
            probe_weights=path if found else None,
            model=model,
            threshold=threshold,
            notes=cfg.get("notes"),
            model_id=model_id,
        ).to_dict()

    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="filepath", label="Image"),
            gr.Dropdown(
                choices=["none", "soft_router"],
                value=default_ensemble if default_ensemble in ("none", "soft_router") else "none",
                label="Ensemble",
                info="soft_router = ProGAN+CIFAKE dual probe (does not beat Hemg ~0.78)",
            ),
        ],
        outputs=gr.JSON(label="Detection {p_fake, label, model_id, notes, version}"),
        title="Deepfake / AIGC Detection (M4 demo)",
        description=(
            "Default single probe: ProGAN → UniFD → CIFAKE. "
            + product_notes()
        ),
    )


def main() -> None:
    cfg = _cfg()
    g = cfg.get("gradio", {})
    demo = build_demo()
    demo.launch(
        server_port=int(g.get("server_port", 7860)),
        share=bool(g.get("share", False)),
    )


if __name__ == "__main__":
    main()
