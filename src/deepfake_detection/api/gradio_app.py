"""Thin Gradio demo stub for local AIGC still-image detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deepfake_detection.models.unifd import detect_image, load_detector


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
    model, _ = load_detector(
        probe_weights=model_cfg.get("probe_weights"),
        backend=model_cfg.get("backend", "open_clip"),
        device=infer_cfg.get("device", "auto"),
        require_weights=False,
    )
    threshold = float(model_cfg.get("threshold", 0.5))

    def predict(image_path: str) -> dict[str, Any]:
        if not image_path:
            return {"error": "no image"}
        return detect_image(
            image_path,
            model=model,
            threshold=threshold,
            notes=cfg.get("notes"),
        ).to_dict()

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Image"),
        outputs=gr.JSON(label="Detection"),
        title="Deepfake / AIGC Detection (M1 stub)",
        description=(
            "UniFD-style CLIP ViT-L/14 + linear probe. "
            "Scores are experimental — not authenticity guarantees."
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
