"""Hugging Face Space: AIGC still-image deepfake detector (CLIP ViT-L/14 + linear probe).

Same JSON schema as the repo CLI / FastAPI / Gradio demo:
    {p_fake, label, model_id, notes, version}

API (for the Vercel frontend / @gradio/client):
    client.predict("/predict", {image: <file>, ensemble: "none" | "soft_router"})
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# Vendored package lives in ./src; weights resolve relative to CWD (weights/...).
sys.path.insert(0, str(HERE / "src"))
os.chdir(HERE)

import gradio as gr  # noqa: E402

from deepfake_detection.infer.pipeline import run_detect  # noqa: E402
from deepfake_detection.infer.resolve import (  # noqa: E402
    CIFAKE_WEIGHTS,
    PROGAN_WEIGHTS,
    product_notes,
    resolve_default_probe_weights,
)
from deepfake_detection.models.unifd import load_detector  # noqa: E402

THRESHOLD = 0.5

# Load the frozen CLIP backbone + ProGAN probe once at startup (CPU).
_PATH, _MODEL_ID, _FOUND = resolve_default_probe_weights(str(PROGAN_WEIGHTS))
_MODEL, _ = load_detector(
    probe_weights=_PATH if _FOUND else None,
    backend="open_clip",
    device="cpu",
    require_weights=False,
)
_MODEL_ID = _MODEL_ID if _FOUND else f"{_MODEL_ID}-untrained"
_HAS_ROUTER = PROGAN_WEIGHTS.is_file() and CIFAKE_WEIGHTS.is_file()
ENSEMBLES = ["none", "soft_router"] if _HAS_ROUTER else ["none"]


def predict(image: str | None, ensemble: str = "none") -> dict[str, Any]:
    """Score one image. Returns {p_fake, label, model_id, notes, version}."""
    if not image:
        return {"error": "no image"}
    mode = None if ensemble in (None, "", "none", "off") else ensemble
    if mode == "soft_router":
        if not _HAS_ROUTER:
            return {"error": "soft_router needs ProGAN + CIFAKE weights (not bundled)"}
        return run_detect(
            image,
            ensemble="soft_router",
            probe_weights=str(PROGAN_WEIGHTS),
            cifake_weights=str(CIFAKE_WEIGHTS),
            model=_MODEL,  # reuse loaded CLIP; only the CIFAKE head is added
            threshold=THRESHOLD,
        ).to_dict()
    return run_detect(
        image,
        ensemble=None,
        probe_weights=str(_PATH) if _FOUND else None,
        model=_MODEL,
        threshold=THRESHOLD,
        model_id=_MODEL_ID,
    ).to_dict()


with gr.Blocks(title="Deepfake / AIGC still-image detector") as demo:
    gr.Markdown(
        "# Deepfake / AIGC still-image detector\n"
        "Frozen OpenAI CLIP ViT-L/14 + linear probe (default: ProGAN-trained). "
        "Returns `{p_fake, label, model_id, notes, version}`.\n\n"
        f"> {product_notes()}"
    )
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="filepath", label="Image")
            ens = gr.Dropdown(
                choices=ENSEMBLES,
                value="none",
                label="Ensemble",
                info="soft_router = ProGAN + CIFAKE dual probe (does not beat Hemg ~0.78)",
            )
            btn = gr.Button("Detect", variant="primary")
        out = gr.JSON(label="Detection")
    btn.click(predict, inputs=[img, ens], outputs=out, api_name="predict")

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()
