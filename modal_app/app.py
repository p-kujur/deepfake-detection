"""Modal deployment: deepfake detector API (CPU).

Deploy (from the repo root, with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set):

    modal deploy modal_app/app.py

Serves FastAPI at https://<workspace>--deepfake-detector-detector-web.modal.run
    GET /health, POST /detect (multipart file + optional ensemble=none|soft_router)
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Make src/deepfake_detection and dfd_service importable at deploy time so
# add_local_python_source can locate them.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

REMOTE_WEIGHTS = "/root/weights"


def _bake_clip() -> None:
    """Download CLIP ViT-L/14 (openai) into the image's HF cache at build time."""
    from deepfake_detection.models.unifd import load_detector

    load_detector(probe_weights=None, backend="open_clip", device="cpu")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "open-clip-torch>=2.24",
        "numpy>=1.24",
        "Pillow>=10.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.3",
        "tqdm>=4.66",
        "fastapi[standard]>=0.110",
        "python-multipart>=0.0.9",
    )
    .env({"DFD_WEIGHTS_DIR": REMOTE_WEIGHTS, "HF_HUB_DISABLE_TELEMETRY": "1"})
    # copy=True so the package is available to the build-time CLIP download step
    .add_local_python_source("deepfake_detection", copy=True)
    .run_function(_bake_clip)
    .add_local_file(ROOT / "weights" / "progan_clip_vit_l14_linear.pth",
                    f"{REMOTE_WEIGHTS}/progan_clip_vit_l14_linear.pth")
    .add_local_file(ROOT / "weights" / "cifake_clip_vit_l14_linear.pth",
                    f"{REMOTE_WEIGHTS}/cifake_clip_vit_l14_linear.pth")
    .add_local_python_source("dfd_service")
)

app = modal.App("deepfake-detector", image=image)


@app.cls(
    cpu=2,
    memory=4096,
    scaledown_window=300,  # stay warm 5 min after the last request
    max_containers=2,      # protect free credits
    timeout=120,
)
@modal.concurrent(max_inputs=4)
class Detector:
    @modal.enter()
    def load(self) -> None:
        from dfd_service import Detector as _Detector

        self.detector = _Detector(REMOTE_WEIGHTS, device="cpu")

    @modal.asgi_app()
    def web(self):
        from dfd_service import create_app

        return create_app(self.detector)
