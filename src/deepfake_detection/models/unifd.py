"""UniFD-inspired detector: frozen CLIP ViT-L/14 + trainable linear probe.

Reference: Ojha et al., CVPR 2023 — UniversalFakeDetect
  Paper: https://arxiv.org/abs/2302.10174
  Code:  https://github.com/WisconsinAIVision/UniversalFakeDetect (MIT)

This wrapper loads CLIP via open_clip (preferred) or transformers, freezes the
backbone, and applies a single linear layer over image embeddings. Optional
pretrained probe weights can be downloaded with ``scripts/download_weights.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from deepfake_detection import __model_id__, __version__
from deepfake_detection.data.transforms import build_clip_preprocess, load_image, tensor_batch

logger = logging.getLogger(__name__)

DEFAULT_NOTES = (
    "UniFD-style frozen CLIP ViT-L/14 + linear probe for AIGC still images. "
    "Not a guarantee of authenticity; face-video is out of scope for v1. "
    "Calibration and cross-generator robustness vary by domain."
)


@dataclass
class DetectResult:
    p_fake: float
    label: str
    model_id: str
    notes: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_fake": self.p_fake,
            "label": self.label,
            "model_id": self.model_id,
            "notes": self.notes,
            "version": self.version,
        }


class UniFDClipLinear(nn.Module):
    """Frozen CLIP image encoder + linear binary probe (fake vs real)."""

    def __init__(
        self,
        backbone: str = "ViT-L-14",
        pretrained: str = "openai",
        backend: str = "open_clip",
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.backend = backend
        self._encode_image = None
        self.visual = None
        self.embed_dim = embed_dim or 768

        if backend == "open_clip":
            self._init_open_clip(backbone, pretrained)
        elif backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.head = nn.Linear(self.embed_dim, 1)
        self._freeze_backbone()

    def _init_open_clip(self, backbone: str, pretrained: str) -> None:
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip is required for backend='open_clip'. "
                "Install with: pip install open-clip-torch"
            ) from e

        # OpenAI CLIP checkpoints use QuickGELU; mismatch silently hurts UniFD probes.
        force_qg = str(pretrained).lower() in {"openai", "openai_clip"}
        model, _, _ = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained, force_quick_gelu=force_qg
        )
        self.visual = model.visual
        # open_clip ViT-L/14 openai embed dim is 768
        with torch.no_grad():
            # Infer embed dim from projection if present
            if hasattr(self.visual, "output_dim"):
                self.embed_dim = int(self.visual.output_dim)
            elif hasattr(model, "visual") and hasattr(model, "text_projection"):
                # image features match text proj rows
                self.embed_dim = int(model.text_projection.shape[1])
            else:
                self.embed_dim = 768

        def _encode(images: torch.Tensor) -> torch.Tensor:
            feats = self.visual(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            return feats

        self._encode_image = _encode

    def _init_transformers(self) -> None:
        try:
            from transformers import CLIPModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for backend='transformers'. "
                "Install with: pip install transformers"
            ) from e

        # OpenAI CLIP ViT-L/14
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.visual = model.vision_model
        self.visual_proj = model.visual_projection
        self.embed_dim = int(model.config.projection_dim)

        def _encode(images: torch.Tensor) -> torch.Tensor:
            out = self.visual(pixel_values=images)
            pooled = out.pooler_output
            return self.visual_proj(pooled)

        self._encode_image = _encode

    def _freeze_backbone(self) -> None:
        if self.visual is not None:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()
        if hasattr(self, "visual_proj"):
            for p in self.visual_proj.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (N, 1).

        Matches UniFD CLIPModel: encode_image → Linear, **without** L2-norm
        before the probe (Ojha et al. CVPR 2023).
        """
        with torch.no_grad():
            feats = self._encode_image(images)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        return self.head(feats.float())

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images)
        return torch.sigmoid(logits).squeeze(-1)

    def load_probe_weights(self, path: str | Path, map_location: str | None = None) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Probe weights not found: {path}. "
                "Run: python scripts/download_weights.py"
            )
        state = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Accept either full module or head-only checkpoints
        if isinstance(state, dict):
            head_state = {}
            for k, v in state.items():
                if k.startswith("head."):
                    head_state[k[len("head.") :]] = v
                elif k in ("weight", "bias"):
                    head_state[k] = v
                elif "fc" in k or "probe" in k or "classifier" in k:
                    # best-effort remap
                    nk = k.split(".")[-1]
                    if nk in ("weight", "bias"):
                        head_state[nk] = v
            if head_state:
                self.head.load_state_dict(head_state, strict=False)
                return
            # try direct head load
            try:
                self.head.load_state_dict(state, strict=False)
                return
            except Exception:
                pass
        raise RuntimeError(
            f"Could not interpret probe checkpoint format at {path}. "
            "Expected a Linear state_dict with 'weight'/'bias' or 'head.*' keys."
        )


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_detector(
    probe_weights: str | Path | None = None,
    backend: str = "open_clip",
    backbone: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "auto",
    require_weights: bool = False,
) -> tuple[UniFDClipLinear, torch.device]:
    """Build detector; optionally load probe weights.

    If ``require_weights`` and path missing → clear FileNotFoundError.
    If path missing and not required → random-init head (documented as untrained).
    """
    dev = resolve_device(device)
    model = UniFDClipLinear(backbone=backbone, pretrained=pretrained, backend=backend)
    model = model.to(dev)
    model.eval()

    if probe_weights:
        p = Path(probe_weights)
        if p.is_file():
            model.load_probe_weights(p, map_location=str(dev))
            logger.info("Loaded probe weights from %s", p)
        elif require_weights:
            raise FileNotFoundError(
                f"Required probe weights missing: {p}. "
                "Run: python scripts/download_weights.py"
            )
        else:
            logger.warning(
                "Probe weights not found at %s — using random head "
                "(predictions are NOT meaningful until weights are loaded).",
                p,
            )
    return model, dev


def detect_image(
    image_path: str | Path,
    model: UniFDClipLinear | None = None,
    device: str = "auto",
    threshold: float = 0.5,
    image_size: int = 224,
    probe_weights: str | Path | None = None,
    backend: str = "open_clip",
    notes: str | None = None,
) -> DetectResult:
    """Run detection on a single image path; return structured result."""
    if model is None:
        model, dev = load_detector(
            probe_weights=probe_weights, backend=backend, device=device
        )
    else:
        dev = next(model.parameters()).device

    preprocess = build_clip_preprocess(image_size)
    img = load_image(image_path)
    batch = tensor_batch(img, preprocess, device=str(dev))

    with torch.inference_mode():
        p_fake = float(model.predict_proba(batch)[0].item())

    label = "fake" if p_fake >= threshold else "real"
    return DetectResult(
        p_fake=round(p_fake, 6),
        label=label,
        model_id=__model_id__,
        notes=notes or DEFAULT_NOTES,
        version=__version__,
    )
