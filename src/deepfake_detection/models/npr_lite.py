"""NPR-lite second head: neighboring-pixel residual stem + tiny CNN.

Full NPR (Tan et al., CVPR 2024 — chuangchuangtan/NPR-DeepfakeDetection) pulls in
a custom ResNet-50 training stack and large pretrained weights. For M3 we ship a
**lightweight NPR-inspired** residual CNN that:

1. Forms ``x - avg_pool(x)`` (local upsampling / neighbor artifact cue).
2. Classifies with a small conv tower (~100k params) trainable on CIFAKE subset.

See ``docs/npr_integration.md`` for the full-NPR integration path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

NPR_LITE_MODEL_ID = "npr-lite-cnn-v1"


class NPRLiteCNN(nn.Module):
    """Tiny CNN over neighboring-pixel residuals (fake vs real logits)."""

    def __init__(self, in_ch: int = 3, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, 1)
        self.model_id = NPR_LITE_MODEL_ID

    @staticmethod
    def neighboring_pixel_residual(x: torch.Tensor) -> torch.Tensor:
        """NPR-inspired residual: center minus 3×3 local average."""
        avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - avg

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits ``(N, 1)``."""
        r = self.neighboring_pixel_residual(images)
        feats = self.features(r).flatten(1)
        return self.head(feats)

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images)).squeeze(-1)


def build_cnn_preprocess(image_size: int = 128):
    """Simple ImageNet-ish preprocess for the NPR-lite CNN (not CLIP norms)."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_npr_lite(
    weights: str | Path | None = None,
    device: str | torch.device = "cpu",
    width: int = 32,
) -> NPRLiteCNN:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model = NPRLiteCNN(width=width).to(dev)
    model.eval()
    if weights:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"NPR-lite weights not found: {path}")
        blob = torch.load(path, map_location=dev, weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        model.load_state_dict(state, strict=True)
        logger.info("Loaded NPR-lite weights from %s", path)
    return model


def save_npr_lite(model: NPRLiteCNN, path: Path, meta: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}, "model_id": model.model_id}, path)
    logger.info("Saved NPR-lite → %s", path)
    return path
