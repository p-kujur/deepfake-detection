"""Image loading and CLIP-compatible preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_image(path: str | Path) -> Image.Image:
    """Load an RGB PIL image from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def build_clip_preprocess(image_size: int = 224) -> Callable[[Image.Image], "torch.Tensor"]:
    """Return a torchvision Compose matching OpenAI CLIP ViT-L/14 norms."""
    import torch
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def tensor_batch(image: Image.Image, preprocess: Callable, device: str = "cpu"):
    """Preprocess a single PIL image to a batched tensor on ``device``."""
    import torch

    t = preprocess(image)
    if not isinstance(t, torch.Tensor):
        raise TypeError("preprocess must return a torch.Tensor")
    return t.unsqueeze(0).to(device)
