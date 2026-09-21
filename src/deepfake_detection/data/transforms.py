"""Image loading and CLIP-compatible preprocessing (+ optional train augs)."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_image(path: str | Path) -> Image.Image:
    """Load an RGB PIL image from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG at ``quality`` and reload (simulates social upload)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


class TrainAugment:
    """Optional Blur / JPEG augmentations applied before CLIP preprocess.

    Probabilities are independent. Disabled when both probs are 0.
    """

    def __init__(
        self,
        jpeg_prob: float = 0.5,
        jpeg_quality_range: tuple[int, int] = (30, 100),
        blur_prob: float = 0.1,
        blur_sigma_range: tuple[float, float] = (0.0, 3.0),
    ) -> None:
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.blur_prob > 0 and random.random() < self.blur_prob:
            lo, hi = self.blur_sigma_range
            radius = random.uniform(lo, hi)
            if radius > 0:
                img = gaussian_blur(img, radius)
        if self.jpeg_prob > 0 and random.random() < self.jpeg_prob:
            q_lo, q_hi = self.jpeg_quality_range
            quality = random.randint(int(q_lo), int(q_hi))
            img = jpeg_compress(img, quality)
        return img


def build_clip_preprocess(image_size: int = 224) -> Callable[[Image.Image], "torch.Tensor"]:
    """Return a torchvision Compose matching OpenAI CLIP ViT-L/14 norms."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_train_preprocess(
    image_size: int = 224,
    jpeg_prob: float = 0.5,
    blur_prob: float = 0.1,
) -> Callable[[Image.Image], "torch.Tensor"]:
    """CLIP preprocess with optional Blur/JPEG aug (UniFD-style robustness)."""
    from torchvision import transforms

    aug = TrainAugment(jpeg_prob=jpeg_prob, blur_prob=blur_prob)
    clip = build_clip_preprocess(image_size)

    def _fn(img: Image.Image):
        return clip(aug(img))

    return _fn


def tensor_batch(image: Image.Image, preprocess: Callable, device: str = "cpu"):
    """Preprocess a single PIL image to a batched tensor on ``device``."""
    import torch

    t = preprocess(image)
    if not isinstance(t, torch.Tensor):
        raise TypeError("preprocess must return a torch.Tensor")
    return t.unsqueeze(0).to(device)


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    """Resize so the shorter side equals ``short`` (aspect preserved)."""
    w, h = img.size
    if min(w, h) == short:
        return img
    if w <= h:
        nw, nh = short, int(round(h * short / w))
    else:
        nw, nh = int(round(w * short / h)), short
    return img.resize((max(nw, 1), max(nh, 1)), Image.Resampling.BICUBIC)


def build_eval_preprocess(
    image_size: int = 224,
    jpeg_quality: int | None = None,
    resize_short: int | None = None,
    clip_norm: bool = True,
):
    """CLIP (or CNN) preprocess with optional JPEG / downscale robustness augs.

    Robustness protocol (plan):
      - JPEG q=70: re-encode then restore to model size
      - resize: short side → 128 (or ``resize_short``), then model native size
    """
    from torchvision import transforms

    if clip_norm:
        normalize = transforms.Normalize(CLIP_MEAN, CLIP_STD)
    else:
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    geo = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    def _fn(img: Image.Image):
        if resize_short is not None:
            img = resize_short_side(img, int(resize_short))
        if jpeg_quality is not None:
            img = jpeg_compress(img, int(jpeg_quality))
        return geo(img)

    return _fn
