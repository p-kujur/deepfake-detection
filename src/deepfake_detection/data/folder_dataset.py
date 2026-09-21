"""Generic real/fake folder dataset for cross-generator eval packs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


class RealFakeFolderDataset(Dataset):
    """Load images from ``real/`` and ``fake/`` subfolders (label 0/1).

    Also accepts alternate names: ``0_real`` / ``1_fake``, ``REAL`` / ``FAKE``.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], Any] | None = None,
        max_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        real_dir, fake_dir = self._resolve_dirs(self.root)
        real_paths = _list_images(real_dir)
        fake_paths = _list_images(fake_dir)
        if max_per_class is not None:
            import random

            rng = random.Random(seed)
            if len(real_paths) > max_per_class:
                real_paths = rng.sample(real_paths, max_per_class)
            if len(fake_paths) > max_per_class:
                fake_paths = rng.sample(fake_paths, max_per_class)
            real_paths = sorted(real_paths)
            fake_paths = sorted(fake_paths)

        self.samples: list[tuple[Path, int]] = [(p, 0) for p in real_paths] + [
            (p, 1) for p in fake_paths
        ]
        if not self.samples:
            raise FileNotFoundError(
                f"No images under {self.root} (expected real/ + fake/ subfolders). "
                "Run: python scripts/download_crossgen.py"
            )
        logger.info(
            "Folder dataset %s: n=%d (real=%d fake=%d)",
            self.root,
            len(self.samples),
            len(real_paths),
            len(fake_paths),
        )

    @staticmethod
    def _resolve_dirs(root: Path) -> tuple[Path, Path]:
        candidates: Sequence[tuple[str, str]] = (
            ("real", "fake"),
            ("REAL", "FAKE"),
            ("0_real", "1_fake"),
            ("RealArt", "AiArtData"),
        )
        for r, f in candidates:
            rp, fp = root / r, root / f
            if rp.is_dir() and fp.is_dir():
                return rp, fp
        # nested Hemg-style RealArt/RealArt
        for r, f in candidates:
            rp, fp = root / r / r, root / f / f
            if rp.is_dir() and fp.is_dir():
                return rp, fp
        raise FileNotFoundError(
            f"Could not find real/fake dirs under {root}. Tried {list(candidates)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, y = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y
