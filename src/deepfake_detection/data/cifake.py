"""CIFAKE dataset loader (Bird & Lotfi, IEEE Access 2024).

Official corpus: 60k real (CIFAR-10) + 60k AI-generated 32×32 images,
100k train / 20k test.

Primary mirror used here:
  https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset
  (train=100000, test=20000; labels FAKE/REAL)

Upstream:
  https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images
  Paper: https://arxiv.org/abs/2303.14126
  Kaggle: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Label convention in this project: **1 = fake**, **0 = real**.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_HF_ID = "dragonintelligence/CIFAKE-image-dataset"
DEFAULT_CACHE_DIR = Path("datasets/cifake")


def _normalize_label(raw: Any, names: Sequence[str] | None = None) -> int:
    """Map dataset label → 1=fake, 0=real."""
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ("fake", "ai", "synthetic", "generated"):
            return 1
        if key in ("real", "authentic", "natural"):
            return 0
        raise ValueError(f"Unknown string label: {raw!r}")

    idx = int(raw)
    if names:
        name = str(names[idx]).lower()
        if "fake" in name or name in ("ai", "synthetic", "generated"):
            return 1
        if "real" in name or name in ("authentic", "natural"):
            return 0
    # Fallback: assume ClassLabel order FAKE=0, REAL=1 (dragonintelligence)
    # or fake=0, real=1 (yanbax) → invert to our convention.
    if names and len(names) == 2:
        return 1 - idx
    # If unknown, treat 1 as fake (common binary convention) — callers should
    # prefer named ClassLabel.
    return 1 if idx == 1 else 0


class CIFAKEDataset(Dataset):
    """PyTorch Dataset wrapping a HuggingFace CIFAKE split.

    Parameters
    ----------
    split:
        ``train`` or ``test`` (official holdout).
    transform:
        Callable ``PIL.Image -> Tensor`` (or any batchable type).
    hf_id:
        HuggingFace dataset id.
    cache_dir:
        Local HF / datasets cache root (also used as download target).
    max_samples:
        Optional cap for quick subset runs (evenly from the start after shuffle
        seed, via ``select``).
    seed:
        Used when ``max_samples`` is set (shuffle before take).
    """

    def __init__(
        self,
        split: str = "train",
        transform: Callable[[Image.Image], Any] | None = None,
        hf_id: str = DEFAULT_HF_ID,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        max_samples: int | None = None,
        seed: int = 42,
        hf_dataset: Any | None = None,
    ) -> None:
        if split not in ("train", "test", "validation"):
            raise ValueError(f"split must be train|test|validation, got {split!r}")
        self.split = "test" if split == "validation" else split
        self.transform = transform
        self.hf_id = hf_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = max_samples
        self.seed = seed

        if hf_dataset is not None:
            self._ds = hf_dataset
        else:
            self._ds = self._load_hf()

        self._label_names: list[str] | None = None
        feat = getattr(self._ds, "features", None)
        if feat is not None and "label" in feat:
            names = getattr(feat["label"], "names", None)
            if names:
                self._label_names = list(names)

        if max_samples is not None and max_samples < len(self._ds):
            # Deterministic subset: shuffle then take head.
            self._ds = self._ds.shuffle(seed=seed).select(range(max_samples))
            logger.info(
                "CIFAKE %s subset: %d / original (seed=%d)",
                self.split,
                max_samples,
                seed,
            )

        n_fake, n_real = self._count_labels()
        logger.info(
            "CIFAKE %s ready: n=%d (fake=%d real=%d) hf_id=%s",
            self.split,
            len(self._ds),
            n_fake,
            n_real,
            self.hf_id,
        )

    def _count_labels(self) -> tuple[int, int]:
        """Count fake/real without decoding images."""
        labels = self._ds["label"]
        n_fake = 0
        for raw in labels:
            if _normalize_label(raw, self._label_names) == 1:
                n_fake += 1
        return n_fake, len(labels) - n_fake

    def _load_hf(self) -> Any:

        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "CIFAKE loader requires the `datasets` package. "
                "Install with: pip install datasets"
            ) from e

        kwargs: dict[str, Any] = {
            "path": self.hf_id,
            "split": self.split,
        }
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self.cache_dir)

        logger.info("Loading CIFAKE from HuggingFace: %s split=%s", self.hf_id, self.split)
        return load_dataset(**kwargs)

    def _label_at(self, index: int) -> int:
        row = self._ds[index]
        return _normalize_label(row["label"], self._label_names)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self._ds[index]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        y = _normalize_label(row["label"], self._label_names)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    @property
    def is_subset(self) -> bool:
        return self.max_samples is not None


def download_cifake(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    hf_id: str = DEFAULT_HF_ID,
    splits: Sequence[str] = ("train", "test"),
) -> Path:
    """Download/cache CIFAKE splits via HuggingFace ``datasets``."""
    from datasets import load_dataset

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for split in splits:
        logger.info("Downloading CIFAKE %s → %s", split, cache)
        load_dataset(hf_id, split=split, cache_dir=str(cache))
    return cache


def cifake_stats(ds: CIFAKEDataset) -> dict[str, Any]:
    n = len(ds)
    n_fake = sum(1 for i in range(n) if ds._label_at(i) == 1)
    return {
        "n": n,
        "n_fake": n_fake,
        "n_real": n - n_fake,
        "split": ds.split,
        "hf_id": ds.hf_id,
        "subset": ds.is_subset,
        "max_samples": ds.max_samples,
    }
