#!/usr/bin/env python3
"""Download modest ProGAN / ForenSynths-style train sources (gitignored).

Sources (kept small):
  1. Oliver1515/ProGAN-Eval — full 1000 real + 1000 fake (~176 MB)
     Used for documented train/val/test split (see make_progan_splits.py).
  2. frp94/progan_val split=train — 521 images (~54 MB parquet)
     Separate HF train split; added to the train pool only.

Does NOT download Hemg / CIFAKE (those stay eval-only for cross-gen / tradeoff).
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_progan_train")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def download_progan_eval(out_root: Path) -> dict:
    """Parallel snapshot of Oliver1515/ProGAN-Eval → real/fake folders."""
    from huggingface_hub import snapshot_download

    repo_id = "Oliver1515/ProGAN-Eval"
    real_out = out_root / "real"
    fake_out = out_root / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    local = Path(
        snapshot_download(repo_id, repo_type="dataset", max_workers=16)
    )
    n_real = n_fake = 0
    for src in tqdm(sorted((local / "0_real").iterdir()), desc="progan_eval/real"):
        if not _is_image(src.name):
            continue
        dst = real_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        n_real += 1
    for src in tqdm(sorted((local / "1_fake").iterdir()), desc="progan_eval/fake"):
        if not _is_image(src.name):
            continue
        dst = fake_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        n_fake += 1

    summary = {
        "repo_id": repo_id,
        "n_real": n_real,
        "n_fake": n_fake,
        "out": str(out_root),
    }
    logger.info("ProGAN-Eval ready: %s", summary)
    return summary


def download_frp94_train(out_root: Path) -> dict:
    """Materialize frp94/progan_val train split as real/fake folders."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    real_out = out_root / "real"
    fake_out = out_root / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    train_pq = hf_hub_download(
        "frp94/progan_val",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    ds = load_dataset("parquet", data_files=train_pq, split="train")
    n_real = n_fake = 0
    for i, ex in enumerate(tqdm(ds, desc="frp94_train")):
        lab = int(ex["label"])
        img = ex["image"].convert("RGB")
        if lab == 0:
            n_real += 1
            path = real_out / f"frp94_real_{n_real:05d}.png"
        else:
            n_fake += 1
            path = fake_out / f"frp94_fake_{n_fake:05d}.png"
        if not path.exists():
            img.save(path, format="PNG", optimize=True)

    summary = {
        "repo_id": "frp94/progan_val",
        "split": "train",
        "n_real": n_real,
        "n_fake": n_fake,
        "out": str(out_root),
    }
    logger.info("frp94 train ready: %s", summary)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="datasets/progan_forensynths")
    p.add_argument(
        "--sources",
        nargs="+",
        default=["progan_eval", "frp94_train"],
        choices=["progan_eval", "frp94_train"],
    )
    args = p.parse_args()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    if "progan_eval" in args.sources:
        download_progan_eval(root / "progan_eval_full")
    if "frp94_train" in args.sources:
        download_frp94_train(root / "frp94_train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
