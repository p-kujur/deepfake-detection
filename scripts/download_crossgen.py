#!/usr/bin/env python3
"""Download modest cross-generator eval packs (ProGAN + wild AI-art).

Default targets (gitignored under datasets/crossgen/):
  - Oliver1515/ProGAN-Eval → GAN family unseen by CIFAKE (SD) probe
  - Hemg/AI-Generated-vs-Real-Images-Datasets → Midjourney-style wild AI art

Keeps downloads small via --max-per-class.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_crossgen")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def download_subset(
    repo_id: str,
    real_prefix: str,
    fake_prefix: str,
    out_root: Path,
    max_per_class: int,
    seed: int,
) -> dict:
    files = list_repo_files(repo_id, repo_type="dataset")
    reals = [f for f in files if f.startswith(real_prefix) and _is_image(f)]
    fakes = [f for f in files if f.startswith(fake_prefix) and _is_image(f)]
    rng = random.Random(seed)
    reals = sorted(rng.sample(reals, max_per_class)) if len(reals) > max_per_class else sorted(reals)
    fakes = sorted(rng.sample(fakes, max_per_class)) if len(fakes) > max_per_class else sorted(fakes)

    real_out = out_root / "real"
    fake_out = out_root / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    def _pull(rel: str, dest_dir: Path) -> None:
        local = hf_hub_download(repo_id, rel, repo_type="dataset")
        target = dest_dir / Path(rel).name
        if not target.exists():
            shutil.copy2(local, target)

    for rel in tqdm(reals, desc=f"{out_root.name}/real"):
        _pull(rel, real_out)
    for rel in tqdm(fakes, desc=f"{out_root.name}/fake"):
        _pull(rel, fake_out)

    summary = {"repo_id": repo_id, "n_real": len(reals), "n_fake": len(fakes), "out": str(out_root)}
    logger.info("Ready: %s", summary)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="datasets/crossgen")
    p.add_argument("--max-per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--packs", nargs="+", default=["progan", "hemg_wild"], choices=["progan", "hemg_wild"])
    args = p.parse_args()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    if "progan" in args.packs:
        download_subset(
            "Oliver1515/ProGAN-Eval",
            "0_real/",
            "1_fake/",
            root / "progan_eval",
            args.max_per_class,
            args.seed,
        )
    if "hemg_wild" in args.packs:
        n = min(args.max_per_class, 80) if args.max_per_class >= 80 else args.max_per_class
        download_subset(
            "Hemg/AI-Generated-vs-Real-Images-Datasets",
            "RealArt/",
            "AiArtData/",
            root / "hemg_wild",
            n,
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
