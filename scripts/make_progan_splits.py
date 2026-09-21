#!/usr/bin/env python3
"""Build documented train/val/test splits for ProGAN UniFD-style probe training.

Protocol (no Hemg / CIFAKE leakage):
  - Pool A: datasets/progan_forensynths/progan_eval_full/{real,fake}
  - Pool B: datasets/progan_forensynths/frp94_train/{real,fake}  → train only
  - Split A with seed (default 42), stratified by class:
      test  = n_test per class   (in-domain holdout; also mirrored to crossgen)
      val   = n_val per class
      train = remainder of A + ALL of B
  - Hemg wild is never included.

Writes:
  datasets/progan_forensynths/splits/{train,val,test}/{real,fake}/  (symlinks or copies)
  datasets/progan_forensynths/splits/manifest.json
  datasets/crossgen/progan_holdout/{real,fake}/  (copy of test for eval harness)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("make_progan_splits")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _split_paths(paths: list[Path], n_test: int, n_val: int, rng: random.Random):
    paths = list(paths)
    rng.shuffle(paths)
    test = paths[:n_test]
    val = paths[n_test : n_test + n_val]
    train = paths[n_test + n_val :]
    return train, val, test


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="datasets/progan_forensynths")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-test", type=int, default=200, help="Per class holdout")
    p.add_argument("--n-val", type=int, default=100, help="Per class val")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--crossgen-holdout", default="datasets/crossgen/progan_holdout")
    args = p.parse_args()

    root = Path(args.root)
    pool_a = root / "progan_eval_full"
    pool_b = root / "frp94_train"
    splits_root = root / "splits"

    a_real = _list_images(pool_a / "real")
    a_fake = _list_images(pool_a / "fake")
    if len(a_real) < args.n_test + args.n_val + 50 or len(a_fake) < args.n_test + args.n_val + 50:
        raise SystemExit(
            f"Need more ProGAN-Eval images under {pool_a}. "
            f"Got real={len(a_real)} fake={len(a_fake)}. "
            "Run: python scripts/download_progan_train.py --sources progan_eval"
        )

    rng = random.Random(args.seed)
    tr_r, va_r, te_r = _split_paths(a_real, args.n_test, args.n_val, rng)
    tr_f, va_f, te_f = _split_paths(a_fake, args.n_test, args.n_val, rng)

    b_real = _list_images(pool_b / "real")
    b_fake = _list_images(pool_b / "fake")
    tr_r = tr_r + b_real
    tr_f = tr_f + b_fake

    # wipe splits
    if splits_root.exists():
        shutil.rmtree(splits_root)

    mapping = {
        "train": (tr_r, tr_f),
        "val": (va_r, va_f),
        "test": (te_r, te_f),
    }
    manifest: dict = {
        "seed": args.seed,
        "n_test_per_class": args.n_test,
        "n_val_per_class": args.n_val,
        "sources": {
            "progan_eval_full": str(pool_a),
            "frp94_train": str(pool_b) if (b_real or b_fake) else None,
        },
        "notes": (
            "Train = ProGAN-Eval remainder + frp94/progan_val train. "
            "Test = ProGAN-Eval holdout (in-domain). "
            "Hemg / CIFAKE never in train. Shuffle seed stratified per class."
        ),
        "splits": {},
    }

    for split, (reals, fakes) in mapping.items():
        for cls, paths in (("real", reals), ("fake", fakes)):
            for src in paths:
                dst = splits_root / split / cls / src.name
                # disambiguate name collisions (frp94 vs eval)
                if dst.exists() or dst.is_symlink():
                    dst = splits_root / split / cls / f"{src.parent.name}__{src.name}"
                _link_or_copy(src, dst, args.mode)
        manifest["splits"][split] = {
            "n_real": len(reals),
            "n_fake": len(fakes),
            "real_files": [p.name for p in reals],
            "fake_files": [p.name for p in fakes],
        }
        logger.info(
            "%s: real=%d fake=%d", split, len(reals), len(fakes)
        )

    # Mirror test → crossgen holdout (copy so eval harness works without following links)
    holdout = Path(args.crossgen_holdout)
    if holdout.exists():
        shutil.rmtree(holdout)
    for cls in ("real", "fake"):
        src_dir = splits_root / "test" / cls
        for src in _list_images(src_dir):
            # resolve symlink
            real_src = src.resolve()
            _link_or_copy(real_src, holdout / cls / src.name, "copy")
    logger.info("Mirrored test holdout → %s", holdout)

    man_path = splits_root / "manifest.json"
    # slim committed-friendly copy without huge file lists in repo — full stays local
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    slim = {
        "seed": manifest["seed"],
        "n_test_per_class": manifest["n_test_per_class"],
        "n_val_per_class": manifest["n_val_per_class"],
        "sources": manifest["sources"],
        "notes": manifest["notes"],
        "counts": {k: {"n_real": v["n_real"], "n_fake": v["n_fake"]} for k, v in manifest["splits"].items()},
    }
    slim_path = Path("configs/progan_splits_summary.json")
    slim_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s and %s", man_path, slim_path)
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
