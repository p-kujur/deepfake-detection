#!/usr/bin/env python3
"""Build tests/fixtures/smoke/ — 10 real + 10 fake from Oliver1515/ProGAN-Eval.

Requires: pip install datasets Pillow
Keeps download streaming / tiny (only 20 images written).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/smoke"),
        help="Output directory (default: tests/fixtures/smoke)",
    )
    parser.add_argument("--per-class", type=int, default=10)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "real").mkdir(parents=True)
    (out / "fake").mkdir(parents=True)

    ds = load_dataset("Oliver1515/ProGAN-Eval", split="train", streaming=True)
    # 0_real / 1_fake convention
    label_to_cls = {0: "real", 1: "fake"}
    saved = {"real": 0, "fake": 0}
    need = args.per_class

    for ex in ds:
        lab = int(ex["label"])
        cls = label_to_cls[lab]
        if saved[cls] >= need:
            if all(saved[c] >= need for c in saved):
                break
            continue
        img = ex["image"].convert("RGB")
        saved[cls] += 1
        path = out / cls / f"{cls}_{saved[cls]:02d}.png"
        img.save(path, format="PNG", optimize=True)
        print(f"wrote {path} size={img.size}")
        if all(saved[c] >= need for c in saved):
            break

    manifest = []
    for cls in ("real", "fake"):
        for p in sorted((out / cls).glob("*.png")):
            manifest.append({"path": p.as_posix(), "label": cls})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"DONE {saved} manifest={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
