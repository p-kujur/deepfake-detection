#!/usr/bin/env python3
"""Create tiny synthetic RGB fixtures for smoke tests (no dataset download).

Writes a few solid-color PNGs under tests/fixtures/. These are NOT real/fake
labeled media — only for I/O and pipeline smoke. Replace with curated pack later.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("smoke_red.png", (220, 40, 40)),
        ("smoke_green.png", (40, 180, 60)),
        ("smoke_blue.png", (40, 80, 220)),
        ("smoke_gray.png", (128, 128, 128)),
    ]
    for name, color in specs:
        img = Image.new("RGB", (256, 256), color)
        path = ROOT / name
        img.save(path)
        print(f"wrote {path}")
    readme = ROOT / "README.md"
    readme.write_text(
        "# Smoke fixtures\n\n"
        "Synthetic solid-color PNGs for pipeline smoke tests only.\n"
        "Not a labeled real/fake evaluation set.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
