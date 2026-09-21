#!/usr/bin/env python3
"""Download / cache CIFAKE via HuggingFace datasets.

Source: dragonintelligence/CIFAKE-image-dataset (100k train / 20k test)
Upstream: jordan-bird CIFAKE (arXiv:2303.14126)

Usage:
  python scripts/download_cifake.py
  python scripts/download_cifake.py --cache-dir datasets/cifake
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running without editable install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detection.data.cifake import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_HF_ID,
    download_cifake,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_cifake")


def main() -> int:
    p = argparse.ArgumentParser(description="Download CIFAKE (HF mirror)")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--hf-id", default=DEFAULT_HF_ID)
    args = p.parse_args()
    try:
        path = download_cifake(cache_dir=args.cache_dir, hf_id=args.hf_id)
    except ImportError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.error("Download failed: %s", e)
        return 1
    logger.info("CIFAKE cached under %s", path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
