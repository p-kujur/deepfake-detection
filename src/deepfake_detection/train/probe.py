"""Train the UniFD-style linear probe on frozen CLIP features.

Placeholder for M2+: load CIFAKE / ForenSynths embeddings and fit ``nn.Linear``.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CLIP linear probe (stub)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None, help="Dataset root (not bundled)")
    args = parser.parse_args(argv)
    logger.warning(
        "Train stub only (M1). Config=%s data_root=%s — implement in M2.",
        args.config,
        args.data_root,
    )
    print("Train stub: no-op in M1 scaffold. See docs/plan.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
