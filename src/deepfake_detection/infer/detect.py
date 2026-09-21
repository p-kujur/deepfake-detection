"""CLI: python -m deepfake_detection.infer.detect IMAGE → JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from deepfake_detection.models.unifd import detect_image


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepfake-detect",
        description="Detect AIGC / synthetic still images (UniFD-style CLIP probe).",
    )
    p.add_argument("image", help="Path to input image")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--weights", default=None, help="Override probe weights path")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--backend", choices=["open_clip", "transformers"], default=None)
    p.add_argument("--require-weights", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("infer", {})

    weights = args.weights or model_cfg.get("probe_weights")
    threshold = (
        args.threshold if args.threshold is not None else float(model_cfg.get("threshold", 0.5))
    )
    device = args.device or infer_cfg.get("device", "auto")
    backend = args.backend or model_cfg.get("backend", "open_clip")
    notes = cfg.get("notes")

    if args.require_weights and weights and not Path(weights).is_file():
        print(
            json.dumps(
                {
                    "error": "probe_weights_missing",
                    "path": weights,
                    "hint": "python scripts/download_weights.py",
                }
            ),
            file=sys.stderr,
        )
        return 2

    try:
        result = detect_image(
            args.image,
            probe_weights=weights,
            backend=backend,
            device=device,
            threshold=threshold,
            image_size=int(model_cfg.get("image_size", 224)),
            notes=notes,
        )
    except FileNotFoundError as e:
        print(json.dumps({"error": "file_not_found", "detail": str(e)}), file=sys.stderr)
        return 1
    except Exception as e:
        print(json.dumps({"error": "inference_failed", "detail": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
