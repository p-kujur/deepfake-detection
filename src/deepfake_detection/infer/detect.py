"""CLI: python -m deepfake_detection.infer.detect IMAGE → JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from deepfake_detection.infer.pipeline import run_detect
from deepfake_detection.infer.resolve import SCHEMA_KEYS, resolve_default_probe_weights


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
        description=(
            "Detect AIGC / synthetic still images (CLIP ViT-L/14 + linear probe). "
            "Default weights: ProGAN → UniFD HF → CIFAKE."
        ),
    )
    p.add_argument("image", help="Path to input image")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--weights",
        default=None,
        help="Override probe weights (single mode) or ProGAN head (soft_router)",
    )
    p.add_argument(
        "--cifake-weights",
        default=None,
        help="CIFAKE probe path for --ensemble soft_router (default: weights/cifake_...)",
    )
    p.add_argument(
        "--ensemble",
        choices=["soft_router"],
        default=None,
        help=(
            "Optional dual-probe fusion. soft_router preserves CIFAKE+ProGAN in-domain "
            "but does not beat ProGAN-only Hemg (~0.78)."
        ),
    )
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

    # Config probe_weights only if not "auto"/empty; else resolve preference chain.
    cfg_weights = model_cfg.get("probe_weights")
    if cfg_weights in (None, "", "auto"):
        cfg_weights = None
    weights = args.weights or cfg_weights
    threshold = (
        args.threshold if args.threshold is not None else float(model_cfg.get("threshold", 0.5))
    )
    device = args.device or infer_cfg.get("device", "auto")
    backend = args.backend or model_cfg.get("backend", "open_clip")
    notes = cfg.get("notes")
    ensemble = args.ensemble or infer_cfg.get("ensemble") or None
    if ensemble in ("", "none", "off", "null"):
        ensemble = None

    if args.require_weights and ensemble != "soft_router":
        path, _, found = resolve_default_probe_weights(weights)
        if not found:
            print(
                json.dumps(
                    {
                        "error": "probe_weights_missing",
                        "path": str(path),
                        "hint": "Prefer weights/progan_clip_vit_l14_linear.pth; "
                        "else UniFD via scripts/download_weights.py; else CIFAKE probe.",
                    }
                ),
                file=sys.stderr,
            )
            return 2

    try:
        result = run_detect(
            args.image,
            ensemble=ensemble,
            probe_weights=weights,
            cifake_weights=args.cifake_weights or model_cfg.get("cifake_weights"),
            backend=backend,
            device=device,
            threshold=threshold,
            image_size=int(model_cfg.get("image_size", 224)),
            notes=notes,
            require_weights=bool(args.require_weights),
        )
    except FileNotFoundError as e:
        print(json.dumps({"error": "file_not_found", "detail": str(e)}), file=sys.stderr)
        return 1
    except Exception as e:
        print(json.dumps({"error": "inference_failed", "detail": str(e)}), file=sys.stderr)
        return 1

    payload = result.to_dict()
    missing = [k for k in SCHEMA_KEYS if k not in payload]
    if missing:
        print(json.dumps({"error": "schema_invalid", "missing": missing}), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
