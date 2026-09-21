"""Inference CLI and helpers (M4 product wrap)."""

from deepfake_detection.infer.pipeline import run_detect
from deepfake_detection.infer.resolve import (
    PRODUCT_NOTES,
    SCHEMA_KEYS,
    resolve_default_probe_weights,
)

__all__ = [
    "run_detect",
    "resolve_default_probe_weights",
    "PRODUCT_NOTES",
    "SCHEMA_KEYS",
]
