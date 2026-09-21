"""Model wrappers (UniFD-style CLIP + linear probe)."""

from .unifd import UniFDClipLinear, load_detector

__all__ = ["UniFDClipLinear", "load_detector"]
