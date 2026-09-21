"""Model wrappers (UniFD CLIP probe + NPR-lite CNN)."""

from .unifd import UniFDClipLinear, load_detector
from .npr_lite import NPRLiteCNN, load_npr_lite

__all__ = ["UniFDClipLinear", "load_detector", "NPRLiteCNN", "load_npr_lite"]
