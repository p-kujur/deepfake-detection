"""Default probe weight resolution and shared product notes (M4).

Preference order for single-probe demo:
  1. weights/progan_clip_vit_l14_linear.pth  (best Hemg cross-gen among shipped heads)
  2. weights/unifd_clip_vit_l14_linear.pth   (HF UniFD mirror)
  3. weights/cifake_clip_vit_l14_linear.pth  (CIFAKE in-domain)

Optional --ensemble soft_router uses ProGAN + CIFAKE heads together
(model_id=dual-probe-soft_router). That preserves CIFAKE but does **not**
raise Hemg above the ProGAN-only near-miss (~0.78).
"""

from __future__ import annotations

from pathlib import Path

from deepfake_detection import __version__

# Preference: ProGAN → UniFD HF → CIFAKE
WEIGHT_CANDIDATES: list[tuple[str, Path]] = [
    ("progan-clip-vit-l14-linear", Path("weights/progan_clip_vit_l14_linear.pth")),
    ("unifd-clip-vit-l14-linear", Path("weights/unifd_clip_vit_l14_linear.pth")),
    ("cifake-clip-vit-l14-linear", Path("weights/cifake_clip_vit_l14_linear.pth")),
]

PROGAN_WEIGHTS = WEIGHT_CANDIDATES[0][1]
CIFAKE_WEIGHTS = WEIGHT_CANDIDATES[2][1]

ENSEMBLE_MODEL_ID = "dual-probe-soft_router"

# Honest product caveats — M3 near-miss must stay visible.
PRODUCT_NOTES = (
    "Experimental AIGC still-image score — not court-grade authenticity evidence. "
    "M3 near-miss: best Hemg cross-gen AUC ≈0.78 with the ProGAN-trained probe; "
    "that same probe has a severe CIFAKE domain tradeoff (AUC ≈0.39). "
    "Cross-generator generalisation is limited; do not treat scores as ≥0.80 cross-gen. "
    "Optional soft_router dual-probe preserves CIFAKE≈0.97 and ProGAN holdout≈0.997 "
    "but does not beat ProGAN-only on Hemg. Face-video is out of scope for v1."
)

SCHEMA_KEYS = ("p_fake", "label", "model_id", "notes", "version")

_STEM_TO_ID = {
    "progan_clip_vit_l14_linear": "progan-clip-vit-l14-linear",
    "unifd_clip_vit_l14_linear": "unifd-clip-vit-l14-linear",
    "cifake_clip_vit_l14_linear": "cifake-clip-vit-l14-linear",
    "fc_weights": "unifd-clip-vit-l14-linear",
}


def model_id_for_weights(path: str | Path | None) -> str:
    """Map a checkpoint path to a stable model_id string."""
    if path is None:
        return "unifd-clip-vit-l14-linear-untrained"
    stem = Path(path).stem
    return _STEM_TO_ID.get(stem, stem.replace("_", "-"))


def resolve_default_probe_weights(
    explicit: str | Path | None = None,
) -> tuple[Path | None, str, bool]:
    """Resolve probe weights for single-head inference.

    Returns ``(path_or_None, model_id, found)``.
    If ``explicit`` is set, that path is used (found iff file exists).
    Otherwise walk WEIGHT_CANDIDATES preference order.
    """
    if explicit is not None and str(explicit).strip():
        p = Path(explicit)
        mid = model_id_for_weights(p)
        if p.is_file():
            return p, mid, True
        return p, mid, False

    for mid, cand in WEIGHT_CANDIDATES:
        if cand.is_file():
            return cand, mid, True

    # Nothing on disk — advertise preferred ProGAN path as the missing default.
    return PROGAN_WEIGHTS, WEIGHT_CANDIDATES[0][0], False


def resolve_ensemble_weights(
    progan: str | Path | None = None,
    cifake: str | Path | None = None,
) -> tuple[Path, Path]:
    """Require both ProGAN and CIFAKE heads for soft_router."""
    pa = Path(progan) if progan else PROGAN_WEIGHTS
    pb = Path(cifake) if cifake else CIFAKE_WEIGHTS
    missing = [str(p) for p in (pa, pb) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "soft_router ensemble needs ProGAN + CIFAKE probe weights; missing: "
            + ", ".join(missing)
            + ". Train/copy to weights/ or pass --weights / --cifake-weights."
        )
    return pa, pb


def product_notes(extra: str | None = None) -> str:
    base = PRODUCT_NOTES
    if extra and str(extra).strip():
        return f"{base} {str(extra).strip()}"
    return base


def schema_dict(
    *,
    p_fake: float,
    label: str,
    model_id: str,
    notes: str | None = None,
    version: str | None = None,
) -> dict:
    return {
        "p_fake": float(p_fake),
        "label": label,
        "model_id": model_id,
        "notes": notes or product_notes(),
        "version": version or __version__,
    }
