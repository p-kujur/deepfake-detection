#!/usr/bin/env python3
"""Download optional UniFD-style linear probe weights.

Product default probe preference (see deepfake_detection.infer.resolve):
  1. weights/progan_clip_vit_l14_linear.pth  (train ProGAN probe)
  2. weights/unifd_clip_vit_l14_linear.pth   (this script)
  3. weights/cifake_clip_vit_l14_linear.pth  (train CIFAKE probe)


UniFD (UniversalFakeDetect) official repo:
  https://github.com/WisconsinAIVision/UniversalFakeDetect

The authors host pretrained linear classifiers for CLIP ViT-L/14 trained on
ForenSynths / ProGAN (see their README "Model Zoo" / Google Drive links).

Documented checkpoint URL (may require Drive cookies / change over time):
  Primary (GitHub release / raw if mirrored): see UNIFD_WEIGHTS_URLS below.
  Official Drive folder (UniFD README):
    https://drive.google.com/drive/folders/1lkK5wYKrOTKdKwUPUFhB7gx1bxm3HtVu

If download fails, this script exits with a clear error and does NOT silently
continue. You can also place a compatible ``.pth`` manually at:
  weights/unifd_clip_vit_l14_linear.pth

Expected tensor keys: Linear ``weight`` [1, 768] and ``bias`` [1], optionally
prefixed with ``head.``.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Candidate direct URLs (tried in order). Drive links often need gdown;
# we try HTTP mirrors first, then print Drive instructions on failure.
UNIFD_WEIGHTS_URLS = [
    # Community mirror of UniFD fc_weights.pth (CLIP ViT-L/14 linear head).
    # Official UniFD repo expects pretrained_weights/fc_weights.pth; authors
    # historically used Drive. This HF mirror is used by other open detectors.
    "https://huggingface.co/siddharthksah/deepsafe-weights/resolve/main/universalfakedetect/fc_weights.pth",
]

DEFAULT_OUT = Path("weights/unifd_clip_vit_l14_linear.pth")

DRIVE_INSTRUCTIONS = """
Could not download UniFD probe weights automatically.

Manual options:
  1. Open the official UniFD Google Drive (from their README Model Zoo):
       https://drive.google.com/drive/folders/1lkK5wYKrOTKdKwUPUFhB7gx1bxm3HtVu
     Download the CLIP ViT-L/14 linear classifier checkpoint trained on
     ForenSynths/ProGAN (filename varies by release).

  2. Optionally use gdown if you have a file id:
       pip install gdown
       gdown --fuzzy 'https://drive.google.com/uc?id=FILE_ID' -O {out}

  3. Place the file at:
       {out}

  4. Or train your own probe:
       python -m deepfake_detection.train.probe --help

Paper: https://arxiv.org/abs/2302.10174
Code:  https://github.com/WisconsinAIVision/UniversalFakeDetect (MIT)
""".strip()


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "deepfake-detection/0.1 (weights-fetch)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if len(data) < 1000:
        raise RuntimeError(f"Download too small ({len(data)} bytes) from {url}")
    dest.write_bytes(data)


def _try_gdown(file_id: str, dest: Path) -> bool:
    try:
        import gdown
    except ImportError:
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = gdown.download(url, str(dest), quiet=False)
    return bool(out) and dest.is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download UniFD probe weights")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Extra direct URL(s) to try before built-in list",
    )
    parser.add_argument(
        "--gdrive-id",
        default=None,
        help="Optional Google Drive file id (uses gdown if installed)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    out: Path = args.out
    if out.is_file() and not args.force:
        print(f"Already exists: {out} (use --force to overwrite)")
        return 0

    urls = list(args.url) + list(UNIFD_WEIGHTS_URLS)
    errors: list[str] = []

    if args.gdrive_id:
        print(f"Trying gdown for Drive id={args.gdrive_id} ...")
        try:
            if _try_gdown(args.gdrive_id, out):
                print(f"Saved: {out} ({out.stat().st_size} bytes)")
                return 0
            errors.append("gdown failed or gdown not installed")
        except Exception as e:
            errors.append(f"gdown error: {e}")

    for url in urls:
        print(f"Trying {url} ...")
        try:
            _download(url, out)
            print(f"Saved: {out} ({out.stat().st_size} bytes)")
            return 0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, OSError) as e:
            errors.append(f"{url}: {e}")
            print(f"  failed: {e}", file=sys.stderr)

    msg = DRIVE_INSTRUCTIONS.format(out=out)
    if errors:
        msg += "\n\nErrors:\n- " + "\n- ".join(errors)
    print(msg, file=sys.stderr)
    print(
        "ERROR: probe weight download failed. See instructions above.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
