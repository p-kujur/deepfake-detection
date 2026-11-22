#!/usr/bin/env bash
# Assemble the Hugging Face Space bundle: vendors src/deepfake_detection and the
# probe weights (gitignored) next to app.py. Then upload space/ to
# https://huggingface.co/spaces/ezra9426/deepfake-detector
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$here/.."
rm -rf "$here/src" "$here/weights"
mkdir -p "$here/src" "$here/weights"
cp -r "$root/src/deepfake_detection" "$here/src/"
find "$here/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$root/weights/progan_clip_vit_l14_linear.pth" "$here/weights/"
cp "$root/weights/cifake_clip_vit_l14_linear.pth" "$here/weights/"   # enables soft_router
echo "Space bundle ready in $here"
