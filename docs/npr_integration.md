# NPR integration path (M3)

## What we shipped (NPR-lite)

Full **NPR** (Tan et al., CVPR 2024 — [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection))
depends on a custom ResNet-50 training stack and large pretrained checkpoints. Pulling that
into this repo would dominate deps and disk for a modest milestone.

Instead M3 ships **NPR-lite** (`deepfake_detection.models.npr_lite`):

1. Neighboring-pixel residual stem: `x - avg_pool3x3(x)` (upsampling / neighbor cue).
2. Tiny conv tower (~100k params) trained on the same CIFAKE subset as the UniFD probe.
3. Mean-probability fusion with the frozen-CLIP linear probe.

## Full NPR upgrade path

1. Vendor or submodule `chuangchuangtan/NPR-DeepfakeDetection` under `third_party/npr/`.
2. Add optional extra: `pip install -e ".[npr]"` (torchvision ResNet weights).
3. Wrap their `Trainer` / `networks.resnet` behind `models/npr_full.py` with the same
   `predict_proba(images) -> (N,)` contract as NPR-lite.
4. Point `--npr-weights` at their `.pth` and reuse `eval.cross_gen` fusion unchanged.

## Train / eval

```bash
python -m deepfake_detection.train.cnn_head --config configs/train_npr_lite_subset.yaml
python -m deepfake_detection.eval.cross_gen \
  --probe-weights weights/cifake_clip_vit_l14_linear.pth \
  --npr-weights weights/npr_lite_cifake.pth
```
