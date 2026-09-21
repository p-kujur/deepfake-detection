# M2 CIFAKE linear-probe run notes

## First metrics (CPU subset)

| Field | Value |
|-------|-------|
| Commit | `9c1a2fa` |
| Dataset | `cifake_subset` (official HF split, capped) |
| Train / test | 4000 / 2000 |
| Acc | **0.9125** |
| AP | **0.973608** |
| AUC | **0.973885** |
| n | 2000 |
| Threshold | 0.5 |
| Latency p50 / p95 | ~321 / ~393 ms / image (CPU, feature extract) |
| Checkpoint | `artifacts/checkpoints/<run>_cifake_probe.pth` (gitignored) |
| Convenience weights | `weights/cifake_clip_vit_l14_linear.pth` (gitignored) |
| Metrics path | `artifacts/runs/<run_id>/metrics.json` (gitignored) |

Aspirational plan targets (Acc≥0.90, AP/AUC≥0.95) **met on this subset**.

## Reproduce

```bash
python scripts/download_cifake.py
python -m deepfake_detection.train.probe --config configs/train_cifake_subset.yaml
# Full official 100k/20k (slow on CPU):
python -m deepfake_detection.train.probe --config configs/train_cifake.yaml
```

HF mirror: `dragonintelligence/CIFAKE-image-dataset`  
Upstream: [jordan-bird/CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) · [arXiv:2303.14126](https://arxiv.org/abs/2303.14126)
