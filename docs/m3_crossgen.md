# M3 — Cross-generator eval + NPR-lite fusion

**Date:** 2026-09-21 (IST) · **Stretch target:** cross-gen AUC ≥ 0.80 · **Result:** **missed** (harness shipped).

## Goal

1. Cross-test the M2 CIFAKE CLIP linear probe on unseen generator families.
2. Add a second head (NPR-lite) + mean-prob fusion.
3. Quick robustness: JPEG q=70 and short-side resize=128.
4. Write `artifacts/runs/*/metrics.json`; commit code/docs (not datasets/weights).

## What shipped

| Piece | Path |
|-------|------|
| Cross-gen download | `scripts/download_crossgen.py` |
| Folder dataset | `src/deepfake_detection/data/folder_dataset.py` |
| Eval + fusion + robustness | `python -m deepfake_detection.eval.cross_gen` |
| NPR-lite model | `src/deepfake_detection/models/npr_lite.py` |
| NPR-lite trainer | `python -m deepfake_detection.train.cnn_head` |
| Full NPR | **Not vendored** — see `docs/npr_integration.md` |

## Datasets (modest HF downloads)

| Pack | HF source | Role | Eval n |
|------|-----------|------|--------|
| `progan_eval` | `Oliver1515/ProGAN-Eval` | GAN family (unseen vs CIFAKE SD) | 80+80 |
| `hemg_wild` | `Hemg/AI-Generated-vs-Real-Images-Datasets` | Midjourney-style wild AI art | 80+80 |

```bash
python scripts/download_crossgen.py --packs progan hemg_wild --max-per-class 80
```

## NPR-lite in-domain (CIFAKE subset, same 4k/2k as M2)

| Metric | Value |
|--------|-------|
| Acc | **0.908** |
| AP | **0.9569** |
| AUC | **0.9665** |
| n | 2000 |
| Weights | `weights/npr_lite_cifake.pth` (gitignored) |

## Cross-generator results (UniFD M2 probe)

Honest numbers — **stretch AUC ≥ 0.80 not met**.

| Pack | Model | Acc | AP | AUC | n |
|------|-------|-----|----|-----|---|
| ProGAN-Eval | UniFD probe | 0.431 | 0.419 | **0.377** | 160 |
| ProGAN-Eval | NPR-lite | 0.500 | 0.500 | 0.500 | 160 |
| ProGAN-Eval | Fusion 0.5/0.5 | 0.500 | 0.419 | 0.377 | 160 |
| Hemg wild | UniFD probe | 0.481 | 0.462 | **0.456** | 160 |
| Hemg wild | NPR-lite | 0.500 | 0.506 | 0.513 | 160 |
| Hemg wild | Fusion 0.5/0.5 | 0.500 | 0.462 | 0.456 | 160 |

### Robustness (UniFD AUC)

| Pack | Clean | JPEG q=70 | Δ | Resize short=128 | Δ |
|------|-------|-----------|---|------------------|---|
| ProGAN-Eval | 0.377 | 0.326 | −0.051 | 0.412 | +0.035 |
| Hemg wild | 0.456 | 0.430 | −0.026 | 0.448 | −0.008 |

JPEG drop on ProGAN ≈ 0.05 (at the plan’s ≤0.05 budget); Hemg JPEG drop is smaller. Absolute AUC remains below chance-to-weak on both packs.

## Interpretation / blockers

1. **Domain gap:** M2 probe was trained on CIFAKE (32×32 Stable Diffusion). ProGAN (GAN, ~256²) and Hemg (web Midjourney-style) are far OOD — collapse toward chance is expected without GenImage-scale training or Blur/JPEG aug + multi-generator data.
2. **NPR-lite** transfers poorly cross-gen (≈ chance) despite strong in-domain CIFAKE scores — residual CNN overfits CIFAKE fingerprint.
3. **Full NPR / GenImage** skipped to keep downloads & deps modest; harness is ready for larger packs later.
4. **No GPU** in this environment — CLIP ViT-L/14 eval is ~0.35 s/img CPU.

## Reproduce

```bash
python scripts/download_crossgen.py --packs progan hemg_wild --max-per-class 80
python -m deepfake_detection.train.cnn_head --config configs/train_npr_lite_subset.yaml
python -m deepfake_detection.eval.cross_gen \
  --probe-weights weights/cifake_clip_vit_l14_linear.pth \
  --npr-weights weights/npr_lite_cifake.pth \
  --packs progan_eval hemg_wild --max-per-class 80
```

Summary artifact (gitignored): `artifacts/runs/<id>-m3-summary/summary.json`.
