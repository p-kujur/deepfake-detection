# M3b — Close domain gap: ProGAN/ForenSynths CLIP linear probe

**Date:** 2026-09-21 (IST) · **Stretch target:** cross-gen AUC ≥ 0.80 · **Result:** **missed** (Hemg AUC **0.7798**).

## Goal

Train (or fine-tune) the frozen CLIP linear probe the UniFD way — on **ProGAN / ForenSynths-style** data instead of CIFAKE (SD 32×32) — then re-run cross-gen tables.

## Approach (chosen)

Modest HF downloads only (~256 MB materialised):

| Source | Role | Size |
|--------|------|------|
| `Oliver1515/ProGAN-Eval` full (1000R+1000F) | Stratified train/val/test | ~204 MB |
| `frp94/progan_val` **train** split only (249R+272F) | Extra train pool (separate HF) | ~52 MB |
| `Hemg/...` wild pack | Cross-gen only — **never in train** | already on disk |
| CIFAKE test subset | In-domain tradeoff re-eval | already on disk |

### Documented split (`configs/progan_splits_summary.json`)

- **Seed 42**, stratified per class on ProGAN-Eval:
  - **test** = 200R+200F in-domain holdout → also `datasets/crossgen/progan_holdout/`
  - **val** = 100R+100F
  - **train** = ProGAN-Eval remainder + all frp94 train → **949R+972F**
- Hemg / CIFAKE never appear in train.

## What shipped

| Piece | Path |
|-------|------|
| Download | `scripts/download_progan_train.py` |
| Split | `scripts/make_progan_splits.py` |
| Train config | `configs/train_progan_subset.yaml` |
| Split summary | `configs/progan_splits_summary.json` |
| Probe trainer | `train.probe` supports `data.name: progan_folder` |
| Weights (gitignored) | `weights/progan_clip_vit_l14_linear.pth` |

## Results (ProGAN-trained UniFD probe)

### In-domain ProGAN holdout

| Metric | Value |
|--------|-------|
| Acc | **0.9875** |
| AP | **0.9997** |
| AUC | **0.9997** |
| n | 400 |

### Cross-generator (Hemg wild Midjourney-style — unseen)

| Metric | CIFAKE probe (M3) | **ProGAN probe (M3b)** | Δ |
|--------|-------------------|------------------------|---|
| Acc | 0.481 | 0.571 | +0.09 |
| AP | 0.462 | **0.8128** | +0.35 |
| AUC | 0.456 | **0.7798** | **+0.32** |
| n | 160 | 177 | — |

**Target AUC ≥ 0.80: not met** (short by ~0.020). Ranking quality improved sharply; Acc@0.5 remains weak (threshold not calibrated for Hemg).

### CIFAKE test tradeoff (subset n=2000, same seed as M2)

| Probe train domain | Acc | AP | AUC |
|--------------------|-----|----|-----|
| CIFAKE (M2) | 0.9125 | 0.9736 | 0.9739 |
| **ProGAN (M3b)** | **0.4215** | **0.4188** | **0.3915** |

Symmetric domain swap: ProGAN probe collapses on CIFAKE SD 32×32 just as CIFAKE probe collapsed on ProGAN/Hemg.

## Interpretation

1. Training on ProGAN-family data closes the UniFD-style **GAN** domain gap for in-domain detection (near-perfect holdout).
2. Transfer to **wild Midjourney-style** art improves a lot (AUC 0.46 → 0.78) but is still slightly under the 0.80 stretch bar on this small pack.
3. Single-generator probe ≠ universal detector — CIFAKE tradeoff shows the cost of specialization.
4. Next levers (not done here): more ForenSynths GAN families, GenImage subset, or score fusion / threshold calibration on a val pack that is neither Hemg nor the final test.

## Reproduce

```bash
python scripts/download_progan_train.py --sources progan_eval frp94_train
python scripts/make_progan_splits.py
python -m deepfake_detection.train.probe --config configs/train_progan_subset.yaml

# In-domain holdout + Hemg cross-gen
python -m deepfake_detection.eval.cross_gen \
  --probe-weights weights/progan_clip_vit_l14_linear.pth \
  --packs progan_holdout hemg_wild --skip-npr --skip-robustness

# CIFAKE tradeoff
python -m deepfake_detection.eval.run_eval \
  --dataset cifake --probe-weights weights/progan_clip_vit_l14_linear.pth \
  --max-samples 2000
```

Train run id: `20260921-222903-cfd86aa4` · Hemg metrics: `artifacts/runs/20260921-224523-bce87d34-cg-hemg_wild-unifd/`.
