# M3c — Dual-probe ensemble/router + near-miss on Hemg ≥0.80

**Date:** 2026-09-21 (IST) · **Stretch target:** cross-gen Hemg AUC ≥ 0.80 · **Result:** **missed** (best Hemg AUC **0.7798**, same as M3b ProGAN probe).

## Framing (course-correct)

Chasing Hemg with a single specialized head tanks CIFAKE (M3b tradeoff AUC ≈0.39). Prefer a **simple ensemble/router** over ProGAN + CIFAKE UniFD probes so in-domain ProGAN and CIFAKE both stay strong. Keep **CIFAKE as a separate tradeoff table**. Never select fusion on Hemg.

## What shipped

| Piece | Path |
|-------|------|
| Dual-probe eval | `python -m deepfake_detection.eval.dual_probe` |
| Logit fusion helper | `fuse_logits` in `eval/cross_gen.py` |
| NPR-lite ProGAN trainer | `train/cnn_head.py` + `configs/train_npr_lite_progan.yaml` |
| Summary JSON | `docs/m3c_dual_probe_summary.json` |

## Selection rule (not Hemg)

On **ProGAN val** + small **CIFAKE test subset (n=800)**, pick mode maximizing  
`0.5 * (AUC_progan_val + AUC_cifake)` among candidates with **ProGAN-val AUC ≥ 0.95**.

**Selected:** `soft_router` (softmax over \|logit\| → mix probs).

## Tables (separate)

### In-domain — ProGAN holdout (n=400)

| Model | Acc | AP | AUC |
|-------|-----|----|-----|
| ProGAN probe alone | 0.9875 | 0.9997 | **0.9997** |
| CIFAKE probe alone | 0.4000 | 0.4199 | 0.4047 |
| **soft_router (selected)** | 0.9700 | 0.9971 | **0.9971** |
| mean 0.5 / logit 0.5 / hard router | — | — | 0.995–0.997 |

ProGAN in-domain **preserved** (≥0.95).

### Cross-generator — Hemg wild (n=177, unseen)

| Model | Acc | AP | AUC |
|-------|-----|----|-----|
| **ProGAN probe alone** | 0.5706 | 0.8128 | **0.7798** |
| CIFAKE probe alone | 0.5141 | 0.5121 | 0.4568 |
| soft_router (selected) | 0.6328 | 0.6482 | 0.6885 |
| mean 0.5 | 0.6328 | 0.7223 | 0.7082 |
| logit 0.5 | 0.6328 | 0.6642 | 0.6927 |
| hard confidence router | 0.6328 | 0.5895 | 0.6317 |

**Gate AUC ≥ 0.80: not met.** Best cross-gen score remains the M3b ProGAN probe (0.7798). Mixing the CIFAKE head **hurts** Hemg ranking (CIFAKE is anti-correlated on this pack).

### CIFAKE tradeoff (separate; subset n=800)

| Model | Acc | AP | AUC |
|-------|-----|----|-----|
| ProGAN probe alone | 0.4163 | 0.4251 | 0.3820 |
| CIFAKE probe alone | 0.9100 | 0.9741 | **0.9728** |
| **soft_router (selected)** | 0.8975 | 0.9647 | **0.9666** |

Dual probe restores CIFAKE without destroying ProGAN holdout — the product-shaped win of M3c.

## NPR-lite on ProGAN (idea 1, aborted as Hemg lever)

Retrained NPR-lite on the same ProGAN train split (`weights/npr_lite_progan.pth`). Holdout AUC **≈0.51** (chance) despite falling train loss — residual CNN does not learn a transferable ProGAN cue at this scale. Fusion with UniFD would not clear 0.80; not used for the gate number.

JPEG/Blur probe aug (idea 2) was **already on** in M3b (`aug=true` in `train_progan_subset.yaml`). Threshold calibration skipped (Acc-only).

## Interpretation

1. **Domain router helps the product** (ProGAN≈1.0 + CIFAKE≈0.97) but cannot invent Hemg signal the ProGAN head lacks.
2. Hemg shortfall is **~0.020 AUC** under the best single head — small-n wild pack (177); ranking is already useful (AP 0.81).
3. Stretch ≥0.80 needs more generator diversity (ForenSynths families / GenImage subset) or a stronger second cue (full NPR), not CIFAKE-head mixing.

## M4 polish path (prepared, not executed)

1. Product default: `soft_router` dual UniFD probes; expose `model_id=dual-probe-soft_router`.
2. Cross-gen reporting: always publish **ProGAN-probe** Hemg row separately from the product ensemble.
3. Optional: GenImage Midjourney/SD subset download + re-eval; full NPR path (`docs/npr_integration.md`).
4. CLI/API schema already stubbed — wire `detect` to dual-probe weights list.
5. Do **not** retune fusion on Hemg; keep val+CIFAKE selection.

## Reproduce

```bash
python -m deepfake_detection.eval.dual_probe \
  --progan-weights weights/progan_clip_vit_l14_linear.pth \
  --cifake-weights weights/cifake_clip_vit_l14_linear.pth \
  --cifake-max 800

# Best Hemg number remains UniFD-only ProGAN probe:
python -m deepfake_detection.eval.cross_gen \
  --probe-weights weights/progan_clip_vit_l14_linear.pth \
  --packs progan_holdout hemg_wild --skip-npr --skip-robustness
```
