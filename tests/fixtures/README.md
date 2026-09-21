# Smoke fixtures

## Labeled M1 pack (`smoke/`) — P1b gate

Curated **20-image** real/fake pack used by `tests/test_smoke_infer.py`:

```
tests/fixtures/smoke/
  real/   # 10 PNGs — photographic / LSUN-style reals
  fake/   # 10 PNGs — ProGAN fakes
  manifest.json
```

**Gate:** with UniFD probe weights present, expect **≥ 18/20** correct at threshold 0.5.

### Source

- Hugging Face dataset [`Oliver1515/ProGAN-Eval`](https://huggingface.co/datasets/Oliver1515/ProGAN-Eval)
  (ProGAN / ForenSynths-style 256×256 eval images; folders `0_real` / `1_fake`).
- Label mapping: `0` → real, `1` → fake (CNNDetection / UniFD convention).
- First 10 examples of each class were exported as PNG (≈2 MB total).
- Compatible with UniFD CLIP ViT-L/14 linear probe trained on ForenSynths/ProGAN
  ([Ojha et al., CVPR 2023](https://arxiv.org/abs/2302.10174)).

Do **not** use unrelated “AI vs Real” web scrapes for this gate — domain shift can
collapse accuracy even when weights load correctly.

### Rebuild (optional)

```bash
# requires: pip install datasets
python scripts/build_smoke_pack.py
```

## Legacy solid-color PNGs

`smoke_*.png` are unlabeled synthetic colors for I/O wiring only
(`scripts/make_smoke_fixtures.py`). They are **not** part of the accuracy gate.
