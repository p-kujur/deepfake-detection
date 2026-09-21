# Deepfake Detection (M1–M4)

AIGC **still-image** detector: frozen **CLIP ViT-L/14** + **linear probe** (UniFD-inspired), with CLI, FastAPI, and Gradio.

> **Not** court-grade authenticity evidence. **Not** a face-video product.  
> **M3 near-miss:** best Hemg cross-gen AUC **≈0.78** (ProGAN probe) — gate ≥0.80 **not met**.  
> ProGAN probe has a severe **CIFAKE tradeoff** (AUC ≈0.39). Do not pretend cross-gen ≥0.80.

| Item | Choice |
|------|--------|
| Modality (v1) | AIGC still images; face-video later |
| Interfaces | CLI + FastAPI + Gradio |
| Stack | Python 3.11+, PyTorch 2.x |
| License (our code) | MIT |
| Default probe | ProGAN → UniFD HF → CIFAKE (auto) |
| Optional ensemble | `--ensemble soft_router` (ProGAN+CIFAKE) |

Paper: [UniFD (CVPR 2023)](https://arxiv.org/abs/2302.10174) · Code: [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) (MIT)

## Install

```bash
cd deepfake-detection
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

No Hugging Face / API tokens required for default `open_clip` OpenAI CLIP weights (downloaded on first use).

### Download / place probe weights

**Preference for the product demo (auto-resolved):**

1. `weights/progan_clip_vit_l14_linear.pth` — best Hemg among shipped heads (~0.78 AUC)  
2. `weights/unifd_clip_vit_l14_linear.pth` — UniFD HF mirror  
3. `weights/cifake_clip_vit_l14_linear.pth` — CIFAKE in-domain probe  

```bash
# UniFD HF mirror → weights/unifd_clip_vit_l14_linear.pth
python scripts/download_weights.py

# ProGAN / CIFAKE heads come from training (gitignored convenience copies):
#   python -m deepfake_detection.train.probe --config configs/train_progan_subset.yaml
#   python -m deepfake_detection.train.probe --config configs/train_cifake_subset.yaml
# → weights/progan_clip_vit_l14_linear.pth
# → weights/cifake_clip_vit_l14_linear.pth
```

Without any probe weights the head is **randomly initialized** — wiring still runs, but **predictions are not meaningful**. `model_id` will end in `-untrained`.

## Run (shared JSON schema)

All three surfaces return the same object:

```json
{
  "p_fake": 0.87,
  "label": "fake",
  "model_id": "progan-clip-vit-l14-linear",
  "notes": "...caveats incl. M3 near-miss...",
  "version": "0.1.0"
}
```

`model_id` reflects which probe (or `dual-probe-soft_router`) actually loaded.  
`notes` always include: not court-grade; Hemg ≈0.78 near-miss; CIFAKE tradeoff; cross-gen limits.

### CLI

```bash
python -m deepfake_detection.infer.detect path/to/image.jpg
# Optional dual-probe product ensemble (needs ProGAN + CIFAKE weights):
python -m deepfake_detection.infer.detect path/to/image.jpg --ensemble soft_router
# Force a specific checkpoint:
python -m deepfake_detection.infer.detect path/to/image.jpg --weights weights/unifd_clip_vit_l14_linear.pth
```

### FastAPI

```bash
uvicorn deepfake_detection.api.app:app --host 0.0.0.0 --port 8000
# GET  /health   → status, model_id, weights, notes
# POST /detect   multipart field: file  (optional form: ensemble=soft_router)
```

### Gradio

```bash
python scripts/run_gradio.py
# or: python -m deepfake_detection.api.gradio_app
# Dropdown: none | soft_router — same JSON schema as CLI/API
```

## Metrics summary (honest)

| Milestone | Setting | Acc | AP | AUC | n | Notes |
|-----------|---------|-----|----|-----|---|-------|
| **M1** smoke | UniFD HF, 20-img pack | ≥18/20 gate | — | — | 20 | Wiring / correctness gate |
| **M2** | CIFAKE subset probe | 0.9125 | 0.9736 | **0.9739** | 2000 | In-domain subset; plan targets met |
| **M3** | CIFAKE probe → Hemg wild | 0.481 | 0.462 | **0.456** | 160 | Cross-gen failure (domain gap) |
| **M3b** | ProGAN probe → holdout | 0.9875 | 0.9997 | **0.9997** | 400 | In-domain strong |
| **M3b** | ProGAN probe → Hemg | 0.5706 | 0.8128 | **0.7798** | 177 | **Near-miss** vs ≥0.80 |
| **M3b** | ProGAN probe → CIFAKE | 0.4215 | 0.4188 | **0.3915** | 2000 | Severe tradeoff |
| **M3c** | soft_router → ProGAN holdout | 0.9700 | 0.9971 | **0.9972** | 400 | Product in-domain OK |
| **M3c** | soft_router → CIFAKE subset | 0.8975 | 0.9647 | **0.9666** | 800 | Restores CIFAKE |
| **M3c** | soft_router → Hemg | 0.6328 | 0.6482 | **0.6885** | 177 | **Worse** than ProGAN-only |
| **M3c** | ProGAN-only Hemg (best) | 0.5706 | 0.8128 | **0.7798** | 177 | Still **&lt; 0.80** |

Sources: `docs/m2_cifake_run.md`, `docs/m3b_progan_probe.md`, `docs/m3c_near_miss.md`, `docs/m3c_dual_probe_summary.json`.

**Product default:** single ProGAN probe when available (best Hemg).  
**Optional `--ensemble soft_router`:** better CIFAKE+ProGAN balance; does **not** clear the Hemg gate.

## Failure modes

| Failure | What happens | Mitigation / honesty |
|---------|--------------|----------------------|
| Cross-generator shift | Hemg / Midjourney-style packs drop ranking | Best shipped Hemg AUC ≈**0.78** (near-miss); not ≥0.80 |
| CIFAKE vs ProGAN domain | ProGAN probe ≈ chance on CIFAKE | Use CIFAKE probe or `soft_router` when CIFAKE-like inputs matter |
| soft_router on Hemg | Mixing CIFAKE head **hurts** Hemg AUC (~0.69) | Prefer single ProGAN probe for cross-gen reporting |
| JPEG / resize | Score drift under social re-encoding | See M3 robustness rows in `docs/m3_crossgen.md` |
| Missing weights | Random head → meaningless `p_fake` | `model_id` …`-untrained`; run `download_weights.py` / train probes |
| Face-swap video | Out of scope | Do not use as a deepfake-video detector |
| Legal / forensic use | Model score ≠ authenticity proof | Notes always say **not court-grade** |

## Tests

```bash
pytest -q
# Schema / CLI JSON / weight preference: tests/test_m4_schema.py
# End-to-end infer smoke skips if UniFD weights absent.
```

## Train / eval (research path)

```bash
# CIFAKE (M2)
python scripts/download_cifake.py
python -m deepfake_detection.train.probe --config configs/train_cifake_subset.yaml

# ProGAN probe (M3b)
python scripts/download_progan_train.py --sources progan_eval frp94_train
python scripts/make_progan_splits.py
python -m deepfake_detection.train.probe --config configs/train_progan_subset.yaml

# Dual-probe tables (M3c)
python -m deepfake_detection.eval.dual_probe \
  --progan-weights weights/progan_clip_vit_l14_linear.pth \
  --cifake-weights weights/cifake_clip_vit_l14_linear.pth \
  --cifake-max 800
```

## Docs

- `docs/plan.md` — project plan  
- `docs/research.md` — prior art  
- `docs/m3_crossgen.md` / `docs/m3b_progan_probe.md` / `docs/m3c_near_miss.md` — milestone write-ups  

## License

MIT for this repository’s original code. Upstream CLIP / UniFD / dataset licenses remain their own.
