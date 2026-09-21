# Deepfake Detection (M1–M2)

AIGC **still-image** detector scaffold: frozen **CLIP ViT-L/14** + **linear probe** (UniFD-inspired), with CLI, FastAPI, and Gradio stubs.

> **Not** a face-video product yet. Scores are experimental probabilities — **not** authenticity guarantees.

| Item | Choice |
|------|--------|
| Modality (v1) | AIGC still images; face-video later |
| Interfaces | CLI + FastAPI + Gradio |
| Stack | Python 3.11+, PyTorch 2.x |
| License (our code) | MIT |
| Model | UniFD-style CLIP linear probe |

Paper: [UniFD (CVPR 2023)](https://arxiv.org/abs/2302.10174) · Code: [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) (MIT)

## Layout

```
deepfake-detection/
  README.md  LICENSE  pyproject.toml  requirements.txt  .gitignore
  configs/default.yaml
  src/deepfake_detection/
    data/  models/  train/  eval/  infer/detect.py  api/app.py  metrics.py
  scripts/download_weights.py  scripts/make_smoke_fixtures.py  scripts/run_gradio.py
  tests/  docs/plan.md  docs/research.md
  artifacts/   # gitignored run outputs
```

## Install (auth-free)

```bash
cd deepfake-detection
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

No Hugging Face / API tokens required for the default `open_clip` OpenAI CLIP weights (downloaded by the library on first use).

### Optional probe weights

```bash
python scripts/download_weights.py
# Default URL: Hugging Face mirror of UniFD fc_weights.pth (tiny linear head).
# If download fails, the script prints Drive / manual instructions and exits non-zero.
# Place a compatible Linear checkpoint at: weights/unifd_clip_vit_l14_linear.pth
```

Without probe weights the head is **randomly initialized** — CLI/API still run for wiring checks, but **predictions are not meaningful**.

Labeled M1 smoke pack (20 ProGAN/ForenSynths-style images under `tests/fixtures/smoke/`):

```bash
# already committed; rebuild with:
# pip install datasets && python scripts/build_smoke_pack.py
pytest tests/test_smoke_infer.py -q   # needs weights/unifd_clip_vit_l14_linear.pth
```

## Run

### CLI → JSON

```bash
python -m deepfake_detection.infer.detect path/to/image.jpg
# → {"p_fake": ..., "label": "real"|"fake", "model_id": "...", "notes": "...", "version": "0.1.0"}
```

### FastAPI

```bash
uvicorn deepfake_detection.api.app:app --host 0.0.0.0 --port 8000
# POST /detect  multipart field: file
# GET  /health
```

### Gradio stub

```bash
python scripts/run_gradio.py
# or: python -m deepfake_detection.api.gradio_app
```

### Metrics stub / CIFAKE eval

```bash
python -m deepfake_detection.eval.run_eval --dataset stub
python -m deepfake_detection.eval.run_eval --dataset cifake --probe-weights weights/cifake_clip_vit_l14_linear.pth
# writes artifacts/runs/<id>/metrics.json
```

Contract fields: `dataset`, `split`, `model_id`, `acc`, `ap`, `auc`, `n`, `threshold`, `latency_p50_ms`, `latency_p95_ms`, `aug`, `git_commit`, `config_hash`.

## CIFAKE data (M2)

Source: [jordan-bird/CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) (Bird & Lotfi, [arXiv:2303.14126](https://arxiv.org/abs/2303.14126)).

HF mirror used by the loader (official 100k train / 20k test):
[`dragonintelligence/CIFAKE-image-dataset`](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset).

```bash
pip install datasets   # or: pip install -e ".[dev]" after pull
python scripts/download_cifake.py
# caches under datasets/cifake/  (gitignored)
```

Optional Kaggle original (~110 MB): `birdy654/cifake-real-and-ai-generated-synthetic-images`.

### Train linear probe (frozen CLIP)

```bash
# CPU-friendly subset (default) → artifacts/runs/<id>/metrics.json
python -m deepfake_detection.train.probe --config configs/train_cifake_subset.yaml

# Full official split (slow on CPU — hours for feature extract)
python -m deepfake_detection.train.probe --config configs/train_cifake.yaml

# Optional Blur/JPEG train aug
python -m deepfake_detection.train.probe --config configs/train_cifake_subset.yaml --aug
```

Checkpoints (gitignored):

- `artifacts/checkpoints/<run_id>_cifake_probe.pth`
- `weights/cifake_clip_vit_l14_linear.pth` (latest convenience copy)

Aspirational in-domain targets (plan): Acc ≥ 0.90, AP/AUC ≥ 0.95 — report actual metrics honestly; subset runs are labeled `dataset: cifake_subset`.

## Tests

```bash
pytest -q
# End-to-end infer smoke skips automatically if weights are absent.
```

## Caveats

- **v1 scope:** full-frame AIGC / synthetic still images. Do not treat as a face-swap video detector.
- **Generalization:** cross-generator AUC is the real goal; in-domain accuracy alone is insufficient (see `docs/plan.md`).
- **Robustness:** social re-encoding (JPEG, resize) can shift scores; log metrics under those augs.
- **Weights:** UniFD checkpoints live on Google Drive and may require manual download; see `scripts/download_weights.py`.
- **Datasets:** CIFAKE via `scripts/download_cifake.py` (not committed). GenImage / FF++ come later.
- **Legal / ethics:** output is a model score with caveats — not a court-grade authenticity claim.

## Docs

- `docs/plan.md` — project plan (problem, metrics, datasets, architecture)
- `docs/research.md` — prior art survey (repos, papers, datasets)

## License

MIT for this repository’s original code. Upstream CLIP / UniFD / dataset licenses remain their own — respect those terms when downloading weights or data.
