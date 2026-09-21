# Deepfake Detection (M1 scaffold)

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

Smoke image fixtures (tiny synthetic PNGs, not a labeled eval set):

```bash
python scripts/make_smoke_fixtures.py
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

### Metrics stub

```bash
python -m deepfake_detection.eval.run_eval
# writes artifacts/runs/<id>/metrics.json
```

Contract fields: `dataset`, `split`, `model_id`, `acc`, `ap`, `auc`, `n`, `threshold`, `latency_p50_ms`, `latency_p95_ms`, `aug`, `git_commit`, `config_hash`.

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
- **No huge datasets** are bundled. CIFAKE / GenImage / FF++ come later via separate download scripts.
- **Legal / ethics:** output is a model score with caveats — not a court-grade authenticity claim.

## Docs

- `docs/plan.md` — project plan (problem, metrics, datasets, architecture)
- `docs/research.md` — prior art survey (repos, papers, datasets)

## License

MIT for this repository’s original code. Upstream CLIP / UniFD / dataset licenses remain their own — respect those terms when downloading weights or data.
