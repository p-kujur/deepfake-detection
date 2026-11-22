# Modal deployment (inference API)

The detector (frozen CLIP ViT-L/14 + ProGAN linear probe, optional CIFAKE head for
`soft_router`) runs on [Modal](https://modal.com) CPU containers. The Vercel page in
`web/` calls it with a multipart `fetch()`.

## Deploy

From the repo root, with the probe weights present in `weights/`
(`progan_clip_vit_l14_linear.pth`, `cifake_clip_vit_l14_linear.pth`; gitignored):

```bash
pip install modal
export MODAL_TOKEN_ID=...      # from modal.com → Settings → API tokens
export MODAL_TOKEN_SECRET=...
modal deploy modal_app/app.py
```

The first deploy builds the image (CPU torch + open_clip, and bakes the ~1.7 GB CLIP
ViT-L/14 openai weights into the image), so it takes several minutes. Later deploys
reuse cached layers.

Endpoint URL (printed by `modal deploy`):

```
https://<workspace>--deepfake-detector-detector-web.modal.run
```

Then set `DEFAULT_API` in `web/app.js` to that URL (or test with `?api=<url>`).

## API

- `GET /health` → `{status, model_id, ensembles, version}`
- `POST /detect` multipart: `file` (image, max 10 MB), optional `ensemble` = `none` | `soft_router`
  → `{p_fake, label, model_id, notes, version}` (same schema and caveat notes as the CLI)

```bash
curl -F file=@tests/fixtures/smoke/fake/fake_01.png \
     https://<workspace>--deepfake-detector-detector-web.modal.run/detect
```

CORS allows all origins.

## Cost guards

`cpu=2`, `memory=4096` MB, `max_containers=2`, `scaledown_window=300` s (containers
stop 5 minutes after the last request, so idle costs nothing). A cold start loads CLIP
from the image, roughly 20–60 s. `@modal.concurrent(max_inputs=4)` per container.

## Local run (no Modal)

```bash
PYTHONPATH=src:modal_app uvicorn --factory dfd_service:create_local_app --port 8000
```

Files: `app.py` (Modal image/app/class), `dfd_service.py` (FastAPI app + detector, no
Modal dependency).

## Caveats

Experimental score, not court-grade. M3 near-miss: best Hemg cross-generator AUC ≈ 0.78.
The ProGAN probe often misses modern generators (Midjourney / SDXL / Flux).
