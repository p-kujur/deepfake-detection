---
title: Deepfake Detector
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: AIGC still-image detector (CLIP ViT-L/14 + linear probe)
---

# Deepfake / AIGC still-image detector

Scores whether a still image is AI-generated. Frozen OpenAI **CLIP ViT-L/14**
(downloaded by `open_clip` on first start) plus a small linear probe trained on
ProGAN fakes (`weights/progan_clip_vit_l14_linear.pth`). Optional
`soft_router` mixes the ProGAN and CIFAKE probes.

Output schema (same as the repo CLI / FastAPI): `{p_fake, label, model_id, notes, version}`.

## API

```js
import { Client } from "@gradio/client";
const app = await Client.connect("ezra9426/deepfake-detector");
const r = await app.predict("/predict", { image: fileOrBlob, ensemble: "none" }); // or "soft_router"
console.log(r.data[0]); // {p_fake, label, model_id, notes, version}
```

## Caveats

- Experimental score, **not court-grade** authenticity evidence.
- M3 near-miss: best Hemg cross-generator AUC ≈ **0.78** (target was ≥ 0.80).
- The ProGAN probe often misses modern generators (Midjourney / SDXL / Flux style
  images can score low / "real"). In-domain and cross-generator results are different tables.
- Face video is out of scope for v1. Runs on free CPU: the first request after
  the Space sleeps can take about a minute.

Source: https://github.com/p-kujur/deepfake-detection
