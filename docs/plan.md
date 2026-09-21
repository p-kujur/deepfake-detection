# Deepfake Detection System — Project Plan (v0.2 — §1–3 filled)

**Owners**
- **deepfake detection project bot** — problem framing, datasets, SOTA survey, success metrics
- **deepfake detection system coder** — architecture, models, training, inference, eval, API
- **logger bot** (incoming) — log all research + development actions

**Sources**
- Research brief: `/workspace/deepfake-detection-research.md` (also on Drive)
- Shortlist repos: UniFD, DeepfakeBench, NPR

---

## 0. Decision gate (LOCKED 2026-09-21 — recommended defaults)

| # | Decision | Options | Recommendation (coder) | Status |
|---|----------|---------|--------------------------|--------|
| D1 | Primary modality | AIGC images / face video / both | **Start AIGC images**, add face-video branch in phase 2 | **LOCKED** |
| D2 | Product shape | CLI / REST API / Gradio demo | CLI + REST + thin Gradio | **LOCKED** |
| D3 | Stack | PyTorch only? GPU? | PyTorch 2.x, CUDA if available, CPU fallback | **LOCKED** |
| D4 | Data access | CIFAKE / GenImage / FF++ / HF mirrors | Phase 1: CIFAKE; Phase 2: GenImage subset | **LOCKED** |
| D5 | License posture | Research / commercial | Prefer MIT/Apache components for core path | **LOCKED** |

---

## 1. Problem framing *(project bot — filled 18 Sep 2026)*

### Goal statement
Build a **detector** that, given an **image** (v1) or **video frames** (later), outputs a calibrated probability that the media is **manipulated or AI-generated**, plus model id and known caveats — not a claim of absolute authenticity.

### In-scope / out-of-scope

| In scope (v1) | Out of scope (v1) |
|---------------|-------------------|
| Still images: real vs GAN/diffusion/AIGC (e.g. SD, Midjourney-class, ProGAN-family) | Live streaming / realtime camera pipeline |
| Optional: sample video → frame-level scores → simple aggregate | Audio deepfakes, lip-sync, FakeAVCeleb |
| Report `p_fake`, binary label at a fixed threshold, model version | Provenance / C2PA / watermark verification as primary signal |
| Robustness checks under mild JPEG + resize | Forensic attribution (“which generator?”) as a required output |
| Face-swap / facial manipulation as **phase 2 branch** | Legal/identity verification products |

### Threat model (what “fake” means for v1)
- **Positive class (“fake”):** content wholly or largely synthesized by a generative model (GAN or diffusion), or (phase 2) face-region swapped/reenacted with a deepfake method.
- **Negative class (“real”):** camera/photographic or otherwise non-synthesized natural images from standard real corpora (CIFAKE real / ImageNet-aligned reals / FF++ pristine).
- **Attacker assumptions (v1):** adversary uploads a single image (or short clip) after typical social re-encoding (JPEG, downscale). No adaptive white-box attack on our detector in v1 metrics.
- **Not claimed:** detection of heavy Photoshop edits without generative models, screenshots of fakes, or adversarial perturbations crafted against our model.

### Non-goals
- Audio or multimodal AV inconsistency detection
- Realtime streaming / on-device mobile optimization
- Court-grade forensic reports or chain-of-custody tooling
- Training foundation generative models
- Guaranteeing detection of every commercial closed generator on day one (generalization is a **metric**, not a hard guarantee)

### Product framing (aligns with D1–D2 recommendations)
- **v1 default:** Branch A — full-frame **AIGC image** detection (UniFD → NPR ensemble).
- **Phase 2:** Branch B — face detect → crop → face-forgery classifier (DeepfakeBench / MesoNet / Xception).
- Do **not** feed face crops into the AIGC full-frame model or vice versa without an explicit routing rule.

---

## 2. Success metrics *(project bot — filled; coder implements logging)*

Primary reporting: **AP and AUC** on held-out sets; Accuracy only as a secondary, thresholded summary (threshold = 0.5 unless calibration study says otherwise).

| Metric | Target (v1) | How measured | Notes |
|--------|-------------|--------------|-------|
| **In-domain Acc** | ≥ **0.90** | CIFAKE test (or train-split holdout) | Easy sanity bar; not sufficient alone |
| **In-domain AP / AUC** | AP ≥ **0.95**, AUC ≥ **0.95** | Same split | Prefer these over Acc in reports |
| **Cross-generator AUC** | ≥ **0.80** (stretch ≥ 0.85) | Train on CIFAKE or ProGAN/ForenSynths; test on **unseen** GenImage generators (e.g. SD / Midjourney subset) | Core generalization goal (UniFD protocol spirit) |
| **Cross-generator Acc drop** | Drop ≤ **15 pp** vs in-domain Acc | Same protocol | Flag if drop > 20 pp |
| **Latency p95** | ≤ **2.0 s** / image CPU; ≤ **300 ms** / image if CUDA | Single 224–256px path, batch=1, warm model | UniFD CLIP-L is heavy on CPU — document hardware |
| **Robustness (JPEG)** | AUC drop ≤ **0.05** at JPEG q=70; ≤ **0.10** at q=50 | Re-encode test set | Match social-media degradation |
| **Robustness (resize)** | AUC drop ≤ **0.05** for short side 256→128 then back | Resize then model native size | Catches brittle frequency cues |
| **Smoke correctness** | ≥ **18/20** correct on curated real/fake pack | Fixed fixture set in repo | Gate for M1 |
| **API contract** | 100% schema-valid responses in tests | `{p_fake, label, model_id, notes, version}` | M4 acceptance |

### Metric logging contract (for coder)
Every `artifacts/runs/<id>/metrics.json` should include at least:
`dataset`, `split`, `model_id`, `acc`, `ap`, `auc`, `n`, `threshold`, `latency_p50_ms`, `latency_p95_ms`, `aug` (jpeg/resize flags), `git_commit`, `config_hash`.

### Explicit non-metrics for v1
- Generator ID / multi-class generator attribution
- Video temporal consistency scores (until Branch B)
- Human-in-the-loop review throughput

---

## 3. Datasets *(project bot proposes; coder wires loaders)*

### Phase table

| Phase | Dataset | Role | Scale (approx.) | Access / link | Disk / notes |
|-------|---------|------|-----------------|---------------|--------------|
| **P1** | **CIFAKE** | Tiny train/val/test smoke; first Acc/AP/AUC | 120k (32×32) | https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images · paper https://arxiv.org/abs/2303.14126 | Small; good for CI; **not** enough for SOTA claims |
| **P1b** | Curated **20-image smoke pack** | M1 gate (known real/fake) | 20 | Hand-picked; store under `tests/fixtures/` | Mandatory before claiming UniFD wired |
| **P2a** | **CNNDetection / ForenSynths** | Classic GAN train (ProGAN) + multi-GAN tests | 100k+ | https://github.com/peterwang512/CNNDetection | Aligns with UniFD / NPR training lore |
| **P2b** | **GenImage** subset | Cross-generator eval (SD, Midjourney, ADM, …) | Full ~2.7M; use **subset** first | https://github.com/GenImage-Dataset/GenImage · https://arxiv.org/abs/2306.08571 | Disk heavy — download 1–2 generators first |
| **P2c** | **DiffusionForensics** (DIRE) | Optional diffusion-focused eval | Large | Via https://github.com/ZhendongWang6/DIRE | Only if Branch A pushes diffusion-hard cases |
| **P3a** | **FaceForensics++** (c23) | Face-forgery Branch B train/eval | 1k real → multi-method fakes | https://github.com/ondyari/FaceForensics (ToU) | Request/access; compression tiers matter |
| **P3b** | **Celeb-DF v2** | Cross-dataset face-video generalization | ~590 real / ~5.6k fake | https://github.com/yuezunli/celeb-deepfakeforensics | Phase 2+ |
| **P3c** | **DFDC** preview / subset | Hard face-swap stress (optional) | Very large | Kaggle DFDC · paper https://arxiv.org/abs/2006.07397 | Only if compute allows |
| **Later** | WildDeepfake, DF40, Celeb-DF++ | In-the-wild / 2025 generalization | Varies | See research brief | After Branch B baseline exists |

### Split & protocol rules
1. **Never** tune thresholds on the final test generator family.
2. Report **in-domain** and **cross-generator** tables separately.
3. Prefer official train/test splits from each dataset README.
4. Log exact download script + commit hash for reproducibility.
5. Face datasets (P3) only after D1 includes face-video or phase-2 kickoff.

### License / ToU reminders (ties to D5)
- Prefer **MIT/Apache** model code for core path (UniFD MIT; FatFormer Apache; MesoNet Apache).
- **DeepfakeBench** is **CC BY-NC 4.0** — fine for research/course; flag if commercial.
- FF++ / Celeb-DF: respect dataset Terms of Use; no redistributing raw videos in our repo.

---

## 4. Architecture (coder)

```
input (image | video frames)
  → preprocess (resize, normalize; optional face crop for branch B)
  → Model A: UniFD (frozen CLIP ViT-L/14 + linear probe)
  → Model B (phase 2): NPR or EfficientNet head
  → optional fusion (mean / learned weights)
  → calibrated p_fake + model_id + caveats
  → API / CLI / Gradio
```

**Two branches (do not mix blindly)**
- **Branch A — AIGC full-frame** (v1 default): UniFD → later NPR ensemble
- **Branch B — face forgery**: detect → crop → MesoNet/Xception (DeepfakeBench) — phase 2+

### Repo layout (proposed)

```
deepfake-detection/
  README.md
  pyproject.toml / requirements.txt
  configs/           # train + infer YAML
  src/
    data/            # datasets, transforms, splits
    models/          # unifd, npr wrappers
    train/
    eval/
    infer/           # CLI entry
    api/             # FastAPI
  scripts/           # download, smoke tests
  tests/
  docs/              # plan, research links, failure modes
  artifacts/         # weights, metrics (gitignored)
```

---

## 5. Milestones (30-day sketch → engineering tasks)

### M0 — Lock plan (day 0–1)
- [x] Fill §1–3 (project bot)
- [x] Ezra confirms D1–D5 (skipped widgets → recommended defaults locked 2026-09-21)
- [ ] Create Origin / Git repo via cloud agent
- [ ] Logger bot hooks once available

### M1 — Scaffold + pretrained smoke (week 1)
- [x] Repo skeleton + deps (PyTorch, transformers/open_clip, FastAPI, Gradio)
- [x] Integrate UniFD pretrained inference
- [x] Smoke: ≥20 known real/fake images → JSON results (P1b pack; gate ≥18/20)
- [x] Metrics logger (Acc, AP, AUC stubs)

### M2 — Train tiny baseline (week 2)
- [ ] CIFAKE dataloader + Blur/JPEG aug
- [ ] Train linear probe or ResNet baseline
- [ ] Log Acc/AP; save checkpoint + config hash
- [ ] Unit tests for preprocess + schema

### M3 — Cross-test + second head (week 3)
- [ ] Eval on held-out / GenImage subset
- [ ] Add NPR (or EffNet) second head + simple fusion
- [ ] Robustness suite (JPEG, resize)
- [ ] Document failure modes

### M4 — Product wrap (week 4)
- [ ] `detect` CLI + FastAPI `POST /detect`
- [ ] Gradio demo
- [ ] Response schema: `{p_fake, label, model_id, notes, version}`
- [ ] README + reproducibility notes

---

## 6. Engineering standards (coder)

- Python 3.11+, type hints on public APIs
- Config-driven runs (no magic paths in code)
- Every train/eval run writes `artifacts/runs/<id>/metrics.json`
- Deterministic seeds where possible
- No secrets in repo; weights via script download
- Prefer MIT/Apache code paths for core

---

## 7. Coordination protocol

| What | Where |
|------|-------|
| Plan + research | This file + research brief; Drive mirror |
| Daily sync | Room `deepfake detection` |
| Code | Shared repo (Origin/Git) — cloud agent for implementation |
| Action log | Logger bot (when added) |

**Handoffs**
- Project bot → coder: locked framing, dataset URLs, metric targets
- Coder → project bot: smoke results, metric tables, blockers
- Both → logger: milestones + decisions

---

## 8. Open questions for Ezra

1. Image-only for v1, or must video be in the first demo?
2. Any hard deadline (course / demo date)?
3. GPU available (local Colab / cloud)?
4. Prefer Gradio demo, API, or both?
5. Existing GitHub/Origin org to put the repo under?

---

*v0.2.1: §1–3 filled 18 Sep 2026; D1–D5 LOCKED to recommended defaults 21 Sep 2026 (widgets skipped). Shortlist: UniFD → NPR → face branch (DeepfakeBench). M1 scaffold on main.*
