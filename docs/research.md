# Deepfake / AI-Generated / Manipulated Media Detection — Research Brief

**Prepared for:** Ezra (student / project bot)  
**Date:** 18 Sep 2026 (Asia/Calcutta)  
**Project goal:** Analyse images or videos and predict whether they may be manipulated or AI-generated.  
**Method:** Links found via WebSearch + Firecrawl search/scrape. Star counts are approximate as of scrape time (Sep 2026) and may drift. Prefer official GitHub + arXiv links below.

---

## 1. Prior open-source projects / GitHub

Aim: classics **and** recent (2023–2026) covering face-swap deepfakes **and** general AI-generated (GAN / diffusion) detection.

| # | Name | Modality | One-line | GitHub | Stars (approx.) | Key technique | License | Why relevant |
|---|------|----------|----------|--------|-----------------|---------------|---------|--------------|
| 1 | **FaceForensics++** | Image + video (frames) | Dataset + Xception baseline for facial manipulation detection | https://github.com/ondyari/FaceForensics | ~2.8k | Xception CNN on face crops; compression benchmarks | Code MIT; data ToU | Seminal face-swap / Face2Face / NeuralTextures stack |
| 2 | **Awesome-Deepfakes-Detection** | N/A (curated list) | Papers, datasets, tools for deepfake detection | https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection | ~1.8k | Curated bibliography | See repo LICENSE | Best navigation map; updated through 2025 (e.g. Celeb-DF++) |
| 3 | **DeepfakeBench** | Image + video | Unified benchmark with many reimplemented detectors + weights | https://github.com/SCLBD/DeepfakeBench | ~1.1k | Xception, MesoNet, EfficientNet, F3Net, CLIP, ViT, video models | CC BY-NC 4.0 | Best “lab in a box” for face forgeries (NeurIPS 2023 D&B); still updated (Effort / DF40 notes) |
| 4 | **CNNDetection** | Image | Detect CNN/GAN-generated images beyond faces | https://github.com/peterwang512/CNNDetection | ~1.0k | ResNet-50 + Blur/JPEG data augmentation | See LICENSE.txt | Classic “universal” GAN detector; data reused by UniFD / NPR / FatFormer |
| 5 | **Awesome-AIGC-Image-Video-Detection** | N/A (curated list) | AIGC image/video detection papers, datasets, tools | https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection | ~454 | Curated AIGC forensics list | Apache-2.0 | Strong for 2025–2026 diffusion / commercial generators (active Sep 2026) |
| 6 | **DIRE** | Image | Diffusion-generated image detection via reconstruction error | https://github.com/ZhendongWang6/DIRE | ~400 | Diffusion Reconstruction Error + CNN | Public code (repo archived Jul 2025) | Core diffusion-forensics idea + DiffusionForensics data |
| 7 | **UniversalFakeDetect (UniFD)** | Image | Cross-generator fake image detection with frozen CLIP | https://github.com/WisconsinAIVision/UniversalFakeDetect | ~378 | CLIP ViT-L/14 features + linear probe | MIT | Strong, simple baseline for GAN→diffusion generalization |
| 8 | **NPR** | Image | Generalizable synthetic-image detection via upsampling cues | https://github.com/chuangchuangtan/NPR-DeepfakeDetection | ~312 | Neighboring Pixel Relationships (NPR) + CNN | See repo | CVPR 2024; HF demo available |
| 9 | **MesoNet** | Video (frames) | Compact facial forgery detector | https://github.com/DariusAf/MesoNet | ~305 | Shallow “mesoscopic” CNNs | Apache-2.0 | Classic lightweight face-deepfake baseline |
| 10 | **FatFormer** | Image | Forgery-aware CLIP adaptation for synthetic image detection | https://github.com/Michel-liu/FatFormer | ~136 | CLIP + forgery-aware adapter (spatial + frequency) + text alignment | Apache-2.0 | CVPR 2024 path beyond frozen UniFD |
| 11 | **Capsule-Forensics-v2** | Image + video | Capsule-network forgery detector with FF++ checkpoints | https://github.com/nii-yamagishilab/Capsule-Forensics-v2 | (verify on page) | Capsule nets + pretrained CNN features | See repo | Classic alternate architecture; still useful pedagogy |
| 12 | **GenImage** | Image (dataset + baselines) | Million-scale AI-generated image detection benchmark | https://github.com/GenImage-Dataset/GenImage | (verify on page) | Dataset + ResNet/DeiT/Swin/CNNSpot/F3Net/… | See repo | Standard AIGC image train/test suite |

### Hugging Face (notable, verified via search)

- NPR demo Space: https://huggingface.co/spaces/tancc/Generalizable_Deepfake_Detection-NPR-CVPR2024  
- Example community models (useful for demos; quality varies):  
  - https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model  
  - https://huggingface.co/spaces/FaceOnLive/Deepfake-Detector  

### Verified examples from the original checklist

| Example to verify | Status |
|-------------------|--------|
| FaceForensics++, MesoNet, Xception baselines | **Confirmed** (ondyari/FaceForensics, DariusAf/MesoNet, DeepfakeBench Xception) |
| EfficientNet / ViT detectors | **Confirmed** inside DeepfakeBench |
| UniversalFakeDetect, DIRE, NPR, FatFormer, CLIP-based | **Confirmed** (repos + papers above) |
| Awesome-Deepfakes-Detection lists | **Confirmed** (Daisy-Zhang + ant-research AIGC list) |
| Microsoft / Facebook DFDC “official detector” monorepo | **Not found as a single OSS product**; DFDC is mainly a **dataset/challenge**. Microsoft Video Authenticator: **no strong public clone found** — mark uncertain |

---

## 2. Key research papers

| # | Title | Authors / year | Venue | Link | Takeaway (1–2 sentences) | Image / video | Build relevance |
|---|-------|----------------|-------|------|--------------------------|---------------|-----------------|
| 1 | FaceForensics++: Learning to Detect Manipulated Facial Images | Rössler et al., 2019 | ICCV | https://arxiv.org/abs/1901.08971 | Large facial manipulation dataset + strong Xception baseline under compression. | Both (frame-level) | Default face-forgery data + baseline |
| 2 | MesoNet: a Compact Facial Video Forgery Detection Network | Afchar et al., 2018 | WIFS | https://arxiv.org/abs/1809.00888 | Tiny CNNs targeting mesoscopic artifacts detect Deepfake/Face2Face well. | Video frames | Easy first face-video prototype |
| 3 | Capsule-Forensics: Using Capsule Networks to Detect Forged Images and Videos | Nguyen et al., 2019 | ICASSP | https://arxiv.org/abs/1810.11215 | Capsules with dynamic routing handle diverse forgery types. | Both | Architecture alternative to plain CNNs |
| 4 | The DeepFake Detection Challenge (DFDC) Dataset | Dolhansky et al., 2020 | arXiv | https://arxiv.org/abs/2006.07397 | Large Meta/Facebook challenge dataset; hard, diverse face-swap videos. | Video | Scale + realism stress test |
| 5 | CNN-generated images are surprisingly easy to spot… for now | Wang et al., 2020 | CVPR | https://arxiv.org/abs/1912.11035 | ProGAN-trained ResNet + Blur/JPEG aug generalizes across many GANs. | Image | Foundation of general AIGC image detection |
| 6 | Towards Universal Fake Image Detectors that Generalize Across Generative Models (UniFD) | Ojha, Li, Lee, 2023 | CVPR | https://arxiv.org/abs/2302.10174 | Frozen CLIP features + linear head beat fully trained CNNs on unseen generators (incl. diffusion). | Image | Best modern starter for AI-generated images |
| 7 | DIRE for Diffusion-Generated Image Detection | Wang et al., 2023 | ICCV | https://arxiv.org/abs/2303.09295 | Diffusion reconstruction error separates real vs diffusion-generated images. | Image | Diffusion-specific feature design |
| 8 | Rethinking the Up-Sampling Operations… (NPR) | Tan et al., 2024 | CVPR | https://arxiv.org/abs/2312.10461 | Neighboring-pixel / upsampling artifacts are a transferable forgery cue. | Image | Strong generalizable image detector |
| 9 | Forgery-aware Adaptive Transformer (FatFormer) | Liu et al., 2024 | CVPR | https://arxiv.org/abs/2312.16649 | Adapt CLIP with spatial+frequency forgery adapters and language-guided alignment. | Image | Upgrade path from frozen UniFD |
| 10 | DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection | Yan et al., 2023 | NeurIPS D&B | https://arxiv.org/abs/2307.01426 | Standardizes data pipelines, metrics, and many SOTA reimplementations. | Image + video | Required before claiming fair comparisons |
| 11 | GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image | Zhu / Bai et al., 2023 | NeurIPS (dataset track) | https://arxiv.org/abs/2306.08571 | Million-scale ImageNet-aligned real vs Midjourney/SD/ADM/etc. | Image | Primary AIGC image benchmark |
| 12 | Leveraging Frequency Analysis for Deep Fake Image Recognition | Frank et al., 2020 | ICML | https://arxiv.org/abs/2003.08685 | GAN upsampling leaves detectable DFT / spectral fingerprints. | Image | Motivation for frequency / NPR-style methods |
| 13 | Watch Your Up-Convolution… | Durall et al., 2020 | CVPR | https://openaccess.thecvf.com/content_CVPR_2020/html/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.html | Transposed-conv generators fail to match real spectral distributions. | Image | Explains *why* frequency detectors work |
| 14 | **Deepfake Detection: A Systematic Literature Review** *(survey)* | Rana, Nobi, Murali, Sung, 2022 | IEEE Access | https://doi.org/10.1109/ACCESS.2022.3154404 | Systematic review of 100+ studies (2018–2020); DL methods dominate. | Both | Required survey / literature map |
| 15 | Combating Digitally Altered Images: Deepfake Detection *(recent survey)* | 2025 | arXiv | https://arxiv.org/abs/2508.16975 | Broader review across image / video / audio / hybrid detection. | Multi | Updated survey context for mid-2020s |

**Audio-visual (brief):** FakeAVCeleb (https://github.com/DASH-Lab/FakeAVCeleb) and AV-inconsistency papers (see Awesome lists) matter if Ezra later adds audio. For v1, RGB image/frame classifiers are enough.

---

## 3. Datasets commonly used

| Dataset | Type | Focus | Rough scale | Public link |
|---------|------|-------|-------------|-------------|
| FaceForensics++ | Video / frames | DF, Face2Face, FaceSwap, NeuralTextures (+FaceShifter) | 1k real → multi-method fakes | https://github.com/ondyari/FaceForensics |
| Google/Jigsaw DeepFakeDetection | Video | Actor face swaps | ~3k+ manip. videos | Via FaceForensics download flow |
| Celeb-DF (v2) | Video | High-quality celebrity swaps | ~590 real / ~5.6k fake | https://github.com/yuezunli/celeb-deepfakeforensics |
| DFDC | Video | Large challenge set | ~100k+ videos | Paper https://arxiv.org/abs/2006.07397 · Data https://www.kaggle.com/c/deepfake-detection-challenge/data |
| WildDeepfake | Video / faces | In-the-wild internet deepfakes | ~7k face sequences | Paper https://arxiv.org/abs/2101.01456 · HF https://huggingface.co/datasets/xingjunm/WildDeepfake |
| DeeperForensics-1.0 | Video | Robustness / perturbations | Large | https://github.com/EndlessSora/DeeperForensics-1.0 |
| CNNDetection / ForenSynths | Image | ProGAN train; many GAN tests | 100k+ | https://github.com/peterwang512/CNNDetection |
| GenImage | Image | Midjourney, SD, ADM, BigGAN, GLIDE, VQDM, Wukong… | ~2.7M | https://github.com/GenImage-Dataset/GenImage · https://arxiv.org/abs/2306.08571 |
| CIFAKE | Image | CIFAR-10 real vs SD1.4 fakes (32×32) | 120k | https://arxiv.org/abs/2303.14126 · https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images |
| DiffusionForensics (DIRE) | Image | Multiple diffusion generators | Large | Links in https://github.com/ZhendongWang6/DIRE |
| DF40 | Image + video | 40 face forgery techniques | 0.1M+ videos / 1M+ images (claimed) | https://github.com/YZY-stack/DF40 · https://arxiv.org/abs/2406.13495 |
| Celeb-DF++ | Video | 2025 generalization benchmark | ~53k fake | https://arxiv.org/abs/2507.18015 · https://github.com/OUC-VAS/Celeb-DF-PP |

---

## 4. Architecture patterns that work well today

### Two problem families (do not mix blindly)

1. **Face-swap / facial manipulation** (FF++, Celeb-DF, DFDC) → face detect → crop → classifier; video temporal models help.  
2. **General AI-generated media** (GAN / diffusion / Midjourney / SD) → full-frame classifiers; CLIP / frequency / upsampling cues transfer better than face-only models.

### First prototype vs stronger systems

| Stage | Recommendation |
|-------|----------------|
| **First prototype (image AIGC)** | UniFD (frozen CLIP + linear head) **or** CNNDetection ResNet; train/eval on CIFAKE then GenImage / CNNDetection tests |
| **First prototype (face video)** | ffmpeg sample frames → RetinaFace/MTCNN crop → MesoNet or Xception (FaceForensics / DeepfakeBench) |
| **Stronger image AIGC** | NPR and/or FatFormer; optional DIRE if compute allows diffusion reconstruction |
| **Stronger face video** | DeepfakeBench SOTA entries (SBI, LSDA, Effort, FTCN, VideoMAE, …); always report cross-dataset AUC |
| **Production-ish bot** | Ensemble: CLIP probe + NPR + (optional) face-crop CNN; expose calibrated `p_fake` + model id + caveats |

### Practical pipeline notes

- **Images:** report Acc **and** AP/AUC; stress-test JPEG, resize, social-media re-encoding.  
- **Videos:** aggregate frame logits (mean/max); tracking reduces ID flicker; temporal nets help but raise IO cost.  
- **Generalization > in-domain Acc:** train on one generator family, test on unseen (UniFD/NPR protocol).  
- **Face vs full-frame:** face crops kill general AIGC cues; full-frame models fail on subtle local face swaps — pick the branch that matches the product goal.

---

## 5. Suggested shortlist for *this* project

### Top 3 GitHub repos to clone first

1. **https://github.com/WisconsinAIVision/UniversalFakeDetect** — modern AI-generated **image** detector; MIT; pretrained CLIP probe.  
2. **https://github.com/SCLBD/DeepfakeBench** — if **face image/video** manipulation is in scope; many baselines + weights.  
3. **https://github.com/chuangchuangtan/NPR-DeepfakeDetection** — CVPR 2024 generalizable detector + HF demo; strong second model for ensembling.

Honorable: FaceForensics++ (data/Xception), CNNDetection (classic GAN baseline), Daisy-Zhang + ant-research awesome lists (navigation).

### Top 3 papers to read first

1. **Ojha et al., UniFD, CVPR 2023** — https://arxiv.org/abs/2302.10174  
2. **Wang et al., CNNDetection, CVPR 2020** — https://arxiv.org/abs/1912.11035  
3. **Rössler et al., FaceForensics++, ICCV 2019** — https://arxiv.org/abs/1901.08971  

Next: DIRE (https://arxiv.org/abs/2303.09295) for diffusion focus; DeepfakeBench paper (https://arxiv.org/abs/2307.01426) before claiming SOTA numbers; Rana et al. survey (https://doi.org/10.1109/ACCESS.2022.3154404) for breadth.

---

## Suggested 30-day build plan

| Week | Milestone |
|------|-----------|
| 1 | Choose modality (AIGC image vs face video). Clone UniFD **or** DeepfakeBench. Run pretrained inference on ~20 known real/fake samples. |
| 2 | Train tiny baseline on CIFAKE or FF++ c23 frames. Log Acc/AP; add JPEG augmentation. |
| 3 | Cross-test GenImage subset or Celeb-DF. Add NPR (or EfficientNet) second head; simple score fusion. |
| 4 | Wrap API: upload → optional face-crop → `{p_fake, model, notes}`; document failure modes (compression, unseen commercial models). |

---

## Verification log

| Item | Status |
|------|--------|
| FaceForensics, MesoNet, CNNDetection, DIRE, UniFD, NPR, FatFormer, DeepfakeBench, both Awesome lists | **Verified** (GitHub scrape and/or search hits) |
| Core arXiv IDs listed above | **Verified** via search results and/or official READMEs |
| Star counts | **Approximate** (Sep 2026 scrape) |
| Microsoft Video Authenticator open-source | **Not found / uncertain** |
| Capsule-Forensics-v2 & GenImage star counts | Repos **confirmed**; stars not re-scraped in final pass |
| Emerging 2026 AIGC benches in Awesome-AIGC | Present on list; treat individual new benches as **emerging** until used |

---

## Quick link index

- FaceForensics++: https://github.com/ondyari/FaceForensics  
- DeepfakeBench: https://github.com/SCLBD/DeepfakeBench · https://arxiv.org/abs/2307.01426  
- UniFD: https://github.com/WisconsinAIVision/UniversalFakeDetect · https://arxiv.org/abs/2302.10174  
- CNNDetection: https://github.com/peterwang512/CNNDetection · https://arxiv.org/abs/1912.11035  
- DIRE: https://github.com/ZhendongWang6/DIRE · https://arxiv.org/abs/2303.09295  
- NPR: https://github.com/chuangchuangtan/NPR-DeepfakeDetection · https://arxiv.org/abs/2312.10461  
- FatFormer: https://github.com/Michel-liu/FatFormer · https://arxiv.org/abs/2312.16649  
- MesoNet: https://github.com/DariusAf/MesoNet · https://arxiv.org/abs/1809.00888  
- GenImage: https://github.com/GenImage-Dataset/GenImage · https://arxiv.org/abs/2306.08571  
- Awesome (face deepfakes): https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection  
- Awesome (AIGC): https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection  
- Survey (IEEE Access 2022): https://doi.org/10.1109/ACCESS.2022.3154404  

---

*End of brief. Clone the shortlist and read UniFD + CNNDetection + FaceForensics++ before expanding scope.*
