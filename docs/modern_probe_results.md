# Modern probe results (final)

train n=12900 (real=6174, fake=6726), with_progan=True, C=0.2

| eval set | head | n | AUC | AP | acc@0.5 | real acc (1-FPR) | fake acc (TPR) |
|---|---|---|---|---|---|---|---|
| openfake_test_ood_clean | progan (old default) | 2717 | 0.5989 | 0.6404 | 0.4476 | 0.9933 | 0.0158 |
| openfake_test_ood_clean | new | 2717 | 0.9357 | 0.9521 | 0.8565 | 0.8775 | 0.8398 |
| openfake_test_ood_clean | soft_router (old) | 2717 | 0.6562 | 0.6933 | 0.5576 | 0.8550 | 0.3223 |
| openfake_test_ood_social_jpeg75 | progan (old default) | 2717 | 0.6105 | 0.6399 | 0.4461 | 0.9983 | 0.0092 |
| openfake_test_ood_social_jpeg75 | new | 2717 | 0.9478 | 0.9603 | 0.8679 | 0.9183 | 0.8279 |
| openfake_test_ood_social_jpeg75 | soft_router (old) | 2717 | 0.6317 | 0.6663 | 0.5197 | 0.8950 | 0.2228 |
| hemg_clean | progan (old default) | 177 | 0.7796 | 0.8126 | 0.5706 | 0.9500 | 0.2577 |
| hemg_clean | new | 177 | 0.8274 | 0.8510 | 0.7288 | 0.4625 | 0.9485 |
| hemg_clean | soft_router (old) | 177 | 0.6885 | 0.6482 | 0.6328 | 0.5875 | 0.6701 |
| hemg_social_jpeg75 | progan (old default) | 177 | 0.7599 | 0.7998 | 0.5706 | 0.9625 | 0.2474 |
| hemg_social_jpeg75 | new | 177 | 0.8139 | 0.8291 | 0.7119 | 0.5125 | 0.8763 |
| hemg_social_jpeg75 | soft_router (old) | 177 | 0.6885 | 0.6496 | 0.6158 | 0.6625 | 0.5773 |
| progan_holdout | progan (old default) | 400 | 0.9997 | 0.9997 | 0.9875 | 1.0000 | 0.9750 |
| progan_holdout | new | 400 | 0.9983 | 0.9984 | 0.9750 | 0.9550 | 0.9950 |
| progan_holdout | soft_router (old) | 400 | 0.9971 | 0.9971 | 0.9700 | 0.9850 | 0.9550 |
| smoke | progan (old default) | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| smoke | new | 20 | 1.0000 | 1.0000 | 0.9500 | 0.9000 | 1.0000 |
| smoke | soft_router (old) | 20 | 1.0000 | 1.0000 | 0.9500 | 0.9000 | 1.0000 |
| cifake_test | progan (old default) | 2000 | 0.3915 | 0.4188 | 0.4215 | 0.1437 | 0.7165 |
| cifake_test | new | 2000 | 0.4550 | 0.4641 | 0.5070 | 0.9369 | 0.0505 |
| ezra_grok_imagine_image | progan (old default) | 1 | | | | | p_fake=0.0499 |
| ezra_grok_imagine_image | new | 1 | | | | | p_fake=0.8950 |
| ezra_grok_imagine_image | soft_router (old) | 1 | | | | | p_fake=0.2390 |

## per-model acc@0.5 — openfake_test_ood_clean

| model | n | old progan | new |
|---|---|---|---|
| fake:aurora-20-1-25 | 40 | 0.000 | 0.725 |
| real:docci | 600 | 0.995 | 0.783 |
| fake:ernie-image | 62 | 0.048 | 0.935 |
| fake:ernie-image-turbo | 100 | 0.010 | 0.900 |
| fake:flux.2-klein-9b | 100 | 0.000 | 0.760 |
| fake:frames-23-1-25 | 40 | 0.025 | 0.750 |
| fake:gpt-image-1.5 | 100 | 0.040 | 0.960 |
| fake:gpt-image-2 | 86 | 0.035 | 0.942 |
| fake:halfmoon-4-4-25 | 33 | 0.030 | 0.697 |
| fake:ideogram-2.0 | 41 | 0.000 | 0.561 |
| fake:illustrious | 100 | 0.000 | 0.980 |
| real:imagenet | 600 | 0.992 | 0.972 |
| fake:lumina-17-2-25 | 71 | 0.042 | 0.930 |
| fake:midjourney-7 | 100 | 0.000 | 0.660 |
| fake:nano-banana-pro | 55 | 0.036 | 0.964 |
| fake:recraft-v2 | 37 | 0.000 | 0.703 |
| fake:recraft-v3 | 100 | 0.000 | 0.480 |
| fake:seedream-v5.0 | 61 | 0.000 | 0.967 |
| fake:sora-2 | 91 | 0.044 | 0.912 |
| fake:veo-3 | 100 | 0.010 | 0.900 |
| fake:wan-video-2.5 | 100 | 0.000 | 0.950 |
| fake:z-image-turbo | 100 | 0.010 | 0.840 |

## per-model acc@0.5 — openfake_test_ood_social_jpeg75

| model | n | old progan | new |
|---|---|---|---|
| fake:aurora-20-1-25 | 40 | 0.000 | 0.750 |
| real:docci | 600 | 0.998 | 0.848 |
| fake:ernie-image | 62 | 0.016 | 0.919 |
| fake:ernie-image-turbo | 100 | 0.010 | 0.900 |
| fake:flux.2-klein-9b | 100 | 0.000 | 0.790 |
| fake:frames-23-1-25 | 40 | 0.025 | 0.600 |
| fake:gpt-image-1.5 | 100 | 0.040 | 0.910 |
| fake:gpt-image-2 | 86 | 0.012 | 0.942 |
| fake:halfmoon-4-4-25 | 33 | 0.030 | 0.636 |
| fake:ideogram-2.0 | 41 | 0.000 | 0.488 |
| fake:illustrious | 100 | 0.000 | 0.980 |
| real:imagenet | 600 | 0.998 | 0.988 |
| fake:lumina-17-2-25 | 71 | 0.014 | 0.958 |
| fake:midjourney-7 | 100 | 0.000 | 0.690 |
| fake:nano-banana-pro | 55 | 0.018 | 0.891 |
| fake:recraft-v2 | 37 | 0.000 | 0.784 |
| fake:recraft-v3 | 100 | 0.000 | 0.490 |
| fake:seedream-v5.0 | 61 | 0.000 | 0.951 |
| fake:sora-2 | 91 | 0.022 | 0.846 |
| fake:veo-3 | 100 | 0.000 | 0.860 |
| fake:wan-video-2.5 | 100 | 0.000 | 0.920 |
| fake:z-image-turbo | 100 | 0.010 | 0.880 |
