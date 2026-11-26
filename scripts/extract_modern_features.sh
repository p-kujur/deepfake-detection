#!/usr/bin/env bash
# Modern-probe feature extraction chain (idempotent + resumable: parts/ checkpoints, skips finished caches).
set -x
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
P=${PYTHON:-python}; FC=artifacts/feature_cache/modern; R=datasets/openfake_raw
run() { local final=$1; shift; [ -f "$final" ] && { echo "skip $final"; return; }; $P scripts/extract_clip_features.py "$@"; }
run $FC/hemg_social.pt --source folder --inputs datasets/crossgen/hemg_wild --views clean social --out $FC/hemg.pt
run $FC/progan_test_social.pt --source folder --inputs datasets/progan_forensynths/splits/test --views clean social --out $FC/progan_test.pt
run $FC/of_train_aug.pt --source openfake --inputs $R/core/validation-00000-of-00015.parquet $R/core/validation-00001-of-00015.parquet $R/core/validation-00002-of-00015.parquet --max-real-per-source 3000 --view aug --out $FC/of_train_aug.pt
run $FC/progan_train_aug.pt --source folder --inputs datasets/progan_forensynths/splits/train --view aug --out $FC/progan_train_aug.pt
run $FC/of_test_social.pt --source openfake --inputs $R/core/test-00000-of-00013.parquet $R/core/test-00007-of-00013.parquet --max-per-model 100 --max-real-per-source 600 --views clean social --out $FC/of_test.pt
run $FC/of_reddit_clean.pt --source openfake --inputs $R/reddit/test-00000-of-00005.parquet --max-per-model 600 --max-real-per-source 600 --view clean --out $FC/of_reddit_clean.pt
echo ALLDONE
