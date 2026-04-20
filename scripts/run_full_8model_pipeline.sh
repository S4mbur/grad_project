#!/usr/bin/env bash
set -euo pipefail

cd /home/byalc/phase1_project
source /home/byalc/phase1_env/bin/activate

mkdir -p /mnt/d/skin_cancer_project/cache
mkdir -p /home/byalc/phase1_project/results

python scripts/rebuild_4class_cache.py --delete-corrupt-melanoma
python scripts/train_all_models_v3.py