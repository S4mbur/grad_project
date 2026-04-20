#!/usr/bin/env bash
set -euo pipefail

cd /home/byalc/phase1_project
source /home/byalc/phase1_env/bin/activate

if ! mountpoint -q /mnt/d; then
  echo "D: drive is not mounted at /mnt/d. Mount it first with:"
  echo "  wsl.exe -u root -d Ubuntu-22.04 bash -lc 'mkdir -p /mnt/d && mount -t drvfs D: /mnt/d'"
  exit 1
fi

if [ ! -d /mnt/d/skin_cancer_project/cache ] || [ ! -d /mnt/d/skin_cancer_project/datasets ]; then
  echo "Expected D: project paths are missing under /mnt/d/skin_cancer_project"
  exit 1
fi

log_dir="/home/byalc/phase1_project/results"
mkdir -p "$log_dir"

echo "[1/2] Training missing normal v3 experiments for ConvNeXt-Small, ConvNeXt-Base, DINOv2-base, and Phikon"
python scripts/train_all_models_v3.py \
  --models ConvNeXt-Small ConvNeXt-Base DINOv2-base Phikon \
  --experiments mel_boost_7x focal_g3 cost_sensitive_strong

echo "[2/2] Training full normal v3 experiment set for UNI and CONCH"
python scripts/train_all_models_v3.py \
  --models UNI CONCH
