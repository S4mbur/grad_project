#!/usr/bin/env python3
"""
=============================================================================
  Extract tiles + features for NEW TCGA-SKCM melanoma slides
  ===========================================================
  This script:
    1. Finds new TCGA-SKCM slides that don't have tiles yet
    2. Extracts tiles (256×256, tissue filter)
    3. Extracts features with ALL 6 backbones
  
  Run after downloading new TCGA slides with gdc-client.
  Usage: python scripts/extract_new_melanoma.py
=============================================================================
"""
import os
import sys
import random
import logging
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
TCGA_DIR = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
TILE_DIR = Path("/home/byalc/phase1_project/data/tiles_4class/class_3")  # Melanoma = class 3
FEATURE_BASE = Path("/home/byalc/phase1_project/data")

TILE_SIZE = 256
MAX_TILES = 200
TISSUE_THRESHOLD = 0.5
BATCH_SIZE = 64
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = [
    {"name": "ResNet18",       "type": "torchvision", "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/resnet18.pth",       "feat_dim": 512,  "loader": "resnet18",       "feat_dir": "features_4class_resnet18"},
    {"name": "ResNet50",       "type": "torchvision", "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/resnet50.pth",       "feat_dim": 2048, "loader": "resnet50",       "feat_dir": "features_4class_resnet50"},
    {"name": "ConvNeXt-Small", "type": "torchvision", "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/convnext_small.pth", "feat_dim": 768,  "loader": "convnext_small", "feat_dir": "features_4class_convnext_small"},
    {"name": "ConvNeXt-Base",  "type": "torchvision", "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/convnext_base.pth",  "feat_dim": 1024, "loader": "convnext_base",  "feat_dir": "features_4class_convnext_base"},
    {"name": "DINOv2-base",    "type": "dinov2",      "weights_path": "/mnt/d/skin_cancer_project/models/vision/dinov2-base",             "feat_dim": 768,  "loader": "dinov2",         "feat_dir": "features_4class_dinov2_base"},
    {"name": "Phikon",         "type": "phikon",      "weights_path": "/mnt/d/skin_cancer_project/models/pathology/phikon",               "feat_dim": 768,  "loader": "phikon",         "feat_dir": "features_4class_phikon"},
]

# ============================================================
# STEP 1: FIND NEW SLIDES
# ============================================================
def find_new_slides():
    """Find TCGA slides that don't have tiles extracted yet."""
    all_svs = sorted(TCGA_DIR.glob("*.svs"))
    logger.info(f"  Total SVS files in TCGA dir: {len(all_svs)}")

    new_slides = []
    existing_slides = []
    for svs in all_svs:
        slide_id = svs.stem
        tile_dir = TILE_DIR / slide_id
        if tile_dir.exists() and len(list(tile_dir.glob("*.png"))) >= 10:
            existing_slides.append(svs)
        else:
            new_slides.append(svs)

    logger.info(f"  Already have tiles: {len(existing_slides)}")
    logger.info(f"  New (need tiles): {len(new_slides)}")
    return new_slides

# ============================================================
# STEP 2: TILE EXTRACTION
# ============================================================
def extract_tiles(slides):
    """Extract tissue tiles from new WSI slides."""
    logger.info(f"\n{'━'*60}")
    logger.info(f"TILE EXTRACTION — {len(slides)} new slides")
    logger.info(f"{'━'*60}")

    try:
        import openslide
    except ImportError:
        logger.error("openslide not installed! pip install openslide-python")
        sys.exit(1)

    TILE_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    extracted = 0
    errors = 0
    total = len(slides)

    for idx, svs in enumerate(slides, 1):
        slide_id = svs.stem
        slide_tile_dir = TILE_DIR / slide_id

        # Skip if already done
        if slide_tile_dir.exists() and len(list(slide_tile_dir.glob("*.png"))) >= 10:
            if idx <= 3:
                logger.info(f"  [{idx}/{total}] Skip {slide_id[:50]} (exists)")
            continue

        slide_tile_dir.mkdir(parents=True, exist_ok=True)

        try:
            wsi = openslide.OpenSlide(str(svs))
            dims = wsi.level_dimensions
            downsamples = wsi.level_downsamples

            level = 0
            if len(dims) > 1 and downsamples[1] <= 4:
                level = 1

            w, h = dims[level]
            step = TILE_SIZE

            candidates = [(x, y) for y in range(0, h - step, step) 
                         for x in range(0, w - step, step)]
            random.shuffle(candidates)

            tile_count = 0
            for (x, y) in candidates:
                if tile_count >= MAX_TILES:
                    break

                scale = int(downsamples[level])
                tile = wsi.read_region((x * scale, y * scale), level, (step, step))
                tile = tile.convert("RGB")

                arr = np.array(tile)
                gray = np.mean(arr, axis=2)
                tissue_frac = np.mean((gray > 30) & (gray < 220))

                if tissue_frac < TISSUE_THRESHOLD:
                    continue

                tile.save(slide_tile_dir / f"tile_{tile_count:03d}.png")
                tile_count += 1

            wsi.close()
            extracted += 1

            if idx % 10 == 0 or idx <= 3 or idx == total:
                logger.info(f"  [{idx:3d}/{total}] ✓ {slide_id[:50]} → {tile_count} tiles")

        except Exception as e:
            errors += 1
            logger.warning(f"  [{idx:3d}/{total}] ✗ {slide_id[:50]}: {str(e)[:60]}")

    logger.info(f"\n  Tile extraction: {extracted} done, {errors} errors")

    # Count total
    all_slide_dirs = [d for d in TILE_DIR.iterdir() if d.is_dir()]
    total_tiles = sum(len(list(d.glob("*.png"))) for d in all_slide_dirs)
    logger.info(f"  Total melanoma slides with tiles: {len(all_slide_dirs)}")
    logger.info(f"  Total melanoma tiles: {total_tiles}")
    return extracted

# ============================================================
# STEP 3: FEATURE EXTRACTION (all 6 backbones)
# ============================================================
def load_feature_extractor(mcfg, device):
    """Load a feature extraction model."""
    name = mcfg["name"]
    mtype = mcfg["type"]
    wpath = mcfg["weights_path"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if mtype == "torchvision":
        loader_name = mcfg["loader"]
        model_fn = getattr(models, loader_name)
        model = model_fn()
        state_dict = torch.load(wpath, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        if "resnet" in loader_name:
            model.fc = nn.Identity()
        elif "convnext" in loader_name:
            model.classifier = nn.Sequential(
                model.classifier[0],
                nn.Flatten(1),
            )

    elif mtype in ("dinov2", "phikon"):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(wpath, local_files_only=True)

    model = model.to(device)
    model.eval()
    return model, transform


def extract_features_for_model(mcfg, device):
    """Extract features for all melanoma slides missing features for this model."""
    name = mcfg["name"]
    feat_dir = FEATURE_BASE / mcfg["feat_dir"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    is_transformer = mcfg["type"] in ("dinov2", "phikon")

    # Find slides that need features
    all_slide_dirs = sorted([d for d in TILE_DIR.iterdir() if d.is_dir()])
    need_features = []
    for d in all_slide_dirs:
        slide_id = d.name
        feat_path = feat_dir / f"{slide_id}.pt"
        if not feat_path.exists():
            tiles = sorted(d.glob("*.png"))
            if len(tiles) >= 5:
                need_features.append((slide_id, d, tiles))

    if not need_features:
        logger.info(f"    {name}: All {len(all_slide_dirs)} slides already have features ✓")
        return 0

    logger.info(f"    {name}: {len(need_features)} slides need features")

    # Load model
    model, transform = load_feature_extractor(mcfg, device)
    logger.info(f"    {name}: Model loaded on {device}")

    extracted = 0
    for idx, (slide_id, tile_dir, tiles) in enumerate(need_features, 1):
        feat_path = feat_dir / f"{slide_id}.pt"

        all_features = []
        for batch_start in range(0, len(tiles), BATCH_SIZE):
            batch_tiles = tiles[batch_start:batch_start + BATCH_SIZE]
            images = [transform(Image.open(t).convert("RGB")) for t in batch_tiles]
            batch = torch.stack(images).to(device)

            with torch.no_grad():
                if is_transformer:
                    out = model(batch)
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    feats = model(batch)
            all_features.append(feats.cpu())

        if all_features:
            features = torch.cat(all_features, dim=0)
            torch.save(features, feat_path)
            extracted += 1

        if idx % 10 == 0 or idx <= 2 or idx == len(need_features):
            logger.info(f"    [{idx}/{len(need_features)}] {slide_id[:40]} → {features.shape}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    total_feats = len(list(feat_dir.glob("*.pt")))
    logger.info(f"    {name}: {extracted} new features → {total_feats} total")
    return extracted


def extract_all_features():
    """Extract features for all 6 backbones."""
    logger.info(f"\n{'━'*60}")
    logger.info(f"FEATURE EXTRACTION — 6 backbone models")
    logger.info(f"{'━'*60}")

    device = torch.device(DEVICE)
    logger.info(f"  Device: {device}")

    for i, mcfg in enumerate(MODEL_CONFIGS, 1):
        logger.info(f"\n  [{i}/6] {mcfg['name']} (feat_dim={mcfg['feat_dim']})")
        t0 = time.time()
        try:
            n = extract_features_for_model(mcfg, device)
            elapsed = time.time() - t0
            logger.info(f"    Done in {elapsed/60:.1f} min")
        except Exception as e:
            logger.error(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("  EXTRACT TILES + FEATURES FOR NEW MELANOMA SLIDES")
    logger.info("=" * 60)

    t0 = time.time()

    # Step 1: Find new slides
    logger.info(f"\n{'━'*60}")
    logger.info("FINDING NEW SLIDES")
    logger.info(f"{'━'*60}")
    new_slides = find_new_slides()

    if not new_slides:
        logger.info("  No new slides to process! Checking features...")
        extract_all_features()
        return

    # Step 2: Tile extraction
    extract_tiles(new_slides)

    # Step 3: Feature extraction (all 6 models)
    extract_all_features()

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"  ALL DONE! Total time: {elapsed/60:.1f} min")
    logger.info(f"{'='*60}")

    # Final summary
    all_mel_tiles = sorted([d for d in TILE_DIR.iterdir() if d.is_dir()])
    logger.info(f"  Total melanoma slides with tiles: {len(all_mel_tiles)}")
    for mcfg in MODEL_CONFIGS:
        feat_dir = FEATURE_BASE / mcfg["feat_dir"]
        n = len(list(feat_dir.glob("*.pt")))
        logger.info(f"  {mcfg['name']:20s} features: {n}")


if __name__ == "__main__":
    main()
