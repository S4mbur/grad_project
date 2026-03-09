#!/usr/bin/env python3

import csv
import logging
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backbone_registry import MODEL_CONFIGS, feature_dir_name, load_feature_extractor, extract_batch_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-5s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Config:
    data_root = Path("/mnt/d/skin_cancer_project/datasets")
    cache_root = Path("/mnt/d/skin_cancer_project/cache")
    tile_dir = cache_root / "tiles_4class"
    base_feature_dir = cache_root

    num_classes = 4
    class_names = ["Normal/Benign", "BCC", "SCC", "Melanoma"]

    tile_size = 256
    max_tiles_per_slide = 200
    tissue_threshold = 0.5
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OOD_CLASS_MAP = {
        "Benign": 0,
        "No abnormalities": 0,
        "Benign sebaceous gland tumor": 0,
        "Cylindroma": 0,
        "Basal cell carcinoma": 1,
        "Squamous cell carcinoma": 2,
        "Melanoma": 3,
        "Melanoma in situ": 3,
        "Merkel cell carcinoma": None,
        "Sebaceous gland carcinoma": None,
        "Microcystic adnexal carcinoma": None,
        "Skin adnexal carcinoma, other": None,
        "Lymphoma": None,
        "Cutaneous metastases": None,
    }


def create_unified_labels(cfg: Config):
    entries = []

    bcc_csv = cfg.data_root / "labels" / "bcc_bcc.csv"
    bcc_dir = cfg.data_root / "cobra_bcc"
    with open(bcc_csv) as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            label = int(row["label"])
            tif_path = bcc_dir / f"{fname}.tif"
            if tif_path.exists():
                entries.append({
                    "slide_path": tif_path,
                    "slide_id": fname,
                    "superclass": 0 if label == 0 else 1,
                    "subclass": "Normal" if label == 0 else "BCC",
                    "source": "cobra_bcc",
                })

    ood_csv = cfg.data_root / "labels" / "ood_disease_types.csv"
    ood_dir = cfg.data_root / "cobra_ood" / "images"
    with open(ood_csv) as f:
        for row in csv.DictReader(f):
            fname, cat = row["filename"], row["category"]
            sc = cfg.OOD_CLASS_MAP.get(cat)
            if sc is None:
                continue
            tif_path = ood_dir / f"{fname}.tif"
            if tif_path.exists():
                entries.append({
                    "slide_path": tif_path,
                    "slide_id": fname,
                    "superclass": sc,
                    "subclass": cat,
                    "source": "cobra_ood",
                })

    tcga_dir = cfg.data_root / "tcga_skcm"
    for svs in sorted(tcga_dir.glob("*.svs")):
        entries.append({
            "slide_path": svs,
            "slide_id": svs.stem,
            "superclass": 3,
            "subclass": "Melanoma (TCGA)",
            "source": "tcga_skcm",
        })

    counts = Counter(e["superclass"] for e in entries)
    logger.info(
        "Labels: %d slides -> %s",
        len(entries),
        ", ".join(f"{cfg.class_names[i]}={counts[i]}" for i in range(cfg.num_classes)),
    )
    return entries


def check_melanoma_integrity(entries, delete_bad=False, workers=8):
    try:
        import openslide
    except ImportError:
        logger.warning("openslide not available, skipping melanoma integrity check")
        return []

    melanoma_paths = [e["slide_path"] for e in entries if e["superclass"] == 3]
    logger.info("Melanoma integrity check: %d slides", len(melanoma_paths))

    def inspect(path: Path):
        try:
            slide = openslide.OpenSlide(str(path))
            _ = slide.level_dimensions[0]
            slide.close()
            return path, None
        except Exception as exc:
            return path, str(exc)

    bad = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect, p): p for p in melanoma_paths}
        for idx, future in enumerate(as_completed(futures), 1):
            path, err = future.result()
            if err:
                bad.append((path, err))
            if idx % 100 == 0 or idx == len(melanoma_paths):
                logger.info("  checked %d/%d", idx, len(melanoma_paths))

    if bad:
        logger.warning("Found %d corrupt melanoma slides", len(bad))
        for path, err in bad[:20]:
            logger.warning("  bad: %s :: %s", path.name, err)
        if delete_bad:
            for path, _ in bad:
                try:
                    path.unlink()
                    logger.warning("  deleted corrupt slide: %s", path)
                except Exception as exc:
                    logger.error("  failed to delete %s: %s", path, exc)
    else:
        logger.info("No corrupt melanoma slides found")
    return bad


def extract_tiles(entries, cfg: Config):
    try:
        import openslide
    except ImportError:
        logger.error("openslide not installed")
        sys.exit(1)

    cfg.tile_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)

    extracted = 0
    skipped = 0
    errors = 0
    total = len(entries)

    for idx, entry in enumerate(entries, 1):
        slide_id = entry["slide_id"]
        slide_path = entry["slide_path"]
        superclass = entry["superclass"]
        slide_tile_dir = cfg.tile_dir / f"class_{superclass}" / slide_id

        if slide_tile_dir.exists() and len(list(slide_tile_dir.glob("*.png"))) >= 10:
            skipped += 1
            if idx % 100 == 0 or idx <= 5:
                logger.info("  [%4d/%4d] skip %s", idx, total, slide_id)
            continue

        slide_tile_dir.mkdir(parents=True, exist_ok=True)
        try:
            wsi = openslide.OpenSlide(str(slide_path))
            dims = wsi.level_dimensions
            downsamples = wsi.level_downsamples
            level = 1 if len(dims) > 1 and downsamples[1] <= 4 else 0
            width, height = dims[level]
            step = cfg.tile_size
            candidates = [(x, y) for y in range(0, height - step, step) for x in range(0, width - step, step)]
            random.shuffle(candidates)

            tile_count = 0
            for x, y in candidates:
                if tile_count >= cfg.max_tiles_per_slide:
                    break
                scale = int(downsamples[level])
                tile = wsi.read_region((x * scale, y * scale), level, (step, step)).convert("RGB")
                arr = np.array(tile)
                gray = np.mean(arr, axis=2)
                tissue_frac = np.mean((gray > 30) & (gray < 220))
                if tissue_frac < cfg.tissue_threshold:
                    continue
                tile.save(slide_tile_dir / f"tile_{tile_count:03d}.png")
                tile_count += 1
            wsi.close()
            extracted += 1
            if idx % 50 == 0 or idx <= 5 or idx == total:
                logger.info("  [%4d/%4d] %s -> %d tiles", idx, total, slide_id[:40], tile_count)
        except Exception as exc:
            errors += 1
            logger.warning("  [%4d/%4d] %s failed: %s", idx, total, slide_id[:40], str(exc)[:120])

    logger.info("Tile extraction done: extracted=%d skipped=%d errors=%d", extracted, skipped, errors)
    for class_idx, class_name in enumerate(cfg.class_names):
        class_dir = cfg.tile_dir / f"class_{class_idx}"
        if not class_dir.exists():
            continue
        slides = [d for d in class_dir.iterdir() if d.is_dir()]
        tiles = sum(len(list(d.glob("*.png"))) for d in slides)
        logger.info("  class_%d %-14s slides=%d tiles=%d", class_idx, class_name, len(slides), tiles)


def extract_features(entries, cfg: Config, models_to_run):
    device = torch.device(cfg.device)
    logger.info("Feature extraction device: %s", device)

    for model_idx, model_cfg in enumerate(models_to_run, 1):
        feature_dir = cfg.base_feature_dir / feature_dir_name(model_cfg)
        feature_dir.mkdir(parents=True, exist_ok=True)

        need_features = []
        for entry in entries:
            slide_id = entry["slide_id"]
            tile_dir = cfg.tile_dir / f"class_{entry['superclass']}" / slide_id
            feat_path = feature_dir / f"{slide_id}.pt"
            if feat_path.exists():
                continue
            if tile_dir.exists() and len(list(tile_dir.glob("*.png"))) >= 5:
                need_features.append((slide_id, tile_dir))

        if not need_features:
            logger.info("[%d/%d] %s: all features already present", model_idx, len(models_to_run), model_cfg["name"])
            continue

        logger.info(
            "[%d/%d] %s: %d slides need features",
            model_idx,
            len(models_to_run),
            model_cfg["name"],
            len(need_features),
        )
        model, transform = load_feature_extractor(model_cfg, cfg.device)
        batch_size = model_cfg.get("batch_size", 32)

        for slide_idx, (slide_id, tile_dir) in enumerate(need_features, 1):
            tiles = sorted(tile_dir.glob("*.png"))
            all_features = []
            for batch_start in range(0, len(tiles), batch_size):
                batch_tiles = tiles[batch_start:batch_start + batch_size]
                images = [transform(Image.open(t).convert("RGB")) for t in batch_tiles]
                batch = torch.stack(images).to(device)
                feats = extract_batch_features(model, model_cfg, batch)
                all_features.append(feats.cpu())
            features = torch.cat(all_features, dim=0)
            torch.save(features, feature_dir / f"{slide_id}.pt")
            if slide_idx % 50 == 0 or slide_idx <= 3 or slide_idx == len(need_features):
                logger.info("    [%4d/%4d] %s -> %s", slide_idx, len(need_features), slide_id[:40], tuple(features.shape))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild 4-class tiles/features on D drive cache")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to extract")
    parser.add_argument("--skip-tiles", action="store_true", help="Skip tile extraction")
    parser.add_argument("--skip-integrity-check", action="store_true", help="Skip melanoma slide integrity scan")
    parser.add_argument("--delete-corrupt-melanoma", action="store_true", help="Delete corrupt melanoma slides if found")
    args = parser.parse_args()

    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    entries = create_unified_labels(cfg)
    if not args.skip_integrity_check:
        bad = check_melanoma_integrity(entries, delete_bad=args.delete_corrupt_melanoma)
        if bad:
            entries = [e for e in entries if e["slide_path"] not in {path for path, _ in bad}]
            logger.warning("Proceeding without %d corrupt melanoma slides", len(bad))

    models_to_run = MODEL_CONFIGS
    if args.models:
        requested = set(args.models)
        models_to_run = [m for m in MODEL_CONFIGS if m["name"] in requested]

    logger.info("Cache root: %s", cfg.cache_root)
    logger.info("Models: %s", [m["name"] for m in models_to_run])

    if not args.skip_tiles:
        extract_tiles(entries, cfg)
    extract_features(entries, cfg, models_to_run)
    logger.info("Rebuild complete")


if __name__ == "__main__":
    main()