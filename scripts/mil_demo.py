#!/usr/bin/env python3
"""
MIL (Multiple Instance Learning) Demo Pipeline
================================================
Binary classification: BCC vs Melanoma using attention-based MIL.

Pipeline steps:
  1. Prepare slide manifest (BCC + Melanoma WSIs)
  2. Extract tiles from melanoma slides (BCC already extracted)
  3. Extract features using pretrained ResNet50
  4. Train attention-based MIL aggregator
  5. Evaluate and report results

Usage:
  python scripts/mil_demo.py
"""

import os
import sys
import csv
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CONFIG
# ============================================================
class MILConfig:
    # Data
    bcc_tile_dir = PROJECT_ROOT / "data" / "tiles" / "train"
    melanoma_wsi_dir = PROJECT_ROOT / "data" / "raw_wsi" / "melanoma"
    melanoma_tile_dir = PROJECT_ROOT / "data" / "tiles" / "melanoma"
    feature_dir = PROJECT_ROOT / "data" / "mil_features"
    output_dir = PROJECT_ROOT / "data" / "mil_demo"

    # Tile extraction
    tile_size = 256
    max_tiles_per_slide = 100  # Demo: keep it small
    min_tissue_fraction = 0.3

    # Feature extraction
    feature_dim = 512  # ResNet18 last layer
    batch_size_feat = 64

    # MIL Training
    num_epochs = 30
    lr = 1e-4
    weight_decay = 1e-5
    patience = 7  # early stopping
    max_bag_size = 100  # max tiles per bag for training
    min_tiles_per_slide = 5

    # Balance
    max_bcc_slides = 40  # match ~melanoma count for balance

    seed = 42

cfg = MILConfig()


# ============================================================
# STEP 1: Prepare Slide Inventory
# ============================================================
def step1_prepare_inventory():
    """Find all available slides and their tiles."""
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing slide inventory")
    logger.info("=" * 60)

    inventory = {"bcc": [], "melanoma": []}

    # BCC: group existing tiles by slide_id
    logger.info("Scanning BCC tiles...")
    bcc_tiles = defaultdict(list)
    if cfg.bcc_tile_dir.exists():
        for f in cfg.bcc_tile_dir.iterdir():
            if f.suffix in (".jpg", ".png"):
                # Name format: {slide_id}_tile_{num}_x{x}_y{y}.jpg
                parts = f.stem.split("_tile_")
                if len(parts) == 2:
                    slide_id = parts[0]
                    bcc_tiles[slide_id].append(str(f))

    logger.info(f"  BCC: {len(bcc_tiles)} slides, {sum(len(v) for v in bcc_tiles.values())} tiles")

    # Limit BCC slides for balance
    bcc_slide_ids = sorted(bcc_tiles.keys())
    random.seed(cfg.seed)
    if len(bcc_slide_ids) > cfg.max_bcc_slides:
        bcc_slide_ids = random.sample(bcc_slide_ids, cfg.max_bcc_slides)
        logger.info(f"  BCC: sampled {cfg.max_bcc_slides} slides for balance")

    for sid in bcc_slide_ids:
        tiles = bcc_tiles[sid][:cfg.max_tiles_per_slide]
        if len(tiles) >= cfg.min_tiles_per_slide:
            inventory["bcc"].append({"slide_id": sid, "tiles": tiles, "label": 0})

    # Melanoma: check for existing tiles or WSIs
    logger.info("Scanning Melanoma data...")
    mel_tiles = defaultdict(list)

    # Check if tiles already extracted
    if cfg.melanoma_tile_dir.exists():
        for f in cfg.melanoma_tile_dir.iterdir():
            if f.suffix in (".jpg", ".png"):
                parts = f.stem.split("_tile_")
                if len(parts) == 2:
                    slide_id = parts[0]
                    mel_tiles[slide_id].append(str(f))

    # Check for WSIs that need tile extraction
    mel_wsis = []
    if cfg.melanoma_wsi_dir.exists():
        mel_wsis = list(cfg.melanoma_wsi_dir.glob("*.svs"))
    logger.info(f"  Melanoma: {len(mel_tiles)} slides with tiles, {len(mel_wsis)} WSIs")

    for sid, tiles in mel_tiles.items():
        tiles = tiles[:cfg.max_tiles_per_slide]
        if len(tiles) >= cfg.min_tiles_per_slide:
            inventory["melanoma"].append({"slide_id": sid, "tiles": tiles, "label": 1})

    # Need to extract tiles from WSIs?
    extracted_ids = set(mel_tiles.keys())
    wsis_needing_extraction = [w for w in mel_wsis
                               if w.stem not in extracted_ids]

    logger.info(f"\n  Summary:")
    logger.info(f"    BCC slides (used): {len(inventory['bcc'])}")
    logger.info(f"    Melanoma slides (with tiles): {len(inventory['melanoma'])}")
    logger.info(f"    Melanoma WSIs needing extraction: {len(wsis_needing_extraction)}")

    return inventory, wsis_needing_extraction


# ============================================================
# STEP 2: Extract Tiles from Melanoma WSIs
# ============================================================
def step2_extract_melanoma_tiles(wsis_to_extract):
    """Extract tiles from melanoma WSI files."""
    if not wsis_to_extract:
        logger.info("STEP 2: No melanoma tiles to extract (already done)")
        return

    logger.info("=" * 60)
    logger.info(f"STEP 2: Extracting tiles from {len(wsis_to_extract)} melanoma WSIs")
    logger.info("=" * 60)

    try:
        import openslide
    except ImportError:
        logger.error("openslide not available! Install: pip install openslide-python")
        logger.info("Falling back to any existing tiles...")
        return

    cfg.melanoma_tile_dir.mkdir(parents=True, exist_ok=True)

    for i, wsi_path in enumerate(wsis_to_extract):
        logger.info(f"  [{i+1}/{len(wsis_to_extract)}] {wsi_path.name}")
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            w, h = slide.dimensions

            # Get MPP
            mpp = float(slide.properties.get("openslide.mpp-x", 0.5))
            target_mpp = 0.5
            scale = mpp / target_mpp
            level = slide.get_best_level_for_downsample(1.0 / scale if scale > 0 else 1.0)
            downsample = slide.level_downsamples[level]
            read_size = int(cfg.tile_size * downsample)

            # Create tissue mask
            thumb_size = 512
            thumb = slide.get_thumbnail((thumb_size, thumb_size))
            thumb_arr = np.array(thumb.convert("RGB"))
            gray = np.mean(thumb_arr, axis=2)
            tissue_mask = (gray < 220) & (gray > 30)

            # Sample tile positions
            scale_x = w / thumb_size
            scale_y = h / thumb_arr.shape[0]

            positions = []
            for ty in range(0, thumb_arr.shape[0], 4):
                for tx in range(0, thumb_size, 4):
                    if ty < tissue_mask.shape[0] and tx < tissue_mask.shape[1]:
                        if tissue_mask[ty, tx]:
                            x = int(tx * scale_x)
                            y = int(ty * scale_y)
                            if x + read_size <= w and y + read_size <= h:
                                positions.append((x, y))

            random.shuffle(positions)
            positions = positions[:cfg.max_tiles_per_slide * 2]  # oversample then filter

            slide_id = wsi_path.stem
            count = 0
            for x, y in positions:
                if count >= cfg.max_tiles_per_slide:
                    break

                region = slide.read_region((x, y), level, (cfg.tile_size, cfg.tile_size))
                tile = region.convert("RGB")
                arr = np.array(tile)

                # Quality check
                gray_tile = np.mean(arr, axis=2)
                tissue_frac = np.mean((gray_tile < 220) & (gray_tile > 30))
                if tissue_frac < cfg.min_tissue_fraction:
                    continue

                # Save
                fname = f"{slide_id}_tile_{count:05d}_x{x}_y{y}.jpg"
                tile.save(str(cfg.melanoma_tile_dir / fname), quality=90)
                count += 1

            logger.info(f"    Extracted {count} tiles")
            slide.close()

        except Exception as e:
            logger.warning(f"    Error: {e}")


# ============================================================
# STEP 3: Feature Extraction
# ============================================================
class TileDataset(Dataset):
    """Dataset for loading tiles for feature extraction."""
    def __init__(self, tile_paths, transform=None):
        self.paths = tile_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def step3_extract_features(inventory):
    """Extract features from tiles using pretrained ResNet18."""
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting features with ResNet18")
    logger.info("=" * 60)

    cfg.feature_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # Remove classifier, get 512-d features
    model = model.to(DEVICE)
    model.eval()

    all_slides = inventory["bcc"] + inventory["melanoma"]
    logger.info(f"  Total slides to process: {len(all_slides)}")

    for i, slide_info in enumerate(all_slides):
        sid = slide_info["slide_id"]
        feat_path = cfg.feature_dir / f"{sid}.pt"

        if feat_path.exists():
            continue

        tiles = slide_info["tiles"]
        if len(tiles) < cfg.min_tiles_per_slide:
            continue

        dataset = TileDataset(tiles)
        loader = DataLoader(dataset, batch_size=cfg.batch_size_feat,
                          shuffle=False, num_workers=0)

        features = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(DEVICE)
                feats = model(imgs)
                features.append(feats.cpu())

        features = torch.cat(features, dim=0)
        torch.save({
            "features": features,
            "label": slide_info["label"],
            "slide_id": sid,
            "n_tiles": len(tiles),
        }, feat_path)

        if (i + 1) % 10 == 0 or i == len(all_slides) - 1:
            logger.info(f"  [{i+1}/{len(all_slides)}] {sid}: {features.shape[0]} tiles -> {features.shape}")

    logger.info("  Feature extraction complete!")


# ============================================================
# STEP 4: MIL Model & Training
# ============================================================
class AttentionMIL(nn.Module):
    """Attention-based MIL aggregator (Ilse et al., 2018)."""
    def __init__(self, input_dim=512, hidden_dim=256, n_classes=2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (N, D) bag of instance features
        Returns:
            logits: (1, n_classes)
            attention_weights: (N, 1)
        """
        # Attention
        a = self.attention(x)  # (N, 1)
        a = F.softmax(a, dim=0)  # normalize

        # Weighted aggregation
        z = torch.mm(a.T, x)  # (1, D)

        # Classification
        logits = self.classifier(z)  # (1, n_classes)

        return logits, a


class MILBagDataset(Dataset):
    """Dataset that loads precomputed features as bags."""
    def __init__(self, feature_files, max_bag_size=100):
        self.bags = []
        for f in feature_files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            feats = data["features"]
            if feats.shape[0] > max_bag_size:
                idx = torch.randperm(feats.shape[0])[:max_bag_size]
                feats = feats[idx]
            self.bags.append({
                "features": feats,
                "label": data["label"],
                "slide_id": data["slide_id"],
            })

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        return bag["features"], bag["label"], bag["slide_id"]


def step4_train_mil(inventory):
    """Train attention-based MIL model."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training Attention-based MIL")
    logger.info("=" * 60)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect feature files
    feature_files = list(cfg.feature_dir.glob("*.pt"))
    logger.info(f"  Feature files: {len(feature_files)}")

    # Split into train/val/test (60/20/20 by slide)
    random.seed(cfg.seed)
    bcc_files = [f for f in feature_files
                 if torch.load(f, map_location="cpu", weights_only=False)["label"] == 0]
    mel_files = [f for f in feature_files
                 if torch.load(f, map_location="cpu", weights_only=False)["label"] == 1]

    random.shuffle(bcc_files)
    random.shuffle(mel_files)

    def split_list(lst, ratios=(0.6, 0.2, 0.2)):
        n = len(lst)
        i1 = int(n * ratios[0])
        i2 = int(n * (ratios[0] + ratios[1]))
        return lst[:i1], lst[i1:i2], lst[i2:]

    bcc_train, bcc_val, bcc_test = split_list(bcc_files)
    mel_train, mel_val, mel_test = split_list(mel_files)

    train_files = bcc_train + mel_train
    val_files = bcc_val + mel_val
    test_files = bcc_test + mel_test

    random.shuffle(train_files)

    logger.info(f"  Train: {len(train_files)} ({len(bcc_train)} BCC, {len(mel_train)} MEL)")
    logger.info(f"  Val:   {len(val_files)} ({len(bcc_val)} BCC, {len(mel_val)} MEL)")
    logger.info(f"  Test:  {len(test_files)} ({len(bcc_test)} BCC, {len(mel_test)} MEL)")

    if len(train_files) < 4 or len(val_files) < 2:
        logger.error("Not enough data for training!")
        return None

    train_dataset = MILBagDataset(train_files, cfg.max_bag_size)
    val_dataset = MILBagDataset(val_files, cfg.max_bag_size)
    test_dataset = MILBagDataset(test_files, cfg.max_bag_size)

    # Model
    model = AttentionMIL(input_dim=cfg.feature_dim, hidden_dim=256, n_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg.num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        for idx in indices:
            features, label, _ = train_dataset[idx]
            features = features.to(DEVICE)
            label_t = torch.tensor([label], dtype=torch.long).to(DEVICE)

            logits, attention = model(features)
            loss = criterion(logits, label_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1).item()
            train_loss += loss.item()
            train_correct += (pred == label)
            train_total += 1

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for idx in range(len(val_dataset)):
                features, label, _ = val_dataset[idx]
                features = features.to(DEVICE)
                label_t = torch.tensor([label], dtype=torch.long).to(DEVICE)

                logits, _ = model(features)
                loss = criterion(logits, label_t)

                pred = logits.argmax(dim=1).item()
                val_loss += loss.item()
                val_correct += (pred == label)
                val_total += 1

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{cfg.num_epochs} | "
                       f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                       f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), cfg.output_dir / "best_mil_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    logger.info(f"  Best val acc: {best_val_acc:.3f} at epoch {best_epoch}")

    return {
        "model": model,
        "test_dataset": test_dataset,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "split": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
    }


# ============================================================
# STEP 5: Evaluation
# ============================================================
def step5_evaluate(results):
    """Evaluate on test set and generate report."""
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluation")
    logger.info("=" * 60)

    model = results["model"]
    test_dataset = results["test_dataset"]

    # Load best model
    best_path = cfg.output_dir / "best_mil_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    slide_results = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            features, label, slide_id = test_dataset[idx]
            features = features.to(DEVICE)

            logits, attention = model(features)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs[0, 1].item())  # P(melanoma)

            slide_results.append({
                "slide_id": slide_id,
                "true_label": "BCC" if label == 0 else "Melanoma",
                "pred_label": "BCC" if pred == 0 else "Melanoma",
                "prob_melanoma": round(probs[0, 1].item(), 4),
                "n_tiles": features.shape[0],
                "correct": pred == label,
            })

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=["BCC", "Melanoma"],
                                   output_dict=True)

    logger.info(f"\n  TEST RESULTS:")
    logger.info(f"  {'='*40}")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"              Pred BCC  Pred MEL")
    logger.info(f"    True BCC:    {cm[0,0]:4d}      {cm[0,1]:4d}")
    logger.info(f"    True MEL:    {cm[1,0]:4d}      {cm[1,1]:4d}")

    logger.info(f"\n  Per-slide results:")
    for sr in slide_results:
        status = "✓" if sr["correct"] else "✗"
        logger.info(f"    {status} {sr['slide_id'][:30]:30s} | "
                    f"True: {sr['true_label']:8s} | "
                    f"Pred: {sr['pred_label']:8s} | "
                    f"P(mel): {sr['prob_melanoma']:.3f}")

    # Save report
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "task": "BCC vs Melanoma (MIL)",
        "method": "Attention-based MIL (Ilse et al., 2018)",
        "feature_extractor": "ResNet18 (ImageNet pretrained)",
        "device": str(DEVICE),
        "config": {
            "tile_size": cfg.tile_size,
            "max_tiles_per_slide": cfg.max_tiles_per_slide,
            "max_bag_size": cfg.max_bag_size,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
        },
        "data_split": results["split"],
        "training": {
            "best_epoch": results["best_epoch"],
            "best_val_acc": results["best_val_acc"],
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "auc_roc": round(auc, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "slide_results": slide_results,
        "history": {k: [round(v, 4) for v in vals] for k, vals in results["history"].items()},
    }

    report_path = cfg.output_dir / "mil_demo_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"\n  Report saved: {report_path}")

    return full_report


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("MIL DEMO: BCC vs MELANOMA")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Step 1: Inventory
    inventory, wsis_to_extract = step1_prepare_inventory()

    # Step 2: Extract melanoma tiles
    step2_extract_melanoma_tiles(wsis_to_extract)

    # Re-inventory after extraction
    if wsis_to_extract:
        inventory, _ = step1_prepare_inventory()

    total_slides = len(inventory["bcc"]) + len(inventory["melanoma"])
    if total_slides < 6:
        logger.error(f"Not enough slides ({total_slides}). Need at least 6.")
        return

    # Step 3: Extract features
    step3_extract_features(inventory)

    # Step 4: Train MIL
    results = step4_train_mil(inventory)

    if results is None:
        logger.error("Training failed!")
        return

    # Step 5: Evaluate
    report = step5_evaluate(results)

    logger.info("\n" + "=" * 60)
    logger.info("MIL DEMO COMPLETE!")
    logger.info(f"  Accuracy: {report['test_metrics']['accuracy']:.4f}")
    logger.info(f"  AUC-ROC:  {report['test_metrics']['auc_roc']:.4f}")
    logger.info(f"  Report:   {cfg.output_dir / 'mil_demo_report.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
