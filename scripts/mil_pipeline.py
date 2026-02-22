#!/usr/bin/env python3
"""
Production MIL Pipeline - Skin Cancer Classification
=====================================================
3-class slide-level classification using Attention-based MIL:
  Class 0: Normal (non-tumor skin)          → COBRA label=0
  Class 1: BCC (basal cell carcinoma)        → COBRA label=1
  Class 2: Melanoma (melanocytic malignant)  → TCGA-SKCM

  ───────────────────────────────────────────────────────
  Superclass mapping for future expansion:
    Normal      → Non-melanocytic Benign
    BCC         → Non-melanocytic Malignant
    Melanoma    → Melanocytic Malignant
    (SCC, IEC, AK, Nevus → to be added later)
  ───────────────────────────────────────────────────────

Pipeline:
  Step 1 → Build slide manifest with labels + splits
  Step 2 → Extract tiles from all WSIs (skip if done)
  Step 3 → Extract features with pretrained encoder
  Step 4 → Train attention-based MIL
  Step 5 → Evaluate with full metrics + visualizations

Usage:
  python scripts/mil_pipeline.py [--skip-tiles] [--skip-features]
"""

import os
import sys
import csv
import json
import random
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configuration
# ============================================================
CLASS_NAMES = {0: "Normal", 1: "BCC", 2: "Melanoma"}
N_CLASSES = 3


class Config:
    """Pipeline configuration."""
    # Paths
    cobra_csv = PROJECT / "data" / "cobra" / "bcc_bcc.csv"
    raw_wsi_dir = PROJECT / "data" / "raw_wsi"
    melanoma_wsi_dir = PROJECT / "data" / "raw_wsi" / "melanoma"
    tile_dir = PROJECT / "data" / "tiles"
    melanoma_tile_dir = PROJECT / "data" / "tiles" / "melanoma"
    feature_dir = PROJECT / "data" / "mil_features_v2"
    output_dir = PROJECT / "data" / "mil_results"
    manifest_path = PROJECT / "data" / "mil_slide_manifest.csv"

    # Tile extraction
    tile_size = 256
    max_tiles_per_slide = 200
    min_tissue_fraction = 0.3

    # Feature extraction
    feature_dim = 512  # ResNet18
    feat_batch_size = 64

    # MIL Training
    hidden_dim = 256
    dropout = 0.25
    lr = 2e-4
    weight_decay = 1e-4
    num_epochs = 50
    patience = 10
    max_bag_size = 200
    min_tiles_per_slide = 10

    # Data
    max_normal_slides = 200   # Cap normal to balance
    max_bcc_slides = 200      # Use all BCC available

    seed = 42


cfg = Config()


# ============================================================
# Step 1: Build Slide Manifest
# ============================================================
def build_manifest() -> Path:
    """Create a unified slide manifest with labels and splits."""
    logger.info("=" * 60)
    logger.info("STEP 1: Building slide manifest")
    logger.info("=" * 60)

    # Read COBRA labels
    cobra_data = {}
    with open(cfg.cobra_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cobra_data[row["filename"]] = {
                "label": int(row["label"]),
                "split": row["split"],
            }
    logger.info(f"  COBRA metadata: {len(cobra_data)} slides")

    # Find slides we actually have
    slides = []

    # Normal + BCC from COBRA
    available_tifs = {p.stem: p for p in cfg.raw_wsi_dir.glob("*.tif")}
    normal_slides = []
    bcc_slides = []

    for slide_id, info in cobra_data.items():
        if slide_id in available_tifs:
            entry = {
                "slide_id": slide_id,
                "path": str(available_tifs[slide_id]),
                "label": 0 if info["label"] == 0 else 1,  # 0=Normal, 1=BCC
                "label_name": "Normal" if info["label"] == 0 else "BCC",
                "source": "COBRA",
                "cobra_split": info["split"],
            }
            if info["label"] == 0:
                normal_slides.append(entry)
            else:
                bcc_slides.append(entry)

    logger.info(f"  Available COBRA: {len(normal_slides)} Normal, {len(bcc_slides)} BCC")

    # Cap normal slides
    random.seed(cfg.seed)
    if len(normal_slides) > cfg.max_normal_slides:
        random.shuffle(normal_slides)
        normal_slides = normal_slides[:cfg.max_normal_slides]
        logger.info(f"  Capped Normal to {cfg.max_normal_slides}")

    if len(bcc_slides) > cfg.max_bcc_slides:
        random.shuffle(bcc_slides)
        bcc_slides = bcc_slides[:cfg.max_bcc_slides]
        logger.info(f"  Capped BCC to {cfg.max_bcc_slides}")

    slides.extend(normal_slides)
    slides.extend(bcc_slides)

    # Melanoma from TCGA
    mel_slides = []
    if cfg.melanoma_wsi_dir.exists():
        for svs in cfg.melanoma_wsi_dir.glob("*.svs"):
            mel_slides.append({
                "slide_id": svs.stem,
                "path": str(svs),
                "label": 2,
                "label_name": "Melanoma",
                "source": "TCGA-SKCM",
                "cobra_split": None,
            })
    slides.extend(mel_slides)
    logger.info(f"  Melanoma slides: {len(mel_slides)}")

    # Assign train/val/test splits (stratified)
    # Try to respect COBRA original splits, assign melanoma randomly
    random.seed(cfg.seed)
    for s in slides:
        if s["cobra_split"] == "train":
            s["split"] = "train"
        elif s["cobra_split"] == "val":
            s["split"] = "val"
        elif s["cobra_split"] == "test":
            s["split"] = "test"
        else:
            # Melanoma: 60/20/20
            r = random.random()
            if r < 0.6:
                s["split"] = "train"
            elif r < 0.8:
                s["split"] = "val"
            else:
                s["split"] = "test"

    # Save manifest
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "slide_id", "path", "label", "label_name", "source", "split"
        ])
        writer.writeheader()
        for s in slides:
            writer.writerow({
                "slide_id": s["slide_id"],
                "path": s["path"],
                "label": s["label"],
                "label_name": s["label_name"],
                "source": s["source"],
                "split": s["split"],
            })

    # Distribution
    for split in ["train", "val", "test"]:
        split_slides = [s for s in slides if s["split"] == split]
        dist = Counter(s["label_name"] for s in split_slides)
        logger.info(f"  {split:5s}: {len(split_slides):3d} slides | "
                    f"Normal={dist.get('Normal',0)} BCC={dist.get('BCC',0)} "
                    f"Melanoma={dist.get('Melanoma',0)}")

    logger.info(f"  Manifest saved: {cfg.manifest_path}")
    return cfg.manifest_path


# ============================================================
# Step 2: Tile Extraction
# ============================================================
def extract_tiles_for_slides(manifest_path: Path, skip_existing: bool = True):
    """Extract tiles from all slides listed in manifest."""
    logger.info("=" * 60)
    logger.info("STEP 2: Tile Extraction")
    logger.info("=" * 60)

    try:
        import openslide
    except ImportError:
        logger.error("openslide not available!")
        return

    slides = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            slides.append(row)

    total = len(slides)
    skipped = 0
    extracted = 0
    errors = 0

    for i, slide in enumerate(slides):
        sid = slide["slide_id"]
        slide_path = slide["path"]
        label_name = slide["label_name"]

        # Output dir per slide
        if label_name == "Melanoma":
            out_dir = cfg.melanoma_tile_dir
        else:
            out_dir = cfg.tile_dir / sid

        # Check if already done
        if skip_existing and out_dir.exists():
            existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
            if len(existing) >= cfg.min_tiles_per_slide:
                skipped += 1
                continue

        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            slide_obj = openslide.OpenSlide(slide_path)
            w, h = slide_obj.dimensions

            # Best level for ~0.5 MPP
            mpp = float(slide_obj.properties.get("openslide.mpp-x", 0.5))
            target_mpp = 0.5
            target_ds = mpp / target_mpp if mpp > 0 else 1.0
            level = slide_obj.get_best_level_for_downsample(max(target_ds, 1.0))
            level_ds = slide_obj.level_downsamples[level]
            read_size = int(cfg.tile_size * level_ds)

            # Tissue mask
            thumb_w = 512
            thumb = slide_obj.get_thumbnail((thumb_w, thumb_w))
            thumb_arr = np.array(thumb.convert("RGB"))
            gray = np.mean(thumb_arr, axis=2)
            tissue_mask = (gray < 220) & (gray > 30)

            # Sample positions
            scale_x = w / thumb_arr.shape[1]
            scale_y = h / thumb_arr.shape[0]

            positions = []
            step = max(1, int(thumb_arr.shape[0] / 50))
            for ty in range(0, thumb_arr.shape[0], step):
                for tx in range(0, thumb_arr.shape[1], step):
                    if tissue_mask[ty, tx]:
                        x = int(tx * scale_x)
                        y = int(ty * scale_y)
                        if x + read_size <= w and y + read_size <= h:
                            positions.append((x, y))

            random.shuffle(positions)
            positions = positions[:cfg.max_tiles_per_slide * 3]

            count = 0
            for x, y in positions:
                if count >= cfg.max_tiles_per_slide:
                    break

                region = slide_obj.read_region((x, y), level, (cfg.tile_size, cfg.tile_size))
                tile = region.convert("RGB")
                arr = np.array(tile)

                # Quality
                gray_t = np.mean(arr, axis=2)
                tissue_frac = np.mean((gray_t < 220) & (gray_t > 30))
                if tissue_frac < cfg.min_tissue_fraction:
                    continue

                fname = f"{sid}_tile_{count:05d}_x{x}_y{y}.jpg"
                tile.save(str(out_dir / fname), quality=90)
                count += 1

            slide_obj.close()
            extracted += 1

            if (extracted) % 20 == 0:
                logger.info(f"  [{i+1}/{total}] {label_name:8s} | {sid[:30]:30s} | {count} tiles")

        except Exception as e:
            errors += 1
            logger.warning(f"  Error [{sid[:30]}]: {e}")

    logger.info(f"  Done: extracted={extracted}, skipped={skipped}, errors={errors}")


# ============================================================
# Step 3: Feature Extraction
# ============================================================
class TileImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), idx
        except Exception:
            return torch.zeros(3, 224, 224), idx


def extract_features(manifest_path: Path):
    """Extract ResNet18 features from tiles."""
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Extraction (ResNet18)")
    logger.info("=" * 60)

    cfg.feature_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(DEVICE)
    model.eval()

    slides = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            slides.append(row)

    processed = 0
    skipped = 0

    for i, slide in enumerate(slides):
        sid = slide["slide_id"]
        label = int(slide["label"])
        feat_path = cfg.feature_dir / f"{sid}.pt"

        if feat_path.exists():
            skipped += 1
            continue

        # Find tiles
        tile_paths = []
        if slide["label_name"] == "Melanoma":
            tile_dir = cfg.melanoma_tile_dir
            for f_tile in tile_dir.glob(f"{sid}_tile_*.jpg"):
                tile_paths.append(str(f_tile))
        else:
            tile_dir = cfg.tile_dir / sid
            if tile_dir.exists():
                tile_paths = [str(p) for p in tile_dir.glob("*.jpg")]
                if not tile_paths:
                    tile_paths = [str(p) for p in tile_dir.glob("*.png")]

            # Also check train/ dir
            if not tile_paths:
                train_tiles = list((cfg.tile_dir / "train").glob(f"{sid}_tile_*.jpg"))
                tile_paths = [str(p) for p in train_tiles]

        if len(tile_paths) < cfg.min_tiles_per_slide:
            continue

        # Cap tiles
        if len(tile_paths) > cfg.max_bag_size:
            random.shuffle(tile_paths)
            tile_paths = tile_paths[:cfg.max_bag_size]

        dataset = TileImageDataset(tile_paths)
        loader = DataLoader(dataset, batch_size=cfg.feat_batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

        features = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(DEVICE)
                feats = model(imgs)
                features.append(feats.cpu())

        if features:
            features = torch.cat(features, dim=0)
            torch.save({
                "features": features,
                "label": label,
                "label_name": slide["label_name"],
                "slide_id": sid,
                "split": slide["split"],
                "n_tiles": len(tile_paths),
            }, feat_path)
            processed += 1

            if processed % 50 == 0:
                logger.info(f"  [{i+1}/{len(slides)}] Processed {processed}, "
                          f"skipped {skipped} | {slide['label_name']:8s} "
                          f"{features.shape}")

    logger.info(f"  Feature extraction done: {processed} processed, {skipped} skipped")


# ============================================================
# Step 4: MIL Model + Training
# ============================================================
class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL (Ilse et al., 2018).
    Uses gated attention mechanism for better feature selection.
    Supports multi-class output.
    """
    def __init__(self, input_dim=512, hidden_dim=256, n_classes=3, dropout=0.25):
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # Gated attention
        v = self.attention_V(x)
        u = self.attention_U(x)
        a = self.attention_w(v * u)
        a = F.softmax(a, dim=0)

        # Aggregate
        z = torch.mm(a.T, x)

        # Classify
        logits = self.classifier(z)
        return logits, a


class MILFeatureDataset(Dataset):
    def __init__(self, feature_files, max_bag_size=200):
        self.data = []
        for f in feature_files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            feats = d["features"]
            if feats.shape[0] > max_bag_size:
                idx = torch.randperm(feats.shape[0])[:max_bag_size]
                feats = feats[idx]
            self.data.append({
                "features": feats,
                "label": d["label"],
                "label_name": d["label_name"],
                "slide_id": d["slide_id"],
                "n_tiles": d["n_tiles"],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["features"], d["label"], d["slide_id"]


def train_mil():
    """Train gated-attention MIL model."""
    logger.info("=" * 60)
    logger.info("STEP 4: MIL Training (Gated Attention)")
    logger.info("=" * 60)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load features and split
    all_features = list(cfg.feature_dir.glob("*.pt"))
    logger.info(f"  Total feature files: {len(all_features)}")

    train_files, val_files, test_files = [], [], []
    class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for f in all_features:
        d = torch.load(f, map_location="cpu", weights_only=False)
        split = d.get("split", "train")
        label = d["label"]

        if split == "train":
            train_files.append(f)
        elif split == "val":
            val_files.append(f)
        else:
            test_files.append(f)
        class_counts[split][CLASS_NAMES[label]] += 1

    for split in ["train", "val", "test"]:
        logger.info(f"  {split:5s}: {sum(class_counts[split].values()):3d} slides | "
                    f"{dict(class_counts[split])}")

    if len(train_files) < 6 or len(val_files) < 3:
        logger.error(f"Not enough data! train={len(train_files)}, val={len(val_files)}")
        return None

    train_dataset = MILFeatureDataset(train_files, cfg.max_bag_size)
    val_dataset = MILFeatureDataset(val_files, cfg.max_bag_size)
    test_dataset = MILFeatureDataset(test_files, cfg.max_bag_size)

    # Class weights for imbalanced data
    train_labels = [d["label"] for d in train_dataset.data]
    class_freq = Counter(train_labels)
    total = sum(class_freq.values())
    weights = torch.tensor([total / (N_CLASSES * class_freq.get(c, 1))
                           for c in range(N_CLASSES)], dtype=torch.float32).to(DEVICE)
    logger.info(f"  Class weights: {weights.tolist()}")

    # Model
    model = GatedAttentionMIL(
        input_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        n_classes=N_CLASSES,
        dropout=cfg.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = 0
    best_epoch = 0
    patience_ctr = 0
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    logger.info(f"\n  {'Epoch':>5s} | {'TrLoss':>7s} {'TrAcc':>6s} {'TrF1':>6s} | "
                f"{'VaLoss':>7s} {'VaAcc':>6s} {'VaF1':>6s} | LR")
    logger.info("  " + "-" * 65)

    for epoch in range(cfg.num_epochs):
        # === TRAIN ===
        model.train()
        t_loss, t_preds, t_labels = 0, [], []
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        for idx in indices:
            feats, label, _ = train_dataset[idx]
            feats = feats.to(DEVICE)
            label_t = torch.tensor([label], dtype=torch.long).to(DEVICE)

            logits, attn = model(feats)
            loss = criterion(logits, label_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item()
            t_preds.append(logits.argmax(1).item())
            t_labels.append(label)

        scheduler.step()

        t_loss /= len(train_dataset)
        t_acc = accuracy_score(t_labels, t_preds)
        t_f1 = f1_score(t_labels, t_preds, average="macro", zero_division=0)

        # === VAL ===
        model.eval()
        v_loss, v_preds, v_labels = 0, [], []

        with torch.no_grad():
            for idx in range(len(val_dataset)):
                feats, label, _ = val_dataset[idx]
                feats = feats.to(DEVICE)
                label_t = torch.tensor([label], dtype=torch.long).to(DEVICE)

                logits, _ = model(feats)
                loss = criterion(logits, label_t)

                v_loss += loss.item()
                v_preds.append(logits.argmax(1).item())
                v_labels.append(label)

        v_loss /= max(len(val_dataset), 1)
        v_acc = accuracy_score(v_labels, v_preds)
        v_f1 = f1_score(v_labels, v_preds, average="macro", zero_division=0)

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["train_f1"].append(round(t_f1, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["val_f1"].append(round(v_f1, 4))

        lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == cfg.num_epochs - 1:
            logger.info(f"  {epoch+1:5d} | {t_loss:7.4f} {t_acc:6.3f} {t_f1:6.3f} | "
                       f"{v_loss:7.4f} {v_acc:6.3f} {v_f1:6.3f} | {lr:.6f}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_epoch = epoch + 1
            patience_ctr = 0
            torch.save(model.state_dict(), cfg.output_dir / "best_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    logger.info(f"\n  Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    return {
        "model": model,
        "test_dataset": test_dataset,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "class_weights": weights.tolist(),
    }


# ============================================================
# Step 5: Evaluation
# ============================================================
def evaluate(results):
    """Full evaluation with metrics and per-slide results."""
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluation")
    logger.info("=" * 60)

    model = results["model"]
    test_ds = results["test_dataset"]

    # Load best checkpoint
    ckpt = cfg.output_dir / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    slide_results = []

    with torch.no_grad():
        for idx in range(len(test_ds)):
            feats, label, sid = test_ds[idx]
            feats = feats.to(DEVICE)
            logits, attn = model(feats)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = logits.argmax(1).item()

            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs)

            slide_results.append({
                "slide_id": sid[:50],
                "true_label": CLASS_NAMES[label],
                "pred_label": CLASS_NAMES[pred],
                "probs": {CLASS_NAMES[c]: round(float(probs[c]), 4) for c in range(N_CLASSES)},
                "n_tiles": feats.shape[0],
                "correct": pred == label,
            })

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)

    # Per-class metrics
    p, r, f, s = precision_recall_fscore_support(all_labels, all_preds,
                                                   labels=[0,1,2], zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])

    logger.info(f"\n  ╔══════════════════════════════════════════╗")
    logger.info(f"  ║        TEST RESULTS (n={len(all_labels):3d})            ║")
    logger.info(f"  ╠══════════════════════════════════════════╣")
    logger.info(f"  ║  Accuracy:     {acc:6.4f}                   ║")
    logger.info(f"  ║  F1 (macro):   {f1_macro:6.4f}                   ║")
    logger.info(f"  ║  F1 (weighted):{f1_weighted:6.4f}                   ║")
    logger.info(f"  ║  AUC-ROC:      {auc:6.4f}                   ║")
    logger.info(f"  ╚══════════════════════════════════════════╝")

    logger.info(f"\n  Per-class metrics:")
    logger.info(f"  {'Class':>10s} | {'Prec':>6s} {'Rec':>6s} {'F1':>6s} | {'Support':>7s}")
    logger.info(f"  " + "-" * 45)
    for c in range(N_CLASSES):
        logger.info(f"  {CLASS_NAMES[c]:>10s} | {p[c]:6.3f} {r[c]:6.3f} {f[c]:6.3f} | {int(s[c]):7d}")

    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"  {'':>12s} {'Pred Norm':>10s} {'Pred BCC':>10s} {'Pred Mel':>10s}")
    for c in range(N_CLASSES):
        logger.info(f"  {'True '+CLASS_NAMES[c]:>12s} {cm[c,0]:10d} {cm[c,1]:10d} {cm[c,2]:10d}")

    logger.info(f"\n  Per-slide predictions:")
    for sr in sorted(slide_results, key=lambda x: (x["true_label"], x["slide_id"])):
        status = "✓" if sr["correct"] else "✗"
        probs_str = " ".join(f"{v:.2f}" for v in sr["probs"].values())
        logger.info(f"    {status} {sr['true_label']:8s} → {sr['pred_label']:8s} | "
                    f"[{probs_str}] | {sr['slide_id'][:40]}")

    # Save full report
    report = {
        "timestamp": datetime.now().isoformat(),
        "task": "3-class skin cancer MIL",
        "classes": CLASS_NAMES,
        "method": "Gated Attention MIL",
        "encoder": "ResNet18 (ImageNet)",
        "device": str(DEVICE),
        "config": {
            "tile_size": cfg.tile_size,
            "max_tiles_per_slide": cfg.max_tiles_per_slide,
            "max_bag_size": cfg.max_bag_size,
            "hidden_dim": cfg.hidden_dim,
            "lr": cfg.lr,
            "num_epochs": cfg.num_epochs,
            "dropout": cfg.dropout,
        },
        "training": {
            "best_epoch": results["best_epoch"],
            "best_val_f1": results["best_val_f1"],
            "class_weights": results["class_weights"],
        },
        "test_metrics": {
            "n_slides": len(all_labels),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "auc_roc": round(auc, 4),
            "per_class": {
                CLASS_NAMES[c]: {
                    "precision": round(float(p[c]), 4),
                    "recall": round(float(r[c]), 4),
                    "f1": round(float(f[c]), 4),
                    "support": int(s[c]),
                }
                for c in range(N_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
        },
        "slide_results": slide_results,
        "history": results["history"],
    }

    out = cfg.output_dir / "mil_results.json"
    with open(out, "w") as fp:
        json.dump(report, fp, indent=2, default=str)
    logger.info(f"\n  Report: {out}")

    # Also save a quick summary text
    summary = cfg.output_dir / "summary.txt"
    with open(summary, "w") as fp:
        fp.write("MIL Skin Cancer Classification Results\n")
        fp.write("=" * 50 + "\n")
        fp.write(f"Date: {datetime.now().isoformat()}\n")
        fp.write(f"Classes: Normal, BCC, Melanoma\n")
        fp.write(f"Method: Gated Attention MIL + ResNet18\n")
        fp.write(f"Test slides: {len(all_labels)}\n\n")
        fp.write(f"Accuracy:     {acc:.4f}\n")
        fp.write(f"F1 (macro):   {f1_macro:.4f}\n")
        fp.write(f"F1 (weighted):{f1_weighted:.4f}\n")
        fp.write(f"AUC-ROC:      {auc:.4f}\n\n")
        fp.write(f"Confusion Matrix:\n")
        fp.write(f"              Pred Normal  Pred BCC  Pred Melanoma\n")
        for c in range(N_CLASSES):
            fp.write(f"True {CLASS_NAMES[c]:8s}   {cm[c,0]:6d}    {cm[c,1]:6d}        {cm[c,2]:6d}\n")
    logger.info(f"  Summary: {summary}")

    return report


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MIL Skin Cancer Pipeline")
    parser.add_argument("--skip-tiles", action="store_true",
                       help="Skip tile extraction")
    parser.add_argument("--skip-features", action="store_true",
                       help="Skip feature extraction")
    args = parser.parse_args()

    start = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   MIL Skin Cancer Classification Pipeline                ║")
    logger.info("║   3-class: Normal / BCC / Melanoma                      ║")
    logger.info(f"║   Device: {str(DEVICE):10s}                                     ║")
    logger.info("╚" + "═" * 58 + "╝")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Step 1: Build manifest
    manifest = build_manifest()

    # Step 2: Tile extraction
    if not args.skip_tiles:
        extract_tiles_for_slides(manifest)
    else:
        logger.info("STEP 2: Skipped (--skip-tiles)")

    # Step 3: Feature extraction
    if not args.skip_features:
        extract_features(manifest)
    else:
        logger.info("STEP 3: Skipped (--skip-features)")

    # Step 4: Train
    results = train_mil()
    if results is None:
        logger.error("Training failed!")
        return

    # Step 5: Evaluate
    report = evaluate(results)

    elapsed = time.time() - start
    logger.info(f"\n  Total time: {elapsed/60:.1f} minutes")
    logger.info("  DONE! ✓")


if __name__ == "__main__":
    main()
