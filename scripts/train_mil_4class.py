#!/usr/bin/env python3
"""
=============================================================================
  4-Class Skin Cancer MIL Pipeline (ResNet18)
  Classes: 0=Normal/Benign, 1=BCC, 2=SCC, 3=Melanoma
  
  Features:
    - Unified label mapping from multiple datasets
    - Tile extraction from WSIs (256x256 @ 20x)
    - ResNet18 feature extraction (offline from D:)
    - Gated Attention MIL with class-weighted loss
    - Stratified train/val/test splits
    - Full logging to console + file
    - Progress bars, per-epoch metrics, confusion matrix
=============================================================================
"""
import os
import sys
import csv
import json
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# ============================================================
# CONFIG
# ============================================================
class Config:
    # Directories
    data_root = Path("/mnt/d/skin_cancer_project/datasets")
    model_path = Path("/mnt/d/skin_cancer_project/models/torchvision/resnet18.pth")
    output_dir = Path("/home/byalc/phase1_project/results/mil_4class_resnet18")
    tile_dir = Path("/home/byalc/phase1_project/data/tiles_4class")
    feature_dir = Path("/home/byalc/phase1_project/data/features_4class")
    
    # Classes
    num_classes = 4
    class_names = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
    
    # Tile extraction
    tile_size = 256
    max_tiles_per_slide = 200
    tissue_threshold = 0.5   # min tissue fraction
    
    # Feature extraction
    feature_dim = 512        # ResNet18 output
    batch_size_feat = 64
    
    # MIL Training
    num_epochs = 50
    lr = 1e-4
    weight_decay = 1e-4
    patience = 10
    mil_hidden = 256
    mil_attention = 128
    dropout = 0.25
    
    # General
    seed = 42
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # OOD disease → superclass mapping
    OOD_CLASS_MAP = {
        # Class 0: Normal/Benign
        "Benign": 0,
        "No abnormalities": 0,
        "Benign sebaceous gland tumor": 0,
        "Cylindroma": 0,
        # Class 1: BCC
        "Basal cell carcinoma": 1,
        # Class 2: SCC
        "Squamous cell carcinoma": 2,
        # Class 3: Melanoma
        "Melanoma": 3,
        "Melanoma in situ": 3,
        # Excluded (rare/ambiguous — too few samples or unclear class)
        "Merkel cell carcinoma": None,
        "Sebaceous gland carcinoma": None,
        "Microcystic adnexal carcinoma": None,
        "Skin adnexal carcinoma, other": None,
        "Lymphoma": None,
        "Cutaneous metastases": None,
    }

# ============================================================
# LOGGING
# ============================================================
def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"training_{timestamp}.log"
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ]
    )
    return logging.getLogger(__name__), log_file

# ============================================================
# STEP 1: UNIFIED LABEL CREATION
# ============================================================
def create_unified_labels(cfg):
    """Create a unified CSV mapping: slide_path, slide_id, superclass, source"""
    logger = logging.getLogger(__name__)
    logger.info("━" * 60)
    logger.info("STEP 1: Creating Unified Labels")
    logger.info("━" * 60)
    
    entries = []
    
    # --- COBRA BCC Group ---
    bcc_csv = cfg.data_root / "labels" / "bcc_bcc.csv"
    bcc_dir = cfg.data_root / "cobra_bcc"
    with open(bcc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            label = int(row["label"])  # 0=Normal, 1=BCC
            tif_path = bcc_dir / f"{fname}.tif"
            if tif_path.exists():
                superclass = 0 if label == 0 else 1  # Normal=0 or BCC=1
                entries.append({
                    "slide_path": str(tif_path),
                    "slide_id": fname,
                    "superclass": superclass,
                    "subclass": "Normal" if label == 0 else "BCC",
                    "source": "cobra_bcc",
                })
    
    logger.info(f"  COBRA BCC: {sum(1 for e in entries if e['source']=='cobra_bcc')} slides")
    
    # --- COBRA OOD Group ---
    ood_csv = cfg.data_root / "labels" / "ood_disease_types.csv"
    ood_dir = cfg.data_root / "cobra_ood" / "images"
    with open(ood_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            category = row["category"]
            superclass = cfg.OOD_CLASS_MAP.get(category, None)
            if superclass is None:
                continue  # Skip excluded categories
            tif_path = ood_dir / f"{fname}.tif"
            if tif_path.exists():
                entries.append({
                    "slide_path": str(tif_path),
                    "slide_id": fname,
                    "superclass": superclass,
                    "subclass": category,
                    "source": "cobra_ood",
                })
    
    logger.info(f"  COBRA OOD: {sum(1 for e in entries if e['source']=='cobra_ood')} slides")
    
    # --- TCGA-SKCM (all melanoma) ---
    tcga_dir = cfg.data_root / "tcga_skcm"
    tcga_count = 0
    for svs in tcga_dir.glob("*.svs"):
        entries.append({
            "slide_path": str(svs),
            "slide_id": svs.stem,
            "superclass": 3,  # Melanoma
            "subclass": "Melanoma (TCGA)",
            "source": "tcga_skcm",
        })
        tcga_count += 1
    
    logger.info(f"  TCGA-SKCM: {tcga_count} slides")
    
    # --- Summary ---
    class_counts = Counter(e["superclass"] for e in entries)
    subclass_counts = Counter(e["subclass"] for e in entries)
    
    logger.info(f"\n  Total: {len(entries)} slides")
    logger.info(f"  ┌─────────────────────────────┬───────┐")
    logger.info(f"  │ Superclass                  │ Count │")
    logger.info(f"  ├─────────────────────────────┼───────┤")
    for i, name in enumerate(cfg.class_names):
        logger.info(f"  │ {i}: {name:24s} │ {class_counts[i]:5d} │")
    logger.info(f"  └─────────────────────────────┴───────┘")
    
    logger.info(f"\n  Subclass breakdown:")
    for sub, cnt in subclass_counts.most_common():
        logger.info(f"    {sub:35s}: {cnt}")
    
    # Save unified labels
    label_path = cfg.output_dir / "unified_labels.csv"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_path", "slide_id", "superclass", "subclass", "source"])
        writer.writeheader()
        writer.writerows(entries)
    
    logger.info(f"\n  Saved: {label_path}")
    return entries

# ============================================================
# STEP 2: TILE EXTRACTION
# ============================================================
def extract_tiles(entries, cfg):
    """Extract tissue tiles from WSIs."""
    logger = logging.getLogger(__name__)
    logger.info("━" * 60)
    logger.info("STEP 2: Tile Extraction")
    logger.info("━" * 60)
    
    try:
        import openslide
    except ImportError:
        logger.error("openslide not installed! Install: pip install openslide-python")
        logger.error("Also: sudo apt-get install openslide-tools")
        sys.exit(1)
    
    from PIL import Image
    
    cfg.tile_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(entries)
    extracted = 0
    skipped = 0
    errors = 0
    
    for idx, entry in enumerate(entries, 1):
        slide_id = entry["slide_id"]
        slide_path = entry["slide_path"]
        superclass = entry["superclass"]
        
        # Output dir per slide
        slide_tile_dir = cfg.tile_dir / f"class_{superclass}" / slide_id
        
        # Skip if already extracted
        if slide_tile_dir.exists():
            existing = list(slide_tile_dir.glob("*.png"))
            if len(existing) >= 10:
                skipped += 1
                if idx % 100 == 0 or idx <= 5:
                    logger.info(f"  [{idx:4d}/{total}] Skip {slide_id} ({len(existing)} tiles)")
                continue
        
        slide_tile_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            wsi = openslide.OpenSlide(slide_path)
            
            # Find best level for ~20x
            dims = wsi.level_dimensions
            downsamples = wsi.level_downsamples
            
            # Use level 0 or 1
            level = 0
            if len(dims) > 1 and downsamples[1] <= 4:
                level = 1
            
            w, h = dims[level]
            step = cfg.tile_size
            
            # Collect candidate positions
            candidates = []
            for y in range(0, h - step, step):
                for x in range(0, w - step, step):
                    candidates.append((x, y))
            
            # Random shuffle and sample
            random.shuffle(candidates)
            
            tile_count = 0
            for (x, y) in candidates:
                if tile_count >= cfg.max_tiles_per_slide:
                    break
                
                # Read tile at level 0 coordinates
                scale = int(downsamples[level])
                tile = wsi.read_region((x * scale, y * scale), level, (step, step))
                tile = tile.convert("RGB")
                
                # Tissue check: not too white, not too dark
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
                logger.info(f"  [{idx:4d}/{total}] ✓ {slide_id[:30]:30s} → {tile_count} tiles")
        
        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning(f"  [{idx:4d}/{total}] ✗ {slide_id[:30]}: {str(e)[:60]}")
    
    logger.info(f"\n  Tile extraction complete:")
    logger.info(f"    Extracted: {extracted}, Skipped: {skipped}, Errors: {errors}")
    
    # Count tiles per class
    for c in range(cfg.num_classes):
        class_dir = cfg.tile_dir / f"class_{c}"
        if class_dir.exists():
            slides = list(class_dir.iterdir())
            tiles = sum(len(list(s.glob("*.png"))) for s in slides if s.is_dir())
            logger.info(f"    Class {c} ({cfg.class_names[c]}): {len(slides)} slides, {tiles} tiles")

# ============================================================
# STEP 3: FEATURE EXTRACTION (ResNet18)
# ============================================================
def extract_features(entries, cfg):
    """Extract features from tiles using pretrained ResNet18."""
    logger = logging.getLogger(__name__)
    logger.info("━" * 60)
    logger.info("STEP 3: Feature Extraction (ResNet18)")
    logger.info("━" * 60)
    
    from torchvision import models
    from PIL import Image
    
    cfg.feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ResNet18
    device = torch.device(cfg.device)
    logger.info(f"  Device: {device}")
    logger.info(f"  Loading ResNet18 from {cfg.model_path}")
    
    model = models.resnet18()
    state_dict = torch.load(cfg.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.fc = nn.Identity()  # Remove classification head → 512-dim features
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    total = len(entries)
    extracted = 0
    skipped = 0
    
    for idx, entry in enumerate(entries, 1):
        slide_id = entry["slide_id"]
        superclass = entry["superclass"]
        
        feat_path = cfg.feature_dir / f"{slide_id}.pt"
        
        # Skip if already extracted
        if feat_path.exists():
            skipped += 1
            continue
        
        tile_dir = cfg.tile_dir / f"class_{superclass}" / slide_id
        if not tile_dir.exists():
            continue
        
        tiles = sorted(tile_dir.glob("*.png"))
        if len(tiles) < 5:
            continue
        
        # Batch process tiles
        all_features = []
        for batch_start in range(0, len(tiles), cfg.batch_size_feat):
            batch_tiles = tiles[batch_start:batch_start + cfg.batch_size_feat]
            images = []
            for t in batch_tiles:
                img = Image.open(t).convert("RGB")
                images.append(transform(img))
            
            batch = torch.stack(images).to(device)
            with torch.no_grad():
                feats = model(batch)
            all_features.append(feats.cpu())
        
        if all_features:
            features = torch.cat(all_features, dim=0)
            torch.save(features, feat_path)
            extracted += 1
        
        if idx % 50 == 0 or idx <= 5 or idx == total:
            n_tiles = len(tiles)
            logger.info(f"  [{idx:4d}/{total}] ✓ {slide_id[:30]:30s} → {n_tiles} tiles → [{features.shape[0]}x{features.shape[1]}]")
    
    logger.info(f"\n  Feature extraction complete: {extracted} new, {skipped} cached")

# ============================================================
# STEP 4: MIL MODEL
# ============================================================
class GatedAttentionMIL(nn.Module):
    """Gated Attention MIL with multi-class output."""
    def __init__(self, feat_dim=512, hidden_dim=256, attn_dim=128, num_classes=4, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        # x: [N_tiles, feat_dim]
        h = self.encoder(x)                    # [N, hidden]
        a_v = self.attention_V(h)              # [N, attn]
        a_u = self.attention_U(h)              # [N, attn]
        a = self.attention_W(a_v * a_u)        # [N, 1]
        a = F.softmax(a, dim=0)                # [N, 1]
        z = torch.sum(a * h, dim=0, keepdim=True)  # [1, hidden]
        logits = self.classifier(z)            # [1, num_classes]
        return logits, a.squeeze()

class SlideDataset(Dataset):
    def __init__(self, slide_list, feature_dir):
        self.slides = []
        for slide_id, label in slide_list:
            feat_path = feature_dir / f"{slide_id}.pt"
            if feat_path.exists():
                self.slides.append((feat_path, label, slide_id))
    
    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, idx):
        feat_path, label, slide_id = self.slides[idx]
        features = torch.load(feat_path, weights_only=True)
        return features, label, slide_id

# ============================================================
# STEP 5: TRAINING
# ============================================================
def train_mil(entries, cfg):
    """Train Gated Attention MIL model."""
    logger = logging.getLogger(__name__)
    logger.info("━" * 60)
    logger.info("STEP 4: MIL Training")
    logger.info("━" * 60)
    
    device = torch.device(cfg.device)
    
    # Prepare slide list
    slide_list = []
    for e in entries:
        feat_path = cfg.feature_dir / f"{e['slide_id']}.pt"
        if feat_path.exists():
            slide_list.append((e["slide_id"], e["superclass"]))
    
    labels = [s[1] for s in slide_list]
    class_counts = Counter(labels)
    logger.info(f"  Slides with features: {len(slide_list)}")
    for c in range(cfg.num_classes):
        logger.info(f"    Class {c} ({cfg.class_names[c]}): {class_counts[c]}")
    
    # Stratified split: 70% train, 15% val, 15% test
    ids = list(range(len(slide_list)))
    train_ids, temp_ids = train_test_split(ids, test_size=0.3, stratify=labels, random_state=cfg.seed)
    temp_labels = [labels[i] for i in temp_ids]
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, stratify=temp_labels, random_state=cfg.seed)
    
    train_slides = [slide_list[i] for i in train_ids]
    val_slides = [slide_list[i] for i in val_ids]
    test_slides = [slide_list[i] for i in test_ids]
    
    logger.info(f"\n  Split: Train={len(train_slides)}, Val={len(val_slides)}, Test={len(test_slides)}")
    
    # Datasets
    train_ds = SlideDataset(train_slides, cfg.feature_dir)
    val_ds = SlideDataset(val_slides, cfg.feature_dir)
    test_ds = SlideDataset(test_slides, cfg.feature_dir)
    
    logger.info(f"  Usable: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # Class weights
    train_labels = [s[1] for s in train_slides]
    total = len(train_labels)
    weights = [total / (cfg.num_classes * train_labels.count(c)) for c in range(cfg.num_classes)]
    class_weights = torch.FloatTensor(weights).to(device)
    logger.info(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    
    # Model
    model = GatedAttentionMIL(
        feat_dim=cfg.feature_dim,
        hidden_dim=cfg.mil_hidden,
        attn_dim=cfg.mil_attention,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    logger.info(f"  Model: GatedAttentionMIL ({sum(p.numel() for p in model.parameters()):,} params)")
    logger.info(f"  Optimizer: Adam (lr={cfg.lr}, wd={cfg.weight_decay})")
    logger.info(f"  Scheduler: CosineAnnealing")
    logger.info(f"\n  {'─'*72}")
    logger.info(f"  {'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>8} │ {'Val Acc':>7} │ {'Val F1':>6} │ {'LR':>8} │ {'Time':>5}")
    logger.info(f"  {'─'*72}")
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = []
    checkpoint_path = cfg.output_dir / "best_model.pt"
    
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        
        # ---- Train ----
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        
        for i in indices:
            features, label, _ = train_ds[i]
            features = features.to(device)
            label_t = torch.LongTensor([label]).to(device)
            
            logits, attn = model(features)
            loss = criterion(logits, label_t)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1).item()
            train_correct += (pred == label)
            train_total += 1
        
        scheduler.step()
        avg_train_loss = train_loss / max(train_total, 1)
        
        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for i in range(len(val_ds)):
                features, label, _ = val_ds[i]
                features = features.to(device)
                label_t = torch.LongTensor([label]).to(device)
                
                logits, _ = model(features)
                loss = criterion(logits, label_t)
                
                val_loss += loss.item()
                val_preds.append(logits.argmax(dim=1).item())
                val_labels.append(label)
        
        avg_val_loss = val_loss / max(len(val_ds), 1)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": lr_now,
        })
        
        # Logging
        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            marker = " ★"
        else:
            patience_counter += 1
        
        logger.info(
            f"  {epoch:5d} │ {avg_train_loss:10.4f} │ {avg_val_loss:8.4f} │ "
            f"{val_acc:6.1%} │ {val_f1:5.1%} │ {lr_now:8.6f} │ {elapsed:4.0f}s{marker}"
        )
        
        # Early stopping
        if patience_counter >= cfg.patience:
            logger.info(f"\n  Early stopping at epoch {epoch} (patience={cfg.patience})")
            break
    
    logger.info(f"  {'─'*72}")
    logger.info(f"  Best: epoch {best_epoch}, val_f1={best_val_f1:.4f}")
    
    # ---- Test Evaluation ----
    logger.info("━" * 60)
    logger.info("STEP 5: Test Evaluation")
    logger.info("━" * 60)
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    test_preds = []
    test_labels = []
    test_probs = []
    test_details = []
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            features, label, slide_id = test_ds[i]
            features = features.to(device)
            
            logits, attn = model(features)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = logits.argmax(dim=1).item()
            
            test_preds.append(pred)
            test_labels.append(label)
            test_probs.append(probs)
            test_details.append({
                "slide_id": slide_id,
                "true": label,
                "pred": pred,
                "true_name": cfg.class_names[label],
                "pred_name": cfg.class_names[pred],
                "probs": probs.tolist(),
                "correct": pred == label,
            })
    
    # Metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_f1_weighted = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    
    # AUC
    try:
        test_probs_np = np.array(test_probs)
        test_auc = roc_auc_score(test_labels, test_probs_np, multi_class="ovr", average="macro")
    except:
        test_auc = 0.0
    
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(cfg.num_classes)))
    
    logger.info(f"\n  ┌──────────────────────────────────┐")
    logger.info(f"  │        TEST RESULTS              │")
    logger.info(f"  ├──────────────────────────────────┤")
    logger.info(f"  │  Accuracy:      {test_acc:6.2%}          │")
    logger.info(f"  │  F1 (macro):    {test_f1_macro:6.2%}          │")
    logger.info(f"  │  F1 (weighted): {test_f1_weighted:6.2%}          │")
    logger.info(f"  │  AUC-ROC:       {test_auc:6.2%}          │")
    logger.info(f"  └──────────────────────────────────┘")
    
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"  {'':20s} Predicted")
    logger.info(f"  {'':20s} {'Normal':>8s} {'BCC':>8s} {'SCC':>8s} {'Melano':>8s}")
    for i, name in enumerate(cfg.class_names):
        row = "  ".join(f"{cm[i][j]:6d}" for j in range(cfg.num_classes))
        logger.info(f"  {name:20s} {row}")
    
    logger.info(f"\n  Classification Report:")
    report = classification_report(
        test_labels, test_preds,
        target_names=cfg.class_names,
        zero_division=0,
    )
    for line in report.split("\n"):
        logger.info(f"  {line}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_classes": cfg.num_classes,
            "class_names": cfg.class_names,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
            "feature_extractor": "ResNet18",
            "device": cfg.device,
        },
        "metrics": {
            "accuracy": test_acc,
            "f1_macro": test_f1_macro,
            "f1_weighted": test_f1_weighted,
            "auc_roc": test_auc,
        },
        "confusion_matrix": cm.tolist(),
        "history": history,
        "test_details": test_details,
        "split_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "class_counts": dict(class_counts),
    }
    
    results_path = cfg.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    summary_path = cfg.output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("4-Class Skin Cancer MIL Results (ResNet18)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy:      {test_acc:.4f}\n")
        f.write(f"F1 (macro):    {test_f1_macro:.4f}\n")
        f.write(f"F1 (weighted): {test_f1_weighted:.4f}\n")
        f.write(f"AUC-ROC:       {test_auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\n{report}\n")
    
    logger.info(f"\n  Results saved: {results_path}")
    logger.info(f"  Summary saved: {summary_path}")
    logger.info(f"  Model saved:   {checkpoint_path}")
    
    return results

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = Config()
    
    # Set seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    logger, log_file = setup_logging(cfg.output_dir)
    
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  4-Class Skin Cancer MIL Pipeline                       ║")
    logger.info("║  Feature Extractor: ResNet18                            ║")
    logger.info("║  Classes: Normal/Benign | BCC | SCC | Melanoma          ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Device:    {cfg.device}")
    logger.info(f"  Log file:  {log_file}")
    logger.info(f"  Output:    {cfg.output_dir}")
    logger.info(f"  Data root: {cfg.data_root}")
    
    t_start = time.time()
    
    # Step 1: Labels
    entries = create_unified_labels(cfg)
    
    # Step 2: Tiles
    extract_tiles(entries, cfg)
    
    # Step 3: Features
    extract_features(entries, cfg)
    
    # Step 4+5: Train & Evaluate
    results = train_mil(entries, cfg)
    
    elapsed = time.time() - t_start
    logger.info(f"\n  Total time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    logger.info("  Done! ✓")

if __name__ == "__main__":
    main()
