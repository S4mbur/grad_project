#!/usr/bin/env python3
"""
=============================================================================
  Weak Supervision Tile-Level Classifier
  
  Pipeline:
    1. Load best MIL model (Phikon v1) → get attention weights per slide
    2. Generate pseudo-labels for tiles:
       - High-attention tiles → slide's class label
       - Low-attention tiles in non-normal slides → "Normal" (background)
       - All tiles in Normal slides → "Normal"
    3. Train a tile-level CNN classifier using pseudo-labels
    4. Evaluate: tile-level accuracy + slide-level aggregation
  
  Why Weak Supervision?
    - MIL gives slide-level predictions but no spatial localization
    - Weak supervision transfers knowledge from MIL → tile-level
    - Enables heatmap generation: per-tile cancer probability maps
    - Can also improve slide-level accuracy via tile aggregation
=============================================================================
"""
import os
import sys
import csv
import json
import time
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
class Config:
    # Paths
    data_root = Path("/mnt/d/skin_cancer_project/datasets")
    tile_dir = Path("/home/byalc/phase1_project/data/tiles_4class")
    feature_dir = Path("/home/byalc/phase1_project/data/features_4class_phikon")
    mil_checkpoint = Path("/home/byalc/phase1_project/results/mil_4class_phikon/best_model.pt")
    resnet_weights = Path("/mnt/d/skin_cancer_project/models/torchvision/resnet18.pth")
    output_dir = Path("/home/byalc/phase1_project/results/weak_supervision")
    
    # Labels
    num_classes = 4
    class_names = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
    
    # Pseudo-label generation
    top_k_percent = 0.5     # top 50% attention tiles get slide label
    confidence_threshold = 0.7  # min MIL prediction confidence
    
    # Tile classifier training
    tile_epochs = 30
    tile_lr = 1e-4
    tile_batch_size = 64
    tile_patience = 10
    label_smoothing = 0.1
    
    # Slide aggregation
    agg_top_k = 50  # top-k tile predictions for slide vote
    
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    OOD_CLASS_MAP = {
        "Benign": 0, "No abnormalities": 0,
        "Benign sebaceous gland tumor": 0, "Cylindroma": 0,
        "Basal cell carcinoma": 1,
        "Squamous cell carcinoma": 2,
        "Melanoma": 3, "Melanoma in situ": 3,
        "Merkel cell carcinoma": None, "Sebaceous gland carcinoma": None,
        "Microcystic adnexal carcinoma": None,
        "Skin adnexal carcinoma, other": None,
        "Lymphoma": None, "Cutaneous metastases": None,
    }

# ============================================================
# MIL MODEL (same architecture as training)
# ============================================================
class GatedAttentionMIL(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=256, attn_dim=128, num_classes=4, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes))
    
    def forward(self, x):
        h = self.encoder(x)
        a = self.attention_W(self.attention_V(h) * self.attention_U(h))
        a = F.softmax(a, dim=0)
        z = torch.sum(a * h, dim=0, keepdim=True)
        return self.classifier(z), a.squeeze()

# ============================================================
# LOGGING
# ============================================================
def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"weak_sup_{timestamp}.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ]
    )
    return logging.getLogger(__name__), log_file

# ============================================================
# STEP 1: CREATE UNIFIED LABELS
# ============================================================
def create_unified_labels(cfg):
    logger = logging.getLogger(__name__)
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
                    "slide_path": str(tif_path), "slide_id": fname,
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
                    "slide_path": str(tif_path), "slide_id": fname,
                    "superclass": sc, "subclass": cat, "source": "cobra_ood",
                })
    
    tcga_dir = cfg.data_root / "tcga_skcm"
    for svs in tcga_dir.glob("*.svs"):
        entries.append({
            "slide_path": str(svs), "slide_id": svs.stem,
            "superclass": 3, "subclass": "Melanoma (TCGA)", "source": "tcga_skcm",
        })
    
    counts = Counter(e["superclass"] for e in entries)
    logger.info(f"  Slides: {len(entries)} -> " +
                ", ".join(f"{cfg.class_names[i]}={counts[i]}" for i in range(cfg.num_classes)))
    return entries

# ============================================================
# STEP 2: GENERATE PSEUDO-LABELS FROM MIL ATTENTION
# ============================================================
def generate_pseudo_labels(entries, cfg):
    """
    Use trained MIL model to assign pseudo-labels to individual tiles.
    
    Strategy:
      - For each slide, run MIL → get prediction + attention weights
      - If MIL predicts correctly (high confidence):
        - Top attention tiles → slide's true class
        - Bottom attention tiles → class 0 (Normal) for non-normal slides
      - Normal slides: all tiles → class 0
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 1: Generating Pseudo-Labels from MIL Attention")
    logger.info("=" * 60)
    
    device = torch.device(cfg.device)
    
    # Load MIL model
    mil_model = GatedAttentionMIL(feat_dim=768, hidden_dim=256, attn_dim=128,
                                   num_classes=4, dropout=0.25)
    mil_model.load_state_dict(torch.load(cfg.mil_checkpoint, map_location=device, weights_only=True))
    mil_model = mil_model.to(device)
    mil_model.eval()
    logger.info(f"  MIL model loaded: {cfg.mil_checkpoint}")
    
    pseudo_labels = []  # list of (tile_path, pseudo_label, confidence, attention_rank)
    stats = {"total_slides": 0, "labeled_slides": 0, "total_tiles": 0,
             "class_dist": Counter()}
    
    for idx, entry in enumerate(entries, 1):
        slide_id = entry["slide_id"]
        true_class = entry["superclass"]
        
        # Load features
        feat_path = cfg.feature_dir / f"{slide_id}.pt"
        if not feat_path.exists():
            continue
        
        # Get tile paths
        tile_dir_path = cfg.tile_dir / f"class_{true_class}" / slide_id
        if not tile_dir_path.exists():
            continue
        tile_paths = sorted(tile_dir_path.glob("*.png"))
        if len(tile_paths) < 5:
            continue
        
        stats["total_slides"] += 1
        
        # Run MIL
        features = torch.load(feat_path, weights_only=True).to(device)
        with torch.no_grad():
            logits, attention = mil_model(features)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = probs.argmax()
            confidence = probs[true_class]
        
        attention = attention.cpu().numpy()
        n_tiles = min(len(tile_paths), len(attention))
        
        if true_class == 0:
            # Normal slide: all tiles are Normal
            for i in range(n_tiles):
                pseudo_labels.append({
                    "tile_path": str(tile_paths[i]),
                    "label": 0,
                    "slide_id": slide_id,
                    "slide_class": true_class,
                    "attention": float(attention[i]),
                    "confidence": float(confidence),
                })
                stats["class_dist"][0] += 1
                stats["total_tiles"] += 1
        else:
            # Disease slide: use attention to assign labels
            if confidence < cfg.confidence_threshold:
                # Low confidence → skip this slide (noisy)
                continue
            
            # Sort tiles by attention
            attn_order = np.argsort(attention)[::-1]  # high → low
            n_top = int(n_tiles * cfg.top_k_percent)
            n_top = max(n_top, 5)  # at least 5
            
            for rank, tile_idx in enumerate(attn_order):
                if tile_idx >= n_tiles:
                    continue
                
                if rank < n_top:
                    # High attention → disease class
                    label = true_class
                else:
                    # Low attention → normal/background
                    label = 0
                
                pseudo_labels.append({
                    "tile_path": str(tile_paths[tile_idx]),
                    "label": label,
                    "slide_id": slide_id,
                    "slide_class": true_class,
                    "attention": float(attention[tile_idx]),
                    "confidence": float(confidence),
                })
                stats["class_dist"][label] += 1
                stats["total_tiles"] += 1
        
        stats["labeled_slides"] += 1
        
        if idx % 200 == 0 or idx <= 3:
            logger.info(f"  [{idx}/{len(entries)}] {slide_id[:25]} "
                        f"class={true_class} conf={confidence:.2f} tiles={n_tiles}")
    
    logger.info(f"\n  Pseudo-label stats:")
    logger.info(f"    Slides processed: {stats['labeled_slides']}/{stats['total_slides']}")
    logger.info(f"    Total tiles labeled: {stats['total_tiles']}")
    for c in range(cfg.num_classes):
        logger.info(f"    Class {c} ({cfg.class_names[c]}): {stats['class_dist'][c]}")
    
    # Save pseudo-labels
    pl_path = cfg.output_dir / "pseudo_labels.json"
    with open(pl_path, "w") as f:
        json.dump(pseudo_labels, f)
    logger.info(f"  Saved: {pl_path}")
    
    return pseudo_labels

# ============================================================
# STEP 3: TILE DATASET
# ============================================================
class TileDataset(Dataset):
    def __init__(self, tile_entries, transform=None):
        self.entries = tile_entries
        self.transform = transform
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["tile_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, entry["label"], entry["slide_id"]

# ============================================================
# STEP 4: TRAIN TILE-LEVEL CLASSIFIER
# ============================================================
def train_tile_classifier(pseudo_labels, entries, cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Training Tile-Level Classifier")
    logger.info("=" * 60)
    
    device = torch.device(cfg.device)
    
    # Split by slide (not by tile!) to avoid data leakage
    slide_ids = list(set(e["slide_id"] for e in pseudo_labels))
    slide_labels = {}
    for e in pseudo_labels:
        slide_labels[e["slide_id"]] = e["slide_class"]
    
    slide_label_list = [slide_labels[s] for s in slide_ids]
    
    train_slides, temp_slides = train_test_split(
        slide_ids, test_size=0.3, stratify=slide_label_list, random_state=cfg.seed)
    temp_label_list = [slide_labels[s] for s in temp_slides]
    val_slides, test_slides = train_test_split(
        temp_slides, test_size=0.5, stratify=temp_label_list, random_state=cfg.seed)
    
    train_set = set(train_slides)
    val_set = set(val_slides)
    test_set = set(test_slides)
    
    train_tiles = [e for e in pseudo_labels if e["slide_id"] in train_set]
    val_tiles = [e for e in pseudo_labels if e["slide_id"] in val_set]
    test_tiles = [e for e in pseudo_labels if e["slide_id"] in test_set]
    
    logger.info(f"  Slide split: Train={len(train_slides)}, Val={len(val_slides)}, Test={len(test_slides)}")
    logger.info(f"  Tile split:  Train={len(train_tiles)}, Val={len(val_tiles)}, Test={len(test_tiles)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_ds = TileDataset(train_tiles, train_transform)
    val_ds = TileDataset(val_tiles, val_transform)
    test_ds = TileDataset(test_tiles, val_transform)
    
    # Weighted sampler for class balance
    train_label_counts = Counter(e["label"] for e in train_tiles)
    sample_weights = []
    for e in train_tiles:
        w = 1.0 / train_label_counts[e["label"]]
        sample_weights.append(w)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.tile_batch_size,
                               sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.tile_batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.tile_batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    
    # Model: ResNet18 fine-tuned
    model = models.resnet18()
    state_dict = torch.load(cfg.resnet_weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # Freeze early layers, fine-tune later layers
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, cfg.num_classes)
    )
    model = model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ResNet18: {total_params:,} total, {trainable:,} trainable")
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg.tile_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.tile_epochs)
    
    # Class weights
    total_train = len(train_tiles)
    weights = [total_train / (cfg.num_classes * max(train_label_counts[c], 1))
               for c in range(cfg.num_classes)]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    
    logger.info(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    
    best_f1 = 0
    best_epoch = 0
    patience_ctr = 0
    ckpt = cfg.output_dir / "best_tile_classifier.pt"
    history = []
    
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    total_train_tiles = len(train_tiles)
    total_val_tiles = len(val_tiles)
    log_every = max(n_train_batches // 10, 1)  # log ~10 times per epoch
    
    logger.info(f"  Batches per epoch: {n_train_batches} train, {n_val_batches} val")
    logger.info(f"  Progress updates every {log_every} batches (~10 per epoch)")
    logger.info(f"\n  {'Epoch':>5} | {'TrLoss':>7} | {'VlLoss':>7} | {'VlAcc':>6} | {'VlF1':>6} | {'LR':>8} | {'Time':>5}")
    logger.info(f"  {'_'*65}")
    
    training_start = time.time()
    
    for epoch in range(1, cfg.tile_epochs + 1):
        t0 = time.time()
        
        # Train
        model.train()
        tr_loss = 0
        tr_correct = 0
        tr_count = 0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            bs = images.size(0)
            tr_loss += loss.item() * bs
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_count += bs
            
            # Intra-epoch progress
            if batch_idx % log_every == 0 or batch_idx == n_train_batches:
                pct = batch_idx / n_train_batches * 100
                run_loss = tr_loss / tr_count
                run_acc = tr_correct / tr_count * 100
                elapsed_batch = time.time() - t0
                tiles_per_sec = tr_count / max(elapsed_batch, 0.1)
                remaining_batches = n_train_batches - batch_idx
                eta_epoch = remaining_batches * (elapsed_batch / batch_idx)
                
                # Total ETA: remaining this epoch + remaining epochs
                epochs_left = cfg.tile_epochs - epoch
                avg_epoch_time = elapsed_batch / (batch_idx / n_train_batches)
                eta_total = eta_epoch + (epochs_left * avg_epoch_time)
                
                eta_str = f"{int(eta_epoch//60)}m{int(eta_epoch%60)}s"
                total_str = f"{int(eta_total//60)}m"
                
                logger.info(
                    f"    E{epoch:02d} [{batch_idx:4d}/{n_train_batches}] "
                    f"{pct:5.1f}% | loss={run_loss:.4f} acc={run_acc:.1f}% | "
                    f"{tiles_per_sec:.0f} tiles/s | ETA epoch: {eta_str} total: {total_str}"
                )
        
        scheduler.step()
        avg_tr = tr_loss / max(tr_count, 1)
        tr_acc = tr_correct / max(tr_count, 1) * 100
        
        # Val
        model.eval()
        vl_loss = 0
        vl_count = 0
        vl_preds, vl_labels = [], []
        val_t0 = time.time()
        with torch.no_grad():
            for val_batch_idx, (images, labels, _) in enumerate(val_loader, 1):
                images = images.to(device)
                labels_dev = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels_dev)
                vl_loss += loss.item() * images.size(0)
                vl_count += images.size(0)
                vl_preds.extend(logits.argmax(1).cpu().tolist())
                vl_labels.extend(labels.tolist())
        
        val_time = time.time() - val_t0
        avg_vl = vl_loss / max(vl_count, 1)
        v_acc = accuracy_score(vl_labels, vl_preds)
        v_f1 = f1_score(vl_labels, vl_preds, average="macro", zero_division=0)
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0
        total_elapsed = time.time() - training_start
        
        history.append({"epoch": epoch, "train_loss": avg_tr, "val_loss": avg_vl,
                        "val_acc": v_acc, "val_f1": v_f1})
        
        marker = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt)
            marker = " ★ BEST"
        else:
            patience_ctr += 1
        
        # Per-class val accuracy
        per_class_acc = {}
        for c in range(cfg.num_classes):
            c_mask = [i for i, l in enumerate(vl_labels) if l == c]
            if c_mask:
                c_correct = sum(1 for i in c_mask if vl_preds[i] == c)
                per_class_acc[cfg.class_names[c]] = c_correct / len(c_mask)
        
        logger.info(f"\n  ── Epoch {epoch}/{cfg.tile_epochs} Summary ──")
        logger.info(f"  Train: loss={avg_tr:.4f}  acc={tr_acc:.1f}%  ({tr_count} tiles)")
        logger.info(f"  Val:   loss={avg_vl:.4f}  acc={v_acc:.1%}  F1={v_f1:.4f}")
        per_cls_str = "  ".join(f"{n[:6]}={a:.0%}" for n, a in per_class_acc.items())
        logger.info(f"  Val per-class: {per_cls_str}")
        logger.info(f"  LR={lr_now:.6f}  Time={epoch_time:.0f}s (train={epoch_time-val_time:.0f}s val={val_time:.0f}s)")
        logger.info(f"  Patience: {patience_ctr}/{cfg.tile_patience}  Best: E{best_epoch} F1={best_f1:.4f}  Total: {total_elapsed/60:.1f}min{marker}")
        
        if patience_ctr >= cfg.tile_patience:
            logger.info(f"\n  ⛔ Early stopping at epoch {epoch}")
            break
    
    logger.info(f"  Best: epoch {best_epoch}, val_f1={best_f1:.4f}")
    
    # ── Test evaluation ──
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Test Evaluation")
    logger.info("=" * 60)
    
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    
    # Tile-level metrics
    t_preds, t_labels = [], []
    slide_predictions = defaultdict(lambda: {"preds": [], "probs": [], "true_class": None})
    
    with torch.no_grad():
        for images, labels, slide_ids in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().tolist()
            
            t_preds.extend(preds)
            t_labels.extend(labels.tolist())
            
            for i, sid in enumerate(slide_ids):
                slide_predictions[sid]["preds"].append(preds[i])
                slide_predictions[sid]["probs"].append(probs[i])
    
    # Assign true class from entries
    entry_map = {e["slide_id"]: e["slide_class"] for e in pseudo_labels}
    for sid in slide_predictions:
        slide_predictions[sid]["true_class"] = entry_map.get(sid, -1)
    
    # Tile-level
    tile_acc = accuracy_score(t_labels, t_preds)
    tile_f1 = f1_score(t_labels, t_preds, average="macro", zero_division=0)
    tile_cm = confusion_matrix(t_labels, t_preds, labels=list(range(cfg.num_classes)))
    tile_report = classification_report(t_labels, t_preds, target_names=cfg.class_names, zero_division=0)
    
    logger.info(f"\n  TILE-LEVEL Results:")
    logger.info(f"    Accuracy: {tile_acc:.4f}")
    logger.info(f"    F1 macro: {tile_f1:.4f}")
    logger.info(f"\n  Confusion Matrix:")
    labels_short = ["Normal", "BCC", "SCC", "Melanoma"]
    header = "True / Pred"
    logger.info(f"    {header:<12}" + "".join(f"{l:>9}" for l in labels_short))
    for i, row in enumerate(tile_cm):
        logger.info(f"    {labels_short[i]:<12}" + "".join(f"{v:9d}" for v in row))
    logger.info(f"\n{tile_report}")
    
    # Slide-level aggregation (majority vote on top-k confident tiles)
    logger.info(f"\n  SLIDE-LEVEL Aggregation (top-{cfg.agg_top_k} tile voting):")
    slide_preds, slide_trues = [], []
    
    for sid, data in slide_predictions.items():
        if data["true_class"] == -1:
            continue
        
        probs = np.array(data["probs"])
        # Use mean probability as slide prediction
        mean_probs = probs.mean(axis=0)
        slide_pred = mean_probs.argmax()
        
        slide_preds.append(slide_pred)
        slide_trues.append(data["true_class"])
    
    if slide_preds:
        slide_acc = accuracy_score(slide_trues, slide_preds)
        slide_f1 = f1_score(slide_trues, slide_preds, average="macro", zero_division=0)
        slide_cm = confusion_matrix(slide_trues, slide_preds, labels=list(range(cfg.num_classes)))
        slide_report = classification_report(slide_trues, slide_preds,
                                              target_names=cfg.class_names, zero_division=0)
        
        logger.info(f"    Slides evaluated: {len(slide_preds)}")
        logger.info(f"    Accuracy: {slide_acc:.4f}")
        logger.info(f"    F1 macro: {slide_f1:.4f}")
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"    {header:<12}" + "".join(f"{l:>9}" for l in labels_short))
        for i, row in enumerate(slide_cm):
            logger.info(f"    {labels_short[i]:<12}" + "".join(f"{v:9d}" for v in row))
        logger.info(f"\n{slide_report}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "weak_supervision",
        "teacher": "Phikon MIL (Gated Attention)",
        "student": "ResNet18 (fine-tuned layer3+4)",
        "tile_metrics": {
            "accuracy": tile_acc,
            "f1_macro": tile_f1,
            "confusion_matrix": tile_cm.tolist(),
        },
        "slide_metrics": {
            "accuracy": slide_acc if slide_preds else 0,
            "f1_macro": slide_f1 if slide_preds else 0,
            "confusion_matrix": slide_cm.tolist() if slide_preds else [],
            "num_slides": len(slide_preds),
        },
        "config": {
            "top_k_percent": cfg.top_k_percent,
            "confidence_threshold": cfg.confidence_threshold,
            "tile_epochs": cfg.tile_epochs,
            "best_epoch": best_epoch,
        },
        "history": history,
    }
    
    rpath = cfg.output_dir / "results.json"
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    
    spath = cfg.output_dir / "summary.txt"
    with open(spath, "w") as f:
        f.write("Weak Supervision Tile Classifier Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Teacher: Phikon MIL\n")
        f.write(f"Student: ResNet18 fine-tuned\n\n")
        f.write(f"Tile-level:  Acc={tile_acc:.4f}  F1={tile_f1:.4f}\n")
        if slide_preds:
            f.write(f"Slide-level: Acc={slide_acc:.4f}  F1={slide_f1:.4f}\n")
        f.write(f"\nTile Confusion Matrix:\n{tile_cm}\n")
        if slide_preds:
            f.write(f"\nSlide Confusion Matrix:\n{slide_cm}\n")
        f.write(f"\n{tile_report}\n")
    
    logger.info(f"\n  Results: {rpath}")
    logger.info(f"  Model:   {ckpt}")
    logger.info(f"  Summary: {spath}")
    
    return results

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    logger, log_file = setup_logging(cfg.output_dir)
    
    logger.info("=" * 60)
    logger.info("  Weak Supervision Pipeline")
    logger.info("  Teacher: Phikon MIL (best model)")
    logger.info("  Student: ResNet18 tile classifier")
    logger.info("  Classes: Normal/Benign | BCC | SCC | Melanoma")
    logger.info("=" * 60)
    logger.info(f"  Device: {cfg.device}")
    logger.info(f"  Log: {log_file}")
    
    t_start = time.time()
    
    # Labels
    entries = create_unified_labels(cfg)
    
    # Step 1: Pseudo-labels from MIL
    pseudo_labels = generate_pseudo_labels(entries, cfg)
    
    # Step 2+3: Train and evaluate tile classifier
    results = train_tile_classifier(pseudo_labels, entries, cfg)
    
    elapsed = time.time() - t_start
    logger.info(f"\n  Total time: {elapsed/60:.1f} min")
    logger.info("  Done!")

if __name__ == "__main__":
    main()
