#!/usr/bin/env python3
"""
    1. ResNet18      (512-d)
    2. ResNet50      (2048-d)
    3. ConvNeXt-Small (768-d)
    4. ConvNeXt-Base  (1024-d)
    5. DINOv2-base   (768-d)
    6. Phikon        (768-d)
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
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from PIL import Image

MODEL_CONFIGS = [
    {
        "name": "ResNet18",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/resnet18.pth",
        "feat_dim": 512,
        "loader": "resnet18",
    },
    {
        "name": "ResNet50",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/resnet50.pth",
        "feat_dim": 2048,
        "loader": "resnet50",
    },
    {
        "name": "ConvNeXt-Small",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/convnext_small.pth",
        "feat_dim": 768,
        "loader": "convnext_small",
    },
    {
        "name": "ConvNeXt-Base",
        "type": "torchvision",
        "weights_path": "/mnt/d/skin_cancer_project/models/torchvision/convnext_base.pth",
        "feat_dim": 1024,
        "loader": "convnext_base",
    },
    {
        "name": "DINOv2-base",
        "type": "dinov2",
        "weights_path": "/mnt/d/skin_cancer_project/models/vision/dinov2-base",
        "feat_dim": 768,
        "loader": "dinov2",
    },
    {
        "name": "Phikon",
        "type": "phikon",
        "weights_path": "/mnt/d/skin_cancer_project/models/pathology/phikon",
        "feat_dim": 768,
        "loader": "phikon",
    },
]

# ============================================================
# GLOBAL CONFIG
# ============================================================
class Config:
    data_root = Path("/mnt/d/skin_cancer_project/datasets")
    base_output = Path("/home/byalc/phase1_project/results")
    tile_dir = Path("/home/byalc/phase1_project/data/tiles_4class")
    base_feature_dir = Path("/home/byalc/phase1_project/data")
    
    num_classes = 4
    class_names = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
    
    tile_size = 256
    max_tiles_per_slide = 200
    tissue_threshold = 0.5
    batch_size_feat = 64
    
    num_epochs = 100
    lr = 2e-4
    weight_decay = 5e-4
    patience = 20          # was 10, more tolerance for F1 fluctuation
    warmup_epochs = 5
    label_smoothing = 0.1
    mil_hidden = 256
    mil_attention = 128
    dropout = 0.3
    
    seed = 42
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # v2 suffix for output dirs (preserves v1 results)
    version = "v2"
    
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
# LOGGING
# ============================================================
def setup_logging(log_file):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ]
    )
    return logging.getLogger(__name__)

# ============================================================
# UNIFIED LABELS (shared across all models)
# ============================================================
def create_unified_labels(cfg):
    logger = logging.getLogger(__name__)
    entries = []
    
    # COBRA BCC
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
    
    # COBRA OOD
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
    
    # TCGA-SKCM
    tcga_dir = cfg.data_root / "tcga_skcm"
    for svs in tcga_dir.glob("*.svs"):
        entries.append({
            "slide_path": str(svs), "slide_id": svs.stem,
            "superclass": 3, "subclass": "Melanoma (TCGA)", "source": "tcga_skcm",
        })
    
    counts = Counter(e["superclass"] for e in entries)
    logger.info(f"  Labels: {len(entries)} slides → " +
                ", ".join(f"{cfg.class_names[i]}={counts[i]}" for i in range(cfg.num_classes)))
    return entries

# ============================================================
# FEATURE EXTRACTOR LOADER
# ============================================================
def load_feature_extractor(model_cfg, device):
    """Load a feature extractor model and return (model, transform)."""
    logger = logging.getLogger(__name__)
    name = model_cfg["name"]
    mtype = model_cfg["type"]
    wpath = model_cfg["weights_path"]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    if mtype == "torchvision":
        loader_name = model_cfg["loader"]
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
        logger.info(f"  Loaded {name} from {wpath}")
    
    elif mtype == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(wpath, local_files_only=True)
        logger.info(f"  Loaded {name} from {wpath}")
    
    elif mtype == "phikon":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(wpath, local_files_only=True)
        logger.info(f"  Loaded {name} from {wpath}")
    
    model = model.to(device)
    model.eval()
    return model, transform

def extract_features_from_model(model, transform, model_cfg, entries, feature_dir, cfg):
    """Extract features for all slides using one model."""
    logger = logging.getLogger(__name__)
    device = torch.device(cfg.device)
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    is_transformer = model_cfg["type"] in ("dinov2", "phikon")
    
    total = len(entries)
    extracted = 0
    skipped = 0
    
    for idx, entry in enumerate(entries, 1):
        slide_id = entry["slide_id"]
        superclass = entry["superclass"]
        feat_path = feature_dir / f"{slide_id}.pt"
        
        if feat_path.exists():
            skipped += 1
            continue
        
        tile_dir = cfg.tile_dir / f"class_{superclass}" / slide_id
        if not tile_dir.exists():
            continue
        
        tiles = sorted(tile_dir.glob("*.png"))
        if len(tiles) < 5:
            continue
        
        all_features = []
        for batch_start in range(0, len(tiles), cfg.batch_size_feat):
            batch_tiles = tiles[batch_start:batch_start + cfg.batch_size_feat]
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
        
        if idx % 100 == 0 or idx <= 3 or idx == total:
            logger.info(f"    [{idx:4d}/{total}] {slide_id[:30]:30s} → {features.shape}")
    
    logger.info(f"    Features: {extracted} new, {skipped} cached")
    return extracted + skipped

# ============================================================
# MIL MODEL + TRAINING
# ============================================================
class GatedAttentionMIL(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, attn_dim=128, num_classes=4, dropout=0.25):
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

class SlideDataset(Dataset):
    def __init__(self, slide_list, feature_dir):
        self.slides = []
        for slide_id, label in slide_list:
            fp = feature_dir / f"{slide_id}.pt"
            if fp.exists():
                self.slides.append((fp, label, slide_id))
    
    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, idx):
        fp, label, sid = self.slides[idx]
        return torch.load(fp, weights_only=True), label, sid

def train_and_evaluate(entries, feature_dir, feat_dim, model_name, output_dir, cfg):
    """Full MIL train + eval for one feature extractor."""
    logger = logging.getLogger(__name__)
    device = torch.device(cfg.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slide_list = [(e["slide_id"], e["superclass"]) for e in entries
                  if (feature_dir / f"{e['slide_id']}.pt").exists()]
    labels = [s[1] for s in slide_list]
    cc = Counter(labels)
    
    logger.info(f"    Slides: {len(slide_list)} → " +
                ", ".join(f"C{i}={cc.get(i,0)}" for i in range(cfg.num_classes)))
    
    ids = list(range(len(slide_list)))
    train_ids, temp_ids = train_test_split(ids, test_size=0.3, stratify=labels, random_state=cfg.seed)
    temp_labels = [labels[i] for i in temp_ids]
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, stratify=temp_labels, random_state=cfg.seed)
    
    train_ds = SlideDataset([slide_list[i] for i in train_ids], feature_dir)
    val_ds = SlideDataset([slide_list[i] for i in val_ids], feature_dir)
    test_ds = SlideDataset([slide_list[i] for i in test_ids], feature_dir)
    
    logger.info(f"    Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_labels = [labels[i] for i in train_ids]
    total = len(train_labels)
    weights = [total / (cfg.num_classes * max(train_labels.count(c), 1)) for c in range(cfg.num_classes)]
    class_weights = torch.FloatTensor(weights).to(device)
    
    model = GatedAttentionMIL(feat_dim, cfg.mil_hidden, cfg.mil_attention,
                               cfg.num_classes, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs  # linear warmup
        progress = (epoch - cfg.warmup_epochs) / max(cfg.num_epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))  # cosine decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"    MIL params: {params:,}")
    logger.info(f"    {'Epoch':>5} │ {'TrLoss':>7} │ {'VlLoss':>7} │ {'VlAcc':>6} │ {'VlF1':>6} │ {'Time':>5}")
    logger.info(f"    {'─'*55}")
    
    best_f1 = 0
    best_epoch = 0
    patience_ctr = 0
    ckpt = output_dir / "best_model.pt"
    history = []
    
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        
        model.train()
        tr_loss = 0
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        for i in indices:
            feat, lab, _ = train_ds[i]
            feat = feat.to(device)
            lab_t = torch.LongTensor([lab]).to(device)
            logits, _ = model(feat)
            loss = criterion(logits, lab_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()
        
        model.eval()
        vl_loss = 0
        vl_preds, vl_labels = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                feat, lab, _ = val_ds[i]
                feat = feat.to(device)
                lab_t = torch.LongTensor([lab]).to(device)
                logits, _ = model(feat)
                vl_loss += criterion(logits, lab_t).item()
                vl_preds.append(logits.argmax(1).item())
                vl_labels.append(lab)
        
        avg_tr = tr_loss / max(len(train_ds), 1)
        avg_vl = vl_loss / max(len(val_ds), 1)
        v_acc = accuracy_score(vl_labels, vl_preds)
        v_f1 = f1_score(vl_labels, vl_preds, average="macro", zero_division=0)
        elapsed = time.time() - t0
        
        history.append({"epoch": epoch, "train_loss": avg_tr, "val_loss": avg_vl,
                        "val_acc": v_acc, "val_f1": v_f1})
        
        marker = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt)
            marker = " ★"
        else:
            patience_ctr += 1
        
        logger.info(f"    {epoch:5d} │ {avg_tr:7.4f} │ {avg_vl:7.4f} │ {v_acc:5.1%} │ {v_f1:5.1%} │ {elapsed:4.0f}s{marker}")
        
        if patience_ctr >= cfg.patience:
            logger.info(f"    Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    t_preds, t_labels, t_probs = [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            feat, lab, _ = test_ds[i]
            feat = feat.to(device)
            logits, _ = model(feat)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            t_preds.append(logits.argmax(1).item())
            t_labels.append(lab)
            t_probs.append(probs)
    
    acc = accuracy_score(t_labels, t_preds)
    f1_mac = f1_score(t_labels, t_preds, average="macro", zero_division=0)
    f1_w = f1_score(t_labels, t_preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(t_labels, np.array(t_probs), multi_class="ovr", average="macro")
    except:
        auc = 0.0
    cm = confusion_matrix(t_labels, t_preds, labels=list(range(cfg.num_classes)))
    
    report = classification_report(t_labels, t_preds, target_names=cfg.class_names, zero_division=0)
    
    logger.info(f"\n    ┌─── {model_name} TEST RESULTS ───┐")
    logger.info(f"    │ Accuracy:   {acc:6.2%}              │")
    logger.info(f"    │ F1 macro:   {f1_mac:6.2%}              │")
    logger.info(f"    │ F1 weight:  {f1_w:6.2%}              │")
    logger.info(f"    │ AUC-ROC:    {auc:6.2%}              │")
    logger.info(f"    └──────────────────────────────┘")
    for line in report.split("\n"):
        logger.info(f"    {line}")
    
    # Save
    results = {
        "model": model_name, "feat_dim": feat_dim,
        "metrics": {"accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_w, "auc_roc": auc},
        "confusion_matrix": cm.tolist(), "history": history,
        "best_epoch": best_epoch, "total_slides": len(slide_list),
        "split": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"4-Class Skin Cancer MIL Results ({model_name})\n{'='*50}\n")
        f.write(f"Accuracy:      {acc:.4f}\nF1 (macro):    {f1_mac:.4f}\n")
        f.write(f"F1 (weighted): {f1_w:.4f}\nAUC-ROC:       {auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n\n{report}\n")
    
    return {"model": model_name, "accuracy": acc, "f1_macro": f1_mac,
            "f1_weighted": f1_w, "auc_roc": auc, "best_epoch": best_epoch}

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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = cfg.base_output / f"multi_model_training_{timestamp}.log"
    cfg.base_output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(master_log)
    
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Multi-Model 4-Class Skin Cancer MIL Pipeline           ║")
    logger.info("║  Feature Extractors: ResNet18/50, ConvNeXt, DINOv2,     ║")
    logger.info("║                      Phikon                             ║")
    logger.info("║  Classes: Normal/Benign | BCC | SCC | Melanoma          ║")
    logger.info("║  Method: MIL (Gated Attention)                          ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Device: {cfg.device}")
    logger.info(f"  Log: {master_log}")
    
    t_total = time.time()
    
    # Step 1: Labels
    logger.info("\n" + "━" * 60)
    logger.info("STEP 1: Unified Labels")
    logger.info("━" * 60)
    entries = create_unified_labels(cfg)
    
    # Check tiles exist
    tiles_exist = cfg.tile_dir.exists() and any(cfg.tile_dir.rglob("*.png"))
    if not tiles_exist:
        logger.error("Tiles not found! Run train_mil_4class.py first for tile extraction.")
        sys.exit(1)
    logger.info(f"  Tiles: cached at {cfg.tile_dir}")
    
    # Step 2: Train each model
    all_results = []
    
    for mi, mcfg in enumerate(MODEL_CONFIGS, 1):
        name = mcfg["name"]
        feat_dim = mcfg["feat_dim"]
        wpath = mcfg["weights_path"]
        
        logger.info(f"\n{'━' * 60}")
        logger.info(f"MODEL {mi}/{len(MODEL_CONFIGS)}: {name} (feat_dim={feat_dim})")
        logger.info(f"{'━' * 60}")
        
        # Check weights exist
        if not Path(wpath).exists():
            logger.warning(f"  ⚠ Weights not found: {wpath} — SKIPPING")
            continue
        
        safe_name = name.lower().replace("-", "_").replace(" ", "_")
        feature_dir = cfg.base_feature_dir / f"features_4class_{safe_name}"
        output_dir = cfg.base_output / f"mil_4class_{safe_name}_{cfg.version}"
        
        t_model = time.time()
        
        # Feature extraction
        logger.info(f"  Feature extraction → {feature_dir}")
        try:
            device = torch.device(cfg.device)
            model, transform = load_feature_extractor(mcfg, device)
            n = extract_features_from_model(model, transform, mcfg, entries, feature_dir, cfg)
            del model
            torch.cuda.empty_cache()
            logger.info(f"  Features ready: {n} slides")
        except Exception as e:
            logger.error(f"  Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # MIL Training
        logger.info(f"  MIL Training → {output_dir}")
        try:
            result = train_and_evaluate(entries, feature_dir, feat_dim, name, output_dir, cfg)
            all_results.append(result)
            logger.info(f"  {name} done in {(time.time()-t_model)/60:.1f} min")
        except Exception as e:
            logger.error(f"  Training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    logger.info(f"\n{'═' * 70}")
    logger.info(f"  FINAL MODEL COMPARISON")
    logger.info(f"{'═' * 70}")
    logger.info(f"  {'Model':<20s} │ {'Accuracy':>8s} │ {'F1 macro':>8s} │ {'F1 wt':>8s} │ {'AUC':>8s} │ {'Epoch':>5s}")
    logger.info(f"  {'─'*70}")
    
    best_model = None
    best_f1 = 0
    for r in sorted(all_results, key=lambda x: x["f1_macro"], reverse=True):
        marker = ""
        if r["f1_macro"] > best_f1:
            best_f1 = r["f1_macro"]
            best_model = r["model"]
            marker = " ◄ BEST"
        logger.info(
            f"  {r['model']:<20s} │ {r['accuracy']:7.2%} │ {r['f1_macro']:7.2%} │ "
            f"{r['f1_weighted']:7.2%} │ {r['auc_roc']:7.2%} │ {r['best_epoch']:5d}{marker}"
        )
    
    logger.info(f"  {'─'*70}")
    logger.info(f"  🏆 Best model: {best_model} (F1 macro = {best_f1:.4f})")
    
    # Save comparison
    comp_path = cfg.base_output / "model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump({"timestamp": timestamp, "results": all_results,
                   "best_model": best_model}, f, indent=2)
    
    elapsed = time.time() - t_total
    logger.info(f"\n  Total time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    logger.info(f"  Comparison saved: {comp_path}")
    logger.info("  All done! ✓")

if __name__ == "__main__":
    main()
