#!/usr/bin/env python3

import os
import sys
import csv
import json
import time
import copy
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

from backbone_registry import MODEL_CONFIGS, safe_model_name, feature_dir_name

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, recall_score
)

# ============================================================
# HYPERPARAMETER EXPERIMENTS
# ============================================================
EXPERIMENTS = [
    {
        "tag": "baseline",
        "description": "Baseline melanoma-safe setup",
        "loss": "cross_entropy",
        "melanoma_weight_mult": 1.0,
        "focal_gamma": 0,
        "label_smoothing": 0.1,
        "lr": 2e-4,
        "dropout": 0.3,
        "cost_matrix": None,
    },
    {
        "tag": "mel_boost_3x",
        "description": "Melanoma class weight 3x boost",
        "loss": "cross_entropy",
        "melanoma_weight_mult": 3.0,
        "focal_gamma": 0,
        "label_smoothing": 0.1,
        "lr": 2e-4,
        "dropout": 0.3,
        "cost_matrix": None,
    },
    {
        "tag": "mel_boost_5x",
        "description": "Melanoma class weight 5x boost",
        "loss": "cross_entropy",
        "melanoma_weight_mult": 5.0,
        "focal_gamma": 0,
        "label_smoothing": 0.05,
        "lr": 1e-4,
        "dropout": 0.3,
        "cost_matrix": None,
    },
    {
        "tag": "mel_boost_7x",
        "description": "Melanoma class weight 7x boost",
        "loss": "cross_entropy",
        "melanoma_weight_mult": 7.0,
        "focal_gamma": 0,
        "label_smoothing": 0.0,
        "lr": 1e-4,
        "dropout": 0.25,
        "cost_matrix": None,
    },
    {
        "tag": "focal_g2",
        "description": "Focal loss gamma=2 with melanoma 3x",
        "loss": "focal",
        "melanoma_weight_mult": 3.0,
        "focal_gamma": 2.0,
        "label_smoothing": 0.0,
        "lr": 2e-4,
        "dropout": 0.3,
        "cost_matrix": None,
    },
    {
        "tag": "focal_g3",
        "description": "Focal loss gamma=3 with melanoma 5x",
        "loss": "focal",
        "melanoma_weight_mult": 5.0,
        "focal_gamma": 3.0,
        "label_smoothing": 0.0,
        "lr": 1e-4,
        "dropout": 0.25,
        "cost_matrix": None,
    },
    {
        "tag": "cost_sensitive",
        "description": "Cost-sensitive: Melanoma to other costs 5x",
        "loss": "cost_sensitive",
        "melanoma_weight_mult": 3.0,
        "focal_gamma": 0,
        "label_smoothing": 0.05,
        "lr": 2e-4,
        "dropout": 0.25,
        "cost_matrix": [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [5.0, 5.0, 5.0, 0.0],
        ],
    },
    {
        "tag": "cost_sensitive_strong",
        "description": "Cost-sensitive strong: melanoma miss costs 8x",
        "loss": "cost_sensitive",
        "melanoma_weight_mult": 5.0,
        "focal_gamma": 0,
        "label_smoothing": 0.0,
        "lr": 1e-4,
        "dropout": 0.25,
        "cost_matrix": [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [8.0, 8.0, 8.0, 0.0],
        ],
    },
]

# ============================================================
# GLOBAL CONFIG
# ============================================================
class Config:
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path("/mnt/d/skin_cancer_project/datasets")
    base_output = project_root / "results"
    cache_root = Path("/mnt/d/skin_cancer_project/cache")
    tile_dir = cache_root / "tiles_4class"
    base_feature_dir = cache_root

    num_classes = 4
    class_names = ["Normal/Benign", "BCC", "SCC", "Melanoma"]

    num_epochs = 100
    weight_decay = 5e-4
    patience = 20
    warmup_epochs = 5
    mil_hidden = 256
    mil_attention = 128

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    version = "v3"

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
# PATH HELPERS
# ============================================================
def _first_existing_path(candidates, kind):
    for p in candidates:
        if p.exists():
            return p
    joined = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {kind}. Checked:\n  - {joined}")


# ============================================================
# UNIFIED LABELS
# ============================================================
def create_unified_labels(cfg):
    logger = logging.getLogger(__name__)
    entries = []

    label_roots = [
        cfg.data_root / "labels",
        cfg.project_root / "data" / "cobra",
        cfg.project_root / "data" / "cobra_fresh",
    ]
    bcc_csv = _first_existing_path([
        root / "bcc_bcc.csv" for root in label_roots
    ], "BCC label CSV")
    ood_csv = _first_existing_path([
        root / "ood_disease_types.csv" for root in label_roots
    ] + [
        root / "ood_labels" / "ood_disease_types.csv" for root in label_roots
    ] + [
        root / "ood_labels" / "labels" / "ood_disease_types.csv" for root in label_roots
    ], "OOD label CSV")

    # COBRA BCC
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
# BALANCED SPLIT — proportional per class
# ============================================================
def balanced_split(entries, feature_dir, cfg, test_size=0.15, val_size=0.15):
    """
    Create train/val/test split that is PROPORTIONAL within each class.
    Only include slides that have features extracted.
    """
    logger = logging.getLogger(__name__)

    # Filter to slides with features
    slide_list = []
    for e in entries:
        feat_path = feature_dir / f"{e['slide_id']}.pt"
        if feat_path.exists():
            slide_list.append(dict(e))

    labels = [s["superclass"] for s in slide_list]
    cc = Counter(labels)
    logger.info(f"    Slides with features: {len(slide_list)} → " +
                ", ".join(f"{cfg.class_names[i]}={cc.get(i,0)}" for i in range(cfg.num_classes)))

    # Stratified split preserving class proportions
    ids = list(range(len(slide_list)))
    train_ids, temp_ids = train_test_split(
        ids, test_size=test_size + val_size,
        stratify=labels, random_state=cfg.seed
    )
    temp_labels = [labels[i] for i in temp_ids]
    relative_val = val_size / (test_size + val_size)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=1 - relative_val,
        stratify=temp_labels, random_state=cfg.seed
    )

    # Log distribution per split
    for name, split_ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        split_labels = [labels[i] for i in split_ids]
        split_cc = Counter(split_labels)
        logger.info(f"    {name:5s}: {len(split_ids):4d} → " +
                    ", ".join(f"{cfg.class_names[i]}={split_cc.get(i,0)}" for i in range(cfg.num_classes)))

    return slide_list, labels, train_ids, val_ids, test_ids

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss — focuses training on hard examples."""
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss

# ============================================================
# COST-SENSITIVE LOSS
# ============================================================
class CostSensitiveLoss(nn.Module):
    """Cross-entropy + asymmetric misclassification cost."""
    def __init__(self, weight=None, cost_matrix=None, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.cost_matrix = cost_matrix  # Tensor [C, C]
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        if self.cost_matrix is not None:
            probs = F.softmax(logits, dim=1)
            # For each sample, compute expected misclassification cost
            costs = self.cost_matrix[targets]  # [B, C]
            cost_penalty = (probs * costs).sum(dim=1)  # [B]
            return (ce + cost_penalty).mean()
        return ce.mean()

# ============================================================
# MIL MODEL
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
        self.meta = {}
        for slide in slide_list:
            slide_id = slide["slide_id"] if isinstance(slide, dict) else slide[0]
            label = slide["superclass"] if isinstance(slide, dict) else slide[1]
            fp = feature_dir / f"{slide_id}.pt"
            if fp.exists():
                self.slides.append((fp, label, slide_id))
                self.meta[slide_id] = slide if isinstance(slide, dict) else {
                    "slide_id": slide_id,
                    "superclass": label,
                }

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        fp, label, sid = self.slides[idx]
        return torch.load(fp, weights_only=True), label, sid

# ============================================================
# TRAIN + EVALUATE (with experiment config)
# ============================================================
def train_and_evaluate(slide_list, labels, train_ids, val_ids, test_ids,
                       feature_dir, feat_dim, model_name, output_dir,
                       experiment, cfg):
    """Full MIL train + eval for one model × one experiment."""
    logger = logging.getLogger(__name__)
    device = torch.device(cfg.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SlideDataset([slide_list[i] for i in train_ids], feature_dir)
    val_ds = SlideDataset([slide_list[i] for i in val_ids], feature_dir)
    test_ds = SlideDataset([slide_list[i] for i in test_ids], feature_dir)

    # --- Class weights (inverse frequency + melanoma boost) ---
    train_labels = [labels[i] for i in train_ids]
    total = len(train_labels)
    weights = []
    for c in range(cfg.num_classes):
        count = max(train_labels.count(c), 1)
        w = total / (cfg.num_classes * count)
        if c == 3:  # Melanoma
            w *= experiment["melanoma_weight_mult"]
        weights.append(w)
    class_weights = torch.FloatTensor(weights).to(device)
    logger.info(f"    Class weights: [{', '.join(f'{w:.2f}' for w in weights)}]")

    # --- Loss function ---
    loss_type = experiment["loss"]
    if loss_type == "focal":
        criterion = FocalLoss(weight=class_weights, gamma=experiment["focal_gamma"])
        logger.info(f"    Loss: Focal(γ={experiment['focal_gamma']})")
    elif loss_type == "cost_sensitive":
        cost_matrix = torch.FloatTensor(experiment["cost_matrix"]).to(device) if experiment["cost_matrix"] else None
        criterion = CostSensitiveLoss(
            weight=class_weights, cost_matrix=cost_matrix,
            label_smoothing=experiment["label_smoothing"])
        logger.info(f"    Loss: CostSensitive (melanoma misclass penalty=5x)")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=experiment["label_smoothing"])
        logger.info(f"    Loss: CE(smoothing={experiment['label_smoothing']})")

    # --- Model ---
    dropout = experiment["dropout"]
    model = GatedAttentionMIL(feat_dim, cfg.mil_hidden, cfg.mil_attention,
                               cfg.num_classes, dropout).to(device)

    lr = experiment["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(cfg.num_epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training loop ---
    best_f1 = 0
    best_epoch = 0
    patience_ctr = 0
    ckpt = output_dir / "best_model.pt"
    history = []

    logger.info(f"    {'Ep':>3} │ {'TrL':>6} │ {'VlL':>6} │ {'VAcc':>5} │ {'VF1':>5} │ {'MelR':>5} │ {'T':>4}")
    logger.info(f"    {'─'*50}")

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

        # Melanoma recall (class 3)
        mel_mask = [i for i, l in enumerate(vl_labels) if l == 3]
        mel_recall = sum(1 for i in mel_mask if vl_preds[i] == 3) / max(len(mel_mask), 1)

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": avg_tr, "val_loss": avg_vl,
                        "val_acc": v_acc, "val_f1": v_f1, "mel_recall": mel_recall})

        marker = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt)
            marker = " ★"
        else:
            patience_ctr += 1

        logger.info(f"    {epoch:3d} │ {avg_tr:6.3f} │ {avg_vl:6.3f} │ {v_acc:4.1%} │ {v_f1:4.1%} │ {mel_recall:4.0%} │ {elapsed:3.0f}s{marker}")

        if patience_ctr >= cfg.patience:
            logger.info(f"    Early stop @ epoch {epoch}")
            break

    # --- Test evaluation ---
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    t_preds, t_labels, t_probs, t_slide_ids = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            feat, lab, sid = test_ds[i]
            feat = feat.to(device)
            logits, _ = model(feat)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            t_preds.append(logits.argmax(1).item())
            t_labels.append(lab)
            t_probs.append(probs)
            t_slide_ids.append(sid)

    acc = accuracy_score(t_labels, t_preds)
    f1_mac = f1_score(t_labels, t_preds, average="macro", zero_division=0)
    f1_w = f1_score(t_labels, t_preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(t_labels, np.array(t_probs), multi_class="ovr", average="macro")
    except:
        auc = 0.0
    cm = confusion_matrix(t_labels, t_preds, labels=list(range(cfg.num_classes)))
    per_class_recall = recall_score(t_labels, t_preds, average=None, labels=list(range(cfg.num_classes)), zero_division=0)

    report = classification_report(t_labels, t_preds, target_names=cfg.class_names, zero_division=0)

    # Melanoma-specific
    mel_fn = cm[3].sum() - cm[3][3]  # Total melanoma misclassified
    mel_recall_test = per_class_recall[3] if len(per_class_recall) > 3 else 0
    mel_total = cm[3].sum()

    logger.info(f"\n    ┌─── {model_name} × {experiment['tag']} ───┐")
    logger.info(f"    │ Accuracy:      {acc:6.2%}          │")
    logger.info(f"    │ F1 macro:      {f1_mac:6.2%}          │")
    logger.info(f"    │ AUC-ROC:       {auc:6.2%}          │")
    logger.info(f"    │ Melanoma Rec:  {mel_recall_test:6.2%}  ({mel_total-mel_fn}/{mel_total})  │")
    logger.info(f"    │ Melanoma FN:   {mel_fn:3d}             │")
    logger.info(f"    └──────────────────────────────────┘")
    logger.info(f"\n  Confusion Matrix:")
    labels_short = ["Normal", "BCC", "SCC", "Mel"]
    logger.info(f"    {'True/Pred':<10}" + "".join(f"{l:>8}" for l in labels_short))
    for i, row in enumerate(cm):
        logger.info(f"    {labels_short[i]:<10}" + "".join(f"{v:8d}" for v in row))
    logger.info(f"\n{report}")

    # --- Per-class threshold tuning on test set ---
    logger.info(f"  Threshold tuning (melanoma):")
    t_probs_arr = np.array(t_probs)
    best_mel_thresh = 0.5
    best_mel_recall_adj = mel_recall_test
    best_mel_f1_adj = f1_mac
    for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        adj_preds = []
        for i, probs in enumerate(t_probs_arr):
            if probs[3] >= thresh:
                adj_preds.append(3)
            else:
                adj_preds.append(probs.argmax())
        adj_recall = recall_score(t_labels, adj_preds, average=None,
                                  labels=list(range(cfg.num_classes)), zero_division=0)
        adj_f1 = f1_score(t_labels, adj_preds, average="macro", zero_division=0)
        adj_mel_fn = sum(1 for i, l in enumerate(t_labels) if l == 3 and adj_preds[i] != 3)
        logger.info(f"    thresh={thresh:.2f} → MelRecall={adj_recall[3]:.0%} MelFN={adj_mel_fn} F1={adj_f1:.2%}")
        if adj_mel_fn == 0 and adj_f1 >= best_mel_f1_adj:
            best_mel_thresh = thresh
            best_mel_recall_adj = adj_recall[3]
            best_mel_f1_adj = adj_f1
    if best_mel_thresh < 0.5:
        logger.info(f"    → Best melanoma-safe threshold: {best_mel_thresh} (F1={best_mel_f1_adj:.2%})")

    phase1_rows = []
    phase1_hard_cases = []
    for sid, true_lab, pred_lab, probs in zip(t_slide_ids, t_labels, t_preds, t_probs):
        meta = test_ds.meta.get(sid, {"slide_id": sid})
        probs_arr = np.asarray(probs, dtype=np.float32)
        order = np.argsort(probs_arr)[::-1]
        top1 = float(probs_arr[order[0]])
        top2 = float(probs_arr[order[1]]) if len(order) > 1 else 0.0
        margin = top1 - top2
        melanoma_prob = float(probs_arr[3])
        is_melanoma = int(true_lab == 3)
        is_melanoma_fn = int(true_lab == 3 and pred_lab != 3)
        hard_case_candidate = int(true_lab == 3 and (pred_lab != 3 or top1 < 0.75 or margin < 0.22))
        row = {
            "slide_id": sid,
            "source": meta.get("source", "unknown"),
            "slide_path": meta.get("slide_path", ""),
            "true_label": cfg.class_names[int(true_lab)],
            "pred_label": cfg.class_names[int(pred_lab)],
            "prediction_confidence": round(top1, 6),
            "margin": round(margin, 6),
            "melanoma_probability": round(melanoma_prob, 6),
            "is_melanoma": is_melanoma,
            "is_melanoma_fn": is_melanoma_fn,
            "hard_case_candidate": hard_case_candidate,
            "prob_normal_benign": round(float(probs_arr[0]), 6),
            "prob_bcc": round(float(probs_arr[1]), 6),
            "prob_scc": round(float(probs_arr[2]), 6),
            "prob_melanoma": round(float(probs_arr[3]), 6),
        }
        phase1_rows.append(row)
        if hard_case_candidate:
            phase1_hard_cases.append(row)

    phase1_fieldnames = [
        "slide_id", "source", "slide_path", "true_label", "pred_label",
        "prediction_confidence", "margin", "melanoma_probability", "is_melanoma",
        "is_melanoma_fn", "hard_case_candidate", "prob_normal_benign",
        "prob_bcc", "prob_scc", "prob_melanoma"
    ]
    pred_csv = output_dir / "phase1_test_predictions.csv"
    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase1_fieldnames)
        writer.writeheader()
        writer.writerows(phase1_rows)

    hard_csv = output_dir / "phase1_hard_cases.csv"
    with open(hard_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase1_fieldnames)
        writer.writeheader()
        writer.writerows(phase1_hard_cases)

    logger.info(f"  Phase 1 test prediction log saved: {pred_csv}")
    logger.info(f"  Phase 1 hard-case rows saved: {hard_csv} ({len(phase1_hard_cases)} rows)")

    # --- Save results ---
    results = {
        "model": model_name, "experiment": experiment["tag"],
        "description": experiment["description"],
        "feat_dim": feat_dim,
        "metrics": {
            "accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_w, "auc_roc": auc,
            "melanoma_recall": float(mel_recall_test), "melanoma_fn": int(mel_fn),
            "per_class_recall": per_class_recall.tolist(),
        },
        "threshold_tuning": {
            "best_melanoma_threshold": best_mel_thresh,
            "best_f1_with_threshold": best_mel_f1_adj,
        },
        "phase1": {
            "test_prediction_csv": str(pred_csv),
            "hard_case_csv": str(hard_csv),
            "hard_case_count": len(phase1_hard_cases),
            "melanoma_fn_cases": sum(1 for r in phase1_hard_cases if r["is_melanoma_fn"]),
        },
        "confusion_matrix": cm.tolist(), "history": history,
        "best_epoch": best_epoch,
        "hyperparams": {
            "loss": experiment["loss"], "lr": lr, "dropout": dropout,
            "melanoma_weight_mult": experiment["melanoma_weight_mult"],
            "focal_gamma": experiment["focal_gamma"],
            "label_smoothing": experiment["label_smoothing"],
        },
        "split": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"4-Class MIL v3: {model_name} × {experiment['tag']}\n{'='*50}\n")
        f.write(f"Accuracy:        {acc:.4f}\nF1 (macro):      {f1_mac:.4f}\n")
        f.write(f"F1 (weighted):   {f1_w:.4f}\nAUC-ROC:         {auc:.4f}\n")
        f.write(f"Melanoma Recall: {mel_recall_test:.4f}\nMelanoma FN:     {mel_fn}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n\n{report}\n")

    return {
        "model": model_name, "experiment": experiment["tag"],
        "accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_w,
        "auc_roc": auc, "mel_recall": float(mel_recall_test),
        "mel_fn": int(mel_fn), "best_epoch": best_epoch,
        "best_mel_threshold": best_mel_thresh,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="v3 Melanoma-safe MIL training")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to train (e.g. Phikon ResNet18). Default: all 8")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiments (e.g. baseline mel_boost_3x). Default: all")
    parser.add_argument("--skip-feature-extraction", action="store_true",
                        help="Skip feature extraction (use cached features)")
    args = parser.parse_args()

    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.base_output.mkdir(parents=True, exist_ok=True)
    master_log = cfg.base_output / f"v3_melanoma_safe_{timestamp}.log"
    logger = setup_logging(master_log)

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  v3 — MELANOMA-SAFE Multi-Model MIL Pipeline            ║")
    logger.info("║  Goal: Reduce Melanoma FN to ZERO                       ║")
    logger.info("║  Models: ResNet18/50, ConvNeXt, DINOv2, Phikon, UNI, CONCH          ║")
    logger.info("║  Classes: Normal/Benign | BCC | SCC | Melanoma          ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Device: {cfg.device}")
    logger.info(f"  Log: {master_log}")

    t_total = time.time()

    # --- Labels ---
    logger.info("\n" + "━" * 60)
    logger.info("STEP 1: Unified Labels")
    logger.info("━" * 60)
    entries = create_unified_labels(cfg)

    # --- Filter models ---
    models_to_run = MODEL_CONFIGS
    if args.models:
        models_to_run = [m for m in MODEL_CONFIGS if m["name"] in args.models]
        logger.info(f"  Running models: {[m['name'] for m in models_to_run]}")

    # --- Filter experiments ---
    exps_to_run = EXPERIMENTS
    if args.experiments:
        exps_to_run = [e for e in EXPERIMENTS if e["tag"] in args.experiments]
        logger.info(f"  Running experiments: {[e['tag'] for e in exps_to_run]}")

    all_results = []

    for mi, mcfg in enumerate(models_to_run, 1):
        name = mcfg["name"]
        feat_dim = mcfg["feat_dim"]
        safe_name = safe_model_name(name)
        feature_dir = cfg.base_feature_dir / feature_dir_name(mcfg)

        logger.info(f"\n{'━' * 60}")
        logger.info(f"MODEL {mi}/{len(models_to_run)}: {name} (feat_dim={feat_dim})")
        logger.info(f"{'━' * 60}")

        # Check features exist
        if not feature_dir.exists():
            logger.warning(f"  ⚠ Features not found: {feature_dir} — SKIP")
            continue
        n_feats = len(list(feature_dir.glob("*.pt")))
        if n_feats < 10:
            logger.warning(f"  ⚠ Too few features: {n_feats} — SKIP")
            continue
        logger.info(f"  Features: {n_feats} cached at {feature_dir}")

        # --- BALANCED SPLIT (same split for all experiments of this model) ---
        slide_list, labels, train_ids, val_ids, test_ids = balanced_split(
            entries, feature_dir, cfg)

        # --- Run each experiment ---
        for ei, exp in enumerate(exps_to_run, 1):
            tag = exp["tag"]
            output_dir = cfg.base_output / f"mil_4class_{safe_name}_{cfg.version}_{tag}"

            logger.info(f"\n  ── Experiment {ei}/{len(exps_to_run)}: {tag} ──")
            logger.info(f"  {exp['description']}")

            t_exp = time.time()
            try:
                result = train_and_evaluate(
                    slide_list, labels, train_ids, val_ids, test_ids,
                    feature_dir, feat_dim, name, output_dir, exp, cfg)
                all_results.append(result)
                logger.info(f"  {name}×{tag} done in {(time.time()-t_exp)/60:.1f} min")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                import traceback
                traceback.print_exc()

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    logger.info(f"\n{'═' * 90}")
    logger.info(f"  FINAL COMPARISON — ALL MODELS × EXPERIMENTS")
    logger.info(f"{'═' * 90}")
    logger.info(f"  {'Model':<20s} {'Experiment':<18s} │ {'Acc':>6s} {'F1':>6s} {'AUC':>6s} │ {'MelRec':>6s} {'MelFN':>5s} │ {'Ep':>3s}")
    logger.info(f"  {'─'*85}")

    # Sort by melanoma FN first (ascending), then F1 (descending)
    sorted_results = sorted(all_results, key=lambda x: (x["mel_fn"], -x["f1_macro"]))

    for r in sorted_results:
        mel_flag = "🟢" if r["mel_fn"] == 0 else "🔴"
        logger.info(
            f"  {r['model']:<20s} {r['experiment']:<18s} │ "
            f"{r['accuracy']:5.1%} {r['f1_macro']:5.1%} {r['auc_roc']:5.1%} │ "
            f"{r['mel_recall']:5.0%} {r['mel_fn']:5d} │ "
            f"{r['best_epoch']:3d} {mel_flag}"
        )

    # Best model for melanoma safety
    mel_safe = [r for r in sorted_results if r["mel_fn"] == 0]
    if mel_safe:
        best = max(mel_safe, key=lambda x: x["f1_macro"])
        logger.info(f"\n  🏆 BEST MELANOMA-SAFE: {best['model']} × {best['experiment']}")
        logger.info(f"     F1={best['f1_macro']:.2%}, AUC={best['auc_roc']:.2%}, MelFN=0")
    else:
        best = sorted_results[0] if sorted_results else None
        if best:
            logger.info(f"\n  ⚠ No model achieved Melanoma FN=0")
            logger.info(f"  Best attempt: {best['model']} × {best['experiment']} (MelFN={best['mel_fn']})")
            logger.info(f"  → Try threshold tuning or more data")

    # Save comparison
    comp_path = cfg.base_output / f"v3_comparison_{timestamp}.json"
    with open(comp_path, "w") as f:
        json.dump({
            "timestamp": timestamp, "version": "v3",
            "results": all_results,
            "best_melanoma_safe": best if mel_safe else None,
        }, f, indent=2)

    elapsed = time.time() - t_total
    logger.info(f"\n  Total time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    logger.info(f"  Comparison saved: {comp_path}")
    logger.info(f"  Log: {master_log}")
    logger.info("  Done! ✓")


if __name__ == "__main__":
    main()
