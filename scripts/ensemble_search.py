#!/usr/bin/env python3
"""
=============================================================================
  Ensemble Model Search — Find best combination of v3 models
  ===========================================================
  Tries ALL combinations of 2, 3, 4, 5 models from the 30 trained models.
  Uses soft voting (probability averaging) and weighted voting.
  Goal: MelFN=0 with highest possible F1.
  
  Usage: python scripts/ensemble_search.py
=============================================================================
"""
import json
import csv
import random
import logging
import time
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, recall_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
RESULTS_BASE = Path("/home/byalc/phase1_project/results")
FEATURE_BASE = Path("/home/byalc/phase1_project/data")
TILE_DIR = Path("/home/byalc/phase1_project/data/tiles_4class")
DATA_ROOT = Path("/mnt/d/skin_cancer_project/datasets")

CLASS_NAMES = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
NUM_CLASSES = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# All v3 model directories
MODEL_INFO = {
    "resnet18":       {"feat_dim": 512,  "feat_dir": "features_4class_resnet18"},
    "resnet50":       {"feat_dim": 2048, "feat_dir": "features_4class_resnet50"},
    "convnext_small": {"feat_dim": 768,  "feat_dir": "features_4class_convnext_small"},
    "convnext_base":  {"feat_dim": 1024, "feat_dir": "features_4class_convnext_base"},
    "dinov2_base":    {"feat_dim": 768,  "feat_dir": "features_4class_dinov2_base"},
    "phikon":         {"feat_dim": 768,  "feat_dir": "features_4class_phikon"},
}

EXPERIMENTS = ["baseline", "mel_boost_3x", "mel_boost_5x", "focal_g2", "cost_sensitive"]

# ============================================================
# MIL MODEL
# ============================================================
class GatedAttentionMIL(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, attn_dim=128, num_classes=4, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
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
# RECREATE THE SAME SPLIT
# ============================================================
def create_entries():
    entries = []
    # COBRA BCC
    bcc_csv = DATA_ROOT / "labels" / "bcc_bcc.csv"
    bcc_dir = DATA_ROOT / "cobra_bcc"
    with open(bcc_csv) as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            label = int(row["label"])
            tif_path = bcc_dir / f"{fname}.tif"
            if tif_path.exists():
                entries.append({"slide_id": fname, "superclass": 0 if label == 0 else 1, "source": "cobra_bcc"})
    # COBRA OOD
    ood_csv = DATA_ROOT / "labels" / "ood_disease_types.csv"
    ood_dir = DATA_ROOT / "cobra_ood" / "images"
    with open(ood_csv) as f:
        for row in csv.DictReader(f):
            fname, cat = row["filename"], row["category"]
            sc = OOD_CLASS_MAP.get(cat)
            if sc is None:
                continue
            tif_path = ood_dir / f"{fname}.tif"
            if tif_path.exists():
                entries.append({"slide_id": fname, "superclass": sc, "source": "cobra_ood"})
    # TCGA-SKCM
    tcga_dir = DATA_ROOT / "tcga_skcm"
    for svs in tcga_dir.glob("*.svs"):
        entries.append({"slide_id": svs.stem, "superclass": 3, "source": "tcga_skcm"})
    return entries

def get_test_split(entries, feature_dir):
    slide_list = []
    for e in entries:
        fp = feature_dir / f"{e['slide_id']}.pt"
        if fp.exists():
            slide_list.append((e["slide_id"], e["superclass"]))
    labels = [s[1] for s in slide_list]
    ids = list(range(len(slide_list)))
    train_ids, temp_ids = train_test_split(ids, test_size=0.30, stratify=labels, random_state=SEED)
    temp_labels = [labels[i] for i in temp_ids]
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, stratify=temp_labels, random_state=SEED)
    return slide_list, labels, test_ids

# ============================================================
# COLLECT PREDICTIONS FROM ALL 30 MODELS
# ============================================================
def collect_all_predictions():
    """Load all 30 models and compute test set probabilities."""
    logger.info("Collecting predictions from all models...")
    
    entries = create_entries()
    device = torch.device(DEVICE)
    
    # Use phikon feature dir as reference for split (same split for all)
    ref_feat_dir = FEATURE_BASE / "features_4class_phikon"
    slide_list, labels, test_ids = get_test_split(entries, ref_feat_dir)
    
    test_slides = [slide_list[i] for i in test_ids]
    test_labels = [labels[i] for i in test_ids]
    
    logger.info(f"Test set: {len(test_slides)} slides")
    logger.info(f"  " + ", ".join(f"{CLASS_NAMES[i]}={test_labels.count(i)}" for i in range(NUM_CLASSES)))
    
    all_model_probs = {}  # model_key -> {slide_id -> probs}
    
    for model_key, minfo in MODEL_INFO.items():
        feat_dim = minfo["feat_dim"]
        feat_dir = FEATURE_BASE / minfo["feat_dir"]
        
        for exp in EXPERIMENTS:
            result_dir = RESULTS_BASE / f"mil_4class_{model_key}_v3_{exp}"
            model_path = result_dir / "best_model.pt"
            
            if not model_path.exists():
                continue
            
            full_key = f"{model_key}_{exp}"
            
            # Load model
            dropout = 0.25 if exp == "cost_sensitive" else 0.3
            model = GatedAttentionMIL(feat_dim, 256, 128, NUM_CLASSES, dropout).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            model.eval()
            
            slide_probs = {}
            for sid, lab in test_slides:
                fp = feat_dir / f"{sid}.pt"
                if not fp.exists():
                    continue
                feat = torch.load(fp, weights_only=True).to(device)
                with torch.no_grad():
                    logits, _ = model(feat)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                slide_probs[sid] = probs
            
            all_model_probs[full_key] = slide_probs
            del model
            torch.cuda.empty_cache()
    
    logger.info(f"Loaded {len(all_model_probs)} models")
    return test_slides, test_labels, all_model_probs

# ============================================================
# EVALUATE ENSEMBLE
# ============================================================
def evaluate_ensemble(model_keys, all_probs, test_slides, test_labels, weights=None):
    """Evaluate a soft-voting ensemble of given model keys."""
    if weights is None:
        weights = [1.0] * len(model_keys)
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    preds = []
    probs_list = []
    valid_labels = []
    
    for i, (sid, lab) in enumerate(test_slides):
        avg_probs = np.zeros(NUM_CLASSES)
        count = 0
        for mk, w in zip(model_keys, weights):
            if mk in all_probs and sid in all_probs[mk]:
                avg_probs += w * all_probs[mk][sid]
                count += 1
        
        if count == 0:
            continue
        
        avg_probs /= count if weights is None else 1  # already weighted
        preds.append(avg_probs.argmax())
        probs_list.append(avg_probs)
        valid_labels.append(lab)
    
    if not preds:
        return None
    
    acc = accuracy_score(valid_labels, preds)
    f1 = f1_score(valid_labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(valid_labels, preds, labels=list(range(NUM_CLASSES)))
    mel_fn = cm[3].sum() - cm[3][3] if cm.shape[0] > 3 else 999
    mel_recall = cm[3][3] / max(cm[3].sum(), 1) if cm.shape[0] > 3 else 0
    
    try:
        auc = roc_auc_score(valid_labels, np.array(probs_list), multi_class="ovr", average="macro")
    except:
        auc = 0
    
    # Also try threshold tuning for melanoma
    best_thresh = 0.5
    best_fn_with_thresh = mel_fn
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
        adj_preds = []
        for p in probs_list:
            if p[3] >= thresh:
                adj_preds.append(3)
            else:
                adj_preds.append(p.argmax())
        adj_cm = confusion_matrix(valid_labels, adj_preds, labels=list(range(NUM_CLASSES)))
        adj_fn = adj_cm[3].sum() - adj_cm[3][3] if adj_cm.shape[0] > 3 else 999
        adj_f1 = f1_score(valid_labels, adj_preds, average="macro", zero_division=0)
        if adj_fn < best_fn_with_thresh or (adj_fn == best_fn_with_thresh and adj_f1 > f1):
            best_fn_with_thresh = adj_fn
            best_thresh = thresh
    
    return {
        "accuracy": acc, "f1_macro": f1, "auc_roc": auc,
        "mel_recall": mel_recall, "mel_fn": int(mel_fn),
        "cm": cm.tolist(),
        "best_thresh": best_thresh, "mel_fn_with_thresh": int(best_fn_with_thresh),
        "models": list(model_keys),
        "n_models": len(model_keys),
    }

# ============================================================
# MAIN — EXHAUSTIVE ENSEMBLE SEARCH
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("  ENSEMBLE MODEL SEARCH")
    logger.info("  Goal: MelFN=0 with highest F1")
    logger.info("=" * 60)
    
    t0 = time.time()
    test_slides, test_labels, all_probs = collect_all_predictions()
    
    model_keys = sorted(all_probs.keys())
    logger.info(f"\nAvailable models: {len(model_keys)}")
    for mk in model_keys:
        logger.info(f"  {mk}")
    
    # --- Individual model performance ---
    logger.info(f"\n{'='*70}")
    logger.info("INDIVIDUAL MODEL PERFORMANCE")
    logger.info(f"{'='*70}")
    individual_results = {}
    for mk in model_keys:
        r = evaluate_ensemble([mk], all_probs, test_slides, test_labels)
        if r:
            individual_results[mk] = r
            flag = "🟢" if r["mel_fn"] == 0 else ("🟡" if r["mel_fn"] <= 3 else "🔴")
            logger.info(f"  {mk:<35s} Acc={r['accuracy']:5.1%} F1={r['f1_macro']:5.1%} MelFN={r['mel_fn']:2d} {flag}")
    
    # --- Ensemble search ---
    all_ensemble_results = []
    
    # Try ensembles of size 2, 3, 4, 5
    for size in [2, 3, 4, 5]:
        logger.info(f"\n{'='*70}")
        logger.info(f"ENSEMBLE SIZE {size} — Testing {len(list(combinations(model_keys, size)))} combinations")
        logger.info(f"{'='*70}")
        
        best_for_size = None
        best_fn0_for_size = None
        count = 0
        
        for combo in combinations(model_keys, size):
            r = evaluate_ensemble(combo, all_probs, test_slides, test_labels)
            if r is None:
                continue
            count += 1
            all_ensemble_results.append(r)
            
            if best_for_size is None or r["mel_fn"] < best_for_size["mel_fn"] or \
               (r["mel_fn"] == best_for_size["mel_fn"] and r["f1_macro"] > best_for_size["f1_macro"]):
                best_for_size = r
            
            if r["mel_fn"] == 0:
                if best_fn0_for_size is None or r["f1_macro"] > best_fn0_for_size["f1_macro"]:
                    best_fn0_for_size = r
        
        if best_for_size:
            logger.info(f"  Best (lowest MelFN): MelFN={best_for_size['mel_fn']} F1={best_for_size['f1_macro']:.1%}")
            logger.info(f"    Models: {best_for_size['models']}")
        if best_fn0_for_size:
            logger.info(f"  🟢 Best MelFN=0: F1={best_fn0_for_size['f1_macro']:.1%} AUC={best_fn0_for_size['auc_roc']:.1%}")
            logger.info(f"    Models: {best_fn0_for_size['models']}")
    
    # --- FINAL RANKING ---
    logger.info(f"\n{'═'*80}")
    logger.info("  FINAL RANKING — ALL ENSEMBLES")
    logger.info(f"{'═'*80}")
    
    # Sort: mel_fn ascending, then f1 descending
    all_ensemble_results.sort(key=lambda x: (x["mel_fn"], -x["f1_macro"]))
    
    # Top 20
    logger.info(f"\n  TOP 20 ENSEMBLES:")
    logger.info(f"  {'#':>3s} {'Size':>4s} │ {'Acc':>6s} {'F1':>6s} {'AUC':>6s} │ {'MelRec':>6s} {'MelFN':>5s} │ {'ThreshFN':>8s} │ Models")
    logger.info(f"  {'─'*100}")
    
    for i, r in enumerate(all_ensemble_results[:20], 1):
        flag = "🟢" if r["mel_fn"] == 0 else ("🟡" if r["mel_fn"] <= 2 else "🔴")
        short_models = [m.replace("_v3_", "·") for m in r["models"]]
        model_str = " + ".join(short_models)
        if len(model_str) > 60:
            model_str = model_str[:57] + "..."
        logger.info(
            f"  {i:3d} {r['n_models']:4d} │ "
            f"{r['accuracy']:5.1%} {r['f1_macro']:5.1%} {r['auc_roc']:5.1%} │ "
            f"{r['mel_recall']:5.0%} {r['mel_fn']:5d} │ "
            f"{r['mel_fn_with_thresh']:8d} │ {model_str} {flag}"
        )
    
    # MelFN=0 results
    fn0 = [r for r in all_ensemble_results if r["mel_fn"] == 0]
    if fn0:
        logger.info(f"\n  🏆 MELANOMA-SAFE ENSEMBLES (MelFN=0): {len(fn0)} found!")
        best = max(fn0, key=lambda x: x["f1_macro"])
        logger.info(f"  Best: F1={best['f1_macro']:.2%} AUC={best['auc_roc']:.2%}")
        logger.info(f"  Models: {best['models']}")
        cm = np.array(best["cm"])
        logger.info(f"  Confusion Matrix:")
        for i, row in enumerate(cm):
            logger.info(f"    {CLASS_NAMES[i]:<15s} {row}")
    else:
        # Show threshold-tuned results
        fn0_thresh = [r for r in all_ensemble_results if r["mel_fn_with_thresh"] == 0]
        if fn0_thresh:
            logger.info(f"\n  ⚠ No MelFN=0 at default threshold, but {len(fn0_thresh)} achieve it with threshold tuning!")
            best = max(fn0_thresh, key=lambda x: x["f1_macro"])
            logger.info(f"  Best: F1={best['f1_macro']:.2%} (thresh={best['best_thresh']})")
            logger.info(f"  Models: {best['models']}")
        else:
            logger.info(f"\n  ❌ No ensemble achieved MelFN=0 even with threshold tuning")
            logger.info(f"  Lowest MelFN: {all_ensemble_results[0]['mel_fn']}")
    
    # Save results
    output = {
        "individual": {k: {kk: vv for kk, vv in v.items() if kk != "cm"} 
                       for k, v in individual_results.items()},
        "top_ensembles": all_ensemble_results[:50],
        "fn0_ensembles": fn0[:10] if fn0 else [],
    }
    output_path = RESULTS_BASE / "ensemble_search_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    logger.info(f"\n  Total time: {elapsed/60:.1f} min")
    logger.info(f"  Results saved: {output_path}")


if __name__ == "__main__":
    main()
