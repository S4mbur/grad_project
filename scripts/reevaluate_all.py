#!/usr/bin/env python3
"""Re-evaluate all 30 v3 models on the corrected test set (2 mislabeled slides removed)."""
import json, csv, sys, random
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report, recall_score)
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_ROOT = Path("/mnt/d/skin_cancer_project/datasets")
FEATURE_BASE = Path("/home/byalc/phase1_project/data")
RESULTS_BASE = Path("/home/byalc/phase1_project/results")
CLASS_NAMES = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OOD_CLASS_MAP = {
    "Benign": 0, "No abnormalities": 0,
    "Benign sebaceous gland tumor": 0, "Cylindroma": 0,
    "Basal cell carcinoma": 1, "Squamous cell carcinoma": 2,
    "Melanoma": 3, "Melanoma in situ": 3,
    "Merkel cell carcinoma": None, "Sebaceous gland carcinoma": None,
    "Microcystic adnexal carcinoma": None,
    "Skin adnexal carcinoma, other": None,
    "Lymphoma": None, "Cutaneous metastases": None,
}

MODEL_CONFIGS = [
    ("ResNet18",       "resnet18",       512,  "features_4class_resnet18"),
    ("ResNet50",       "resnet50",       2048, "features_4class_resnet50"),
    ("ConvNeXt-Small", "convnext_small", 768,  "features_4class_convnext_small"),
    ("ConvNeXt-Base",  "convnext_base",  1024, "features_4class_convnext_base"),
    ("DINOv2-base",    "dinov2_base",    768,  "features_4class_dinov2_base"),
    ("Phikon",         "phikon",         768,  "features_4class_phikon"),
]
EXPERIMENTS = ["baseline", "mel_boost_3x", "mel_boost_5x", "focal_g2", "cost_sensitive"]

class GatedAttentionMIL(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, attn_dim=128, num_classes=4, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))
    def forward(self, x):
        h = self.encoder(x)
        a = self.attention_W(self.attention_V(h) * self.attention_U(h))
        a = F.softmax(a, dim=0)
        z = torch.sum(a * h, dim=0, keepdim=True)
        return self.classifier(z), a.squeeze()

def create_entries():
    entries = []
    bcc_csv = DATA_ROOT / "labels" / "bcc_bcc.csv"
    with open(bcc_csv) as f:
        for row in csv.DictReader(f):
            fname, label = row["filename"], int(row["label"])
            if (DATA_ROOT / "cobra_bcc" / f"{fname}.tif").exists():
                entries.append({"slide_id": fname, "superclass": 0 if label == 0 else 1})
    ood_csv = DATA_ROOT / "labels" / "ood_disease_types.csv"
    with open(ood_csv) as f:
        for row in csv.DictReader(f):
            fname, cat = row["filename"], row["category"]
            sc = OOD_CLASS_MAP.get(cat)
            if sc is None: continue
            if (DATA_ROOT / "cobra_ood" / "images" / f"{fname}.tif").exists():
                entries.append({"slide_id": fname, "superclass": sc})
    for svs in (DATA_ROOT / "tcga_skcm").glob("*.svs"):
        entries.append({"slide_id": svs.stem, "superclass": 3})
    return entries

def get_splits(entries, feature_dir):
    slide_list = []
    for e in entries:
        if (feature_dir / f"{e['slide_id']}.pt").exists():
            slide_list.append((e["slide_id"], e["superclass"]))
    labels = [s[1] for s in slide_list]
    ids = list(range(len(slide_list)))
    train_ids, temp_ids = train_test_split(ids, test_size=0.30, stratify=labels, random_state=SEED)
    temp_labels = [labels[i] for i in temp_ids]
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, stratify=temp_labels, random_state=SEED)
    return slide_list, labels, train_ids, val_ids, test_ids

def evaluate_model(model, slide_list, test_ids, feature_dir, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for idx in test_ids:
        sid, lab = slide_list[idx]
        fp = feature_dir / f"{sid}.pt"
        if not fp.exists(): continue
        feat = torch.load(fp, weights_only=True).to(device)
        with torch.no_grad():
            logits, _ = model(feat)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = probs.argmax()
        all_preds.append(pred)
        all_labels.append(lab)
        all_probs.append(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main():
    entries = create_entries()
    device = torch.device(DEVICE)
    
    # Use phikon features as reference for common slide set
    ref_dir = FEATURE_BASE / "features_4class_phikon"
    slide_list, labels, train_ids, val_ids, test_ids = get_splits(entries, ref_dir)
    
    test_labels_ref = [labels[i] for i in test_ids]
    print(f"Dataset after cleanup:")
    print(f"  Total slides with features: {len(slide_list)}")
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    for c in range(NUM_CLASSES):
        print(f"  Test {CLASS_NAMES[c]}: {test_labels_ref.count(c)}")
    print()
    
    all_results = []
    
    for model_name, model_key, feat_dim, feat_dir_name in MODEL_CONFIGS:
        feat_dir = FEATURE_BASE / feat_dir_name
        # Re-split with this feature set
        sl, lb, tr, vl, te = get_splits(entries, feat_dir)
        
        for exp in EXPERIMENTS:
            result_dir = RESULTS_BASE / f"mil_4class_{model_key}_v3_{exp}"
            model_path = result_dir / "best_model.pt"
            if not model_path.exists():
                continue
            
            dropout = 0.25 if exp == "cost_sensitive" else 0.3
            mil = GatedAttentionMIL(feat_dim, 256, 128, NUM_CLASSES, dropout).to(device)
            mil.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            
            true_labels, preds, probs = evaluate_model(mil, sl, te, feat_dir, device)
            del mil; torch.cuda.empty_cache()
            
            acc = accuracy_score(true_labels, preds)
            f1_macro = f1_score(true_labels, preds, average="macro", zero_division=0)
            f1_weighted = f1_score(true_labels, preds, average="weighted", zero_division=0)
            try:
                auc = roc_auc_score(true_labels, probs, multi_class="ovr", average="macro")
            except:
                auc = 0
            cm = confusion_matrix(true_labels, preds, labels=list(range(NUM_CLASSES)))
            mel_total = cm[3].sum()
            mel_tp = cm[3][3]
            mel_fn = mel_total - mel_tp
            mel_recall = mel_tp / mel_total if mel_total > 0 else 0
            mel_precision = cm[3][3] / cm[:,3].sum() if cm[:,3].sum() > 0 else 0
            
            report = classification_report(true_labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
            
            r = {
                "model": model_name, "model_key": model_key, "experiment": exp,
                "full_key": f"{model_key}_{exp}",
                "accuracy": round(acc, 4), "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4), "auc_roc": round(auc, 4),
                "mel_recall": round(mel_recall, 4), "mel_precision": round(mel_precision, 4),
                "mel_fn": int(mel_fn), "mel_tp": int(mel_tp), "mel_total": int(mel_total),
                "cm": cm.tolist(),
                "per_class": {cn: {
                    "precision": round(report[cn]["precision"], 4),
                    "recall": round(report[cn]["recall"], 4),
                    "f1": round(report[cn]["f1-score"], 4),
                    "support": int(report[cn]["support"]),
                } for cn in CLASS_NAMES},
            }
            all_results.append(r)
            print(f"  {model_name:15s} {exp:18s} Acc={acc:.1%} F1={f1_macro:.1%} AUC={auc:.1%} MelFN={mel_fn}")
    
    # Sort by F1 descending
    all_results.sort(key=lambda x: (-x["f1_macro"]))
    
    # Save
    output_path = RESULTS_BASE / "v3_corrected_results.json"
    with open(output_path, "w") as f:
        json.dump({"test_size": len(test_ids), "results": all_results}, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {output_path}")

if __name__ == "__main__":
    main()
