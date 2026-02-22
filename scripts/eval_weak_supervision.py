#!/usr/bin/env python3
"""
Evaluate the trained weak supervision tile classifier.
Loads pseudo_labels.json and best_tile_classifier.pt, runs test evaluation.
"""
import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    output_dir = Path("/home/byalc/phase1_project/results/weak_supervision")
    pseudo_labels_path = output_dir / "pseudo_labels.json"
    model_path = output_dir / "best_tile_classifier.pt"

    num_classes = 4
    class_names = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
    tile_batch_size = 64
    agg_top_k = 50
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATASET
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
# MAIN
# ============================================================
def main():
    cfg = Config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)
    device = torch.device(cfg.device)

    logger.info("=" * 60)
    logger.info("  Weak Supervision — Standalone Evaluation")
    logger.info("=" * 60)

    # Load pseudo-labels
    logger.info(f"  Loading pseudo-labels: {cfg.pseudo_labels_path}")
    with open(cfg.pseudo_labels_path) as f:
        pseudo_labels = json.load(f)
    logger.info(f"  Total tile entries: {len(pseudo_labels)}")

    # Reproduce the same slide split (same seed, same logic as training)
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

    test_set = set(test_slides)
    test_tiles = [e for e in pseudo_labels if e["slide_id"] in test_set]
    logger.info(f"  Test slides: {len(test_slides)}, Test tiles: {len(test_tiles)}")

    # Transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = TileDataset(test_tiles, val_transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.tile_batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, cfg.num_classes)
    )
    model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    logger.info(f"  Model loaded: {cfg.model_path}")

    # Evaluate
    t_preds, t_labels = [], []
    slide_predictions = defaultdict(lambda: {"preds": [], "probs": [], "true_class": None})

    with torch.no_grad():
        for images, labels, slide_ids_batch in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().tolist()

            t_preds.extend(preds)
            t_labels.extend(labels.tolist())

            for i, sid in enumerate(slide_ids_batch):
                slide_predictions[sid]["preds"].append(preds[i])
                slide_predictions[sid]["probs"].append(probs[i])

    # Assign true class
    entry_map = {e["slide_id"]: e["slide_class"] for e in pseudo_labels}
    for sid in slide_predictions:
        slide_predictions[sid]["true_class"] = entry_map.get(sid, -1)

    # ── Tile-level results ──
    tile_acc = accuracy_score(t_labels, t_preds)
    tile_f1 = f1_score(t_labels, t_preds, average="macro", zero_division=0)
    tile_cm = confusion_matrix(t_labels, t_preds, labels=list(range(cfg.num_classes)))
    tile_report = classification_report(t_labels, t_preds,
                                        target_names=cfg.class_names, zero_division=0)

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

    # ── Slide-level aggregation ──
    logger.info(f"\n  SLIDE-LEVEL Aggregation (mean probability):")
    slide_preds, slide_trues = [], []

    for sid, data in slide_predictions.items():
        if data["true_class"] == -1:
            continue
        probs = np.array(data["probs"])
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

    logger.info(f"\n  Results saved: {rpath}")
    logger.info(f"  Summary saved: {spath}")
    logger.info("  Done!")


if __name__ == "__main__":
    main()
