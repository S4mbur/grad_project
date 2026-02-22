#!/usr/bin/env python3
"""
Multi-Class Training Script
============================

Bu script, multi-class skin cancer classification için model eğitir.

Sınıflar:
    0 = Normal (benign tissue)
    1 = BCC (Basal Cell Carcinoma)
    2 = Melanoma
    (3 = SCC - sonra eklenecek)

Kullanım:
    python scripts/train_multiclass.py
    python scripts/train_multiclass.py --epochs 30 --batch-size 64
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset.tile_dataset import TileDataset, load_tile_manifest
from src.models import create_model
from src.training import Trainer, TrainerConfig
from src.utils.transforms import get_train_transforms, get_eval_transforms


# Class names for multi-class
CLASS_NAMES = ["normal", "bcc", "melanoma"]


def compute_class_weights(labels, num_classes):
    """Compute class weights for imbalanced dataset."""
    counter = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    # Normalize
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    
    return weights


def load_multiclass_tile_manifest(manifest_path, split):
    """Load tile manifest with multi-class labels."""
    import pandas as pd
    
    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split]
    
    tiles = []
    for _, row in df.iterrows():
        tiles.append({
            "path": row["tile_path"],
            "label": int(row["label"]),
            "slide_id": row["slide_id"],
        })
    
    return tiles


class MultiClassTileDataset(torch.utils.data.Dataset):
    """Tile dataset supporting multi-class labels."""
    
    def __init__(self, tiles, transform=None, num_classes=3):
        self.tiles = tiles
        self.transform = transform
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]
        
        from PIL import Image
        img = Image.open(tile["path"]).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        label = tile["label"]
        
        return img, label
    
    def get_labels(self):
        """Get all labels for computing class weights."""
        return [t["label"] for t in self.tiles]
    
    def stats(self):
        """Get class distribution statistics."""
        labels = self.get_labels()
        counter = Counter(labels)
        return {CLASS_NAMES[i]: counter.get(i, 0) for i in range(self.num_classes)}


def main():
    parser = argparse.ArgumentParser(description="Train multi-class patch classifier")
    parser.add_argument("--tile-manifest", type=str, 
                        default="data/manifests/multiclass_tile_manifest.csv",
                        help="Path to multi-class tile manifest")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of classes")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--use-class-weights", action="store_true", default=True,
                        help="Use class weights for imbalanced data")
    parser.add_argument("--checkpoint-name", type=str, 
                        default="multiclass_patch_classifier.pt",
                        help="Checkpoint filename")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    logging.info(f"Number of classes: {args.num_classes}")
    
    tile_manifest = PROJECT_ROOT / args.tile_manifest
    
    if not tile_manifest.exists():
        logging.error(f"Tile manifest not found: {tile_manifest}")
        logging.error("Run create_multiclass_manifest.py and extract tiles first")
        sys.exit(1)
    
    # Load datasets
    logging.info("Loading datasets...")
    
    train_tiles = load_multiclass_tile_manifest(str(tile_manifest), split="train")
    val_tiles = load_multiclass_tile_manifest(str(tile_manifest), split="val")
    
    logging.info(f"Train tiles: {len(train_tiles)}")
    logging.info(f"Val tiles: {len(val_tiles)}")
    
    # Create transforms
    train_transform = get_train_transforms(img_size=224, augmentation=True)
    eval_transform = get_eval_transforms(img_size=224)
    
    # Create datasets
    train_dataset = MultiClassTileDataset(
        train_tiles, transform=train_transform, num_classes=args.num_classes
    )
    val_dataset = MultiClassTileDataset(
        val_tiles, transform=eval_transform, num_classes=args.num_classes
    )
    
    # Log class distribution
    train_stats = train_dataset.stats()
    val_stats = val_dataset.stats()
    
    logging.info("Train class distribution:")
    for cls_name, count in train_stats.items():
        logging.info(f"  {cls_name}: {count}")
    
    logging.info("Val class distribution:")
    for cls_name, count in val_stats.items():
        logging.info(f"  {cls_name}: {count}")
    
    # Compute class weights if needed
    class_weights = None
    if args.use_class_weights:
        train_labels = train_dataset.get_labels()
        class_weights = compute_class_weights(train_labels, args.num_classes)
        logging.info(f"Class weights: {class_weights.tolist()}")
        class_weights = class_weights.to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    logging.info("Creating multi-class model...")
    model = create_model(
        num_classes=args.num_classes,
        pretrained=True,
        dropout=0.5,
        architecture="resnet18",
        device=device,
    )
    
    # Create trainer config
    config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.0001,
        early_stopping_patience=7,
        device=device,
        checkpoint_dir=str(PROJECT_ROOT / "logs" / "checkpoints"),
    )
    
    logging.info("="*60)
    logging.info("MULTI-CLASS TRAINING")
    logging.info(f"Classes: {CLASS_NAMES[:args.num_classes]}")
    logging.info("="*60)
    
    # Create trainer with custom loss if using class weights
    trainer = Trainer(model, train_loader, val_loader, config)
    
    if class_weights is not None:
        trainer.criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info("Using weighted CrossEntropyLoss")
    
    # Train
    results = trainer.train(checkpoint_name=args.checkpoint_name)
    
    logging.info("="*60)
    logging.info("Training complete!")
    logging.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    logging.info(f"Checkpoint saved to: {results['checkpoint_path']}")
    logging.info("="*60)
    
    return results


if __name__ == "__main__":
    main()
