#!/usr/bin/env python3
"""
Balanced Demo Training for Multi-Class Skin Cancer Classification
=================================================================

Bu script:
1. Her sınıftan dengeli sayıda tile seçer
2. Küçük bir demo model eğitir (hızlı iterasyon için)
3. Class weighting kullanarak dengesiz veriyi kompanse eder

Sınıflar:
    0 = normal (COBRA benign)
    1 = bcc (COBRA malignant)
    2 = melanoma (TCGA-SKCM)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patch_classifier import create_model


# Constants
NUM_CLASSES = 3
CLASS_NAMES = ["normal", "bcc", "melanoma"]
TILES_PER_CLASS = 1000  # For balanced demo training


class BalancedTileDataset(Dataset):
    """Dataset with balanced sampling from each class."""
    
    def __init__(self, manifest_path, split="train", tiles_per_class=1000, transform=None):
        self.transform = transform
        self.tiles_per_class = tiles_per_class
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split]
        
        # Sample balanced tiles from each class
        sampled_dfs = []
        for label in sorted(df["label"].unique()):
            class_df = df[df["label"] == label]
            n_samples = min(len(class_df), tiles_per_class)
            sampled = class_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled)
            print(f"  Class {label}: sampled {n_samples} tiles")
        
        self.df = pd.concat(sampled_dfs, ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=42)  # Shuffle
        
        print(f"  Total: {len(self.df)} tiles")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row["tile_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return random noise if image fails
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        label = int(row["label"])
        return image, label


def get_transforms(split="train"):
    """Get data transforms for training/validation."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def compute_class_weights(df):
    """Compute class weights for imbalanced data."""
    class_counts = df["label"].value_counts().sort_index()
    total = len(df)
    weights = []
    for label in sorted(class_counts.index):
        weight = total / (len(class_counts) * class_counts[label])
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Balanced Demo Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--tiles-per-class", type=int, default=500, help="Tiles per class for balanced training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Setup logging
    log_dir = PROJECT_ROOT / "logs" / "demo_training"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"demo_train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print("="*60)
    print("BALANCED DEMO TRAINING - MULTI-CLASS SKIN CANCER")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tiles per class: {args.tiles_per_class}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Check manifest
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "multiclass_tile_manifest.csv"
    if not manifest_path.exists():
        print("❌ Multiclass tile manifest not found!")
        print("   Run: python scripts/convert_tiles_multiclass.py")
        return
    
    # Check class distribution
    df = pd.read_csv(manifest_path)
    print("Full dataset class distribution:")
    for label in sorted(df["label"].unique()):
        name = df[df["label"] == label]["label_name"].iloc[0]
        count = len(df[df["label"] == label])
        print(f"  {label} = {name}: {count:,} tiles")
    print()
    
    # Check if melanoma tiles exist
    melanoma_count = len(df[df["label"] == 2])
    if melanoma_count == 0:
        print("⚠️  No melanoma tiles found!")
        print("   Run tile extraction after downloading melanoma slides.")
        print("   Proceeding with 2-class demo (normal vs bcc)...")
        # Filter to only normal and bcc
        df = df[df["label"].isin([0, 1])]
        num_classes = 2
        class_names = ["normal", "bcc"]
    else:
        num_classes = NUM_CLASSES
        class_names = CLASS_NAMES
    
    # Create datasets
    print(f"\nCreating balanced datasets ({args.tiles_per_class} tiles/class)...")
    print("Train set:")
    train_dataset = BalancedTileDataset(
        manifest_path, split="train", 
        tiles_per_class=args.tiles_per_class,
        transform=get_transforms("train")
    )
    
    print("\nValidation set:")
    val_dataset = BalancedTileDataset(
        manifest_path, split="val",
        tiles_per_class=args.tiles_per_class // 2,
        transform=get_transforms("val")
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print(f"\nCreating model (ResNet18, {num_classes} classes)...")
    device = torch.device(args.device)
    model = create_model(
        num_classes=num_classes,
        pretrained=True,
        dropout=0.5,
        architecture="resnet18",
        device=device
    )
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset.df).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_acc = 0
    best_model_path = log_dir / f"best_model_{timestamp}.pth"
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'class_names': class_names
            }, best_model_path)
            print(f"  ✓ Saved best model (acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print(f"\nBest Validation Accuracy: {val_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(cm)
    
    print(f"\n✓ Training complete! Best model saved to: {best_model_path}")
    logging.info(f"Training complete. Best val acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
