#!/usr/bin/env python3
"""Quick evaluation of the trained demo model."""

import sys
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_balanced_demo import BalancedTileDataset, get_transforms, validate
from src.models.patch_classifier import create_model

def main():
    print("="*60)
    print("DEMO MODEL EVALUATION")
    print("="*60)
    
    # Find latest model
    log_dir = PROJECT_ROOT / "logs" / "demo_training"
    model_files = list(log_dir.glob("best_model_*.pth"))
    if not model_files:
        print("❌ No model found!")
        return
    
    model_path = sorted(model_files)[-1]
    print(f"Loading model: {model_path.name}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    num_classes = checkpoint.get('num_classes', 3)
    class_names = checkpoint.get('class_names', ['normal', 'bcc', 'melanoma'])
    
    print(f"Classes: {num_classes} - {class_names}")
    print(f"Best val accuracy: {checkpoint.get('val_acc', 0):.2f}%")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        num_classes=num_classes,
        pretrained=False,
        dropout=0.5,
        architecture="resnet18",
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "multiclass_tile_manifest.csv"
    
    print("\nTest set:")
    from torch.utils.data import DataLoader
    test_dataset = BalancedTileDataset(
        manifest_path, split="test",
        tiles_per_class=200,
        transform=get_transforms("val")
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate
    print("\nEvaluating on test set...")
    criterion = torch.nn.CrossEntropyLoss()
    _, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*60}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names[:num_classes]))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print("               Predicted")
    print(f"              {' '.join([f'{n:>8}' for n in class_names[:num_classes]])}")
    print("Actual")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>10}  {' '.join([f'{v:>8}' for v in row])}")
    
    print(f"\n✓ Evaluation complete!")

if __name__ == "__main__":
    main()
