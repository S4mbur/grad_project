#!/usr/bin/env python3
"""
PyTorch → ONNX Model Export Script
===================================
Exports the ResNet18 encoder and GatedAttentionMIL model to ONNX format
for use with the .NET SkinSight application (ONNX Runtime).

Usage:
    python export_to_onnx.py

Output:
    data/mil_results/encoder.onnx
    data/mil_results/best_model.onnx
"""

import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import torch
import torch.nn as nn
from torchvision import models


def export_encoder():
    """Export ResNet18 feature extractor to ONNX."""
    print("Exporting ResNet18 encoder...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = PROJECT_DIR / "data" / "mil_results" / "encoder.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"  ✓ Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_mil_model():
    """Export GatedAttentionMIL to ONNX."""
    print("Exporting GatedAttentionMIL...")

    class GatedAttentionMIL(nn.Module):
        def __init__(self, input_dim=512, hidden_dim=256, n_classes=3, dropout=0.25):
            super().__init__()
            self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
            self.attention_w = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            v = self.attention_V(x)
            u = self.attention_U(x)
            a = self.attention_w(v * u)
            a = torch.nn.functional.softmax(a, dim=0)
            z = torch.mm(a.T, x)
            logits = self.classifier(z)
            return logits, a

    ckpt_path = PROJECT_DIR / "data" / "mil_results" / "best_model.pt"
    output_path = PROJECT_DIR / "data" / "mil_results" / "best_model.onnx"

    model = GatedAttentionMIL(input_dim=512, hidden_dim=256, n_classes=3, dropout=0.25)

    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded weights from {ckpt_path}")
    else:
        print(f"  ⚠ Checkpoint not found at {ckpt_path}, exporting with random weights")

    model.eval()

    # MIL model takes variable number of tile features
    dummy_input = torch.randn(50, 512)  # 50 tiles, 512-dim features

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["logits", "attention"],
        dynamic_axes={
            "features": {0: "n_tiles"},
            "attention": {0: "n_tiles"},
        },
        opset_version=17,
    )
    print(f"  ✓ Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    export_encoder()
    export_mil_model()
    print("\n✓ All models exported! The .NET app can now use these ONNX files.")
