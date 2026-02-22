#!/usr/bin/env python3
"""Quick dependency check."""
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import openslide
    print(f"OpenSlide: OK ({openslide.__version__})")
except ImportError:
    print("OpenSlide: NOT INSTALLED - run: pip install openslide-python")

try:
    import sklearn
    print(f"sklearn: OK ({sklearn.__version__})")
except ImportError:
    print("sklearn: NOT INSTALLED - run: pip install scikit-learn")

try:
    from torchvision import models
    print("torchvision: OK")
except ImportError:
    print("torchvision: NOT INSTALLED")

try:
    from PIL import Image
    print("Pillow: OK")
except ImportError:
    print("Pillow: NOT INSTALLED")

# Check ResNet18 model file
from pathlib import Path
model_path = Path("/mnt/d/skin_cancer_project/models/torchvision/resnet18.pth")
print(f"\nResNet18 model: {'OK' if model_path.exists() else 'MISSING'} ({model_path})")

# Check D: mount
data_root = Path("/mnt/d/skin_cancer_project/datasets")
print(f"D: data root: {'OK' if data_root.exists() else 'NOT MOUNTED'}")
