#!/usr/bin/env python3
"""Quick analysis of Histo-Seg - just check mask colors on small samples."""
import os
from pathlib import Path
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None  # Disable bomb check

PROJECT = Path(__file__).parent.parent
BASE = PROJECT / "data/downloads/histo_seg/extracted/Histo-Seg H&E Whole Slide Image Segmentation Datas/Histo-Seg"
IMG_DIR = BASE / "Images"
MASK_DIR = BASE / "Masks"

images = sorted(IMG_DIR.glob("*.jpg"))
masks = sorted(MASK_DIR.glob("*.png"))

print(f"Images: {len(images)}, Masks: {len(masks)}")

# Quick image listing
print("\nAll images:")
for i, p in enumerate(images):
    sz = p.stat().st_size / (1024**2)
    print(f"  {i+1:2d}. {sz:7.1f} MB | {p.name}")

# Analyze just 3 masks quickly by sampling
print("\nMask analysis (sampling center region):")
for mask_path in masks[:5]:
    try:
        im = Image.open(mask_path)
        w, h = im.size
        # Sample a center crop
        cx, cy = w//2, h//2
        crop = im.crop((cx-500, cy-500, cx+500, cy+500))
        arr = np.array(crop)
        
        print(f"\n  {mask_path.name}: {w}x{h}, mode={im.mode}")
        
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[2])
            unique = np.unique(flat, axis=0)
            for c in unique:
                cnt = np.sum(np.all(arr == c, axis=2))
                pct = cnt / (arr.shape[0]*arr.shape[1]) * 100
                if pct > 0.1:
                    print(f"    RGB({c[0]:3d},{c[1]:3d},{c[2]:3d}): {pct:5.1f}%")
        else:
            for v in np.unique(arr):
                cnt = np.sum(arr == v)
                pct = cnt / arr.size * 100
                if pct > 0.1:
                    print(f"    Val {v}: {pct:5.1f}%")
        im.close()
    except Exception as e:
        print(f"  Error: {e}")

# Also sample edges for full unique colors from first mask
print("\nFull unique colors (first mask, downsampled):")
try:
    im = Image.open(masks[0])
    w, h = im.size
    small = im.resize((w//10, h//10), Image.NEAREST)
    arr = np.array(small)
    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[2])
        unique = np.unique(flat, axis=0)
        print(f"  Mask: {masks[0].name} ({w}x{h})")
        print(f"  Unique colors: {len(unique)}")
        for c in unique:
            cnt = np.sum(np.all(arr == c, axis=2))
            pct = cnt / (arr.shape[0]*arr.shape[1]) * 100
            print(f"    RGB({c[0]:3d},{c[1]:3d},{c[2]:3d}): {pct:5.1f}%")
    im.close()
except Exception as e:
    print(f"  Error: {e}")

# Check if mask filenames give class info
print("\n\nFilename patterns:")
img_names = [p.stem for p in images]
mask_names = [p.stem for p in masks]

# Check overlap
common = set(img_names) & set(mask_names)
print(f"  Images with matching masks: {len(common)}/{len(images)}")

# Group by case ID
cases = {}
for name in img_names:
    # Extract case ID (before parenthesis)
    case_id = name.split("(")[0] if "(" in name else name
    if case_id not in cases:
        cases[case_id] = []
    cases[case_id].append(name)

print(f"\n  Unique cases: {len(cases)}")
for cid, files in sorted(cases.items()):
    print(f"    {cid}: {len(files)} slides")
