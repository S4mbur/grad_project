#!/usr/bin/env python3
"""Check current data inventory for MIL training."""
import os
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/home/byalc/phase1_project")
TILES = PROJECT / "data" / "tiles"
RAW_WSI = PROJECT / "data" / "raw_wsi"

print("=" * 60)
print("DATA INVENTORY CHECK")
print("=" * 60)

# Check BCC raw WSIs
bcc_wsis = list(RAW_WSI.glob("*.tif"))
print(f"\nBCC raw WSIs: {len(bcc_wsis)}")

# Check melanoma raw WSIs
mel_wsis = list((RAW_WSI / "melanoma").glob("*.svs")) if (RAW_WSI / "melanoma").exists() else []
print(f"Melanoma raw WSIs: {len(mel_wsis)}")

# Check tile directories
print(f"\nTile directory structure:")
special_dirs = {"train", "val", "test", "demo", "melanoma"}
slide_tile_dirs = []
for d in TILES.iterdir():
    if d.is_dir() and d.name not in special_dirs:
        slide_tile_dirs.append(d)

print(f"  Slide-based tile dirs (BCC): {len(slide_tile_dirs)}")

# Count tiles per slide dir
bcc_slide_tiles = {}
for d in slide_tile_dirs:
    tiles = list(d.glob("*.jpg")) + list(d.glob("*.png"))
    if tiles:
        bcc_slide_tiles[d.name] = len(tiles)

print(f"  BCC slides with tiles: {len(bcc_slide_tiles)}")
print(f"  Total BCC tiles: {sum(bcc_slide_tiles.values())}")
if bcc_slide_tiles:
    vals = list(bcc_slide_tiles.values())
    print(f"  Tiles per slide: min={min(vals)}, max={max(vals)}, avg={sum(vals)/len(vals):.0f}")

# Check train dir (old-style BCC tiles)
train_dir = TILES / "train"
if train_dir.exists():
    train_tiles = list(train_dir.glob("*.jpg"))
    train_slide_ids = set()
    for t in train_tiles:
        parts = t.stem.split("_tile_")
        if len(parts) >= 2:
            train_slide_ids.add(parts[0])
    print(f"\n  Old train/ dir: {len(train_tiles)} tiles from {len(train_slide_ids)} slides")

# Melanoma tiles
mel_tile_dir = TILES / "melanoma"
if mel_tile_dir.exists():
    mel_tiles_list = list(mel_tile_dir.glob("*.jpg"))
    mel_slide_ids = set()
    for t in mel_tiles_list:
        parts = t.stem.split("_tile_")
        if len(parts) >= 2:
            mel_slide_ids.add(parts[0])
    print(f"\n  Melanoma tiles dir: {len(mel_tiles_list)} tiles from {len(mel_slide_ids)} slides")

# Check COBRA original labels
# Look for any label/manifest files
print(f"\nLooking for label information...")
for csv_file in PROJECT.rglob("*.csv"):
    rel = csv_file.relative_to(PROJECT)
    try:
        with open(csv_file) as f:
            header = f.readline().strip()
            n_lines = sum(1 for _ in f)
        print(f"  {rel}: {n_lines} rows, header: {header[:100]}")
    except:
        pass

# Check if COBRA has normal slides
# The COBRA dataset is originally BCC binary: tumor vs normal
# So the 575 .tif files should include both BCC tumor and normal tissue
print(f"\n{'='*60}")
print("NOTE: COBRA dataset is binary classification:")
print("  - BCC (tumor) slides")
print("  - Normal (non-tumor) slides")
print("  Need to check which is which from the original labels")
print("=" * 60)

# Check for label files in data dir
for name in ["labels.csv", "manifest.csv", "metadata.csv", "train.csv", "test.csv"]:
    for root in [PROJECT / "data", PROJECT / "data" / "raw_wsi", PROJECT]:
        p = root / name
        if p.exists():
            print(f"\nFound: {p}")
            with open(p) as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"  {line.strip()}")
