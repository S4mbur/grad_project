#!/usr/bin/env python3
"""
Convert Existing Tile Manifest to Multi-Class Format
=====================================================

Bu script:
1. Mevcut tile manifest'i multi-class formatına dönüştürür
2. Melanoma slide'ları için tile extraction yapar
3. Birleşik multi-class tile manifest oluşturur

Sınıflar:
    0 = normal (COBRA benign)
    1 = bcc (COBRA malignant)
    2 = melanoma (TCGA-SKCM)
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.tile_extractor import TileExtractor, TileExtractionConfig


# Label mapping
OLD_TO_NEW_LABEL = {
    "benign": "normal",
    "malignant": "bcc",
}

LABEL_TO_INT = {
    "normal": 0,
    "bcc": 1,
    "melanoma": 2,
}


def convert_existing_manifest():
    """Convert existing tile manifest to multi-class format."""
    print("[1/3] Converting existing tile manifest...")
    
    old_manifest = PROJECT_ROOT / "data" / "manifests" / "tile_manifest.csv"
    if not old_manifest.exists():
        print(f"❌ Tile manifest not found: {old_manifest}")
        return None
    
    df = pd.read_csv(old_manifest)
    print(f"  Loaded {len(df)} tiles from existing manifest")
    
    # Convert labels
    df["label_name"] = df["label"].map(OLD_TO_NEW_LABEL)
    df["label"] = df["label_name"].map(LABEL_TO_INT)
    
    # Add source column
    df["source"] = "cobra"
    
    # Reorder columns
    new_columns = ["tile_path", "slide_id", "label", "label_name", "source", "split", 
                   "x", "y", "level", "tissue_fraction", "blur_score", "nuclei_score"]
    df = df[new_columns]
    
    print(f"  ✓ Converted to multi-class format")
    print(f"    - Normal: {len(df[df['label'] == 0])} tiles")
    print(f"    - BCC: {len(df[df['label'] == 1])} tiles")
    
    return df


def extract_melanoma_tiles():
    """Extract tiles from melanoma slides."""
    print("\n[2/3] Extracting tiles from melanoma slides...")
    
    # Get melanoma slides from multiclass manifest
    slide_manifest = PROJECT_ROOT / "data" / "manifests" / "multiclass_slide_manifest.csv"
    if not slide_manifest.exists():
        print(f"❌ Multiclass slide manifest not found: {slide_manifest}")
        return None
    
    slides_df = pd.read_csv(slide_manifest)
    melanoma_slides = slides_df[slides_df["label"] == 2]
    
    if len(melanoma_slides) == 0:
        print("⚠️  No melanoma slides found in manifest")
        return pd.DataFrame()
    
    print(f"  Found {len(melanoma_slides)} melanoma slide(s)")
    
    # Create tile extraction config
    config = TileExtractionConfig(
        tile_size=512,
        target_mpp=0.5,
        max_tiles_per_slide=500,
        min_tissue_fraction=0.3,
        blur_threshold=80.0,
        jpeg_quality=90,
    )
    
    tiles_dir = PROJECT_ROOT / "data" / "tiles_multiclass"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = TileExtractor(config)
    
    all_tiles = []
    for _, row in melanoma_slides.iterrows():
        slide_path = Path(row["local_path"])
        slide_id = row["slide_id"]
        split = row["split"]
        
        if not slide_path.exists():
            print(f"  ⚠️  Slide not found: {slide_path}")
            continue
        
        print(f"  Processing: {slide_id}")
        
        # Create output directory for this slide
        slide_tiles_dir = tiles_dir / slide_id
        slide_tiles_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract tiles with all required arguments
            tiles = extractor.extract_tiles(
                slide_path=str(slide_path),
                output_dir=str(tiles_dir),
                slide_id=slide_id,
                label="melanoma",
                split=split
            )
            
            # Add extra metadata
            for tile in tiles:
                tile["label"] = 2
                tile["label_name"] = "melanoma"
                tile["source"] = "tcga"
            
            all_tiles.extend(tiles)
            print(f"    ✓ Extracted {len(tiles)} tiles")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue
    
    if not all_tiles:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_tiles)
    
    # Rename columns to match
    column_mapping = {
        "path": "tile_path",
        "coords": None,  # Will extract x, y separately
    }
    
    if "path" in df.columns:
        df = df.rename(columns={"path": "tile_path"})
    
    # Ensure required columns exist
    if "x" not in df.columns and "coords" in df.columns:
        df["x"] = df["coords"].apply(lambda c: c[0] if c else 0)
        df["y"] = df["coords"].apply(lambda c: c[1] if c else 0)
    
    # Add default values for missing columns
    for col in ["level", "tissue_fraction", "blur_score", "nuclei_score"]:
        if col not in df.columns:
            df[col] = 0.0
    
    return df


def create_multiclass_tile_manifest(cobra_df, melanoma_df):
    """Create combined multi-class tile manifest."""
    print("\n[3/3] Creating multi-class tile manifest...")
    
    combined = pd.concat([cobra_df, melanoma_df], ignore_index=True)
    
    # Summary
    print(f"\nTotal tiles: {len(combined)}")
    print("\nBy class:")
    for label_name in ["normal", "bcc", "melanoma"]:
        count = len(combined[combined["label_name"] == label_name])
        label = LABEL_TO_INT[label_name]
        print(f"  {label} = {label_name}: {count} tiles")
    
    print("\nBy split:")
    for split in combined["split"].unique():
        count = len(combined[combined["split"] == split])
        print(f"  {split}: {count} tiles")
    
    # Save
    output_path = PROJECT_ROOT / "data" / "manifests" / "multiclass_tile_manifest.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return combined


def main():
    print("="*60)
    print("CONVERTING TILE MANIFEST TO MULTI-CLASS FORMAT")
    print("="*60)
    
    # Step 1: Convert existing COBRA tiles
    cobra_df = convert_existing_manifest()
    if cobra_df is None:
        print("❌ Failed to convert existing manifest")
        return
    
    # Step 2: Extract melanoma tiles (if any)
    melanoma_df = extract_melanoma_tiles()
    
    # Step 3: Combine and save
    create_multiclass_tile_manifest(cobra_df, melanoma_df)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("""
Next steps:
1. Download more TCGA-SKCM melanoma slides (max 5GB)
2. Add them to multiclass_slide_manifest.csv
3. Re-run this script to extract their tiles
4. Train with: python scripts/train_multiclass.py --num-classes 3
""")


if __name__ == "__main__":
    main()
