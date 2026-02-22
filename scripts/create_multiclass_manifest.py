#!/usr/bin/env python3
"""
Create Multi-Class Slide Manifest
==================================

Bu script, mevcut COBRA verilerini ve CMU-1.svs melanoma örneğini
birleştirerek multi-class manifest oluşturur.

Sınıflar:
    0 = normal (COBRA benign)
    1 = bcc (COBRA malignant - BCC)
    2 = melanoma (CMU-1.svs test slide)

Kullanım:
    python scripts/create_multiclass_manifest.py

Not: SCC daha sonra UQ NMSC dataset'i ile eklenecek.
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Multi-class label mapping
LABEL_MAP = {
    "normal": 0,      # COBRA benign (non-BCC)
    "bcc": 1,         # COBRA malignant (BCC)
    "melanoma": 2,    # TCGA-SKCM
    # "scc": 3,       # UQ NMSC - sonra eklenecek
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def load_cobra_manifest():
    """Load existing COBRA slide manifest."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "slide_manifest.csv"
    
    if not manifest_path.exists():
        print(f"❌ COBRA manifest not found: {manifest_path}")
        return None
    
    df = pd.read_csv(manifest_path)
    print(f"✓ Loaded COBRA manifest: {len(df)} slides")
    return df


def convert_cobra_labels(df):
    """Convert COBRA binary labels to multi-class format."""
    # Original: benign->0, malignant->1
    # New: normal->0 (was benign), bcc->1 (was malignant)
    
    df = df.copy()
    
    # Map labels
    label_conversion = {
        "benign": "normal",
        "malignant": "bcc",
    }
    
    df["label_name"] = df["label"].map(label_conversion)
    df["label"] = df["label_name"].map(LABEL_MAP)
    df["source"] = "cobra"
    
    # Ensure paths are correct
    df["local_path"] = df["local_path"].apply(
        lambda p: str(PROJECT_ROOT / p) if not Path(p).is_absolute() else p
    )
    
    print(f"  - Normal (benign): {len(df[df['label'] == 0])} slides")
    print(f"  - BCC (malignant): {len(df[df['label'] == 1])} slides")
    
    return df


def get_melanoma_slides():
    """Get melanoma slides (TCGA or test SVS)."""
    raw_wsi_dir = PROJECT_ROOT / "data" / "raw_wsi"
    
    # Look for SVS files (TCGA format)
    svs_files = list(raw_wsi_dir.glob("*.svs"))
    
    if not svs_files:
        print("⚠️  No melanoma SVS files found")
        return pd.DataFrame()
    
    # Filter to meaningful files (>1MB)
    valid_svs = [f for f in svs_files if f.stat().st_size > 1_000_000]
    
    if not valid_svs:
        print("⚠️  No valid melanoma slides found")
        return pd.DataFrame()
    
    # Create melanoma entries
    melanoma_data = []
    for svs_path in valid_svs:
        slide_id = svs_path.stem
        melanoma_data.append({
            "slide_id": slide_id,
            "patient_id": slide_id,
            "local_path": str(svs_path),
            "label": LABEL_MAP["melanoma"],
            "label_name": "melanoma",
            "source": "tcga",
            "split": "train",  # Sadece 1 slide olduğu için train'e koyuyoruz
        })
    
    df = pd.DataFrame(melanoma_data)
    print(f"✓ Found {len(df)} melanoma slide(s)")
    
    return df


def balance_splits(df, min_per_split=1):
    """Ensure all classes have representation in all splits."""
    # Bu fonksiyon daha sonra SCC eklendiğinde kullanılacak
    # Şimdilik melanoma için sadece train kullanıyoruz
    return df


def create_multiclass_manifest():
    """Create the unified multi-class manifest."""
    print("\n" + "="*60)
    print("CREATING MULTI-CLASS SLIDE MANIFEST")
    print("="*60 + "\n")
    
    # Step 1: Load COBRA data
    print("[1/3] Loading COBRA slides...")
    cobra_df = load_cobra_manifest()
    if cobra_df is None:
        return None
    
    # Step 2: Convert COBRA labels
    print("\n[2/3] Converting COBRA labels to multi-class...")
    cobra_multi = convert_cobra_labels(cobra_df)
    
    # Step 3: Add melanoma slides
    print("\n[3/3] Adding melanoma slides...")
    melanoma_df = get_melanoma_slides()
    
    # Combine all sources
    all_dfs = [cobra_multi]
    if len(melanoma_df) > 0:
        all_dfs.append(melanoma_df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Standardize columns
    final_columns = ["slide_id", "patient_id", "local_path", "label", "label_name", "source", "split"]
    combined_df = combined_df[final_columns]
    
    # Summary
    print("\n" + "="*60)
    print("MANIFEST SUMMARY")
    print("="*60)
    
    print(f"\nTotal slides: {len(combined_df)}")
    print("\nBy class:")
    for label_idx in sorted(combined_df["label"].unique()):
        label_name = LABEL_NAMES[label_idx]
        count = len(combined_df[combined_df["label"] == label_idx])
        print(f"  {label_idx} = {label_name}: {count} slides")
    
    print("\nBy split:")
    for split in ["train", "val", "test"]:
        count = len(combined_df[combined_df["split"] == split])
        print(f"  {split}: {count} slides")
    
    print("\nBy source:")
    for source in combined_df["source"].unique():
        count = len(combined_df[combined_df["source"] == source])
        print(f"  {source}: {count} slides")
    
    # Save manifest
    output_path = PROJECT_ROOT / "data" / "manifests" / "multiclass_slide_manifest.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved manifest to: {output_path}")
    
    return combined_df


def main():
    manifest = create_multiclass_manifest()
    
    if manifest is not None:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("""
1. Run tile extraction for multi-class:
   python scripts/02_extract_tiles.py --manifest data/manifests/multiclass_slide_manifest.csv

2. Train multi-class model:
   python scripts/train_multiclass.py --num-classes 3

3. (Later) Add SCC slides from UQ NMSC dataset
""")


if __name__ == "__main__":
    main()
