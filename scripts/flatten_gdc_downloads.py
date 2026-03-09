#!/usr/bin/env python3
"""
Flatten gdc-client UUID directories
====================================
gdc-client downloads files into UUID subdirectories. This script 
moves all .svs files up to the parent directory.

Usage: python scripts/flatten_gdc_downloads.py
"""
import os
import shutil
from pathlib import Path

OUT_DIR = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")

def main():
    print("=" * 60)
    print("  Flatten gdc-client UUID directories")
    print(f"  Directory: {OUT_DIR}")
    print("=" * 60)

    # Find all SVS files in subdirectories
    moved = 0
    skipped = 0
    errors = 0

    for uuid_dir in sorted(OUT_DIR.iterdir()):
        if not uuid_dir.is_dir():
            continue
        # Skip non-UUID dirs (they have hyphens and are 36 chars)
        if len(uuid_dir.name) != 36 or uuid_dir.name.count("-") != 4:
            continue

        for svs_file in uuid_dir.glob("*.svs"):
            target = OUT_DIR / svs_file.name

            if target.exists():
                # Already exists at top level
                if target.stat().st_size >= svs_file.stat().st_size * 0.95:
                    skipped += 1
                    continue

            try:
                shutil.move(str(svs_file), str(target))
                moved += 1
                print(f"  Moved: {svs_file.name} ({svs_file.stat().st_size/(1024**2):.0f} MB)")
            except Exception as e:
                errors += 1
                print(f"  Error moving {svs_file.name}: {e}")

        # Remove empty UUID directory
        remaining = list(uuid_dir.iterdir())
        # Only remove if empty or only has logs
        non_svs = [f for f in remaining if f.suffix != ".svs"]
        svs_left = [f for f in remaining if f.suffix == ".svs"]
        if not svs_left:
            try:
                shutil.rmtree(uuid_dir)
            except:
                pass

    # Final count
    total_svs = len(list(OUT_DIR.glob("*.svs")))
    total_gb = sum(f.stat().st_size for f in OUT_DIR.glob("*.svs")) / (1024**3)
    remaining_dirs = len([d for d in OUT_DIR.iterdir() if d.is_dir()])

    print(f"\n  Summary:")
    print(f"    Moved: {moved}")
    print(f"    Skipped (exists): {skipped}")
    print(f"    Errors: {errors}")
    print(f"    Total SVS files: {total_svs} ({total_gb:.1f} GB)")
    print(f"    Remaining subdirs: {remaining_dirs}")


if __name__ == "__main__":
    main()
