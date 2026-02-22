#!/usr/bin/env python3
"""Analyze BCC labels and prepare D: transfer."""
import csv
import os
import shutil
from pathlib import Path
from collections import Counter

# 1. Read BCC labels
bcc_csv = "/home/byalc/phase1_project/data/cobra/bcc_bcc.csv"
with open(bcc_csv) as f:
    rows = list(csv.DictReader(f))

labels = Counter(r["label"] for r in rows)
print(f"BCC label file: {len(rows)} entries")
print(f"  Label 0 (Normal): {labels['0']}")
print(f"  Label 1 (BCC):    {labels['1']}")

# 2. Match with actual TIF files
raw_wsi = Path("/home/byalc/phase1_project/data/raw_wsi")
actual_tifs = {f.name for f in raw_wsi.glob("*.tif")}
print(f"\nActual TIF files in raw_wsi: {len(actual_tifs)}")

bcc_names = {r["filename"] + ".tif" for r in rows if r["label"] == "1"}
normal_names = {r["filename"] + ".tif" for r in rows if r["label"] == "0"}

bcc_present = bcc_names & actual_tifs
normal_present = normal_names & actual_tifs

print(f"  BCC TIFs present:    {len(bcc_present)}")
print(f"  Normal TIFs present: {len(normal_present)}")
print(f"  Total matched:       {len(bcc_present) + len(normal_present)}")

# 3. D: drive targets
d_bcc_dir = Path("/mnt/d/skin_cancer_project/datasets/cobra_bcc")
d_labels_dir = Path("/mnt/d/skin_cancer_project/datasets/labels")

print(f"\n{'='*50}")
print("TRANSFER PLAN")
print(f"{'='*50}")
print(f"  Source: {raw_wsi}")
print(f"  Target: {d_bcc_dir}")
print(f"  Files: {len(actual_tifs)} TIFs")

total_size = sum((raw_wsi / f).stat().st_size for f in actual_tifs) / (1024**3)
print(f"  Size: {total_size:.1f} GB")

# 4. TCGA melanoma
tcga_src = raw_wsi / "melanoma"
tcga_svs = list(tcga_src.glob("*.svs"))
d_tcga = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
d_tcga_existing = {f.name for f in d_tcga.glob("*.svs")}
tcga_new = [f for f in tcga_svs if f.name not in d_tcga_existing]

print(f"\n  TCGA melanoma (WSL): {len(tcga_svs)} SVS")
print(f"  Already on D:: {len(d_tcga_existing)}")
print(f"  New to copy: {len(tcga_new)}")
if tcga_new:
    new_sz = sum(f.stat().st_size for f in tcga_new) / (1024**3)
    print(f"  New size: {new_sz:.1f} GB")

# Summary
print(f"\n{'='*50}")
print("READY TO TRANSFER:")
print(f"  1. {len(actual_tifs)} COBRA BCC TIFs -> D:/datasets/cobra_bcc/")
print(f"  2. {len(tcga_new)} TCGA melanoma SVS -> D:/datasets/tcga_skcm/")
print(f"  3. Label CSVs -> D:/datasets/labels/")
