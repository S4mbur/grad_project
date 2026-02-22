#!/usr/bin/env python3
"""
Transfer all WSL data to D: and prepare unified training-ready structure.
"""
import csv
import json
import shutil
from pathlib import Path
from collections import Counter

def hr(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ============================================================
# STEP 1: Inventory all WSI files in WSL
# ============================================================
hr("STEP 1: WSL Data Inventory")

# COBRA BCC TIFs
cobra_tifs = list(Path("/home/byalc/phase1_project/data/raw_wsi").glob("*.tif"))
print(f"  data/raw_wsi/*.tif: {len(cobra_tifs)}")

# TCGA Melanoma SVS
tcga_svs = list(Path("/home/byalc/phase1_project/data/raw_wsi/melanoma").glob("*.svs"))
print(f"  data/raw_wsi/melanoma/*.svs: {len(tcga_svs)}")

# Any SCC
scc_dir = Path("/home/byalc/phase1_project/data/raw_wsi/scc")
scc_files = list(scc_dir.glob("*")) if scc_dir.exists() else []
print(f"  data/raw_wsi/scc/*: {len(scc_files)}")

# ============================================================
# STEP 2: Analyze labels
# ============================================================
hr("STEP 2: Label Analysis")

# BCC labels - bcc_bcc.csv tells which are BCC
bcc_csv = Path("/home/byalc/phase1_project/data/cobra/bcc_bcc.csv")
bcc_images_csv = Path("/home/byalc/phase1_project/data/cobra/bcc_images.csv")

bcc_slides = set()
if bcc_csv.exists():
    with open(bcc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "")
            if fname:
                bcc_slides.add(fname)
    print(f"  BCC slides (bcc_bcc.csv): {len(bcc_slides)}")

# All BCC group images
all_bcc_group = set()
if bcc_images_csv.exists():
    with open(bcc_images_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "")
            if fname:
                all_bcc_group.add(fname)
    print(f"  All BCC group images (bcc_images.csv): {len(all_bcc_group)}")

# Normal = BCC group - BCC
normal_slides = all_bcc_group - bcc_slides
print(f"  Normal slides: {len(normal_slides)}")
print(f"  BCC slides: {len(bcc_slides)}")

# Match with actual files
cobra_tif_names = {f.name for f in cobra_tifs}
matched_bcc = bcc_slides & cobra_tif_names
matched_normal = normal_slides & cobra_tif_names
unmatched = cobra_tif_names - all_bcc_group
print(f"\n  Matched BCC TIFs: {len(matched_bcc)}")
print(f"  Matched Normal TIFs: {len(matched_normal)}")
print(f"  Unmatched TIFs: {len(unmatched)}")

# OOD labels
ood_csv = Path("/home/byalc/phase1_project/data/cobra/ood_labels/labels/ood_disease_types.csv")
ood_diseases = Counter()
ood_map = {}
if ood_csv.exists():
    with open(ood_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                ood_diseases[row[1]] += 1
                ood_map[row[0]] = row[1]
    print(f"\n  OOD Disease Types ({sum(ood_diseases.values())} total):")
    for d, c in ood_diseases.most_common():
        print(f"    {d:35s}: {c}")

# ============================================================
# STEP 3: D: drive current status
# ============================================================
hr("STEP 3: D: Drive Status")

d_ood = list(Path("/mnt/d/skin_cancer_project/datasets/cobra_ood/images").glob("*.tif"))
d_tcga = list(Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm").glob("*.svs"))
d_tcga_01z = [f for f in d_tcga if "-01Z-" in f.name]

print(f"  COBRA OOD images: {len(d_ood)}")
print(f"  TCGA-SKCM SVS: {len(d_tcga)} ({len(d_tcga_01z)} primary)")

# Check if BCC group already on D:
d_bcc = Path("/mnt/d/skin_cancer_project/datasets/cobra_bcc")
if d_bcc.exists():
    d_bcc_tifs = list(d_bcc.rglob("*.tif"))
    print(f"  COBRA BCC on D:: {len(d_bcc_tifs)}")
else:
    print(f"  COBRA BCC on D:: NOT YET COPIED")

# ============================================================
# STEP 4: What needs to be copied
# ============================================================
hr("STEP 4: Transfer Plan")

# BCC TIFs -> D:/skin_cancer_project/datasets/cobra_bcc/
print(f"  COBRA BCC TIFs to copy: {len(cobra_tifs)} files")
total_bcc_size = sum(f.stat().st_size for f in cobra_tifs) / (1024**3)
print(f"    Size: {total_bcc_size:.1f} GB")

# TCGA melanoma SVS -> already some on D:, check overlap
d_tcga_names = {f.name for f in d_tcga}
wsl_tcga_new = [f for f in tcga_svs if f.name not in d_tcga_names]
print(f"  TCGA melanoma SVS to copy: {len(wsl_tcga_new)} new (of {len(tcga_svs)} total)")
if wsl_tcga_new:
    new_size = sum(f.stat().st_size for f in wsl_tcga_new) / (1024**3)
    print(f"    Size: {new_size:.1f} GB")

# Labels to copy
print(f"  Label files to copy: bcc_bcc.csv, bcc_images.csv, bcc_patient.csv, ood_disease_types.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
hr("FINAL TRAINING-READY SUMMARY (after transfer)")

mel_count = ood_diseases.get("Melanoma", 0) + ood_diseases.get("Melanoma in situ", 0) + len(tcga_svs) + len(d_tcga)
# Remove duplicates
d_tcga_unique = len(d_tcga_names | {f.name for f in tcga_svs})
mel_total = ood_diseases.get("Melanoma", 0) + ood_diseases.get("Melanoma in situ", 0) + d_tcga_unique

bcc_total = len(matched_bcc) + ood_diseases.get("Basal cell carcinoma", 0)
scc_total = ood_diseases.get("Squamous cell carcinoma", 0)
normal_total = len(matched_normal) + ood_diseases.get("Benign", 0) + ood_diseases.get("No abnormalities", 0)

print(f"\n  3-CLASS TOTALS:")
print(f"    Melanoma:          {mel_total}")
print(f"    BCC + SCC:         {bcc_total + scc_total} (BCC:{bcc_total}, SCC:{scc_total})")
print(f"    Normal/Benign:     {normal_total}")
print(f"    Total usable:      {mel_total + bcc_total + scc_total + normal_total}")

if __name__ == "__main__":
    main = None  # just run top-level
