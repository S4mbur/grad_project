#!/usr/bin/env python3
"""Analyze TCGA-SKCM download status and slide types."""
import json
from pathlib import Path

manifest_path = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm/download_manifest.json")
dl_dir = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")

# Load manifest
with open(manifest_path) as f:
    files = json.load(f)

print(f"Total files in manifest: {len(files)}")
total_gb = sum(f["file_size"] for f in files) / (1024**3)
print(f"Total size: {total_gb:.1f} GB")

# Analyze slide types
# TCGA: -01Z- = primary tumor diagnostic, -06Z- = metastatic
diag_01 = [f for f in files if "-01Z-" in f["file_name"]]
meta_06 = [f for f in files if "-06Z-" in f["file_name"]]
other = [f for f in files if "-01Z-" not in f["file_name"] and "-06Z-" not in f["file_name"]]

print()
print(f"Primary tumor (01Z): {len(diag_01)} slides ({sum(f['file_size'] for f in diag_01)/(1024**3):.1f} GB)")
print(f"Metastatic (06Z):    {len(meta_06)} slides ({sum(f['file_size'] for f in meta_06)/(1024**3):.1f} GB)")
if other:
    print(f"Other:               {len(other)} slides")
    for o in other[:5]:
        print(f"  {o['file_name']}")

# Check downloaded
downloaded = list(dl_dir.glob("*.svs"))
print()
print(f"Downloaded so far: {len(downloaded)}")
dl_01 = [f for f in downloaded if "-01Z-" in f.name]
dl_06 = [f for f in downloaded if "-06Z-" in f.name]
print(f"  Primary (01Z): {len(dl_01)}")
print(f"  Metastatic (06Z): {len(dl_06)}")
dl_size = sum(f.stat().st_size for f in downloaded) / (1024**3)
print(f"  Total downloaded: {dl_size:.1f} GB")

# For melanoma MIL, we want primary tumor slides (01Z)
print()
print("=" * 50)
print("RECOMMENDATION:")
print(f"  Primary tumor (01Z) slides are the ones we need for melanoma classification.")
print(f"  We need: {len(diag_01)} slides ({sum(f['file_size'] for f in diag_01)/(1024**3):.1f} GB)")
print(f"  Metastatic (06Z) could also be useful but are from lymph node metastases.")
print(f"  Already downloaded 01Z: {len(dl_01)} / {len(diag_01)}")
print(f"  Already downloaded 06Z: {len(dl_06)} / {len(meta_06)}")
