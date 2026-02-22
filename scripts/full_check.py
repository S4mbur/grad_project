#!/usr/bin/env python3
"""Comprehensive check of all downloaded datasets and models."""
import os
import json
from pathlib import Path

def hr(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_dir(path, ext=None):
    p = Path(path)
    if not p.exists():
        return 0, 0
    if ext:
        files = list(p.glob(f"*.{ext}"))
    else:
        files = [f for f in p.iterdir() if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    return len(files), total

hr("FULL SYSTEM CHECK")
print(f"  Checking all datasets and models...")

# ============================================================
# 1. COBRA BCC Group (WSL local)
# ============================================================
hr("1. COBRA BCC Group (WSL)")
cobra_bcc_path = Path("/home/byalc/phase1_project/data/cobra")
if cobra_bcc_path.exists():
    tifs = list(cobra_bcc_path.rglob("*.tif"))
    print(f"  Path: {cobra_bcc_path}")
    print(f"  TIF files: {len(tifs)}")
    # Check for labels
    labels_path = cobra_bcc_path / "labels"
    if labels_path.exists():
        csvs = list(labels_path.glob("*.csv"))
        print(f"  Label CSVs: {[c.name for c in csvs]}")
    else:
        # Check other possible label locations
        csvs = list(cobra_bcc_path.rglob("*.csv"))
        print(f"  CSVs found: {[c.name for c in csvs]}")
    print(f"  Status: {'OK' if len(tifs) > 0 else 'MISSING'}")
else:
    print(f"  Path: {cobra_bcc_path} NOT FOUND")
    # Try alternate paths
    alt = Path("/home/byalc/phase1_project/data")
    if alt.exists():
        print(f"  Available in /data: {[d.name for d in alt.iterdir()]}")

# ============================================================
# 2. COBRA OOD (D: drive)
# ============================================================
hr("2. COBRA OOD (D: drive)")
ood_path = Path("/mnt/d/skin_cancer_project/datasets/cobra_ood")
n_img, sz_img = check_dir(ood_path / "images", "tif")
print(f"  Images: {n_img} TIF files ({sz_img/1e9:.1f} GB)")

meta_path = ood_path / "metadata"
if meta_path.exists():
    mfiles = list(meta_path.iterdir())
    print(f"  Metadata: {[f.name for f in mfiles]}")
else:
    print(f"  Metadata: MISSING")

ann_path = ood_path / "annotations"
if ann_path.exists():
    afiles = list(ann_path.iterdir())
    print(f"  Annotations: {[f.name for f in afiles]}")
else:
    print(f"  Annotations: MISSING")

# Verify label file has data
label_csv = ann_path / "ood_disease_types.csv" if ann_path.exists() else None
if label_csv and label_csv.exists():
    with open(label_csv) as f:
        lines = f.readlines()
    print(f"  Label entries: {len(lines)-1}")
else:
    # Check metadata for labels
    img_csv = meta_path / "ood_images.csv" if meta_path.exists() else None
    if img_csv and img_csv.exists():
        with open(img_csv) as f:
            lines = f.readlines()
        print(f"  ood_images.csv entries: {len(lines)-1}")

print(f"  Status: {'OK' if n_img >= 1200 else 'INCOMPLETE'} ({n_img}/1248)")

# ============================================================
# 3. TCGA-SKCM (D: drive)
# ============================================================
hr("3. TCGA-SKCM (D: drive)")
tcga_path = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
n_svs, sz_svs = check_dir(tcga_path, "svs")
svs_files = list(tcga_path.glob("*.svs"))
n_01z = len([f for f in svs_files if "-01Z-" in f.name])
n_06z = len([f for f in svs_files if "-06Z-" in f.name])

print(f"  SVS files: {n_svs} ({sz_svs/1e9:.1f} GB)")
print(f"  Primary (01Z): {n_01z}")
print(f"  Metastatic (06Z): {n_06z}")

# Check for partial downloads
manifest = tcga_path / "download_manifest.json"
if manifest.exists():
    with open(manifest) as f:
        mdata = json.load(f)
    expected = {f["file_name"]: f["file_size"] for f in mdata}
    partial = 0
    for svs in svs_files:
        exp = expected.get(svs.name)
        if exp and svs.stat().st_size < exp:
            partial += 1
            print(f"  WARNING: Partial file: {svs.name}")
    if partial == 0:
        print(f"  Integrity: All files complete (no partials)")
    print(f"  Manifest: {len(mdata)} total slides listed")
print(f"  Status: {n_svs} slides ready")

# ============================================================
# 4. TCGA-SKCM (WSL local - old)
# ============================================================
hr("4. TCGA-SKCM (WSL local)")
tcga_wsl = Path("/home/byalc/phase1_project/data/tcga_skcm")
if tcga_wsl.exists():
    n_wsl, sz_wsl = check_dir(tcga_wsl, "svs")
    print(f"  SVS files: {n_wsl} ({sz_wsl/1e9:.1f} GB)")
else:
    # Search for TCGA data
    for p in ["/home/byalc/phase1_project/data"]:
        pp = Path(p)
        if pp.exists():
            tcga_dirs = [d for d in pp.rglob("*tcga*")]
            if tcga_dirs:
                print(f"  Found: {[str(d) for d in tcga_dirs[:5]]}")
    print(f"  Status: Not found at expected path")

# ============================================================
# 5. Models (D: drive)
# ============================================================
hr("5. PRETRAINED MODELS (D: drive)")
models_base = Path("/mnt/d/skin_cancer_project/models")

model_checks = [
    ("ResNet18", "torchvision/resnet18.pth"),
    ("ResNet50", "torchvision/resnet50.pth"),
    ("ConvNeXt-Small", "torchvision/convnext_small.pth"),
    ("ConvNeXt-Base", "torchvision/convnext_base.pth"),
    ("DINOv2-base", "vision/dinov2-base/model.safetensors"),
    ("Phikon", "pathology/phikon/model.safetensors"),
    ("UNI", "pathology/uni/pytorch_model.bin"),
    ("UNI (safetensors)", "pathology/uni/model.safetensors"),
    ("CONCH", "pathology/conch/pytorch_model.bin"),
    ("CTransPath", "pathology/ctranspath.pth"),
]

ok_count = 0
for name, rel_path in model_checks:
    full = models_base / rel_path
    if full.exists():
        sz = full.stat().st_size / 1e6
        print(f"  ✅ {name:20s}: {sz:.1f} MB")
        ok_count += 1
    else:
        print(f"  ❌ {name:20s}: NOT FOUND")

# ============================================================
# SUMMARY
# ============================================================
hr("SUMMARY")
print(f"  COBRA BCC (WSL):     {'✅' if len(list(Path('/home/byalc/phase1_project/data/cobra').rglob('*.tif'))) > 0 else '❌'}")
print(f"  COBRA OOD (D:):      {'✅' if n_img >= 1200 else '❌'} {n_img}/1248 images")
print(f"  TCGA-SKCM (D:):      {'✅' if n_svs > 0 else '❌'} {n_svs} slides ({n_01z} primary)")
print(f"  Models:              {'✅' if ok_count >= 6 else '⚠️'} {ok_count}/10 downloaded")
print()
print(f"  3-CLASS SLIDE COUNTS:")
# COBRA OOD labels
ood_label_file = ann_path / "ood_disease_types.csv" if ann_path.exists() else None
mel_total = n_01z + n_06z  # TCGA D:
if ood_label_file and ood_label_file.exists():
    import csv
    with open(ood_label_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # Count by disease type
    from collections import Counter
    diseases = Counter()
    for row in rows:
        if len(row) >= 2:
            diseases[row[1]] += 1
    mel_ood = diseases.get("Melanoma", 0) + diseases.get("Melanoma in situ", 0)
    mel_total += mel_ood
    print(f"    Melanoma:    {mel_total} (OOD:{mel_ood} + TCGA-D:{n_01z+n_06z})")
    
    bcc_scc = diseases.get("SCC", 0) + diseases.get("BCC", 0)
    print(f"    BCC/SCC:     {bcc_scc + 201} (OOD:{bcc_scc} + COBRA-BCC:201)")
    print(f"    Normal:      374+ (COBRA-BCC normals + OOD benign)")
    print(f"    OOD disease breakdown:")
    for d, c in diseases.most_common():
        print(f"      {d:30s}: {c}")
