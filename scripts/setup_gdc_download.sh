#!/bin/bash
# Create GDC manifest and download with gdc-client (FAST!)
set -e

GDC_CLIENT="/tmp/gdc-client-bin/gdc-client"
OUT_DIR="/mnt/d/skin_cancer_project/datasets/tcga_skcm"
MANIFEST="/tmp/gdc_download_manifest.txt" 
MAX_GB=151

echo "=============================================="
echo "  TCGA-SKCM Fast Download with gdc-client"
echo "=============================================="

# Step 1: Query GDC API and create manifest
source ~/phase1_env/bin/activate
python3 -u << 'PYEOF'
import json
import requests
from pathlib import Path

OUT_DIR = Path("/mnt/c/Users/byalcn/Desktop/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_GB = 151

# Query GDC
print("[1/2] Querying GDC API...")
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-SKCM"}},
        {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
        {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}}
    ]
}
params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,file_size,md5sum",
    "format": "JSON",
    "size": 1000
}
resp = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=60)
resp.raise_for_status()
hits = resp.json()["data"]["hits"]
svs = [h for h in hits if h.get("file_name", "").endswith(".svs")]
print(f"  Total slides: {len(svs)}")

# Save manifest JSON for future use
with open(OUT_DIR / "download_manifest.json", "w") as f:
    json.dump(svs, f, indent=2)

# Check existing
existing = {}
for fp in OUT_DIR.glob("*.svs"):
    existing[fp.name] = fp.stat().st_size
existing_gb = sum(existing.values()) / (1024**3)
print(f"  Already on disk: {len(existing)} ({existing_gb:.1f} GB)")

# Build GDC manifest (TSV format)
svs.sort(key=lambda x: x.get("file_size", 0))
lines = ["id\tfilename\tmd5\tsize\tstate"]
running_gb = existing_gb
count = 0

for f in svs:
    fname = f["file_name"]
    fsize = f["file_size"]
    fsize_gb = fsize / (1024**3)
    
    if fname in existing and existing[fname] >= fsize * 0.95:
        continue
    if running_gb + fsize_gb > MAX_GB:
        continue
    
    md5 = f.get("md5sum", "")
    lines.append(f'{f["file_id"]}\t{fname}\t{md5}\t{fsize}\tvalidated')
    running_gb += fsize_gb
    count += 1

manifest_path = "/tmp/gdc_download_manifest.txt"
with open(manifest_path, "w") as out:
    out.write("\n".join(lines))

new_gb = running_gb - existing_gb
print(f"\n[2/2] Manifest created:")
print(f"  To download: {count} files ({new_gb:.1f} GB)")
print(f"  Total after: {running_gb:.1f} GB")
print(f"  File: {manifest_path}")
PYEOF

echo ""
echo "Starting gdc-client download (8 processes)..."
echo "This should be MUCH faster than Python requests!"
echo ""

# Step 2: Download with gdc-client
$GDC_CLIENT download \
    -m /tmp/gdc_download_manifest.txt \
    -d "$OUT_DIR" \
    --n-processes 8 \
    --retry-amount 3

echo ""
echo "=============================================="
echo "  Download complete!"
echo "  Total slides: $(ls $OUT_DIR/*.svs 2>/dev/null | wc -l)"
echo "=============================================="
