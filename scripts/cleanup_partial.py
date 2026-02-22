#!/usr/bin/env python3
"""Clean up partial TCGA-SKCM downloads by checking against manifest sizes."""
import json
from pathlib import Path

dl_dir = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
manifest_path = dl_dir / "download_manifest.json"

with open(manifest_path) as f:
    manifest = json.load(f)

# Build size lookup
expected = {f["file_name"]: f["file_size"] for f in manifest}

svs_files = list(dl_dir.glob("*.svs"))
print(f"Total SVS files: {len(svs_files)}")

partial = []
complete = []
for svs in svs_files:
    actual = svs.stat().st_size
    exp = expected.get(svs.name, None)
    if exp is None:
        print(f"  ? {svs.name} - not in manifest")
        continue
    if actual < exp:
        diff_mb = (exp - actual) / 1e6
        print(f"  PARTIAL: {svs.name} ({actual/1e6:.0f}/{exp/1e6:.0f} MB, missing {diff_mb:.0f} MB)")
        partial.append(svs)
    else:
        complete.append(svs)

print(f"\nComplete: {len(complete)}")
print(f"Partial:  {len(partial)}")

if partial:
    print("\nDeleting partial files...")
    for p in partial:
        p.unlink()
        print(f"  Deleted: {p.name}")
    print("Done!")

# Final count
remaining = list(dl_dir.glob("*.svs"))
primary = [f for f in remaining if "-01Z-" in f.name]
print(f"\nFinal: {len(remaining)} SVS total, {len(primary)} primary (01Z)")
