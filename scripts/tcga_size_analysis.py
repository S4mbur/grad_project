#!/usr/bin/env python3
"""Analyze TCGA slide sizes and estimate download times."""
import json, random
from pathlib import Path

with open("/mnt/d/skin_cancer_project/datasets/tcga_skcm/download_manifest.json") as f:
    files = json.load(f)

primary = [f for f in files if "-01Z-" in f["file_name"]]
existing = {f.name for f in Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm").glob("*.svs")}
existing_01z = [n for n in existing if "-01Z-" in n]
remaining = [f for f in primary if f["file_name"] not in existing]
remaining.sort(key=lambda x: x["file_size"])

print(f"Total 01Z in manifest: {len(primary)}")
print(f"Already downloaded 01Z: {len(existing_01z)}")
print(f"Remaining 01Z: {len(remaining)}")
med = remaining[len(remaining)//2]
print(f"Size range: {remaining[0]['file_size']/1e6:.0f} MB - {remaining[-1]['file_size']/1e6:.0f} MB")
print(f"Median: {med['file_size']/1e6:.0f} MB")
print()

# Options: pick N smallest
print("=" * 55)
print("OPTIONS (by picking SMALLEST slides, @ 3 MB/s):")
print("=" * 55)
need = 50 - len(existing_01z) 
print(f"Need {need} more to reach 50 total 01Z slides")
print()

for n in [need, 20, 15, 10]:
    if n > len(remaining):
        n = len(remaining)
    batch = remaining[:n]
    gb = sum(f["file_size"] for f in batch) / (1024**3)
    mins = gb * 1024 / 3 / 60
    avg_mb = gb * 1024 / n
    print(f"  Smallest {n:2d}: {gb:5.1f} GB (avg {avg_mb:.0f} MB) -> ~{mins:.0f} min ({mins/60:.1f} hr)")

print()
print("=" * 55)
print("COMPARISON: smallest 32 vs random 32:")
print("=" * 55)
smallest_32 = remaining[:32]
random.seed(42)
rand_32 = random.sample(remaining, 32)
gs = sum(f["file_size"] for f in smallest_32) / (1024**3)
gr = sum(f["file_size"] for f in rand_32) / (1024**3)
print(f"  Smallest 32: {gs:.1f} GB -> ~{gs*1024/3/60:.0f} min")
print(f"  Random 32:   {gr:.1f} GB -> ~{gr*1024/3/60:.0f} min")
print(f"  Savings:     {gr-gs:.1f} GB  ({(1-gs/gr)*100:.0f}% less)")
