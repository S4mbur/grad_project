#!/usr/bin/env python3
"""
Download MORE TCGA-SKCM Melanoma Slides (up to 20GB budget)
============================================================
Mevcut 15 slide'a ek olarak daha fazla melanoma WSI indirir.
Zaten indirilen dosyaları atlar.
"""

import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MELANOMA_DIR = DATA_DIR / "raw_wsi" / "melanoma"
MANIFEST_DIR = DATA_DIR / "manifests"

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

MAX_TOTAL_GB = 20


def get_all_skcm_slides():
    """Query ALL TCGA-SKCM diagnostic slides."""
    print("[1/4] Querying TCGA-SKCM slides from GDC API...")

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
        "fields": "file_id,file_name,file_size,cases.case_id,cases.submitter_id",
        "format": "JSON",
        "size": 1000
    }

    try:
        response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        hits = data.get("data", {}).get("hits", [])
        svs = [h for h in hits if h.get("file_name", "").endswith(".svs")]
        print(f"  Found {len(svs)} total diagnostic SVS slides")
        return svs
    except requests.RequestException as e:
        print(f"  Error: {e}")
        return []


def select_new_slides(slides, max_total_gb=20):
    """Select slides to download, skipping already downloaded ones."""
    print(f"\n[2/4] Selecting slides (budget: {max_total_gb} GB)...")

    MELANOMA_DIR.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in MELANOMA_DIR.iterdir() if f.suffix == ".svs"}
    existing_size_mb = sum(
        f.stat().st_size / (1024 * 1024)
        for f in MELANOMA_DIR.iterdir() if f.suffix == ".svs"
    )
    print(f"  Already downloaded: {len(existing)} slides ({existing_size_mb:.0f} MB)")

    # Add size info and sort
    for s in slides:
        s["size_mb"] = s.get("file_size", 0) / (1024 * 1024)
    slides.sort(key=lambda x: x["size_mb"])

    # Separate existing vs new
    already = [s for s in slides if s["file_name"] in existing]
    candidates = [s for s in slides if s["file_name"] not in existing]

    budget_mb = max_total_gb * 1024
    running_total_mb = existing_size_mb

    selected = []
    for s in candidates:
        if running_total_mb + s["size_mb"] <= budget_mb:
            selected.append(s)
            running_total_mb += s["size_mb"]

    new_total_mb = sum(s["size_mb"] for s in selected)
    print(f"  New slides to download: {len(selected)} ({new_total_mb:.0f} MB)")
    print(f"  Total after download: {len(existing) + len(selected)} slides ({running_total_mb:.0f} MB / {running_total_mb/1024:.1f} GB)")

    return selected


def download_slide(file_id, file_name, output_dir):
    """Download a single slide from GDC."""
    output_path = output_dir / file_name

    if output_path.exists() and output_path.stat().st_size > 10000:
        return True

    url = f"{GDC_DATA_ENDPOINT}/{file_id}"

    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                     desc=f"    {file_name[:40]}", leave=True) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except requests.RequestException as e:
        print(f"    Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_all(selected):
    """Download all selected slides."""
    print(f"\n[3/4] Downloading {len(selected)} new slides...")

    MELANOMA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = []
    failed = []

    for i, slide in enumerate(selected, 1):
        file_id = slide["file_id"]
        file_name = slide["file_name"]
        size_mb = slide["size_mb"]

        print(f"\n  [{i}/{len(selected)}] {file_name} ({size_mb:.0f} MB)")

        if download_slide(file_id, file_name, MELANOMA_DIR):
            downloaded.append({
                "file_id": file_id,
                "file_name": file_name,
                "size_mb": size_mb,
                "local_path": str(MELANOMA_DIR / file_name)
            })
        else:
            failed.append(file_name)

    print(f"\n  Downloaded: {len(downloaded)}")
    if failed:
        print(f"  Failed: {len(failed)}")

    return downloaded


def update_manifest():
    """Rebuild melanoma entries in manifest from actual files on disk."""
    print("\n[4/4] Updating slide manifest...")

    manifest_path = MANIFEST_DIR / "multiclass_slide_manifest.csv"

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        df = df[df["label"] != 2]  # Remove old melanoma entries
    else:
        df = pd.DataFrame()

    # Scan actual files
    svs_files = sorted(MELANOMA_DIR.glob("*.svs"))
    print(f"  Found {len(svs_files)} melanoma SVS files on disk")

    new_rows = []
    n = len(svs_files)
    for i, svs in enumerate(svs_files):
        # 80/10/10 split
        if i < n * 0.8:
            split = "train"
        elif i < n * 0.9:
            split = "val"
        else:
            split = "test"

        slide_id = svs.stem
        new_rows.append({
            "slide_id": slide_id,
            "patient_id": "-".join(slide_id.split("-")[:3]) if "-" in slide_id else slide_id,
            "local_path": str(svs),
            "label": 2,
            "label_name": "melanoma",
            "source": "tcga",
            "split": split
        })

    melanoma_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, melanoma_df], ignore_index=True)
    combined.to_csv(manifest_path, index=False)

    print(f"  Manifest updated: {len(new_rows)} melanoma slides")
    print(f"  Total slides: {len(combined)}")

    print("\n  Class distribution:")
    for label in sorted(combined["label"].unique()):
        sub = combined[combined["label"] == label]
        name = sub["label_name"].iloc[0]
        print(f"    {label} = {name}: {len(sub)} slides")


def main():
    print("=" * 60)
    print("TCGA-SKCM MELANOMA - BULK DOWNLOAD (20GB budget)")
    print("=" * 60)

    slides = get_all_skcm_slides()
    if not slides:
        print("No slides found.")
        return

    selected = select_new_slides(slides, MAX_TOTAL_GB)
    if not selected:
        print("Nothing new to download (budget full or all downloaded).")
        update_manifest()
        return

    total_gb = sum(s["size_mb"] for s in selected) / 1024
    print(f"\n  About to download {len(selected)} slides ({total_gb:.1f} GB)")

    downloaded = download_all(selected)

    update_manifest()

    # Final stats
    total_size = sum(f.stat().st_size for f in MELANOMA_DIR.glob("*.svs")) / (1024**3)
    total_count = len(list(MELANOMA_DIR.glob("*.svs")))
    print(f"\n{'='*60}")
    print(f"DONE! {total_count} melanoma slides ({total_size:.1f} GB)")
    print(f"{'='*60}")
    print("\nNext: python scripts/convert_tiles_multiclass.py")


if __name__ == "__main__":
    main()
