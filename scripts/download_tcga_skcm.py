#!/usr/bin/env python3
"""
Download TCGA-SKCM diagnostic slides via GDC API.
Downloads all diagnostic SVS slides for Skin Cutaneous Melanoma.
"""
import os
import sys
import json
import hashlib
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
GDC_API = "https://api.gdc.cancer.gov"
CHUNK_SIZE = 8192 * 16  # 128KB chunks


def get_diagnostic_slide_files():
    """Query GDC API for TCGA-SKCM diagnostic slide files."""
    logger.info("Querying GDC API for TCGA-SKCM diagnostic slides...")
    
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-SKCM"]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "files.experimental_strategy", "value": ["Diagnostic Slide"]}},
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,md5sum,cases.case_id,cases.submitter_id",
        "format": "JSON",
        "size": 1000,
    }
    
    resp = requests.get(f"{GDC_API}/files", params=params)
    resp.raise_for_status()
    data = resp.json()
    
    files = []
    for hit in data["data"]["hits"]:
        files.append({
            "file_id": hit["file_id"],
            "file_name": hit["file_name"],
            "file_size": hit["file_size"],
            "md5sum": hit.get("md5sum", ""),
            "case_id": hit["cases"][0]["case_id"] if hit.get("cases") else "",
        })
    
    logger.info(f"  Found {len(files)} diagnostic slides")
    total_gb = sum(f["file_size"] for f in files) / (1024**3)
    logger.info(f"  Total size: {total_gb:.1f} GB")
    
    return files


def download_file(file_info, out_dir):
    """Download a single file from GDC."""
    file_id = file_info["file_id"]
    file_name = file_info["file_name"]
    file_size = file_info["file_size"]
    out_path = out_dir / file_name
    
    # Skip if already downloaded and correct size
    if out_path.exists() and out_path.stat().st_size == file_size:
        return file_name, "skipped"
    
    try:
        url = f"{GDC_API}/data/{file_id}"
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        
        with open(out_path, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
        
        return file_name, "downloaded"
    except Exception as e:
        # Clean up partial download
        if out_path.exists():
            out_path.unlink()
        return file_name, f"error: {e}"


def main():
    logger.info("=" * 60)
    logger.info("TCGA-SKCM Diagnostic Slide Download")
    logger.info(f"Output: {OUT_DIR}")
    logger.info("=" * 60)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get file list
    files = get_diagnostic_slide_files()
    
    if not files:
        logger.error("No files found!")
        return
    
    # Save manifest
    manifest_path = OUT_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(files, f, indent=2)
    logger.info(f"  Manifest saved: {manifest_path}")
    
    # Check existing
    existing = {f.name for f in OUT_DIR.glob("*.svs")}
    to_download = [f for f in files if f["file_name"] not in existing]
    logger.info(f"  Already downloaded: {len(existing)}")
    logger.info(f"  To download: {len(to_download)}")
    
    if not to_download:
        logger.info("  All files already downloaded!")
        return
    
    # Download with thread pool (2 parallel downloads)
    downloaded = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(download_file, f, OUT_DIR): f 
            for f in to_download
        }
        
        for future in as_completed(futures):
            file_info = futures[future]
            name, status = future.result()
            
            if status == "downloaded":
                downloaded += 1
                size_mb = file_info["file_size"] / (1024**2)
                if downloaded % 10 == 0 or downloaded <= 3:
                    logger.info(f"  [{downloaded}/{len(to_download)}] ✓ {name} ({size_mb:.0f} MB)")
            elif status == "skipped":
                pass
            else:
                errors += 1
                logger.warning(f"  ✗ {name}: {status}")
    
    logger.info(f"\n  Done! Downloaded: {downloaded}, Errors: {errors}")
    logger.info(f"  Total slides in {OUT_DIR}: {len(list(OUT_DIR.glob('*.svs')))}")


if __name__ == "__main__":
    main()
