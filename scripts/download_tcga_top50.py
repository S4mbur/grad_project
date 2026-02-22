#!/usr/bin/env python3
"""Download 33 more TCGA-SKCM primary tumor (01Z) slides to reach 50 total."""
import json
import random
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
GDC_API = "https://api.gdc.cancer.gov"
CHUNK_SIZE = 8192 * 16
TARGET_NEW = 33

def main():
    # Load manifest
    with open(OUT_DIR / "download_manifest.json") as f:
        all_files = json.load(f)
    
    # Filter: only primary tumor (01Z) = diagnostic melanoma slides
    primary = [f for f in all_files if "-01Z-" in f["file_name"]]
    logger.info(f"Total primary (01Z) in manifest: {len(primary)}")
    
    # Already downloaded
    existing = {f.name for f in OUT_DIR.glob("*.svs")}
    existing_01z = [n for n in existing if "-01Z-" in n]
    logger.info(f"Already downloaded (01Z): {len(existing_01z)}")
    
    # Remaining to download
    remaining = [f for f in primary if f["file_name"] not in existing]
    logger.info(f"Remaining available: {len(remaining)}")
    
    # Random select 33
    random.seed(42)
    to_download = random.sample(remaining, min(TARGET_NEW, len(remaining)))
    total_gb = sum(f["file_size"] for f in to_download) / (1024**3)
    logger.info(f"Selected {len(to_download)} random slides ({total_gb:.1f} GB)")
    
    # Download
    downloaded = 0
    errors = 0
    for i, finfo in enumerate(to_download, 1):
        fname = finfo["file_name"]
        fsize = finfo["file_size"]
        out_path = OUT_DIR / fname
        
        if out_path.exists() and out_path.stat().st_size == fsize:
            logger.info(f"  [{i}/{len(to_download)}] Skip (exists): {fname}")
            downloaded += 1
            continue
        
        try:
            url = f"{GDC_API}/data/{finfo['file_id']}"
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()
            
            with open(out_path, "wb") as f:
                dl = 0
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    dl += len(chunk)
            
            downloaded += 1
            size_mb = fsize / (1024**2)
            logger.info(f"  [{i}/{len(to_download)}] OK {fname[:60]}... ({size_mb:.0f} MB)")
        except Exception as e:
            errors += 1
            logger.warning(f"  [{i}/{len(to_download)}] FAIL {fname}: {e}")
            if out_path.exists():
                out_path.unlink()
    
    # Final count
    final_01z = len([f for f in OUT_DIR.glob("*.svs") if "-01Z-" in f.name])
    final_all = len(list(OUT_DIR.glob("*.svs")))
    logger.info(f"\nDone! Downloaded: {downloaded}, Errors: {errors}")
    logger.info(f"Total 01Z slides: {final_01z}")
    logger.info(f"Total all slides: {final_all}")

if __name__ == "__main__":
    main()
