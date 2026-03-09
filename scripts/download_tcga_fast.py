#!/usr/bin/env python3
"""
Fast Parallel TCGA-SKCM Melanoma Downloader
=============================================
Uses existing download_manifest.json (same as working download_tcga_top50.py),
4 parallel workers, and 128KB chunks — proven to work.

Budget: ~151 GB total (existing + new)
"""

import json
import time
import logging
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
GDC_API = "https://api.gdc.cancer.gov"
CHUNK_SIZE = 8192 * 16   # 128KB — same as working script
MAX_TOTAL_GB = 151
PARALLEL_WORKERS = 4     # moderate parallelism, avoid GDC throttle

# Thread-safe counters
lock = threading.Lock()
stats = {"done": 0, "fail": 0, "bytes": 0, "t0": 0}


def download_one(finfo, idx, total):
    """Download a single file — same logic as working download_tcga_top50.py."""
    fname = finfo["file_name"]
    fsize = finfo["file_size"]
    out_path = OUT_DIR / fname

    # Skip if already complete
    if out_path.exists() and out_path.stat().st_size >= fsize * 0.95:
        with lock:
            stats["done"] += 1
            stats["bytes"] += fsize
        return fname, "skip"

    try:
        url = f"{GDC_API}/data/{finfo['file_id']}"
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()

        dl = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                dl += len(chunk)

        with lock:
            stats["done"] += 1
            stats["bytes"] += dl
            elapsed = time.time() - stats["t0"]
            speed = stats["bytes"] / (1024**2) / max(elapsed, 1)
            gb = stats["bytes"] / (1024**3)
            d = stats["done"]
            logger.info(
                f"  [{d}/{total}] OK {fname[:55]} "
                f"({fsize/(1024**2):.0f}MB) | "
                f"{speed:.1f} MB/s | {gb:.1f}/{MAX_TOTAL_GB} GB | "
                f"{elapsed/60:.0f}min"
            )
        return fname, "ok"

    except Exception as e:
        with lock:
            stats["fail"] += 1
        logger.warning(f"  FAIL {fname[:55]}: {e}")
        if out_path.exists():
            try:
                out_path.unlink()
            except:
                pass
        return fname, f"err: {e}"


def main():
    logger.info("=" * 60)
    logger.info(f"TCGA-SKCM MELANOMA — PARALLEL DOWNLOAD")
    logger.info(f"Budget: {MAX_TOTAL_GB} GB | Workers: {PARALLEL_WORKERS}")
    logger.info(f"Chunk: {CHUNK_SIZE//1024} KB (same as working script)")
    logger.info("=" * 60)

    # Use existing manifest — same as working script
    manifest_path = OUT_DIR / "download_manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.info("Run download_tcga_skcm.py first to create it.")
        return

    with open(manifest_path) as f:
        all_files = json.load(f)
    logger.info(f"Manifest: {len(all_files)} slides")

    # Already downloaded
    existing = {}
    for fp in OUT_DIR.glob("*.svs"):
        existing[fp.name] = fp.stat().st_size
    existing_gb = sum(existing.values()) / (1024**3)
    logger.info(f"Already downloaded: {len(existing)} slides ({existing_gb:.1f} GB)")

    # Select what to download (within budget)
    all_files.sort(key=lambda x: x.get("file_size", 0))  # smallest first
    
    to_download = []
    running_gb = existing_gb
    skipped_exists = 0
    
    for f in all_files:
        fname = f["file_name"]
        fsize = f["file_size"]
        fsize_gb = fsize / (1024**3)

        # Already fully downloaded?
        if fname in existing and existing[fname] >= fsize * 0.95:
            skipped_exists += 1
            continue

        # Budget check
        if running_gb + fsize_gb > MAX_TOTAL_GB:
            continue

        to_download.append(f)
        running_gb += fsize_gb

    new_gb = sum(f["file_size"] for f in to_download) / (1024**3)
    logger.info(f"To download: {len(to_download)} slides ({new_gb:.1f} GB)")
    logger.info(f"Skipped (exists): {skipped_exists}")
    logger.info(f"Projected total: {running_gb:.1f} GB")

    if not to_download:
        logger.info("Nothing to download!")
        return

    # Download with parallel workers
    stats["t0"] = time.time()
    stats["done"] = 0
    stats["fail"] = 0
    stats["bytes"] = 0
    total = len(to_download)

    logger.info(f"\nStarting {PARALLEL_WORKERS} parallel downloads...")

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {
            pool.submit(download_one, f, i, total): f
            for i, f in enumerate(to_download, 1)
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions

    elapsed = time.time() - stats["t0"]
    total_gb = stats["bytes"] / (1024**3)
    speed = stats["bytes"] / (1024**2) / max(elapsed, 1)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE!")
    logger.info(f"  Downloaded: {stats['done']}, Failed: {stats['fail']}")
    logger.info(f"  Data: {total_gb:.1f} GB in {elapsed/60:.1f} min")
    logger.info(f"  Speed: {speed:.1f} MB/s average")

    final = len(list(OUT_DIR.glob("*.svs")))
    final_gb = sum(f.stat().st_size for f in OUT_DIR.glob("*.svs")) / (1024**3)
    logger.info(f"  Total on disk: {final} slides ({final_gb:.1f} GB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
