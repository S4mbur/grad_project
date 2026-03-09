#!/usr/bin/env python3
"""
Download all remaining TCGA-SKCM diagnostic slides.

Designed for the current project layout:
  /mnt/d/skin_cancer_project/datasets/tcga_skcm

Behavior:
  - queries GDC if local manifest is missing
  - skips already complete .svs files
  - downloads the full remaining set by default
  - writes to .part first, then renames on success
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
DEFAULT_OUT_DIR = Path("/mnt/d/skin_cancer_project/datasets/tcga_skcm")
DEFAULT_MANIFEST = DEFAULT_OUT_DIR / "download_manifest.json"
CHUNK_SIZE = 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

stats_lock = threading.Lock()
stats = {
    "done": 0,
    "failed": 0,
    "skipped": 0,
    "bytes": 0,
    "t0": 0.0,
}


def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit(
            f"Another download_tcga_skcm_full.py instance is already running.\n"
            f"Lock file: {lock_path}"
        )
    fh.write(str(Path("/proc/self").resolve().parent.name))
    fh.flush()
    return fh


def query_manifest() -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-SKCM"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,md5sum,cases.case_id,cases.submitter_id",
        "format": "JSON",
        "size": 1000,
    }
    resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json().get("data", {}).get("hits", [])
    slides = [h for h in hits if h.get("file_name", "").endswith(".svs")]
    slides.sort(key=lambda x: x.get("file_size", 0))
    return slides


def load_or_create_manifest(manifest_path: Path, refresh: bool) -> list[dict]:
    if manifest_path.exists() and not refresh:
        logger.info("Using existing manifest: %s", manifest_path)
        return json.loads(manifest_path.read_text())

    logger.info("Querying GDC for TCGA-SKCM diagnostic slides...")
    slides = query_manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(slides, indent=2))
    logger.info("Saved manifest: %s", manifest_path)
    return slides


def select_remaining(slides: list[dict], out_dir: Path, max_total_gb: float | None) -> tuple[list[dict], int, float]:
    existing = {fp.name: fp.stat().st_size for fp in out_dir.glob("*.svs")}
    existing_complete = 0
    selected: list[dict] = []

    existing_bytes = 0
    for slide in slides:
        fname = slide["file_name"]
        fsize = int(slide.get("file_size", 0))
        if fname in existing and existing[fname] >= fsize * 0.95:
            existing_complete += 1
            existing_bytes += fsize

    running_gb = existing_bytes / (1024 ** 3)
    for slide in slides:
        fname = slide["file_name"]
        fsize = int(slide.get("file_size", 0))
        if fname in existing and existing[fname] >= fsize * 0.95:
            continue
        slide_gb = fsize / (1024 ** 3)
        if max_total_gb is not None and running_gb + slide_gb > max_total_gb:
            continue
        selected.append(slide)
        running_gb += slide_gb

    return selected, existing_complete, running_gb


def human_gb(num_bytes: float) -> float:
    return round(num_bytes / (1024 ** 3), 2)


def download_one(slide: dict, out_dir: Path, total: int, retries: int) -> tuple[str, str]:
    fname = slide["file_name"]
    file_id = slide["file_id"]
    fsize = int(slide.get("file_size", 0))
    final_path = out_dir / fname
    part_path = out_dir / f"{fname}.part"

    if final_path.exists() and final_path.stat().st_size >= fsize * 0.95:
        with stats_lock:
            stats["skipped"] += 1
        return fname, "skipped"

    url = f"{GDC_DATA_ENDPOINT}/{file_id}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)

            if part_path.stat().st_size < fsize * 0.95:
                raise RuntimeError(
                    f"incomplete download ({part_path.stat().st_size} / {fsize} bytes)"
                )

            part_path.replace(final_path)
            with stats_lock:
                stats["done"] += 1
                stats["bytes"] += final_path.stat().st_size
                elapsed = max(time.time() - stats["t0"], 1.0)
                speed = stats["bytes"] / (1024 ** 2) / elapsed
                logger.info(
                    "[%d/%d] OK %s | %.1f MB | %.1f MB/s",
                    stats["done"],
                    total,
                    fname[:80],
                    final_path.stat().st_size / (1024 ** 2),
                    speed,
                )
            return fname, "ok"
        except Exception as exc:
            last_err = exc
            try:
                part_path.unlink(missing_ok=True)
            except TypeError:
                if part_path.exists():
                    try:
                        part_path.unlink()
                    except FileNotFoundError:
                        pass
            logger.warning("Retry %d/%d failed for %s: %s", attempt, retries, fname[:80], exc)

    with stats_lock:
        stats["failed"] += 1
    return fname, f"failed: {last_err}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all remaining TCGA-SKCM slides")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--max-total-gb",
        type=float,
        default=None,
        help="Optional total disk cap including existing files. Default: no cap.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Ignore local manifest and query GDC again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    manifest_path = args.manifest

    if not out_dir.parent.exists():
        raise SystemExit(
            f"Output parent path missing: {out_dir.parent}\n"
            "If /mnt/d is empty, mount D: first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = acquire_lock(out_dir / ".download_tcga_skcm_full.lock")

    slides = load_or_create_manifest(manifest_path, args.refresh_manifest)
    total_manifest_gb = human_gb(sum(int(s.get("file_size", 0)) for s in slides))
    logger.info("Manifest slides: %d (%.2f GB)", len(slides), total_manifest_gb)

    remaining, existing_complete, projected_total_gb = select_remaining(
        slides, out_dir, args.max_total_gb
    )
    remaining_gb = human_gb(sum(int(s.get("file_size", 0)) for s in remaining))

    logger.info("Already complete: %d", existing_complete)
    logger.info("To download: %d (%.2f GB)", len(remaining), remaining_gb)
    logger.info("Projected total on disk: %.2f GB", projected_total_gb)

    if not remaining:
        logger.info("Nothing to download.")
        return

    stats["t0"] = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, slide, out_dir, len(remaining), args.retries): slide
            for slide in remaining
        }
        for future in as_completed(futures):
            fname, status = future.result()
            if status.startswith("failed:"):
                logger.error("%s -> %s", fname, status)

    elapsed = max(time.time() - stats["t0"], 1.0)
    logger.info("Finished in %.1f min", elapsed / 60)
    logger.info(
        "Summary: downloaded=%d skipped=%d failed=%d data=%.2f GB",
        stats["done"],
        stats["skipped"],
        stats["failed"],
        human_gb(stats["bytes"]),
    )
    lock_fh.close()


if __name__ == "__main__":
    main()
