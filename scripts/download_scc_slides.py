#!/usr/bin/env python3
"""
Download Histo-Seg dataset from Mendeley Data
==============================================
DOI: 10.17632/vccj8mp2cg.2
Content: 38 H&E WSI (BCC + SCC + IEC) with segmentation masks
"""

import os
import sys
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads" / "histo_seg"
SCC_DIR = DATA_DIR / "raw_wsi" / "scc"


def get_mendeley_files(doi="vccj8mp2cg", version="2"):
    """Get file listing from Mendeley Data API."""
    print("[1/3] Querying Mendeley Data API...")
    
    # Mendeley Data API endpoints
    urls = [
        f"https://data.mendeley.com/api/datasets/{doi}/versions/{version}",
        f"https://data.mendeley.com/api/datasets/{doi}",
    ]
    
    for url in urls:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, timeout=30)
            print(f"  Status: {r.status_code}")
            
            if r.status_code == 200:
                data = r.json()
                
                # Navigate to the data
                if isinstance(data, dict):
                    inner = data.get("data", data)
                    if isinstance(inner, dict):
                        # Look for files
                        files = inner.get("files", inner.get("data_files", []))
                        if files:
                            print(f"  Found {len(files)} files")
                            for f in files:
                                if isinstance(f, dict):
                                    name = f.get("filename", f.get("name", "?"))
                                    size = f.get("size", 0) / (1024**2)
                                    print(f"    {name}: {size:.0f} MB")
                            return files, inner
                        
                        # Try to get download URL for full dataset
                        download_url = inner.get("download_url", "")
                        if download_url:
                            print(f"  Download URL: {download_url}")
                            return [], inner
                        
                        print(f"  Keys: {list(inner.keys())}")
                        print(f"  Full response: {json.dumps(inner, indent=2, default=str)[:2000]}")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    return [], {}


def download_dataset():
    """Download the Histo-Seg dataset."""
    print("\n[2/3] Downloading Histo-Seg dataset...")
    
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Known S3 cache URL pattern for Mendeley
    urls = [
        "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/vccj8mp2cg-2.zip",
        "https://data.mendeley.com/datasets/vccj8mp2cg/2/files/download",
    ]
    
    output_file = DOWNLOAD_DIR / "histo_seg.zip"
    
    if output_file.exists() and output_file.stat().st_size > 1000000:
        print(f"  Already downloaded: {output_file} ({output_file.stat().st_size/(1024**2):.0f} MB)")
        return output_file
    
    for url in urls:
        print(f"\n  Trying: {url[:70]}...")
        try:
            r = requests.get(url, stream=True, timeout=600, allow_redirects=True)
            
            # Check for redirect
            if r.status_code in [301, 302, 307]:
                redirect_url = r.headers.get("Location", "")
                print(f"  Redirect to: {redirect_url[:80]}...")
                if redirect_url:
                    r = requests.get(redirect_url, stream=True, timeout=600)
            
            print(f"  Status: {r.status_code}")
            content_type = r.headers.get("content-type", "")
            print(f"  Content-Type: {content_type}")
            
            if r.status_code == 200 and "html" not in content_type.lower():
                total = int(r.headers.get("content-length", 0))
                if total > 0:
                    print(f"  Size: {total/(1024**2):.0f} MB")
                
                with open(output_file, "wb") as f:
                    with tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading") as pbar:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # Verify it's not HTML
                with open(output_file, "rb") as f:
                    header = f.read(4)
                
                if header[:2] == b"PK" or header[:4] == b"Rar!":
                    print(f"  Downloaded: {output_file.stat().st_size/(1024**2):.0f} MB")
                    return output_file
                elif header[:1] == b"<":
                    print("  Got HTML instead of data file, trying next URL...")
                    output_file.unlink()
                    continue
                else:
                    print(f"  File header: {header.hex()}")
                    return output_file
            
            print(f"  Failed (status={r.status_code})")
        except Exception as e:
            print(f"  Error: {e}")
    
    return None


def extract_and_organize(zip_path):
    """Extract and organize SCC files."""
    print("\n[3/3] Extracting and organizing...")
    
    if not zip_path or not zip_path.exists():
        print("  No file to extract")
        return
    
    SCC_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            print(f"  Files in archive: {len(names)}")
            
            # Show structure
            exts = {}
            for n in names:
                ext = Path(n).suffix.lower()
                exts[ext] = exts.get(ext, 0) + 1
            print(f"  File types: {exts}")
            
            # Find SCC-related files
            scc_files = [n for n in names if "scc" in n.lower() or "squamous" in n.lower()]
            print(f"  SCC-related files: {len(scc_files)}")
            
            # Show top-level dirs
            top_dirs = set()
            for n in names[:100]:
                parts = Path(n).parts
                if parts:
                    top_dirs.add(parts[0])
            print(f"  Top directories: {sorted(top_dirs)[:10]}")
            
            # Extract all
            extract_dir = DOWNLOAD_DIR / "extracted"
            extract_dir.mkdir(exist_ok=True)
            print("  Extracting all files...")
            zf.extractall(extract_dir)
            print("  Done!")
            
            # Show extracted structure
            for item in sorted(extract_dir.rglob("*"))[:30]:
                if item.is_file():
                    size_mb = item.stat().st_size / (1024**2)
                    print(f"    {item.relative_to(extract_dir)} ({size_mb:.1f} MB)")
    
    except zipfile.BadZipFile:
        print("  Not a valid zip file")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    print("=" * 60)
    print("HISTO-SEG DATASET DOWNLOADER (SCC WSI)")
    print("=" * 60)
    print("DOI: 10.17632/vccj8mp2cg.2")
    print("Content: 38 H&E WSI with BCC+SCC+IEC segmentation")
    print()
    
    # Step 1: Get file info
    files, metadata = get_mendeley_files()
    
    # Step 2: Download
    zip_path = download_dataset()
    
    # Step 3: Extract
    if zip_path:
        extract_and_organize(zip_path)
    else:
        print("\n Download failed.")
        print("Manual download:")
        print("  https://data.mendeley.com/datasets/vccj8mp2cg/2")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
