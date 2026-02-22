#!/usr/bin/env python3
"""
Download cutaneous SCC data from available public sources.

Strategy:
1. Try UQ NMSC dataset (60 SCC WSIs, TIF format)
2. UQ dataset is available at: https://espace.library.uq.edu.au/view/UQ:356b5ab
   Files are accessible via direct download links

The UQ NMSC dataset structure:
- SCC folder: 60 cases of Squamous Cell Carcinoma
- Images in TIF format (1x, 2x, 5x, 10x downsample)
- Segmentation masks in PNG format
"""

import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCC_DIR = DATA_DIR / "raw_wsi" / "scc"
MANIFEST_DIR = DATA_DIR / "manifests"

MAX_TOTAL_GB = 20


def check_uq_nmsc():
    """Check UQ eSpace dataset availability."""
    print("="*60)
    print("Checking UQ NMSC Dataset availability...")
    print("="*60)
    
    # UQ eSpace record ID: UQ:356b5ab
    # Known file structure: SCC.zip, BCC.zip, IEC.zip, masks, margins
    base_url = "https://espace.library.uq.edu.au"
    record_id = "UQ_356b5ab"
    
    # Try known file patterns
    potential_files = [
        f"{base_url}/data/{record_id}/SCC_1x.zip",
        f"{base_url}/data/{record_id}/SCC.zip", 
        f"{base_url}/data/{record_id}/images_1x.zip",
    ]
    
    print("\nChecking direct download links:")
    for url in potential_files:
        try:
            r = requests.head(url, timeout=10, allow_redirects=True)
            size_mb = int(r.headers.get('content-length', 0)) / (1024*1024)
            print(f"  {url.split('/')[-1]:30s} -> {r.status_code} ({size_mb:.0f} MB)")
        except Exception as e:
            print(f"  {url.split('/')[-1]:30s} -> Error: {e}")
    
    # Try view page
    try:
        r = requests.get(f"{base_url}/view/{record_id.replace('_', ':')}", timeout=10)
        print(f"\n  Record page status: {r.status_code}")
        if r.status_code == 200:
            # Look for download links in HTML
            import re
            links = re.findall(r'href="([^"]*(?:SCC|scc)[^"]*\.(?:zip|tif))"', r.text, re.IGNORECASE)
            if links:
                print(f"  Found SCC download links: {links[:5]}")
            
            # Look for any data links
            data_links = re.findall(r'href="(/data/[^"]+)"', r.text)
            if data_links:
                print(f"\n  Data files found ({len(data_links)}):")
                for link in data_links[:20]:
                    print(f"    {link}")
    except Exception as e:
        print(f"  Page error: {e}")


def check_mendeley():
    """Check Mendeley skin cancer dataset."""
    print("\n" + "="*60)
    print("Checking Mendeley Skin Cancer Dataset...")
    print("="*60)
    print("  DOI: 10.17632/d48b5zybck.1")
    print("  Size: 3.37 GB (RAR)")
    print("  Content: 4,357 histology patches")
    print("  Classes: BCC, SCC, Melanoma, Benign")
    print("  Note: These are patches NOT full WSIs")
    print("        But good for training!")
    
    # Try Mendeley API
    try:
        api_url = "https://data.mendeley.com/api/datasets/d48b5zybck"
        r = requests.get(api_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"\n  Title: {data.get('name', 'N/A')}")
            
            versions = data.get('versions', [])
            if versions:
                v = versions[-1] if isinstance(versions, list) else versions
                print(f"  Version: {v}")
            
            files = data.get('files', [])
            if files:
                for f in files[:10]:
                    fname = f.get('filename', f.get('name', 'unknown'))
                    fsize = f.get('size', 0) / (1024*1024)
                    print(f"  File: {fname} ({fsize:.0f} MB)")
        else:
            print(f"  API status: {r.status_code}")
    except Exception as e:
        print(f"  API error: {e}")
    
    # Try version 1 API
    try:
        api_v1 = "https://data.mendeley.com/api/datasets/d48b5zybck/versions/1/files"
        r = requests.get(api_v1, timeout=10)
        if r.status_code == 200:
            files = r.json()
            print(f"\n  Files in dataset ({len(files)}):")
            total = 0
            for f in files:
                fname = f.get('filename', f.get('name', 'unknown'))
                fsize = f.get('size', 0) / (1024*1024)
                total += fsize
                dl = f.get('download_url', f.get('content_details', {}).get('download_url', ''))
                print(f"    {fname}: {fsize:.0f} MB")
                if dl:
                    print(f"      Download: {dl[:80]}...")
            print(f"  Total: {total:.0f} MB ({total/1024:.2f} GB)")
    except Exception as e:
        print(f"  Files API error: {e}")


def check_figshare():
    """Check Figshare for skin SCC datasets."""
    print("\n" + "="*60)
    print("Checking Figshare for cSCC datasets...")
    print("="*60)
    
    try:
        api = "https://api.figshare.com/v2/articles/search"
        payload = {
            "search_for": "skin squamous cell carcinoma whole slide image histopathology",
            "page_size": 5
        }
        r = requests.post(api, json=payload, timeout=10)
        if r.status_code == 200:
            results = r.json()
            print(f"  Found {len(results)} results:")
            for item in results:
                print(f"\n  Title: {item.get('title', 'N/A')[:80]}")
                print(f"  DOI: {item.get('doi', 'N/A')}")
                print(f"  URL: {item.get('url_public_html', 'N/A')}")
                print(f"  Published: {item.get('published_date', 'N/A')[:10]}")
                
                # Get files
                article_id = item.get('id')
                if article_id:
                    fr = requests.get(f"https://api.figshare.com/v2/articles/{article_id}/files", timeout=10)
                    if fr.status_code == 200:
                        files = fr.json()
                        total_gb = sum(f.get('size', 0) for f in files) / (1024**3)
                        print(f"  Files: {len(files)}, Total: {total_gb:.2f} GB")
                        for f in files[:5]:
                            print(f"    {f.get('name','?'):50s} {f.get('size',0)/(1024**2):.0f} MB")
    except Exception as e:
        print(f"  Figshare error: {e}")


def main():
    print("CUTANEOUS SCC DATA SOURCE EXPLORER")
    print("="*60)
    
    check_uq_nmsc()
    check_mendeley()
    check_figshare()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)


if __name__ == "__main__":
    main()
