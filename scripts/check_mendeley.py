#!/usr/bin/env python3
"""Scrape Mendeley page for actual download links."""
import requests
import re

url = "https://data.mendeley.com/datasets/vccj8mp2cg/2"
r = requests.get(url, timeout=30)
print(f"Status: {r.status_code}")
print(f"Content length: {len(r.text)}")

# Find download links
download_links = re.findall(r'href="([^"]*download[^"]*)"', r.text, re.IGNORECASE)
print(f"\nDownload links ({len(download_links)}):")
for l in download_links[:20]:
    print(f"  {l}")

# Find any file links
file_links = re.findall(r'href="([^"]*/files/[^"]*)"', r.text, re.IGNORECASE)
print(f"\nFile links ({len(file_links)}):")
for l in file_links[:20]:
    print(f"  {l}")

# Find zip/rar links
zip_links = re.findall(r'href="([^"]*\.(zip|rar|tif|svs)[^"]*)"', r.text, re.IGNORECASE)
print(f"\nArchive links ({len(zip_links)}):")
for l in zip_links[:10]:
    print(f"  {l[0]}")

# Find S3/CDN links
cdn_links = re.findall(r'(https?://[^"]*amazonaws[^"]*)', r.text)
print(f"\nCDN links ({len(cdn_links)}):")
for l in cdn_links[:10]:
    print(f"  {l[:100]}")

# Also check json API
api_url = "https://data.mendeley.com/api/datasets-v2/ds/vccj8mp2cg"
try:
    r2 = requests.get(api_url, timeout=10)
    print(f"\nAPI v2 status: {r2.status_code}")
    if r2.status_code == 200:
        import json
        data = r2.json()
        print(json.dumps(data, indent=2, default=str)[:2000])
except Exception as e:
    print(f"API v2 error: {e}")
