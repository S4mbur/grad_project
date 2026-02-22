#!/usr/bin/env python3
"""
Trial Pipeline for Multi-Source WSI Testing
============================================

Bu script, her dataset kaynağından (COBRA, TCGA-SKCM, UQ-NMSC) 
birer örnek slide indirip/kullanıp tile extraction'ı test eder.

Amaç: Farklı formatların uyumluluğunu doğrulamak

Kullanım:
    python scripts/trial_multiclass_pipeline.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Trial configuration
TRIAL_CONFIG = {
    "output_dir": PROJECT_ROOT / "data" / "trial_multiclass",
    "tile_size": 512,
    "target_mpp": 0.5,
    "max_tiles": 10,  # Just 10 tiles for quick test
    "sources": {
        "cobra_bcc": {
            "label": 1,
            "label_name": "bcc",
            "format": "tif",
            "source_type": "local",  # Already downloaded
        },
        "cobra_benign": {
            "label": 0,
            "label_name": "benign",
            "format": "tif",
            "source_type": "local",
        },
        "tcga_melanoma": {
            "label": 3,
            "label_name": "melanoma",
            "format": "svs",
            "source_type": "gdc",
            "example_uuid": "3f66b550-8e59-4d6b-8725-e04603d1d09f",  # Example from GDC
        },
        "uq_scc": {
            "label": 2,
            "label_name": "scc",
            "format": "tif",
            "source_type": "uq",
            "download_url": "https://espace.library.uq.edu.au/view/UQ:e8d9af2",
        }
    }
}


def setup_directories():
    """Create trial output directories."""
    dirs = [
        TRIAL_CONFIG["output_dir"],
        TRIAL_CONFIG["output_dir"] / "slides",
        TRIAL_CONFIG["output_dir"] / "tiles",
        TRIAL_CONFIG["output_dir"] / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created trial directories at {TRIAL_CONFIG['output_dir']}")
    return True


def get_cobra_sample_slides():
    """Get sample slides from existing COBRA data."""
    raw_wsi_dir = PROJECT_ROOT / "data" / "raw_wsi"
    
    results = {"bcc": None, "benign": None}
    
    # Direct approach: just pick first available TIF files
    # (Manifest may have different column names or not exist)
    tif_files = sorted(list(raw_wsi_dir.glob("*.tif")))
    
    if len(tif_files) >= 2:
        results["bcc"] = str(tif_files[0])
        results["benign"] = str(tif_files[1])
        print(f"✓ Found COBRA BCC sample: {tif_files[0].name}")
        print(f"✓ Found COBRA Benign sample: {tif_files[1].name}")
    elif len(tif_files) == 1:
        results["bcc"] = str(tif_files[0])
        print(f"✓ Found COBRA sample: {tif_files[0].name}")
    else:
        print("✗ No COBRA TIF files found in raw_wsi directory")
    
    return results


def check_tcga_sample():
    """Check if TCGA sample exists or provide download instructions."""
    raw_wsi_dir = PROJECT_ROOT / "data" / "raw_wsi"
    
    # Check for any existing SVS file (TCGA format)
    svs_files = list(raw_wsi_dir.glob("*.svs"))
    
    if svs_files:
        # Filter for meaningful SVS files (not test files)
        valid_svs = [f for f in svs_files if f.stat().st_size > 1_000_000]  # > 1MB
        if valid_svs:
            print(f"✓ Found existing SVS file: {valid_svs[0].name}")
            return str(valid_svs[0])
    
    # No TCGA sample found - provide instructions
    print("\n" + "="*60)
    print("⚠️  TCGA-SKCM Sample Not Found")
    print("="*60)
    print("""
Melanoma için TCGA'dan bir örnek WSI indirmeniz gerekiyor.

ADIMLAR:
1. GDC Portal'a gidin: https://portal.gdc.cancer.gov

2. Filters uygulayın:
   - Project: TCGA-SKCM
   - Data Category: Biospecimen
   - Data Type: Slide Image
   
3. Tek bir dosya seçin ve "Download" tıklayın

4. İndirilen .svs dosyasını buraya taşıyın:
   {raw_wsi_dir}

Veya GDC Client ile:
   gdc-client download <file-uuid>
   
Örnek UUID: 3f66b550-8e59-4d6b-8725-e04603d1d09f
""".format(raw_wsi_dir=raw_wsi_dir))
    print("="*60 + "\n")
    
    return None


def check_uq_sample():
    """Check if UQ NMSC sample exists or provide download instructions."""
    trial_dir = TRIAL_CONFIG["output_dir"] / "slides"
    
    # Check for any UQ-style files
    uq_files = list(trial_dir.glob("*scc*.tif")) + list(trial_dir.glob("*SCC*.tif"))
    
    if uq_files:
        print(f"✓ Found UQ NMSC sample: {uq_files[0].name}")
        return str(uq_files[0])
    
    # No UQ sample found - provide instructions
    print("\n" + "="*60)
    print("⚠️  UQ NMSC Dataset Sample Not Found")
    print("="*60)
    print("""
SCC için UQ NMSC dataset'inden bir örnek indirmeniz gerekiyor.

ADIMLAR:
1. UQ eSpace'e gidin:
   https://espace.library.uq.edu.au/view/UQ:e8d9af2

2. Kullanım şartlarını kabul edin

3. 'data_1x.zip' dosyasını indirin (en küçük versiyon)

4. ZIP'i açın ve bir SCC örneği buraya kopyalayın:
   {trial_dir}

Örnek dosya adı: G45-SCC-1.tif
""".format(trial_dir=trial_dir))
    print("="*60 + "\n")
    
    return None


def test_openslide_compatibility(slide_path, source_name):
    """Test if a slide can be opened with OpenSlide."""
    try:
        import openslide
        
        slide = openslide.OpenSlide(slide_path)
        
        info = {
            "source": source_name,
            "path": slide_path,
            "dimensions": slide.dimensions,
            "level_count": slide.level_count,
            "level_dimensions": slide.level_dimensions,
            "properties": dict(slide.properties),
            "mpp": None,
            "status": "success"
        }
        
        # Try to get MPP
        if 'openslide.mpp-x' in slide.properties:
            info["mpp"] = float(slide.properties['openslide.mpp-x'])
        elif 'tiff.XResolution' in slide.properties:
            try:
                xres = float(slide.properties['tiff.XResolution'])
                if xres > 0:
                    info["mpp"] = 10000.0 / xres  # Convert to µm/px
            except:
                pass
        
        slide.close()
        
        print(f"\n✓ OpenSlide test PASSED for {source_name}")
        print(f"  Dimensions: {info['dimensions']}")
        print(f"  Levels: {info['level_count']}")
        print(f"  MPP: {info['mpp']}")
        
        return info
        
    except Exception as e:
        print(f"\n✗ OpenSlide test FAILED for {source_name}: {e}")
        return {
            "source": source_name,
            "path": slide_path,
            "status": "failed",
            "error": str(e)
        }


def test_tile_extraction(slide_path, source_name, output_dir):
    """Test tile extraction from a slide."""
    try:
        from src.dataset.tile_extractor import TileExtractor, TileExtractionConfig
        
        # Create output directory for this source
        source_output = output_dir / source_name
        source_output.mkdir(parents=True, exist_ok=True)
        
        # Create config with quick test settings
        tile_config = TileExtractionConfig(
            tile_size=TRIAL_CONFIG["tile_size"],
            target_mpp=TRIAL_CONFIG["target_mpp"],
            max_tiles_per_slide=TRIAL_CONFIG["max_tiles"],
            min_tissue_fraction=0.3,
            jpeg_quality=90
        )
        
        extractor = TileExtractor(config=tile_config)
        
        # Extract tiles using correct API
        slide_id = Path(slide_path).stem
        result = extractor.extract_tiles(
            slide_path=slide_path,
            output_dir=str(source_output),
            slide_id=slide_id,
            label=str(TRIAL_CONFIG["sources"].get(source_name, {}).get("label", 0)),
            split="trial"
        )
        
        # Count extracted tiles
        tile_dir = source_output / slide_id
        if tile_dir.exists():
            tile_files = list(tile_dir.glob("*.jpg"))
            num_tiles = len(tile_files)
        else:
            num_tiles = len(result) if isinstance(result, list) else 0
        
        print(f"\n✓ Tile extraction PASSED for {source_name}")
        print(f"  Tiles extracted: {num_tiles}")
        print(f"  Output: {source_output}")
        
        return {
            "source": source_name,
            "status": "success",
            "num_tiles": num_tiles,
            "output_dir": str(source_output)
        }
        
    except Exception as e:
        import traceback
        print(f"\n✗ Tile extraction FAILED for {source_name}: {e}")
        traceback.print_exc()
        return {
            "source": source_name,
            "status": "failed",
            "error": str(e)
        }


def generate_trial_report(results):
    """Generate a summary report of the trial."""
    report_path = TRIAL_CONFIG["output_dir"] / "trial_report.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "tile_size": TRIAL_CONFIG["tile_size"],
            "target_mpp": TRIAL_CONFIG["target_mpp"],
            "max_tiles": TRIAL_CONFIG["max_tiles"]
        },
        "results": results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("TRIAL PIPELINE SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results.values() if r and r.get("status") == "success")
    total_count = len([r for r in results.values() if r])
    
    print(f"\nResults: {success_count}/{total_count} sources tested successfully")
    
    for source, result in results.items():
        if result:
            status = "✓" if result.get("status") == "success" else "✗"
            print(f"  {status} {source}: {result.get('status', 'not tested')}")
    
    print(f"\nFull report saved to: {report_path}")
    print("="*60)
    
    return report


def main():
    """Run the trial pipeline."""
    print("\n" + "="*60)
    print("MULTI-SOURCE WSI TRIAL PIPELINE")
    print("Testing compatibility across different dataset sources")
    print("="*60 + "\n")
    
    # Step 1: Setup
    print("[1/5] Setting up directories...")
    setup_directories()
    
    # Step 2: Find sample slides
    print("\n[2/5] Locating sample slides...")
    
    # COBRA samples (should exist)
    cobra_samples = get_cobra_sample_slides()
    
    # TCGA sample
    tcga_sample = check_tcga_sample()
    
    # UQ sample
    uq_sample = check_uq_sample()
    
    # Step 3: Test OpenSlide compatibility
    print("\n[3/5] Testing OpenSlide compatibility...")
    openslide_results = {}
    
    if cobra_samples["bcc"]:
        openslide_results["cobra_bcc"] = test_openslide_compatibility(
            cobra_samples["bcc"], "cobra_bcc"
        )
    
    if cobra_samples["benign"]:
        openslide_results["cobra_benign"] = test_openslide_compatibility(
            cobra_samples["benign"], "cobra_benign"
        )
    
    if tcga_sample:
        openslide_results["tcga_melanoma"] = test_openslide_compatibility(
            tcga_sample, "tcga_melanoma"
        )
    
    if uq_sample:
        openslide_results["uq_scc"] = test_openslide_compatibility(
            uq_sample, "uq_scc"
        )
    
    # Step 4: Test tile extraction
    print("\n[4/5] Testing tile extraction...")
    tile_results = {}
    tiles_output = TRIAL_CONFIG["output_dir"] / "tiles"
    
    for source_name, slide_path in [
        ("cobra_bcc", cobra_samples.get("bcc")),
        ("cobra_benign", cobra_samples.get("benign")),
        ("tcga_melanoma", tcga_sample),
        ("uq_scc", uq_sample),
    ]:
        if slide_path and openslide_results.get(source_name, {}).get("status") == "success":
            tile_results[source_name] = test_tile_extraction(
                slide_path, source_name, tiles_output
            )
        else:
            tile_results[source_name] = None
    
    # Step 5: Generate report
    print("\n[5/5] Generating trial report...")
    
    # Combine results
    all_results = {}
    for source in ["cobra_bcc", "cobra_benign", "tcga_melanoma", "uq_scc"]:
        all_results[source] = {
            "openslide": openslide_results.get(source),
            "tile_extraction": tile_results.get(source)
        }
    
    report = generate_trial_report(all_results)
    
    # Provide next steps
    missing = []
    if not tcga_sample:
        missing.append("TCGA-SKCM (Melanoma)")
    if not uq_sample:
        missing.append("UQ NMSC (SCC)")
    
    if missing:
        print("\n⚠️  Missing samples for:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease download as instructed above, then re-run this script.")
    else:
        print("\n🎉 All samples available! Ready for full dataset integration.")
    
    return report


if __name__ == "__main__":
    main()
