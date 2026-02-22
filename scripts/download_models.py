#!/usr/bin/env python3
"""
Download all pretrained models to D: drive for offline use.
Models: ResNet18, ResNet50, ConvNeXt-Small, ConvNeXt-Base, DINOv2, Phikon, CTransPath
"""
import os
import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/mnt/d/skin_cancer_project/models")

def download_torchvision_models():
    """Download ResNet and ConvNeXt from torchvision."""
    from torchvision import models
    
    out = BASE / "torchvision"
    out.mkdir(parents=True, exist_ok=True)
    
    model_configs = [
        ("resnet18", models.resnet18, models.ResNet18_Weights.DEFAULT),
        ("resnet50", models.resnet50, models.ResNet50_Weights.DEFAULT),
        ("convnext_small", models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
        ("convnext_base", models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT),
    ]
    
    for name, model_fn, weights in model_configs:
        save_path = out / f"{name}.pth"
        if save_path.exists():
            logger.info(f"  ✓ {name} already exists ({save_path.stat().st_size/1e6:.1f} MB)")
            continue
        
        logger.info(f"  Downloading {name}...")
        model = model_fn(weights=weights)
        torch.save(model.state_dict(), save_path)
        logger.info(f"  ✓ {name} saved ({save_path.stat().st_size/1e6:.1f} MB)")


def download_dinov2():
    """Download DINOv2 ViT-Base from HuggingFace."""
    out = BASE / "vision" / "dinov2-base"
    if (out / "model.safetensors").exists() or (out / "pytorch_model.bin").exists():
        logger.info("  ✓ DINOv2-base already exists")
        return
    
    logger.info("  Downloading DINOv2-base...")
    try:
        from transformers import AutoModel, AutoConfig
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        model.save_pretrained(str(out))
        logger.info(f"  ✓ DINOv2-base saved to {out}")
    except Exception as e:
        logger.warning(f"  Transformers failed: {e}")
        logger.info("  Trying torch.hub...")
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            torch.save(model.state_dict(), out / "dinov2_vitb14.pth")
            logger.info(f"  ✓ DINOv2-base saved via torch.hub")
        except Exception as e2:
            logger.error(f"  Failed: {e2}")


def download_phikon():
    """Download Phikon (pathology ViT) from HuggingFace."""
    out = BASE / "pathology" / "phikon"
    if (out / "model.safetensors").exists() or (out / "pytorch_model.bin").exists():
        logger.info("  ✓ Phikon already exists")
        return
    
    logger.info("  Downloading Phikon (owkin/phikon)...")
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("owkin/phikon")
        model.save_pretrained(str(out))
        logger.info(f"  ✓ Phikon saved to {out}")
    except Exception as e:
        logger.error(f"  Phikon download failed: {e}")


def download_uni():
    """Try to download UNI from HuggingFace (gated - may fail)."""
    out = BASE / "pathology" / "uni"
    if (out / "pytorch_model.bin").exists() or (out / "model.safetensors").exists():
        logger.info("  ✓ UNI already exists")
        return
    
    logger.info("  Downloading UNI (MahmoodLab/UNI)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="MahmoodLab/UNI",
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )
        logger.info(f"  ✓ UNI saved to {out}")
    except Exception as e:
        logger.warning(f"  UNI download failed (gated access?): {e}")
        logger.info("  → Request access at https://huggingface.co/MahmoodLab/UNI")


def download_conch():
    """Try to download CONCH from HuggingFace (gated - may fail)."""
    out = BASE / "pathology" / "conch"
    if (out / "pytorch_model.bin").exists() or (out / "model.safetensors").exists():
        logger.info("  ✓ CONCH already exists")
        return
    
    logger.info("  Downloading CONCH (MahmoodLab/CONCH)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="MahmoodLab/CONCH",
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )
        logger.info(f"  ✓ CONCH saved to {out}")
    except Exception as e:
        logger.warning(f"  CONCH download failed (gated access?): {e}")
        logger.info("  → Request access at https://huggingface.co/MahmoodLab/CONCH")


def download_ctranspath():
    """Download CTransPath checkpoint."""
    out = BASE / "pathology"
    save_path = out / "ctranspath.pth"
    if save_path.exists():
        logger.info(f"  ✓ CTransPath already exists ({save_path.stat().st_size/1e6:.1f} MB)")
        return
    
    logger.info("  Downloading CTransPath...")
    try:
        from huggingface_hub import hf_hub_download
        # CTransPath is available on HuggingFace
        path = hf_hub_download(
            repo_id="jamessyx/CTransPath",
            filename="ctranspath.pth",
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )
        logger.info(f"  ✓ CTransPath saved to {save_path}")
    except Exception as e:
        logger.warning(f"  CTransPath HF failed: {e}")
        logger.info("  → Manual download from https://github.com/Xiyue-Wang/TransPath")


def main():
    logger.info("=" * 60)
    logger.info("DOWNLOADING ALL PRETRAINED MODELS → D:")
    logger.info("=" * 60)
    
    # 1. Torchvision models
    logger.info("\n[1/6] Torchvision Models (ResNet18, ResNet50, ConvNeXt-S/B)")
    download_torchvision_models()
    
    # 2. DINOv2
    logger.info("\n[2/6] DINOv2 (facebook/dinov2-base)")
    download_dinov2()
    
    # 3. Phikon
    logger.info("\n[3/6] Phikon (owkin/phikon)")
    download_phikon()
    
    # 4. CTransPath
    logger.info("\n[4/6] CTransPath")
    download_ctranspath()
    
    # 5. UNI (gated)
    logger.info("\n[5/6] UNI (gated - may need access approval)")
    download_uni()
    
    # 6. CONCH (gated)
    logger.info("\n[6/6] CONCH (gated - may need access approval)")
    download_conch()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for root, dirs, files in os.walk(BASE):
        level = root.replace(str(BASE), '').count(os.sep)
        indent = '  ' * level
        folder = os.path.basename(root)
        logger.info(f"{indent}{folder}/")
        for f in files:
            fp = Path(root) / f
            size = fp.stat().st_size / 1e6
            logger.info(f"{indent}  {f} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
