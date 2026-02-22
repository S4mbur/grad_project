#!/usr/bin/env python3
"""
SkinSight – Whole Slide Image Analysis Server  (v3 – multi-model)
================================================================
Key features:
  • 4-class skin cancer classification (Normal/Benign, BCC, SCC, Melanoma)
  • 6 feature extractor models + Ensemble mode
  • On-demand DZI tile serving (no pre-generation)
  • Attention-based heatmaps with top-tile navigation
  • Model selection per analysis
"""

import os
import sys
import io
import json
import uuid
import time
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, abort, Response
from flask_cors import CORS
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent
UPLOAD_DIR = APP_DIR / "uploads"
RESULTS_DIR = APP_DIR / "results"
STATIC_DIR = APP_DIR / "static"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class AppConfig:
    """Server configuration – all tunables in one place."""
    DELETE_SLIDE_AFTER_ANALYSIS = False
    RESULT_RETENTION_MINUTES = 60
    MAX_UPLOAD_SIZE_GB = 5

    # Analysis
    MAX_TILES_FOR_ANALYSIS = 200
    TILE_SIZE = 256
    MIN_TISSUE_FRACTION = 0.3
    FEATURE_BATCH_SIZE = 32

    # DZI (on-demand)
    DZI_TILE_SIZE = 254
    DZI_OVERLAP = 1
    DZI_QUALITY = 75
    SLIDE_CACHE_SIZE = 4

cfg = AppConfig()

# ---------------------------------------------------------------------------
# 4-Class Setup
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "Normal/Benign", 1: "BCC", 2: "SCC", 3: "Melanoma"}
CLASS_KEYS = ["normal", "bcc", "scc", "melanoma"]
N_CLASSES = 4

# ---------------------------------------------------------------------------
# Model Registry — best version of each model
# ---------------------------------------------------------------------------
MODELS_DIR = Path("/mnt/d/skin_cancer_project/models")
RESULTS_BASE = PROJECT_DIR / "results"

MODEL_REGISTRY = {
    "phikon": {
        "name": "Phikon",
        "display": "Phikon (Pathology Foundation)",
        "type": "phikon",
        "weights_path": str(MODELS_DIR / "pathology" / "phikon"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_phikon" / "best_model.pt"),
        "feat_dim": 768,
        "f1": 0.9250,
        "auc": 0.9811,
        "description": "Pathology-specialized ViT, highest F1 (92.5%)",
    },
    "convnext_small": {
        "name": "ConvNeXt-Small",
        "display": "ConvNeXt-Small",
        "type": "torchvision",
        "loader": "convnext_small",
        "weights_path": str(MODELS_DIR / "torchvision" / "convnext_small.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_convnext_small_v2" / "best_model.pt"),
        "feat_dim": 768,
        "f1": 0.8716,
        "auc": 0.9551,
        "description": "Modern CNN, strong performer (87.2% F1)",
    },
    "convnext_base": {
        "name": "ConvNeXt-Base",
        "display": "ConvNeXt-Base",
        "type": "torchvision",
        "loader": "convnext_base",
        "weights_path": str(MODELS_DIR / "torchvision" / "convnext_base.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_convnext_base_v2" / "best_model.pt"),
        "feat_dim": 1024,
        "f1": 0.8681,
        "auc": 0.9663,
        "description": "Larger ConvNeXt, high AUC (96.6%)",
    },
    "dinov2": {
        "name": "DINOv2-base",
        "display": "DINOv2-base (Self-Supervised)",
        "type": "dinov2",
        "weights_path": str(MODELS_DIR / "dinov2" / "dinov2_vitb14_pretrain"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_dinov2_base_v2" / "best_model.pt"),
        "feat_dim": 768,
        "f1": 0.8198,
        "auc": 0.9477,
        "description": "Self-supervised ViT from Meta (82.0% F1)",
    },
    "resnet18": {
        "name": "ResNet18",
        "display": "ResNet18 (Baseline)",
        "type": "torchvision",
        "loader": "resnet18",
        "weights_path": str(MODELS_DIR / "torchvision" / "resnet18.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_resnet18_v2" / "best_model.pt"),
        "feat_dim": 512,
        "f1": 0.8155,
        "auc": 0.9414,
        "description": "Lightweight baseline CNN (81.6% F1)",
    },
    "resnet50": {
        "name": "ResNet50",
        "display": "ResNet50",
        "type": "torchvision",
        "loader": "resnet50",
        "weights_path": str(MODELS_DIR / "torchvision" / "resnet50.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_resnet50_v2" / "best_model.pt"),
        "feat_dim": 2048,
        "f1": 0.7988,
        "auc": 0.9404,
        "description": "Deeper ResNet (79.9% F1)",
    },
}

# Ensemble uses top-3 models
ENSEMBLE_MODELS = ["phikon", "convnext_small", "convnext_base"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("skinsight")

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_SIZE_GB * 1024 * 1024 * 1024

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
analyses = {}
analyses_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Slide Handle Cache
# ---------------------------------------------------------------------------
class SlideCache:
    """Thread-safe LRU cache of OpenSlide objects."""
    def __init__(self, maxsize=4):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, path: str):
        openslide = _ensure_openslide()
        from openslide.deepzoom import DeepZoomGenerator
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                return self._cache[path]
            slide = openslide.OpenSlide(path)
            dz = DeepZoomGenerator(slide, tile_size=cfg.DZI_TILE_SIZE,
                                   overlap=cfg.DZI_OVERLAP, limit_bounds=True)
            self._cache[path] = (slide, dz)
            while len(self._cache) > self._maxsize:
                old_path, (old_slide, _) = self._cache.popitem(last=False)
                try: old_slide.close()
                except: pass
            return slide, dz

    def remove(self, path: str):
        with self._lock:
            if path in self._cache:
                slide, _ = self._cache.pop(path)
                try: slide.close()
                except: pass

    def clear(self):
        with self._lock:
            for _, (slide, _) in self._cache.items():
                try: slide.close()
                except: pass
            self._cache.clear()

slide_cache = SlideCache(maxsize=cfg.SLIDE_CACHE_SIZE)

# ---------------------------------------------------------------------------
# Lazy-loaded heavy modules & model cache
# ---------------------------------------------------------------------------
_torch = None
_openslide = None
_device = None

# Cache loaded encoders and MIL models by key
_encoder_cache = {}       # model_key -> (encoder, transform)
_mil_model_cache = {}     # model_key -> GatedAttentionMIL


def _ensure_torch():
    global _torch, _device
    if _torch is None:
        import torch
        _torch = torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch loaded – device: {_device}")
    return _torch, _device


def _ensure_openslide():
    global _openslide
    if _openslide is None:
        import openslide
        _openslide = openslide
        logger.info("OpenSlide loaded")
    return _openslide


# ---------------------------------------------------------------------------
# GatedAttentionMIL — matches train_all_models.py architecture exactly
# ---------------------------------------------------------------------------
def _build_mil_model(feat_dim, num_classes=4, hidden_dim=256, attn_dim=128, dropout=0.25):
    torch, device = _ensure_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class GatedAttentionMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
            self.attention_W = nn.Linear(attn_dim, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, num_classes))

        def forward(self, x):
            h = self.encoder(x)
            a = self.attention_W(self.attention_V(h) * self.attention_U(h))
            a = F.softmax(a, dim=0)
            z = torch.sum(a * h, dim=0, keepdim=True)
            return self.classifier(z), a.squeeze()

    return GatedAttentionMIL()


# ---------------------------------------------------------------------------
# Encoder loader (lazily cached per model key)
# ---------------------------------------------------------------------------
def _get_encoder(model_key):
    """Load or return cached encoder + transform for a model."""
    if model_key in _encoder_cache:
        return _encoder_cache[model_key]

    torch, device = _ensure_torch()
    import torch.nn as nn
    from torchvision import transforms, models

    mcfg = MODEL_REGISTRY[model_key]
    mtype = mcfg["type"]
    wpath = mcfg["weights_path"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if mtype == "torchvision":
        loader_name = mcfg["loader"]
        model_fn = getattr(models, loader_name)
        model = model_fn()
        state_dict = torch.load(wpath, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        if "resnet" in loader_name:
            model.fc = nn.Identity()
        elif "convnext" in loader_name:
            model.classifier = nn.Sequential(
                model.classifier[0],     # LayerNorm
                nn.Flatten(1),           # no Linear → raw features
            )

    elif mtype == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(wpath, local_files_only=True)

    elif mtype == "phikon":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(wpath, local_files_only=True)

    else:
        raise ValueError(f"Unknown model type: {mtype}")

    model = model.to(device)
    model.eval()
    logger.info(f"Encoder loaded: {mcfg['name']} ({mtype}, {mcfg['feat_dim']}d)")

    _encoder_cache[model_key] = (model, transform, mtype)
    return model, transform, mtype


def _get_mil_model(model_key):
    """Load or return cached MIL model for a model key."""
    if model_key in _mil_model_cache:
        return _mil_model_cache[model_key]

    torch, device = _ensure_torch()
    mcfg = MODEL_REGISTRY[model_key]
    ckpt_path = mcfg["mil_checkpoint"]

    if not Path(ckpt_path).exists():
        logger.warning(f"MIL checkpoint not found: {ckpt_path}")
        return None

    model = _build_mil_model(feat_dim=mcfg["feat_dim"])
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    _mil_model_cache[model_key] = model
    logger.info(f"MIL model loaded: {mcfg['name']} from {ckpt_path}")
    return model


# ===================================================================
# Analysis Pipeline
# ===================================================================
def run_analysis(job_id: str, slide_path: str, model_key: str):
    """Run full analysis pipeline in background thread."""
    try:
        is_ensemble = (model_key == "ensemble")
        model_display = "Ensemble (Top-3)" if is_ensemble else MODEL_REGISTRY[model_key]["name"]

        _update_job(job_id, status="processing", progress=5,
                    message=f"Opening slide... (Model: {model_display})")

        openslide = _ensure_openslide()
        slide = openslide.OpenSlide(slide_path)
        w, h = slide.dimensions
        mpp = float(slide.properties.get("openslide.mpp-x", 0.5))

        slide_info = {
            "width": w, "height": h,
            "mpp": round(mpp, 4),
            "vendor": slide.properties.get("openslide.vendor", "unknown"),
            "level_count": slide.level_count,
        }
        _update_job(job_id, progress=10,
                    message="Extracting tiles for analysis...",
                    slide_info=slide_info)

        # Step 1: Extract tiles
        tiles, tile_coords = _extract_tiles(slide, job_id)
        if not tiles:
            _update_job(job_id, status="error",
                        message="No tissue tiles found in slide.")
            slide.close()
            return

        _update_job(job_id, progress=30,
                    message=f"Extracted {len(tiles)} tiles. Loading {model_display}...")

        if is_ensemble:
            # Run ensemble: extract features & run MIL for each model
            all_probs = []
            all_attns = []
            model_results = []

            for i, mkey in enumerate(ENSEMBLE_MODELS):
                pct = 30 + int(50 * i / len(ENSEMBLE_MODELS))
                mname = MODEL_REGISTRY[mkey]["name"]
                _update_job(job_id, progress=pct,
                            message=f"Ensemble: running {mname} ({i+1}/{len(ENSEMBLE_MODELS)})...")

                features = _extract_features(tiles, mkey)
                pred, probs, attn = _run_mil_inference(features, mkey)
                all_probs.append(probs)
                all_attns.append(attn)
                model_results.append({
                    "model": mname,
                    "prediction": CLASS_NAMES[pred],
                    "probabilities": {CLASS_NAMES[c]: round(float(probs[c]), 4) for c in range(N_CLASSES)},
                })

            # Average probabilities
            avg_probs = np.mean(all_probs, axis=0)
            prediction = int(avg_probs.argmax())
            probabilities = avg_probs
            # Average attention for heatmap
            attention_weights = np.mean(all_attns, axis=0)

        else:
            # Single model
            _update_job(job_id, progress=40,
                        message=f"Extracting features with {model_display}...")
            features = _extract_features(tiles, model_key)

            _update_job(job_id, progress=65, message="Running MIL inference...")
            prediction, probabilities, attention_weights = _run_mil_inference(features, model_key)
            model_results = None

        _update_job(job_id, progress=80, message="Generating heatmap...")

        # Step 4: Generate heatmap
        heatmap_ok = _generate_heatmap(slide, tile_coords, attention_weights,
                                       probabilities, job_id)

        # Step 5: Top attention tiles
        top_tiles = _get_top_attention_tiles(tiles, tile_coords,
                                             attention_weights, job_id)

        slide.close()

        result = {
            "prediction": CLASS_NAMES[prediction],
            "prediction_id": int(prediction),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]), 4)
                for i in range(N_CLASSES)
            },
            "n_tiles": len(tiles),
            "top_tiles": top_tiles,
            "heatmap_available": heatmap_ok is not None,
            "model_used": model_display,
            "model_key": model_key,
            "timestamp": datetime.now().isoformat(),
        }

        # For ensemble, include individual model predictions
        if model_results:
            result["ensemble_details"] = model_results

        _update_job(job_id, status="completed", progress=100,
                    message="Analysis complete!", result=result)
        logger.info(f"Analysis complete for {job_id}: {CLASS_NAMES[prediction]} ({model_display})")

        if cfg.DELETE_SLIDE_AFTER_ANALYSIS:
            _cleanup_slide(job_id, slide_path)

    except Exception as e:
        logger.exception(f"Analysis failed for {job_id}")
        _update_job(job_id, status="error", message=str(e))


def _cleanup_slide(job_id: str, slide_path: str):
    try:
        slide_cache.remove(slide_path)
        p = Path(slide_path)
        if p.exists():
            p.unlink()
        upload_dir = UPLOAD_DIR / job_id
        if upload_dir.exists() and not any(upload_dir.iterdir()):
            upload_dir.rmdir()
    except Exception as e:
        logger.warning(f"[cleanup] Could not delete slide for {job_id}: {e}")


def _cleanup_old_results():
    if cfg.RESULT_RETENTION_MINUTES <= 0:
        return
    cutoff = datetime.now() - timedelta(minutes=cfg.RESULT_RETENTION_MINUTES)
    with analyses_lock:
        expired = [jid for jid, job in analyses.items()
                   if job.get("created_at") and
                   datetime.fromisoformat(job["created_at"]) < cutoff]
    for jid in expired:
        for d in (RESULTS_DIR / jid, UPLOAD_DIR / jid):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        with analyses_lock:
            analyses.pop(jid, None)
        logger.info(f"[cleanup] Expired job {jid}")


# ───────────────── Tile Extraction ──────────────────────

def _extract_tiles(slide, job_id):
    """Extract tissue tiles from a WSI for analysis."""
    import random

    w, h = slide.dimensions
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5))
    target_ds = mpp / 0.5 if mpp > 0 else 1.0
    level = slide.get_best_level_for_downsample(max(target_ds, 1.0))
    level_ds = slide.level_downsamples[level]
    read_size = int(cfg.TILE_SIZE * level_ds)

    thumb = slide.get_thumbnail((512, 512))
    thumb_arr = np.array(thumb.convert("RGB"))
    gray = np.mean(thumb_arr, axis=2)
    tissue_mask = (gray < 220) & (gray > 30)

    scale_x = w / thumb_arr.shape[1]
    scale_y = h / thumb_arr.shape[0]

    positions = []
    step = max(1, int(thumb_arr.shape[0] / 50))
    for ty in range(0, thumb_arr.shape[0], step):
        for tx in range(0, thumb_arr.shape[1], step):
            if tissue_mask[ty, tx]:
                x = int(tx * scale_x)
                y = int(ty * scale_y)
                if x + read_size <= w and y + read_size <= h:
                    positions.append((x, y))

    random.seed(42)
    random.shuffle(positions)
    positions = positions[:cfg.MAX_TILES_FOR_ANALYSIS * 3]

    tiles = []
    coords = []
    tile_save_dir = RESULTS_DIR / job_id / "tiles"
    tile_save_dir.mkdir(parents=True, exist_ok=True)

    for x, y in positions:
        if len(tiles) >= cfg.MAX_TILES_FOR_ANALYSIS:
            break
        region = slide.read_region((x, y), level, (cfg.TILE_SIZE, cfg.TILE_SIZE))
        tile = region.convert("RGB")
        arr = np.array(tile)
        gray_t = np.mean(arr, axis=2)
        tissue_frac = np.mean((gray_t < 220) & (gray_t > 30))
        if tissue_frac < cfg.MIN_TISSUE_FRACTION:
            continue

        idx = len(tiles)
        tiles.append(tile)
        coords.append({
            "x": x, "y": y,
            "level": level,
            "size": cfg.TILE_SIZE,
            "read_size": read_size,
            "level_ds": level_ds,
        })
        tile.save(str(tile_save_dir / f"tile_{idx:04d}.jpg"), quality=85)

    logger.info(f"Extracted {len(tiles)} tiles from {len(positions)} candidates")
    return tiles, coords


# ───────────────── Feature Extraction ───────────────────

def _extract_features(tiles, model_key):
    """Extract features from tile images using specified encoder."""
    torch, device = _ensure_torch()

    encoder, transform, mtype = _get_encoder(model_key)
    is_transformer = mtype in ("dinov2", "phikon")

    features = []
    bs = cfg.FEATURE_BATCH_SIZE
    for i in range(0, len(tiles), bs):
        batch = tiles[i:i + bs]
        tensors = torch.stack([transform(t) for t in batch]).to(device)
        with torch.no_grad():
            if is_transformer:
                out = encoder(tensors)
                feats = out.last_hidden_state[:, 0, :]  # CLS token
            else:
                feats = encoder(tensors)
        features.append(feats.cpu())

    return torch.cat(features, dim=0)


# ───────────────── MIL Inference ────────────────────────

def _run_mil_inference(features, model_key):
    """Run MIL model on extracted features."""
    torch, device = _ensure_torch()
    model = _get_mil_model(model_key)

    if model is None:
        logger.warning(f"MIL model not available for {model_key}, returning uniform")
        n = features.shape[0]
        return 0, np.ones(N_CLASSES) / N_CLASSES, np.ones(n) / n

    features = features.to(device)
    with torch.no_grad():
        logits, attn = model(features)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        pred = logits.argmax(1).item()
        attn_np = attn.cpu().numpy().flatten()

    return pred, probs, attn_np


# ───────────────── Heatmap ──────────────────────────────

def _make_clinical_colormap(n=256):
    cmap = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        t = i / (n - 1)
        if t < 0.25:
            s = t / 0.25
            cmap[i] = [1.0, 0.95 - 0.15 * s, 0.2 * (1 - s)]
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            cmap[i] = [1.0, 0.8 - 0.35 * s, 0.0]
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            cmap[i] = [1.0, 0.45 - 0.4 * s, 0.0 + 0.05 * s]
        else:
            s = (t - 0.75) / 0.25
            cmap[i] = [1.0, 0.05, 0.05 + 0.45 * s]
    return cmap


def _generate_heatmap(slide, tile_coords, attention_weights, probabilities, job_id):
    import cv2
    try:
        use_level = slide.level_count - 1
        for lv in range(slide.level_count):
            lw, lh = slide.level_dimensions[lv]
            if lw <= 4096 and lh <= 4096:
                use_level = lv
                break

        wL, hL = slide.level_dimensions[use_level]
        ds = float(slide.level_downsamples[use_level])

        thumb = slide.read_region((0, 0), use_level, (wL, hL)).convert("RGB")
        thumb_np = np.array(thumb).astype(np.float32) / 255.0

        gray = np.mean(thumb_np, axis=2)
        tissue_mask = ((gray < 0.90) & (gray > 0.08)).astype(np.float32)
        tissue_mask = cv2.GaussianBlur(tissue_mask, (0, 0), sigmaX=5, sigmaY=5)

        heat = np.zeros((hL, wL), dtype=np.float32)
        attn = attention_weights.copy()
        if attn.max() > attn.min():
            attn = (attn - attn.min()) / (attn.max() - attn.min())

        min_stamp = max(4, int(min(wL, hL) * 0.005))

        for i, coord in enumerate(tile_coords):
            if i >= len(attn):
                break
            x0, y0 = coord["x"], coord["y"]
            xL, yL = int(x0 / ds), int(y0 / ds)
            read_size = coord.get("read_size", coord["size"] * coord.get("level_ds", 1))
            stamp = max(min_stamp, int(read_size / ds))
            x2 = min(wL, xL + stamp)
            y2 = min(hL, yL + stamp)
            if xL < 0 or yL < 0 or xL >= wL or yL >= hL:
                continue
            heat[yL:y2, xL:x2] = np.maximum(heat[yL:y2, xL:x2], attn[i])

        sigma1 = max(3, min(wL, hL) / 200)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
        sigma2 = max(5, min(wL, hL) / 80)
        heat_blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma2, sigmaY=sigma2)

        hmin, hmax = float(heat_blur.min()), float(heat_blur.max())
        heat_norm = (heat_blur - hmin) / (hmax - hmin) if hmax > hmin else heat_blur
        heat_norm = np.power(heat_norm, 0.65)
        heat_norm[heat_norm < 0.02] = 0.0
        heat_norm *= tissue_mask

        cmap = _make_clinical_colormap(256)
        heat_idx = np.clip((heat_norm * 255).astype(np.int32), 0, 255)
        heat_color = cmap[heat_idx]

        alpha_map = np.clip(heat_norm * 0.65, 0, 0.65)
        alpha_3ch = alpha_map[:, :, np.newaxis]
        overlay = np.clip((1 - alpha_3ch) * thumb_np + alpha_3ch * heat_color, 0, 1)
        overlay_u8 = (overlay * 255).astype(np.uint8)

        out_dir = RESULTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(overlay_u8).save(str(out_dir / "heatmap.jpg"), quality=90)
        Image.fromarray((thumb_np * 255).astype(np.uint8)).save(
            str(out_dir / "thumbnail.jpg"), quality=90)

        heat_rgba = np.zeros((hL, wL, 4), dtype=np.uint8)
        heat_rgba[:, :, :3] = (heat_color * 255).astype(np.uint8)
        visible_alpha = np.clip(heat_norm * 200, 0, 200).astype(np.uint8)
        heat_rgba[:, :, 3] = visible_alpha
        Image.fromarray(heat_rgba).save(str(out_dir / "heatmap_only.png"))

        logger.info(f"Heatmap saved for {job_id} ({wL}×{hL})")
        return str(out_dir / "heatmap.jpg")

    except Exception as e:
        logger.exception(f"Heatmap generation failed: {e}")
        return None


def _get_top_attention_tiles(tiles, tile_coords, attention_weights, job_id, top_k=8):
    indices = np.argsort(attention_weights)[::-1][:top_k]
    top_tiles = []
    for rank, idx in enumerate(indices):
        if idx < len(tiles) and idx < len(tile_coords):
            top_tiles.append({
                "rank": rank + 1,
                "tile_index": int(idx),
                "attention": round(float(attention_weights[idx]), 6),
                "coord": tile_coords[idx],
                "image_url": f"/api/results/{job_id}/tiles/tile_{idx:04d}.jpg",
            })
    return top_tiles


def _update_job(job_id, **kwargs):
    with analyses_lock:
        if job_id in analyses:
            analyses[job_id].update(kwargs)
        else:
            analyses[job_id] = kwargs


# ===================================================================
#  API Routes
# ===================================================================

@app.route("/")
def serve_index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(str(STATIC_DIR), path)


# ───────── Models ─────────

@app.route("/api/models")
def list_models():
    """List available analysis models."""
    models = []
    for key, mcfg in MODEL_REGISTRY.items():
        ckpt_exists = Path(mcfg["mil_checkpoint"]).exists()
        models.append({
            "key": key,
            "name": mcfg["name"],
            "display": mcfg["display"],
            "f1": mcfg["f1"],
            "auc": mcfg["auc"],
            "description": mcfg["description"],
            "available": ckpt_exists,
        })
    # Sort by F1 descending
    models.sort(key=lambda x: x["f1"], reverse=True)

    return jsonify({
        "models": models,
        "ensemble": {
            "key": "ensemble",
            "name": "Ensemble",
            "display": "Ensemble (Top-3 Models)",
            "description": f"Averages predictions from {', '.join(MODEL_REGISTRY[m]['name'] for m in ENSEMBLE_MODELS)}",
            "models": ENSEMBLE_MODELS,
        },
        "default": "phikon",
    })


# ───────── Upload ─────────

@app.route("/api/upload", methods=["POST"])
def upload_slide():
    """Upload a WSI file and start analysis."""
    if "slide" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["slide"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn"):
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Model selection
    model_key = request.form.get("model", "phikon")
    if model_key != "ensemble" and model_key not in MODEL_REGISTRY:
        model_key = "phikon"

    job_id = str(uuid.uuid4())[:8]
    slide_dir = UPLOAD_DIR / job_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    slide_path = slide_dir / file.filename
    file.save(str(slide_path))

    file_size_mb = slide_path.stat().st_size / (1024 * 1024)
    model_display = "Ensemble" if model_key == "ensemble" else MODEL_REGISTRY[model_key]["name"]
    logger.info(f"Uploaded {file.filename} ({file_size_mb:.1f} MB) -> {job_id} [model: {model_display}]")

    with analyses_lock:
        analyses[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Upload complete. Starting analysis...",
            "filename": file.filename,
            "slide_path": str(slide_path),
            "file_size_mb": round(file_size_mb, 1),
            "model_key": model_key,
            "model_display": model_display,
            "created_at": datetime.now().isoformat(),
            "result": None,
        }

    thread = threading.Thread(target=run_analysis,
                              args=(job_id, str(slide_path), model_key), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "filename": file.filename,
                     "size_mb": round(file_size_mb, 1),
                     "model": model_display})


# ───────── Status ─────────

@app.route("/api/status/<job_id>")
def get_status(job_id):
    with analyses_lock:
        job = analyses.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    safe = {k: v for k, v in job.items() if k != "slide_path"}
    return jsonify(safe)


# ───────── On-demand DZI tile serving ─────────

@app.route("/api/results/<job_id>/dzi/slide.dzi")
def serve_dzi_descriptor(job_id):
    slide_path = _get_slide_path(job_id)
    if not slide_path:
        abort(404)
    slide, dz = slide_cache.get(slide_path)
    w, h = dz.level_dimensions[-1]
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpeg" Overlap="{cfg.DZI_OVERLAP}" TileSize="{cfg.DZI_TILE_SIZE}">
  <Size Width="{w}" Height="{h}"/>
</Image>"""
    return Response(xml, mimetype="application/xml")


@app.route("/api/results/<job_id>/dzi/slide_files/<int:level>/<path:tile_name>")
def serve_dzi_tile(job_id, level, tile_name):
    slide_path = _get_slide_path(job_id)
    if not slide_path:
        abort(404)
    try:
        name = tile_name.rsplit(".", 1)[0]
        col, row = map(int, name.split("_"))
    except (ValueError, IndexError):
        abort(400)

    slide, dz = slide_cache.get(slide_path)
    if level < 0 or level >= dz.level_count:
        abort(404)
    cols, rows = dz.level_tiles[level]
    if col < 0 or col >= cols or row < 0 or row >= rows:
        abort(404)

    tile = dz.get_tile(level, (col, row))
    buf = io.BytesIO()
    tile.save(buf, "JPEG", quality=cfg.DZI_QUALITY)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


def _get_slide_path(job_id: str) -> str | None:
    with analyses_lock:
        job = analyses.get(job_id)
    if not job:
        return None
    sp = job.get("slide_path")
    if sp and Path(sp).exists():
        return sp
    return None


# ───────── Result assets ─────────

@app.route("/api/results/<job_id>/heatmap")
def serve_heatmap(job_id):
    p = RESULTS_DIR / job_id / "heatmap.jpg"
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="image/jpeg")

@app.route("/api/results/<job_id>/heatmap_only")
def serve_heatmap_only(job_id):
    p = RESULTS_DIR / job_id / "heatmap_only.png"
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="image/png")

@app.route("/api/results/<job_id>/thumbnail")
def serve_thumbnail(job_id):
    p = RESULTS_DIR / job_id / "thumbnail.jpg"
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="image/jpeg")

@app.route("/api/results/<job_id>/tiles/<path:filename>")
def serve_tile(job_id, filename):
    p = RESULTS_DIR / job_id / "tiles" / filename
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="image/jpeg")


# ───────── History & Export ─────────

@app.route("/api/history")
def get_history():
    _cleanup_old_results()
    with analyses_lock:
        history = []
        for jid, job in sorted(analyses.items(),
                                key=lambda x: x[1].get("created_at", ""),
                                reverse=True):
            history.append({
                "job_id": jid,
                "filename": job.get("filename", "unknown"),
                "status": job.get("status", "unknown"),
                "created_at": job.get("created_at", ""),
                "model": job.get("model_display", ""),
                "result": job.get("result"),
            })
    return jsonify(history)


@app.route("/api/results/<job_id>/export")
def export_results(job_id):
    with analyses_lock:
        job = analyses.get(job_id)
    if not job or not job.get("result"):
        return jsonify({"error": "No results available"}), 404

    export_data = {
        "job_id": job_id,
        "filename": job.get("filename"),
        "analysis_date": job.get("created_at"),
        "model": job.get("model_display"),
        "result": job["result"],
        "slide_info": job.get("slide_info"),
    }

    export_path = RESULTS_DIR / job_id / "export.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2)

    return send_file(str(export_path), mimetype="application/json",
                     as_attachment=True,
                     download_name=f"skinsight_report_{job_id}.json")


@app.route("/api/results/<job_id>/delete", methods=["POST"])
def delete_results(job_id):
    for d in (RESULTS_DIR / job_id, UPLOAD_DIR / job_id):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    with analyses_lock:
        job = analyses.pop(job_id, None)
    if job:
        slide_cache.remove(job.get("slide_path", ""))
    return jsonify({"deleted": job_id})


# ───────── Info ─────────

@app.route("/api/info")
def server_info():
    upload_size = sum(f.stat().st_size for f in UPLOAD_DIR.rglob("*") if f.is_file())
    results_size = sum(f.stat().st_size for f in RESULTS_DIR.rglob("*") if f.is_file())
    with analyses_lock:
        n_jobs = len(analyses)
    return jsonify({
        "status": "ok",
        "jobs": n_jobs,
        "uploads_mb": round(upload_size / 1e6, 1),
        "results_mb": round(results_size / 1e6, 1),
        "n_models": len(MODEL_REGISTRY),
        "classes": list(CLASS_NAMES.values()),
    })


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"Starting SkinSight server on port {port}")
    logger.info(f"  Models: {len(MODEL_REGISTRY)} + Ensemble")
    logger.info(f"  Classes: {', '.join(CLASS_NAMES.values())}")
    logger.info(f"  Ensemble: {', '.join(MODEL_REGISTRY[m]['name'] for m in ENSEMBLE_MODELS)}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
