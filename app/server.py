#!/usr/bin/env python3
"""
SkinSight â€“ Whole Slide Image Analysis Server  (v3 â€“ multi-model)
================================================================
Key features:
  â€¢ 4-class skin cancer classification (Normal/Benign, BCC, SCC, Melanoma)
  â€¢ 6 feature extractor models + Ensemble mode
  â€¢ On-demand DZI tile serving (no pre-generation)
  â€¢ Attention-based heatmaps with top-tile navigation
  â€¢ Model selection per analysis
"""

import os
import sys
import io
import json
import uuid
import time
import csv
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
PHASE1_BANK_DIR = PROJECT_DIR / "results" / "phase1_hard_case_bank"
PHASE2_DIR = PROJECT_DIR / "results" / "phase2_safety"
PHASE2_CALIBRATION_PATH = PHASE2_DIR / "calibration_registry.json"
PHASE2_OOD_PATH = PHASE2_DIR / "ood_registry.json"
PHASE4_DIR = PROJECT_DIR / "results" / "phase4_retrieval"
PHASE4_RETRIEVAL_PATH = PHASE4_DIR / "retrieval_registry.json"
PHASE4_EMBEDDINGS_PATH = PHASE4_DIR / "retrieval_embeddings.npz"
PHASE4_THUMB_DIR = PHASE4_DIR / "thumbnails"
PHASE0_DIR = PROJECT_DIR / "results" / "phase0_registry"
PHASE0_THRESHOLD_PATH = PHASE0_DIR / "threshold_registry.json"
PHASE0_EXPERIMENT_PATH = PHASE0_DIR / "experiment_registry.json"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PHASE1_BANK_DIR.mkdir(parents=True, exist_ok=True)
PHASE2_DIR.mkdir(parents=True, exist_ok=True)
PHASE4_DIR.mkdir(parents=True, exist_ok=True)
PHASE4_THUMB_DIR.mkdir(parents=True, exist_ok=True)
PHASE0_DIR.mkdir(parents=True, exist_ok=True)

phase1_case_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class AppConfig:
    """Server configuration â€“ all tunables in one place."""
    DELETE_SLIDE_AFTER_ANALYSIS = False
    RESULT_RETENTION_MINUTES = 60
    MAX_UPLOAD_SIZE_GB = 5

    # Analysis
    MAX_TILES_FOR_ANALYSIS = 200
    TILE_SIZE = 256
    MIN_TISSUE_FRACTION = 0.3
    FEATURE_BATCH_SIZE = 32

    # Phase 1 safety thresholds
    ABSTAIN_CONFIDENCE_THRESHOLD = 0.62
    HIGH_UNCERTAINTY_THRESHOLD = 0.58
    MODERATE_UNCERTAINTY_THRESHOLD = 0.42
    LOW_MARGIN_THRESHOLD = 0.18
    MELANOMA_BORDERLINE_PROB = 0.20
    MELANOMA_HIGH_RISK_PROB = 0.35
    ENSEMBLE_DISAGREEMENT_THRESHOLD = 0.34

    # Phase 2 safety thresholds
    OOD_STRONG_THRESHOLD = 1.35
    OOD_MODERATE_THRESHOLD = 1.05
    UNIFIED_SAFETY_HIGH = 0.72
    UNIFIED_SAFETY_MODERATE = 0.48
    DEFAULT_TEMPERATURE = 1.0

    # Cost-aware retrieval
    RETRIEVAL_ACTIVE_METHOD = "trlq_quotient_v2"

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
# Model Registry â€” best version of each model
# ---------------------------------------------------------------------------
MODELS_DIR = Path("/mnt/d/skin_cancer_project/models")
RESULTS_BASE = PROJECT_DIR / "results"


def _windows_style_from_wsl_path(path_str: str):
    path_str = str(path_str)
    if path_str.startswith("/mnt/") and len(path_str) > 6:
        drive = path_str[5]
        rest = path_str[7:].replace("/", "\\")
        return f"{drive.upper()}:\\{rest}"
    return None


def _likely_unmounted_windows_drive(path_str: str) -> bool:
    path = str(path_str)
    if not path.startswith("/mnt/"):
        return False
    parts = Path(path).parts
    if len(parts) < 3:
        return False
    mount_root = Path(parts[0]) / parts[1] / parts[2]
    try:
        return mount_root.exists() and not any(mount_root.iterdir())
    except Exception:
        return False


def _weights_missing_message(model_name: str, path_str: str) -> str:
    windows_hint = _windows_style_from_wsl_path(path_str)
    message = f"Required weights for {model_name} were not found at {path_str}."
    if _likely_unmounted_windows_drive(path_str):
        mount_cmd = "sudo mount -t drvfs D: /mnt/d"
        message += f" The Windows D: drive appears to be unmounted in WSL. Mount it with `{mount_cmd}` and restart the server."
    elif windows_hint:
        message += f" Expected Windows-side location: `{windows_hint}`."
    return message

MODEL_REGISTRY = {
    # Phikon (pathology foundation encoder)
    "phikon_baseline": {
        "name": "Phikon", "group": "Phikon",
        "display": "Phikon - Baseline",
        "type": "phikon",
        "weights_path": str(MODELS_DIR / "pathology" / "phikon"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_phikon_v3_baseline" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.9184, "auc": 0.9795,
        "mel_fn": 2, "description": "Phikon encoder with baseline cross-entropy training. Macro F1 91.8%, Melanoma FN=2.",
    },
    "phikon_mel_boost_3x": {
        "name": "Phikon", "group": "Phikon",
        "display": "Phikon - Mel Boost 3x",
        "type": "phikon",
        "weights_path": str(MODELS_DIR / "pathology" / "phikon"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_phikon_v3_mel_boost_3x" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.9326, "auc": 0.9869,
        "mel_fn": 1, "description": "Phikon encoder with 3x melanoma class weighting. Melanoma recall 97.4%, Macro F1 93.3%, Melanoma FN=1.",
    },
    "phikon_mel_boost_5x": {
        "name": "Phikon", "group": "Phikon",
        "display": "Phikon - Mel Boost 5x",
        "type": "phikon",
        "weights_path": str(MODELS_DIR / "pathology" / "phikon"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_phikon_v3_mel_boost_5x" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.9404, "auc": 0.9872,
        "mel_fn": 3, "description": "Phikon encoder with 5x melanoma class weighting. Highest legacy single-model Macro F1 at 94.0%, Melanoma FN=3.",
    },
    "phikon_focal_g2": {
        "name": "Phikon", "group": "Phikon",
        "display": "Phikon - Focal G2",
        "type": "phikon",
        "weights_path": str(MODELS_DIR / "pathology" / "phikon"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_phikon_v3_focal_g2" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.9205, "auc": 0.9908,
        "mel_fn": 0, "description": "Phikon encoder with focal loss gamma=2 and melanoma up-weighting. Melanoma FN=0, melanoma recall 100.0%, AUC 99.1%.",
    },
    "phikon_cost_sensitive": {
        "name": "Phikon", "group": "Phikon",
        "display": "Phikon - Cost-Sensitive",
        "type": "phikon",
        "weights_path": str(MODELS_DIR / "pathology" / "phikon"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_phikon_v3_cost_sensitive" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.9429, "auc": 0.9905,
        "mel_fn": 1, "description": "Phikon encoder with cost-sensitive loss that penalizes melanoma misses more heavily. Accuracy 94.3%, Macro F1 94.3%, AUC 99.1%, Melanoma FN=1.",
    },
    "phikon_cost_sensitive_strong": {
        "name": "Phikon", "group": "Phikon",
        "display": "Phikon - Cost-Sensitive Strong",
        "type": "phikon",
        "weights_path": str(MODELS_DIR / "pathology" / "phikon"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_phikon_v3_fast_cost_sensitive_strong" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.9424, "auc": 0.9938,
        "mel_fn": 3, "description": "Phikon encoder with stronger melanoma-miss penalty. Best fast-run Phikon shortlist model with Macro F1 94.2% and Melanoma FN=3.",
    },
    # UNI / CONCH (pathology foundation encoders)
    "uni_cost_sensitive_strong": {
        "name": "UNI", "group": "UNI",
        "display": "UNI - Cost-Sensitive Strong",
        "type": "uni",
        "weights_path": str(MODELS_DIR / "pathology" / "uni" / "pytorch_model.bin"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_uni_v3_fast_cost_sensitive_strong" / "best_model.pt"),
        "feat_dim": 1024, "f1": 0.9541, "auc": 0.9957,
        "mel_fn": 1, "description": "UNI encoder with strong cost-sensitive melanoma penalty. Current best overall run: Macro F1 95.4%, AUC 99.6%, Melanoma FN=1.",
    },
    "uni_focal_g3": {
        "name": "UNI", "group": "UNI",
        "display": "UNI - Focal G3",
        "type": "uni",
        "weights_path": str(MODELS_DIR / "pathology" / "uni" / "pytorch_model.bin"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_uni_v3_fast_focal_g3" / "best_model.pt"),
        "feat_dim": 1024, "f1": 0.9514, "auc": 0.9958,
        "mel_fn": 3, "description": "UNI encoder with focal loss gamma=3 and 5x melanoma weighting. Alternate UNI shortlist run with Macro F1 95.1% and Melanoma FN=3.",
    },
    "conch_cost_sensitive_strong": {
        "name": "CONCH", "group": "CONCH",
        "display": "CONCH - Cost-Sensitive Strong",
        "type": "conch",
        "weights_path": str(MODELS_DIR / "pathology" / "conch" / "pytorch_model.bin"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_conch_v3_fast_cost_sensitive_strong" / "best_model.pt"),
        "feat_dim": 512, "f1": 0.9323, "auc": 0.9881,
        "mel_fn": 4, "description": "CONCH encoder with strong cost-sensitive melanoma penalty. Best CONCH run with Macro F1 93.2% and Melanoma FN=4.",
    },
    # ConvNeXt-Base
    "convnext_base_mel_boost_3x": {
        "name": "ConvNeXt-Base", "group": "ConvNeXt-Base",
        "display": "ConvNeXt-Base - Mel Boost 3x",
        "type": "torchvision", "loader": "convnext_base",
        "weights_path": str(MODELS_DIR / "torchvision" / "convnext_base.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_convnext_base_v3_mel_boost_3x" / "best_model.pt"),
        "feat_dim": 1024, "f1": 0.8773, "auc": 0.9666,
        "mel_fn": 3, "description": "ConvNeXt-Base encoder with 3x melanoma class weighting. Best ConvNeXt-Base run with Macro F1 87.7% and Melanoma FN=3.",
    },
    "convnext_base_focal_g2": {
        "name": "ConvNeXt-Base", "group": "ConvNeXt-Base",
        "display": "ConvNeXt-Base - Focal G2",
        "type": "torchvision", "loader": "convnext_base",
        "weights_path": str(MODELS_DIR / "torchvision" / "convnext_base.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_convnext_base_v3_focal_g2" / "best_model.pt"),
        "feat_dim": 1024, "f1": 0.8514, "auc": 0.9668,
        "mel_fn": 1, "description": "ConvNeXt-Base encoder with focal loss gamma=2. Macro F1 85.1%, Melanoma FN=1.",
    },
    # ConvNeXt-Small
    "convnext_small_mel_boost_3x": {
        "name": "ConvNeXt-Small", "group": "ConvNeXt-Small",
        "display": "ConvNeXt-Small - Mel Boost 3x",
        "type": "torchvision", "loader": "convnext_small",
        "weights_path": str(MODELS_DIR / "torchvision" / "convnext_small.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_convnext_small_v3_mel_boost_3x" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.8632, "auc": 0.9563,
        "mel_fn": 2, "description": "ConvNeXt-Small encoder with 3x melanoma class weighting. Best ConvNeXt-Small run with Macro F1 86.3% and Melanoma FN=2.",
    },
    "convnext_small_focal_g2": {
        "name": "ConvNeXt-Small", "group": "ConvNeXt-Small",
        "display": "ConvNeXt-Small - Focal G2",
        "type": "torchvision", "loader": "convnext_small",
        "weights_path": str(MODELS_DIR / "torchvision" / "convnext_small.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_convnext_small_v3_focal_g2" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.8495, "auc": 0.9638,
        "mel_fn": 6, "description": "ConvNeXt-Small encoder with focal loss gamma=2. Macro F1 85.0%, Melanoma FN=6.",
    },
    # DINOv2-base
    "dinov2_base_focal_g2": {
        "name": "DINOv2-base", "group": "DINOv2",
        "display": "DINOv2-Base - Focal G2",
        "type": "dinov2",
        "weights_path": str(MODELS_DIR / "vision" / "dinov2-base"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_dinov2_base_v3_focal_g2" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.8535, "auc": 0.9643,
        "mel_fn": 3, "description": "DINOv2-base encoder with focal loss gamma=2. Best DINOv2 run with Macro F1 85.3% and Melanoma FN=3.",
    },
    "dinov2_base_mel_boost_5x": {
        "name": "DINOv2-base", "group": "DINOv2",
        "display": "DINOv2-Base - Mel Boost 5x",
        "type": "dinov2",
        "weights_path": str(MODELS_DIR / "vision" / "dinov2-base"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_dinov2_base_v3_mel_boost_5x" / "best_model.pt"),
        "feat_dim": 768, "f1": 0.8319, "auc": 0.9557,
        "mel_fn": 8, "description": "DINOv2-base encoder with 5x melanoma class weighting. Macro F1 83.2%, Melanoma FN=8.",
    },
    # ResNet50
    "resnet50_focal_g2": {
        "name": "ResNet50", "group": "ResNet",
        "display": "ResNet50 - Focal G2",
        "type": "torchvision", "loader": "resnet50",
        "weights_path": str(MODELS_DIR / "torchvision" / "resnet50.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_resnet50_v3_focal_g2" / "best_model.pt"),
        "feat_dim": 2048, "f1": 0.8345, "auc": 0.9687,
        "mel_fn": 3, "description": "ResNet50 encoder with focal loss gamma=2. Macro F1 83.5%, Melanoma FN=3.",
    },
    # ResNet18
    "resnet18_focal_g2": {
        "name": "ResNet18", "group": "ResNet",
        "display": "ResNet18 - Focal G2",
        "type": "torchvision", "loader": "resnet18",
        "weights_path": str(MODELS_DIR / "torchvision" / "resnet18.pth"),
        "mil_checkpoint": str(RESULTS_BASE / "mil_4class_resnet18_v3_focal_g2" / "best_model.pt"),
        "feat_dim": 512, "f1": 0.8412, "auc": 0.9588,
        "mel_fn": 6, "description": "ResNet18 encoder with focal loss gamma=2. Lightweight backbone run with Macro F1 84.1% and Melanoma FN=6.",
    },
}

# Ensemble presets from exhaustive search (MelFN=0 validated!)
ENSEMBLE_PRESETS = {
    "ensemble_2_best": {
        "name": "Ensemble-2 (Best Pathology Pair)",
        "display": "Ensemble 2-Model (UNI + Phikon)",
        "description": "Average-probability ensemble of UNI - Cost-Sensitive Strong and Phikon - Cost-Sensitive Strong. Chosen as the strongest two-model pathology pair from the completed runs.",
        "models": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong"],
        "f1": 0.948, "auc": 0.995,
    },
    "ensemble_3_best": {
        "name": "Ensemble-3 (Best Pathology Trio)",
        "display": "Ensemble 3-Model (UNI + Phikon + CONCH)",
        "description": "Average-probability ensemble of UNI - Cost-Sensitive Strong, Phikon - Cost-Sensitive Strong, and CONCH - Cost-Sensitive Strong. This is the current default pathology trio.",
        "models": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"],
        "f1": 0.943, "auc": 0.993,
    },
    "ensemble_3": {
        "name": "Ensemble-3 (MelFN=0)",
        "display": "Ensemble 3-Model (Legacy Best)",
        "description": "Legacy validated ensemble of Phikon - Cost-Sensitive, Phikon - Mel Boost 5x, and ResNet50 - Focal G2. Historical reference run with Melanoma FN=0 validation behavior.",
        "models": ["phikon_cost_sensitive", "phikon_mel_boost_5x", "resnet50_focal_g2"],
        "f1": 0.961, "auc": 0.988,
    },
    "ensemble_4": {
        "name": "Ensemble-4 (Multi-backbone)",
        "display": "Ensemble 4-Model (Legacy Multi-backbone)",
        "description": "Legacy multi-backbone ensemble of ConvNeXt-Base - Focal G2, DINOv2-Base - Focal G2, Phikon - Cost-Sensitive, and Phikon - Mel Boost 5x.",
        "models": ["convnext_base_focal_g2", "dinov2_base_focal_g2", "phikon_cost_sensitive", "phikon_mel_boost_5x"],
        "f1": 0.961, "auc": 0.987,
    },
    "ensemble_5": {
        "name": "Ensemble-5 (Maximum)",
        "display": "Ensemble 5-Model (Legacy Maximum)",
        "description": "Legacy five-model ensemble of ConvNeXt-Small - Focal G2, Phikon - Cost-Sensitive, Phikon - Mel Boost 3x, Phikon - Mel Boost 5x, and ResNet50 - Focal G2. Highest historical ensemble AUC in the older search space.",
        "models": ["convnext_small_focal_g2", "phikon_cost_sensitive", "phikon_mel_boost_3x", "phikon_mel_boost_5x", "resnet50_focal_g2"],
        "f1": 0.961, "auc": 0.989,
    },
}

# Default ensemble
ENSEMBLE_MODELS = ENSEMBLE_PRESETS["ensemble_3_best"]["models"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("skinsight")

if not MODELS_DIR.exists():
    logger.warning(
        "Model directory %s is not visible inside WSL. If your models are on Windows D:, mount it with `sudo mount -t drvfs D: /mnt/d` before starting the server.",
        MODELS_DIR,
    )

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
_phase2_registry_cache = {}
_phase4_registry_cache = {}
_phase0_registry_cache = {}
_retrieval_signal_cache = {}


def _ensure_torch():
    global _torch, _device
    if _torch is None:
        import torch
        _torch = torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch loaded â€“ device: {_device}")
    return _torch, _device


def _ensure_openslide():
    global _openslide
    if _openslide is None:
        import openslide
        _openslide = openslide
        logger.info("OpenSlide loaded")
    return _openslide


# ---------------------------------------------------------------------------
# GatedAttentionMIL â€” matches train_all_models.py architecture exactly
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
            logits = self.classifier(z)
            return logits, a.squeeze(), z.squeeze(0), h

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

    if not Path(wpath).exists():
        message = _weights_missing_message(mcfg["name"], wpath)
        logger.error(message)
        raise FileNotFoundError(message)

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
                nn.Flatten(1),           # no Linear â†’ raw features
            )

    elif mtype == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(wpath, local_files_only=True)

    elif mtype == "phikon":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(wpath, local_files_only=True)

    elif mtype == "uni":
        import timm
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        try:
            state_dict = torch.load(wpath, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(wpath, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    elif mtype == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        model, transform = create_model_from_pretrained("conch_ViT-B-16", wpath)

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
        is_ensemble = model_key.startswith("ensemble")
        if is_ensemble:
            preset_key = model_key if model_key in ENSEMBLE_PRESETS else "ensemble_3"
            ensemble_cfg = ENSEMBLE_PRESETS[preset_key]
            ENSEMBLE_MODELS_RUN = ensemble_cfg["models"]
            model_display = ensemble_cfg["name"]
        else:
            model_display = MODEL_REGISTRY[model_key]["display"]

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
            all_raw_probs = []
            all_attns = []
            all_preds = []
            all_bag_embeddings = []
            all_calibrations = []
            all_contrastive_views = []
            model_results = []

            for i, mkey in enumerate(ENSEMBLE_MODELS_RUN):
                pct = 30 + int(50 * i / len(ENSEMBLE_MODELS_RUN))
                mname = MODEL_REGISTRY[mkey]["display"]
                _update_job(job_id, progress=pct,
                            message=f"Ensemble: running {mname} ({i+1}/{len(ENSEMBLE_MODELS_RUN)})...")

                features = _extract_features(tiles, mkey)
                pred, probs, attn, bag_embedding, raw_probs, contrastive_views = _run_mil_inference(features, mkey)
                all_probs.append(probs)
                all_raw_probs.append(raw_probs)
                all_attns.append(attn)
                all_preds.append(pred)
                all_bag_embeddings.append(bag_embedding)
                all_contrastive_views.append(contrastive_views)
                calibration_meta = _get_calibration_entry(mkey) or {}
                all_calibrations.append({
                    "model_key": mkey,
                    "model_display": MODEL_REGISTRY[mkey]["display"],
                    "available": bool(calibration_meta),
                    "temperature": round(float(calibration_meta.get("temperature", cfg.DEFAULT_TEMPERATURE)), 4),
                    "ece_before": calibration_meta.get("ece_before"),
                    "ece_after": calibration_meta.get("ece_after"),
                    "mce_before": calibration_meta.get("mce_before"),
                    "mce_after": calibration_meta.get("mce_after"),
                })
                model_results.append({
                    "model": mname,
                    "prediction": CLASS_NAMES[pred],
                    "probabilities": {CLASS_NAMES[c]: round(float(probs[c]), 4) for c in range(N_CLASSES)},
                })

            # Average probabilities
            avg_probs = np.mean(all_probs, axis=0)
            avg_raw_probs = np.mean(all_raw_probs, axis=0)
            prediction = int(avg_probs.argmax())
            probabilities = avg_probs
            attention_views = _build_ensemble_attention_views(all_attns)
            attention_views.update(_aggregate_attention_view_dicts(all_contrastive_views))
            attention_weights = attention_views.get("consensus", np.mean(all_attns, axis=0))
            safety = _build_phase1_safety(prediction, probabilities, ensemble_predictions=all_preds)
            safety = _merge_phase2_ensemble_safety(
                safety,
                probabilities,
                ENSEMBLE_MODELS_RUN,
                all_bag_embeddings,
                all_calibrations,
                raw_probabilities=avg_raw_probs,
            )
            safety = _annotate_threshold_policy(safety, model_key)
            retrieval = _retrieve_similar_cases(
                model_key,
                ensemble_model_keys=ENSEMBLE_MODELS_RUN,
                ensemble_bag_embeddings=all_bag_embeddings,
                probabilities=probabilities,
                safety=safety,
                query_slide_id=Path(slide_path).stem,
            )

        else:
            # Single model
            _update_job(job_id, progress=40,
                        message=f"Extracting features with {model_display}...")
            features = _extract_features(tiles, model_key)

            _update_job(job_id, progress=65, message="Running MIL inference...")
            prediction, probabilities, attention_weights, bag_embedding, raw_probabilities, contrastive_views = _run_mil_inference(features, model_key)
            model_results = None
            attention_views = {"attention": _normalize_attention_weights(attention_weights)}
            attention_views.update(contrastive_views)
            safety = _build_phase1_safety(prediction, probabilities)
            safety = _merge_phase2_safety(
                safety,
                probabilities,
                model_key,
                bag_embedding,
                _get_calibration_entry(model_key) or {},
                raw_probabilities=raw_probabilities,
            )
            safety = _annotate_threshold_policy(safety, model_key)
            retrieval = _retrieve_similar_cases(
                model_key,
                bag_embedding=bag_embedding,
                probabilities=probabilities,
                safety=safety,
                query_slide_id=Path(slide_path).stem,
            )

        _update_job(job_id, progress=80, message="Generating heatmap...")

        # Step 4: Generate heatmap
        heatmap_ok = _generate_heatmap(slide, tile_coords, attention_weights,
                                       probabilities, job_id, variant="attention")
        if is_ensemble:
            _generate_heatmap(slide, tile_coords, attention_views["consensus"], probabilities, job_id, variant="consensus")
            _generate_heatmap(slide, tile_coords, attention_views["disagreement"], probabilities, job_id, variant="disagreement")
            _generate_heatmap(slide, tile_coords, attention_views["shared"], probabilities, job_id, variant="shared")
        for contrastive_key in [k for k in attention_views.keys() if k.startswith("contrast_")]:
            _generate_heatmap(slide, tile_coords, attention_views[contrastive_key], probabilities, job_id, variant=contrastive_key)

        # Step 5: Top attention tiles
        top_tiles = _get_top_attention_tiles(tiles, tile_coords, attention_weights, job_id)
        if is_ensemble:
            top_tiles = _annotate_top_tiles(
                _get_top_attention_tiles(tiles, tile_coords, attention_views["shared"], job_id),
                {
                    "consensus_score": attention_views["consensus"],
                    "disagreement_score": attention_views["disagreement"],
                    "shared_score": attention_views["shared"],
                },
            )

        slide.close()

        result = {
            "prediction": safety["display_prediction"],
            "raw_prediction": CLASS_NAMES[prediction],
            "prediction_key": safety["prediction_key"],
            "decision_status": safety["decision_status"],
            "prediction_id": int(prediction),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]), 4)
                for i in range(N_CLASSES)
            },
            "safety": safety,
            "threshold_policy": safety.get("threshold_policy") or _build_threshold_policy(model_key),
            "retrieval": retrieval,
            "n_tiles": len(tiles),
            "top_tiles": top_tiles,
            "top_tiles_title": "Areas of Strongest Shared Attention" if is_ensemble else "Top Attention Tiles",
            "top_tiles_mode": "shared_consensus" if is_ensemble else "single_attention",
            "heatmap_available": heatmap_ok is not None,
            "heatmap_views": _build_heatmap_view_list([
                key for key in (
                    (["consensus", "disagreement", "shared"] if is_ensemble else ["attention"]) +
                    [k for k in attention_views.keys() if k.startswith("contrast_")]
                )
                if key in attention_views or key in ("attention", "consensus", "disagreement", "shared")
            ]),
            "default_heatmap_view": "consensus" if is_ensemble else "attention",
            "model_used": model_display,
            "model_key": model_key,
            "timestamp": datetime.now().isoformat(),
        }
        # For ensemble, include individual model predictions
        if model_results:
            result["ensemble_details"] = model_results

        result["calculation_details"] = _build_result_calculation_details(result, slide_info)
        result["artifacts"] = _build_result_artifacts(job_id, result)

        _record_phase1_inference_case(job_id, slide_path, model_key, model_display, slide_info, result)

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Tile Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Feature Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            elif mtype == "conch":
                feats = encoder.encode_image(tensors, proj_contrast=False, normalize=False)
            else:
                feats = encoder(tensors)
        features.append(feats.cpu())

    return torch.cat(features, dim=0)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ MIL Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _run_mil_inference(features, model_key):
    """Run MIL model on extracted features."""
    torch, device = _ensure_torch()
    model = _get_mil_model(model_key)

    if model is None:
        logger.warning(f"MIL model not available for {model_key}, returning uniform")
        n = features.shape[0]
        return 0, np.ones(N_CLASSES) / N_CLASSES, np.ones(n) / n, np.zeros(256, dtype=np.float32), np.ones(N_CLASSES) / N_CLASSES, {}

    features = features.to(device)
    with torch.no_grad():
        logits, attn, bag_embedding, tile_hidden = model(features)
        tile_logits = model.classifier(tile_hidden)
        raw_probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        probs, _ = _apply_probability_calibration(raw_probs, model_key)
        pred = int(np.argmax(probs))
        attn_np = attn.cpu().numpy().flatten()
        bag_np = bag_embedding.detach().cpu().numpy().astype(np.float32)
        contrastive_views = _build_contrastive_attention_views(
            attn_np,
            tile_logits.detach().cpu().numpy().astype(np.float32),
        )

    return pred, probs, attn_np, bag_np, raw_probs, contrastive_views


def _normalized_entropy(probabilities):
    probs = np.asarray(probabilities, dtype=np.float32)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum() / np.log(len(probs)))


def _normalize_attention_weights(attention_weights):
    attn = np.asarray(attention_weights, dtype=np.float32).reshape(-1)
    if attn.size == 0:
        return attn
    amin = float(attn.min())
    amax = float(attn.max())
    if amax > amin:
        return (attn - amin) / (amax - amin)
    return np.zeros_like(attn, dtype=np.float32)


def _build_ensemble_attention_views(attention_list):
    if not attention_list:
        return {}

    normalized = [_normalize_attention_weights(attn) for attn in attention_list]
    stack = np.stack(normalized, axis=0)
    consensus = stack.mean(axis=0)
    disagreement = stack.std(axis=0)
    shared = consensus * (1.0 - np.clip(disagreement, 0.0, 1.0))

    return {
        "consensus": _normalize_attention_weights(consensus),
        "disagreement": _normalize_attention_weights(disagreement),
        "shared": _normalize_attention_weights(shared),
    }


CONTRASTIVE_CLASS_PAIRS = [
    ("melanoma", "scc"),
    ("melanoma", "bcc"),
]


HEATMAP_VIEW_METADATA = {
    "attention": {
        "label": "Attention",
        "description": "Single-model MIL attention heatmap.",
    },
    "consensus": {
        "label": "Consensus",
        "description": "Regions jointly emphasized by ensemble members.",
    },
    "disagreement": {
        "label": "Disagreement",
        "description": "Regions where ensemble attention diverges.",
    },
    "shared": {
        "label": "Shared Focus",
        "description": "Consensus weighted by low disagreement.",
    },
    "contrast_melanoma_vs_scc": {
        "label": "Mel vs SCC",
        "description": "Class-contrastive heatmap showing evidence for melanoma relative to SCC.",
    },
    "contrast_melanoma_vs_bcc": {
        "label": "Mel vs BCC",
        "description": "Class-contrastive heatmap showing evidence for melanoma relative to BCC.",
    },
}


def _build_contrastive_attention_views(tile_attention, tile_class_scores):
    attention = _normalize_attention_weights(tile_attention)
    class_scores = np.asarray(tile_class_scores, dtype=np.float32)
    if class_scores.ndim != 2 or class_scores.shape[1] != N_CLASSES:
        return {}

    views = {}
    for pos_key, neg_key in CONTRASTIVE_CLASS_PAIRS:
        pos_idx = CLASS_KEYS.index(pos_key)
        neg_idx = CLASS_KEYS.index(neg_key)
        pos_view = _normalize_attention_weights(class_scores[:, pos_idx])
        contrast_view = _normalize_attention_weights(class_scores[:, pos_idx] - class_scores[:, neg_idx])
        combined = _normalize_attention_weights(0.25 * attention + 0.25 * pos_view + 0.50 * contrast_view)
        views[f"contrast_{pos_key}_vs_{neg_key}"] = combined
    return views


def _aggregate_attention_view_dicts(view_dicts):
    if not view_dicts:
        return {}
    out = {}
    all_keys = sorted({k for view_dict in view_dicts for k in view_dict.keys()})
    for key in all_keys:
        arrays = [np.asarray(view_dict[key], dtype=np.float32) for view_dict in view_dicts if key in view_dict]
        if not arrays:
            continue
        out[key] = _normalize_attention_weights(np.mean(np.stack(arrays, axis=0), axis=0))
    return out


def _build_heatmap_view_list(view_keys):
    views = []
    for key in view_keys:
        meta = HEATMAP_VIEW_METADATA.get(key, {})
        views.append({
            "key": key,
            "label": meta.get("label", key),
            "description": meta.get("description", key),
        })
    return views


def _load_phase2_registry(path):
    cache_key = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _phase2_registry_cache.pop(cache_key, None)
        return {}

    cached = _phase2_registry_cache.get(cache_key)
    if cached and cached["mtime"] == mtime:
        return cached["data"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load Phase 2 registry: %s", path)
        data = {}

    _phase2_registry_cache[cache_key] = {"mtime": mtime, "data": data}
    return data


def _load_phase4_registry():
    cache_key = f"{PHASE4_RETRIEVAL_PATH}|{PHASE4_EMBEDDINGS_PATH}"
    try:
        mtimes = (
            PHASE4_RETRIEVAL_PATH.stat().st_mtime,
            PHASE4_EMBEDDINGS_PATH.stat().st_mtime,
        )
    except FileNotFoundError:
        _phase4_registry_cache.pop(cache_key, None)
        return {}, {}

    cached = _phase4_registry_cache.get(cache_key)
    if cached and cached["mtimes"] == mtimes:
        return cached["registry"], cached["arrays"]

    try:
        registry = json.loads(PHASE4_RETRIEVAL_PATH.read_text(encoding="utf-8"))
        with np.load(PHASE4_EMBEDDINGS_PATH, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except Exception:
        logger.exception("Failed to load Phase 4 retrieval artifacts")
        registry, arrays = {}, {}

    _phase4_registry_cache[cache_key] = {
        "mtimes": mtimes,
        "registry": registry,
        "arrays": arrays,
    }
    return registry, arrays


def _load_phase0_registry(path):
    cache_key = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _phase0_registry_cache.pop(cache_key, None)
        return {}

    cached = _phase0_registry_cache.get(cache_key)
    if cached and cached["mtime"] == mtime:
        return cached["data"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load Phase 0 registry: %s", path)
        data = {}

    _phase0_registry_cache[cache_key] = {"mtime": mtime, "data": data}
    return data


def _get_threshold_registry():
    return _load_phase0_registry(PHASE0_THRESHOLD_PATH)


def _threshold_entry_from_registry(registry, model_key):
    if not isinstance(registry, dict):
        return {}
    if model_key in registry:
        return registry.get(model_key) or {}
    for bucket_name in ("models", "ensembles", "thresholds", "entries"):
        bucket = registry.get(bucket_name)
        if isinstance(bucket, dict) and model_key in bucket:
            return bucket.get(model_key) or {}
    return {}


def _coerce_threshold_value(entry):
    for key in (
        "melanoma_safe_threshold",
        "best_melanoma_threshold",
        "selected_threshold",
        "melanoma_threshold",
        "threshold",
    ):
        value = entry.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _build_threshold_policy(model_key):
    registry = _get_threshold_registry()
    entry = _threshold_entry_from_registry(registry, model_key)
    threshold_value = _coerce_threshold_value(entry) if entry else None

    policy = {
        "available": bool(entry),
        "model_key": model_key,
        "default_threshold": float(entry.get("default_threshold", 0.5)) if entry else 0.5,
        "melanoma_safe_threshold": threshold_value,
        "selection_basis": entry.get("selection_basis") or entry.get("policy") or entry.get("threshold_basis"),
        "source_run": entry.get("source_run"),
        "source_csv": entry.get("source_csv"),
        "source_results": entry.get("source_results"),
        "evaluation_split": entry.get("evaluation_split"),
        "notes": entry.get("notes"),
    }
    if model_key in ENSEMBLE_PRESETS:
        policy["components"] = ENSEMBLE_PRESETS[model_key]["models"]
    label_parts = []
    if threshold_value is not None:
        label_parts.append(f"Mel review {threshold_value:.2f}")
    if policy["selection_basis"]:
        label_parts.append(str(policy["selection_basis"]))
    policy["label"] = " | ".join(label_parts) if label_parts else "Default threshold only"
    return policy


def _annotate_threshold_policy(safety, model_key):
    policy = _build_threshold_policy(model_key)
    safety["threshold_policy"] = policy
    threshold_value = policy.get("melanoma_safe_threshold")
    details = dict(safety.get("details") or {})
    details["threshold_policy"] = {
        "title": "Melanoma review threshold policy",
        "summary": "A calibrated melanoma review threshold is applied after the raw model prediction so borderline melanoma evidence is not hidden by a different top class.",
        "clinical_context": [
            "This policy is designed around melanoma false-negative reduction: if melanoma probability is clinically non-trivial, the app should prefer review over silent BCC/SCC/benign finalization.",
            "The threshold does not diagnose melanoma by itself. It changes the workflow status by surfacing cases that deserve expert review."
        ],
        "technical_context": [
            "The threshold value is loaded from the phase-0 threshold registry when available for the selected model or ensemble.",
            "The rule is evaluated on the calibrated model probability P(Melanoma), not on the heatmap intensity or retrieval score."
        ],
        "formulae": [
            "threshold_triggered = P(Melanoma) >= melanoma_safe_threshold",
            "review_signal = threshold_triggered and raw_prediction != Melanoma",
        ],
        "inputs": {
            "raw_prediction": safety.get("raw_prediction"),
            "melanoma_probability": round(float(safety.get("melanoma_probability", 0.0)), 4),
            "melanoma_safe_threshold": threshold_value,
            "policy_available": bool(policy.get("available")),
            "selection_basis": policy.get("selection_basis"),
            "source_run": policy.get("source_run"),
            "evaluation_split": policy.get("evaluation_split"),
        },
        "replication_steps": [
            "Load the selected model's threshold registry entry.",
            "Read melanoma_probability from the calibrated slide-level probability vector.",
            "Compare melanoma_probability against melanoma_safe_threshold.",
            "If the threshold is crossed while the raw class is not Melanoma, mark a review signal."
        ],
    }
    safety["details"] = details
    if threshold_value is None:
        return safety

    melanoma_probability = float(safety.get("melanoma_probability", 0.0))
    threshold_triggered = melanoma_probability >= float(threshold_value)
    safety["threshold_policy"]["threshold_triggered"] = threshold_triggered
    safety["details"]["threshold_policy"]["inputs"]["threshold_triggered"] = threshold_triggered
    if threshold_triggered and safety.get("raw_prediction") != "Melanoma":
        reasons = list(safety.get("reasons", []))
        trigger_reason = "Melanoma probability crossed the tuned review threshold"
        if trigger_reason not in reasons:
            reasons.append(trigger_reason)
        safety["reasons"] = reasons
        safety["threshold_policy"]["review_signal"] = True
        safety["details"]["threshold_policy"]["inputs"]["review_signal"] = True
    else:
        safety["threshold_policy"]["review_signal"] = False
        safety["details"]["threshold_policy"]["inputs"]["review_signal"] = False
    return safety


def _normalize_embedding(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _softmax_from_scaled_probs(probabilities, temperature):
    probs = np.asarray(probabilities, dtype=np.float32)
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs)
    scaled = logits / max(float(temperature), 1e-4)
    scaled -= scaled.max()
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum()


def _get_calibration_entry(model_key):
    registry = _load_phase2_registry(PHASE2_CALIBRATION_PATH)
    return registry.get(model_key)


def _apply_probability_calibration(probabilities, model_key):
    entry = _get_calibration_entry(model_key) or {}
    temperature = float(entry.get("temperature", cfg.DEFAULT_TEMPERATURE))
    calibrated = _softmax_from_scaled_probs(probabilities, temperature)
    meta = {
        "available": bool(entry),
        "method": entry.get("method", "temperature_scaling" if entry else "none"),
        "temperature": round(temperature, 4),
        "ece_before": entry.get("ece_before"),
        "ece_after": entry.get("ece_after"),
        "mce_before": entry.get("mce_before"),
        "mce_after": entry.get("mce_after"),
        "source_run": entry.get("source_run"),
        "source_csv": entry.get("source_csv"),
    }
    return calibrated, meta


def _get_ood_entry(model_key):
    registry = _load_phase2_registry(PHASE2_OOD_PATH)
    return registry.get(model_key)


def _estimate_ood_from_embedding(bag_embedding, model_key):
    entry = _get_ood_entry(model_key) or {}
    if bag_embedding is None or not entry:
        return {
            "available": False,
            "ood_score": None,
            "ood_flag": False,
            "ood_level": "unavailable",
            "nearest_class": None,
            "nearest_distance": None,
            "normalized_distance": None,
            "id_support_score": None,
        }

    embedding = np.asarray(bag_embedding, dtype=np.float32)
    centroids = entry.get("class_centroids", {})
    thresholds = entry.get("class_thresholds", {})
    if not centroids:
        return {
            "available": False,
            "ood_score": None,
            "ood_flag": False,
            "ood_level": "unavailable",
            "nearest_class": None,
            "nearest_distance": None,
            "normalized_distance": None,
            "id_support_score": None,
        }

    distances = {}
    normalized = {}
    for class_name, centroid in centroids.items():
        centroid_np = np.asarray(centroid, dtype=np.float32)
        dist = float(np.linalg.norm(embedding - centroid_np))
        threshold = max(float(thresholds.get(class_name, 1.0)), 1e-6)
        distances[class_name] = dist
        normalized[class_name] = dist / threshold

    nearest_class = min(normalized, key=normalized.get)
    nearest_distance = distances[nearest_class]
    normalized_distance = float(normalized[nearest_class])
    ood_score = float(np.clip((normalized_distance - 1.0) / 0.8, 0.0, 1.0))
    id_support = float(np.clip(1.0 - ood_score, 0.0, 1.0))

    if normalized_distance >= cfg.OOD_STRONG_THRESHOLD:
        ood_level = "strong"
        ood_flag = True
    elif normalized_distance >= cfg.OOD_MODERATE_THRESHOLD:
        ood_level = "moderate"
        ood_flag = False
    else:
        ood_level = "low"
        ood_flag = False

    return {
        "available": True,
        "ood_score": round(ood_score, 4),
        "ood_flag": ood_flag,
        "ood_level": ood_level,
        "nearest_class": nearest_class,
        "nearest_distance": round(nearest_distance, 4),
        "normalized_distance": round(normalized_distance, 4),
        "id_support_score": round(id_support, 4),
    }


def _merge_phase2_safety(phase1_safety, probabilities, model_key, bag_embedding, calibration_meta, raw_probabilities=None):
    safety = dict(phase1_safety)
    probs = np.asarray(probabilities, dtype=np.float32)
    raw_probs = np.asarray(raw_probabilities, dtype=np.float32) if raw_probabilities is not None else probs
    ood = _estimate_ood_from_embedding(bag_embedding, model_key)
    disagreement = safety.get("ensemble_disagreement")

    components = [
        float(safety.get("uncertainty", 0.0)),
        float(np.clip(1.0 - safety.get("margin", 0.0), 0.0, 1.0)),
        float(ood["ood_score"]) if ood["ood_score"] is not None else 0.0,
    ]
    if disagreement is not None:
        components.append(float(disagreement))
    unified_safety_score = float(np.mean(components))
    component_details = [
        {
            "name": "uncertainty",
            "value": round(float(safety.get("uncertainty", 0.0)), 4),
            "meaning": "Normalized entropy of the calibrated class probabilities.",
        },
        {
            "name": "inverse_margin",
            "value": round(float(np.clip(1.0 - safety.get("margin", 0.0), 0.0, 1.0)), 4),
            "meaning": "1 - margin; higher value means top classes are closer.",
        },
        {
            "name": "ood_score",
            "value": round(float(ood["ood_score"]), 4) if ood["ood_score"] is not None else 0.0,
            "meaning": "Out-of-distribution shift score estimated from class-centroid distance.",
        },
    ]
    if disagreement is not None:
        component_details.append({
            "name": "ensemble_disagreement",
            "value": round(float(disagreement), 4),
            "meaning": "Fraction of ensemble votes outside the majority class.",
        })

    reasons = list(safety.get("reasons", []))
    if ood["available"] and ood["ood_level"] == "moderate":
        reasons.append(f"Moderate OOD shift toward {ood['nearest_class']}")
    if ood["ood_flag"]:
        reasons.append(f"Strong OOD signal beyond {ood['nearest_class']}")

    if ood["ood_flag"]:
        safety["display_prediction"] = "Needs Expert Review"
        safety["decision_status"] = "abstain"
        safety["prediction_key"] = "abstain"
        safety["abstain_recommended"] = True
        safety["risk_level"] = "urgent review recommended"
        safety["recommendation"] = "Potential out-of-distribution case detected; defer to expert review."
    elif unified_safety_score >= cfg.UNIFIED_SAFETY_HIGH and safety.get("risk_level") != "urgent review recommended":
        safety["risk_level"] = "high risk"
    elif unified_safety_score >= cfg.UNIFIED_SAFETY_MODERATE and safety.get("risk_level") == "low risk":
        safety["risk_level"] = "moderate risk"

    safety["reasons"] = reasons
    safety["phase"] = "phase2"
    safety["raw_probabilities"] = {
        CLASS_NAMES[i]: round(float(raw_probs[i]), 4)
        for i in range(N_CLASSES)
    }
    safety["calibration"] = {
        "available": bool(calibration_meta),
        "method": calibration_meta.get("method", "temperature_scaling" if calibration_meta else "none"),
        "temperature": round(float(calibration_meta.get("temperature", cfg.DEFAULT_TEMPERATURE)), 4),
        "ece_before": calibration_meta.get("ece_before"),
        "ece_after": calibration_meta.get("ece_after"),
        "mce_before": calibration_meta.get("mce_before"),
        "mce_after": calibration_meta.get("mce_after"),
        "source_run": calibration_meta.get("source_run"),
        "source_csv": calibration_meta.get("source_csv"),
    }
    safety["ood"] = ood
    safety["unified_safety_score"] = round(unified_safety_score, 4)
    safety["safety_score"] = round(unified_safety_score, 4)
    safety["id_support_score"] = ood["id_support_score"]
    details = dict(safety.get("details") or {})
    details["phase2"] = {
        "title": "Phase 2 unified safety calculation",
        "summary": "Phase 2 combines calibrated uncertainty, class-margin risk, and OOD evidence into a single review-oriented safety score.",
        "clinical_context": [
            "OOD evidence means the slide embedding does not sit close to the training/reference distribution, so the model may be extrapolating.",
            "The unified score is designed to make risk visible even when the raw class probability looks confident."
        ],
        "technical_context": [
            "OOD distance is estimated in bag-embedding space against class centroids stored in the phase-2 registry.",
            "The final score is an arithmetic mean of normalized risk components so every component stays in a comparable 0-1 range."
        ],
        "formulae": [
            "inverse_margin = clip(1 - margin, 0, 1)",
            "ood_score = clip((nearest_normalized_centroid_distance - 1.0) / 0.8, 0, 1)",
            "unified_safety_score = mean(uncertainty, inverse_margin, ood_score, optional ensemble_disagreement)",
            f"high risk if unified_safety_score >= {cfg.UNIFIED_SAFETY_HIGH:.2f}",
            f"moderate risk if unified_safety_score >= {cfg.UNIFIED_SAFETY_MODERATE:.2f}",
        ],
        "components": component_details,
        "ood": ood,
        "thresholds": {
            "ood_moderate_normalized_distance": cfg.OOD_MODERATE_THRESHOLD,
            "ood_strong_normalized_distance": cfg.OOD_STRONG_THRESHOLD,
            "unified_safety_moderate": cfg.UNIFIED_SAFETY_MODERATE,
            "unified_safety_high": cfg.UNIFIED_SAFETY_HIGH,
        },
        "outputs": {
            "unified_safety_score": round(unified_safety_score, 4),
            "risk_level": safety.get("risk_level"),
            "decision_status": safety.get("decision_status"),
            "abstain_recommended": safety.get("abstain_recommended"),
            "reasons": reasons,
        },
        "replication_steps": [
            "Read uncertainty and margin from Phase 1.",
            "Compute inverse_margin = 1 - margin.",
            "Estimate nearest class-centroid distance in embedding space and convert it to ood_score.",
            "Average all available risk components.",
            "Map the unified score to low/moderate/high risk thresholds."
        ],
    }
    safety["details"] = details
    return safety


def _merge_phase2_ensemble_safety(phase1_safety, probabilities, model_keys, bag_embeddings, calibration_metas, raw_probabilities=None):
    safety = dict(phase1_safety)
    per_model_ood = []
    for mkey, emb in zip(model_keys, bag_embeddings):
        per_model_ood.append({
            "model_key": mkey,
            "model_display": MODEL_REGISTRY.get(mkey, {}).get("display", mkey),
            **_estimate_ood_from_embedding(emb, mkey),
        })

    available_ood = [x for x in per_model_ood if x.get("available")]
    if available_ood:
        ood_score = float(np.mean([x["ood_score"] for x in available_ood if x["ood_score"] is not None]))
        id_support = float(np.mean([x["id_support_score"] for x in available_ood if x["id_support_score"] is not None]))
        strongest = max(available_ood, key=lambda x: x["ood_score"])
        ood_flag = any(x["ood_flag"] for x in available_ood)
        ood_level = "strong" if ood_flag else ("moderate" if any(x["ood_level"] == "moderate" for x in available_ood) else "low")
        ood = {
            "available": True,
            "ood_score": round(ood_score, 4),
            "ood_flag": ood_flag,
            "ood_level": ood_level,
            "nearest_class": strongest.get("nearest_class"),
            "nearest_distance": strongest.get("nearest_distance"),
            "normalized_distance": strongest.get("normalized_distance"),
            "id_support_score": round(id_support, 4),
            "per_model": per_model_ood,
        }
    else:
        ood = {
            "available": False,
            "ood_score": None,
            "ood_flag": False,
            "ood_level": "unavailable",
            "nearest_class": None,
            "nearest_distance": None,
            "normalized_distance": None,
            "id_support_score": None,
            "per_model": per_model_ood,
        }

    disagreement = safety.get("ensemble_disagreement")
    components = [
        float(safety.get("uncertainty", 0.0)),
        float(np.clip(1.0 - safety.get("margin", 0.0), 0.0, 1.0)),
        float(ood["ood_score"]) if ood["ood_score"] is not None else 0.0,
    ]
    if disagreement is not None:
        components.append(float(disagreement))
    unified_safety_score = float(np.mean(components))
    component_details = [
        {
            "name": "uncertainty",
            "value": round(float(safety.get("uncertainty", 0.0)), 4),
            "meaning": "Normalized entropy of the ensemble-averaged class probabilities.",
        },
        {
            "name": "inverse_margin",
            "value": round(float(np.clip(1.0 - safety.get("margin", 0.0), 0.0, 1.0)), 4),
            "meaning": "1 - margin; higher value means top classes are closer.",
        },
        {
            "name": "mean_ood_score",
            "value": round(float(ood["ood_score"]), 4) if ood["ood_score"] is not None else 0.0,
            "meaning": "Mean out-of-distribution score across available ensemble components.",
        },
    ]
    if disagreement is not None:
        component_details.append({
            "name": "ensemble_disagreement",
            "value": round(float(disagreement), 4),
            "meaning": "Fraction of component predictions outside the majority class.",
        })

    reasons = list(safety.get("reasons", []))
    if ood["available"] and ood["ood_level"] == "moderate":
        reasons.append(f"Moderate ensemble OOD shift toward {ood['nearest_class']}")
    if ood["ood_flag"]:
        reasons.append(f"Strong ensemble OOD signal beyond {ood['nearest_class']}")

    if ood["ood_flag"]:
        safety["display_prediction"] = "Needs Expert Review"
        safety["decision_status"] = "abstain"
        safety["prediction_key"] = "abstain"
        safety["abstain_recommended"] = True
        safety["risk_level"] = "urgent review recommended"
        safety["recommendation"] = "Potential out-of-distribution ensemble case detected; defer to expert review."
    elif unified_safety_score >= cfg.UNIFIED_SAFETY_HIGH and safety.get("risk_level") != "urgent review recommended":
        safety["risk_level"] = "high risk"
    elif unified_safety_score >= cfg.UNIFIED_SAFETY_MODERATE and safety.get("risk_level") == "low risk":
        safety["risk_level"] = "moderate risk"

    probs = np.asarray(probabilities, dtype=np.float32)
    raw_probs = np.asarray(raw_probabilities, dtype=np.float32) if raw_probabilities is not None else probs
    safety["reasons"] = reasons
    safety["phase"] = "phase2"
    safety["raw_probabilities"] = {
        CLASS_NAMES[i]: round(float(raw_probs[i]), 4)
        for i in range(N_CLASSES)
    }
    safety["calibration"] = {
        "available": any(meta.get("available") for meta in calibration_metas),
        "method": "per-model temperature scaling",
        "ensemble": calibration_metas,
    }
    safety["ood"] = ood
    safety["unified_safety_score"] = round(unified_safety_score, 4)
    safety["safety_score"] = round(unified_safety_score, 4)
    safety["id_support_score"] = ood["id_support_score"]
    details = dict(safety.get("details") or {})
    details["phase2"] = {
        "title": "Phase 2 ensemble safety calculation",
        "summary": "For ensembles, Phase 2 uses the same safety idea but averages OOD evidence across available component models and includes component disagreement.",
        "clinical_context": [
            "If component models disagree or several component embeddings look out-of-distribution, the case should be reviewed even if the averaged probability has a clear top class.",
            "This is especially useful for melanoma-vs-SCC or melanoma-vs-benign borderline cases where different encoders can emphasize different histologic patterns."
        ],
        "technical_context": [
            "Each component embedding is scored against its own OOD registry when available.",
            "The ensemble OOD score is the mean of available per-model OOD scores; ensemble disagreement is included as an additional normalized risk component."
        ],
        "formulae": [
            "inverse_margin = clip(1 - margin, 0, 1)",
            "ensemble_ood_score = mean(per_model_ood_score)",
            "unified_safety_score = mean(uncertainty, inverse_margin, ensemble_ood_score, ensemble_disagreement)",
            f"high risk if unified_safety_score >= {cfg.UNIFIED_SAFETY_HIGH:.2f}",
            f"moderate risk if unified_safety_score >= {cfg.UNIFIED_SAFETY_MODERATE:.2f}",
        ],
        "components": component_details,
        "ood": ood,
        "thresholds": {
            "ood_moderate_normalized_distance": cfg.OOD_MODERATE_THRESHOLD,
            "ood_strong_normalized_distance": cfg.OOD_STRONG_THRESHOLD,
            "unified_safety_moderate": cfg.UNIFIED_SAFETY_MODERATE,
            "unified_safety_high": cfg.UNIFIED_SAFETY_HIGH,
        },
        "outputs": {
            "unified_safety_score": round(unified_safety_score, 4),
            "risk_level": safety.get("risk_level"),
            "decision_status": safety.get("decision_status"),
            "abstain_recommended": safety.get("abstain_recommended"),
            "reasons": reasons,
        },
        "replication_steps": [
            "Compute ensemble uncertainty and margin from averaged probabilities.",
            "Compute inverse_margin = 1 - margin.",
            "Compute OOD score for each component model with an available registry.",
            "Average component OOD scores.",
            "Average uncertainty, inverse margin, ensemble OOD, and disagreement into the unified safety score."
        ],
    }
    safety["details"] = details
    return safety


def _build_retrieval_query_embedding(model_key, bag_embedding=None, ensemble_model_keys=None, ensemble_bag_embeddings=None):
    if model_key in ENSEMBLE_PRESETS:
        component_models = ENSEMBLE_PRESETS[model_key]["models"]
        if not ensemble_model_keys or not ensemble_bag_embeddings:
            return None
        by_model = {
            mkey: _normalize_embedding(emb)
            for mkey, emb in zip(ensemble_model_keys, ensemble_bag_embeddings)
            if emb is not None
        }
        if not all(mkey in by_model for mkey in component_models):
            return None
        return _normalize_embedding(np.concatenate([by_model[mkey] for mkey in component_models], axis=0))

    if bag_embedding is None:
        return None
    return _normalize_embedding(bag_embedding)


def _retrieval_prediction_csv(model_key):
    mcfg = MODEL_REGISTRY.get(model_key)
    if not mcfg:
        return None
    checkpoint = mcfg.get("mil_checkpoint")
    if not checkpoint:
        return None
    return Path(checkpoint).parent / "phase1_test_predictions.csv"


def _load_retrieval_prediction_map(model_key):
    path = _retrieval_prediction_csv(model_key)
    if path is None or not path.exists():
        return {}
    cache_key = f"predictions|{model_key}|{path}"
    mtime = path.stat().st_mtime
    cached = _retrieval_signal_cache.get(cache_key)
    if cached and cached["mtime"] == mtime:
        return cached["rows"]

    rows = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("slide_id")
                if sid:
                    rows[sid] = row
    except Exception:
        logger.exception("Failed to load retrieval prediction CSV: %s", path)
        rows = {}

    _retrieval_signal_cache[cache_key] = {"mtime": mtime, "rows": rows}
    return rows


def _prob_dict_from_row(row, fallback_label=None):
    if row:
        vals = {
            "Normal/Benign": row.get("prob_normal_benign"),
            "BCC": row.get("prob_bcc"),
            "SCC": row.get("prob_scc"),
            "Melanoma": row.get("prob_melanoma"),
        }
        try:
            probs = {k: float(v) for k, v in vals.items()}
            total = sum(max(v, 0.0) for v in probs.values())
            if total > 0:
                return {k: max(v, 0.0) / total for k, v in probs.items()}
        except (TypeError, ValueError):
            pass

    probs = {name: 0.05 for name in CLASS_NAMES.values()}
    if fallback_label in probs:
        probs[fallback_label] = 0.85
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def _signal_from_prob_dict(probs, pred_label=None, hard_case_candidate=False):
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1 = ordered[0] if ordered else ("Normal/Benign", 0.0)
    top2 = ordered[1] if len(ordered) > 1 else ("", 0.0)
    pred_label = pred_label or top1[0]
    return {
        "probs": {name: float(probs.get(name, 0.0)) for name in CLASS_NAMES.values()},
        "pred_label": pred_label,
        "confidence": float(top1[1]),
        "margin": float(top1[1] - top2[1]),
        "melanoma_probability": float(probs.get("Melanoma", 0.0)),
        "hard_case_candidate": bool(hard_case_candidate),
    }


def _query_signal_from_result(probabilities, safety):
    if probabilities is None:
        return None
    probs_arr = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probs_arr.size < N_CLASSES:
        return None
    probs = {CLASS_NAMES[i]: float(probs_arr[i]) for i in range(N_CLASSES)}
    pred = CLASS_NAMES[int(np.argmax(probs_arr))]
    return _signal_from_prob_dict(
        probs,
        pred_label=(safety or {}).get("raw_prediction") or pred,
        hard_case_candidate=bool((safety or {}).get("hard_case_candidate")),
    )


def _signal_entropy(signal):
    probs = np.asarray([signal["probs"][CLASS_NAMES[i]] for i in range(N_CLASSES)], dtype=np.float32)
    probs = np.clip(probs, 1e-8, 1.0)
    return float(-(probs * np.log(probs)).sum() / np.log(N_CLASSES))


def _clinical_signature_from_signal(signal):
    probs = signal["probs"]
    return np.asarray(
        [
            probs["Normal/Benign"],
            probs["BCC"],
            probs["SCC"],
            probs["Melanoma"],
            signal["confidence"],
            signal["margin"],
            signal["melanoma_probability"],
            _signal_entropy(signal),
        ],
        dtype=np.float32,
    )


def _pathology_axis_from_signal(signal):
    probs = signal["probs"]
    return np.asarray(
        [
            signal["melanoma_probability"],
            probs["BCC"] + probs["SCC"],
            probs["Normal/Benign"],
            1.0 - signal["margin"],
            _signal_entropy(signal),
            float(signal["hard_case_candidate"]),
        ],
        dtype=np.float32,
    )


def _clinical_similarity(query_sig, candidate_sigs):
    weights = np.asarray([0.5, 0.5, 0.5, 1.3, 0.5, 0.7, 1.2, 0.8], dtype=np.float32)
    diff = np.abs(candidate_sigs - query_sig[None, :])
    return np.clip(1.0 - (diff @ weights) / 5.0, 0.0, 1.0)


def _pathology_axis_similarity(query_axis, candidate_axes):
    weights = np.asarray([1.6, 0.8, 0.5, 1.0, 1.0, 1.2], dtype=np.float32)
    diff = np.abs(candidate_axes - query_axis[None, :])
    return np.clip(1.0 - (diff @ weights) / weights.sum(), 0.0, 1.0)


def _risk_tier(signal):
    melanoma_guard = signal["pred_label"] != "Melanoma" and signal["melanoma_probability"] >= 0.20
    if melanoma_guard or signal["hard_case_candidate"] or signal["confidence"] < 0.62 or signal["margin"] < 0.18:
        return "high"
    if signal["confidence"] < 0.80 or signal["margin"] < 0.35 or signal["melanoma_probability"] >= 0.10:
        return "moderate"
    return "low"


def _label_risk_rank(label):
    if label == "Melanoma":
        return 3
    if label in {"BCC", "SCC"}:
        return 1
    return 0


def _signal_risk_rank(signal):
    if signal["pred_label"] == "Melanoma" or signal["melanoma_probability"] >= 0.35:
        return 3
    if signal["melanoma_probability"] >= 0.20 or signal["hard_case_candidate"]:
        return 2
    if signal["pred_label"] in {"BCC", "SCC"}:
        return 1
    return 0


def _risk_lattice_similarity(query_rank, candidate_ranks):
    candidate_ranks = np.asarray(candidate_ranks, dtype=np.float32)
    base = 1.0 - np.abs(candidate_ranks - float(query_rank)) / 3.0
    if query_rank >= 2:
        base = np.where(candidate_ranks >= 2, np.maximum(base, 0.86), base)
        base = np.where(candidate_ranks == 3, base + 0.08, base)
    elif query_rank == 1:
        base = np.where(candidate_ranks == 1, base + 0.05, base)
    return np.clip(base, 0.05, 1.0)


def _candidate_label_bonus(candidate_labels, signal, tier):
    labels = np.asarray(candidate_labels)
    bonus = np.zeros(len(labels), dtype=np.float32)
    bonus[labels == signal["pred_label"]] += 0.08
    if tier == "high" or signal["melanoma_probability"] >= 0.20:
        bonus[labels == "Melanoma"] += 0.14
    elif signal["melanoma_probability"] >= 0.10:
        bonus[labels == "Melanoma"] += 0.05
    return bonus


def _unit_similarity_from_dot(dot_scores):
    return np.clip((np.asarray(dot_scores, dtype=np.float32) + 1.0) / 2.0, 1e-6, 1.0)


def _diagnostic_contrast_similarity(query_sig, candidate_sigs):
    query_profile = np.asarray(
        [
            query_sig[6],
            query_sig[2],
            query_sig[1],
            query_sig[6] - query_sig[2],
            query_sig[6] - max(query_sig[1], query_sig[2]),
        ],
        dtype=np.float32,
    )
    cand_profiles = np.stack(
        [
            candidate_sigs[:, 6],
            candidate_sigs[:, 2],
            candidate_sigs[:, 1],
            candidate_sigs[:, 6] - candidate_sigs[:, 2],
            candidate_sigs[:, 6] - np.maximum(candidate_sigs[:, 1], candidate_sigs[:, 2]),
        ],
        axis=1,
    ).astype(np.float32)
    weights = np.asarray([1.5, 0.8, 0.5, 1.2, 1.2], dtype=np.float32)
    diff = np.abs(cand_profiles - query_profile[None, :])
    return np.clip(1.0 - (diff @ weights) / weights.sum(), 0.0, 1.0)


def _top_feature_proxy_similarity(query, candidates):
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    candidates = np.asarray(candidates, dtype=np.float32)
    n_top = max(8, min(32, query.shape[0] // 8))
    q_top = np.argsort(np.abs(query))[::-1][:n_top]
    q_mask = np.zeros(query.shape[0], dtype=bool)
    q_mask[q_top] = True
    sims = []
    for cand in candidates:
        c_top = np.argsort(np.abs(cand))[::-1][:n_top]
        c_mask = np.zeros(cand.shape[0], dtype=bool)
        c_mask[c_top] = True
        inter = float(np.logical_and(q_mask, c_mask).sum())
        union = float(np.logical_or(q_mask, c_mask).sum())
        sims.append(inter / union if union else 0.0)
    return np.asarray(sims, dtype=np.float32)


def _build_diagnostic_quotient(embeddings, signatures):
    probs = np.asarray(signatures[:, :N_CLASSES], dtype=np.float32)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    weights = probs.sum(axis=0)
    center = embeddings.mean(axis=0, keepdims=True)
    centroids = (probs.T @ embeddings) / np.maximum(weights[:, None], 1e-8)
    directions = centroids - center
    _, sigma, vt = np.linalg.svd(directions, full_matrices=False)
    if sigma.size == 0:
        return {"basis": np.zeros((embeddings.shape[1], 0), dtype=np.float32), "center": center[0], "scale": 1.0, "dim": 0}
    tol = max(float(sigma.max()), 1.0) * 1e-5
    rank = int(np.sum(sigma > tol))
    rank = max(1, min(rank, N_CLASSES - 1, vt.shape[0]))
    basis = vt[:rank].T.astype(np.float32)
    coords = ((embeddings - center) @ basis).astype(np.float32)
    coords = coords / np.maximum(coords.std(axis=0, keepdims=True), 1e-6)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.maximum((diffs * diffs).sum(axis=-1), 0.0))
    off_diag = dists[~np.eye(len(coords), dtype=bool)]
    scale = float(np.median(off_diag)) if len(off_diag) else 1.0
    return {
        "basis": basis,
        "center": center[0].astype(np.float32),
        "coords": coords,
        "scale": max(scale, 1e-3),
        "dim": rank,
    }


def _diagnostic_quotient_similarity(query, quotient, candidate_indices):
    if quotient.get("dim", 0) <= 0 or len(candidate_indices) == 0:
        return np.ones(len(candidate_indices), dtype=np.float32)
    q = ((query - quotient["center"]) @ quotient["basis"]).astype(np.float32)
    std = np.maximum(quotient["coords"].std(axis=0), 1e-6)
    q = q / std
    c = quotient["coords"][candidate_indices]
    diff = c - q[None, :]
    dist2 = np.maximum((diff * diff).sum(axis=1), 0.0)
    scale2 = max(float(quotient.get("scale", 1.0)) ** 2, 1e-6)
    return np.exp(-dist2 / (2.0 * scale2)).astype(np.float32)


def _candidate_mask_for_cost_aware(labels, signal, tier, exclude_index=None):
    labels_arr = np.asarray(labels)
    mask = np.ones(len(labels_arr), dtype=bool)
    if exclude_index is not None and 0 <= exclude_index < len(mask):
        mask[exclude_index] = False
    if tier == "high":
        return mask
    ordered_probs = sorted(signal["probs"].items(), key=lambda kv: kv[1], reverse=True)
    candidate_labels = {signal["pred_label"]}
    if tier == "moderate":
        candidate_labels.update([ordered_probs[0][0], ordered_probs[1][0]])
        if signal["melanoma_probability"] >= 0.10:
            candidate_labels.add("Melanoma")
    routed = np.isin(labels_arr, list(candidate_labels))
    if exclude_index is not None and 0 <= exclude_index < len(mask):
        routed[exclude_index] = False
    return routed


def _bank_retrieval_signals(bank_key, bank, embeddings, registry):
    case_ids = bank.get("case_ids", [])
    cache_key = f"bank_signals|{bank_key}|{PHASE4_RETRIEVAL_PATH.stat().st_mtime if PHASE4_RETRIEVAL_PATH.exists() else 0}"
    cached = _retrieval_signal_cache.get(cache_key)
    if cached:
        return cached

    component_models = bank.get("component_models") or ([bank_key] if bank_key in MODEL_REGISTRY else [])
    prediction_maps = {mkey: _load_retrieval_prediction_map(mkey) for mkey in component_models}
    cases = registry.get("cases", {})
    labels = []
    signals = []
    for sid in case_ids:
        meta = cases.get(sid) or {}
        true_label = meta.get("true_label")
        labels.append(true_label or "Unknown")
        probs_list = []
        hard_flags = [bool(meta.get("is_hard_melanoma"))]
        pred_labels = []
        for mkey, pred_map in prediction_maps.items():
            row = pred_map.get(sid)
            if row:
                probs_list.append(_prob_dict_from_row(row, fallback_label=true_label))
                hard_flags.append(str(row.get("hard_case_candidate", "0")).lower() in {"1", "true", "yes"})
                pred_labels.append(row.get("pred_label"))
        if probs_list:
            probs = {
                cls: float(np.mean([p.get(cls, 0.0) for p in probs_list]))
                for cls in CLASS_NAMES.values()
            }
        else:
            probs = _prob_dict_from_row(None, fallback_label=true_label)
        signal = _signal_from_prob_dict(probs, hard_case_candidate=any(hard_flags))
        if pred_labels and len(component_models) == 1 and pred_labels[0]:
            signal["pred_label"] = pred_labels[0]
        signals.append(signal)

    signatures = np.stack([_clinical_signature_from_signal(s) for s in signals], axis=0)
    axes = np.stack([_pathology_axis_from_signal(s) for s in signals], axis=0)
    risk_ranks = np.asarray([max(_signal_risk_rank(s), _label_risk_rank(lbl)) for s, lbl in zip(signals, labels)], dtype=np.int64)
    quotient = _build_diagnostic_quotient(embeddings, signatures)
    payload = {
        "labels": labels,
        "signals": signals,
        "signatures": signatures,
        "axes": axes,
        "risk_ranks": risk_ranks,
        "quotient": quotient,
    }
    _retrieval_signal_cache[cache_key] = payload
    return payload


def _cost_aware_component_scores(query, candidate_indices, embeddings, bank_signals, query_signal):
    labels = np.asarray(bank_signals["labels"])
    signatures = bank_signals["signatures"]
    axes = bank_signals["axes"]
    quotient = bank_signals["quotient"]
    query_sig = _clinical_signature_from_signal(query_signal)
    query_axis = _pathology_axis_from_signal(query_signal)
    query_rank = _signal_risk_rank(query_signal)
    emb_scores = _unit_similarity_from_dot(embeddings[candidate_indices] @ query)
    quotient_scores = _diagnostic_quotient_similarity(query, quotient, candidate_indices)
    clinical_scores = _clinical_similarity(query_sig, signatures[candidate_indices])
    axis_scores = _pathology_axis_similarity(query_axis, axes[candidate_indices])
    tile_scores = _top_feature_proxy_similarity(query, embeddings[candidate_indices])
    contrast_scores = _diagnostic_contrast_similarity(query_sig, signatures[candidate_indices])
    lattice_scores = _risk_lattice_similarity(query_rank, bank_signals["risk_ranks"][candidate_indices])
    candidate_labels = labels[candidate_indices]
    evidence_scores = np.full(len(candidate_indices), 0.62, dtype=np.float32)
    evidence_scores[candidate_labels == query_signal["pred_label"]] = 0.86
    if query_signal["melanoma_probability"] >= 0.20 or query_signal["pred_label"] == "Melanoma":
        evidence_scores[candidate_labels == "Melanoma"] = 0.96
    elif query_signal["melanoma_probability"] >= 0.10:
        evidence_scores[candidate_labels == "Melanoma"] = 0.78
    return {
        "embedding": np.clip(emb_scores, 1e-6, 1.0),
        "quotient": np.clip(quotient_scores, 1e-6, 1.0),
        "clinical": np.clip(clinical_scores, 1e-6, 1.0),
        "axis": np.clip(axis_scores, 1e-6, 1.0),
        "tile": np.clip(tile_scores, 1e-6, 1.0),
        "contrast": np.clip(contrast_scores, 1e-6, 1.0),
        "lattice": np.clip(lattice_scores, 1e-6, 1.0),
        "evidence": np.clip(evidence_scores, 1e-6, 1.0),
    }


def _cost_aware_weights(method):
    if method == "aags_product_v1":
        return {"embedding": 0.34, "clinical": 0.17, "axis": 0.14, "tile": 0.10, "contrast": 0.11, "lattice": 0.09, "evidence": 0.05}
    if method == "aags_quotient_v2":
        return {"embedding": 0.24, "quotient": 0.18, "clinical": 0.15, "axis": 0.13, "tile": 0.09, "contrast": 0.10, "lattice": 0.07, "evidence": 0.04}
    if method == "trlq_tropical_v1":
        return {"embedding": 0.36, "clinical": 0.15, "axis": 0.13, "tile": 0.09, "contrast": 0.12, "lattice": 0.10, "evidence": 0.05}
    return {"embedding": 0.25, "quotient": 0.18, "clinical": 0.14, "axis": 0.12, "tile": 0.08, "contrast": 0.10, "lattice": 0.09, "evidence": 0.04}


def _combine_cost_aware_scores(components, method):
    weights = _cost_aware_weights(method)
    if method.startswith("aags"):
        scores = np.ones(len(next(iter(components.values()))), dtype=np.float32)
        for key, weight in weights.items():
            scores *= np.power(components[key], weight)
        return scores
    cost = np.zeros(len(next(iter(components.values()))), dtype=np.float32)
    for key, weight in weights.items():
        cost += weight * (-np.log(components[key]))
    return -cost


def _display_similarity_from_active_score(score, method):
    score = float(score)
    if method.startswith("trlq"):
        return float(np.exp(score))
    return float(np.clip(score, 0.0, 1.0))


def _cost_aware_search(query, embeddings, case_ids, bank_signals, query_signal, exclude_index=None, top_k=5, method=None):
    method = method or cfg.RETRIEVAL_ACTIVE_METHOD
    labels = bank_signals["labels"]
    tier = _risk_tier(query_signal)
    mask = _candidate_mask_for_cost_aware(labels, query_signal, tier, exclude_index=exclude_index)
    candidate_indices = np.flatnonzero(mask)
    if len(candidate_indices) == 0:
        return {"indices": [], "scores": np.asarray([], dtype=np.float32), "components": {}, "tier": tier, "cost": {}}

    query_sig = _clinical_signature_from_signal(query_signal)
    query_axis = _pathology_axis_from_signal(query_signal)
    query_rank = _signal_risk_rank(query_signal)
    signatures = bank_signals["signatures"]
    axes = bank_signals["axes"]
    quotient = bank_signals["quotient"]
    candidate_labels = [labels[idx] for idx in candidate_indices]
    clinical_scores = _clinical_similarity(query_sig, signatures[candidate_indices])
    lattice_scores = _risk_lattice_similarity(query_rank, bank_signals["risk_ranks"][candidate_indices])
    contrast_scores = _diagnostic_contrast_similarity(query_sig, signatures[candidate_indices])
    quotient_scores = _diagnostic_quotient_similarity(query, quotient, candidate_indices)
    label_bonus = _candidate_label_bonus(candidate_labels, query_signal, tier)
    routing_scores = (
        0.38 * clinical_scores +
        0.20 * lattice_scores +
        0.18 * contrast_scores +
        0.14 * quotient_scores +
        0.10 * label_bonus
    )

    if tier == "low":
        budget = min(32, len(candidate_indices))
        rerank_budget = min(14, budget)
    elif tier == "moderate":
        budget = min(80, len(candidate_indices))
        rerank_budget = min(28, budget)
    else:
        budget = min(160, len(candidate_indices))
        rerank_budget = min(54, budget)

    preselect_local = np.argsort(routing_scores)[::-1][:budget]
    preselect = candidate_indices[preselect_local]
    preselect_routing = routing_scores[preselect_local]
    emb_unit = _unit_similarity_from_dot(embeddings[preselect] @ query)
    stage_scores = 0.58 * emb_unit + 0.42 * preselect_routing
    rerank_local = np.argsort(stage_scores)[::-1][:rerank_budget]
    rerank_indices = preselect[rerank_local]
    components = _cost_aware_component_scores(query, rerank_indices, embeddings, bank_signals, query_signal)
    final_scores = _combine_cost_aware_scores(components, method)
    order_local = np.argsort(final_scores)[::-1]
    ordered = rerank_indices[order_local].tolist()
    ordered_scores = final_scores[order_local]

    melanoma_guard = query_signal["pred_label"] != "Melanoma" and query_signal["melanoma_probability"] >= 0.20
    extra_melanoma_scan = 0
    if melanoma_guard and not any(labels[idx] == "Melanoma" for idx in ordered[:top_k]):
        mel_mask = np.asarray(labels) == "Melanoma"
        if exclude_index is not None and 0 <= exclude_index < len(mel_mask):
            mel_mask[exclude_index] = False
        mel_candidates = np.flatnonzero(mel_mask)
        if len(mel_candidates):
            mel_components = _cost_aware_component_scores(query, mel_candidates, embeddings, bank_signals, query_signal)
            mel_scores = 0.45 * mel_components["embedding"] + 0.25 * mel_components["axis"] + 0.20 * mel_components["contrast"] + 0.10 * mel_components["lattice"]
            best_mel = int(mel_candidates[int(np.argmax(mel_scores))])
            if best_mel not in ordered:
                ordered = ordered[: max(0, top_k - 1)] + [best_mel] + ordered[top_k - 1:]
                ordered_scores = np.concatenate([ordered_scores[: max(0, top_k - 1)], np.asarray([float(np.max(mel_scores))], dtype=np.float32), ordered_scores[top_k - 1:]])
            extra_melanoma_scan = int(len(mel_candidates))

    signature_dim = signatures.shape[1]
    axis_dim = axes.shape[1]
    quotient_dim = int(quotient.get("dim", 0))
    embedding_dim = embeddings.shape[1]
    routing_equivalent = len(candidate_indices) * ((signature_dim + axis_dim + quotient_dim + 2) / max(embedding_dim, 1))
    rerank_equivalent = rerank_budget * (0.45 + (axis_dim + quotient_dim) / max(embedding_dim, 1))
    equivalent_cost = float(routing_equivalent + budget + rerank_equivalent + extra_melanoma_scan)
    cost = {
        "candidate_count": int(len(candidate_indices)),
        "preselect_budget": int(budget),
        "rerank_budget": int(rerank_budget),
        "melanoma_guard_extra_scan": int(extra_melanoma_scan),
        "embedding_dot_products_executed": int(budget + extra_melanoma_scan),
        "equivalent_full_vector_comparisons": round(equivalent_cost, 4),
        "routing_equivalent_comparisons": round(float(routing_equivalent), 4),
        "rerank_equivalent_comparisons": round(float(rerank_equivalent), 4),
    }
    return {
        "indices": ordered[:top_k],
        "scores": ordered_scores[:top_k],
        "components": components,
        "component_indices": rerank_indices.tolist(),
        "tier": tier,
        "method": method,
        "cost": cost,
    }


def _score_specific_candidates(query, embeddings, bank_signals, query_signal, candidate_indices, method):
    if len(candidate_indices) == 0:
        return np.asarray([], dtype=np.float32), {}
    components = _cost_aware_component_scores(query, np.asarray(candidate_indices, dtype=np.int64), embeddings, bank_signals, query_signal)
    scores = _combine_cost_aware_scores(components, method)
    return scores, components


def _format_component_snapshot(components, local_idx):
    out = {}
    for key, values in components.items():
        try:
            out[key] = round(float(values[local_idx]), 4)
        except Exception:
            pass
    return out


def _format_retrieval_case(case_meta, similarity, method="cosine", component_scores=None, rank_score=None):
    similarity_value = round(float(similarity), 4)
    return {
        "slide_id": case_meta["slide_id"],
        "filename": case_meta.get("filename"),
        "true_label": case_meta.get("true_label"),
        "source": case_meta.get("source"),
        "similarity": similarity_value,
        "thumbnail_url": case_meta.get("thumbnail_url"),
        "is_hard_melanoma": bool(case_meta.get("is_hard_melanoma")),
        "detail": {
            "metric": method,
            "score": similarity_value,
            "rank_score": None if rank_score is None else round(float(rank_score), 4),
            "component_scores": component_scores or {},
            "formula": "TRLQ uses weighted tropical cost: score = exp(-sum_i w_i * -log(component_i)); AAGS uses product_i component_i^w_i.",
            "computed_from": "Slide-level MIL bag embeddings, not raw pixels and not metadata.",
            "interpretation": "Higher score means stronger agreement across embedding similarity, diagnostic quotient, clinical probability profile, risk lattice, melanoma contrast, and top-feature proxy.",
            "medical_use": "Use as case-based evidence: compare morphology, predicted class, safety flags, and attention regions before trusting the analogy.",
        },
    }


def _retrieve_similar_cases(
    model_key,
    bag_embedding=None,
    ensemble_model_keys=None,
    ensemble_bag_embeddings=None,
    probabilities=None,
    safety=None,
    query_slide_id=None,
    top_k=5,
    hard_top_k=3,
):
    registry, arrays = _load_phase4_registry()
    banks = registry.get("banks", {})
    bank = banks.get(model_key)
    if not bank:
        return {
            "available": False,
            "bank_key": model_key,
            "similar_cases": [],
            "hard_melanoma_matches": [],
        }

    embeddings = arrays.get(model_key)
    if embeddings is None or not len(embeddings):
        return {
            "available": False,
            "bank_key": model_key,
            "similar_cases": [],
            "hard_melanoma_matches": [],
        }

    query = _build_retrieval_query_embedding(
        model_key,
        bag_embedding=bag_embedding,
        ensemble_model_keys=ensemble_model_keys,
        ensemble_bag_embeddings=ensemble_bag_embeddings,
    )
    if query is None or query.shape[0] != embeddings.shape[1]:
        return {
            "available": False,
            "bank_key": model_key,
            "similar_cases": [],
            "hard_melanoma_matches": [],
        }

    case_lookup = registry.get("cases", {})
    case_ids = bank.get("case_ids", [])
    exhaustive_comparisons = int(len(case_ids))
    embedding_dim = int(query.shape[0])
    exclude_index = None
    if query_slide_id and query_slide_id in case_ids:
        exclude_index = case_ids.index(query_slide_id)

    query_signal = _query_signal_from_result(probabilities, safety)
    if query_signal is None:
        query_signal = _signal_from_prob_dict(
            {CLASS_NAMES[i]: 1.0 / N_CLASSES for i in range(N_CLASSES)},
            pred_label=(safety or {}).get("raw_prediction"),
            hard_case_candidate=bool((safety or {}).get("hard_case_candidate")),
        )

    bank_signals = _bank_retrieval_signals(model_key, bank, embeddings, registry)
    active_method = cfg.RETRIEVAL_ACTIVE_METHOD
    active = _cost_aware_search(
        query,
        embeddings,
        case_ids,
        bank_signals,
        query_signal,
        exclude_index=exclude_index,
        top_k=top_k,
        method=active_method,
    )

    method_summaries = {}
    for method in ("macs_attention_v1", "aags_quotient_v2", "trlq_quotient_v2"):
        if method == "macs_attention_v1":
            # MACS is represented by the first routing+embedding stage; it is
            # included as a cost comparator and fallback interpretation.
            probe = _cost_aware_search(
                query,
                embeddings,
                case_ids,
                bank_signals,
                query_signal,
                exclude_index=exclude_index,
                top_k=top_k,
                method="aags_product_v1",
            )
            method_summaries[method] = {
                "role": "clinical shortlist plus attention/pathology-aware rerank comparator",
                "tier": probe.get("tier"),
                "equivalent_full_vector_comparisons": probe.get("cost", {}).get("equivalent_full_vector_comparisons"),
                "embedding_dot_products_executed": probe.get("cost", {}).get("embedding_dot_products_executed"),
            }
        else:
            probe = active if method == active_method else _cost_aware_search(
                query,
                embeddings,
                case_ids,
                bank_signals,
                query_signal,
                exclude_index=exclude_index,
                top_k=top_k,
                method=method,
            )
            method_summaries[method] = {
                "role": "active" if method == active_method else "available comparator",
                "tier": probe.get("tier"),
                "equivalent_full_vector_comparisons": probe.get("cost", {}).get("equivalent_full_vector_comparisons"),
                "embedding_dot_products_executed": probe.get("cost", {}).get("embedding_dot_products_executed"),
            }

    baseline_cost = {
        "method": "full_cosine_exhaustive",
        "comparisons": exhaustive_comparisons - (1 if exclude_index is not None else 0),
        "multiply_adds_estimate": int((exhaustive_comparisons - (1 if exclude_index is not None else 0)) * embedding_dim),
        "complexity": "O(N * D) full embedding dot products plus sorting.",
        "executed": False,
        "note": "Shown as baseline cost only; the active UI path no longer executes full cosine search before retrieval.",
    }
    active_cost = active.get("cost", {})
    active_equiv = float(active_cost.get("equivalent_full_vector_comparisons", exhaustive_comparisons) or exhaustive_comparisons)

    similar_cases = []
    active_component_index = {idx: pos for pos, idx in enumerate(active.get("component_indices", []))}
    active_components = active.get("components", {})
    for idx, rank_score in zip(active.get("indices", []), active.get("scores", [])):
        if idx >= len(case_ids):
            continue
        slide_id = case_ids[idx]
        case_meta = case_lookup.get(slide_id)
        if not case_meta:
            continue
        local_idx = active_component_index.get(idx)
        component_scores = _format_component_snapshot(active_components, local_idx) if local_idx is not None else {}
        similar_cases.append(_format_retrieval_case(
            case_meta,
            _display_similarity_from_active_score(rank_score, active_method),
            method=active_method,
            component_scores=component_scores,
            rank_score=rank_score,
        ))

    hard_indices = [
        idx for idx, sid in enumerate(case_ids)
        if idx != exclude_index and bool((case_lookup.get(sid) or {}).get("is_hard_melanoma"))
    ]
    hard_scores, hard_components = _score_specific_candidates(
        query,
        embeddings,
        bank_signals,
        query_signal,
        hard_indices,
        active_method,
    )
    hard_order = np.argsort(hard_scores)[::-1][:hard_top_k] if len(hard_scores) else []
    hard_cases = []
    for local_pos in hard_order:
        idx = int(hard_indices[int(local_pos)])
        slide_id = case_ids[idx]
        case_meta = case_lookup.get(slide_id)
        if not case_meta:
            continue
        hard_cases.append(_format_retrieval_case(
            case_meta,
            _display_similarity_from_active_score(hard_scores[int(local_pos)], active_method),
            method=active_method,
            component_scores=_format_component_snapshot(hard_components, int(local_pos)),
            rank_score=hard_scores[int(local_pos)],
        ))
    hard_scan_count = int(len(hard_indices))
    active_equiv_total = active_equiv + hard_scan_count
    saved_equiv = max(float(baseline_cost["comparisons"]) - active_equiv_total, 0.0)

    return {
        "available": True,
        "bank_key": model_key,
        "bank_display": bank.get("display", model_key),
        "bank_type": bank.get("type", "single_model"),
        "bank_size": int(bank.get("n_cases", len(case_ids))),
        "hard_case_count": int(bank.get("hard_case_count", 0)),
        "similar_cases": similar_cases,
        "hard_melanoma_matches": hard_cases,
        "details": {
            "title": "Cost-aware pathology retrieval calculation",
            "summary": "The retrieval panel now uses TRLQ/AAGS-style cost-aware pathology search instead of exhaustive cosine. It routes by cheap clinical/pathology signals, then spends full embedding comparisons only on a short candidate list.",
            "clinical_context": [
                "Similar cases are not additional labels. They are visual and statistical evidence that helps a reviewer ask whether the current slide resembles known BCC, SCC, melanoma, benign, or hard melanoma examples.",
                "The hard-melanoma list is useful when the model predicts another class but the feature space still places the case near difficult melanoma examples.",
                "The risk tier controls search breadth: high-risk or melanoma-borderline queries deliberately search more widely; low-risk queries spend less retrieval cost."
            ],
            "technical_context": [
                "The query vector is the MIL bag embedding z for a single model. For an ensemble bank, normalized component embeddings are concatenated in the ensemble component order and normalized again.",
                "Each bank vector was precomputed from the same feature extractor/MIL family, then L2-normalized before storage.",
                "A cheap clinical signature p=[P(Normal),P(BCC),P(SCC),P(Melanoma),confidence,margin,P(Melanoma),entropy] and pathology axis vector are used to route candidates before full embedding scoring.",
                "AAGS combines component similarities by a weighted product. TRLQ maps component similarities to tropical costs with -log and minimizes accumulated evidence penalty.",
                "The active production path is TRLQ quotient v2; full cosine is retained only as a displayed baseline cost, not as the executed search."
            ],
            "metric": active_method,
            "metric_formula": "TRLQ score(q,x)=exp(-sum_i w_i * -log(s_i(q,x))); AAGS score(q,x)=prod_i s_i(q,x)^w_i",
            "query_embedding_dim": embedding_dim,
            "ranking_rule": "Candidates are first routed by clinical/pathology signals, then reranked by TRLQ over embedding, diagnostic quotient, clinical profile, pathology axis, top-feature proxy, melanoma contrast, risk lattice, and label evidence.",
            "cost": {
                "active_runtime_mode": active_method,
                "risk_tier": active.get("tier"),
                "bank_size": exhaustive_comparisons,
                "full_cosine_baseline_comparisons": baseline_cost["comparisons"],
                "full_cosine_baseline_multiply_adds": baseline_cost["multiply_adds_estimate"],
                "active_equivalent_full_vector_comparisons": round(active_equiv_total, 4),
                "active_embedding_dot_products_executed": int((active_cost.get("embedding_dot_products_executed") or 0) + hard_scan_count),
                "saved_equivalent_comparisons_vs_cosine": round(saved_equiv, 4),
                "cost_ratio_vs_full_cosine": round(active_equiv_total / max(float(baseline_cost["comparisons"]), 1.0), 4),
                "estimated_cost_reduction_percent": round(100.0 * saved_equiv / max(float(baseline_cost["comparisons"]), 1.0), 2),
                "candidate_count_after_safe_routing": active_cost.get("candidate_count"),
                "preselect_budget": active_cost.get("preselect_budget"),
                "rerank_budget": active_cost.get("rerank_budget"),
                "melanoma_guard_extra_scan": active_cost.get("melanoma_guard_extra_scan"),
                "hard_melanoma_evidence_scan": hard_scan_count,
                "baseline_note": baseline_cost["note"],
            },
            "method_comparison": method_summaries,
            "formulae": [
                "clinical_signature = [P(N), P(BCC), P(SCC), P(Mel), confidence, margin, P(Mel), entropy]",
                "pathology_axis = [P(Mel), P(BCC)+P(SCC), P(N), 1-margin, entropy, hard_case_flag]",
                "SAFE-R tier = high if melanoma guard / hard case / low confidence / low margin; moderate for borderline uncertainty; else low",
                "routing_score = 0.38 clinical + 0.20 risk_lattice + 0.18 melanoma_contrast + 0.14 diagnostic_quotient + 0.10 label_bonus",
                "stage_score = 0.58 embedding_similarity + 0.42 routing_score",
                "MACS = SAFE-R candidate routing + clinical/pathology preselection + embedding rerank on the shortlist",
                "AAGS = product_i component_i ^ weight_i",
                "TRLQ = exp(-sum_i weight_i * (-log(component_i)))",
                "full_cosine_cost = N * D multiply-adds",
                "active_cost ~= routed_low_dim_cost + preselect_embedding_dots + rerank_component_cost + melanoma_guard_scan",
            ],
            "score_distribution": {
                "max": max((c["similarity"] for c in similar_cases), default=None),
                "mean": round(float(np.mean([c["similarity"] for c in similar_cases])), 4) if similar_cases else None,
                "min": min((c["similarity"] for c in similar_cases), default=None),
            },
            "steps": [
                "Build query embedding from the selected model output.",
                "L2-normalize the query embedding.",
                "Build query clinical signature, pathology axis, melanoma contrast and risk-lattice rank from the model probabilities and safety flags.",
                "Use SAFE-R routing to choose a low/moderate/high risk candidate pool.",
                "Use cheap routing scores to preselect a short list instead of scoring every bank embedding.",
                "Run TRLQ quotient v2 on the shortlist and return the top similar cases.",
                "Run a hard-melanoma evidence pass so dangerous analogues remain visible.",
            ],
            "replication_steps": [
                "Run the same encoder and MIL model on the query WSI to obtain bag embedding z.",
                "Normalize z with L2 norm.",
                "Load the selected bank embedding matrix E with shape [N, D].",
                "Load phase1_test_predictions.csv for the same bank to build clinical signatures for each reference case.",
                "Compute SAFE-R tier and candidate mask from query probabilities and melanoma safety flags.",
                "Compute routing scores on low-dimensional signatures.",
                "Compute full embedding/component scores only for the preselected shortlist.",
                "Rank by active TRLQ score and map indices back to case metadata."
            ],
            "limitations": [
                "Cost numbers are equivalent full-vector comparisons; low-dimensional routing operations are converted to comparable units for transparency.",
                "A retrieved case supports explanation and review; it is not a ground-truth diagnosis for the current WSI.",
                "TRLQ/AAGS are pathology-specific because their components depend on melanoma probability, diagnostic quotient, risk lattice and differential diagnosis structure."
            ],
        },
    }


def _get_retrieval_case_meta(slide_id):
    registry, _ = _load_phase4_registry()
    return (registry.get("cases") or {}).get(slide_id)


def _comparison_job_id(slide_id, model_key):
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in model_key)[:32]
    return f"retrcmp_{slide_id[:12]}_{safe_model}"


def _build_result_artifacts(job_id, result):
    views = result.get("heatmap_views") or []
    overlay_urls = {}
    mask_urls = {}
    for view in views:
        key = view["key"]
        if key in ("attention", "default"):
            overlay_urls[key] = f"/api/results/{job_id}/heatmap"
            mask_urls[key] = f"/api/results/{job_id}/heatmap_only"
        else:
            overlay_urls[key] = f"/api/results/{job_id}/heatmap/{key}"
            mask_urls[key] = f"/api/results/{job_id}/heatmap_only/{key}"
    return {
        "thumbnail_url": f"/api/results/{job_id}/thumbnail",
        "heatmap_overlay_urls": overlay_urls,
        "heatmap_mask_urls": mask_urls,
        "tile_base_url": f"/api/results/{job_id}/tiles",
        "export_url": f"/api/results/{job_id}/export",
    }


def _build_result_calculation_details(result, slide_info):
    probabilities = result.get("probabilities") or {}
    sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top1 = sorted_probs[0] if sorted_probs else (None, 0.0)
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)
    margin = float(top1[1]) - float(top2[1])
    safety = result.get("safety") or {}
    retrieval = result.get("retrieval") or {}
    heatmap_views = result.get("heatmap_views") or []
    top_tiles = result.get("top_tiles") or []
    ensemble_details = result.get("ensemble_details") or []
    model_key = result.get("model_key")
    if model_key in ENSEMBLE_PRESETS:
        model_kind = "ensemble"
        model_components = ENSEMBLE_PRESETS[model_key].get("models", [])
        model_feature_dim = "concatenated component bag embeddings"
    else:
        model_cfg = MODEL_REGISTRY.get(model_key, {})
        model_kind = model_cfg.get("type", "unknown")
        model_components = []
        model_feature_dim = model_cfg.get("feat_dim")
    width = slide_info.get("width")
    height = slide_info.get("height")
    mpp = slide_info.get("mpp")
    physical_width_mm = round((float(width) * float(mpp)) / 1000.0, 2) if width and mpp else None
    physical_height_mm = round((float(height) * float(mpp)) / 1000.0, 2) if height and mpp else None

    details = {
        "mode": "demo_audit",
        "summary": "Collapsed details for explaining how each visible result was computed.",
        "pipeline": {
            "title": "End-to-end analysis pipeline",
            "summary": "This pipeline treats one whole-slide image as one slide-level diagnostic bag: many tissue tiles go in, one calibrated slide prediction plus safety and retrieval evidence comes out.",
            "clinical_context": [
                "The input is a digitized dermatopathology whole-slide image, typically an H&E-stained skin tissue section scanned at high resolution.",
                "The four displayed classes are Normal/Benign, basal cell carcinoma (BCC), squamous cell carcinoma (SCC), and melanoma.",
                "Because melanoma false negatives are clinically more dangerous than many false positives, the pipeline includes safety logic after classification."
            ],
            "technical_context": [
                "OpenSlide reads WSI metadata and pixel regions without loading the entire slide into memory.",
                "Tissue-containing tiles are sampled from the WSI, embedded by the selected foundation/CNN encoder, and aggregated by a gated-attention MIL classifier.",
                "Post-processing adds calibrated safety, OOD estimation, attention visualization, and case-based retrieval."
            ],
            "stages": [
                {
                    "stage": "OpenSlide metadata",
                    "description": "Read slide dimensions, scanner metadata, pyramid levels, and microns-per-pixel. This defines the coordinate system used by tiles and heatmaps.",
                    "outputs": {
                        "width": slide_info.get("width"),
                        "height": slide_info.get("height"),
                        "mpp": slide_info.get("mpp"),
                        "physical_width_mm_estimate": physical_width_mm,
                        "physical_height_mm_estimate": physical_height_mm,
                        "levels": slide_info.get("level_count"),
                        "vendor": slide_info.get("vendor"),
                    },
                },
                {
                    "stage": "Tile extraction",
                    "description": "The WSI is too large for a classifier directly, so it is converted into a bag of tissue tiles. Background-heavy tiles are dropped.",
                    "formula": "keep tile if tissue_fraction >= MIN_TISSUE_FRACTION; use at most MAX_TILES_FOR_ANALYSIS tiles",
                    "inputs": {
                        "wsi_width": width,
                        "wsi_height": height,
                    },
                    "outputs": {
                        "tile_size": cfg.TILE_SIZE,
                        "min_tissue_fraction": cfg.MIN_TISSUE_FRACTION,
                        "max_tiles_for_analysis": cfg.MAX_TILES_FOR_ANALYSIS,
                        "tiles_used": result.get("n_tiles"),
                    },
                },
                {
                    "stage": "Feature extraction",
                    "description": "Each retained RGB tile is transformed into a dense vector by the selected visual encoder. These vectors are the model's numerical view of tissue morphology.",
                    "formula": "tile RGB image -> encoder -> feature vector f_i",
                    "outputs": {
                        "model": result.get("model_used"),
                        "model_key": result.get("model_key"),
                        "model_kind": model_kind,
                        "feature_dim": model_feature_dim,
                        "ensemble_components": model_components,
                    },
                },
                {
                    "stage": "MIL aggregation",
                    "description": "Multiple Instance Learning aggregates many tile vectors into a single slide vector while learning which tiles deserve more weight.",
                    "formula": "h_i = encoder_head(f_i); a_i = softmax(W(tanh(Vh_i) * sigmoid(Uh_i))); z = sum_i a_i h_i; logits = classifier(z)",
                    "outputs": {
                        "prediction": result.get("raw_prediction"),
                        "display_prediction": result.get("prediction"),
                    },
                },
                {
                    "stage": "Safety layer",
                    "description": "The safety layer converts raw class probabilities into review-aware signals: uncertainty, narrow class margin, OOD shift, melanoma risk, and optional ensemble disagreement.",
                    "formula": "safety_score = mean(uncertainty, 1 - margin, OOD score, optional ensemble disagreement)",
                    "outputs": {
                        "risk_level": safety.get("risk_level"),
                        "decision_status": safety.get("decision_status"),
                        "safety_score": safety.get("safety_score"),
                    },
                },
                {
                    "stage": "Retrieval",
                    "description": "The final slide embedding is compared with a reference bank to retrieve similar historical/research cases for explanation.",
                    "formula": retrieval.get("details", {}).get("metric_formula"),
                    "outputs": {
                        "bank": retrieval.get("bank_display") or retrieval.get("bank_key"),
                        "bank_size": retrieval.get("bank_size"),
                        "top_k": len(retrieval.get("similar_cases") or []),
                    },
                },
            ],
            "replication_steps": [
                "Open the WSI with OpenSlide and record dimensions/mpp.",
                "Extract 256x256 tissue tiles and discard tiles below the tissue-fraction threshold.",
                "Encode each tile with the selected model family.",
                "Run the gated-attention MIL head to obtain probabilities, attention, and bag embedding.",
                "Apply safety, OOD, retrieval, and visualization post-processing."
            ],
        },
        "prediction": {
            "title": "Prediction calculation",
            "summary": "The prediction is the highest-probability class after the MIL model converts all selected tissue tiles into one slide-level probability vector.",
            "clinical_context": [
                "The displayed class is a decision-support output for the whole slide, not a substitute for a pathologist's final report.",
                "BCC, SCC, and melanoma can share visual patterns in some regions, so the probability margin is shown to expose close differential diagnoses."
            ],
            "technical_context": [
                "The model produces one logit per class. Softmax converts logits into probabilities that sum to 1.",
                "The raw prediction is argmax(probabilities). The displayed prediction can be overridden to Needs Expert Review by the safety layer."
            ],
            "formulae": [
                "P(class_i) = exp(logit_i) / sum_j exp(logit_j)",
                "prediction = argmax(P(class))",
                "confidence = max(P(class))",
                "margin = top1_probability - top2_probability",
                "display_prediction may be replaced by Needs Expert Review when safety abstain triggers",
            ],
            "inputs": {
                "probabilities": probabilities,
                "top1_class": top1[0],
                "top1_probability": round(float(top1[1]), 4),
                "top2_class": top2[0],
                "top2_probability": round(float(top2[1]), 4),
                "margin": round(margin, 4),
            },
            "outputs": {
                "raw_prediction": result.get("raw_prediction"),
                "display_prediction": result.get("prediction"),
                "decision_status": result.get("decision_status"),
            },
            "replication_steps": [
                "Collect the slide-level probability vector returned by MIL inference.",
                "Sort classes by probability.",
                "Use the highest-probability class as raw_prediction.",
                "Compute confidence and margin from the sorted probabilities.",
                "Pass raw_prediction and probabilities through the safety layer before displaying the final label."
            ],
            "limitations": [
                "A high probability means model confidence, not guaranteed biological truth.",
                "A low margin means the model sees evidence for multiple diagnostic classes and should be interpreted cautiously."
            ],
        },
        "attention": {
            "title": "Attention heatmap and top tile calculation",
            "summary": "Attention explains which tissue tiles most influenced the MIL bag representation used for slide classification.",
            "clinical_context": [
                "High-attention tiles should be reviewed as candidate diagnostically informative regions, such as tumor nests, atypical melanocytic proliferation, keratinizing squamous areas, or other discriminative morphology.",
                "Attention is not a pixel-level tumor segmentation mask; it is a model-weighted importance map at tile level."
            ],
            "technical_context": [
                "The MIL head uses gated attention: tanh and sigmoid gates are multiplied, projected to one score per tile, and normalized with softmax across the bag.",
                "For ensemble mode, consensus/shared attention is derived from component attention maps so the displayed regions represent agreement across models."
            ],
            "formulae": [
                "h_i = ReLU(W_encoder f_i + b_encoder)",
                "raw_attention_i = W_attn(tanh(V h_i) * sigmoid(U h_i))",
                "a_i = softmax(raw_attention_i over all tiles in the slide bag)",
                "z = sum_i a_i h_i",
                "visual_attention_i = minmax(a_i) for heatmap display",
                "top tiles are sorted by the active attention/shared-attention score",
                "heatmap stamps each tile score back onto a low-resolution WSI thumbnail",
            ],
            "inputs": {
                "top_tiles_mode": result.get("top_tiles_mode"),
                "heatmap_views": [view.get("key") for view in heatmap_views],
                "default_heatmap_view": result.get("default_heatmap_view"),
            },
            "outputs": {
                "top_tile_count": len(top_tiles),
                "heatmap_available": result.get("heatmap_available"),
                "top_tiles_title": result.get("top_tiles_title"),
            },
            "replication_steps": [
                "Keep the attention vector returned by the MIL forward pass.",
                "Normalize attention scores for visualization.",
                "Sort tile indices by the selected attention view.",
                "Save top tile crops and overlay the tile scores on a downsampled WSI thumbnail."
            ],
            "limitations": [
                "Attention highlights influential tiles, not necessarily all tumor tissue.",
                "A low-attention tile can still contain clinically relevant tissue; this is an explanation layer, not an exhaustive pathology annotation."
            ],
        },
    }

    if ensemble_details:
        vote_counts = {}
        for model_result in ensemble_details:
            pred = model_result.get("prediction")
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        majority_count = max(vote_counts.values()) if vote_counts else 0
        details["ensemble"] = {
            "title": "Ensemble aggregation calculation",
            "summary": "The ensemble combines multiple trained MIL models by averaging their probability vectors and reporting disagreement as a safety signal.",
            "clinical_context": [
                "Ensembling is useful when model families emphasize different histologic cues. Agreement raises confidence; disagreement exposes ambiguous or borderline morphology.",
                "A melanoma-sensitive ensemble can still abstain when one or more components produce clinically relevant melanoma evidence."
            ],
            "technical_context": [
                "Each component model runs its own encoder and MIL head on the same tile bag.",
                "The final probability vector is an arithmetic mean of component probabilities; the vote count is shown only as an interpretability aid.",
                "Ensemble disagreement is used by safety scoring because disagreement often correlates with uncertain decision boundaries."
            ],
            "formulae": [
                "P_ensemble(class) = mean(P_model_1(class), ..., P_model_n(class))",
                "ensemble_prediction = argmax(P_ensemble(class))",
                "majority_vote = most frequent per-model predicted class",
                "ensemble_disagreement = 1 - majority_vote_count / number_of_models",
            ],
            "inputs": {
                "models": [m.get("model") for m in ensemble_details],
                "per_model_predictions": {
                    m.get("model"): m.get("prediction") for m in ensemble_details
                },
                "per_model_probabilities": {
                    m.get("model"): m.get("probabilities") for m in ensemble_details
                },
            },
            "outputs": {
                "ensemble_probabilities": probabilities,
                "ensemble_prediction": result.get("raw_prediction"),
                "display_prediction": result.get("prediction"),
                "vote_counts": vote_counts,
                "majority_vote_count": majority_count,
                "ensemble_disagreement": safety.get("ensemble_disagreement"),
            },
            "replication_steps": [
                "Run each listed model on the same extracted tile bag.",
                "Store each model's class probabilities and argmax class.",
                "Average probabilities class-wise.",
                "Compute the ensemble argmax from averaged probabilities.",
                "Compute disagreement from per-model votes and pass it to the safety layer."
            ],
        }

    return details


def _build_phase1_safety(prediction, probabilities, ensemble_predictions=None):
    probs = np.asarray(probabilities, dtype=np.float32)
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = top1 - top2
    entropy_norm = _normalized_entropy(probs)
    melanoma_prob = float(probs[3])
    raw_prediction = CLASS_NAMES[int(prediction)]

    disagreement = None
    if ensemble_predictions:
        votes = [int(v) for v in ensemble_predictions]
        if votes:
            majority_votes = max(votes.count(v) for v in set(votes))
            disagreement = float(1.0 - (majority_votes / len(votes)))

    reasons = []
    if top1 < cfg.ABSTAIN_CONFIDENCE_THRESHOLD:
        reasons.append('Low top-class confidence')
    if entropy_norm >= cfg.HIGH_UNCERTAINTY_THRESHOLD:
        reasons.append('High predictive uncertainty')
    if margin < cfg.LOW_MARGIN_THRESHOLD:
        reasons.append('Narrow margin between top classes')
    if disagreement is not None and disagreement >= cfg.ENSEMBLE_DISAGREEMENT_THRESHOLD:
        reasons.append('High ensemble disagreement')

    melanoma_first_guard = raw_prediction != 'Melanoma' and melanoma_prob >= cfg.MELANOMA_BORDERLINE_PROB
    if melanoma_first_guard:
        reasons.append('Melanoma-first safeguard triggered')

    abstain_recommended = bool(
        melanoma_first_guard and (
            top1 < 0.75 or
            entropy_norm >= cfg.MODERATE_UNCERTAINTY_THRESHOLD or
            margin < 0.22 or
            (disagreement is not None and disagreement >= cfg.ENSEMBLE_DISAGREEMENT_THRESHOLD)
        )
    )

    if abstain_recommended:
        risk_level = 'urgent review recommended'
        recommendation = 'Do not finalize diagnosis automatically; send for expert review.'
        display_prediction = 'Needs Expert Review'
        decision_status = 'abstain'
        prediction_key = 'abstain'
    elif raw_prediction == 'Melanoma' or melanoma_prob >= cfg.MELANOMA_HIGH_RISK_PROB:
        risk_level = 'high risk'
        recommendation = 'Melanoma-sensitive review recommended.'
        display_prediction = raw_prediction
        decision_status = 'predicted'
        prediction_key = CLASS_KEYS[int(prediction)]
    elif reasons:
        risk_level = 'moderate risk'
        recommendation = 'Prediction available, but review caution flags before final use.'
        display_prediction = raw_prediction
        decision_status = 'predicted'
        prediction_key = CLASS_KEYS[int(prediction)]
    else:
        risk_level = 'low risk'
        recommendation = 'No Phase 1 safety warning triggered.'
        display_prediction = raw_prediction
        decision_status = 'predicted'
        prediction_key = CLASS_KEYS[int(prediction)]

    phase1_detail = {
        "title": "Phase 1 melanoma-sensitive safety calculation",
        "summary": "Phase 1 is a melanoma false-negative guard. It checks whether the raw predicted class should be trusted or converted into an expert-review recommendation.",
        "clinical_context": [
            "A missed melanoma is clinically more dangerous than sending an uncertain case to review, so melanoma probability receives asymmetric treatment.",
            "The guard is most relevant when the raw top class is BCC, SCC, or benign but P(Melanoma) remains above a borderline threshold."
        ],
        "technical_context": [
            "Uncertainty is normalized entropy, so it is comparable across the four-class output.",
            "Margin measures how separated the top two classes are; small margin means the classifier is close to changing its decision.",
            "The abstain rule is intentionally conjunctive: melanoma evidence must be present, then at least one weakness signal must make automatic finalization unsafe."
        ],
        "formulae": [
            "confidence = max(P(class))",
            "margin = top1_probability - top2_probability",
            "uncertainty = -sum_i p_i log(p_i) / log(number_of_classes)",
            f"melanoma_first_guard = raw_prediction != Melanoma and P(Melanoma) >= {cfg.MELANOMA_BORDERLINE_PROB:.2f}",
            f"abstain = melanoma_first_guard and (confidence < 0.75 or uncertainty >= {cfg.MODERATE_UNCERTAINTY_THRESHOLD:.2f} or margin < 0.22 or ensemble_disagreement >= {cfg.ENSEMBLE_DISAGREEMENT_THRESHOLD:.2f})",
        ],
        "inputs": {
            "raw_prediction": raw_prediction,
            "top1_class": CLASS_NAMES[int(order[0])],
            "top1_probability": round(top1, 4),
            "top2_class": CLASS_NAMES[int(order[1])] if len(order) > 1 else None,
            "top2_probability": round(top2, 4),
            "margin": round(margin, 4),
            "uncertainty": round(entropy_norm, 4),
            "melanoma_probability": round(melanoma_prob, 4),
            "ensemble_disagreement": None if disagreement is None else round(disagreement, 4),
        },
        "thresholds": {
            "abstain_confidence": cfg.ABSTAIN_CONFIDENCE_THRESHOLD,
            "high_uncertainty": cfg.HIGH_UNCERTAINTY_THRESHOLD,
            "moderate_uncertainty": cfg.MODERATE_UNCERTAINTY_THRESHOLD,
            "low_margin": cfg.LOW_MARGIN_THRESHOLD,
            "melanoma_borderline_probability": cfg.MELANOMA_BORDERLINE_PROB,
            "melanoma_high_risk_probability": cfg.MELANOMA_HIGH_RISK_PROB,
            "ensemble_disagreement": cfg.ENSEMBLE_DISAGREEMENT_THRESHOLD,
        },
        "outputs": {
            "melanoma_first_guard": melanoma_first_guard,
            "abstain_recommended": abstain_recommended,
            "risk_level": risk_level,
            "decision_status": decision_status,
            "display_prediction": display_prediction,
            "reasons": reasons,
        },
        "replication_steps": [
            "Sort the four class probabilities.",
            "Compute confidence from the top probability and margin from the top-two difference.",
            "Compute normalized entropy over all class probabilities.",
            "Check melanoma_first_guard using P(Melanoma) and raw_prediction.",
            "If the melanoma guard is active and confidence/margin/uncertainty/disagreement is unsafe, set decision_status to abstain."
        ],
    }

    return {
        'raw_prediction': raw_prediction,
        'display_prediction': display_prediction,
        'decision_status': decision_status,
        'prediction_key': prediction_key,
        'confidence': round(top1, 4),
        'margin': round(margin, 4),
        'uncertainty': round(entropy_norm, 4),
        'melanoma_probability': round(melanoma_prob, 4),
        'ensemble_disagreement': None if disagreement is None else round(disagreement, 4),
        'melanoma_first_guard': melanoma_first_guard,
        'abstain_recommended': abstain_recommended,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'reasons': reasons,
        'hard_case_candidate': bool(melanoma_first_guard or (raw_prediction == 'Melanoma' and top1 < 0.75)),
        'details': {
            'phase1': phase1_detail,
        },
    }


def _record_phase1_inference_case(job_id, slide_path, model_key, model_display, slide_info, result):
    safety = result.get('safety') or {}
    if not (safety.get('hard_case_candidate') or safety.get('abstain_recommended')):
        return

    record = {
        'job_id': job_id,
        'timestamp': result.get('timestamp'),
        'slide_path': slide_path,
        'filename': Path(slide_path).name,
        'model_key': model_key,
        'model_display': model_display,
        'prediction': result.get('prediction'),
        'raw_prediction': result.get('raw_prediction'),
        'prediction_key': result.get('prediction_key'),
        'decision_status': result.get('decision_status'),
        'probabilities': result.get('probabilities'),
        'safety': safety,
        'slide_info': slide_info,
    }

    case_dir = PHASE1_BANK_DIR / 'inference_cases'
    case_dir.mkdir(parents=True, exist_ok=True)
    case_path = case_dir / f'{job_id}.json'
    with open(case_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)

    manifest_path = PHASE1_BANK_DIR / 'inference_candidates.jsonl'
    with phase1_case_lock:
        with open(manifest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")


# ??????????????????????????????????????????????????? Heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


def _heatmap_asset_paths(job_id, variant="attention"):
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if variant in ("attention", "default", None):
        return out_dir / "heatmap.jpg", out_dir / "heatmap_only.png", out_dir / "thumbnail.jpg"

    safe_variant = variant.replace("/", "_").replace("\\", "_")
    return (
        out_dir / f"{safe_variant}_heatmap.jpg",
        out_dir / f"{safe_variant}_heatmap_only.png",
        out_dir / "thumbnail.jpg",
    )


def _generate_heatmap(slide, tile_coords, attention_weights, probabilities, job_id, variant="attention"):
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

        overlay_path, heat_only_path, thumb_path = _heatmap_asset_paths(job_id, variant)
        Image.fromarray(overlay_u8).save(str(overlay_path), quality=90)
        Image.fromarray((thumb_np * 255).astype(np.uint8)).save(str(thumb_path), quality=90)

        heat_rgba = np.zeros((hL, wL, 4), dtype=np.uint8)
        heat_rgba[:, :, :3] = (heat_color * 255).astype(np.uint8)
        visible_alpha = np.clip(heat_norm * 200, 0, 200).astype(np.uint8)
        heat_rgba[:, :, 3] = visible_alpha
        Image.fromarray(heat_rgba).save(str(heat_only_path))

        logger.info(f"Heatmap saved for {job_id} [{variant}] ({wL}Ã—{hL})")
        return str(overlay_path)

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


def _annotate_top_tiles(base_tiles, extra_fields):
    extra_fields = extra_fields or {}
    out = []
    for tile in base_tiles:
        tile_copy = dict(tile)
        idx = tile_copy.get("tile_index")
        for key, values in extra_fields.items():
            if values is None or idx is None or idx >= len(values):
                continue
            tile_copy[key] = round(float(values[idx]), 6)
        out.append(tile_copy)
    return out


def _default_heatmap_views(is_ensemble):
    base_keys = ["consensus", "disagreement", "shared"] if is_ensemble else ["attention"]
    return _build_heatmap_view_list(base_keys)


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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€ Models â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/api/models")
def list_models():
    """List available analysis models."""
    models = []
    for key, mcfg in MODEL_REGISTRY.items():
        ckpt_exists = Path(mcfg["mil_checkpoint"]).exists()
        threshold_policy = _build_threshold_policy(key)
        models.append({
            "key": key,
            "name": mcfg["name"],
            "display": mcfg["display"],
            "group": mcfg.get("group", "Other"),
            "f1": mcfg["f1"],
            "auc": mcfg["auc"],
            "mel_fn": mcfg.get("mel_fn", "?"),
            "description": mcfg["description"],
            "available": ckpt_exists,
            "threshold_policy": threshold_policy,
            "threshold_label": threshold_policy.get("label"),
        })
    # Sort by F1 descending
    models.sort(key=lambda x: x["f1"], reverse=True)

    # Ensemble presets
    ensembles = []
    for ekey, ecfg in ENSEMBLE_PRESETS.items():
        threshold_policy = _build_threshold_policy(ekey)
        ensembles.append({
            "key": ekey,
            "name": ecfg["name"],
            "display": ecfg["display"],
            "description": ecfg["description"],
            "models": ecfg["models"],
            "f1": ecfg["f1"],
            "auc": ecfg["auc"],
            "mel_fn": 0,
            "threshold_policy": threshold_policy,
            "threshold_label": threshold_policy.get("label"),
        })

    return jsonify({
        "models": models,
        "ensembles": ensembles,
        "default": "ensemble_3_best",
    })


# â”€â”€â”€â”€â”€â”€â”€â”€â”€ Upload â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    model_key = request.form.get("model", "ensemble_3_best")
    if not model_key.startswith("ensemble") and model_key not in MODEL_REGISTRY:
        model_key = "ensemble_3_best"

    job_id = str(uuid.uuid4())[:8]
    slide_dir = UPLOAD_DIR / job_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    slide_path = slide_dir / file.filename
    file.save(str(slide_path))

    file_size_mb = slide_path.stat().st_size / (1024 * 1024)
    if model_key in ENSEMBLE_PRESETS:
        model_display = ENSEMBLE_PRESETS[model_key]["name"]
    elif model_key in MODEL_REGISTRY:
        model_display = MODEL_REGISTRY[model_key]["display"]
    else:
        model_display = model_key
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€ Status â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/api/status/<job_id>")
def get_status(job_id):
    with analyses_lock:
        job = analyses.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    safe = {k: v for k, v in job.items() if k != "slide_path"}
    return jsonify(safe)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€ On-demand DZI tile serving â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€ Result assets â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/api/results/<job_id>/heatmap")
def serve_heatmap(job_id):
    p = RESULTS_DIR / job_id / "heatmap.jpg"
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="image/jpeg")

@app.route("/api/results/<job_id>/heatmap/<variant>")
def serve_heatmap_variant(job_id, variant):
    p, _, _ = _heatmap_asset_paths(job_id, variant)
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/jpeg")

@app.route("/api/results/<job_id>/heatmap_only")
def serve_heatmap_only(job_id):
    p = RESULTS_DIR / job_id / "heatmap_only.png"
    if not p.exists(): abort(404)
    return send_file(str(p), mimetype="image/png")

@app.route("/api/results/<job_id>/heatmap_only/<variant>")
def serve_heatmap_only_variant(job_id, variant):
    _, p, _ = _heatmap_asset_paths(job_id, variant)
    if not p.exists():
        abort(404)
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


@app.route("/api/retrieval/thumbnails/<slide_id>.jpg")
def serve_retrieval_thumbnail(slide_id):
    p = PHASE4_THUMB_DIR / f"{slide_id}.jpg"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/jpeg")


@app.route("/api/retrieval/cases/<slide_id>/compare")
def compare_retrieval_case(slide_id):
    case_meta = _get_retrieval_case_meta(slide_id)
    if not case_meta:
        return jsonify({"error": "Retrieval case not found"}), 404

    model_key = request.args.get("model", "ensemble_3_best")
    if model_key not in ENSEMBLE_PRESETS and model_key not in MODEL_REGISTRY:
        model_key = "ensemble_3_best"
    if model_key in ENSEMBLE_PRESETS:
        model_display = ENSEMBLE_PRESETS[model_key]["display"]
    else:
        model_display = MODEL_REGISTRY[model_key]["display"]

    job_id = _comparison_job_id(slide_id, model_key)
    with analyses_lock:
        job = analyses.get(job_id)

    if not job or job.get("slide_path") != case_meta["slide_path"] or not job.get("result"):
        with analyses_lock:
            analyses[job_id] = {
                "status": "queued",
                "progress": 0,
                "message": "Queued for retrieval comparison.",
                "filename": case_meta["filename"],
                "slide_path": case_meta["slide_path"],
                "model_key": model_key,
                "model_display": model_display,
                "created_at": datetime.now().isoformat(),
                "result": None,
            }
        run_analysis(job_id, case_meta["slide_path"], model_key)
        with analyses_lock:
            job = analyses.get(job_id)

    if not job or job.get("status") != "completed" or not job.get("result"):
        return jsonify({
            "error": "Comparison analysis failed",
            "status": (job or {}).get("status", "error"),
            "message": (job or {}).get("message", "Unknown error"),
        }), 500

    result = dict(job["result"])
    result.setdefault("artifacts", _build_result_artifacts(job_id, result))
    return jsonify({
        "job_id": job_id,
        "filename": case_meta["filename"],
        "slide_id": slide_id,
        "true_label": case_meta.get("true_label"),
        "source": case_meta.get("source"),
        "is_hard_melanoma": bool(case_meta.get("is_hard_melanoma")),
        "model_key": model_key,
        "model_display": model_display,
        "result": result,
    })


# â”€â”€â”€â”€â”€â”€â”€â”€â”€ History & Export â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        "model_key": job.get("model_key"),
        "result": job["result"],
        "slide_info": job.get("slide_info"),
        "decision_policy": {
            "abstain_confidence_threshold": cfg.ABSTAIN_CONFIDENCE_THRESHOLD,
            "high_uncertainty_threshold": cfg.HIGH_UNCERTAINTY_THRESHOLD,
            "moderate_uncertainty_threshold": cfg.MODERATE_UNCERTAINTY_THRESHOLD,
            "low_margin_threshold": cfg.LOW_MARGIN_THRESHOLD,
            "melanoma_borderline_probability": cfg.MELANOMA_BORDERLINE_PROB,
            "melanoma_high_risk_probability": cfg.MELANOMA_HIGH_RISK_PROB,
            "ensemble_disagreement_threshold": cfg.ENSEMBLE_DISAGREEMENT_THRESHOLD,
            "ood_strong_threshold": cfg.OOD_STRONG_THRESHOLD,
            "ood_moderate_threshold": cfg.OOD_MODERATE_THRESHOLD,
            "unified_safety_high": cfg.UNIFIED_SAFETY_HIGH,
            "unified_safety_moderate": cfg.UNIFIED_SAFETY_MODERATE,
        },
        "threshold_policy": (job["result"].get("threshold_policy") or _build_threshold_policy(job.get("model_key"))),
        "artifacts": job["result"].get("artifacts") or _build_result_artifacts(job_id, job["result"]),
        "retrieval_summary": (job["result"].get("retrieval") or {}),
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€ Info â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/api/info")
def server_info():
    upload_size = sum(f.stat().st_size for f in UPLOAD_DIR.rglob("*") if f.is_file())
    results_size = sum(f.stat().st_size for f in RESULTS_DIR.rglob("*") if f.is_file())
    with analyses_lock:
        n_jobs = len(analyses)
    retrieval_registry, _ = _load_phase4_registry()
    threshold_registry = _get_threshold_registry()
    threshold_count = 0
    if isinstance(threshold_registry, dict):
        if "models" in threshold_registry or "ensembles" in threshold_registry:
            threshold_count += len(threshold_registry.get("models", {}))
            threshold_count += len(threshold_registry.get("ensembles", {}))
        else:
            threshold_count = len(threshold_registry)
    return jsonify({
        "status": "ok",
        "jobs": n_jobs,
        "uploads_mb": round(upload_size / 1e6, 1),
        "results_mb": round(results_size / 1e6, 1),
        "n_models": len(MODEL_REGISTRY),
        "classes": list(CLASS_NAMES.values()),
        "retrieval_banks": sorted((retrieval_registry.get("banks") or {}).keys()),
        "phase0_threshold_registry_loaded": bool(threshold_count),
        "phase0_threshold_entries": threshold_count,
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
