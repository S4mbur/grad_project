#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import server


PHASE4_DIR = PROJECT_DIR / "results" / "phase4_retrieval"
THUMB_DIR = PHASE4_DIR / "thumbnails"
SOURCE_CSV = PROJECT_DIR / "results" / "phase1_hard_case_bank" / "all_test_predictions.csv"
HARD_CASE_CSV = PROJECT_DIR / "results" / "phase1_hard_case_bank" / "hard_case_bank.csv"
REGISTRY_JSON = PHASE4_DIR / "retrieval_registry.json"
EMBEDDINGS_NPZ = PHASE4_DIR / "retrieval_embeddings.npz"
FEATURE_ROOT = Path("/mnt/d/skin_cancer_project/cache")
DEFAULT_MODELS = [
    "uni_cost_sensitive_strong",
    "phikon_cost_sensitive_strong",
    "conch_cost_sensitive_strong",
]


def parse_args():
    p = argparse.ArgumentParser(description="Build Phase 4 similar-case retrieval artifacts.")
    p.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    p.add_argument("--hard-case-csv", type=Path, default=HARD_CASE_CSV)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--thumb-size", type=int, default=192)
    return p.parse_args()


def feature_dir_for_model(model_key: str) -> Path:
    cfg = server.MODEL_REGISTRY[model_key]
    mtype = cfg["type"]
    if mtype == "torchvision":
        loader = cfg.get("loader", "")
        if loader == "convnext_base":
            suffix = "convnext_base"
        elif loader == "convnext_small":
            suffix = "convnext_small"
        elif loader == "resnet50":
            suffix = "resnet50"
        elif loader == "resnet18":
            suffix = "resnet18"
        else:
            raise ValueError(f"Unsupported torchvision loader: {loader}")
    elif mtype == "dinov2":
        suffix = "dinov2_base"
    else:
        suffix = mtype
    return FEATURE_ROOT / f"features_4class_{suffix}"


def normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cases(source_csv: Path, hard_case_csv: Path):
    hard_rows = read_rows(hard_case_csv)
    hard_case_ids = {row["slide_id"] for row in hard_rows if row.get("true_label") == "Melanoma"}

    case_lookup = {}
    for row in read_rows(source_csv):
        slide_id = row["slide_id"]
        slide_path = Path(row["slide_path"])
        if slide_id in case_lookup:
            continue
        if not slide_path.exists():
            continue
        case_lookup[slide_id] = {
            "slide_id": slide_id,
            "filename": slide_path.name,
            "slide_path": str(slide_path),
            "true_label": row["true_label"],
            "source": row["source"],
            "is_hard_melanoma": slide_id in hard_case_ids,
            "thumbnail_url": f"/api/retrieval/thumbnails/{slide_id}.jpg",
        }
    ordered_cases = [case_lookup[k] for k in sorted(case_lookup)]
    return ordered_cases, hard_case_ids


def save_thumbnail(case, thumb_size: int):
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = THUMB_DIR / f"{case['slide_id']}.jpg"
    if out_path.exists():
        return out_path

    openslide = server._ensure_openslide()
    try:
        slide = openslide.OpenSlide(case["slide_path"])
        thumb = slide.get_thumbnail((thumb_size, thumb_size)).convert("RGB")
        slide.close()
    except Exception:
        thumb = Image.new("RGB", (thumb_size, thumb_size), color=(36, 44, 54))
    thumb.save(out_path, quality=90)
    return out_path


def slide_embedding(model, feat_path: Path):
    torch, device = server._ensure_torch()
    feats = torch.load(str(feat_path), map_location=device)
    feats = feats.to(device)
    with torch.no_grad():
        _, _, bag_embedding, _ = model(feats)
    return bag_embedding.detach().cpu().numpy().astype(np.float32).reshape(-1)


def build_model_vectors(cases, model_key: str):
    feature_dir = feature_dir_for_model(model_key)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Missing feature directory for {model_key}: {feature_dir}")

    model = server._get_mil_model(model_key)
    vectors = {}
    for idx, case in enumerate(cases, 1):
        feat_path = feature_dir / f"{case['slide_id']}.pt"
        if not feat_path.exists():
            continue
        vectors[case["slide_id"]] = normalize(slide_embedding(model, feat_path))
        if idx % 50 == 0:
            print(f"[{model_key}] {idx}/{len(cases)} cases processed")

    torch, _ = server._ensure_torch()
    server._mil_model_cache.pop(model_key, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectors


def build_ensemble_vectors(model_vectors, ensemble_key: str, component_models):
    case_sets = [set(model_vectors[mkey]) for mkey in component_models if mkey in model_vectors]
    if len(case_sets) != len(component_models) or not case_sets:
        return {}

    case_ids = sorted(set.intersection(*case_sets))
    vectors = {}
    for slide_id in case_ids:
        concat = np.concatenate([model_vectors[mkey][slide_id] for mkey in component_models], axis=0)
        vectors[slide_id] = normalize(concat)
    print(f"[{ensemble_key}] built from {len(case_ids)} shared cases")
    return vectors


def main():
    args = parse_args()
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    cases, hard_case_ids = load_cases(args.source_csv, args.hard_case_csv)
    for case in cases:
        save_thumbnail(case, args.thumb_size)

    model_keys = [m for m in args.models if m in server.MODEL_REGISTRY]
    print(f"Building Phase 4 retrieval bank for {len(cases)} cases and {len(model_keys)} models")

    model_vectors = {}
    arrays = {}
    banks = {}

    for model_key in model_keys:
        vectors = build_model_vectors(cases, model_key)
        model_vectors[model_key] = vectors
        case_ids = [case["slide_id"] for case in cases if case["slide_id"] in vectors]
        if not case_ids:
            continue
        arrays[model_key] = np.stack([vectors[slide_id] for slide_id in case_ids]).astype(np.float32)
        banks[model_key] = {
            "type": "single_model",
            "display": server.MODEL_REGISTRY[model_key]["display"],
            "case_ids": case_ids,
            "embedding_dim": int(arrays[model_key].shape[1]),
            "n_cases": int(len(case_ids)),
            "hard_case_count": int(sum(1 for slide_id in case_ids if slide_id in hard_case_ids)),
            "metric": "cosine",
        }
        print(f"[{model_key}] ready with {len(case_ids)} cases")

    for ensemble_key in ("ensemble_2_best", "ensemble_3_best"):
        if ensemble_key not in server.ENSEMBLE_PRESETS:
            continue
        component_models = server.ENSEMBLE_PRESETS[ensemble_key]["models"]
        if not all(model_key in model_vectors for model_key in component_models):
            continue
        vectors = build_ensemble_vectors(model_vectors, ensemble_key, component_models)
        if not vectors:
            continue
        case_ids = [case["slide_id"] for case in cases if case["slide_id"] in vectors]
        arrays[ensemble_key] = np.stack([vectors[slide_id] for slide_id in case_ids]).astype(np.float32)
        banks[ensemble_key] = {
            "type": "ensemble",
            "display": server.ENSEMBLE_PRESETS[ensemble_key]["display"],
            "component_models": component_models,
            "case_ids": case_ids,
            "embedding_dim": int(arrays[ensemble_key].shape[1]),
            "n_cases": int(len(case_ids)),
            "hard_case_count": int(sum(1 for slide_id in case_ids if slide_id in hard_case_ids)),
            "metric": "cosine",
        }

    np.savez_compressed(EMBEDDINGS_NPZ, **arrays)
    registry = {
        "generated_at": datetime.now().isoformat(),
        "source_csv": str(args.source_csv),
        "hard_case_csv": str(args.hard_case_csv),
        "thumbnail_dir": str(THUMB_DIR),
        "cases": {case["slide_id"]: case for case in cases},
        "hard_case_slide_ids": sorted(hard_case_ids),
        "banks": banks,
    }
    REGISTRY_JSON.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"Registry: {REGISTRY_JSON}")
    print(f"Embeddings: {EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()
