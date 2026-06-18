#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import server

PHASE2_DIR = PROJECT_DIR / 'results' / 'phase2_safety'
PHASE2_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(
    os.environ.get('SKINSIGHT_DATA_ROOT', '/mnt/d/skin_cancer_project/datasets')
).expanduser()
FEATURE_ROOT = Path(
    os.environ.get('SKINSIGHT_CACHE_ROOT', '/mnt/d/skin_cancer_project/cache')
).expanduser()
MULTICLASS_MANIFEST = Path(
    os.environ.get(
        'SKINSIGHT_MULTICLASS_MANIFEST',
        str(DATA_ROOT / 'manifests' / 'multiclass_slide_manifest.csv'),
    )
).expanduser()
LABEL_TO_ID = {name: idx for idx, name in server.CLASS_NAMES.items()}


def feature_dir_for_model(model_key: str) -> Path:
    cfg = server.MODEL_REGISTRY[model_key]
    mtype = cfg['type']
    if mtype == 'torchvision':
        loader = cfg.get('loader', '')
        if loader == 'convnext_base':
            suffix = 'convnext_base'
        elif loader == 'convnext_small':
            suffix = 'convnext_small'
        elif loader == 'resnet50':
            suffix = 'resnet50'
        elif loader == 'resnet18':
            suffix = 'resnet18'
        else:
            raise ValueError(f'Unsupported torchvision loader: {loader}')
    elif mtype == 'dinov2':
        suffix = 'dinov2_base'
    else:
        suffix = mtype
    return FEATURE_ROOT / f'features_4class_{suffix}'


def softmax_temp(probabilities, temperature: float):
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs)
    scaled = logits / max(float(temperature), 1e-4)
    scaled -= scaled.max(axis=1, keepdims=True)
    exps = np.exp(scaled)
    return exps / exps.sum(axis=1, keepdims=True)


def nll(probs, labels):
    idx = np.arange(len(labels))
    return float(-np.mean(np.log(np.clip(probs[idx, labels], 1e-8, 1.0))))


def calibration_error(probs, labels, bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc = float(correctness[mask].mean())
        conf = float(confidences[mask].mean())
        gap = abs(acc - conf)
        ece += gap * (mask.sum() / len(labels))
        mce = max(mce, gap)
    return round(float(ece), 6), round(float(mce), 6)


def fit_temperature(probabilities, labels):
    candidates = np.concatenate([
        np.linspace(0.6, 2.0, 29),
        np.linspace(2.1, 4.0, 20),
    ])
    best_t = 1.0
    best_loss = nll(probabilities, labels)
    for temp in candidates:
        scaled = softmax_temp(probabilities, float(temp))
        loss = nll(scaled, labels)
        if loss < best_loss:
            best_loss = loss
            best_t = float(temp)
    return round(best_t, 4)


def load_phase1_predictions(csv_path: Path):
    probs = []
    labels = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(LABEL_TO_ID[row['true_label']])
            probs.append([
                float(row['prob_normal_benign']),
                float(row['prob_bcc']),
                float(row['prob_scc']),
                float(row['prob_melanoma']),
            ])
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def build_calibration_registry(model_keys):
    registry = {}
    for model_key in model_keys:
        run_dir = Path(server.MODEL_REGISTRY[model_key]['mil_checkpoint']).parent
        csv_path = run_dir / 'phase1_test_predictions.csv'
        if not csv_path.exists():
            print(f'[calibration] skip {model_key}: missing {csv_path}')
            continue
        probs, labels = load_phase1_predictions(csv_path)
        temp = fit_temperature(probs, labels)
        calibrated = softmax_temp(probs, temp)
        ece_before, mce_before = calibration_error(probs, labels)
        ece_after, mce_after = calibration_error(calibrated, labels)
        registry[model_key] = {
            'method': 'temperature_scaling',
            'temperature': temp,
            'ece_before': ece_before,
            'ece_after': ece_after,
            'mce_before': mce_before,
            'mce_after': mce_after,
            'source_run': run_dir.name,
            'source_csv': str(csv_path),
            'n_cases': int(len(labels)),
        }
        print(f'[calibration] {model_key}: T={temp:.4f} ECE {ece_before:.4f}->{ece_after:.4f}')
    out_path = PHASE2_DIR / 'calibration_registry.json'
    out_path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
    return out_path


def load_train_manifest():
    rows = []
    with MULTICLASS_MANIFEST.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') == 'train':
                rows.append(row)
    return rows


def slide_embedding(model, feat_path: Path):
    torch, device = server._ensure_torch()
    feats = torch.load(str(feat_path), map_location=device)
    feats = feats.to(device)
    with torch.no_grad():
        _, _, bag_embedding, _ = model(feats)
    return bag_embedding.detach().cpu().numpy().astype(np.float32)


def build_ood_registry(model_keys):
    train_rows = load_train_manifest()
    registry = {}
    torch, _ = server._ensure_torch()
    for model_key in model_keys:
        feature_dir = feature_dir_for_model(model_key)
        if not feature_dir.exists():
            print(f'[ood] skip {model_key}: missing {feature_dir}')
            continue
        model = server._get_mil_model(model_key)
        by_class = {name: [] for name in server.CLASS_NAMES.values()}
        used = 0
        for row in train_rows:
            feat_path = feature_dir / f"{row['slide_id']}.pt"
            if not feat_path.exists():
                continue
            class_name = server.CLASS_NAMES[int(row['label'])]
            emb = slide_embedding(model, feat_path)
            by_class[class_name].append(emb)
            used += 1
        centroids = {}
        thresholds = {}
        counts = {}
        for class_name, vectors in by_class.items():
            if not vectors:
                continue
            arr = np.stack(vectors)
            centroid = arr.mean(axis=0)
            dists = np.linalg.norm(arr - centroid, axis=1)
            centroids[class_name] = centroid.tolist()
            thresholds[class_name] = round(float(np.quantile(dists, 0.95)), 6)
            counts[class_name] = int(arr.shape[0])
        registry[model_key] = {
            'feature_dir': str(feature_dir),
            'embedding_dim': len(next(iter(centroids.values()))) if centroids else None,
            'class_centroids': centroids,
            'class_thresholds': thresholds,
            'train_counts': counts,
            'source_run': Path(server.MODEL_REGISTRY[model_key]['mil_checkpoint']).parent.name,
            'method': 'bag_embedding_nearest_centroid',
            'n_train_embeddings': used,
        }
        print(f'[ood] {model_key}: built centroids from {used} train slides')
        server._mil_model_cache.pop(model_key, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out_path = PHASE2_DIR / 'ood_registry.json'
    out_path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description='Build Phase 2 calibration and OOD artifacts for app inference.')
    p.add_argument('--models', nargs='*', default=sorted(server.MODEL_REGISTRY.keys()), help='Model keys to process')
    p.add_argument('--skip-calibration', action='store_true')
    p.add_argument('--skip-ood', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    model_keys = [m for m in args.models if m in server.MODEL_REGISTRY]
    print(f'Building Phase 2 artifacts for {len(model_keys)} models')
    if not args.skip_calibration:
        out = build_calibration_registry(model_keys)
        print(f'[done] calibration registry -> {out}')
    if not args.skip_ood:
        out = build_ood_registry(model_keys)
        print(f'[done] ood registry -> {out}')


if __name__ == '__main__':
    main()
