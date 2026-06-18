#!/usr/bin/env python3
"""Evaluate post-hoc OOD baselines when cached features are available.

Implemented baselines:
- MSP: 1 - max softmax probability
- MaxLogit: -max raw logit
- Energy: -T logsumexp(logits / T), used as a high-is-OOD score
- Mahalanobis: distance to nearest training class centroid with pooled covariance

If OOD split features are missing, the script writes a readiness report instead
of fabricating AUROC numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import server  # noqa: E402
from scripts.reproducibility import set_global_seed  # noqa: E402


PHASE0_DIR = PROJECT_ROOT / "results" / "phase0_registry"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase10_ood_posthoc"

CACHE_ROOT = Path(
    os.environ.get("SKINSIGHT_CACHE_ROOT", "/mnt/d/skin_cancer_project/cache")
).expanduser()
MODEL_FEATURE_DIRS = {
    "uni_cost_sensitive_strong": CACHE_ROOT / "features_4class_uni",
    "phikon_cost_sensitive_strong": CACHE_ROOT / "features_4class_phikon",
    "conch_cost_sensitive_strong": CACHE_ROOT / "features_4class_conch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="uni_cost_sensitive_strong", choices=sorted(MODEL_FEATURE_DIRS))
    parser.add_argument("--max-train", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def split_manifest(split: str) -> pd.DataFrame:
    return pd.read_csv(PHASE0_DIR / "split_manifests" / f"{split}.csv")


def feature_path(feature_dir: Path, slide_id: str) -> Path:
    return feature_dir / f"{slide_id}.pt"


def available_rows(df: pd.DataFrame, feature_dir: Path) -> pd.DataFrame:
    mask = [feature_path(feature_dir, sid).exists() for sid in df["slide_id"].astype(str)]
    return df.loc[mask].copy()


def run_mil_on_cached_features(model_key: str, feature_file: Path) -> Dict[str, np.ndarray]:
    torch, device = server._ensure_torch()
    model = server._get_mil_model(model_key)
    features = torch.load(feature_file, map_location="cpu")
    if isinstance(features, dict):
        features = features.get("features") or features.get("x")
    features = features.to(device)
    with torch.no_grad():
        logits, attn, bag_embedding, tile_hidden = model(features)
        logits_np = logits.detach().cpu().numpy()[0].astype(np.float64)
        probs_np = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float64)
        bag_np = bag_embedding.detach().cpu().numpy().astype(np.float64)
    return {"logits": logits_np, "probs": probs_np, "embedding": bag_np}


def collect_outputs(model_key: str, rows: pd.DataFrame, feature_dir: Path) -> Dict[str, np.ndarray]:
    logits = []
    probs = []
    embeddings = []
    labels = []
    slide_ids = []
    for _, row in rows.iterrows():
        sid = str(row["slide_id"])
        out = run_mil_on_cached_features(model_key, feature_path(feature_dir, sid))
        logits.append(out["logits"])
        probs.append(out["probs"])
        embeddings.append(out["embedding"])
        labels.append(row.get("target_class"))
        slide_ids.append(sid)
    return {
        "slide_id": np.asarray(slide_ids),
        "label": np.asarray(labels),
        "logits": np.stack(logits, axis=0),
        "probs": np.stack(probs, axis=0),
        "embeddings": np.stack(embeddings, axis=0),
    }


def fit_mahalanobis(train_outputs: Dict[str, np.ndarray]) -> Dict[str, object]:
    labels = train_outputs["label"]
    embeddings = train_outputs["embeddings"]
    classes = sorted(label for label in set(labels) if isinstance(label, str) and label)
    centroids = {}
    residuals = []
    for label in classes:
        cls = embeddings[labels == label]
        if len(cls) < 2:
            continue
        centroid = cls.mean(axis=0)
        centroids[label] = centroid
        residuals.append(cls - centroid)
    residual_matrix = np.concatenate(residuals, axis=0)
    covariance = LedoitWolf().fit(residual_matrix)
    return {"classes": list(centroids), "centroids": centroids, "precision": covariance.precision_}


def mahalanobis_score(outputs: Dict[str, np.ndarray], model: Dict[str, object]) -> np.ndarray:
    scores = []
    precision = model["precision"]
    for emb in outputs["embeddings"]:
        distances = []
        for label in model["classes"]:
            delta = emb - model["centroids"][label]
            distances.append(float(delta @ precision @ delta.T))
        scores.append(min(distances))
    return np.asarray(scores, dtype=np.float64)


def ood_scores(outputs: Dict[str, np.ndarray], maha_model: Dict[str, object], temperature: float) -> Dict[str, np.ndarray]:
    logits = outputs["logits"]
    probs = outputs["probs"]
    energy = -temperature * np.log(np.exp(logits / temperature).sum(axis=1))
    return {
        "msp": 1.0 - probs.max(axis=1),
        "maxlogit": -logits.max(axis=1),
        "energy": energy,
        "mahalanobis": mahalanobis_score(outputs, maha_model),
    }


def fpr_at_tpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    candidates = fpr[tpr >= 0.95]
    return float(candidates[0]) if len(candidates) else 1.0


def evaluate_scores(id_scores: Dict[str, np.ndarray], ood_scores_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    y = np.concatenate([np.zeros_like(next(iter(id_scores.values()))), np.ones_like(next(iter(ood_scores_dict.values())))])
    for name in sorted(id_scores):
        scores = np.concatenate([id_scores[name], ood_scores_dict[name]])
        rows.append({
            "score": name,
            "auroc": float(roc_auc_score(y, scores)),
            "aupr_ood": float(average_precision_score(y, scores)),
            "fpr95": fpr_at_tpr95(y, scores),
            "id_mean": float(np.mean(id_scores[name])),
            "ood_mean": float(np.mean(ood_scores_dict[name])),
        })
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def write_readiness_report(output_dir: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Phase 10 Post-Hoc OOD Baseline Readiness",
        "",
        "This script evaluates Mahalanobis, Energy, MaxLogit, and MSP OOD baselines only when cached features exist for train/test/OOD splits.",
        "",
        "## Current Artifact Status",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    if not payload["can_evaluate"]:
        lines.extend([
            "## Result",
            "",
            "OOD AUROC was not computed because OOD split features are missing. Reporting numbers without OOD features would be misleading.",
            "",
            "## Next Step",
            "",
            "Run feature extraction for `results/phase0_registry/split_manifests/ood.csv` with the selected model cache, then rerun this script.",
        ])
    else:
        lines.extend([
            "## Result",
            "",
            "Required features are present; see `ood_posthoc_metrics.csv`.",
        ])
    (output_dir / "ood_posthoc_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = MODEL_FEATURE_DIRS[args.model_key]

    train_df = split_manifest("train")
    test_df = split_manifest("test")
    ood_df = split_manifest("ood")
    train_available = available_rows(train_df, feature_dir)
    test_available = available_rows(test_df, feature_dir)
    ood_available = available_rows(ood_df, feature_dir)
    payload = {
        "model_key": args.model_key,
        "feature_dir": str(feature_dir),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "ood_rows": int(len(ood_df)),
        "train_features_available": int(len(train_available)),
        "test_features_available": int(len(test_available)),
        "ood_features_available": int(len(ood_available)),
        "can_evaluate": bool(len(train_available) >= 20 and len(test_available) >= 20 and len(ood_available) >= 20),
    }
    (args.output_dir / "ood_posthoc_readiness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_readiness_report(args.output_dir, payload)

    if not payload["can_evaluate"]:
        print(f"Wrote readiness report to {args.output_dir}; OOD features are missing.")
        return

    train_available = train_available.sample(n=min(args.max_train, len(train_available)), random_state=args.seed)
    train_outputs = collect_outputs(args.model_key, train_available, feature_dir)
    id_outputs = collect_outputs(args.model_key, test_available, feature_dir)
    ood_outputs = collect_outputs(args.model_key, ood_available, feature_dir)
    maha_model = fit_mahalanobis(train_outputs)
    id_score_dict = ood_scores(id_outputs, maha_model, args.temperature)
    ood_score_dict = ood_scores(ood_outputs, maha_model, args.temperature)
    metrics = evaluate_scores(id_score_dict, ood_score_dict)
    metrics.to_csv(args.output_dir / "ood_posthoc_metrics.csv", index=False)
    write_readiness_report(args.output_dir, {**payload, "can_evaluate": True})
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
