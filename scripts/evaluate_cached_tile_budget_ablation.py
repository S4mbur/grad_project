#!/usr/bin/env python3
"""Evaluate MIL performance when using fewer cached tile features.

This is a fast proxy for the "why 200 tiles?" question.  It does not re-run
the expensive WSI encoders; instead it truncates each cached feature bag to
the first N tiles and runs the trained MIL heads.  Therefore:

* classification deltas are meaningful for bag-size sensitivity
* latency is MIL-head latency only
* feature-extraction cost is reported as a proportional tile-call proxy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase11_tile_budget_ablation"
MANIFEST_PATH = PROJECT_ROOT / "results" / "phase0_registry" / "split_manifests" / "test.csv"
CALIBRATION_PATH = PROJECT_ROOT / "results" / "phase2_safety" / "calibration_registry.json"
FEATURE_ROOT = Path("/mnt/d/skin_cancer_project/cache")

LABELS = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
MELANOMA_INDEX = LABELS.index("Melanoma")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    feat_dim: int
    feature_dir: Path
    checkpoint: Path
    temperature: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--budgets", nargs="+", type=int, default=[25, 50, 100, 150, 200])
    parser.add_argument("--sample-size", type=int, default=0, help="0 means all available test slides")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class GatedAttentionMIL(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int = 4, hidden_dim: int = 256, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        a = self.attention_W(self.attention_V(h) * self.attention_U(h))
        a = F.softmax(a, dim=0)
        z = torch.sum(a * h, dim=0, keepdim=True)
        logits = self.classifier(z)
        return logits


def load_temperature_registry() -> Dict[str, float]:
    if not CALIBRATION_PATH.exists():
        return {}
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return {k: float(v.get("temperature", 1.0)) for k, v in data.items()}


def build_specs() -> Dict[str, ModelSpec]:
    temps = load_temperature_registry()
    return {
        "uni_cost_sensitive_strong": ModelSpec(
            key="uni_cost_sensitive_strong",
            display="UNI",
            feat_dim=1024,
            feature_dir=FEATURE_ROOT / "features_4class_uni",
            checkpoint=PROJECT_ROOT / "results" / "mil_4class_uni_v3_fast_cost_sensitive_strong" / "best_model.pt",
            temperature=temps.get("uni_cost_sensitive_strong", 1.0),
        ),
        "phikon_cost_sensitive_strong": ModelSpec(
            key="phikon_cost_sensitive_strong",
            display="Phikon",
            feat_dim=768,
            feature_dir=FEATURE_ROOT / "features_4class_phikon",
            checkpoint=PROJECT_ROOT / "results" / "mil_4class_phikon_v3_fast_cost_sensitive_strong" / "best_model.pt",
            temperature=temps.get("phikon_cost_sensitive_strong", 1.0),
        ),
        "conch_cost_sensitive_strong": ModelSpec(
            key="conch_cost_sensitive_strong",
            display="CONCH",
            feat_dim=512,
            feature_dir=FEATURE_ROOT / "features_4class_conch",
            checkpoint=PROJECT_ROOT / "results" / "mil_4class_conch_v3_fast_cost_sensitive_strong" / "best_model.pt",
            temperature=temps.get("conch_cost_sensitive_strong", 1.0),
        ),
    }


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_models(specs: Dict[str, ModelSpec], device: torch.device) -> Dict[str, GatedAttentionMIL]:
    models = {}
    for key, spec in specs.items():
        model = GatedAttentionMIL(spec.feat_dim)
        state = safe_torch_load(spec.checkpoint, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models[key] = model
    return models


def load_features(spec: ModelSpec, slide_id: str) -> torch.Tensor | None:
    path = spec.feature_dir / f"{slide_id}.pt"
    if not path.exists():
        return None
    tensor = safe_torch_load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        return None
    return tensor.float()


def predict_probs(model: GatedAttentionMIL, features: torch.Tensor, temperature: float, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(features.to(device))
        probs = F.softmax(logits / max(float(temperature), 1e-6), dim=1).detach().cpu().numpy()[0]
    return probs.astype(np.float64)


def should_escalate(prob_1d: np.ndarray) -> bool:
    ordered = np.sort(prob_1d)
    confidence = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    pred = LABELS[int(np.argmax(prob_1d))]
    mel_prob = float(prob_1d[MELANOMA_INDEX])
    return confidence < 0.70 or margin < 0.20 or (pred != "Melanoma" and mel_prob >= 0.20)


def labels_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.asarray([LABELS[int(i)] for i in np.argmax(probs, axis=1)])


def metrics_row(method: str, budget: int, y_true: List[str], probs: np.ndarray, latencies: List[float], model_counts: List[int]) -> Dict[str, float | int | str]:
    pred = labels_from_probs(probs)
    y = np.asarray(y_true)
    return {
        "method": method,
        "budget": int(budget),
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "melanoma_recall": float(recall_score(y, pred, labels=["Melanoma"], average="macro", zero_division=0)),
        "melanoma_fn": int(np.sum((y == "Melanoma") & (pred != "Melanoma"))),
        "avg_mil_ms_per_slide": float(np.mean(latencies) * 1000.0),
        "avg_models_run": float(np.mean(model_counts)),
        "tile_call_proxy": float(np.mean(model_counts) * budget),
        "cost_ratio_vs_full_3model_200tile": float((np.mean(model_counts) * budget) / (3.0 * 200.0)),
    }


def evaluate_budget(
    manifest: pd.DataFrame,
    specs: Dict[str, ModelSpec],
    models: Dict[str, GatedAttentionMIL],
    budget: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_probs: Dict[str, List[np.ndarray]] = {"uni": [], "ensemble3": [], "gated": []}
    method_latencies: Dict[str, List[float]] = {"uni": [], "ensemble3": [], "gated": []}
    method_counts: Dict[str, List[int]] = {"uni": [], "ensemble3": [], "gated": []}
    y_true: List[str] = []
    detail_rows = []
    order = ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"]

    for _, row in manifest.iterrows():
        slide_id = str(row["slide_id"])
        true_label = str(row["target_class"])
        feature_bags = {}
        n_tiles_by_model = {}
        missing = False
        for key in order:
            feats = load_features(specs[key], slide_id)
            if feats is None:
                missing = True
                break
            n_use = min(int(budget), int(feats.shape[0]))
            if n_use <= 0:
                missing = True
                break
            feature_bags[key] = feats[:n_use]
            n_tiles_by_model[key] = n_use
        if missing:
            continue

        y_true.append(true_label)

        per_model_probs = {}
        per_model_latency = {}
        for key in order:
            t0 = time.perf_counter()
            per_model_probs[key] = predict_probs(models[key], feature_bags[key], specs[key].temperature, device)
            per_model_latency[key] = time.perf_counter() - t0

        uni_probs = per_model_probs[order[0]]
        method_probs["uni"].append(uni_probs)
        method_latencies["uni"].append(per_model_latency[order[0]])
        method_counts["uni"].append(1)

        ens_probs = np.mean([per_model_probs[k] for k in order], axis=0)
        method_probs["ensemble3"].append(ens_probs)
        method_latencies["ensemble3"].append(sum(per_model_latency[k] for k in order))
        method_counts["ensemble3"].append(3)

        running = []
        invoked = []
        gated_latency = 0.0
        for key in order:
            running.append(per_model_probs[key])
            invoked.append(key)
            gated_latency += per_model_latency[key]
            avg = np.mean(running, axis=0)
            if key == order[-1] or not should_escalate(avg):
                break
        gated_probs = np.mean(running, axis=0)
        method_probs["gated"].append(gated_probs)
        method_latencies["gated"].append(gated_latency)
        method_counts["gated"].append(len(invoked))

        detail_rows.append({
            "budget": int(budget),
            "slide_id": slide_id,
            "source": row.get("source", ""),
            "true_label": true_label,
            "uni_pred": LABELS[int(np.argmax(uni_probs))],
            "ensemble3_pred": LABELS[int(np.argmax(ens_probs))],
            "gated_pred": LABELS[int(np.argmax(gated_probs))],
            "gated_models_run": len(invoked),
            "gated_models": ",".join(invoked),
            "gated_p_mel": float(gated_probs[MELANOMA_INDEX]),
        })

    metric_rows = []
    for method, probs_list in method_probs.items():
        if not probs_list:
            continue
        metric_rows.append(metrics_row(
            method,
            budget,
            y_true,
            np.vstack(probs_list),
            method_latencies[method],
            method_counts[method],
        ))
    return pd.DataFrame(metric_rows), pd.DataFrame(detail_rows)


def plot_ablation(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160)
    colors = {"uni": "#e67700", "ensemble3": "#5f3dc4", "gated": "#0b7285"}
    for method, sub in metrics_df.groupby("method"):
        sub = sub.sort_values("budget")
        axes[0].plot(sub["budget"], sub["macro_f1"], marker="o", label=method, color=colors.get(method))
        axes[1].plot(sub["budget"], sub["melanoma_recall"], marker="o", label=method, color=colors.get(method))
        axes[2].plot(sub["budget"], sub["cost_ratio_vs_full_3model_200tile"], marker="o", label=method, color=colors.get(method))
    axes[0].set_title("Macro F1 vs tile budget")
    axes[1].set_title("Melanoma recall vs tile budget")
    axes[2].set_title("Proxy cost vs full 3x200")
    for ax in axes:
        ax.set_xlabel("Cached tiles per model")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Macro F1")
    axes[1].set_ylabel("Melanoma recall")
    axes[2].set_ylabel("Cost ratio")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    view = df.copy()
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["target_class"].isin(LABELS)].copy()
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    if args.sample_size and args.sample_size > 0 and args.sample_size < len(manifest):
        keep = rng.choice(len(manifest), size=args.sample_size, replace=False)
        manifest = manifest.iloc[np.sort(keep)].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = build_specs()
    models = load_models(specs, device)

    metric_frames = []
    detail_frames = []
    for budget in args.budgets:
        print(f"Evaluating budget={budget} on {len(manifest)} candidate slides")
        metrics, details = evaluate_budget(manifest, specs, models, int(budget), device)
        metric_frames.append(metrics)
        detail_frames.append(details)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    details_df = pd.concat(detail_frames, ignore_index=True)
    metrics_df.to_csv(args.output_dir / "cached_tile_budget_ablation.csv", index=False)
    details_df.to_csv(args.output_dir / "cached_tile_budget_predictions.csv", index=False)
    plot_ablation(metrics_df, args.output_dir / "cached_tile_budget_ablation.png")

    best = metrics_df.sort_values(["method", "budget"]).copy()
    lines = [
        "# Phase 11 Cached Tile-Budget Ablation",
        "",
        "This is a cached-feature proxy experiment. It truncates existing feature bags to different tile budgets and re-runs the trained MIL heads.",
        "",
        f"Device: `{device}`. Candidate test slides: {len(manifest)}.",
        "",
        f"Figure: `{(args.output_dir / 'cached_tile_budget_ablation.png').as_posix()}`",
        "",
        dataframe_to_markdown(best),
        "",
        "## Interpretation",
        "",
        "- The experiment answers whether the MIL classifier is sensitive to fewer tiles once features already exist.",
        "- The cost column is a proportional encoder-call proxy against a full 3-model, 200-tile baseline.",
        "- It does not measure WSI tile extraction or foundation-encoder wall time; the real WSI benchmark remains the production latency reference.",
        "- If a lower budget keeps melanoma recall stable, it becomes a candidate for the app after real WSI wall-time verification.",
    ]
    (args.output_dir / "cached_tile_budget_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
