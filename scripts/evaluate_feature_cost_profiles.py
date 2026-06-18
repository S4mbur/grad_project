#!/usr/bin/env python3
"""Profile feature-extraction cost strategies from existing MIL predictions.

This script intentionally separates three questions:
1. Which ensembles are worth running when all model predictions are available?
2. Can sequential ensemble gating reduce the number of encoders that need to run?
3. How does tile budget multiply feature extraction cost?

It does not claim tile-budget accuracy unless tile-level features are available.
The current project stores most WSI/tile features on D:, so this script reports
whether real feature benchmarking is possible and otherwise emits proxy cost
curves that can be used to decide the next production integration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "phase9_feature_cost_profile"

LABELS = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
PROB_COLS = [
    "prob_normal_benign",
    "prob_bcc",
    "prob_scc",
    "prob_melanoma",
]

APP_THREE_ORDER = ["uni", "phikon", "conch"]
APP_PREFERRED_ORDER = [
    "uni",
    "phikon",
    "conch",
    "convnext_base",
    "convnext_small",
    "dinov2_base",
    "resnet50",
    "resnet18",
]

# These are policy thresholds over the currently averaged probability vector.
# They are inference-time observables and do not use the true label.
GATING_POLICIES = {
    "cheap_conf70_margin20_mel20": {
        "confidence_below": 0.70,
        "margin_below": 0.20,
        "mel_prob_at_least_if_not_mel": 0.20,
        "confirm_predicted_melanoma": False,
    },
    "balanced_conf80_margin35_mel10": {
        "confidence_below": 0.80,
        "margin_below": 0.35,
        "mel_prob_at_least_if_not_mel": 0.10,
        "confirm_predicted_melanoma": False,
    },
    "mel_safe_conf85_margin45_mel05": {
        "confidence_below": 0.85,
        "margin_below": 0.45,
        "mel_prob_at_least_if_not_mel": 0.05,
        "confirm_predicted_melanoma": False,
    },
    "conservative_conf90_margin55_mel03": {
        "confidence_below": 0.90,
        "margin_below": 0.55,
        "mel_prob_at_least_if_not_mel": 0.03,
        "confirm_predicted_melanoma": True,
    },
}

TILE_BUDGETS = [32, 64, 96, 128, 160, 200, 256, 320]
BASE_TILE_BUDGET = 200


@dataclass
class PredictionBank:
    model_keys: List[str]
    slide_ids: List[str]
    true_labels: List[str]
    probs: Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate feature-cost proxy, ensemble gating, and tile-budget cost curves."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Project results directory containing mil_4class_* prediction folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where CSV, plots, and markdown report will be written.",
    )
    parser.add_argument(
        "--prediction-glob",
        default="mil_4class_*_v3_fast_cost_sensitive_strong/phase1_test_predictions.csv",
        help="Glob under results-root for prediction CSVs.",
    )
    return parser.parse_args()


def model_key_from_path(path: Path) -> str:
    name = path.parent.name
    prefix = "mil_4class_"
    suffix = "_v3_fast_cost_sensitive_strong"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def load_prediction_bank(results_root: Path, prediction_glob: str) -> PredictionBank:
    csv_paths = sorted(results_root.glob(prediction_glob))
    if not csv_paths:
        raise FileNotFoundError(
            f"No prediction CSVs found under {results_root} with glob {prediction_glob!r}"
        )

    frames: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_paths:
        model_key = model_key_from_path(csv_path)
        df = pd.read_csv(csv_path)
        required = {"slide_id", "true_label", *PROB_COLS}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        df = df[["slide_id", "true_label", *PROB_COLS]].copy()
        df["slide_id"] = df["slide_id"].astype(str)
        frames[model_key] = df

    common_ids = None
    for df in frames.values():
        ids = set(df["slide_id"])
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        raise ValueError("Prediction CSVs have no common slide_id values.")

    first_key = sorted(frames)[0]
    base = frames[first_key][frames[first_key]["slide_id"].isin(common_ids)].copy()
    base = base.sort_values("slide_id")
    slide_ids = base["slide_id"].tolist()
    true_labels = base["true_label"].tolist()

    probs: Dict[str, np.ndarray] = {}
    for model_key, df in frames.items():
        aligned = df[df["slide_id"].isin(common_ids)].set_index("slide_id").loc[slide_ids]
        if aligned["true_label"].tolist() != true_labels:
            raise ValueError(f"True-label mismatch after alignment for model {model_key}")
        probs[model_key] = aligned[PROB_COLS].to_numpy(dtype=float)

    ordered_keys = [m for m in APP_PREFERRED_ORDER if m in probs]
    ordered_keys += [m for m in sorted(probs) if m not in ordered_keys]
    return PredictionBank(ordered_keys, slide_ids, true_labels, probs)


def pred_from_probs(probs: np.ndarray) -> List[str]:
    return [LABELS[int(i)] for i in np.argmax(probs, axis=1)]


def margin_from_probs(probs: np.ndarray) -> np.ndarray:
    ordered = np.sort(probs, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def compute_metrics(
    true_labels: Sequence[str],
    probs: np.ndarray,
    name: str,
    method_type: str,
    models: Sequence[str],
    avg_models_run: float,
    max_models_considered: int,
    stop_counts: Dict[int, int] | None = None,
) -> Dict[str, object]:
    preds = pred_from_probs(probs)
    y_true = list(true_labels)
    y_pred = preds
    melanoma_fn = sum(t == "Melanoma" and p != "Melanoma" for t, p in zip(y_true, y_pred))
    melanoma_total = sum(t == "Melanoma" for t in y_true)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    melanoma_recall = recall_score(
        y_true,
        y_pred,
        labels=["Melanoma"],
        average="macro",
        zero_division=0,
    )
    stop_counts = stop_counts or {len(models): len(y_true)}
    n = len(y_true)
    row: Dict[str, object] = {
        "name": name,
        "type": method_type,
        "models": "+".join(models),
        "num_models_available": len(models),
        "max_models_considered": max_models_considered,
        "avg_models_run": avg_models_run,
        "encoder_cost_ratio_vs_available": avg_models_run / max_models_considered,
        "encoder_cost_ratio_vs_3model_baseline": avg_models_run / 3.0,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "melanoma_recall": melanoma_recall,
        "melanoma_fn": melanoma_fn,
        "melanoma_total": melanoma_total,
        "mean_confidence": float(np.max(probs, axis=1).mean()),
        "mean_margin": float(margin_from_probs(probs).mean()),
    }
    for i in range(1, max_models_considered + 1):
        row[f"stop_after_{i}_pct"] = stop_counts.get(i, 0) / n
    return row


def average_model_probs(bank: PredictionBank, models: Sequence[str]) -> np.ndarray:
    arrays = [bank.probs[m] for m in models]
    return np.mean(arrays, axis=0)


def should_escalate(probs_1d: np.ndarray, policy: Dict[str, object]) -> bool:
    sorted_probs = np.sort(probs_1d)
    confidence = float(sorted_probs[-1])
    margin = float(sorted_probs[-1] - sorted_probs[-2])
    pred = LABELS[int(np.argmax(probs_1d))]
    mel_prob = float(probs_1d[LABELS.index("Melanoma")])

    if confidence < float(policy["confidence_below"]):
        return True
    if margin < float(policy["margin_below"]):
        return True
    if pred != "Melanoma" and mel_prob >= float(policy["mel_prob_at_least_if_not_mel"]):
        return True
    if bool(policy["confirm_predicted_melanoma"]) and pred == "Melanoma":
        return True
    return False


def run_gated_sequence(
    bank: PredictionBank,
    order: Sequence[str],
    policy_name: str,
    policy: Dict[str, object],
) -> Tuple[np.ndarray, float, Dict[int, int]]:
    n = len(bank.slide_ids)
    final_probs = np.zeros((n, len(LABELS)), dtype=float)
    invoked_counts: List[int] = []

    for i in range(n):
        running: List[np.ndarray] = []
        invoked = 0
        for model_key in order:
            running.append(bank.probs[model_key][i])
            invoked += 1
            current = np.mean(running, axis=0)
            if invoked == len(order):
                break
            if not should_escalate(current, policy):
                break
        final_probs[i] = np.mean(running, axis=0)
        invoked_counts.append(invoked)

    stop_counts: Dict[int, int] = {}
    for count in invoked_counts:
        stop_counts[count] = stop_counts.get(count, 0) + 1
    return final_probs, float(np.mean(invoked_counts)), stop_counts


def evaluate_all(bank: PredictionBank) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for model_key in bank.model_keys:
        rows.append(
            compute_metrics(
                bank.true_labels,
                bank.probs[model_key],
                name=f"single_{model_key}",
                method_type="single",
                models=[model_key],
                avg_models_run=1.0,
                max_models_considered=1,
            )
        )

    single_df = pd.DataFrame(rows)
    rank_df = single_df.sort_values(
        ["melanoma_fn", "macro_f1", "accuracy"],
        ascending=[True, False, False],
    )
    best_by_mel_then_macro = rank_df["models"].tolist()
    best_by_macro = single_df.sort_values("macro_f1", ascending=False)["models"].tolist()

    ensemble_specs: List[Tuple[str, List[str]]] = []
    available_app_three = [m for m in APP_THREE_ORDER if m in bank.probs]
    if len(available_app_three) >= 2:
        ensemble_specs.append(("ensemble_app_first2", available_app_three[:2]))
    if len(available_app_three) >= 3:
        ensemble_specs.append(("ensemble_app_first3", available_app_three[:3]))
    if len(best_by_macro) >= 2:
        ensemble_specs.append(("ensemble_top2_macro", best_by_macro[:2]))
    if len(best_by_macro) >= 3:
        ensemble_specs.append(("ensemble_top3_macro", best_by_macro[:3]))
    if len(best_by_mel_then_macro) >= 3:
        ensemble_specs.append(("ensemble_top3_mel_then_macro", best_by_mel_then_macro[:3]))
    if len(best_by_macro) >= 4:
        ensemble_specs.append(("ensemble_top4_macro", best_by_macro[:4]))
    if len(bank.model_keys) >= 2:
        ensemble_specs.append(("ensemble_all_available", bank.model_keys))

    seen_ensembles = set()
    for name, models in ensemble_specs:
        key = tuple(models)
        if key in seen_ensembles:
            continue
        seen_ensembles.add(key)
        probs = average_model_probs(bank, models)
        rows.append(
            compute_metrics(
                bank.true_labels,
                probs,
                name=name,
                method_type="full_ensemble",
                models=models,
                avg_models_run=float(len(models)),
                max_models_considered=len(models),
            )
        )

    order_specs: List[Tuple[str, List[str]]] = []
    if len(available_app_three) >= 2:
        order_specs.append(("gated_app_order", available_app_three))
    if len(best_by_macro) >= 3:
        order_specs.append(("gated_top3_macro_order", best_by_macro[:3]))
    if len(best_by_mel_then_macro) >= 3:
        order_specs.append(("gated_top3_mel_then_macro_order", best_by_mel_then_macro[:3]))
    if len(best_by_macro) >= 5:
        order_specs.append(("gated_top5_macro_order", best_by_macro[:5]))
    if len(bank.model_keys) >= 3:
        order_specs.append(("gated_all_app_preferred_order", bank.model_keys))

    seen_gated = set()
    for order_name, order in order_specs:
        key = tuple(order)
        if key in seen_gated:
            continue
        seen_gated.add(key)
        for policy_name, policy in GATING_POLICIES.items():
            probs, avg_run, stop_counts = run_gated_sequence(bank, order, policy_name, policy)
            rows.append(
                compute_metrics(
                    bank.true_labels,
                    probs,
                    name=f"{order_name}_{policy_name}",
                    method_type="sequential_gating",
                    models=order,
                    avg_models_run=avg_run,
                    max_models_considered=len(order),
                    stop_counts=stop_counts,
                )
            )

    return pd.DataFrame(rows)


def make_tile_budget_curve(gating_df: pd.DataFrame) -> pd.DataFrame:
    selected_names = []
    for candidate in [
        "single_uni",
        "ensemble_app_first2",
        "ensemble_app_first3",
        "ensemble_all_available",
        "gated_app_order_cheap_conf70_margin20_mel20",
        "gated_app_order_balanced_conf80_margin35_mel10",
        "gated_app_order_mel_safe_conf85_margin45_mel05",
        "gated_all_app_preferred_order_cheap_conf70_margin20_mel20",
        "gated_all_app_preferred_order_balanced_conf80_margin35_mel10",
        "gated_all_app_preferred_order_mel_safe_conf85_margin45_mel05",
    ]:
        if candidate in set(gating_df["name"]):
            selected_names.append(candidate)

    rows: List[Dict[str, object]] = []
    selected = gating_df[gating_df["name"].isin(selected_names)].copy()
    for _, item in selected.iterrows():
        for tile_budget in TILE_BUDGETS:
            tile_ratio = tile_budget / BASE_TILE_BUDGET
            rows.append(
                {
                    "name": item["name"],
                    "type": item["type"],
                    "models": item["models"],
                    "tile_budget": tile_budget,
                    "avg_models_run": item["avg_models_run"],
                    "avg_tile_encoder_calls": item["avg_models_run"] * tile_budget,
                    "tile_cost_ratio_vs_same_method_200_tiles": tile_ratio,
                    "feature_cost_ratio_vs_3model_200tile": (
                        item["avg_models_run"] * tile_budget
                    )
                    / (3 * BASE_TILE_BUDGET),
                    "feature_cost_ratio_vs_8model_200tile": (
                        item["avg_models_run"] * tile_budget
                    )
                    / (8 * BASE_TILE_BUDGET),
                    "macro_f1_at_200tile_prediction_proxy": item["macro_f1"],
                    "melanoma_fn_at_200tile_prediction_proxy": item["melanoma_fn"],
                }
            )
    return pd.DataFrame(rows)


def check_real_feature_benchmark_status() -> Dict[str, object]:
    data_root = Path(
        os.environ.get("SKINSIGHT_DATA_ROOT", "/mnt/d/skin_cancer_project/datasets")
    ).expanduser()
    models_root = Path(
        os.environ.get("SKINSIGHT_MODELS_ROOT", "/mnt/d/skin_cancer_project/models")
    ).expanduser()
    cache_root = Path(
        os.environ.get("SKINSIGHT_CACHE_ROOT", "/mnt/d/skin_cancer_project/cache")
    ).expanduser()
    checks = {
        "data_root_exists": data_root.exists(),
        "data_root_has_entries": data_root.exists() and any(data_root.iterdir()),
        "uni_weights": (models_root / "pathology/uni/pytorch_model.bin").exists(),
        "conch_weights": (models_root / "pathology/conch/pytorch_model.bin").exists(),
        "phikon_weights_dir": (models_root / "pathology/phikon").exists(),
        "tcga_skcm_dir": (data_root / "tcga_skcm").exists(),
        "feature_dir": cache_root.exists(),
    }
    checks["can_run_real_wsi_feature_benchmark"] = all(
        [
            checks["data_root_exists"],
            checks["data_root_has_entries"],
            checks["uni_weights"],
            checks["conch_weights"],
            checks["phikon_weights_dir"],
        ]
    )
    return checks


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_f):
        return ""
    return f"{value_f:.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 30) -> str:
    view = df.loc[:, columns].head(max_rows).copy()
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in view.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(fmt_float(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_gating_frontier(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.copy()
    plt.figure(figsize=(12, 7))
    colors = {
        "single": "#777777",
        "full_ensemble": "#1f77b4",
        "sequential_gating": "#d62728",
    }
    for method_type, group in plot_df.groupby("type"):
        plt.scatter(
            group["encoder_cost_ratio_vs_3model_baseline"],
            group["macro_f1"],
            s=80 + 35 * (group["melanoma_fn"].max() - group["melanoma_fn"] + 1),
            alpha=0.78,
            label=method_type,
            c=colors.get(method_type, "#333333"),
            edgecolors="white",
            linewidths=0.8,
        )
    top = plot_df.sort_values(["melanoma_fn", "macro_f1"], ascending=[True, False]).head(12)
    for _, row in top.iterrows():
        label = str(row["name"]).replace("gated_", "").replace("ensemble_", "ens_")
        plt.annotate(
            label[:42],
            (row["encoder_cost_ratio_vs_3model_baseline"], row["macro_f1"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("Encoder cost ratio vs current 3-model production baseline")
    plt.ylabel("Macro F1")
    plt.title("Feature-cost proxy frontier: full ensembles vs sequential gating")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tile_curve(tile_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    for name, group in tile_df.groupby("name"):
        plt.plot(
            group["tile_budget"],
            group["feature_cost_ratio_vs_3model_200tile"],
            marker="o",
            linewidth=1.8,
            label=name.replace("gated_", "g_").replace("ensemble_", "ens_")[:45],
        )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    plt.xlabel("Tile budget per invoked encoder")
    plt.ylabel("Feature cost ratio vs 3-model x 200-tile baseline")
    plt.title("Tile budget multiplies encoder-gating cost linearly")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    bank: PredictionBank,
    metrics_df: pd.DataFrame,
    tile_df: pd.DataFrame,
    status: Dict[str, object],
) -> None:
    sorted_metrics = metrics_df.sort_values(
        ["melanoma_fn", "macro_f1", "avg_models_run"],
        ascending=[True, False, True],
    )
    app3 = metrics_df[metrics_df["name"] == "ensemble_app_first3"]
    if not app3.empty:
        reference = app3.iloc[0]
    else:
        reference = sorted_metrics.iloc[0]

    suitable = metrics_df[
        (metrics_df["melanoma_fn"] <= reference["melanoma_fn"])
        & (metrics_df["macro_f1"] >= float(reference["macro_f1"]) - 0.01)
        & (metrics_df["encoder_cost_ratio_vs_3model_baseline"] <= 0.75)
    ].sort_values(
        ["melanoma_fn", "macro_f1", "avg_models_run"],
        ascending=[True, False, True],
    )

    md: List[str] = []
    md.append("# Phase 9 Feature-Cost and Ensemble-Gating Profile")
    md.append("")
    md.append("## Scope")
    md.append("")
    md.append(
        "This run evaluates retrieval-independent feature-cost controls. It uses existing "
        "MIL prediction CSVs to simulate which encoders would be invoked under different "
        "ensemble policies, and it computes tile-budget cost curves. It does not claim "
        "tile-budget accuracy unless tile-level features are available."
    )
    md.append("")
    md.append("## Available Prediction Bank")
    md.append("")
    md.append(f"- Slides in aligned test set: {len(bank.slide_ids)}")
    md.append(f"- Models loaded: {', '.join(bank.model_keys)}")
    md.append("")
    md.append("## Real WSI Feature Benchmark Status")
    md.append("")
    for key, value in status.items():
        md.append(f"- {key}: {value}")
    if not status["can_run_real_wsi_feature_benchmark"]:
        md.append("")
        md.append(
            "Real WSI feature extraction timing was skipped because the required D: "
            "artifacts are not visible from WSL. Mounting D: and rerunning this script "
            "can convert this from a proxy profile into a real encoder benchmark."
        )
    md.append("")
    md.append("## Best Methods by Safety-First Ranking")
    md.append("")
    md.append(
        dataframe_to_markdown(
            sorted_metrics,
            [
                "name",
                "type",
                "models",
                "avg_models_run",
                "encoder_cost_ratio_vs_3model_baseline",
                "accuracy",
                "macro_f1",
                "melanoma_recall",
                "melanoma_fn",
                "mean_margin",
            ],
            max_rows=25,
        )
    )
    md.append("")
    md.append("## Integration Candidates")
    md.append("")
    if suitable.empty:
        md.append(
            "No sequential/full method met the strict integration filter against the "
            "current 3-model reference. Keep the current production path until a real "
            "WSI feature benchmark is available."
        )
    else:
        md.append(
            "Candidates below match or improve melanoma FN against the current 3-model "
            "reference, stay within 0.01 macro-F1, and use <= 75% of the current "
            "UNI+Phikon+CONCH encoder budget."
        )
        md.append("")
        md.append(
            dataframe_to_markdown(
                suitable,
                [
                    "name",
                    "models",
                    "avg_models_run",
                    "encoder_cost_ratio_vs_3model_baseline",
                    "macro_f1",
                    "melanoma_recall",
                    "melanoma_fn",
                    "stop_after_1_pct",
                    "stop_after_2_pct",
                    "stop_after_3_pct",
                ],
                max_rows=15,
            )
        )
    md.append("")
    md.append("## Tile Budget Cost Profile")
    md.append("")
    md.append(
        "Feature extraction cost is approximately proportional to `tile_budget * "
        "number_of_invoked_encoders`. Therefore, reducing from 200 to 100 tiles halves "
        "encoder FLOPs for the same method, but accuracy must be validated with tile-level "
        "features before changing production defaults."
    )
    md.append("")
    if not tile_df.empty:
        md.append(
            dataframe_to_markdown(
                tile_df.sort_values(
                    ["feature_cost_ratio_vs_3model_200tile", "name", "tile_budget"]
                ),
                [
                    "name",
                    "tile_budget",
                    "avg_models_run",
                    "avg_tile_encoder_calls",
                    "feature_cost_ratio_vs_3model_200tile",
                    "macro_f1_at_200tile_prediction_proxy",
                    "melanoma_fn_at_200tile_prediction_proxy",
                ],
                max_rows=35,
            )
        )
    md.append("")
    md.append("## Recommendation")
    md.append("")
    if not suitable.empty:
        best = suitable.iloc[0]
        md.append(
            f"Use `{best['name']}` as the first production candidate after D: is mounted "
            "and a small real WSI timing test confirms that encoder load/switch overhead "
            "does not erase the predicted savings."
        )
    else:
        md.append(
            "Do not wire a production switch yet. First run a real WSI feature benchmark "
            "with mounted D: artifacts."
        )
    md.append("")
    md.append("## Output Files")
    md.append("")
    md.append("- gating_policy_results.csv")
    md.append("- tile_budget_cost_curve.csv")
    md.append("- real_feature_benchmark_status.json")
    md.append("- gating_cost_quality_frontier.png")
    md.append("- tile_budget_cost_curve.png")

    (output_dir / "feature_cost_profile.md").write_text("\n".join(md) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bank = load_prediction_bank(args.results_root, args.prediction_glob)
    metrics_df = evaluate_all(bank)
    tile_df = make_tile_budget_curve(metrics_df)
    status = check_real_feature_benchmark_status()

    metrics_df.to_csv(args.output_dir / "gating_policy_results.csv", index=False)
    tile_df.to_csv(args.output_dir / "tile_budget_cost_curve.csv", index=False)
    (args.output_dir / "real_feature_benchmark_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    plot_gating_frontier(metrics_df, args.output_dir / "gating_cost_quality_frontier.png")
    if not tile_df.empty:
        plot_tile_curve(tile_df, args.output_dir / "tile_budget_cost_curve.png")
    write_report(args.output_dir, bank, metrics_df, tile_df, status)

    print(f"Wrote profile outputs to {args.output_dir}")
    print(
        metrics_df.sort_values(
            ["melanoma_fn", "macro_f1", "avg_models_run"],
            ascending=[True, False, True],
        )
        .head(12)[
            [
                "name",
                "models",
                "avg_models_run",
                "encoder_cost_ratio_vs_3model_baseline",
                "macro_f1",
                "melanoma_recall",
                "melanoma_fn",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
