#!/usr/bin/env python3
"""Generate bootstrap CIs, source-stratified metrics, and McNemar tests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.seed import set_global_seed


RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "phase10_statistical_robustness"

LABELS = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
PROB_COLS = ["prob_normal_benign", "prob_bcc", "prob_scc", "prob_melanoma"]

MODEL_SPECS = {
    "uni_cost_sensitive_strong": RESULTS_ROOT / "mil_4class_uni_v3_fast_cost_sensitive_strong" / "phase1_test_predictions.csv",
    "phikon_cost_sensitive_strong": RESULTS_ROOT / "mil_4class_phikon_v3_fast_cost_sensitive_strong" / "phase1_test_predictions.csv",
    "conch_cost_sensitive_strong": RESULTS_ROOT / "mil_4class_conch_v3_fast_cost_sensitive_strong" / "phase1_test_predictions.csv",
}

ENSEMBLE_SPECS = {
    "ensemble_2_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong"],
    "ensemble_3_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"],
}

GATED_POLICY = {
    "name": "gated_app_order_cheap_conf70_margin20_mel20",
    "order": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"],
    "confidence_below": 0.70,
    "margin_below": 0.20,
    "mel_prob_at_least_if_not_mel": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_base_predictions() -> Dict[str, pd.DataFrame]:
    frames = {}
    for name, path in MODEL_SPECS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"slide_id", "source", "true_label", *PROB_COLS}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames[name] = df.copy()
    return frames


def align_frames(frames: Dict[str, pd.DataFrame]) -> Tuple[List[str], pd.Series, pd.Series, Dict[str, np.ndarray]]:
    common = None
    for df in frames.values():
        ids = set(df["slide_id"].astype(str))
        common = ids if common is None else common & ids
    if not common:
        raise ValueError("No common slide_id values across model predictions.")

    first_name = sorted(frames)[0]
    base = frames[first_name][frames[first_name]["slide_id"].astype(str).isin(common)].copy()
    base["slide_id"] = base["slide_id"].astype(str)
    base = base.sort_values("slide_id")
    slide_ids = base["slide_id"].tolist()
    y_true = base["true_label"].reset_index(drop=True)
    sources = base["source"].reset_index(drop=True)

    probs = {}
    for name, df in frames.items():
        aligned = df.copy()
        aligned["slide_id"] = aligned["slide_id"].astype(str)
        aligned = aligned[aligned["slide_id"].isin(common)].set_index("slide_id").loc[slide_ids]
        if aligned["true_label"].tolist() != y_true.tolist():
            raise ValueError(f"True-label mismatch for {name}")
        probs[name] = aligned[PROB_COLS].to_numpy(dtype=np.float64)
    return slide_ids, y_true, sources, probs


def labels_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.asarray([LABELS[int(i)] for i in np.argmax(probs, axis=1)])


def build_method_probabilities(probs_by_model: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    methods = dict(probs_by_model)
    for name, models in ENSEMBLE_SPECS.items():
        methods[name] = np.mean([probs_by_model[m] for m in models], axis=0)
    methods[GATED_POLICY["name"]] = simulate_gated_probabilities(probs_by_model)
    return methods


def should_escalate(prob_1d: np.ndarray) -> bool:
    ordered = np.sort(prob_1d)
    confidence = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    pred = LABELS[int(np.argmax(prob_1d))]
    mel_prob = float(prob_1d[LABELS.index("Melanoma")])
    return (
        confidence < GATED_POLICY["confidence_below"]
        or margin < GATED_POLICY["margin_below"]
        or (pred != "Melanoma" and mel_prob >= GATED_POLICY["mel_prob_at_least_if_not_mel"])
    )


def simulate_gated_probabilities(probs_by_model: Dict[str, np.ndarray]) -> np.ndarray:
    order = GATED_POLICY["order"]
    n = probs_by_model[order[0]].shape[0]
    out = np.zeros_like(probs_by_model[order[0]])
    for i in range(n):
        running = []
        for pos, model_name in enumerate(order):
            running.append(probs_by_model[model_name][i])
            avg = np.mean(running, axis=0)
            if pos == len(order) - 1 or not should_escalate(avg):
                break
        out[i] = np.mean(running, axis=0)
    return out


def macro_auc(y_true: Sequence[str], probs: np.ndarray) -> float:
    y = np.asarray([LABELS.index(v) for v in y_true])
    try:
        return float(roc_auc_score(y, probs, labels=list(range(len(LABELS))), multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def metrics_for(y_true: Sequence[str], probs: np.ndarray) -> Dict[str, float]:
    pred = labels_from_probs(probs)
    y_true_arr = np.asarray(y_true)
    melanoma_fn = int(np.sum((y_true_arr == "Melanoma") & (pred != "Melanoma")))
    return {
        "n": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, pred)),
        "macro_f1": float(f1_score(y_true_arr, pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, pred, labels=LABELS, average="macro", zero_division=0)),
        "melanoma_recall": float(recall_score(y_true_arr, pred, labels=["Melanoma"], average="macro", zero_division=0)),
        "melanoma_precision": float(precision_score(y_true_arr, pred, labels=["Melanoma"], average="macro", zero_division=0)),
        "melanoma_fn": melanoma_fn,
        "macro_auc_ovr": macro_auc(y_true_arr, probs),
    }


def bootstrap_ci(y_true: Sequence[str], probs: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr = np.asarray(y_true)
    samples: Dict[str, List[float]] = {}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = metrics_for(y_true_arr[idx], probs[idx])
        for key, value in m.items():
            if key == "n":
                continue
            samples.setdefault(key, []).append(value)

    ci = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        ci[key] = (
            float(np.nanpercentile(arr, 2.5)),
            float(np.nanpercentile(arr, 97.5)),
        )
    return ci


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = int(b + c)
    if n == 0:
        return 1.0
    tail = min(b, c)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * prob))


def mcnemar_row(name_a: str, probs_a: np.ndarray, name_b: str, probs_b: np.ndarray, y_true: Sequence[str]) -> Dict[str, object]:
    y = np.asarray(y_true)
    pred_a = labels_from_probs(probs_a)
    pred_b = labels_from_probs(probs_b)
    correct_a = pred_a == y
    correct_b = pred_b == y
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    return {
        "model_a": name_a,
        "model_b": name_b,
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "mcnemar_exact_p": exact_mcnemar_pvalue(b, c),
    }


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, metrics_df: pd.DataFrame, source_df: pd.DataFrame, mcnemar_df: pd.DataFrame) -> None:
    best = metrics_df.sort_values(["melanoma_fn", "macro_f1"], ascending=[True, False]).head(12)
    lines = [
        "# Phase 10 Statistical Robustness Report",
        "",
        "This report adds paired statistical evidence without retraining: bootstrap confidence intervals, source-stratified subgroup metrics, and exact McNemar tests.",
        "",
        "## Overall Metrics with 95% Bootstrap CI",
        "",
        dataframe_to_markdown(best),
        "",
        "## Source-Stratified Metrics",
        "",
        dataframe_to_markdown(source_df.sort_values(["method", "source"]).head(80), max_rows=80),
        "",
        "## Paired McNemar Tests",
        "",
        dataframe_to_markdown(mcnemar_df),
        "",
        "## Interpretation Notes",
        "",
        "- Bootstrap CIs quantify test-set uncertainty for the 318-case paired evaluation.",
        "- Source-stratified rows expose whether performance is dominated by TCGA melanoma or COBRA-derived BCC/SCC/benign cases.",
        "- McNemar tests compare paired correctness patterns; they are more appropriate than independent tests because every method sees the same slides.",
        "- This does not replace external validation; it makes the current internal validation statistically more defensible.",
    ]
    (output_dir / "statistical_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_base_predictions()
    slide_ids, y_true, sources, probs_by_model = align_frames(frames)
    methods = build_method_probabilities(probs_by_model)

    metric_rows = []
    for name, probs in methods.items():
        row = {"method": name, **metrics_for(y_true, probs)}
        ci = bootstrap_ci(y_true, probs, args.bootstrap, args.seed)
        for metric, (lo, hi) in ci.items():
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows).sort_values(["melanoma_fn", "macro_f1"], ascending=[True, False])

    source_rows = []
    y_true_arr = np.asarray(y_true)
    source_arr = np.asarray(sources)
    for name, probs in methods.items():
        pred = labels_from_probs(probs)
        for source in sorted(set(source_arr)):
            mask = source_arr == source
            row = {"method": name, "source": source, **metrics_for(y_true_arr[mask], probs[mask])}
            cm = confusion_matrix(y_true_arr[mask], pred[mask], labels=LABELS)
            row["confusion_matrix_json"] = json.dumps(cm.tolist())
            source_rows.append(row)
    source_df = pd.DataFrame(source_rows)

    compare_pairs = [
        ("uni_cost_sensitive_strong", GATED_POLICY["name"]),
        ("ensemble_2_best", GATED_POLICY["name"]),
        ("ensemble_3_best", GATED_POLICY["name"]),
        ("uni_cost_sensitive_strong", "ensemble_3_best"),
    ]
    mcnemar_df = pd.DataFrame([
        mcnemar_row(a, methods[a], b, methods[b], y_true)
        for a, b in compare_pairs
        if a in methods and b in methods
    ])

    metrics_df.to_csv(args.output_dir / "metrics_with_bootstrap_ci.csv", index=False)
    source_df.to_csv(args.output_dir / "per_source_metrics.csv", index=False)
    mcnemar_df.to_csv(args.output_dir / "mcnemar_tests.csv", index=False)
    pd.DataFrame({"slide_id": slide_ids, "source": sources, "true_label": y_true}).to_csv(
        args.output_dir / "evaluation_slide_index.csv",
        index=False,
    )
    write_report(args.output_dir, metrics_df, source_df, mcnemar_df)
    print(f"Wrote {args.output_dir}")
    print(metrics_df[["method", "accuracy", "macro_f1", "melanoma_recall", "melanoma_fn", "macro_auc_ovr"]].to_string(index=False))


if __name__ == "__main__":
    main()
