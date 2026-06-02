#!/usr/bin/env python3
"""Decision Curve Analysis for melanoma triage operating points.

The curve treats melanoma detection as the clinical action:

    intervene/review if P(Melanoma) >= threshold

Net benefit is:

    TP / N - FP / N * threshold / (1 - threshold)

This makes the cost of a false-positive review explicit as the threshold
odds.  It is useful for thesis defense because it translates probability
outputs into a clinical utility curve rather than another pure accuracy
metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.statistical_robustness_report import (  # noqa: E402
    GATED_POLICY,
    LABELS,
    build_method_probabilities,
    load_base_predictions,
    align_frames,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase11_decision_curve"
MELANOMA_INDEX = LABELS.index("Melanoma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=99)
    return parser.parse_args()


def net_benefit(y_true_mel: np.ndarray, p_mel: np.ndarray, threshold: float) -> Dict[str, float]:
    predicted_positive = p_mel >= threshold
    tp = int(np.sum(predicted_positive & y_true_mel))
    fp = int(np.sum(predicted_positive & ~y_true_mel))
    fn = int(np.sum(~predicted_positive & y_true_mel))
    tn = int(np.sum(~predicted_positive & ~y_true_mel))
    n = int(len(y_true_mel))
    odds = threshold / max(1.0 - threshold, 1e-12)
    nb = (tp / n) - (fp / n) * odds
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    review_rate = (tp + fp) / max(n, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "net_benefit": float(nb),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "review_rate": float(review_rate),
    }


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.head(max_rows).copy()
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


def plot_curves(curves_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=160)
    method_order = [
        "uni_cost_sensitive_strong",
        "phikon_cost_sensitive_strong",
        "conch_cost_sensitive_strong",
        "ensemble_2_best",
        "ensemble_3_best",
        GATED_POLICY["name"],
        "treat_all",
        "treat_none",
    ]
    colors = {
        GATED_POLICY["name"]: "#0b7285",
        "ensemble_3_best": "#5f3dc4",
        "ensemble_2_best": "#2b8a3e",
        "uni_cost_sensitive_strong": "#e67700",
        "phikon_cost_sensitive_strong": "#c92a2a",
        "conch_cost_sensitive_strong": "#495057",
        "treat_all": "#868e96",
        "treat_none": "#212529",
    }
    for method in method_order:
        sub = curves_df[curves_df["method"] == method]
        if sub.empty:
            continue
        style = "--" if method.startswith("treat_") else "-"
        width = 1.5 if method.startswith("treat_") else 2.0
        ax.plot(
            sub["threshold"],
            sub["net_benefit"],
            label=method.replace("_", " "),
            color=colors.get(method),
            linestyle=style,
            linewidth=width,
        )
    ax.set_title("Decision Curve Analysis: melanoma review action")
    ax.set_xlabel("Melanoma probability threshold")
    ax.set_ylabel("Net benefit")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_base_predictions()
    slide_ids, y_true, sources, probs_by_model = align_frames(frames)
    methods = build_method_probabilities(probs_by_model)
    y_true_mel = np.asarray(y_true) == "Melanoma"
    prevalence = float(np.mean(y_true_mel))
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)

    rows = []
    for threshold in thresholds:
        treat_all_nb = prevalence - (1.0 - prevalence) * threshold / max(1.0 - threshold, 1e-12)
        rows.append({
            "method": "treat_all",
            "threshold": float(threshold),
            "net_benefit": float(treat_all_nb),
            "tp": int(y_true_mel.sum()),
            "fp": int((~y_true_mel).sum()),
            "fn": 0,
            "tn": 0,
            "sensitivity": 1.0,
            "specificity": 0.0,
            "review_rate": 1.0,
        })
        rows.append({
            "method": "treat_none",
            "threshold": float(threshold),
            "net_benefit": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": int(y_true_mel.sum()),
            "tn": int((~y_true_mel).sum()),
            "sensitivity": 0.0,
            "specificity": 1.0,
            "review_rate": 0.0,
        })
        for method, probs in methods.items():
            p_mel = probs[:, MELANOMA_INDEX]
            rows.append({"method": method, "threshold": float(threshold), **net_benefit(y_true_mel, p_mel, float(threshold))})

    curves_df = pd.DataFrame(rows)
    curves_df.to_csv(args.output_dir / "decision_curve_analysis.csv", index=False)
    plot_curves(curves_df, args.output_dir / "decision_curve_analysis.png")

    model_rows = curves_df[~curves_df["method"].isin(["treat_all", "treat_none"])].copy()
    best_rows = model_rows.sort_values(["net_benefit", "sensitivity"], ascending=[False, False]).groupby("method", as_index=False).head(1)
    high_sens = model_rows[model_rows["sensitivity"] >= 0.99].copy()
    if not high_sens.empty:
        high_sens_best = high_sens.sort_values(["net_benefit", "review_rate"], ascending=[False, True]).groupby("method", as_index=False).head(1)
    else:
        high_sens_best = pd.DataFrame(columns=model_rows.columns)
    best_rows.to_csv(args.output_dir / "best_net_benefit_operating_points.csv", index=False)
    high_sens_best.to_csv(args.output_dir / "high_sensitivity_operating_points.csv", index=False)

    lines = [
        "# Phase 11 Decision Curve Analysis",
        "",
        f"Samples: {len(slide_ids)}. Melanoma prevalence: {prevalence:.4f}.",
        "",
        "Clinical action: send the case to melanoma-focused review when `P(Melanoma) >= threshold`.",
        "",
        "Net benefit formula:",
        "",
        "`NB(t) = TP/N - FP/N * t/(1-t)`",
        "",
        "The second term is the threshold-odds penalty for unnecessary melanoma review. This is why DCA is a better clinical utility figure than accuracy alone.",
        "",
        f"Figure: `{(args.output_dir / 'decision_curve_analysis.png').as_posix()}`",
        "",
        "## Best Net-Benefit Point Per Method",
        "",
        dataframe_to_markdown(best_rows.sort_values("net_benefit", ascending=False)),
        "",
        "## Best High-Sensitivity Point Per Method",
        "",
        dataframe_to_markdown(high_sens_best.sort_values("net_benefit", ascending=False)),
        "",
        "## Interpretation",
        "",
        "- Curves above treat false-negative melanoma as the critical event by scanning thresholds directly on melanoma probability.",
        "- A method is clinically useful where its net-benefit curve is above both treat-all and treat-none.",
        "- High-sensitivity rows show whether the operating point can keep melanoma misses near zero without reviewing every case.",
        "- This is still an internal test-set analysis; it supports clinical triage reasoning but does not replace external validation.",
    ]
    (args.output_dir / "decision_curve_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    print(best_rows[["method", "threshold", "net_benefit", "sensitivity", "specificity", "review_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
