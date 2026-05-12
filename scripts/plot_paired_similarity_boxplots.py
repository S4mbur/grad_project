#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_registry.json"
DEFAULT_EMBEDDINGS = PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_embeddings.npz"
DEFAULT_PREDICTIONS = PROJECT_DIR / "results" / "phase1_hard_case_bank" / "all_test_predictions.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "phase4_retrieval" / "paired_similarity_plots"

CLASS_ORDER = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
PLOT_CLASS_ORDER = ["Melanoma", "SCC", "BCC", "Normal/Benign"]
METRIC_ORDER = [
    "cosine",
    "sphere_geodesic",
    "rbf_kernel",
    "diagnostic_quotient_kernel",
    "aags_quotient",
    "trlq_quotient",
    "augmax_cosine",
]
METRIC_LABELS = {
    "cosine": "Cosine",
    "sphere_geodesic": "Geodesic",
    "rbf_kernel": "RBF",
    "diagnostic_quotient_kernel": "Quotient\nkernel",
    "aags_quotient": "AAGS-Q",
    "trlq_quotient": "TRLQ-Q",
    "augmax_cosine": "AugMax\ncosine",
}


def _load_retrieval_helpers():
    path = PROJECT_DIR / "scripts" / "evaluate_safe_r_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_safe_r_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


retr = _load_retrieval_helpers()


def parse_args():
    p = argparse.ArgumentParser(
        description="Build paired same-vs-other class similarity boxplots for retrieval metrics."
    )
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--banks",
        nargs="*",
        default=["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong", "ensemble_2_best", "ensemble_3_best"],
    )
    p.add_argument(
        "--augmented-embeddings",
        type=Path,
        default=None,
        help=(
            "Optional npz with arrays named '<bank>__rot90', '<bank>__rot180', "
            "'<bank>__flip'. Enables augmax_cosine."
        ),
    )
    return p.parse_args()


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def labels_for_bank(registry, bank_key):
    bank = registry["banks"][bank_key]
    cases = registry["cases"]
    case_ids = bank["case_ids"]
    return case_ids, np.asarray([cases[cid]["true_label"] for cid in case_ids], dtype=object)


def signals_and_components(bank_key, case_ids, labels, predictions, available_runs):
    signals = [
        retr.signals_for_model(bank_key, slide_id, labels[idx], predictions, available_runs)
        for idx, slide_id in enumerate(case_ids)
    ]
    signatures = np.stack([retr.clinical_signature(sig) for sig in signals], axis=0)
    axes = np.stack([retr.pathology_axes(sig) for sig in signals], axis=0)
    risk_ranks = np.asarray(
        [retr.risk_lattice_rank(sig, labels[idx]) for idx, sig in enumerate(signals)],
        dtype=np.float32,
    )
    return signals, signatures, axes, risk_ranks


def rbf_kernel_matrix(embeddings):
    x = np.asarray(embeddings, dtype=np.float32)
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0)
    off_diag = dist2[~np.eye(len(x), dtype=bool)]
    sigma2 = float(np.median(off_diag)) if len(off_diag) else 1.0
    sigma2 = max(sigma2, 1e-6)
    return np.exp(-dist2 / (2.0 * sigma2)).astype(np.float32)


def sphere_geodesic_similarity(cosine_matrix):
    clipped = np.clip(cosine_matrix, -1.0, 1.0)
    return (1.0 - np.arccos(clipped) / np.pi).astype(np.float32)


def quotient_kernel_matrix(quotient):
    coords = quotient["coords"]
    if quotient.get("dim", 0) <= 0:
        return np.ones((len(coords), len(coords)), dtype=np.float32)
    sq = np.sum(coords * coords, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * (coords @ coords.T), 0.0)
    scale2 = max(float(quotient.get("scale", 1.0)) ** 2, 1e-6)
    return np.exp(-dist2 / (2.0 * scale2)).astype(np.float32)


def algebraic_full_matrix(embeddings, labels, signatures, axes, quotient, risk_ranks, signals, mode):
    n = len(embeddings)
    out = np.full((n, n), np.nan, dtype=np.float32)
    all_idx = np.arange(n)
    if mode == "aags_quotient":
        weights = {
            "embedding": 0.24,
            "quotient": 0.18,
            "clinical": 0.15,
            "axis": 0.13,
            "tile": 0.09,
            "contrast": 0.10,
            "lattice": 0.07,
            "evidence": 0.04,
        }
    elif mode == "trlq_quotient":
        weights = {
            "embedding": 0.25,
            "quotient": 0.18,
            "clinical": 0.14,
            "axis": 0.12,
            "tile": 0.08,
            "contrast": 0.10,
            "lattice": 0.09,
            "evidence": 0.04,
        }
    else:
        raise ValueError(mode)

    for q_idx in range(n):
        candidates = all_idx[all_idx != q_idx]
        comps = retr.algebraic_component_scores(
            embeddings,
            labels.tolist(),
            signatures,
            axes,
            quotient,
            risk_ranks,
            q_idx,
            candidates,
            signals[q_idx],
        )
        if mode == "aags_quotient":
            score = np.ones(len(candidates), dtype=np.float32)
            for key, weight in weights.items():
                score *= np.power(comps[key], weight)
        else:
            cost = np.zeros(len(candidates), dtype=np.float32)
            for key, weight in weights.items():
                cost += weight * (-np.log(comps[key]))
            score = np.exp(-cost)
        out[q_idx, candidates] = score.astype(np.float32)
    return out


def augmax_cosine_matrix(bank_key, base_embeddings, augmented):
    if augmented is None:
        return None
    required = [f"{bank_key}__rot90", f"{bank_key}__rot180", f"{bank_key}__flip"]
    if not all(key in augmented for key in required):
        return None
    base = normalize_rows(base_embeddings)
    mats = [base @ base.T]
    for key in required:
        aug = normalize_rows(augmented[key])
        if aug.shape != base.shape:
            return None
        mats.append(base @ aug.T)
    return np.maximum.reduce(mats).astype(np.float32)


def topk_mean(values, k):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    k = min(int(k), len(values))
    return float(np.sort(values)[::-1][:k].mean())


def collect_paired_rows(bank_key, labels, metric_name, sim_matrix, top_k):
    rows = []
    n = len(labels)
    for q_idx in range(n):
        query_class = labels[q_idx]
        same_mask = labels == query_class
        same_mask[q_idx] = False
        other_mask = labels != query_class
        same_score = topk_mean(sim_matrix[q_idx, same_mask], top_k)
        other_score = topk_mean(sim_matrix[q_idx, other_mask], top_k)
        if np.isfinite(same_score) and np.isfinite(other_score):
            rows.append({
                "bank": bank_key,
                "metric": metric_name,
                "query_class": query_class,
                "query_index": q_idx,
                "same_topk_mean": same_score,
                "other_topk_mean": other_score,
                "delta_same_minus_other": same_score - other_score,
            })
    return rows


def summarize_pairs(df):
    rows = []
    for (bank, metric, query_class), group in df.groupby(["bank", "metric", "query_class"]):
        delta = group["delta_same_minus_other"].to_numpy(dtype=float)
        same = group["same_topk_mean"].to_numpy(dtype=float)
        other = group["other_topk_mean"].to_numpy(dtype=float)
        try:
            p = float(wilcoxon(same, other, alternative="greater").pvalue)
        except ValueError:
            p = np.nan
        rows.append({
            "bank": bank,
            "metric": metric,
            "query_class": query_class,
            "n_queries": int(len(group)),
            "same_mean": float(np.mean(same)),
            "other_mean": float(np.mean(other)),
            "delta_mean": float(np.mean(delta)),
            "delta_median": float(np.median(delta)),
            "delta_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
            "cohen_dz": float(np.mean(delta) / np.std(delta, ddof=1)) if len(delta) > 1 and np.std(delta, ddof=1) > 0 else np.nan,
            "wilcoxon_p_same_gt_other": p,
        })
    return pd.DataFrame(rows)


def plot_delta_boxplot(df, bank_key, out_dir):
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    sub = df[df["bank"] == bank_key].copy()
    order = [m for m in METRIC_ORDER if m in set(sub["metric"])]
    sub["metric_label"] = sub["metric"].map(METRIC_LABELS)
    label_order = [METRIC_LABELS[m] for m in order]
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=sub,
        x="metric_label",
        y="delta_same_minus_other",
        hue="query_class",
        order=label_order,
        hue_order=[c for c in PLOT_CLASS_ORDER if c in set(sub["query_class"])],
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=sub.sample(min(len(sub), 900), random_state=7),
        x="metric_label",
        y="delta_same_minus_other",
        hue="query_class",
        order=label_order,
        hue_order=[c for c in PLOT_CLASS_ORDER if c in set(sub["query_class"])],
        dodge=True,
        alpha=0.22,
        size=2.2,
        linewidth=0,
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color="#333333", linewidth=1.2, linestyle="--")
    ax.set_title(f"Paired same-class vs other-class similarity gap ({bank_key})")
    ax.set_xlabel("Similarity metric")
    ax.set_ylabel("mean top-k same-class similarity - other-class similarity")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Query class", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    path = out_dir / f"{bank_key}__paired_delta_boxplot.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_same_other_boxplot(df, bank_key, out_dir):
    sub = df[df["bank"] == bank_key].copy()
    order = [m for m in METRIC_ORDER if m in set(sub["metric"])]
    label_order = [METRIC_LABELS[m] for m in order]
    long_rows = []
    for row in sub.to_dict("records"):
        long_rows.append({
            "metric": METRIC_LABELS[row["metric"]],
            "query_class": row["query_class"],
            "group": "same class",
            "score": row["same_topk_mean"],
        })
        long_rows.append({
            "metric": METRIC_LABELS[row["metric"]],
            "query_class": row["query_class"],
            "group": "other classes",
            "score": row["other_topk_mean"],
        })
    long_df = pd.DataFrame(long_rows)
    classes = [c for c in PLOT_CLASS_ORDER if c in set(long_df["query_class"])]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=False)
    axes = axes.ravel()
    palette = {"same class": "#287c71", "other classes": "#c75b3a"}
    for ax, cls in zip(axes, classes):
        cls_df = long_df[long_df["query_class"] == cls]
        sns.boxplot(
            data=cls_df,
            x="metric",
            y="score",
            hue="group",
            order=label_order,
            showfliers=False,
            palette=palette,
            ax=ax,
        )
        ax.set_title(cls)
        ax.set_xlabel("")
        ax.set_ylabel("mean top-k similarity")
        ax.tick_params(axis="x", rotation=0)
        ax.legend_.remove()
    for ax in axes[len(classes):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Pair member", loc="upper center", ncol=2)
    fig.suptitle(f"Paired same-vs-other similarity distributions ({bank_key})", y=1.02)
    fig.tight_layout()
    path = out_dir / f"{bank_key}__same_vs_other_boxplot.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_heatmap(summary, bank_key, out_dir):
    sub = summary[summary["bank"] == bank_key].copy()
    pivot = sub.pivot(index="metric", columns="query_class", values="delta_median")
    row_order = [m for m in METRIC_ORDER if m in pivot.index]
    col_order = [c for c in PLOT_CLASS_ORDER if c in pivot.columns]
    pivot = pivot.loc[row_order, col_order]
    pivot.index = [METRIC_LABELS[m] for m in pivot.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "median paired gap"},
        ax=ax,
    )
    ax.set_title(f"Median same-other paired gap by class ({bank_key})")
    ax.set_xlabel("Query class")
    ax.set_ylabel("Metric")
    fig.tight_layout()
    path = out_dir / f"{bank_key}__median_delta_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(out_dir, paired_csv, summary_csv, plot_paths, top_k, aug_available):
    lines = [
        "# Paired Similarity Metric Boxplots",
        "",
        f"Top-k aggregation: `{top_k}` nearest candidates within each paired group.",
        "",
        "For every query slide, the script computes:",
        "",
        "```text",
        "same_score  = mean top-k similarity to slides with the same ground-truth class",
        "other_score = mean top-k similarity to slides with all other classes",
        "delta       = same_score - other_score",
        "```",
        "",
        "Positive delta means the metric separates same-class cases from other-class cases for that query.",
        "",
        "## Metrics",
        "",
        "- `cosine`: standard normalized embedding dot product.",
        "- `sphere_geodesic`: hypersphere geodesic similarity, `1 - arccos(cosine) / pi`.",
        "- `rbf_kernel`: RBF kernel on slide-level retrieval embeddings.",
        "- `diagnostic_quotient_kernel`: RBF kernel on model-induced diagnostic quotient coordinates.",
        "- `aags_quotient`: product/t-norm algebraic similarity with quotient component.",
        "- `trlq_quotient`: tropical-cost algebraic similarity converted back to `[0, 1]` via `exp(-cost)`.",
    ]
    if aug_available:
        lines.append("- `augmax_cosine`: max cosine over original, rot90, rot180, and flip candidate embeddings.")
    else:
        lines.extend([
            "- `augmax_cosine`: not computed in this run.",
            "",
            "Rotation/flip invariant cosine requires augmented embeddings created by re-running tile/slide feature extraction after geometric transforms. A vector itself cannot be rotated meaningfully after extraction.",
        ])
    lines.extend([
        "",
        "## Data Files",
        "",
        f"- Paired query-level rows: `{paired_csv}`",
        f"- Metric/class summary: `{summary_csv}`",
        "",
        "## Plots",
        "",
    ])
    for path in plot_paths:
        rel = path.relative_to(PROJECT_DIR)
        lines.append(f"- `{rel}`")
    lines.append("")
    (out_dir / "paired_similarity_plots.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    predictions, available_runs = retr.load_predictions(args.predictions)
    arrays = {}
    with np.load(args.embeddings, allow_pickle=False) as data:
        for key in data.files:
            arrays[key] = normalize_rows(data[key].astype(np.float32))

    augmented = None
    if args.augmented_embeddings is not None and args.augmented_embeddings.exists():
        with np.load(args.augmented_embeddings, allow_pickle=False) as data:
            augmented = {key: data[key].astype(np.float32) for key in data.files}

    all_rows = []
    aug_available = False
    for bank_key in args.banks:
        if bank_key not in arrays or bank_key not in registry.get("banks", {}):
            continue
        embeddings = arrays[bank_key]
        case_ids, labels = labels_for_bank(registry, bank_key)
        signals, signatures, axes, risk_ranks = signals_and_components(
            bank_key, case_ids, labels, predictions, available_runs
        )
        quotient = retr.build_diagnostic_quotient(embeddings, signatures)

        cosine = embeddings @ embeddings.T
        metrics = {
            "cosine": cosine.astype(np.float32),
            "sphere_geodesic": sphere_geodesic_similarity(cosine),
            "rbf_kernel": rbf_kernel_matrix(embeddings),
            "diagnostic_quotient_kernel": quotient_kernel_matrix(quotient),
            "aags_quotient": algebraic_full_matrix(
                embeddings, labels, signatures, axes, quotient, risk_ranks, signals, "aags_quotient"
            ),
            "trlq_quotient": algebraic_full_matrix(
                embeddings, labels, signatures, axes, quotient, risk_ranks, signals, "trlq_quotient"
            ),
        }
        augmax = augmax_cosine_matrix(bank_key, embeddings, augmented)
        if augmax is not None:
            metrics["augmax_cosine"] = augmax
            aug_available = True

        for matrix in metrics.values():
            np.fill_diagonal(matrix, np.nan)

        for metric_name, matrix in metrics.items():
            all_rows.extend(collect_paired_rows(bank_key, labels.copy(), metric_name, matrix, args.top_k))

    paired = pd.DataFrame(all_rows)
    paired["metric"] = pd.Categorical(paired["metric"], categories=METRIC_ORDER, ordered=True)
    paired["query_class"] = pd.Categorical(paired["query_class"], categories=PLOT_CLASS_ORDER, ordered=True)
    paired_csv = args.output / "paired_similarity_rows.csv"
    paired.to_csv(paired_csv, index=False)

    summary = summarize_pairs(paired)
    summary["metric"] = pd.Categorical(summary["metric"], categories=METRIC_ORDER, ordered=True)
    summary["query_class"] = pd.Categorical(summary["query_class"], categories=PLOT_CLASS_ORDER, ordered=True)
    summary = summary.sort_values(["bank", "metric", "query_class"])
    summary_csv = args.output / "paired_similarity_summary.csv"
    summary.to_csv(summary_csv, index=False)

    plot_paths = []
    for bank_key in sorted(paired["bank"].dropna().unique()):
        plot_paths.append(plot_delta_boxplot(paired, bank_key, args.output))
        plot_paths.append(plot_same_other_boxplot(paired, bank_key, args.output))
        plot_paths.append(plot_delta_heatmap(summary, bank_key, args.output))

    write_report(args.output, paired_csv, summary_csv, plot_paths, args.top_k, aug_available)
    print(f"Wrote {paired_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {args.output / 'paired_similarity_plots.md'}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
