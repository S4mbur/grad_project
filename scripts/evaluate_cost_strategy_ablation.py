#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_registry.json"
DEFAULT_EMBEDDINGS = PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_embeddings.npz"
DEFAULT_PREDICTIONS = PROJECT_DIR / "results" / "phase1_hard_case_bank" / "all_test_predictions.csv"
DEFAULT_SAFE_R = PROJECT_DIR / "results" / "phase4_retrieval" / "safe_r_study" / "safe_r_study.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "phase8_cost_strategy_ablation"

CLASS_ORDER = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_ORDER)}

sys.path.insert(0, str(PROJECT_DIR))
from scripts.evaluate_safe_r_retrieval import (  # noqa: E402
    candidate_mask_for_safe_r,
    clinical_signature,
    clinical_similarity,
    fallback_signals,
    load_predictions,
    risk_tier,
    signals_for_model,
)


@dataclass
class SearchOutput:
    rankings: list[list[int]]
    avg_equiv_comparisons: float
    avg_raw_comparisons: float
    avg_memory_bytes_per_vector: float
    avg_ms_per_query: float
    notes: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate general and algebraic cost-reduction strategies for WSI retrieval.")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--safe-r", type=Path, default=DEFAULT_SAFE_R)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--banks", nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cascade-top-m", type=int, default=80)
    p.add_argument("--pq-m", type=int, default=16)
    p.add_argument("--pq-k", type=int, default=16)
    return p.parse_args()


def l2_normalise(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)
    return x / denom


def unit_cosine_scores(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    return np.clip((bank @ query + 1.0) / 2.0, 1e-6, 1.0)


def labels_for_bank(case_ids: list[str], registry: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    cases = registry.get("cases", {})
    labels = []
    label_names = []
    hard = []
    for cid in case_ids:
        meta = cases.get(cid, {})
        label = meta.get("true_label", "")
        label_names.append(label)
        labels.append(CLASS_INDEX.get(label, -1))
        hard.append(bool(meta.get("is_hard_melanoma")))
    return np.asarray(labels, dtype=np.int64), label_names, np.asarray(hard, dtype=bool)


def load_arrays(embeddings_path: Path, registry_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    with np.load(embeddings_path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    return arrays, registry


def build_signatures(
    bank_key: str,
    case_ids: list[str],
    label_names: list[str],
    predictions: dict,
    available_runs: set[str],
) -> np.ndarray:
    signatures = []
    for cid, label in zip(case_ids, label_names):
        sig = signals_for_model(bank_key, cid, label, predictions, available_runs)
        signatures.append(clinical_signature(sig))
    return np.asarray(signatures, dtype=np.float32)


def diagnostic_basis_from_labels(bank: np.ndarray, labels: np.ndarray) -> np.ndarray:
    valid = labels >= 0
    if valid.sum() < len(CLASS_ORDER):
        return np.zeros((bank.shape[1], 1), dtype=np.float32)
    center = bank[valid].mean(axis=0, keepdims=True)
    centroids = []
    for cls in range(len(CLASS_ORDER)):
        mask = labels == cls
        if mask.sum() == 0:
            centroids.append(center[0])
        else:
            centroids.append(bank[mask].mean(axis=0))
    directions = np.asarray(centroids, dtype=np.float32) - center
    _, sigma, vt = np.linalg.svd(directions, full_matrices=False)
    rank = int((sigma > max(float(sigma.max()) if sigma.size else 0.0, 1.0) * 1e-6).sum())
    rank = max(1, rank)
    return vt[:rank].T.astype(np.float32)


def diagnostic_basis_from_logreg(bank: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return diagnostic_basis_from_labels(bank, labels)
    valid = labels >= 0
    if valid.sum() < len(CLASS_ORDER):
        return diagnostic_basis_from_labels(bank, labels)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(bank[valid], labels[valid])
    W = np.zeros((len(CLASS_ORDER), bank.shape[1]), dtype=np.float32)
    for row, cls in enumerate(clf.classes_):
        W[int(cls)] = clf.coef_[row]
    _, sigma, vt = np.linalg.svd(W, full_matrices=False)
    rank = int((sigma > max(float(sigma.max()) if sigma.size else 0.0, 1.0) * 1e-6).sum())
    rank = max(1, rank)
    return vt[:rank].T.astype(np.float32)


def random_projection_matrix(dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 1.0 / math.sqrt(out_dim), size=(dim, out_dim)).astype(np.float32)
    return mat


def pca_projection(bank: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    out_dim = min(out_dim, bank.shape[0] - 1, bank.shape[1])
    pca = PCA(n_components=out_dim, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(bank).astype(np.float32)


def top_indices(scores: np.ndarray, query_idx: int, top_k: int | None = None) -> list[int]:
    scores = np.asarray(scores, dtype=np.float32).copy()
    scores[query_idx] = -np.inf
    order = np.argsort(scores)[::-1]
    if top_k is not None:
        order = order[:top_k]
    return order.astype(int).tolist()


def search_full_cosine(bank_norm: np.ndarray, top_k: int) -> SearchOutput:
    n, dim = bank_norm.shape
    t0 = time.perf_counter()
    rankings = [top_indices(bank_norm @ bank_norm[i], i, top_k) for i in range(n)]
    elapsed = time.perf_counter() - t0
    return SearchOutput(rankings, n - 1, n - 1, dim * 4, elapsed * 1000.0 / n, "Exact full cosine over float32 embeddings.")


def search_projected_cosine(name: str, coords: np.ndarray, full_dim: int, bytes_per_value: float, top_k: int) -> SearchOutput:
    coords = l2_normalise(coords)
    n, dim = coords.shape
    t0 = time.perf_counter()
    rankings = [top_indices(coords @ coords[i], i, top_k) for i in range(n)]
    elapsed = time.perf_counter() - t0
    equiv = (n - 1) * dim / full_dim
    return SearchOutput(rankings, equiv, n - 1, dim * bytes_per_value, elapsed * 1000.0 / n, name)


def search_int8_cosine(bank_norm: np.ndarray, top_k: int) -> SearchOutput:
    n, dim = bank_norm.shape
    qbank = np.round(np.clip(bank_norm, -1.0, 1.0) * 127).astype(np.int8)
    qbank_i = qbank.astype(np.int32)
    t0 = time.perf_counter()
    rankings = [top_indices(qbank_i @ qbank_i[i], i, top_k) for i in range(n)]
    elapsed = time.perf_counter() - t0
    # int8 dot is reported as 0.25 full-float equivalent: conservative proxy, not a hardware benchmark.
    equiv = (n - 1) * 0.25
    return SearchOutput(rankings, equiv, n - 1, dim, elapsed * 1000.0 / n, "Scalar int8 quantized cosine proxy.")


def search_binary_hash(bank_norm: np.ndarray, bits: int, seed: int, top_k: int) -> SearchOutput:
    n, dim = bank_norm.shape
    proj = random_projection_matrix(dim, bits, seed)
    codes = (bank_norm @ proj) >= 0
    t0 = time.perf_counter()
    rankings: list[list[int]] = []
    for i in range(n):
        same_bits = (codes == codes[i]).sum(axis=1).astype(np.float32)
        rankings.append(top_indices(same_bits, i, top_k))
    elapsed = time.perf_counter() - t0
    # Hamming compare on packed bits, expressed as rough float-equivalent comparison cost.
    equiv = (n - 1) * (bits / 32.0) / dim
    return SearchOutput(rankings, equiv, n - 1, bits / 8.0, elapsed * 1000.0 / n, f"{bits}-bit random-sign hash.")


def search_pq_asymmetric(bank_norm: np.ndarray, m: int, k: int, top_k: int, seed: int) -> SearchOutput:
    from sklearn.cluster import MiniBatchKMeans

    n, dim = bank_norm.shape
    m = min(m, dim)
    while dim % m != 0 and m > 1:
        m -= 1
    subdim = dim // m
    codes = np.zeros((n, m), dtype=np.int16)
    codebooks = []
    for part in range(m):
        sl = slice(part * subdim, (part + 1) * subdim)
        X = bank_norm[:, sl]
        km = MiniBatchKMeans(
            n_clusters=min(k, n),
            random_state=seed + part,
            n_init=3,
            batch_size=max(64, n),
            max_iter=100,
        )
        codes[:, part] = km.fit_predict(X)
        codebooks.append(km.cluster_centers_.astype(np.float32))

    t0 = time.perf_counter()
    rankings: list[list[int]] = []
    for qi in range(n):
        scores = np.zeros(n, dtype=np.float32)
        for part, centers in enumerate(codebooks):
            sl = slice(part * subdim, (part + 1) * subdim)
            lookup = centers @ bank_norm[qi, sl]
            scores += lookup[codes[:, part]]
        rankings.append(top_indices(scores, qi, top_k))
    elapsed = time.perf_counter() - t0

    precompute_ops = k * dim
    lookup_ops = (n - 1) * m
    equiv = (precompute_ops + lookup_ops) / dim
    bytes_per_vector = m if k <= 256 else 2 * m
    return SearchOutput(
        rankings,
        equiv,
        n - 1,
        bytes_per_vector,
        elapsed * 1000.0 / n,
        f"Product quantization ADC proxy, m={m}, k={k}.",
    )


def search_cascade(
    coarse_name: str,
    coarse_coords: np.ndarray,
    full_bank_norm: np.ndarray,
    top_m: int,
    top_k: int,
    bytes_per_value: float,
) -> SearchOutput:
    coarse = l2_normalise(coarse_coords)
    n, full_dim = full_bank_norm.shape
    dim = coarse.shape[1]
    top_m = min(top_m, n - 1)
    t0 = time.perf_counter()
    rankings: list[list[int]] = []
    for qi in range(n):
        coarse_scores = coarse @ coarse[qi]
        candidates = top_indices(coarse_scores, qi, top_m)
        full_scores = full_bank_norm[candidates] @ full_bank_norm[qi]
        order = np.asarray(candidates)[np.argsort(full_scores)[::-1]][:top_k]
        rankings.append(order.astype(int).tolist())
    elapsed = time.perf_counter() - t0
    equiv = ((n - 1) * dim + top_m * full_dim) / full_dim
    mem = dim * bytes_per_value + full_dim * 4
    return SearchOutput(
        rankings,
        equiv,
        top_m,
        mem,
        elapsed * 1000.0 / n,
        f"Cascade: {coarse_name} shortlist top-{top_m}, then exact full cosine rerank.",
    )


def search_selective_safe_r(
    bank_norm: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    signatures: np.ndarray,
    bank_key: str,
    case_ids: list[str],
    predictions: dict,
    available_runs: set[str],
    top_k: int,
) -> SearchOutput:
    n, dim = bank_norm.shape
    t0 = time.perf_counter()
    rankings: list[list[int]] = []
    costs = []
    for qi, (cid, label) in enumerate(zip(case_ids, label_names)):
        sig = signals_for_model(bank_key, cid, label, predictions, available_runs)
        tier = risk_tier(sig)
        mask = candidate_mask_for_safe_r(label_names, qi, sig, tier)
        cands = np.flatnonzero(mask)
        if cands.size == 0:
            rankings.append([])
            costs.append(0)
            continue
        scores = bank_norm[cands] @ bank_norm[qi]
        order = cands[np.argsort(scores)[::-1]][:top_k]
        rankings.append(order.astype(int).tolist())
        costs.append(int(cands.size))
    elapsed = time.perf_counter() - t0
    return SearchOutput(
        rankings,
        float(np.mean(costs)),
        float(np.mean(costs)),
        dim * 4,
        elapsed * 1000.0 / n,
        "Selective SAFE-R route using predicted risk tier and label-conditioned candidate pools.",
    )


def search_macs_clinical_cascade(
    bank_norm: np.ndarray,
    signatures: np.ndarray,
    top_m: int,
    top_k: int,
) -> SearchOutput:
    n, dim = bank_norm.shape
    sig_dim = signatures.shape[1]
    top_m = min(top_m, n - 1)
    t0 = time.perf_counter()
    rankings: list[list[int]] = []
    for qi in range(n):
        cs = clinical_similarity(signatures[qi], signatures)
        cands = top_indices(cs, qi, top_m)
        scores = bank_norm[cands] @ bank_norm[qi]
        order = np.asarray(cands)[np.argsort(scores)[::-1]][:top_k]
        rankings.append(order.astype(int).tolist())
    elapsed = time.perf_counter() - t0
    equiv = ((n - 1) * sig_dim + top_m * dim) / dim
    mem = sig_dim * 4 + dim * 4
    return SearchOutput(
        rankings,
        equiv,
        top_m,
        mem,
        elapsed * 1000.0 / n,
        f"General two-stage MACS: clinical signature top-{top_m}, then full cosine rerank.",
    )


def search_algebraic_upper_bound(
    bank_norm: np.ndarray,
    signatures: np.ndarray,
    quotient_coords: np.ndarray,
    top_k: int,
) -> SearchOutput:
    quotient = l2_normalise(quotient_coords)
    n, dim = bank_norm.shape
    qdim = quotient.shape[1]
    sig_dim = signatures.shape[1]
    t0 = time.perf_counter()
    rankings: list[list[int]] = []
    expensive_counts = []
    for qi in range(n):
        clinical = np.clip(clinical_similarity(signatures[qi], signatures), 1e-6, 1.0)
        qsim = unit_cosine_scores(quotient[qi], quotient)
        partial = np.power(clinical, 0.55) * np.power(qsim, 0.45)
        partial[qi] = -np.inf

        order = np.argsort(partial)[::-1]
        best: list[tuple[float, int]] = []
        kth = -np.inf
        expensive = 0
        for idx in order:
            if len(best) >= top_k and partial[idx] <= kth:
                break
            emb = float(np.clip((bank_norm[idx] @ bank_norm[qi] + 1.0) / 2.0, 1e-6, 1.0))
            # Product/t-norm style final score. Since emb <= 1, partial is an upper bound.
            score = float(partial[idx] * (emb ** 0.60))
            expensive += 1
            best.append((score, int(idx)))
            best.sort(reverse=True, key=lambda x: x[0])
            best = best[:top_k]
            if len(best) >= top_k:
                kth = best[-1][0]
        rankings.append([idx for _, idx in best])
        expensive_counts.append(expensive)
    elapsed = time.perf_counter() - t0
    cheap_ops = (n - 1) * (sig_dim + qdim)
    expensive_ops = float(np.mean(expensive_counts)) * dim
    equiv = (cheap_ops + expensive_ops) / dim
    mem = (sig_dim + qdim + dim) * 4
    return SearchOutput(
        rankings,
        equiv,
        float(np.mean(expensive_counts)),
        mem,
        elapsed * 1000.0 / n,
        "AAGS-style branch-and-bound: cheap clinical+quotient upper bound, full embedding only while candidate can enter top-k.",
    )


def evaluate_rankings(
    rankings: list[list[int]],
    labels: np.ndarray,
    hard_flags: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    valid = labels >= 0
    same_hits = []
    top1 = []
    mel_hit = []
    mel_p = []
    hard_first_ranks = []
    any_hit = []
    for qi, rank in enumerate(rankings):
        if not valid[qi]:
            continue
        rank = rank[:top_k]
        if not rank:
            same_hits.append(0.0)
            top1.append(0.0)
            any_hit.append(0.0)
            continue
        same = [1.0 if labels[j] == labels[qi] else 0.0 for j in rank]
        same_hits.append(float(np.mean(same)))
        top1.append(float(same[0]))
        any_hit.append(float(any(same)))
        if labels[qi] == CLASS_INDEX["Melanoma"]:
            mel_flags = [labels[j] == CLASS_INDEX["Melanoma"] for j in rank]
            mel_hit.append(float(any(mel_flags)))
            mel_p.append(float(np.mean(mel_flags)))
        if hard_flags[qi] and labels[qi] == CLASS_INDEX["Melanoma"]:
            first = None
            for pos, j in enumerate(rank, start=1):
                if labels[j] == CLASS_INDEX["Melanoma"]:
                    first = pos
                    break
            hard_first_ranks.append(float(first if first is not None else top_k + 1))
    return {
        "same_label_p@k": float(np.mean(same_hits)) if same_hits else 0.0,
        "same_label_top1": float(np.mean(top1)) if top1 else 0.0,
        "same_label_hit@k": float(np.mean(any_hit)) if any_hit else 0.0,
        "melanoma_hit@k": float(np.mean(mel_hit)) if mel_hit else 0.0,
        "melanoma_p@k": float(np.mean(mel_p)) if mel_p else 0.0,
        "hard_mel_first_rank@k": float(np.mean(hard_first_ranks)) if hard_first_ranks else float("nan"),
    }


def plot_frontier(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        "exact": "#2f4858",
        "compression": "#33658a",
        "hash": "#86bbd8",
        "pq": "#f6ae2d",
        "cascade": "#f26419",
        "selective": "#7a306c",
        "algebraic": "#00876c",
    }
    for family, sub in df.groupby("family"):
        ax.scatter(
            sub["cost_ratio"],
            sub["same_label_p@5"],
            s=80,
            label=family,
            alpha=0.85,
            color=colors.get(family, None),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Estimated cost ratio vs full cosine (lower is better)")
    ax.set_ylabel("Same-label precision@5")
    ax.set_title("Cost-quality frontier across retrieval cost-reduction strategies")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "cost_quality_frontier.png", dpi=220)
    plt.close(fig)


def plot_melanoma(df: pd.DataFrame, output: Path) -> None:
    methods = [
        "full_cosine",
        "int8_full_cosine",
        "pca64_full",
        "quotient_logreg_full",
        "pq_adc",
        "cascade_quotient80_full",
        "macs_clinical80_full",
        "aags_upper_bound_prune",
        "safe_r_selective",
    ]
    sub = df[df["method"].isin(methods)].copy()
    if sub.empty:
        return
    pivot = sub.groupby("method")["melanoma_hit@5"].mean().reindex(methods).dropna()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(pivot.index, pivot.values, color="#d62828")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Melanoma hit@5")
    ax.set_title("Melanoma retrieval preservation under cost reduction")
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(pivot.values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "melanoma_hit_by_method.png", dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, output: Path, safe_r_path: Path) -> None:
    agg = (
        df.groupby(["method", "family"], as_index=False)
        .agg(
            same_label_p5=("same_label_p@5", "mean"),
            melanoma_hit5=("melanoma_hit@5", "mean"),
            melanoma_p5=("melanoma_p@5", "mean"),
            hard_mel_rank=("hard_mel_first_rank@5", "mean"),
            cost_ratio=("cost_ratio", "mean"),
            equiv_comparisons=("equiv_comparisons", "mean"),
            memory_bytes=("memory_bytes_per_vector", "mean"),
            ms_per_query=("ms_per_query", "mean"),
        )
        .sort_values(["same_label_p5", "cost_ratio"], ascending=[False, True])
    )
    efficient = agg[(agg["same_label_p5"] >= agg.loc[agg["method"] == "full_cosine", "same_label_p5"].mean() - 0.005)]
    efficient = efficient.sort_values("cost_ratio").head(10)

    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in frame.iterrows():
            vals = []
            for col in headers:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Phase 8 Cost Strategy Ablation",
        "",
        "This is a local ablation over the existing Phase 4 retrieval bank. It does not retrain encoders and does not run full WSI feature extraction.",
        "",
        "## Files",
        "",
        f"- Per-bank rows: `{output / 'cost_strategy_ablation.csv'}`",
        f"- Aggregated rows: `{output / 'cost_strategy_aggregate.csv'}`",
        f"- Cost-quality frontier: `{output / 'cost_quality_frontier.png'}`",
        f"- Melanoma preservation plot: `{output / 'melanoma_hit_by_method.png'}`",
        "",
        "## Method Families",
        "",
        "- `exact`: full float32 cosine baseline.",
        "- `compression`: PCA, random projection, int8 and classifier-induced quotient full scans.",
        "- `pq`: product-quantized asymmetric distance computation proxy.",
        "- `cascade`: cheap compressed/quotient shortlist followed by full embedding rerank.",
        "- `selective`: risk/clinical routing similar to SAFE-R/MACS.",
        "- `algebraic`: branch-and-bound using a product/t-norm upper bound over clinical and quotient evidence.",
        "",
        "## Aggregate Results",
        "",
        md_table(agg),
        "",
        "## Most Efficient Rows Within 0.005 Same-label P@5 of Full Cosine",
        "",
        md_table(efficient) if not efficient.empty else "No method matched the full-cosine quality window.",
        "",
        "## Initial Interpretation",
        "",
        "1. Methods with low cost ratio and stable melanoma hit@5 are candidates for real app integration.",
        "2. If compression-only methods preserve quality, they reduce general compute independently of selective routing.",
        "3. If cascade methods preserve quality, they are stronger than pure selective routing because they lower the cost for every query.",
        "4. If algebraic upper-bound pruning preserves quality, it gives the cleanest bridge between abstract algebra and computation: the product/tropical score supplies a safe early-stop bound.",
        "5. Product quantization is a proxy here because the bank is small. It becomes more meaningful when the retrieval bank reaches tens or hundreds of thousands of slides.",
        "",
        f"Existing SAFE-R study for comparison: `{safe_r_path}`",
    ]
    (output / "cost_strategy_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    agg.to_csv(output / "cost_strategy_aggregate.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    arrays, registry = load_arrays(args.embeddings, args.registry)
    predictions, available_runs = load_predictions(args.predictions)

    banks = args.banks or list(registry.get("banks", {}).keys())
    rows = []
    for bank_key in banks:
        if bank_key not in arrays:
            continue
        bank = np.asarray(arrays[bank_key], dtype=np.float32)
        bank_norm = l2_normalise(bank)
        n, dim = bank_norm.shape
        case_ids = list(registry["banks"][bank_key]["case_ids"])
        labels, label_names, hard_flags = labels_for_bank(case_ids, registry)
        signatures = build_signatures(bank_key, case_ids, label_names, predictions, available_runs)
        quotient_label = bank_norm @ diagnostic_basis_from_labels(bank_norm, labels)
        quotient_logreg = bank_norm @ diagnostic_basis_from_logreg(bank_norm, labels, args.seed)

        methods: list[tuple[str, str, Callable[[], SearchOutput]]] = []
        methods.append(("full_cosine", "exact", lambda bank_norm=bank_norm: search_full_cosine(bank_norm, args.top_k)))
        methods.append(("int8_full_cosine", "compression", lambda bank_norm=bank_norm: search_int8_cosine(bank_norm, args.top_k)))
        for d in [16, 32, 64, 128, 256]:
            if d < dim and d < n:
                methods.append((f"pca{d}_full", "compression", lambda d=d, bank_norm=bank_norm: search_projected_cosine(f"PCA-{d} full scan.", pca_projection(bank_norm, d, args.seed), dim, 4, args.top_k)))
        for d in [32, 64, 128]:
            if d < dim:
                methods.append((f"rp{d}_full", "compression", lambda d=d, bank_norm=bank_norm: search_projected_cosine(f"Random projection {d}D full scan.", bank_norm @ random_projection_matrix(dim, d, args.seed + d), dim, 4, args.top_k)))
        methods.append(("quotient_label_full", "compression", lambda quotient_label=quotient_label: search_projected_cosine("Class-centroid diagnostic quotient full scan.", quotient_label, dim, 4, args.top_k)))
        methods.append(("quotient_logreg_full", "compression", lambda quotient_logreg=quotient_logreg: search_projected_cosine("Classifier-induced logreg quotient full scan.", quotient_logreg, dim, 4, args.top_k)))
        methods.append(("binary128_hash", "hash", lambda bank_norm=bank_norm: search_binary_hash(bank_norm, 128, args.seed, args.top_k)))
        methods.append(("pq_adc", "pq", lambda bank_norm=bank_norm: search_pq_asymmetric(bank_norm, args.pq_m, args.pq_k, args.top_k, args.seed)))
        methods.append(("cascade_pca64_full", "cascade", lambda bank_norm=bank_norm: search_cascade("PCA-64", pca_projection(bank_norm, min(64, n - 1, dim - 1), args.seed), bank_norm, args.cascade_top_m, args.top_k, 4)))
        methods.append(("cascade_quotient80_full", "cascade", lambda quotient_logreg=quotient_logreg, bank_norm=bank_norm: search_cascade("logreg quotient", quotient_logreg, bank_norm, args.cascade_top_m, args.top_k, 4)))
        methods.append(("macs_clinical80_full", "selective", lambda bank_norm=bank_norm, signatures=signatures: search_macs_clinical_cascade(bank_norm, signatures, args.cascade_top_m, args.top_k)))
        methods.append(("safe_r_selective", "selective", lambda bank_norm=bank_norm, labels=labels, label_names=label_names, signatures=signatures, bank_key=bank_key, case_ids=case_ids: search_selective_safe_r(bank_norm, labels, label_names, signatures, bank_key, case_ids, predictions, available_runs, args.top_k)))
        methods.append(("aags_upper_bound_prune", "algebraic", lambda bank_norm=bank_norm, signatures=signatures, quotient_logreg=quotient_logreg: search_algebraic_upper_bound(bank_norm, signatures, quotient_logreg, args.top_k)))

        baseline_equiv = n - 1
        for method, family, fn in methods:
            out = fn()
            metrics = evaluate_rankings(out.rankings, labels, hard_flags, args.top_k)
            rows.append({
                "bank": bank_key,
                "n_cases": n,
                "dim": dim,
                "method": method,
                "family": family,
                "equiv_comparisons": out.avg_equiv_comparisons,
                "raw_comparisons_or_candidates": out.avg_raw_comparisons,
                "cost_ratio": out.avg_equiv_comparisons / baseline_equiv if baseline_equiv else float("nan"),
                "memory_bytes_per_vector": out.avg_memory_bytes_per_vector,
                "memory_ratio": out.avg_memory_bytes_per_vector / (dim * 4),
                "ms_per_query": out.avg_ms_per_query,
                "same_label_p@5": metrics["same_label_p@k"],
                "same_label_top1": metrics["same_label_top1"],
                "same_label_hit@5": metrics["same_label_hit@k"],
                "melanoma_hit@5": metrics["melanoma_hit@k"],
                "melanoma_p@5": metrics["melanoma_p@k"],
                "hard_mel_first_rank@5": metrics["hard_mel_first_rank@k"],
                "notes": out.notes,
            })
            print(f"{bank_key:28s} {method:28s} P@5={metrics['same_label_p@k']:.4f} cost={out.avg_equiv_comparisons / baseline_equiv:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output / "cost_strategy_ablation.csv", index=False)
    plot_frontier(df, args.output)
    plot_melanoma(df, args.output)
    write_report(df, args.output, args.safe_r)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
