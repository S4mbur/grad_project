#!/usr/bin/env python3
"""
diagnose_phikon_hard_mel.py
===========================
Why does Mahalanobis hurt the Phikon hard-melanoma slice?

The leave-one-out benchmark in
``results/phase4_retrieval/metric_study/metric_study.md`` shows that on
``phikon_cost_sensitive_strong``:

  * cosine first_hit on hard_mel  = 1.38
  * mahalanobis first_hit on hard_mel = 2.69

That is the only bank where the global mAP winner (Mahalanobis) also
loses badly on the operationally critical hard-melanoma slice.  This
script reproduces the slice case-by-case, identifies the queries that
get demoted by Mahalanobis, and projects the offending shift onto the
top eigenvectors of the bank covariance to give an interpretable
diagnosis.

Drop at ``scripts/diagnose_phikon_hard_mel.py`` and run from the repo
root:

    python scripts/diagnose_phikon_hard_mel.py \
        --output results/phase4_retrieval/metric_study/phikon_hard_mel_diagnostic.md

Optional flags let you pick a different bank (``--bank``) or a
different number of top-K cases to inspect (``--top-k``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_DIR / "app"
PHASE4_DIR = PROJECT_DIR / "results" / "phase4_retrieval"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(APP_DIR))

from app import similarity_metrics as smet  # noqa: E402

CLASS_NAMES = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", default="phikon_cost_sensitive_strong")
    p.add_argument("--registry", type=Path, default=PHASE4_DIR / "retrieval_registry.json")
    p.add_argument("--embeddings", type=Path, default=PHASE4_DIR / "retrieval_embeddings.npz")
    p.add_argument("--output", type=Path, default=PHASE4_DIR / "metric_study" / "phikon_hard_mel_diagnostic.md")
    p.add_argument("--spectrum-output", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=5, help="Top-K candidates inspected per query")
    p.add_argument("--top-directions", type=int, default=4, help="Number of dominant covariance directions to break the divergence into")
    return p.parse_args()


def load_bank(args: argparse.Namespace):
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    with np.load(args.embeddings, allow_pickle=False) as data:
        if args.bank not in data.files:
            raise SystemExit(f"Bank '{args.bank}' not in embeddings file")
        bank = np.asarray(data[args.bank], dtype=np.float32)
    bank_meta = registry["banks"][args.bank]
    case_ids: List[str] = bank_meta["case_ids"]
    cases: Dict[str, dict] = registry.get("cases", {})
    labels = np.array([CLASS_INDEX.get((cases.get(cid) or {}).get("true_label"), -1) for cid in case_ids])
    hard_flags = np.array([bool((cases.get(cid) or {}).get("is_hard_melanoma")) for cid in case_ids])
    return bank, labels, hard_flags, case_ids, cases


def case_label(cases: Dict[str, dict], slide_id: str) -> str:
    meta = cases.get(slide_id, {})
    label = meta.get("true_label", "?")
    src = meta.get("source", "?")
    hard = "[HARD]" if meta.get("is_hard_melanoma") else ""
    return f"{label} {src} {hard}".strip()


def topk(scores: np.ndarray, k: int, exclude: int) -> List[int]:
    s = scores.copy()
    s[exclude] = -np.inf
    return np.argsort(s)[::-1][:k].tolist()


def shrunk_class_pooled_covariance(bank: np.ndarray, labels: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Mirror app.similarity_metrics.fit_inverse_covariance for diagnostics.

    The diagnostic needs the covariance spectrum, not only the inverse matrix.
    Keeping this local copy makes the figure exactly comparable to the
    Mahalanobis matrix used below.
    """
    x = np.asarray(bank, dtype=np.float64)
    labels_arr = np.asarray(labels)
    covs = []
    weights = []
    for c in np.unique(labels_arr):
        x_c = x[labels_arr == c]
        if x_c.shape[0] < 2:
            continue
        covs.append(np.cov(x_c, rowvar=False, bias=True))
        weights.append(x_c.shape[0])
    if covs:
        weights_arr = np.asarray(weights, dtype=np.float64)
        cov = sum(w * c for w, c in zip(weights_arr, covs)) / weights_arr.sum()
    else:
        cov = np.cov(x, rowvar=False, bias=True)
    diag_target = np.eye(cov.shape[0]) * float(np.trace(cov) / max(cov.shape[0], 1))
    return (1.0 - shrinkage) * cov + shrinkage * diag_target


def first_hit_rank(order_full: np.ndarray, query_idx: int, labels: np.ndarray, target_label: int) -> int:
    """Return 1-based rank of the first target-label neighbor, excluding self.

    Previous diagnostic code accidentally removed the item at position
    ``query_idx`` in the ranking array.  The ranking stores case indices, so
    self-exclusion must filter values equal to ``query_idx``.
    """
    rank = 0
    for j in order_full:
        j = int(j)
        if j == query_idx:
            continue
        rank += 1
        if int(labels[j]) == int(target_label):
            return rank
    return max(rank, 1)


def plot_eigenspectrum(cov_eigvals: np.ndarray, inv_eigvals: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cov = np.sort(np.asarray(cov_eigvals, dtype=np.float64))[::-1]
    inv = np.sort(np.asarray(inv_eigvals, dtype=np.float64))[::-1]
    cov = np.maximum(cov, 1e-12)
    inv = np.maximum(inv, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    axes[0].plot(np.arange(1, len(cov) + 1), cov, linewidth=1.5)
    axes[0].set_title("Shrunk covariance eigenvalues")
    axes[0].set_xlabel("Eigenvalue rank")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    axes[1].plot(np.arange(1, len(inv) + 1), inv, color="#b44", linewidth=1.5)
    axes[1].set_title("Inverse-covariance eigenvalues")
    axes[1].set_xlabel("Eigenvalue rank")
    axes[1].set_ylabel("Inverse eigenvalue")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Phikon Mahalanobis diagnostic spectrum")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spectrum_output = args.spectrum_output or args.output.with_name(args.output.stem + "_eigenspectrum.png")
    bank, labels, hard_flags, case_ids, cases = load_bank(args)

    # Compute the metric scores once per query.
    bank_norm = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8)
    inv_cov = smet.fit_inverse_covariance(bank, labels=labels.tolist(), shrinkage=0.1)
    shrunk_cov = shrunk_class_pooled_covariance(bank, labels, shrinkage=0.1)

    # Eigendecomposition of inv_cov for direction-level diagnostic.
    eigvals, eigvecs = np.linalg.eigh(inv_cov.astype(np.float64))
    order_eig = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order_eig]
    eigvecs = eigvecs[:, order_eig]
    top_dirs = eigvecs[:, : args.top_directions]
    top_dir_eigvals = eigvals[: args.top_directions]
    cov_eigvals = np.linalg.eigvalsh(shrunk_cov.astype(np.float64))
    inv_eigvals_all = np.linalg.eigvalsh(inv_cov.astype(np.float64))
    plot_eigenspectrum(cov_eigvals, inv_eigvals_all, spectrum_output)

    hard_mel_indices = [
        i for i in range(len(case_ids))
        if hard_flags[i] and labels[i] == CLASS_INDEX["Melanoma"]
    ]
    if not hard_mel_indices:
        print("No hard-melanoma cases found in bank; aborting")
        return 1
    print(f"{len(hard_mel_indices)} hard-melanoma queries found in {args.bank}")

    lines: List[str] = []
    lines.append(f"# Phikon hard-melanoma Mahalanobis diagnostic")
    lines.append("")
    lines.append(f"Bank: `{args.bank}`  -- N = {bank.shape[0]}, dim = {bank.shape[1]}")
    lines.append(f"Hard-melanoma queries: {len(hard_mel_indices)}")
    lines.append("")
    lines.append("Each subsection lists, for one hard-melanoma query, the top-K")
    lines.append("retrieval ordering under cosine and under Mahalanobis, then a")
    lines.append("decomposition of the offending vector onto the top inverse-")
    lines.append("covariance eigenvectors (the directions Mahalanobis amplifies).")
    lines.append("")
    lines.append(f"Eigenspectrum figure: `{spectrum_output.as_posix()}`")
    lines.append("")
    lines.append("## Covariance Spectrum Summary")
    lines.append("")
    positive_cov = cov_eigvals[cov_eigvals > 1e-12]
    positive_inv = inv_eigvals_all[inv_eigvals_all > 1e-12]
    effective_rank = float((positive_cov.sum() ** 2) / np.maximum((positive_cov ** 2).sum(), 1e-12))
    lines.append(f"- Effective covariance rank: {effective_rank:.2f} / {bank.shape[1]}")
    lines.append(f"- Covariance eigenvalue range: {positive_cov.min():.6e} to {positive_cov.max():.6e}")
    lines.append(f"- Inverse eigenvalue range: {positive_inv.min():.6e} to {positive_inv.max():.6e}")
    lines.append("- Direction projections below are shown in scientific notation to avoid hiding small but amplified components.")
    lines.append("")

    cosine_first_hits: List[float] = []
    maha_first_hits: List[float] = []
    divergence_scores = []  # (idx, cos_top1_label, maha_top1_label, divergence_magnitude)

    for q_idx in hard_mel_indices:
        q = bank[q_idx]
        # cosine
        cos_scores = (bank_norm @ (q / max(np.linalg.norm(q), 1e-8))).astype(np.float32)
        # mahalanobis as similarity = -distance
        diff = bank - q[None, :]
        maha_d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, inv_cov, diff), 0.0))
        cos_order = topk(cos_scores, args.top_k, exclude=q_idx)
        maha_order = topk(-maha_d, args.top_k, exclude=q_idx)

        cos_full = np.argsort(cos_scores)[::-1]
        maha_full = np.argsort(-maha_d)
        cos_fh = first_hit_rank(cos_full, q_idx, labels, CLASS_INDEX["Melanoma"])
        maha_fh = first_hit_rank(maha_full, q_idx, labels, CLASS_INDEX["Melanoma"])
        cosine_first_hits.append(cos_fh)
        maha_first_hits.append(maha_fh)
        divergence = maha_fh - cos_fh
        divergence_scores.append((q_idx, cos_fh, maha_fh, divergence))

        slide_id = case_ids[q_idx]
        lines.append(f"## Query {slide_id}  ({case_label(cases, slide_id)})")
        lines.append("")
        lines.append(f"Cosine first-hit rank: **{cos_fh}**, Mahalanobis first-hit rank: **{maha_fh}**")
        lines.append("")
        lines.append("### Top-K under cosine")
        lines.append("")
        lines.append("| rank | slide_id | true_label | source | cosine | maha_d |")
        lines.append("|---|---|---|---|---|---|")
        for r, j in enumerate(cos_order, 1):
            lines.append(
                f"| {r} | {case_ids[j]} | {labels_text(j, labels)} | "
                f"{(cases.get(case_ids[j]) or {}).get('source', '?')} | "
                f"{cos_scores[j]:.4f} | {maha_d[j]:.4f} |"
            )
        lines.append("")
        lines.append("### Top-K under Mahalanobis")
        lines.append("")
        lines.append("| rank | slide_id | true_label | source | cosine | maha_d |")
        lines.append("|---|---|---|---|---|---|")
        for r, j in enumerate(maha_order, 1):
            lines.append(
                f"| {r} | {case_ids[j]} | {labels_text(j, labels)} | "
                f"{(cases.get(case_ids[j]) or {}).get('source', '?')} | "
                f"{cos_scores[j]:.4f} | {maha_d[j]:.4f} |"
            )
        lines.append("")
        # Direction-level analysis: project the difference between the
        # first non-melanoma and the true melanoma onto the top inverse-
        # covariance eigenvectors.
        first_non_mel = next((j for j in maha_full if j != q_idx and labels[j] != CLASS_INDEX["Melanoma"]), None)
        first_mel = next(
            (j for j in maha_full if j != q_idx and labels[j] == CLASS_INDEX["Melanoma"]), None,
        )
        if first_non_mel is not None and first_mel is not None and first_non_mel != first_mel:
            d_nm = bank[first_non_mel] - q
            d_mel = bank[first_mel] - q
            proj_nm = (top_dirs.T @ d_nm).astype(np.float32)
            proj_mel = (top_dirs.T @ d_mel).astype(np.float32)
            lines.append("### Direction decomposition (top inv-cov eigenvectors)")
            lines.append("")
            lines.append("| direction | eigval | proj of (top non-Mel - q) | proj of (top Mel - q) | diff^2 weighted |")
            lines.append("|---|---|---|---|---|")
            for d in range(args.top_directions):
                wnm = float(proj_nm[d] ** 2 * top_dir_eigvals[d])
                wmel = float(proj_mel[d] ** 2 * top_dir_eigvals[d])
                lines.append(
                    f"| {d} | {top_dir_eigvals[d]:.6e} | "
                    f"{float(proj_nm[d]):+.6e} | {float(proj_mel[d]):+.6e} | "
                    f"non-mel={wnm:.6e}, mel={wmel:.6e} |"
                )
            lines.append("")
            lines.append(
                "*Mahalanobis amplifies directions with high inv-cov eigenvalues. "
                "When the true melanoma's projection on those directions is larger "
                "than the false-positive's, the metric demotes the melanoma even "
                "though cosine kept it on top. The columns above pinpoint which "
                "direction caused the swap.*"
            )
            lines.append("")

    # Summary
    div_array = np.asarray([s[3] for s in divergence_scores])
    lines.insert(5, f"Mean cosine first-hit:    {np.mean(cosine_first_hits):.2f}")
    lines.insert(6, f"Mean Mahalanobis first-hit: {np.mean(maha_first_hits):.2f}")
    lines.insert(7, f"Demoted by Mahalanobis: {(div_array > 0).sum()} / {len(divergence_scores)} queries "
                  f"(mean shift {div_array[div_array > 0].mean():.2f} ranks)" if (div_array > 0).any() else "No demotion observed")
    lines.insert(8, "")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Mean cosine first-hit:    {np.mean(cosine_first_hits):.2f}")
    print(f"Mean Mahalanobis first-hit: {np.mean(maha_first_hits):.2f}")
    if (div_array > 0).any():
        print(f"Mahalanobis demoted {int((div_array > 0).sum())} / {len(divergence_scores)} queries")
    return 0


def labels_text(idx: int, labels: np.ndarray) -> str:
    label_id = int(labels[idx])
    if label_id < 0:
        return "?"
    return CLASS_NAMES[label_id]


if __name__ == "__main__":
    raise SystemExit(main())
