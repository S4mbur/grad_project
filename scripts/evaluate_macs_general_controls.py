#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "phase4_retrieval" / "general_control_study"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MACS-style similarity on non-pathology vector search controls.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=318)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--classes", type=int, default=4)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


def make_clustered_vectors(rng, n, dim, n_classes):
    labels = np.arange(n) % n_classes
    rng.shuffle(labels)
    centers = normalize_rows(rng.normal(size=(n_classes, dim)).astype(np.float32))
    vectors = centers[labels] + 0.45 * rng.normal(size=(n, dim)).astype(np.float32)
    return normalize_rows(vectors), labels


def make_digits_vectors(dim):
    digits = load_digits()
    x = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    x = StandardScaler().fit_transform(x)
    n_components = min(dim, x.shape[1], len(x) - 1)
    z = PCA(n_components=n_components, random_state=0).fit_transform(x).astype(np.float32)
    if n_components < dim:
        padded = np.zeros((z.shape[0], dim), dtype=np.float32)
        padded[:, :n_components] = z
        z = padded
    return normalize_rows(z), y


def make_random_metadata(rng, n, n_classes):
    probs = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float32)
    ordered = np.sort(probs, axis=1)[:, ::-1]
    confidence = ordered[:, 0]
    margin = ordered[:, 0] - ordered[:, 1]
    melanoma_like = probs[:, -1]
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1) / np.log(n_classes)
    return np.concatenate(
        [probs, confidence[:, None], margin[:, None], melanoma_like[:, None], entropy[:, None]],
        axis=1,
    ).astype(np.float32)


def make_leaky_metadata(labels, n_classes):
    probs = np.full((len(labels), n_classes), 0.04, dtype=np.float32)
    probs[np.arange(len(labels)), labels] = 0.88
    confidence = np.full((len(labels), 1), 0.88, dtype=np.float32)
    margin = np.full((len(labels), 1), 0.84, dtype=np.float32)
    melanoma_like = probs[:, -1:]
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1, keepdims=True) / np.log(n_classes)
    return np.concatenate([probs, confidence, margin, melanoma_like, entropy], axis=1).astype(np.float32)


def clinical_similarity(query_sig, candidate_sigs):
    weights = np.asarray([0.5, 0.5, 0.5, 1.3, 0.5, 0.7, 1.2, 0.8], dtype=np.float32)
    if candidate_sigs.shape[1] != len(weights):
        weights = np.ones(candidate_sigs.shape[1], dtype=np.float32)
    diff = np.abs(candidate_sigs - query_sig[None, :])
    return np.clip(1.0 - (diff @ weights) / max(float(weights.sum()), 1e-8), 0.0, 1.0)


def retrieve_cosine(vectors, query_idx, top_k):
    scores = vectors @ vectors[query_idx]
    scores[query_idx] = -np.inf
    return np.argsort(scores)[::-1][:top_k]


def retrieve_macs_like(vectors, signatures, query_idx, top_k, shortlist=50):
    sig_scores = clinical_similarity(signatures[query_idx], signatures)
    sig_scores[query_idx] = -np.inf
    pre = np.argsort(sig_scores)[::-1][:shortlist]
    emb_scores = vectors[pre] @ vectors[query_idx]
    final = 0.52 * emb_scores + 0.48 * sig_scores[pre]
    return pre[np.argsort(final)[::-1][:top_k]]


def unit_similarity_from_dot(dot_scores):
    return np.clip((np.asarray(dot_scores, dtype=np.float32) + 1.0) / 2.0, 1e-6, 1.0)


def build_signature_quotient(vectors, signatures):
    """Control analogue of the pathology diagnostic quotient.

    For real pathology this quotient is induced by clinically meaningful
    model probabilities. In the random-metadata controls it should be the
    wrong projection and therefore should not beat cosine.
    """
    n_classes = max(1, signatures.shape[1] - 4)
    probs = np.asarray(signatures[:, :n_classes], dtype=np.float32)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    weights = probs.sum(axis=0)
    center = vectors.mean(axis=0, keepdims=True)
    centroids = (probs.T @ vectors) / np.maximum(weights[:, None], 1e-8)
    directions = centroids - center
    _, sigma, vt = np.linalg.svd(directions, full_matrices=False)
    if sigma.size == 0:
        return {"coords": np.zeros((len(vectors), 1), dtype=np.float32), "scale": 1.0, "dim": 0}
    tol = max(float(sigma.max()), 1.0) * 1e-5
    rank = int(np.sum(sigma > tol))
    rank = max(1, min(rank, n_classes - 1 if n_classes > 1 else 1, vt.shape[0]))
    basis = vt[:rank].T.astype(np.float32)
    coords = ((vectors - center) @ basis).astype(np.float32)
    coords = coords / np.maximum(coords.std(axis=0, keepdims=True), 1e-6)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.maximum((diffs * diffs).sum(axis=-1), 0.0))
    off_diag = dists[~np.eye(len(coords), dtype=bool)]
    scale = float(np.median(off_diag)) if len(off_diag) else 1.0
    return {"coords": coords, "scale": max(scale, 1e-3), "dim": rank}


def quotient_similarity(quotient, query_idx, candidate_indices):
    coords = quotient["coords"]
    if quotient.get("dim", 0) <= 0:
        return np.ones(len(candidate_indices), dtype=np.float32)
    q = coords[query_idx]
    c = coords[candidate_indices]
    dist2 = np.maximum(((c - q[None, :]) ** 2).sum(axis=1), 0.0)
    scale2 = max(float(quotient.get("scale", 1.0)) ** 2, 1e-6)
    return np.exp(-dist2 / (2.0 * scale2)).astype(np.float32)


def generic_contrast_similarity(signatures, query_idx, candidate_indices):
    """Deliberately generic stand-in for the pathology contrast component.

    If the metadata is arbitrary, this component should not reliably improve
    object-like vector retrieval. Leaky labels are kept as an unrealistic upper
    bound control.
    """
    q = signatures[query_idx]
    c = signatures[candidate_indices]
    if signatures.shape[1] >= 8:
        q_profile = np.asarray([q[-2], q[-3], q[-2] - q[-3], q[-4]], dtype=np.float32)
        c_profile = np.stack([c[:, -2], c[:, -3], c[:, -2] - c[:, -3], c[:, -4]], axis=1)
    else:
        q_profile = q
        c_profile = c
    diff = np.abs(c_profile - q_profile[None, :])
    return np.clip(1.0 - diff.mean(axis=1), 0.0, 1.0)


def retrieve_algebraic_like(vectors, signatures, quotient, query_idx, top_k, shortlist=50):
    """AAGS-style product metric applied to non-pathology controls."""
    sig_scores = clinical_similarity(signatures[query_idx], signatures)
    contrast_scores = generic_contrast_similarity(signatures, query_idx, np.arange(len(signatures)))
    quotient_scores = quotient_similarity(quotient, query_idx, np.arange(len(signatures)))
    risk_like = 1.0 - np.abs(signatures[:, -2] - signatures[query_idx, -2])
    routing = 0.34 * sig_scores + 0.24 * contrast_scores + 0.22 * quotient_scores + 0.20 * risk_like
    routing[query_idx] = -np.inf

    pre = np.argsort(routing)[::-1][:shortlist]
    emb_scores = unit_similarity_from_dot(vectors[pre] @ vectors[query_idx])
    sig_pre = np.clip(sig_scores[pre], 1e-6, 1.0)
    contrast_pre = np.clip(contrast_scores[pre], 1e-6, 1.0)
    quotient_pre = np.clip(quotient_scores[pre], 1e-6, 1.0)
    risk_pre = np.clip(risk_like[pre], 1e-6, 1.0)

    final = (
        np.power(emb_scores, 0.30) *
        np.power(sig_pre, 0.22) *
        np.power(contrast_pre, 0.18) *
        np.power(quotient_pre, 0.18) *
        np.power(risk_pre, 0.12)
    )
    return pre[np.argsort(final)[::-1][:top_k]]


def precision_at_k(order, labels, query_label):
    return float(np.mean(labels[order] == query_label))


def evaluate_dataset(name, vectors, labels, signatures, top_k):
    cosine_scores = []
    macs_scores = []
    algebraic_scores = []
    quotient = build_signature_quotient(vectors, signatures)
    for i in range(len(vectors)):
        cosine_order = retrieve_cosine(vectors, i, top_k)
        macs_order = retrieve_macs_like(vectors, signatures, i, top_k)
        algebraic_order = retrieve_algebraic_like(vectors, signatures, quotient, i, top_k)
        cosine_scores.append(precision_at_k(cosine_order, labels, labels[i]))
        macs_scores.append(precision_at_k(macs_order, labels, labels[i]))
        algebraic_scores.append(precision_at_k(algebraic_order, labels, labels[i]))
    return {
        "dataset": name,
        f"cosine_same_label_p@{top_k}": float(np.mean(cosine_scores)),
        f"macs_like_same_label_p@{top_k}": float(np.mean(macs_scores)),
        f"algebraic_like_same_label_p@{top_k}": float(np.mean(algebraic_scores)),
        "macs_delta": float(np.mean(macs_scores) - np.mean(cosine_scores)),
        "algebraic_delta": float(np.mean(algebraic_scores) - np.mean(cosine_scores)),
        "interpretation": (
            "MACS-style metadata is useful only if metadata is semantically aligned with the target task."
        ),
    }


def write_outputs(rows, output: Path, top_k: int):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "general_control_study.csv"
    json_path = output / "general_control_study.json"
    md_path = output / "general_control_study.md"

    fields = [
        "dataset",
        f"cosine_same_label_p@{top_k}",
        f"macs_like_same_label_p@{top_k}",
        f"algebraic_like_same_label_p@{top_k}",
        "macs_delta",
        "algebraic_delta",
        "interpretation",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps({"generated_at": datetime.now().isoformat(), "rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# MACS General-Control Study",
        "",
        "This control checks whether MACS-style metadata helps generic vector search when the metadata is not clinically meaningful.",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"| dataset | cosine P@{top_k} | MACS-like P@{top_k} | algebraic-like P@{top_k} | MACS delta | algebraic delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row[f'cosine_same_label_p@{top_k}']:.4f} | "
            f"{row[f'macs_like_same_label_p@{top_k}']:.4f} | "
            f"{row[f'algebraic_like_same_label_p@{top_k}']:.4f} | "
            f"{row['macs_delta']:.4f} | {row['algebraic_delta']:.4f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- `random_metadata_control` simulates a generic vector search task where MACS-style clinical metadata is unrelated to the target.",
        "- `leaky_metadata_control` is an intentionally unrealistic upper bound where metadata directly encodes class identity.",
        "- `algebraic-like` applies product-style composition plus a quotient-style projection to generic vectors; with random metadata it should not behave like a universal replacement for cosine.",
        "- If MACS helps only in pathology and in the unrealistic leaky control, the argument is that its value comes from clinically meaningful risk signals rather than from generic vector-search mechanics.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    vectors, labels = make_clustered_vectors(rng, args.n, args.dim, args.classes)

    random_signatures = make_random_metadata(rng, args.n, args.classes)
    leaky_signatures = make_leaky_metadata(labels, args.classes)

    digits_vectors, digits_labels = make_digits_vectors(args.dim)
    digits_random_signatures = make_random_metadata(rng, len(digits_labels), 10)
    digits_leaky_signatures = make_leaky_metadata(digits_labels, 10)

    rows = [
        evaluate_dataset("random_metadata_control", vectors, labels, random_signatures, args.top_k),
        evaluate_dataset("leaky_metadata_control", vectors, labels, leaky_signatures, args.top_k),
        evaluate_dataset("sklearn_digits_random_metadata", digits_vectors, digits_labels, digits_random_signatures, args.top_k),
        evaluate_dataset("sklearn_digits_leaky_metadata", digits_vectors, digits_labels, digits_leaky_signatures, args.top_k),
    ]
    paths = write_outputs(rows, args.output, args.top_k)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
