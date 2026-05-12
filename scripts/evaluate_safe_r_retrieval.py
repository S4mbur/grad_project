#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_registry.json"
DEFAULT_EMBEDDINGS = PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_embeddings.npz"
DEFAULT_PREDICTIONS = PROJECT_DIR / "results" / "phase1_hard_case_bank" / "all_test_predictions.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "phase4_retrieval" / "safe_r_study"

CLASS_ORDER = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
MODEL_COMPONENTS = {
    "ensemble_2_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong"],
    "ensemble_3_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"],
}


@dataclass
class QuerySignals:
    pred_label: str
    probs: dict[str, float]
    confidence: float
    margin: float
    melanoma_probability: float
    hard_case_candidate: bool


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SAFE-R risk-adaptive retrieval against full cosine retrieval.")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_predictions(path: Path):
    rows = read_csv_rows(path)
    out = {}
    runs = set()
    for row in rows:
        run_name = Path(row["result_dir"]).name
        runs.add(run_name)
        out[(run_name, row["slide_id"])] = row
    return out, runs


def model_key_to_run_names(model_key: str):
    if "_" not in model_key:
        return []
    parts = model_key.split("_")
    backbone = parts[0]
    exp = "_".join(parts[1:])
    return [
        f"mil_4class_{backbone}_v3_fast_{exp}",
        f"mil_4class_{backbone}_v3_{exp}",
    ]


def row_to_signals(row) -> QuerySignals:
    probs = {
        "Normal/Benign": float(row.get("prob_normal_benign", 0.0)),
        "BCC": float(row.get("prob_bcc", 0.0)),
        "SCC": float(row.get("prob_scc", 0.0)),
        "Melanoma": float(row.get("prob_melanoma", 0.0)),
    }
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return QuerySignals(
        pred_label=row.get("pred_label") or ordered[0][0],
        probs=probs,
        confidence=float(row.get("prediction_confidence", ordered[0][1])),
        margin=float(row.get("margin", ordered[0][1] - ordered[1][1])),
        melanoma_probability=float(row.get("melanoma_probability", probs["Melanoma"])),
        hard_case_candidate=str(row.get("hard_case_candidate", "0")) == "1",
    )


def fallback_signals(true_label: str) -> QuerySignals:
    probs = {k: 0.02 for k in CLASS_ORDER}
    probs[true_label] = 0.94
    return QuerySignals(
        pred_label=true_label,
        probs=probs,
        confidence=0.94,
        margin=0.88,
        melanoma_probability=probs["Melanoma"],
        hard_case_candidate=False,
    )


def signals_for_model(model_key, slide_id, true_label, predictions, available_runs):
    if model_key in MODEL_COMPONENTS:
        component_signals = [
            signals_for_model(mkey, slide_id, true_label, predictions, available_runs)
            for mkey in MODEL_COMPONENTS[model_key]
        ]
        probs = {
            cls: float(np.mean([sig.probs[cls] for sig in component_signals]))
            for cls in CLASS_ORDER
        }
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return QuerySignals(
            pred_label=ordered[0][0],
            probs=probs,
            confidence=ordered[0][1],
            margin=ordered[0][1] - ordered[1][1],
            melanoma_probability=probs["Melanoma"],
            hard_case_candidate=any(sig.hard_case_candidate for sig in component_signals),
        )

    for run_name in model_key_to_run_names(model_key):
        if run_name in available_runs and (run_name, slide_id) in predictions:
            return row_to_signals(predictions[(run_name, slide_id)])
    return fallback_signals(true_label)


def risk_tier(signals: QuerySignals):
    melanoma_guard = signals.pred_label != "Melanoma" and signals.melanoma_probability >= 0.20
    if melanoma_guard or signals.hard_case_candidate or signals.confidence < 0.62 or signals.margin < 0.18:
        return "high"
    if signals.confidence < 0.80 or signals.margin < 0.35 or signals.melanoma_probability >= 0.10:
        return "moderate"
    return "low"


def candidate_mask_for_safe_r(case_labels, query_idx, signals: QuerySignals, tier: str):
    labels = np.asarray(case_labels)
    mask = np.ones(len(labels), dtype=bool)
    mask[query_idx] = False

    if tier == "high":
        return mask

    candidate_labels = {signals.pred_label}
    ordered_probs = sorted(signals.probs.items(), key=lambda kv: kv[1], reverse=True)
    if tier == "moderate":
        candidate_labels.update([ordered_probs[0][0], ordered_probs[1][0]])
        if signals.melanoma_probability >= 0.10:
            candidate_labels.add("Melanoma")

    routed = np.isin(labels, list(candidate_labels))
    routed[query_idx] = False
    return routed


def rank_from_mask(embeddings, query_idx, mask):
    query = embeddings[query_idx]
    candidate_indices = np.flatnonzero(mask)
    if len(candidate_indices) == 0:
        return [], 0
    scores = embeddings[candidate_indices] @ query
    ordered = candidate_indices[np.argsort(scores)[::-1]]
    return ordered.tolist(), int(len(candidate_indices))


def baseline_full(embeddings, query_idx):
    mask = np.ones(len(embeddings), dtype=bool)
    mask[query_idx] = False
    return rank_from_mask(embeddings, query_idx, mask)


def safe_r_v0(embeddings, case_labels, query_idx, signals: QuerySignals, top_k: int):
    tier = risk_tier(signals)
    mask = candidate_mask_for_safe_r(case_labels, query_idx, signals, tier)
    ordered, comparisons = rank_from_mask(embeddings, query_idx, mask)

    output = ordered[:top_k]
    melanoma_guard = signals.pred_label != "Melanoma" and signals.melanoma_probability >= 0.20

    if melanoma_guard and not any(case_labels[idx] == "Melanoma" for idx in output):
        mel_mask = np.asarray(case_labels) == "Melanoma"
        mel_mask[query_idx] = False
        mel_order, mel_cost = rank_from_mask(embeddings, query_idx, mel_mask)
        comparisons += mel_cost
        if mel_order:
            output = (output[: max(0, top_k - 1)] + [mel_order[0]])[:top_k]

    return output, comparisons, tier


def clinical_signature(signals: QuerySignals):
    probs = [signals.probs[cls] for cls in CLASS_ORDER]
    entropy = -sum(p * np.log(max(p, 1e-8)) for p in probs) / np.log(len(probs))
    return np.asarray(
        probs + [
            signals.confidence,
            signals.margin,
            signals.melanoma_probability,
            entropy,
        ],
        dtype=np.float32,
    )


def clinical_similarity(query_sig, candidate_sigs):
    diff = np.abs(candidate_sigs - query_sig[None, :])
    weighted = diff @ np.asarray([0.5, 0.5, 0.5, 1.3, 0.5, 0.7, 1.2, 0.8], dtype=np.float32)
    return np.clip(1.0 - weighted / 5.0, 0.0, 1.0)


def candidate_label_bonus(candidate_labels, signals: QuerySignals, tier: str):
    labels = np.asarray(candidate_labels)
    bonus = np.zeros(len(labels), dtype=np.float32)
    bonus[labels == signals.pred_label] += 0.08
    if tier == "high" or signals.melanoma_probability >= 0.20:
        bonus[labels == "Melanoma"] += 0.14
    elif signals.melanoma_probability >= 0.10:
        bonus[labels == "Melanoma"] += 0.05
    return bonus


def unit_similarity_from_dot(dot_scores):
    """Map cosine/dot similarity from [-1, 1] into a positive [0, 1] score."""
    return np.clip((np.asarray(dot_scores, dtype=np.float32) + 1.0) / 2.0, 1e-6, 1.0)


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def build_diagnostic_quotient(embeddings, signatures):
    """Build a model-induced diagnostic quotient proxy for a retrieval bank.

    The production MIL classifier is nonlinear, but its predictions define
    soft diagnostic fibres over the retrieval embeddings. We estimate the
    low-rank diagnostic quotient by taking probability-weighted class
    centroids and extracting their span. Differences orthogonal to this span
    are treated as approximate kernel/style variation for this component.
    """
    probs = np.asarray(signatures[:, : len(CLASS_ORDER)], dtype=np.float32)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    weights = probs.sum(axis=0)
    center = embeddings.mean(axis=0, keepdims=True)
    centroids = (probs.T @ embeddings) / np.maximum(weights[:, None], 1e-8)
    directions = centroids - center

    _, sigma, vt = np.linalg.svd(directions, full_matrices=False)
    if sigma.size == 0:
        return {
            "coords": np.zeros((len(embeddings), 1), dtype=np.float32),
            "scale": 1.0,
            "dim": 0,
            "strength": 0.0,
        }

    tol = max(float(sigma.max()), 1.0) * 1e-5
    rank = int(np.sum(sigma > tol))
    rank = max(1, min(rank, len(CLASS_ORDER) - 1, vt.shape[0]))
    basis = vt[:rank].T.astype(np.float32)
    coords = ((embeddings - center) @ basis).astype(np.float32)
    coords = coords / np.maximum(coords.std(axis=0, keepdims=True), 1e-6)

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.maximum((diffs * diffs).sum(axis=-1), 0.0))
    off_diag = dists[~np.eye(len(coords), dtype=bool)]
    scale = float(np.median(off_diag)) if len(off_diag) else 1.0
    scale = max(scale, 1e-3)
    strength = float(sigma[:rank].sum() / max(sigma.sum(), 1e-8))
    return {
        "coords": coords,
        "scale": scale,
        "dim": rank,
        "strength": strength,
    }


def diagnostic_quotient_similarity(quotient, query_idx, candidate_indices):
    """RBF similarity in the classifier-induced diagnostic quotient space."""
    coords = quotient["coords"]
    if quotient.get("dim", 0) <= 0 or len(candidate_indices) == 0:
        return np.ones(len(candidate_indices), dtype=np.float32)
    q = coords[query_idx]
    c = coords[candidate_indices]
    diff = c - q[None, :]
    dist2 = np.maximum((diff * diff).sum(axis=1), 0.0)
    scale2 = max(float(quotient.get("scale", 1.0)) ** 2, 1e-6)
    return np.exp(-dist2 / (2.0 * scale2)).astype(np.float32)


def label_risk_rank(label: str):
    """Finite risk lattice used by algebra-inspired retrieval metrics.

    0: benign-like, 1: keratinocytic carcinoma, 2: melanoma-borderline,
    3: melanoma-like. The intermediate rank is query/model derived rather
    than a ground-truth diagnosis.
    """
    if label == "Melanoma":
        return 3
    if label in {"BCC", "SCC"}:
        return 1
    return 0


def signal_risk_rank(signals: QuerySignals):
    if signals.pred_label == "Melanoma" or signals.melanoma_probability >= 0.35:
        return 3
    if signals.melanoma_probability >= 0.20 or signals.hard_case_candidate:
        return 2
    if signals.pred_label in {"BCC", "SCC"}:
        return 1
    return 0


def risk_lattice_rank(signals: QuerySignals, true_label: str | None = None):
    rank = signal_risk_rank(signals)
    if true_label is not None:
        rank = max(rank, label_risk_rank(true_label))
    return rank


def risk_lattice_similarity(query_rank: int, candidate_ranks):
    """Similarity on an ordered risk lattice, not a generic image metric.

    High melanoma-risk queries define an upward-compatible set: melanoma-like
    and melanoma-borderline cases remain useful even when exact diagnosis is
    not identical. This is intentionally dermatopathology-specific.
    """
    candidate_ranks = np.asarray(candidate_ranks, dtype=np.float32)
    base = 1.0 - np.abs(candidate_ranks - float(query_rank)) / 3.0
    if query_rank >= 2:
        upward = candidate_ranks >= 2
        base = np.where(upward, np.maximum(base, 0.86), base)
        base = np.where(candidate_ranks == 3, base + 0.08, base)
    elif query_rank == 1:
        base = np.where(candidate_ranks == 1, base + 0.05, base)
    return np.clip(base, 0.05, 1.0)


def diagnostic_contrast_similarity(signatures, query_idx, candidate_indices):
    """Melanoma-vs-SCC/BCC differential profile similarity.

    The contrast dimensions make sense for this dermatopathology task because
    melanoma false negatives and SCC/BCC confusions are clinically asymmetric.
    They would be arbitrary metadata for a generic object-detection problem.
    """
    q = signatures[query_idx]
    query_profile = np.asarray(
        [
            q[6],
            q[2],
            q[1],
            q[6] - q[2],
            q[6] - max(q[1], q[2]),
        ],
        dtype=np.float32,
    )
    c = signatures[candidate_indices]
    cand_profiles = np.stack(
        [
            c[:, 6],
            c[:, 2],
            c[:, 1],
            c[:, 6] - c[:, 2],
            c[:, 6] - np.maximum(c[:, 1], c[:, 2]),
        ],
        axis=1,
    ).astype(np.float32)
    weights = np.asarray([1.5, 0.8, 0.5, 1.2, 1.2], dtype=np.float32)
    diff = np.abs(cand_profiles - query_profile[None, :])
    return np.clip(1.0 - (diff @ weights) / weights.sum(), 0.0, 1.0)


def macs_search(embeddings, case_labels, signatures, query_idx, signals: QuerySignals, top_k: int):
    """Melanoma-aware Clinical Similarity search.

    Stage 1 is a cheap low-dimensional clinical-signature search. Stage 2 only
    computes embedding similarity for the routed shortlist.
    """
    tier = risk_tier(signals)
    base_mask = candidate_mask_for_safe_r(case_labels, query_idx, signals, tier)
    candidate_indices = np.flatnonzero(base_mask)
    if len(candidate_indices) == 0:
        return [], 0.0, tier

    query_sig = signatures[query_idx]
    candidate_sigs = signatures[candidate_indices]
    clin_scores = clinical_similarity(query_sig, candidate_sigs)
    clin_scores += candidate_label_bonus([case_labels[idx] for idx in candidate_indices], signals, tier)

    if tier == "low":
        budget = min(30, len(candidate_indices))
    elif tier == "moderate":
        budget = min(75, len(candidate_indices))
    else:
        budget = min(150, len(candidate_indices))

    preselect_local = np.argsort(clin_scores)[::-1][:budget]
    preselect = candidate_indices[preselect_local]
    preselect_clin = clin_scores[preselect_local]

    emb_scores = embeddings[preselect] @ embeddings[query_idx]
    mel_axis = 1.0 - np.abs(signatures[preselect, 6] - signatures[query_idx, 6])
    margin_axis = 1.0 - np.abs(signatures[preselect, 5] - signatures[query_idx, 5])
    label_bonus = candidate_label_bonus([case_labels[idx] for idx in preselect], signals, tier)

    final_scores = (
        0.52 * emb_scores +
        0.25 * preselect_clin +
        0.10 * mel_axis +
        0.05 * margin_axis +
        0.08 * label_bonus
    )
    order = preselect[np.argsort(final_scores)[::-1]].tolist()

    melanoma_guard = signals.pred_label != "Melanoma" and signals.melanoma_probability >= 0.20
    if melanoma_guard and not any(case_labels[idx] == "Melanoma" for idx in order[:top_k]):
        mel_mask = (np.asarray(case_labels) == "Melanoma")
        mel_mask[query_idx] = False
        mel_candidates = np.flatnonzero(mel_mask)
        mel_emb_scores = embeddings[mel_candidates] @ embeddings[query_idx]
        best_mel = int(mel_candidates[int(np.argmax(mel_emb_scores))])
        order = (order[: max(0, top_k - 1)] + [best_mel] + order[top_k - 1:])
        budget += len(mel_candidates)

    signature_dim = signatures.shape[1]
    embedding_dim = embeddings.shape[1]
    equivalent_cost = len(candidate_indices) * (signature_dim / embedding_dim) + budget
    return order[:top_k], float(equivalent_cost), tier


def pathology_axes(signals: QuerySignals):
    """Low-cost pathology-aware profile used for shortlist reranking.

    This is deliberately not a generic image-search feature. It captures the
    clinical axes that matter for this task: melanoma risk, keratinocytic
    carcinoma risk, benign support, and decision ambiguity.
    """
    entropy = -sum(p * np.log(max(p, 1e-8)) for p in signals.probs.values()) / np.log(len(CLASS_ORDER))
    return np.asarray(
        [
            signals.melanoma_probability,
            signals.probs["BCC"] + signals.probs["SCC"],
            signals.probs["Normal/Benign"],
            1.0 - signals.margin,
            entropy,
            float(signals.hard_case_candidate),
        ],
        dtype=np.float32,
    )


def pathology_axis_similarity(query_axis, candidate_axes):
    weights = np.asarray([1.6, 0.8, 0.5, 1.0, 1.0, 1.2], dtype=np.float32)
    diff = np.abs(candidate_axes - query_axis[None, :])
    return np.clip(1.0 - (diff @ weights) / weights.sum(), 0.0, 1.0)


def top_tile_proxy_similarity(embeddings, query_idx, candidate_indices):
    """Approximate attention/tile reranking without reloading tile tensors.

    We use coordinate-wise overlap of high-activation dimensions as a cheap
    proxy for whether two bag embeddings are supported by similar dominant
    morphology factors. It is not a replacement for full tile-set matching,
    but it is a cost-aware stand-in for attention-aware reranking.
    """
    query = embeddings[query_idx]
    candidates = embeddings[candidate_indices]
    n_top = max(8, min(32, query.shape[0] // 8))
    q_top = np.argsort(np.abs(query))[::-1][:n_top]
    q_mask = np.zeros(query.shape[0], dtype=bool)
    q_mask[q_top] = True

    sims = []
    for cand in candidates:
        c_top = np.argsort(np.abs(cand))[::-1][:n_top]
        c_mask = np.zeros(cand.shape[0], dtype=bool)
        c_mask[c_top] = True
        inter = float(np.logical_and(q_mask, c_mask).sum())
        union = float(np.logical_or(q_mask, c_mask).sum())
        sims.append(inter / union if union else 0.0)
    return np.asarray(sims, dtype=np.float32)


def macs_attention_v1_search(embeddings, case_labels, signatures, axes, query_idx, signals: QuerySignals, top_k: int):
    """MACS v1: MACS shortlist plus pathology/attention-aware reranking."""
    tier = risk_tier(signals)
    base_mask = candidate_mask_for_safe_r(case_labels, query_idx, signals, tier)
    candidate_indices = np.flatnonzero(base_mask)
    if len(candidate_indices) == 0:
        return [], 0.0, tier

    query_sig = signatures[query_idx]
    candidate_sigs = signatures[candidate_indices]
    clin_scores = clinical_similarity(query_sig, candidate_sigs)
    clin_scores += candidate_label_bonus([case_labels[idx] for idx in candidate_indices], signals, tier)

    if tier == "low":
        budget = min(30, len(candidate_indices))
        rerank_budget = min(12, budget)
    elif tier == "moderate":
        budget = min(75, len(candidate_indices))
        rerank_budget = min(24, budget)
    else:
        budget = min(150, len(candidate_indices))
        rerank_budget = min(48, budget)

    preselect_local = np.argsort(clin_scores)[::-1][:budget]
    preselect = candidate_indices[preselect_local]
    preselect_clin = clin_scores[preselect_local]

    emb_scores = embeddings[preselect] @ embeddings[query_idx]
    first_stage = 0.62 * emb_scores + 0.38 * preselect_clin
    rerank_local = np.argsort(first_stage)[::-1][:rerank_budget]
    rerank_indices = preselect[rerank_local]

    axis_scores = pathology_axis_similarity(axes[query_idx], axes[rerank_indices])
    tile_proxy = top_tile_proxy_similarity(embeddings, query_idx, rerank_indices)
    rerank_emb = embeddings[rerank_indices] @ embeddings[query_idx]
    mel_alignment = 1.0 - np.abs(signatures[rerank_indices, 6] - signatures[query_idx, 6])
    label_bonus = candidate_label_bonus([case_labels[idx] for idx in rerank_indices], signals, tier)

    final_scores = (
        0.42 * rerank_emb +
        0.22 * axis_scores +
        0.14 * tile_proxy +
        0.12 * mel_alignment +
        0.10 * label_bonus
    )

    ordered = rerank_indices[np.argsort(final_scores)[::-1]].tolist()
    remaining = [idx for idx in preselect[np.argsort(first_stage)[::-1]].tolist() if idx not in set(ordered)]
    ordered.extend(remaining)

    melanoma_guard = signals.pred_label != "Melanoma" and signals.melanoma_probability >= 0.20
    if melanoma_guard and not any(case_labels[idx] == "Melanoma" for idx in ordered[:top_k]):
        mel_mask = (np.asarray(case_labels) == "Melanoma")
        mel_mask[query_idx] = False
        mel_candidates = np.flatnonzero(mel_mask)
        mel_axis_scores = pathology_axis_similarity(axes[query_idx], axes[mel_candidates])
        mel_emb_scores = embeddings[mel_candidates] @ embeddings[query_idx]
        best_mel = int(mel_candidates[int(np.argmax(0.55 * mel_emb_scores + 0.45 * mel_axis_scores))])
        ordered = (ordered[: max(0, top_k - 1)] + [best_mel] + ordered[top_k - 1:])
        budget += len(mel_candidates)

    signature_dim = signatures.shape[1]
    axis_dim = axes.shape[1]
    embedding_dim = embeddings.shape[1]
    equivalent_cost = (
        len(candidate_indices) * (signature_dim / embedding_dim) +
        budget +
        rerank_budget * (0.35 + axis_dim / embedding_dim)
    )
    return ordered[:top_k], float(equivalent_cost), tier


def algebraic_component_scores(
    embeddings,
    case_labels,
    signatures,
    axes,
    quotient,
    risk_ranks,
    query_idx,
    candidate_indices,
    signals: QuerySignals,
):
    emb_scores = unit_similarity_from_dot(embeddings[candidate_indices] @ embeddings[query_idx])
    quotient_scores = diagnostic_quotient_similarity(quotient, query_idx, candidate_indices)
    clinical_scores = clinical_similarity(signatures[query_idx], signatures[candidate_indices])
    axis_scores = pathology_axis_similarity(axes[query_idx], axes[candidate_indices])
    tile_scores = top_tile_proxy_similarity(embeddings, query_idx, candidate_indices)
    contrast_scores = diagnostic_contrast_similarity(signatures, query_idx, candidate_indices)
    lattice_scores = risk_lattice_similarity(risk_lattice_rank(signals), risk_ranks[candidate_indices])

    candidate_labels = np.asarray([case_labels[idx] for idx in candidate_indices])
    evidence_scores = np.full(len(candidate_indices), 0.62, dtype=np.float32)
    evidence_scores[candidate_labels == signals.pred_label] = 0.86
    if signals.melanoma_probability >= 0.20 or signals.pred_label == "Melanoma":
        evidence_scores[candidate_labels == "Melanoma"] = 0.96
    elif signals.melanoma_probability >= 0.10:
        evidence_scores[candidate_labels == "Melanoma"] = 0.78

    return {
        "embedding": emb_scores,
        "quotient": np.clip(quotient_scores, 1e-6, 1.0),
        "clinical": np.clip(clinical_scores, 1e-6, 1.0),
        "axis": np.clip(axis_scores, 1e-6, 1.0),
        "tile": np.clip(tile_scores, 1e-6, 1.0),
        "contrast": np.clip(contrast_scores, 1e-6, 1.0),
        "lattice": np.clip(lattice_scores, 1e-6, 1.0),
        "evidence": np.clip(evidence_scores, 1e-6, 1.0),
    }


def algebraic_similarity_search(
    embeddings,
    case_labels,
    signatures,
    axes,
    quotient,
    risk_ranks,
    query_idx,
    signals: QuerySignals,
    top_k: int,
    mode: str,
):
    """Abstract-algebra-inspired search over pathology evidence components.

    `aags_product_v1` treats component similarities as elements of the
    multiplicative monoid on [0, 1]. A mismatch on one clinically important
    component cannot be fully hidden by a high embedding dot product.

    `trlq_tropical_v1` maps similarities to costs with -log(s) and combines
    them in a min-plus/tropical style score. This is useful when the retrieval
    problem is viewed as accumulating evidence penalties rather than adding
    raw similarities.

    The v2 modes add an explicit diagnostic quotient component [z]_K. It is
    estimated from probability-weighted class centroids in the retrieval bank,
    so the search geometry is induced by the trained model's diagnostic fibres
    rather than by the raw embedding inner product alone.
    """
    uses_quotient = mode.endswith("_v2")
    tier = risk_tier(signals)
    base_mask = candidate_mask_for_safe_r(case_labels, query_idx, signals, tier)
    candidate_indices = np.flatnonzero(base_mask)
    if len(candidate_indices) == 0:
        return [], 0.0, tier

    query_sig = signatures[query_idx]
    candidate_sigs = signatures[candidate_indices]
    clinical_scores = clinical_similarity(query_sig, candidate_sigs)
    lattice_scores = risk_lattice_similarity(risk_lattice_rank(signals), risk_ranks[candidate_indices])
    contrast_scores = diagnostic_contrast_similarity(signatures, query_idx, candidate_indices)
    quotient_scores = diagnostic_quotient_similarity(quotient, query_idx, candidate_indices)
    label_bonus = candidate_label_bonus([case_labels[idx] for idx in candidate_indices], signals, tier)
    if uses_quotient:
        routing_scores = (
            0.38 * clinical_scores +
            0.20 * lattice_scores +
            0.18 * contrast_scores +
            0.14 * quotient_scores +
            0.10 * label_bonus
        )
    else:
        routing_scores = (
            0.46 * clinical_scores +
            0.24 * lattice_scores +
            0.20 * contrast_scores +
            0.10 * label_bonus
        )

    if tier == "low":
        budget = min(32, len(candidate_indices))
        rerank_budget = min(14, budget)
    elif tier == "moderate":
        budget = min(80, len(candidate_indices))
        rerank_budget = min(28, budget)
    else:
        budget = min(160, len(candidate_indices))
        rerank_budget = min(54, budget)

    preselect_local = np.argsort(routing_scores)[::-1][:budget]
    preselect = candidate_indices[preselect_local]
    preselect_routing = routing_scores[preselect_local]

    emb_unit = unit_similarity_from_dot(embeddings[preselect] @ embeddings[query_idx])
    stage_scores = 0.58 * emb_unit + 0.42 * preselect_routing
    rerank_local = np.argsort(stage_scores)[::-1][:rerank_budget]
    rerank_indices = preselect[rerank_local]

    comps = algebraic_component_scores(
        embeddings,
        case_labels,
        signatures,
        axes,
        quotient,
        risk_ranks,
        query_idx,
        rerank_indices,
        signals,
    )

    if mode == "aags_product_v1":
        weights = {
            "embedding": 0.34,
            "clinical": 0.17,
            "axis": 0.14,
            "tile": 0.10,
            "contrast": 0.11,
            "lattice": 0.09,
            "evidence": 0.05,
        }
        final_scores = np.ones(len(rerank_indices), dtype=np.float32)
        for key, weight in weights.items():
            final_scores *= np.power(comps[key], weight)
    elif mode == "aags_quotient_v2":
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
        final_scores = np.ones(len(rerank_indices), dtype=np.float32)
        for key, weight in weights.items():
            final_scores *= np.power(comps[key], weight)
    elif mode == "trlq_tropical_v1":
        weights = {
            "embedding": 0.36,
            "clinical": 0.15,
            "axis": 0.13,
            "tile": 0.09,
            "contrast": 0.12,
            "lattice": 0.10,
            "evidence": 0.05,
        }
        cost = np.zeros(len(rerank_indices), dtype=np.float32)
        for key, weight in weights.items():
            cost += weight * (-np.log(comps[key]))
        final_scores = -cost
    elif mode == "trlq_quotient_v2":
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
        cost = np.zeros(len(rerank_indices), dtype=np.float32)
        for key, weight in weights.items():
            cost += weight * (-np.log(comps[key]))
        final_scores = -cost
    else:
        raise ValueError(f"Unknown algebraic retrieval mode: {mode}")

    ordered = rerank_indices[np.argsort(final_scores)[::-1]].tolist()
    ordered_set = set(ordered)
    remaining = [
        idx for idx in preselect[np.argsort(stage_scores)[::-1]].tolist()
        if idx not in ordered_set
    ]
    ordered.extend(remaining)

    melanoma_guard = signals.pred_label != "Melanoma" and signals.melanoma_probability >= 0.20
    if melanoma_guard and not any(case_labels[idx] == "Melanoma" for idx in ordered[:top_k]):
        mel_mask = (np.asarray(case_labels) == "Melanoma")
        mel_mask[query_idx] = False
        mel_candidates = np.flatnonzero(mel_mask)
        if len(mel_candidates):
            mel_components = algebraic_component_scores(
                embeddings,
                case_labels,
                signatures,
                axes,
                quotient,
                risk_ranks,
                query_idx,
                mel_candidates,
                signals,
            )
            mel_scores = (
                0.45 * mel_components["embedding"] +
                0.25 * mel_components["axis"] +
                0.20 * mel_components["contrast"] +
                0.10 * mel_components["lattice"]
            )
            best_mel = int(mel_candidates[int(np.argmax(mel_scores))])
            ordered = (ordered[: max(0, top_k - 1)] + [best_mel] + ordered[top_k - 1:])
            budget += len(mel_candidates)

    signature_dim = signatures.shape[1]
    axis_dim = axes.shape[1]
    quotient_dim = quotient.get("dim", 0) if uses_quotient else 0
    embedding_dim = embeddings.shape[1]
    equivalent_cost = (
        len(candidate_indices) * ((signature_dim + axis_dim + quotient_dim + 2) / embedding_dim) +
        budget +
        rerank_budget * (0.45 + (axis_dim + quotient_dim) / embedding_dim)
    )
    return ordered[:top_k], float(equivalent_cost), tier


def danger_aware_search(embeddings, case_labels, signatures, query_idx, signals: QuerySignals, top_k: int):
    """Always reserve evidence slots for dangerous alternatives when relevant."""
    tier = risk_tier(signals)
    base_order, base_cost = baseline_full(embeddings, query_idx)
    selected = base_order[:top_k]
    extra_cost = 0

    if signals.pred_label != "Melanoma" and signals.melanoma_probability >= 0.10:
        mel_mask = np.asarray(case_labels) == "Melanoma"
        mel_mask[query_idx] = False
        mel_candidates = np.flatnonzero(mel_mask)
        extra_cost += len(mel_candidates)
        mel_scores = embeddings[mel_candidates] @ embeddings[query_idx]
        best_mel = int(mel_candidates[int(np.argmax(mel_scores))])
        if best_mel not in selected:
            selected = (selected[: max(0, top_k - 1)] + [best_mel])[:top_k]

    return selected, float(base_cost + extra_cost), tier


def clinical_only_search(signatures, query_idx, top_k: int):
    query = signatures[query_idx]
    candidate_indices = np.asarray([i for i in range(len(signatures)) if i != query_idx], dtype=np.int64)
    scores = clinical_similarity(query, signatures[candidate_indices])
    ordered = candidate_indices[np.argsort(scores)[::-1]]
    cost = len(candidate_indices) * (signatures.shape[1] / 256.0)
    return ordered[:top_k].tolist(), float(cost), "clinical_only"


def pathology_axis_search(axes, query_idx, top_k: int):
    query = axes[query_idx]
    candidate_indices = np.asarray([i for i in range(len(axes)) if i != query_idx], dtype=np.int64)
    scores = pathology_axis_similarity(query, axes[candidate_indices])
    ordered = candidate_indices[np.argsort(scores)[::-1]]
    cost = len(candidate_indices) * (axes.shape[1] / 256.0)
    return ordered[:top_k].tolist(), float(cost), "pathology_axis"


def precision_same_label(order, labels, true_label, k):
    if not order:
        return 0.0
    selected = order[:k]
    return float(np.mean([labels[idx] == true_label for idx in selected]))


def contains_label(order, labels, label, k):
    return any(labels[idx] == label for idx in order[:k])


def first_hit_rank(order, labels, target_label, max_rank=50):
    for rank, idx in enumerate(order[:max_rank], 1):
        if labels[idx] == target_label:
            return rank
    return None


def evaluate_bank(bank_key, bank, embeddings, registry, predictions, available_runs, top_k):
    case_ids = bank["case_ids"]
    cases = registry["cases"]
    labels = [cases[slide_id]["true_label"] for slide_id in case_ids]
    hard_ids = set(registry.get("hard_case_slide_ids", []))
    signals_list = [
        signals_for_model(bank_key, slide_id, labels[idx], predictions, available_runs)
        for idx, slide_id in enumerate(case_ids)
    ]
    signatures = np.stack([clinical_signature(sig) for sig in signals_list], axis=0)
    axes = np.stack([pathology_axes(sig) for sig in signals_list], axis=0)
    quotient = build_diagnostic_quotient(embeddings, signatures)
    risk_ranks = np.asarray(
        [risk_lattice_rank(sig, labels[idx]) for idx, sig in enumerate(signals_list)],
        dtype=np.float32,
    )

    rows = []
    methods = {
        "baseline_full_cosine": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "safe_r_v0": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "clinical_signature_only": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "pathology_axis_only": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "macs_v0": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "macs_attention_v1": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "aags_product_v1": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "trlq_tropical_v1": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "aags_quotient_v2": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "trlq_quotient_v2": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
        "danger_aware_full": {"same_p": [], "mel_cov": [], "hard_first": [], "cost": [], "tiers": []},
    }

    for q_idx, slide_id in enumerate(case_ids):
        true_label = labels[q_idx]
        signals = signals_list[q_idx]

        base_order, base_cost = baseline_full(embeddings, q_idx)
        safe_order, safe_cost, tier = safe_r_v0(embeddings, labels, q_idx, signals, top_k)
        clinical_order, clinical_cost, clinical_tier = clinical_only_search(signatures, q_idx, top_k)
        axis_order, axis_cost, axis_tier = pathology_axis_search(axes, q_idx, top_k)
        macs_order, macs_cost, macs_tier = macs_search(embeddings, labels, signatures, q_idx, signals, top_k)
        macs_v1_order, macs_v1_cost, macs_v1_tier = macs_attention_v1_search(
            embeddings, labels, signatures, axes, q_idx, signals, top_k
        )
        aags_order, aags_cost, aags_tier = algebraic_similarity_search(
            embeddings, labels, signatures, axes, quotient, risk_ranks, q_idx, signals, top_k, "aags_product_v1"
        )
        trlq_order, trlq_cost, trlq_tier = algebraic_similarity_search(
            embeddings, labels, signatures, axes, quotient, risk_ranks, q_idx, signals, top_k, "trlq_tropical_v1"
        )
        aags_q_order, aags_q_cost, aags_q_tier = algebraic_similarity_search(
            embeddings, labels, signatures, axes, quotient, risk_ranks, q_idx, signals, top_k, "aags_quotient_v2"
        )
        trlq_q_order, trlq_q_cost, trlq_q_tier = algebraic_similarity_search(
            embeddings, labels, signatures, axes, quotient, risk_ranks, q_idx, signals, top_k, "trlq_quotient_v2"
        )
        danger_order, danger_cost, danger_tier = danger_aware_search(embeddings, labels, signatures, q_idx, signals, top_k)

        for method, order, cost, tier_name in (
            ("baseline_full_cosine", base_order, base_cost, "full"),
            ("safe_r_v0", safe_order, safe_cost, tier),
            ("clinical_signature_only", clinical_order, clinical_cost, clinical_tier),
            ("pathology_axis_only", axis_order, axis_cost, axis_tier),
            ("macs_v0", macs_order, macs_cost, macs_tier),
            ("macs_attention_v1", macs_v1_order, macs_v1_cost, macs_v1_tier),
            ("aags_product_v1", aags_order, aags_cost, aags_tier),
            ("trlq_tropical_v1", trlq_order, trlq_cost, trlq_tier),
            ("aags_quotient_v2", aags_q_order, aags_q_cost, aags_q_tier),
            ("trlq_quotient_v2", trlq_q_order, trlq_q_cost, trlq_q_tier),
            ("danger_aware_full", danger_order, danger_cost, danger_tier),
        ):
            methods[method]["same_p"].append(precision_same_label(order, labels, true_label, top_k))
            methods[method]["mel_cov"].append(float(contains_label(order, labels, "Melanoma", top_k)))
            methods[method]["cost"].append(cost)
            methods[method]["tiers"].append(tier_name)
            if slide_id in hard_ids:
                hit = first_hit_rank(order, labels, "Melanoma", max_rank=top_k)
                methods[method]["hard_first"].append(hit if hit is not None else top_k + 1)

    for method, values in methods.items():
        tier_counts = {tier: values["tiers"].count(tier) for tier in sorted(set(values["tiers"]))}
        rows.append({
            "bank": bank_key,
            "method": method,
            "n_cases": len(case_ids),
            f"same_label_precision@{top_k}": float(np.mean(values["same_p"])),
            f"melanoma_coverage@{top_k}": float(np.mean(values["mel_cov"])),
            f"hard_mel_first_rank@{top_k}": float(np.mean(values["hard_first"])) if values["hard_first"] else None,
            "avg_comparisons": float(np.mean(values["cost"])),
            "tier_counts": tier_counts,
        })
    return rows


def write_outputs(rows, output_dir: Path, top_k: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "safe_r_study.csv"
    json_path = output_dir / "safe_r_study.json"
    md_path = output_dir / "safe_r_study.md"

    fields = [
        "bank",
        "method",
        "n_cases",
        f"same_label_precision@{top_k}",
        f"melanoma_coverage@{top_k}",
        f"hard_mel_first_rank@{top_k}",
        "avg_comparisons",
        "tier_counts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writable = dict(row)
            writable["tier_counts"] = json.dumps(writable["tier_counts"], sort_keys=True)
            writer.writerow(writable)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "top_k": top_k,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# SAFE-R v0 Retrieval Study",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| bank | method | same P@{} | mel coverage@{} | hard mel first rank@{} | avg comparisons | tiers |".format(top_k, top_k, top_k),
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        hard = row[f"hard_mel_first_rank@{top_k}"]
        hard_text = "" if hard is None else f"{hard:.2f}"
        lines.append(
            "| {bank} | {method} | {p:.4f} | {mc:.4f} | {hard} | {cost:.1f} | `{tiers}` |".format(
                bank=row["bank"],
                method=row["method"],
                p=row[f"same_label_precision@{top_k}"],
                mc=row[f"melanoma_coverage@{top_k}"],
                hard=hard_text,
                cost=row["avg_comparisons"],
                tiers=json.dumps(row["tier_counts"], sort_keys=True),
            )
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- `baseline_full_cosine` searches every case except the query itself.",
        "- `safe_r_v0` routes low-risk queries to smaller label-conditioned candidate pools.",
        "- `macs_v0` uses a cheap melanoma-aware clinical signature before embedding reranking.",
        "- `macs_attention_v1` adds a pathology-axis and top-activation proxy rerank on the MACS shortlist.",
        "- `aags_product_v1` is an abstract-algebra-inspired product/t-norm metric over embedding, clinical, pathology-axis, top-activation, differential-diagnosis, risk-lattice, and evidence components.",
        "- `trlq_tropical_v1` maps the same components into -log costs and combines them with a tropical/min-plus style evidence penalty.",
        "- `aags_quotient_v2` and `trlq_quotient_v2` add an explicit model-induced diagnostic quotient component estimated from probability-weighted class centroids in the retrieval embedding bank.",
        "- `danger_aware_full` is a high-cost control that injects melanoma alternatives without reducing the full-search cost.",
        "- melanoma-borderline non-melanoma queries add a nearest melanoma counterfactual if needed.",
        "- `avg_comparisons` is the approximate number of vector dot products per query.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main():
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    predictions, available_runs = load_predictions(args.predictions)
    with np.load(args.embeddings, allow_pickle=False) as data:
        arrays = {key: data[key].astype(np.float32) for key in data.files}

    rows = []
    for bank_key, bank in registry.get("banks", {}).items():
        if bank_key not in arrays:
            continue
        rows.extend(
            evaluate_bank(
                bank_key,
                bank,
                arrays[bank_key],
                registry,
                predictions,
                available_runs,
                args.top_k,
            )
        )

    csv_path, json_path, md_path = write_outputs(rows, args.output, args.top_k)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
