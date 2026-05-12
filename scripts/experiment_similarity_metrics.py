#!/usr/bin/env python3
"""
experiment_similarity_metrics.py
================================
Leave-one-out retrieval benchmark for the SkinSight Phase 4 case bank.

Drop this file at ``scripts/experiment_similarity_metrics.py``.  Run from
the repository root, after ``app/similarity_metrics.py`` is in place:

    python scripts/experiment_similarity_metrics.py \
        --logits-source model \
        --output results/phase4_retrieval/metric_study

The script

  1. loads ``results/phase4_retrieval/retrieval_embeddings.npz`` plus
     ``retrieval_registry.json`` (the artefacts written by
     ``build_phase4_retrieval_bank.py``),
  2. derives a 4-class linear classifier head ``W`` for each bank,
  3. evaluates every metric in ``app.similarity_metrics`` with strict
     leave-one-out retrieval (the query is removed from the bank), and
  4. writes a CSV summary, a per-query JSON dump, and a MD report card.

Stratified evaluation
---------------------
For each metric we report numbers in three slices:

  * ``all``       -- every labelled query
  * ``melanoma``  -- queries whose true_label is ``Melanoma``
  * ``hard_mel``  -- queries flagged ``is_hard_melanoma`` in the registry
                     (i.e.\ historical hard cases the safety pipeline
                     wants the comparison panel to do well on)

The classifier head can come from two sources:

  * ``--logits-source model``  -- import the actual MIL classifier from
    the checkpoint registered in ``app/server.py`` and linearise it via
    Jacobian at the bank centroid.  Requires Torch and the model weights.
  * ``--logits-source logreg`` -- fit a multinomial logistic regression
    on the bank embeddings using the case ``true_label`` as supervision.
    Self-contained, runs anywhere with NumPy + SciPy / scikit-learn.

For the ensemble bank we apply the per-component classifier to the
concatenated embedding by chunking it back into the originally
concatenated component vectors and averaging the per-component logits;
this matches what the ensemble would produce at inference time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Repository layout helpers
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_DIR / "app"
PHASE4_DIR = PROJECT_DIR / "results" / "phase4_retrieval"
DEFAULT_REGISTRY = PHASE4_DIR / "retrieval_registry.json"
DEFAULT_EMBEDDINGS = PHASE4_DIR / "retrieval_embeddings.npz"
DEFAULT_COVARIANCES = PHASE4_DIR / "attention_covariances.npz"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(APP_DIR))

from app import similarity_metrics as smet  # noqa: E402

CLASS_NAMES = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument(
        "--attention-covariances",
        type=Path,
        default=DEFAULT_COVARIANCES,
        help="NPZ produced by build_attention_covariance_bank.py.  When "
             "missing the ABD / DBRD metrics are skipped.",
    )
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=[0.1, 1.0, 10.0],
        help="DBRD lambda sweep (mixing weight on the Bures term).",
    )
    p.add_argument("--output", type=Path, default=PHASE4_DIR / "metric_study")
    p.add_argument(
        "--logits-source",
        choices=["model", "logreg", "auto"],
        default="auto",
        help="How to obtain the per-bank classifier head W.",
    )
    p.add_argument(
        "--banks",
        nargs="*",
        default=None,
        help="Restrict to these bank keys (default: all banks in the registry).",
    )
    p.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.3, 0.5, 0.7, 0.9],
        help="Mixing weights to sweep for the decomposed metric.",
    )
    p.add_argument(
        "--ks",
        type=int,
        nargs="*",
        default=[1, 3, 5, 10],
        help="Cut-offs at which to compute Recall, Precision, mAP and class purity.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for pullback metrics.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the optional logistic regression.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bank loading
# ---------------------------------------------------------------------------

def load_bank_arrays(embeddings_path: Path, registry_path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file missing: {embeddings_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file missing: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    with np.load(embeddings_path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    return arrays, registry


def load_attention_covariances(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Returns {bank_key: {"mu", "U", "s"}}.  Missing file or bank is OK
    -- the caller skips the ABD / DBRD metrics in that case."""
    if not path.exists():
        print(f"  [covariances] file not found at {path}; ABD/DBRD will be skipped")
        return {}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        bank_keys = sorted({k.split("__")[0] for k in keys if "__" in k})
        for bank_key in bank_keys:
            try:
                out[bank_key] = {
                    "mu": np.asarray(data[f"{bank_key}__mu"], dtype=np.float32),
                    "U": np.asarray(data[f"{bank_key}__U"], dtype=np.float32),
                    "s": np.asarray(data[f"{bank_key}__s"], dtype=np.float32),
                }
            except KeyError:
                pass
    return out


def bank_labels(case_ids: Sequence[str], registry: dict) -> Tuple[np.ndarray, np.ndarray]:
    cases = registry.get("cases", {})
    labels: List[int] = []
    hard_flags: List[bool] = []
    for cid in case_ids:
        meta = cases.get(cid) or {}
        labels.append(CLASS_INDEX.get(meta.get("true_label"), -1))
        hard_flags.append(bool(meta.get("is_hard_melanoma")))
    return np.asarray(labels, dtype=np.int64), np.asarray(hard_flags, dtype=bool)


# ---------------------------------------------------------------------------
# Classifier head retrieval
# ---------------------------------------------------------------------------

def fit_logreg_head(
    bank: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
) -> smet.ClassifierHead:
    """Multinomial logistic regression fit on the bank itself."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scikit-learn is required for the logreg classifier head"
        ) from exc

    valid = labels >= 0
    X = bank[valid]
    y = labels[valid]
    if X.shape[0] < len(CLASS_NAMES):
        raise ValueError("Not enough labelled samples to fit a classifier head")

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=seed,
    )
    classifier.fit(X, y)
    W = np.zeros((len(CLASS_NAMES), bank.shape[1]), dtype=np.float32)
    b = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for row, cls in enumerate(classifier.classes_):
        W[int(cls)] = classifier.coef_[row]
        b[int(cls)] = classifier.intercept_[row]
    z_ref = bank.mean(axis=0)
    return smet.ClassifierHead(W=W, b=b, z_ref=z_ref, label="logreg")


def linearise_torch_classifier(
    classifier_module,
    z_ref: np.ndarray,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Jacobian and offset of a torch classifier at z_ref.

    The leaf tensor lives in (1, d) shape so that .grad behaves correctly;
    we only ever read autograd outputs through ``torch.autograd.grad``,
    never through ``.grad`` on a non-leaf tensor.
    """
    import torch

    classifier_module = classifier_module.to(device).eval()
    z_leaf = torch.tensor(
        z_ref.reshape(1, -1),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    # Disable dropout deterministically.
    out = classifier_module(z_leaf).squeeze(0)
    C = out.shape[0]
    W = torch.zeros((C, z_ref.shape[0]), device=device)
    for c in range(C):
        out_c = classifier_module(z_leaf).squeeze(0)
        grad = torch.autograd.grad(out_c[c], z_leaf, retain_graph=False)[0]
        W[c] = grad.squeeze(0)
    out_final = classifier_module(z_leaf).squeeze(0).detach()
    b = (out_final - W @ z_leaf.squeeze(0)).detach().cpu().numpy()
    return W.detach().cpu().numpy(), b


def head_from_model(
    bank_key: str,
    bank: np.ndarray,
    registry: dict,
) -> Optional[smet.ClassifierHead]:
    """Try to build a head from the actual MIL checkpoint via app.server."""
    try:
        from app import server  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  [model head] could not import app.server ({exc}); skipping")
        return None

    z_ref = bank.mean(axis=0)
    component_models: Optional[List[str]] = None
    bank_meta = registry.get("banks", {}).get(bank_key, {})
    if bank_meta.get("type") == "ensemble":
        component_models = bank_meta.get("component_models")
    if component_models:
        chunk = z_ref.shape[0] // len(component_models)
        Ws = []
        bs = []
        for idx, mkey in enumerate(component_models):
            try:
                model = server._get_mil_model(mkey)
            except Exception as exc:
                print(f"  [model head] component {mkey} unavailable ({exc}); abort")
                return None
            slice_z = z_ref[idx * chunk:(idx + 1) * chunk]
            W_part, b_part = linearise_torch_classifier(
                model.classifier,
                slice_z,
                device="cpu",
            )
            full = np.zeros((W_part.shape[0], z_ref.shape[0]), dtype=np.float32)
            full[:, idx * chunk:(idx + 1) * chunk] = W_part
            Ws.append(full)
            bs.append(b_part)
        W = np.mean(Ws, axis=0)
        b = np.mean(bs, axis=0)
        return smet.ClassifierHead(W=W, b=b, z_ref=z_ref, label=f"model:{bank_key}")

    try:
        model = server._get_mil_model(bank_key)
    except Exception as exc:
        print(f"  [model head] could not load {bank_key} ({exc})")
        return None
    W, b = linearise_torch_classifier(model.classifier, z_ref, device="cpu")
    return smet.ClassifierHead(W=W, b=b, z_ref=z_ref, label=f"model:{bank_key}")


def derive_head(
    bank_key: str,
    bank: np.ndarray,
    labels: np.ndarray,
    registry: dict,
    source: str,
    seed: int,
) -> Optional[smet.ClassifierHead]:
    if source in {"model", "auto"}:
        head = head_from_model(bank_key, bank, registry)
        if head is not None:
            return head
        if source == "model":
            print(f"  [head] model unavailable for {bank_key}; skipping bank")
            return None
    return fit_logreg_head(bank, labels, seed=seed)


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def average_precision(relevance: np.ndarray, k: Optional[int] = None) -> float:
    """Average precision of a binary relevance vector."""
    if k is not None:
        relevance = relevance[:k]
    relevance = np.asarray(relevance, dtype=np.float64)
    if relevance.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(relevance)
    ranks = np.arange(1, len(relevance) + 1)
    precision_at_i = cum_hits / ranks
    return float((precision_at_i * relevance).sum() / relevance.sum())


def evaluate_metric(
    metric: str,
    bank: np.ndarray,
    labels: np.ndarray,
    hard_flags: np.ndarray,
    context: smet.MetricContext,
    ks: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    n = bank.shape[0]
    valid_query = labels >= 0
    per_query_scores = np.full((n, n), -np.inf, dtype=np.float32)
    spreads: List[float] = []
    for q_idx in range(n):
        if not valid_query[q_idx]:
            continue
        scores = smet.score(metric, bank[q_idx], context).astype(np.float32)
        scores[q_idx] = -np.inf
        per_query_scores[q_idx] = scores
        spreads.append(smet.discriminative_spread(np.delete(scores, q_idx)))

    relevance_matrices: Dict[int, np.ndarray] = {}
    for q_idx in range(n):
        if not valid_query[q_idx]:
            continue
        order = np.argsort(per_query_scores[q_idx])[::-1]
        relevance_matrices[q_idx] = (labels[order] == labels[q_idx]).astype(np.int32)

    melanoma_idx = set(np.where(labels == CLASS_INDEX["Melanoma"])[0].tolist())
    hard_mel_idx = set(np.where(hard_flags & (labels == CLASS_INDEX["Melanoma"]))[0].tolist())
    slices = {
        "all": list(relevance_matrices.keys()),
        "melanoma": [i for i in relevance_matrices if i in melanoma_idx],
        "hard_mel": [i for i in relevance_matrices if i in hard_mel_idx],
    }

    out: Dict[str, Dict[str, float]] = {}
    for slice_name, indices in slices.items():
        scores: Dict[str, float] = {}
        if not indices:
            scores["n_queries"] = 0
            out[slice_name] = scores
            continue
        for k in ks:
            recalls, precisions, aps, mrr_scores = [], [], [], []
            for q_idx in indices:
                rel = relevance_matrices[q_idx]
                top = rel[:k]
                n_relevant_total = int((labels == labels[q_idx]).sum() - 1)
                recall = top.sum() / max(n_relevant_total, 1)
                precision = top.sum() / max(k, 1)
                ap = average_precision(rel, k=k)
                first_hit = np.argmax(rel) if rel.any() else -1
                mrr = 1.0 / (first_hit + 1) if first_hit >= 0 else 0.0
                recalls.append(recall)
                precisions.append(precision)
                aps.append(ap)
                mrr_scores.append(mrr)
            scores[f"recall@{k}"] = float(np.mean(recalls))
            scores[f"precision@{k}"] = float(np.mean(precisions))
            scores[f"map@{k}"] = float(np.mean(aps))
            scores[f"mrr@{k}"] = float(np.mean(mrr_scores))
        # Mean reciprocal rank of first true class match (no truncation).
        first_hits = []
        for q_idx in indices:
            rel = relevance_matrices[q_idx]
            first_hit = np.argmax(rel) if rel.any() else -1
            first_hits.append((first_hit + 1) if first_hit >= 0 else float("nan"))
        first_hits = np.asarray(first_hits, dtype=np.float64)
        finite = first_hits[~np.isnan(first_hits)]
        scores["first_hit_rank"] = float(np.mean(finite)) if finite.size else float("nan")
        scores["n_queries"] = len(indices)
        out[slice_name] = scores

    out["all"]["mean_spread"] = float(np.mean(spreads)) if spreads else 0.0
    out["all"]["min_spread"] = float(np.min(spreads)) if spreads else 0.0
    return out


def evaluate_bank(
    bank_key: str,
    bank: np.ndarray,
    registry: dict,
    *,
    logits_source: str,
    seed: int,
    alphas: Sequence[float],
    lambdas: Sequence[float],
    ks: Sequence[int],
    temperature: float,
    cov_bank: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, dict]]:
    bank_meta = registry.get("banks", {}).get(bank_key, {})
    case_ids = bank_meta.get("case_ids") or []
    if not case_ids:
        print(f"  [bank {bank_key}] no case ids; skipping")
        return None
    if len(case_ids) != bank.shape[0]:
        print(f"  [bank {bank_key}] case_ids/embedding mismatch ({len(case_ids)} vs {bank.shape[0]})")
        return None

    labels, hard_flags = bank_labels(case_ids, registry)
    n_labelled = int((labels >= 0).sum())
    if n_labelled < 4:
        print(f"  [bank {bank_key}] too few labelled cases ({n_labelled}); skipping")
        return None

    head = derive_head(bank_key, bank, labels, registry, logits_source, seed)
    inv_cov = smet.fit_inverse_covariance(bank, labels=labels.tolist(), shrinkage=0.1)
    bank_norm = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8)
    self_cosine = (bank_norm @ bank_norm.T).astype(np.float32)
    if head is not None:
        head.temperature = temperature

    context = smet.MetricContext(
        bank=bank,
        head=head,
        inv_cov=inv_cov,
        bank_self_cosine=self_cosine,
        decomposed_alpha=alphas[0] if alphas else 0.7,
    )
    if cov_bank is not None and bank_key in cov_bank:
        cov_payload = cov_bank[bank_key]
        cov_mu = cov_payload["mu"]
        if cov_mu.shape[0] == bank.shape[0]:
            context.bank_cov_U = cov_payload["U"]
            context.bank_cov_s = cov_payload["s"]
            print(f"  [bank {bank_key}] attention covariances loaded "
                  f"(U={context.bank_cov_U.shape}, s={context.bank_cov_s.shape})")
        else:
            print(f"  [bank {bank_key}] attention-cov mu rowcount mismatch "
                  f"({cov_mu.shape[0]} vs {bank.shape[0]}); skipping ABD/DBRD")
    if head is not None:
        context.ensure_logits()

    metric_names = list(smet.available_metrics(context))
    extra_metric_records: Dict[str, str] = {}

    print(f"  [bank {bank_key}] metrics: {metric_names}")
    metric_records: Dict[str, dict] = {}
    for metric in metric_names:
        if metric in {"decomposed", "dbrd"}:
            continue  # handled with parameter sweep below
        try:
            metric_records[metric] = evaluate_metric(metric, bank, labels, hard_flags, context, ks)
        except Exception as exc:
            print(f"    metric {metric} failed: {exc}")
            extra_metric_records[metric] = str(exc)

    if head is not None:
        for alpha in alphas:
            context.decomposed_alpha = float(alpha)
            try:
                key = f"decomposed_a{alpha:.2f}"
                metric_records[key] = evaluate_metric(
                    "decomposed", bank, labels, hard_flags, context, ks,
                )
            except Exception as exc:
                print(f"    decomposed alpha={alpha} failed: {exc}")
                extra_metric_records[f"decomposed_a{alpha:.2f}"] = str(exc)

    if context.bank_cov_U is not None and head is not None:
        for lam in lambdas:
            context.dbrd_lambda = float(lam)
            try:
                key = f"dbrd_l{lam:g}"
                metric_records[key] = evaluate_metric(
                    "dbrd", bank, labels, hard_flags, context, ks,
                )
            except Exception as exc:
                print(f"    dbrd lambda={lam} failed: {exc}")
                extra_metric_records[f"dbrd_l{lam:g}"] = str(exc)

    summary = {
        "bank_key": bank_key,
        "bank_size": bank.shape[0],
        "embedding_dim": bank.shape[1],
        "n_labelled": n_labelled,
        "n_melanoma": int((labels == CLASS_INDEX["Melanoma"]).sum()),
        "n_hard_melanoma": int((hard_flags & (labels == CLASS_INDEX["Melanoma"])).sum()),
        "head_label": head.label if head else "n/a",
        "metric_results": metric_records,
        "metric_failures": extra_metric_records,
    }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(summary: Dict[str, dict], path: Path) -> None:
    rows = []
    for bank_key, info in summary.items():
        if not info:
            continue
        for metric_name, slice_records in info["metric_results"].items():
            for slice_name, scores in slice_records.items():
                row = {"bank": bank_key, "metric": metric_name, "slice": slice_name}
                row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()})
                rows.append(row)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(summary: Dict[str, dict], path: Path, ks: Sequence[int]) -> None:
    lines = ["# Similarity-metric study", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    headers = ["metric"] + [f"R@{k}" for k in ks] + [f"mAP@{k}" for k in ks] + ["spread", "first_hit"]
    for bank_key, info in summary.items():
        if not info:
            continue
        lines.append(
            f"## Bank: `{bank_key}` (n={info['bank_size']}, dim={info['embedding_dim']}, "
            f"head={info['head_label']}, "
            f"melanoma={info['n_melanoma']}, hard_mel={info['n_hard_melanoma']})"
        )
        lines.append("")
        for slice_name in ("all", "melanoma", "hard_mel"):
            lines.append(f"### Slice: {slice_name}")
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for metric_name, slice_records in info["metric_results"].items():
                scores = slice_records.get(slice_name, {})
                if not scores or scores.get("n_queries", 0) == 0:
                    continue
                row = [metric_name]
                for k in ks:
                    row.append(f"{scores.get(f'recall@{k}', 0.0):.3f}")
                for k in ks:
                    row.append(f"{scores.get(f'map@{k}', 0.0):.3f}")
                row.append(f"{slice_records.get('all', {}).get('mean_spread', 0.0):.4f}")
                fh = scores.get("first_hit_rank")
                row.append(f"{fh:.2f}" if isinstance(fh, float) and not np.isnan(fh) else "n/a")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        if info["metric_failures"]:
            lines.append("Failures: " + ", ".join(f"`{k}` ({v})" for k, v in info["metric_failures"].items()))
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading registry from {args.registry}")
    arrays, registry = load_bank_arrays(args.embeddings, args.registry)
    cov_bank = load_attention_covariances(args.attention_covariances)
    if cov_bank:
        print(f"Loaded attention covariances for banks: {sorted(cov_bank.keys())}")
    bank_keys = args.banks or sorted(arrays.keys())
    print(f"Banks to evaluate: {bank_keys}")

    summary: Dict[str, dict] = {}
    for bank_key in bank_keys:
        if bank_key not in arrays:
            print(f"  bank {bank_key} not in embeddings file; skipping")
            continue
        print(f"\n=== Bank {bank_key} ===")
        summary[bank_key] = evaluate_bank(
            bank_key,
            arrays[bank_key],
            registry,
            logits_source=args.logits_source,
            seed=args.seed,
            alphas=args.alphas,
            lambdas=args.lambdas,
            ks=args.ks,
            temperature=args.temperature,
            cov_bank=cov_bank,
        )

    json_path = args.output / "metric_study.json"
    csv_path = args.output / "metric_study.csv"
    md_path = args.output / "metric_study.md"
    json_path.write_text(json.dumps({"summary": summary, "args": vars(args)}, default=str, indent=2), encoding="utf-8")
    write_csv(summary, csv_path)
    write_markdown(summary, md_path, args.ks)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())