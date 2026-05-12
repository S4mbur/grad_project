"""
similarity_router.py
====================
Glue layer that lets ``app/server.py`` switch between similarity metrics
without growing a long if/else chain.

Drop this file at ``app/similarity_router.py`` next to ``server.py`` and
``similarity_metrics.py``.  See ``docs/similarity_metric_integration.md``
for the matching ``server.py`` patch.

The router maintains a per-bank cache of ``MetricContext`` objects.  The
classifier head ``W`` is constructed lazily the first time a non-cosine
metric is requested for a bank; for the cosine metric the router never
loads Torch, so existing cosine traffic stays as cheap as before.

The router treats every bank in ``retrieval_registry.json`` independently
and keeps the cache invalidated when the registry mtime changes (the same
mechanism ``server._load_phase4_registry`` already uses).
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from app import similarity_metrics as smet


SUPPORTED_METRICS = (
    "cosine",
    "negative_euclidean",
    "mahalanobis",
    "pullback_euclidean",
    "symmetrised_kl",
    "fisher_rao",
    "bhattacharyya",
    "style_cosine",
    "decomposed",
    "mutual_proximity_cosine",
    # Original metrics introduced by the SkinSight thesis (Section 8 of
    # docs/similarity_metric_study.md).
    "igpd",
    "abd",
    "dbrd",
    "mmpc",  # Mahalanobis composed with mutual-proximity hubness fix
)
DEFAULT_METRIC = "cosine"

# Optional path to the attention-covariance bank built by
# scripts/build_attention_covariance_bank.py.  When present the router
# loads the per-bank entries lazily on first use.
ATTENTION_COVARIANCES_PATH: "Optional[object]" = None


class _RouterCache:
    def __init__(self) -> None:
        self._lock = Lock()
        # key -> (registry_mtime, embeddings_mtime, MetricContext)
        self._contexts: Dict[str, Tuple[Tuple[float, float], smet.MetricContext]] = {}

    def get(
        self,
        bank_key: str,
        bank_embeddings: np.ndarray,
        registry: dict,
        registry_mtime: float,
        embeddings_mtime: float,
        head_factory,
    ) -> smet.MetricContext:
        cached = self._contexts.get(bank_key)
        if cached and cached[0] == (registry_mtime, embeddings_mtime):
            return cached[1]
        with self._lock:
            cached = self._contexts.get(bank_key)
            if cached and cached[0] == (registry_mtime, embeddings_mtime):
                return cached[1]
            ctx = self._build_context(bank_key, bank_embeddings, registry, head_factory)
            self._contexts[bank_key] = ((registry_mtime, embeddings_mtime), ctx)
            return ctx

    def _maybe_load_covariances(self, bank_key: str) -> "Tuple[Optional[np.ndarray], Optional[np.ndarray]]":
        path = ATTENTION_COVARIANCES_PATH
        if path is None:
            return None, None
        try:
            from pathlib import Path
            path = Path(path)
            if not path.exists():
                return None, None
            with np.load(path, allow_pickle=False) as data:
                key_U = f"{bank_key}__U"
                key_s = f"{bank_key}__s"
                if key_U not in data.files or key_s not in data.files:
                    return None, None
                return (
                    np.asarray(data[key_U], dtype=np.float32),
                    np.asarray(data[key_s], dtype=np.float32),
                )
        except Exception:
            return None, None

    def _build_context(
        self,
        bank_key: str,
        bank_embeddings: np.ndarray,
        registry: dict,
        head_factory,
    ) -> smet.MetricContext:
        bank_meta = registry.get("banks", {}).get(bank_key, {})
        case_ids = bank_meta.get("case_ids", [])
        cases = registry.get("cases", {})
        labels: List[int] = []
        class_index = {"Normal/Benign": 0, "BCC": 1, "SCC": 2, "Melanoma": 3}
        for cid in case_ids:
            label = class_index.get((cases.get(cid) or {}).get("true_label"), -1)
            labels.append(label)
        labels_arr = np.asarray(labels, dtype=np.int64)
        valid_labels = labels_arr[labels_arr >= 0]
        inv_cov = None
        if valid_labels.size:
            inv_cov = smet.fit_inverse_covariance(
                bank_embeddings, labels=labels_arr.tolist(), shrinkage=0.1,
            )
        bank_norm = bank_embeddings / np.maximum(
            np.linalg.norm(bank_embeddings, axis=1, keepdims=True), 1e-8,
        )
        self_cosine = (bank_norm @ bank_norm.T).astype(np.float32)
        head = None
        try:
            head = head_factory(bank_key, bank_embeddings, registry, labels_arr)
        except Exception as exc:  # pragma: no cover - defensive
            head = None
        ctx = smet.MetricContext(
            bank=bank_embeddings,
            head=head,
            inv_cov=inv_cov,
            bank_self_cosine=self_cosine,
        )
        bank_cov_U, bank_cov_s = self._maybe_load_covariances(bank_key)
        if bank_cov_U is not None and bank_cov_s is not None:
            if bank_cov_U.shape[0] == bank_embeddings.shape[0]:
                ctx.bank_cov_U = bank_cov_U
                ctx.bank_cov_s = bank_cov_s
        if head is not None:
            ctx.ensure_logits()
        return ctx


_cache = _RouterCache()


def resolve_metric(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_METRIC
    value = str(value).strip().lower()
    if value not in SUPPORTED_METRICS:
        return DEFAULT_METRIC
    return value


def fallback_logreg_head(
    bank_key: str,
    bank: np.ndarray,
    registry: dict,
    labels: np.ndarray,
) -> Optional[smet.ClassifierHead]:
    """Logistic-regression head used when the actual MIL classifier is
    unavailable (e.g. weights not on disk).  Mirrors what the experiment
    harness does."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    valid = labels >= 0
    if valid.sum() < 4:
        return None
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
    )
    classifier.fit(bank[valid], labels[valid])
    W = np.zeros((4, bank.shape[1]), dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    for row, cls in enumerate(classifier.classes_):
        W[int(cls)] = classifier.coef_[row]
        b[int(cls)] = classifier.intercept_[row]
    z_ref = bank.mean(axis=0)
    return smet.ClassifierHead(W=W, b=b, z_ref=z_ref, label="logreg")


def model_classifier_head(
    bank_key: str,
    bank: np.ndarray,
    registry: dict,
    labels: np.ndarray,
) -> Optional[smet.ClassifierHead]:
    """Linearise the actual MIL classifier head at the bank centroid."""
    try:
        import torch
        from app import server  # type: ignore
    except Exception:
        return fallback_logreg_head(bank_key, bank, registry, labels)

    z_ref = bank.mean(axis=0)
    bank_meta = registry.get("banks", {}).get(bank_key, {})
    component_models = bank_meta.get("component_models")

    def linearise(classifier_module, ref_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        classifier_module = classifier_module.cpu().eval()
        z = torch.tensor(ref_vec, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        out = classifier_module(z).squeeze(0)
        C = out.shape[0]
        W_local = torch.zeros((C, ref_vec.shape[0]))
        for c in range(C):
            grad = torch.autograd.grad(
                classifier_module(z).squeeze(0)[c],
                z,
                retain_graph=False,
            )[0]
            W_local[c] = grad.squeeze(0)
        b_local = (out - W_local @ z.squeeze(0)).detach().numpy()
        return W_local.detach().numpy(), b_local

    if component_models:
        chunk = z_ref.shape[0] // len(component_models)
        Ws = []
        bs = []
        for idx, mkey in enumerate(component_models):
            try:
                mil = server._get_mil_model(mkey)
            except Exception:
                return fallback_logreg_head(bank_key, bank, registry, labels)
            slice_z = z_ref[idx * chunk:(idx + 1) * chunk]
            W_part, b_part = linearise(mil.classifier, slice_z)
            full = np.zeros((W_part.shape[0], z_ref.shape[0]), dtype=np.float32)
            full[:, idx * chunk:(idx + 1) * chunk] = W_part
            Ws.append(full)
            bs.append(b_part)
        W = np.mean(Ws, axis=0).astype(np.float32)
        b = np.mean(bs, axis=0).astype(np.float32)
    else:
        try:
            mil = server._get_mil_model(bank_key)
        except Exception:
            return fallback_logreg_head(bank_key, bank, registry, labels)
        W, b = linearise(mil.classifier, z_ref)
    head = smet.ClassifierHead(W=W.astype(np.float32), b=b.astype(np.float32), z_ref=z_ref, label=f"model:{bank_key}")
    return head


def context_for(
    bank_key: str,
    bank_embeddings: np.ndarray,
    registry: dict,
    registry_mtime: float,
    embeddings_mtime: float,
    *,
    head_factory=model_classifier_head,
) -> smet.MetricContext:
    return _cache.get(
        bank_key,
        bank_embeddings,
        registry,
        registry_mtime,
        embeddings_mtime,
        head_factory,
    )


def score_query(
    metric: str,
    query: np.ndarray,
    context: smet.MetricContext,
    *,
    decomposed_alpha: float = 0.7,
) -> np.ndarray:
    if metric == "decomposed":
        context.decomposed_alpha = float(decomposed_alpha)
    try:
        return smet.score(metric, query, context)
    except ValueError:
        # graceful degrade: if the requested metric is unavailable for
        # this bank (e.g. no classifier head), fall back to cosine.
        return smet.score("cosine", query, context)


__all__ = [
    "DEFAULT_METRIC",
    "SUPPORTED_METRICS",
    "context_for",
    "fallback_logreg_head",
    "model_classifier_head",
    "resolve_metric",
    "score_query",
]