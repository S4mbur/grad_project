#!/usr/bin/env python3
"""Reusable melanoma-focused metric helpers.

This module keeps the Phase 0 metric contract in one place so training,
evaluation, registry building, and future scripts can share the same
definitions for melanoma recall, melanoma false negatives, and standardized
error-bank records.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception:  # pragma: no cover - sklearn is expected in this repo
    accuracy_score = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


DEFAULT_CLASS_NAMES = ("Normal/Benign", "BCC", "SCC", "Melanoma")
DEFAULT_MELANOMA_INDEX = 3


def _as_array(values, dtype=None):
    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def _label_to_index(label, class_names: Sequence[str]) -> int:
    if isinstance(label, (int, np.integer)):
        return int(label)
    if label is None:
        raise ValueError("label cannot be None")
    label_str = str(label).strip()
    for idx, name in enumerate(class_names):
        if label_str == name or label_str.lower() == name.lower():
            return idx
    raise ValueError(f"Unknown label: {label_str}")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def multiclass_summary(
    y_true,
    y_pred,
    y_prob=None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    melanoma_index: int = DEFAULT_MELANOMA_INDEX,
):
    """Return a JSON-friendly multiclass summary with melanoma-centric fields."""
    y_true = _as_array(y_true, dtype=np.int64)
    y_pred = _as_array(y_pred, dtype=np.int64)
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred are required")
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")

    n_classes = len(class_names)
    labels = np.arange(n_classes, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels) if confusion_matrix else None

    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    ).tolist() if recall_score else [0.0] * n_classes
    per_class_precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    ).tolist() if precision_score else [0.0] * n_classes

    summary = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6) if accuracy_score else float(np.mean(y_true == y_pred)),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6) if f1_score else 0.0,
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6) if f1_score else 0.0,
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6) if recall_score else 0.0,
        "per_class_recall": [round(float(v), 6) for v in per_class_recall],
        "per_class_precision": [round(float(v), 6) for v in per_class_precision],
        "melanoma_recall": round(float(recall_score(y_true == melanoma_index, y_pred == melanoma_index, zero_division=0)), 6) if recall_score else 0.0,
        "melanoma_precision": round(float(precision_score(y_true == melanoma_index, y_pred == melanoma_index, zero_division=0)), 6) if precision_score else 0.0,
        "melanoma_fn": int(np.sum((y_true == melanoma_index) & (y_pred != melanoma_index))),
        "melanoma_tp": int(np.sum((y_true == melanoma_index) & (y_pred == melanoma_index))),
        "melanoma_fp": int(np.sum((y_true != melanoma_index) & (y_pred == melanoma_index))),
        "melanoma_support": int(np.sum(y_true == melanoma_index)),
        "confusion_matrix": cm.tolist() if cm is not None else [],
    }

    if y_prob is not None:
        y_prob = _as_array(y_prob, dtype=np.float32)
        try:
            if y_prob.ndim == 2 and y_prob.shape[0] == len(y_true):
                summary["auc_ovr"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr")), 6)
            else:
                summary["auc_ovr"] = None
        except Exception:
            summary["auc_ovr"] = None
    else:
        summary["auc_ovr"] = None

    return summary


def melanoma_threshold_sweep(
    y_true,
    y_pred,
    melanoma_prob,
    thresholds: Sequence[float] | None = None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    melanoma_index: int = DEFAULT_MELANOMA_INDEX,
):
    """Sweep a melanoma override threshold and return the best operating point."""
    y_true = _as_array(y_true, dtype=np.int64)
    y_pred = _as_array(y_pred, dtype=np.int64)
    melanoma_prob = _as_array(melanoma_prob, dtype=np.float32)
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 101), 2)
    results = []
    for thr in thresholds:
        adjusted = np.asarray(y_pred, dtype=np.int64).copy()
        adjusted[melanoma_prob >= float(thr)] = melanoma_index
        summary = multiclass_summary(
            y_true,
            adjusted,
            y_prob=None,
            class_names=class_names,
            melanoma_index=melanoma_index,
        )
        results.append({
            "threshold": float(thr),
            "macro_f1": summary["macro_f1"],
            "melanoma_recall": summary["melanoma_recall"],
            "melanoma_fn": summary["melanoma_fn"],
            "summary": summary,
        })

    best = max(
        results,
        key=lambda r: (r["melanoma_recall"], r["macro_f1"], -r["threshold"]),
    ) if results else None
    return {
        "candidates": results,
        "best": best,
    }


def standardize_error_row(
    *,
    slide_id: str,
    source: str,
    result_dir: str,
    true_label,
    probabilities,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    model_name: str | None = None,
    experiment: str | None = None,
    extra: Mapping[str, object] | None = None,
    hard_case_candidate: bool | None = None,
):
    """Create a normalized per-slide, per-run error-analysis row."""
    class_names = tuple(class_names)
    probs = _as_array(probabilities, dtype=np.float32)
    if probs is None or probs.ndim != 1:
        raise ValueError("probabilities must be a 1D sequence")
    if len(probs) != len(class_names):
        raise ValueError("probabilities length must match class_names")

    true_index = _label_to_index(true_label, class_names)
    pred_index = int(np.argmax(probs))
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = float(top1 - top2)
    melanoma_prob = float(probs[DEFAULT_MELANOMA_INDEX])

    if hard_case_candidate is None:
        hard_case_candidate = bool(
            true_index == DEFAULT_MELANOMA_INDEX
            and (pred_index != DEFAULT_MELANOMA_INDEX or top1 < 0.75 or margin < 0.22)
        )

    row = {
        "slide_id": str(slide_id),
        "source": str(source or ""),
        "result_dir": str(result_dir or ""),
        "model_name": str(model_name or ""),
        "experiment": str(experiment or ""),
        "true_label": class_names[true_index],
        "pred_label": class_names[pred_index],
        "prediction_confidence": round(top1, 6),
        "margin": round(margin, 6),
        "melanoma_probability": round(melanoma_prob, 6),
        "is_melanoma": int(true_index == DEFAULT_MELANOMA_INDEX),
        "is_melanoma_fn": int(true_index == DEFAULT_MELANOMA_INDEX and pred_index != DEFAULT_MELANOMA_INDEX),
        "hard_case_candidate": int(bool(hard_case_candidate)),
        "is_correct": int(true_index == pred_index),
        "error_type": "correct" if true_index == pred_index else (
            "melanoma_fn" if true_index == DEFAULT_MELANOMA_INDEX else "mismatch"
        ),
        "prob_normal_benign": round(float(probs[0]), 6),
        "prob_bcc": round(float(probs[1]), 6),
        "prob_scc": round(float(probs[2]), 6),
        "prob_melanoma": round(float(probs[3]), 6),
    }
    if extra:
        for key, value in extra.items():
            row[key] = value
    return row


def aggregate_error_bank(rows: Iterable[Mapping[str, object]]):
    """Aggregate standardized error rows to a slide-level summary."""
    grouped = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("slide_id", "")),
            str(row.get("source", "")),
            str(row.get("true_label", "")),
        )
        grouped[key].append(dict(row))

    summaries = []
    for (slide_id, source, true_label), items in sorted(grouped.items(), key=lambda x: x[0]):
        result_dirs = sorted({str(item.get("result_dir", "")) for item in items if item.get("result_dir")})
        model_names = sorted({str(item.get("model_name", "")) for item in items if item.get("model_name")})
        experiments = sorted({str(item.get("experiment", "")) for item in items if item.get("experiment")})
        preds = [str(item.get("pred_label", "")) for item in items]
        hard_flags = [int(item.get("hard_case_candidate", 0)) for item in items]
        melanoma_fn_flags = [int(item.get("is_melanoma_fn", 0)) for item in items]
        confidences = [float(item.get("prediction_confidence", 0.0)) for item in items]
        margins = [float(item.get("margin", 0.0)) for item in items]
        mel_probs = [float(item.get("melanoma_probability", 0.0)) for item in items]
        correct_count = sum(int(item.get("is_correct", 0)) for item in items)

        summaries.append({
            "slide_id": slide_id,
            "source": source,
            "true_label": true_label,
            "run_count": len(items),
            "correct_count": int(correct_count),
            "error_count": int(len(items) - correct_count),
            "melanoma_fn_count": int(sum(melanoma_fn_flags)),
            "hard_case_count": int(sum(hard_flags)),
            "unique_pred_labels": ";".join(sorted(set(preds))),
            "result_dirs": ";".join(result_dirs),
            "model_names": ";".join(model_names),
            "experiments": ";".join(experiments),
            "min_confidence": round(min(confidences), 6) if confidences else 0.0,
            "max_confidence": round(max(confidences), 6) if confidences else 0.0,
            "min_margin": round(min(margins), 6) if margins else 0.0,
            "max_margin": round(max(margins), 6) if margins else 0.0,
            "min_melanoma_probability": round(min(mel_probs), 6) if mel_probs else 0.0,
            "max_melanoma_probability": round(max(mel_probs), 6) if mel_probs else 0.0,
            "missed_by_models": ";".join(sorted({str(item.get("model_name", "")) for item in items if int(item.get("is_melanoma_fn", 0))})),
        })
    return summaries

