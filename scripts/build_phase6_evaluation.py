#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import server


OUT_DIR = PROJECT_DIR / "results" / "phase6_evaluation"
CAL_DIR = OUT_DIR / "calibration"
EXPL_DIR = OUT_DIR / "explanation"
RETR_DIR = OUT_DIR / "retrieval"
OOD_DIR = OUT_DIR / "ood"
DATA_ROOT = Path(
    os.environ.get("SKINSIGHT_DATA_ROOT", "/mnt/d/skin_cancer_project/datasets")
).expanduser()
FEATURE_ROOT = Path(
    os.environ.get("SKINSIGHT_CACHE_ROOT", "/mnt/d/skin_cancer_project/cache")
).expanduser()
OOD_LABELS_CSV = DATA_ROOT / "labels" / "ood_disease_types.csv"
OOD_IMAGES_DIR = DATA_ROOT / "cobra_ood" / "images"

SHORTLIST = [
    "uni_cost_sensitive_strong",
    "phikon_cost_sensitive_strong",
    "conch_cost_sensitive_strong",
]
ENSEMBLE_SETTINGS = {
    "ensemble_2_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong"],
    "ensemble_3_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"],
}
LABEL_NAMES = list(server.CLASS_NAMES.values())
LABEL_TO_ID = {v: k for k, v in server.CLASS_NAMES.items()}
OOD_ID_CATEGORIES = {"Squamous cell carcinoma", "Basal cell carcinoma", "Melanoma", "Benign", "No abnormalities"}
OOD_TARGET_MODEL = "uni_cost_sensitive_strong"
EMBED_CACHE = {}


def ensure_dirs():
    for d in (OUT_DIR, CAL_DIR, EXPL_DIR, RETR_DIR, OOD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_phase1_rows(model_key):
    csv_path = Path(server.MODEL_REGISTRY[model_key]["mil_checkpoint"]).parent / "phase1_test_predictions.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_to_probs(rows):
    probs = []
    labels = []
    for row in rows:
        labels.append(LABEL_TO_ID[row["true_label"]])
        probs.append([
            float(row["prob_normal_benign"]),
            float(row["prob_bcc"]),
            float(row["prob_scc"]),
            float(row["prob_melanoma"]),
        ])
    return np.asarray(probs, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def feature_dir_for_model(model_key):
    cfg = server.MODEL_REGISTRY[model_key]
    mtype = cfg["type"]
    if mtype == "torchvision":
        loader = cfg.get("loader", "")
        suffix_map = {
            "convnext_base": "convnext_base",
            "convnext_small": "convnext_small",
            "resnet50": "resnet50",
            "resnet18": "resnet18",
        }
        suffix = suffix_map[loader]
    elif mtype == "dinov2":
        suffix = "dinov2_base"
    else:
        suffix = mtype
    return FEATURE_ROOT / f"features_4class_{suffix}"


def slide_embedding_from_feature(model_key, slide_id):
    cache_key = (model_key, slide_id)
    if cache_key in EMBED_CACHE:
        return EMBED_CACHE[cache_key]
    torch, device = server._ensure_torch()
    feat_path = feature_dir_for_model(model_key) / f"{slide_id}.pt"
    feats = torch.load(str(feat_path), map_location=device)
    feats = feats.to(device)
    model = server._get_mil_model(model_key)
    with torch.no_grad():
        _, _, bag_embedding, _ = model(feats)
    emb = bag_embedding.detach().cpu().numpy().astype(np.float32).reshape(-1)
    EMBED_CACHE[cache_key] = emb
    return emb


def multiclass_metrics(y_true, probs, predictions):
    metrics = {
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro")), 4),
        "macro_recall": round(float(recall_score(y_true, predictions, average="macro")), 4),
        "melanoma_recall": round(float(recall_score(y_true == 3, predictions == 3)), 4),
        "melanoma_fn": int(np.sum((y_true == 3) & (predictions != 3))),
        "coverage": round(float(len(predictions) / len(y_true)), 4),
    }
    try:
        metrics["auc_ovr"] = round(float(roc_auc_score(y_true, probs, multi_class="ovr")), 4)
    except Exception:
        metrics["auc_ovr"] = None
    return metrics


def build_calibration_analysis():
    summary_rows = []
    for model_key in SHORTLIST:
        rows = load_phase1_rows(model_key)
        raw_probs, labels = rows_to_probs(rows)
        calibrated = np.asarray([
            server._apply_probability_calibration(prob, model_key)[0]
            for prob in raw_probs
        ], dtype=np.float32)

        conf_raw = raw_probs.max(axis=1)
        pred_raw = raw_probs.argmax(axis=1)
        conf_cal = calibrated.max(axis=1)
        pred_cal = calibrated.argmax(axis=1)
        correct_raw = (pred_raw == labels).astype(np.float32)
        correct_cal = (pred_cal == labels).astype(np.float32)

        bins = np.linspace(0.0, 1.0, 11)
        centers = (bins[:-1] + bins[1:]) / 2.0
        acc_raw, avg_raw, acc_cal, avg_cal = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            if hi == 1.0:
                mask_raw = (conf_raw >= lo) & (conf_raw <= hi)
                mask_cal = (conf_cal >= lo) & (conf_cal <= hi)
            else:
                mask_raw = (conf_raw >= lo) & (conf_raw < hi)
                mask_cal = (conf_cal >= lo) & (conf_cal < hi)
            acc_raw.append(float(correct_raw[mask_raw].mean()) if np.any(mask_raw) else np.nan)
            avg_raw.append(float(conf_raw[mask_raw].mean()) if np.any(mask_raw) else np.nan)
            acc_cal.append(float(correct_cal[mask_cal].mean()) if np.any(mask_cal) else np.nan)
            avg_cal.append(float(conf_cal[mask_cal].mean()) if np.any(mask_cal) else np.nan)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Perfect calibration")
        ax.plot(avg_raw, acc_raw, marker="o", color="#ef4444", label="Before scaling")
        ax.plot(avg_cal, acc_cal, marker="o", color="#0ea5e9", label="After scaling")
        ax.set_title(f"Reliability Diagram - {server.MODEL_REGISTRY[model_key]['display']}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        out_png = CAL_DIR / f"{model_key}_reliability.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

        reg = server._get_calibration_entry(model_key) or {}
        summary_rows.append({
            "model_key": model_key,
            "model_display": server.MODEL_REGISTRY[model_key]["display"],
            "ece_before": reg.get("ece_before"),
            "ece_after": reg.get("ece_after"),
            "mce_before": reg.get("mce_before"),
            "mce_after": reg.get("mce_after"),
            "temperature": reg.get("temperature"),
            "plot_path": str(out_png),
        })

    out_json = CAL_DIR / "calibration_summary.json"
    out_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return summary_rows


def build_ensemble_ablation():
    model_rows = {m: load_phase1_rows(m) for m in SHORTLIST}
    aligned = {}
    for m, rows in model_rows.items():
        aligned[m] = {row["slide_id"]: row for row in rows}
    slide_ids = sorted(set.intersection(*(set(x.keys()) for x in aligned.values())))

    y_true = np.asarray([LABEL_TO_ID[aligned[SHORTLIST[0]][sid]["true_label"]] for sid in slide_ids], dtype=np.int64)
    probs_by_model = {}
    for model_key in SHORTLIST:
        probs_by_model[model_key] = np.asarray([
            server._apply_probability_calibration(np.asarray([
                float(aligned[model_key][sid]["prob_normal_benign"]),
                float(aligned[model_key][sid]["prob_bcc"]),
                float(aligned[model_key][sid]["prob_scc"]),
                float(aligned[model_key][sid]["prob_melanoma"]),
            ], dtype=np.float32), model_key)[0]
            for sid in slide_ids
        ], dtype=np.float32)

    rows = []
    for model_key in SHORTLIST:
        probs = probs_by_model[model_key]
        preds = probs.argmax(axis=1)
        rows.append({
            "setting": model_key,
            "display": server.MODEL_REGISTRY[model_key]["display"],
            **multiclass_metrics(y_true, probs, preds),
        })

    ensemble2 = np.mean([probs_by_model[m] for m in ENSEMBLE_SETTINGS["ensemble_2_best"]], axis=0)
    rows.append({
        "setting": "ensemble_2_best",
        "display": server.ENSEMBLE_PRESETS["ensemble_2_best"]["display"],
        **multiclass_metrics(y_true, ensemble2, ensemble2.argmax(axis=1)),
    })

    ensemble3 = np.mean([probs_by_model[m] for m in ENSEMBLE_SETTINGS["ensemble_3_best"]], axis=0)
    rows.append({
        "setting": "ensemble_3_best",
        "display": server.ENSEMBLE_PRESETS["ensemble_3_best"]["display"],
        **multiclass_metrics(y_true, ensemble3, ensemble3.argmax(axis=1)),
    })

    abstained = []
    kept_probs = []
    kept_true = []
    for idx, sid in enumerate(slide_ids):
        per_model_embs = [slide_embedding_from_feature(m, sid) for m in ENSEMBLE_SETTINGS["ensemble_3_best"]]
        safety = server._build_phase1_safety(int(ensemble3[idx].argmax()), ensemble3[idx], ensemble_predictions=[int(probs_by_model[m][idx].argmax()) for m in ENSEMBLE_SETTINGS["ensemble_3_best"]])
        cal_meta = []
        for m in ENSEMBLE_SETTINGS["ensemble_3_best"]:
            reg = server._get_calibration_entry(m) or {}
            cal_meta.append({
                "available": bool(reg),
                "temperature": reg.get("temperature", 1.0),
                "ece_before": reg.get("ece_before"),
                "ece_after": reg.get("ece_after"),
            })
        safety = server._merge_phase2_ensemble_safety(safety, ensemble3[idx], ENSEMBLE_SETTINGS["ensemble_3_best"], per_model_embs, cal_meta, raw_probabilities=ensemble3[idx])
        abstained.append(int(safety["decision_status"] == "abstain"))
        if safety["decision_status"] != "abstain":
            kept_probs.append(ensemble3[idx])
            kept_true.append(y_true[idx])

    kept_probs = np.asarray(kept_probs, dtype=np.float32)
    kept_true = np.asarray(kept_true, dtype=np.int64)
    rows.append({
        "setting": "ensemble_3_best_safety",
        "display": "Ensemble 3-Model + Safety Abstain",
        **multiclass_metrics(kept_true, kept_probs, kept_probs.argmax(axis=1)),
        "coverage": round(float(1.0 - np.mean(abstained)), 4),
    })

    out_csv = OUT_DIR / "ensemble_ablation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_risk_coverage():
    aligned = {m: {row["slide_id"]: row for row in load_phase1_rows(m)} for m in SHORTLIST}
    slide_ids = sorted(set.intersection(*(set(x.keys()) for x in aligned.values())))
    probs = []
    labels = []
    safety_scores = []
    abstain_flags = []

    for sid in slide_ids:
        ensemble_probs = []
        ensemble_preds = []
        ensemble_embs = []
        for model_key in ENSEMBLE_SETTINGS["ensemble_3_best"]:
            row = aligned[model_key][sid]
            raw = np.asarray([
                float(row["prob_normal_benign"]),
                float(row["prob_bcc"]),
                float(row["prob_scc"]),
                float(row["prob_melanoma"]),
            ], dtype=np.float32)
            calibrated, _ = server._apply_probability_calibration(raw, model_key)
            ensemble_probs.append(calibrated)
            ensemble_preds.append(int(calibrated.argmax()))
            ensemble_embs.append(slide_embedding_from_feature(model_key, sid))
        avg = np.mean(ensemble_probs, axis=0)
        safety = server._build_phase1_safety(int(avg.argmax()), avg, ensemble_predictions=ensemble_preds)
        cal_meta = []
        for model_key in ENSEMBLE_SETTINGS["ensemble_3_best"]:
            reg = server._get_calibration_entry(model_key) or {}
            cal_meta.append({
                "available": bool(reg),
                "temperature": reg.get("temperature", 1.0),
                "ece_before": reg.get("ece_before"),
                "ece_after": reg.get("ece_after"),
            })
        safety = server._merge_phase2_ensemble_safety(safety, avg, ENSEMBLE_SETTINGS["ensemble_3_best"], ensemble_embs, cal_meta, raw_probabilities=avg)
        probs.append(avg)
        labels.append(LABEL_TO_ID[aligned[SHORTLIST[0]][sid]["true_label"]])
        safety_scores.append(float(safety.get("safety_score", 0.0)))
        abstain_flags.append(int(safety["decision_status"] == "abstain"))

    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    safety_scores = np.asarray(safety_scores, dtype=np.float32)
    abstain_flags = np.asarray(abstain_flags, dtype=np.int64)

    levels = np.linspace(0.4, 1.0, 7)
    rows = []
    for coverage_target in levels:
        keep_n = max(1, int(round(len(labels) * coverage_target)))
        keep_idx = np.argsort(safety_scores)[:keep_n]
        kept_true = labels[keep_idx]
        kept_probs = probs[keep_idx]
        preds = kept_probs.argmax(axis=1)
        rows.append({
            "target_coverage": round(float(coverage_target), 4),
            "realized_coverage": round(float(len(keep_idx) / len(labels)), 4),
            "macro_f1": round(float(f1_score(kept_true, preds, average="macro")), 4),
            "melanoma_recall": round(float(recall_score(kept_true == 3, preds == 3)), 4),
            "melanoma_fn": int(np.sum((kept_true == 3) & (preds != 3))),
            "mean_safety_score": round(float(np.mean(safety_scores[keep_idx])), 4),
        })

    rows.append({
        "target_coverage": "abstain_policy",
        "realized_coverage": round(float(1.0 - np.mean(abstain_flags)), 4),
        "macro_f1": round(float(f1_score(labels[abstain_flags == 0], probs[abstain_flags == 0].argmax(axis=1), average="macro")), 4),
        "melanoma_recall": round(float(recall_score(labels[abstain_flags == 0] == 3, probs[abstain_flags == 0].argmax(axis=1) == 3)), 4),
        "melanoma_fn": int(np.sum((labels[abstain_flags == 0] == 3) & (probs[abstain_flags == 0].argmax(axis=1) != 3))),
        "mean_safety_score": round(float(np.mean(safety_scores[abstain_flags == 0])), 4) if np.any(abstain_flags == 0) else None,
    })

    out_csv = OUT_DIR / "risk_coverage.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = [float(r["realized_coverage"]) for r in rows[:-1]]
    y = [float(r["melanoma_fn"]) for r in rows[:-1]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", color="#dc2626")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Melanoma FN")
    ax.set_title("Risk-Coverage Curve - Ensemble 3 + Safety")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_coverage_curve.png", dpi=160)
    plt.close(fig)
    return rows


def build_explanation_ablation():
    case_json = PROJECT_DIR / "results" / "phase3_case_study" / "phase3_case_study_cases.json"
    cases = json.loads(case_json.read_text(encoding="utf-8"))
    rows = []
    md = [
        "# Phase 6 Explanation Ablation",
        "",
        "This report compares the explanation variants generated for the 10-case Phase 3 set.",
        "",
        "| Case | True Label | Prediction | Consensus | Disagreement | Shared | Mel vs SCC | Mel vs BCC |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, case in enumerate(cases, 1):
        if case.get("status") != "completed":
            continue
        heat = case["heatmap_paths"]
        rows.append({
            "case": idx,
            "filename": case["filename"],
            "true_label": case["true_label"],
            "prediction": case["pred_label"],
            "consensus": heat["consensus"],
            "disagreement": heat["disagreement"],
            "shared": heat["shared"],
            "mel_vs_scc": heat["mel_vs_scc"],
            "mel_vs_bcc": heat["mel_vs_bcc"],
        })
        md.append(
            f"| {idx} | {case['true_label']} | {case['pred_label']} | "
            f"[Consensus]({heat['consensus']}) | [Disagreement]({heat['disagreement']}) | "
            f"[Shared]({heat['shared']}) | [Mel vs SCC]({heat['mel_vs_scc']}) | [Mel vs BCC]({heat['mel_vs_bcc']}) |"
        )
    (EXPL_DIR / "explanation_ablation.md").write_text("\n".join(md), encoding="utf-8")
    with (EXPL_DIR / "explanation_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def build_retrieval_ablation():
    reg = json.loads((PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_registry.json").read_text(encoding="utf-8"))
    case_study = json.loads((PROJECT_DIR / "results" / "phase3_case_study" / "phase3_case_study_cases.json").read_text(encoding="utf-8"))
    hard_ids = set(reg.get("hard_case_slide_ids") or [])
    report = [
        "# Phase 6 Retrieval Ablation",
        "",
        "This report contrasts base case interpretation with retrieval-supported review.",
        "",
        "| Case | True Label | Prediction | Retrieval Support | Hard Melanoma Context |",
        "| --- | --- | --- | --- | --- |",
    ]

    all_cases = reg.get("cases", {})
    bank = reg.get("banks", {}).get("ensemble_3_best", {})
    case_ids = bank.get("case_ids", [])
    vectors = np.load(PROJECT_DIR / "results" / "phase4_retrieval" / "retrieval_embeddings.npz", allow_pickle=False)["ensemble_3_best"]
    id_to_idx = {sid: idx for idx, sid in enumerate(case_ids)}

    rows = []
    for idx, case in enumerate(case_study, 1):
        sid = Path(case["slide_path"]).stem
        if sid not in id_to_idx:
            continue
        vec = vectors[id_to_idx[sid]]
        scores = vectors @ vec
        order = np.argsort(scores)[::-1]
        matches = []
        for j in order:
            mid = case_ids[j]
            if mid == sid:
                continue
            meta = all_cases.get(mid)
            if not meta:
                continue
            matches.append({
                "slide_id": mid,
                "true_label": meta["true_label"],
                "source": meta["source"],
                "similarity": round(float(scores[j]), 4),
                "is_hard_melanoma": mid in hard_ids,
            })
            if len(matches) >= 3:
                break
        hard_support = any(m["is_hard_melanoma"] for m in matches)
        support_text = "; ".join(f"{m['true_label']} ({m['similarity']:.3f})" for m in matches)
        report.append(f"| {idx} | {case['true_label']} | {case['pred_label']} | {support_text} | {'Yes' if hard_support else 'No'} |")
        rows.append({
            "case": idx,
            "slide_id": sid,
            "true_label": case["true_label"],
            "prediction": case["pred_label"],
            "matches": matches,
            "hard_support": hard_support,
        })

    (RETR_DIR / "retrieval_ablation.md").write_text("\n".join(report), encoding="utf-8")
    with (RETR_DIR / "retrieval_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def ood_category_rows():
    rows = list(csv.DictReader(OOD_LABELS_CSV.open()))
    existing = {p.stem for p in OOD_IMAGES_DIR.glob("*.tif")}
    out = []
    for row in rows:
        if row["filename"] not in existing:
            continue
        if row["category"] in OOD_ID_CATEGORIES:
            continue
        out.append({
            "slide_id": row["filename"],
            "slide_path": str(OOD_IMAGES_DIR / f"{row['filename']}.tif"),
            "category": row["category"],
        })
    return out


def build_ood_benchmark():
    cache_json = OOD_DIR / f"{OOD_TARGET_MODEL}_ood_scores.json"
    if cache_json.exists():
        scores = json.loads(cache_json.read_text(encoding="utf-8"))
    else:
        id_rows = load_phase1_rows(OOD_TARGET_MODEL)
        scores = {"id": [], "ood": []}
        for row in id_rows:
            emb = slide_embedding_from_feature(OOD_TARGET_MODEL, row["slide_id"])
            ood = server._estimate_ood_from_embedding(emb, OOD_TARGET_MODEL)
            scores["id"].append({
                "slide_id": row["slide_id"],
                "true_label": row["true_label"],
                "ood_score": ood["ood_score"],
            })
        for idx, row in enumerate(ood_category_rows(), 1):
            openslide = server._ensure_openslide()
            slide = openslide.OpenSlide(row["slide_path"])
            tiles, _ = server._extract_tiles(slide, f"p6ood_{idx:04d}")
            slide.close()
            features = server._extract_features(tiles, OOD_TARGET_MODEL)
            _, _, _, bag_embedding, _, _ = server._run_mil_inference(features, OOD_TARGET_MODEL)
            ood = server._estimate_ood_from_embedding(bag_embedding, OOD_TARGET_MODEL)
            scores["ood"].append({
                "slide_id": row["slide_id"],
                "category": row["category"],
                "ood_score": ood["ood_score"],
            })
            if idx % 25 == 0:
                print(f"[ood] processed {idx}/{len(ood_category_rows())}")
        cache_json.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    y_true = np.asarray([0] * len(scores["id"]) + [1] * len(scores["ood"]), dtype=np.int64)
    y_score = np.asarray([x["ood_score"] for x in scores["id"]] + [x["ood_score"] for x in scores["ood"]], dtype=np.float32)
    auroc = float(roc_auc_score(y_true, y_score))
    aupr = float(average_precision_score(y_true, y_score))

    thresholds = np.sort(np.unique(y_score))[::-1]
    fpr95 = None
    for thr in thresholds:
        pred = (y_score >= thr).astype(np.int64)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if tpr >= 0.95:
            fpr95 = float(fpr)
            break

    per_class = []
    grouped = defaultdict(list)
    id_scores = [x["ood_score"] for x in scores["id"]]
    for row in scores["ood"]:
        grouped[row["category"]].append(row["ood_score"])
    for category, ood_scores in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        y_true_cls = np.asarray([0] * len(id_scores) + [1] * len(ood_scores), dtype=np.int64)
        y_score_cls = np.asarray(id_scores + ood_scores, dtype=np.float32)
        per_class.append({
            "category": category,
            "n_cases": len(ood_scores),
            "auroc": round(float(roc_auc_score(y_true_cls, y_score_cls)), 4),
            "aupr": round(float(average_precision_score(y_true_cls, y_score_cls)), 4),
            "mean_ood_score": round(float(np.mean(ood_scores)), 4),
        })

    summary = {
        "model_key": OOD_TARGET_MODEL,
        "model_display": server.MODEL_REGISTRY[OOD_TARGET_MODEL]["display"],
        "id_cases": len(scores["id"]),
        "ood_cases": len(scores["ood"]),
        "auroc": round(auroc, 4),
        "aupr": round(aupr, 4),
        "fpr95": round(fpr95, 4) if fpr95 is not None else None,
        "per_class": per_class,
    }
    (OOD_DIR / "ood_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OOD_DIR / "ood_benchmark_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)
    return summary


def build_master_report(cal_rows, risk_rows, ensemble_rows, ood_summary):
    lines = [
        "# Phase 6 Evaluation Report",
        "",
        f"- Generated: {datetime.now().isoformat()}",
        "",
        "## REQ-601 Risk-Coverage",
        "",
        "| Coverage | Macro F1 | Melanoma Recall | Melanoma FN | Mean Safety Score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in risk_rows:
        lines.append(
            f"| {row['realized_coverage']} | {row['macro_f1']} | {row['melanoma_recall']} | {row['melanoma_fn']} | {row['mean_safety_score']} |"
        )

    lines.extend([
        "",
        "## REQ-603 Calibration",
        "",
        "| Model | ECE Before | ECE After | MCE Before | MCE After | Temperature | Plot |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in cal_rows:
        lines.append(
            f"| {row['model_display']} | {row['ece_before']} | {row['ece_after']} | {row['mce_before']} | {row['mce_after']} | {row['temperature']} | [plot]({row['plot_path']}) |"
        )

    lines.extend([
        "",
        "## REQ-604 Ensemble Ablation",
        "",
        "| Setting | Coverage | Macro F1 | Melanoma Recall | Melanoma FN | AUC OVR |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for row in ensemble_rows:
        lines.append(
            f"| {row['display']} | {row['coverage']} | {row['macro_f1']} | {row['melanoma_recall']} | {row['melanoma_fn']} | {row['auc_ovr']} |"
        )

    lines.extend([
        "",
        "## REQ-602 OOD Benchmark",
        "",
        f"- Model: `{ood_summary['model_display']}`",
        f"- ID cases: `{ood_summary['id_cases']}`",
        f"- OOD cases: `{ood_summary['ood_cases']}`",
        f"- AUROC: `{ood_summary['auroc']}`",
        f"- AUPR: `{ood_summary['aupr']}`",
        f"- FPR95: `{ood_summary['fpr95']}`",
        "",
        "| OOD Category | Cases | AUROC | AUPR | Mean OOD Score |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in ood_summary["per_class"]:
        lines.append(
            f"| {row['category']} | {row['n_cases']} | {row['auroc']} | {row['aupr']} | {row['mean_ood_score']} |"
        )

    lines.extend([
        "",
        "## REQ-605 / REQ-606",
        "",
        f"- Explanation ablation: [explanation_ablation.md]({EXPL_DIR / 'explanation_ablation.md'})",
        f"- Retrieval ablation: [retrieval_ablation.md]({RETR_DIR / 'retrieval_ablation.md'})",
    ])
    (OUT_DIR / "phase6_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    cal_rows = build_calibration_analysis()
    ensemble_rows = build_ensemble_ablation()
    risk_rows = build_risk_coverage()
    build_explanation_ablation()
    build_retrieval_ablation()
    ood_summary = build_ood_benchmark()
    build_master_report(cal_rows, risk_rows, ensemble_rows, ood_summary)
    print(f"Phase 6 outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
