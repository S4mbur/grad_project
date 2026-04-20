#!/usr/bin/env python3
"""Build canonical Phase 0 registry artifacts.

This script consolidates the project's fragmented dataset, run, threshold, and
error-analysis metadata into one reproducible Phase 0 bundle under
`results/phase0_registry/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_metrics import aggregate_error_bank  # noqa: E402


def resolve_data_root() -> Path:
    candidates = [
        Path("/mnt/d/skin_cancer_project/datasets"),
        Path("D:/skin_cancer_project/datasets"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_ROOT = resolve_data_root()
LABEL_ROOT = DATA_ROOT / "labels"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = RESULTS_ROOT / "phase0_registry"
PHASE1_BANK_CSV = RESULTS_ROOT / "phase1_hard_case_bank" / "all_test_predictions.csv"
PHASE1_HARD_BANK_CSV = RESULTS_ROOT / "phase1_hard_case_bank" / "hard_case_bank.csv"
ENSEMBLE_SEARCH_JSON = RESULTS_ROOT / "ensemble_search_results.json"

CLASS_NAMES = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
OOD_CLASS_MAP = {
    "Benign": 0,
    "No abnormalities": 0,
    "Benign sebaceous gland tumor": 0,
    "Cylindroma": 0,
    "Basal cell carcinoma": 1,
    "Squamous cell carcinoma": 2,
    "Melanoma": 3,
    "Melanoma in situ": 3,
    "Merkel cell carcinoma": None,
    "Sebaceous gland carcinoma": None,
    "Microcystic adnexal carcinoma": None,
    "Skin adnexal carcinoma, other": None,
    "Lymphoma": None,
    "Cutaneous metastases": None,
}
SHORTLIST_APP_MODEL_KEYS = {
    "uni_cost_sensitive_strong",
    "uni_focal_g3",
    "phikon_cost_sensitive_strong",
    "conch_cost_sensitive_strong",
}
ENSEMBLE_COMPONENTS = {
    "ensemble_2_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong"],
    "ensemble_3_best": ["uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong"],
}


@dataclass
class SlideProbe:
    valid: bool
    corrupt: bool
    width: int | None
    height: int | None
    mpp: float | None
    vendor: str | None
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical Phase 0 registry artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction of remaining ID pool.")
    parser.add_argument("--workers", type=int, default=6, help="Integrity probe worker count.")
    parser.add_argument("--skip-probe", action="store_true", help="Skip file-level slide open checks.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def first_existing_path(candidates: Iterable[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n  - {checked}")


def deterministic_stratified_split(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["target_class"]].append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for class_name, class_rows in sorted(grouped.items()):
        rng = random.Random(f"{seed}:{class_name}")
        ordered = list(class_rows)
        rng.shuffle(ordered)
        desired_val = int(round(len(ordered) * val_fraction))
        min_val = 1 if len(ordered) >= 2 else 0
        max_val = max(len(ordered) - 1, 0)
        val_count = min(max(desired_val, min_val), max_val)
        val_rows.extend(ordered[:val_count])
        train_rows.extend(ordered[val_count:])
    return train_rows, val_rows


def slide_probe(path: Path) -> SlideProbe:
    if not path.exists():
        return SlideProbe(False, True, None, None, None, None, "missing")

    suffix = path.suffix.lower()
    try:
        if suffix in {".svs", ".ndpi", ".mrxs", ".scn"}:
            import openslide  # type: ignore

            slide = openslide.OpenSlide(str(path))
            width, height = slide.dimensions
            mpp = safe_float(slide.properties.get("openslide.mpp-x"))
            vendor = slide.properties.get("openslide.vendor")
            slide.close()
            return SlideProbe(True, False, int(width), int(height), mpp, vendor, "")

        with Image.open(path) as image:
            width, height = image.size
        return SlideProbe(True, False, int(width), int(height), None, None, "")
    except Exception as exc:  # pragma: no cover - depends on local slide set
        return SlideProbe(False, True, None, None, None, None, str(exc)[:240])


def build_source_rows() -> list[dict]:
    rows: list[dict] = []

    label_roots = [
        LABEL_ROOT,
        PROJECT_ROOT / "data" / "cobra",
        PROJECT_ROOT / "data" / "cobra_fresh",
    ]
    bcc_csv = first_existing_path((root / "bcc_bcc.csv" for root in label_roots), "BCC label CSV")
    ood_csv = first_existing_path(
        list(root / "ood_disease_types.csv" for root in label_roots)
        + list(root / "ood_labels" / "ood_disease_types.csv" for root in label_roots)
        + list(root / "ood_labels" / "labels" / "ood_disease_types.csv" for root in label_roots),
        "OOD label CSV",
    )

    bcc_dir = DATA_ROOT / "cobra_bcc"
    for item in read_csv(bcc_csv):
        slide_id = item["filename"]
        path = bcc_dir / f"{slide_id}.tif"
        label = "Normal/Benign" if item["label"] == "0" else "BCC"
        rows.append({
            "slide_id": slide_id,
            "source": "cobra_bcc",
            "raw_path": str(path),
            "downloaded": int(path.exists()),
            "original_label": label,
            "target_class": label,
            "used_for_ood": 0,
            "notes": "",
        })

    ood_dir = DATA_ROOT / "cobra_ood" / "images"
    for item in read_csv(ood_csv):
        slide_id = item["filename"]
        path = ood_dir / f"{slide_id}.tif"
        category = item["category"]
        mapped = OOD_CLASS_MAP.get(category)
        rows.append({
            "slide_id": slide_id,
            "source": "cobra_ood",
            "raw_path": str(path),
            "downloaded": int(path.exists()),
            "original_label": category,
            "target_class": CLASS_NAMES[mapped] if mapped is not None else "",
            "used_for_ood": int(mapped is None),
            "notes": "",
        })

    tcga_dir = DATA_ROOT / "tcga_skcm"
    for path in sorted(tcga_dir.glob("*.svs")):
        rows.append({
            "slide_id": path.stem,
            "source": "tcga_skcm",
            "raw_path": str(path),
            "downloaded": 1,
            "original_label": "Melanoma (TCGA)",
            "target_class": "Melanoma",
            "used_for_ood": 0,
            "notes": "",
        })

    deduped = {}
    for row in rows:
        deduped[(row["slide_id"], row["source"])] = row
    return list(deduped.values())


def build_eval_bank() -> dict[tuple[str, str], dict]:
    eval_bank = {}
    for row in read_csv(PHASE1_BANK_CSV):
        key = (row["slide_id"], row["source"])
        eval_bank[key] = row
    return eval_bank


def assign_splits(rows: list[dict], eval_bank: dict[tuple[str, str], dict], seed: int, val_fraction: float) -> list[dict]:
    id_rows = []
    ood_rows = []
    for row in rows:
        key = (row["slide_id"], row["source"])
        row["used_for_test"] = int(key in eval_bank)
        row["used_for_training"] = 0
        row["used_for_validation"] = 0
        row["split"] = ""
        if row["used_for_ood"]:
            row["split"] = "ood"
            ood_rows.append(row)
        elif row["target_class"]:
            if row["used_for_test"]:
                row["split"] = "test"
            id_rows.append(row)

    remaining = [row for row in id_rows if row["split"] != "test"]
    if remaining:
        val_size = max(val_fraction, len(CLASS_NAMES) / max(len(remaining), 1))
        train_rows, val_rows = deterministic_stratified_split(
            remaining,
            val_fraction=val_size,
            seed=seed,
        )
        for row in train_rows:
            row["split"] = "train"
            row["used_for_training"] = 1
        for row in val_rows:
            row["split"] = "val"
            row["used_for_validation"] = 1

    for row in id_rows:
        if row["split"] == "test":
            row["used_for_test"] = 1
    for row in ood_rows:
        row["used_for_ood"] = 1
    return rows


def enrich_probe(rows: list[dict], workers: int, skip_probe: bool) -> None:
    if skip_probe:
        for row in rows:
            path = Path(row["raw_path"])
            row["file_size_gb"] = round(path.stat().st_size / (1024 ** 3), 6) if path.exists() else None
            row["valid"] = int(path.exists())
            row["corrupt"] = int(not path.exists())
            row["width"] = None
            row["height"] = None
            row["mpp"] = None
            row["vendor"] = None
            row["probe_note"] = "skipped"
        return

    def one(row: dict) -> dict:
        path = Path(row["raw_path"])
        probe = slide_probe(path)
        row["file_size_gb"] = round(path.stat().st_size / (1024 ** 3), 6) if path.exists() else None
        row["valid"] = int(probe.valid)
        row["corrupt"] = int(probe.corrupt)
        row["width"] = probe.width
        row["height"] = probe.height
        row["mpp"] = round(probe.mpp, 6) if probe.mpp is not None else None
        row["vendor"] = probe.vendor
        row["probe_note"] = probe.note
        return row

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        enriched = list(pool.map(one, rows))
    rows[:] = enriched


def run_id_to_model_key(run_id: str) -> str | None:
    if not run_id.startswith("mil_4class_"):
        return None
    name = run_id.replace("mil_4class_", "", 1)
    if "_v3_fast_" in name:
        backbone, experiment = name.split("_v3_fast_", 1)
    elif "_v3_" in name:
        backbone, experiment = name.split("_v3_", 1)
    else:
        return None
    return f"{backbone}_{experiment}"


def classify_training_mode(run_id: str) -> str:
    return "fast" if "_v3_fast_" in run_id else "normal"


def build_experiment_registry() -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    by_model_key: dict[str, dict] = {}
    for result_dir in sorted(RESULTS_ROOT.glob("mil_4class_*")):
        results_path = result_dir / "results.json"
        if not results_path.exists():
            continue
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        run_id = result_dir.name
        model_key = run_id_to_model_key(run_id)
        metrics = payload.get("metrics", {})
        threshold = payload.get("threshold_tuning", {})
        row = {
            "run_id": run_id,
            "model_name": payload.get("model"),
            "experiment": payload.get("experiment"),
            "description": payload.get("description"),
            "training_mode": classify_training_mode(run_id),
            "app_model_key": model_key or "",
            "is_shortlist_model": int((model_key or "") in SHORTLIST_APP_MODEL_KEYS),
            "results_json": str(results_path),
            "checkpoint_path": str(result_dir / "best_model.pt"),
            "summary_path": str(result_dir / "summary.txt"),
            "predictions_csv": str(result_dir / "phase1_test_predictions.csv"),
            "hard_cases_csv": str(result_dir / "phase1_hard_cases.csv"),
            "best_epoch": payload.get("best_epoch"),
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "auc_roc": metrics.get("auc_roc"),
            "melanoma_recall": metrics.get("melanoma_recall"),
            "melanoma_fn": metrics.get("melanoma_fn"),
            "best_melanoma_threshold": threshold.get("best_melanoma_threshold"),
            "best_f1_with_threshold": threshold.get("best_f1_with_threshold"),
        }
        rows.append(row)
        if model_key:
            by_model_key[model_key] = row
    return rows, by_model_key


def build_threshold_registry(experiment_rows: dict[str, dict]) -> dict:
    registry = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": {},
        "ensembles": {},
    }

    for model_key, row in sorted(experiment_rows.items()):
        registry["models"][model_key] = {
            "default_threshold": 0.5,
            "melanoma_safe_threshold": safe_float(row.get("best_melanoma_threshold")),
            "selection_basis": "stored results.json threshold_tuning sweep",
            "source_run": row.get("run_id"),
            "source_results": row.get("results_json"),
            "source_csv": row.get("predictions_csv"),
            "evaluation_split": "test",
            "notes": "Use as review/triage threshold, not as a replacement for multiclass argmax.",
        }

    if ENSEMBLE_SEARCH_JSON.exists():
        payload = json.loads(ENSEMBLE_SEARCH_JSON.read_text(encoding="utf-8"))
        for bucket in ("pair", "triple", "quad", "quint"):
            for key, value in (payload.get(bucket) or {}).items():
                if not isinstance(value, dict):
                    continue
                registry["ensembles"][key] = {
                    "default_threshold": 0.5,
                    "melanoma_safe_threshold": safe_float(value.get("best_thresh")),
                    "selection_basis": "ensemble_search_results.json melanoma-safe sweep",
                    "source_run": key,
                    "evaluation_split": "test",
                    "notes": "Legacy ensemble threshold derived from ensemble search artifact.",
                }

    for ensemble_key, components in ENSEMBLE_COMPONENTS.items():
        component_thresholds = [
            registry["models"][component]["melanoma_safe_threshold"]
            for component in components
            if component in registry["models"] and registry["models"][component]["melanoma_safe_threshold"] is not None
        ]
        if component_thresholds:
            registry["ensembles"][ensemble_key] = {
                "default_threshold": 0.5,
                "melanoma_safe_threshold": round(float(np.mean(component_thresholds)), 4),
                "selection_basis": "mean of component shortlist thresholds",
                "source_run": ",".join(components),
                "evaluation_split": "test",
                "notes": "Shortlist ensemble review threshold aggregated from component runs.",
            }

    return registry


def build_error_bank() -> tuple[list[dict], list[dict]]:
    rows = read_csv(PHASE1_BANK_CSV)
    normalized = []
    for row in rows:
        normalized.append({
            "slide_id": row["slide_id"],
            "source": row["source"],
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "prediction_confidence": row.get("prediction_confidence"),
            "margin": row.get("margin"),
            "melanoma_probability": row.get("melanoma_probability"),
            "is_melanoma": row.get("is_melanoma"),
            "is_melanoma_fn": row.get("is_melanoma_fn"),
            "hard_case_candidate": row.get("hard_case_candidate"),
            "prob_normal_benign": row.get("prob_normal_benign"),
            "prob_bcc": row.get("prob_bcc"),
            "prob_scc": row.get("prob_scc"),
            "prob_melanoma": row.get("prob_melanoma"),
            "result_dir": row.get("result_dir", ""),
            "model_name": run_id_to_model_key(Path(row.get("result_dir", "")).name) or "",
            "experiment": Path(row.get("result_dir", "")).name,
            "is_correct": int(row["true_label"] == row["pred_label"]),
            "error_type": "melanoma_fn" if row.get("is_melanoma_fn") == "1" else ("correct" if row["true_label"] == row["pred_label"] else "mismatch"),
        })
    return normalized, aggregate_error_bank(normalized)


def build_source_counts(rows: list[dict]) -> dict:
    source_counts: dict[str, dict] = defaultdict(lambda: {
        "downloaded": 0,
        "valid": 0,
        "corrupt": 0,
        "ood_pool": 0,
        "train": 0,
        "val": 0,
        "test": 0,
        "class_counts": Counter(),
    })
    for row in rows:
        bucket = source_counts[row["source"]]
        bucket["downloaded"] += int(row["downloaded"])
        bucket["valid"] += int(row["valid"])
        bucket["corrupt"] += int(row["corrupt"])
        bucket["ood_pool"] += int(row["used_for_ood"])
        if row["split"] in ("train", "val", "test"):
            bucket[row["split"]] += 1
        if row["target_class"]:
            bucket["class_counts"][row["target_class"]] += 1

    serializable = {}
    for source, payload in source_counts.items():
        serializable[source] = dict(payload)
        serializable[source]["class_counts"] = dict(payload["class_counts"])
    return serializable


def build_readme(rows: list[dict], run_rows: list[dict], threshold_registry: dict, error_rows: list[dict], error_summary: list[dict]) -> str:
    split_counts = Counter(row["split"] for row in rows)
    class_counts = Counter(row["target_class"] for row in rows if row["target_class"])
    shortlist_count = sum(1 for row in run_rows if row["is_shortlist_model"])
    return f"""# Phase 0 Canonical Registry

Generated: {datetime.now().isoformat(timespec='seconds')}

## Summary

- Total slide records: {len(rows)}
- Train slides: {split_counts.get('train', 0)}
- Validation slides: {split_counts.get('val', 0)}
- Test slides: {split_counts.get('test', 0)}
- OOD pool slides: {split_counts.get('ood', 0)}
- 4-class distribution: {dict(class_counts)}
- Experiment registry rows: {len(run_rows)}
- Shortlist app-linked runs: {shortlist_count}
- Threshold registry entries:
  - models: {len(threshold_registry.get('models', {}))}
  - ensembles: {len(threshold_registry.get('ensembles', {}))}
- Error-analysis rows: {len(error_rows)}
- Error-analysis slide summaries: {len(error_summary)}

## Main Artifacts

- `master_slide_manifest.csv`
- `master_slide_manifest.json`
- `source_count_report.json`
- `split_manifests/train.csv`
- `split_manifests/val.csv`
- `split_manifests/test.csv`
- `split_manifests/ood.csv`
- `experiment_registry.csv`
- `experiment_registry.json`
- `threshold_registry.json`
- `error_analysis_bank.csv`
- `error_analysis_slide_summary.csv`

## Notes

- The test split is frozen from the real evaluation bank used by completed runs.
- The train/validation split is deterministically regenerated from the remaining 4-class in-distribution pool using seed 42.
- Threshold entries are stored as review/triage thresholds and should not be treated as a direct replacement for multiclass argmax.
"""


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "split_manifests").mkdir(parents=True, exist_ok=True)

    rows = build_source_rows()
    eval_bank = build_eval_bank()
    rows = assign_splits(rows, eval_bank, seed=args.seed, val_fraction=args.val_fraction)
    enrich_probe(rows, workers=args.workers, skip_probe=args.skip_probe)

    run_rows, run_lookup = build_experiment_registry()
    threshold_registry = build_threshold_registry(run_lookup)
    error_rows, error_summary = build_error_bank()
    source_counts = build_source_counts(rows)

    manifest_fieldnames = [
        "slide_id", "source", "raw_path", "downloaded", "valid", "corrupt",
        "file_size_gb", "width", "height", "mpp", "vendor",
        "original_label", "target_class", "split",
        "used_for_training", "used_for_validation", "used_for_test", "used_for_ood",
        "probe_note", "notes",
    ]
    experiment_fieldnames = [
        "run_id", "model_name", "experiment", "description", "training_mode",
        "app_model_key", "is_shortlist_model", "results_json", "checkpoint_path",
        "summary_path", "predictions_csv", "hard_cases_csv", "best_epoch",
        "accuracy", "f1_macro", "f1_weighted", "auc_roc", "melanoma_recall",
        "melanoma_fn", "best_melanoma_threshold", "best_f1_with_threshold",
    ]
    error_fieldnames = list(error_rows[0].keys()) if error_rows else []
    error_summary_fields = list(error_summary[0].keys()) if error_summary else []

    write_csv(OUTPUT_ROOT / "master_slide_manifest.csv", rows, manifest_fieldnames)
    (OUTPUT_ROOT / "master_slide_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "source_count_report.json").write_text(json.dumps(source_counts, indent=2), encoding="utf-8")
    write_csv(OUTPUT_ROOT / "split_manifests" / "train.csv", [r for r in rows if r["split"] == "train"], manifest_fieldnames)
    write_csv(OUTPUT_ROOT / "split_manifests" / "val.csv", [r for r in rows if r["split"] == "val"], manifest_fieldnames)
    write_csv(OUTPUT_ROOT / "split_manifests" / "test.csv", [r for r in rows if r["split"] == "test"], manifest_fieldnames)
    write_csv(OUTPUT_ROOT / "split_manifests" / "ood.csv", [r for r in rows if r["split"] == "ood"], manifest_fieldnames)
    write_csv(OUTPUT_ROOT / "experiment_registry.csv", run_rows, experiment_fieldnames)
    (OUTPUT_ROOT / "experiment_registry.json").write_text(json.dumps(run_rows, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "threshold_registry.json").write_text(json.dumps(threshold_registry, indent=2), encoding="utf-8")
    if error_rows:
        write_csv(OUTPUT_ROOT / "error_analysis_bank.csv", error_rows, error_fieldnames)
    if error_summary:
        write_csv(OUTPUT_ROOT / "error_analysis_slide_summary.csv", error_summary, error_summary_fields)
    (OUTPUT_ROOT / "README.md").write_text(
        build_readme(rows, run_rows, threshold_registry, error_rows, error_summary),
        encoding="utf-8",
    )

    print(f"Phase 0 registry written to: {OUTPUT_ROOT}")
    print(f"Master manifest rows: {len(rows)}")
    print(f"Experiment registry rows: {len(run_rows)}")
    print(f"Threshold registry entries: {len(threshold_registry.get('models', {})) + len(threshold_registry.get('ensembles', {}))}")
    print(f"Error-analysis rows: {len(error_rows)}")


if __name__ == "__main__":
    main()
