#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import server


DEFAULT_SOURCE_CSV = PROJECT_DIR / "results" / "mil_4class_uni_v3_fast_cost_sensitive_strong" / "phase1_test_predictions.csv"
DEFAULT_HARD_CASE_CSV = PROJECT_DIR / "results" / "mil_4class_uni_v3_fast_cost_sensitive_strong" / "phase1_hard_cases.csv"
REPORT_DIR = PROJECT_DIR / "results" / "phase3_case_study"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Build a qualitative Phase 3 heatmap case-study report.")
    p.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    p.add_argument("--hard-case-csv", type=Path, default=DEFAULT_HARD_CASE_CSV)
    p.add_argument("--model", default="ensemble_3_best")
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--extra-melanoma-hard-cases", type=int, default=2)
    p.add_argument("--limit", type=int, default=10)
    return p.parse_args()


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_cases(source_rows, hard_rows, per_class=2, extra_melanoma_hard_cases=2, limit=10):
    chosen = []
    chosen_ids = set()
    by_label = defaultdict(list)
    for row in source_rows:
        if Path(row["slide_path"]).exists():
            by_label[row["true_label"]].append(row)

    for label in ["Normal/Benign", "BCC", "SCC", "Melanoma"]:
        for row in by_label.get(label, [])[:per_class]:
            if row["slide_id"] not in chosen_ids:
                chosen.append(row)
                chosen_ids.add(row["slide_id"])

    melanoma_hard = [
        row for row in hard_rows
        if row.get("true_label") == "Melanoma" and Path(row["slide_path"]).exists()
    ]
    for row in melanoma_hard:
        if len([x for x in chosen if x["true_label"] == "Melanoma"]) >= per_class + extra_melanoma_hard_cases:
            break
        if row["slide_id"] not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(row["slide_id"])

    if len(chosen) < limit:
        for row in source_rows:
            if len(chosen) >= limit:
                break
            if row["slide_id"] in chosen_ids:
                continue
            if not Path(row["slide_path"]).exists():
                continue
            chosen.append(row)
            chosen_ids.add(row["slide_id"])

    return chosen[:limit]


def run_case(case_row, model_key, case_idx):
    slide_path = case_row["slide_path"]
    filename = Path(slide_path).name
    job_id = f"p3case{case_idx:02d}_{uuid.uuid4().hex[:6]}"
    model_display = server.ENSEMBLE_PRESETS.get(model_key, {}).get("name", model_key)

    with server.analyses_lock:
        server.analyses[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued for case-study analysis.",
            "filename": filename,
            "slide_path": slide_path,
            "model_key": model_key,
            "model_display": model_display,
            "created_at": datetime.now().isoformat(),
            "result": None,
        }

    server.run_analysis(job_id, slide_path, model_key)

    with server.analyses_lock:
        job = dict(server.analyses[job_id])

    if job.get("status") != "completed" or not job.get("result"):
        return {
            "job_id": job_id,
            "status": job.get("status", "error"),
            "error": job.get("message", "Unknown error"),
            "slide_path": slide_path,
            "filename": filename,
            "true_label": case_row["true_label"],
        }

    export_data = {
        "job_id": job_id,
        "filename": filename,
        "analysis_date": job.get("created_at"),
        "model": job.get("model_display"),
        "source_case": case_row,
        "result": job["result"],
        "slide_info": job.get("slide_info"),
    }
    export_path = PROJECT_DIR / "app" / "results" / job_id / "export.json"
    export_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")

    result = job["result"]
    result_dir = PROJECT_DIR / "app" / "results" / job_id
    return {
        "job_id": job_id,
        "status": "completed",
        "filename": filename,
        "slide_path": slide_path,
        "true_label": case_row["true_label"],
        "pred_label": result.get("prediction"),
        "raw_prediction": result.get("raw_prediction"),
        "model": job.get("model_display"),
        "decision_status": result.get("decision_status"),
        "safety": result.get("safety", {}),
        "export_path": str(export_path),
        "result_dir": str(result_dir),
        "heatmap_paths": {
            "consensus": str(result_dir / "consensus_heatmap.jpg"),
            "disagreement": str(result_dir / "disagreement_heatmap.jpg"),
            "shared": str(result_dir / "shared_heatmap.jpg"),
            "mel_vs_scc": str(result_dir / "contrast_melanoma_vs_scc_heatmap.jpg"),
            "mel_vs_bcc": str(result_dir / "contrast_melanoma_vs_bcc_heatmap.jpg"),
        },
    }


def write_report(cases, report_path: Path, model_key: str):
    completed = [c for c in cases if c.get("status") == "completed"]
    lines = [
        "# Phase 3 Heatmap Reliability Case Study",
        "",
        f"- Generated: {datetime.now().isoformat()}",
        f"- Model: `{model_key}`",
        f"- Total cases requested: {len(cases)}",
        f"- Completed analyses: {len(completed)}",
        "",
        "## Summary",
        "",
        "| Case | True Label | Prediction | Decision | Risk | Safety Score | OOD Score | Export |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for idx, case in enumerate(cases, 1):
        if case.get("status") != "completed":
            lines.append(f"| {idx} | {case.get('true_label','?')} | ERROR | {case.get('status')} | - | - | - | - |")
            continue
        safety = case.get("safety", {})
        lines.append(
            f"| {idx} | {case['true_label']} | {case['pred_label']} | {case['decision_status']} | "
            f"{safety.get('risk_level','-')} | {safety.get('safety_score','-')} | "
            f"{(safety.get('ood') or {}).get('ood_score','-')} | [export]({case['export_path']}) |"
        )

    lines.extend([
        "",
        "## Case-by-Case Review",
        "",
    ])

    for idx, case in enumerate(cases, 1):
        lines.append(f"### Case {idx}: {case.get('filename', 'Unknown')}")
        if case.get("status") != "completed":
            lines.extend([
                "",
                f"- Status: `{case.get('status')}`",
                f"- Error: {case.get('error', 'Unknown error')}",
                "",
            ])
            continue

        safety = case.get("safety", {})
        ood = safety.get("ood") or {}
        lines.extend([
            "",
            f"- True Label: `{case['true_label']}`",
            f"- Prediction: `{case['pred_label']}`",
            f"- Raw Prediction: `{case['raw_prediction']}`",
            f"- Decision Status: `{case['decision_status']}`",
            f"- Risk Level: `{safety.get('risk_level', '-')}`",
            f"- Safety Score: `{safety.get('safety_score', '-')}`",
            f"- OOD Score: `{ood.get('ood_score', '-')}`",
            f"- Recommendation: {safety.get('recommendation', '-')}",
            f"- Export: [export.json]({case['export_path']})",
            f"- Result Directory: [{case['result_dir']}]({case['result_dir']})",
            "",
            "Heatmaps:",
            f"- [Consensus]({case['heatmap_paths']['consensus']})",
            f"- [Disagreement]({case['heatmap_paths']['disagreement']})",
            f"- [Shared Focus]({case['heatmap_paths']['shared']})",
            f"- [Melanoma vs SCC]({case['heatmap_paths']['mel_vs_scc']})",
            f"- [Melanoma vs BCC]({case['heatmap_paths']['mel_vs_bcc']})",
            "",
        ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    source_rows = read_csv_rows(args.source_csv)
    hard_rows = read_csv_rows(args.hard_case_csv) if args.hard_case_csv.exists() else []
    selected = select_cases(
        source_rows,
        hard_rows,
        per_class=args.per_class,
        extra_melanoma_hard_cases=args.extra_melanoma_hard_cases,
        limit=args.limit,
    )
    print(f"Selected {len(selected)} cases for Phase 3 case study")

    outputs = []
    for idx, row in enumerate(selected, 1):
        print(f"[{idx}/{len(selected)}] {Path(row['slide_path']).name} ({row['true_label']})")
        outputs.append(run_case(row, args.model, idx))

    json_path = REPORT_DIR / "phase3_case_study_cases.json"
    json_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    report_path = REPORT_DIR / "phase3_case_study_report.md"
    write_report(outputs, report_path, args.model)
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
