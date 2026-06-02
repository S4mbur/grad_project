#!/usr/bin/env python3
"""Verify Phase 0 manifest integrity and split consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE0_DIR = PROJECT_ROOT / "results" / "phase0_registry"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase10_dataset_integrity"
LABELS = ["Normal/Benign", "BCC", "SCC", "Melanoma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PHASE0_DIR / "master_slide_manifest.csv")
    parser.add_argument("--split-dir", type=Path, default=PHASE0_DIR / "split_manifests")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if blocking issues are found.")
    return parser.parse_args()


def wsl_path(path_value: object) -> Path | None:
    if pd.isna(path_value):
        return None
    text = str(path_value)
    if not text:
        return None
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def add_issue(issues: List[Dict[str, object]], severity: str, check: str, message: str, count: int = 0) -> None:
    issues.append({
        "severity": severity,
        "check": check,
        "message": message,
        "count": int(count),
    })


def canonical_sort_key(row: pd.Series) -> tuple:
    split_rank = {"test": 0, "val": 1, "train": 2, "ood": 3}
    return (
        -int(row.get("valid", 0) or 0),
        -int(row.get("downloaded", 0) or 0),
        int(row.get("corrupt", 0) or 0),
        split_rank.get(str(row.get("split")), 9),
    )


def write_canonical_manifest(df: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    kept_rows = []
    duplicate_resolution_rows = []
    for slide_id, group in df.groupby("slide_id", sort=False):
        if len(group) == 1:
            kept_rows.append(group.iloc[0])
            continue
        ranked = sorted((canonical_sort_key(row), idx, row) for idx, row in group.iterrows())
        keep = ranked[0][2]
        kept_rows.append(keep)
        for _, idx, row in ranked[1:]:
            duplicate_resolution_rows.append({
                "slide_id": slide_id,
                "dropped_index": int(idx),
                "dropped_source": row.get("source"),
                "dropped_split": row.get("split"),
                "dropped_valid": row.get("valid"),
                "dropped_corrupt": row.get("corrupt"),
                "kept_source": keep.get("source"),
                "kept_split": keep.get("split"),
                "kept_valid": keep.get("valid"),
                "kept_corrupt": keep.get("corrupt"),
                "reason": "Prefer valid/downloaded/non-corrupt records, then evaluation splits before train.",
            })

    canonical = pd.DataFrame(kept_rows).reset_index(drop=True)
    canonical.to_csv(output_dir / "canonical_master_slide_manifest.csv", index=False)
    (output_dir / "canonical_master_slide_manifest.json").write_text(
        canonical.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(duplicate_resolution_rows).to_csv(output_dir / "duplicate_resolution.csv", index=False)
    return {
        "canonical_rows": int(len(canonical)),
        "dropped_duplicate_rows": int(len(duplicate_resolution_rows)),
        "canonical_csv": str(output_dir / "canonical_master_slide_manifest.csv"),
        "duplicate_resolution_csv": str(output_dir / "duplicate_resolution.csv"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    issues: List[Dict[str, object]] = []

    required = {
        "slide_id",
        "source",
        "raw_path",
        "downloaded",
        "valid",
        "corrupt",
        "target_class",
        "split",
        "used_for_training",
        "used_for_validation",
        "used_for_test",
        "used_for_ood",
    }
    missing = required - set(df.columns)
    if missing:
        add_issue(issues, "blocking", "required_columns", f"Missing required columns: {sorted(missing)}", len(missing))

    duplicate_count = int(df["slide_id"].duplicated().sum()) if "slide_id" in df else 0
    if duplicate_count:
        add_issue(issues, "blocking", "duplicate_slide_id", "Duplicate slide_id values exist.", duplicate_count)

    if not missing:
        flag_sum = df[["used_for_training", "used_for_validation", "used_for_test", "used_for_ood"]].fillna(0).astype(int).sum(axis=1)
        invalid_flag_rows = int((flag_sum != 1).sum())
        if invalid_flag_rows:
            add_issue(issues, "warning", "split_flags", "Rows should belong to exactly one split flag.", invalid_flag_rows)

        split_map = {
            "train": "used_for_training",
            "val": "used_for_validation",
            "test": "used_for_test",
            "ood": "used_for_ood",
        }
        mismatch_count = 0
        for split_name, flag_col in split_map.items():
            mismatch_count += int(((df["split"] == split_name) != (df[flag_col].fillna(0).astype(int) == 1)).sum())
        if mismatch_count:
            add_issue(issues, "warning", "split_column_flag_mismatch", "Split column and split flags disagree.", mismatch_count)

        valid_downloaded = df[(df["downloaded"].fillna(0).astype(int) == 1) & (df["valid"].fillna(0).astype(int) == 1)]
        missing_files = []
        for _, row in valid_downloaded.iterrows():
            path = wsl_path(row["raw_path"])
            if path is None or not path.exists():
                missing_files.append(row["slide_id"])
        if missing_files:
            add_issue(issues, "blocking", "valid_downloaded_missing_file", "Valid/downloaded slides are missing on disk.", len(missing_files))
            pd.DataFrame({"slide_id": missing_files}).to_csv(args.output_dir / "missing_valid_files.csv", index=False)

        valid_and_corrupt = int(((df["valid"].fillna(0).astype(int) == 1) & (df["corrupt"].fillna(0).astype(int) == 1)).sum())
        if valid_and_corrupt:
            add_issue(issues, "blocking", "valid_and_corrupt", "Rows marked both valid and corrupt.", valid_and_corrupt)

    summary = {
        "manifest": str(args.manifest),
        "n_rows": int(len(df)),
        "n_unique_slides": int(df["slide_id"].nunique()) if "slide_id" in df else None,
        "source_counts": df["source"].value_counts(dropna=False).to_dict() if "source" in df else {},
        "split_counts": df["split"].value_counts(dropna=False).to_dict() if "split" in df else {},
        "class_counts": df["target_class"].value_counts(dropna=False).to_dict() if "target_class" in df else {},
        "valid_count": int(df["valid"].fillna(0).astype(int).sum()) if "valid" in df else None,
        "downloaded_count": int(df["downloaded"].fillna(0).astype(int).sum()) if "downloaded" in df else None,
        "corrupt_count": int(df["corrupt"].fillna(0).astype(int).sum()) if "corrupt" in df else None,
        "issues": issues,
    }

    split_rows = []
    for split_path in sorted(args.split_dir.glob("*.csv")):
        split_df = pd.read_csv(split_path)
        split_rows.append({
            "split_file": split_path.name,
            "n_rows": len(split_df),
            "unique_slide_ids": split_df["slide_id"].nunique() if "slide_id" in split_df else None,
            "source_counts": json.dumps(split_df["source"].value_counts(dropna=False).to_dict()) if "source" in split_df else "{}",
            "class_counts": json.dumps(split_df["target_class"].value_counts(dropna=False).to_dict()) if "target_class" in split_df else "{}",
        })
    split_df = pd.DataFrame(split_rows)

    issue_df = pd.DataFrame(issues)
    canonical_summary = write_canonical_manifest(df, args.output_dir)
    issue_df.to_csv(args.output_dir / "manifest_issues.csv", index=False)
    split_df.to_csv(args.output_dir / "split_summary.csv", index=False)
    (args.output_dir / "manifest_integrity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 10 Dataset Manifest Integrity Report",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Unique slides: {summary['n_unique_slides']}",
        f"- Downloaded rows: {summary['downloaded_count']}",
        f"- Valid rows: {summary['valid_count']}",
        f"- Corrupt rows: {summary['corrupt_count']}",
        "",
        "## Source Counts",
        "",
        "```json",
        json.dumps(summary["source_counts"], indent=2),
        "```",
        "",
        "## Split Counts",
        "",
        "```json",
        json.dumps(summary["split_counts"], indent=2),
        "```",
        "",
        "## Class Counts",
        "",
        "```json",
        json.dumps(summary["class_counts"], indent=2),
        "```",
        "",
        "## Issues",
        "",
    ]
    if issues:
        lines.append("| severity | check | count | message |")
        lines.append("| --- | --- | --- | --- |")
        for issue in issues:
            lines.append(f"| {issue['severity']} | {issue['check']} | {issue['count']} | {issue['message']} |")
    else:
        lines.append("No blocking or warning issues detected by this verifier.")
    lines.extend([
        "",
        "## Canonical Manifest Output",
        "",
        f"- Canonical rows: {canonical_summary['canonical_rows']}",
        f"- Dropped duplicate rows: {canonical_summary['dropped_duplicate_rows']}",
        "- Output CSV: `canonical_master_slide_manifest.csv`",
        "- Duplicate resolution log: `duplicate_resolution.csv`",
        "",
        "The original Phase 0 manifest is not overwritten by this verifier. The canonical output is a deterministic, audit-safe replacement candidate.",
        "",
        "## Notes",
        "",
        "- Corrupt rows can exist in the master manifest as historical audit records; they should not be used for model training/evaluation.",
        "- `valid_downloaded_missing_file` is blocking because it means the registry points to a slide that cannot be read on the current machine.",
        "- This script makes Phase 0 the single registry authority and converts Windows `D:\\...` paths to WSL `/mnt/d/...` for disk checks.",
    ])
    (args.output_dir / "manifest_integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_dir}")
    if args.strict and any(issue["severity"] == "blocking" for issue in issues):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
