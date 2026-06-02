#!/usr/bin/env python3
"""Build a compact gallery of melanoma false-negative and high-risk cases.

The goal is not to create another metric table.  The figure gives the thesis
discussion section a visual error-analysis artifact: which real slides were
unsafe for one or more models, which were rescued by the gated policy, and
which non-melanoma slides looked melanoma-like.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.statistical_robustness_report import (  # noqa: E402
    GATED_POLICY,
    LABELS,
    build_method_probabilities,
    load_base_predictions,
    align_frames,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase11_error_gallery"
MELANOMA_INDEX = LABELS.index("Melanoma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--thumb-size", type=int, default=512)
    return parser.parse_args()


def label_from_probs(probs: np.ndarray) -> str:
    return LABELS[int(np.argmax(probs))]


def short_method_name(method: str) -> str:
    return (
        method.replace("_cost_sensitive_strong", "")
        .replace("gated_app_order_cheap_conf70_margin20_mel20", "gated")
        .replace("ensemble_2_best", "ens2")
        .replace("ensemble_3_best", "ens3")
        .replace("_", "-")
    )


def load_thumbnail(slide_path: str, size: int) -> Image.Image:
    path = Path(str(slide_path))
    if path.exists():
        try:
            import openslide

            slide = openslide.OpenSlide(str(path))
            thumb = slide.get_thumbnail((size, size)).convert("RGB")
            slide.close()
            return thumb
        except Exception as exc:
            return placeholder_thumbnail(path.name, f"OpenSlide error: {exc}", size)
    return placeholder_thumbnail(path.name, "Slide not found", size)


def placeholder_thumbnail(title: str, reason: str, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (245, 241, 232))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, size - 8, size - 8], outline=(130, 90, 70), width=3)
    draw.text((24, 28), "Thumbnail unavailable", fill=(60, 45, 35))
    draw.text((24, 64), title[:48], fill=(60, 45, 35))
    draw.text((24, 100), reason[:72], fill=(120, 50, 40))
    return img


def build_case_rows(
    slide_ids: List[str],
    y_true: pd.Series,
    sources: pd.Series,
    methods: Dict[str, np.ndarray],
    metadata: Dict[str, dict],
    max_cases: int,
) -> pd.DataFrame:
    rows = []
    y_arr = np.asarray(y_true)
    method_order = [
        "uni_cost_sensitive_strong",
        "phikon_cost_sensitive_strong",
        "conch_cost_sensitive_strong",
        "ensemble_2_best",
        "ensemble_3_best",
        GATED_POLICY["name"],
    ]
    method_order = [m for m in method_order if m in methods]

    for i, slide_id in enumerate(slide_ids):
        true_label = str(y_arr[i])
        preds = {m: label_from_probs(methods[m][i]) for m in method_order}
        p_mel = {m: float(methods[m][i, MELANOMA_INDEX]) for m in method_order}
        failed = [m for m in method_order if true_label == "Melanoma" and preds[m] != "Melanoma"]
        false_positive_mel = [m for m in method_order if true_label != "Melanoma" and preds[m] == "Melanoma"]
        gated_pred = preds.get(GATED_POLICY["name"], "")

        if failed:
            case_type = "melanoma_fn_or_rescued"
            priority = 100 + len(failed) * 10 - min(p_mel[m] for m in failed)
        elif true_label == "Melanoma":
            case_type = "hard_melanoma_correct"
            priority = 40 - max(p_mel.values())
        elif false_positive_mel:
            case_type = "melanoma_false_positive"
            priority = 20 + max(p_mel[m] for m in false_positive_mel)
        else:
            continue

        wrong_summary = ",".join(short_method_name(m) for m in failed or false_positive_mel) or "none"
        rows.append({
            "priority": float(priority),
            "case_type": case_type,
            "slide_id": slide_id,
            "source": str(sources.iloc[i]),
            "slide_path": metadata.get(slide_id, {}).get("slide_path", ""),
            "true_label": true_label,
            "gated_pred": gated_pred,
            "methods_triggered": wrong_summary,
            "min_p_mel": min(p_mel.values()),
            "max_p_mel": max(p_mel.values()),
            "uni_pred": preds.get("uni_cost_sensitive_strong", ""),
            "phikon_pred": preds.get("phikon_cost_sensitive_strong", ""),
            "conch_pred": preds.get("conch_cost_sensitive_strong", ""),
            "ens2_pred": preds.get("ensemble_2_best", ""),
            "ens3_pred": preds.get("ensemble_3_best", ""),
            "gated_p_mel": p_mel.get(GATED_POLICY["name"], np.nan),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["priority", "slide_id"], ascending=[False, True]).head(max_cases).reset_index(drop=True)


def draw_gallery(cases_df: pd.DataFrame, output_path: Path, thumb_size: int) -> None:
    n = len(cases_df)
    cols = 4
    rows = int(np.ceil(max(n, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, 4.8 * rows), dpi=160)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")

    for idx, row in cases_df.iterrows():
        ax = axes_arr[idx]
        img = load_thumbnail(str(row["slide_path"]), thumb_size)
        ax.imshow(img)
        ax.axis("off")
        title = (
            f"{row['slide_id'][:12]} | {row['source']}\n"
            f"{row['case_type']} | true={row['true_label']} gated={row['gated_pred']}\n"
            f"triggered={row['methods_triggered']} | Pmel {row['min_p_mel']:.3f}-{row['max_p_mel']:.3f}"
        )
        ax.set_title(title, fontsize=8, loc="left")

    fig.suptitle("Melanoma false-negative / high-risk case gallery", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    view = df.head(max_rows).copy()
    keep_cols = [
        "case_type",
        "slide_id",
        "source",
        "true_label",
        "gated_pred",
        "methods_triggered",
        "min_p_mel",
        "max_p_mel",
    ]
    view = view[[c for c in keep_cols if c in view.columns]]
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_base_predictions()
    first_frame = next(iter(frames.values())).copy()
    first_frame["slide_id"] = first_frame["slide_id"].astype(str)
    metadata = first_frame.set_index("slide_id").to_dict(orient="index")

    slide_ids, y_true, sources, probs_by_model = align_frames(frames)
    methods = build_method_probabilities(probs_by_model)
    cases_df = build_case_rows(slide_ids, y_true, sources, methods, metadata, args.max_cases)
    cases_df.to_csv(args.output_dir / "error_case_gallery_index.csv", index=False)

    if not cases_df.empty:
        draw_gallery(cases_df, args.output_dir / "error_case_gallery.png", args.thumb_size)

    lines = [
        "# Phase 11 Error Case Gallery",
        "",
        "This artifact turns the numerical melanoma safety story into inspectable cases.",
        "",
        f"Figure: `{(args.output_dir / 'error_case_gallery.png').as_posix()}`",
        "",
        "Selection priority:",
        "",
        "1. Melanoma slides that at least one model called non-melanoma.",
        "2. Melanoma slides that all models handled correctly but had low melanoma probability.",
        "3. Non-melanoma slides that at least one model called melanoma.",
        "",
        dataframe_to_markdown(cases_df),
        "",
        "Interpretation note: this is a discussion figure, not an independent validation set. It is meant to support failure-mode analysis and defense Q&A.",
    ]
    (args.output_dir / "error_case_gallery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    if not cases_df.empty:
        print(cases_df[["case_type", "slide_id", "source", "true_label", "gated_pred", "methods_triggered", "min_p_mel", "max_p_mel"]].to_string(index=False))


if __name__ == "__main__":
    main()
