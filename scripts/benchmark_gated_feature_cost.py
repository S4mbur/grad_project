#!/usr/bin/env python3
"""Benchmark gated feature-extraction cost on mounted WSI files.

The app's production tile budget stays at 200. This helper is for offline
profiling only: it extracts tiles in memory, runs the gated UNI -> Phikon ->
CONCH sequence, and optionally measures the full three-model baseline on the
same slide/tile budget.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import server  # noqa: E402


DATA_ROOT = Path(
    os.environ.get("SKINSIGHT_DATA_ROOT", "/mnt/d/skin_cancer_project/datasets")
).expanduser()
DEFAULT_SLIDE_ROOTS = [
    DATA_ROOT / "tcga_skcm",
    DATA_ROOT / "cobra_ood" / "images",
    DATA_ROOT / "cobra_bcc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real WSI gated feature-cost benchmark.")
    parser.add_argument("--slides", nargs="*", type=Path, default=None, help="Explicit WSI paths.")
    parser.add_argument("--max-slides", type=int, default=3, help="Maximum slides to auto-discover.")
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 160, 200], help="Tile budgets to test.")
    parser.add_argument(
        "--measure-full-baseline",
        action="store_true",
        help="Also run skipped encoders so wall-time can be compared with the full three-model baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase9_feature_cost_profile" / "real_wsi_gated_benchmark.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def discover_slides(max_slides: int) -> List[Path]:
    slides: List[Path] = []
    for root in DEFAULT_SLIDE_ROOTS:
        if not root.exists():
            continue
        for pattern in ("*.svs", "*.tif", "*.tiff"):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    slides.append(path)
                if len(slides) >= max_slides:
                    return slides
    return slides[:max_slides]


def extract_tiles_in_memory(slide, tile_budget: int):
    w, h = slide.dimensions
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5))
    target_ds = mpp / 0.5 if mpp > 0 else 1.0
    level = slide.get_best_level_for_downsample(max(target_ds, 1.0))
    level_ds = slide.level_downsamples[level]
    read_size = int(server.cfg.TILE_SIZE * level_ds)

    thumb = slide.get_thumbnail((512, 512))
    thumb_arr = np.array(thumb.convert("RGB"))
    gray = np.mean(thumb_arr, axis=2)
    tissue_mask = (gray < 220) & (gray > 30)

    scale_x = w / thumb_arr.shape[1]
    scale_y = h / thumb_arr.shape[0]
    positions = []
    step = max(1, int(thumb_arr.shape[0] / 50))
    for ty in range(0, thumb_arr.shape[0], step):
        for tx in range(0, thumb_arr.shape[1], step):
            if tissue_mask[ty, tx]:
                x = int(tx * scale_x)
                y = int(ty * scale_y)
                if x + read_size <= w and y + read_size <= h:
                    positions.append((x, y))

    random.seed(42)
    random.shuffle(positions)
    positions = positions[: tile_budget * 3]

    tiles = []
    for x, y in positions:
        if len(tiles) >= tile_budget:
            break
        region = slide.read_region((x, y), level, (server.cfg.TILE_SIZE, server.cfg.TILE_SIZE))
        tile = region.convert("RGB")
        arr = np.array(tile)
        gray_t = np.mean(arr, axis=2)
        tissue_frac = np.mean((gray_t < 220) & (gray_t > 30))
        if tissue_frac >= server.cfg.MIN_TISSUE_FRACTION:
            tiles.append(tile)
    return tiles


def run_model(tiles, model_key: str) -> Dict[str, object]:
    feature_started = time.perf_counter()
    features = server._extract_features(tiles, model_key)
    feature_seconds = time.perf_counter() - feature_started
    mil_started = time.perf_counter()
    pred, probs, attn, bag_embedding, raw_probs, contrastive_views = server._run_mil_inference(features, model_key)
    mil_seconds = time.perf_counter() - mil_started
    return {
        "model_key": model_key,
        "prediction": server.CLASS_NAMES[pred],
        "probabilities": probs,
        "bag_embedding": bag_embedding,
        "feature_seconds": feature_seconds,
        "mil_seconds": mil_seconds,
        "total_seconds": feature_seconds + mil_seconds,
    }


def benchmark_slide(slide_path: Path, budget: int, measure_full_baseline: bool) -> Dict[str, object]:
    openslide = server._ensure_openslide()
    preset = server.ENSEMBLE_PRESETS[server.DEFAULT_MODEL_KEY]
    model_order = list(preset["models"])
    policy = dict(preset["gating_policy"])

    slide = openslide.OpenSlide(str(slide_path))
    tile_started = time.perf_counter()
    tiles = extract_tiles_in_memory(slide, budget)
    tile_seconds = time.perf_counter() - tile_started
    slide.close()

    results: Dict[str, Dict[str, object]] = {}
    invoked = []
    gating_decisions = []
    gated_started = time.perf_counter()
    for i, model_key in enumerate(model_order):
        results[model_key] = run_model(tiles, model_key)
        invoked.append(model_key)
        avg_probs = np.mean([results[m]["probabilities"] for m in invoked], axis=0)
        escalate, reasons, stats = server._gated_escalation_decision(
            avg_probs,
            policy,
            is_last_model=(i == len(model_order) - 1),
        )
        gating_decisions.append({
            "step": len(invoked),
            "prediction": stats["prediction"],
            "confidence": stats["confidence"],
            "margin": stats["margin"],
            "melanoma_probability": stats["melanoma_probability"],
            "escalated": escalate,
            "reasons": "; ".join(reasons),
        })
        if not escalate:
            break
    gated_model_seconds = time.perf_counter() - gated_started

    if measure_full_baseline:
        for model_key in model_order:
            if model_key not in results:
                results[model_key] = run_model(tiles, model_key)

    gated_probs = np.mean([results[m]["probabilities"] for m in invoked], axis=0)
    full_probs = np.mean([results[m]["probabilities"] for m in model_order if m in results], axis=0)
    gated_pred = server.CLASS_NAMES[int(np.argmax(gated_probs))]
    full_pred = server.CLASS_NAMES[int(np.argmax(full_probs))]
    full_model_seconds = sum(float(results[m]["total_seconds"]) for m in model_order if m in results)

    actual_calls = len(tiles) * len(invoked)
    same_slide_full_calls = len(tiles) * len(model_order)
    fixed_200_calls = 200 * 3

    return {
        "slide_path": str(slide_path),
        "slide_id": slide_path.stem,
        "tile_budget": budget,
        "tiles_used": len(tiles),
        "tile_extraction_seconds": round(tile_seconds, 4),
        "gated_model_seconds": round(gated_model_seconds, 4),
        "gated_wall_seconds_including_tiles": round(tile_seconds + gated_model_seconds, 4),
        "full_model_seconds_measured_or_partial": round(full_model_seconds, 4),
        "full_wall_seconds_measured_or_partial": round(tile_seconds + full_model_seconds, 4),
        "full_baseline_measured": bool(measure_full_baseline),
        "models_run": "+".join(invoked),
        "num_models_run": len(invoked),
        "actual_tile_encoder_calls": actual_calls,
        "same_slide_full_candidate_tile_encoder_calls": same_slide_full_calls,
        "fixed_3model_200tile_baseline_calls": fixed_200_calls,
        "cost_ratio_vs_same_slide_full": round(actual_calls / max(float(same_slide_full_calls), 1.0), 4),
        "cost_ratio_vs_3model_200tile": round(actual_calls / max(float(fixed_200_calls), 1.0), 4),
        "gated_prediction": gated_pred,
        "full_prediction_measured_or_partial": full_pred,
        "gated_melanoma_probability": round(float(gated_probs[server.CLASS_KEYS.index("melanoma")]), 4),
        "gating_decisions": " | ".join(
            f"{d['step']}:{d['prediction']} conf={d['confidence']:.3f} margin={d['margin']:.3f} mel={d['melanoma_probability']:.3f} escalate={d['escalated']} ({d['reasons']})"
            for d in gating_decisions
        ),
        "per_model_seconds": "; ".join(
            f"{m}={results[m]['total_seconds']:.4f}s" for m in model_order if m in results
        ),
    }


def write_summary(output_csv: Path, rows: Sequence[Dict[str, object]]) -> None:
    summary_path = output_csv.with_suffix(".md")
    if not rows:
        summary_path.write_text("# Real WSI Gated Benchmark\n\nNo rows were produced.\n")
        return

    cost_ratios = [float(r["cost_ratio_vs_same_slide_full"]) for r in rows]
    gated_wall = [float(r["gated_wall_seconds_including_tiles"]) for r in rows]
    lines = [
        "# Real WSI Gated Feature-Cost Benchmark",
        "",
        f"- Rows: {len(rows)}",
        f"- Mean cost ratio vs same-slide full 3-model ensemble: {np.mean(cost_ratios):.4f}",
        f"- Mean gated wall time including tiles: {np.mean(gated_wall):.4f} seconds",
        "",
        "This is a wall-time smoke benchmark on the current machine. It is separate from the Phase 9 proxy quality profile.",
        "",
        "## Rows",
        "",
        "| slide_id | tile_budget | tiles_used | models_run | cost_ratio | gated_wall_s | prediction |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['slide_id']} | {row['tile_budget']} | {row['tiles_used']} | {row['models_run']} | "
            f"{row['cost_ratio_vs_same_slide_full']} | {row['gated_wall_seconds_including_tiles']} | {row['gated_prediction']} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    slides = args.slides if args.slides else discover_slides(args.max_slides)
    slides = [Path(p) for p in slides if Path(p).exists()]
    if not slides:
        raise FileNotFoundError(
            "No WSI slides found. Pass --slides or set SKINSIGHT_DATA_ROOT."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for slide_path in slides[: args.max_slides]:
        for budget in args.budgets:
            print(f"Benchmarking {slide_path.name} at {budget} tiles...")
            rows.append(benchmark_slide(slide_path, int(budget), args.measure_full_baseline))

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(args.output, rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
