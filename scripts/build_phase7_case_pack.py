from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_RESULTS = REPO_ROOT / "app" / "results"
OUTPUT_ROOT = REPO_ROOT / "results" / "phase7_case_pack"
PHASE4_ROOT = REPO_ROOT / "results" / "phase4_retrieval"
RETRIEVAL_REGISTRY = PHASE4_ROOT / "retrieval_registry.json"
RETRIEVAL_EMBEDDINGS = PHASE4_ROOT / "retrieval_embeddings.npz"


@dataclass
class SelectedCase:
    pack_id: str
    source_job_id: str
    source_dir: str
    source_slide_id: str
    source_label: str
    predicted_label: str
    decision_status: str
    risk_level: str
    melanoma_probability: float
    confidence: float
    uncertainty: float
    ood_score: float
    ood_flag: bool
    hard_case_candidate: bool
    rationale: str
    model_key: str
    retrieval_bank_key: str
    similar_case_count: int
    hard_case_match_count: int
    copied_dir: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def iter_export_dirs(root: Path) -> Iterable[Path]:
    for export_path in root.rglob("export.json"):
        yield export_path.parent


def load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case_dir in iter_export_dirs(APP_RESULTS):
        export_path = case_dir / "export.json"
        try:
            data = read_json(export_path)
        except Exception:
            continue

        source = data.get("source_case", {})
        result = data.get("result", {})
        safety = result.get("safety", {})
        ood = safety.get("ood", {})
        source_label = source.get("true_label") or data.get("true_label") or ""
        predicted_label = (
            result.get("display_prediction")
            or result.get("prediction")
            or result.get("raw_prediction")
            or data.get("pred_label")
            or ""
        )

        records.append(
            {
                "job_id": data.get("job_id") or case_dir.name,
                "case_dir": case_dir,
                "export_path": export_path,
                "slide_id": source.get("slide_id") or data.get("slide_id") or data.get("filename", "").split(".")[0],
                "source_label": source_label,
                "predicted_label": predicted_label,
                "decision_status": safety.get("decision_status") or result.get("decision_status") or "",
                "risk_level": safety.get("risk_level") or "",
                "melanoma_probability": as_float(safety.get("melanoma_probability")),
                "confidence": as_float(safety.get("confidence")),
                "uncertainty": as_float(safety.get("uncertainty")),
                "ood_score": as_float(ood.get("ood_score")),
                "ood_flag": as_bool(ood.get("ood_flag")),
                "hard_case_candidate": as_bool(safety.get("hard_case_candidate") or source.get("hard_case_candidate")),
                "abstain_recommended": as_bool(safety.get("abstain_recommended")),
                "is_melanoma_fn": as_bool(source.get("is_melanoma_fn")),
                "model_key": result.get("model_key") or data.get("model_key") or "",
                "raw_prediction": result.get("raw_prediction") or "",
                "display_prediction": result.get("display_prediction") or result.get("prediction") or "",
                "recommendation": safety.get("recommendation") or "",
            }
        )
    return records


def select_case(records: list[dict[str, Any]], predicate, sort_key, fallback_message: str) -> dict[str, Any]:
    candidates = [r for r in records if predicate(r)]
    if not candidates:
        raise RuntimeError(fallback_message)
    candidates.sort(key=sort_key)
    return candidates[0]


def select_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    melanoma_tp = select_case(
        records,
        lambda r: r["source_label"] == "Melanoma" and r["predicted_label"] == "Melanoma" and r["decision_status"] == "predicted",
        lambda r: (-r["confidence"], r["uncertainty"], r["ood_score"], r["job_id"]),
        "No melanoma true-positive candidate found.",
    )

    melanoma_abstain = select_case(
        records,
        lambda r: r["source_label"] == "Melanoma" and (r["decision_status"] == "abstain" or r["is_melanoma_fn"]),
        lambda r: (
            -int(r["hard_case_candidate"]),
            -int(r["abstain_recommended"]),
            -r["melanoma_probability"],
            -r["confidence"],
            r["uncertainty"],
            r["job_id"],
        ),
        "No melanoma abstain candidate found.",
    )

    normal_abstain = [r for r in records if r["source_label"] == "Normal/Benign" and r["decision_status"] == "abstain"]
    if normal_abstain:
        normal_abstain.sort(key=lambda r: (-int(r["ood_flag"]), -r["ood_score"], -int(r["abstain_recommended"]), -r["confidence"], r["job_id"]))
        normal_case = normal_abstain[0]
        rationale = "Normal/Benign case with abstain behavior and OOD warning."
    else:
        normal_candidates = [r for r in records if r["source_label"] == "Normal/Benign" and r["predicted_label"] == "Normal/Benign"]
        if not normal_candidates:
            raise RuntimeError("No Normal/Benign control candidate found.")
        normal_candidates.sort(key=lambda r: (-r["confidence"], r["uncertainty"], r["ood_score"], r["job_id"]))
        normal_case = normal_candidates[0]
        rationale = "Clean Normal/Benign control case."

    selected = [
        {
            "pack_id": "case_01_melanoma_true_positive",
            "rationale": "Confident melanoma true positive that shows the model can detect an obvious positive case.",
            "record": melanoma_tp,
        },
        {
            "pack_id": "case_02_melanoma_abstain",
            "rationale": "Melanoma hard case that triggers abstention and demonstrates melanoma-first safety behavior.",
            "record": melanoma_abstain,
        },
        {
            "pack_id": "case_03_normal_ood_abstain" if normal_case["decision_status"] == "abstain" else "case_03_normal_control",
            "rationale": rationale,
            "record": normal_case,
        },
    ]
    return selected


def copy_case(record: dict[str, Any], pack_id: str) -> Path:
    src_dir: Path = record["case_dir"]
    dest_dir = OUTPUT_ROOT / pack_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)
    return dest_dir


def load_phase4_assets() -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    registry = read_json(RETRIEVAL_REGISTRY)
    embeddings = np.load(RETRIEVAL_EMBEDDINGS, allow_pickle=True)
    thumbnail_dir = Path(registry.get("thumbnail_dir") or (PHASE4_ROOT / "thumbnails"))
    return registry, {key: embeddings[key] for key in embeddings.files}, thumbnail_dir


def choose_retrieval_bank(model_key: str, registry: dict[str, Any]) -> str:
    banks = registry.get("banks", {})
    if model_key in banks:
        return model_key
    if "ensemble_3_best" in banks:
        return "ensemble_3_best"
    if banks:
        return next(iter(banks))
    raise RuntimeError("No retrieval banks available in registry.")


def build_case_retrieval_payload(
    slide_id: str,
    bank_key: str,
    registry: dict[str, Any],
    embeddings: dict[str, np.ndarray],
    thumbnail_dir: Path,
    *,
    top_k: int = 5,
    hard_k: int = 3,
) -> dict[str, Any]:
    bank = registry["banks"][bank_key]
    case_ids: list[str] = list(bank.get("case_ids") or [])
    if slide_id not in case_ids:
        return {
            "available": False,
            "bank_key": bank_key,
            "reason": "slide_id_not_found_in_retrieval_bank",
            "similar_cases": [],
            "hard_melanoma_matches": [],
        }

    vectors = embeddings[bank_key]
    case_index = case_ids.index(slide_id)
    anchor = vectors[case_index]
    similarities = vectors @ anchor
    order = np.argsort(-similarities)

    all_cases = registry.get("cases", {})
    hard_case_ids = set(registry.get("hard_case_slide_ids") or [])

    def serialize_match(match_slide_id: str, similarity: float) -> dict[str, Any]:
        case_meta = dict(all_cases.get(match_slide_id) or {})
        thumbnail_src = thumbnail_dir / f"{match_slide_id}.jpg"
        case_meta.update(
            {
                "slide_id": match_slide_id,
                "similarity": round(float(similarity), 6),
                "thumbnail_source_path": str(thumbnail_src) if thumbnail_src.exists() else "",
            }
        )
        return case_meta

    similar_cases: list[dict[str, Any]] = []
    for idx in order:
        match_slide_id = case_ids[int(idx)]
        if match_slide_id == slide_id:
            continue
        similar_cases.append(serialize_match(match_slide_id, similarities[int(idx)]))
        if len(similar_cases) >= top_k:
            break

    hard_melanoma_matches: list[dict[str, Any]] = []
    for idx in order:
        match_slide_id = case_ids[int(idx)]
        if match_slide_id == slide_id or match_slide_id not in hard_case_ids:
            continue
        hard_melanoma_matches.append(serialize_match(match_slide_id, similarities[int(idx)]))
        if len(hard_melanoma_matches) >= hard_k:
            break

    return {
        "available": True,
        "bank_key": bank_key,
        "bank_display": bank.get("display") or bank_key,
        "bank_size": int(bank.get("n_cases") or len(case_ids)),
        "hard_case_count": int(bank.get("hard_case_count") or 0),
        "metric": bank.get("metric") or "cosine",
        "similar_cases": similar_cases,
        "hard_melanoma_matches": hard_melanoma_matches,
    }


def copy_retrieval_thumbnails(dest_dir: Path, payload: dict[str, Any]) -> None:
    retrieval_dir = dest_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    for prefix, items in (
        ("similar", payload.get("similar_cases") or []),
        ("hard", payload.get("hard_melanoma_matches") or []),
    ):
        bucket_dir = retrieval_dir / prefix
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for rank, item in enumerate(items, start=1):
            src = Path(item.get("thumbnail_source_path") or "")
            if not src.exists():
                continue
            target_name = f"{rank:02d}_{item.get('slide_id', 'case')}.jpg"
            shutil.copy2(src, bucket_dir / target_name)


def write_retrieval_summary(dest_dir: Path, payload: dict[str, Any]) -> None:
    retrieval_dir = dest_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    sanitized = dict(payload)
    for key in ("similar_cases", "hard_melanoma_matches"):
        cleaned_items = []
        for item in sanitized.get(key) or []:
            cleaned = dict(item)
            cleaned.pop("thumbnail_source_path", None)
            cleaned_items.append(cleaned)
        sanitized[key] = cleaned_items

    (retrieval_dir / "retrieval_summary.json").write_text(json.dumps(sanitized, indent=2), encoding="utf-8")

    lines = [
        "# Retrieval Summary",
        "",
        f"- Available: `{str(bool(payload.get('available'))).lower()}`",
        f"- Retrieval bank: `{payload.get('bank_display') or payload.get('bank_key')}`",
        f"- Bank size: `{payload.get('bank_size', 0)}`",
        f"- Hard melanoma cases in bank: `{payload.get('hard_case_count', 0)}`",
        f"- Metric: `{payload.get('metric') or 'cosine'}`",
        "",
        "## Top Similar Cases",
        "",
    ]
    similar_cases = sanitized.get("similar_cases") or []
    if similar_cases:
        for idx, item in enumerate(similar_cases, start=1):
            lines.extend(
                [
                    f"### Similar {idx}",
                    f"- Slide ID: `{item.get('slide_id', '')}`",
                    f"- Label: `{item.get('true_label', '')}`",
                    f"- Source: `{item.get('source', '')}`",
                    f"- Similarity: `{float(item.get('similarity', 0.0)):.4f}`",
                    "",
                ]
            )
    else:
        lines.extend(["No similar cases available.", ""])

    lines.extend(["## Hard Melanoma Matches", ""])
    hard_matches = sanitized.get("hard_melanoma_matches") or []
    if hard_matches:
        for idx, item in enumerate(hard_matches, start=1):
            lines.extend(
                [
                    f"### Hard Match {idx}",
                    f"- Slide ID: `{item.get('slide_id', '')}`",
                    f"- Label: `{item.get('true_label', '')}`",
                    f"- Source: `{item.get('source', '')}`",
                    f"- Similarity: `{float(item.get('similarity', 0.0)):.4f}`",
                    "",
                ]
            )
    else:
        lines.extend(["No hard melanoma matches available.", ""])

    (retrieval_dir / "retrieval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def enrich_export_with_retrieval(dest_dir: Path, payload: dict[str, Any]) -> None:
    export_path = dest_dir / "export.json"
    export_data = read_json(export_path)
    export_data["retrieval_summary"] = {
        "available": bool(payload.get("available")),
        "bank_key": payload.get("bank_key"),
        "bank_display": payload.get("bank_display"),
        "bank_size": payload.get("bank_size"),
        "similar_case_count": len(payload.get("similar_cases") or []),
        "hard_case_match_count": len(payload.get("hard_melanoma_matches") or []),
    }
    export_data.setdefault("result", {})["retrieval"] = {
        "available": bool(payload.get("available")),
        "bank_key": payload.get("bank_key"),
        "bank_display": payload.get("bank_display"),
        "bank_size": payload.get("bank_size"),
        "hard_case_count": payload.get("hard_case_count"),
        "metric": payload.get("metric"),
        "similar_cases": payload.get("similar_cases") or [],
        "hard_melanoma_matches": payload.get("hard_melanoma_matches") or [],
    }
    export_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")


def build_manifest(selected: list[dict[str, Any]]) -> list[SelectedCase]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    registry, embeddings, thumbnail_dir = load_phase4_assets()
    result: list[SelectedCase] = []
    for item in selected:
        record = item["record"]
        copied_dir = copy_case(record, item["pack_id"])
        bank_key = choose_retrieval_bank(record.get("model_key") or "", registry)
        retrieval_payload = build_case_retrieval_payload(
            str(record["slide_id"]),
            bank_key,
            registry,
            embeddings,
            thumbnail_dir,
        )
        copy_retrieval_thumbnails(copied_dir, retrieval_payload)
        write_retrieval_summary(copied_dir, retrieval_payload)
        enrich_export_with_retrieval(copied_dir, retrieval_payload)
        result.append(
            SelectedCase(
                pack_id=item["pack_id"],
                source_job_id=record["job_id"],
                source_dir=str(record["case_dir"]),
                source_slide_id=str(record["slide_id"]),
                source_label=str(record["source_label"]),
                predicted_label=str(record["predicted_label"]),
                decision_status=str(record["decision_status"]),
                risk_level=str(record["risk_level"]),
                melanoma_probability=record["melanoma_probability"],
                confidence=record["confidence"],
                uncertainty=record["uncertainty"],
                ood_score=record["ood_score"],
                ood_flag=record["ood_flag"],
                hard_case_candidate=record["hard_case_candidate"],
                rationale=item["rationale"],
                model_key=str(record.get("model_key") or ""),
                retrieval_bank_key=bank_key,
                similar_case_count=len(retrieval_payload.get("similar_cases") or []),
                hard_case_match_count=len(retrieval_payload.get("hard_melanoma_matches") or []),
                copied_dir=str(copied_dir),
            )
        )
    return result


def write_outputs(manifest: list[SelectedCase], total_candidates: int) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(APP_RESULTS),
        "output_root": str(OUTPUT_ROOT),
        "total_candidates": total_candidates,
        "selected_count": len(manifest),
        "selected_cases": [asdict(item) for item in manifest],
    }
    (OUTPUT_ROOT / "selected_cases.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 7 Presentation Case Pack",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This pack curates three representative cases from existing app results:",
        "",
    ]
    for idx, item in enumerate(manifest, start=1):
        lines.extend(
            [
                f"## Case {idx}: {item.pack_id}",
                "",
                f"- Source job: `{item.source_job_id}`",
                f"- Source label: `{item.source_label}`",
                f"- Predicted label: `{item.predicted_label}`",
                f"- Decision status: `{item.decision_status}`",
                f"- Risk level: `{item.risk_level}`",
                f"- Melanoma probability: `{item.melanoma_probability:.4f}`",
                f"- Confidence: `{item.confidence:.4f}`",
                f"- Uncertainty: `{item.uncertainty:.4f}`",
                f"- OOD score: `{item.ood_score:.4f}`",
                f"- OOD flag: `{str(item.ood_flag).lower()}`",
                f"- Hard case candidate: `{str(item.hard_case_candidate).lower()}`",
                f"- Model key: `{item.model_key}`",
                f"- Retrieval bank: `{item.retrieval_bank_key}`",
                f"- Retrieved similar cases copied: `{item.similar_case_count}`",
                f"- Hard melanoma retrieval matches copied: `{item.hard_case_match_count}`",
                f"- Rationale: {item.rationale}",
                f"- Assets folder: `{item.copied_dir}`",
                "",
                "Primary assets in each copied folder:",
                "- `export.json`",
                "- `retrieval/retrieval_summary.json`",
                "- `retrieval/retrieval_summary.md`",
                "- copied retrieval thumbnails under `retrieval/similar/` and `retrieval/hard/`",
                "- `thumbnail.jpg`",
                "- `heatmap.jpg`",
                "- `consensus_heatmap.jpg`",
                "- `disagreement_heatmap.jpg`",
                "- `shared_heatmap.jpg`",
                "- contrastive heatmaps when available",
                "- tile thumbnails under `tiles/`",
                "",
            ]
        )

    lines.extend(
        [
            "## How to Use This Pack",
            "",
            "1. Open the copied case folders and inspect `export.json` first.",
            "2. Use the thumbnail and heatmap images for slides or thesis figures.",
            "3. If you need a short narrative, use the three cases as:",
            "   - confident melanoma true positive",
            "   - melanoma hard case with abstain behavior",
            "   - normal/OOD cautionary case",
            "",
        ]
    )
    (OUTPUT_ROOT / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    records = load_records()
    if not records:
        raise SystemExit("No app result cases with export.json were found.")
    selected = select_cases(records)
    manifest = build_manifest(selected)
    write_outputs(manifest, len(records))
    print(f"Created phase 7 case pack at: {OUTPUT_ROOT}")
    for item in manifest:
        print(f"- {item.pack_id} <- {item.source_job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
