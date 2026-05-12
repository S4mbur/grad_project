#!/usr/bin/env python3
"""
build_attention_covariance_bank.py
==================================
Pre-compute the per-slide attention-weighted covariance spectra needed by
the original ABD / DBRD metrics in ``app/similarity_metrics.py``.

For each case in ``results/phase4_retrieval/retrieval_registry.json`` the
script

  1. loads the cached tile feature tensor (the same .pt file used by
     ``build_phase4_retrieval_bank.py``),
  2. runs the bank's MIL model to obtain attention weights ``a_i`` and
     the encoder hidden representations ``h_i``,
  3. forms the attention-weighted covariance
     ``Sigma_z = sum_i a_i (h_i - mu_z)(h_i - mu_z)^T``,
  4. truncates it to its top-``rank`` eigenvectors via SVD on the
     weighted hidden matrix (no need to materialise the d x d matrix),
  5. saves ``mu``, ``U``, ``s`` for every bank into a single
     ``attention_covariances.npz``.

Storage:
    keys are ``<bank_key>__mu``, ``<bank_key>__U``, ``<bank_key>__s``;
    shapes are ``(N, d)``, ``(N, d, k)``, ``(N, k)`` respectively.

Drop this file at ``scripts/build_attention_covariance_bank.py`` and
run from the repo root with the analyse-server environment active.

For the ensemble bank the script processes each component model on its
own slice and concatenates ``mu``, stacks ``U`` block-diagonally and
concatenates ``s``; this keeps the Bures distance compatible with the
concatenated bank embedding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "app"))

from app import similarity_metrics as smet  # noqa: E402

PHASE4_DIR = PROJECT_DIR / "results" / "phase4_retrieval"
DEFAULT_REGISTRY = PHASE4_DIR / "retrieval_registry.json"
DEFAULT_OUTPUT = PHASE4_DIR / "attention_covariances.npz"
FEATURE_ROOT = Path("/mnt/d/skin_cancer_project/cache")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--rank", type=int, default=32, help="Top-k eigenvectors to keep per slide")
    p.add_argument("--banks", nargs="*", default=None, help="Restrict to these bank keys")
    p.add_argument("--device", default="cuda", help="Torch device for MIL forward pass")
    return p.parse_args()


def feature_dir_for_model(model_key: str, server) -> Path:
    cfg = server.MODEL_REGISTRY[model_key]
    mtype = cfg["type"]
    if mtype == "torchvision":
        loader = cfg.get("loader", "")
        suffix = loader
    elif mtype == "dinov2":
        suffix = "dinov2_base"
    else:
        suffix = mtype
    return FEATURE_ROOT / f"features_4class_{suffix}"


def slide_attention_and_hidden(model, feats, device):
    """Forward pass that exposes the encoder hidden states ``h_i`` and
    attention weights ``a_i``.

    The MIL classes used by ``app/server.py`` return
    ``(logits, attn, bag_embedding, tile_hidden)`` from ``forward(...)``,
    where ``tile_hidden`` is exactly the post-encoder ``h_i`` array we
    need for the covariance.
    """
    import torch

    feats = feats.to(device)
    with torch.no_grad():
        out = model(feats)
    if not isinstance(out, tuple) or len(out) < 4:
        raise RuntimeError(
            "MIL model.forward() did not return (logits, attn, bag, tile_hidden); "
            "this indexing script needs the 4-tuple variant used by the server."
        )
    _, attn, _, tile_hidden = out
    return (
        tile_hidden.detach().cpu().numpy().astype(np.float32),
        attn.detach().cpu().numpy().astype(np.float32).reshape(-1),
    )


def covariance_for_case(
    server, model_key: str, model, case_meta: dict, feature_dir: Path,
    *, rank: int, device: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import torch

    slide_id = case_meta["slide_id"]
    feat_path = feature_dir / f"{slide_id}.pt"
    if not feat_path.exists():
        return None
    feats = torch.load(str(feat_path), map_location=device)
    H, a = slide_attention_and_hidden(model, feats, device)
    if a.size == 0 or H.shape[0] == 0:
        return None
    if a.sum() <= 0:
        a = np.ones_like(a)
    a = a / a.sum()
    mu, U, s = smet.low_rank_covariance(H, a, rank=rank)
    return mu, U, s


def build_for_single_model(server, model_key: str, registry: dict, *, rank: int, device: str):
    bank_meta = registry.get("banks", {}).get(model_key, {})
    case_ids = bank_meta.get("case_ids") or []
    if not case_ids:
        print(f"  [{model_key}] no case ids; skipping")
        return None
    feature_dir = feature_dir_for_model(model_key, server)
    if not feature_dir.exists():
        print(f"  [{model_key}] missing feature dir {feature_dir}; skipping")
        return None
    model = server._get_mil_model(model_key)
    mus, Us, ss = [], [], []
    valid = []
    for idx, slide_id in enumerate(case_ids, 1):
        meta = registry["cases"].get(slide_id, {})
        result = covariance_for_case(
            server, model_key, model, {**meta, "slide_id": slide_id},
            feature_dir, rank=rank, device=device,
        )
        if result is None:
            mus.append(None); Us.append(None); ss.append(None)
            continue
        mu, U, s = result
        # Right-pad U to (d, rank) and s to (rank,) for clean stacking.
        d = U.shape[0]
        U_pad = np.zeros((d, rank), dtype=np.float32)
        U_pad[:, : U.shape[1]] = U
        s_pad = np.zeros(rank, dtype=np.float32)
        s_pad[: s.shape[0]] = s
        mus.append(mu); Us.append(U_pad); ss.append(s_pad)
        valid.append(idx - 1)
        if idx % 25 == 0:
            print(f"  [{model_key}] {idx}/{len(case_ids)} processed")
    # Replace any failed entries with the first successful one (so shapes
    # match) and remember which indices were dropped.
    if not valid:
        return None
    fallback_mu, fallback_U, fallback_s = mus[valid[0]], Us[valid[0]], ss[valid[0]]
    for i in range(len(case_ids)):
        if mus[i] is None:
            mus[i] = fallback_mu
            Us[i] = fallback_U
            ss[i] = fallback_s
    return {
        "mu": np.stack(mus, axis=0).astype(np.float32),
        "U": np.stack(Us, axis=0).astype(np.float32),
        "s": np.stack(ss, axis=0).astype(np.float32),
        "valid_indices": np.asarray(valid, dtype=np.int32),
    }


def build_for_ensemble(server, model_key: str, registry: dict, *, rank: int, device: str):
    bank_meta = registry["banks"].get(model_key, {})
    case_ids = bank_meta.get("case_ids") or []
    component_models = bank_meta.get("component_models") or []
    if not case_ids or not component_models:
        print(f"  [{model_key}] missing case_ids or component_models; skipping")
        return None
    component_results = {}
    for mkey in component_models:
        print(f"  [{model_key}] building component {mkey}")
        component_results[mkey] = build_for_single_model(
            server, mkey, registry, rank=rank, device=device,
        )
        if component_results[mkey] is None:
            print(f"  [{model_key}] component {mkey} failed; aborting")
            return None
    # Each component_results[mkey]["mu"] is (N, d_c) where d_c = 256 for
    # MIL hidden.  We concatenate means and block-diagonal-stack the U
    # matrices so that the resulting covariance is the direct sum of the
    # components -- which matches the concatenated bag embedding.
    mus = np.concatenate([component_results[m]["mu"] for m in component_models], axis=1)
    block_Us = []
    block_ss = []
    for case_idx in range(len(case_ids)):
        rows = []
        s_concat = []
        for mkey in component_models:
            U_c = component_results[mkey]["U"][case_idx]
            s_c = component_results[mkey]["s"][case_idx]
            rows.append(U_c)
            s_concat.append(s_c)
        # Block diagonal: each component's U occupies its own slice.
        d_total = mus.shape[1]
        k_total = sum(r.shape[1] for r in rows)
        U_full = np.zeros((d_total, k_total), dtype=np.float32)
        cursor_row = 0
        cursor_col = 0
        for r in rows:
            d_c, k_c = r.shape
            U_full[cursor_row:cursor_row + d_c, cursor_col:cursor_col + k_c] = r
            cursor_row += d_c
            cursor_col += k_c
        s_full = np.concatenate(s_concat, axis=0)
        block_Us.append(U_full)
        block_ss.append(s_full)
    return {
        "mu": mus,
        "U": np.stack(block_Us, axis=0).astype(np.float32),
        "s": np.stack(block_ss, axis=0).astype(np.float32),
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from app import server  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"Could not import app.server (Torch / model weights required): {exc}"
        )

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    bank_keys = args.banks or sorted(registry.get("banks", {}).keys())
    print(f"Banks to process: {bank_keys}")

    arrays_to_save: Dict[str, np.ndarray] = {}
    for bank_key in bank_keys:
        print(f"\n=== Bank {bank_key} ===")
        bank_meta = registry["banks"].get(bank_key, {})
        if bank_meta.get("type") == "ensemble":
            result = build_for_ensemble(server, bank_key, registry, rank=args.rank, device=args.device)
        else:
            result = build_for_single_model(server, bank_key, registry, rank=args.rank, device=args.device)
        if result is None:
            print(f"  [{bank_key}] failed")
            continue
        arrays_to_save[f"{bank_key}__mu"] = result["mu"]
        arrays_to_save[f"{bank_key}__U"] = result["U"]
        arrays_to_save[f"{bank_key}__s"] = result["s"]
        print(f"  [{bank_key}] saved shapes: mu={result['mu'].shape}, U={result['U'].shape}, s={result['s'].shape}")

    np.savez_compressed(args.output, **arrays_to_save)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())