#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backbone_registry import MODEL_CONFIGS, feature_dir_name
import train_all_models_v3 as train_v3


def load_state_dict_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_phase1_outputs_for_run(run_dir: Path, force: bool = False):
    results_json = run_dir / 'results.json'
    ckpt = run_dir / 'best_model.pt'
    if not results_json.exists() or not ckpt.exists():
        return {'run_dir': str(run_dir), 'status': 'skipped', 'reason': 'missing results.json or best_model.pt'}

    pred_csv = run_dir / 'phase1_test_predictions.csv'
    hard_csv = run_dir / 'phase1_hard_cases.csv'
    if pred_csv.exists() and hard_csv.exists() and not force:
        return {'run_dir': str(run_dir), 'status': 'skipped', 'reason': 'phase1 outputs already exist'}

    with open(results_json, encoding='utf-8') as f:
        meta = json.load(f)

    model_name = meta['model']
    feat_dim = meta['feat_dim']
    dropout = meta.get('hyperparams', {}).get('dropout', 0.25)
    cfg = train_v3.Config()
    device = torch.device(cfg.device)

    model_cfg = next((m for m in MODEL_CONFIGS if m['name'] == model_name), None)
    if model_cfg is None:
        return {'run_dir': str(run_dir), 'status': 'skipped', 'reason': f'unknown model {model_name}'}

    feature_dir = cfg.base_feature_dir / feature_dir_name(model_cfg)
    if not feature_dir.exists():
        return {'run_dir': str(run_dir), 'status': 'skipped', 'reason': f'missing feature dir {feature_dir}'}
    entries = train_v3.create_unified_labels(cfg)
    slide_list, labels, train_ids, val_ids, test_ids = train_v3.balanced_split(entries, feature_dir, cfg)
    if len(slide_list) == 0:
        return {'run_dir': str(run_dir), 'status': 'skipped', 'reason': 'no slides with features'}
    test_ds = train_v3.SlideDataset([slide_list[i] for i in test_ids], feature_dir)

    model = train_v3.GatedAttentionMIL(
        feat_dim, cfg.mil_hidden, cfg.mil_attention, cfg.num_classes, dropout
    ).to(device)
    model.load_state_dict(load_state_dict_compat(ckpt, device))
    model.eval()

    t_preds, t_labels, t_probs, t_slide_ids = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            feat, lab, sid = test_ds[i]
            feat = feat.to(device)
            logits, _ = model(feat)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            t_preds.append(logits.argmax(1).item())
            t_labels.append(lab)
            t_probs.append(probs)
            t_slide_ids.append(sid)

    rows = []
    hard_rows = []
    for sid, true_lab, pred_lab, probs in zip(t_slide_ids, t_labels, t_preds, t_probs):
        meta_row = test_ds.meta.get(sid, {'slide_id': sid})
        probs_arr = np.asarray(probs, dtype=np.float32)
        order = np.argsort(probs_arr)[::-1]
        top1 = float(probs_arr[order[0]])
        top2 = float(probs_arr[order[1]]) if len(order) > 1 else 0.0
        margin = top1 - top2
        melanoma_prob = float(probs_arr[3])
        is_melanoma = int(true_lab == 3)
        is_melanoma_fn = int(true_lab == 3 and pred_lab != 3)
        hard_case_candidate = int(true_lab == 3 and (pred_lab != 3 or top1 < 0.75 or margin < 0.22))
        row = {
            'slide_id': sid,
            'source': meta_row.get('source', 'unknown'),
            'slide_path': meta_row.get('slide_path', ''),
            'true_label': cfg.class_names[int(true_lab)],
            'pred_label': cfg.class_names[int(pred_lab)],
            'prediction_confidence': round(top1, 6),
            'margin': round(margin, 6),
            'melanoma_probability': round(melanoma_prob, 6),
            'is_melanoma': is_melanoma,
            'is_melanoma_fn': is_melanoma_fn,
            'hard_case_candidate': hard_case_candidate,
            'prob_normal_benign': round(float(probs_arr[0]), 6),
            'prob_bcc': round(float(probs_arr[1]), 6),
            'prob_scc': round(float(probs_arr[2]), 6),
            'prob_melanoma': round(float(probs_arr[3]), 6),
        }
        rows.append(row)
        if hard_case_candidate:
            hard_rows.append(row)

    fieldnames = [
        'slide_id', 'source', 'slide_path', 'true_label', 'pred_label',
        'prediction_confidence', 'margin', 'melanoma_probability', 'is_melanoma',
        'is_melanoma_fn', 'hard_case_candidate', 'prob_normal_benign',
        'prob_bcc', 'prob_scc', 'prob_melanoma'
    ]
    with open(pred_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(hard_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hard_rows)

    meta.setdefault('phase1', {})
    meta['phase1'].update({
        'test_prediction_csv': str(pred_csv),
        'hard_case_csv': str(hard_csv),
        'hard_case_count': len(hard_rows),
        'melanoma_fn_cases': sum(1 for r in hard_rows if r['is_melanoma_fn']),
        'backfilled': True,
    })
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return {'run_dir': str(run_dir), 'status': 'ok', 'rows': len(rows), 'hard_rows': len(hard_rows)}


def main():
    parser = argparse.ArgumentParser(description='Backfill Phase 1 hard-case outputs for completed runs')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--runs', nargs='*', default=None, help='Optional specific result directory names')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / 'results'
    run_dirs = sorted([p for p in root.glob('mil_4class_*_v3*') if p.is_dir()])
    if args.runs:
        run_names = set(args.runs)
        run_dirs = [p for p in run_dirs if p.name in run_names]

    summary = []
    for run_dir in run_dirs:
        out = build_phase1_outputs_for_run(run_dir, force=args.force)
        summary.append(out)
        print(out)

    ok = sum(1 for s in summary if s['status'] == 'ok')
    skipped = sum(1 for s in summary if s['status'] == 'skipped')
    print({'processed': len(summary), 'ok': ok, 'skipped': skipped})


if __name__ == '__main__':
    main()
