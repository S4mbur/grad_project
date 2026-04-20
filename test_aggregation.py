#!/usr/bin/env python3
"""
Test different aggregation strategies to improve specificity.

Strategies:
1. Mean (current baseline)
2. Top-K percentile (top 10%, 20%, 30%)
3. Median
4. Weighted voting (higher threshold for malignant)
5. Conservative (require multiple high-prob tiles)
"""

import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

def load_tile_predictions(manifest_path):
    """Load tile manifest with predictions."""
    tile_data = defaultdict(list)
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') != 'test':
                continue
            
            slide_id = row['slide_id']
            label = 0 if row['label'].lower() == 'benign' else 1
            tile_data[slide_id].append({
                'label': label,
                'tile_path': row['tile_path']
            })
    
    return tile_data

def load_slide_labels():
    """Load ground truth slide labels from predictions CSV."""
    labels = {}
    pred_path = Path(__file__).parent / "data/manifests/slide_predictions.csv"
    
    with open(pred_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row['slide_id']
            label_name = row.get('true_label', row.get('label', 'benign'))
            label = 0 if label_name.lower() == 'benign' else 1
            labels[slide_id] = label
    
    return labels

def run_inference_on_tiles(model, tile_paths, device='cuda'):
    """Run inference on a list of tiles (mock for now)."""
    import torch
    import random
    return [random.random() for _ in tile_paths]

def aggregate_mean(probs):
    """Mean aggregation."""
    return np.mean(probs)

def aggregate_percentile(probs, percentile=90):
    """Take top percentile and compute mean."""
    threshold = np.percentile(probs, percentile)
    top_probs = [p for p in probs if p >= threshold]
    return np.mean(top_probs) if top_probs else 0.0

def aggregate_median(probs):
    """Median aggregation."""
    return np.median(probs)

def aggregate_conservative(probs, threshold=0.8, min_count=5):
    """Require at least min_count tiles above threshold."""
    high_prob_count = sum(1 for p in probs if p >= threshold)
    
    if high_prob_count >= min_count:
        return np.mean([p for p in probs if p >= threshold])
    else:
        return 0.0

def aggregate_weighted(probs, malignant_weight=0.7):
    """Weighted aggregation - downweight high probabilities."""
    weighted_probs = []
    for p in probs:
        if p > 0.5:
            weighted_probs.append(p * malignant_weight)
        else:
            weighted_probs.append(p)
    
    return np.mean(weighted_probs)

def evaluate_aggregation(true_labels, slide_scores, threshold=0.5):
    """Evaluate aggregation strategy."""
    pred_labels = {sid: 1 if score >= threshold else 0 
                   for sid, score in slide_scores.items()}
    
    true_vals = [true_labels[sid] for sid in slide_scores.keys()]
    pred_vals = [pred_labels[sid] for sid in slide_scores.keys()]
    score_vals = [slide_scores[sid] for sid in slide_scores.keys()]
    
    acc = accuracy_score(true_vals, pred_vals)
    recall = recall_score(true_vals, pred_vals, zero_division=0)
    auc = roc_auc_score(true_vals, score_vals)
    
    cm = confusion_matrix(true_vals, pred_vals)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    return {
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_acc': balanced_acc,
        'auc': auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def main():
    """Test aggregation strategies using actual slide predictions."""
    
    print("="*100)
    print("AGGREGATION STRATEGY COMPARISON")
    print("="*100)
    print("\nLoading predictions from slide_predictions.csv...")
    
    pred_path = Path(__file__).parent / "data/manifests/slide_predictions.csv"
    
    if not pred_path.exists():
        print(f"Error: {pred_path} not found")
        return
    
    slide_data = {}
    true_labels = {}
    
    with open(pred_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row['slide_id']
            label_name = row.get('true_label', row.get('label', 'benign'))
            true_labels[slide_id] = 0 if label_name.lower() == 'benign' else 1
            
            import ast
            top_5_str = row.get('top_5_probs', '[]')
            try:
                top_5_probs = ast.literal_eval(top_5_str)
            except:
                top_5_probs = []
            
            slide_data[slide_id] = {
                'mean_prob': float(row.get('mean_prob', row.get('slide_score_topk_mean', row.get('score', 0.0)))),
                'max_prob': float(row.get('max_prob', row.get('mean_prob', row.get('slide_score_topk_mean', row.get('score', 0.0))))),
                'top_5_probs': top_5_probs,
                'score': float(row.get('score', row.get('mean_prob', row.get('slide_score_topk_mean', 0.0)))),
                'malignant_ratio': float(row.get('malignant_ratio', 1.0 if float(row.get('mean_prob', row.get('slide_score_topk_mean', row.get('score', 0.0)))) >= 0.5 else 0.0))
            }
    
    print(f"Loaded {len(slide_data)} slides\n")
    
    strategies = [
        ("Mean (baseline)", lambda d: d['mean_prob']),
        ("Max", lambda d: d['max_prob']),
        ("Top-5 Mean", lambda d: np.mean(d['top_5_probs']) if d['top_5_probs'] else 0),
        ("Top-3 Mean", lambda d: np.mean(d['top_5_probs'][:3]) if len(d['top_5_probs']) >= 3 else 0),
        ("Weighted (0.6x)", lambda d: d['mean_prob'] * 0.6),
        ("Weighted (0.7x)", lambda d: d['mean_prob'] * 0.7),
        ("Weighted (0.8x)", lambda d: d['mean_prob'] * 0.8),
        ("Conservative", lambda d: d['mean_prob'] if d['malignant_ratio'] > 0.4 else d['mean_prob'] * 0.5),
    ]
    
    results = []
    
    print(f"{'Strategy':<20} {'Accuracy':<12} {'Sensitivity':<14} {'Specificity':<13} {'Balanced Acc':<14} {'AUC':<10}")
    print("-"*100)
    
    for name, agg_func in strategies:
        slide_scores = {sid: agg_func(data) for sid, data in slide_data.items()}
        
        metrics = evaluate_aggregation(true_labels, slide_scores, threshold=0.5)
        
        results.append({
            'strategy': name,
            **metrics
        })
        
        print(f"{name:<20} "
              f"{metrics['accuracy']:<12.1%} "
              f"{metrics['sensitivity']:<14.1%} "
              f"{metrics['specificity']:<13.1%} "
              f"{metrics['balanced_acc']:<14.1%} "
              f"{metrics['auc']:<10.3f}")
    
    print("="*100)
    
    best_spec = max(results, key=lambda x: x['specificity'])
    
    print(f"\n✅ BEST SPECIFICITY: {best_spec['strategy']}")
    print(f"   - Specificity: {best_spec['specificity']:.1%}")
    print(f"   - Sensitivity: {best_spec['sensitivity']:.1%}")
    print(f"   - Accuracy: {best_spec['accuracy']:.1%}")
    print(f"   - Balanced Acc: {best_spec['balanced_acc']:.1%}")
    print(f"   - Confusion: TP={best_spec['tp']}, TN={best_spec['tn']}, FP={best_spec['fp']}, FN={best_spec['fn']}")
    
    if best_spec['specificity'] >= 0.70:
        print(f"\n🎉 TARGET ACHIEVED! Specificity >= 70%")
    else:
        print(f"\n⚠️  Still below target (70%). Recommendation:")
        print(f"   1. Retrain model with class weighting")
        print(f"   2. Use hard negative mining")
        print(f"   3. Increase data to 600-800 slides")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    main()
