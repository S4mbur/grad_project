#!/usr/bin/env python3
"""
MIL Model Performance Summary
Generates a clean performance overview of all trained models (v1 & v2).
"""
import json
import numpy as np
from pathlib import Path

results_dir = Path("/home/byalc/phase1_project/results")

# Collect all results
all_models = []
for d in sorted(results_dir.glob("mil_4class_*")):
    rfile = d / "results.json"
    if not rfile.exists():
        continue
    r = json.load(open(rfile))
    m = r["metrics"]
    cm = np.array(r.get("confusion_matrix", []))
    h = r.get("history", [])
    ver = "v2" if "_v2" in d.name else "v1"
    
    # Per-class metrics from confusion matrix
    per_class = {}
    class_names = ["Normal/Benign", "BCC", "SCC", "Melanoma"]
    if cm.size > 0:
        for i, cn in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            per_class[cn] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i].sum())}
    
    all_models.append({
        "name": r.get("model", d.name),
        "version": ver,
        "dir": d.name,
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
        "auc_roc": m["auc_roc"],
        "best_epoch": r.get("best_epoch", 0),
        "total_epochs": len(h),
        "cm": cm.tolist() if cm.size > 0 else [],
        "per_class": per_class,
        "split": r.get("split", {}),
    })

# Sort by F1 macro descending
all_models.sort(key=lambda x: x["f1_macro"], reverse=True)

# ── Print summary ──
print("=" * 80)
print("  4-CLASS SKIN CANCER MIL — ALL MODEL PERFORMANCES")
print("  Classes: Normal/Benign | BCC | SCC | Melanoma")
print("=" * 80)

print(f"\n{'─'*80}")
print(f"  {'#':>2} {'Model':<18s} {'Ver':>3} │ {'Acc':>7} │ {'F1mac':>7} │ {'F1wt':>7} │ {'AUC':>7} │ {'BestEp':>6} │ {'TotEp':>5}")
print(f"{'─'*80}")

for i, m in enumerate(all_models, 1):
    tag = " ◄" if i == 1 else ""
    print(f"  {i:2d} {m['name']:<18s} {m['version']:>3} │ {m['accuracy']:6.2%} │ {m['f1_macro']:6.2%} │ "
          f"{m['f1_weighted']:6.2%} │ {m['auc_roc']:6.2%} │ {m['best_epoch']:6d} │ {m['total_epochs']:5d}{tag}")

print(f"{'─'*80}")

# Best model per-class breakdown
best = all_models[0]
print(f"\n{'='*60}")
print(f"  BEST MODEL: {best['name']} ({best['version']})")
print(f"{'='*60}")
print(f"  Accuracy: {best['accuracy']:.4f}  |  F1-macro: {best['f1_macro']:.4f}  |  AUC: {best['auc_roc']:.4f}")
print(f"\n  Per-class breakdown:")
print(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print(f"  {'─'*58}")
for cn in ["Normal/Benign", "BCC", "SCC", "Melanoma"]:
    if cn in best["per_class"]:
        c = best["per_class"][cn]
        print(f"  {cn:<16} {c['precision']:9.2%} {c['recall']:9.2%} {c['f1']:9.2%} {c['support']:10d}")

# Confusion Matrix
if best["cm"]:
    cm = np.array(best["cm"])
    print(f"\n  Confusion Matrix:")
    labels = ["Normal", "BCC", "SCC", "Melanoma"]
    header = "True \\ Pred"
    print(f"  {header:<12}" + "".join(f"{l:>9}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i]:<12}" + "".join(f"{v:9d}" for v in row))

# Best model per version
print(f"\n{'='*60}")
print(f"  BEST PER VERSION")
print(f"{'='*60}")
for ver in ["v1", "v2"]:
    ver_models = [m for m in all_models if m["version"] == ver]
    if ver_models:
        b = ver_models[0]
        print(f"  {ver}: {b['name']:<18s} F1={b['f1_macro']:.4f}  Acc={b['accuracy']:.4f}  AUC={b['auc_roc']:.4f}")

# Data split info
if best.get("split"):
    s = best["split"]
    print(f"\n  Data split: Train={s.get('train',0)}, Val={s.get('val',0)}, Test={s.get('test',0)}")

print(f"\n{'='*80}")
print(f"  Total models trained: {len(all_models)}")
print(f"  Results dir: {results_dir}")
print(f"{'='*80}")
