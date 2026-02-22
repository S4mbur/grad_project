#!/usr/bin/env python3
"""Compare v1 vs v2 results side by side."""
import json
from pathlib import Path

results_dir = Path("/home/byalc/phase1_project/results")

# Collect all results
models = {}
for d in sorted(results_dir.glob("mil_4class_*")):
    rfile = d / "results.json"
    if not rfile.exists():
        continue
    r = json.load(open(rfile))
    name = r.get("model", d.name)
    version = "v2" if "_v2" in d.name else "v1"
    if name not in models:
        models[name] = {}
    models[name][version] = r

print(f"{'='*85}")
print(f"  MODEL COMPARISON: v1 vs v2")
print(f"{'='*85}")
print(f"  {'Model':<16s} │ {'Version':>4s} │ {'Accuracy':>8s} │ {'F1 macro':>8s} │ {'AUC-ROC':>8s} │ {'Best Ep':>7s} │ {'Total Ep':>8s}")
print(f"  {'─'*85}")

for name in ["ResNet18", "ResNet50", "ConvNeXt-Small", "ConvNeXt-Base", "DINOv2-base", "Phikon"]:
    if name not in models:
        continue
    for ver in ["v1", "v2"]:
        if ver not in models[name]:
            continue
        r = models[name][ver]
        m = r["metrics"]
        h = r.get("history", [])
        total_ep = len(h)
        best_ep = r.get("best_epoch", 0)
        print(f"  {name:<16s} │ {ver:>4s} │ {m['accuracy']:7.2%} │ {m['f1_macro']:7.2%} │ {m['auc_roc']:7.2%} │ {best_ep:7d} │ {total_ep:8d}")
    
    # Delta
    if "v1" in models[name] and "v2" in models[name]:
        d_acc = models[name]["v2"]["metrics"]["accuracy"] - models[name]["v1"]["metrics"]["accuracy"]
        d_f1 = models[name]["v2"]["metrics"]["f1_macro"] - models[name]["v1"]["metrics"]["f1_macro"]
        d_auc = models[name]["v2"]["metrics"]["auc_roc"] - models[name]["v1"]["metrics"]["auc_roc"]
        arrow = lambda x: f"+{x:.2%}" if x > 0 else f"{x:.2%}"
        print(f"  {'  Δ change':<16s} │ {'':>4s} │ {arrow(d_acc):>8s} │ {arrow(d_f1):>8s} │ {arrow(d_auc):>8s} │")
    print(f"  {'─'*85}")

# Best confusion matrices
print(f"\n{'='*60}")
print(f"  BEST MODEL CONFUSION MATRICES")
print(f"{'='*60}")

for ver in ["v1", "v2"]:
    if "Phikon" in models and ver in models["Phikon"]:
        r = models["Phikon"][ver]
        cm = r.get("confusion_matrix", [])
        print(f"\n  Phikon {ver}:")
        names = ["Normal", "BCC", "SCC", "Melanoma"]
        print(f"  {'':12s} {'Normal':>8s} {'BCC':>8s} {'SCC':>8s} {'Melano':>8s}")
        for i, row in enumerate(cm):
            print(f"  {names[i]:12s} " + "".join(f"{v:8d}" for v in row))
