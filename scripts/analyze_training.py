#!/usr/bin/env python3
"""Analyze training history across all models."""
import json
from pathlib import Path

results_dir = Path("/home/byalc/phase1_project/results")
for d in sorted(results_dir.glob("mil_4class_*")):
    rfile = d / "results.json"
    if not rfile.exists():
        continue
    r = json.load(open(rfile))
    h = r["history"]
    name = r.get("model", d.name)
    f1s = [x["val_f1"] for x in h]
    losses = [x["val_loss"] for x in h]
    accs = [x["val_acc"] for x in h]
    best = r["best_epoch"]
    
    print(f"{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Total epochs: {len(h)}, Best epoch: {best}")
    print(f"  Best val F1:  {max(f1s):.4f}")
    print(f"  Test acc:     {r['metrics']['accuracy']:.4f}")
    print(f"  Test F1:      {r['metrics']['f1_macro']:.4f}")
    print(f"  Test AUC:     {r['metrics']['auc_roc']:.4f}")
    print(f"  Last 5 val F1:   {[round(x,4) for x in f1s[-5:]]}")
    print(f"  Last 5 val loss: {[round(x,4) for x in losses[-5:]]}")
    print(f"  F1 at best:      {f1s[best-1]:.4f}")
    print(f"  F1 plateau?      {f1s[best-1] - f1s[-1]:.4f} drop from best to last")
    print()
