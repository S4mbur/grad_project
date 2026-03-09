#!/usr/bin/env python3
"""
Priority training wrapper for an ~8 hour budget.
Uses the existing v3 trainer with:
- FN-focused config subset
- shorter training schedule
- requested model order
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_all_models_v3 as base
from backbone_registry import MODEL_CONFIGS as REGISTRY_MODELS


PRIORITY_MODEL_ORDER = [
    "UNI",
    "CONCH",
    "Phikon",
    "ConvNeXt-Base",
    "ConvNeXt-Small",
    "DINOv2-base",
    "ResNet50",
    "ResNet18",
]

# Keep only the most FN-focused configs for this short run.
PRIORITY_EXPERIMENT_TAGS = [
    "cost_sensitive_strong",
    "focal_g3",
]


def ordered_models():
    by_name = {m["name"]: m for m in REGISTRY_MODELS}
    return [by_name[name] for name in PRIORITY_MODEL_ORDER if name in by_name]


def ordered_experiments():
    by_tag = {e["tag"]: e for e in base.EXPERIMENTS}
    return [by_tag[tag] for tag in PRIORITY_EXPERIMENT_TAGS if tag in by_tag]


def main():
    base.MODEL_CONFIGS = ordered_models()
    base.EXPERIMENTS = ordered_experiments()

    # Short budget schedule.
    base.Config.num_epochs = 50
    base.Config.patience = 10
    base.Config.warmup_epochs = 3
    base.Config.version = "v3_fast"

    # Preserve explicit user args if they are passed later.
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "--models", *PRIORITY_MODEL_ORDER,
            "--experiments", *PRIORITY_EXPERIMENT_TAGS,
        ]

    print("Priority training plan")
    print("  Models:", PRIORITY_MODEL_ORDER)
    print("  Experiments:", PRIORITY_EXPERIMENT_TAGS)
    print("  Epochs:", base.Config.num_epochs)
    print("  Patience:", base.Config.patience)
    print("  Warmup:", base.Config.warmup_epochs)

    base.main()


if __name__ == "__main__":
    main()