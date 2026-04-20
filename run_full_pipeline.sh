#!/bin/bash

# Full training pipeline script for 400-slide COBRA dataset
# This script:
# 1. Trains the model
# 2. Waits for training completion
# 3. Runs inference
# 4. Runs evaluation
# 5. Compares with baseline

set -e

cd ~/phase1_project
source ~/phase1_env/bin/activate

echo "========================================================"
echo "Starting Full Pipeline: Training -> Inference -> Eval"
echo "========================================================"

# Step 1: Train model (if not already running)
echo ""
echo "[1/4] Model Training (400 slides, 20 epochs)"
echo "========================================================"
python3 scripts/03_train.py --epochs 20 --batch-size 32

# Step 2: Wait a bit for checkpoint to be written
echo ""
echo "Waiting for training checkpoint to be written..."
sleep 2

# Step 3: Run inference
echo ""
echo "[2/4] Slide-Level Inference"
echo "========================================================"
python3 scripts/04_inference.py --split test

# Step 4: Run evaluation
echo ""
echo "[3/4] Model Evaluation"
echo "========================================================"
python3 scripts/05_evaluate.py

# Step 5: Comparison summary
echo ""
echo "[4/4] Results Summary"
echo "========================================================"
echo ""
echo "Baseline Results (200 slides):"
echo "  - Accuracy:   72%"
echo "  - AUC-ROC:    90.32%"
echo "  - Sensitivity: 96%"
echo "  - Specificity: 48%"
echo ""
echo "Check reports/ and data/manifests/slide_predictions.csv"
echo "for detailed results with 400 slides."
echo ""
echo "========================================================"
echo "Pipeline Complete!"
echo "========================================================"
