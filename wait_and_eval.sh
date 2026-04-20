#!/bin/bash

# Wait for training to complete and then run inference + evaluation
# Checks training.log for completion marker

cd ~/phase1_project
source ~/phase1_env/bin/activate

echo "========================================================"
echo "Waiting for training to complete..."
echo "========================================================"

# Wait for training log to show completion
TRAINING_LOG="training.log"
WAIT_INTERVAL=5
MAX_WAIT=1800  # 30 minutes max

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if [ -f "$TRAINING_LOG" ]; then
        # Check if training is complete (look for "Epoch 20" completion or error)
        if tail -20 "$TRAINING_LOG" | grep -q "Best val"; then
            echo "[$(date)] Training completed!"
            break
        fi
    fi
    
    # Show progress
    if [ -f "$TRAINING_LOG" ]; then
        tail -1 "$TRAINING_LOG" | grep -E "Epoch|Best"
    fi
    
    sleep $WAIT_INTERVAL
    elapsed=$((elapsed + WAIT_INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo "Timeout waiting for training to complete"
    exit 1
fi

echo ""
echo "========================================================"
echo "Training Complete. Starting Inference and Evaluation..."
echo "========================================================"
echo ""

# Run inference
echo "[$(date)] Running Slide-Level Inference..."
python3 scripts/04_inference.py --split test 2>&1 | tee inference.log

# Run evaluation
echo "[$(date)] Running Evaluation..."
python3 scripts/05_evaluate.py 2>&1 | tee evaluation.log

echo ""
echo "========================================================"
echo "Pipeline Complete!"
echo "Results saved to:"
echo "  - data/manifests/slide_predictions.csv"
echo "  - reports/"
echo "========================================================"
