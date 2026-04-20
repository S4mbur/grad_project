#!/bin/bash
# ============================================================
# Master Download Script - D: drive
# Downloads: COBRA OOD, TCGA-SKCM, All Models
# ============================================================

set -e
BASE="/mnt/d/skin_cancer_project"
LOG_DIR="/home/byalc/download_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  MASTER DOWNLOAD SCRIPT"
echo "  Target: $BASE"
echo "  Date: $(date)"
echo "============================================================"

# ============================================================
# 1. COBRA OOD - Metadata & Labels
# ============================================================
echo ""
echo "[1/4] COBRA OOD Metadata & Labels..."
mkdir -p "$BASE/datasets/cobra_ood/metadata"
mkdir -p "$BASE/datasets/cobra_ood/annotations"

aws s3 cp --no-sign-request \
    s3://cobra-pathology/packages/ood/metadata/ood_images.csv \
    "$BASE/datasets/cobra_ood/metadata/" 2>/dev/null || true

aws s3 cp --no-sign-request \
    s3://cobra-pathology/packages/ood/metadata/ood_patient.csv \
    "$BASE/datasets/cobra_ood/metadata/" 2>/dev/null || true

aws s3 cp --no-sign-request --recursive \
    s3://cobra-pathology/packages/ood/annotations/ \
    "$BASE/datasets/cobra_ood/annotations/" 2>/dev/null || true

echo "  Metadata files:"
ls -la "$BASE/datasets/cobra_ood/metadata/" 2>/dev/null
echo "  ✓ Metadata done"

# ============================================================
# 2. COBRA OOD - WSI Images (LARGE - ~200GB)
# ============================================================
echo ""
echo "[2/4] COBRA OOD WSI Images (1,248 files, ~200GB)..."
echo "  Starting aws s3 sync..."
mkdir -p "$BASE/datasets/cobra_ood/images"

aws s3 sync --no-sign-request \
    s3://cobra-pathology/packages/ood/images/ \
    "$BASE/datasets/cobra_ood/images/" \
    2>&1 | tee "$LOG_DIR/cobra_ood_images.log" &
COBRA_PID=$!
echo "  COBRA download PID: $COBRA_PID"

# ============================================================
# 3. TCGA-SKCM - Diagnostic Slides
# ============================================================
echo ""
echo "[3/4] TCGA-SKCM download starting..."
cd /home/byalc/phase1_project
source ~/phase1_env/bin/activate
python scripts/download_tcga_skcm.py 2>&1 | tee "$LOG_DIR/tcga_skcm.log" &
TCGA_PID=$!
echo "  TCGA download PID: $TCGA_PID"

# ============================================================
# 4. Models
# ============================================================
echo ""
echo "[4/4] Model downloads starting..."
python scripts/download_models.py 2>&1 | tee "$LOG_DIR/models.log" &
MODEL_PID=$!
echo "  Model download PID: $MODEL_PID"

# ============================================================
# Monitor progress
# ============================================================
echo ""
echo "============================================================"
echo "All downloads started in parallel!"
echo "  COBRA OOD PID: $COBRA_PID"
echo "  TCGA-SKCM PID: $TCGA_PID"
echo "  Models PID:     $MODEL_PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/cobra_ood_images.log"
echo "  tail -f $LOG_DIR/tcga_skcm.log"
echo "  tail -f $LOG_DIR/models.log"
echo "============================================================"

# Wait for models first (smallest)
echo ""
echo "Waiting for model downloads..."
wait $MODEL_PID 2>/dev/null
echo "  ✓ Models complete!"

# Show progress periodically
while kill -0 $COBRA_PID 2>/dev/null || kill -0 $TCGA_PID 2>/dev/null; do
    echo ""
    echo "--- Progress check $(date +%H:%M:%S) ---"
    COBRA_COUNT=$(ls "$BASE/datasets/cobra_ood/images/"*.tif 2>/dev/null | wc -l)
    TCGA_COUNT=$(ls "$BASE/datasets/tcga_skcm/"*.svs 2>/dev/null | wc -l)
    echo "  COBRA OOD: $COBRA_COUNT / 1248 images"
    echo "  TCGA-SKCM: $TCGA_COUNT slides"
    sleep 120
done

echo ""
echo "============================================================"
echo "  ALL DOWNLOADS COMPLETE!"
echo "============================================================"
COBRA_FINAL=$(ls "$BASE/datasets/cobra_ood/images/"*.tif 2>/dev/null | wc -l)
TCGA_FINAL=$(ls "$BASE/datasets/tcga_skcm/"*.svs 2>/dev/null | wc -l)
echo "  COBRA OOD: $COBRA_FINAL images"
echo "  TCGA-SKCM: $TCGA_FINAL slides"
echo "  Models: $(find $BASE/models -type f | wc -l) files"
echo "  Total D: usage: $(du -sh $BASE 2>/dev/null | cut -f1)"
