#!/bin/bash
set -e

echo "================================================================"
echo "🚀 STEP 1: Running the training and merging script..."
echo "================================================================"
python train_and_merge.py

echo ""
echo "================================================================"
echo "✅ STEP 2: Training complete. Running the inference script..."
echo "================================================================"
python test_merged_model.py

echo ""
echo "================================================================"
echo "🎉 All steps completed successfully!"
echo "================================================================"
