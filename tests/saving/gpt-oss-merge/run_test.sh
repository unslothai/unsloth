#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
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
