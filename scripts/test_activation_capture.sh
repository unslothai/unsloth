#!/usr/bin/env bash
# =============================================================================
# Quick test: run unsloth-cli with activation capture on a tiny model/dataset.
#
# Usage (in WSL, with venv active):
#   source ~/unsloth-dev/bin/activate
#   bash /mnt/m/Unsloth_Work/unsloth/scripts/test_activation_capture.sh
#
# Or all-in-one from WSL:
#   source ~/unsloth-dev/bin/activate && \
#       bash /mnt/m/Unsloth_Work/unsloth/scripts/test_activation_capture.sh
# =============================================================================
set -euo pipefail

REPO="/mnt/m/Unsloth_Work/unsloth"
OUT="$REPO/test_activation_output"
LOG="$OUT/activation_logs"

echo "==> Running unsloth-cli with TinyLlama + alpaca-cleaned (20 steps)"
python "$REPO/unsloth-cli.py" \
    --model_name        "unsloth/tinyllama"          \
    --max_seq_length    512                           \
    --load_in_4bit                                    \
    --dataset           "yahma/alpaca-cleaned"        \
    --r                 8                             \
    --lora_alpha        8                             \
    --max_steps         20                            \
    --per_device_train_batch_size 1                   \
    --gradient_accumulation_steps 1                   \
    --warmup_steps      2                             \
    --learning_rate     2e-4                          \
    --optim             "adamw_8bit"                  \
    --logging_steps     1                             \
    --output_dir        "$OUT/train_output"           \
    --report_to         "none"                        \
    --capture_activations                             \
    --capture_output_dir "$LOG"                       \
    --capture_interval  5                             \
    --capture_max_channels 32

echo ""
echo "==> Generating visualization"
python "$REPO/visualize_activations.py" "$LOG" \
    --output "$LOG/viz.html" \
    --open

echo ""
echo "✅  Test complete."
echo "    Activation log : $LOG/activation_log.jsonl"
echo "    Visualization  : $LOG/viz.html"
