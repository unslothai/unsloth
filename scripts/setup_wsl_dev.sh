#!/usr/bin/env bash
# =============================================================================
# Unsloth WSL dev environment setup
# Creates a Python venv at ~/unsloth-dev and installs the local repo.
# CUDA 12.8 wheels are used (compatible with CUDA 13.0 driver via BC).
#
# Usage:
#   bash /mnt/m/Unsloth_Work/unsloth/scripts/setup_wsl_dev.sh
# =============================================================================
set -euo pipefail

REPO_PATH="/mnt/m/Unsloth_Work/unsloth"
VENV_PATH="$HOME/unsloth-dev"
PYTHON="python3.12"

echo "==> Creating venv at $VENV_PATH"
$PYTHON -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "==> Upgrading pip / wheel"
pip install --upgrade pip wheel setuptools packaging

echo "==> Installing PyTorch 2.7 (CUDA 12.8 wheels)"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing triton (Linux)"
pip install triton

echo "==> Installing bitsandbytes"
pip install "bitsandbytes>=0.45.5"

echo "==> Installing unsloth_zoo"
pip install "unsloth_zoo>=2026.3.2"

echo "==> Installing unsloth core deps (no torch, no zoo - already installed)"
pip install \
    "peft>=0.18.0" \
    "trl>=0.18.2" \
    "transformers>=4.51.3" \
    "accelerate>=0.34.1" \
    "datasets>=3.4.1" \
    "huggingface_hub>=0.34.0" \
    "hf_transfer" \
    "sentencepiece>=0.2.0" \
    "protobuf" \
    "psutil" \
    "tqdm" \
    "numpy" \
    "tyro" \
    "xformers" \
    "sentence-transformers"

echo "==> Installing local unsloth repo in editable mode"
pip install -e "$REPO_PATH" --no-build-isolation --no-deps

echo ""
echo "✅  Done!  Activate with:"
echo "        source $VENV_PATH/bin/activate"
echo ""
echo "    Then run the test:"
echo "        python $REPO_PATH/scripts/test_activation_capture.sh"
