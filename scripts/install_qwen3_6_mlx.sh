#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -euo pipefail

# Qwen3.6 MLX one-command setup. The uv installer payload is pinned by SHA-256;
# rotate with: curl -sSLf https://astral.sh/uv/install.sh | shasum -a 256

RULE=""
_rule_i=0
while [ "$_rule_i" -lt 52 ]; do
    RULE="${RULE}─"
    _rule_i=$((_rule_i + 1))
done

if [ -n "${NO_COLOR:-}" ]; then
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
elif [ -t 1 ] || [ -n "${FORCE_COLOR:-}" ]; then
    _ESC="$(printf '\033')"
    C_TITLE="${_ESC}[38;5;117m"
    C_DIM="${_ESC}[38;5;245m"
    C_OK="${_ESC}[38;5;108m"
    C_WARN="${_ESC}[38;5;136m"
    C_ERR="${_ESC}[91m"
    C_RST="${_ESC}[0m"
else
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
fi

step()    { printf "  ${C_DIM}%-18.18s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  ${C_DIM}%-18s${2:-$C_DIM}%s${C_RST}\n" "" "$1"; }
fail()    { step "error" "$1" "$C_ERR"; exit 1; }

VENV_DIR=""
_next_is_venv=false

for arg in "$@"; do
    if [ "$_next_is_venv" = true ]; then
        VENV_DIR="$arg"
        _next_is_venv=false
        continue
    fi
    case "$arg" in
        --venv-dir)  _next_is_venv=true ;;
    esac
done

if [ -z "$VENV_DIR" ]; then
    VENV_DIR="$HOME/.unsloth/unsloth_qwen3_6_mlx"
fi

echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "Qwen3.6 MLX Installer"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""

if [ "$(uname)" != "Darwin" ]; then
    fail "MLX requires macOS with Apple Silicon. Detected: $(uname)"
fi

_ARCH=$(uname -m)
if [ "$_ARCH" != "arm64" ]; then
    step "warning" "Apple Silicon recommended (detected: $_ARCH)" "$C_WARN"
fi

step "platform" "macOS ($_ARCH)"

PYTHON=""
for _candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$_candidate" >/dev/null 2>&1; then
        PYTHON="$_candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python 3 not found. Install via: brew install python@3.12"
fi

_PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
step "python" "$PYTHON ($_PY_VERSION)"

if [ -x "$VENV_DIR/bin/python" ]; then
    step "venv" "using existing environment"
    substep "$VENV_DIR"
else
    step "venv" "creating virtual environment"
    substep "$VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Fetch to a temp file, verify the digest, and only then execute.
_UV_INSTALLER_SHA256="48cd5aca5d5671a3b3d5f61538cc8622e4434af63319115159990d8b0dd02416"

if ! command -v uv >/dev/null 2>&1; then
    step "uv" "installing uv package manager..."
    _uv_tmp=$(mktemp)
    curl -LsSf "https://astral.sh/uv/install.sh" -o "$_uv_tmp"
    _uv_actual=$(shasum -a 256 "$_uv_tmp" | awk '{print $1}')
    if [ "$_uv_actual" != "$_UV_INSTALLER_SHA256" ]; then
        rm -f "$_uv_tmp"
        fail "uv installer SHA-256 mismatch: got $_uv_actual expected $_UV_INSTALLER_SHA256 (refusing to execute)"
    fi
    sh "$_uv_tmp" </dev/null
    rm -f "$_uv_tmp"
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    substep "done"
else
    step "uv" "found $(uv --version 2>/dev/null || echo 'uv')"
fi

_VENV_PY="$VENV_DIR/bin/python"

step "install" "installing current mlx-vlm..."
# Reinstall even if satisfied: older copies of this installer modified mlx-vlm in place.
uv pip install --python "$_VENV_PY" -q \
    --upgrade-package mlx-vlm --reinstall-package mlx-vlm mlx-vlm
# Remove the legacy module the old installer added, only if the distribution does not own it.
"$_VENV_PY" - <<'PY'
from importlib.metadata import distribution

dist = distribution("mlx-vlm")
legacy_module = dist.locate_file("mlx_vlm/generate.py")
owned_files = {str(path) for path in (dist.files or ())}
if legacy_module.is_file() and "mlx_vlm/generate.py" not in owned_files:
    legacy_module.unlink()
PY
substep "done"

step "install" "installing transformers>=5.2.0..."
if uv pip install --python "$_VENV_PY" -q "transformers>=5.2.0"; then
    substep "installed from PyPI"
else
    substep "PyPI install failed, trying GitHub..."
    if uv pip install --python "$_VENV_PY" -q "git+https://github.com/huggingface/transformers.git"; then
        substep "installed from huggingface/transformers main"
    else
        fail "Could not install transformers>=5.2.0 (required for Qwen3.5/3.6 model support). Please check your Python version (>=3.10 required) and network connection, then try again."
    fi
fi

step "install" "installing torch + torchvision (needed for Qwen3 VL processor)..."
uv pip install --python "$_VENV_PY" -q torch torchvision
substep "done"

if "$_VENV_PY" -c "import mlx_vlm; import torch; import torchvision; import transformers"; then
    substep "mlx-vlm + torch + transformers verified"
else
    fail "Installation verification failed. Please ensure Python >=3.10 and try again."
fi

if "$_VENV_PY" - <<'PY'
from mlx_vlm.generate import stream_generate
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.qwen3_5.config import ModelConfig
from mlx_vlm.models.qwen3_5.qwen3_5 import Model, ModelConfig as RuntimeModelConfig, sanitize_key

required_fields = {"position_ids", "rope_deltas"}
available_fields = set(getattr(InputEmbeddingsFeatures, "__dataclass_fields__", ()))
if ModelConfig is not RuntimeModelConfig:
    raise RuntimeError("Qwen3.5 model and config modules are inconsistent")
if not callable(Model) or not callable(sanitize_key) or not callable(stream_generate):
    raise RuntimeError("Qwen3.5 model, sanitizer, or generation entry point is unavailable")
if not required_fields.issubset(available_fields):
    raise RuntimeError("mlx-vlm lacks the Qwen3.5 request-owned position interface")
PY
then
    substep "Qwen3.5 model + generation runtime verified"
else
    fail "Installed mlx-vlm does not provide a coherent Qwen3.5/3.6 runtime. Please retry with a current mlx-vlm release."
fi

echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "Qwen3.6 MLX installed!"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""
step "available models" "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit"
substep "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
substep "unsloth/Qwen3.6-35B-A3B-MLX-8bit"
echo ""
step "venv activate" "source ${VENV_DIR}/bin/activate"
echo ""
step "vision chat" "python -m mlx_vlm.chat --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
substep "Use /image path/to/image.jpg to load an image"
echo ""
step "gradio UI" "python -m mlx_vlm.chat_ui --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
echo ""
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""
