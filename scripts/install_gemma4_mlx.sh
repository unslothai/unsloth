#!/bin/bash
set -euo pipefail

# ============================================================
# Gemma 4 MLX — One-command setup + inference
#
# Supply-chain hardening: the uv installer payload is pinned by
# SHA-256. Rotate by running:
#   curl -sSLf https://astral.sh/uv/install.sh | shasum -a 256
# and updating _UV_INSTALLER_SHA256 below.
# ============================================================
#
# Usage:
#   bash install_gemma4_mlx.sh [--venv-dir DIR]
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs uv, mlx-vlm, transformers
# ============================================================

# ── Output style (inspired by unsloth/install.sh) ─────────────
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

# ── Parse flags ───────────────────────────────────────────────
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

# Default venv location
if [ -z "$VENV_DIR" ]; then
    VENV_DIR="$HOME/.unsloth/unsloth_gemma4_mlx"
fi

# ── Banner ────────────────────────────────────────────────────
echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "💎 Gemma 4 MLX Installer"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""

# ── Platform check ────────────────────────────────────────────
if [ "$(uname)" != "Darwin" ]; then
    fail "MLX requires macOS with Apple Silicon. Detected: $(uname)"
fi

_ARCH=$(uname -m)
if [ "$_ARCH" != "arm64" ]; then
    step "warning" "Apple Silicon recommended (detected: $_ARCH)" "$C_WARN"
fi

step "platform" "macOS ($_ARCH)"

# ── Detect Python ─────────────────────────────────────────────
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

# ── Create virtual environment ────────────────────────────────
if [ -x "$VENV_DIR/bin/python" ]; then
    step "venv" "using existing environment"
    substep "$VENV_DIR"
else
    step "venv" "creating virtual environment"
    substep "$VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ── Install uv ───────────────────────────────────────────────
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
    sh "$_uv_tmp" </dev/null >/dev/null 2>&1
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

# ── Install dependencies ──────────────────────────────────────
step "install" "installing mlx-vlm..."
uv pip install --python "$_VENV_PY" -q mlx-vlm
substep "done"

step "install" "installing transformers>=5.5.0..."
if uv pip install --python "$_VENV_PY" -q "transformers>=5.5.0" 2>/dev/null; then
    substep "installed from PyPI"
else
    substep "PyPI install failed (Python <3.10?), trying GitHub..."
    if uv pip install --python "$_VENV_PY" -q "git+https://github.com/huggingface/transformers.git@v5.5-release" 2>/dev/null; then
        substep "installed from huggingface/transformers v5.5-release"
    else
        step "warning" "could not install transformers>=5.5.0" "$C_WARN"
        substep "tried: PyPI, huggingface/transformers v5.5-release"
    fi
fi

# ── Verify installation ──────────────────────────────────────
if "$_VENV_PY" -c "import mlx_vlm"; then
    substep "mlx-vlm verified"
else
    fail "Installation verification failed."
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "Gemma 4 MLX installed!"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""
step "available models" "unsloth/gemma-4-E2B-it-UD-MLX-4bit"
substep "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
substep "unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit"
substep "unsloth/gemma-4-31b-it-UD-MLX-4bit"
echo ""
step "venv activate" "source ${VENV_DIR}/bin/activate"
echo ""
step "text chat" "python -m mlx_vlm.chat --model unsloth/gemma-4-E2B-it-UD-MLX-4bit"
echo ""
step "vision chat" "python -m mlx_vlm.chat --model unsloth/gemma-4-31b-it-UD-MLX-4bit"
substep "Use /image path/to/image.jpg to load an image"
echo ""
step "gradio UI" "python -m mlx_vlm.chat_ui --model unsloth/gemma-4-31b-it-UD-MLX-4bit"
echo ""
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""
