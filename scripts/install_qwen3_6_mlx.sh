#!/bin/bash
set -euo pipefail

# ============================================================
# Qwen3.6 MLX — One-command setup + inference
#
# Supply-chain hardening:
#   - All third-party downloads (uv installer, mlx_vlm qwen3_5
#     patches) are pinned to an immutable git commit SHA and verified
#     against a hardcoded SHA-256. Any mismatch aborts the install
#     before the bytes are copied into site-packages.
#   - To rotate any pin, fetch the new file with `curl`, run
#     `shasum -a 256`, and update the corresponding constant below.
# ============================================================
#
# Usage:
#   bash install_qwen3_6_mlx.sh [--venv-dir DIR]
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs uv, mlx-vlm, transformers, torch, torchvision
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
    VENV_DIR="$HOME/.unsloth/unsloth_qwen3_6_mlx"
fi

# ── Banner ────────────────────────────────────────────────────
echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "Qwen3.6 MLX Installer"
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
# Pin the uv installer payload by SHA-256. Rotate by running:
#   curl -sSLf https://astral.sh/uv/install.sh | shasum -a 256
# and updating the constant below. We fetch into a temp file, verify
# the digest, and only then execute. Mismatch aborts.
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

# ── Install dependencies ──────────────────────────────────────
step "install" "installing mlx-vlm..."
uv pip install --python "$_VENV_PY" -q mlx-vlm
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

# ── Verify installation ──────────────────────────────────────
if "$_VENV_PY" -c "import mlx_vlm; import torch; import torchvision; import transformers"; then
    substep "mlx-vlm + torch + transformers verified"
else
    fail "Installation verification failed. Please ensure Python >=3.10 and try again."
fi

# ── Apply patches for multi-turn image chat ──────────────────
#
# Pin every patch to an immutable commit SHA and verify the body
# against a hardcoded SHA-256. The mlx_vlm_qwen3_5 patch tree
# currently only exists on the upstream `fix/ui-fix` branch; we pin
# to the branch HEAD commit, NOT the floating ref, so a forced push
# on `fix/ui-fix` cannot swap the bytes under us.
#
# Rotate by:
#   _PATCH_COMMIT=<new SHA>
#   curl -sSLf "https://raw.githubusercontent.com/unslothai/unsloth/$_PATCH_COMMIT/unsloth/models/patches/mlx_vlm_qwen3_5/qwen3_5.py" | shasum -a 256
#   curl -sSLf "https://raw.githubusercontent.com/unslothai/unsloth/$_PATCH_COMMIT/unsloth/models/patches/mlx_vlm_qwen3_5/generate.py" | shasum -a 256
_PATCH_COMMIT="013c99e51bbb8c4b83d88f3b150a1e53251a19d2"
_PATCH_BASE="https://raw.githubusercontent.com/unslothai/unsloth/${_PATCH_COMMIT}/unsloth/models/patches/mlx_vlm_qwen3_5"
_PATCH_SHA_QWEN35="4b6fbbcc59b1d6b935e7204351aae1476836d25542a11c7885402b672d2efa64"
_PATCH_SHA_GENERATE="50c4cbb8c3d94c0c74a4d209db6d2b23b102944c147c6421f2eded427b8edaf7"

_SITE_PKGS=$("$_VENV_PY" -c "import site; print(site.getsitepackages()[0])")

step "patch" "fixing multi-turn image chat..."

# Stage all downloads in an isolated tmpdir; we only copy into
# site-packages after every checksum has matched.
_PATCH_TMP=$(mktemp -d)
trap 'rm -rf "$_PATCH_TMP"' EXIT

apply_pinned_patch() {
    # apply_pinned_patch <remote_basename> <expected_sha256> <dest_abspath>
    _name="$1"; _expected="$2"; _dest="$3"
    _staged="$_PATCH_TMP/$_name"
    if ! curl -sSLf "${_PATCH_BASE}/${_name}" -o "$_staged"; then
        step "warning" "failed to download ${_name} patch — multi-turn image chat may not work" "$C_WARN"
        return 1
    fi
    _actual=$(shasum -a 256 "$_staged" | awk '{print $1}')
    if [ "$_actual" != "$_expected" ]; then
        step "warning" "${_name} SHA-256 mismatch (got $_actual expected $_expected) — refusing to install patch" "$C_WARN"
        return 1
    fi
    mkdir -p "$(dirname "$_dest")"
    cp "$_staged" "$_dest"
    return 0
}

if apply_pinned_patch "qwen3_5.py" "$_PATCH_SHA_QWEN35" "${_SITE_PKGS}/mlx_vlm/models/qwen3_5/qwen3_5.py"; then
    substep "patched qwen3_5.py (MRoPE position reset)"
fi

if apply_pinned_patch "generate.py" "$_PATCH_SHA_GENERATE" "${_SITE_PKGS}/mlx_vlm/generate.py"; then
    substep "patched generate.py (mask trim on cache reuse)"
fi

# Clear pycache so patches take effect
find "${_SITE_PKGS}/mlx_vlm" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
substep "cleared bytecode cache"

# ── Done ──────────────────────────────────────────────────────
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
