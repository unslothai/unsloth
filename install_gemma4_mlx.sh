#!/bin/bash
set -e

# ============================================================
# Gemma 4 MLX — One-command setup + inference
#
# Usage:
#   bash install_gemma4_mlx.sh [--venv-dir DIR]
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs uv, mlx, mlx-lm, transformers
#   3. Downloads gemma4.py and gemma4_text.py from unsloth repo
#   4. Installs them into mlx-lm's models directory
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
if ! command -v uv >/dev/null 2>&1; then
    step "uv" "installing uv package manager..."
    _uv_tmp=$(mktemp)
    curl -LsSf "https://astral.sh/uv/install.sh" -o "$_uv_tmp"
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

# ── Repo config ──────────────────────────────────────────────
BRANCH="fix/ui-fix"
REPO_URL="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/${BRANCH}"

# ── Install dependencies ──────────────────────────────────────
step "install" "installing mlx, mlx-lm..."
uv pip install --python "$_VENV_PY" -q mlx mlx-lm 2>/dev/null
substep "done"

step "install" "installing transformers>=5.5.0..."
uv pip install --python "$_VENV_PY" -q "transformers>=5.5.0" 2>/dev/null
substep "done"

# ── Find mlx-lm models directory ─────────────────────────────
MLX_MODELS=$("$_VENV_PY" -c "import mlx_lm; print(mlx_lm.__path__[0])")/models
step "models dir" "$MLX_MODELS"

# ── Download and install Gemma 4 model files ──────────────────

step "download" "installing Gemma 4 model files..."

_install_model_file() {
    _fname="$1"
    if curl -fsSL "${REPO_URL}/unsloth/models/${_fname}" -o "${MLX_MODELS}/${_fname}" 2>/dev/null; then
        substep "downloaded ${_fname} from branch ${BRANCH}"
    elif [ -f "./${_fname}" ]; then
        substep "using local ./${_fname}"
        cp "./${_fname}" "${MLX_MODELS}/${_fname}"
    else
        fail "Could not install ${_fname}. Tried:
                    1) ${REPO_URL}/unsloth/models/${_fname}
                    2) Local file ./${_fname}

                    To fix, download the file manually and place it in the current directory,
                    then re-run this script."
    fi
}

_install_model_file "gemma4.py"
_install_model_file "gemma4_text.py"

# Verify files were installed correctly
if "$_VENV_PY" -c "from mlx_lm.models.gemma4_text import ProportionalRoPE" 2>/dev/null; then
    substep "model files verified"
else
    fail "Model files installed but verification failed (ProportionalRoPE import error).
                    Try manually from: https://github.com/unslothai/unsloth/tree/feature/${BRANCH}"
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "Gemma 4 MLX installed!"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""
step "available models" "unsloth/gemma-4-E2B-it-UD-MLX-4bit (/BF16)"
substep "unsloth/gemma-4-E4B-it-UD-MLX-4bit (/BF16)"
echo ""
step "venv activate" "source ${VENV_DIR}/bin/activate"
echo ""
step "quick start" "python -m mlx_lm chat --model unsloth/gemma-4-E2B-it-UD-MLX-4bit --max-tokens 200"
echo ""
step "python API" "from mlx_lm import load, generate"
substep "model, tokenizer = load('unsloth/gemma-4-E2B-it-UD-MLX-4bit')"
substep "messages = [{'role': 'user', 'content': 'Hello!'}]"
substep "prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)"
substep "print(generate(model, tokenizer, prompt=prompt, max_tokens=200))"
echo ""
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""
