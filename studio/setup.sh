#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RULE=$(printf '\342\224\200%.0s' {1..52})

# ── Parse flags ──
# --local: install from the local repo checkout (overlays unsloth as editable
# and unsloth-zoo from git main). Mirrors install.sh --local for the Colab
# path that runs setup.sh directly without going through install.sh.
if [ "$#" -gt 0 ]; then
    for _arg in "$@"; do
        case "$_arg" in
            --local)
                export STUDIO_LOCAL_INSTALL=1
                export STUDIO_LOCAL_REPO="$REPO_ROOT"
                ;;
        esac
    done
fi

# ── Maintainer-editable defaults ──────────────────────────────────────────
# Change these in the GitHub-hosted script so all users get updated defaults.
# User environment variables always override these baked-in values.
#
#   _DEFAULT_LLAMA_PR_FORCE : PR number to build by default ("" = normal path)
#   _DEFAULT_LLAMA_SOURCE   : git clone URL for source builds
#   _DEFAULT_LLAMA_TAG      : llama.cpp ref to build ("latest" = newest release,
#                             "master" = bleeding-edge, "bNNNN" = specific tag)
#                             Prefer "latest" over "master" -- "master" bypasses
#                             the prebuilt resolver (no matching GitHub release),
#                             forces a source build, and causes HTTP 422 errors.
#                             Only use "master" temporarily when the latest release
#                             is missing support for a new model architecture.
# ──────────────────────────────────────────────────────────────────────────
_DEFAULT_LLAMA_PR_FORCE=""
_DEFAULT_LLAMA_SOURCE="https://github.com/ggml-org/llama.cpp"
_DEFAULT_LLAMA_TAG="latest"
_DEFAULT_LLAMA_FORCE_COMPILE_REF="master"

# ── Colors (same palette as startup_banner / install_python_stack) ──
if [ -n "${NO_COLOR:-}" ]; then
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
elif [ -t 1 ] || [ -n "${FORCE_COLOR:-}" ]; then
    C_TITLE=$'\033[38;5;150m'
    C_DIM=$'\033[38;5;245m'
    C_OK=$'\033[38;5;108m'
    C_WARN=$'\033[38;5;136m'
    C_ERR=$'\033[91m'
    C_RST=$'\033[0m'
else
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
fi

# ── Output helpers ──
# Consistent column layout: 2-space indent, 15-char label (fits llama-quantize), then value.
# Usage: step <label> <message> [color]   (color defaults to C_OK)
step()    { printf "  ${C_DIM}%-15.15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "$1"; }

_is_verbose() {
    [ "${UNSLOTH_VERBOSE:-0}" = "1" ]
}

verbose_substep() {
    if _is_verbose; then
        substep "$1"
    fi
    return 0
}

run_maybe_quiet() {
    if _is_verbose; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

# ── Helper: run command quietly, show output only on failure ──
_run_quiet() {
    local on_fail=$1
    local label=$2
    shift 2

    if _is_verbose; then
        local exit_code
        "$@" && return 0
        exit_code=$?
        step "error" "$label failed (exit code $exit_code)" "$C_ERR" >&2
        if [ "$on_fail" = "exit" ]; then
            exit "$exit_code"
        else
            return "$exit_code"
        fi
    fi

    local tmplog
    tmplog=$(mktemp) || {
        step "error" "Failed to create temporary file" "$C_ERR" >&2
        [ "$on_fail" = "exit" ] && exit 1 || return 1
    }

    if "$@" >"$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        step "error" "$label failed (exit code $exit_code)" "$C_ERR" >&2
        cat "$tmplog" >&2
        rm -f "$tmplog"

        if [ "$on_fail" = "exit" ]; then
            exit "$exit_code"
        else
            return "$exit_code"
        fi
    fi
}

run_quiet() {
    _run_quiet exit "$@"
}

run_quiet_no_exit() {
    _run_quiet return "$@"
}

print_llama_error_log() {
    local log_file=$1
    [ -s "$log_file" ] || return 0
    substep "llama.cpp diagnostics (last 120 lines):"
    tail -n 120 "$log_file" | sed 's/^/   | /' >&2
}

installed_llama_prebuilt_release() {
    local install_dir=${1:-}
    local metadata_path="$install_dir/UNSLOTH_PREBUILT_INFO.json"
    [ -f "$metadata_path" ] || return 0
    python - "$metadata_path" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

if not isinstance(payload, dict):
    raise SystemExit(0)

repo = str(payload.get("published_repo") or "").strip()
release_tag = str(payload.get("release_tag") or "").strip()
llama_tag = str(payload.get("tag") or "").strip()
if not repo or not release_tag:
    raise SystemExit(0)

message = f"installed release: {repo}@{release_tag}"
if llama_tag and llama_tag != release_tag:
    message += f" (tag {llama_tag})"
print(message)
PY
}

print_installed_llama_prebuilt_release() {
    local install_dir=${1:-}
    local installed_release
    installed_release="$(installed_llama_prebuilt_release "$install_dir")"
    if [ -n "$installed_release" ]; then
        substep "$installed_release"
    fi
}

# ── Banner ──
echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "🦥 Unsloth Studio Setup"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
verbose_substep "verbose diagnostics enabled"
_LLAMA_ONLY="${UNSLOTH_STUDIO_LLAMA_ONLY:-0}"
if [ "$_LLAMA_ONLY" = "1" ]; then
    substep "llama.cpp only mode"
fi
if [ "${STUDIO_LOCAL_INSTALL:-0}" = "1" ]; then
    substep "local mode: overlaying $REPO_ROOT (editable) + unsloth-zoo from git main"
fi
# ── Clean up stale caches ──
rm -rf "$REPO_ROOT/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/backend/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/tmp/unsloth_compiled_cache"

# ── Detect Colab ──
IS_COLAB=false
keynames=$'\n'$(printenv | cut -d= -f1)
if [[ "$keynames" == *$'\nCOLAB_'* ]]; then
    IS_COLAB=true
fi

if [ "$_LLAMA_ONLY" != "1" ]; then
# ── Detect whether frontend needs building ──
# Skip if SKIP_STUDIO_FRONTEND=1 (Tauri desktop app bundles its own frontend),
# or if dist/ exists AND no tracked input is newer than dist/.
if [ "${SKIP_STUDIO_FRONTEND:-0}" = "1" ]; then
    _NEED_FRONTEND_BUILD=false
    step "frontend" "bundled (Tauri)"
else
_NEED_FRONTEND_BUILD=true
if [ -d "$SCRIPT_DIR/frontend/dist" ]; then
    _changed=$(find "$SCRIPT_DIR/frontend" -maxdepth 1 -type f \
        ! -name 'bun.lock' \
        -newer "$SCRIPT_DIR/frontend/dist" -print -quit 2>/dev/null)
    if [ -z "$_changed" ]; then
        _changed=$(find "$SCRIPT_DIR/frontend/src" "$SCRIPT_DIR/frontend/public" \
            -type f -newer "$SCRIPT_DIR/frontend/dist" -print -quit 2>/dev/null) || true
    fi
    [ -z "$_changed" ] && _NEED_FRONTEND_BUILD=false
fi
fi  # end SKIP_STUDIO_FRONTEND guard

if [ "$_NEED_FRONTEND_BUILD" = false ]; then
    step "frontend" "up to date"
    verbose_substep "frontend dist is newer than source inputs"
else

# ── Node ──
NEED_NODE=true
if command -v node &>/dev/null && command -v npm &>/dev/null; then
    NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
    NODE_MINOR=$(node -v | sed 's/v//' | cut -d. -f2)
    NPM_MAJOR=$(npm -v | cut -d. -f1)
    # Vite 8 requires Node ^20.19.0 || >=22.12.0
    NODE_OK=false
    if [ "$NODE_MAJOR" -eq 20 ] && [ "$NODE_MINOR" -ge 19 ]; then NODE_OK=true; fi
    if [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -ge 12 ]; then NODE_OK=true; fi
    if [ "$NODE_MAJOR" -ge 23 ]; then NODE_OK=true; fi
    if [ "$NODE_OK" = true ] && [ "$NPM_MAJOR" -ge 11 ]; then
        NEED_NODE=false
    else
        if [ "$IS_COLAB" = true ] && [ "$NODE_OK" = true ]; then
            # In Colab, just upgrade npm directly - nvm doesn't work well
            if [ "$NPM_MAJOR" -lt 11 ]; then
                substep "upgrading npm..."
                run_maybe_quiet npm install -g npm@latest
            fi
            NEED_NODE=false
        fi
    fi
fi

if [ "$NEED_NODE" = true ]; then
    substep "installing nvm..."
    export NODE_OPTIONS=--dns-result-order=ipv4first
    if _is_verbose; then
        curl -so- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    else
        curl -so- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash > /dev/null 2>&1
    fi

    export NVM_DIR="$HOME/.nvm"
    set +u
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    if [ -f "$HOME/.npmrc" ]; then
        if grep -qE '^\s*(prefix|globalconfig)\s*=' "$HOME/.npmrc"; then
            sed -i.bak '/^\s*\(prefix\|globalconfig\)\s*=/d' "$HOME/.npmrc"
        fi
    fi

    substep "installing Node LTS..."
    run_quiet "nvm install" nvm install --lts
    if _is_verbose; then
        nvm use --lts
    else
        nvm use --lts > /dev/null 2>&1
    fi
    set -u

    NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
    NPM_MAJOR=$(npm -v | cut -d. -f1)

    if [ "$NODE_MAJOR" -lt 20 ]; then
        step "node" "FAILED -- version must be >= 20 (got $(node -v))" "$C_ERR"
        exit 1
    fi
    if [ "$NPM_MAJOR" -lt 11 ]; then
        substep "upgrading npm..."
        run_quiet "npm update" npm install -g npm@latest
    fi
fi

step "node" "$(node -v) | npm $(npm -v)"
verbose_substep "node check: NEED_NODE=$NEED_NODE NODE_OK=${NODE_OK:-unknown} NPM_MAJOR=${NPM_MAJOR:-unknown}"

# ── Install bun (optional, faster package installs) ──
# Uses npm to install bun globally -- Node is already guaranteed above,
# avoids platform-specific installers, PATH issues, and admin requirements.
if ! command -v bun &>/dev/null; then
    substep "installing bun..."
    if run_maybe_quiet npm install -g bun && command -v bun &>/dev/null; then
        substep "bun installed ($(bun --version))"
    else
        substep "bun install skipped (npm will be used instead)"
    fi
else
    substep "bun already installed ($(bun --version))"
fi

# ── Build frontend ──
substep "building frontend..."
cd "$SCRIPT_DIR/frontend"
_HIDDEN_GITIGNORES=()
_dir="$(pwd)"
while [ "$_dir" != "/" ]; do
    _dir="$(dirname "$_dir")"
    if [ -f "$_dir/.gitignore" ] && grep -qx '\*' "$_dir/.gitignore" 2>/dev/null; then
        mv "$_dir/.gitignore" "$_dir/.gitignore._twbuild"
        _HIDDEN_GITIGNORES+=("$_dir/.gitignore")
    fi
done

_restore_gitignores() {
    for _gi in "${_HIDDEN_GITIGNORES[@]+"${_HIDDEN_GITIGNORES[@]}"}"; do
        mv "${_gi}._twbuild" "$_gi" 2>/dev/null || true
    done
}
trap _restore_gitignores EXIT

# Use bun for install if available (faster), fall back to npm.
# Build always uses npm (Node runtime -- avoids bun runtime issues on some platforms).
# NOTE: We intentionally avoid run_quiet for the bun install attempt because
# run_quiet calls exit on failure, which would kill the script before the npm
# fallback can run. Instead we capture output manually and only show it on failure.
#
# IMPORTANT: bun's package cache can become corrupt -- packages get stored
# with only metadata (package.json, README) but no actual content (bin/,
# lib/). When this happens bun install exits 0 but leaves binaries missing.
# We verify critical binaries after install. If missing, we clear the cache
# and retry once before falling back to npm.
_try_bun_install() {
    local _log _exit_code=0
    _log=$(mktemp)
    bun install >"$_log" 2>&1 || _exit_code=$?

    # bun may create .exe shims on Windows (Git Bash / MSYS2) instead of plain scripts
    if [ "$_exit_code" -eq 0 ] \
        && { [ -x node_modules/.bin/tsc ] || [ -f node_modules/.bin/tsc.exe ] || [ -f node_modules/.bin/tsc.bunx ]; } \
        && { [ -x node_modules/.bin/vite ] || [ -f node_modules/.bin/vite.exe ] || [ -f node_modules/.bin/vite.bunx ]; }; then
        rm -f "$_log"
        return 0
    fi

    # Either bun install failed or it exited 0 but left packages missing
    if [ "$_exit_code" -ne 0 ]; then
        echo "   bun install failed (exit code $_exit_code):"
    else
        echo "   bun install exited 0 but critical binaries are missing:"
    fi
    sed 's/^/   | /' "$_log" >&2
    rm -f "$_log"
    rm -rf node_modules
    return 1
}

_bun_install_ok=false
if command -v bun &>/dev/null; then
    substep "using bun for package install (faster)"
    if _try_bun_install; then
        _bun_install_ok=true
    else
        # First attempt failed, likely due to corrupt cache entries.
        # Clear the cache and retry once.
        echo "   Clearing bun cache and retrying..."
        run_maybe_quiet bun pm cache rm || true
        if _try_bun_install; then
            _bun_install_ok=true
        fi
    fi
fi
if [ "$_bun_install_ok" = false ]; then
    run_quiet_no_exit "npm install" npm install --no-fund --no-audit --loglevel=error
    _npm_install_rc=$?
    if [ "$_npm_install_rc" -ne 0 ]; then
        exit "$_npm_install_rc"
    fi
fi
run_quiet "npm run build" npm run build

_restore_gitignores
trap - EXIT

_MAX_CSS=$(find "$SCRIPT_DIR/frontend/dist/assets" -name '*.css' -exec wc -c {} + 2>/dev/null | sort -n | tail -1 | awk '{print $1}')
if [ -z "$_MAX_CSS" ]; then
    step "frontend" "built (warning: no CSS emitted)" "$C_WARN"
elif [ "$_MAX_CSS" -lt 100000 ]; then
    step "frontend" "built (warning: CSS may be truncated)" "$C_WARN"
else
    step "frontend" "built"
fi

cd "$SCRIPT_DIR"

fi  # end frontend build check

# ── oxc-validator runtime ──
if [ -d "$SCRIPT_DIR/backend/core/data_recipe/oxc-validator" ] && command -v npm &>/dev/null; then
    cd "$SCRIPT_DIR/backend/core/data_recipe/oxc-validator"
    run_quiet_no_exit "npm install (oxc validator runtime)" npm install --no-fund --no-audit --loglevel=error
    _oxc_install_rc=$?
    if [ "$_oxc_install_rc" -ne 0 ]; then
        exit "$_oxc_install_rc"
    fi
    cd "$SCRIPT_DIR"
fi

# ── Python venv + deps ──
STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth/studio}"
VENV_DIR="$STUDIO_HOME/unsloth_studio"
VENV_T5_530_DIR="$STUDIO_HOME/.venv_t5_530"
VENV_T5_550_DIR="$STUDIO_HOME/.venv_t5_550"

[ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
[ -d "$REPO_ROOT/.venv_overlay" ] && rm -rf "$REPO_ROOT/.venv_overlay"
[ -d "$REPO_ROOT/.venv_t5" ] && rm -rf "$REPO_ROOT/.venv_t5"
[ -d "$REPO_ROOT/.venv_t5_530" ] && rm -rf "$REPO_ROOT/.venv_t5_530"
[ -d "$REPO_ROOT/.venv_t5_550" ] && rm -rf "$REPO_ROOT/.venv_t5_550"
# Note: do NOT delete $STUDIO_HOME/.venv here — install.sh handles migration

_COLAB_NO_VENV=false
_DOCKER_NO_VENV=false
if [ -n "$UNSLOTH_DOCKER" ]; then
    # Docker: packages already in /opt/conda — skip venv entirely.
    # Only pre-install .venv_t5 for transformers 5.x switching (handled below).
    _DOCKER_NO_VENV=true
elif [ ! -x "$VENV_DIR/bin/python" ]; then
    if [ "$IS_COLAB" = true ]; then
        # On Colab there is no Studio venv -- install backend deps into system Python.
        # Strip all version constraints so pip keeps Colab's pre-installed
        # packages (huggingface-hub, datasets, transformers) and only pulls
        # in genuinely missing ones (structlog, fastapi, etc.).
        substep "Colab detected, installing Studio backend dependencies..."
        _COLAB_REQS_TMP="$(mktemp)"
        sed 's/[><=!~;].*//' "$SCRIPT_DIR/backend/requirements/studio.txt" \
            | grep -v '^#' | grep -v '^$' > "$_COLAB_REQS_TMP"
        if [ -s "$_COLAB_REQS_TMP" ]; then
            if ! run_quiet_no_exit "install Colab backend deps" pip install -q -r "$_COLAB_REQS_TMP"; then
                rm -f "$_COLAB_REQS_TMP"
                step "python" "Colab backend dependency install failed" "$C_ERR"
                exit 1
            fi
        else
            step "python" "no Colab backend dependencies resolved from requirements file" "$C_WARN"
        fi
        rm -f "$_COLAB_REQS_TMP"
        _COLAB_NO_VENV=true
    else
        step "python" "venv not found at $VENV_DIR" "$C_ERR"
        substep "Run install.sh first to create the environment:"
        substep "curl -fsSL https://unsloth.ai/install.sh | sh"
        exit 1
    fi
else
    source "$VENV_DIR/bin/activate"
fi

install_python_stack() {
    python "$SCRIPT_DIR/install_python_stack.py"
}

USE_UV=false
if command -v uv &>/dev/null; then
    USE_UV=true
elif {
    if _is_verbose; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    fi
}; then
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null && USE_UV=true
fi

fast_install() {
    if [ "$USE_UV" = true ]; then
        uv pip install --python "$(command -v python)" "$@" && return 0
    fi
    python -m pip install "$@"
}

cd "$SCRIPT_DIR"

# On Colab without a venv, skip venv-dependent Python deps sections but
# continue to llama.cpp install so GGUF inference is available.
if [ "$_COLAB_NO_VENV" = true ]; then
    step "python" "backend deps installed into system Python"
    substep "continuing to llama.cpp install for GGUF inference support"
fi

# In Docker, packages are pre-installed in /opt/conda — only install missing
# studio/data-designer deps and pre-install .venv_t5 for transformers 5.x.
if [ "$_DOCKER_NO_VENV" = true ]; then
    echo "   Docker detected — skipping venv activation."

    # Install branch's unsloth/unsloth_cli/studio into /opt/conda
    # (overwrites PyPI version with Docker-aware code)
    echo "   Installing local unsloth from branch..."
    pip install --force-reinstall --no-deps "$REPO_ROOT"

    # Install only missing deps (studio, data-designer, plugin, metadata patch).
    # Heavy packages (torch, unsloth, vllm, etc.) are already in /opt/conda.
    # install_python_stack.py has steps 1-5 commented out for this branch.
    python "$SCRIPT_DIR/install_python_stack.py"

    # Pre-install transformers 5.x into .venv_t5
    echo ""
    echo "   Pre-installing transformers 5.x for newer model support..."
    mkdir -p "$VENV_T5_DIR"
    pip install --target "$VENV_T5_DIR" --no-deps "transformers==5.3.0" 2>/dev/null
    pip install --target "$VENV_T5_DIR" --no-deps "huggingface_hub==1.7.1" 2>/dev/null
    pip install --target "$VENV_T5_DIR" --no-deps "hf_xet==1.4.2" 2>/dev/null
    pip install --target "$VENV_T5_DIR" "tiktoken" 2>/dev/null
    echo "✅ Transformers 5.x pre-installed to $VENV_T5_DIR/"

    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║     Docker Studio Setup Complete!    ║"
    echo "╚══════════════════════════════════════╝"
    exit 0
fi

# ── Check if Python deps need updating ──
# Compare installed package version against PyPI latest.
# Skip all Python dependency work if versions match (fast update path).
# On Colab (no venv), skip this version check (it needs $VENV_DIR/bin/python)
# but still run install_python_stack below (it uses sys.executable).
_SKIP_PYTHON_DEPS=false
_SKIP_VERSION_CHECK=false
if [ "$_COLAB_NO_VENV" = true ]; then
    _SKIP_VERSION_CHECK=true
fi
_PKG_NAME="${STUDIO_PACKAGE_NAME:-unsloth}"
if [ "$_SKIP_VERSION_CHECK" != true ] && [ "${SKIP_STUDIO_BASE:-0}" != "1" ] && [ "${STUDIO_LOCAL_INSTALL:-0}" != "1" ]; then
    # Only check when NOT called from install.sh (which just installed the package)
    INSTALLED_VER=$("$VENV_DIR/bin/python" -c "
import sys; from importlib.metadata import version
print(version(sys.argv[1]))
" "$_PKG_NAME" 2>/dev/null || echo "")

    LATEST_VER=$(curl -fsSL --max-time 5 "https://pypi.org/pypi/$_PKG_NAME/json" 2>/dev/null \
        | "$VENV_DIR/bin/python" -c "import sys,json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null \
        || echo "")

    if [ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ] && [ "$INSTALLED_VER" = "$LATEST_VER" ]; then
        step "python" "$_PKG_NAME $INSTALLED_VER is up to date"
        _SKIP_PYTHON_DEPS=true
    elif [ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ]; then
        substep "$_PKG_NAME $INSTALLED_VER -> $LATEST_VER available, updating..."
    elif [ -z "$LATEST_VER" ]; then
        substep "could not reach PyPI, updating to be safe..."
    fi
fi

if [ "$_SKIP_PYTHON_DEPS" = false ]; then
    install_python_stack

    # ── 6b. Pre-install transformers 5.x into .venv_t5_530/ and .venv_t5_550/ ──
    # Models like GLM-4.7-Flash, Qwen3 MoE need transformers>=5.3.0.
    # Gemma 4 models need transformers>=5.5.0.
    # Pre-install into separate directories to avoid runtime pip overhead.
    # The training subprocess prepends the appropriate dir to sys.path.

    # Clean up legacy single .venv_t5 directory
    [ -d "$STUDIO_HOME/.venv_t5" ] && rm -rf "$STUDIO_HOME/.venv_t5"

    [ -d "$VENV_T5_530_DIR" ] && rm -rf "$VENV_T5_530_DIR"
    mkdir -p "$VENV_T5_530_DIR"
    run_quiet "install transformers 5.3.0" fast_install --target "$VENV_T5_530_DIR" --no-deps "transformers==5.3.0"
    run_quiet "install huggingface_hub for t5_530" fast_install --target "$VENV_T5_530_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_530" fast_install --target "$VENV_T5_530_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_530" fast_install --target "$VENV_T5_530_DIR" "tiktoken"
    step "transformers" "5.3.0 pre-installed"

    [ -d "$VENV_T5_550_DIR" ] && rm -rf "$VENV_T5_550_DIR"
    mkdir -p "$VENV_T5_550_DIR"
    run_quiet "install transformers 5.5.0" fast_install --target "$VENV_T5_550_DIR" --no-deps "transformers==5.5.0"
    run_quiet "install huggingface_hub for t5_550" fast_install --target "$VENV_T5_550_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_550" fast_install --target "$VENV_T5_550_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_550" fast_install --target "$VENV_T5_550_DIR" "tiktoken"
    step "transformers" "5.5.0 pre-installed"
else
    step "python" "dependencies up to date"
    verbose_substep "python deps check: installed=$_PKG_NAME@${INSTALLED_VER:-unknown} latest=${LATEST_VER:-unknown}"
fi

# ── 6b. Pre-install transformers 5.x into .venv_t5_530/ and .venv_t5_550/ ──
# Models like GLM-4.7-Flash, Qwen3 MoE need transformers>=5.3.0.
# Gemma 4 models need transformers>=5.5.0.
# Pre-install into separate directories to avoid runtime pip overhead.
# The training subprocess prepends the appropriate dir to sys.path.
#
# Runs outside the _SKIP_PYTHON_DEPS gate so that upgrades from legacy
# single .venv_t5 are always migrated to the tiered layout.
_NEED_T5_INSTALL=false
if [ -d "$STUDIO_HOME/.venv_t5" ]; then
    # Legacy layout — migrate
    rm -rf "$STUDIO_HOME/.venv_t5"
    _NEED_T5_INSTALL=true
fi
[ ! -d "$VENV_T5_530_DIR" ] && _NEED_T5_INSTALL=true
[ ! -d "$VENV_T5_550_DIR" ] && _NEED_T5_INSTALL=true
# Also reinstall when python deps were updated (packages may need rebuild)
[ "$_SKIP_PYTHON_DEPS" = false ] && _NEED_T5_INSTALL=true

if [ "$_NEED_T5_INSTALL" = true ]; then
    [ -d "$VENV_T5_530_DIR" ] && rm -rf "$VENV_T5_530_DIR"
    mkdir -p "$VENV_T5_530_DIR"
    run_quiet "install transformers 5.3.0" fast_install --target "$VENV_T5_530_DIR" --no-deps "transformers==5.3.0"
    run_quiet "install huggingface_hub for t5_530" fast_install --target "$VENV_T5_530_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_530" fast_install --target "$VENV_T5_530_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_530" fast_install --target "$VENV_T5_530_DIR" "tiktoken"
    step "transformers" "5.3.0 pre-installed"

    [ -d "$VENV_T5_550_DIR" ] && rm -rf "$VENV_T5_550_DIR"
    mkdir -p "$VENV_T5_550_DIR"
    run_quiet "install transformers 5.5.0" fast_install --target "$VENV_T5_550_DIR" --no-deps "transformers==5.5.0"
    run_quiet "install huggingface_hub for t5_550" fast_install --target "$VENV_T5_550_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_550" fast_install --target "$VENV_T5_550_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_550" fast_install --target "$VENV_T5_550_DIR" "tiktoken"
    step "transformers" "5.5.0 pre-installed"
fi
fi

# ── 7. Prefer prebuilt llama.cpp bundles before any source build path ──
if [ "$_DOCKER_NO_VENV" = true ]; then
    step "llama.cpp" "skipped (Docker)"
else  # begin non-Docker llama.cpp block
UNSLOTH_HOME="$HOME/.unsloth"
mkdir -p "$UNSLOTH_HOME"
LLAMA_CPP_DIR="$UNSLOTH_HOME/llama.cpp"
LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
_NEED_LLAMA_SOURCE_BUILD=false
_LLAMA_CPP_DEGRADED=false
_LLAMA_FORCE_COMPILE="${UNSLOTH_LLAMA_FORCE_COMPILE:-0}"
_REQUESTED_LLAMA_TAG="${UNSLOTH_LLAMA_TAG:-${_DEFAULT_LLAMA_TAG}}"
_HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
if [ "$_HOST_SYSTEM" = "Darwin" ]; then
    _HELPER_RELEASE_REPO="ggml-org/llama.cpp"
else
    _HELPER_RELEASE_REPO="unslothai/llama.cpp"
fi
_LLAMA_PR="${UNSLOTH_LLAMA_PR:-}"
_SKIP_PREBUILT_INSTALL=false
_LLAMA_PR_FORCE="${UNSLOTH_LLAMA_PR_FORCE:-${_DEFAULT_LLAMA_PR_FORCE}}"
_LLAMA_SOURCE="${_DEFAULT_LLAMA_SOURCE}"
_LLAMA_SOURCE="${_LLAMA_SOURCE%.git}"  # normalize: strip trailing .git
_RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
_RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
_RESOLVED_SOURCE_REF_KIND="tag"
_RESOLVED_LLAMA_TAG="$_REQUESTED_LLAMA_TAG"

if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
    _NEED_LLAMA_SOURCE_BUILD=true
    _SKIP_PREBUILT_INSTALL=true
fi

# Baked-in PR_FORCE promotes to _LLAMA_PR when user hasn't set one.
if [ -z "$_LLAMA_PR" ] && [ -n "$_LLAMA_PR_FORCE" ] && \
   [[ "$_LLAMA_PR_FORCE" =~ ^[0-9]+$ ]] && [ "$_LLAMA_PR_FORCE" -gt 0 ]; then
    _LLAMA_PR="$_LLAMA_PR_FORCE"
    step "llama.cpp" "baked-in PR_FORCE=$_LLAMA_PR_FORCE" "$C_WARN"
fi

if [ -n "$_LLAMA_PR" ]; then
    if ! [[ "$_LLAMA_PR" =~ ^[0-9]+$ ]] || [ "$_LLAMA_PR" -le 0 ]; then
        step "llama.cpp" "UNSLOTH_LLAMA_PR=$_LLAMA_PR is not a valid PR number" "$C_ERR"
        exit 1
    fi
    step "llama.cpp" "UNSLOTH_LLAMA_PR=$_LLAMA_PR -- will build from PR head" "$C_WARN"
    _RESOLVED_LLAMA_TAG="pr-$_LLAMA_PR"
    _RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
    _RESOLVED_SOURCE_REF="pr-$_LLAMA_PR"
    _RESOLVED_SOURCE_REF_KIND="pull"
    _NEED_LLAMA_SOURCE_BUILD=true
    _SKIP_PREBUILT_INSTALL=true
fi

verbose_substep "requested llama.cpp tag: $_REQUESTED_LLAMA_TAG (repo: $_HELPER_RELEASE_REPO)"

if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
    step "llama.cpp" "UNSLOTH_LLAMA_FORCE_COMPILE=1 -- skipping prebuilt" "$C_WARN"
    _NEED_LLAMA_SOURCE_BUILD=true
elif [ "${_SKIP_PREBUILT_INSTALL:-false}" = true ]; then
    substep "prebuilt install skipped -- falling back to source build"
else
    substep "installing prebuilt llama.cpp..."
    if [ -d "$LLAMA_CPP_DIR" ]; then
        substep "existing install detected -- validating update"
    fi
    _PREBUILT_CMD=(
        python "$SCRIPT_DIR/install_llama_prebuilt.py"
        --install-dir "$LLAMA_CPP_DIR"
        --llama-tag "$_REQUESTED_LLAMA_TAG"
        --published-repo "$_HELPER_RELEASE_REPO"
        --simple-policy
    )
    if [ -n "${UNSLOTH_LLAMA_RELEASE_TAG:-}" ]; then
        _PREBUILT_CMD+=(--published-release-tag "$UNSLOTH_LLAMA_RELEASE_TAG")
    fi
    _PREBUILT_LOG="$(mktemp)"
    set +e
    if _is_verbose; then
        "${_PREBUILT_CMD[@]}" 2>&1 | tee "$_PREBUILT_LOG"
        _PREBUILT_STATUS=${PIPESTATUS[0]}
    else
        "${_PREBUILT_CMD[@]}" >"$_PREBUILT_LOG" 2>&1
        _PREBUILT_STATUS=$?
    fi
    set -e

    if [ "$_PREBUILT_STATUS" -eq 0 ]; then
        if grep -Fq "already matches" "$_PREBUILT_LOG"; then
            step "llama.cpp" "prebuilt up to date and validated"
        else
            step "llama.cpp" "prebuilt installed and validated"
        fi
        print_installed_llama_prebuilt_release "$LLAMA_CPP_DIR"
        verbose_substep "llama.cpp install dir: $LLAMA_CPP_DIR"
        rm -f "$_PREBUILT_LOG"
    elif [ "$_PREBUILT_STATUS" -eq 3 ]; then
        step "llama.cpp" "install blocked by active llama.cpp process" "$C_WARN"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "existing install was restored"
        fi
        substep "close Studio or other llama.cpp users and retry"
        exit 3
    else
        step "llama.cpp" "prebuilt install failed (continuing)" "$C_WARN"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "prebuilt update failed; existing install restored"
        fi
        substep "falling back to source build"
        _NEED_LLAMA_SOURCE_BUILD=true
    fi
fi

# ── 8. WSL: pre-install GGUF build dependencies for fallback source builds ──
# On WSL, sudo requires a password and can't be entered during GGUF export
# (runs in a non-interactive subprocess). Install build deps here instead.
if [ "$_NEED_LLAMA_SOURCE_BUILD" = true ] && grep -qi microsoft /proc/version 2>/dev/null; then
    _GGUF_DEPS="pciutils build-essential cmake curl git libcurl4-openssl-dev"
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y $_GGUF_DEPS >/dev/null 2>&1 || true

    _STILL_MISSING=""
    for _pkg in $_GGUF_DEPS; do
        case "$_pkg" in
            build-essential) command -v gcc >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            pciutils) command -v lspci >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            libcurl4-openssl-dev) command -v curl-config >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            *) command -v "$_pkg" >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
        esac
    done
    _STILL_MISSING=$(echo "$_STILL_MISSING" | sed 's/^ *//')

    if [ -z "$_STILL_MISSING" ]; then
        step "gguf deps" "installed"
    elif command -v sudo >/dev/null 2>&1; then
        step "gguf deps" "sudo required for: $_STILL_MISSING" "$C_WARN"
        printf "  %-15s" ""
        printf "accept? [Y/n] "
        if [ -r /dev/tty ]; then
            read -r REPLY </dev/tty || REPLY="y"
        else
            REPLY="y"
        fi
        case "$REPLY" in
            [nN]*)
                substep "skipped -- run manually:"
                substep "sudo apt-get install -y $_STILL_MISSING"
                _SKIP_GGUF_BUILD=true
                ;;
            *)
                sudo apt-get update -y
                sudo apt-get install -y $_STILL_MISSING
                step "gguf deps" "installed"
                ;;
        esac
    else
        step "gguf deps" "missing (no sudo) -- install manually:" "$C_WARN"
        substep "apt-get install -y $_STILL_MISSING"
        _SKIP_GGUF_BUILD=true
    fi
fi

# ── 9. Build llama.cpp binaries for GGUF inference + export when prebuilt install fails ──
# Builds at ~/.unsloth/llama.cpp — a single shared location under the user's
# home directory. This is used by both the inference server and the GGUF
# export pipeline (unsloth-zoo).
#   - llama-server: for GGUF model inference
#   - llama-quantize: for GGUF export quantization (symlinked to root for check_llama_cpp())
if [ "$_NEED_LLAMA_SOURCE_BUILD" = false ]; then
    :
elif [ "${_SKIP_GGUF_BUILD:-}" = true ]; then
    step "llama.cpp" "skipped (missing build deps)" "$C_WARN"
    [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
else
{
    if ! command -v cmake &>/dev/null; then
        step "llama.cpp" "skipped (cmake not found)" "$C_WARN"
        [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
    elif ! command -v git &>/dev/null; then
        step "llama.cpp" "skipped (git not found)" "$C_WARN"
        [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
    else
        if [ -z "$_LLAMA_PR" ]; then
            _RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
            if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
                if [ "$_REQUESTED_LLAMA_TAG" = "latest" ]; then
                    _RESOLVED_SOURCE_REF="${UNSLOTH_LLAMA_FORCE_COMPILE_REF:-${_DEFAULT_LLAMA_FORCE_COMPILE_REF}}"
                    _RESOLVED_SOURCE_REF_KIND="branch"
                else
                    _RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
                    _RESOLVED_SOURCE_REF_KIND="tag"
                fi
            elif [ "$_REQUESTED_LLAMA_TAG" = "latest" ]; then
                _RESOLVE_TAG_ARGS=(--resolve-llama-tag latest --published-repo "ggml-org/llama.cpp" --output-format json)
                set +e
                _RESOLVE_TAG_JSON="$(python "$SCRIPT_DIR/install_llama_prebuilt.py" "${_RESOLVE_TAG_ARGS[@]}" 2>/dev/null)"
                _RESOLVE_TAG_STATUS=$?
                set -e
                if [ "$_RESOLVE_TAG_STATUS" -eq 0 ] && [ -n "${_RESOLVE_TAG_JSON:-}" ]; then
                    _RESOLVED_SOURCE_REF="$(
                        printf '%s' "$_RESOLVE_TAG_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("llama_tag",""))' 2>/dev/null || true
                    )"
                else
                    _RESOLVED_SOURCE_REF=""
                fi
                if [ -z "$_RESOLVED_SOURCE_REF" ]; then
                    _RESOLVED_SOURCE_REF="latest"
                fi
                _RESOLVED_SOURCE_REF_KIND="tag"
            else
                _RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
                _RESOLVED_SOURCE_REF_KIND="tag"
            fi
            if [ -z "$_RESOLVED_SOURCE_URL" ]; then
                _RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
            fi
            if [ -z "$_RESOLVED_SOURCE_REF" ]; then
                _RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
            fi
        fi
        verbose_substep "source build repo: $_RESOLVED_SOURCE_URL"
        verbose_substep "source build ref: ${_RESOLVED_SOURCE_REF:-latest} (${_RESOLVED_SOURCE_REF_KIND})"
        BUILD_OK=true
        mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
        _BUILD_TMP="${LLAMA_CPP_DIR}.build.$$"
        rm -rf "$_BUILD_TMP"
        if [ -n "$_LLAMA_PR" ]; then
            run_quiet_no_exit "clone llama.cpp" \
                git clone --depth 1 "${_LLAMA_SOURCE}.git" "$_BUILD_TMP" || BUILD_OK=false
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "fetch PR #$_LLAMA_PR" \
                    git -C "$_BUILD_TMP" fetch --depth 1 origin "pull/$_LLAMA_PR/head:pr-$_LLAMA_PR" || BUILD_OK=false
            fi
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "checkout PR #$_LLAMA_PR" \
                    git -C "$_BUILD_TMP" checkout "pr-$_LLAMA_PR" || BUILD_OK=false
            fi
        elif [ "$_RESOLVED_SOURCE_REF_KIND" = "pull" ] && [ -n "$_RESOLVED_SOURCE_REF" ]; then
            run_quiet_no_exit "clone llama.cpp" \
                git clone --depth 1 "${_RESOLVED_SOURCE_URL}.git" "$_BUILD_TMP" || BUILD_OK=false
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "fetch source PR ref" \
                    git -C "$_BUILD_TMP" fetch --depth 1 origin "$_RESOLVED_SOURCE_REF" || BUILD_OK=false
            fi
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "checkout source PR ref" \
                    git -C "$_BUILD_TMP" checkout -B unsloth-llama-build FETCH_HEAD || BUILD_OK=false
            fi
        elif [ "$_RESOLVED_SOURCE_REF_KIND" = "commit" ] && [ -n "$_RESOLVED_SOURCE_REF" ]; then
            run_quiet_no_exit "clone llama.cpp" \
                git clone --depth 1 "${_RESOLVED_SOURCE_URL}.git" "$_BUILD_TMP" || BUILD_OK=false
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "fetch source commit" \
                    git -C "$_BUILD_TMP" fetch --depth 1 origin "$_RESOLVED_SOURCE_REF" || BUILD_OK=false
            fi
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "checkout source commit" \
                    git -C "$_BUILD_TMP" checkout -B unsloth-llama-build FETCH_HEAD || BUILD_OK=false
            fi
        else
            _CLONE_ARGS=(git clone --depth 1)
            if [ "$_RESOLVED_SOURCE_REF" != "latest" ] && [ -n "$_RESOLVED_SOURCE_REF" ]; then
                _CLONE_ARGS+=(--branch "$_RESOLVED_SOURCE_REF")
            fi
            _CLONE_ARGS+=("${_RESOLVED_SOURCE_URL}.git" "$_BUILD_TMP")
            run_quiet_no_exit "clone llama.cpp" \
                "${_CLONE_ARGS[@]}" || BUILD_OK=false
        fi

        if [ "$BUILD_OK" = true ]; then
            CMAKE_ARGS="-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=ON"
            _TRY_METAL_CPU_FALLBACK=false
            _HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
            _HOST_MACHINE="$(uname -m 2>/dev/null || true)"
            _IS_MACOS_ARM64=false
            if [ "$_HOST_SYSTEM" = "Darwin" ] && { [ "$_HOST_MACHINE" = "arm64" ] || [ "$_HOST_MACHINE" = "aarch64" ]; }; then
                _IS_MACOS_ARM64=true
            fi

            if command -v ccache &>/dev/null; then
                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
            fi
            CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"

            GPU_BACKEND=""
            NVCC_PATH=""
            if command -v nvcc &>/dev/null; then
                NVCC_PATH="$(command -v nvcc)"
                GPU_BACKEND="cuda"
            elif [ -x /usr/local/cuda/bin/nvcc ]; then
                NVCC_PATH="/usr/local/cuda/bin/nvcc"
                export PATH="/usr/local/cuda/bin:$PATH"
                GPU_BACKEND="cuda"
            elif ls /usr/local/cuda-*/bin/nvcc &>/dev/null 2>&1; then
                # Pick the newest cuda-XX.X directory
                NVCC_PATH="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
                export PATH="$(dirname "$NVCC_PATH"):$PATH"
                GPU_BACKEND="cuda"
            fi

            # Check for ROCm (AMD) only if CUDA was not already selected
            ROCM_HIPCC=""
            if [ -z "$GPU_BACKEND" ]; then
                if command -v hipcc &>/dev/null; then
                    ROCM_HIPCC="$(command -v hipcc)"
                    GPU_BACKEND="rocm"
                elif [ -x /opt/rocm/bin/hipcc ]; then
                    ROCM_HIPCC="/opt/rocm/bin/hipcc"
                    export PATH="/opt/rocm/bin:$PATH"
                    GPU_BACKEND="rocm"
                elif ls /opt/rocm-*/bin/hipcc &>/dev/null 2>&1; then
                    ROCM_HIPCC="$(ls -d /opt/rocm-*/bin/hipcc 2>/dev/null | sort -V | tail -1)"
                    export PATH="$(dirname "$ROCM_HIPCC"):$PATH"
                    GPU_BACKEND="rocm"
                fi
            fi

            _BUILD_DESC="building"
            if [ "$_IS_MACOS_ARM64" = true ]; then
                # Metal takes precedence on Apple Silicon (CUDA/ROCm not functional on macOS)
                _BUILD_DESC="building (Metal)"
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_USE_BF16=ON -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
                CPU_FALLBACK_CMAKE_ARGS="$CPU_FALLBACK_CMAKE_ARGS -DGGML_METAL=OFF"
                _TRY_METAL_CPU_FALLBACK=true
            elif [ -n "$NVCC_PATH" ]; then
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"

                CUDA_ARCHS=""
                if command -v nvidia-smi &>/dev/null; then
                    _raw_caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
                    while IFS= read -r _cap; do
                        _cap=$(echo "$_cap" | tr -d '[:space:]')
                        if [[ "$_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
                            _arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
                            # Append if not already present
                            case ";$CUDA_ARCHS;" in
                                *";$_arch;"*) ;;
                                *) CUDA_ARCHS="${CUDA_ARCHS:+$CUDA_ARCHS;}$_arch" ;;
                            esac
                        fi
                    done <<< "$_raw_caps"
                fi

                if [ -n "$CUDA_ARCHS" ]; then
                    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
                    _BUILD_DESC="building (CUDA, sm_${CUDA_ARCHS//;/+sm_})"
                else
                    _BUILD_DESC="building (CUDA)"
                fi

                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS=--threads=0"
            elif [ "$GPU_BACKEND" = "rocm" ]; then
                # Resolve hipcc symlinks to find the real ROCm root
                _HIPCC_REAL="$(readlink -f "$ROCM_HIPCC" 2>/dev/null || printf '%s' "$ROCM_HIPCC")"
                ROCM_ROOT=""
                if command -v hipconfig &>/dev/null; then
                    ROCM_ROOT="$(hipconfig -R 2>/dev/null || true)"
                fi
                if [ -z "$ROCM_ROOT" ]; then
                    ROCM_ROOT="$(cd "$(dirname "$_HIPCC_REAL")/.." 2>/dev/null && pwd)"
                fi

                _BUILD_DESC="building (ROCm)"
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"
                export ROCM_PATH="$ROCM_ROOT"
                export HIP_PATH="$ROCM_ROOT"

                # Use upstream-recommended HIP compiler (not legacy hipcc-as-CXX)
                if command -v hipconfig &>/dev/null; then
                    _HIP_CLANG_DIR="$(hipconfig -l 2>/dev/null || true)"
                    [ -n "$_HIP_CLANG_DIR" ] && export HIPCXX="$_HIP_CLANG_DIR/clang"
                fi

                # Detect AMD GPU architecture (gfx target)
                GPU_TARGETS=""
                if command -v rocminfo &>/dev/null; then
                    _gfx_list=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]{2,4}[a-z]?' | sort -u || true)
                    _valid_gfx=""
                    for _gfx in $_gfx_list; do
                        if [[ "$_gfx" =~ ^gfx[0-9]{2,4}[a-z]?$ ]]; then
                            # Drop bare family-level targets (gfx10, gfx11, gfx12, ...)
                            # when a specific sibling is present in the same list.
                            # rocminfo on ROCm 6.1+ emits both the specific GPU and
                            # the LLVM generic family line (e.g. gfx1100 alongside
                            # gfx11-generic), and the outer grep above captures the
                            # bare family prefix from the generic line. Passing that
                            # bare prefix to -DGPU_TARGETS breaks the HIP/llama.cpp
                            # build because clang only accepts specific gfxNNN ids.
                            # No real AMD GPU has a 2-digit gfx id, so this filter
                            # can only ever drop family prefixes, never real targets.
                            if [[ "$_gfx" =~ ^gfx[0-9]{2}$ ]] \
                               && echo "$_gfx_list" | grep -qE "^${_gfx}[0-9][0-9a-z]?$"; then
                                continue
                            fi
                            _valid_gfx="${_valid_gfx}${_valid_gfx:+;}$_gfx"
                        fi
                    done
                    [ -n "$_valid_gfx" ] && GPU_TARGETS="$_valid_gfx"
                fi

                if [ -n "$GPU_TARGETS" ]; then
                    CMAKE_ARGS="$CMAKE_ARGS -DGPU_TARGETS=${GPU_TARGETS}"
                    _BUILD_DESC="building (ROCm, ${GPU_TARGETS//;/+})"
                fi
            elif [ -d /usr/local/cuda ] || nvidia-smi &>/dev/null; then
                _BUILD_DESC="building (CPU, CUDA driver found but nvcc missing)"
            elif [ -d /opt/rocm ] || command -v rocm-smi &>/dev/null; then
                _BUILD_DESC="building (CPU, ROCm driver found but hipcc missing)"
            else
                _BUILD_DESC="building (CPU)"
            fi

            substep "$_BUILD_DESC..."

            NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
            CMAKE_GENERATOR_ARGS=""
            if command -v ninja &>/dev/null; then
                CMAKE_GENERATOR_ARGS="-G Ninja"
            fi

            if ! run_quiet_no_exit "cmake llama.cpp" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CMAKE_ARGS; then
                if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                    _TRY_METAL_CPU_FALLBACK=false
                    substep "Metal configure failed; retrying CPU build..." "$C_WARN"
                    rm -rf "$_BUILD_TMP/build"
                    run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS || BUILD_OK=false
                    if [ "$BUILD_OK" = true ]; then
                        _BUILD_DESC="building (CPU fallback)"
                    fi
                else
                    BUILD_OK=false
                fi
            fi
        fi

        if [ "$BUILD_OK" = true ]; then
            if ! run_quiet_no_exit "build llama-server" cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU"; then
                if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                    _TRY_METAL_CPU_FALLBACK=false
                    substep "Metal build failed; retrying CPU build..." "$C_WARN"
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback)"
                        run_quiet_no_exit "build llama-server (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU" || BUILD_OK=false
                    else
                        BUILD_OK=false
                    fi
                else
                    BUILD_OK=false
                fi
            fi
        fi

        if [ "$BUILD_OK" = true ]; then
            run_quiet_no_exit "build llama-quantize" cmake --build "$_BUILD_TMP/build" --config Release --target llama-quantize -j"$NCPU" || true
        fi

        # Swap only after build succeeds -- preserves existing install on failure
        if [ "$BUILD_OK" = true ]; then
            rm -rf "$LLAMA_CPP_DIR"
            mv "$_BUILD_TMP" "$LLAMA_CPP_DIR"
            # Symlink to llama.cpp root -- check_llama_cpp() looks for the binary there
            QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
            if [ -f "$QUANTIZE_BIN" ]; then
                ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
            fi
        else
            rm -rf "$_BUILD_TMP"
        fi

        if [ "$BUILD_OK" = true ] && [ -f "$LLAMA_SERVER_BIN" ]; then
            step "llama.cpp" "built"
            [ -f "$LLAMA_CPP_DIR/llama-quantize" ] && step "llama-quantize" "built"
        elif [ "$BUILD_OK" = true ]; then
            step "llama.cpp" "binary not found after build" "$C_WARN"
            _LLAMA_CPP_DEGRADED=true
        else
            step "llama.cpp" "build failed" "$C_ERR"
            [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
        fi
    fi
}
fi  # end _SKIP_GGUF_BUILD check
fi  # end non-Docker llama.cpp block

# ── Footer ──
if [ "$_LLAMA_ONLY" = "1" ]; then
    echo ""
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "llama.cpp update finished (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "llama.cpp update finished"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
elif [ "$IS_COLAB" = true ]; then
    echo ""
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "Unsloth Studio Setup Complete (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "Unsloth Studio Setup Complete"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    substep "from colab import start"
    substep "start()"
else
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "Unsloth Studio Installed (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "Unsloth Studio Installed"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_DIM}%-15s${C_WARN}%s${C_RST}\n" "launch" "unsloth studio -p 8888"
    else
        printf "  ${C_DIM}%-15s${C_OK}%s${C_RST}\n" "launch" "unsloth studio -p 8888"
    fi
    printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "(add -H 0.0.0.0 to allow network / cloud access)"
fi
echo ""

# When called from install.sh (SKIP_STUDIO_BASE=1), exit non-zero so the
# installer can report the GGUF failure after finishing PATH/shortcut setup.
# When called directly via 'unsloth studio update', keep the install
# successful -- the footer above already reports the limitation and Studio
# is still usable for non-GGUF workflows.
if [ "$_LLAMA_CPP_DEGRADED" = true ] && [ "${SKIP_STUDIO_BASE:-0}" = "1" ]; then
    exit 1
fi
