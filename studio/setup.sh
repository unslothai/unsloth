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
# Usage: substep <message> [color]         (color defaults to C_DIM)
step()    { printf "  ${C_DIM}%-15.15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  %-15s${2:-$C_DIM}%s${C_RST}\n" "" "$1"; }

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

_nvcc_meets_llama_minimum() {
    # Echo "ok|too_old|unknown" then the parsed "X.Y" version, one per line.
    # llama.cpp needs CUDA toolkit >= 12.4 (#4437; setup.ps1 aborts via #4517).
    _nvcc_bin=$1
    [ -n "$_nvcc_bin" ] || { echo "unknown"; echo ""; return 0; }
    _raw=$("$_nvcc_bin" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | head -1)
    if [ -z "$_raw" ]; then
        echo "unknown"; echo ""; return 0
    fi
    _maj=${_raw%%.*}
    _min_raw=${_raw#*.}
    _min=${_min_raw%%.*}
    if [ "$_maj" -lt 12 ] 2>/dev/null; then
        echo "too_old"
    elif [ "$_maj" -eq 12 ] && [ "$_min" -lt 4 ] 2>/dev/null; then
        echo "too_old"
    else
        echo "ok"
    fi
    echo "$_raw"
}

# Run a GPU probe under a 10s timeout when `timeout` is available so a wedged
# NVIDIA driver cannot hang setup; fall back to a bare call where it is not.
_setup_run_smi() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 10 "$@"
    else
        "$@"
    fi
}

# Returns 0 when CUDA_VISIBLE_DEVICES is set to "" or "-1", i.e. every NVIDIA
# device is deliberately hidden (mixed AMD+NVIDIA hosts steering work to the
# AMD card). Unset means all devices visible. nvidia-smi ignores this env var,
# so the probes below cannot see the distinction on their own.
_setup_cvd_hides_nvidia() {
    [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] || return 1
    _setup_cvd_trim=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]')
    [ -z "$_setup_cvd_trim" ] || [ "$_setup_cvd_trim" = "-1" ]
}

# Returns 0 when an NVIDIA GPU is present and usable. Primary probe is
# `nvidia-smi -L` (timeout-bounded). Fallback is /proc/driver/nvidia/gpus,
# which the driver populates per GPU regardless of nvidia-smi state -- handles
# PATH gaps and driver init races. Mirrors install.sh _has_usable_nvidia_gpu
# (PR 6174) so setup routes the same way as the torch installer. A GPU hidden
# via CUDA_VISIBLE_DEVICES=""/-1 counts as NOT usable (matches
# install_llama_prebuilt.py has_usable_nvidia), so the AMD probes still run
# and a mixed host steered to its AMD card keeps the ROCm route.
_setup_has_usable_nvidia_gpu() {
    if _setup_cvd_hides_nvidia; then
        return 1
    fi
    _setup_nvsmi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        _setup_nvsmi="nvidia-smi"
    elif [ -x "/usr/bin/nvidia-smi" ]; then
        _setup_nvsmi="/usr/bin/nvidia-smi"
    fi
    if [ -n "$_setup_nvsmi" ]; then
        if _setup_run_smi "$_setup_nvsmi" -L 2>/dev/null \
           | awk '/^GPU[[:space:]]+[0-9]+:/{found=1} END{exit !found}'; then
            return 0
        fi
    fi
    if [ -d /proc/driver/nvidia/gpus ] && \
       [ -n "$(ls -A /proc/driver/nvidia/gpus 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

_cuda_driver_max_version() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    _setup_run_smi nvidia-smi 2>/dev/null \
        | sed -nE 's/.*CUDA( UMD)? Version:[[:space:]]*([0-9]+)\.([0-9]+).*/\2.\3/p' \
        | head -1 || true
}

_cuda_version_gt() {
    local _left=${1:-}
    local _right=${2:-}
    if ! [[ "$_left" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _left_major=$((10#${BASH_REMATCH[1]}))
    local _left_minor=$((10#${BASH_REMATCH[2]}))
    if ! [[ "$_right" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _right_major=$((10#${BASH_REMATCH[1]}))
    local _right_minor=$((10#${BASH_REMATCH[2]}))

    if [ "$_left_major" -gt "$_right_major" ]; then
        return 0
    fi
    if [ "$_left_major" -eq "$_right_major" ] && [ "$_left_minor" -gt "$_right_minor" ]; then
        return 0
    fi
    return 1
}

_cuda_toolkit_major_gt_driver() {
    local _toolkit_version=${1:-}
    local _driver_version=${2:-}
    if ! [[ "$_toolkit_version" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _toolkit_major=$((10#${BASH_REMATCH[1]}))
    if ! [[ "$_driver_version" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _driver_major=$((10#${BASH_REMATCH[1]}))
    [ "$_toolkit_major" -gt "$_driver_major" ]
}

_cuda_nvcc_candidate_paths() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
    fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        printf '%s\n' "/usr/local/cuda/bin/nvcc"
    fi
    ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V -r 2>/dev/null || true
}

_cuda_find_compatible_nvcc_for_driver() {
    local _driver_version=$1
    local _exclude_path=${2:-}
    local _candidate _seen _check _status _version
    local _best_path="" _best_version=""
    _seen="
"
    while IFS= read -r _candidate; do
        [ -n "$_candidate" ] || continue
        [ "$_candidate" != "$_exclude_path" ] || continue
        [ -x "$_candidate" ] || continue
        case "$_seen" in
            *"
$_candidate
"*) continue ;;
        esac
        _seen="${_seen}${_candidate}
"
        _check="$(_nvcc_meets_llama_minimum "$_candidate")"
        _status="$(printf '%s\n' "$_check" | sed -n '1p')"
        _version="$(printf '%s\n' "$_check" | sed -n '2p')"
        [ "$_status" = "ok" ] || continue
        [ -n "$_version" ] || continue
        if _cuda_toolkit_major_gt_driver "$_version" "$_driver_version"; then
            continue
        fi
        if [ -z "$_best_version" ] || _cuda_version_gt "$_version" "$_best_version"; then
            _best_path="$_candidate"
            _best_version="$_version"
        fi
    done <<EOF
$(_cuda_nvcc_candidate_paths)
EOF
    [ -n "$_best_path" ] || return 1
    printf '%s\n%s\n' "$_best_path" "$_best_version"
}

_print_cuda_driver_toolkit_mismatch() {
    local _toolkit_version=$1
    local _driver_version=$2
    local _toolkit_major=${_toolkit_version%%.*}
    local _driver_major=${_driver_version%%.*}
    substep "CUDA Toolkit $_toolkit_version is a major-version mismatch: toolkit major $_toolkit_major exceeds driver CUDA major $_driver_major ($_driver_version)." "$C_WARN"
    substep "Update the NVIDIA GPU driver to run CUDA Toolkit $_toolkit_version, or install a CUDA $_driver_major.x toolkit." "$C_WARN"
    substep "Or let Studio use the prebuilt CUDA bundle; it does not need the local toolkit." "$C_WARN"
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
source = str(payload.get("source") or "").strip()
binary_repo = str(payload.get("binary_repo") or "").strip()
binary_tag = str(payload.get("binary_release_tag") or "").strip()
if not repo or not release_tag:
    raise SystemExit(0)

# For non-fork sources (e.g. ggml-org upstream prebuilts) the published_repo/
# release_tag refer to the unsloth source tree while the actual binaries came
# from a different repo. Show both so the log is unambiguous.
if source and source != "upstream" and binary_repo and binary_tag and binary_repo != repo:
    message = f"installed release: {repo}@{release_tag} + {source}@{binary_tag}"
else:
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
    # --allow-scripts=bun: npm >=11.16 gates install scripts and bun's
    # postinstall fetches its binary; without it the install is a broken stub.
    if run_maybe_quiet npm install -g bun --allow-scripts=bun && command -v bun &>/dev/null; then
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
# UNSLOTH_STUDIO_HOME (or STUDIO_HOME alias) overrides the install root
# (mirrors install.sh). UNSLOTH_STUDIO_HOME wins when both are set.
_studio_override_var=""
_studio_override="${UNSLOTH_STUDIO_HOME:-}"
if [ -n "$_studio_override" ]; then
    _studio_override_var="UNSLOTH_STUDIO_HOME"
else
    _studio_override="${STUDIO_HOME:-}"
    [ -n "$_studio_override" ] && _studio_override_var="STUDIO_HOME"
fi
# Strip whitespace so " " is treated as unset (matches Python .strip()).
_studio_override=$(printf '%s' "$_studio_override" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
case "$_studio_override" in
    "~") _studio_override="$HOME" ;;
    "~/"*) _studio_override="$HOME/${_studio_override#'~/'}" ;;
esac
if [ -n "$_studio_override" ]; then
    # setup.sh runs against an existing install (via 'unsloth studio update');
    # a typo in the override must fail fast instead of materializing an
    # empty workspace dir. Mirrors setup.ps1 behavior.
    if [ ! -d "$_studio_override" ]; then
        echo "ERROR: $_studio_override_var=$_studio_override does not exist." >&2
        echo "       Run install.sh to create the install root before 'unsloth studio update'." >&2
        exit 1
    fi
    [ -w "$_studio_override" ] || { echo "ERROR: $_studio_override_var=$_studio_override is not writable." >&2; exit 1; }
    STUDIO_HOME="$(CDPATH= cd -P -- "$_studio_override" && pwd -P)" || exit 1
else
    STUDIO_HOME="$HOME/.unsloth/studio"
fi
VENV_DIR="$STUDIO_HOME/unsloth_studio"
VENV_T5_530_DIR="$STUDIO_HOME/.venv_t5_530"
VENV_T5_550_DIR="$STUDIO_HOME/.venv_t5_550"
VENV_T5_510_DIR="$STUDIO_HOME/.venv_t5_510"

[ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
[ -d "$REPO_ROOT/.venv_overlay" ] && rm -rf "$REPO_ROOT/.venv_overlay"
[ -d "$REPO_ROOT/.venv_t5" ] && rm -rf "$REPO_ROOT/.venv_t5"
[ -d "$REPO_ROOT/.venv_t5_530" ] && rm -rf "$REPO_ROOT/.venv_t5_530"
[ -d "$REPO_ROOT/.venv_t5_550" ] && rm -rf "$REPO_ROOT/.venv_t5_550"
# Note: do NOT delete $STUDIO_HOME/.venv here — install.sh handles migration

_COLAB_NO_VENV=false
if [ ! -x "$VENV_DIR/bin/python" ]; then
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
else
    step "python" "dependencies up to date"
    verbose_substep "python deps check: installed=$_PKG_NAME@${INSTALLED_VER:-unknown} latest=${LATEST_VER:-unknown}"
fi

# ── 6b. Pre-install transformers 5.x into .venv_t5_530/, .venv_t5_550/, and .venv_t5_510/ ──
# Models like GLM-4.7-Flash, Qwen3 MoE need transformers>=5.3.0.
# Gemma 4 models need transformers>=5.5.0; Gemma 4 Unified needs 5.10.x.
# Pre-install into separate directories to avoid runtime pip overhead.
# The training subprocess prepends the appropriate dir to sys.path.
#
# Runs outside the _SKIP_PYTHON_DEPS gate so that upgrades from legacy
# single .venv_t5 are always migrated to the tiered layout.
# why: in env-override mode $STUDIO_HOME is user-chosen; require the
# ownership marker before rm -rf so unrelated dirs survive. Gated on the
# canonical comparison so an override pointing at the legacy default still
# behaves like a default install.
_STUDIO_OWNED_MARKER=".unsloth-studio-owned"
_LEGACY_STUDIO_HOME="$HOME/.unsloth/studio"
_studio_home_canon="$STUDIO_HOME"
if [ -d "$_studio_home_canon" ]; then
    _studio_home_canon=$(CDPATH= cd -P -- "$_studio_home_canon" 2>/dev/null && pwd -P) \
        || _studio_home_canon="$STUDIO_HOME"
fi
if [ -d "$_LEGACY_STUDIO_HOME" ]; then
    _LEGACY_STUDIO_HOME=$(CDPATH= cd -P -- "$_LEGACY_STUDIO_HOME" 2>/dev/null && pwd -P) \
        || _LEGACY_STUDIO_HOME="$HOME/.unsloth/studio"
fi
_STUDIO_HOME_IS_CUSTOM=false
if [ "$_studio_home_canon" != "$_LEGACY_STUDIO_HOME" ]; then
    _STUDIO_HOME_IS_CUSTOM=true
fi
_assert_studio_owned_or_absent() {
    _aso_dir="$1"
    _aso_label="$2"
    [ -d "$_aso_dir" ] || return 0
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ ! -f "$_aso_dir/$_STUDIO_OWNED_MARKER" ]; then
        echo "ERROR: $_aso_dir already exists and is not marked as a Studio-owned $_aso_label." >&2
        echo "       Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." >&2
        exit 1
    fi
}
_target_has_pkg_version() {
    _thpv_dir="$1"
    _thpv_pkg="$2"
    _thpv_version="$3"
    [ -d "$_thpv_dir" ] || return 1
    _thpv_pkg_norm=$(printf '%s' "$_thpv_pkg" | tr '-' '_')
    for _thpv_metadata in \
        "$_thpv_dir"/"$_thpv_pkg_norm"-*.dist-info/METADATA \
        "$_thpv_dir"/"$_thpv_pkg"-*.dist-info/METADATA
    do
        [ -f "$_thpv_metadata" ] || continue
        grep -qx "Version: $_thpv_version" "$_thpv_metadata" && return 0
    done
    return 1
}
_NEED_T5_INSTALL=false
if [ -d "$STUDIO_HOME/.venv_t5" ]; then
    # Legacy layout — migrate
    _assert_studio_owned_or_absent "$STUDIO_HOME/.venv_t5" "legacy transformers sidecar venv"
    rm -rf "$STUDIO_HOME/.venv_t5"
    _NEED_T5_INSTALL=true
fi
[ ! -d "$VENV_T5_530_DIR" ] && _NEED_T5_INSTALL=true
[ ! -d "$VENV_T5_550_DIR" ] && _NEED_T5_INSTALL=true
[ ! -d "$VENV_T5_510_DIR" ] && _NEED_T5_INSTALL=true
_target_has_pkg_version "$VENV_T5_530_DIR" "transformers" "5.3.0" || _NEED_T5_INSTALL=true
_target_has_pkg_version "$VENV_T5_550_DIR" "transformers" "5.5.0" || _NEED_T5_INSTALL=true
_target_has_pkg_version "$VENV_T5_510_DIR" "transformers" "5.10.2" || _NEED_T5_INSTALL=true
# Also reinstall when python deps were updated (packages may need rebuild)
[ "$_SKIP_PYTHON_DEPS" = false ] && _NEED_T5_INSTALL=true

if [ "$_NEED_T5_INSTALL" = true ]; then
    _assert_studio_owned_or_absent "$VENV_T5_530_DIR" "transformers 5.3 sidecar venv"
    [ -d "$VENV_T5_530_DIR" ] && rm -rf "$VENV_T5_530_DIR"
    mkdir -p "$VENV_T5_530_DIR"
    : > "$VENV_T5_530_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    run_quiet "install transformers 5.3.0" fast_install --target "$VENV_T5_530_DIR" --no-deps "transformers==5.3.0"
    run_quiet "install huggingface_hub for t5_530" fast_install --target "$VENV_T5_530_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_530" fast_install --target "$VENV_T5_530_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_530" fast_install --target "$VENV_T5_530_DIR" "tiktoken"
    step "transformers" "5.3.0 pre-installed"

    _assert_studio_owned_or_absent "$VENV_T5_550_DIR" "transformers 5.5 sidecar venv"
    [ -d "$VENV_T5_550_DIR" ] && rm -rf "$VENV_T5_550_DIR"
    mkdir -p "$VENV_T5_550_DIR"
    : > "$VENV_T5_550_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    run_quiet "install transformers 5.5.0" fast_install --target "$VENV_T5_550_DIR" --no-deps "transformers==5.5.0"
    run_quiet "install huggingface_hub for t5_550" fast_install --target "$VENV_T5_550_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_550" fast_install --target "$VENV_T5_550_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_550" fast_install --target "$VENV_T5_550_DIR" "tiktoken"
    step "transformers" "5.5.0 pre-installed"

    _assert_studio_owned_or_absent "$VENV_T5_510_DIR" "transformers 5.10 sidecar venv"
    [ -d "$VENV_T5_510_DIR" ] && rm -rf "$VENV_T5_510_DIR"
    mkdir -p "$VENV_T5_510_DIR"
    : > "$VENV_T5_510_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    run_quiet "install transformers 5.10.2" fast_install --target "$VENV_T5_510_DIR" --no-deps "transformers==5.10.2"
    run_quiet "install huggingface_hub for t5_510" fast_install --target "$VENV_T5_510_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_510" fast_install --target "$VENV_T5_510_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_510" fast_install --target "$VENV_T5_510_DIR" "tiktoken"
    step "transformers" "5.10.2 pre-installed"
fi
fi

# ── GPU detection summary (mirrors setup.ps1 step "gpu" block) ──
# WSL2 ROCDXG: the system rocminfo enumerates the GPU over /dev/dxg only when
# HSA_ENABLE_DXG_DETECTION=1 (a no-op on bare metal), and /opt/rocm/bin can be
# off PATH outside login shells (the profile.d drop-in). Seed both before the
# probes or a ROCDXG WSL host is misdetected as CPU-only.
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
if ! command -v rocminfo >/dev/null 2>&1 && [ -x /opt/rocm/bin/rocminfo ]; then
    PATH="$PATH:/opt/rocm/bin"
fi
_setup_amd_detected=false
_setup_nvidia_usable=false
_setup_gfx_all=""
_setup_mkt=""
# NVIDIA priority: classify NVIDIA first and skip the AMD probes entirely on
# a usable-NVIDIA host (mirrors _has_rocm_gpu in install_python_stack.py).
# This also keeps a wedged rocminfo/amd-smi from hanging setup before the
# host is classified; the AMD probes themselves run under _setup_run_smi.
if _setup_has_usable_nvidia_gpu; then
    _setup_nvidia_usable=true
fi
if [ "$_setup_nvidia_usable" != true ]; then
    if command -v rocminfo >/dev/null 2>&1 && \
       _setup_run_smi rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx[1-9][0-9]/{found=1} END{exit !found}'; then
        _setup_amd_detected=true
        _setup_gfx_all=$(_setup_run_smi rocminfo 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        _setup_mkt=$(_setup_run_smi rocminfo 2>/dev/null | awk -F': ' \
            '/Marketing Name:/{gsub(/^[[:space:]]+|[[:space:]]+$/,"", $2); if($2){print $2; exit}}' || true)
    elif command -v amd-smi >/dev/null 2>&1 && \
         _setup_run_smi amd-smi list 2>/dev/null | awk '/^GPU[[:space:]]*[:\[][[:space:]]*[0-9]/{ found=1 } END{ exit !found }'; then
        _setup_amd_detected=true
        _setup_gfx_all=$(_setup_run_smi amd-smi list 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        [ -z "$_setup_gfx_all" ] && \
            _setup_gfx_all=$(_setup_run_smi amd-smi static --asic 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        _setup_mkt=$(_setup_run_smi amd-smi static --asic 2>/dev/null | awk -F'[:|]' \
            '/[Mm]arket.?[Nn]ame/{gsub(/^[[:space:]]+|[[:space:]]+$/,"", $2); if($2){print $2; exit}}' || true)
    elif [ -e /dev/kfd ] && \
         awk 'FNR==1{ gpu=0; amd=0 } /gpu_id/{ gpu=($2+0>0) } /vendor_id/{ amd=($2==4098) } \
              gpu && amd { found=1 } END{ exit !found }' \
             /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null; then
        # KFD sysfs fallback, AMD vendor_id 4098 only (mirrors install.sh
        # _has_amd_rocm_gpu): covers AMD hosts where rocminfo/amd-smi are
        # missing but the kernel exposes the GPU, so the source-build gate
        # below does not drop them to a CPU llama.cpp build. No gfx arch is
        # available from this path; name-based inference handles it.
        _setup_amd_detected=true
    fi
fi

if [ "$_setup_nvidia_usable" = true ]; then
    step "gpu" "NVIDIA GPU detected"
elif [ "$_setup_amd_detected" = true ]; then
    _setup_vis="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}"
    _setup_vis_idx=0
    if [ -n "$_setup_vis" ] && [ "$_setup_vis" != "-1" ]; then
        _setup_first="${_setup_vis%%,*}"
        case "$_setup_first" in ''|*[!0-9]*) ;; *) _setup_vis_idx=$_setup_first ;; esac
    fi
    _setup_gfx=$(printf '%s\n' "$_setup_gfx_all" | awk -v idx="$_setup_vis_idx" \
        'NF && !seen[$0]++ { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx] }')
    # UNSLOTH_ROCM_GFX_ARCH env override (mirrors setup.ps1)
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        _setup_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
        substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $_setup_gfx"
    # Name-based arch inference when tools don't report gfx (mirrors setup.ps1 nameArchTable)
    elif [ -z "$_setup_gfx" ] && [ -n "$_setup_mkt" ]; then
        # Kept in sync with the table in install.sh (and the PS nameArchTable).
        # gfx1102 matched BEFORE gfx1100 so the spaceless "RX 7700S" lands on
        # gfx1102 (bash case has no negative lookahead like the PS tables).
        case "$_setup_mkt" in
            *"9070 XT"*|*9080*)                                                                            _setup_gfx="gfx1201" ;;  # RDNA 4
            *9070*|*9060*)                                                                                 _setup_gfx="gfx1200" ;;  # RDNA 4
            *"8060S"*|*"8050S"*|*"8040S"*|*"Strix Halo"*|*"Ryzen AI Max"*|*"AI Max"*) _setup_gfx="gfx1151" ;;  # RDNA 3.5 (Strix Halo: Radeon 8060S/8050S/8040S iGPU, Ryzen AI Max+)
            *"890M"*|*"880M"*|*"860M"*|*"840M"*|*"Strix Point"*|*"Krackan"*|*"HX 37"*|*"AI 9 HX"*|*"AI 9 36"*|*"AI 7 35"*|*"AI 5 34"*|*"AI 7 PRO 35"*|*"AI 5 33"*) _setup_gfx="gfx1150" ;;  # RDNA 3.5 (Strix/Krackan Point: Radeon 890M/880M iGPU, Ryzen AI 9 HX 370/375)
            *"RX 7600"*|*"RX 7700S"*|*"RX 7650"*|*"PRO W7600"*|*"PRO W7500"*|*"PRO V710"*)                  _setup_gfx="gfx1102" ;;  # RDNA 3 (Navi 33)
            *"RX 7900"*|*"RX 7800"*|*"RX 7700"*|*"PRO W7900"*|*"PRO W7800"*|*"PRO W7700"*)                  _setup_gfx="gfx1100" ;;  # RDNA 3 desktop / workstation (Navi 31)
            *"780M"*|*"760M"*|*"740M"*|*"Phoenix"*|*"Hawk Point"*|*"Z1 Extreme"*|*"Z2 Extreme"*)            _setup_gfx="gfx1103" ;;  # RDNA 3 iGPU (Phoenix / Hawk Point)
            *"RX 6900"*|*"RX 6800"*|*"RX 6750"*|*"RX 6700"*|*"PRO W6800"*|*"PRO W6900"*)                    _setup_gfx="gfx1030" ;;  # RDNA 2 (Navi 21)
            *"RX 6650"*|*"RX 6600"*|*"PRO W6600"*|*"PRO W6650"*)                                            _setup_gfx="gfx1032" ;;  # RDNA 2 (Navi 23)
            *"RX 6500"*|*"RX 6400"*|*"RX 6300"*|*"PRO W6400"*|*"PRO W6500"*)                                _setup_gfx="gfx1034" ;;  # RDNA 2 (Navi 24)
        esac
        if [ -n "$_setup_gfx" ]; then
            substep "gfx arch inferred from GPU name: $_setup_gfx"
            substep "Tip: set UNSLOTH_ROCM_GFX_ARCH=$_setup_gfx to skip inference next time"
        fi
    fi
    # ROCm version via hipconfig, then amd-smi
    _setup_rocm_ver=""
    if command -v hipconfig >/dev/null 2>&1; then
        _setup_rocm_ver=$(hipconfig --version 2>/dev/null | awk 'NR==1 && /^[0-9]/{print; exit}' || true)
    fi
    if [ -z "$_setup_rocm_ver" ] && command -v amd-smi >/dev/null 2>&1; then
        _setup_rocm_ver=$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
            'NF>1{gsub(/[[:space:]]/,"", $2); print $2; exit}' || true)
    fi
    if [ -n "$_setup_gfx" ]; then
        step "gpu" "AMD ROCm ($_setup_gfx)"
    else
        step "gpu" "AMD ROCm"
    fi
    _setup_rocm_root="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
    substep "ROCm: $_setup_rocm_root"
    [ -n "$_setup_rocm_ver" ] && substep "hipconfig: $_setup_rocm_ver"
    [ -n "$_setup_mkt" ] && [ -n "$_setup_gfx" ] && substep "GPU: $_setup_mkt"
else
    step "gpu" "none (chat-only / GGUF)" "$C_WARN"
    substep "Training and GPU inference require an NVIDIA or AMD ROCm GPU."
fi

# ── 7. Prefer prebuilt llama.cpp bundles before any source build path ──
# Nest llama.cpp under $STUDIO_HOME only for real env-overrides; legacy
# default keeps ~/.unsloth/llama.cpp so pre-PR builds are still discovered.
if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
    UNSLOTH_HOME="$STUDIO_HOME"
else
    UNSLOTH_HOME="$HOME/.unsloth"
fi
mkdir -p "$UNSLOTH_HOME"
LLAMA_CPP_DIR="$UNSLOTH_HOME/llama.cpp"
LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
_NEED_LLAMA_SOURCE_BUILD=false
_LLAMA_CPP_DEGRADED=false
_LLAMA_FORCE_COMPILE="${UNSLOTH_LLAMA_FORCE_COMPILE:-0}"
_REQUESTED_LLAMA_TAG="${UNSLOTH_LLAMA_TAG:-${_DEFAULT_LLAMA_TAG}}"
_HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
_HOST_MACHINE="$(uname -m 2>/dev/null || true)"

# Pick the release repo install_llama_prebuilt.py plans against.
# The fork ships CUDA (Linux x64/arm64, Windows), ROCm (Linux/Windows) and
# macOS bundles. Only the plain CPU/Vulkan bundles still come from ggml-org, so
# CPU-only Linux (x86_64 and arm64) routes there; GPU Linux, Windows and macOS
# use unslothai.
_LINUX_HAS_GPU=false
# Route to the fork only for a usable GPU. NVIDIA counts only when a device is
# actually enumerated and not hidden via CUDA_VISIBLE_DEVICES=""/-1
# (_setup_nvidia_usable, from _setup_has_usable_nvidia_gpu above) -- mirroring
# install_llama_prebuilt.py's has_usable_nvidia. Mere nvidia-smi presence
# (CPU-only CUDA-toolkit containers, broken drivers) or a hidden GPU therefore
# takes the ggml-org CPU prebuilt instead of a slow source build. AMD is
# deliberately left on tooling presence, not usability: an unusable NVIDIA host
# has a good CPU prebuilt to fall back to, whereas tightening AMD would regress
# ROCm hosts exposing only hipconfig/hipinfo into an unnecessary CPU build.
if [ "$_setup_nvidia_usable" = true ]; then
    _LINUX_HAS_GPU=true
else
    for _GPU_TOOL in rocminfo amd-smi hipconfig hipinfo; do
        if command -v "$_GPU_TOOL" >/dev/null 2>&1; then
            _LINUX_HAS_GPU=true
            break
        fi
    done
fi

if [ "$_HOST_SYSTEM" = "Linux" ] \
        && [ "$_HOST_MACHINE" = "x86_64" ] \
        && [ "$_LINUX_HAS_GPU" = false ]; then
    _HELPER_RELEASE_REPO="ggml-org/llama.cpp"
elif [ "$_HOST_SYSTEM" = "Linux" ] \
        && { [ "$_HOST_MACHINE" = "aarch64" ] || [ "$_HOST_MACHINE" = "arm64" ]; } \
        && [ "$_LINUX_HAS_GPU" = false ]; then
    # CPU-only Linux ARM64 (Ampere Altra, Raspberry Pi 5, GitHub
    # `ubuntu-24.04-arm`, CPU-only Jetson rescue mode, ...). The fork ships no
    # arm64 CPU bundle, so without this branch the prebuilt resolver returns 0
    # attempts and the installer falls back to a source build. ggml-org ships
    # llama-bNNNN-bin-ubuntu-arm64.tar.gz from at least b9072 onward.
    _HELPER_RELEASE_REPO="ggml-org/llama.cpp"
else
    # GPU Linux (x64 CUDA/ROCm, arm64 CUDA), Windows (CUDA/ROCm), and macOS.
    _HELPER_RELEASE_REPO="unslothai/llama.cpp"
fi
unset _GPU_TOOL
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
    # why: install_llama_prebuilt.py uses os.replace(), which would displace
    # an unrelated $UNSLOTH_STUDIO_HOME/llama.cpp before the source-build
    # ownership check below ever runs.
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"
    fi
    _PREBUILT_CMD=(
        python "$SCRIPT_DIR/install_llama_prebuilt.py"
        --install-dir "$LLAMA_CPP_DIR"
        --llama-tag "$_REQUESTED_LLAMA_TAG"
        --published-repo "$_HELPER_RELEASE_REPO"
    )
    if [ -n "${UNSLOTH_LLAMA_RELEASE_TAG:-}" ]; then
        _PREBUILT_CMD+=(--published-release-tag "$UNSLOTH_LLAMA_RELEASE_TAG")
    fi
    # Forward the gfx arch resolved above so the per-gfx ROCm prebuilt is picked
    # even when the installer's own probe cannot report it (amd-smi-only hosts,
    # name-inferred arch). Implies --has-rocm on the installer side.
    if [ -n "${_setup_gfx:-}" ]; then
        _PREBUILT_CMD+=(--rocm-gfx "$_setup_gfx")
    elif [ "$_setup_amd_detected" = true ]; then
        # AMD was detected but gfx resolution failed; tell the installer ROCm is
        # present so it can still attempt a prebuilt. Mirrors setup.ps1 behaviour.
        _PREBUILT_CMD+=(--has-rocm)
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
        if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ -d "$LLAMA_CPP_DIR" ]; then
            : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
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

# Source-built llama.cpp installs do not have the prebuilt metadata used above
# for exact release matching. Reuse a complete local source build unless the
# caller explicitly requested a rebuild or a PR-specific llama.cpp checkout.
if [ "$_NEED_LLAMA_SOURCE_BUILD" = true ] && \
   [ "$_LLAMA_FORCE_COMPILE" != "1" ] && \
   [ -z "$_LLAMA_PR" ] && \
   [ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ] && \
   [ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    step "llama.cpp" "existing source build found; skipping rebuild"
    ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    fi
    _NEED_LLAMA_SOURCE_BUILD=false
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
            # Set Release explicitly (llama.cpp only defaults to it on non-MSVC/Xcode).
            CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=ON"
            _TRY_METAL_CPU_FALLBACK=false
            _HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
            _HOST_MACHINE="$(uname -m 2>/dev/null || true)"
            _IS_MACOS_ARM64=false
            if [ "$_HOST_SYSTEM" = "Darwin" ] && { [ "$_HOST_MACHINE" = "arm64" ] || [ "$_HOST_MACHINE" = "aarch64" ]; }; then
                _IS_MACOS_ARM64=true
            fi

            # macOS: pin a low deployment target so the source build loads on
            # older macOS too (else a macOS 26 host stamps minos=26). Set before
            # CPU_FALLBACK_CMAKE_ARGS copies CMAKE_ARGS so both paths inherit it.
            if [ "$_HOST_SYSTEM" = "Darwin" ]; then
                _MACOS_DEPLOYMENT_TARGET="${UNSLOTH_MACOS_DEPLOYMENT_TARGET:-13.3}"
                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_OSX_DEPLOYMENT_TARGET=${_MACOS_DEPLOYMENT_TARGET}"
                export MACOSX_DEPLOYMENT_TARGET="${_MACOS_DEPLOYMENT_TARGET}"
            fi

            if command -v ccache &>/dev/null; then
                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
            fi
            CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"

            GPU_BACKEND=""
            NVCC_PATH=""
            # Gate the CUDA toolkit search on an actually-usable NVIDIA GPU
            # (_setup_nvidia_usable, computed in the GPU summary block above;
            # already false when hidden via CUDA_VISIBLE_DEVICES=""/-1).
            # A CUDA toolkit alone (CPU-only build container, leftover packages)
            # is not proof of a GPU: building with -DGGML_CUDA=ON there yields a
            # binary that fails at runtime, so fall through to the CPU build.
            if [ "$_setup_nvidia_usable" = true ]; then
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
            fi

            # Check for ROCm (AMD) only if CUDA was not already selected, and
            # only when an AMD GPU was actually detected (_setup_amd_detected).
            # hipcc presence alone (HIP SDK, no GPU) must not select a HIP build.
            # NVIDIA-usable hosts never build HIP (defense in depth: the AMD
            # probes above are already skipped when NVIDIA is usable).
            ROCM_HIPCC=""
            if [ -z "$GPU_BACKEND" ] && [ "$_setup_nvidia_usable" != true ] && [ "$_setup_amd_detected" = true ]; then
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
                # Returns "ok|too_old|unknown\nX.Y" on stdout.
                _NVCC_CHECK="$(_nvcc_meets_llama_minimum "$NVCC_PATH")"
                _NVCC_STATUS="$(printf '%s\n' "$_NVCC_CHECK" | sed -n '1p')"
                _NVCC_VER="$(printf '%s\n' "$_NVCC_CHECK" | sed -n '2p')"

                if [ "$_NVCC_STATUS" = "too_old" ]; then
                    substep "CUDA toolkit $_NVCC_VER is below llama.cpp minimum (12.4)." "$C_ERR"
                    substep "install a newer CUDA toolkit: https://developer.nvidia.com/cuda-toolkit-archive" "$C_WARN"
                    substep "falling back to CPU llama.cpp build for this run." "$C_WARN"
                    NVCC_PATH=""
                    GPU_BACKEND=""
                    _BUILD_DESC="building (CPU, CUDA toolkit < 12.4)"
                else
                    _DRIVER_MAX_CUDA="$(_cuda_driver_max_version)"
                    _CUDA_TOOLKIT_ALLOWED=true
                    if [ -n "$_NVCC_VER" ] && [ -n "$_DRIVER_MAX_CUDA" ] && \
                       _cuda_toolkit_major_gt_driver "$_NVCC_VER" "$_DRIVER_MAX_CUDA"; then
                        _BLOCKED_NVCC_VER="$_NVCC_VER"
                        if _ALT_NVCC_CHECK="$(_cuda_find_compatible_nvcc_for_driver "$_DRIVER_MAX_CUDA" "$NVCC_PATH")"; then
                            NVCC_PATH="$(printf '%s\n' "$_ALT_NVCC_CHECK" | sed -n '1p')"
                            _NVCC_VER="$(printf '%s\n' "$_ALT_NVCC_CHECK" | sed -n '2p')"
                            GPU_BACKEND="cuda"
                            export PATH="$(dirname "$NVCC_PATH"):$PATH"
                            substep "CUDA Toolkit $_BLOCKED_NVCC_VER is a major-version mismatch with driver CUDA $_DRIVER_MAX_CUDA; using compatible CUDA Toolkit $_NVCC_VER at $NVCC_PATH." "$C_WARN"
                        else
                            _print_cuda_driver_toolkit_mismatch "$_NVCC_VER" "$_DRIVER_MAX_CUDA"
                            substep "falling back to CPU llama.cpp build for this run." "$C_WARN"
                            NVCC_PATH=""
                            GPU_BACKEND=""
                            _BUILD_DESC="building (CPU, CUDA toolkit major > driver)"
                            _CUDA_TOOLKIT_ALLOWED=false
                        fi
                    fi

                    if [ "$_CUDA_TOOLKIT_ALLOWED" = true ]; then
                        CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"

                        CUDA_ARCHS=""
                        if command -v nvidia-smi &>/dev/null; then
                            _raw_caps=$(_setup_run_smi nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
                            while IFS= read -r _cap; do
                                _cap=$(echo "$_cap" | tr -d '[:space:]')
                                if [[ "$_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
                                    _arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
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

                        # Allow a host gcc/clang newer than nvcc's whitelist (else a fresh
                        # toolkit aborts with "unsupported GNU version"); via env to avoid word-splitting.
                        export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }-allow-unsupported-compiler"
                    fi
                fi
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

                # ROCm 7.x ships clang-20 which on Ubuntu 24.04+ defaults to the
                # highest-numbered gcc lib dir (/usr/lib/gcc/x86_64-linux-gnu/14/)
                # which contains runtime objects but NOT C++ headers, causing:
                #   fatal error: 'cstdlib' file not found
                # Find the newest gcc install dir that actually has both the
                # runtime dir AND /usr/include/c++/<ver> headers, then pass it
                # to clang via --gcc-install-dir so HIP builds succeed.
                _GCC_INSTALL_DIR=""
                _gcc_pm="$(gcc -print-multiarch 2>/dev/null)"
                case "$_gcc_pm" in
                    *-linux-gnu*) _GCC_MULTIARCH="$_gcc_pm" ;;
                    *) _GCC_MULTIARCH="$(uname -m)-linux-gnu" ;;
                esac
                for _gcc_ver in 14 13 12 11; do
                    if [ -d "/usr/lib/gcc/$_GCC_MULTIARCH/$_gcc_ver/include" ] && \
                       [ -d "/usr/include/c++/$_gcc_ver" ]; then
                        _GCC_INSTALL_DIR="/usr/lib/gcc/$_GCC_MULTIARCH/$_gcc_ver"
                        break
                    fi
                done
                if [ -n "$_GCC_INSTALL_DIR" ]; then
                    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_HIP_FLAGS=--gcc-install-dir=\"$_GCC_INSTALL_DIR\""
                    substep "ROCm HIP gcc install dir: $_GCC_INSTALL_DIR"
                fi

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
            elif [ -d /usr/local/cuda ] || _setup_run_smi nvidia-smi &>/dev/null; then
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

            # GPU label for the CPU-fallback message: Metal, else GPU_BACKEND
            # (cuda/rocm). Empty on a bare CPU build (nothing to fall back from).
            _gpu_fallback_label() {
                if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                    echo "Metal"
                elif [ -n "$GPU_BACKEND" ]; then
                    printf '%s' "$GPU_BACKEND" | tr '[:lower:]' '[:upper:]'
                fi
            }

            if ! run_quiet_no_exit "cmake llama.cpp" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CMAKE_ARGS; then
                _FB_LABEL="$(_gpu_fallback_label)"
                if [ -n "$_FB_LABEL" ]; then
                    _TRY_METAL_CPU_FALLBACK=false
                    substep "$_FB_LABEL configure failed; retrying CPU build..." "$C_WARN"
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback after $_FB_LABEL configure failed)"
                        # Now configured for CPU; clear GPU_BACKEND so a later
                        # build-step failure won't re-enter fallback on this config.
                        GPU_BACKEND=""
                    else
                        BUILD_OK=false
                    fi
                else
                    BUILD_OK=false
                fi
            fi
        fi

        if [ "$BUILD_OK" = true ]; then
            if ! run_quiet_no_exit "build llama-server" cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU"; then
                _FB_LABEL="$(_gpu_fallback_label)"
                if [ -n "$_FB_LABEL" ]; then
                    _TRY_METAL_CPU_FALLBACK=false
                    substep "$_FB_LABEL build failed; retrying CPU build..." "$C_WARN"
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback after $_FB_LABEL build failed)"
                        GPU_BACKEND=""
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
            _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"
            rm -rf "$LLAMA_CPP_DIR"
            mv "$_BUILD_TMP" "$LLAMA_CPP_DIR"
            : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
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

# ── arm64 Linux GPU: CPU prebuilt as a last resort ──
# arm64 Linux with a GPU has no CUDA prebuilt anywhere (the unslothai fork is
# x64 only; ggml-org ships no Linux CUDA build), so it source-builds for the
# GPU above. If that produced no binary, install ggml-org's arm64 CPU prebuilt
# instead of leaving the host without llama.cpp.
if [ "$_LLAMA_CPP_DEGRADED" = true ] \
        && [ "$_HOST_SYSTEM" = "Linux" ] \
        && { [ "$_HOST_MACHINE" = "aarch64" ] || [ "$_HOST_MACHINE" = "arm64" ]; }; then
    substep "GPU source build unavailable; trying ggml-org arm64 CPU prebuilt..."
    _ARM64_CPU_CMD=(
        python "$SCRIPT_DIR/install_llama_prebuilt.py"
        --install-dir "$LLAMA_CPP_DIR"
        --llama-tag "$_REQUESTED_LLAMA_TAG"
        --published-repo "ggml-org/llama.cpp"
        --cpu-fallback
    )
    # Trust the installer's exit code: it validates the server before exiting 0,
    # the same signal the primary prebuilt path above relies on.
    if run_quiet_no_exit "arm64 CPU prebuilt" "${_ARM64_CPU_CMD[@]}"; then
        step "llama.cpp" "arm64 CPU prebuilt installed (GPU build unavailable)" "$C_WARN"
        _LLAMA_CPP_DEGRADED=false
        print_installed_llama_prebuilt_release "$LLAMA_CPP_DIR"
    fi
fi

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
