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

_remove_agent_instruction_files() {
    local _root
    for _root in "$@"; do
        [ -d "$_root" ] || continue
        [ -L "$_root" ] && continue
        find "$_root" \( -type f -o -type l \) \( -name 'AGENTS.md' -o -name 'CLAUDE.md' \) \
            -exec rm -f {} + 2>/dev/null || true
    done
}

# ── Corporate-mirror / proxy escape hatch for the frontend npm/bun install (#6491) ──
# studio/frontend/.npmrc pins registry=https://registry.npmjs.org/ as a supply-chain
# lock. A project-level pin overrides a corporate user's ~/.npmrc proxy, so the install
# hits npmjs.org directly and a firewall returns 403. UNSLOTH_NPM_REGISTRY is a
# deliberate opt-in: when set we thread it as `--registry <url>` into every npm/bun
# install. `--registry` is the highest-precedence override for BOTH tools and leaves
# min-release-age / save-exact in force. Empty array (the default) expands to nothing
# under `set -u`, so normal installs are unchanged.
_NPM_REGISTRY_ARGS=()
if [ -n "${UNSLOTH_NPM_REGISTRY:-}" ]; then
    _NPM_REGISTRY_ARGS=(--registry "$UNSLOTH_NPM_REGISTRY")
fi
# Failure-path capture log consumed by _suggest_npm_registry. Set to a temp file
# around the npm/bun installs; "" elsewhere so unrelated run_quiet calls don't capture.
_CAPTURE_LOG=""

# Print actionable guidance when a frontend/OXC npm/bun install fails and the registry
# lock is the likely cause (corporate firewall/proxy). No-op once the user has opted in
# via UNSLOTH_NPM_REGISTRY. We never switch registries automatically -- we only guide.
# $1 = path to a captured install log (may be empty/missing).
_suggest_npm_registry() {
    [ -n "${UNSLOTH_NPM_REGISTRY:-}" ] && return 0
    local _log="${1:-}"
    # If we captured output and it does NOT look like a registry/network problem, stay
    # quiet -- the raw error already shown is more useful than a misleading hint.
    if [ -n "$_log" ] && [ -s "$_log" ] \
        && ! grep -Eqi '40[13]|ENOTFOUND|ECONNREFUSED|ECONNRESET|ETIMEDOUT|EAI_AGAIN|ConnectionRefused|failed to resolve|registry\.npmjs\.org|getaddrinfo|tunneling socket|network|proxy|self.?signed|unable to (get|verify)' "$_log"; then
        return 0
    fi
    # Best-effort: surface a mirror the user already configured (env or ~/.npmrc).
    # Read npm config from / (a dir with no project .npmrc) so the frontend's pinned
    # registry= does not mask the user's ~/.npmrc / global mirror -- the caller is
    # still inside studio/frontend when this runs.
    local _mirror="${NPM_CONFIG_REGISTRY:-${npm_config_registry:-}}"
    if [ -z "$_mirror" ] && command -v npm >/dev/null 2>&1; then
        _mirror="$( (cd / 2>/dev/null && npm config get registry) 2>/dev/null || true )"
    fi
    case "$_mirror" in
        ""|undefined|null|https://registry.npmjs.org|https://registry.npmjs.org/) _mirror="" ;;
    esac
    printf '\n' >&2
    step "frontend" "registry.npmjs.org looks blocked (corporate firewall/proxy?)" "$C_WARN" >&2
    if [ -n "$_mirror" ]; then
        substep "Unsloth pins the public npm registry; your mirror is being ignored." >&2
        substep "Detected a registry in your npm config:" >&2
        substep "  $_mirror" >&2
        substep "Re-run pointing Unsloth at it:" >&2
        substep "  UNSLOTH_NPM_REGISTRY=$_mirror ./install.sh --local" >&2
    else
        substep "If you use a private mirror/proxy, point Unsloth at it and re-run:" >&2
        substep "  UNSLOTH_NPM_REGISTRY=https://your-mirror.example/api/npm/ ./install.sh --local" >&2
    fi
    substep "(min-release-age and save-exact stay enforced.)" >&2
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
        if [ -n "${_CAPTURE_LOG:-}" ]; then cat "$tmplog" >> "$_CAPTURE_LOG" 2>/dev/null || true; fi
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

# Echo a ';'-separated CUDA arch list (e.g. "86;120"). Override ($2,
# UNSLOTH_LLAMA_CUDA_ARCHS) wins verbatim; else parse+dedupe compute_cap text
# ($1). Empty means "no arch detected", so the caller builds CPU instead of a
# PTX-only binary that fails on an old driver (#5854).
_resolve_cuda_archs() {
    local _raw_caps=$1
    local _arch_override=$2
    if [ -n "$_arch_override" ]; then
        printf '%s' "$_arch_override"
        return 0
    fi
    local _archs="" _cap _arch
    while IFS= read -r _cap; do
        _cap=$(printf '%s' "$_cap" | tr -d '[:space:]')
        if [[ "$_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
            _arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
            case ";$_archs;" in
                *";$_arch;"*) ;;
                *) _archs="${_archs:+$_archs;}$_arch" ;;
            esac
        fi
    done <<< "$_raw_caps"
    printf '%s' "$_archs"
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
    substep "Or let Unsloth use the prebuilt CUDA bundle; it does not need the local toolkit." "$C_WARN"
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

# Resolve studio home + ownership marker before the llama-only split: the
# llama.cpp section needs STUDIO_HOME / _STUDIO_HOME_IS_CUSTOM, but
# UNSLOTH_STUDIO_LLAMA_ONLY=1 ('unsloth studio update') skips the base install.
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
# Directory-local evidence Unsloth created "$1": only prebuilt-installer metadata
# counts (UNSLOTH_PREBUILT_INFO.json for llama.cpp, UNSLOTH_NODE_PREBUILT_INFO.json
# for Node), both written only by our installers. Mirrors the setup.ps1 Node guard.
# A markerless source build stays strict since this runs right before an rm -rf.
_studio_owned_adoptable() {
    [ -f "$1/UNSLOTH_PREBUILT_INFO.json" ] && return 0
    [ -f "$1/UNSLOTH_NODE_PREBUILT_INFO.json" ] && return 0
    return 1
}
_assert_studio_owned_or_absent() {
    _aso_dir="$1"
    _aso_label="$2"
    [ -d "$_aso_dir" ] || return 0
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ ! -f "$_aso_dir/$_STUDIO_OWNED_MARKER" ]; then
        if _studio_owned_adoptable "$_aso_dir"; then
            : > "$_aso_dir/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
            return 0
        fi
        echo "ERROR: $_aso_dir already exists and is not marked as an Unsloth-owned $_aso_label." >&2
        echo "       Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." >&2
        exit 1
    fi
}

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

# OXC validator runtime (below) needs node/npm whenever its dir exists, regardless
# of dist staleness; provision Node when the frontend builds OR the OXC dir exists.
_OXC_DIR="$SCRIPT_DIR/backend/core/data_recipe/oxc-validator"
if [ "$_NEED_FRONTEND_BUILD" = false ] && [ ! -d "$_OXC_DIR" ]; then
    step "frontend" "up to date"
    verbose_substep "frontend dist is newer than source inputs"
else

# ── Node (isolated; never touches the system Node/npm) ──
# Unsloth's frontend (Vite 8) needs Node ^20.19 || >=22.12 || >=23 and npm >= 11.
# Three sources:
#   system  -- system Node + npm already satisfy both; used read-only.
#   bundled -- install a pinned isolated Node under $UNSLOTH_HOME/node, build-only.
#   skip    -- UNSLOTH_SKIP_NODE_INSTALL=1 and system unsuitable; print manual fix.
# decide_node_source(node_v, npm_v, skip_flag) -> system | bundled | skip
# (pure; unit-tested in tests/sh/test_node_decision.sh).
decide_node_source() {
    _dns_node="${1#v}"
    _dns_npm="$2"
    _dns_skip="$3"
    # Treat empty or non-numeric versions as "missing".
    case "$_dns_node" in ''|*[!0-9.]*) _dns_node='' ;; esac
    case "$_dns_npm"  in ''|*[!0-9.]*) _dns_npm=''  ;; esac
    if [ -n "$_dns_node" ] && [ -n "$_dns_npm" ]; then
        _dns_nmaj="${_dns_node%%.*}"
        case "$_dns_node" in
            *.*) _dns_rest="${_dns_node#*.}"; _dns_nmin="${_dns_rest%%.*}" ;;
            *)   _dns_nmin=0 ;;
        esac
        case "$_dns_nmin" in ''|*[!0-9]*) _dns_nmin=0 ;; esac
        _dns_pmaj="${_dns_npm%%.*}"
        _dns_ok=false
        if [ "$_dns_nmaj" -eq 20 ] && [ "$_dns_nmin" -ge 19 ]; then _dns_ok=true; fi
        if [ "$_dns_nmaj" -eq 22 ] && [ "$_dns_nmin" -ge 12 ]; then _dns_ok=true; fi
        if [ "$_dns_nmaj" -ge 23 ]; then _dns_ok=true; fi
        if [ "$_dns_ok" = true ] && [ "$_dns_pmaj" -ge 11 ]; then
            echo system
            return 0
        fi
    fi
    if [ "$_dns_skip" = "1" ]; then
        echo skip
        return 0
    fi
    echo bundled
}

# Mirror the llama.cpp UNSLOTH_HOME derivation; the frontend build runs first.
if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
    _NODE_PARENT="$STUDIO_HOME"
else
    _NODE_PARENT="$HOME/.unsloth"
fi
NODE_DIR="$_NODE_PARENT/node"

_SYS_NODE_VER="$(node -v 2>/dev/null || true)"
_SYS_NPM_VER="$(npm -v 2>/dev/null || true)"
NODE_SOURCE="$(decide_node_source "$_SYS_NODE_VER" "$_SYS_NPM_VER" "${UNSLOTH_SKIP_NODE_INSTALL:-0}")"
_FRONTEND_SKIP=false

if [ "$NODE_SOURCE" = system ]; then
    step "node" "$(node -v) | npm $(npm -v) (system)"
elif [ "$NODE_SOURCE" = bundled ]; then
    mkdir -p "$_NODE_PARENT"
    # install_node_prebuilt.py uses os.replace(); guard a custom-home dir so we
    # never displace a user-owned $UNSLOTH_STUDIO_HOME/node.
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$NODE_DIR" "Node install"
    fi
    substep "installing isolated Node (system Node/npm left untouched)..."
    # Runs before the venv is activated, so bare `python` may be absent; resolve
    # venv python, then python3, then python.
    if [ -x "$VENV_DIR/bin/python" ]; then
        _NODE_PY="$VENV_DIR/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        _NODE_PY="python3"
    else
        _NODE_PY="python"
    fi
    _NODE_LOG="$(mktemp)"
    set +e
    if _is_verbose; then
        "$_NODE_PY" "$SCRIPT_DIR/install_node_prebuilt.py" --install-dir "$NODE_DIR" 2>&1 | tee "$_NODE_LOG"
        _NODE_STATUS=${PIPESTATUS[0]}
    else
        "$_NODE_PY" "$SCRIPT_DIR/install_node_prebuilt.py" --install-dir "$NODE_DIR" >"$_NODE_LOG" 2>&1
        _NODE_STATUS=$?
    fi
    set -e
    if [ "$_NODE_STATUS" -eq 3 ]; then
        step "node" "install blocked by another active Unsloth install" "$C_ERR"
        sed 's/^/   | /' "$_NODE_LOG" >&2; rm -f "$_NODE_LOG"
        substep "close other Unsloth installs and retry"
        exit 3
    elif [ "$_NODE_STATUS" -ne 0 ]; then
        step "node" "isolated Node install failed" "$C_ERR"
        sed 's/^/   | /' "$_NODE_LOG" >&2; rm -f "$_NODE_LOG"
        substep "install Node >= 20.19 (with npm >= 11) yourself and re-run, or check your network"
        exit 1
    fi
    grep -Fq "already matches" "$_NODE_LOG" && verbose_substep "isolated Node already up to date"
    rm -f "$_NODE_LOG"
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ -d "$NODE_DIR" ]; then
        : > "$NODE_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    fi
    # Prepend the isolated bin (this process only) so node/npm/bun resolve here.
    export PATH="$NODE_DIR/bin:$PATH"
    # Keep npm and module resolution inside the isolated Node.
    export NPM_CONFIG_PREFIX="$NODE_DIR"
    export npm_config_prefix="$NODE_DIR"
    unset NODE_PATH
    hash -r 2>/dev/null || true
    step "node" "$(node -v) | npm $(npm -v) (isolated)"
else
    _FRONTEND_SKIP=true
    step "frontend" "skipped (no suitable Node; system left untouched)" "$C_WARN"
    substep "found Node='${_SYS_NODE_VER:-none}' npm='${_SYS_NPM_VER:-none}'; Unsloth needs Node >=20.19/22.12/23 and npm >= 11"
    substep "install a suitable Node + npm, or unset UNSLOTH_SKIP_NODE_INSTALL to let Unsloth manage an isolated Node"
fi
verbose_substep "node source: $NODE_SOURCE (sys node=${_SYS_NODE_VER:-none} npm=${_SYS_NPM_VER:-none}) dir=$NODE_DIR"

if [ "$_FRONTEND_SKIP" = true ]; then
    : # no suitable Node (skip source): message already shown above; nothing to build
elif [ "$_NEED_FRONTEND_BUILD" = false ]; then
    # Node was provisioned only for the OXC runtime; the dist is already current.
    step "frontend" "up to date"
    verbose_substep "frontend dist is newer than source inputs"
else

# ── Install bun (optional, faster package installs) ──
# Install bun via npm only when we manage the isolated Node (npm -g lands in the
# isolated prefix); on a system Node we install nothing global. Build falls back to npm.
if command -v bun &>/dev/null; then
    substep "bun already installed ($(bun --version))"
elif [ "$NODE_SOURCE" = bundled ]; then
    substep "installing bun..."
    # --allow-scripts=bun: npm >=11.16 gates install scripts and bun's
    # postinstall fetches its binary; without it the install is a broken stub.
    if run_maybe_quiet npm install -g bun --allow-scripts=bun "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" && command -v bun &>/dev/null; then
        substep "bun installed ($(bun --version))"
    else
        substep "bun install skipped (npm will be used instead)"
    fi
else
    verbose_substep "skipping global bun install on system Node (npm will be used)"
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
    bun install "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" >"$_log" 2>&1 || _exit_code=$?

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
    if [ -n "${_CAPTURE_LOG:-}" ]; then cat "$_log" >> "$_CAPTURE_LOG" 2>/dev/null || true; fi
    rm -f "$_log"
    rm -rf node_modules
    return 1
}

# Capture install output (bun + npm fallback) so we can detect a registry block.
_FRONTEND_INSTALL_LOG=$(mktemp)
_CAPTURE_LOG="$_FRONTEND_INSTALL_LOG"
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
    # `|| _npm_install_rc=$?` keeps this off `set -e`'s exit path (run_quiet_no_exit
    # returns non-zero on failure) so the hint branch is reachable; it also captures
    # the exact exit code. Mirrors the `|| BUILD_OK=false` idiom used below.
    _npm_install_rc=0
    run_quiet_no_exit "npm install" npm install --no-fund --no-audit --loglevel=error "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" || _npm_install_rc=$?
    if [ "$_npm_install_rc" -ne 0 ]; then
        _suggest_npm_registry "$_FRONTEND_INSTALL_LOG"
        rm -f "$_FRONTEND_INSTALL_LOG"
        exit "$_npm_install_rc"
    fi
fi
_CAPTURE_LOG=""
rm -f "$_FRONTEND_INSTALL_LOG"
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

fi  # end _FRONTEND_SKIP guard (Node available: system or isolated)

fi  # end frontend build check

# ── oxc-validator runtime ──
# Skip when the user opted out of Node (NODE_SOURCE=skip): there is no suitable
# Node, so do not run npm install against an unsuitable/absent system Node.
if [ -d "$_OXC_DIR" ] && [ "${NODE_SOURCE:-}" != skip ] && command -v npm &>/dev/null; then
    cd "$_OXC_DIR"
    _OXC_INSTALL_LOG=$(mktemp)
    _CAPTURE_LOG="$_OXC_INSTALL_LOG"
    # `|| _oxc_install_rc=$?` keeps this off `set -e`'s exit path so the hint branch
    # below is reachable; it also captures the exact exit code.
    _oxc_install_rc=0
    run_quiet_no_exit "npm install (oxc validator runtime)" npm install --no-fund --no-audit --loglevel=error "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" || _oxc_install_rc=$?
    _CAPTURE_LOG=""
    if [ "$_oxc_install_rc" -ne 0 ]; then
        _suggest_npm_registry "$_OXC_INSTALL_LOG"
        rm -f "$_OXC_INSTALL_LOG"
        exit "$_oxc_install_rc"
    fi
    rm -f "$_OXC_INSTALL_LOG"
    cd "$SCRIPT_DIR"
elif [ -d "$_OXC_DIR" ] && [ "${NODE_SOURCE:-}" != skip ]; then
    # No npm on PATH: skip rather than abort; the backend Node resolver degrades
    # the validator gracefully. Mirrors setup.ps1's elseif on this block.
    substep "OXC validator runtime skipped (no npm found); code validation degrades until Node is available" "$C_WARN"
fi

_remove_agent_instruction_files \
    "$SCRIPT_DIR/frontend/node_modules" \
    "$_OXC_DIR/node_modules"

# ── Python venv + deps ──

[ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
[ -d "$REPO_ROOT/.venv_overlay" ] && rm -rf "$REPO_ROOT/.venv_overlay"
[ -d "$REPO_ROOT/.venv_t5" ] && rm -rf "$REPO_ROOT/.venv_t5"
[ -d "$REPO_ROOT/.venv_t5_530" ] && rm -rf "$REPO_ROOT/.venv_t5_530"
[ -d "$REPO_ROOT/.venv_t5_550" ] && rm -rf "$REPO_ROOT/.venv_t5_550"
# Note: do NOT delete $STUDIO_HOME/.venv here — install.sh handles migration

_COLAB_NO_VENV=false
if [ ! -x "$VENV_DIR/bin/python" ]; then
    if [ "$IS_COLAB" = true ]; then
        # On Colab there is no Unsloth venv -- install backend deps into system Python.
        # Strip all version constraints so pip keeps Colab's pre-installed
        # packages (huggingface-hub, datasets, transformers) and only pulls
        # in genuinely missing ones (structlog, fastapi, etc.).
        substep "Colab detected, installing Unsloth backend dependencies..."
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
        # A pre-#6483-fix install can be stuck on anyio>=4.14 even though
        # $_PKG_NAME itself is current; the fast path above would otherwise
        # never reach install_python_stack's anyio repair (#6797).
        if "$VENV_DIR/bin/python" -c "
import re, sys
from importlib.metadata import version, PackageNotFoundError
try:
    parts = version('anyio').split('.')
    major = int(parts[0])
    minor = int(re.sub(r'[^0-9].*', '', parts[1])) if len(parts) > 1 else 0
except (PackageNotFoundError, ValueError, IndexError):
    sys.exit(1)
sys.exit(0 if (major, minor) >= (4, 14) else 1)
" 2>/dev/null; then
            substep "anyio >=4.14 found (#6483) -- forcing dependency pass to repair..."
            _SKIP_PYTHON_DEPS=false
        fi
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
elif [ "$(uname -s 2>/dev/null)" = "Darwin" ] && [ "$(uname -m 2>/dev/null)" = "arm64" ]; then
    # Apple Silicon: llama.cpp builds with Metal over unified memory, so not a CPU-only host.
    step "gpu" "Apple Silicon (Metal, unified memory)"
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

# Pick the release repo install_llama_prebuilt.py plans against. Every host this
# installer supports now pulls its llama.cpp prebuilt from the unslothai fork: it
# ships the CUDA (Linux x64/arm64, Windows), ROCm (Linux/Windows) and macOS
# bundles, plus the CPU bundles for Linux/Windows on both x86_64 and arm64.
# ggml-org artifacts are no longer used by default.
_HELPER_RELEASE_REPO="unslothai/llama.cpp"
# UNSLOTH_ROCM_GFX_ARCH may be set on a host where no probe fired, so the override
# nested in the AMD-detected branch above never ran and _setup_gfx is still empty.
# Honour it here so the --rocm-gfx forwarding below still sees it
# (install_llama_prebuilt.py reads the same env var as the --rocm-gfx default).
if [ "${_setup_nvidia_usable:-}" != true ] && [ -z "${_setup_gfx:-}" ] && [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
    _setup_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
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

# GGUF export's check_llama_cpp() looks for a llama-quantize shim at the root of
# the install dir, but a source build keeps the binary under build/bin/. Mirror
# the source-build-reuse step and create the shim when the reused tree has one
# but no root shim yet. Best-effort: the tree may be read-only (shared/CI cache),
# and under `set -e` a failed ln would otherwise abort an good reuse.
_link_local_llama_quantize_shim() {
    if [ -x "$1/build/bin/llama-quantize" ] && [ ! -e "$1/llama-quantize" ]; then
        ln -sf build/bin/llama-quantize "$1/llama-quantize" 2>/dev/null || \
            substep "could not create llama-quantize shim in linked dir (read-only?); GGUF export may be unavailable"
    fi
}

# Accept any layout LlamaCppBackend._layout_candidates() resolves so the flag
# never rejects a tree Unsloth could actually run: a root-level llama-server (a
# `make` build or a flat-extracted release) or the CMake build/bin/llama-server.
_has_local_llama_server() {
    [ -x "$1/llama-server" ] || [ -x "$1/build/bin/llama-server" ]
}

_LOCAL_LLAMA_CPP_LINKED=false
if [ -n "${UNSLOTH_LOCAL_LLAMA_CPP_DIR:-}" ]; then
    if [ ! -d "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" ]; then
        step "llama.cpp" "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $UNSLOTH_LOCAL_LLAMA_CPP_DIR" "$C_ERR"
        exit 1
    fi
    _RESOLVED_LOCAL="$(CDPATH= cd -P -- "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" && pwd -P)"
    # Canonicalize the install path the same way before comparing: _RESOLVED_LOCAL
    # is fully resolved, but LLAMA_CPP_DIR is textual ($UNSLOTH_HOME/llama.cpp). If
    # $HOME (or UNSLOTH_HOME) contains a symlink, the two never match even when the
    # user pointed the flag at the canonical install itself -- and the rm -rf below
    # would then wipe the very tree they asked to reuse. Resolve via the parent so
    # this works whether or not the leaf currently exists.
    _CANON_LLAMA_CPP_DIR="$LLAMA_CPP_DIR"
    _LLAMA_CPP_PARENT="$(dirname "$LLAMA_CPP_DIR")"
    if [ -d "$_LLAMA_CPP_PARENT" ]; then
        _CANON_LLAMA_CPP_DIR="$(CDPATH= cd -P -- "$_LLAMA_CPP_PARENT" && pwd -P)/$(basename "$LLAMA_CPP_DIR")"
    fi
    if [ "$_RESOLVED_LOCAL" = "$_CANON_LLAMA_CPP_DIR" ]; then
        # Points at the canonical install location itself: never delete-then-link
        # it onto itself. If a usable build is already there, reuse it and skip
        # both the prebuilt download and the source build -- the prebuilt installer
        # uses os.replace() and would otherwise clobber an existing source build at
        # this path. If nothing is built there yet, fall through to the normal
        # install so it gets built in place exactly as it would without the flag.
        if _has_local_llama_server "$LLAMA_CPP_DIR"; then
            substep "UNSLOTH_LOCAL_LLAMA_CPP_DIR is the canonical install location and already holds a build; reusing it"
            _link_local_llama_quantize_shim "$LLAMA_CPP_DIR"
            _LOCAL_LLAMA_CPP_LINKED=true
            _NEED_LLAMA_SOURCE_BUILD=false
            _SKIP_PREBUILT_INSTALL=true
        else
            substep "UNSLOTH_LOCAL_LLAMA_CPP_DIR points to the canonical install location with nothing built there yet; running the normal install"
        fi
    else
        # Reusing disables BOTH the prebuilt download and the source build, so the
        # linked tree must already contain a runnable llama-server in one of the
        # layouts the backend resolves (root-level or build/bin/). Fail clearly
        # rather than link an unbuilt or wrong-platform checkout and leave Unsloth
        # with no usable binary.
        if ! _has_local_llama_server "$_RESOLVED_LOCAL"; then
            step "llama.cpp" "no llama-server under $_RESOLVED_LOCAL (looked for ./llama-server and ./build/bin/llama-server) -- build llama.cpp there first, or drop --with-llama-cpp-dir" "$C_ERR"
            exit 1
        fi
        # A stale link from a previous --with-llama-cpp-dir run isn't Unsloth-owned
        # content; drop it before the ownership check so re-runs stay idempotent
        # for a custom UNSLOTH_STUDIO_HOME (the assert would otherwise follow the
        # link into the user's dir and reject it as unowned).
        [ -L "$LLAMA_CPP_DIR" ] && rm -f "$LLAMA_CPP_DIR"
        if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
            _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"
        fi
        rm -rf "$LLAMA_CPP_DIR"
        ln -sfn "$_RESOLVED_LOCAL" "$LLAMA_CPP_DIR"
        _link_local_llama_quantize_shim "$LLAMA_CPP_DIR"
        step "llama.cpp" "linked local directory: $_RESOLVED_LOCAL"
        _LOCAL_LLAMA_CPP_LINKED=true
        _NEED_LLAMA_SOURCE_BUILD=false
        _SKIP_PREBUILT_INSTALL=true
    fi
fi

if [ "$_LOCAL_LLAMA_CPP_LINKED" = true ]; then
    : # local directory linked above; skip prebuilt install
elif [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
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
        substep "close Unsloth or other llama.cpp users and retry"
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
                        # Resolve the arch list before committing to a CUDA build;
                        # an empty list means CPU instead of a PTX-only binary (#5854).
                        _raw_caps=""
                        # Resolve nvidia-smi as _setup_has_usable_nvidia_gpu does
                        # (PATH, then /usr/bin); `command -v` alone would miss an
                        # off-PATH binary and wrongly drop a CUDA host to CPU.
                        _smi_bin=""
                        if command -v nvidia-smi >/dev/null 2>&1; then
                            _smi_bin="nvidia-smi"
                        elif [ -x "/usr/bin/nvidia-smi" ]; then
                            _smi_bin="/usr/bin/nvidia-smi"
                        fi
                        if [ -n "$_smi_bin" ]; then
                            _raw_caps=$(_setup_run_smi "$_smi_bin" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
                        fi
                        CUDA_ARCHS="$(_resolve_cuda_archs "$_raw_caps" "${UNSLOTH_LLAMA_CUDA_ARCHS:-}")"

                        if [ -n "$CUDA_ARCHS" ]; then
                            CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
                            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS=--threads=0"
                            _BUILD_DESC="building (CUDA, sm_${CUDA_ARCHS//;/+sm_})"

                            # Allow a host gcc/clang newer than nvcc's whitelist (else a fresh
                            # toolkit aborts with "unsupported GNU version"); via env to avoid word-splitting.
                            export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }-allow-unsupported-compiler"
                        else
                            # No detectable arch: build CPU (CMAKE_ARGS has no
                            # -DGGML_CUDA=ON yet, so clearing GPU_BACKEND yields CPU).
                            substep "could not detect a CUDA compute capability; building CPU llama.cpp instead of a PTX-only binary (set UNSLOTH_LLAMA_CUDA_ARCHS, e.g. \"120\", to force a CUDA build)." "$C_WARN"
                            GPU_BACKEND=""
                            _BUILD_DESC="building (CPU, CUDA arch undetectable)"
                        fi
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
            # Best-effort: the DiffusionGemma visual server (an example target, present
            # on llama.cpp PR #24423). No-op when the diffusion example is not configured.
            run_quiet_no_exit "build diffusion visual server" cmake --build "$_BUILD_TMP/build" --config Release --target llama-diffusion-gemma-visual-server -j"$NCPU" || true
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
            # DiffusionGemma visual server, if it was built (PR #24423): link next to
            # llama-server so Unsloth serves DiffusionGemma GGUFs without DG_VISUAL_BIN.
            if [ -f "$LLAMA_CPP_DIR/build/bin/llama-diffusion-gemma-visual-server" ]; then
                ln -sf build/bin/llama-diffusion-gemma-visual-server "$LLAMA_CPP_DIR/llama-diffusion-gemma-visual-server"
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
# An arm64 Linux GPU host source-builds for the GPU above. If that produced no
# binary, install the fork's arm64 CPU prebuilt (app-<tag>-linux-arm64-cpu.tar.gz)
# instead of leaving the host without llama.cpp. --cpu-fallback drops the GPU
# attributes so the CPU bundle is selected rather than re-attempting CUDA.
if [ "$_LLAMA_CPP_DEGRADED" = true ] \
        && [ "$_HOST_SYSTEM" = "Linux" ] \
        && { [ "$_HOST_MACHINE" = "aarch64" ] || [ "$_HOST_MACHINE" = "arm64" ]; }; then
    substep "GPU source build unavailable; trying arm64 CPU prebuilt..."
    _ARM64_CPU_CMD=(
        python "$SCRIPT_DIR/install_llama_prebuilt.py"
        --install-dir "$LLAMA_CPP_DIR"
        --llama-tag "$_REQUESTED_LLAMA_TAG"
        --published-repo "unslothai/llama.cpp"
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

if [ ! -L "$LLAMA_CPP_DIR" ] && {
    [ "$_STUDIO_HOME_IS_CUSTOM" != true ] ||
        [ -f "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" ] ||
        _studio_owned_adoptable "$LLAMA_CPP_DIR"
}; then
    _remove_agent_instruction_files "$LLAMA_CPP_DIR"
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
    printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
    printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
fi
echo ""

# When called from install.sh (SKIP_STUDIO_BASE=1), exit non-zero so the
# installer can report the GGUF failure after finishing PATH/shortcut setup.
# When called directly via 'unsloth studio update', keep the install
# successful -- the footer above already reports the limitation and Unsloth
# is still usable for non-GGUF workflows.
if [ "$_LLAMA_CPP_DEGRADED" = true ] && [ "${SKIP_STUDIO_BASE:-0}" = "1" ]; then
    exit 1
fi
