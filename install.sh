#!/bin/sh
#
# Unsloth Studio Installer
#
# Usage:  curl -fsSL https://unsloth.ai/install.sh | sh
#         wget  -qO- https://unsloth.ai/install.sh | sh
#         ./install.sh --local   (install from a cloned repo instead of PyPI)
#
# Piped installs take options as env vars after the pipe (a bare `| sh --no-torch`
# makes sh reject --no-torch as its own option). Flags still work via ./install.sh:
#   curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_NO_TORCH=1 sh       # skip PyTorch (GGUF-only)
#   curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_SKIP_AUTOSTART=1 sh # do not prompt to launch
#   curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_PYTHON=3.12 sh      # pin Python version
#   curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_STUDIO_HOME=/abs/path sh
# Equivalent flags: ./install.sh --no-torch --python 3.12  (or pipe them: sh -s -- --no-torch)
#
# Install dir priority: UNSLOTH_STUDIO_HOME > STUDIO_HOME (alias) > $HOME/.unsloth/studio
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e

# ── Output style (aligned with studio/setup.sh) ──
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
    C_TITLE="${_ESC}[38;5;150m"
    C_DIM="${_ESC}[38;5;245m"
    C_OK="${_ESC}[38;5;108m"
    C_WARN="${_ESC}[38;5;136m"
    C_ERR="${_ESC}[91m"
    C_RST="${_ESC}[0m"
else
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
fi

step()    { printf "  ${C_DIM}%-15.15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  ${C_DIM}%-15s${2:-$C_DIM}%s${C_RST}\n" "" "$1"; }

# ── Parse flags ──
STUDIO_LOCAL_INSTALL=false
PACKAGE_NAME="unsloth"
TAURI_MODE=false
_USER_PYTHON=""
_NO_TORCH_FLAG=false
_SKIP_AUTOSTART=false
_VERBOSE=false
_SHORTCUTS_ONLY=false
_next_is_package=false
_next_is_python=false
_next_is_llama_cpp_dir=false
# Seed from the environment so a caller who exports UNSLOTH_LOCAL_LLAMA_CPP_DIR
# (the documented piped-install style) is honored; the --with-llama-cpp-dir
# flag below overrides it when given.
_WITH_LLAMA_CPP_DIR="${UNSLOTH_LOCAL_LLAMA_CPP_DIR:-}"
for arg in "$@"; do
    if [ "$_next_is_package" = true ]; then
        PACKAGE_NAME="$arg"
        _next_is_package=false
        continue
    fi
    if [ "$_next_is_python" = true ]; then
        _USER_PYTHON="$arg"
        _next_is_python=false
        continue
    fi
    if [ "$_next_is_llama_cpp_dir" = true ]; then
        _WITH_LLAMA_CPP_DIR="$arg"
        _next_is_llama_cpp_dir=false
        continue
    fi
    case "$arg" in
        --local) STUDIO_LOCAL_INSTALL=true ;;
        --package) _next_is_package=true ;;
        --tauri) TAURI_MODE=true ;;
        --python) _next_is_python=true ;;
        --no-torch) _NO_TORCH_FLAG=true ;;
        --verbose|-v) _VERBOSE=true ;;
        --shortcuts-only) _SHORTCUTS_ONLY=true ;;
        --with-llama-cpp-dir) _next_is_llama_cpp_dir=true ;;
    esac
done

# Env-var equivalents for piped installs; an explicit flag still wins.
case "${UNSLOTH_NO_TORCH:-}" in 1|true|TRUE|yes|YES|on|ON) _NO_TORCH_FLAG=true ;; esac
case "${UNSLOTH_SKIP_AUTOSTART:-}" in 1|true|TRUE|yes|YES|on|ON) _SKIP_AUTOSTART=true ;; esac
[ -z "$_USER_PYTHON" ] && [ -n "${UNSLOTH_PYTHON:-}" ] && _USER_PYTHON="$UNSLOTH_PYTHON"

if [ "$_VERBOSE" = true ]; then
    export UNSLOTH_VERBOSE=1
fi

# Custom Unsloth roots are not supported with --tauri (desktop app still
# resolves ~/.unsloth/studio). Pass through if the override == legacy default.
if [ "$TAURI_MODE" = true ]; then
    _tauri_override_var=""
    _tauri_override="${UNSLOTH_STUDIO_HOME:-}"
    if [ -n "$_tauri_override" ]; then
        _tauri_override_var="UNSLOTH_STUDIO_HOME"
    else
        _tauri_override="${STUDIO_HOME:-}"
        [ -n "$_tauri_override" ] && _tauri_override_var="STUDIO_HOME"
    fi
    # Strip whitespace so " " is treated as unset (matches Python .strip()).
    _tauri_override=$(printf '%s' "$_tauri_override" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    if [ -n "$_tauri_override" ]; then
        case "$_tauri_override" in
            "~") _tauri_override="$HOME" ;;
            "~/"*) _tauri_override="$HOME/${_tauri_override#'~/'}" ;;
        esac
        # Canonicalize both sides (CDPATH=, -P) so a CDPATH-set env or
        # symlinked $HOME doesn't break the legacy-equality comparison.
        if [ -d "$_tauri_override" ]; then
            _tauri_override_abs=$(CDPATH= cd -P -- "$_tauri_override" 2>/dev/null && pwd -P) \
                || _tauri_override_abs="$_tauri_override"
        else
            _tauri_override_abs="$_tauri_override"
        fi
        # Strip trailing separators so ".../studio/" matches ".../studio".
        while [ "$_tauri_override_abs" != "/" ] \
            && [ "${_tauri_override_abs%/}" != "$_tauri_override_abs" ]; do
            _tauri_override_abs=${_tauri_override_abs%/}
        done
        _tauri_legacy_root="$HOME/.unsloth/studio"
        if [ -d "$_tauri_legacy_root" ]; then
            _tauri_legacy_root=$(CDPATH= cd -P -- "$_tauri_legacy_root" 2>/dev/null && pwd -P) \
                || _tauri_legacy_root="$HOME/.unsloth/studio"
        fi
        while [ "$_tauri_legacy_root" != "/" ] \
            && [ "${_tauri_legacy_root%/}" != "$_tauri_legacy_root" ]; do
            _tauri_legacy_root=${_tauri_legacy_root%/}
        done
        if [ "$_tauri_override_abs" != "$_tauri_legacy_root" ]; then
            echo "ERROR: $_tauri_override_var is not supported with --tauri." >&2
            echo "       The desktop app still uses the legacy ~/.unsloth/studio root." >&2
            echo "       Run install.sh without --tauri for custom-root shell installs," >&2
            echo "       or unset the env var for default desktop installs." >&2
            exit 1
        fi
    fi
fi

_is_verbose() {
    [ "${UNSLOTH_VERBOSE:-0}" = "1" ]
}

run_maybe_quiet() {
    if _is_verbose; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

# Trim trailing slashes from the URL PATH only, preserving ?query / #fragment: a whole-URL
# strip corrupts a token ending in "/", a single strip leaves .../cu128// empty. Shared.
_trim_index_path_slashes() {
    _tips_v="$1"
    case "$_tips_v" in
        *[?#]*)
            _tips_head="${_tips_v%%[?#]*}"
            _tips_tail="${_tips_v#"$_tips_head"}"
            ;;
        *)
            _tips_head="$_tips_v"
            _tips_tail=""
            ;;
    esac
    while [ -n "$_tips_head" ] && [ "${_tips_head%/}" != "$_tips_head" ]; do
        _tips_head="${_tips_head%/}"
    done
    printf '%s%s' "$_tips_head" "$_tips_tail"
}

# Redact index-URL credentials (userinfo + ?query= + #fragment) from captured installer
# output before printing on failure; uv/pip errors echo the failing --index-url verbatim.
# Mirrors the other installers. Verbose mode streams uncaptured, so it isn't redacted.
_redact_install_output() {
    sed -E \
        -e 's#(https?://)[^/@[:space:]`]+@#\1<redacted>@#g' \
        -e 's#([?&][^=[:space:]&`]+)=[^&#[:space:]`]+#\1=<redacted>#g' \
        -e 's|(https?://[^[:space:]`#]+)#[^[:space:]`]+|\1#<redacted>|g' \
        "$@"
}

run_install_cmd() {
    _label="$1"
    shift
    # Installer-pinned index installs (torch) must beat an inherited uv mirror (#6898):
    # for --default-index, neutralize the uv index/backend/config vars (UV_TORCH_BACKEND
    # redirects torch; UV_NO_CONFIG=1 + dropping UV_CONFIG_FILE stops a uv.toml/pyproject
    # index outranking the CLI pin, uv 0.10).
    case " $* " in
        *" --default-index "*) set -- env -u UV_DEFAULT_INDEX -u UV_INDEX_URL -u UV_INDEX -u UV_EXTRA_INDEX_URL -u UV_TORCH_BACKEND -u UV_FIND_LINKS -u UV_CONFIG_FILE UV_NO_CONFIG=1 "$@" ;;
    esac
    if _is_verbose; then
        # Stream through the redactor: uv echoes index URLs (credentials and
        # all) in its errors, and verbose mode previously bypassed the
        # redaction the quiet path applies. The rc file preserves the
        # command's exit code across the pipe without relying on pipefail
        # (this script runs under plain sh).
        _rcf=$(mktemp)
        { "$@" 2>&1; printf '%s' "$?" > "$_rcf"; } | _redact_install_output
        _rc=$(cat "$_rcf" 2>/dev/null || echo 1)
        rm -f "$_rcf"
        [ "${_rc:-1}" -eq 0 ] 2>/dev/null && return 0
        step "error" "$_label failed (exit code $_rc)" "$C_ERR" >&2
        return "$_rc"
    fi
    _log=$(mktemp)
    "$@" >"$_log" 2>&1 && { rm -f "$_log"; return 0; }
    _rc=$?
    step "error" "$_label failed (exit code $_rc)" "$C_ERR" >&2
    _redact_install_output "$_log" >&2
    rm -f "$_log"
    return $_rc
}

# Retry run_install_cmd on transient uv download failures with backoff. Returns
# the last exit code on permanent failure so the set -e rollback trap still fires.
: "${UNSLOTH_INSTALL_RETRIES:=3}"
: "${UNSLOTH_INSTALL_RETRY_DELAY:=3}"
run_install_cmd_retry() {
    _ricr_label="$1"
    # Sanitize overrides to a default of 3 (a typo must not disable retries; =1 disables).
    # Length guard precedes the numeric test so a huge value can't overflow `[ -ge ]`.
    # 0?* rejects leading-zero delays ("08"/"09" break the later $((delay*2)) as octal);
    # bare "0" stays valid. Bounds: 1..100 retries, 0..3600s base delay.
    case "$UNSLOTH_INSTALL_RETRIES" in
        ''|*[!0-9]*|0) _ricr_max=3 ;;
        *) if [ "${#UNSLOTH_INSTALL_RETRIES}" -le 3 ] && [ "$UNSLOTH_INSTALL_RETRIES" -ge 1 ] 2>/dev/null && [ "$UNSLOTH_INSTALL_RETRIES" -le 100 ] 2>/dev/null; then _ricr_max=$UNSLOTH_INSTALL_RETRIES; else _ricr_max=3; fi ;;
    esac
    case "$UNSLOTH_INSTALL_RETRY_DELAY" in
        ''|*[!0-9]*|0?*) _ricr_delay=3 ;;
        *) if [ "${#UNSLOTH_INSTALL_RETRY_DELAY}" -le 4 ] && [ "$UNSLOTH_INSTALL_RETRY_DELAY" -ge 0 ] 2>/dev/null && [ "$UNSLOTH_INSTALL_RETRY_DELAY" -le 3600 ] 2>/dev/null; then _ricr_delay=$UNSLOTH_INSTALL_RETRY_DELAY; else _ricr_delay=3; fi ;;
    esac
    _ricr_attempt=1
    while :; do
        # AND-OR (not `if`) preserves the real failure code: $? after a non-taken
        # `if` is 0 in sh/dash/bash, which would break the rollback path.
        run_install_cmd "$@" && return 0
        _ricr_rc=$?
        if [ "$_ricr_attempt" -ge "$_ricr_max" ]; then
            return "$_ricr_rc"
        fi
        substep "retrying \"$_ricr_label\" after transient failure (attempt $((_ricr_attempt + 1))/$_ricr_max, waiting ${_ricr_delay}s)..." "$C_WARN"
        sleep "$_ricr_delay" || true
        _ricr_attempt=$((_ricr_attempt + 1))
        _ricr_delay=$((_ricr_delay * 2))
    done
}

# Install bitsandbytes on AMD ROCm hosts. Uses the continuous-release_main
# wheel for the ROCm 4-bit GEMV fix (bnb PR #1887, post-0.49.2); bnb <= 0.49.2
# NaNs at decode shape on every AMD GPU. Falls back to PyPI >=0.49.1 if the
# pre-release URL is unreachable. Drop the pin once bnb 0.50+ ships on PyPI.
_install_bnb_rocm() {
    _label="$1"
    _venv_py="$2"
    case "$_ARCH" in
        x86_64|amd64)
            _bnb_whl_url="https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"
            ;;
        aarch64|arm64)
            _bnb_whl_url="https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl"
            ;;
        *)
            _bnb_whl_url=""
            ;;
    esac
    # uv rejects the continuous-release_main bitsandbytes wheel because the
    # filename version (1.33.7rc0) does not match the embedded metadata version
    # (0.50.0.dev0). pip accepts the mismatch, so bootstrap pip and use it.
    if ! "$_venv_py" -m pip --version >/dev/null 2>&1; then
        if ! run_maybe_quiet "$_venv_py" -m ensurepip --upgrade; then
            run_maybe_quiet uv pip install --python "$_venv_py" pip || \
                substep "[WARN] could not bootstrap pip; bitsandbytes install will likely fail" "$C_WARN"
        fi
    fi
    if [ -n "$_bnb_whl_url" ]; then
        substep "installing bitsandbytes for AMD ROCm (pre-release, PR #1887)..."
        _bnb_log=$(mktemp)
        if "$_venv_py" -m pip install \
            --disable-pip-version-check \
            --force-reinstall --no-cache-dir --no-deps \
            --retries 8 --timeout 90 \
            "$_bnb_whl_url" >"$_bnb_log" 2>&1; then
            rm -f "$_bnb_log"
            return 0
        fi
        _bnb_rc=$?
        if _is_verbose; then
            _redact_install_output "$_bnb_log" >&2
        fi
        rm -f "$_bnb_log"
        step "warning" "$_label (pre-release) failed (exit code $_bnb_rc)" "$C_WARN" >&2
        substep "[WARN] bnb pre-release install failed; falling back to PyPI (4-bit decode broken on ROCm)" "$C_WARN"
    fi
    run_install_cmd "$_label (pypi fallback)" "$_venv_py" -m pip install \
        --force-reinstall --no-cache-dir --no-deps "bitsandbytes>=0.49.1"
}

if [ "$_next_is_package" = true ]; then
    echo "❌ ERROR: --package requires an argument." >&2
    exit 1
fi
if [ "$_next_is_python" = true ]; then
    echo "❌ ERROR: --python requires a version argument (e.g. --python 3.12)." >&2
    exit 1
fi
if [ "$_next_is_llama_cpp_dir" = true ]; then
    echo "❌ ERROR: --with-llama-cpp-dir requires a path argument." >&2
    exit 1
fi

# Validate --package to prevent injection into shell/Python commands.
# Must start with a letter/digit (rejects leading dashes that uv would parse as flags).
case "$PACKAGE_NAME" in
    [!a-zA-Z0-9]*)
        echo "❌ ERROR: --package name must start with a letter or digit." >&2
        exit 1 ;;
    *[!a-zA-Z0-9._-]*)
        echo "❌ ERROR: --package name contains invalid characters (allowed: a-z A-Z 0-9 . _ -)" >&2
        exit 1 ;;
esac

# ── Tauri structured output ──
tauri_log() {
    if [ "$TAURI_MODE" = true ]; then
        echo "[TAURI:$1] $2"
    fi
}

tauri_diag_marker() {
    _diag_gpu_branch="${1:-unknown}"
    _diag_torch_index_family="${2:-none}"
    tauri_log "DIAG" "diag_schema=1 platform=${OS:-unknown} arch=${_ARCH:-unknown} python_version=${PYTHON_VERSION:-unknown} skip_torch=${SKIP_TORCH:-false} mac_intel=${MAC_INTEL:-false} gpu_branch=${_diag_gpu_branch} torch_index_family=${_diag_torch_index_family}"
}

_tauri_torch_index_family() {
    if [ "${SKIP_TORCH:-false}" = true ]; then
        echo "none"
        return
    fi
    _diag_url="${1:-}"
    # Strip query/fragment AND a trailing slash before classifying (like _torch_index_url_leaf):
    # a token isn't echoed into [TAURI:DIAG], and .../cu128/?token=x still classifies as cu128.
    _diag_url="${_diag_url%%\?*}"
    _diag_url="${_diag_url%%#*}"
    _diag_url="${_diag_url%/}"
    case "$_diag_url" in
        */cu118) echo "cu118" ;;
        */cu124) echo "cu124" ;;
        */cu126) echo "cu126" ;;
        */cu128) echo "cu128" ;;
        */cu130) echo "cu130" ;;
        */cpu) echo "cpu" ;;
        */rocm[0-9]*.[0-9]*)
            _diag_family=${_diag_url##*/}
            case "$_diag_family" in
                rocm[0-9]*.[0-9]*) echo "$_diag_family" ;;
                *) echo "auto" ;;
            esac ;;
        # AMD arch-specific index (e.g. repo.amd.com/rocm/whl/gfx1151/) --
        # used for Strix Halo/Point where torch 2.11+rocm7.13 has the real fix.
        *repo.amd.com/rocm/whl/gfx*|*rocm/whl/gfx*) echo "rocm7.13" ;;
        "") echo "none" ;;
        *) echo "auto" ;;
    esac
}

_tauri_gpu_branch() {
    _diag_family="${1:-unknown}"
    _diag_radeon="${2:-false}"
    if [ "${SKIP_TORCH:-false}" = true ]; then
        echo "no_torch"
        return
    fi
    if [ "${OS:-}" = "macos" ]; then
        echo "mac"
        return
    fi
    case "$_diag_family" in
        # Require a digit after cu so /current or /custom isn't branded CUDA (parity ^cu[0-9]).
        cu[0-9]*) echo "cuda" ;;
        rocm*)
            if [ "$_diag_radeon" = true ]; then
                echo "rocm_radeon"
            else
                echo "rocm"
            fi ;;
        radeon) echo "rocm_radeon" ;;
        cpu) echo "cpu" ;;
        none) echo "no_torch" ;;
        *) echo "unknown" ;;
    esac
}

PYTHON_VERSION=""  # resolved after platform detection

# Resolve install destinations: env override, HOME-redirect (best-effort
# via getent/dscl), or default. Env-var priority: UNSLOTH_STUDIO_HOME wins
# over STUDIO_HOME (the more specific signal beats the generic alias).
_resolve_studio_destinations() {
    _override_var=""
    _override="${UNSLOTH_STUDIO_HOME:-}"
    if [ -n "$_override" ]; then
        _override_var="UNSLOTH_STUDIO_HOME"
    else
        _override="${STUDIO_HOME:-}"
        [ -n "$_override" ] && _override_var="STUDIO_HOME"
    fi
    # Strip surrounding whitespace so " " is treated as unset (matches the
    # Python resolvers' .strip()), preventing install/runtime layout drift.
    _override=$(printf '%s' "$_override" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    # Tilde expansion: env vars are not subject to it when quoted on assignment.
    case "$_override" in
        "~") _override="$HOME" ;;
        "~/"*) _override="$HOME/${_override#'~/'}" ;;
    esac
    if [ -n "$_override" ]; then
        mkdir -p -- "$_override" 2>/dev/null || { echo "ERROR: $_override_var=$_override cannot be created." >&2; exit 1; }
        [ -w "$_override" ] || { echo "ERROR: $_override_var=$_override is not writable." >&2; exit 1; }
        STUDIO_HOME="$(CDPATH= cd -P -- "$_override" && pwd -P)" || exit 1
        DATA_DIR="$STUDIO_HOME/share"
        _LOCAL_BIN="$STUDIO_HOME/bin"
        _STUDIO_HOME_REDIRECT=env
        substep "custom $_override_var=$STUDIO_HOME"
        return 0
    fi
    _default_home=""
    if command -v getent >/dev/null 2>&1; then
        _default_home=$(getent passwd "${USER:-$(whoami)}" 2>/dev/null | cut -d: -f6)
    elif [ "$(uname)" = "Darwin" ] && command -v dscl >/dev/null 2>&1; then
        _default_home=$(dscl . -read "/Users/${USER:-$(whoami)}" NFSHomeDirectory 2>/dev/null | awk '{print $2}')
    fi
    # Canonicalize both sides so a trailing slash on $HOME (or symlink mismatch
    # with passwd-DB output) doesn't misfire the redirection branch.
    _home_canon="$HOME"
    if [ -d "$_home_canon" ]; then
        _home_canon=$(CDPATH= cd -P -- "$_home_canon" 2>/dev/null && pwd -P) || _home_canon="$HOME"
    fi
    _default_home_canon="$_default_home"
    if [ -n "$_default_home_canon" ] && [ -d "$_default_home_canon" ]; then
        _default_home_canon=$(CDPATH= cd -P -- "$_default_home_canon" 2>/dev/null && pwd -P) || _default_home_canon="$_default_home"
    fi
    if [ -n "$_default_home_canon" ] && [ "$_home_canon" != "$_default_home_canon" ]; then
        STUDIO_HOME="$HOME/.unsloth/studio"
        DATA_DIR="$HOME/.local/share/unsloth"
        _LOCAL_BIN="$HOME/.local/bin"
        _STUDIO_HOME_REDIRECT=home
        substep "HOME redirected ($HOME); install follows \$HOME"
        return 0
    fi
    STUDIO_HOME="$HOME/.unsloth/studio"
    DATA_DIR="$HOME/.local/share/unsloth"
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT=default
}
_resolve_studio_destinations
VENV_DIR="$STUDIO_HOME/unsloth_studio"
_VENV_ROLLBACK_DIR=""
_VENV_ROLLBACK_TARGET="$VENV_DIR"
_VENV_ROLLBACK_ACTIVE=false

_start_studio_venv_replacement() {
    _existing_dir="$1"
    _stamp=$(date +%Y%m%d%H%M%S 2>/dev/null || echo "time")
    _candidate="$STUDIO_HOME/unsloth_studio.rollback.$_stamp.$$"
    _suffix=0
    while [ -e "$_candidate" ]; do
        _suffix=$((_suffix + 1))
        _candidate="$STUDIO_HOME/unsloth_studio.rollback.$_stamp.$$.$_suffix"
    done
    mv "$_existing_dir" "$_candidate"
    _VENV_ROLLBACK_DIR="$_candidate"
    _VENV_ROLLBACK_TARGET="$_existing_dir"
    _VENV_ROLLBACK_ACTIVE=true
    substep "previous environment preserved for rollback"
}

_restore_studio_venv_replacement() {
    [ "$_VENV_ROLLBACK_ACTIVE" = true ] || return 0
    [ -n "$_VENV_ROLLBACK_DIR" ] && [ -d "$_VENV_ROLLBACK_DIR" ] || {
        _VENV_ROLLBACK_ACTIVE=false
        return 0
    }
    substep "restoring previous environment after failed install..." "$C_WARN"
    rm -rf "$_VENV_ROLLBACK_TARGET"
    if mv "$_VENV_ROLLBACK_DIR" "$_VENV_ROLLBACK_TARGET"; then
        substep "restored previous environment"
        _VENV_ROLLBACK_ACTIVE=false
        _VENV_ROLLBACK_DIR=""
    else
        echo "⚠️  Could not restore previous environment from $_VENV_ROLLBACK_DIR to $_VENV_ROLLBACK_TARGET" >&2
    fi
}

_commit_studio_venv_replacement() {
    [ "$_VENV_ROLLBACK_ACTIVE" = true ] || return 0
    if [ -n "$_VENV_ROLLBACK_DIR" ] && [ -d "$_VENV_ROLLBACK_DIR" ]; then
        rm -rf "$_VENV_ROLLBACK_DIR" || true
    fi
    _VENV_ROLLBACK_ACTIVE=false
    _VENV_ROLLBACK_DIR=""
}

_on_install_exit() {
    _status=$?
    if [ "$_status" -ne 0 ]; then
        _restore_studio_venv_replacement
    fi
    [ -n "${_UV_OVERRIDE_TMPDIR:-}" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
    [ -n "${_UNSLOTH_TORCH_OVERRIDES:-}" ] && rm -f "$_UNSLOTH_TORCH_OVERRIDES" 2>/dev/null || true
    exit "$_status"
}
# Empty so an inherited value never reaches the trap's rm; only temp paths this
# script creates below (spaced-path dir, torch-trio overrides) are removed.
_UV_OVERRIDE_TMPDIR=""
_UNSLOTH_TORCH_OVERRIDES=""
trap _on_install_exit EXIT

# ── Helper: download a URL to a file (supports curl and wget) ──
download() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$2" "$1"
    else
        echo "Error: neither curl nor wget found. Install one and re-run."
        exit 1
    fi
}

# ── Helper: check if a single package is available on the system ──
_is_pkg_installed() {
    case "$1" in
        build-essential) command -v gcc >/dev/null 2>&1 ;;
        libcurl4-openssl-dev)
            command -v dpkg >/dev/null 2>&1 && dpkg -s "$1" >/dev/null 2>&1 ;;
        pciutils)
            command -v lspci >/dev/null 2>&1 ;;
        *) command -v "$1" >/dev/null 2>&1 ;;
    esac
}

# ── Helper: install packages via apt, escalating to sudo only if needed ──
# Usage: _smart_apt_install pkg1 pkg2 pkg3 ...
_smart_apt_install() {
    _PKGS="$*"

    # Step 1: Try installing without sudo (works when already root)
    apt-get update -y </dev/null >/dev/null 2>&1 || true
    apt-get install -y $_PKGS </dev/null >/dev/null 2>&1 || true

    # Step 2: Check which packages are still missing
    _STILL_MISSING=""
    for _pkg in $_PKGS; do
        if ! _is_pkg_installed "$_pkg"; then
            _STILL_MISSING="$_STILL_MISSING $_pkg"
        fi
    done
    _STILL_MISSING=$(echo "$_STILL_MISSING" | sed 's/^ *//')

    if [ -z "$_STILL_MISSING" ]; then
        return 0
    fi

    # In Tauri mode, report needed packages and exit — Rust handles elevation
    if [ "$TAURI_MODE" = true ]; then
        tauri_log "NEED_SUDO" "$_STILL_MISSING"
        exit 2
    fi

    # Step 3: Escalate -- need elevated permissions for remaining packages
    if command -v sudo >/dev/null 2>&1; then
        echo ""
        echo "    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "    WARNING: We require sudo elevated permissions to install:"
        echo "    $_STILL_MISSING"
        echo "    If you accept, we'll run sudo now, and it'll prompt your password."
        echo "    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
        printf "    Accept? [Y/n] "
        if [ -r /dev/tty ]; then
            read -r REPLY </dev/tty || REPLY="y"
        else
            REPLY="y"
        fi
        case "$REPLY" in
            [nN]*)
                echo ""
                echo "    Please install these packages first, then re-run Unsloth Studio setup:"
                echo "    sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                exit 1
                ;;
            *)
                sudo apt-get update -y </dev/null
                sudo apt-get install -y $_STILL_MISSING </dev/null
                ;;
        esac
    else
        echo ""
        echo "    sudo is not available on this system."
        echo "    Please install these packages as root, then re-run Unsloth Studio setup:"
        echo "    apt-get update -y && apt-get install -y $_STILL_MISSING"
        exit 1
    fi
}

# ── Helper: create desktop shortcuts and launcher script ──
# Usage: create_studio_shortcuts <unsloth_exe> <os>
# Creates ~/.local/share/unsloth/launch-studio.sh (shared launcher),
# plus platform-specific shortcuts (Linux .desktop / macOS .app bundle /
# WSL Windows Desktop+Start Menu .lnk).
create_studio_shortcuts() {
    _css_exe="$1"
    _css_os="$2"

    # Validate exe
    if [ ! -x "$_css_exe" ]; then
        echo "[WARN] Cannot create shortcuts: unsloth not found at $_css_exe"
        return 0
    fi

    # Resolve absolute path
    _css_exe_dir=$(cd "$(dirname "$_css_exe")" && pwd)
    _css_exe="$_css_exe_dir/$(basename "$_css_exe")"

    _css_data_dir="$DATA_DIR"
    _css_launcher="$_css_data_dir/launch-studio.sh"
    _css_icon_png="$_css_data_dir/unsloth-studio.png"
    _css_gem_png="$_css_data_dir/unsloth-gem.png"

    mkdir -p "$_css_data_dir"

    # Same-install discriminator: per-install opaque id written once at install
    # time and read by both this launcher and the backend (/api/health). Replaces
    # the older sha256(canonical $STUDIO_HOME) scheme to (a) avoid leaking the
    # install path on -H 0.0.0.0 deployments and (b) sidestep launcher/backend
    # canonicalization drift (cd -P vs Path.resolve() symlink/junction handling).
    # Lives at $STUDIO_HOME/share/ (not $DATA_DIR) so the backend can find it
    # via _STUDIO_ROOT_RESOLVED / "share" / "studio_install_id" regardless of
    # mode (in env-mode $STUDIO_HOME/share == $DATA_DIR; in default mode they
    # diverge but the backend only knows the studio_root). 32 bytes of urandom
    # -> 64 hex chars, byte-compatible with the prior digest so launcher
    # placeholder, _check_health, and tests stay length-agnostic.
    _css_id_dir="$STUDIO_HOME/share"
    mkdir -p "$_css_id_dir"
    _css_id_file="$_css_id_dir/studio_install_id"
    if [ ! -s "$_css_id_file" ]; then
        if [ -r /dev/urandom ]; then
            _css_new_id=$(od -An -N32 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')
        fi
        if [ -z "${_css_new_id:-}" ] && command -v python3 >/dev/null 2>&1; then
            _css_new_id=$(python3 -c 'import secrets; print(secrets.token_hex(32))' 2>/dev/null)
        fi
        if [ -z "${_css_new_id:-}" ]; then
            echo "[WARN] Cannot create launcher: no entropy source for studio_install_id" >&2
            return 1
        fi
        # Atomic write so a partial install can't leave a half-written id.
        _css_id_tmp="$_css_id_file.$$.tmp"
        printf '%s' "$_css_new_id" > "$_css_id_tmp" \
            && mv "$_css_id_tmp" "$_css_id_file"
        chmod 600 "$_css_id_file" 2>/dev/null || true
        unset _css_new_id _css_id_tmp
    fi
    _css_studio_root_id=$(cat "$_css_id_file" 2>/dev/null)
    if [ -z "$_css_studio_root_id" ]; then
        echo "[WARN] Cannot create launcher: failed to read $_css_id_file" >&2
        return 1
    fi
    _css_is_env_mode=false
    [ "$_STUDIO_HOME_REDIRECT" = "env" ] && _css_is_env_mode=true

    # ── Write launcher script ──
    # Single-quoted heredoc; @@DATA_DIR@@, @@STUDIO_ROOT_ID@@, and
    # @@INSTALLED_IS_ENV_MODE@@ are substituted via sed below.
    cat > "$_css_launcher" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
# Unsloth Studio Launcher
# Auto-generated by install.sh -- do not edit manually.
set -euo pipefail

DATA_DIR='@@DATA_DIR@@'
_EXPECTED_STUDIO_ROOT_ID='@@STUDIO_ROOT_ID@@'
_INSTALLED_IS_ENV_MODE='@@INSTALLED_IS_ENV_MODE@@'

# Read exe path from config written at install time.
# Sourcing is safe: the config file is written by install.sh, not user input.
if [ -f "$DATA_DIR/studio.conf" ]; then
    . "$DATA_DIR/studio.conf"
fi
if [ -z "${UNSLOTH_EXE:-}" ] || [ ! -x "${UNSLOTH_EXE:-}" ]; then
    echo "Error: UNSLOTH_EXE not set or not executable. Re-run the installer." >&2
    exit 1
fi

BASE_PORT=8888
MAX_PORT_OFFSET=20
TIMEOUT_SEC=60
POLL_INTERVAL_SEC=0.25
LOG_FILE="$DATA_DIR/studio.log"
# why: in env-override mode multiple installs share an OS user; namespace the
# lock and remember our own healthy port so we never attach to an unrelated
# Unsloth listening on the global 8888..8908 range.
LOCK_DIR="${XDG_RUNTIME_DIR:-/tmp}/unsloth-studio-launcher-$(id -u).lock"
PORT_FILE=""
# why: gate on the install-time mode (baked above) instead of the runtime env
# var; sourcing a custom-root studio.conf in shell must not flip a default-mode
# launcher into env-mode behavior with stale state.
if [ "$_INSTALLED_IS_ENV_MODE" = "true" ]; then
    if command -v cksum >/dev/null 2>&1; then
        _LOCK_KEY=$(printf '%s' "$DATA_DIR" | cksum | awk '{print $1}')
    else
        _LOCK_KEY=""
    fi
    [ -n "$_LOCK_KEY" ] && LOCK_DIR="${XDG_RUNTIME_DIR:-/tmp}/unsloth-studio-launcher-$(id -u)-${_LOCK_KEY}.lock"
    PORT_FILE="$DATA_DIR/studio.port"
fi

# ── HTTP GET helper (supports curl and wget) ──
_http_get() {
    _url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsS --max-time 1 "$_url" 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- --timeout=1 "$_url" 2>/dev/null
    else
        return 1
    fi
}

# ── Health check ──
_check_health() {
    _port=$1
    _resp=$(_http_get "http://127.0.0.1:$_port/api/health") || return 1
    case "$_resp" in
        *'"status"'*'"healthy"'*'"service"'*'"Unsloth UI Backend"'*) ;;
        *'"service"'*'"Unsloth UI Backend"'*'"status"'*'"healthy"'*) ;;
        *) return 1 ;;
    esac
    # why: verify the backend belongs to THIS install. Baked hex digest avoids
    # JSON-escape mismatches on paths with `\`/`"` and avoids leaking the raw
    # install path to unauthenticated callers.
    if [ -n "$_EXPECTED_STUDIO_ROOT_ID" ]; then
        case "$_resp" in
            *"\"studio_root_id\":\"$_EXPECTED_STUDIO_ROOT_ID\""*|*"\"studio_root_id\": \"$_EXPECTED_STUDIO_ROOT_ID\""*) return 0 ;;
            *) return 1 ;;
        esac
    fi
    return 0
}

# ── Port scanning ──
_candidate_ports() {
    echo "$BASE_PORT"
    _max_port=$((BASE_PORT + MAX_PORT_OFFSET))
    if command -v ss >/dev/null 2>&1; then
        ss -tlnH 2>/dev/null | awk '{print $4}' | grep -oE '[0-9]+$' | \
            awk -v lo="$BASE_PORT" -v hi="$_max_port" '$1 >= lo && $1 <= hi && $1 != lo {print}' || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP -sTCP:LISTEN -nP 2>/dev/null | awk '{print $9}' | grep -oE '[0-9]+$' | \
            awk -v lo="$BASE_PORT" -v hi="$_max_port" '$1 >= lo && $1 <= hi && $1 != lo {print}' || true
    else
        _offset=1
        while [ "$_offset" -le "$MAX_PORT_OFFSET" ]; do
            echo $((BASE_PORT + _offset))
            _offset=$((_offset + 1))
        done
    fi
}

_find_healthy_port() {
    if [ -n "$PORT_FILE" ] && [ -f "$PORT_FILE" ]; then
        # why: env-mode installs only attach to a port we previously launched
        # ourselves; never to a sibling Unsloth that happens to be healthy.
        _p=$(cat "$PORT_FILE" 2>/dev/null || true)
        case "$_p" in
            ''|*[!0-9]*) ;;
            *)
                if _check_health "$_p"; then
                    echo "$_p"
                    return 0
                fi
                rm -f "$PORT_FILE"
                ;;
        esac
        return 1
    fi
    if [ -n "$PORT_FILE" ]; then
        return 1
    fi
    for _p in $(_candidate_ports | sort -un); do
        if _check_health "$_p"; then
            echo "$_p"
            return 0
        fi
    done
    return 1
}

# ── Check if a port is busy ──
_is_port_busy() {
    _port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -tlnH 2>/dev/null | awk '{print $4}' | grep -qE "[.:]$_port$"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$_port" -sTCP:LISTEN -nP >/dev/null 2>&1
    else
        return 1
    fi
}

# ── Find a free port in range ──
_find_launch_port() {
    _offset=0
    while [ "$_offset" -le "$MAX_PORT_OFFSET" ]; do
        _candidate=$((BASE_PORT + _offset))
        if ! _is_port_busy "$_candidate"; then
            echo "$_candidate"
            return 0
        fi
        _offset=$((_offset + 1))
    done
    return 1
}

# ── Open browser ──
_open_browser() {
    _url="$1"
    if [ "$(uname)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
        open "$_url"
    elif grep -qi microsoft /proc/version 2>/dev/null; then
        # WSL: xdg-open is unreliable; use Windows browser via PowerShell or cmd
        if command -v powershell.exe >/dev/null 2>&1; then
            powershell.exe -NoProfile -Command "Start-Process '$_url'" >/dev/null 2>&1 &
        elif command -v cmd.exe >/dev/null 2>&1; then
            cmd.exe /c start "" "$_url" >/dev/null 2>&1 &
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$_url" >/dev/null 2>&1 &
        else
            echo "Open in your browser: $_url" >&2
        fi
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$_url" >/dev/null 2>&1 &
    else
        echo "Open in your browser: $_url" >&2
    fi
}

# ── Spawn terminal with studio command ──
_spawn_terminal() {
    _cmd="$1"
    _os=$(uname)
    if [ "$_os" = "Darwin" ]; then
        # AppleEvents are TCC-denied from unsigned .app bundles; spawn
        # Terminal via a .command file + Launch Services instead. Server
        # is nohup'd so warm relaunches hit the fast-path; watcher + trap
        # in the .command couple Terminal close <-> server shutdown.
        # `exec` keeps the recorded PID equal to the studio process so
        # signals reach studio directly rather than a wrapper shell.
        nohup sh -c "exec $_cmd" >> "$LOG_FILE" 2>&1 &
        _server_pid=$!
        _pid_file="$DATA_DIR/studio-$_launch_port.pid"
        printf '%d\n' "$_server_pid" > "$_pid_file" 2>/dev/null || true

        _cmd_file="$DATA_DIR/launch-terminal.command"
        _logfile_q=$(printf '%s' "$LOG_FILE" | sed "s/'/'\\\\''/g")
        _pidfile_q=$(printf '%s' "$_pid_file" | sed "s/'/'\\\\''/g")
        if {
            {
                printf '#!/bin/bash\n'
                printf "SERVER_PID=%s\n" "$_server_pid"
                printf "PID_FILE='%s'\n" "$_pidfile_q"
                # Wait up to 12s for graceful shutdown before SIGKILL.
                printf 'shutdown_studio() {\n'
                printf '  kill -TERM "$SERVER_PID" 2>/dev/null\n'
                printf '  _i=0\n'
                printf '  while kill -0 "$SERVER_PID" 2>/dev/null && [ "$_i" -lt 24 ]; do\n'
                printf '    sleep 0.5\n'
                printf '    _i=$((_i + 1))\n'
                printf '  done\n'
                printf '  kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL "$SERVER_PID" 2>/dev/null\n'
                printf '  rm -f "$PID_FILE" 2>/dev/null\n'
                printf '}\n'
                printf "tail -n 100 -F '%s' &\n" "$_logfile_q"
                printf 'TAIL_PID=$!\n'
                # Server gone -> kill tail so bash exits cleanly.
                printf '(\n'
                printf '  while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 1; done\n'
                printf '  kill "$TAIL_PID" 2>/dev/null\n'
                printf ') &\n'
                printf 'WATCHER_PID=$!\n'
                printf "trap 'shutdown_studio; kill \"\$WATCHER_PID\" \"\$TAIL_PID\" 2>/dev/null; exit' HUP INT TERM\n"
                printf "trap 'rm -f \"\$PID_FILE\" 2>/dev/null' EXIT\n"
                printf 'wait "$TAIL_PID" 2>/dev/null\n'
            } > "$_cmd_file" 2>/dev/null \
                && chmod +x "$_cmd_file" 2>/dev/null \
                && open -a Terminal "$_cmd_file" 2>/dev/null
        }; then
            # Foreground Terminal (Launch Services spawns us backgrounded).
            osascript -e 'tell application "Terminal" to activate' >/dev/null 2>&1 || true
            return 0
        fi
        # .command/open failed: kill orphan, fall through to generic fallback.
        kill -TERM "$_server_pid" 2>/dev/null || true
        _i=0
        while kill -0 "$_server_pid" 2>/dev/null && [ "$_i" -lt 6 ]; do
            sleep 0.5
            _i=$((_i + 1))
        done
        kill -0 "$_server_pid" 2>/dev/null && kill -KILL "$_server_pid" 2>/dev/null || true
        rm -f "$_pid_file" 2>/dev/null || true
        echo "[WARN] Could not open Terminal; falling back to background launch" >&2
    else
        for _term in gnome-terminal konsole xfce4-terminal mate-terminal lxterminal xterm; do
            if command -v "$_term" >/dev/null 2>&1; then
                case "$_term" in
                    gnome-terminal) "$_term" -- sh -c "$_cmd" & return 0 ;;
                    konsole)        "$_term" -e sh -c "$_cmd" & return 0 ;;
                    xterm)          "$_term" -e sh -c "$_cmd" & return 0 ;;
                    *)              "$_term" -e sh -c "$_cmd" & return 0 ;;
                esac
            fi
        done
    fi
    # Fallback: background with log
    echo "No terminal emulator found; running in background. Logs: $LOG_FILE" >&2
    nohup sh -c "$_cmd" >> "$LOG_FILE" 2>&1 &
    return 0
}

# ── Atomic directory-based single-instance guard ──
_acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "$$" > "$LOCK_DIR/pid"
        return 0
    fi

    # Lock dir exists -- check if owner is still alive
    _old_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    if [ -n "$_old_pid" ] && kill -0 "$_old_pid" 2>/dev/null; then
        # Another launcher is running; wait for it to bring Unsloth up
        _deadline=$(($(date +%s) + TIMEOUT_SEC))
        while [ "$(date +%s)" -lt "$_deadline" ]; do
            _port=$(_find_healthy_port) && {
                _open_browser "http://localhost:$_port"
                exit 0
            }
            sleep "$POLL_INTERVAL_SEC"
        done
        echo "Timed out waiting for other launcher (PID $_old_pid)" >&2
        exit 0
    fi

    # Stale lock -- reclaim
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null || return 1
    echo "$$" > "$LOCK_DIR/pid"
}

_release_lock() {
    [ -d "$LOCK_DIR" ] || return 0
    [ "$(cat "$LOCK_DIR/pid" 2>/dev/null)" = "$$" ] || return 0
    rm -rf "$LOCK_DIR"
}

# ── Main ──
# Fast path: already healthy
_port=$(_find_healthy_port) && {
    _open_browser "http://localhost:$_port"
    exit 0
}

_acquire_lock
trap '_release_lock' EXIT INT TERM

# Post-lock re-check (handles race with another launcher)
_port=$(_find_healthy_port) && {
    _open_browser "http://localhost:$_port"
    exit 0
}

# Find a free port in range
_launch_port=$(_find_launch_port) || {
    echo "No free port found in range ${BASE_PORT}-$((BASE_PORT + MAX_PORT_OFFSET))" >&2
    exit 1
}

if [ -t 1 ]; then
    # ── Foreground mode (TTY available) ──
    # Background subshell: wait for studio to become healthy, release the
    # single-instance lock, then open the browser. The lock stays held until
    # health is confirmed so a second launcher cannot race during startup.
    (
        _obwr_deadline=$(($(date +%s) + TIMEOUT_SEC))
        while [ "$(date +%s)" -lt "$_obwr_deadline" ]; do
            if _check_health "$_launch_port"; then
                [ -n "$PORT_FILE" ] && printf '%s\n' "$_launch_port" > "$PORT_FILE" 2>/dev/null || true
                _release_lock
                _open_browser "http://localhost:$_launch_port"
                exit 0
            fi
            sleep "$POLL_INTERVAL_SEC"
        done
        # Timed out -- release the lock anyway so future launches are not blocked
        _release_lock
    ) &
    # Clear traps so exec does not trigger _release_lock (the subshell owns it)
    trap - EXIT INT TERM
    exec "$UNSLOTH_EXE" studio -p "$_launch_port"
else
    # ── Background mode (no TTY) ──
    # Used by macOS .app and headless invocations.
    _launch_cmd=$(printf '%q ' "$UNSLOTH_EXE" studio -p "$_launch_port")
    _launch_cmd=${_launch_cmd% }
    _spawn_terminal "$_launch_cmd"

    # Poll for health on the specific port we launched on
    _deadline=$(($(date +%s) + TIMEOUT_SEC))
    while [ "$(date +%s)" -lt "$_deadline" ]; do
        if _check_health "$_launch_port"; then
            [ -n "$PORT_FILE" ] && printf '%s\n' "$_launch_port" > "$PORT_FILE" 2>/dev/null || true
            _open_browser "http://localhost:$_launch_port"
            exit 0
        fi
        sleep "$POLL_INTERVAL_SEC"
    done

    echo "Unsloth Studio did not become healthy within ${TIMEOUT_SEC}s." >&2
    echo "Check logs at: $LOG_FILE" >&2
    exit 1
fi
LAUNCHER_EOF

    # why: bake non-user-controlled placeholders FIRST so a literal
    # `@@STUDIO_ROOT_ID@@` inside $DATA_DIR cannot be rewritten below.
    sed -e "s|@@STUDIO_ROOT_ID@@|$_css_studio_root_id|g" \
        -e "s|@@INSTALLED_IS_ENV_MODE@@|$_css_is_env_mode|g" \
        "$_css_launcher" > "$_css_launcher.tmp" \
        && mv "$_css_launcher.tmp" "$_css_launcher"

    # Env-mode bakes an absolute DATA_DIR (root fixed at install time);
    # default / HOME-redirect keeps the literal $HOME/.local/share/unsloth
    # so behavior is byte-identical to pre-override.
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
        # Two-stage escape: (1) `'` -> `'\''` for shell single-quote embedding,
        # (2) backslash/&/| escape so the value survives the s|...|VALUE| sed
        # below. Verified end-to-end with apostrophes, spaces, &, |, $.
        _sq_escaped=$(printf '%s' "$DATA_DIR" | sed "s/'/'\\\\''/g")
        _sed_safe=$(printf '%s' "$_sq_escaped" | sed 's/[\\&|]/\\&/g')
        sed "s|@@DATA_DIR@@|$_sed_safe|g" "$_css_launcher" > "$_css_launcher.tmp" \
            && mv "$_css_launcher.tmp" "$_css_launcher"
    else
        sed "s|DATA_DIR='@@DATA_DIR@@'|DATA_DIR=\"\$HOME/.local/share/unsloth\"|" \
            "$_css_launcher" > "$_css_launcher.tmp" \
            && mv "$_css_launcher.tmp" "$_css_launcher"
    fi

    chmod +x "$_css_launcher"

    # studio.conf: exe path + (env-mode only) persisted env vars so fresh
    # shells launch the right install without re-exporting.
    _css_quoted_exe=$(printf '%s' "$_css_exe" | sed "s/'/'\\\\''/g")
    {
        printf '%s\n' "UNSLOTH_EXE='$_css_quoted_exe'"
        if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
            # When an override resolves to the legacy default, llama.cpp
            # still lives at ~/.unsloth/llama.cpp (one shared build).
            # Canonicalize the legacy side so a symlinked $HOME doesn't
            # break the comparison.
            _css_legacy_studio="$HOME/.unsloth/studio"
            if [ -d "$_css_legacy_studio" ]; then
                _css_legacy_studio=$(CDPATH= cd -P -- "$_css_legacy_studio" 2>/dev/null && pwd -P) \
                    || _css_legacy_studio="$HOME/.unsloth/studio"
            fi
            if [ "$STUDIO_HOME" = "$_css_legacy_studio" ]; then
                _css_llama_path="$HOME/.unsloth/llama.cpp"
            else
                _css_llama_path="$STUDIO_HOME/llama.cpp"
            fi
            _css_quoted_home=$(printf '%s' "$STUDIO_HOME" | sed "s/'/'\\\\''/g")
            _css_quoted_llama=$(printf '%s' "$_css_llama_path" | sed "s/'/'\\\\''/g")
            printf '%s\n' "export UNSLOTH_STUDIO_HOME='$_css_quoted_home'"
            # UNSLOTH_LLAMA_CPP_PATH is a pre-existing user-controlled
            # llama.cpp dir override; only default it if unset.
            printf '%s\n' 'if [ -z "${UNSLOTH_LLAMA_CPP_PATH:-}" ]; then'
            printf '%s\n' "    export UNSLOTH_LLAMA_CPP_PATH='$_css_quoted_llama'"
            printf '%s\n' 'fi'
        fi
    } > "$_css_data_dir/studio.conf"

    # ── Icon: try bundled, then download ──
    # rounded-512.png used for both Linux and macOS icons
    _css_script_dir=""
    if [ -n "${0:-}" ] && [ -f "$0" ]; then
        _css_script_dir=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || true
    fi

    # Try to find rounded-512.png from installed package (site-packages) or local repo
    _css_found_icon=""
    _css_venv_dir=$(dirname "$(dirname "$_css_exe")")
    # Check site-packages
    for _sp in "$_css_venv_dir"/lib/python*/site-packages/unsloth/studio/frontend/public; do
        if [ -f "$_sp/rounded-512.png" ]; then
            _css_found_icon="$_sp/rounded-512.png"
        fi
    done
    # Check local repo (when running from clone)
    if [ -z "$_css_found_icon" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/frontend/public/rounded-512.png" ]; then
        _css_found_icon="$_css_script_dir/studio/frontend/public/rounded-512.png"
    fi

    # Copy or download rounded-512.png (used for both Linux icon and macOS icns)
    if [ -n "$_css_found_icon" ]; then
        cp "$_css_found_icon" "$_css_icon_png" 2>/dev/null || true
        cp "$_css_found_icon" "$_css_gem_png" 2>/dev/null || true
    else
        download "https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/rounded-512.png" "$_css_icon_png" 2>/dev/null || true
        cp "$_css_icon_png" "$_css_gem_png" 2>/dev/null || true
    fi

    # Validate PNG header (first 4 bytes: \x89PNG)
    _css_validate_png() {
        [ -f "$1" ] || return 1
        _hdr=$(od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' ')
        [ "$_hdr" = "89504e47" ]
    }
    if [ -f "$_css_icon_png" ] && ! _css_validate_png "$_css_icon_png"; then
        rm -f "$_css_icon_png"
    fi
    if [ -f "$_css_gem_png" ] && ! _css_validate_png "$_css_gem_png"; then
        rm -f "$_css_gem_png"
    fi

    # ── Platform-specific shortcuts ──
    # Env-mode installs are workspace-scoped: skip persistent desktop /
    # Start-Menu / dock launchers that may point at a deleted workspace.
    # Runtime launcher + studio.conf + icon are still written above.
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
        substep "wrote launcher at $_css_launcher (persistent shortcuts skipped in env-override mode)"
        return 0
    fi

    _css_created=0

    if [ "$_css_os" = "linux" ]; then
        # ── Linux: .desktop file ──
        _css_app_dir="$HOME/.local/share/applications"
        mkdir -p "$_css_app_dir"

        _css_desktop="$_css_app_dir/unsloth-studio.desktop"
        # Escape backslashes and double-quotes for .desktop Exec= field
        _css_exec_escaped=$(printf '%s' "$_css_launcher" | sed 's/\\/\\\\/g; s/"/\\"/g')
        _css_icon_escaped=$(printf '%s' "$_css_icon_png" | sed 's/\\/\\\\/g; s/"/\\"/g')
        cat > "$_css_desktop" << DESKTOP_EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Unsloth Studio
Comment=Launch Unsloth Studio
Exec="$_css_exec_escaped"
Icon=$_css_icon_escaped
Terminal=true
StartupNotify=true
Categories=Development;Science;
DESKTOP_EOF
        chmod +x "$_css_desktop"

        # Copy to ~/Desktop if it exists
        if [ -d "$HOME/Desktop" ]; then
            cp "$_css_desktop" "$HOME/Desktop/unsloth-studio.desktop" 2>/dev/null || true
            chmod +x "$HOME/Desktop/unsloth-studio.desktop" 2>/dev/null || true
            # Mark as trusted so GNOME/Nautilus allows launching via double-click
            if command -v gio >/dev/null 2>&1; then
                gio set "$HOME/Desktop/unsloth-studio.desktop" metadata::trusted true 2>/dev/null || true
            fi
        fi

        # Best-effort update database
        update-desktop-database "$_css_app_dir" 2>/dev/null || true
        _css_created=1

    elif [ "$_css_os" = "macos" ]; then
        # ── macOS: .app bundle ──
        _css_app="$HOME/Applications/Unsloth Studio.app"
        _css_contents="$_css_app/Contents"
        _css_macos_dir="$_css_contents/MacOS"
        _css_res_dir="$_css_contents/Resources"
        # Recreate bundle if root or any subpath is a symlink (mkdir -p follows them).
        if [ -L "$_css_app" ] || [ -L "$_css_contents" ] \
            || [ -L "$_css_macos_dir" ] || [ -L "$_css_res_dir" ]; then
            rm -rf "$_css_app" 2>/dev/null || {
                echo "[ERROR] $_css_app contains a symlinked bundle path; remove manually and re-run install" >&2
                return 1
            }
        elif [ -e "$_css_app" ] && [ ! -d "$_css_app" ]; then
            echo "[ERROR] $_css_app exists but is not a directory; remove manually and re-run install" >&2
            return 1
        fi
        mkdir -p "$_css_macos_dir" "$_css_res_dir"

        # Info.plist
        cat > "$_css_contents/Info.plist" << 'PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>ai.unsloth.studio</string>
    <key>CFBundleName</key>
    <string>Unsloth Studio</string>
    <key>CFBundleDisplayName</key>
    <string>Unsloth Studio</string>
    <key>CFBundleExecutable</key>
    <string>launch-studio</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST_EOF

        # Executable stub: same single-quoted-heredoc + sed-substitute
        # pattern as launch-studio.sh so $-vars in $_css_data_dir don't
        # expand at .app launch time.
        _css_sq_dir=$(printf '%s' "$_css_data_dir" | sed "s/'/'\\\\''/g")
        _css_sed_dir=$(printf '%s' "$_css_sq_dir" | sed 's/[\\&|]/\\&/g')
        cat > "$_css_macos_dir/launch-studio" << 'STUB_EOF'
#!/bin/sh
exec '@@DATA_DIR@@/launch-studio.sh' "$@"
STUB_EOF
        sed "s|@@DATA_DIR@@|$_css_sed_dir|g" "$_css_macos_dir/launch-studio" \
            > "$_css_macos_dir/launch-studio.tmp" \
            && mv "$_css_macos_dir/launch-studio.tmp" "$_css_macos_dir/launch-studio"
        chmod +x "$_css_macos_dir/launch-studio"

        # Build AppIcon.icns from unsloth-gem.png (2240x2240)
        if [ -f "$_css_gem_png" ] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
            _css_tmpdir=$(mktemp -d 2>/dev/null)
            if [ -d "$_css_tmpdir" ]; then
                _css_iconset="$_css_tmpdir/AppIcon.iconset"
                mkdir -p "$_css_iconset"
                _css_icon_ok=true
                for _sz in 16 32 128 256 512; do
                    _sz2=$((_sz * 2))
                    sips -z "$_sz" "$_sz" "$_css_gem_png" --out "$_css_iconset/icon_${_sz}x${_sz}.png" >/dev/null 2>&1 || _css_icon_ok=false
                    sips -z "$_sz2" "$_sz2" "$_css_gem_png" --out "$_css_iconset/icon_${_sz}x${_sz}@2x.png" >/dev/null 2>&1 || _css_icon_ok=false
                done
                if [ "$_css_icon_ok" = "true" ]; then
                    iconutil -c icns "$_css_iconset" -o "$_css_res_dir/AppIcon.icns" 2>/dev/null || true
                fi
                rm -rf "$_css_tmpdir"
            fi
        fi
        # Fallback: copy PNG as icon
        if [ ! -f "$_css_res_dir/AppIcon.icns" ] && [ -f "$_css_icon_png" ]; then
            cp "$_css_icon_png" "$_css_res_dir/AppIcon.icns" 2>/dev/null || true
        fi

        # Touch so Finder indexes it
        touch "$_css_app"

        # Symlink on Desktop
        if [ -d "$HOME/Desktop" ]; then
            ln -sf "$_css_app" "$HOME/Desktop/Unsloth Studio" 2>/dev/null || true
        fi
        _css_created=1

    elif [ "$_css_os" = "wsl" ]; then
        # ── WSL: create Windows Desktop and Start Menu shortcuts ──
        # Detect current WSL distro for targeted shortcut
        _css_distro="${WSL_DISTRO_NAME:-}"

        # Build the wsl.exe arguments.
        # Double-quote distro name and launcher path for Windows command line
        # parsing so values with spaces (e.g. "Ubuntu Preview") are kept as
        # single arguments.
        _css_wsl_args=""
        if [ -n "$_css_distro" ]; then
            _css_wsl_args="-d \"$_css_distro\" "
        fi
        _css_wsl_args="${_css_wsl_args}-- bash -l -c \"exec \\\"$_css_launcher\\\"\""

        # Detect whether Windows Terminal (wt.exe) is available (better UX)
        _css_use_wt=false
        if command -v wt.exe >/dev/null 2>&1; then
            _css_use_wt=true
        fi

        if [ "$_css_use_wt" = true ]; then
            _css_sc_target='wt.exe'
            _css_sc_args="wsl.exe $_css_wsl_args"
        else
            _css_sc_target='wsl.exe'
            _css_sc_args="$_css_wsl_args"
        fi

        # Escape single quotes for PowerShell single-quoted string embedding
        _css_sc_args_ps=$(printf '%s' "$_css_sc_args" | sed "s/'/''/g")

        # DISTINCT shortcut name so the WSL launcher never clobbers a native
        # install's "Unsloth Studio.lnk" in the same folder. Per-distro suffix.
        if [ -n "$_css_distro" ]; then
            _css_lnk_name="Unsloth Studio (WSL - ${_css_distro}).lnk"
        else
            _css_lnk_name="Unsloth Studio (WSL).lnk"
        fi
        _css_lnk_name_ps=$(printf '%s' "$_css_lnk_name" | sed "s/'/''/g")

        # Create shortcuts via a temp PowerShell script to avoid escaping issues
        _css_ps1_tmp=$(mktemp /tmp/unsloth-shortcut-XXXXXX.ps1 2>/dev/null) || true
        if [ -n "$_css_ps1_tmp" ]; then
            cat > "$_css_ps1_tmp" << WSLPS1_EOF
\$WshShell = New-Object -ComObject WScript.Shell
\$targetExe = (Get-Command '$_css_sc_target' -ErrorAction SilentlyContinue).Source
if (-not \$targetExe) { exit 1 }
# Best-effort: fetch the Unsloth icon to a stable Windows path (shared with a
# native install if one exists) so the WSL shortcut shows the proper icon.
\$iconDir = Join-Path \$env:LOCALAPPDATA 'Unsloth Studio'
\$iconPath = Join-Path \$iconDir 'unsloth.ico'
\$preIconHash = \$null
if (Test-Path -LiteralPath \$iconPath) {
    try { \$preIconHash = (Get-FileHash -LiteralPath \$iconPath -Algorithm SHA256).Hash } catch {}
}
if (-not (Test-Path -LiteralPath \$iconPath)) {
    try {
        New-Item -ItemType Directory -Force -Path \$iconDir | Out-Null
        Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/unsloth.ico' -OutFile \$iconPath -UseBasicParsing -ErrorAction Stop
    } catch {}
}
\$hasIcon = \$false
if (Test-Path -LiteralPath \$iconPath) {
    try { \$b = [System.IO.File]::ReadAllBytes(\$iconPath); if (\$b.Length -ge 4 -and \$b[0] -eq 0 -and \$b[1] -eq 0 -and \$b[2] -eq 1 -and \$b[3] -eq 0) { \$hasIcon = \$true } } catch {}
}
\$locations = @(
    [Environment]::GetFolderPath('Desktop'),
    (Join-Path \$env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
)
\$created = @()
\$firstShortcut = \$false
foreach (\$dir in \$locations) {
    if (-not \$dir -or -not (Test-Path \$dir)) { continue }
    \$linkPath = Join-Path \$dir '$_css_lnk_name_ps'
    if (-not (Test-Path -LiteralPath \$linkPath)) { \$firstShortcut = \$true }
    \$shortcut = \$WshShell.CreateShortcut(\$linkPath)
    \$shortcut.TargetPath = \$targetExe
    \$shortcut.Arguments = '$_css_sc_args_ps'
    \$shortcut.Description = 'Launch Unsloth Studio (WSL)'
    if (\$hasIcon) { \$shortcut.IconLocation = "\$iconPath,0" }
    \$shortcut.Save()
    \$created += \$linkPath
}
\$iconChanged = \$false
if (\$hasIcon) {
    if (-not \$preIconHash) {
        \$iconChanged = \$true
    } else {
        try {
            \$postIconHash = (Get-FileHash -LiteralPath \$iconPath -Algorithm SHA256).Hash
            \$iconChanged = (\$postIconHash -ne \$preIconHash)
        } catch { \$iconChanged = \$true }
    }
} elseif (\$preIconHash) {
    \$iconChanged = \$true
}
# Per-item refresh always (cheap, non-disruptive) so the rewritten .lnk renders
# immediately instead of a stale/blank (generic) icon. The reliable fix (no
# explorer restart) is a PER-ITEM SHChangeNotify(SHCNE_UPDATEITEM, SHCNF_PATHW,
# <lnk>) -- the global SHCNE_ASSOCCHANGED alone does not recover a stale item.
try {
    Add-Type -Namespace UnslothShell -Name IconRefresh -MemberDefinition '[System.Runtime.InteropServices.DllImport("shell32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)] public static extern void SHChangeNotify(int e, uint f, string a, System.IntPtr b);' -ErrorAction SilentlyContinue
    foreach (\$p in \$created) { try { [UnslothShell.IconRefresh]::SHChangeNotify(0x00002000, 0x0005, \$p, [System.IntPtr]::Zero) } catch {} }
    [UnslothShell.IconRefresh]::SHChangeNotify(0x08000000, 0, \$null, [System.IntPtr]::Zero)
} catch {}
# Heavier on-disk icon-cache clear + StartMenuExperienceHost tile rebuild
# (preserve start2.bin) only on first install or a real icon change, so a no-op
# WSL reinstall does not run a dropper-like clear-cache + kill cluster each time.
if (\$created.Count -gt 0 -and (\$firstShortcut -or \$iconChanged)) {
    try { & "\$env:SystemRoot\System32\ie4uinit.exe" -ClearIconCache } catch {}
    try { & "\$env:SystemRoot\System32\ie4uinit.exe" -show } catch {}
    try {
        \$smeh = Join-Path \$env:LOCALAPPDATA 'Packages\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\TempState'
        if (Test-Path -LiteralPath \$smeh) {
            Get-ChildItem -LiteralPath \$smeh -Filter 'TileCache_*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath (Join-Path \$smeh 'StartUnifiedTileModelCache.dat') -Force -ErrorAction SilentlyContinue
            Stop-Process -Name StartMenuExperienceHost -Force -ErrorAction SilentlyContinue
        }
    } catch {}
}
WSLPS1_EOF

            # Convert WSL path to Windows path for powershell.exe
            _css_ps1_win=$(wslpath -w "$_css_ps1_tmp" 2>/dev/null)
            if [ -n "$_css_ps1_win" ]; then
                powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$_css_ps1_win" >/dev/null 2>&1 && _css_created=1
            fi
            rm -f "$_css_ps1_tmp"
        fi
        # If WSL interop is disabled (powershell.exe "Exec format error"), the
        # shortcut wasn't created; tell the user how to launch / re-enable it.
        if [ "$_css_created" -ne 1 ]; then
            substep "Couldn't create the Windows shortcut (WSL interop may be disabled)." "$C_WARN"
            substep "  Launch Unsloth from Windows:  wsl -d \"$_css_distro\" -- bash -lc 'unsloth studio'" "$C_WARN"
            substep "  (re-enable shortcuts: turn WSL interop back on, e.g. run 'wsl --shutdown' then reopen WSL.)" "$C_WARN"
        fi
    fi

    if [ "$_css_created" -eq 1 ]; then
        substep "Created Unsloth Studio shortcut"
    fi
}

echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "🦥 Unsloth Studio Installer"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""

# ── Detect platform ──
tauri_log "STEP" "Detecting platform"
OS="linux"
if [ "$(uname)" = "Darwin" ]; then
    OS="macos"
elif grep -qi microsoft /proc/version 2>/dev/null; then
    OS="wsl"
fi
step "platform" "$OS"

# Regen launcher/shortcuts only; used by `unsloth studio update`.
if [ "$_SHORTCUTS_ONLY" = true ]; then
    # Tauri owns its own shortcuts.
    if [ "$TAURI_MODE" != true ]; then
        VENV_ABS_BIN="$VENV_DIR/bin"
        if [ ! -x "$VENV_ABS_BIN/unsloth" ]; then
            echo "ERROR: unsloth binary missing at '$VENV_ABS_BIN/unsloth'; run install.sh first." >&2
            exit 1
        fi
        create_studio_shortcuts "$VENV_ABS_BIN/unsloth" "$OS"
    fi
    exit 0
fi

# ── Architecture detection & Python version ──
_ARCH=$(uname -m)
MAC_INTEL=false
if [ "$OS" = "macos" ] && [ "$_ARCH" = "x86_64" ]; then
    # Guard against Apple Silicon running under Rosetta (reports x86_64).
    # sysctl hw.optional.arm64 returns "1" on Apple Silicon even in Rosetta.
    if [ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
        echo ""
        echo "  WARNING: Apple Silicon detected, but this shell is running under Rosetta (x86_64)."
        echo "  Re-run install.sh from a native arm64 terminal for full PyTorch support."
        echo "  Continuing in GGUF-only mode for now."
        echo ""
    fi
    MAC_INTEL=true
fi

if [ -n "$_USER_PYTHON" ]; then
    PYTHON_VERSION="$_USER_PYTHON"
    echo "  Using user-specified Python $PYTHON_VERSION (--python override)"
elif [ "$MAC_INTEL" = true ]; then
    PYTHON_VERSION="3.12"
else
    PYTHON_VERSION="3.13"
fi

if [ "$MAC_INTEL" = true ]; then
    echo ""
    echo "  NOTE: Intel Mac (x86_64) detected."
    echo "  PyTorch is unavailable for this platform (dropped Jan 2024)."
    echo "  Unsloth will install in GGUF-only mode."
    echo "  Chat, inference via GGUF, and data recipes will work."
    echo "  Training requires Apple Silicon or Linux with GPU."
    echo ""
fi

# ── Unified SKIP_TORCH: --no-torch flag OR Intel Mac auto-detection ──
SKIP_TORCH=false
if [ "$_NO_TORCH_FLAG" = true ] || [ "$MAC_INTEL" = true ]; then
    SKIP_TORCH=true
fi

# Apple Silicon: exclude broken mlx-lm 0.31.3 (QK-norm load regression for
# gemma4 / qwen3_5; mlx-lm #1242). A curl-piped install has no overrides file
# and skips the guarded MLX step (SKIP_STUDIO_BASE=1), so this is the only cover.
_MLX_LM_EXCLUDE_ARG=""

# Apple Silicon: override mlx-vlm / mlx-lm's transformers pin (see overrides file).
if [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _MLX_LM_EXCLUDE_ARG="mlx-lm!=0.31.3"
    _OVERRIDES_FILE="$(cd "$(dirname "$0" 2>/dev/null || echo ".")" && pwd)/studio/backend/requirements/single-env/overrides-darwin-arm64.txt"
    if [ -f "$_OVERRIDES_FILE" ]; then
        # uv splits UV_OVERRIDE on whitespace, so a repo path with whitespace
        # truncates it and aborts every later uv call (issue #6503). Hand uv a copy.
        case "$_OVERRIDES_FILE" in
            *[[:space:]]*)
                _UV_OVERRIDE_TMPDIR=$(mktemp -d 2>/dev/null) || _UV_OVERRIDE_TMPDIR=""
                case "$_UV_OVERRIDE_TMPDIR" in
                    "") ;;
                    *[[:space:]]*) rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true; _UV_OVERRIDE_TMPDIR="" ;;
                    *)
                        if cp "$_OVERRIDES_FILE" "$_UV_OVERRIDE_TMPDIR/overrides-darwin-arm64.txt" 2>/dev/null; then
                            _OVERRIDES_FILE="$_UV_OVERRIDE_TMPDIR/overrides-darwin-arm64.txt"
                        else
                            rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
                            _UV_OVERRIDE_TMPDIR=""
                        fi
                        ;;
                esac
                ;;
        esac
        export UV_OVERRIDE="$_OVERRIDES_FILE"
    fi
fi

_TAURI_INITIAL_GPU_BRANCH="unknown"
if [ "$SKIP_TORCH" = true ]; then
    _TAURI_INITIAL_GPU_BRANCH="no_torch"
elif [ "$OS" = "macos" ]; then
    _TAURI_INITIAL_GPU_BRANCH="mac"
fi
tauri_diag_marker "$_TAURI_INITIAL_GPU_BRANCH" "none"

# AMD GPU name from the Windows host via WMI, or empty. Discrete cards aren't in
# /proc/cpuinfo, so ask Windows. Cached ("-" = negative), self-contained, bounded
# to 10s. Defined here so the reroute below can use it before _run_bounded exists.
_WSL_AMD_GPU_NAME_CACHE=""
_wsl_amd_gpu_name() {
    if [ -n "$_WSL_AMD_GPU_NAME_CACHE" ]; then
        [ "$_WSL_AMD_GPU_NAME_CACHE" = "-" ] && return 1
        printf '%s' "$_WSL_AMD_GPU_NAME_CACHE"; return 0
    fi
    command -v powershell.exe >/dev/null 2>&1 || { _WSL_AMD_GPU_NAME_CACHE="-"; return 1; }
    _wag_ps="(Get-CimInstance Win32_VideoController | Where-Object { \$_.Name -match 'AMD|Radeon' } | Select-Object -First 1).Name"
    if command -v timeout >/dev/null 2>&1; then
        _wag_n="$(timeout 10 powershell.exe -NoProfile -Command "$_wag_ps" 2>/dev/null | tr -d '\r\n\000')"
    else
        _wag_n="$(powershell.exe -NoProfile -Command "$_wag_ps" 2>/dev/null | tr -d '\r\n\000')"
    fi
    if [ -n "$_wag_n" ]; then _WSL_AMD_GPU_NAME_CACHE="$_wag_n"; printf '%s' "$_wag_n"; return 0; fi
    _WSL_AMD_GPU_NAME_CACHE="-"; return 1
}

# ── Bounded command runner ──
# Runs a command under a 10s timeout when the `timeout` binary is available,
# otherwise runs it unbounded. Keeps a wedged nvidia-smi (blocking during
# driver init or after a reset) from hanging the installer: a timed-out probe
# exits nonzero and is treated exactly like a failed probe. No-op semantics on
# hosts without `timeout` (e.g. macOS) or when the probe is healthy.
_run_bounded() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 10 "$@"
    else
        "$@"
    fi
}

# Returns 0 (true) when CUDA_VISIBLE_DEVICES is set to "" or "-1", i.e. every
# NVIDIA device is deliberately hidden (mixed AMD+NVIDIA hosts steering work to
# the AMD card). Unset means all devices visible. nvidia-smi ignores this env
# var, so the probes below cannot see the distinction on their own.
_cvd_hides_nvidia() {
    [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] || return 1
    _cvd_trim=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]')
    [ -z "$_cvd_trim" ] || [ "$_cvd_trim" = "-1" ]
}

# ── NVIDIA usable-GPU helper ──
# Returns 0 (true) if an NVIDIA GPU is present and usable.
# Primary probe: nvidia-smi -L. Fallback: /proc/driver/nvidia/gpus/ sysfs,
# which the NVIDIA driver populates on Linux regardless of nvidia-smi state
# -- handles PATH gaps, subprocess timeouts, and driver init races that
# could otherwise cause nvidia-smi to fail and silence NVIDIA detection.
# A GPU hidden via CUDA_VISIBLE_DEVICES=""/-1 counts as NOT usable (matches
# install_llama_prebuilt.py has_usable_nvidia), so AMD/CPU routing still runs.
_has_usable_nvidia_gpu() {
    if _cvd_hides_nvidia; then
        return 1
    fi
    _nvsmi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        _nvsmi="nvidia-smi"
    elif [ -x "/usr/bin/nvidia-smi" ]; then
        _nvsmi="/usr/bin/nvidia-smi"
    fi
    if [ -n "$_nvsmi" ]; then
        if _run_bounded "$_nvsmi" -L 2>/dev/null | awk '/^GPU[[:space:]]+[0-9]+:/{found=1} END{exit !found}'; then
            return 0
        fi
    fi
    # Fallback: NVIDIA driver exposes one subdir per GPU under this path.
    if [ -d /proc/driver/nvidia/gpus ] && \
       [ -n "$(ls -A /proc/driver/nvidia/gpus 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

# Strix Halo ROCm-on-WSL only targets Ubuntu 24.04. On a newer distro (e.g. 26.04)
# with a 24.04 distro present, re-run the install there and stop; else fall through
# to CPU + the `wsl --install` hint below (never auto-create a distro). Runs before
# the STUDIO_HOME mkdir/venv so the origin distro is untouched.
_maybe_reroute_strixhalo_to_2404() {
    [ "${OS:-}" = "wsl" ] || return 0
    # An explicit index pin skips every GPU-driven reroute (same contract as
    # the later Radeon/Strix guard): the pin is honored in THIS distro rather
    # than probing the GPU and switching distributions. Whitespace-only
    # overrides do not gate (parity with get_torch_index_url).
    _rr_pin=$(printf '%s' "${UNSLOTH_TORCH_INDEX_URL:-}${UNSLOTH_TORCH_INDEX_FAMILY:-}" | tr -d '[:space:]')
    [ -n "$_rr_pin" ] && return 0
    [ "${SKIP_TORCH:-false}" = "false" ] || return 0
    [ "${UNSLOTH_SKIP_ROCM_WSL_SETUP:-0}" = "1" ] && return 0
    [ "${UNSLOTH_WSL_REROUTED:-0}" = "1" ] && return 0
    [ -e /dev/dxg ] || return 0
    # A usable NVIDIA GPU (common on hybrid AMD+NVIDIA hosts) means the CUDA path works on
    # this distro, so don't reroute for AMD. _has_usable_nvidia_gpu (moved above) honors
    # CUDA_VISIBLE_DEVICES=""/-1 and the /proc/driver/nvidia fallback for PATH/timeout gaps.
    if _has_usable_nvidia_gpu; then return 0; fi
    # Strix APUs show in /proc/cpuinfo; discrete cards don't, so also try WMI. Either reroutes.
    if ! grep -qiE 'Ryzen AI Max|Radeon 80[0-9][05]S|Strix Halo' /proc/cpuinfo 2>/dev/null \
       && ! _wsl_amd_gpu_name >/dev/null 2>&1; then
        return 0
    fi
    # Already ROCm-on-WSL? leave a working GPU alone, whatever the version.
    if [ -e /opt/rocm/lib/librocdxg.so ] || [ -e /opt/rocm/lib64/librocdxg.so ]; then
        return 0
    fi
    _rr_ver=""
    [ -r /etc/os-release ] && _rr_ver=$(. /etc/os-release 2>/dev/null; printf '%s' "${VERSION_ID:-}")
    # The bootstrap (scripts/install_rocm_wsl_strixhalo.sh) dies on any VERSION_ID but
    # 24.04 and pins the noble repo, so 24.04 is the sole GPU-supported target; leave a
    # 24.04 user alone. (Working ROCm on other versions was caught by librocdxg above.)
    case "$_rr_ver" in 24.04) return 0 ;; esac
    # Distro is now unsupported. If we can't reroute to a 24.04 target, stay CPU-only
    # AND skip the later origin-distro ROCm bootstrap (it ignores distro version, so it
    # would otherwise install ROCm into 26.04 etc.).
    command -v wsl.exe >/dev/null 2>&1 || { UNSLOTH_SKIP_ROCM_WSL_SETUP=1; return 0; }
    # Route only to an installed Ubuntu-24.04 (bootstrap's only target). Match the whole
    # line (one distro per line from wsl.exe -l -q), not a substring, so "Ubuntu-24.04-test"
    # can't masquerade as it and then fail `wsl -d`.
    # || true: no match is expected, not an error (script runs under set -e).
    _rr_distros=$(wsl.exe -l -q 2>/dev/null | tr -d '\000\r')
    _rr_target=$(printf '%s\n' "$_rr_distros" | grep -ixF "Ubuntu-24.04" | head -n1) || true
    [ -n "$_rr_target" ] || {
        substep "ROCm-on-WSL (GPU) needs Ubuntu 24.04; this distro is Ubuntu ${_rr_ver:-unknown}." "$C_WARN"
        substep "No Ubuntu-24.04 WSL distro found; staying CPU-only. Install Ubuntu-24.04 and re-run there for GPU." "$C_WARN"
        UNSLOTH_SKIP_ROCM_WSL_SETUP=1
        return 0
    }

    echo ""
    substep "ROCm-on-WSL (GPU) needs Ubuntu 24.04; this distro is Ubuntu ${_rr_ver:-unknown}." "$C_WARN"
    substep "Found an existing $_rr_target distro -- continuing the GPU install there." "$C_OK"
    # A --local checkout can't be replayed via curl|sh (the repo isn't in the target
    # distro), so tell the user to re-run there rather than silently run a different install.
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        substep "This is a --local install; re-run it from $_rr_target instead:" "$C_WARN"
        substep "  wsl -d $_rr_target -- bash -lc 'cd <your checkout> && ./install.sh --local'" "$C_WARN"
        substep "Continuing CPU-only in Ubuntu ${_rr_ver:-this distro} for now." "$C_WARN"
        # Unsupported distro, can't reroute a --local checkout: skip the origin ROCm bootstrap.
        UNSLOTH_SKIP_ROCM_WSL_SETUP=1
        return 0
    fi
    # Forward the caller's options/env (custom package/python/home) so the rerouted
    # install matches what was asked for, not a default install.
    _rr_q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
    _rr_exports="set -o pipefail; export UNSLOTH_WSL_REROUTED=1"
    [ "$_STUDIO_HOME_REDIRECT" = "env" ] && _rr_exports="$_rr_exports; export UNSLOTH_STUDIO_HOME=$(_rr_q "$STUDIO_HOME")"
    # Forward explicit ROCm-bootstrap consent (e.g. Tauri) so the child auto-enables the
    # GPU instead of falling back to the desktop-app prompt path.
    [ "${UNSLOTH_ROCM_WSL_AUTO:-0}" = "1" ] && _rr_exports="$_rr_exports; export UNSLOTH_ROCM_WSL_AUTO=1"
    # Forward a pinned torch index into the rerouted distro; dropping it would
    # silently revert the child install to auto-detection.
    [ -n "${UNSLOTH_TORCH_INDEX_URL:-}" ] && _rr_exports="$_rr_exports; export UNSLOTH_TORCH_INDEX_URL=$(_rr_q "$UNSLOTH_TORCH_INDEX_URL")"
    [ -n "${UNSLOTH_TORCH_INDEX_FAMILY:-}" ] && _rr_exports="$_rr_exports; export UNSLOTH_TORCH_INDEX_FAMILY=$(_rr_q "$UNSLOTH_TORCH_INDEX_FAMILY")"
    [ "$_SKIP_AUTOSTART" = true ] && _rr_exports="$_rr_exports; export UNSLOTH_SKIP_AUTOSTART=1"
    _rr_args=""
    [ "$PACKAGE_NAME" != "unsloth" ] && _rr_args="$_rr_args --package $(_rr_q "$PACKAGE_NAME")"
    [ -n "$_USER_PYTHON" ] && _rr_args="$_rr_args --python $(_rr_q "$_USER_PYTHON")"
    [ "$_VERBOSE" = true ] && _rr_args="$_rr_args --verbose"
    [ "$TAURI_MODE" = true ] && _rr_args="$_rr_args --tauri"
    if [ -n "${UNSLOTH_WSL_REROUTE_CMD:-}" ]; then
        _rr_cmd="$UNSLOTH_WSL_REROUTE_CMD"               # user took full control
    elif [ -n "$_rr_args" ]; then
        _rr_cmd="curl -fsSL https://unsloth.ai/install.sh | sh -s --$_rr_args"
    else
        _rr_cmd="curl -fsSL https://unsloth.ai/install.sh | sh"
    fi
    # pipefail so a failed curl in `curl | sh` isn't masked by sh exiting 0 on empty
    # input (which would wrongly report success and exit 0 the parent installer).
    _rr_rc=0
    wsl.exe -d "$_rr_target" -- bash -lc "$_rr_exports; $_rr_cmd" || _rr_rc=$?
    if [ "$_rr_rc" -eq 0 ]; then
        exit 0
    fi
    # In Tauri mode the child uses exit 2 ([TAURI:NEED_SUDO]) to ask the desktop app to
    # elevate for the target distro; the child already printed the NEED_SUDO line, so
    # propagate the code instead of masking it as a reroute failure and dropping to CPU.
    if [ "$TAURI_MODE" = true ] && [ "$_rr_rc" -eq 2 ]; then
        exit 2
    fi
    substep "Could not auto-continue in $_rr_target; run it yourself:" "$C_WARN"
    substep "  wsl -d $_rr_target -- bash -lc 'curl -fsSL https://unsloth.ai/install.sh | sh'"
    substep "Continuing CPU-only in Ubuntu ${_rr_ver:-this distro} for now." "$C_WARN"
    # Reroute failed; don't let the later bootstrap install ROCm into this unsupported
    # distro -- stay CPU-only.
    UNSLOTH_SKIP_ROCM_WSL_SETUP=1
    return 0
}
_maybe_reroute_strixhalo_to_2404 || true

# ── Check system dependencies ──
# cmake/git are only needed to *build* llama.cpp from source. Unsloth downloads a
# prebuilt by default, and setup.sh self-skips the source build when they're
# absent -- so macOS doesn't block on cmake (requiring it would force a manual
# Homebrew install). Linux keeps requiring them; its package manager has them.
tauri_log "STEP" "Checking system dependencies"

case "$OS" in
    macos)
        # Xcode Command Line Tools provide the C/C++ compiler and git.
        if ! xcode-select -p >/dev/null 2>&1; then
            echo ""
            echo "==> Xcode Command Line Tools are required."
            echo "    Installing (a system dialog will appear)..."
            xcode-select --install </dev/null 2>/dev/null || true
            echo "    After the installation completes, please re-run this script."
            exit 1
        fi
        # cmake is only needed for a source build; the default prebuilt path
        # doesn't use it, so its absence is not fatal -- no Homebrew prerequisite.
        if command -v cmake >/dev/null 2>&1; then
            step "deps" "all system dependencies found"
        else
            step "deps" "using prebuilt llama.cpp (cmake not found)" "$C_WARN"
            substep "Install cmake only if you want a source build: brew install cmake"
        fi
        ;;
    linux|wsl)
        MISSING=""
        command -v cmake >/dev/null 2>&1 || MISSING="$MISSING cmake"
        command -v git   >/dev/null 2>&1 || MISSING="$MISSING git"
        # curl or wget is needed for downloads; check both
        if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
            MISSING="$MISSING curl"
        fi
        command -v gcc  >/dev/null 2>&1 || MISSING="$MISSING build-essential"
        # libcurl dev headers for llama.cpp HTTPS support
        command -v curl-config >/dev/null 2>&1 || MISSING="$MISSING libcurl4-openssl-dev"

        MISSING=$(echo "$MISSING" | sed 's/^ *//')
        if [ -n "$MISSING" ]; then
            echo ""
            step "deps" "missing: $MISSING" "$C_WARN"
            substep "These are needed to build the GGUF inference engine."
            if command -v apt-get >/dev/null 2>&1; then
                _smart_apt_install $MISSING
            else
                echo "    Automatic system package installation is supported on apt-based"
                echo "    Linux distributions (Ubuntu/Debian) only. Please install the"
                echo "    missing dependencies with your package manager, then re-run setup:"
                echo "    $MISSING"
                echo ""
                echo "    Examples:"
                echo "      Fedora/RHEL: sudo dnf install cmake git gcc gcc-c++ make libcurl-devel"
                echo "      Arch:       sudo pacman -S --needed cmake git base-devel curl"
                echo "      openSUSE:   sudo zypper install cmake git gcc gcc-c++ make libcurl-devel"
                exit 1
            fi
            echo ""
        else
            step "deps" "all system dependencies found"
        fi
        ;;
esac

# ── Install uv ──
tauri_log "STEP" "Installing uv package manager"
UV_MIN_VERSION="0.8.16"

# When bytecode compilation is enabled, large installs can exceed uv's 60s default on slow machines. Default to 180s, preserving overrides ("0" disables).
: "${UV_COMPILE_BYTECODE_TIMEOUT:=180}"
export UV_COMPILE_BYTECODE_TIMEOUT

# uv >= 0.8.16 retries HTTP/2 streaming body errors; raise retries and read
# timeout for large wheel downloads. ":=" preserves any user override.
: "${UV_HTTP_RETRIES:=5}"
export UV_HTTP_RETRIES
: "${UV_HTTP_TIMEOUT:=180}"
export UV_HTTP_TIMEOUT

# macOS: trust the system Keychain so uv uses SecureTransport instead of rustls.
# Required behind TLS-inspecting proxies (Cisco Umbrella, Zscaler, etc.) which
# present their own CA certificate. rustls (uv's default) ignores the Keychain
# and rejects intercepted connections with "invalid peer certificate: UnknownIssuer".
# Set both vars: UV_SYSTEM_CERTS is the modern one (uv >= 0.11), UV_NATIVE_TLS the
# legacy one understood by uv 0.8.16-0.10.x, which the installer keeps if already
# present (UV_MIN_VERSION) and which ignores UV_SYSTEM_CERTS. Mirror the choice onto
# both so it works on either uv. Opt out with UV_SYSTEM_CERTS=0.
if [ "$OS" = "macos" ]; then
    : "${UV_SYSTEM_CERTS:=1}"
    : "${UV_NATIVE_TLS:=$UV_SYSTEM_CERTS}"
fi
[ -n "${UV_SYSTEM_CERTS:-}" ] && export UV_SYSTEM_CERTS
[ -n "${UV_NATIVE_TLS:-}" ] && export UV_NATIVE_TLS

version_ge() {
    # returns 0 if $1 >= $2
    _a=$1
    _b=$2

    while [ -n "$_a" ] || [ -n "$_b" ]; do
        _a_part=${_a%%.*}
        _b_part=${_b%%.*}

        [ "$_a" = "$_a_part" ] && _a="" || _a=${_a#*.}
        [ "$_b" = "$_b_part" ] && _b="" || _b=${_b#*.}

        [ -z "$_a_part" ] && _a_part=0
        [ -z "$_b_part" ] && _b_part=0

        if [ "$_a_part" -gt "$_b_part" ]; then
            return 0
        fi
        if [ "$_a_part" -lt "$_b_part" ]; then
            return 1
        fi
    done

    return 0
}

_uv_version_ok() {
    _raw=$("$1" --version 2>/dev/null | awk '{print $2}') || return 1
    [ -n "$_raw" ] || return 1
    _ver=${_raw%%[-+]*}
    case "$_ver" in
        ''|*[!0-9.]*) return 1 ;;
    esac
    version_ge "$_ver" "$UV_MIN_VERSION" || return 1
    # Prerelease of the exact minimum (e.g. 0.7.14-rc1) is still below stable 0.7.14
    [ "$_ver" = "$UV_MIN_VERSION" ] && [ "$_raw" != "$_ver" ] && return 1
    return 0
}

if ! command -v uv >/dev/null 2>&1 || ! _uv_version_ok uv; then
    substep "installing uv package manager..."
    _uv_tmp=$(mktemp)
    download "https://astral.sh/uv/install.sh" "$_uv_tmp"
    run_maybe_quiet sh "$_uv_tmp" </dev/null
    rm -f "$_uv_tmp"
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Create venv (migrate old layout if possible, otherwise fresh) ──
tauri_log "STEP" "Creating virtual environment"
mkdir -p "$STUDIO_HOME"

_MIGRATED=false
# Empty so an inherited value can never masquerade as a probed torch version.
_PREV_TORCH_VER=""

if [ -x "$VENV_DIR/bin/python" ]; then
    # why: matching guard to the .venv branch below -- in env-mode
    # $STUDIO_HOME is a user-chosen workspace, so refuse to nuke an
    # existing $STUDIO_HOME/unsloth_studio that lacks Unsloth sentinels.
    # Accept the in-VENV ownership marker so partial-install retries are
    # not blocked. Sentinels must be regular files: -f follows symlinks
    # to files (the legitimate ln -s shim shape) but rejects directories
    # and broken/dir-targeted symlinks.
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ] \
       && [ ! -f "$VENV_DIR/.unsloth-studio-owned" ] \
       && [ ! -f "$STUDIO_HOME/share/studio.conf" ] \
       && [ ! -f "$STUDIO_HOME/bin/unsloth" ]; then
        echo "ERROR: $VENV_DIR already exists but does not look like an Unsloth Studio install." >&2
        echo "       Move it aside or choose an empty UNSLOTH_STUDIO_HOME." >&2
        exit 1
    fi
    # Record the existing venv's torch BEFORE the replacement moves it aside: a re-run
    # rebuilds the venv for clean state, but must keep the torch release the user
    # already has (see _previous_torch_pin below). Last line only: sitecustomize or
    # import-hook noise on stdout must not corrupt the version.
    _PREV_TORCH_VER=$("$VENV_DIR/bin/python" -c \
        "import torch; print(torch.__version__)" 2>/dev/null | tail -n 1 || true)
    # New layout already exists — replace only after preserving rollback copy.
    substep "preserving existing environment for rollback..."
    _start_studio_venv_replacement "$VENV_DIR"
elif [ "$_STUDIO_HOME_REDIRECT" != "env" ] && [ -x "$STUDIO_HOME/.venv/bin/python" ]; then
    # Old layout exists — validate before migrating.
    # Skip in env-mode so we don't rm -rf an unrelated .venv at the
    # workspace root (e.g. user's existing project Python venv).
    # In no-torch mode, a missing torch package is expected; validate Python only.
    substep "found legacy Unsloth environment, validating..."
    _legacy_ok=false
    if [ "$SKIP_TORCH" = true ]; then
        if "$STUDIO_HOME/.venv/bin/python" -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
            _legacy_ok=true
        fi
    elif "$STUDIO_HOME/.venv/bin/python" -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
A = torch.ones((10, 10), device=device)
B = torch.ones((10, 10), device=device)
C = torch.ones((10, 10), device=device)
D = A + B
E = D @ C
torch.testing.assert_close(torch.unique(E), torch.tensor((20,), device=E.device, dtype=E.dtype))
" >/dev/null 2>&1; then
        _legacy_ok=true
    fi
    if [ "$_legacy_ok" = true ]; then
        echo "✅ Legacy environment is healthy — migrating..."
        mv "$STUDIO_HOME/.venv" "$VENV_DIR"
        echo "   Moved ~/.unsloth/studio/.venv → $VENV_DIR"
        _MIGRATED=true
    else
        echo "⚠️  Legacy environment failed validation — creating fresh environment"
        _invalid_venv="$STUDIO_HOME/.venv.invalid.$(date +%Y%m%d%H%M%S 2>/dev/null || echo time).$$"
        mv "$STUDIO_HOME/.venv" "$_invalid_venv" 2>/dev/null || true
    fi
fi

# If an Intel Mac has a stale 3.13 venv from a previous failed install, recreate
# (skip when the user explicitly chose a version via --python)
if [ "$SKIP_TORCH" = true ] && [ "$MAC_INTEL" = true ] && [ -z "$_USER_PYTHON" ] && [ -x "$VENV_DIR/bin/python" ]; then
    _PY_MM=$("$VENV_DIR/bin/python" -c \
        "import sys; print('{}.{}'.format(*sys.version_info[:2]))" 2>/dev/null || echo "")
    if [ "$_PY_MM" != "3.12" ]; then
        echo "  Recreating Intel Mac environment with Python 3.12 (was $_PY_MM)..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    step "venv" "creating Python ${PYTHON_VERSION} virtual environment"
    substep "$VENV_DIR"
    if [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ] && [ -z "$_USER_PYTHON" ]; then
        # Apple Silicon: request an arch-explicit arm64 CPython so uv cannot
        # reuse a cached x86_64 (Rosetta) build. torch ships no macOS x86_64
        # wheels since 2.2.2, so an x86_64 venv makes the torch install
        # unresolvable. The arm64 guard below is kept as a backstop for
        # migrated / pre-existing venvs.
        run_install_cmd "create venv" uv venv "$VENV_DIR" \
            --python "cpython-${PYTHON_VERSION}-macos-aarch64-none"
    else
        run_install_cmd "create venv" uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
fi

# Mark the freshly-created venv as Unsloth-owned so a partial install can be
# repaired by re-running install.sh; the env-mode deletion guard above accepts
# this marker as the primary sentinel.
if [ -x "$VENV_DIR/bin/python" ]; then
    : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
fi

# Guard against two independent Apple Silicon venv problems, in order:
#   1. uv may create the venv from a cached x86_64 (Rosetta) Python when a
#      same-version x86_64 build is already cached (often because uv itself
#      is an x86_64 build). That venv reports x86_64 to wheel resolvers, and
#      PyTorch ships no macOS wheels on the CPU index for any architecture,
#      so the torch install can never resolve. Recreate it with an
#      arch-explicit arm64 CPython.
#   2. Python 3.13.8 has a known torch import bug.
# The two are independent: a venv may be x86_64 and, once recreated, still
# land on 3.13.8. So we re-inspect the interpreter between the checks instead
# of chaining them with elif, guaranteeing both invariants hold on whatever
# venv we end up with. Skip both when the user explicitly chose an interpreter
# via --python.
if [ -z "$_USER_PYTHON" ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _inspect_venv() {
        "$VENV_DIR/bin/python" -c \
            "import platform, sys; print(platform.machine(), '{}.{}.{}'.format(*sys.version_info[:3]))" \
            2>/dev/null || echo " "
    }
    _info=$(_inspect_venv)
    _VENV_ARCH=${_info%% *}
    _PY_VER=${_info##* }
    # If the interpreter could not be executed (an x86_64 venv python on a Mac
    # without Rosetta installed), the probe above yields an empty arch. Fall
    # back to reading the binary's Mach-O arch statically so the x86_64
    # recreate below still triggers instead of letting uv fail later.
    if [ -z "$_VENV_ARCH" ] && [ -x "$VENV_DIR/bin/python" ]; then
        # uv symlinks bin/python to the base interpreter, so dereference with
        # file -L (lipo already follows the link). Trailing || true keeps the
        # installer alive under set -e when neither tool is present.
        _archs=$(lipo -archs "$VENV_DIR/bin/python" 2>/dev/null \
            || file -L "$VENV_DIR/bin/python" 2>/dev/null || true)
        case "$_archs" in
            *arm64*)  _VENV_ARCH=arm64 ;;
            *x86_64*) _VENV_ARCH=x86_64 ;;
        esac
    fi

    if [ "$_VENV_ARCH" = "x86_64" ]; then
        echo "  WARNING: venv was created with an x86_64 (Rosetta) Python on Apple Silicon."
        echo "  Recreating venv with native arm64 Python ${PYTHON_VERSION}..."
        rm -rf "$VENV_DIR"
        run_install_cmd "recreate venv (arm64)" uv venv "$VENV_DIR" \
            --python "cpython-${PYTHON_VERSION}-macos-aarch64-none"
        if [ -x "$VENV_DIR/bin/python" ]; then
            : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
        fi
        # Re-inspect: the recreated arm64 venv may still be 3.13.8.
        _info=$(_inspect_venv)
        _VENV_ARCH=${_info%% *}
        _PY_VER=${_info##* }
    fi

    if [ "$_PY_VER" = "3.13.8" ]; then
        echo "  WARNING: Python 3.13.8 has a known torch import bug."
        echo "  Recreating venv with Python 3.12..."
        rm -rf "$VENV_DIR"
        PYTHON_VERSION="3.12"
        run_install_cmd "recreate venv" uv venv "$VENV_DIR" \
            --python "cpython-${PYTHON_VERSION}-macos-aarch64-none"
        if [ -x "$VENV_DIR/bin/python" ]; then
            : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
        fi
    fi
fi

if [ -x "$VENV_DIR/bin/python" ]; then
    step "venv" "using environment"
    substep "${VENV_DIR}"
fi

# Default torch constraint -- tightened for Python 3.13+ on arm64 macOS
# (torch <2.6 has no cp313 macOS arm64 wheels)
TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
if [ "$SKIP_TORCH" = false ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _PY_MINOR=$("$VENV_DIR/bin/python" -c \
        "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$_PY_MINOR" -ge 13 ] 2>/dev/null; then
        TORCH_CONSTRAINT="torch>=2.6,<2.11.0"
    fi
fi
# Companion (torchvision/torchaudio) constraints, bounded to torch's window.
# torchaudio 2.11 dropped its exact torch pin, so a bare companion next to a
# <2.11-capped torch resolves torchaudio 2.11 (verified: cpu leaf installed
# torch 2.10.0+cpu with torchaudio 2.11.0+cpu). torchvision still exact-pins
# torch and self-corrects, but is bounded for symmetry. Widened alongside the
# cu* torch window below; the torch-2.11 AMD paths (rocm7.2 / per-gfx / Strix)
# pin their own trio.
TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.26.0"
TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<2.11.0"

# ── Resolve repo root (for --local installs) ──
_REPO_ROOT="$(cd "$(dirname "$0" 2>/dev/null || echo ".")" && pwd)"

# ── Helper: find no-torch-runtime.txt (local repo or site-packages) ──
_find_no_torch_runtime() {
    # Check local repo first (for --local installs)
    if [ -f "$_REPO_ROOT/studio/backend/requirements/no-torch-runtime.txt" ]; then
        echo "$_REPO_ROOT/studio/backend/requirements/no-torch-runtime.txt"
        return
    fi
    # Check inside installed package
    _rt=$(find "$VENV_DIR" -path "*/studio/backend/requirements/no-torch-runtime.txt" -print -quit 2>/dev/null || echo "")
    if [ -n "$_rt" ]; then
        echo "$_rt"
        return
    fi
}

# ── AMD ROCm GPU detection helper ──
# WSL2 ROCDXG: the system rocminfo enumerates the GPU over /dev/dxg only when
# HSA_ENABLE_DXG_DETECTION=1 (a no-op on bare metal), and /opt/rocm/bin can be
# off PATH outside login shells (the profile.d drop-in). Seed both before any
# rocminfo probe or a ROCDXG WSL host is misdetected as CPU-only.
_ensure_rocm_probe_env() {
    export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
    if ! command -v rocminfo >/dev/null 2>&1 && [ -x /opt/rocm/bin/rocminfo ]; then
        PATH="$PATH:/opt/rocm/bin"
    fi
}

# Returns 0 if an AMD GPU is present. Checks rocminfo, amd-smi, then sysfs
# KFD topology (env-var-independent fallback for when HIP/ROCR_VISIBLE_DEVICES hides devices).
# Always returns 1 (false) when an NVIDIA GPU is present: blocks every
# detection path (rocminfo, amd-smi, KFD sysfs) from producing a false
# positive on NVIDIA-only or NVIDIA-primary hosts, even when ROCm tools
# are co-installed.
_has_amd_rocm_gpu() {
    _ensure_rocm_probe_env
    if _has_usable_nvidia_gpu; then
        return 1
    fi
    if command -v rocminfo >/dev/null 2>&1 && \
       rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx[1-9][0-9]/{found=1} END{exit !found}'; then
        return 0
    elif command -v amd-smi >/dev/null 2>&1 && \
         amd-smi list 2>/dev/null | awk '/^GPU[[:space:]]*[:\[][[:space:]]*[0-9]/{ found=1 } END{ exit !found }'; then
        return 0
    elif [ -e /dev/kfd ] && \
         awk '/vendor_id/ && $2 == 4098 { found = 1 } END { exit !found }' \
             /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null; then
        # vendor_id 4098 = 0x1002 (AMD) marks a GPU node: the KFD CPU node
        # reports vendor_id 0, so any 4098 node is an AMD GPU. NVIDIA's open
        # kernel module (driver 560+) registers KFD nodes as vendor_id 4318
        # (0x10DE), so this never false-positives on NVIDIA-only hosts.
        # The prior check also required a gpu_id line, but gpu_id is a SIBLING
        # sysfs file, not a line in properties -- it never matched, so the
        # fallback silently missed every ROCm-less AMD host (issue: fresh
        # Arch/CachyOS boxes reporting "no GPU detected").
        return 0
    fi
    return 1
}

# Returns 0 if an AMD display GPU is on the PCI bus even when ROCm can't use it
# (e.g. a Strix Halo iGPU with no /dev/kfd). Only sharpens the "no GPU detected"
# hint. vendor 0x1002 = AMD/ATI; class 0x03* = display controller.
_amd_gpu_present_via_pci() {
    [ -d /sys/bus/pci/devices ] || return 1
    for _pci_vendor in /sys/bus/pci/devices/*/vendor; do
        [ -r "$_pci_vendor" ] || continue
        read -r _v < "$_pci_vendor" 2>/dev/null || continue
        [ "$_v" = "0x1002" ] || continue
        _cls="${_pci_vendor%vendor}class"
        [ -r "$_cls" ] || continue
        read -r _c < "$_cls" 2>/dev/null || continue
        case "$_c" in 0x03*) return 0 ;; esac
    done
    return 1
}

# ── Detect GPU and choose PyTorch index URL ──
# Mirrors Get-TorchIndexUrl in install.ps1.
# On CPU-only machines this returns the cpu index, avoiding the solver
# dead-end where --torch-backend=auto resolves to unsloth==2024.8.
get_torch_index_url() {
    _base="${UNSLOTH_PYTORCH_MIRROR:-https://download.pytorch.org/whl}"
    _base="${_base%/}"
    # Explicit override -- skip ALL GPU probing (headless / container / CI / cross-install).
    # UNSLOTH_TORCH_INDEX_URL wins (full URL, verbatim); _FAMILY is the leaf (cpu, cu128, ...)
    # appended to the mirror base. Trim whitespace so a whitespace-only value is unset.
    _url="${UNSLOTH_TORCH_INDEX_URL:-}"
    _url="${_url#"${_url%%[![:space:]]*}"}"; _url="${_url%"${_url##*[![:space:]]}"}"
    if [ -n "$_url" ]; then
        # Trim trailing PATH slashes (a multi-slash path 404s on strict pip proxies) while
        # preserving a ?query/#fragment token (a whole-URL strip would eat a "/"-ending token).
        _url=$(_trim_index_path_slashes "$_url")
        echo "$_url"; return
    fi
    _family="${UNSLOTH_TORCH_INDEX_FAMILY:-}"
    _family="${_family#"${_family%%[![:space:]]*}"}"; _family="${_family%"${_family##*[![:space:]]}"}"
    if [ -n "$_family" ]; then
        while [ "${_family#/}" != "$_family" ]; do _family="${_family#/}"; done
        while [ "${_family%/}" != "$_family" ]; do _family="${_family%/}"; done
        echo "$_base/$_family"; return
    fi
    # macOS: always CPU (no CUDA support)
    case "$(uname -s)" in Darwin) echo "$_base/cpu"; return ;; esac
    # Try nvidia-smi -- require the binary to actually list a usable GPU.
    # Presence of the binary alone (container leftovers, stale driver
    # packages) is not sufficient: otherwise an AMD-only host would
    # silently install CUDA wheels.
    _smi=""
    _nvidia_detected=0
    if _has_usable_nvidia_gpu; then
        _nvidia_detected=1
        if command -v nvidia-smi >/dev/null 2>&1; then
            _smi="nvidia-smi"
        elif [ -x "/usr/bin/nvidia-smi" ]; then
            _smi="/usr/bin/nvidia-smi"
        fi
    fi
    if [ "$_nvidia_detected" -eq 0 ]; then
        # No NVIDIA GPU -- check for AMD ROCm GPU.
        # PyTorch only publishes ROCm wheels for linux-x86_64; skip the
        # ROCm branch entirely on aarch64 / arm64 / other architectures
        # so non-x86_64 Linux hosts fall back cleanly to CPU wheels.
        case "$(uname -m)" in
            x86_64|amd64) : ;;
            *) echo "$_base/cpu"; return ;;
        esac
        if ! _has_amd_rocm_gpu; then
            echo "$_base/cpu"; return
        fi
        # AMD GPU confirmed -- detect ROCm version
        _rocm_tag=""
        _rocm_tag=$({ command -v amd-smi >/dev/null 2>&1 && \
            amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
                'NF>1{gsub(/[^0-9.]/, "", $2); split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}'; } || \
            { [ -r /opt/rocm/.info/version ] && \
                awk -F. '{print "rocm"$1"."$2; exit}' /opt/rocm/.info/version; } || \
            { command -v hipconfig >/dev/null 2>&1 && \
                hipconfig --version 2>/dev/null | awk 'NR==1 && /^[0-9]/{split($1,a,"."); if(a[1]+0>0){print "rocm"a[1]"."a[2]; found=1}} END{exit !found}'; } || \
            { command -v dpkg-query >/dev/null 2>&1 && \
                ver="$(dpkg-query -W -f='${Version}\n' rocm-core 2>/dev/null)" && \
                [ -n "$ver" ] && \
                printf '%s\n' "$ver" | sed 's/^[0-9]*://' | awk -F'[.-]' '{print "rocm"$1"."$2; exit}'; } || \
            { command -v rpm >/dev/null 2>&1 && \
                ver="$(rpm -q --qf '%{VERSION}\n' rocm-core 2>/dev/null)" && \
                [ -n "$ver" ] && \
                printf '%s\n' "$ver" | awk -F'[.-]' '{print "rocm"$1"."$2; exit}'; }) 2>/dev/null
        # Validate _rocm_tag: must match "rocmX.Y" with major >= 1
        case "$_rocm_tag" in
            rocm[1-9]*.[0-9]*) : ;;  # valid (major >= 1)
            *) _rocm_tag="" ;;        # reject malformed (empty, garbled, or major=0)
        esac
        if [ -n "$_rocm_tag" ]; then
            # Minimum supported: ROCm 6.0 (no PyTorch wheels exist for older)
            case "$_rocm_tag" in
                rocm[1-5].*)
                    echo "[WARN] ROCm $_rocm_tag detected but PyTorch ROCm wheels require ROCm 6.0+ -- falling back to CPU-only PyTorch" >&2
                    echo "[WARN] Upgrade ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html" >&2
                    echo "$_base/cpu"; return ;;
            esac
            # Supported tags; 6.5+ clips to rocm6.4, 7.3+ caps to rocm7.2.
            # PyTorch publishes major.minor URLs only (no patch level), so
            # rocm7.2.1 / rocm6.0.2 / etc. must normalise to rocm7.2 / rocm6.0.
            case "$_rocm_tag" in
                rocm6.0|rocm6.0.*) echo "$_base/rocm6.0" ;;
                rocm6.1|rocm6.1.*) echo "$_base/rocm6.1" ;;
                rocm6.2|rocm6.2.*) echo "$_base/rocm6.2" ;;
                rocm6.3|rocm6.3.*) echo "$_base/rocm6.3" ;;
                rocm6.4|rocm6.4.*) echo "$_base/rocm6.4" ;;
                rocm7.0|rocm7.0.*) echo "$_base/rocm7.0" ;;
                rocm7.1|rocm7.1.*) echo "$_base/rocm7.1" ;;
                rocm7.2|rocm7.2.*) echo "$_base/rocm7.2" ;;
                rocm6.*)
                    # ROCm 6.5+ (no published PyTorch wheels): clip down
                    # to the last supported 6.x wheel set.
                    echo "$_base/rocm6.4" ;;
                *)
                    # ROCm 7.3+ (future): cap to rocm7.2 (latest known)
                    echo "$_base/rocm7.2" ;;
            esac
            return
        fi
        # AMD GPU confirmed (rocminfo/amd-smi or the KFD topology fallback) but
        # no ROCm/HIP install was found to read the version from (amd-smi,
        # /opt/rocm/.info/version, hipconfig, dpkg, rpm). This is the common
        # fresh-install case: the GPU is real, but with no ROCm userspace the
        # correct PyTorch build can't be selected. Warn with an actionable fix
        # rather than silently installing CPU PyTorch.
        echo "[WARN] AMD GPU detected, but no ROCm/HIP install was found to select the matching GPU PyTorch build -- falling back to CPU-only PyTorch." >&2
        echo "[WARN] Install the ROCm/HIP SDK, then re-run this installer:" >&2
        echo "[WARN]   Arch / CachyOS : sudo pacman -S rocm-hip-sdk" >&2
        echo "[WARN]   other distros  : https://rocm.docs.amd.com/en/latest/deploy/linux/index.html" >&2
        echo "[WARN] Minimum required for version detection: amd-smi, hipconfig, /opt/rocm/.info/version, or the rocm-core package." >&2
        echo "$_base/cpu"; return
    fi
    # Parse CUDA version from nvidia-smi output (POSIX-safe, no grep -P).
    # Newer NVIDIA drivers (e.g. 610.x) print "CUDA UMD Version: X.Y" instead
    # of the legacy "CUDA Version: X.Y"; accept both with two BRE expressions
    # (POSIX sed does not support "?" without -E).  The two patterns are
    # mutually exclusive per line, so head -1 picks the first emitted match.
    # Bound the call (a wedged nvidia-smi would otherwise hang here) and force
    # the C locale for stable parsing. LC_ALL is exported inside this command
    # substitution subshell so it reaches nvidia-smi through _run_bounded
    # without depending on `env`; the export is scoped to the subshell.
    _cuda_ver=$(export LC_ALL=C; _run_bounded "$_smi" 2>/dev/null \
        | sed -n \
            -e 's/.*CUDA UMD Version:[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
            -e 's/.*CUDA Version:[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | head -1)
    if [ -z "$_cuda_ver" ]; then
        echo "[WARN] Could not determine CUDA version from nvidia-smi, defaulting to cu126" >&2
        echo "$_base/cu126"; return
    fi
    _major=${_cuda_ver%%.*}
    _minor=${_cuda_ver#*.}
    if [ "$_major" -ge 13 ]; then echo "$_base/cu130"
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 8 ]; then echo "$_base/cu128"
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 6 ]; then echo "$_base/cu126"
    elif [ "$_major" -ge 12 ]; then echo "$_base/cu124"
    elif [ "$_major" -ge 11 ]; then echo "$_base/cu118"
    else echo "$_base/cpu"; fi
}

# ── Torch flavor helpers (to repair a stale CPU / wrong-CUDA wheel) ──
# torch.__version__ ($1) -> flavor tag (cuXXX / rocm / cpu); untagged wheel = cpu.
_torch_flavor_tag() {
    case "$1" in
        *+cu[0-9]*) printf '%s\n' "$1" | sed -n 's/.*+\(cu[0-9][0-9]*\).*/\1/p' ;;
        *+rocm*)    echo "rocm" ;;
        *+cpu*)     echo "cpu" ;;
        "")         echo "" ;;
        *)          echo "cpu" ;;
    esac
}

# Final path segment of a wheel index URL ($1), lowercased, query/fragment stripped first
# so a token-authenticated pin (.../cu128?token=x) classifies as cu128 (else it reinstalls
# every update). Classification only. Shared with the py / ps1 leaf extractors.
_torch_index_url_leaf() {
    _tl_u="${1%%\?*}"
    _tl_u="${_tl_u%%#*}"
    # Strip ALL trailing slashes, not one: .../rocm7.2// must yield rocm7.2, not an empty leaf.
    while [ -n "$_tl_u" ] && [ "${_tl_u%/}" != "$_tl_u" ]; do
        _tl_u="${_tl_u%/}"
    done
    printf '%s' "${_tl_u##*/}" | tr '[:upper:]' '[:lower:]'
}

# True (exit 0) when a lowercased leaf is an EXACT pip ROCm family: rocm<digits>[.<digits>]
# or a gfx ARCHITECTURE leaf (gfx followed by a digit: gfx90a, gfx1151, gfx120x-all). A leaf
# that merely starts with rocm/gfx (rocm7.2-private, gfx-private) is a custom verbatim pin.
# Matches the py / ps1 sides.
_is_pip_rocm_family_leaf() {
    case "$1" in
        gfx[0-9]*) return 0 ;;
        rocm[0-9]*)
            # Exact rocm<digits>[.<digits>]: both major and minor must be non-empty all-digits
            # (rocm7., rocm7.2.1, rocm7.2-private are all custom pins, not a family).
            _rocm_rest="${1#rocm}"
            case "$_rocm_rest" in
                *.*.*) return 1 ;;
                *.*)
                    _rocm_minor="${_rocm_rest#*.}"
                    case "${_rocm_rest%%.*}" in "" | *[!0-9]*) return 1 ;; esac
                    case "$_rocm_minor" in "" | *[!0-9]*) return 1 ;; esac
                    ;;
                *[!0-9]*) return 1 ;;
            esac
            return 0
            ;;
        *) return 1 ;;
    esac
}

# Whether release base $1 (X.Y[.Z...]) falls inside constraint window $2
# ("torch>=A.B[.C],<D.E.F"). Compares at major.minor granularity, which is exact
# for the windows this script uses (ceilings are always X.Y.0); a non-.0 ceiling
# would only make this conservative (excludes the whole ceiling minor). Anything
# unparseable answers "no" so the caller fails toward the supported range.
_torch_release_in_window() {
    _trw_con="$2"
    case "$_trw_con" in
        "torch>="*",<"*) ;;
        *) echo "no"; return ;;
    esac
    _trw_floor="${_trw_con#torch>=}"; _trw_floor="${_trw_floor%%,*}"
    _trw_ceil="${_trw_con##*,<}"
    _v_maj="${1%%.*}";          _v_rest="${1#*.}";          _v_min="${_v_rest%%.*}"
    _f_maj="${_trw_floor%%.*}"; _f_rest="${_trw_floor#*.}"; _f_min="${_f_rest%%.*}"
    _c_maj="${_trw_ceil%%.*}";  _c_rest="${_trw_ceil#*.}";  _c_min="${_c_rest%%.*}"
    for _trw_n in "$_v_maj" "$_v_min" "$_f_maj" "$_f_min" "$_c_maj" "$_c_min"; do
        case "$_trw_n" in ''|*[!0-9]*) echo "no"; return ;; esac
    done
    if [ "$_v_maj" -gt "$_f_maj" ] || { [ "$_v_maj" -eq "$_f_maj" ] && [ "$_v_min" -ge "$_f_min" ]; }; then
        if [ "$_v_maj" -lt "$_c_maj" ] || { [ "$_v_maj" -eq "$_c_maj" ] && [ "$_v_min" -lt "$_c_min" ]; }; then
            echo "yes"
            return
        fi
    fi
    echo "no"
}

# Keep the previous venv's torch on a re-run: echo "torch==X.Y.Z" when the probed
# version ($1) is inside the active constraint window ($2), else "". The RELEASE is kept
# regardless of flavor tag; the pin installs from the freshly chosen index, so flavor
# follows the machine (cpu <-> cuda, cu126 -> cu130, PyPI bare -> +cu130) while the
# release follows the user. Gating on flavor was wrong: a PyPI torch reports a BARE
# version (on Linux the PyPI wheel IS CUDA), misclassified "cpu", so a healthy 2.10 on a
# cu130 host was moved to 2.11. Per-leaf floors still win (rocm7.2 / gfx >=2.11 for the
# Strix _grouped_mm fix, out-of-window manual installs) and are never pinned; the caller's
# _PREV_FALLBACK_CONSTRAINT installs the newest supported release when the index lacks the
# exact one. Opt out with UNSLOTH_TORCH_UPGRADE=1.
_previous_torch_pin() {
    _ptp_ver="$1"
    _ptp_con="$2"
    [ -n "$_ptp_ver" ] || { echo ""; return; }
    [ "${UNSLOTH_TORCH_UPGRADE:-0}" = "1" ] && { echo ""; return; }
    _ptp_base="${_ptp_ver%%+*}"
    # Base must be a plain numeric release (X.Y[.Z]); probe noise and
    # nightly/dev/source builds (2.11.0.dev20250704, 2.9.0a0) must never
    # become a pin -- no stable index carries them, so pinning would only
    # print "keeping it" and then burn a doomed resolve before falling back.
    case "$_ptp_base" in
        *[!0-9.]* | *..* | .* | *.) echo ""; return ;;
        [0-9]*.[0-9]*) ;;
        *) echo ""; return ;;
    esac
    [ "$(_torch_release_in_window "$_ptp_base" "$_ptp_con")" = "yes" ] || { echo ""; return; }
    echo "torch==$_ptp_base"
}

# Install torch from TORCH_INDEX_URL honoring a kept-release pin: with _PREV_TORCH_PIN
# set, TORCH_CONSTRAINT is the exact previous release; fall back to the supported range
# if the index lacks it (pruned mirror) rather than failing. Used by every --default-index
# path (NVIDIA cu*, AMD rocm/gfx fallbacks, cpu/mac, ROCm repairs) so preservation is
# uniform. Extra args (e.g. --force-reinstall) are passed through to uv.
_install_torch_default_index() {
    if [ -n "$_PREV_TORCH_PIN" ]; then
        # Pair the companions with the kept torch minor: torchaudio no longer
        # exact-pins torch in its metadata, so leaving it unconstrained resolves
        # a newer mismatched build (a kept torch 2.9.0 pulled torchaudio 2.11.0).
        _itdi_base="${_PREV_TORCH_PIN#torch==}"
        _itdi_minor="${_itdi_base#*.}"
        _itdi_minor="${_itdi_minor%%.*}"
        _itdi_tv="torchvision"
        _itdi_ta="torchaudio"
        case "$_itdi_base" in
            2.*)
                _itdi_tv="torchvision==0.$((_itdi_minor + 15)).*"
                _itdi_ta="torchaudio==2.${_itdi_minor}.*"
                ;;
        esac
        if ! run_install_cmd_retry "install PyTorch (kept release)" uv pip install --python "$_VENV_PY" "$TORCH_CONSTRAINT" "$_itdi_tv" "$_itdi_ta" \
            --default-index "$TORCH_INDEX_URL" "$@"; then
            substep "[WARN] $_PREV_TORCH_PIN is not installable from $(_strip_index_url_credentials "$TORCH_INDEX_URL") -- installing the newest supported release instead" "$C_WARN"
            TORCH_CONSTRAINT="$_PREV_FALLBACK_CONSTRAINT"
            _PREV_TORCH_PIN=""
            run_install_cmd_retry "install PyTorch" uv pip install --python "$_VENV_PY" "$TORCH_CONSTRAINT" "$TORCHVISION_CONSTRAINT" "$TORCHAUDIO_CONSTRAINT" \
                --default-index "$TORCH_INDEX_URL" "$@"
        fi
    else
        run_install_cmd_retry "install PyTorch" uv pip install --python "$_VENV_PY" "$TORCH_CONSTRAINT" "$TORCHVISION_CONSTRAINT" "$TORCHAUDIO_CONSTRAINT" \
            --default-index "$TORCH_INDEX_URL" "$@"
    fi
}

# Expected tag from the index leaf ($1): cuXXX / cpu / rocm (rocmX.Y and gfx* ->
# rocm). Empty on an unknown leaf (odd mirror) so the repair safely no-ops.
_expected_torch_flavor_tag() {
    _leaf=$(_torch_index_url_leaf "$1")
    case "$_leaf" in
        cu[0-9]*)
            # Exact cu + digits only; a cu*-suffixed leaf (cu128-private) -> "" (custom),
            # else a correct +cu128 wheel is force-reinstalled every run.
            case "${_leaf#cu}" in
                *[!0-9]*) echo "" ;;
                *)        echo "$_leaf" ;;
            esac
            ;;
        cpu)          echo "cpu" ;;
        # Exact rocm/gfx families only; a custom rocm*-suffixed leaf -> "" (custom).
        *)
            if _is_pip_rocm_family_leaf "$_leaf"; then echo "rocm"; else echo ""; fi
            ;;
    esac
}

# Whether index ($1) supports a plain --default-index reinstall. pytorch.org cuXXX /
# rocmX.Y AND the repo.amd.com gfx* indexes are all PEP 503 simple indexes that uv
# resolves (torch + every transitive dep) via --default-index -- the same URLs the
# fresh-install paths above already use -- so a stale wheel is auto-repairable.
# Unknown/odd-mirror leaves -> no, so we warn rather than risk a wrong reinstall.
_torch_index_repairable() {
    _leaf=$(_torch_index_url_leaf "$1")
    case "$_leaf" in
        cu[0-9]*) echo "yes" ;;
        # Only EXACT rocm/gfx families resolve via --default-index; a suffixed leaf is verbatim.
        *)
            if _is_pip_rocm_family_leaf "$_leaf"; then echo "yes"; else echo "no"; fi
            ;;
    esac
}

# Remove credentials from a wheel index URL ($1) so an authenticated pin never leaks:
# drops userinfo AND query/fragment; scheme/host/path stay exact. Shared with py / ps1.
_strip_index_url_credentials() {
    _sic_url="$1"
    case "$_sic_url" in
        *://*) ;;
        *) printf '%s' "$_sic_url"; return ;;
    esac
    _sic_scheme="${_sic_url%%://*}"
    _sic_rest="${_sic_url#*://}"
    # Drop query / fragment (may hold auth tokens).
    _sic_rest="${_sic_rest%%\?*}"
    _sic_rest="${_sic_rest%%#*}"
    _sic_auth="${_sic_rest%%/*}"
    # Drop user:pass@ userinfo if present.
    case "$_sic_auth" in
        *@*) _sic_host="${_sic_auth##*@}" ;;
        *)   _sic_host="$_sic_auth" ;;
    esac
    if [ "$_sic_auth" = "$_sic_rest" ]; then
        printf '%s://%s' "$_sic_scheme" "$_sic_host"
    else
        printf '%s://%s/%s' "$_sic_scheme" "$_sic_host" "${_sic_rest#*/}"
    fi
}

get_radeon_wheel_url() {
    # Only meaningful on Linux. Picks a repo.radeon.com base URL whose listing
    # contains torch wheels. Tries paths like rocm-rel-7.2.1/, rocm-rel-7.2/,
    # rocm-rel-7.1.1/, rocm-rel-7.1/ (AMD publishes both M.m and M.m.p dirs).
    # Accepts both X.Y and X.Y.Z host versions since /opt/rocm/.info/version
    # and hipconfig --version can return either shape.
    case "$(uname -s)" in Linux) ;; *) echo ""; return ;; esac

    # Detect ROCm version (X.Y or X.Y.Z) -- try amd-smi, then
    # /opt/rocm/.info/version, then hipconfig.
    _full_ver=""
    _full_ver=$({ command -v amd-smi >/dev/null 2>&1 && \
        amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
            'NF>1{if(match($2,/[0-9]+\.[0-9]+(\.[0-9]+)?/)){print substr($2,RSTART,RLENGTH); ok=1; exit}} END{exit !ok}'; } || \
        { [ -r /opt/rocm/.info/version ] && \
            awk 'match($0,/[0-9]+\.[0-9]+(\.[0-9]+)?/){print substr($0,RSTART,RLENGTH); found=1; exit} END{exit !found}' /opt/rocm/.info/version; } || \
        { command -v hipconfig >/dev/null 2>&1 && \
            hipconfig --version 2>/dev/null | awk 'NR==1 && match($0,/[0-9]+\.[0-9]+(\.[0-9]+)?/){print substr($0,RSTART,RLENGTH); found=1} END{exit !found}'; }) 2>/dev/null

    # Validate: must be X.Y or X.Y.Z with X >= 1
    case "$_full_ver" in
        [1-9]*.[0-9]*.[0-9]*) : ;;  # X.Y.Z
        [1-9]*.[0-9]*) : ;;          # X.Y
        *) echo ""; return ;;
    esac
    echo "https://repo.radeon.com/rocm/manylinux/rocm-rel-${_full_ver}/"
}

# ── Radeon repo wheel selection helpers ──────────────────────────────────────
# Fetches the Radeon repo directory listing once into _RADEON_LISTING (global).
# _RADEON_PYTAG holds the CPython tag for the running interpreter (e.g. cp312).
# _RADEON_BASE_URL holds the base URL for relative-href resolution.
_RADEON_LISTING=""
_RADEON_PYTAG=""
_RADEON_BASE_URL=""

_radeon_fetch_listing() {
    # Usage: _radeon_fetch_listing BASE_URL
    # Populates _RADEON_LISTING, _RADEON_PYTAG, _RADEON_BASE_URL.
    _RADEON_BASE_URL="$1"
    _RADEON_PYTAG=$("$_VENV_PY" -c "
import sys
print('cp{}{}'.format(sys.version_info.major, sys.version_info.minor))
" 2>/dev/null) || return 1
    if command -v curl >/dev/null 2>&1; then
        _RADEON_LISTING=$(curl -fsSL --max-time 20 "$_RADEON_BASE_URL" 2>/dev/null)
    elif command -v wget >/dev/null 2>&1; then
        _RADEON_LISTING=$(wget -qO- --timeout=20 "$_RADEON_BASE_URL" 2>/dev/null)
    fi
    [ -n "$_RADEON_LISTING" ] || return 1
}

_pick_radeon_wheel() {
    # Usage: _pick_radeon_wheel PACKAGE_NAME [VERSION_PREFIX]
    # Scans $_RADEON_LISTING for the newest wheel whose filename starts exactly
    # with PACKAGE_NAME- (and optionally VERSION_PREFIX) and matches _RADEON_PYTAG + linux_x86_64.
    # Prints the full URL (resolving relative hrefs against _RADEON_BASE_URL).
    #
    # POSIX-compliant pipeline: all href parsing, filtering, and version
    # selection is done inside a single awk script rather than reaching
    # for GNU extensions (grep -o, sort -V) that would break under BSD
    # or BusyBox coreutils.
    _pkg="$1"
    _ver_prefix="${2:-}"
    [ -n "$_RADEON_LISTING" ] || return 1
    [ -n "$_RADEON_PYTAG"   ] || return 1
    _tag="$_RADEON_PYTAG"
    _href=$(printf '%s\n' "$_RADEON_LISTING" \
        | awk -v pkg="$_pkg" -v tag="$_tag" -v ver_prefix="$_ver_prefix" '
            BEGIN { max_pad = ""; max_url = "" }
            {
                line = $0
                while (match(line, /href="[^"]*"/)) {
                    # Strip the leading href=" (6 chars) and trailing " (1 char)
                    url = substr(line, RSTART + 6, RLENGTH - 7)
                    line = substr(line, RSTART + RLENGTH)

                    # Extract basename, strip query / fragment
                    n = split(url, p, "/")
                    base = p[n]
                    sub(/[?#].*/, "", base)

                    prefix = pkg "-" ver_prefix
                    # Match cpXY-cpXY or cpXY-abi3 with any linux x86_64
                    # platform tag (linux_x86_64, manylinux_2_28_x86_64,
                    # manylinux2014_x86_64, etc.)
                    if (substr(base, 1, length(prefix)) == prefix &&
                            index(base, "-" tag "-") > 0 &&
                            match(base, /x86_64\.whl$/)) {
                        # Extract the version component (first
                        # dotted-number run) and pad each piece so a
                        # plain lexical comparison gives us the newest.
                        if (match(base, /[0-9]+\.[0-9]+(\.[0-9]+)?/)) {
                            ver = substr(base, RSTART, RLENGTH)
                            m = split(ver, v, ".")
                            pad = ""
                            for (i = 1; i <= m; i++)
                                pad = pad sprintf("%08d", v[i])
                            if (pad > max_pad) {
                                max_pad = pad
                                max_url = url
                            }
                        }
                    }
                }
            }
            END { if (max_url != "") print max_url }')
    [ -z "$_href" ] && return 1
    case "$_href" in
        http*) printf '%s\n' "$_href" ;;
        *)     printf '%s\n' "${_RADEON_BASE_URL%/}/${_href#/}" ;;
    esac
}

# ── ROCm-on-WSL bootstrap for AMD Strix Halo (gfx1151) ───────────────────────
# No-op everywhere except: WSL + GPU wanted + no usable GPU yet + /dev/dxg +
# Strix Halo APU. Every other config (NVIDIA, native-Linux ROCm, macOS, Windows,
# CPU, non-Strix WSL) skips it and normal detection runs unchanged. NEVER aborts
# the installer -- always returns 0. Runs the idempotent helper (ROCm 7.2 +
# librocdxg), then sources the env it persisted so detection finds the GPU.
# Export the ROCm-on-WSL env into this process and persist it to /etc/profile.d
# so non-login Unsloth/llama launches inherit it. Idempotent (writes only when
# the drop-in is missing); no-op without librocdxg, so never fires off WSL.
# /etc/profile.d is root-owned -- sudo-tee when not root, else ROCm vanishes
# after this shell on a non-root reinstall. Best-effort either way.
_persist_rocm_wsl_dropin() {
    [ -e /opt/rocm/lib/librocdxg.so ] || [ -e /opt/rocm/lib64/librocdxg.so ] || return 0
    _rw_rocm=/opt/rocm
    export HSA_ENABLE_DXG_DETECTION=1
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    case ":${PATH}:" in
        *":${_rw_rocm}/bin:"*) ;;
        *) export PATH="${_rw_rocm}/bin:${PATH}" ;;
    esac
    export LD_LIBRARY_PATH="${_rw_rocm}/lib:${LD_LIBRARY_PATH:-}"
    [ -r /etc/profile.d/unsloth-rocm-wsl.sh ] && return 0
    _rw_dropin="$(
        printf '# >>> Unsloth ROCm-on-WSL (gfx1151) >>>\n'
        printf 'export HSA_ENABLE_DXG_DETECTION=1\n'
        printf 'export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1\n'
        printf 'export PATH="%s/bin:${PATH}"\n' "${_rw_rocm}"
        printf 'export LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n' "${_rw_rocm}"
        printf '# <<< Unsloth ROCm-on-WSL (gfx1151) <<<\n'
    )"
    if [ "$(id -u)" = "0" ]; then
        printf '%s\n' "$_rw_dropin" > /etc/profile.d/unsloth-rocm-wsl.sh 2>/dev/null || true
    elif command -v sudo >/dev/null 2>&1; then
        printf '%s\n' "$_rw_dropin" | sudo tee /etc/profile.d/unsloth-rocm-wsl.sh >/dev/null 2>&1 || true
    fi
}

# _wsl_amd_gpu_name is defined earlier so both the reroute and this bootstrap can use it.
_maybe_bootstrap_rocm_wsl() {
    [ "${OS:-}" = "wsl" ] || return 0
    [ "${SKIP_TORCH:-false}" = "false" ] || return 0
    [ "${UNSLOTH_SKIP_ROCM_WSL_SETUP:-0}" = "1" ] && return 0
    # Leave any already-usable GPU completely alone (NVIDIA, or working ROCm).
    if _has_usable_nvidia_gpu; then return 0; fi
    # Usable ROCm = rocminfo enumerates a real GPU agent: gfx[1-9] (excludes gfx000,
    # the CPU agent) and not the "gfx11-generic" fallback. awk consumes all input so
    # rocminfo isn't SIGPIPE'd like `grep -q` under pipefail.
    _ensure_rocm_probe_env
    if command -v rocminfo >/dev/null 2>&1 && \
       rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx[1-9]/ && !/generic/{found=1} END{exit !found}'; then
        # rocminfo may work only via the transient env _ensure_rocm_probe_env
        # just set, which dies with the installer. Persist the drop-in so login
        # shells (Unsloth, llama.cpp) inherit it -- else a reinstall over an
        # existing /opt/rocm (uninstall keeps ROCm but drops it) loses the GPU.
        _persist_rocm_wsl_dropin
        return 0
    fi
    # WSL GPU passthrough device must exist (present on any WSL2 GPU host).
    [ -e /dev/dxg ] || return 0
    # Strix APUs show in /proc/cpuinfo (the CPU model); discrete cards don't, so also
    # ask the Windows host. Either signal suffices; the bootstrap detects arch from rocminfo.
    if ! grep -qiE 'Ryzen AI Max|Radeon 80[0-9][05]S|Strix Halo' /proc/cpuinfo 2>/dev/null \
       && ! _wsl_amd_gpu_name >/dev/null 2>&1; then
        return 0
    fi
    command -v bash >/dev/null 2>&1 || return 0

    # Fast path: already configured (librocdxg present) but launched from a
    # non-login shell so the persisted env wasn't loaded -- just load it.
    if [ -e /opt/rocm/lib/librocdxg.so ] || [ -e /opt/rocm/lib64/librocdxg.so ]; then
        if [ -r /etc/profile.d/unsloth-rocm-wsl.sh ]; then
            # shellcheck disable=SC1091
            . /etc/profile.d/unsloth-rocm-wsl.sh || true
        else
            # librocdxg present but the env drop-in is gone (e.g. an Unsloth
            # uninstall removed it while keeping shared ROCm). Restore the env.
            _persist_rocm_wsl_dropin
        fi
        return 0
    fi

    echo ""
    _rw_gpu="$(_wsl_amd_gpu_name 2>/dev/null || true)"; [ -n "$_rw_gpu" ] || _rw_gpu="an AMD GPU"
    substep "Detected ${_rw_gpu} in WSL with no ROCm runtime yet." "$C_WARN"
    substep "Setting up ROCm-on-WSL (ROCm 7.2 + librocdxg) automatically to enable this GPU."
    substep "One-time, uses sudo and a large download. (skip: re-run with UNSLOTH_SKIP_ROCM_WSL_SETUP=1)"

    # Locate the helper: prefer the copy shipped beside install.sh, else fetch it.
    _rw_helper="${_REPO_ROOT:-.}/scripts/install_rocm_wsl_strixhalo.sh"
    _rw_tmp=""
    if [ ! -r "$_rw_helper" ]; then
        _rw_tmp="$(mktemp 2>/dev/null || echo /tmp/_unsloth_rocm_wsl.sh)"
        if download "https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/install_rocm_wsl_strixhalo.sh" "$_rw_tmp" 2>/dev/null; then
            _rw_helper="$_rw_tmp"
        else
            substep "Could not fetch the ROCm-on-WSL helper; using CPU fallback." "$C_WARN"
            [ -n "$_rw_tmp" ] && rm -f "$_rw_tmp"
            return 0
        fi
    fi

    # Consent: the narrow guarded case is exactly the GPU setup the user ran the
    # installer for, so it proceeds AUTOMATICALLY by default (works with no TTY,
    # e.g. `curl ... | sh`). Opt out via UNSLOTH_SKIP_ROCM_WSL_SETUP=1 (top of
    # function). The Tauri app drives its own consent UI, so under TAURI_MODE it
    # only runs when the app passes UNSLOTH_ROCM_WSL_AUTO=1; else surface and wait.
    _rw_go=1
    if [ "${TAURI_MODE:-false}" = "true" ] && [ "${UNSLOTH_ROCM_WSL_AUTO:-0}" != "1" ]; then
        tauri_log "ROCM_WSL_AVAILABLE" "strixhalo"
        substep "Enable the GPU from the desktop app (or set UNSLOTH_ROCM_WSL_AUTO=1)." "$C_WARN"
        _rw_go=0
    fi

    if [ "$_rw_go" = "1" ]; then
        # Helper does its own sudo + is idempotent. SMOKE_TEST=0: install.sh
        # installs torch itself right after, into the real venv.
        if UNSLOTH_WSL_SMOKE_TEST=0 bash "$_rw_helper"; then
            # Pull the helper's persisted env into THIS shell so detection
            # (rocminfo) now enumerates the GPU and routes to gfx1151.
            if [ -r /etc/profile.d/unsloth-rocm-wsl.sh ]; then
                # shellcheck disable=SC1091
                . /etc/profile.d/unsloth-rocm-wsl.sh || true
            fi
            substep "ROCm-on-WSL ready; continuing with GPU install." "$C_OK"
        else
            substep "ROCm-on-WSL setup did not complete; falling back to CPU-only." "$C_WARN"
        fi
    fi
    [ -n "$_rw_tmp" ] && rm -f "$_rw_tmp"
    return 0
}
# When the caller pins the wheel index (UNSLOTH_TORCH_INDEX_URL / _FAMILY), honour it
# everywhere: skip the WSL ROCm bootstrap and the Radeon/Strix reroute below (which would
# re-probe the GPU and overwrite the pin). Trim whitespace first (parity with
# get_torch_index_url): a whitespace-only override is unset there, so must not flip this true.
_torch_index_pinned=false
_ti_url_trim="${UNSLOTH_TORCH_INDEX_URL:-}"
_ti_url_trim="${_ti_url_trim#"${_ti_url_trim%%[![:space:]]*}"}"; _ti_url_trim="${_ti_url_trim%"${_ti_url_trim##*[![:space:]]}"}"
_ti_family_trim="${UNSLOTH_TORCH_INDEX_FAMILY:-}"
_ti_family_trim="${_ti_family_trim#"${_ti_family_trim%%[![:space:]]*}"}"; _ti_family_trim="${_ti_family_trim%"${_ti_family_trim##*[![:space:]]}"}"
if [ -n "$_ti_url_trim" ] || [ -n "$_ti_family_trim" ]; then
    _torch_index_pinned=true
fi
[ "$_torch_index_pinned" = true ] || _maybe_bootstrap_rocm_wsl || true

TORCH_INDEX_URL=$(get_torch_index_url)

# Export the resolved torch backend ("cuda", "rocm", or "cpu") so that
# downstream scripts (setup.sh -> install_python_stack.py) know what was
# chosen here and can skip ROCm-specific repair steps on CUDA/CPU hosts.
# Classify on the FINAL path segment only: a custom UNSLOTH_PYTORCH_MIRROR
# whose base path happens to contain "rocm" or "gfx" must not mislabel a
# cu*/cpu index as ROCm (radeon repo URLs end in rocm-rel-X.Y/, Strix
# overrides in gfxNNNN/, so the trailing slash is stripped first).
# Lowercase the leaf so every gfx*/rocm*/cu* arm matches regardless of case (canonical AMD
# RDNA4 leaf is gfx120X-all). CUDA is branded only on a real cu[0-9]* leaf, so a mirror
# leaf (/current) does NOT commit a CUDA backend; an unknown leaf leaves the var unset so
# the stack probes the GPU. Query/fragment dropped first, then ALL trailing slashes (in
# lockstep with the shared _torch_index_url_leaf extractor).
_torch_index_leaf="${TORCH_INDEX_URL%%\?*}"
_torch_index_leaf="${_torch_index_leaf%%#*}"
# Strip ALL trailing slashes, not one: .../cu128// must yield cu128, not an empty leaf.
while [ -n "$_torch_index_leaf" ] && [ "${_torch_index_leaf%/}" != "$_torch_index_leaf" ]; do
    _torch_index_leaf="${_torch_index_leaf%/}"
done
_torch_index_leaf="${_torch_index_leaf##*/}"
_torch_index_leaf=$(printf '%s' "$_torch_index_leaf" | tr '[:upper:]' '[:lower:]')
case "$_torch_index_leaf" in
    rocm*|gfx*) export UNSLOTH_TORCH_BACKEND="rocm" ;;
    cpu)        export UNSLOTH_TORCH_BACKEND="cpu"  ;;
    cu[0-9]*)   export UNSLOTH_TORCH_BACKEND="cuda" ;;
    # Unknown leaf (odd mirror, /current): unset so a stale inherited value can't leak and
    # the stack probes the GPU.
    *)          unset UNSLOTH_TORCH_BACKEND ;;
esac

# Whether TORCH_INDEX_URL names an actual pip ROCm family (rocm<digit>* / gfx*), gating the
# ROCm-only side effects below (AMD bitsandbytes, ROCm-torch repair). Digit-gated so a leaf
# merely STARTING with "rocm" isn't force-repaired from the wrong path.
if _is_pip_rocm_family_leaf "$_torch_index_leaf"; then
    _torch_index_is_rocm_family=true
else
    _torch_index_is_rocm_family=false
fi

# rocm7.2 and the per-gfx indexes with the _grouped_mm <2.11 bug (gfx120X-all, gfx1151,
# gfx1150) ship torch 2.11.0 -- raise the floor (also covers a pinned override that skipped
# the Strix reroute). Pin the companions too: the per-gfx index publishes them independently
# and a bare name can resolve a 2.12 ABI-mismatched wheel. Match on the FINAL leaf so a
# custom mirror with a gfx/rocm7.2 path segment but a cu*/cpu family isn't forced.
case "$_torch_index_leaf" in
    rocm7.2|gfx120x-all|gfx1151|gfx1150)
        TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
        TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
        TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
        ;;
    # CUDA cu12x/cu13x indexes ship torch 2.11.x: widen the ceiling to <2.12.0 (matches
    # _CUDA_TORCH_PKG_SPEC) and widen the companions with it so the trio stays paired.
    cu[0-9]*)
        TORCH_CONSTRAINT="torch>=2.4,<2.12.0"
        TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.27.0"
        TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<2.12.0"
        ;;
esac

# A pinned custom/unknown-leaf index (/simple, /current, /cu128-private) has no curated
# companion set, so bound torchvision/torchaudio to the same <2.11 range the Python path pins
# (else a mirror with newer companions resolves a 2.12 ABI-mismatched wheel). Known families
# keep their curated companions above (_expected_torch_flavor_tag returns "" only for custom).
if [ "$_torch_index_pinned" = true ] && \
   [ -z "$(_expected_torch_flavor_tag "$TORCH_INDEX_URL")" ]; then
    TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.26.0"
    TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<2.11.0"
fi

# Auto-detect GPU for AMD ROCm based
# get_torch_index_url must have chosen */rocm*
# (gfx in rocminfo or amd-smi list). Then require rocminfo "Marketing Name:.*Radeon".
# Skipped when the index is pinned: an explicit override must not be rerouted to the
# Radeon/Strix repos by GPU probing.
_amd_gpu_radeon=false
if [ "$_torch_index_pinned" = false ]; then
case "$TORCH_INDEX_URL" in
    */rocm*)
        if _has_amd_rocm_gpu && command -v rocminfo >/dev/null 2>&1 && \
           rocminfo 2>/dev/null | grep -q 'Marketing Name:.*Radeon'; then
            _amd_gpu_radeon=true
        fi
        ;;
esac
# 0 when a rocmX.Y index leaf ($1, the final path segment) is older than floor
# $2.$3 (int compare, so rocm7.2 < rocm7.13). Non-rocm leaves (gfx*, cu*, cpu) and
# non-numeric versions return 1. Leaf-based (like $_torch_index_leaf) so a mirror
# base holding its own rocm token compares the family leaf, not the base path.
_rocm_leaf_below() {
    case "$1" in rocm[0-9]*.[0-9]*) : ;; *) return 1 ;; esac
    _rb=${1#rocm}; _maj=${_rb%%.*}; _min=${_rb#*.}; _min=${_min%%.*}
    case "$_maj$_min" in *[!0-9]*) return 1 ;; esac
    if [ "$_maj" -lt "$2" ]; then return 0; fi
    if [ "$_maj" -eq "$2" ] && [ "$_min" -lt "$3" ]; then return 0; fi
    return 1
}
# ── Strix Halo / Strix Point: route to the AMD arch-specific index ───────────
# gfx1151/gfx1150 need torch 2.11+rocm7.13 from repo.amd.com/rocm/whl/gfx<arch>/,
# which carries AMD's real fixes (the rocm7.1 _grouped_mm segfault, moe_utils.py:167,
# and later Strix kernel bugs). Every generic pytorch.org index below rocm7.13 lacks
# them (and the Radeon repo can be offline, unslothai#7264), so reroute a detected
# Strix GPU whenever the picked index is older than the arch build -- covers today's
# rocm6.0-7.2 and any future 7.x < 7.13; rocm7.13+ already has the fixes, so leave it.
case "$_torch_index_leaf" in
    rocm[0-9]*)
        # Collect every gfx token in rocminfo / amd-smi enumeration order
        # (skip duplicates), then index by HIP_VISIBLE_DEVICES /
        # ROCR_VISIBLE_DEVICES so a mixed Strix iGPU + non-Strix dGPU box
        # where the user selected the dGPU does NOT get rerouted to the
        # Strix per-gfx index.
        # || true on each probe: no gfx match makes grep exit 1, which under
        # set -euo pipefail would abort the installer before the next fallback
        # runs (now that the case matches every rocm* index, not just rocm7.1).
        _gfx_all=""
        if command -v rocminfo >/dev/null 2>&1; then
            _gfx_all=$(rocminfo 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        fi
        if [ -z "$_gfx_all" ] && command -v amd-smi >/dev/null 2>&1; then
            _gfx_all=$(amd-smi list 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
            # PowerShell paths also probe `amd-smi static --asic`; mirror it
            # so a host with hipinfo-less amd-smi reports the gfx target.
            if [ -z "$_gfx_all" ]; then
                _gfx_all=$(amd-smi static --asic 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
            fi
        fi
        _runtime_gfx=""
        if [ -n "$_gfx_all" ]; then
            _vis="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}"
            _idx=0
            if [ -n "$_vis" ] && [ "$_vis" != "-1" ]; then
                _first=${_vis%%,*}
                case "$_first" in
                    ''|*[!0-9]*) _idx=0 ;;
                    *) _idx=$_first ;;
                esac
            fi
            _runtime_gfx=$(printf '%s\n' "$_gfx_all" | awk -v idx="$_idx" '
                NF && !seen[$0]++ { vals[n++] = $0 }
                END {
                    if (idx < 0 || idx >= n) idx = 0
                    if (n > 0) print vals[idx]
                }')
        fi
        _strix_gfx=""
        case "$_runtime_gfx" in
            gfx1151|gfx1150) _strix_gfx="$_runtime_gfx" ;;
        esac
        # Skip rocm7.13+ generic indexes: they already ship the fixes, so the
        # arch build (rocm7.13) would be a downgrade rather than a rescue.
        if [ -n "$_strix_gfx" ] && _rocm_leaf_below "$_torch_index_leaf" 7 13; then
            echo "" >&2
            echo "  [WARN] $_strix_gfx (Strix) detected -- routing to the AMD arch-specific index" >&2
            echo "  [WARN] torch 2.11+rocm7.13 has AMD's real gfx1150/gfx1151 fixes (the ROCm 7.1" >&2
            echo "  [WARN] _grouped_mm segfault, moe_utils.py:167, and later Strix kernel bugs)," >&2
            echo "  [WARN] and is more reliable than the rocm7.2 index or an offline Radeon repo." >&2
            echo "" >&2
            # AMD's arch-specific index serves torch 2.11.0+rocm7.13.0 which has AMD's
            # actual fix for the gfx1151/gfx1150 _grouped_mm kernel bug -- preferred
            # over the pytorch.org rocm7.2 fallback because it exercises the real GPU
            # kernel path. Set UNSLOTH_AMD_ROCM_MIRROR to override for air-gapped installs.
            _amd_strix_base="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}"
            # Strip ALL trailing slashes to match Python's .rstrip("/") -- a
            # double-/triple-slash mirror URL would otherwise produce 404s on
            # strict pip proxies (artifactory, sonatype).
            while [ "${_amd_strix_base%/}" != "$_amd_strix_base" ]; do
                _amd_strix_base="${_amd_strix_base%/}"
            done
            TORCH_INDEX_URL="${_amd_strix_base}/${_strix_gfx}/"
            TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
            # Pin companions to 2.11 (per-gfx index publishes them independently).
            TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
            TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
            _amd_gpu_radeon=false
        fi
        ;;
esac
fi  # _torch_index_pinned guard (Radeon + Strix reroute)
# Re-run over an existing install: keep the previous venv's torch RELEASE; the fresh
# index above supplies the right flavor for this machine. Evaluated HERE, after every
# index/constraint decision including the Strix reroute, so the window checked is the
# final one and a raised floor (rocm7.2 / Strix gfx) rejects an older release.
# _PREV_FALLBACK_CONSTRAINT keeps the range so the install can fall back when the exact
# release is not on the chosen index (mirrors may prune old wheels). Skipped for --no-torch.
_PREV_TORCH_PIN=""
_PREV_FALLBACK_CONSTRAINT="$TORCH_CONSTRAINT"
if [ "$SKIP_TORCH" = false ]; then
    _prev_pin=$(_previous_torch_pin "$_PREV_TORCH_VER" "$TORCH_CONSTRAINT")
    if [ -n "$_prev_pin" ]; then
        _PREV_TORCH_PIN="$_prev_pin"
        TORCH_CONSTRAINT="$_prev_pin"
        substep "existing install has torch $_PREV_TORCH_VER -- keeping it (set UNSLOTH_TORCH_UPGRADE=1 to get the newest release)"
    fi
fi

_TAURI_TORCH_INDEX_FAMILY=$(_tauri_torch_index_family "$TORCH_INDEX_URL")
if [ "$_amd_gpu_radeon" = true ] && [ "$SKIP_TORCH" = false ]; then
    _TAURI_TORCH_INDEX_FAMILY="radeon"
fi
_TAURI_GPU_BRANCH=$(_tauri_gpu_branch "$_TAURI_TORCH_INDEX_FAMILY" "$_amd_gpu_radeon")
tauri_diag_marker "$_TAURI_GPU_BRANCH" "$_TAURI_TORCH_INDEX_FAMILY"

# ── GPU detection summary (mirrors install.ps1 step "gpu" block) ──
if _has_usable_nvidia_gpu; then
    step "gpu" "NVIDIA GPU detected"
elif case "$TORCH_INDEX_URL" in */rocm*|*/gfx*) true ;; *) false ;; esac; then
    # Probe gfx arch for the display label, honouring HIP_VISIBLE_DEVICES
    _ensure_rocm_probe_env
    _gpu_disp_gfx_all=""
    _gpu_disp_mkt=""
    if command -v rocminfo >/dev/null 2>&1; then
        _gpu_disp_gfx_all=$(rocminfo 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        _gpu_disp_mkt=$(rocminfo 2>/dev/null | awk -F': ' \
            '/Marketing Name:/{gsub(/^[[:space:]]+|[[:space:]]+$/,"", $2); if($2){print $2; exit}}' || true)
    fi
    if [ -z "$_gpu_disp_gfx_all" ] && command -v amd-smi >/dev/null 2>&1; then
        _gpu_disp_gfx_all=$(amd-smi list 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        [ -z "$_gpu_disp_gfx_all" ] && \
            _gpu_disp_gfx_all=$(amd-smi static --asic 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
    fi
    if [ -z "$_gpu_disp_mkt" ] && command -v amd-smi >/dev/null 2>&1; then
        _gpu_disp_mkt=$(amd-smi static --asic 2>/dev/null | awk -F'[:|]' \
            '/[Mm]arket.?[Nn]ame/{gsub(/^[[:space:]]+|[[:space:]]+$/,"", $2); if($2){print $2; exit}}' || true)
    fi
    _gpu_vis="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}"
    _gpu_vis_idx=0
    if [ -n "$_gpu_vis" ] && [ "$_gpu_vis" != "-1" ]; then
        _gpu_first="${_gpu_vis%%,*}"
        case "$_gpu_first" in ''|*[!0-9]*) ;; *) _gpu_vis_idx=$_gpu_first ;; esac
    fi
    _gpu_disp_gfx=$(printf '%s\n' "$_gpu_disp_gfx_all" | awk -v idx="$_gpu_vis_idx" \
        'NF && !seen[$0]++ { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx] }')
    # UNSLOTH_ROCM_GFX_ARCH env override (mirrors install.ps1)
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        _gpu_disp_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
        substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $_gpu_disp_gfx"
    # Name-based arch inference when tools don't report gfx (mirrors install.ps1 nameArchTable)
    elif [ -z "$_gpu_disp_gfx" ] && [ -n "$_gpu_disp_mkt" ]; then
        # Kept in sync with the nameArchTable in install.ps1 / setup.ps1.
        # gfx1102 matched BEFORE gfx1100 so the spaceless "RX 7700S" lands on
        # gfx1102 (bash case has no negative lookahead like the PS tables).
        case "$_gpu_disp_mkt" in
            *"9070 XT"*|*9080*)                                                                            _gpu_disp_gfx="gfx1201" ;;  # RDNA 4
            *9070*|*9060*)                                                                                 _gpu_disp_gfx="gfx1200" ;;  # RDNA 4
            *"8065S"*|*"8060S"*|*"8050S"*|*"8040S"*|*"Strix Halo"*|*"Ryzen AI Max"*|*"AI Max"*) _gpu_disp_gfx="gfx1151" ;;  # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
            *"890M"*|*"880M"*|*"860M"*|*"840M"*|*"Strix Point"*|*"Krackan"*|*"HX 37"*|*"AI 9 HX"*|*"AI 9 36"*|*"AI 7 35"*|*"AI 5 34"*|*"AI 7 PRO 35"*|*"AI 5 33"*) _gpu_disp_gfx="gfx1150" ;;  # RDNA 3.5 (Strix/Krackan Point: Radeon 890M/880M iGPU, Ryzen AI 9 HX 370/375)
            *"RX 7600"*|*"RX 7700S"*|*"RX 7650"*|*"PRO W7600"*|*"PRO W7500"*|*"PRO V710"*)                  _gpu_disp_gfx="gfx1102" ;;  # RDNA 3 (Navi 33)
            *"RX 7900"*|*"RX 7800"*|*"RX 7700"*|*"PRO W7900"*|*"PRO W7800"*|*"PRO W7700"*)                  _gpu_disp_gfx="gfx1100" ;;  # RDNA 3 desktop / workstation (Navi 31)
            *"780M"*|*"760M"*|*"740M"*|*"Phoenix"*|*"Hawk Point"*|*"Z1 Extreme"*|*"Z2 Extreme"*)            _gpu_disp_gfx="gfx1103" ;;  # RDNA 3 iGPU (Phoenix / Hawk Point)
            *"RX 6900"*|*"RX 6800"*|*"RX 6750"*|*"RX 6700"*|*"PRO W6800"*|*"PRO W6900"*)                    _gpu_disp_gfx="gfx1030" ;;  # RDNA 2 (Navi 21)
            *"RX 6650"*|*"RX 6600"*|*"PRO W6600"*|*"PRO W6650"*)                                            _gpu_disp_gfx="gfx1032" ;;  # RDNA 2 (Navi 23)
            *"RX 6500"*|*"RX 6400"*|*"RX 6300"*|*"PRO W6400"*|*"PRO W6500"*)                                _gpu_disp_gfx="gfx1034" ;;  # RDNA 2 (Navi 24)
        esac
        if [ -n "$_gpu_disp_gfx" ]; then
            substep "gfx arch inferred from GPU name: $_gpu_disp_gfx"
            substep "Tip: set UNSLOTH_ROCM_GFX_ARCH=$_gpu_disp_gfx to skip inference next time"
        fi
    fi
    # ROCm version via hipconfig, then amd-smi
    _gpu_rocm_ver=""
    if command -v hipconfig >/dev/null 2>&1; then
        _gpu_rocm_ver=$(hipconfig --version 2>/dev/null | awk 'NR==1 && /^[0-9]/{print; exit}' || true)
    fi
    if [ -z "$_gpu_rocm_ver" ] && command -v amd-smi >/dev/null 2>&1; then
        _gpu_rocm_ver=$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
            'NF>1{gsub(/[[:space:]]/,"", $2); print $2; exit}' || true)
    fi
    if [ -n "$_gpu_disp_gfx" ]; then
        step "gpu" "AMD ROCm ($_gpu_disp_gfx)"
    else
        step "gpu" "AMD ROCm"
    fi
    _rocm_root="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
    substep "ROCm: $_rocm_root"
    [ -n "$_gpu_rocm_ver" ] && substep "hipconfig: $_gpu_rocm_ver"
    [ -n "$_gpu_disp_mkt" ] && [ -n "$_gpu_disp_gfx" ] && substep "GPU: $_gpu_disp_mkt"
elif [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    # Apple Silicon: PyTorch gets Metal (MPS) acceleration over unified memory, so not CPU-only.
    step "gpu" "Apple Silicon (Metal, unified memory)"
elif _has_amd_rocm_gpu; then
    # AMD GPU visible to the kernel but the torch index stayed CPU: no usable
    # ROCm userspace to pick a wheel. "none" would repeat the false diagnosis
    # this installer used to give.
    step "gpu" "AMD GPU (no usable ROCm -- CPU fallback)" "$C_WARN"
else
    step "gpu" "none (CPU-only)" "$C_WARN"
fi

# ── PyTorch wheel index note ──
case "$TORCH_INDEX_URL" in
    */cpu)
        if [ "$SKIP_TORCH" = false ] && [ "$OS" != "macos" ]; then
            if _has_amd_rocm_gpu; then
                substep "AMD GPU detected, but no usable ROCm/HIP install -- installing CPU-only PyTorch." "$C_WARN"
                substep "Install the ROCm/HIP SDK and re-run this installer for GPU PyTorch." "$C_WARN"
            else
                substep "No GPU detected -- installing CPU-only PyTorch." "$C_WARN"
            fi
            if [ "$OS" = "wsl" ]; then
                # WSL + no GPU detected (detection above found nothing). Common
                # cause: an AMD GPU whose ROCm-on-WSL runtime isn't exposed yet --
                # /dev/dxg present (graphics) but no ROCm runtime.
                _wsl_ubu_ver=""
                [ -r /etc/os-release ] && _wsl_ubu_ver=$(. /etc/os-release 2>/dev/null; printf '%s' "${VERSION_ID:-}")
                if [ -e /dev/dxg ]; then
                    substep "A GPU is plumbed into WSL (/dev/dxg) but no ROCm runtime is exposed to it." "$C_WARN"
                fi
                substep "For an AMD GPU, ROCm-on-WSL currently needs ALL of:"
                substep "  1. AMD Adrenalin Edition 26.1.1+ on Windows (26.2.2+ for Strix Halo / Ryzen AI Max+)."
                substep "     Older drivers lack production ROCDXG/WSL support, so ROCm can't see the GPU."
                substep "     Get it from AMD (open in a browser -- direct downloads are referrer-gated):"
                substep "       https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html"
                substep "  2. ROCm 7.2.1 + librocdxg inside WSL (with HSA_ENABLE_DXG_DETECTION=1)."
                substep "  3. A WSL distro AMD supports for ROCm -- Ubuntu 24.04 is the known-good one."
                if [ -n "$_wsl_ubu_ver" ] && [ "$_wsl_ubu_ver" != "24.04" ]; then
                    substep "  This distro is Ubuntu $_wsl_ubu_ver, which AMD may not support for ROCm-on-WSL yet." "$C_WARN"
                fi
                substep "Set up the GPU in WSL with a dedicated Ubuntu 24.04 distro:"
                substep "  wsl --install Ubuntu-24.04        # run in Windows PowerShell, then reopen WSL"
                substep "  # then re-run this installer inside Ubuntu-24.04 -- it will detect the GPU."
                substep "AMD ROCm-on-WSL docs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/"
                substep "Strix Halo (gfx1151): this installer auto-offers ROCm-on-WSL setup once the"
                substep "  driver is current; or run unsloth/scripts/install_rocm_wsl_strixhalo.sh yourself."
            else
                substep "AMD ROCm users: see https://docs.unsloth.ai/get-started/install-and-update/amd"
                # Only when ROCm truly can't see the GPU: a detected-but-too-old
                # ROCm (rocminfo works, wheels need 6.0+) has its own guidance.
                if ! _has_amd_rocm_gpu && _amd_gpu_present_via_pci; then
                    substep "An AMD GPU is on the PCI bus but ROCm cannot see it (no /dev/kfd," "$C_WARN"
                    substep "  rocminfo, or amd-smi). Install the ROCm kernel stack so /dev/kfd exists;"
                    substep "  Strix Halo (gfx1151/gfx1150) needs a recent kernel (6.11+) and ROCm 7.x."
                fi
            fi
            substep "Re-run with --no-torch for GGUF-only (faster, no PyTorch):"
            substep "  curl -fsSL https://unsloth.ai/install.sh | sh -s -- --no-torch"
        fi
        ;;
    */rocm*|*/gfx*)
        if [ "$_amd_gpu_radeon" = true ]; then
            substep "wheels: repo.radeon.com (Radeon)"
        else
            substep "wheels: $(_strip_index_url_credentials "$TORCH_INDEX_URL")"
        fi
        ;;
esac

# ── Install unsloth directly into the venv (no activation needed) ──
tauri_log "STEP" "Installing PyTorch"
_VENV_PY="$VENV_DIR/bin/python"

# A released unsloth wheel can pin an older torch (unsloth 2026.7.2 declares
# torch<2.11.0); a with-deps PyPI resolve then downgrades the whole trio,
# swapping the pinned +cuXXX/+rocm build for PyPI's default. The flavor guard
# below misses this (PyPI's torch 2.10 default is itself cu128-flavored), so
# freeze the trio via uv --overrides (overrides replace dependency requirements
# during resolution) while unsloth's other deps resolve normally. Sets
# _UNSLOTH_TORCH_OVERRIDES from the trio in the venv; every with-deps unsloth
# install (migrated and fresh) must call this before resolving and rm it after.
_build_unsloth_torch_overrides() {
    _UNSLOTH_TORCH_OVERRIDES=""
    [ "$SKIP_TORCH" = false ] || return 0
    _torch_trio_pins=$("$_VENV_PY" -c "
from importlib.metadata import version, PackageNotFoundError
for _p in ('torch', 'torchvision', 'torchaudio'):
    try:
        print(_p + '==' + version(_p))
    except PackageNotFoundError:
        pass
" 2>/dev/null) || _torch_trio_pins=""
    case "$_torch_trio_pins" in
        torch==*)
            _UNSLOTH_TORCH_OVERRIDES=$(mktemp)
            printf '%s\n' "$_torch_trio_pins" > "$_UNSLOTH_TORCH_OVERRIDES"
            # The CLI --overrides flag replaces any UV_OVERRIDE env file (same
            # uv setting; macOS arm64 exports one here), so fold its pins in.
            # awk, not cat: it drops inherited torch-trio lines (uv intersects
            # duplicate overrides, so a conflicting pin would make resolution
            # unsatisfiable) and newline-terminates the last line so an
            # unterminated file cannot join two requirements into one.
            for _ov_file in ${UV_OVERRIDE:-}; do
                [ -f "$_ov_file" ] && awk '!/^[[:space:]]*torch(vision|audio)?([[:space:]<>=!~;@[]|$)/' "$_ov_file" >> "$_UNSLOTH_TORCH_OVERRIDES"
            done
            ;;
    esac
}

if [ "$_MIGRATED" = true ]; then
    # Migrated env: force-reinstall unsloth+unsloth-zoo for a clean state, preserving
    # existing torch/CUDA unless the ROCm repair below fires.
    substep "upgrading unsloth in migrated environment..."
    if [ "$SKIP_TORCH" = true ]; then
        # No-torch: install unsloth + unsloth-zoo with --no-deps (current
        # PyPI metadata still declares torch as a hard dep), then install
        # runtime deps (typer, safetensors, transformers, etc.) with --no-deps
        # to prevent transitive torch resolution.
        run_install_cmd_retry "install unsloth (migrated no-torch)" uv pip install --python "$_VENV_PY" --no-deps \
            --reinstall-package unsloth --reinstall-package unsloth-zoo \
            "unsloth>=2026.7.4" "unsloth-zoo>=2026.7.4"
        # Resolve pydantic WITH deps so pip pins pydantic-core to the
        # matching version (no-torch-runtime.txt below is --no-deps).
        # All transitive deps are torch-free.
        run_install_cmd_retry "install pydantic (with deps for compatible core)" \
            uv pip install --python "$_VENV_PY" pydantic
        _NO_TORCH_RT="$(_find_no_torch_runtime)"
        if [ -n "$_NO_TORCH_RT" ]; then
            run_install_cmd_retry "install no-torch runtime deps" uv pip install --python "$_VENV_PY" --no-deps -r "$_NO_TORCH_RT"
        fi
    else
        # Pin mlx-lm away from 0.31.3 here too: a curl-piped migration has no
        # overrides file, so UV_OVERRIDE is unset and this positional is the only cover.
        _build_unsloth_torch_overrides
        run_install_cmd_retry "install unsloth (migrated)" uv pip install --python "$_VENV_PY" \
            ${_UNSLOTH_TORCH_OVERRIDES:+--overrides "$_UNSLOTH_TORCH_OVERRIDES"} \
            --reinstall-package unsloth --reinstall-package unsloth-zoo \
            "unsloth>=2026.7.4" "unsloth-zoo>=2026.7.4" ${_MLX_LM_EXCLUDE_ARG:-}
        [ -n "$_UNSLOTH_TORCH_OVERRIDES" ] && rm -f "$_UNSLOTH_TORCH_OVERRIDES"
        _UNSLOTH_TORCH_OVERRIDES=""
    fi
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git main..."
        run_install_cmd_retry "overlay unsloth-zoo (git main)" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
    fi
    # AMD ROCm: install bitsandbytes even in migrated environments so
    # existing ROCm installs gain the AMD bitsandbytes build without a
    # fresh reinstall.
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = true ]; then
        _install_bnb_rocm "install bitsandbytes (AMD)" "$_VENV_PY"
        # Repair ROCm torch if overwritten during migrated install
        _has_hip=$("$_VENV_PY" -c "import torch; print(getattr(torch.version,'hip','') or '')" 2>/dev/null || true)
        if [ -z "$_has_hip" ]; then
            substep "repairing ROCm torch (overwritten by dependency resolution)..."
            _install_torch_default_index --force-reinstall
        fi
    fi
elif [ -n "$TORCH_INDEX_URL" ]; then
    # Fresh: Step 1 - install torch from explicit index (skip when --no-torch or Intel Mac)
    if [ "$SKIP_TORCH" = true ]; then
        substep "skipping PyTorch (--no-torch or Intel Mac x86_64)." "$C_WARN"
    elif [ "$_amd_gpu_radeon" = true ]; then
        _radeon_url=$(get_radeon_wheel_url)
        if [ -n "$_radeon_url" ]; then
            _radeon_listing_ok=false
            if _radeon_fetch_listing "$_radeon_url" 2>/dev/null; then
                _radeon_listing_ok=true
            else
                # Try shorter X.Y path (AMD publishes both X.Y.Z and X.Y dirs)
                _radeon_url_short=$(printf '%s\n' "$_radeon_url" \
                    | sed 's|rocm-rel-\([0-9]*\)\.\([0-9]*\)\.[0-9]*/|rocm-rel-\1.\2/|')
                if [ "$_radeon_url_short" != "$_radeon_url" ] && \
                   _radeon_fetch_listing "$_radeon_url_short" 2>/dev/null; then
                    _radeon_listing_ok=true
                fi
            fi

            if [ "$_radeon_listing_ok" = true ]; then
                # Require torch, torchvision, torchaudio wheels to all resolve
                # from the Radeon listing. The repo often publishes multiple
                # generations simultaneously, so picking the highest-version
                # for each package independently can assemble a mismatched trio
                # (e.g. torch 2.10 + torchvision 0.24). To prevent this,
                # we identify the highest common minor version and downpair
                # wheels if necessary to ensure a compatible set.
                _torch_whl=$(_pick_radeon_wheel "torch"       2>/dev/null) || _torch_whl=""
                _tv_whl=$(_pick_radeon_wheel    "torchvision" 2>/dev/null) || _tv_whl=""
                _ta_whl=$(_pick_radeon_wheel    "torchaudio"  2>/dev/null) || _ta_whl=""
                _tri_whl=$(_pick_radeon_wheel   "triton"      2>/dev/null) || _tri_whl=""

                # Check that torch and torchaudio share the same X.Y public
                # version prefix, and that torchvision's minor correctly
                # pairs with torch's minor (torchvision = torch.minor + 15
                # since torch 2.4 -> torchvision 0.19 -> torch 2.9 ->
                # torchvision 0.24).
                #
                # URL-decode each wheel name so %2B -> + before version
                # extraction. Real Radeon wheel hrefs are percent-encoded
                # (torch-2.10.0%2Brocm7.2.0...), so a plain [+-] terminator
                # in the sed regex below would never match and
                # _radeon_versions_match would stay false for every real
                # listing, silently forcing a fallback to the generic
                # ROCm index.
                _extract_version() {
                    _whl=$1
                    _pkg=$2
                    if [ -n "$_whl" ]; then
                        _name=$(printf '%s' "${_whl##*/}" | sed 's/%2[Bb]/+/g')
                        printf '%s\n' "$_name" | sed -n "s|^${_pkg}-\([0-9][0-9]*\.[0-9][0-9]*\)\(\.[0-9][0-9]*\)\{0,1\}[+-].*|\1|p"
                    fi
                }

                _torch_ver=$(_extract_version "$_torch_whl" "torch")
                _tv_ver=$(_extract_version "$_tv_whl" "torchvision")
                _ta_ver=$(_extract_version "$_ta_whl" "torchaudio")

                _radeon_versions_match=false
                # Kept release (_PREV_TORCH_PIN) wins here too: pick its exact
                # patch (else the newest patch of its minor) plus the paired
                # vision/audio wheels. Any gap falls back to the newest-trio
                # search below, mirroring _install_torch_default_index, so a
                # rerun never drifts to another release nor below the kept one.
                if [ -n "$_PREV_TORCH_PIN" ]; then
                    _prev_kept_base="${_PREV_TORCH_PIN#torch==}"
                    _prev_kept_minor="${_prev_kept_base#*.}"
                    _prev_kept_minor="${_prev_kept_minor%%.*}"
                    case "$_prev_kept_minor" in
                        ''|*[!0-9]*) ;;
                        *)
                            _kept_torch=$(_pick_radeon_wheel "torch" "${_prev_kept_base}" 2>/dev/null) || _kept_torch=""
                            [ -z "$_kept_torch" ] && { _kept_torch=$(_pick_radeon_wheel "torch" "2.${_prev_kept_minor}." 2>/dev/null) || _kept_torch=""; }
                            _kept_tv=$(_pick_radeon_wheel "torchvision" "0.$((_prev_kept_minor + 15))." 2>/dev/null) || _kept_tv=""
                            _kept_ta=$(_pick_radeon_wheel "torchaudio" "2.${_prev_kept_minor}." 2>/dev/null) || _kept_ta=""
                            if [ -n "$_kept_torch" ] && [ -n "$_kept_tv" ] && [ -n "$_kept_ta" ]; then
                                _torch_whl=$_kept_torch
                                _tv_whl=$_kept_tv
                                _ta_whl=$_kept_ta
                                _tri_whl=""
                                _radeon_versions_match=true
                                # Say so when the listing pruned the exact patch
                                # and a same-series build is installed instead.
                                case "$(printf '%s' "${_kept_torch##*/}" | sed 's/%2[Bb]/+/g')" in
                                    "torch-${_prev_kept_base}"[+-]*) ;;
                                    *) substep "kept release ${_prev_kept_base} is not in the Radeon listing -- installing the closest 2.${_prev_kept_minor} series build instead" ;;
                                esac
                            else
                                substep "[WARN] Radeon repo lacks a complete wheel set for kept $_PREV_TORCH_PIN -- installing the newest compatible set instead" "$C_WARN"
                            fi
                            ;;
                    esac
                fi
                if [ "$_radeon_versions_match" != true ] && \
                   [ -n "$_torch_ver" ] && [ -n "$_tv_ver" ] && [ -n "$_ta_ver" ]; then
                    _torch_minor=${_torch_ver#*.}
                    _ta_minor=${_ta_ver#*.}
                    _tv_minor=${_tv_ver#*.}
                    _tv_equiv_minor=$((_tv_minor - 15))

                    # Determine initial target minor (lowest common denominator)
                    _target_minor=$_torch_minor
                    [ "$_tv_equiv_minor" -lt "$_target_minor" ] && _target_minor=$_tv_equiv_minor
                    [ "$_ta_minor" -lt "$_target_minor" ] && _target_minor=$_ta_minor

                    # Loop downwards to find the first complete matching trio.
                    # This avoids aborting if the repo has gaps.
                    _attempts=0
                    while [ "$_attempts" -lt 5 ] && [ "$_target_minor" -ge 0 ]; do
                        _expected_tv_minor=$((_target_minor + 15))

                        _curr_torch=$(_pick_radeon_wheel "torch"       "2.${_target_minor}." 2>/dev/null) || _curr_torch=""
                        _curr_tv=$(_pick_radeon_wheel    "torchvision" "0.${_expected_tv_minor}." 2>/dev/null) || _curr_tv=""
                        _curr_ta=$(_pick_radeon_wheel    "torchaudio"  "2.${_target_minor}." 2>/dev/null) || _curr_ta=""

                        if [ -n "$_curr_torch" ] && [ -n "$_curr_tv" ] && [ -n "$_curr_ta" ]; then
                            # Extract versions from the wheels found in this iteration
                            _c_torch_ver=$(_extract_version "$_curr_torch" "torch")
                            _c_tv_ver=$(_extract_version "$_curr_tv" "torchvision")
                            _c_ta_ver=$(_extract_version "$_curr_ta" "torchaudio")

                            # Parse Major.Minor for validation
                            _c_torch_major=${_c_torch_ver%%.*}
                            _c_torch_minor=${_c_torch_ver#*.}
                            _c_ta_major=${_c_ta_ver%%.*}
                            _c_ta_minor=${_c_ta_ver#*.}
                            _c_tv_major=${_c_tv_ver%%.*}
                            _c_tv_minor=${_c_tv_ver#*.}

                            # Strict X.Y validation: allow patch versions to differ (e.g. torch 2.9.1 + vision 0.24.0)
                            # as long as the Major and Minor pairing is correct.
                            if [ "$_c_torch_major" = "$_c_ta_major" ] && \
                               [ "$_c_torch_minor" = "$_c_ta_minor" ] && \
                               [ "$_c_tv_major" = "0" ] && \
                               [ "$_c_tv_minor" = "$((_c_torch_minor + 15))" ]; then

                                _torch_whl=$_curr_torch
                                _tv_whl=$_curr_tv
                                _ta_whl=$_curr_ta
                                _tri_whl=""
                                _radeon_versions_match=true
                                break
                            fi
                        fi
                        _target_minor=$((_target_minor - 1))
                        _attempts=$((_attempts + 1))
                    done
                fi

                if [ -z "$_torch_whl" ] || [ -z "$_tv_whl" ] || [ -z "$_ta_whl" ] || \
                   [ "$_radeon_versions_match" != true ]; then
                    substep "[WARN] Radeon repo lacks a compatible wheel set for this Python; falling back to ROCm index ($(_strip_index_url_credentials "$TORCH_INDEX_URL"))" "$C_WARN"
                    _install_torch_default_index
                else
                    substep "installing PyTorch from Radeon repo (${_RADEON_BASE_URL})..."
                    # Pass explicit wheel URLs so the matched trio is
                    # installed together. --find-links lets uv discover
                    # the Radeon listing for any local lookup, and PyPI
                    # (not disabled) provides transitive deps like
                    # filelock / sympy / networkx which are not in the
                    # Radeon listing.
                    if [ -n "$_tri_whl" ]; then
                        run_install_cmd_retry "install triton + PyTorch" uv pip install --python "$_VENV_PY" \
                            --find-links "$_RADEON_BASE_URL" \
                            "$_tri_whl" "$_torch_whl" "$_tv_whl" "$_ta_whl"
                    else
                        run_install_cmd_retry "install PyTorch" uv pip install --python "$_VENV_PY" \
                            --find-links "$_RADEON_BASE_URL" \
                            "$_torch_whl" "$_tv_whl" "$_ta_whl"
                    fi
                fi
            else
                substep "[WARN] Radeon repo unavailable; falling back to ROCm index ($(_strip_index_url_credentials "$TORCH_INDEX_URL"))" "$C_WARN"
                _install_torch_default_index
            fi
        else
            substep "[WARN] Radeon GPU detected but could not detect full ROCm version; falling back to ROCm index" "$C_WARN"
            _install_torch_default_index
        fi
    else
        substep "installing PyTorch ($(_strip_index_url_credentials "$TORCH_INDEX_URL"))..."
        _install_torch_default_index
    fi
    # AMD ROCm: install bitsandbytes (once, after torch, for all ROCm paths).
    # Gate on SKIP_TORCH=false so a user running with --no-torch on a ROCm
    # host stays in GGUF-only mode rather than pulling in bitsandbytes,
    # which is only useful once torch is present for training.
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = true ]; then
        _install_bnb_rocm "install bitsandbytes (AMD)" "$_VENV_PY"
    fi
    # Fresh: Step 2 - install unsloth, preserving the torch Step 1 installed
    tauri_log "STEP" "Installing Unsloth"
    substep "installing unsloth (this may take a few minutes)..."
    _build_unsloth_torch_overrides
    if [ "$SKIP_TORCH" = true ]; then
        # No-torch: install unsloth + unsloth-zoo with --no-deps, then
        # runtime deps (typer, safetensors, transformers, etc.) with --no-deps.
        run_install_cmd_retry "install unsloth (no-torch)" uv pip install --python "$_VENV_PY" --no-deps \
            --upgrade-package unsloth --upgrade-package unsloth-zoo \
            "unsloth>=2026.7.4" "unsloth-zoo>=2026.7.4"
        # Same pydantic-with-deps trick as the migrated branch.
        run_install_cmd_retry "install pydantic (with deps for compatible core)" \
            uv pip install --python "$_VENV_PY" pydantic
        _NO_TORCH_RT="$(_find_no_torch_runtime)"
        if [ -n "$_NO_TORCH_RT" ]; then
            run_install_cmd_retry "install no-torch runtime deps" uv pip install --python "$_VENV_PY" --no-deps -r "$_NO_TORCH_RT"
        fi
        if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
            substep "overlaying local repo (editable)..."
            run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
            substep "overlaying unsloth-zoo from git main..."
            run_install_cmd_retry "overlay unsloth-zoo (git main)" uv pip install --python "$_VENV_PY" \
                --no-deps --reinstall-package unsloth-zoo \
                "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
        fi
    elif [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        run_install_cmd_retry "install unsloth (local)" uv pip install --python "$_VENV_PY" \
            ${_UNSLOTH_TORCH_OVERRIDES:+--overrides "$_UNSLOTH_TORCH_OVERRIDES"} \
            --upgrade-package unsloth "unsloth>=2026.7.4" "unsloth-zoo>=2026.7.4"
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git main..."
        run_install_cmd_retry "overlay unsloth-zoo (git main)" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
    else
        run_install_cmd_retry "install unsloth" uv pip install --python "$_VENV_PY" \
            ${_UNSLOTH_TORCH_OVERRIDES:+--overrides "$_UNSLOTH_TORCH_OVERRIDES"} \
            --upgrade-package unsloth -- "$PACKAGE_NAME" ${_MLX_LM_EXCLUDE_ARG:-}
    fi
    [ -n "$_UNSLOTH_TORCH_OVERRIDES" ] && rm -f "$_UNSLOTH_TORCH_OVERRIDES"
    _UNSLOTH_TORCH_OVERRIDES=""
    # AMD ROCm: repair torch if the unsloth/unsloth-zoo install pulled in
    # CUDA torch from PyPI, overwriting the ROCm wheels installed in Step 1.
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = true ]; then
        _has_hip=$("$_VENV_PY" -c "import torch; print(getattr(torch.version,'hip','') or '')" 2>/dev/null || true)
        if [ -z "$_has_hip" ]; then
            substep "repairing ROCm torch (overwritten by dependency resolution)..."
            _install_torch_default_index --force-reinstall
        fi
    fi
else
    # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
    tauri_log "STEP" "Installing Unsloth"
    substep "installing unsloth (this may take a few minutes)..."
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        run_install_cmd_retry "install unsloth (auto torch backend)" uv pip install --python "$_VENV_PY" "unsloth-zoo>=2026.7.4" "unsloth>=2026.7.4" --torch-backend=auto
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git main..."
        run_install_cmd_retry "overlay unsloth-zoo (git main)" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
    else
        run_install_cmd_retry "install unsloth (auto torch backend)" uv pip install --python "$_VENV_PY" --torch-backend=auto -- "$PACKAGE_NAME"
    fi
fi

_installed_package_version=$("$_VENV_PY" -c \
    'from importlib.metadata import version; import sys; print(version(sys.argv[1]))' \
    "$PACKAGE_NAME" 2>/dev/null || true)
if [ -n "$_installed_package_version" ]; then
    step "$PACKAGE_NAME" "$_installed_package_version installed"
else
    substep "[WARN] installed $PACKAGE_NAME version could not be determined" "$C_WARN"
fi

# ── Enforce the installed torch flavor matches the detected GPU build ──
# PEP 440 ignores the +cpu/+cuXXX/+rocm local label in a version range, so uv
# keeps a stale torch==X+cpu against a GPU index and the venv silently trains on
# CPU. Reinstall the right wheel triplet when a GPU build is expected; if it
# can't be reinstalled, warn loudly. --no-torch / CPU-only / macOS: no-op.
if [ "$SKIP_TORCH" = false ] && [ -n "${TORCH_INDEX_URL:-}" ]; then
    _expected_torch_tag=$(_expected_torch_flavor_tag "$TORCH_INDEX_URL")
    # Only act when a GPU build is expected (cuXXX / rocm); cpu and unknown skip.
    if [ -n "$_expected_torch_tag" ] && [ "$_expected_torch_tag" != "cpu" ]; then
        _installed_torch_ver=$("$_VENV_PY" -c "import torch; print(torch.__version__)" 2>/dev/null || true)
        _installed_torch_tag=""
        [ -n "$_installed_torch_ver" ] && _installed_torch_tag=$(_torch_flavor_tag "$_installed_torch_ver")
        # Repair when flavor is wrong AND the index is plain --default-index reinstallable
        # (cuXXX / rocmX.Y / repo.amd.com gfx*); an unknown mirror leaf -> warn only.
        if [ -n "$_installed_torch_tag" ] && [ "$_installed_torch_tag" != "$_expected_torch_tag" ] \
           && [ "$(_torch_index_repairable "$TORCH_INDEX_URL")" = "yes" ]; then
            substep "PyTorch flavor mismatch (installed $_installed_torch_tag, need $_expected_torch_tag) -- reinstalling correct build..."
            _install_torch_default_index \
                --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
            _installed_torch_ver=$("$_VENV_PY" -c "import torch; print(torch.__version__)" 2>/dev/null || true)
            _installed_torch_tag=""
            [ -n "$_installed_torch_ver" ] && _installed_torch_tag=$(_torch_flavor_tag "$_installed_torch_ver")
        fi
        # Safety net (incl. AMD/WSL): GPU build expected but still CPU -> warn loudly.
        if [ "$_installed_torch_tag" = "cpu" ]; then
            substep "[WARN] PyTorch is CPU-only but a $_expected_torch_tag GPU build was expected for this machine." "$C_WARN"
            substep "[WARN] Training and GPU inference will run on CPU until this is fixed." "$C_WARN"
            substep "[WARN] Re-run this installer, or reinstall the GPU build manually:" "$C_WARN"
            substep "[WARN]   uv pip install --python \"$_VENV_PY\" \"$TORCH_CONSTRAINT\" \"$TORCHVISION_CONSTRAINT\" \"$TORCHAUDIO_CONSTRAINT\" --default-index $(_strip_index_url_credentials "$TORCH_INDEX_URL") --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio" "$C_WARN"
        fi
    fi
fi

# ── Run studio setup ──
tauri_log "STEP" "Running Unsloth setup"
# When --local, use the repo's own setup.sh directly.
# Otherwise, find it inside the installed package.
SETUP_SH=""
if [ "$STUDIO_LOCAL_INSTALL" = true ] && [ -f "$_REPO_ROOT/studio/setup.sh" ]; then
    SETUP_SH="$_REPO_ROOT/studio/setup.sh"
fi

if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    SETUP_SH=$("$VENV_DIR/bin/python" -c "
import importlib.resources
print(importlib.resources.files('studio') / 'setup.sh')
" 2>/dev/null || echo "")
fi

# Fallback: search site-packages
if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    SETUP_SH=$(find "$VENV_DIR" -path "*/studio/setup.sh" -print -quit 2>/dev/null || echo "")
fi

if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    tauri_log "ERROR" "Could not find studio/setup.sh in the installed package"
    echo "❌ ERROR: Could not find studio/setup.sh in the installed package."
    exit 1
fi

# Ensure the venv's Python is on PATH so setup.sh can find it.
VENV_ABS_BIN="$(cd "$VENV_DIR/bin" && pwd)"
if [ -n "$VENV_ABS_BIN" ]; then
    export PATH="$VENV_ABS_BIN:$PATH"
fi

if ! command -v bash >/dev/null 2>&1; then
    step "setup" "bash is required to run studio setup" "$C_ERR"
    substep "Please install bash and re-run install.sh"
    exit 1
fi

step "setup" "running unsloth studio update..."
_SKIP_BASE=1
_SETUP_EXIT=0
# Tauri desktop app bundles its own frontend — skip Node/npm/frontend build
_SKIP_FRONTEND=0
if [ "$TAURI_MODE" = true ]; then
    _SKIP_FRONTEND=1
fi
# Prepend UNSLOTH_STUDIO_HOME=$STUDIO_HOME to "$@" for env-override installs
# without word-splitting on whitespace paths.
_run_setup_with_studio_home() {
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
        UNSLOTH_STUDIO_HOME="$STUDIO_HOME" "$@"
    else
        "$@"
    fi
}
if [ -n "$_WITH_LLAMA_CPP_DIR" ]; then
    if [ ! -d "$_WITH_LLAMA_CPP_DIR" ]; then
        echo "[ERROR] --with-llama-cpp-dir path does not exist: $_WITH_LLAMA_CPP_DIR" >&2
        exit 1
    fi
    _WITH_LLAMA_CPP_DIR="$(CDPATH= cd -P -- "$_WITH_LLAMA_CPP_DIR" && pwd -P)"
fi
if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
    _run_setup_with_studio_home env \
    SKIP_STUDIO_BASE="$_SKIP_BASE" \
    SKIP_STUDIO_FRONTEND="$_SKIP_FRONTEND" \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    STUDIO_LOCAL_INSTALL=1 \
    STUDIO_LOCAL_REPO="$_REPO_ROOT" \
    UNSLOTH_NO_TORCH="$SKIP_TORCH" \
    UNSLOTH_LOCAL_LLAMA_CPP_DIR="$_WITH_LLAMA_CPP_DIR" \
    bash "$SETUP_SH" </dev/null || _SETUP_EXIT=$?
else
    # Explicitly reset STUDIO_LOCAL_INSTALL / STUDIO_LOCAL_REPO so a stale
    # value inherited from the parent shell (e.g. a previous --local run in
    # the same session) does not silently flip a normal install onto the
    # local-dev path in setup.sh and install_python_stack.py. Mirrors the
    # reset already done in install.ps1 for PowerShell.
    _run_setup_with_studio_home env \
    SKIP_STUDIO_BASE="$_SKIP_BASE" \
    SKIP_STUDIO_FRONTEND="$_SKIP_FRONTEND" \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    STUDIO_LOCAL_INSTALL=0 \
    STUDIO_LOCAL_REPO= \
    UNSLOTH_NO_TORCH="$SKIP_TORCH" \
    UNSLOTH_LOCAL_LLAMA_CPP_DIR="$_WITH_LLAMA_CPP_DIR" \
    bash "$SETUP_SH" </dev/null || _SETUP_EXIT=$?
fi

# ── Make 'unsloth' available via $_LOCAL_BIN (resolved earlier) ──
# Env-mode: $_LOCAL_BIN is $STUDIO_HOME/bin; skip shell-rc PATH append so we
# don't pollute the user's profile with a workspace-scoped path.
mkdir -p "$_LOCAL_BIN"
# ln -sf into an existing dir creates link inside it. Refuse to delete a
# real directory at the shim path -- that could destroy unrelated user data.
_shim_path="$_LOCAL_BIN/unsloth"
if [ -d "$_shim_path" ] && [ ! -L "$_shim_path" ]; then
    echo "ERROR: $_shim_path is a directory; refusing to delete it." >&2
    echo "       Move or remove it manually, then re-run the installer." >&2
    exit 1
fi
# why: -sfn is atomic and -n prevents descent into a symlink-to-directory at
# the shim path (the directory guard above already rejects a real directory).
ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path"

case ":$PATH:" in
    *":$_LOCAL_BIN:"*) ;;  # already on PATH
    *)
        if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
            export PATH="$_LOCAL_BIN:$PATH"
            step "path" "exported $_LOCAL_BIN for this session (no rc-file append in env-override mode)"
        else
            _SHELL_PROFILE=""
            if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
                _SHELL_PROFILE="$HOME/.zshrc"
            elif [ -f "$HOME/.bashrc" ]; then
                _SHELL_PROFILE="$HOME/.bashrc"
            elif [ -f "$HOME/.profile" ]; then
                _SHELL_PROFILE="$HOME/.profile"
            fi
            if [ -n "$_SHELL_PROFILE" ]; then
                if ! grep -q '\.local/bin' "$_SHELL_PROFILE" 2>/dev/null; then
                    echo '' >> "$_SHELL_PROFILE"
                    echo '# Added by Unsloth installer' >> "$_SHELL_PROFILE"
                    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$_SHELL_PROFILE"
                    step "path" "added ~/.local/bin to PATH in $_SHELL_PROFILE"
                fi
            fi
            export PATH="$_LOCAL_BIN:$PATH"
        fi
        ;;
esac

# Non-Tauri installs keep shortcuts even if setup reports failure.
# create_studio_shortcuts gates persistent menu shortcuts on env-mode;
# launcher + studio.conf + icon are always written.
if [ "$TAURI_MODE" != true ]; then
    create_studio_shortcuts "$VENV_ABS_BIN/unsloth" "$OS"
fi

# If setup.sh failed, report and exit now.
# PATH and shortcuts are already set up so the user can fix and retry.
if [ "$_SETUP_EXIT" -ne 0 ]; then
    echo ""
    step "error" "studio setup failed (exit code $_SETUP_EXIT)" "$C_ERR"
    echo ""
    exit "$_SETUP_EXIT"
fi

_commit_studio_venv_replacement

# ── Tauri mode: done, skip shortcuts and auto-launch ──
if [ "$TAURI_MODE" = true ]; then
    tauri_log "DONE" ""
    exit 0
fi

# Warn if another 'unsloth' wins on PATH (different venv, system pip, etc).
# Users typing `unsloth studio` later would hit that binary instead of the
# one just installed; the runtime now falls back via UNSLOTH_STUDIO_HOME
# but the absolute path is still the most reliable launch.
# Uses the venv python (just created above) for path canonicalization so
# this works on macOS (BSD readlink has no -f) as well as Linux/WSL.
_installed_bin="$VENV_DIR/bin/unsloth"
_path_unsloth=$(command -v unsloth 2>/dev/null || true)
if [ -n "$_path_unsloth" ] && [ -x "$VENV_DIR/bin/python" ]; then
    # Canonicalize via the venv python (BSD readlink lacks -f on macOS).
    # If either side fails to resolve, skip the check entirely rather than
    # comparing raw paths (which would false-trigger on symlink targets).
    _canon() {
        "$VENV_DIR/bin/python" -c \
            'import os, sys; print(os.path.realpath(sys.argv[1]))' \
            "$1" 2>/dev/null
    }
    _installed_real=$(_canon "$_installed_bin")
    _path_real=$(_canon "$_path_unsloth")
    if [ -n "$_installed_real" ] && [ -n "$_path_real" ] \
        && [ "$_installed_real" != "$_path_real" ]; then
        echo ""
        step "warning" "another 'unsloth' wins on PATH:" "$C_WARN"
        substep "$_path_unsloth"
        substep "this installer's binary is at:"
        substep "$_installed_bin"
        substep "to use this install, run the absolute path above,"
        substep "alias unsloth, or put its dir earlier on PATH."
        echo ""
    fi
fi

echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "Unsloth Studio installed!"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
echo ""

# In interactive terminals, ask the user before starting Unsloth unless the
# caller explicitly disabled the post-install prompt.
# In non-interactive environments (Docker, CI, cloud-init) just print instructions.
if [ "$_SKIP_AUTOSTART" != true ] && [ -t 1 ]; then
    echo ""
    printf "  Start Unsloth Studio now? [Y/n] "
    # No readable answer (closed/EOF tty) defaults to no; Enter is still yes.
    if [ -r /dev/tty ]; then
        read -r _reply </dev/tty || _reply="n"
    else
        _reply="n"
    fi
    case "${_reply:-y}" in
        [Yy]*|"")
            step "launch" "starting Unsloth Studio..."
            # Detach stdin from the `curl | sh` pipe: as a foreground server the
            # studio would otherwise drain the rest of this piped script, leaving
            # the shell to die parsing the now-truncated tail (`unexpected fi`).
            # trap '' INT: wait for studio's shutdown instead of racing the prompt.
            # Subshell resets INT so the child still gets Ctrl+C (no inherited ignore).
            trap '' INT
            # `|| ...`: capture the exit code without set -e aborting first.
            _LAUNCH_EXIT=0
            (trap - INT; exec "$VENV_DIR/bin/unsloth" studio -p 8888 </dev/null) || _LAUNCH_EXIT=$?
            if [ "$_LAUNCH_EXIT" -ne 0 ] && [ "$_MIGRATED" = true ]; then
                echo ""
                echo "⚠️  Unsloth Studio failed to start after migration."
                echo "   Your migrated environment may be incompatible."
                echo "   To fix, remove the environment and reinstall:"
                echo ""
                echo "   rm -rf $VENV_DIR"
                echo "   curl -fsSL https://unsloth.ai/install.sh | sh"
                echo ""
            fi
            exit "$_LAUNCH_EXIT"
            ;;
        *)
            step "launch" "to start later, run:"
            substep "unsloth studio -p 8888"
            substep "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
            substep "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
            echo ""
            ;;
    esac
else
    step "launch" "manual commands:"
    # Single-quote-escape so paths with spaces / apostrophes copy-paste cleanly.
    _li_shim_q="'$(printf '%s' "${_LOCAL_BIN}/unsloth" | sed "s/'/'\\\\''/g")'"
    _li_act_q="'$(printf '%s' "${VENV_DIR}/bin/activate" | sed "s/'/'\\\\''/g")'"
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
        # Env-mode skips the rc PATH append, so print the absolute shim path.
        substep "$_li_shim_q studio -p 8888"
        substep "or activate env first:"
        substep "source $_li_act_q"
        substep "unsloth studio -p 8888"
    else
        substep "unsloth studio -p 8888"
        substep "or activate env first:"
        substep "source $_li_act_q"
        substep "unsloth studio -p 8888"
    fi
    substep "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
    substep "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
    echo ""
fi
