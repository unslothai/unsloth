#!/bin/sh
#
# Unsloth Studio Installer
#
# Usage, supported options and the web one-liner are documented in the repository README under
# "Unsloth Studio (web UI)": https://github.com/unslothai/unsloth#unsloth-studio-web-ui.
# They are not repeated here: this file ships inside the Linux desktop bundle, where a header
# rehearsing download-and-run command lines is the first thing a generic script classifier reads,
# and nothing in the script consults it.
#
# A piped install takes options as environment variables after the pipe (UNSLOTH_NO_TORCH,
# UNSLOTH_SKIP_AUTOSTART, UNSLOTH_PYTHON, UNSLOTH_STUDIO_HOME) because a bare `--no-torch` after
# the pipe would be read as an option to sh itself; a local run takes the equivalent flags
# (--no-torch, --python, --local).
#
# Install dir priority: UNSLOTH_STUDIO_HOME > STUDIO_HOME (alias) > $HOME/.unsloth/studio
#
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e
# ── Why the installer lives in a function ──
# Under a piped web install, sh is the pipe READER. This file is ~150KB, so a top-level
# `exit` left most of it unread, the write end failed, and curl tacked
# "(56) Failure writing output to destination" onto our own error message. Wrapping
# the body forces sh to parse to the closing brace first, so the pipe always drains
# (install.ps1 has always had this shape).
#
# Body is deliberately NOT reindented: reflowing 4000+ lines would bury the change,
# and `exit` still exits the shell from inside a function. Do not add
# `exec < /dev/null`: for a piped shell that closes the script's own source.
_unsloth_main() {

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
# Seed from env (piped-install style); --with-llama-cpp-dir below overrides it.
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

# Custom Unsloth roots are unsupported with --tauri unless override == legacy default.
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
        # Canonicalize both sides so CDPATH / symlinked $HOME can't break equality.
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

# Trim trailing slashes from the URL PATH only, preserving ?query / #fragment. Shared.
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

# Redact index-URL credentials (userinfo + ?query= + #fragment) from captured installer output.
_redact_install_output() {
    sed -E \
        -e 's#(https?://)[^/@[:space:]`]+@#\1<redacted>@#g' \
        -e 's#([?&][^=[:space:]&`]+)=[^&#[:space:]`]+#\1=<redacted>#g' \
        -e 's|(https?://[^[:space:]`#]+)#[^[:space:]`]+|\1#<redacted>|g' \
        "$@"
}

# Large downloads become markers the app consumes and does not display; forwarding uv's
# own chatter would put dozens of lines in front of the user.
: "${UNSLOTH_DL_MARKER_MIN_BYTES:=52428800}"

# $1 is the child's output sink: a log file for the quiet path, empty to pass it along
# stdout for the verbose one. Markers go to stderr to stay clear of the verbose path's
# redactor -- sed block-buffers, so a marker queued behind it would arrive only once the
# download it announces had finished.
_uv_download_markers() {
    # Minimal images ship without awk, which the uv version probe below also allows for.
    # This pipe now carries every install command, so a missing awk must cost the markers
    # and nothing else: without this the pipeline closes and the child dies of SIGPIPE.
    if ! command -v awk >/dev/null 2>&1; then
        if [ -n "$1" ]; then cat >> "$1"; else cat; fi
        return
    fi
    awk -v logf="$1" -v minb="$2" -v tauri="${TAURI_MODE:-false}" -v err=/dev/stderr '
        { if (logf == "") print; else print >> logf }
        tauri != "true" { next }
        # Field-relative so a leading status glyph cannot shift the match.
        /(^| )Downloading [^ ]+ \([0-9.]+[KMG]iB\)$/ {
            size = $NF
            gsub(/[()]/, "", size)
            n = size; sub(/[KMG]iB$/, "", n)
            u = size; sub(/^[0-9.]+/, "", u)
            mult = (u == "GiB") ? 1073741824 : (u == "MiB") ? 1048576 : 1024
            if (n * mult >= minb) {
                announced[$(NF - 1)] = 1
                print "[TAURI:DL] " $(NF - 1) " " size > err
                fflush(err)
            }
            next
        }
        # Only close what was opened: uv also reports completion for unannounced packages.
        /(^| )Downloaded [^ ]+$/ && ($NF in announced) {
            delete announced[$NF]
            print "[TAURI:DL_DONE] " $NF > err
            fflush(err)
        }
    '
}

run_install_cmd() {
    _label="$1"
    shift
    # For --default-index, neutralize inherited uv index/backend/config vars so a uv.toml/pyproject index can't outrank the CLI pin.
    case " $* " in
        *" --default-index "*) set -- env -u UV_DEFAULT_INDEX -u UV_INDEX_URL -u UV_INDEX -u UV_EXTRA_INDEX_URL -u UV_TORCH_BACKEND -u UV_FIND_LINKS -u UV_CONFIG_FILE UV_NO_CONFIG=1 "$@" ;;
    esac
    if _is_verbose; then
        # Stream through the redactor; rc file carries the exit code across the pipe (no pipefail in plain sh).
        _rcf=$(mktemp)
        tauri_stream_log stdout "OUTPUT_CLEAR" "$_label"
        {
            if "$@" 2>&1; then
                _cmd_rc=0
            else
                _cmd_rc=$?
            fi
            printf '%s' "$_cmd_rc" > "$_rcf"
        } | _uv_download_markers "" "$UNSLOTH_DL_MARKER_MIN_BYTES" | _redact_install_output
        _rc=$(cat "$_rcf" 2>/dev/null || echo 1)
        rm -f "$_rcf"
        _rc=${_rc:-1}
        if [ "$_rc" -eq 0 ] 2>/dev/null; then
            tauri_clear_install_error "$_label recovered"
            return 0
        fi
        tauri_stream_log stdout "ERROR_OUTPUT" "$_label failed (exit code $_rc)"
        step "error" "$_label failed (exit code $_rc)" "$C_ERR" >&2
        return "$_rc"
    fi
    _log=$(mktemp)
    _rcf=$(mktemp)
    tauri_stream_log stderr "OUTPUT_CLEAR" "$_label"
    # rc file because the marker filter is a pipe, and plain sh reports only its last stage.
    {
        if "$@" 2>&1; then
            _cmd_rc=0
        else
            _cmd_rc=$?
        fi
        printf '%s' "$_cmd_rc" > "$_rcf"
    } | _uv_download_markers "$_log" "$UNSLOTH_DL_MARKER_MIN_BYTES"
    _rc=$(cat "$_rcf" 2>/dev/null || echo 1)
    rm -f "$_rcf"
    _rc=${_rc:-1}
    if [ "$_rc" -eq 0 ] 2>/dev/null; then
        rm -f "$_log"
        tauri_clear_install_error "$_label recovered"
        return 0
    fi
    step "error" "$_label failed (exit code $_rc)" "$C_ERR" >&2
    _redact_install_output "$_log" >&2
    tauri_stream_log stderr "ERROR_OUTPUT" "$_label failed (exit code $_rc)"
    rm -f "$_log"
    return $_rc
}

# Retry run_install_cmd with backoff; returns the last exit code so the set -e rollback trap still fires.
: "${UNSLOTH_INSTALL_RETRIES:=3}"
: "${UNSLOTH_INSTALL_RETRY_DELAY:=3}"
run_install_cmd_retry() {
    _ricr_label="$1"
    # Sanitize to default 3; length guard before `[ -ge ]`, 0?* rejects leading-zero (octal) delays. Bounds: 1..100 retries, 0..3600s delay.
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
        # AND-OR (not `if`) preserves the real failure code for the rollback path.
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

# True when the runtime target is gfx906 (MI50/Radeon VII): the prebuilt AMD
# bitsandbytes wheel carries no gfx906 kernels, and force-reinstalling it would
# clobber a user's source-built bnb (the only 4-bit path on this arch) on every
# `studio update`. So skip the auto-install and leave whatever bnb is present.
# _gfx906_target is set during torch-index resolution; also honor an explicit
# UNSLOTH_ROCM_GFX_ARCH so a pinned-index install still skips. The override is
# normalized (gfx906:sramecc-:xnack- -> gfx906) so a copied HIP gcnArchName counts.
_is_gfx906_bnb_skip() {
    [ "${_gfx906_target:-false}" = true ] && return 0
    _bnb_gfx_env=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
    _bnb_gfx_env=${_bnb_gfx_env%%:*}
    [ "$_bnb_gfx_env" = "gfx906" ] && return 0
    # A pinned index (UNSLOTH_TORCH_INDEX_URL/_FAMILY) skips the reroute block that
    # sets _gfx906_target, so a real gfx906 host with a pinned rocm6.3 index and no
    # UNSLOTH_ROCM_GFX_ARCH would otherwise clobber a source-built bnb. Probe here
    # in that gap; skip only when gfx906 is the SOLE distinct arch (mixed hosts
    # opt in via the env var, mirroring the reroute block's de-dup rule).
    if [ -z "$_bnb_gfx_env" ] && [ "${_torch_index_pinned:-false}" = true ]; then
        _bnb_gfx_probe=$(_probe_amd_gfx_arch | awk 'NF && !seen[$0]++')
        [ "$_bnb_gfx_probe" = "gfx906" ] && return 0
    fi
    return 1
}

# `pip install unsloth` resolves its unconditional bitsandbytes dep to a generic
# CUDA wheel (no gfx906 kernels) once we skip the prebuilt one. Snapshot bnb before
# the unsloth install, then drop a freshly pulled wheel afterwards while leaving a
# pre-existing source build in place.
_gfx906_bnb_installed() {
    "$_VENV_PY" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('bitsandbytes') else 1)" >/dev/null 2>&1
}
_gfx906_bnb_snapshot() {
    _gfx906_bnb_absent_before=false
    _is_gfx906_bnb_skip || return 0
    _gfx906_bnb_installed || _gfx906_bnb_absent_before=true
}
_gfx906_bnb_prune() {
    _is_gfx906_bnb_skip || return 0
    [ "${_gfx906_bnb_absent_before:-false}" = true ] || return 0
    _gfx906_bnb_installed || return 0
    substep "gfx906: removing generic bitsandbytes pulled in as a dependency (no gfx906 kernels; build from source for 4-bit QLoRA)" "$C_WARN"
    uv pip uninstall --python "$_VENV_PY" bitsandbytes >/dev/null 2>&1 \
        || "$_VENV_PY" -m pip uninstall -y bitsandbytes >/dev/null 2>&1 || true
}

# Install bitsandbytes on AMD ROCm hosts. bnb <= 0.49.2 NaNs at 4-bit decode
# shape on every AMD GPU; the fix (bnb #1887) ships in continuous-release_main
# and, on PyPI, first in 0.50.0. Keep this floor in step with the amd extra in
# pyproject.toml and studio/install_python_stack.py.
_BNB_ROCM_PYPI_FALLBACK="bitsandbytes>=0.50.0"
# Intel XPU: separate constant, same floor by coincidence. 0.50.0 manylinux is the first with
# libbitsandbytes_xpu2025.so / _xpu2026.so; studio/setup.ps1's XPU pass uses the same floor.
_BNB_XPU_SPEC="bitsandbytes>=0.50.0"
# bitsandbytes ships no ROCm binary in its aarch64 wheel at any version: the PyPI
# 0.50.0 and continuous-release_main aarch64 wheels both carry only
# libbitsandbytes_cpu.so plus CUDA variants. So neither install path below gives
# aarch64 a 4-bit backend, and the messages must not claim one. Cf. gfx906.
_bnb_rocm_arch_has_binary() {
    case "$_ARCH" in
        aarch64|arm64) return 1 ;;
        *) return 0 ;;
    esac
}
_warn_bnb_no_rocm_binary() {
    _bnb_rocm_arch_has_binary && return 0
    substep "[WARN] aarch64: bitsandbytes ships no ROCm kernels on this arch; 4-bit QLoRA needs a source build -- https://docs.unsloth.ai/get-started/install-and-update/amd" "$C_WARN"
}
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
    # uv rejects the pre-release wheel: filename version (1.33.7rc0) does not
    # match metadata (0.50.x.dev0). pip accepts it, so bootstrap pip and use it.
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
            _warn_bnb_no_rocm_binary
            return 0
        fi
        _bnb_rc=$?
        if _is_verbose; then
            _redact_install_output "$_bnb_log" >&2
        fi
        rm -f "$_bnb_log"
        step "warning" "$_label (pre-release) failed (exit code $_bnb_rc)" "$C_WARN" >&2
        if _bnb_rocm_arch_has_binary; then
            substep "[WARN] bnb pre-release install failed; falling back to PyPI $_BNB_ROCM_PYPI_FALLBACK, which carries the ROCm 4-bit fix" "$C_WARN"
        else
            substep "[WARN] bnb pre-release install failed; falling back to PyPI $_BNB_ROCM_PYPI_FALLBACK" "$C_WARN"
        fi
    fi
    run_install_cmd "$_label (pypi fallback)" "$_venv_py" -m pip install \
        --force-reinstall --no-cache-dir --no-deps "$_BNB_ROCM_PYPI_FALLBACK"
    _bnb_pypi_rc=$?
    _warn_bnb_no_rocm_binary
    return $_bnb_pypi_rc
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

# Validate --package (injection guard); must start with a letter/digit so uv can't parse it as a flag.
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

tauri_stream_log() {
    _tsl_stream="$1"
    _tsl_tag="$2"
    shift 2
    if [ "$TAURI_MODE" = true ]; then
        if [ "$_tsl_stream" = stderr ]; then
            printf '[TAURI:%s] %s\n' "$_tsl_tag" "$*" >&2
        else
            printf '[TAURI:%s] %s\n' "$_tsl_tag" "$*"
        fi
    fi
}

rollback_substep() {
    if [ "$TAURI_MODE" = true ]; then
        tauri_log "PROGRESS" "$1"
    else
        substep "$@"
    fi
}

tauri_clear_install_error() {
    if [ "$TAURI_MODE" = true ]; then
        tauri_log "ERROR_CLEAR" "$1"
        printf '[TAURI:ERROR_CLEAR] %s\n' "$1" >&2
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
    # Strip query/fragment and trailing slash before classifying (like _torch_index_url_leaf).
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
        */xpu) echo "xpu" ;;
        */rocm[0-9]*.[0-9]*)
            _diag_family=${_diag_url##*/}
            case "$_diag_family" in
                rocm[0-9]*.[0-9]*) echo "$_diag_family" ;;
                *) echo "auto" ;;
            esac ;;
        # AMD arch-specific index (Strix Halo/Point; torch 2.11+rocm7.13 has the real fix).
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
        # Require a digit after cu so /current or /custom isn't branded CUDA.
        cu[0-9]*) echo "cuda" ;;
        rocm*)
            if [ "$_diag_radeon" = true ]; then
                echo "rocm_radeon"
            else
                echo "rocm"
            fi ;;
        radeon) echo "rocm_radeon" ;;
        xpu) echo "xpu" ;;
        cpu) echo "cpu" ;;
        none) echo "no_torch" ;;
        *) echo "unknown" ;;
    esac
}

PYTHON_VERSION=""  # resolved after platform detection

# Resolve install destinations; UNSLOTH_STUDIO_HOME wins over the STUDIO_HOME alias.
_resolve_studio_destinations() {
    _override_var=""
    _override="${UNSLOTH_STUDIO_HOME:-}"
    if [ -n "$_override" ]; then
        _override_var="UNSLOTH_STUDIO_HOME"
    else
        _override="${STUDIO_HOME:-}"
        [ -n "$_override" ] && _override_var="STUDIO_HOME"
    fi
    # Strip surrounding whitespace so " " is treated as unset (matches Python .strip()).
    _override=$(printf '%s' "$_override" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    # Tilde expansion: quoted env vars aren't subject to it on assignment.
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
    # Canonicalize both sides so a trailing slash / symlink mismatch doesn't misfire redirection.
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
# The PATH we inherited, before anything below prepends to it. The shim setup at the end asks
# whether a NEW login shell will find _LOCAL_BIN, and by then this process has prepended it
# several times (uv bootstrap, venv), so testing $PATH there answers yes for a shell that would
# answer no and the profile entry never gets written. astral's installer used to write that line
# for us; the pinned path does not.
_UNSLOTH_LOGIN_PATH="$PATH"
VENV_DIR="$STUDIO_HOME/unsloth_studio"
_VENV_ROLLBACK_DIR=""
_VENV_ROLLBACK_TARGET="$VENV_DIR"
_VENV_ROLLBACK_ACTIVE=false

_start_studio_venv_replacement() {
    _existing_dir="$1"
    _stamp=$(date +%Y%m%d%H%M%S 2>/dev/null || echo "time")
    _candidate="$STUDIO_HOME/unsloth_studio.rollback.$_stamp.$$"
    _suffix=0
    while [ -e "$_candidate" ] || [ -L "$_candidate" ]; do
        _suffix=$((_suffix + 1))
        _candidate="$STUDIO_HOME/unsloth_studio.rollback.$_stamp.$$.$_suffix"
    done
    _VENV_ROLLBACK_DIR="$_candidate"
    _VENV_ROLLBACK_TARGET="$_existing_dir"
    _VENV_ROLLBACK_ACTIVE=true
    # Publish the rollback state before the atomic rename so a signal cannot
    # land after mv but before the exit handlers know where the old venv went.
    if ! mv "$_existing_dir" "$_candidate"; then
        _VENV_ROLLBACK_ACTIVE=false
        _VENV_ROLLBACK_DIR=""
        return 1
    fi
    substep "previous environment preserved for rollback"
}

# uv creates only into a path that is absent or an empty directory. Everything
# else is occupied, hidden entries and non-resolving symlinks included.
_dir_has_entries() {  # dir
    if [ ! -d "$1" ]; then
        # Still an existing path to mkdir(2), which answers EEXIST for a file or
        # for a symlink "dangling or not", so uv refuses it too. -d follows the
        # link and -e misses a dangling one, hence the -L.
        { [ -e "$1" ] || [ -L "$1" ]; } && return 0
        return 1
    fi
    # Not enumerable: the globs cannot expand without read, and the tests below
    # fail on every name without search, so it would read as empty. Fail closed
    # like install.ps1's catch; the rename only needs write on the parent.
    { [ -r "$1" ] && [ -x "$1" ]; } || return 0
    # The globs are the whole check, so a caller's set -f would make every
    # directory look empty. Mirrors _path_has_dir, which saves the flag too.
    _dhe_glob=on
    case $- in *f*) _dhe_glob=off ;; esac
    set +f
    _dhe_found=1
    for _dhe_entry in "$1"/* "$1"/.[!.]* "$1"/..?*; do
        if [ -e "$_dhe_entry" ] || [ -L "$_dhe_entry" ]; then
            _dhe_found=0
            break
        fi
    done
    [ "$_dhe_glob" = off ] && set -f
    return "$_dhe_found"
}

# Clear $VENV_DIR for a recreate without ever destroying the only copy. The
# legacy-layout migration below moves $STUDIO_HOME/.venv straight into $VENV_DIR
# without going through _start_studio_venv_replacement, so a plain `rm -rf` there
# is unrecoverable: if the `uv venv` that follows cannot resolve an interpreter
# (offline, or a uv whose managed-Python manifest predates the patch being asked
# for) the user is left with no environment at all. Move it aside instead and let
# the exit/signal traps put it back. When a replacement is already in flight the
# rollback copy holds the user's real environment and $VENV_DIR is this run's own
# work, so plain removal stays correct.
_discard_venv_for_recreate() {  # venv dir
    if [ "$_VENV_ROLLBACK_ACTIVE" != true ] && [ -d "$1" ] \
       && _start_studio_venv_replacement "$1"; then
        return 0
    fi
    rm -rf "$1"
}

_restore_studio_venv_replacement() {
    [ "$_VENV_ROLLBACK_ACTIVE" = true ] || return 0
    # -e/-L, not -d: a rollback holds whatever _dir_has_entries called occupied,
    # and -d would drop a file or a dangling link and strand the original.
    [ -n "$_VENV_ROLLBACK_DIR" ] \
        && { [ -e "$_VENV_ROLLBACK_DIR" ] || [ -L "$_VENV_ROLLBACK_DIR" ]; } || {
        _VENV_ROLLBACK_ACTIVE=false
        return 0
    }
    rollback_substep "restoring previous environment after failed install..." "$C_WARN"
    rm -rf "$_VENV_ROLLBACK_TARGET"
    if mv "$_VENV_ROLLBACK_DIR" "$_VENV_ROLLBACK_TARGET"; then
        rollback_substep "restored previous environment"
        _VENV_ROLLBACK_ACTIVE=false
        _VENV_ROLLBACK_DIR=""
    else
        echo "⚠️  Could not restore previous environment from $_VENV_ROLLBACK_DIR to $_VENV_ROLLBACK_TARGET" >&2
    fi
}

_studio_venv_rollback_must_be_preserved() {
    _rollback_name=${1##*/}
    _rollback_metadata=${_rollback_name#unsloth_studio.rollback.}
    _rollback_stamp=${_rollback_metadata%%.*}
    _rollback_process=${_rollback_metadata#*.}
    # Preserve anything outside the installer's timestamp.PID[.suffix] format.
    [ "$_rollback_process" != "$_rollback_metadata" ] || return 0
    case "$_rollback_stamp" in
        time) ;;
        ''|*[!0-9]*) return 0 ;;
        *) [ "${#_rollback_stamp}" -eq 14 ] || return 0 ;;
    esac
    _rollback_pid=${_rollback_process%%.*}
    case "$_rollback_pid" in
        ''|*[!0-9]*) return 0 ;;
    esac
    _rollback_suffix=${_rollback_process#*.}
    if [ "$_rollback_suffix" != "$_rollback_process" ]; then
        case "$_rollback_suffix" in ''|*[!0-9]*) return 0 ;; esac
    fi
    kill -0 "$_rollback_pid" 2>/dev/null
}

_prune_stale_studio_venv_rollbacks() {
    for _stale_rollback in "$STUDIO_HOME"/unsloth_studio.rollback.*; do
        [ -d "$_stale_rollback" ] || continue
        if [ -L "$_stale_rollback" ]; then
            echo "⚠️  Refusing to remove rollback symlink $_stale_rollback" >&2
            continue
        fi
        # A concurrent installer may have moved its live venv aside. The PID in
        # the generated name keeps this successful run from deleting its rescue copy.
        _studio_venv_rollback_must_be_preserved "$_stale_rollback" && continue
        if rm -rf "$_stale_rollback"; then
            substep "removed stale environment rollback ${_stale_rollback##*/}"
        else
            echo "⚠️  Could not remove stale environment rollback $_stale_rollback" >&2
        fi
    done
}

_commit_studio_venv_replacement() {
    if [ "$_VENV_ROLLBACK_ACTIVE" = true ]; then
        _rollback_to_remove="$_VENV_ROLLBACK_DIR"
        # The new environment is already committed. Clear the restore state
        # before deletion so an interrupt cannot replace it with a half-deleted backup.
        _VENV_ROLLBACK_ACTIVE=false
        _VENV_ROLLBACK_DIR=""
        # Same shapes as the restore, or such a backup is never cleaned up.
        if [ -n "$_rollback_to_remove" ] \
           && { [ -e "$_rollback_to_remove" ] || [ -L "$_rollback_to_remove" ]; }; then
            if ! rm -rf "$_rollback_to_remove"; then
                echo "⚠️  Could not remove environment rollback $_rollback_to_remove" >&2
            fi
        fi
    fi
    # Only prune older orphaned copies after the replacement has succeeded, so
    # an interrupted install never discards the last known-good environment.
    _prune_stale_studio_venv_rollbacks
}

_cleanup_install_temporaries() {
    [ -n "${_UV_OVERRIDE_TMPDIR:-}" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
    [ -n "${_UV_INSTALL_NAME_TOOL_SHIM_DIR:-}" ] && rm -rf "$_UV_INSTALL_NAME_TOOL_SHIM_DIR" 2>/dev/null || true
    [ -n "${_UV_VENV_CAPTURE_DIR:-}" ] && rm -rf "$_UV_VENV_CAPTURE_DIR" 2>/dev/null || true
    [ -n "${_UNSLOTH_TORCH_OVERRIDES:-}" ] && rm -f "$_UNSLOTH_TORCH_OVERRIDES" 2>/dev/null || true
    # The pinned uv path's own cleanup only runs when that function returns, so a Ctrl-C left
    # the unpacked archive behind plus a staging file inside a directory that is on PATH.
    [ -n "${_UIP_WORK:-}" ] && rm -rf "$_UIP_WORK" 2>/dev/null || true
    [ -n "${_UIP_STAGE:-}" ] && rm -f "$_UIP_STAGE" 2>/dev/null || true
    [ -n "${_UIP_STAGE2:-}" ] && rm -f "$_UIP_STAGE2" 2>/dev/null || true
    [ -n "${_ROCM_TAG_MEMO_DIR:-}" ] && rm -rf "$_ROCM_TAG_MEMO_DIR" 2>/dev/null || true
}

_on_install_exit() {
    _status=$?
    if [ "$_status" -ne 0 ]; then
        _restore_studio_venv_replacement
    fi
    _cleanup_install_temporaries
    exit "$_status"
}

_on_install_signal() {
    _signal_status="$1"
    # EXIT is disabled to avoid a second cleanup pass. Ignore further termination
    # signals until the old environment is back in place.
    trap - EXIT
    trap '' HUP INT TERM
    _restore_studio_venv_replacement
    _cleanup_install_temporaries
    exit "$_signal_status"
}
# Clear inherited cleanup targets before installing traps.
_UV_OVERRIDE_TMPDIR=""
_UV_INSTALL_NAME_TOOL_SHIM_DIR=""
_UV_VENV_CAPTURE_DIR=""
_UNSLOTH_TORCH_OVERRIDES=""
_UIP_WORK=""
_UIP_STAGE=""
_UIP_STAGE2=""
_ROCM_TAG_MEMO_DIR=""
_ROCM_TAG_MEMO=""
trap _on_install_exit EXIT
trap '_on_install_signal 129' HUP
trap '_on_install_signal 130' INT
trap '_on_install_signal 143' TERM

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

# ── Helper: human-readable apt distro label for the sudo package prompt (#6207) ──
# Reads /etc/os-release so the Accept? prompt can say which distro we detected and
# that packages come from that distro's official apt repos (not a tarball).
_apt_distro_description() {
    # Plain ( ... ) subshell — not $() — so case/;; stays bash-3.2-safe on macOS.
    # Bash 3.2 misparses case arms inside command substitution and errors on `;;`.
    (
        if [ ! -r /etc/os-release ]; then
            printf 'a debian-like system'
            exit 0
        fi
        # shellcheck disable=SC1091
        . /etc/os-release 2>/dev/null || true
        if [ -n "${NAME:-}" ] && [ -n "${VERSION_ID:-}" ]; then
            _ad_label="$NAME $VERSION_ID"
        elif [ -n "${PRETTY_NAME:-}" ]; then
            _ad_label="$PRETTY_NAME"
        elif [ -n "${NAME:-}" ]; then
            _ad_label="$NAME"
        else
            printf 'a debian-like system'
            exit 0
        fi
        case " ${ID:-} ${ID_LIKE:-} " in
            *" debian "*|*" ubuntu "*) _ad_label="${_ad_label} (debian-like)" ;;
        esac
        printf '%s' "$_ad_label"
    )
}

# ── Helper: can the controlling terminal actually be opened for reading? ──
# `test -r` only checks permission bits, which look fine in containers and
# systemd units where open() then fails with ENXIO. Probe with a real open.
# The subshell is required: in dash a failed redirection on the special
# builtin `:` exits the whole script.
_can_read_tty() {
    ( : </dev/tty ) >/dev/null 2>&1
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

    # Optional callers never elevate, in any mode: nothing on the consumer path
    # builds anything, so neither the terminal sudo prompt below nor the Tauri
    # NEED_SUDO dialog (whose Cancel leaves the user not installed) may gate the
    # run over unused tools. The caller falls through to prebuilt llama.cpp.
    # Required packages such as curl still escalate.
    if [ "${_SMART_APT_OPTIONAL:-false}" = true ]; then
        return 2
    fi

    if [ "$TAURI_MODE" = true ]; then
        # Report needed packages and exit — Rust handles elevation.
        tauri_log "NEED_SUDO" "$_STILL_MISSING"
        exit 2
    fi

    # Step 3: Escalate -- need elevated permissions for remaining packages
    if command -v sudo >/dev/null 2>&1; then
        _ad_desc="$(_apt_distro_description)"
        echo ""
        echo "    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "    WARNING: We require sudo elevated permissions to install:"
        echo "    $_STILL_MISSING"
        echo "    Detected ${_ad_desc}."
        echo "    If you accept, we'll run sudo apt-get to install these packages"
        echo "    from your distro's official repositories (not a third-party tarball)."
        echo "    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
        if _can_read_tty; then
            printf "    Accept? [Y/n] "
            # The device opened, so a failed read is EOF, not consent: decline,
            # as the autostart prompt below does. Enter is still yes (a
            # successful read of an empty line).
            read -r REPLY </dev/tty || REPLY="n"
            case "$REPLY" in
                [nN]*)
                    echo ""
                    echo "    Please install these packages first, then re-run Unsloth Studio setup:"
                    echo "    sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                    exit 1
                    ;;
            esac
            # Mirror the headless branch: on a sudoers denial, a wrong password
            # or an apt error, say what to run by hand instead of letting set -e
            # abort on a bare sudo/apt message.
            if sudo apt-get update -y </dev/null &&
                sudo apt-get install -y $_STILL_MISSING </dev/null; then
                :
            else
                echo ""
                echo "    Could not install these packages: $_STILL_MISSING"
                echo "    See the error above."
                echo "    Please install them first, then re-run Unsloth Studio setup:"
                echo "    sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                exit 1
            fi
        else
            # Nobody can answer a prompt or type a password here. -n makes sudo
            # refuse rather than prompt into a closed stdin, which is how #7307
            # died. Probe with the real commands: `sudo -l` answers whether they
            # are *authorized*, not whether running them needs authentication.
            # -k ignores any cached timestamp, so only a real NOPASSWD rule gets
            # through, not someone's sudo in another shell minutes ago. Per
            # sudo(8), -k alongside a command ignores the cached credentials and
            # "will not update" them, so other sessions keep theirs.
            echo "    No terminal to confirm on; trying passwordless sudo."
            if sudo -n -k apt-get update -y </dev/null &&
                sudo -n -k apt-get install -y $_STILL_MISSING </dev/null; then
                echo "    Installed with passwordless sudo."
            else
                echo ""
                echo "    Could not install these packages: $_STILL_MISSING"
                echo "    Detected ${_ad_desc}."
                # Either sudo refused, or apt failed on a bad repo, dpkg lock or
                # network outage. sudo exits 1 on an auth/config problem and
                # when the command cannot be executed, but otherwise passes the
                # command's own status through, so state both causes.
                echo "    Either sudo needs a password here, or apt-get itself"
                echo "    failed; see the error above. With no terminal to"
                echo "    authenticate on, this cannot be done unattended."
                echo "    Please install them first, then re-run Unsloth Studio setup:"
                echo "    sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                exit 1
            fi
        fi
    else
        echo ""
        echo "    sudo is not available on this system."
        echo "    Please install these packages as root, then re-run Unsloth Studio setup:"
        echo "    apt-get update -y && apt-get install -y $_STILL_MISSING"
        exit 1
    fi
}

# ── Helper: the studio_install_id contract ──
# 64 lowercase hex, as in the backend (_STUDIO_INSTALL_ID_RE) and the desktop
# app (is_valid_studio_root_id). Nothing else is an id: no backend reports it,
# and the launcher holds it in a single-quoted assignment, so a planted value
# with a quote in it would be launcher code.
# Subshell bodies scope LC_ALL=C to the check: the classes below must mean the
# same bytes in any inherited locale, and the contract is pure ASCII.
_css_install_id_is_valid() (
    LC_ALL=C
    export LC_ALL
    case "${1:-}" in
        "" | *[!0123456789abcdef]*) return 1 ;;
    esac
    [ "${#1}" -eq 64 ]
)

# Echoes the id at $1 when it satisfies the contract, nothing otherwise.
# Returns 1 when the path could not be READ, a different answer: a failed read
# may still be sitting on a valid id.
_css_read_valid_install_id() (
    LC_ALL=C
    export LC_ALL
    # Regular files only: a FIFO here (or a symlink to one, or to a device)
    # would park the installer on the open, waiting for a writer forever.
    [ -f "$1" ] || return 0
    # -s answers "no id" from stat, without a read, so an empty file we also
    # cannot read is replaced as it was pre-validation instead of failing the
    # install. A real id is 64 bytes and never reaches this.
    [ -s "$1" ] || return 0
    # A NUL cannot live in a shell variable, so command substitution drops it
    # and <32 hex>\0<32 hex> would read back valid while the backend, which
    # keeps the byte, reports "". Catch it by mapping NULs to a real character.
    if [ -n "$({ tr -dc '\000' < "$1" | tr '\000' 'N'; } 2>/dev/null)" ]; then
        return 0
    fi
    # Group the redirect, or the shell's own "cannot open" escapes 2>/dev/null.
    # A failed read is reported, never flattened into "no id": permissions or a
    # transient NFS/FUSE fault must not license a rewrite.
    _cvi_id=$({ cat "$1"; } 2>/dev/null) || return 1
    # Trim what the backend's .strip() trims, SURROUNDING whitespace only.
    # Deleting interior whitespace would mint a 64-hex token out of bytes the
    # backend reads otherwise, leaving the launcher holding an id it never
    # reports.
    _cvi_id=${_cvi_id#"${_cvi_id%%[![:space:]]*}"}
    _cvi_id=${_cvi_id%"${_cvi_id##*[![:space:]]}"}
    if _css_install_id_is_valid "$_cvi_id"; then
        printf '%s' "$_cvi_id"
    fi
)

# ── Helper: create desktop shortcuts and launcher script ──
# Usage: create_studio_shortcuts <unsloth_exe> <os> -- writes launch-studio.sh + platform shortcuts.
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

    # Same-install discriminator: per-install opaque id read by launcher + backend (/api/health); lives at $STUDIO_HOME/share/ so the backend finds it via studio_root.
    _css_id_dir="$STUDIO_HOME/share"
    mkdir -p "$_css_id_dir"
    _css_id_file="$_css_id_dir/studio_install_id"
    # Reuse an existing id only when it matches the contract above: a re-run
    # over a normal install is then a no-op, and a pre-populated custom root
    # cannot reach the launcher.
    # Unreadable is not malformed: in a shared root the id can be a good one
    # owned by someone else and already reported by a running backend, so
    # regenerating would break that install. Refuse, as this did pre-validation.
    if ! _css_studio_root_id=$(_css_read_valid_install_id "$_css_id_file"); then
        echo "[WARN] Cannot create launcher: cannot read $_css_id_file" >&2
        return 1
    fi
    if [ -z "$_css_studio_root_id" ]; then
        if [ -r /dev/urandom ]; then
            _css_new_id=$(od -An -N32 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')
        fi
        if ! _css_install_id_is_valid "${_css_new_id:-}" && command -v python3 >/dev/null 2>&1; then
            _css_new_id=$(python3 -c 'import secrets; print(secrets.token_hex(32))' 2>/dev/null)
        fi
        if ! _css_install_id_is_valid "${_css_new_id:-}"; then
            echo "[WARN] Cannot create launcher: no entropy source for studio_install_id" >&2
            return 1
        fi
        # Publish no-clobber: the desktop app mints this same id, so a plain mv
        # could replace one a running backend already reported. ln fails with
        # EEXIST instead and we adopt the winner; its lock is not shareable
        # portably (no flock(1) on macOS). The id is in the temp name because
        # $$ is the parent's pid inside a subshell in some shells.
        _css_id_tmp="$_css_id_file.$$.$(printf '%.8s' "$_css_new_id").tmp"
        if printf '%s' "$_css_new_id" > "$_css_id_tmp"; then
            if ! ln "$_css_id_tmp" "$_css_id_file" 2>/dev/null; then
                # A usable incumbent wins, but only a valid one: zero-length
                # or malformed is an interrupted write or a planted value, so
                # replace it with one rename (no unlink, the path never
                # vanishes). Also covers filesystems without hard links
                # (exFAT/FAT32). -d because renaming onto a directory moves
                # the temp inside it instead of replacing it.
                if _css_incumbent=$(_css_read_valid_install_id "$_css_id_file") \
                    && [ -z "$_css_incumbent" ] && [ ! -d "$_css_id_file" ]; then
                    mv "$_css_id_tmp" "$_css_id_file" 2>/dev/null || true
                fi
            fi
        fi
        rm -f "$_css_id_tmp"
        if [ -f "$_css_id_file" ]; then
            chmod 600 "$_css_id_file" 2>/dev/null || true
        fi
        # Bake what is on disk, not what we meant to write: that is what the
        # backend reports from /api/health, whoever won the race. An unwritable
        # or non-regular path leaves this empty and no launcher is generated.
        _css_studio_root_id=$(_css_read_valid_install_id "$_css_id_file") || true
        unset _css_new_id _css_id_tmp _css_incumbent
    fi
    if [ -z "$_css_studio_root_id" ]; then
        echo "[WARN] Cannot create launcher: failed to read $_css_id_file" >&2
        return 1
    fi
    _css_is_env_mode=false
    [ "$_STUDIO_HOME_REDIRECT" = "env" ] && _css_is_env_mode=true

    # ── Write launcher script ──
    # Single-quoted heredoc; @@ placeholders substituted via sed below.
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

    # Bake non-user-controlled placeholders FIRST so a literal @@STUDIO_ROOT_ID@@ in $DATA_DIR can't be rewritten below.
    sed -e "s|@@STUDIO_ROOT_ID@@|$_css_studio_root_id|g" \
        -e "s|@@INSTALLED_IS_ENV_MODE@@|$_css_is_env_mode|g" \
        "$_css_launcher" > "$_css_launcher.tmp" \
        && mv "$_css_launcher.tmp" "$_css_launcher"

    # Env-mode bakes an absolute DATA_DIR; default / HOME-redirect keeps the literal $HOME/.local/share/unsloth.
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
        # Two-stage escape: single-quote embedding, then backslash/&/| for the sed below.
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

    # studio.conf: exe path + (env-mode only) persisted env vars for fresh shells.
    _css_quoted_exe=$(printf '%s' "$_css_exe" | sed "s/'/'\\\\''/g")
    {
        printf '%s\n' "UNSLOTH_EXE='$_css_quoted_exe'"
        if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
            # An override resolving to the legacy default shares ~/.unsloth/llama.cpp; canonicalize the legacy side.
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
            # UNSLOTH_LLAMA_CPP_PATH is user-controlled; only default it if unset.
            printf '%s\n' 'if [ -z "${UNSLOTH_LLAMA_CPP_PATH:-}" ]; then'
            printf '%s\n' "    export UNSLOTH_LLAMA_CPP_PATH='$_css_quoted_llama'"
            printf '%s\n' 'fi'
        fi
    } > "$_css_data_dir/studio.conf"

    # ── Icon: try bundled, then download ──
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

    # Also try to find the pre-built Tauri icon.icns (1024×1024, professionally
    # built with all required sizes).  Prefer it over the sips-generated icon
    # for the macOS .app bundle — it ships in the pip package at
    # studio/src-tauri/icons/icon.icns and is higher quality with @2x variants.
    _css_tauri_icns=""
    for _sp in "$_css_venv_dir"/lib/python*/site-packages/studio/src-tauri/icons; do
        if [ -f "$_sp/icon.icns" ]; then
            _css_tauri_icns="$_sp/icon.icns"
        fi
    done
    if [ -z "$_css_tauri_icns" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/src-tauri/icons/icon.icns" ]; then
        _css_tauri_icns="$_css_script_dir/studio/src-tauri/icons/icon.icns"
    fi

    # Also look for the higher-resolution Tauri icon.png (1024×1024) for
    # the Linux .desktop icon — better than the 512px rounded variant.
    _css_tauri_png=""
    for _sp in "$_css_venv_dir"/lib/python*/site-packages/studio/src-tauri/icons; do
        if [ -f "$_sp/icon.png" ]; then
            _css_tauri_png="$_sp/icon.png"
        fi
    done
    if [ -z "$_css_tauri_png" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/src-tauri/icons/icon.png" ]; then
        _css_tauri_png="$_css_script_dir/studio/src-tauri/icons/icon.png"
    fi

    # ── Platform-specific shortcuts ──
    # Env-mode installs are workspace-scoped: skip persistent launchers that may point at a deleted workspace.
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
        # Prefer the higher-resolution Tauri icon.png, but persist it under the
        # installed data directory so local-checkout shortcuts survive repo moves.
        _css_desktop_icon="$_css_icon_png"
        if [ -f "$_css_tauri_png" ]; then
            _css_desktop_icon_tmp="${_css_icon_png}.tmp"
            if cp "$_css_tauri_png" "$_css_desktop_icon_tmp" 2>/dev/null \
                && mv "$_css_desktop_icon_tmp" "$_css_icon_png" 2>/dev/null; then
                :
            else
                rm -f "$_css_desktop_icon_tmp"
            fi
        fi
        _css_icon_escaped=$(printf '%s' "$_css_desktop_icon" | sed 's/\\/\\\\/g; s/"/\\"/g')
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
        # Older installs linked the Desktop shortcut with `ln -sf`, which followed the
        # existing link and planted a self-referential copy one level inside the bundle.
        if [ -L "$_css_app/Unsloth Studio.app" ]; then
            rm -f "$_css_app/Unsloth Studio.app" 2>/dev/null || true
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

        # Executable stub: single-quoted heredoc + sed so $-vars in $_css_data_dir don't expand at launch.
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

        # ── AppIcon ──
        # Prefer the pre-built Tauri icon.icns (1024×1024, professionally built
        # with all sizes).  Fall back to generating one from the gem PNG via
        # sips+iconutil, then to a plain PNG copy.
        if [ -f "$_css_tauri_icns" ] \
            && cp "$_css_tauri_icns" "$_css_res_dir/AppIcon.icns" 2>/dev/null; then
            :
        elif [ -f "$_css_gem_png" ] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
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
        # Last-resort fallback: copy PNG as icon
        if [ ! -f "$_css_res_dir/AppIcon.icns" ] && [ -f "$_css_icon_png" ]; then
            cp "$_css_icon_png" "$_css_res_dir/AppIcon.icns" 2>/dev/null || true
        fi

        # Touch so Finder indexes it
        touch "$_css_app"

        # Symlink on Desktop. -n is required: without it a re-run follows the existing
        # link into the bundle and creates the new one inside it, as the CLI shim guards.
        if [ -d "$HOME/Desktop" ]; then
            ln -sfn "$_css_app" "$HOME/Desktop/Unsloth Studio" 2>/dev/null || true
        fi
        _css_created=1

    elif [ "$_css_os" = "wsl" ]; then
        # ── WSL: create Windows Desktop and Start Menu shortcuts ──
        _css_distro="${WSL_DISTRO_NAME:-}"

        # Build wsl.exe args; double-quote so spaced values ("Ubuntu Preview") stay single args.
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

        # DISTINCT per-distro shortcut name so the WSL launcher never clobbers a native "Unsloth Studio.lnk".
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
        # WSL interop disabled ("Exec format error"): no shortcut; tell the user how.
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
# Rosetta is a property of the shell, not of the machine, so it is tracked apart from
# MAC_INTEL: torch and the Python version rightly follow the x86_64 shell, but anything
# that reasons about the HARDWARE (the /usr/bin CLT shims in _has_working_git) must see
# an Apple Silicon Mac here, not an Intel one.
_MAC_ROSETTA=false
if [ "$OS" = "macos" ] && [ "$_ARCH" = "x86_64" ]; then
    # Apple Silicon under Rosetta reports x86_64; hw.optional.arm64 stays "1".
    if [ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
        _MAC_ROSETTA=true
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

# Apple Silicon: override mlx-vlm / mlx-lm's transformers pin (see overrides file).
if [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _OVERRIDES_FILE="$(cd "$(dirname "$0" 2>/dev/null || echo ".")" && pwd)/studio/backend/requirements/single-env/overrides-darwin-arm64.txt"
    if [ -f "$_OVERRIDES_FILE" ]; then
        # uv splits UV_OVERRIDE on whitespace; hand uv a copy in a whitespace-free temp dir.
        case "$_OVERRIDES_FILE" in
            *[[:space:]]*)
                _UV_OVERRIDE_TMP_ROOT=${TMPDIR:-/tmp}
                case "$_UV_OVERRIDE_TMP_ROOT" in *[[:space:]]*) _UV_OVERRIDE_TMP_ROOT=/tmp ;; esac
                _UV_OVERRIDE_TMPDIR=$(mktemp -d "$_UV_OVERRIDE_TMP_ROOT/unsloth_uv.XXXXXX" 2>/dev/null) || _UV_OVERRIDE_TMPDIR=""
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

# AMD GPU name from the Windows host via WMI (discrete cards aren't in /proc/cpuinfo); cached ("-" = negative), bounded to 10s.
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
# 10s timeout when `timeout` exists, so a wedged nvidia-smi can't hang the installer.
_run_bounded() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 10 "$@"
    else
        "$@"
    fi
}

# True when CUDA_VISIBLE_DEVICES is "" or "-1" (every NVIDIA device deliberately hidden); nvidia-smi ignores it.
_cvd_hides_nvidia() {
    [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] || return 1
    _cvd_trim=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]')
    [ -z "$_cvd_trim" ] || [ "$_cvd_trim" = "-1" ]
}

# ── NVIDIA usable-GPU helper ──
# nvidia-smi -L primary, /proc/driver/nvidia/gpus/ sysfs fallback; a GPU hidden via CUDA_VISIBLE_DEVICES=""/-1 counts as NOT usable.
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
    # Fallback: one subdir per GPU under this path.
    if [ -d /proc/driver/nvidia/gpus ] && \
       [ -n "$(ls -A /proc/driver/nvidia/gpus 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

# Strix Halo ROCm-on-WSL only targets Ubuntu 24.04: re-run the install in an installed 24.04 distro, else fall through to CPU (never auto-create a distro).
_maybe_reroute_strixhalo_to_2404() {
    [ "${OS:-}" = "wsl" ] || return 0
    # An explicit index pin skips every GPU-driven reroute; whitespace-only overrides don't gate.
    _rr_pin=$(printf '%s' "${UNSLOTH_TORCH_INDEX_URL:-}${UNSLOTH_TORCH_INDEX_FAMILY:-}" | tr -d '[:space:]')
    [ -n "$_rr_pin" ] && return 0
    [ "${SKIP_TORCH:-false}" = "false" ] || return 0
    [ "${UNSLOTH_SKIP_ROCM_WSL_SETUP:-0}" = "1" ] && return 0
    [ "${UNSLOTH_WSL_REROUTED:-0}" = "1" ] && return 0
    [ -e /dev/dxg ] || return 0
    # A usable NVIDIA GPU means the CUDA path works here, so don't reroute for AMD.
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
    # The bootstrap only supports 24.04, so leave a 24.04 user alone.
    case "$_rr_ver" in 24.04) return 0 ;; esac
    # Without a 24.04 reroute target, stay CPU-only AND skip the origin-distro ROCm bootstrap.
    command -v wsl.exe >/dev/null 2>&1 || { UNSLOTH_SKIP_ROCM_WSL_SETUP=1; return 0; }
    # Whole-line match so "Ubuntu-24.04-test" can't masquerade; || true: no match is fine.
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
    # A --local checkout can't be replayed by a piped web install (the repo isn't in the target
    # distro), so tell the user to re-run there rather than silently run a different install.
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        substep "This is a --local install; re-run it from $_rr_target instead:" "$C_WARN"
        substep "  wsl -d $_rr_target -- bash -lc 'cd <your checkout> && ./install.sh --local'" "$C_WARN"
        substep "Continuing CPU-only in Ubuntu ${_rr_ver:-this distro} for now." "$C_WARN"
        UNSLOTH_SKIP_ROCM_WSL_SETUP=1
        return 0
    fi
    # Forward the caller's options/env so the rerouted install matches what was asked for.
    _rr_q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
    _rr_exports="set -o pipefail; export UNSLOTH_WSL_REROUTED=1"
    [ "$_STUDIO_HOME_REDIRECT" = "env" ] && _rr_exports="$_rr_exports; export UNSLOTH_STUDIO_HOME=$(_rr_q "$STUDIO_HOME")"
    # Forward explicit ROCm-bootstrap consent (e.g. Tauri) so the child auto-enables the GPU.
    [ "${UNSLOTH_ROCM_WSL_AUTO:-0}" = "1" ] && _rr_exports="$_rr_exports; export UNSLOTH_ROCM_WSL_AUTO=1"
    # Forward a pinned torch index; dropping it would revert the child to auto-detection.
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
    # pipefail so a failed download in a piped web install isn't masked by sh exiting 0 on empty
    # input (which would wrongly report success and exit 0 the parent installer).
    _rr_rc=0
    wsl.exe -d "$_rr_target" -- bash -lc "$_rr_exports; $_rr_cmd" || _rr_rc=$?
    if [ "$_rr_rc" -eq 0 ]; then
        exit 0
    fi
    # Tauri child exit 2 ([TAURI:NEED_SUDO]) asks the desktop app to elevate; propagate it.
    if [ "$TAURI_MODE" = true ] && [ "$_rr_rc" -eq 2 ]; then
        exit 2
    fi
    substep "Could not auto-continue in $_rr_target; run it yourself:" "$C_WARN"
    substep "  wsl -d $_rr_target -- bash -lc 'curl -fsSL https://unsloth.ai/install.sh | sh'"
    substep "Continuing CPU-only in Ubuntu ${_rr_ver:-this distro} for now." "$C_WARN"
    # Reroute failed; keep the later bootstrap from installing ROCm into this distro.
    UNSLOTH_SKIP_ROCM_WSL_SETUP=1
    return 0
}
_maybe_reroute_strixhalo_to_2404 || true

# ── Check system dependencies ──
tauri_log "STEP" "Checking system dependencies"

# Without the Xcode CLT, macOS still ships /usr/bin/git as a stub that errors and pops
# a GUI dialog, so `command -v git` is not enough -- only running it tells the truth.
_has_working_git() {
    command -v git >/dev/null 2>&1 || return 1
    # Executing the probe is the problem on macOS: /usr/bin/git is a Command Line Tools
    # shim, so `git --version` against it raises the "install the command line developer
    # tools" GUI dialog -- the probe firing the dialog it exists to detect. Answer from the
    # resolved path instead when it is that exact shim and no toolchain is selected.
    #
    # Narrow on purpose. Only /usr/bin/git is a shim: a Homebrew, MacPorts or Xcode.app git
    # earlier on PATH is a real binary and is still probed by executing it. Intel macOS can
    # also ship a working /usr/bin/git after CLT masking, so MAC_INTEL must probe that path;
    # only Apple Silicon treats it as the known dialog shim without execution. xcode-select
    # -p only asks which toolchain is selected and never prompts.
    # The hardware decides, not the shell: MAC_INTEL is also true for an x86_64 shell under
    # Rosetta, where /usr/bin/git is still the arm64 machine's dialog shim, so _MAC_ROSETTA
    # puts that host back on the non-executing branch.
    # $OS is the platform detected above, not a fresh `uname` call: this can run with a
    # scrubbed PATH where uname is not resolvable, and a failed probe there would silently
    # fall through to executing the shim. _CLT_GIT_SHIM is the shim path, overridable so
    # the branch is testable without a /usr/bin write.
    if [ "${OS:-}" = "macos" ] &&
       { [ "${MAC_INTEL:-false}" != true ] || [ "${_MAC_ROSETTA:-false}" = true ]; } &&
       [ "$(command -v git)" = "${_CLT_GIT_SHIM:-/usr/bin/git}" ] &&
       ! xcode-select -p >/dev/null 2>&1; then
        return 1
    fi
    git --version >/dev/null 2>&1
}

# macOS system-dependency check. A function so tests/sh can sed-extract it; the old
# inline form was untestable, which is why this gate shipped broken.
#
# The consumer install needs no developer toolchain: uv is a prebuilt binary, CPython
# is uv-managed, llama.cpp/whisper.cpp/Node are prebuilt downloads, and triton is
# skipped on macOS. Only `--local` needs git, for the unsloth-zoo git+https URL.
_check_macos_deps() {
    _clt_missing=false
    xcode-select -p >/dev/null 2>&1 || _clt_missing=true

    if [ "$STUDIO_LOCAL_INSTALL" = true ] && ! _has_working_git; then
        echo ""
        step "deps" "git is required for --local installs" "$C_ERR"
        substep "--local installs unsloth-zoo from git+https://github.com/unslothai/unsloth-zoo,"
        substep "which needs a working git. Install the Xcode Command Line Tools:"
        substep "  xcode-select --install"
        substep "Then re-run this script. A normal (non---local) install needs no compiler"
        substep "and no git -- it uses prebuilt binaries and wheels only."
        tauri_log "NEED_XCODE_CLT" "git"
        return 1
    fi

    if [ "$_clt_missing" = true ]; then
        # Not fatal, and no GUI dialog: firing xcode-select --install and exiting is
        # what stranded clean Macs.
        step "deps" "no Xcode Command Line Tools (not required)" "$C_WARN"
        substep "Unsloth installs prebuilt binaries and wheels, so no compiler is needed."
        substep "Install them only for a llama.cpp source build: xcode-select --install"
    elif command -v cmake >/dev/null 2>&1; then
        step "deps" "all system dependencies found"
    else
        # cmake is only for a source build, so its absence is not fatal.
        step "deps" "using prebuilt llama.cpp (cmake not found)" "$C_WARN"
        substep "Install cmake only if you want a source build: brew install cmake"
    fi
    return 0
}

# Linux/WSL system-dependency check. Same split as macOS, and a function for the same
# reason: tests/sh can extract it.
#
# Only a download transport is required. cmake, gcc and the libcurl headers exist
# solely for a llama.cpp source build the consumer path never does -- unslothai/
# llama.cpp publishes linux-x64/arm64 prebuilts for cpu, cuda12, cuda13, rocm and
# vulkan. Requiring them turned every non-apt distro into a hard exit 1 over unused
# tooling. git follows macOS: --local only.
_check_linux_deps() {
    _transport_missing=false
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        _transport_missing=true
    fi

    # Wanted, never required: git fetches the triton_kernels git+https requirement (a
    # training speedup), the rest serve the optional source build. Warn, never stop.
    _optional_missing=""
    command -v cmake       >/dev/null 2>&1 || _optional_missing="$_optional_missing cmake"
    _has_working_git                       || _optional_missing="$_optional_missing git"
    command -v gcc         >/dev/null 2>&1 || _optional_missing="$_optional_missing build-essential"
    command -v curl-config >/dev/null 2>&1 || _optional_missing="$_optional_missing libcurl4-openssl-dev"
    # Parameter expansion, not `sed`: sed may be absent on a minimal image, and a
    # failed `$(... | sed ...)` yields "" -- "all found" on a machine that has none.
    _optional_missing="${_optional_missing# }"

    if [ "$STUDIO_LOCAL_INSTALL" = true ] && ! _has_working_git; then
        echo ""
        step "deps" "git is required for --local installs" "$C_ERR"
        substep "--local installs unsloth-zoo from git+https://github.com/unslothai/unsloth-zoo,"
        substep "which needs git. Install it with your package manager, then re-run."
        substep "A normal (non---local) install needs no git and no compiler."
        return 1
    fi

    # The one fatal case: nothing can be downloaded. apt is the only distro family we
    # can drive unattended.
    if [ "$_transport_missing" = true ]; then
        if command -v apt-get >/dev/null 2>&1; then
            echo ""
            step "deps" "missing: curl" "$C_WARN"
            substep "Needed to download uv, Python and the prebuilt inference engine."
            _smart_apt_install curl
            echo ""
        else
            echo ""
            step "deps" "missing: curl (or wget)" "$C_ERR"
            substep "Unsloth needs one of them to download uv, Python and the prebuilt"
            substep "inference engine. Install one, then re-run setup:"
            substep "  Fedora/RHEL: sudo dnf install curl"
            substep "  Arch:        sudo pacman -S --needed curl"
            substep "  openSUSE:    sudo zypper install curl"
            return 1
        fi
    fi

    # Try apt for the optional set too; failing only costs the features warned about
    # below.
    if [ -n "$_optional_missing" ] && command -v apt-get >/dev/null 2>&1; then
        step "deps" "installing optional build tools: $_optional_missing" "$C_DIM"
        # Subshell because _smart_apt_install exits rather than returns, so `|| true`
        # alone would not catch it. _SMART_APT_OPTIONAL suppresses every escalation
        # path, so no install hinges on a prompt for tools nothing here needs.
        ( _SMART_APT_OPTIONAL=true; _smart_apt_install $_optional_missing ) || true
        _optional_missing=""
        command -v cmake       >/dev/null 2>&1 || _optional_missing="$_optional_missing cmake"
        _has_working_git                       || _optional_missing="$_optional_missing git"
        command -v gcc         >/dev/null 2>&1 || _optional_missing="$_optional_missing build-essential"
        command -v curl-config >/dev/null 2>&1 || _optional_missing="$_optional_missing libcurl4-openssl-dev"
        _optional_missing="${_optional_missing# }"
    fi

    if [ -n "$_optional_missing" ]; then
        step "deps" "using prebuilt llama.cpp (missing: $_optional_missing)" "$C_WARN"
        substep "Not required to run: Unsloth downloads a prebuilt inference engine."
        case " $_optional_missing " in
            *" git "*) substep "Without git the triton kernels training speedup is skipped." ;;
        esac
    else
        step "deps" "all system dependencies found"
    fi
    return 0
}

case "$OS" in
    macos)
        _check_macos_deps || exit 1
        ;;
    linux|wsl)
        _check_linux_deps || exit 1
        ;;
esac

# ── Install uv ──
tauri_log "STEP" "Installing uv package manager"
# 0.9.3 is the first uv whose managed-Python manifest carries CPython 3.13.9.
# Anything older tops out at 3.13.8, which cannot import torch (see PYTHON_SKIP),
# so a bare "3.13" request on an older uv resolves straight to the broken patch.
UV_MIN_VERSION="0.9.3"
# The floor before this raised it. An offline host may keep a uv between the two,
# since those installs worked without touching the network; below it the
# installer rejected the uv outright and still has to, or it proceeds on a uv
# missing flags it is about to be handed (--default-index, --torch-backend).
UV_OFFLINE_MIN_VERSION="0.8.16"

# Large bytecode-compiled installs can exceed uv's 60s default; use 180s ("0" disables).
: "${UV_COMPILE_BYTECODE_TIMEOUT:=180}"
export UV_COMPILE_BYTECODE_TIMEOUT

# Raise retries and read timeout for large wheel downloads (":=" keeps overrides).
: "${UV_HTTP_RETRIES:=5}"
export UV_HTTP_RETRIES
: "${UV_HTTP_TIMEOUT:=180}"
export UV_HTTP_TIMEOUT

# macOS: trust the system Keychain (TLS-inspecting proxies) via both UV_SYSTEM_CERTS (uv >= 0.11) and UV_NATIVE_TLS (uv 0.8.16-0.10.x); opt out with UV_SYSTEM_CERTS=0.
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

# Patch releases the stack cannot run, space separated.
#   3.13.8: python/cpython#139783 makes inspect.getsourcelines() drop a function
#   body when a decorator is followed by a comment, which is the shape torch
#   2.11's nn/modules/rnn.py has, and _overload_method reads its own source at
#   import time -- so `import torch` dies with IndentationError. 3.13.9 was an
#   expedited release carrying only that fix.
PYTHON_SKIP="3.13.8"

# Every entry above is skipped for one reason: it cannot `import torch`. A
# --no-torch install never imports it, so refusing the interpreter there would
# fail a GGUF-only setup on a locked-down host over a package it will not
# install. SKIP_TORCH is set well before any of this runs.
_python_skip_applies() {
    [ "$SKIP_TORCH" != true ]
}

_python_is_skipped() {  # full x.y.z version
    _python_skip_applies || return 1
    for _bad in $PYTHON_SKIP; do
        [ "$1" = "$_bad" ] && return 0
    done
    return 1
}

# uv picks the patch itself for a bare "3.13", so name a range it cannot satisfy
# with a skipped release rather than checking afterwards. uv accepts a PEP 440
# specifier as a python request, and the exclusions come straight from
# PYTHON_SKIP so there is one list to maintain.
#
# The exclusion is spelled "!=3.13.8" rather than ">=3.13.9" on purpose: a host
# that is offline, or whose uv is too old to know 3.13.9, may still have a
# perfectly good cached 3.13.7, and a floor would refuse it and fail the install
# outright. Measured against uv 0.10.7 with only 3.13.7 and 3.13.8 installed and
# --offline: "3.13" gives 3.13.8, ">=3.13.9,<3.14" errors, ">=3.13,<3.14,!=3.13.8"
# gives 3.13.7.
_python_request() {  # requested version -> what uv is asked for
    _python_skip_applies || { echo "$1"; return 0; }
    case "$1" in
        # An explicit patch is the caller's own choice, and a path or a uv
        # download name is not a version at all. Pass those through untouched.
        [0-9]*.[0-9]*.*|*/*|*\\*) echo "$1"; return 0 ;;
        [0-9]*.[0-9]*) ;;
        *) echo "$1"; return 0 ;;
    esac
    _req_minor=${1#*.}
    # Only a plain X.Y gets a range. "3.13rc1", or a relative path like
    # "3.13/bin/python" that slipped past the globs above, would otherwise reach
    # the arithmetic below, and dash aborts the whole install on "Illegal number".
    case "$_req_minor" in
        ''|*[!0-9]*) echo "$1"; return 0 ;;
    esac
    _req=">=$1,<${1%%.*}.$((_req_minor + 1))"
    for _bad in $PYTHON_SKIP; do
        case "$_bad" in
            "$1".*) _req="$_req,!=$_bad" ;;
        esac
    done
    echo "$_req"
}

_uv_version_ok() {  # uv command, floor (defaults to UV_MIN_VERSION)
    _floor=${2:-$UV_MIN_VERSION}
    _raw=$("$1" --version 2>/dev/null | awk '{print $2}') || return 1
    [ -n "$_raw" ] || return 1
    _ver=${_raw%%[-+]*}
    case "$_ver" in
        ''|*[!0-9.]*) return 1 ;;
    esac
    version_ge "$_ver" "$_floor" || return 1
    # Prerelease of the exact minimum (e.g. 0.7.14-rc1) is still below stable 0.7.14
    [ "$_ver" = "$_floor" ] && [ "$_raw" != "$_ver" ] && return 1
    return 0
}

# ── uv from a pinned release ──
# Same archive, destination and PATH treatment as astral's installer, but it fetches a
# data file with a pinned SHA-256 instead of a script it runs and deletes. Mirrors
# Install-UvFromRelease in install.ps1. Bumping the version means bumping every hash:
#   curl -sL https://github.com/astral-sh/uv/releases/download/<ver>/<asset>.sha256
#
# Only the four mainstream targets are pinned. musl, armv7 and the rest fall through to
# the caller's existing path rather than risk a wrong triple.
UV_PINNED_VERSION="0.12.1"

# Echoes the glibc minor version (the N in 2.N), or nothing when this is not a glibc host or
# the version cannot be read. "not musl" is not the same as "a glibc new enough to run the GNU
# build": astral's installer checks a minimum and drops to its musl-static archive below it, so
# a host we cannot positively confirm has to reach the fallback rather than take a binary that
# will not exec.
_uv_glibc_minor() {
    _ugm_line=$( (ldd --version 2>/dev/null || true) | head -1 )
    case "$_ugm_line" in *[Mm]usl*) return 1 ;; esac
    _ugm_ver=$(printf '%s\n' "$_ugm_line" | awk '{print $NF}')
    # getconf is the fallback for an ldd that prints no version, and for hosts with no ldd.
    case "$_ugm_ver" in
        2.[0-9]*) : ;;
        *) _ugm_ver=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $NF}') ;;
    esac
    case "$_ugm_ver" in 2.[0-9]*) : ;; *) return 1 ;; esac
    _ugm_minor=${_ugm_ver#2.}
    _ugm_minor=${_ugm_minor%%.*}
    case "$_ugm_minor" in "" | *[!0-9]*) return 1 ;; esac
    echo "$_ugm_minor"
    return 0
}

# Prints "<asset> <sha256>" for this host, or nothing when the host is not pinned.
_uv_pinned_asset() {
    _upa_os=$(uname -s 2>/dev/null || echo unknown)
    _upa_arch=$(uname -m 2>/dev/null || echo unknown)
    case "$_upa_os" in
        Linux)
            # A 64-bit kernel under a 32-bit userland reports x86_64 from uname but cannot load
            # a 64-bit binary, so ask the userland, not the kernel.
            [ "$(getconf LONG_BIT 2>/dev/null || echo 0)" = "64" ] || return 1
            # Rejects musl, an unreadable libc, and a glibc below astral's floor for the triple.
            _upa_glibc=$(_uv_glibc_minor) || return 1
            case "$_upa_arch" in
                x86_64|amd64)
                    [ "$_upa_glibc" -ge 17 ] 2>/dev/null || return 1
                    echo "uv-x86_64-unknown-linux-gnu.tar.gz 90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb" ;;
                aarch64|arm64)
                    [ "$_upa_glibc" -ge 28 ] 2>/dev/null || return 1
                    echo "uv-aarch64-unknown-linux-gnu.tar.gz 769d373e146692c639b5fbaae33b331c297a32e03d30448772051902df52bbf4" ;;
                *) return 1 ;;
            esac
            ;;
        Darwin)
            # Under Rosetta 2 a translated shell reports x86_64 on an Apple Silicon Mac. astral
            # reads the same sysctl and ships the native build; matching it keeps the uv the user
            # ends up with identical to the one they had before.
            if [ "$_upa_arch" = "x86_64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
                _upa_arch=arm64
            fi
            case "$_upa_arch" in
                x86_64)
                    echo "uv-x86_64-apple-darwin.tar.gz 69d9f9a00337f25a50dcb13882052da08b8469bac11091c98c5694c3c6721467" ;;
                arm64|aarch64)
                    echo "uv-aarch64-apple-darwin.tar.gz 77d2906988e8074fd43f2f329ec452ebbf9b0c257ba1c66451c71de70a6baf42" ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
    return 0
}

# Echoes the SHA-256 of "$1", or nothing when the host has no digest tool.
_uv_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
    fi
}

# Can a freshly downloaded binary run at all? Both ways it could hang are closed off: no stdin,
# so a build that prompts reads EOF, and a ceiling where `timeout` exists (stock macOS has none).
# A healthy uv answers in milliseconds, so only a binary we would refuse reaches the ceiling.
_uv_probe_exec() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 20 "$1" --version >/dev/null 2>&1 </dev/null
    else
        "$1" --version >/dev/null 2>&1 </dev/null
    fi
}

_uv_install_pinned() {
    _uip_spec=$(_uv_pinned_asset) || return 1
    [ -n "$_uip_spec" ] || return 1
    _uip_asset=${_uip_spec%% *}
    _uip_want=${_uip_spec##* }
    # Unverified is worth less than astral's own release flow, so decline instead.
    command -v tar >/dev/null 2>&1 || return 1
    if [ -z "$(_uv_sha256 /dev/null)" ]; then return 1; fi

    # astral's destination priority, so an existing uv is replaced in place and the
    # PATH lines below still find it.
    _uip_dest=""
    for _uip_candidate in "${UV_INSTALL_DIR:-}" "${UV_UNMANAGED_INSTALL:-}" "${XDG_BIN_HOME:-}"; do
        if [ -n "$_uip_candidate" ]; then _uip_dest="$_uip_candidate"; break; fi
    done
    if [ -z "$_uip_dest" ] && [ -n "${XDG_DATA_HOME:-}" ]; then _uip_dest="$XDG_DATA_HOME/../bin"; fi
    if [ -z "$_uip_dest" ]; then
        [ -n "${HOME:-}" ] || return 1
        _uip_dest="$HOME/.local/bin"
    fi

    # 2>/dev/null: this is a speculative attempt whose failure falls back to astral's
    # installer, so an unusable $TMPDIR must not print a line the user cannot act on.
    _uip_work=$(mktemp -d 2>/dev/null) || return 1
    _UIP_WORK="$_uip_work"
    _uip_rc=1
    # astral's mirrors and precedence; each serves the identical asset, so one pin holds. A
    # configured mirror is EXCLUSIVE, as it is for astral: a restricted network sets one because
    # the public hosts are unreachable, and download() has no timeout, so trying them first
    # would hang rather than fall through.
    if [ -n "${UV_DOWNLOAD_URL:-}" ]; then
        _uip_bases="${UV_DOWNLOAD_URL%/}"
    elif [ -n "${INSTALLER_DOWNLOAD_URL:-}" ]; then
        _uip_bases="${INSTALLER_DOWNLOAD_URL%/}"
    elif [ -n "${UV_INSTALLER_GHE_BASE_URL:-}" ]; then
        _uip_bases="${UV_INSTALLER_GHE_BASE_URL%/}/astral-sh/uv/releases/download/$UV_PINNED_VERSION"
    elif [ -n "${UV_INSTALLER_GITHUB_BASE_URL:-}" ]; then
        _uip_bases="${UV_INSTALLER_GITHUB_BASE_URL%/}/astral-sh/uv/releases/download/$UV_PINNED_VERSION"
    else
        _uip_bases="https://releases.astral.sh/github/uv/releases/download/$UV_PINNED_VERSION
https://github.com/astral-sh/uv/releases/download/$UV_PINNED_VERSION"
    fi
    for _uip_base in $_uip_bases; do
        # 2>/dev/null: curl -sS prints its own errors and these attempts are speculative, so an
        # unreachable mirror stays off the console when the install still succeeds.
        if ! download "$_uip_base/$_uip_asset" "$_uip_work/$_uip_asset" 2>/dev/null; then continue; fi
        _uip_got=$(_uv_sha256 "$_uip_work/$_uip_asset")
        if [ "$_uip_got" != "$_uip_want" ]; then
            # Not tauri_log: [TAURI:WARN] is a marker install.sh has never emitted, and the app
            # forwards unknown markers to its progress UI verbatim. Verbose only, since the next
            # mirror or the fallback still runs.
            if _is_verbose; then
                echo "uv archive digest mismatch from $_uip_base, trying the next source" >&2
            fi
            continue
        fi
        # The POSIX archives hold uv and uvx under a uv-<triple>/ directory.
        if ! tar -xzf "$_uip_work/$_uip_asset" -C "$_uip_work" 2>/dev/null; then continue; fi
        mkdir -p "$_uip_dest" 2>/dev/null || break
        _uip_placed=0
        # uv first, and either half failing aborts the placement: the two ship as a set, and a
        # pinned uvx beside the host's older uv is a pairing we never built or tested.
        # Stage both, then publish both: the renames sit next to each other so the pair is
        # replaced as one, and a failure anywhere before them leaves the destination untouched.
        _uip_ready=1
        for _uip_exe in uv uvx; do
            # `mv f d` moves f INTO d and reports success, and a searchable directory passes -x
            # too, so a directory called uv at the destination would look like a published
            # binary and skip the fallback. The installer already refuses one for its own shim.
            if [ -d "$_uip_dest/$_uip_exe" ]; then _uip_ready=0; break; fi
            _uip_src=$(find "$_uip_work" -type f -name "$_uip_exe" 2>/dev/null | head -1)
            if [ -z "$_uip_src" ] || [ ! -f "$_uip_src" ]; then _uip_ready=0; break; fi
            # cp onto a symlinked destination writes through it and would rewrite, say, the
            # Homebrew binary `~/.local/bin/uv` points at; rename replaces the link. mktemp, not
            # a fixed name, so two installers racing here cannot publish each other's file.
            _uip_stage=$(mktemp "$_uip_dest/.$_uip_exe.XXXXXX" 2>/dev/null) || { _uip_ready=0; break; }
            if [ "$_uip_exe" = "uv" ]; then _UIP_STAGE="$_uip_stage"; else _UIP_STAGE2="$_uip_stage"; fi
            if ! cp -f "$_uip_src" "$_uip_stage" 2>/dev/null; then _uip_ready=0; break; fi
            # 0755, not +x: the staging file carries the umask default and +x only adds execute
            # where read was allowed, so umask 077 would leave uv unusable for every other
            # account. astral ships these 0755.
            chmod 0755 "$_uip_stage" 2>/dev/null || true
            # Validate BEFORE publishing: the rename destroys the incumbent, and a missing loader
            # or a noexec mount would leave the host with neither. The staging file is on the
            # destination filesystem, so this answers noexec too.
            if [ "$_uip_exe" = "uv" ] && ! _uv_probe_exec "$_uip_stage"; then _uip_ready=0; break; fi
        done
        if [ "$_uip_ready" = "1" ] &&
           mv -f "$_UIP_STAGE" "$_uip_dest/uv" 2>/dev/null &&
           mv -f "$_UIP_STAGE2" "$_uip_dest/uvx" 2>/dev/null; then
            _uip_placed=1
        fi
        rm -f "$_UIP_STAGE" "$_UIP_STAGE2" 2>/dev/null || true
        _UIP_STAGE=""
        _UIP_STAGE2=""
        # The staged binary already answered --version above, before it replaced anything.
        if [ "$_uip_placed" = "1" ] && [ -x "$_uip_dest/uv" ]; then
            export PATH="$_uip_dest:$PATH"
            # Where uv landed, for the profile write below: UV_INSTALL_DIR and friends can put
            # it outside ~/.local/bin, and that directory has to reach a new shell too.
            _UNSLOTH_UV_BIN_DIR="$_uip_dest"
            _uip_rc=0
        fi
        break
    done
    rm -rf "$_uip_work"
    _UIP_WORK=""
    _UIP_STAGE=""
    _UIP_STAGE2=""
    # Nothing is unwound on the failure path on purpose: the fallback installs over whatever is
    # at the destination, and deleting there would take out a working uv the host already had.
    return "$_uip_rc"
}

if ! command -v uv >/dev/null 2>&1 || ! _uv_version_ok uv; then
    # Raising the floor pulled every 0.8.16-0.9.2 host into this block, and those
    # installs used to succeed without touching the network, so a download
    # failure must not be fatal for them. An unreadable version counts as
    # present: that is a minimal image without awk, not an old uv.
    _uv_present_before=false
    if command -v uv >/dev/null 2>&1; then
        # `|| _uv_prev_ver=`: on an image with no awk the pipeline exits 127 and
        # set -e would kill the install here, which is exactly the host this
        # block exists to keep working.
        _uv_prev_ver=$(uv --version 2>/dev/null | awk '{print $2}' 2>/dev/null) \
            || _uv_prev_ver=""
        if [ -z "$_uv_prev_ver" ] || _uv_version_ok uv "$UV_OFFLINE_MIN_VERSION"; then
            _uv_present_before=true
        fi
    fi
    substep "installing uv package manager..."
    _uv_refreshed=true
    # download() exits the shell outright when neither curl nor wget is present,
    # which an `if` cannot catch, so probe first: a minimal image with uv copied
    # in but no downloader must keep the install it had before the floor moved.
    if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then
        # Pinned release first: a digest-checked data file scores far lower than
        # download-run-delete, which is the literal shape of a dropper.
        if _uv_install_pinned; then
            :
        else
            # Unpinned hosts keep the path they have always had: a wrong triple
            # breaks the install outright, which costs more than the fallback's score.
            _uv_tmp=$(mktemp)
            if download "https://astral.sh/uv/install.sh" "$_uv_tmp"; then
                run_maybe_quiet sh "$_uv_tmp" </dev/null || _uv_refreshed=false
            else
                _uv_refreshed=false
            fi
            rm -f "$_uv_tmp"
        fi
    else
        _uv_refreshed=false
    fi
    if [ "$_uv_refreshed" = false ] && [ "$_uv_present_before" = false ]; then
        tauri_log "ERROR" "Could not install uv"
        step "error" "could not download uv, and none is installed" "$C_ERR"
        substep "Check the network, or install uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    # ...and put the pinned destination back in front. UV_INSTALL_DIR and friends can put uv
    # somewhere other than ~/.local/bin, and both the line above and astral's env file prepend
    # ~/.local/bin, so a stale uv there would shadow the 0.12.1 we just verified.
    if [ -n "${_UNSLOTH_UV_BIN_DIR:-}" ] && [ "$_UNSLOTH_UV_BIN_DIR" != "$HOME/.local/bin" ]; then
        export PATH="$_UNSLOTH_UV_BIN_DIR:$PATH"
    fi
fi

# ── Create venv (migrate old layout if possible, otherwise fresh) ──
tauri_log "STEP" "Creating virtual environment"
mkdir -p "$STUDIO_HOME"

_MIGRATED=false
# Empty so an inherited value can never masquerade as a probed torch version.
_PREV_TORCH_VER=""

# Replace occupied venvs even when bin/python is missing or dangling, as in
# the repair loop reported in #9479.
if [ -x "$VENV_DIR/bin/python" ] || _dir_has_entries "$VENV_DIR"; then
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
    # Disk first, no interpreter: `import torch` can block forever on a wedged Intel driver,
    # and this runs before setup.sh's bounded probes. version.py carries the same label; the
    # interpreter stays as the fallback for a layout without one.
    _PREV_TORCH_VER=""
    for _prev_tv in "$VENV_DIR"/lib/python*/site-packages/torch/version.py; do
        [ -f "$_prev_tv" ] || continue
        _PREV_TORCH_VER=$(sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" "$_prev_tv" | head -n 1)
        break
    done
    # _run_bounded on the fallback: the disk read above already avoids the interpreter in
    # the normal case, but a layout without version.py still lands on `import torch`, which
    # is exactly what blocks forever on a wedged driver. No-ops where `timeout` is absent.
    [ -n "$_PREV_TORCH_VER" ] || _PREV_TORCH_VER=$(_run_bounded "$VENV_DIR/bin/python" -c \
        "import torch; print(torch.__version__)" 2>/dev/null | tail -n 1 || true)
    # New layout already exists — replace only after preserving rollback copy.
    substep "preserving existing environment for rollback..."
    # A bare call still aborts under `set -e`, but shows only mv's own stderr.
    # install.ps1 reports this step; say the same here and name the directory.
    if ! _start_studio_venv_replacement "$VENV_DIR"; then
        echo "ERROR: could not move $VENV_DIR aside to reinstall." >&2
        echo "       Check that $STUDIO_HOME is writable, or move $VENV_DIR yourself and re-run." >&2
        exit 1
    fi
elif [ "$_STUDIO_HOME_REDIRECT" != "env" ] && [ -x "$STUDIO_HOME/.venv/bin/python" ]; then
    # Old layout: validate before migrating (env-mode skips so an unrelated workspace .venv isn't rm -rf'd); no-torch validates Python only.
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
        # `mv` into an existing directory nests the environment inside it
        # ($VENV_DIR/.venv) rather than renaming it, and uv then refuses that
        # target as in #9479. This branch already means $VENV_DIR is absent or
        # empty, so clear it: rmdir cannot take one that gained an entry since
        # the check, and unlinking a symlink never touches its target.
        if [ -L "$VENV_DIR" ]; then
            rm -f "$VENV_DIR"
        elif [ -d "$VENV_DIR" ] && ! rmdir "$VENV_DIR" 2>/dev/null; then
            echo "ERROR: $VENV_DIR is in the way of the legacy migration." >&2
            echo "       Move it aside and re-run." >&2
            exit 1
        fi
        mv "$STUDIO_HOME/.venv" "$VENV_DIR"
        echo "   Moved ~/.unsloth/studio/.venv → $VENV_DIR"
        _MIGRATED=true
    else
        echo "⚠️  Legacy environment failed validation — creating fresh environment"
        _invalid_venv="$STUDIO_HOME/.venv.invalid.$(date +%Y%m%d%H%M%S 2>/dev/null || echo time).$$"
        mv "$STUDIO_HOME/.venv" "$_invalid_venv" 2>/dev/null || true
    fi
fi

# Recreate a stale Intel Mac 3.13 venv (skip when the user chose --python).
if [ "$SKIP_TORCH" = true ] && [ "$MAC_INTEL" = true ] && [ -z "$_USER_PYTHON" ] && [ -x "$VENV_DIR/bin/python" ]; then
    _PY_MM=$("$VENV_DIR/bin/python" -c \
        "import sys; print('{}.{}'.format(*sys.version_info[:2]))" 2>/dev/null || echo "")
    if [ "$_PY_MM" != "3.12" ]; then
        echo "  Recreating Intel Mac environment with Python 3.12 (was $_PY_MM)..."
        rm -rf "$VENV_DIR"
    fi
fi

# uv unconditionally invokes install_name_tool after downloading managed CPython on
# macOS. On a consumer Mac without developer tools, Apple's /usr/bin shim opens the
# Command Line Tools installer even though uv treats patch failure as a warning. There
# is no supported uv opt-out yet (https://github.com/astral-sh/uv/issues/14893).
#
# Do not execute install_name_tool or xcrun to probe it: either probe can launch the
# same dialog. A selected standalone CLT and full Xcode have stable on-disk locations.
_macos_has_selected_install_name_tool() {
    _uvv_developer_dir=$(xcode-select -p 2>/dev/null) || return 1
    [ -n "$_uvv_developer_dir" ] && [ -d "$_uvv_developer_dir" ] || return 1

    # DEVELOPER_DIR may select a custom path or symlink, so do not require Apple's
    # standard directory names. Reject only candidates that are the base-system
    # /usr/bin dialog shim itself; comparing file identity does not execute the tool.
    for _uvv_tool in \
        "$_uvv_developer_dir/usr/bin/install_name_tool" \
        "$_uvv_developer_dir/Toolchains/XcodeDefault.xctoolchain/usr/bin/install_name_tool"; do
        [ -x "$_uvv_tool" ] || continue
        if [ -e /usr/bin/install_name_tool ] \
           && [ "$_uvv_tool" -ef /usr/bin/install_name_tool ] 2>/dev/null; then
            continue
        fi
        return 0
    done
    return 1
}

# Run one uv venv command with its argv unchanged. If no real selected macOS tool is
# present, put a non-success shim ahead of /usr/bin only for uv's process. Returning
# nonzero is intentional: uv must retain its warning path rather than being told that
# an unpatched dylib was successfully modified.
_run_uv_venv() {  # label, uv-venv args...
    _uvv_label="$1"
    shift
    if [ "$OS" != "macos" ] || _macos_has_selected_install_name_tool; then
        run_install_cmd "$_uvv_label" uv venv "$@"
        return $?
    fi

    _UV_INSTALL_NAME_TOOL_SHIM_DIR=$(mktemp -d \
        "${TMPDIR:-/tmp}/unsloth-uv-install-name-tool.XXXXXX") || {
        echo "ERROR: could not create the temporary macOS uv guard." >&2
        tauri_stream_log stderr "ERROR_OUTPUT" "$_uvv_label failed (temporary guard)"
        return 1
    }
    if ! printf '%s\n' '#!/bin/sh' 'exit 1' \
            > "$_UV_INSTALL_NAME_TOOL_SHIM_DIR/install_name_tool" \
       || ! chmod +x "$_UV_INSTALL_NAME_TOOL_SHIM_DIR/install_name_tool"; then
        echo "ERROR: could not prepare the temporary macOS uv guard." >&2
        rm -rf "$_UV_INSTALL_NAME_TOOL_SHIM_DIR" 2>/dev/null || true
        _UV_INSTALL_NAME_TOOL_SHIM_DIR=""
        tauri_stream_log stderr "ERROR_OUTPUT" "$_uvv_label failed (temporary guard)"
        return 1
    fi

    if run_install_cmd "$_uvv_label" env \
        PATH="$_UV_INSTALL_NAME_TOOL_SHIM_DIR:$PATH" uv venv "$@"; then
        _uvv_status=0
    else
        _uvv_status=$?
    fi
    rm -rf "$_UV_INSTALL_NAME_TOOL_SHIM_DIR" 2>/dev/null || true
    _UV_INSTALL_NAME_TOOL_SHIM_DIR=""
    return "$_uvv_status"
}

# Apple Silicon venv. The arch-explicit arm64 CPython stops uv reusing a cached
# x86_64 (Rosetta) build: torch ships no macOS x86_64 wheels since 2.2.2, so an
# x86_64 venv cannot resolve torch. The arm64 guard below backstops older venvs.
#
# only-managed stops uv walking PATH and *executing* every interpreter it finds to
# read its version. Without CLT, /usr/bin/python3 is Apple's xcode_select shim, so
# that probe pops the "command line developer tools" dialog for tools this install
# never needs. Spelled --python-preference, not the --managed-python alias: same
# effect, accepted since uv 0.4.30 rather than 0.8.16.
#
# It also drops uv's system-interpreter fallback, so a host that is offline or has
# UV_PYTHON_DOWNLOADS=never is left with nothing to resolve. Retry unflagged for
# them: the dialog is worth removing, a failed install is not.
_uv_venv_arm64() {  # label
    _run_uv_venv "$1" "$VENV_DIR" \
        --python-preference only-managed \
        --python "cpython-${PYTHON_VERSION}-macos-aarch64-none" \
    || _run_uv_venv "$1 (system Python)" "$VENV_DIR" \
        --python "cpython-${PYTHON_VERSION}-macos-aarch64-none"
}

# Fedora sets python-downloads = "manual", so uv venv cannot fetch a matching
# interpreter. Install it only after that hint, then retry. An explicit install
# still honors "never"; UV_PYTHON_DOWNLOADS=automatic would not.
_uv_venv_requested() {  # label
    _uvvr_label="$1"
    _uvvr_req="$(_python_request "$PYTHON_VERSION")"
    # Capture the hint while streaming Unsloth output live. If capture setup fails,
    # use the original venv path. The global directory is owned by trap cleanup.
    _UV_VENV_CAPTURE_DIR=""
    if ! command -v tee >/dev/null 2>&1 \
       || ! _UV_VENV_CAPTURE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/unsloth-uv-venv.XXXXXX") \
       || ! mkfifo "$_UV_VENV_CAPTURE_DIR/out_pipe" "$_UV_VENV_CAPTURE_DIR/err_pipe"; then
        [ -n "$_UV_VENV_CAPTURE_DIR" ] && rm -rf "$_UV_VENV_CAPTURE_DIR" || true
        _UV_VENV_CAPTURE_DIR=""
        _run_uv_venv "$_uvvr_label" "$VENV_DIR" --python "$_uvvr_req"
        return $?
    fi
    _uvvr_out="$_UV_VENV_CAPTURE_DIR/out"
    _uvvr_err="$_UV_VENV_CAPTURE_DIR/err"
    # BSD tee supports -u; GNU tee is already unbuffered.
    tee -u /dev/null </dev/null >/dev/null 2>&1 && _uvvr_tee_u=-u || _uvvr_tee_u=
    tee $_uvvr_tee_u "$_uvvr_out" < "$_UV_VENV_CAPTURE_DIR/out_pipe" &
    _uvvr_tee_out=$!
    tee $_uvvr_tee_u "$_uvvr_err" < "$_UV_VENV_CAPTURE_DIR/err_pipe" >&2 &
    _uvvr_tee_err=$!
    if _run_uv_venv "$_uvvr_label" "$VENV_DIR" --python "$_uvvr_req" \
            >"$_UV_VENV_CAPTURE_DIR/out_pipe" 2>"$_UV_VENV_CAPTURE_DIR/err_pipe"; then
        _uvvr_status=0
    else
        _uvvr_status=$?
    fi
    wait "$_uvvr_tee_out" "$_uvvr_tee_err" 2>/dev/null || true
    if [ "$_uvvr_status" -eq 0 ]; then
        rm -rf "$_UV_VENV_CAPTURE_DIR"
        _UV_VENV_CAPTURE_DIR=""
        return 0
    fi
    if grep -q "Python downloads are set to 'manual'" "$_uvvr_out" "$_uvvr_err" 2>/dev/null \
       || grep -q "python-downloads" "$_uvvr_out" "$_uvvr_err" 2>/dev/null; then
        rm -rf "$_UV_VENV_CAPTURE_DIR"
        _UV_VENV_CAPTURE_DIR=""
        run_install_cmd "$_uvvr_label (managed Python)" \
            uv python install "$_uvvr_req" || return $?
        _run_uv_venv "$_uvvr_label" "$VENV_DIR" --python "$_uvvr_req" || return $?
        return 0
    fi
    rm -rf "$_UV_VENV_CAPTURE_DIR"
    _UV_VENV_CAPTURE_DIR=""
    return "$_uvvr_status"
}

if [ ! -x "$VENV_DIR/bin/python" ]; then
    step "venv" "creating Python ${PYTHON_VERSION} virtual environment"
    substep "$VENV_DIR"
    if [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ] && [ -z "$_USER_PYTHON" ]; then
        _uv_venv_arm64 "create venv"
    else
        _uv_venv_requested "create venv"
    fi
fi

# Mark the freshly-created venv as Unsloth-owned (env-mode deletion guard's primary sentinel).
if [ -x "$VENV_DIR/bin/python" ]; then
    : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
fi

# Two independent Apple Silicon venv guards: (1) x86_64 (Rosetta) venv -> recreate arm64; (2) Python 3.13.8 torch import bug. Re-inspect between checks (not elif); skip under --python.
if [ -z "$_USER_PYTHON" ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _inspect_venv() {
        "$VENV_DIR/bin/python" -c \
            "import platform, sys; print(platform.machine(), '{}.{}.{}'.format(*sys.version_info[:3]))" \
            2>/dev/null || echo " "
    }
    _info=$(_inspect_venv)
    _VENV_ARCH=${_info%% *}
    _PY_VER=${_info##* }
    # An unexecutable x86_64 venv python (no Rosetta) yields an empty arch; read the binary's Mach-O arch statically.
    if [ -z "$_VENV_ARCH" ] && [ -x "$VENV_DIR/bin/python" ]; then
        # uv symlinks bin/python to the base interpreter, so dereference with
        # file -L (lipo already follows the link). Trailing || true keeps the
        # installer alive under set -e when neither tool is present.
        #
        # file -L FIRST, lipo only as the fallback: lipo is a Command Line Tools shim, so
        # on a Mac without CLT merely running it raises the "install the command line
        # developer tools" dialog -- 2>/dev/null hides its stderr but not a GUI dialog.
        # file is base-system and always answers. Both spellings feed the same case below
        # ("Mach-O 64-bit executable arm64", or "universal binary ... [x86_64] [arm64]",
        # against lipo's "arm64" / "x86_64 arm64"), so the branch taken is unchanged.
        # clean-machine-assert.sh:152 already made exactly this swap for its own use.
        _archs=$(file -L "$VENV_DIR/bin/python" 2>/dev/null \
            || lipo -archs "$VENV_DIR/bin/python" 2>/dev/null || true)
        case "$_archs" in
            *arm64*)  _VENV_ARCH=arm64 ;;
            *x86_64*) _VENV_ARCH=x86_64 ;;
        esac
    fi

    if [ "$_VENV_ARCH" = "x86_64" ]; then
        echo "  WARNING: venv was created with an x86_64 (Rosetta) Python on Apple Silicon."
        echo "  Recreating venv with native arm64 Python ${PYTHON_VERSION}..."
        _discard_venv_for_recreate "$VENV_DIR"
        _uv_venv_arm64 "recreate venv (arm64)"
        if [ -x "$VENV_DIR/bin/python" ]; then
            : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
        fi
        # Re-inspect: the recreated arm64 venv may still be 3.13.8.
        _info=$(_inspect_venv)
        _VENV_ARCH=${_info%% *}
        _PY_VER=${_info##* }
    fi

    if _python_is_skipped "$_PY_VER"; then
        echo "  WARNING: Python $_PY_VER cannot import torch."
        echo "  Recreating venv with Python 3.12..."
        _discard_venv_for_recreate "$VENV_DIR"
        PYTHON_VERSION="3.12"
        _uv_venv_arm64 "recreate venv"
        if [ -x "$VENV_DIR/bin/python" ]; then
            : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
        fi
    fi
fi

# The request above only decides what a *new* venv gets. A venv from an earlier
# run, on any platform, can still hold a skipped interpreter, and reusing it is
# how the reported installs stayed broken across re-runs. Honour --python.
if [ -z "$_USER_PYTHON" ] && [ -x "$VENV_DIR/bin/python" ]; then
    _PY_VER=$("$VENV_DIR/bin/python" -c \
        'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))' 2>/dev/null || echo "")
    if _python_is_skipped "$_PY_VER"; then
        echo "  WARNING: Python $_PY_VER cannot import torch."
        echo "  Recreating venv..."
        _discard_venv_for_recreate "$VENV_DIR"
        _uv_venv_requested "recreate venv"
        if [ -x "$VENV_DIR/bin/python" ]; then
            : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
        fi
    fi
fi

if [ -x "$VENV_DIR/bin/python" ]; then
    step "venv" "using environment"
    substep "${VENV_DIR}"
fi

# Supported torch line: the default range admits torch 2.11 (wheels verified on
# cpu/cu126/cu128/cu130/rocm7.1+/mac arm64). Bump the three ceilings together
# when the next torch minor is validated; curated ROCm floors below stay literal.
_TORCH_CEILING="2.12.0"
_TORCHVISION_CEILING="0.27.0"
_TORCHAUDIO_CEILING="2.12.0"
# Default torch constraint; tightened for Python 3.13+ on arm64 macOS (torch <2.6 has no cp313 macOS arm64 wheels).
TORCH_CONSTRAINT="torch>=2.4,<${_TORCH_CEILING}"
if [ "$SKIP_TORCH" = false ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _PY_MINOR=$("$VENV_DIR/bin/python" -c \
        "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$_PY_MINOR" -ge 13 ] 2>/dev/null; then
        TORCH_CONSTRAINT="torch>=2.6,<${_TORCH_CEILING}"
    fi
fi
# Companion constraints bounded to torch's window: torchaudio 2.11 dropped its torch pin, so a bare companion can drift from a capped torch.
TORCHVISION_CONSTRAINT="torchvision>=0.19,<${_TORCHVISION_CEILING}"
TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<${_TORCHAUDIO_CEILING}"

# ── Resolve repo root (for --local installs) ──
_REPO_ROOT="$(cd "$(dirname "$0" 2>/dev/null || echo ".")" && pwd)"
# Whether the scripts next to install.sh may be trusted. A piped web install
# has $0 = "sh", so _REPO_ROOT is just the caller's cwd and a file planted there would run.
# Marker files cannot decide this (whoever can plant a helper can plant those), so require
# the explicit --local intent AND a run from the file itself; else fetch the official copy.
_REPO_IS_CHECKOUT=0
case "$0" in
    */install.sh|install.sh)
        [ "$STUDIO_LOCAL_INSTALL" = true ] && [ -r "$0" ] && _REPO_IS_CHECKOUT=1 ;;
esac

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
# WSL2 ROCDXG: rocminfo needs HSA_ENABLE_DXG_DETECTION=1 and /opt/rocm/bin can be off PATH; seed both or a ROCDXG WSL host misdetects as CPU-only.
_ensure_rocm_probe_env() {
    export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
    if ! command -v rocminfo >/dev/null 2>&1 && [ -x /opt/rocm/bin/rocminfo ]; then
        PATH="$PATH:/opt/rocm/bin"
    fi
}

# True if an AMD GPU is present (rocminfo, amd-smi, then KFD sysfs); always false when an NVIDIA GPU is present.
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

# rocminfo names each agent twice, so "gfx1201\ngfx1201" is one device, not two.
_amd_probe_arches() {
    printf '%s\n' "$1" | sed 's/:.*$//' | tr '[:upper:]' '[:lower:]' | awk 'NF' | sort -u
}

# The wheels must work on every AMD GPU in the box, so every agent has to land in the same
# family: routing on whichever the kernel enumerated first puts gfx1151 wheels on a 9070 XT.
_amd_agreed_index_family() {
    _aif_family=""
    for _aif_a in $(_amd_probe_arches "$1"); do
        _aif_f=$(_amd_arch_index_family_for_gfx "$_aif_a") || return 1
        [ -z "$_aif_family" ] || [ "$_aif_f" = "$_aif_family" ] || return 1
        _aif_family="$_aif_f"
    done
    [ -n "$_aif_family" ] || return 1
    printf '%s\n' "$_aif_family"
}

# setup.sh takes UNSLOTH_ROCM_GFX_ARCH over its own visibility-aware selection, so naming an
# arbitrary member of an agreed family (gfx1200 beside gfx1201) overrules a correct choice.
_amd_sole_index_arch() {
    _sia=$(_amd_probe_arches "$1")
    [ -n "$_sia" ] || return 1
    [ "$(printf '%s\n' "$_sia" | awk 'END{print NR}')" -eq 1 ] || return 1
    _amd_arch_index_family_for_gfx "$_sia" >/dev/null 2>&1 || return 1
    printf '%s\n' "$_sia"
}

# Map a gfx arch to the AMD pip index family (mirrors install.ps1 $archFamilyMap).
_amd_arch_index_family_for_gfx() {
    case "$1" in
        gfx1201|gfx1200) echo gfx120X-all ;;
        gfx1151) echo gfx1151 ;;
        gfx1150) echo gfx1150 ;;
        gfx1152) echo gfx1152 ;;
        gfx1103|gfx1102|gfx1101|gfx1100) echo gfx110X-all ;;
        gfx1036|gfx1035|gfx1034|gfx1033|gfx1032|gfx1031|gfx1030) echo gfx103X-all ;;
        gfx90a) echo gfx90a ;;
        gfx908) echo gfx908 ;;
        *) return 1 ;;
    esac
}

# Map a GPU marketing name to gfx arch (kept in sync with install.ps1 nameArchTable).
_infer_amd_gfx_arch_from_gpu_name() {
    case "$1" in
        *9070*|*9080*|*"R9700"*) echo gfx1201 ;;
        *9060*) echo gfx1200 ;;
        *"8065S"*|*"8060S"*|*"8050S"*|*"8040S"*|*"Strix Halo"*|*"Ryzen AI Max"*|*"AI Max"*) echo gfx1151 ;;
        *"890M"*|*"880M"*|*"Strix Point"*|*"HX 37"*|*"AI 9 HX"*|*"AI 9 36"*) echo gfx1150 ;;
        *"860M"*|*"840M"*|*"Krackan"*|*"AI 7 35"*|*"AI 5 34"*|*"AI 7 PRO 35"*|*"AI 5 33"*) echo gfx1152 ;;
        *"RX 7600"*|*"RX 7700S"*|*"RX 7650"*|*"PRO W7600"*|*"PRO W7500"*) echo gfx1102 ;;
        *"RX 7800"*|*"RX 7700"*|*"PRO W7700"*|*"PRO V710"*) echo gfx1101 ;;
        *"RX 7900"*|*"PRO W7900"*|*"PRO W7800"*) echo gfx1100 ;;
        *"780M"*|*"760M"*|*"740M"*|*"Phoenix"*|*"Hawk Point"*|*"Z1 Extreme"*|*"Z2 Extreme"*) echo gfx1103 ;;
        *"RX 6900"*|*"RX 6800"*|*"RX 6750"*|*"RX 6700"*|*"PRO W6800"*|*"PRO W6900"*) echo gfx1030 ;;
        *"RX 6650"*|*"RX 6600"*|*"PRO W6600"*|*"PRO W6650"*) echo gfx1032 ;;
        *"RX 6500"*|*"RX 6400"*|*"RX 6300"*|*"PRO W6400"*|*"PRO W6500"*) echo gfx1034 ;;
        *) return 1 ;;
    esac
}

# GPU name -> gfx arch for AMD generations Unsloth's ROCm wheels do NOT cover: RDNA 1
# and Polaris 10/20/30 (unslothai#8529). SEPARATE from _infer_amd_gfx_arch_from_gpu_name
# on purpose: messaging only, nothing here may route to a wheel index. AMD's TheRock
# ships RDNA 1 wheels, but not on the repo.amd.com indexes routed here, and never gfx803.
# ORDER IS LOAD-BEARING: `case` has no negative lookahead, so a *"RX 570"* arm would
# swallow an "RX 5700 XT" -- RDNA 1 arms come FIRST and Polaris last.
# Names from LLVM's AMDGPU tables plus libdrm amdgpu.ids/pci.ids for the Navi 10/14
# professional parts LLVM omits; nothing is guessed, so Polaris 11/12 (RX 460/550/560,
# a different die) is left out. Case-sensitive, unlike the regex copies: every source
# here (WMI, amd-smi, lspci) spells these names as pci.ids does.
_infer_unsupported_amd_gfx_arch_from_gpu_name() {
    case "$1" in
        *"Radeon Pro V520"*|*"Radeon Pro 5600M"*) echo gfx1011 ;;  # RDNA 1
        *"RX 5700"*|*"RX 5600"*|*"Radeon Pro 5600 XT"*|*"Radeon Pro 5700"*|*"Radeon Pro W5700"*) echo gfx1010 ;;  # RDNA 1 (Navi 10)
        *"RX 5500"*|*"RX 5300"*|*"Radeon Pro W5500"*|*"Radeon Pro W5300"*) echo gfx1012 ;;  # RDNA 1 (Navi 14)
        *"RX 470"|*"RX 470"[!0]*|*"RX 480"|*"RX 480"[!0]*|*"RX 570"|*"RX 570"[!0]*|*"RX 580"|*"RX 580"[!0]*|*"RX 590"|*"RX 590"[!0]*|*"Radeon Pro WX 7100"*|*"Radeon Pro WX 5100"*) echo gfx803 ;;  # Polaris 10/20/30
        *) return 1 ;;
    esac
}

# Linux counterpart: first AMD display-class lspci line naming a generation ROCm
# does not cover. Messaging only -- never feed the result into index selection.
_infer_linux_unsupported_amd_gfx_arch() {
    command -v lspci >/dev/null 2>&1 || return 1
    _unsup_disp=$(lspci -nn 2>/dev/null | grep -E 'VGA compatible controller|3D controller|Display controller' | grep -E 'AMD|ATI' || true)
    while IFS= read -r _unsup_ln; do
        [ -n "$_unsup_ln" ] || continue
        if _unsup_gfx=$(_infer_unsupported_amd_gfx_arch_from_gpu_name "$_unsup_ln"); then
            echo "$_unsup_gfx"
            return 0
        fi
    done <<EOF
$_unsup_disp
EOF
    return 1
}

# Best-effort gfx inference when ROCm tools can't see the GPU (unslothai#7301).
# Mirrors install.ps1 arch resolution on Windows ($HasROCm false, $ROCmGfxArch set).
_infer_linux_amd_gfx_arch() {
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        printf '%s\n' "$(printf '%s' "$UNSLOTH_ROCM_GFX_ARCH" | tr '[:upper:]' '[:lower:]')"
        return 0
    fi
    # On WSL /proc/cpuinfo and lspci still report the host APU, but without the
    # ROCDXG bridge (librocdxg over /dev/dxg) the AMD wheels can't reach the GPU;
    # keep the CPU fallback there unless that runtime is present (the explicit
    # override above still wins). Mirrors install_python_stack.py.
    _gpu_evidence=""
    if [ -e /dev/dxg ] || grep -qi microsoft /proc/version 2>/dev/null; then
        for _d in /opt/rocm/lib /opt/rocm/lib64 /opt/rocm-*/lib /opt/rocm-*/lib64; do
            { [ -e "$_d/librocdxg.so" ] || [ -e "$_d/librocdxg.so.1" ]; } && _rocdxg=1 && break
        done
        [ -n "${_rocdxg:-}" ] || return 1
        # WSL enumerates no PCI display device; /dev/dxg + librocdxg IS the
        # GPU evidence there.
        _gpu_evidence=1
    elif _amd_gpu_present_via_pci; then
        _gpu_evidence=1
    fi
    # /proc/cpuinfo leaks the HOST CPU model into VMs/containers that received
    # no AMD GPU, so the CPU-model text alone is not GPU evidence: require an
    # AMD display device (PCI vendor 0x1002, class 0x03*) before trusting it.
    # The lspci fallback below needs no gate; an AMD display line IS evidence.
    if [ -n "$_gpu_evidence" ] && grep -qiE 'Ryzen AI Max|Radeon 80[0-9][05]S|Strix Halo' /proc/cpuinfo 2>/dev/null; then
        echo gfx1151
        return 0
    fi
    if [ -n "$_gpu_evidence" ] && grep -qiE '890M|880M|Strix Point|HX 37[05]|AI 9 HX|AI 9 36[05]' /proc/cpuinfo 2>/dev/null; then
        echo gfx1150
        return 0
    fi
    if [ -n "$_gpu_evidence" ] && grep -qiE '860M|840M|Krackan|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33' /proc/cpuinfo 2>/dev/null; then
        echo gfx1152
        return 0
    fi
    if command -v lspci >/dev/null 2>&1; then
        # A non-AMD controller can enumerate first (Intel/ASPEED before an AMD
        # dGPU), so scan every display-class line and take the first AMD one
        # that maps. The vendor guard is case-SENSITIVE (a -i "ATI" would match
        # "CorporATIon" on every Intel/NVIDIA line); whole-line matching also
        # survives the 0000: PCI domain prefix. Mirrors install_python_stack.py.
        _amd_disp=$(lspci -nn 2>/dev/null | grep -E 'VGA compatible controller|3D controller|Display controller' | grep -E 'AMD|ATI' || true)
        while IFS= read -r _ln; do
            [ -n "$_ln" ] || continue
            if _gfx=$(_infer_amd_gfx_arch_from_gpu_name "$_ln"); then
                echo "$_gfx"
                return 0
            fi
        done <<EOF
$_amd_disp
EOF
    fi
    return 1
}

# gfx arch named by an HSA_OVERRIDE_GFX_VERSION value ($1), or nothing if it is not a
# readable major.minor.stepping triple. ROCr builds gfx<major><minor><stepping in hex>,
# which is why 9.0.10 is gfx90a: 11.0.0 -> gfx1100, 11.5.1 -> gfx1151, 10.3.0 -> gfx1030.
# Kept in sync with _hsa_override_gfx_arch in studio/install_python_stack.py.
_hsa_override_gfx_arch() {
    printf '%s' "${1:-}" | awk '
        {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "")
            if ($0 !~ /^[0-9]+\.[0-9]+\.[0-9]+$/) exit
            split($0, p, ".")
            maj = p[1] + 0; min = p[2] + 0; step = p[3] + 0
            # Steppings are a single hex nibble; wider is not a real target.
            if (maj <= 0 || min > 9 || step > 15) exit
            printf "gfx%d%d%x", maj, min, step
        }'
}

# gfx arches the KERNEL sees, one line per AMD GPU node, from KFD topology sysfs.
# amdkfd writes gfx_target_version itself, so it is immune to HSA_OVERRIDE_GFX_VERSION
# (which ROCr applies in userland) -- the ground truth for unslothai#7331. Encoding is
# major*10000 + minor*100 + stepping in hex: 110000 -> gfx1100, 110501 -> gfx1151,
# 90010 -> gfx90a. CPU nodes carry no gfx_target_version and drop out; vendor_id 4098
# (0x1002) keeps NVIDIA's open-driver KFD nodes out, mirroring _has_amd_rocm_gpu.
# Kept in sync with _kfd_gfx_targets in studio/install_python_stack.py.
_kfd_gfx_targets() {
    [ -d /sys/class/kfd/kfd/topology/nodes ] || return 0
    for _kfd_node in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$_kfd_node" ] || continue
        awk '
            /^[[:space:]]*vendor_id[[:space:]]/     { vendor = $2 }
            /^[[:space:]]*gfx_target_version[[:space:]]/ { gtv = $2 + 0 }
            END {
                if (vendor != 4098 || gtv <= 0) exit
                maj = int(gtv / 10000) % 100
                min = int(gtv / 100) % 100
                step = gtv % 100
                if (maj <= 0 || min > 9 || step > 15) exit
                printf "gfx%d%d%x\n", maj, min, step
            }' "$_kfd_node" 2>/dev/null || true
    done
    return 0
}

# Physical gfx arch when the ISA probe is an HSA_OVERRIDE_GFX_VERSION spoof
# (unslothai#7331): $1 = the arch inferred from the product name, $2 = the probed gfx
# token list. Prints the physical arch, or nothing to mean "believe the probe" (default).
#
# Requires ALL of the following, which keeps a mixed Strix APU + discrete AMD GPU host
# (the reason the probe outranks the product name at all) out of reach:
#   * HSA_OVERRIDE_GFX_VERSION is set -- with no override there is nothing to doubt;
#   * the product name inferred a spoofable RDNA 3.5 APU arch and the probe reported
#     a DIFFERENT one;
#   * the probe saw exactly ONE arch (rocminfo repeats the token per agent, so this
#     counts DISTINCT tokens -- a pre-filter, not the safety property);
#   * the variable names EXACTLY the arch that was reported: ROCr can only spoof to
#     the target the variable names, so any other reading is real silicon;
#   * a source the override cannot reach agrees with the product name: KFD sysfs
#     first (the kernel), then rocminfo re-run with the variable unset.
# Corroboration is REQUIRED, with deliberately no "the variable names the reported arch,
# so assume a spoof" fallback: that shape is identical on a host telling the truth (a real
# gfx1100 dGPU in a Ryzen AI Max chassis whose owner set the override for unrelated
# reasons), and rerouting a working machine to the wrong wheels is worse than
# unslothai#7331 itself.
# Kept in sync with _hsa_spoofed_physical_gfx in studio/install_python_stack.py.
_hsa_spoofed_physical_gfx() {
    _hsp_inferred="${1:-}"
    _hsp_probed_all="${2:-}"
    [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] || return 0
    case "$_hsp_inferred" in
        gfx1151|gfx1150|gfx1152) : ;;
        *) return 0 ;;
    esac
    # Exactly one DISTINCT arch, else the single-arch premise fails. Deduplicated because
    # the caller passes raw `rocminfo | grep -oE gfx...`, which repeats the token per
    # Name/ISA line -- counting lines would never fire on #7331's own host.
    _hsp_n=$(printf '%s\n' "$_hsp_probed_all" | awk 'NF && !seen[$0]++ { n++ } END { print n + 0 }')
    [ "${_hsp_n:-0}" -ne 1 ] && return 0
    _hsp_probed=$(printf '%s\n' "$_hsp_probed_all" | awk 'NF { print; exit }')
    [ -n "$_hsp_probed" ] || return 0
    [ "$_hsp_probed" = "$_hsp_inferred" ] && return 0
    # Only the arch the variable names can be a spoof of that variable's doing.
    [ "$(_hsa_override_gfx_arch "$HSA_OVERRIDE_GFX_VERSION")" = "$_hsp_probed" ] || return 0

    echo "  [WARN] HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is set; ROCm reports" >&2
    echo "  [WARN] $_hsp_probed but this host's product name is $_hsp_inferred. Checking for a spoof." >&2

    # 1. The kernel, which the override cannot reach. Decisive either way: if it answers
    # at all, no weaker source gets to overrule it.
    _hsp_kfd=$(_kfd_gfx_targets | awk 'NF')
    if [ -n "$_hsp_kfd" ]; then
        if [ "$_hsp_kfd" = "$_hsp_inferred" ]; then
            echo "  [WARN] KFD topology sysfs reports $_hsp_inferred -- $_hsp_probed is a spoof." >&2
            printf '%s\n' "$_hsp_inferred"
        else
            # On a real gfx1100 card in a Ryzen AI Max chassis this is the CORRECT outcome.
            echo "  [WARN] The kernel does not corroborate a spoof; keeping $_hsp_probed." >&2
        fi
        # Several GPU nodes: the single-arch premise was wrong (the spoof collapsed a
        # mixed host into one apparent arch). Decline.
        return 0
    fi

    # 2. The runtime, asked again without the override (ROCr getenv()s it while building
    # agent names, so stripping it retracts the spoofed name) and without the visible
    # masks, so a mask cannot hide the second GPU that would veto the correction. A
    # re-probe that still answers $_hsp_probed is evidence FOR the probe: the name did
    # not move, so it is real silicon. That is what keeps a genuine gfx1100 dGPU in a
    # Ryzen AI Max chassis on its own wheels.
    _hsp_re=""
    if command -v rocminfo >/dev/null 2>&1; then
        _hsp_re=$( (unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; \
                    rocminfo 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' | awk 'NF && !seen[$0]++' || true)
    fi
    # rocminfo FAILS on the very host this exists for: strip the override on a ROCm stack
    # predating the physical arch and ROCr has no ISA entry for it, so hsa_init errors and
    # no agent is listed. amd-smi reads the driver and is override-immune, and
    # _detect_amd_gfx_codes falls through to it, so mirror that here.
    if [ -z "$_hsp_re" ] && command -v amd-smi >/dev/null 2>&1; then
        _hsp_re=$( (unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; \
                    amd-smi list 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' | awk 'NF && !seen[$0]++' || true)
        if [ -z "$_hsp_re" ]; then
            _hsp_re=$( (unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; \
                        amd-smi static --asic 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' | awk 'NF && !seen[$0]++' || true)
        fi
    fi
    if [ -z "$_hsp_re" ]; then
        echo "  [WARN] Nothing left to re-probe and no KFD sysfs; keeping $_hsp_probed." >&2
    elif [ "$_hsp_re" = "$_hsp_inferred" ]; then
        echo "  [WARN] $_hsp_inferred reported with HSA_OVERRIDE_GFX_VERSION unset -- spoof confirmed." >&2
        printf '%s\n' "$_hsp_inferred"
    else
        echo "  [WARN] the re-probe does not corroborate a spoof; keeping $_hsp_probed." >&2
    fi
    return 0
}

# Reads the AMD gfx arch for wheel-index decisions: a user-set
# UNSLOTH_ROCM_GFX_ARCH is authoritative (lowercased), else rocminfo, then
# amd-smi. rocminfo/amd-smi honor ROCR/HIP_VISIBLE_DEVICES, so a container mask
# (e.g. ROCR_VISIBLE_DEVICES=-1) would hide a GPU that the env-independent KFD
# detection still sees -- the tool probes run with the masks cleared. Prints the
# gfx token(s) or nothing when unreadable, and always returns 0 (a failing probe
# as the last command would trip set -e in callers' assignments). Shared by
# get_torch_index_url's gfx gate and the runtime-less reroute gate so the two
# can never disagree on what "readable" means.
_probe_amd_gfx_arch() {
    _ensure_rocm_probe_env
    _pg=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]')
    if [ -z "$_pg" ] && command -v rocminfo >/dev/null 2>&1; then
        _pg=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; rocminfo 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
    fi
    if [ -z "$_pg" ] && command -v amd-smi >/dev/null 2>&1; then
        _pg=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; amd-smi list 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        if [ -z "$_pg" ]; then
            _pg=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; amd-smi static --asic 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        fi
    fi
    printf '%s\n' "$_pg"
}

# Classify the physical NVIDIA inventory for a cu126 fallback: "cu126" when it covers
# every GPU, "uncovered" for an incompatible mix, empty when no fallback is needed or the
# inventory is unreadable. CUDA_VISIBLE_DEVICES is ignored because the wheel must support
# the host. Shared decision with install.ps1 / setup.ps1 / install_python_stack.py.
_nvidia_cu126_verdict() {
    [ -n "$1" ] || return 0
    _ncv_caps=$(_run_bounded "$1" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null) || return 0
    printf '%s\n' "$_ncv_caps" | awk '
        { gsub(/^[[:space:]]+|[[:space:]]+$/, "") }   # match the .Trim()/.strip() siblings
        /^[0-9]+\.[0-9]+$/ {
            split($0, _sm, ".")
            _n = (_sm[1] * 10) + _sm[2]
            seen = 1
            if (_n < 75) legacy = 1
            if (_n < 50 || _n > 90) outside_cu126 = 1
            next
        }
        /./ { unreadable = 1 }
        END {
            if (!seen || unreadable || !legacy) exit
            print outside_cu126 ? "uncovered" : "cu126"
        }
    '
}

# Cap cu128/cu130 at cu126 when it covers every physical GPU: PyTorch 2.11's cu128/cu130
# start at sm_75, cu126 spans sm_50-90. Non-x86_64 keeps driver-only selection.
_cap_cuda_family_for_pre_turing() {
    case "$_ARCH" in
        x86_64|amd64) ;;
        *) printf '%s\n' "$1"; return ;;
    esac
    case "$1" in
        cu128|cu130) ;;
        *) printf '%s\n' "$1"; return ;;
    esac
    case "$(_nvidia_cu126_verdict "$2")" in
        cu126)
            echo "[WARN] Pre-Turing NVIDIA GPUs (sm_<75) are present -- selecting cu126, because PyTorch 2.11's $1 wheels start at sm_75." >&2
            printf '%s\n' "cu126"
            return
            ;;
        uncovered)
            echo "[WARN] This host mixes pre-Turing NVIDIA GPUs with GPUs that cu126 cannot serve; no PyTorch 2.11 CUDA family covers both." >&2
            echo "[WARN] Keeping $1, so the pre-Turing GPUs will be unusable. Set UNSLOTH_TORCH_INDEX_FAMILY=cu126 to choose the other way." >&2
            ;;
    esac
    printf '%s\n' "$1"
}

# ── ROCm version sources ──
# One helper per source, each returning 0 unconditionally: under set -e a failing source
# would kill the installer before the actionable warning at the end of the ROCm branch.
# Every source that execs runs through _run_bounded: highest-wins consults all five, so a
# single wedged probe would hang the installer. A timed-out probe just declined to answer.
_rocm_tag_from_amd_smi() {
    command -v amd-smi >/dev/null 2>&1 || return 0
    # Cut at the field separator and require digits: the line is pipe-delimited
    # ("... | ROCm version: N/A | amdgpu version: 6.10.10 | ..."), so stripping every
    # non-digit fabricated rocm6.10 out of the amdgpu driver version. Position used to
    # hide that; under highest-wins a fabricated reading outvotes a real 6.1.
    _run_bounded amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
        'NF>1{v=$2; sub(/[ \t|].*$/, "", v); if (v ~ /^[0-9]+\.[0-9]+/) {split(v,a,"."); print "rocm"a[1]"."a[2]} exit}' || return 0
}

_rocm_tag_from_version_file() {
    [ -r /opt/rocm/.info/version ] || return 0
    awk -F. '{print "rocm"$1"."$2; exit}' /opt/rocm/.info/version || return 0
}

# Naming pacman unconditionally told the unslothai#8731 reporter to run it on an
# immutable Fedora image, where it does not exist.
_rocm_sdk_install_hint() {
    if command -v pacman >/dev/null 2>&1; then
        echo "sudo pacman -S rocm-hip-sdk"
    elif command -v dnf >/dev/null 2>&1; then
        echo "sudo dnf install rocm-hip rocm-runtime   (rpm-ostree install ... on atomic images)"
    elif command -v zypper >/dev/null 2>&1; then
        echo "sudo zypper install rocm-hip"
    elif command -v apt-get >/dev/null 2>&1; then
        echo "see https://rocm.docs.amd.com/en/latest/deploy/linux/index.html for the AMD apt repo"
    else
        echo "https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
    fi
}

_rocm_tag_from_hipconfig() {
    # AMD's own installer puts hipconfig in ROCM_PATH/bin and leaves it off PATH unless
    # its profile.d snippet ran, so a PATH-only lookup misses a tree that is right there.
    # Not the Fedora shape: Fedora's hipconfig is /usr/bin/hipconfig, owned by hipcc.
    _rt_hipconfig=""
    if command -v hipconfig >/dev/null 2>&1; then
        _rt_hipconfig=hipconfig
    elif [ -x "${ROCM_PATH:-/opt/rocm}/bin/hipconfig" ]; then
        _rt_hipconfig="${ROCM_PATH:-/opt/rocm}/bin/hipconfig"
    else
        return 0
    fi
    _run_bounded "$_rt_hipconfig" --version 2>/dev/null \
        | awk 'NR==1 && /^[0-9]/{split($1,a,"."); if(a[1]+0>0){print "rocm"a[1]"."a[2]}}' || return 0
}

_rocm_tag_from_dpkg() {
    command -v dpkg-query >/dev/null 2>&1 || return 0
    # Require the status word "installed" ($4 of the three-word ${Status}): dpkg-query -W lists
    # every package except purged ones, so a removed-but-not-purged entry still reports its old
    # version and could outrank the live runtime under highest-wins. ${Status} over
    # ${db:Status-Status}: documented showformat field with no dpkg version floor, and an
    # unrecognised field renders empty rather than failing, so a dpkg lacking it goes silent.
    # `|| true` is load-bearing: dpkg-query exits nonzero when either package is absent while
    # still printing the other's line. rocm-core wins outright; libhsa-runtime64-1 is read only
    # in its absence (Debian ships no rocm-core), comes from the distro archive, and can be older.
    { _run_bounded dpkg-query -W -f='${Package} ${Status} ${Version}\n' rocm-core libhsa-runtime64-1 2>/dev/null || true; } \
        | awk '
            $4 == "installed" && $5 != "" {
                v = $5
                sub(/^[0-9]+:/, "", v)
                split(v, a, /[.-]/)
                if (a[1] !~ /^[0-9]+$/ || a[2] !~ /^[0-9]+$/) next
                if ($1 == "rocm-core") _core[_nc++] = "rocm" a[1] "." a[2]
                else                   _hsa[_nh++]  = "rocm" a[1] "." a[2]
            }
            END {
                if (_nc) { for (_i = 0; _i < _nc; _i++) print _core[_i] }
                else     { for (_i = 0; _i < _nh; _i++) print _hsa[_i] }
            }
        ' || return 0
}

_rocm_tag_from_rpm() {
    command -v rpm >/dev/null 2>&1 || return 0
    # Bounded, alone among the five, because this is the one source highest-wins
    # newly made unconditional that can block forever: it used to be LAST in a
    # first-answer-wins `||` chain, so /opt/rocm/.info/version answered at position
    # two and rpm was never invoked on a normal RHEL/SLES install. `rpm -q` is not a
    # lock-free read -- a leftover /var/lib/rpm/__db.00* from a killed rpm/yum wedges
    # plain queries in futex on the BerkeleyDB backend (rpm < 4.16, i.e. RHEL 8 /
    # SLES 15; rhbz#485780, rhbz#73097), and rpm 6.0.x deadlocks `rpm --query`
    # against a running dnf (rhbz#2463435). A version probe must not hang the
    # installer, and a timed-out probe is just a source that declined to answer.
    # _run_bounded no-ops where `timeout` is absent, so this adds no dependency.
    # Fedora ships rocm-core but nothing except the `rocm` metapackage requires it, so a
    # host running rocm-hip/rocm-runtime answered nothing (unslothai#8731). All names in
    # ONE query, since looping would pay the timeout above once per name; rpm reports
    # misses on stdout, so keep only lines starting with a digit. Every installed
    # component is emitted, not just the first: these are all AMD packages from the same
    # repo, so a partial upgrade can leave rocm-core 5.7 beside rocm-runtime 6.4, and
    # ranking by argument order would read that host as 5.7 and send a supported runtime
    # to CPU wheels. _highest_rocm_tag ranks them, as it does across the other sources.
    _rt_ver=$(_run_bounded rpm -q --qf '%{VERSION}\n' rocm-core rocm-runtime rocm-hip 2>/dev/null \
        | awk '/^[0-9]/{print}') || return 0
    [ -n "$_rt_ver" ] || return 0
    printf '%s\n' "$_rt_ver" | awk -F'[.-]' 'NF{print "rocm"$1"."$2}' || return 0
}

# Highest "rocmX.Y" line on stdin (major >= 1), or nothing when no line is usable.
_highest_rocm_tag() {
    awk '
        /^rocm[0-9]+\.[0-9]+$/ {
            split(substr($0, 5), a, ".")
            maj = a[1] + 0; min = a[2] + 0
            if (maj < 1) next
            if (!seen || maj > best_maj || (maj == best_maj && min > best_min)) {
                best_maj = maj; best_min = min; seen = 1
            }
        }
        END { if (seen) printf "rocm%d.%d\n", best_maj, best_min }
    '
}

# Consult EVERY source and take the highest, not the first that answers. Distros with split
# ROCm packaging ship one component well behind the runtime the GPU actually uses: Debian 13
# (and Linux Mint on top of it) packages hipconfig at 5.7.x next to a 6.1.x rocminfo/HSA, so
# first-answer resolution reported rocm5.7 on a working gfx1100 and the 6.0+ gate below sent
# it to CPU-only wheels (issue #8402). A source reading lower than another on the same host
# is stale packaging, not a downgrade. Overshoot is bounded: PyTorch's ROCm wheels vendor
# their own userspace and need only an amdgpu/KFD driver AMD documents as compatible +/- 2
# releases, the normalisation below can only emit a leaf PyTorch publishes, package-manager
# sources that can report an uninstalled tree are filtered above, and any disagreement is
# named on stderr for the install log.

# Path of the cross-subshell answer cache; set by the parent shell, empty when unused.
_ROCM_TAG_MEMO=""
_detect_rocm_version_tag() {
    if [ -n "${_ROCM_TAG_MEMO:-}" ] && [ -f "$_ROCM_TAG_MEMO" ]; then
        cat "$_ROCM_TAG_MEMO"
        return
    fi
    _rt_readings=$({
        _rocm_tag_from_amd_smi
        _rocm_tag_from_version_file
        _rocm_tag_from_hipconfig
        _rocm_tag_from_dpkg
        _rocm_tag_from_rpm
    } 2>/dev/null) || _rt_readings=""
    _rt_best=$(printf '%s\n' "$_rt_readings" | _highest_rocm_tag) || _rt_best=""
    if [ -n "$_rt_best" ]; then
        # Same shape gate as _highest_rocm_tag: a reading that was never a candidate
        # must not be named as a dissenting opinion.
        _rt_seen=$(printf '%s\n' "$_rt_readings" \
            | grep '^rocm[1-9][0-9]*\.[0-9][0-9]*$' | sort -u | tr '\n' ' ') || _rt_seen=""
        case "$_rt_seen" in
            ""|"$_rt_best ") : ;;  # one reading, or every source agreeing
            *) echo "[WARN] ROCm version sources disagree (${_rt_seen% }) -- using the highest, $_rt_best." >&2 ;;
        esac
    fi
    if [ -n "${_ROCM_TAG_MEMO:-}" ]; then
        printf '%s\n' "$_rt_best" > "$_ROCM_TAG_MEMO" 2>/dev/null || true
    fi
    printf '%s\n' "$_rt_best"
}

# ── Detect GPU and choose PyTorch index URL ──
# Mirrors Get-TorchIndexUrl in install.ps1.
# On CPU-only machines this returns the cpu index, avoiding the solver
# dead-end where --torch-backend=auto resolves to unsloth==2024.8.
get_torch_index_url() {
    _base="${UNSLOTH_PYTORCH_MIRROR:-https://download.pytorch.org/whl}"
    _base="${_base%/}"
    # Explicit override skips ALL GPU probing: UNSLOTH_TORCH_INDEX_URL wins (verbatim); UNSLOTH_TORCH_INDEX_FAMILY is the leaf appended to the mirror base; whitespace-only = unset.
    _url="${UNSLOTH_TORCH_INDEX_URL:-}"
    _url="${_url#"${_url%%[![:space:]]*}"}"; _url="${_url%"${_url##*[![:space:]]}"}"
    if [ -n "$_url" ]; then
        # Trim trailing PATH slashes (multi-slash 404s on strict proxies), preserving ?query/#fragment.
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
    # Require nvidia-smi to actually list a usable GPU; the binary alone would install CUDA wheels on AMD.
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
        # No NVIDIA GPU: check AMD ROCm. ROCm wheels are linux-x86_64 only; other arches fall back to CPU.
        case "$(uname -m)" in
            x86_64|amd64) : ;;
            *) echo "$_base/cpu"; return ;;
        esac
        if ! _has_amd_rocm_gpu; then
            echo "$_base/cpu"; return
        fi
        # A generic rocm index is only safe when the gfx arch is readable: the
        # Strix reroute (gfx1150/1151 -> arch-specific index) learns gfx from
        # rocminfo/amd-smi, so if those are missing OR do not enumerate the GPU, an
        # unknown-arch box might be Strix and would get the broken _grouped_mm
        # wheels. Probe via the shared helper (override first, then rocminfo/amd-smi
        # with visibility masks cleared); if the arch is unreadable, never guess a
        # rocm index. A KFD-only host whose arch is still inferable from hardware
        # IDs (PCI/cpuinfo/lspci) returns the cpu index and lets the runtime-less
        # reroute below upgrade it to AMD per-arch wheels -- the reroute gate uses
        # this same probe, so the handoff can't misfire. Only when inference fails
        # too is CPU final, with the actionable warning.
        _amd_gfx_probe=$(_probe_amd_gfx_arch)
        if [ -z "$_amd_gfx_probe" ]; then
            if _amd_inferred_gfx=$(_infer_linux_amd_gfx_arch 2>/dev/null) && \
               [ -n "$_amd_inferred_gfx" ] && \
               _amd_arch_index_family_for_gfx "$_amd_inferred_gfx" >/dev/null 2>&1; then
                echo "[WARN] AMD GPU detected but rocminfo/amd-smi can't read its gfx arch -- inferring $_amd_inferred_gfx from hardware IDs." >&2
                echo "$_base/cpu"; return
            fi
            # Repairing rocminfo cannot help here: the arch would read fine and still
            # have no wheels (unslothai#8529). Advice only, same CPU index either way.
            if _amd_unsup_gfx=$(_infer_linux_unsupported_amd_gfx_arch 2>/dev/null); then
                # Scoped to the card named, never to the host: a second AMD GPU here
                # may well have wheels, and nothing has looked at it yet.
                echo "[WARN] AMD GPU detected ($_amd_unsup_gfx) -- Unsloth has no ROCm PyTorch wheels for that arch, installing CPU PyTorch." >&2
                echo "[WARN] This is expected on this GPU; repairing rocminfo/amd-smi or setting UNSLOTH_ROCM_GFX_ARCH will not give it ROCm PyTorch." >&2
                # Torch ends here, llama.cpp does not. `export` is load-bearing: a bare
                # assignment never reaches the re-run (the unslothai#8458 mistake).
                echo "[INFO] GGUF chat can still use this GPU through Vulkan: export UNSLOTH_LLAMA_CPP_BACKEND=vulkan and re-run this installer (it selects the llama.cpp bundle at install time)." >&2
                echo "$_base/cpu"; return
            fi
            echo "[WARN] AMD GPU detected but its gfx arch can't be read (rocminfo/amd-smi missing or not enumerating the GPU) -- installing CPU-only PyTorch." >&2
            echo "[WARN] For GPU PyTorch, install or repair rocminfo/amd-smi (e.g. sudo pacman -S rocm-hip-sdk) and re-run this installer." >&2
            echo "$_base/cpu"; return
        fi
        # AMD GPU confirmed -- detect ROCm version
        _rocm_tag=""
        _rocm_tag=$(_detect_rocm_version_tag) || _rocm_tag=""
        # ^ || guard: belt and braces on the set -e contract the helpers hold, so a
        # fresh AMD host with no version source at all still reaches the actionable
        # no-version WARN below. stderr is deliberately not redirected: each source
        # already silences its own noise, leaving only the sources-disagree
        # breadcrumb, which belongs in the install log.
        # Shape gate on "rocmX.Y" with major >= 1: _highest_rocm_tag enforces it too,
        # kept so a future source cannot leak garbage into the cases below.
        case "$_rocm_tag" in
            rocm[1-9]*.[0-9]*) : ;;  # valid (major >= 1)
            *) _rocm_tag="" ;;        # reject malformed (empty, garbled, or major=0)
        esac
        if [ -n "$_rocm_tag" ]; then
            # Minimum supported: ROCm 6.0.
            case "$_rocm_tag" in
                rocm[1-5].*)
                    echo "[WARN] ROCm $_rocm_tag detected but PyTorch ROCm wheels require ROCm 6.0+ -- falling back to CPU-only PyTorch" >&2
                    echo "[WARN] $_rocm_tag is the HIGHEST version detected from usable ROCm sources; where dpkg has no rocm-core (Debian) the installed libhsa-runtime64-1 is read instead." >&2
                    echo "[WARN] Upgrade ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html" >&2
                    echo "[WARN] If this host really runs ROCm 6.0+ and only its packaging says otherwise, pin the wheels and re-run:" >&2
                    echo "[WARN]   UNSLOTH_TORCH_INDEX_FAMILY=rocm6.4   (a PyTorch wheel leaf: rocm6.0-6.4, rocm7.0-7.2)" >&2
                    echo "[WARN]   UNSLOTH_TORCH_INDEX_URL=<full index URL>   (takes precedence, used verbatim)" >&2
                    echo "$_base/cpu"; return ;;
            esac
            # Normalise to major.minor (no patch-level URLs); 6.5+ clips to rocm6.4, 7.3+ caps to rocm7.2.
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
                    # ROCm 6.5+: clip to the last supported 6.x wheel set.
                    echo "$_base/rocm6.4" ;;
                *)
                    # ROCm 7.3+: cap to rocm7.2 (latest known).
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
        # The version only picks between the generic rocmX.Y leaves, so an arch with its
        # own repo.amd.com/rocm/whl/gfx* index does not need one (unslothai#8731).
        _amd_gfx_family=$(_amd_agreed_index_family "$_amd_gfx_probe") || _amd_gfx_family=""
        if [ -n "$_amd_gfx_family" ]; then
            _amd_gfx_first=$(_amd_sole_index_arch "$_amd_gfx_probe") || _amd_gfx_first=""
            if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
                echo "[WARN] AMD GPU detected with no readable ROCm version, but UNSLOTH_ROCM_GFX_ARCH=${_amd_gfx_first:-$_amd_gfx_family} is set -- routing to AMD per-arch wheels." >&2
            else
                echo "[WARN] AMD ${_amd_gfx_first:-$_amd_gfx_family} detected but no ROCm version could be read -- routing to AMD per-arch wheels, which do not need one." >&2
            fi
            echo "$_base/cpu"; return
        fi
        echo "[WARN] AMD GPU detected, but no ROCm version could be read to select the matching GPU PyTorch build -- falling back to CPU-only PyTorch." >&2
        if [ -d "${ROCM_PATH:-/opt/rocm}" ]; then
            # Telling someone with a populated ROCm tree that "no ROCm install was found"
            # sends them off to install a package they already have. Fedora is the OTHER
            # branch: it owns no path under /opt/rocm, so it lands on the SDK hint.
            echo "[WARN] ${ROCM_PATH:-/opt/rocm} exists, so ROCm is likely installed but not reporting a version this installer can read." >&2
            echo "[WARN] Pin the wheels and re-run: UNSLOTH_TORCH_INDEX_FAMILY=rocm6.4   (a PyTorch wheel leaf: rocm6.0-6.4, rocm7.0-7.2)" >&2
        else
            echo "[WARN] Install the ROCm/HIP SDK, then re-run this installer:" >&2
            echo "[WARN]   $(_rocm_sdk_install_hint)" >&2
        fi
        echo "[WARN] Version sources checked: amd-smi, /opt/rocm/.info/version, hipconfig, dpkg, rpm (Debian runtime package: libhsa-runtime64-1)." >&2
        echo "$_base/cpu"; return
    fi
    # Parse CUDA version from nvidia-smi (POSIX-safe): accept both "CUDA Version:" and the newer "CUDA UMD Version:". Bounded, C locale.
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
    if [ "$_major" -ge 13 ]; then _cuda_tag=cu130
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 8 ]; then _cuda_tag=cu128
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 6 ]; then _cuda_tag=cu126
    elif [ "$_major" -ge 12 ]; then _cuda_tag=cu124
    elif [ "$_major" -ge 11 ]; then _cuda_tag=cu118
    else echo "$_base/cpu"; return; fi
    echo "$_base/$(_cap_cuda_family_for_pre_turing "$_cuda_tag" "$_smi")"
}

# ── Torch flavor helpers (to repair a stale CPU / wrong-CUDA wheel) ──
# torch.__version__ ($1) -> flavor tag (cuXXX / rocm / xpu / cpu); untagged wheel = cpu.
# The xpu arm is load-bearing: without it a +xpu wheel reads as "cpu" and is force-reinstalled
# on every run. Parity with the ps1 side.
_torch_flavor_tag() {
    case "$1" in
        *+cu[0-9]*) printf '%s\n' "$1" | sed -n 's/.*+\(cu[0-9][0-9]*\).*/\1/p' ;;
        *+rocm*)    echo "rocm" ;;
        *+xpu*)     echo "xpu" ;;
        *+cpu*)     echo "cpu" ;;
        "")         echo "" ;;
        *)          echo "cpu" ;;
    esac
}

# Final path segment of a wheel index URL ($1), lowercased, query/fragment stripped so .../cu128?token=x classifies as cu128. Shared with py / ps1.
_torch_index_url_leaf() {
    _tl_u="${1%%\?*}"
    _tl_u="${_tl_u%%#*}"
    # Strip ALL trailing slashes: .../rocm7.2// must yield rocm7.2, not empty.
    while [ -n "$_tl_u" ] && [ "${_tl_u%/}" != "$_tl_u" ]; do
        _tl_u="${_tl_u%/}"
    done
    printf '%s' "${_tl_u##*/}" | tr '[:upper:]' '[:lower:]'
}

# True when a lowercased leaf is an EXACT pip ROCm family (rocm<digits>[.<digits>] or gfx<digit>*); a leaf merely starting with rocm/gfx is a custom verbatim pin.
_is_pip_rocm_family_leaf() {
    case "$1" in
        gfx[0-9]*) return 0 ;;
        rocm[0-9]*)
            # Major/minor both non-empty all-digits (rocm7., rocm7.2.1, rocm7.2-private are custom pins).
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

# Whether release base $1 falls inside constraint window $2 ("torch>=A.B,<D.E.F") at major.minor granularity; unparseable answers "no".
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

# Keep the previous venv's torch RELEASE on a re-run when inside the constraint window; flavor follows the freshly chosen index. Opt out with UNSLOTH_TORCH_UPGRADE=1.
_previous_torch_pin() {
    _ptp_ver="$1"
    _ptp_con="$2"
    [ -n "$_ptp_ver" ] || { echo ""; return; }
    [ "${UNSLOTH_TORCH_UPGRADE:-0}" = "1" ] && { echo ""; return; }
    _ptp_base="${_ptp_ver%%+*}"
    # Base must be a plain numeric release; nightly/dev/source builds must never become a pin.
    case "$_ptp_base" in
        *[!0-9.]* | *..* | .* | *.) echo ""; return ;;
        [0-9]*.[0-9]*) ;;
        *) echo ""; return ;;
    esac
    [ "$(_torch_release_in_window "$_ptp_base" "$_ptp_con")" = "yes" ] || { echo ""; return; }
    echo "torch==$_ptp_base"
}

# Install torch from TORCH_INDEX_URL honoring a kept-release pin, falling back to the supported range if the index lacks it; used by every --default-index path.
_install_torch_default_index() {
    if [ -n "$_PREV_TORCH_PIN" ]; then
        # Pair companions with the kept torch minor (torchaudio no longer exact-pins torch).
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

# Expected tag from the index leaf ($1): cuXXX / cpu / xpu / rocm (rocmX.Y and gfx* ->
# rocm). Empty on an unknown leaf (odd mirror) so the repair safely no-ops.
_expected_torch_flavor_tag() {
    _leaf=$(_torch_index_url_leaf "$1")
    case "$_leaf" in
        cu[0-9]*)
            # Exact cu + digits only; a cu*-suffixed leaf (cu128-private) -> "" (custom).
            case "${_leaf#cu}" in
                *[!0-9]*) echo "" ;;
                *)        echo "$_leaf" ;;
            esac
            ;;
        cpu)          echo "cpu" ;;
        # Intel XPU (SYCL) is a GPU flavor, so a pinned xpu index repairs a stale CPU wheel.
        xpu)          echo "xpu" ;;
        # Exact rocm/gfx families only; a custom rocm*-suffixed leaf -> "" (custom).
        *)
            if _is_pip_rocm_family_leaf "$_leaf"; then echo "rocm"; else echo ""; fi
            ;;
    esac
}

# Installed torch's version label, for expected-flavor tag $1. The xpu path reads it off disk
# (as setup.sh's fast-path escape does): `import torch` can block forever on a wedged Intel
# driver. Other families keep the interpreter read.
_installed_torch_version_for_tag() {
    if [ "$1" = "xpu" ]; then
        for _itv in "$VENV_DIR"/lib/python*/site-packages/torch/version.py; do
            [ -f "$_itv" ] || continue
            sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" "$_itv" | head -n 1
            return
        done
        return
    fi
    "$_VENV_PY" -c "import torch; print(torch.__version__)" 2>/dev/null || true
}

# Whether index ($1) supports a plain --default-index reinstall. pytorch.org cuXXX /
# xpu / rocmX.Y AND the repo.amd.com gfx* indexes are all PEP 503 simple indexes that uv
# resolves (torch + every transitive dep) via --default-index -- the same URLs the
# fresh-install paths above already use -- so a stale wheel is auto-repairable.
# Unknown/odd-mirror leaves -> no, so we warn rather than risk a wrong reinstall.
_torch_index_repairable() {
    _leaf=$(_torch_index_url_leaf "$1")
    case "$_leaf" in
        cu[0-9]*) echo "yes" ;;
        # /whl/xpu is a plain PEP 503 index (oneAPI runtime and triton-xpu are ordinary deps).
        xpu)      echo "yes" ;;
        # Only EXACT rocm/gfx families resolve via --default-index; a suffixed leaf is verbatim.
        *)
            if _is_pip_rocm_family_leaf "$_leaf"; then echo "yes"; else echo "no"; fi
            ;;
    esac
}

# Remove credentials from a wheel index URL ($1): drops userinfo AND query/fragment. Shared with py / ps1.
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

# 0 when host version $1 (x.y or x.y.z) is no older than leaf version $2 (x.y), or $2 is empty
_radeon_host_ver_not_older() {
    [ -n "$1" ] || return 1
    [ -n "$2" ] || return 0
    _rh_maj=${1%%.*}; _rh_rest=${1#*.}; _rh_min=${_rh_rest%%.*}
    _rl_maj=${2%%.*}; _rl_rest=${2#*.}; _rl_min=${_rl_rest%%.*}
    case "$_rh_maj$_rh_min$_rl_maj$_rl_min" in *[!0-9]*) return 1 ;; esac
    if [ "$_rh_maj" -gt "$_rl_maj" ]; then return 0; fi
    if [ "$_rh_maj" -lt "$_rl_maj" ]; then return 1; fi
    [ "$_rh_min" -ge "$_rl_min" ]
}

get_radeon_wheel_url() {
    # Only meaningful on Linux. AMD publishes both M.m and M.m.p rocm-rel directories, so
    # both X.Y and X.Y.Z are valid leaf names here.
    case "$(uname -s)" in Linux) ;; *) echo ""; return ;; esac

    _full_ver=""
    _resolved_tag="${1:-}"
    _resolved_ver=""
    case "$_resolved_tag" in
        rocm[1-9]*.[0-9]*)
            _resolved_ver=$(printf '%s\n' "$_resolved_tag" \
                | awk '/^rocm[1-9][0-9]*\.[0-9][0-9]*$/ {sub(/^rocm/, ""); print; exit}')
            ;;
    esac
    _host_ver=$({ command -v amd-smi >/dev/null 2>&1 && \
        _run_bounded amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
            'NF>1{if(match($2,/[0-9]+\.[0-9]+(\.[0-9]+)?/)){print substr($2,RSTART,RLENGTH); ok=1; exit}} END{exit !ok}'; } || \
        { [ -r /opt/rocm/.info/version ] && \
            awk 'match($0,/[0-9]+\.[0-9]+(\.[0-9]+)?/){print substr($0,RSTART,RLENGTH); found=1; exit} END{exit !found}' /opt/rocm/.info/version; } || \
        { command -v hipconfig >/dev/null 2>&1 && \
            _run_bounded hipconfig --version 2>/dev/null | awk 'NR==1 && match($0,/[0-9]+\.[0-9]+(\.[0-9]+)?/){print substr($0,RSTART,RLENGTH); found=1} END{exit !found}'; }) 2>/dev/null || _host_ver=""
    if _radeon_host_ver_not_older "$_host_ver" "$_resolved_ver"; then
        _full_ver="$_host_ver"
    else
        _full_ver="$_resolved_ver"
    fi

    # Validate: must be X.Y or X.Y.Z with X >= 1
    case "$_full_ver" in
        [1-9]*.[0-9]*.[0-9]*) : ;;  # X.Y.Z
        [1-9]*.[0-9]*) : ;;          # X.Y
        *) echo ""; return ;;
    esac
    echo "https://repo.radeon.com/rocm/manylinux/rocm-rel-${_full_ver}/"
}

# ── Radeon repo wheel selection helpers ──────────────────────────────────────
_RADEON_LISTING=""
_RADEON_PYTAG=""
_RADEON_BASE_URL=""

_radeon_fetch_listing() {
    # Usage: _radeon_fetch_listing BASE_URL -- populates _RADEON_LISTING, _RADEON_PYTAG, _RADEON_BASE_URL.
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
    # Usage: _pick_radeon_wheel PACKAGE_NAME [VERSION_PREFIX] -- newest matching cpXY linux_x86_64 wheel URL; POSIX awk only (no grep -o / sort -V).
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
# Export the ROCm-on-WSL env + persist to /etc/profile.d (sudo-tee when not root); idempotent, no-op without librocdxg, best-effort.
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

_maybe_bootstrap_rocm_wsl() {
    [ "${OS:-}" = "wsl" ] || return 0
    [ "${SKIP_TORCH:-false}" = "false" ] || return 0
    [ "${UNSLOTH_SKIP_ROCM_WSL_SETUP:-0}" = "1" ] && return 0
    # Leave any already-usable GPU alone (NVIDIA, or working ROCm).
    if _has_usable_nvidia_gpu; then return 0; fi
    # Usable ROCm = rocminfo enumerates a real gfx[1-9] agent (not gfx000 / generic); awk consumes all input so rocminfo isn't SIGPIPE'd.
    _ensure_rocm_probe_env
    if command -v rocminfo >/dev/null 2>&1 && \
       rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx[1-9]/ && !/generic/{found=1} END{exit !found}'; then
        # Persist the drop-in so login shells inherit the transient probe env.
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

    # Fast path: configured (librocdxg present) but launched from a non-login shell -- just load it.
    if [ -e /opt/rocm/lib/librocdxg.so ] || [ -e /opt/rocm/lib64/librocdxg.so ]; then
        if [ -r /etc/profile.d/unsloth-rocm-wsl.sh ]; then
            # shellcheck disable=SC1091
            . /etc/profile.d/unsloth-rocm-wsl.sh || true
        else
            # librocdxg present but the env drop-in is gone (uninstall dropped it); restore it.
            _persist_rocm_wsl_dropin
        fi
        return 0
    fi

    echo ""
    _rw_gpu="$(_wsl_amd_gpu_name 2>/dev/null || true)"; [ -n "$_rw_gpu" ] || _rw_gpu="an AMD GPU"
    substep "Detected ${_rw_gpu} in WSL with no ROCm runtime yet." "$C_WARN"
    substep "Setting up ROCm-on-WSL (ROCm 7.2 + librocdxg) automatically to enable this GPU."
    substep "One-time, uses sudo and a large download. (skip: re-run with UNSLOTH_SKIP_ROCM_WSL_SETUP=1)"

    # Locate the helper: prefer the copy shipped beside install.sh, else fetch it. The local
    # copy counts only for a --local checkout run, since this executes with no prompt and
    # _REPO_ROOT may otherwise be the caller's cwd. The fetch pulls the same script.
    #
    # PINNED, never a branch: this runs unattended and installs with sudo, so a moving ref
    # would turn any rewrite of that branch into root code on every affected WSL box. Bump
    # it whenever the helper changes; lagging only means an older helper, and the gate below
    # rejects one too old to be safe.
    _ROCM_WSL_HELPER_REF="d3367edd9a1de7a0ac15aa899bd9cb97173679dc"
    # librocdxg pin (v1.2.2), forwarded to the helper. The ref IS the commit, so an older
    # helper that ignores the SHA still resolves this exact revision: its `--branch <sha>`
    # attempt fails and the full clone plus checkout land on it. Kept equal to the helper's
    # defaults; a test enforces that. A user-set ref wins and, with no SHA of its own, turns
    # the helper's check off rather than failing against our pin.
    _rw_dxg_ref="${UNSLOTH_LIBROCDXG_REF:-}"
    _rw_dxg_sha="${UNSLOTH_LIBROCDXG_SHA:-}"
    if [ -z "$_rw_dxg_ref" ]; then
        _rw_dxg_ref="4955d12888a3ec57057f1cf8660c2485e415e74c"
        [ -n "$_rw_dxg_sha" ] || _rw_dxg_sha="$_rw_dxg_ref"
    fi
    # A known SHA is authoritative, so forward it AS the ref: an operator pinning a branch
    # plus its expected commit would otherwise have that symbolic ref cloned unverified by a
    # helper old enough to ignore the SHA.
    if [ -n "$_rw_dxg_sha" ]; then
        _rw_dxg_ref="$_rw_dxg_sha"
    fi
    _rw_helper="${_REPO_ROOT:-.}/scripts/install_rocm_wsl_strixhalo.sh"
    _rw_tmp=""
    if [ "$_REPO_IS_CHECKOUT" != "1" ] || [ ! -r "$_rw_helper" ]; then
        _rw_tmp="$(mktemp 2>/dev/null || echo /tmp/_unsloth_rocm_wsl.sh)"
        if download "https://raw.githubusercontent.com/unslothai/unsloth/${_ROCM_WSL_HELPER_REF}/scripts/install_rocm_wsl_strixhalo.sh" "$_rw_tmp" 2>/dev/null; then
            _rw_helper="$_rw_tmp"
        else
            substep "Could not fetch the ROCm-on-WSL helper; using CPU fallback." "$C_WARN"
            [ -n "$_rw_tmp" ] && rm -f "$_rw_tmp"
            return 0
        fi
    fi

    # Run ONLY a helper declaring the contract (defined in its header): verifies the clone
    # against the pinned SHA, and treats an unresolvable checkout as fatal. One without it
    # swallows that failure and would build the repo's default HEAD as root once the pinned
    # ref stopped existing. Gating on the declaration is what makes this fail closed whatever
    # the pin, or a user's older checkout, supplies.
    if ! grep -q "^UNSLOTH_ROCM_WSL_HELPER_CONTRACT=2$" "$_rw_helper" 2>/dev/null; then
        substep "ROCm-on-WSL helper predates the pinned-source check; using CPU fallback." "$C_WARN"
        [ -n "$_rw_tmp" ] && rm -f "$_rw_tmp"
        return 0
    fi

    # Consent: the narrow guarded case is exactly the GPU setup the user ran the
    # installer for, so it proceeds AUTOMATICALLY by default (works with no TTY,
    # e.g. a piped web install). Opt out via UNSLOTH_SKIP_ROCM_WSL_SETUP=1 (top of
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
        if UNSLOTH_WSL_SMOKE_TEST=0 \
           UNSLOTH_LIBROCDXG_REF="$_rw_dxg_ref" UNSLOTH_LIBROCDXG_SHA="$_rw_dxg_sha" \
           bash "$_rw_helper"; then
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
# A pinned wheel index (UNSLOTH_TORCH_INDEX_URL / _FAMILY) skips the WSL ROCm bootstrap and the Radeon/Strix reroute below; whitespace trimmed first.
_torch_index_pinned=false
_ti_url_trim="${UNSLOTH_TORCH_INDEX_URL:-}"
_ti_url_trim="${_ti_url_trim#"${_ti_url_trim%%[![:space:]]*}"}"; _ti_url_trim="${_ti_url_trim%"${_ti_url_trim##*[![:space:]]}"}"
_ti_family_trim="${UNSLOTH_TORCH_INDEX_FAMILY:-}"
_ti_family_trim="${_ti_family_trim#"${_ti_family_trim%%[![:space:]]*}"}"; _ti_family_trim="${_ti_family_trim%"${_ti_family_trim##*[![:space:]]}"}"
if [ -n "$_ti_url_trim" ] || [ -n "$_ti_family_trim" ]; then
    _torch_index_pinned=true
fi
[ "$_torch_index_pinned" = true ] || _maybe_bootstrap_rocm_wsl || true

# Created here, not inside get_torch_index_url: that runs in a command substitution, so only
# a file outlives it. mktemp -d, never a $$-derived name -- a predictable path under a
# world-writable /tmp can be pre-created as a symlink, feeding the probe a chosen version.
_ROCM_TAG_MEMO_DIR=$(mktemp -d "${TMPDIR:-/tmp}/unsloth-rocm.XXXXXX" 2>/dev/null) \
    && _ROCM_TAG_MEMO="$_ROCM_TAG_MEMO_DIR/tag" || _ROCM_TAG_MEMO=""

TORCH_INDEX_URL=$(get_torch_index_url)

# Linux: ROCm runtime missing but a supported AMD gfx arch is inferable (Strix Halo
# in /proc/cpuinfo, lspci marketing name, UNSLOTH_ROCM_GFX_ARCH). Route to AMD's
# per-arch wheels like install.ps1 does on Windows (unslothai#7301).
# Gated on the runtime probes NOT naming a gfx: either no AMD GPU is detected at
# all (_has_amd_rocm_gpu false), or the GPU is visible only through the
# env-independent KFD topology while rocminfo/amd-smi can't read its arch
# (KFD-only host, unslothai#7314 -- before the KFD detection fix these hosts
# reached this reroute via the false branch, so the empty-probe condition
# preserves that routing). A */cpu index chosen WITH a readable gfx and a readable but
# UNSUPPORTED ROCm version is a deliberate fallback, and stays excluded because the shared
# probe returns its gfx. An UNREADABLE version is only a detection miss, so it gets its own
# way in below (unslothai#8731). UNSLOTH_ROCM_GFX_ARCH stays authoritative either way.

_amd_no_rocm_version_reroute=false
_amd_probed_gfx_first=""
case "$TORCH_INDEX_URL" in
    */cpu)
        if [ "$_torch_index_pinned" = false ] && [ "$SKIP_TORCH" = false ] && \
           [ -z "${UNSLOTH_ROCM_GFX_ARCH:-}" ] && \
           ! _has_usable_nvidia_gpu && _has_amd_rocm_gpu; then
            _amd_probe_out=$(_probe_amd_gfx_arch)
            # HSA_OVERRIDE_GFX_VERSION=11.0.0 is the standard Strix Halo workaround, and
            # ROCr then reports the spoofed gfx1100. The llama.cpp path corrects that
            # further down, which is too late for this branch: both the wheel family and
            # the exported arch are derived from this probe, so an uncorrected gfx1151
            # would take gfx110X-all wheels and export gfx1100 to setup.sh.
            _amd_spoof_inferred=$(_infer_linux_amd_gfx_arch 2>/dev/null || true)
            _amd_spoof_physical=$(_hsa_spoofed_physical_gfx "$_amd_spoof_inferred" "$_amd_probe_out")
            if [ -n "${_amd_spoof_physical:-}" ]; then
                _amd_probe_out="$_amd_spoof_physical"
            fi
            # An empty correction has two meanings and only one of them is "no spoof": the
            # helper also declines whenever the KFD reports more than one GPU node, because
            # the override can collapse several physical targets into one reported token.
            # Counting NODES rather than distinct arches deliberately matches the helper's
            # own rule -- it declines on two nodes even where they agree -- so a singleton
            # probe it refused to vouch for never picks a family on its own.
            if [ -z "${_amd_spoof_physical:-}" ] && [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && \
               [ "$(_kfd_gfx_targets | awk 'NF { n++ } END { print n + 0 }')" -gt 1 ]; then
                _amd_probe_out=""
            fi
            _amd_probed_family=$(_amd_agreed_index_family "$_amd_probe_out") \
                || _amd_probed_family=""
            _amd_probed_gfx_first=$(_amd_sole_index_arch "$_amd_probe_out") \
                || _amd_probed_gfx_first=""
            if [ -n "${_amd_probed_family:-}" ] && \
               [ -z "$(_detect_rocm_version_tag 2>/dev/null)" ]; then
                _amd_no_rocm_version_reroute=true
            fi
        fi
        ;;
esac
if [ "$_torch_index_pinned" = false ] && [ "$SKIP_TORCH" = false ] && \
   ! _has_usable_nvidia_gpu && \
   { [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ] || ! _has_amd_rocm_gpu || \
     [ -z "$(_probe_amd_gfx_arch)" ] || \
     [ "${_amd_no_rocm_version_reroute:-false}" = true ]; } && \
   case "$(uname -s)" in Linux) true ;; *) false ;; esac && \
   case "$_ARCH" in x86_64|amd64) true ;; *) false ;; esac; then
    # ROCm torch wheels are x86_64-only; get_torch_index_url returns CPU on other
    # arches, so an inferred/overridden gfx must not reroute arm64 to AMD wheels.
    case "$TORCH_INDEX_URL" in
        */cpu)
            _linux_inferred_gfx=$(_infer_linux_amd_gfx_arch 2>/dev/null || true)
            # Inference hands an explicit override back verbatim, and HIP's gcnArchName
            # carries feature flags (gfx1201:sramecc+:xnack-) the index table has no arm
            # for, so the suffix silently cost the reroute. Normalise as the caller did.
            _linux_inferred_gfx=$(_amd_sole_index_arch "$_linux_inferred_gfx") \
                || _linux_inferred_gfx=""
            # Route on the arch the probe READ, not on lspci marketing-name inference:
            # the two disagree on a mixed APU + discrete host, and inference's answer
            # would install wheels for a GPU the reroute decision never looked at.
            if [ "${_amd_no_rocm_version_reroute:-false}" = true ]; then
                _linux_inferred_gfx="${_amd_probed_gfx_first:-}"
            fi
            _amd_family=""
            if [ "${_amd_no_rocm_version_reroute:-false}" = true ]; then
                _amd_family="${_amd_probed_family:-}"
            elif [ -n "$_linux_inferred_gfx" ]; then
                _amd_family=$(_amd_arch_index_family_for_gfx "$_linux_inferred_gfx") || _amd_family=""
            fi
            if [ -n "$_amd_family" ]; then
                    _amd_mirror="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}"
                    while [ "${_amd_mirror%/}" != "$_amd_mirror" ]; do
                        _amd_mirror="${_amd_mirror%/}"
                    done
                    TORCH_INDEX_URL="${_amd_mirror}/${_amd_family}/"
                    # Hand the arch to setup.sh (llama.cpp): it re-probes ROCm on its
                    # own, and on these runtime-less hosts its probes find nothing, so
                    # without this it classifies the box as non-ROCm and installs the
                    # CPU prebuilt while torch just got AMD per-arch wheels. setup.sh
                    # and install_llama_prebuilt.py both honor UNSLOTH_ROCM_GFX_ARCH,
                    # so exporting it is the whole handoff (a user-set override
                    # re-exports unchanged).
                    if [ -n "$_linux_inferred_gfx" ]; then
                        export UNSLOTH_ROCM_GFX_ARCH="$_linux_inferred_gfx"
                    fi
                    # A corroborated spoof has to be cleared here as well. The rocm* leaf
                    # below does it, but this reroute produces a gfx* leaf, which that case
                    # never matches: the host would install native $_linux_inferred_gfx
                    # wheels while ROCr kept reporting the spoofed arch, leaving the kernels
                    # in those wheels unusable. SKIP_TORCH is false on this branch by its
                    # own guard, so the wheels really are going in.
                    if [ "${_amd_no_rocm_version_reroute:-false}" = true ] && \
                       [ -n "${_amd_spoof_physical:-}" ]; then
                        unset HSA_OVERRIDE_GFX_VERSION
                        echo "  [WARN] Clearing HSA_OVERRIDE_GFX_VERSION for the rest of this install:" >&2
                        echo "  [WARN] these wheels carry $_linux_inferred_gfx kernels, so the runtime has" >&2
                        echo "  [WARN] to report the real arch. Remove the export from your shell profile" >&2
                        echo "  [WARN] (~/.bashrc, ~/.profile) as well, or the next terminal restores it." >&2
                    fi
                    # Off the family, not the arch: no family straddles this boundary.
                    case "$_amd_family" in
                        gfx120X-all|gfx1151|gfx1150|gfx1152)
                            TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
                            TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
                            TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
                            ;;
                    esac
                    echo "" >&2
                    # KFD-only hosts reach this reroute with /dev/kfd present
                    # (that's what detected them), so don't claim it's missing.
                    if [ "${_amd_no_rocm_version_reroute:-false}" = true ]; then
                        echo "  [WARN] AMD ${_linux_inferred_gfx:-$_amd_family} detected, but no ROCm version could be read (checked amd-smi, /opt/rocm/.info/version, hipconfig, dpkg, rpm)." >&2
                        echo "  [WARN] The per-arch index is keyed on the arch alone, so the version is not needed." >&2
                    elif _has_amd_rocm_gpu; then
                        echo "  [WARN] AMD GPU visible via the kernel driver (KFD) but rocminfo/amd-smi can't read its gfx arch; using $_linux_inferred_gfx." >&2
                    else
                        echo "  [WARN] ROCm runtime not visible (/dev/kfd, rocminfo, amd-smi) but $_linux_inferred_gfx inferred." >&2
                    fi
                    echo "  [WARN] Routing to AMD arch-specific wheels ($(_strip_index_url_credentials "$TORCH_INDEX_URL"))." >&2
                    echo "  [WARN] These wheels bundle their own ROCm runtime; install the kernel stack for native compute:" >&2
                    echo "  [WARN]   https://docs.unsloth.ai/get-started/install-and-update/amd" >&2
                    if [ -n "$_linux_inferred_gfx" ]; then
                        echo "  [WARN] Tip: set UNSLOTH_ROCM_GFX_ARCH=$_linux_inferred_gfx to skip inference next time." >&2
                    else
                        echo "  [WARN] Two AMD GPUs of different archs share this wheel family; set UNSLOTH_ROCM_GFX_ARCH to name the one llama.cpp should build for." >&2
                    fi
                    echo "" >&2
            fi
            ;;
    esac
fi

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
# Strip ALL trailing slashes: .../cu128// must yield cu128, not empty.
while [ -n "$_torch_index_leaf" ] && [ "${_torch_index_leaf%/}" != "$_torch_index_leaf" ]; do
    _torch_index_leaf="${_torch_index_leaf%/}"
done
_torch_index_leaf="${_torch_index_leaf##*/}"
_torch_index_leaf=$(printf '%s' "$_torch_index_leaf" | tr '[:upper:]' '[:lower:]')
# Whether the caller had already STATED a backend before the assignment below overwrites it.
# setup.sh documents UNSLOTH_TORCH_BACKEND=cpu as the way to keep a deliberate CPU install,
# and on a GPU-less host the resolved value is cpu too, so without this the manifest cannot
# tell a stated choice from the automatic answer.
if [ -n "${UNSLOTH_TORCH_BACKEND:-}" ]; then
    _torch_backend_was_stated=true
    _torch_backend_stated_value=$(printf '%s' "$UNSLOTH_TORCH_BACKEND" | tr '[:upper:]' '[:lower:]')
else
    _torch_backend_was_stated=false
    _torch_backend_stated_value=""
fi
case "$_torch_index_leaf" in
    rocm*|gfx*) export UNSLOTH_TORCH_BACKEND="rocm" ;;
    cpu)        export UNSLOTH_TORCH_BACKEND="cpu"  ;;
    cu[0-9]*)   export UNSLOTH_TORCH_BACKEND="cuda" ;;
    # Unknown leaf: unset so a stale inherited value can't leak and the stack probes the GPU.
    *)          unset UNSLOTH_TORCH_BACKEND ;;
esac

# Derived from the index this script RESOLVED, which on a GPU-less machine is "cpu" whether
# or not anyone asked. Without the marker every ordinary Linux CPU install is recorded as a
# deliberate choice, and a machine that later gains a GPU is never offered the repair.
# Only when the stated family SURVIVED the resolution. The case above has already
# overwritten the variable, so a caller who said "cuda" on a machine with no visible GPU
# now carries the resolved "cpu" -- and treating that as stated records a deliberate CPU
# flavor for a host that asked for CUDA and did not get it. It would then be denied the
# repair for good if the GPU ever became visible.
if [ -n "${UNSLOTH_TORCH_BACKEND:-}" ] &&
   { [ "$_torch_backend_was_stated" != true ] ||
     [ "$_torch_backend_stated_value" != "$UNSLOTH_TORCH_BACKEND" ]; }; then
    export UNSLOTH_TORCH_BACKEND_SOURCE="resolved"
else
    unset UNSLOTH_TORCH_BACKEND_SOURCE
fi

# Whether TORCH_INDEX_URL names an actual pip ROCm family (rocm<digit>* / gfx*), gating the
# ROCm-only side effects below (AMD bitsandbytes, ROCm-torch repair). Digit-gated so a leaf
# merely STARTING with "rocm" isn't force-repaired from the wrong path.
if _is_pip_rocm_family_leaf "$_torch_index_leaf"; then
    _torch_index_is_rocm_family=true
else
    _torch_index_is_rocm_family=false
fi

# rocm7.2 and the per-gfx indexes (Strix _grouped_mm fix) ship torch 2.11.0: raise the floor and pin companions; match the FINAL leaf only.
# (cu*/cpu/custom leaves all use the default <2.12 trio above.)
case "$_torch_index_leaf" in
    rocm7.2|gfx120x-all|gfx1151|gfx1150|gfx1152)
        TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
        TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
        TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
        ;;
    # Floor 2.6, not the generic 2.4: unsloth/models/_utils.py raises at import for an XPU
    # device below it, so a mirror serving an older +xpu wheel would install something that
    # cannot run. Reached only through an explicit pin (no Intel autodetect on this side).
    xpu)
        TORCH_CONSTRAINT="torch>=2.6,<2.11.0"
        TORCHVISION_CONSTRAINT="torchvision>=0.21,<0.26.0"
        TORCHAUDIO_CONSTRAINT="torchaudio>=2.6,<2.11.0"
        ;;
esac

# Detect a Radeon card (*/rocm* index + rocminfo "Marketing Name:.*Radeon"); skipped when the index is pinned.
_amd_gpu_radeon=false
if [ "$_torch_index_pinned" = false ]; then
# On the LEAF, like every other index classifier here: the AMD per-arch mirror is
# https://repo.amd.com/ROCM/whl/gfx120X-all/, so a whole-URL */rocm* glob brands every
# per-arch reroute as Radeon and the summary then reports repo.radeon.com wheels that
# were never fetched. The two older per-arch reroutes each clear the flag by hand
# afterwards; matching the leaf is what stops the next one from having to.
case "$_torch_index_leaf" in
    rocm*)
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
        # A user-supplied UNSLOTH_ROCM_GFX_ARCH overrides probing (mirrors setup.sh
        # and the display block), so a Strix override still reaches the arch index.
        _gfx_all=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]')
        if [ -z "$_gfx_all" ] && command -v rocminfo >/dev/null 2>&1; then
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
        # get_torch_index_url reads the arch with ROCR/HIP masks cleared, so a
        # mask hiding every agent (e.g. ROCR_VISIBLE_DEVICES=-1) still lands
        # here on a generic rocm index; re-probe unmasked or a masked-out Strix
        # box keeps the broken generic wheels. Partial masks never get here
        # (they enumerate at least one agent above) and keep their selection.
        # ${VAR+x} (not :-): a SET-but-empty mask also hides every agent and
        # must trigger the re-probe too.
        if [ -z "$_gfx_all" ] && [ -n "${ROCR_VISIBLE_DEVICES+x}${HIP_VISIBLE_DEVICES+x}" ]; then
            if command -v rocminfo >/dev/null 2>&1; then
                _gfx_all=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; rocminfo 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
            fi
            if [ -z "$_gfx_all" ] && command -v amd-smi >/dev/null 2>&1; then
                _gfx_all=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; amd-smi list 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
                [ -z "$_gfx_all" ] && \
                    _gfx_all=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; amd-smi static --asic 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
            fi
        fi
        # HSA_OVERRIDE_GFX_VERSION=11.0.0 (the circulated Strix workaround) makes ROCr
        # hand rocminfo the SPOOFED ISA, so a gfx1151 host reports gfx1100 and the Strix
        # case below never matches (unslothai#7331). Correct the reading back to the
        # physical arch first, only in the narrow shape that cannot be a real mixed host.
        _spoof_physical=""
        if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ -n "$_gfx_all" ]; then
            _spoof_inferred=$(_infer_linux_amd_gfx_arch 2>/dev/null || true)
            _spoof_physical=$(_hsa_spoofed_physical_gfx "$_spoof_inferred" "$_gfx_all")
            [ -n "$_spoof_physical" ] && _gfx_all="$_spoof_physical"
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
        # An explicit UNSLOTH_ROCM_GFX_ARCH=gfx906 pins the runtime target to the
        # MI50 / Radeon VII path and must win over Strix probe-order detection on a
        # mixed Strix + MI50 host, so the Strix reroute is suppressed when it is set.
        # Normalize a copied HIP gcnArchName (gfx906:sramecc-:xnack- -> gfx906) and
        # trim whitespace (mirrors the Python .strip()) so the feature-flag suffix or
        # a stray newline does not defeat the exact gfx906 comparisons below.
        _gfx906_env=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
        _gfx906_env=${_gfx906_env%%:*}
        _strix_gfx=""
        if [ "$_gfx906_env" != "gfx906" ]; then
            case "$_runtime_gfx" in
                gfx1151|gfx1150|gfx1152) _strix_gfx="$_runtime_gfx" ;;
            esac
        fi
        # Skip rocm7.13+ generic indexes: they already ship the fixes, so the
        # arch build (rocm7.13) would be a downgrade rather than a rescue.
        if [ -n "$_strix_gfx" ] && _rocm_leaf_below "$_torch_index_leaf" 7 13; then
            echo "" >&2
            echo "  [WARN] $_strix_gfx (Strix) detected -- routing to the AMD arch-specific index" >&2
            echo "  [WARN] torch 2.11+rocm7.13 has AMD's real gfx1150/gfx1151 fixes (the ROCm 7.1" >&2
            echo "  [WARN] _grouped_mm segfault, moe_utils.py:167, and later Strix kernel bugs)," >&2
            echo "  [WARN] and is more reliable than the rocm7.2 index or an offline Radeon repo." >&2
            echo "" >&2
            # AMD's arch-specific index has the real _grouped_mm fix (torch 2.11.0+rocm7.13.0); UNSLOTH_AMD_ROCM_MIRROR overrides for air-gapped installs.
            _amd_strix_base="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}"
            # Strip ALL trailing slashes (match Python .rstrip("/")); multi-slash 404s on strict proxies.
            while [ "${_amd_strix_base%/}" != "$_amd_strix_base" ]; do
                _amd_strix_base="${_amd_strix_base%/}"
            done
            TORCH_INDEX_URL="${_amd_strix_base}/${_strix_gfx}/"
            TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
            # Pin companions to 2.11 (per-gfx index publishes them independently).
            TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
            TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
            _amd_gpu_radeon=false
            # Routing the wheels is only half of unslothai#7331: ROCr rebuilds the agent
            # from HSA_OVERRIDE_GFX_VERSION in every LATER process (and this shell execs
            # Unsloth further down), so leaving it set hands the freshly installed per-gfx
            # wheels a device whose reported ISA matches none of their code. Only on this
            # branch, where the spoof was corroborated and native $_strix_gfx wheels are
            # going in; paths that keep generic wheels need the override as their only
            # source of kernels. SKIP_TORCH is the other half of "the wheels are going in":
            # --no-torch reaches this branch and installs nothing, so clearing the override
            # there would strand the host with generic wheels AND no override.
            # Mirrors _clear_confirmed_hsa_spoof in studio/install_python_stack.py.
            if [ -n "$_spoof_physical" ] && [ "$SKIP_TORCH" = false ]; then
                unset HSA_OVERRIDE_GFX_VERSION
                echo "  [WARN] Clearing HSA_OVERRIDE_GFX_VERSION for the rest of this install:" >&2
                echo "  [WARN] the $_strix_gfx wheels carry $_strix_gfx kernels, so the runtime has" >&2
                echo "  [WARN] to report the real arch. Remove the export from your shell profile" >&2
                echo "  [WARN] (~/.bashrc, ~/.profile) as well, or the next terminal restores it." >&2
            fi
        fi
        # ── MI50 / Radeon VII (gfx906, Vega 20): legacy community-supported path ──
        # Newer rocm wheel families bundle ROCm libraries whose Tensile kernels
        # dropped gfx906 (rocBLAS "TensileLibrary.dat ... not read for gfx906",
        # ROCm/TheRock#1844), so a rocm6.4+/7.x index installs a torch that fails
        # at the first BLAS call. The rocm6.3 index is the last one whose wheels
        # run on gfx906 (torch 2.7.0 verified on MI50 32GB; up to 2.9 in community
        # use). Reroute any newer picked index; leave rocm6.0-6.3 alone.
        #
        # Target resolution: an explicit UNSLOTH_ROCM_GFX_ARCH wins (lets a host
        # whose rocminfo/amd-smi emit no gfx token still opt in; _gfx906_env was
        # lowercased above, before the Strix block it suppresses). Otherwise only
        # treat gfx906 as the target when it is the SOLE distinct arch present:
        # _gfx_all is de-duplicated by visible index, which loses per-device
        # ordinals on a mixed host, so a non-gfx906 selection must never be
        # downgraded to rocm6.3 -- such hosts set UNSLOTH_ROCM_GFX_ARCH to opt in.
        _gfx906_target=false
        if [ -n "$_gfx906_env" ]; then
            [ "$_gfx906_env" = "gfx906" ] && _gfx906_target=true
        elif [ -n "$_gfx_all" ]; then
            _gfx906_uniq=$(printf '%s\n' "$_gfx_all" | awk 'NF && !seen[$0]++')
            [ "$_gfx906_uniq" = "gfx906" ] && _gfx906_target=true
        fi
        # gfx906 always trains from the PyTorch rocm6.3 wheels, never the Radeon repo
        # (repo.radeon.com wheels carry no gfx906 BLAS kernels). Clear the Radeon
        # marketing-name flag as soon as gfx906 is the target -- even when the host
        # already picks rocm6.0-6.3 and the reroute below is a no-op -- so a Radeon VII
        # does not divert to the radeon branch on those versions.
        if [ "$_gfx906_target" = true ]; then
            _amd_gpu_radeon=false
        fi
        if [ "$_gfx906_target" = true ] && ! _rocm_leaf_below "$_torch_index_leaf" 6 4; then
            echo "" >&2
            echo "  [WARN] gfx906 (MI50 / Radeon VII / Vega 20) detected -- routing torch to the" >&2
            echo "  [WARN] rocm6.3 index: it is the last wheel family that runs on gfx906 (newer" >&2
            echo "  [WARN] rocm wheels ship without gfx906 BLAS kernels and fail at first use)." >&2
            echo "  [WARN] gfx906 is a community-maintained legacy path: 16-bit LoRA and full" >&2
            echo "  [WARN] finetuning work out of the box; bitsandbytes 4-bit QLoRA requires a" >&2
            echo "  [WARN] source build of bitsandbytes for gfx906 (see docs.unsloth.ai/amd)." >&2
            echo "" >&2
            _amd_gfx906_base="${UNSLOTH_PYTORCH_MIRROR:-https://download.pytorch.org/whl}"
            while [ "${_amd_gfx906_base%/}" != "$_amd_gfx906_base" ]; do
                _amd_gfx906_base="${_amd_gfx906_base%/}"
            done
            TORCH_INDEX_URL="${_amd_gfx906_base}/rocm6.3"
            # Cap below 2.11 (narrower than the default <2.12 window): a rocm7.2 pick
            # raised the floor to 2.11 above, which the rocm6.3 index (torch <= 2.9.x)
            # cannot satisfy.
            TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
            TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.26.0"
            TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<2.11.0"
            # (_amd_gpu_radeon already cleared above for every gfx906 target.)
        fi
        ;;
esac
fi  # _torch_index_pinned guard (Radeon + Strix reroute)
# Keep the previous venv's torch RELEASE; evaluated after every index/constraint decision (incl. the Strix reroute) so a raised floor rejects an older release.
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

# Pair each rocminfo GPU gfx id with its marketing name instead of using the CPU-first
# global name (#7307). Blank names keep device ordinals; no GPU keeps the old fallback.
# Keep in sync with studio/setup.sh.
_rocminfo_gpu_records() {
    awk '
        # Split at the first colon so embedded colons survive.
        function value(line,   v) {
            v = line
            sub(/^[^:]*:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            return v
        }
        /^[[:space:]]*Name:/ {
            # Keep a slot for a nameless GPU.
            if (gfx != "" && !named) { print gfx "|"; gpus++ }
            gfx = ""; named = 0
            name = value($0)
            # Accept target suffixes such as gfx90a:sramecc+, but reject ISA names.
            if (match(name, /^gfx[1-9][0-9a-z][0-9a-z][0-9a-z]?/)) {
                rest = substr(name, RLENGTH + 1)
                if (rest == "" || rest ~ /^[^0-9a-z]/) gfx = substr(name, 1, RLENGTH)
            }
            next
        }
        /^[[:space:]]*Marketing Name:/ {
            mkt = value($0)
            if (gfx != "" && !named) { print gfx "|" mkt; gpus++; named = 1 }
            else if (first == "") first = mkt
            next
        }
        END {
            if (gfx != "" && !named) { print gfx "|"; gpus++ }
            if (gpus == 0 && first != "") print "|" first
        }
    '
}

# amd-smi enumerates in discovery order over its KFD view; HIP_VISIBLE_DEVICES and
# ROCR_VISIBLE_DEVICES index HIP/ROCr order, which the library derives from the KFD node
# id instead. The two disagree on real hardware (MI350X SPX/NPS1), and _gfx here becomes
# --rocm-gfx, so an untranslated ordinal can fetch a prebuilt for another card's arch.
# `amd-smi list -e` is the map AMD publishes for this (HIP_ID, ROCm 6.4.0+); the Python
# side reads the same field in utils/hardware/amd.py get_hip_id_by_gpu_index.
# Keep in sync with studio/setup.sh.
_amd_smi_hip_order() {
    # POSIX awk forbids a physical newline in a -v value (gawk --posix makes it fatal),
    # so the records arrive on stdin ahead of the map, separated by a sentinel. The first
    # output line reports which index space the records came back in; the caller needs to
    # know, because a mask cannot be applied to an untranslated list of unlike adapters.
    { printf '%s\n' "$1"; echo "@@hip-map@@"; cat; } | awk '
        function value(line,   v) {
            v = line
            sub(/^[^:]*:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            return v
        }
        function keep(   i) { print "discovery"; for (i = 1; i <= r; i++) print rec[i] }
        !split_seen && $0 == "@@hip-map@@" { split_seen = 1; next }
        !split_seen { if ($0 != "") rec[++r] = $0; next }
        /^[[:space:]]*GPU:[[:space:]]*[0-9]/ { n++; hip[n] = -1; next }
        n && tolower($0) ~ /hip.?id/ {
            if (hip[n] < 0) { v = value($0); if (v ~ /^[0-9]+$/) hip[n] = v + 0 }
            next
        }
        END {
            # All or nothing, like get_hip_id_by_gpu_index: an older CLI rejects -e, and
            # hip_id reads N/A when the library cannot reach a KFD node. A partial or
            # colliding map is not a 1:1 device mapping, so keep discovery order.
            if (r == 0 || n != r) { keep(); exit }
            for (i = 1; i <= n; i++) {
                if (hip[i] < 0 || hip[i] >= r || (hip[i] in used)) { keep(); exit }
                used[hip[i]] = 1
                out[hip[i]] = rec[i]
            }
            print "hip"
            for (i = 0; i < r; i++) print out[i]
        }
    '
}

# One `gfx|marketing name` per adapter, in `GPU: N` order, so the mask picks both halves
# of one device. Was: arch indexed, name always adapter 0's -- and on amd-smi 6.1.1, which
# has no TARGET_GRAPHICS_VERSION, that name is what --rocm-gfx is inferred from.
# Keep in sync with studio/setup.sh.
_amd_smi_gpu_records() {
    awk '
        function value(line,   v) {
            v = line
            sub(/^[^:]*:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            return v
        }
        function flush() {
            if (started) print gfx "|" mkt
            gfx = ""; mkt = ""
        }
        # amd-smi upper-cases every key (amdsmi_logger.py _capitalize_keys): MARKET_NAME,
        # TARGET_GRAPHICS_VERSION. Matched case-folded so older spellings work too.
        /^[[:space:]]*GPU:[[:space:]]*[0-9]/ { flush(); started = 1; next }
        !started { next }
        tolower($0) ~ /market.?name/ { if (mkt == "") mkt = value($0); next }
        tolower($0) ~ /target.?graphics.?version/ {
            v = value($0)
            if (gfx == "" && v ~ /^gfx[1-9][0-9a-z][0-9a-z][0-9a-z]?$/) gfx = v
            next
        }
        END { flush() }
    '
}

# ── GPU detection summary (mirrors install.ps1 step "gpu" block) ──
if _has_usable_nvidia_gpu; then
    step "gpu" "NVIDIA GPU detected"
elif case "$TORCH_INDEX_URL" in */rocm*|*/gfx*) true ;; *) false ;; esac; then
    # Probe gfx arch for the display label, honouring HIP_VISIBLE_DEVICES
    _ensure_rocm_probe_env
    _gpu_disp_gfx_all=""
    _gpu_disp_gfx=""
    _gpu_disp_hip_map_missing=0
    _gpu_disp_mkt=""
    _gpu_disp_records=""
    if command -v rocminfo >/dev/null 2>&1; then
        _gpu_disp_records=$(rocminfo 2>/dev/null | _rocminfo_gpu_records || true)
        _gpu_disp_gfx_all=$(printf '%s\n' "$_gpu_disp_records" | awk -F'|' '$1 != "" { print $1 }')
    fi
    if [ -z "$_gpu_disp_gfx_all" ] && command -v amd-smi >/dev/null 2>&1; then
        _gpu_disp_smi_records=$(amd-smi static --asic 2>/dev/null | _amd_smi_gpu_records || true)
        if [ -n "$_gpu_disp_smi_records" ]; then
            _gpu_disp_smi_out=$(amd-smi list -e 2>/dev/null \
                | _amd_smi_hip_order "$_gpu_disp_smi_records" || true)
            _gpu_disp_smi_space=$(printf '%s\n' "$_gpu_disp_smi_out" | head -n 1)
            _gpu_disp_smi_records=$(printf '%s\n' "$_gpu_disp_smi_out" | tail -n +2)
            # No map, and the adapters are not interchangeable: the mask indexes HIP order
            # while these records are in discovery order, so any ordinal is a guess. Report
            # nothing rather than name one card while the mask selects another. amd-smi
            # 6.1.1 reports no TARGET_GRAPHICS_VERSION at all and the arch is then inferred
            # from the name, so an archless record is compared on its name instead.
            # Interchangeable adapters are unaffected: every ordinal gives the same answer.
            if [ "$_gpu_disp_smi_space" != hip ] && \
               [ "$(printf '%s\n' "$_gpu_disp_smi_records" | awk -F'|' \
                    'NF { k = ($1 != "" ? $1 : "name:" $2); if (!(k in seen)) { seen[k]; n++ } }
                     END { print n + 0 }')" -gt 1 ]; then
                _gpu_disp_smi_records=""
                _gpu_disp_gfx_all=""
                _gpu_disp_hip_map_missing=1
            fi
        fi
        _gpu_disp_gfx_all=$(amd-smi list 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        [ -z "$_gpu_disp_gfx_all" ] && \
            _gpu_disp_gfx_all=$(printf '%s\n' "$_gpu_disp_smi_records" | awk -F'|' '$1 != "" { print $1 }')
        # A silent amd-smi does not own the device list: keep rocminfo's APU fallback.
        [ -n "$_gpu_disp_smi_records" ] && _gpu_disp_records="$_gpu_disp_smi_records"
    fi
    _gpu_vis="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}"
    _gpu_vis_idx=0
    if [ -n "$_gpu_vis" ] && [ "$_gpu_vis" != "-1" ]; then
        _gpu_first="${_gpu_vis%%,*}"
        case "$_gpu_first" in ''|*[!0-9]*) ;; *) _gpu_vis_idx=$_gpu_first ;; esac
    fi
    if [ -n "$_gpu_disp_records" ]; then
        # Records already preserve device ordinals, including duplicate arches.
        _gpu_disp_record=$(printf '%s\n' "$_gpu_disp_records" | awk -v idx="$_gpu_vis_idx" \
            'NF { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx] }')
        _gpu_disp_gfx=${_gpu_disp_record%%|*}
        _gpu_disp_mkt=${_gpu_disp_record#*|}
    fi
    # Only pre-TARGET_GRAPHICS_VERSION amd-smi lands here: names but no arch in the record.
    if [ -z "$_gpu_disp_gfx" ]; then
        _gpu_disp_gfx=$(printf '%s\n' "$_gpu_disp_gfx_all" | awk -v idx="$_gpu_vis_idx" \
            'NF && !seen[$0]++ { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx] }')
    fi
    # UNSLOTH_ROCM_GFX_ARCH env override (mirrors install.ps1)
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        _gpu_disp_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
        substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $_gpu_disp_gfx"
    # Name-based arch inference when tools don't report gfx (mirrors install.ps1 nameArchTable)
    elif [ -z "$_gpu_disp_gfx" ] && [ -n "$_gpu_disp_mkt" ]; then
        # Kept in sync with install.ps1 nameArchTable; gfx1102 matched before gfx1100 ("RX 7700S").
        case "$_gpu_disp_mkt" in
            *9070*|*9080*|*"R9700"*)                                                                       _gpu_disp_gfx="gfx1201" ;;  # RDNA 4 (Navi 48: RX 9070 / 9080, Radeon AI PRO R9700)
            *9060*)                                                                                        _gpu_disp_gfx="gfx1200" ;;  # RDNA 4 (Navi 44)
            *"8065S"*|*"8060S"*|*"8050S"*|*"8040S"*|*"Strix Halo"*|*"Ryzen AI Max"*|*"AI Max"*) _gpu_disp_gfx="gfx1151" ;;  # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
            *"890M"*|*"880M"*|*"Strix Point"*|*"HX 37"*|*"AI 9 HX"*|*"AI 9 36"*) _gpu_disp_gfx="gfx1150" ;;  # RDNA 3.5 (Strix Point: Radeon 890M/880M, Ryzen AI 9 HX 370/375)
            *"860M"*|*"840M"*|*"Krackan"*|*"AI 7 35"*|*"AI 5 34"*|*"AI 7 PRO 35"*|*"AI 5 33"*) _gpu_disp_gfx="gfx1152" ;;  # RDNA 3.5 (Krackan Point: Radeon 860M/840M, Ryzen AI 7 350 / AI 5 340)
            *"RX 7600"*|*"RX 7700S"*|*"RX 7650"*|*"PRO W7600"*|*"PRO W7500"*)                              _gpu_disp_gfx="gfx1102" ;;  # RDNA 3 (Navi 33)
            *"RX 7800"*|*"RX 7700"*|*"PRO W7700"*|*"PRO V710"*)                                            _gpu_disp_gfx="gfx1101" ;;  # RDNA 3 (Navi 32)
            *"RX 7900"*|*"PRO W7900"*|*"PRO W7800"*)                                                       _gpu_disp_gfx="gfx1100" ;;  # RDNA 3 desktop / workstation (Navi 31)
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
    if [ "$_torch_index_pinned" = true ]; then
        # An explicit UNSLOTH_TORCH_INDEX_URL/_FAMILY pin skipped all probing;
        # do not claim ROCm is unusable when a CPU/other index was requested.
        step "gpu" "AMD GPU (torch index pinned: $_torch_index_leaf)" "$C_WARN"
    else
        # AMD GPU visible to the kernel but the torch index stayed CPU: no usable
        # ROCm userspace to pick a wheel. "none" would repeat the false diagnosis
        # this installer used to give.
        step "gpu" "AMD GPU (no usable ROCm -- CPU fallback)" "$C_WARN"
    fi
else
    step "gpu" "none (CPU-only)" "$C_WARN"
fi

# ── PyTorch wheel index note ──
case "$TORCH_INDEX_URL" in
    */cpu)
        if [ "$SKIP_TORCH" = false ] && [ "$OS" != "macos" ]; then
            if [ "$_torch_index_pinned" = true ]; then
                # An explicit CPU pin is a request, not a detection failure:
                # skip the SDK guidance (ROCm may be perfectly healthy here).
                substep "CPU-only PyTorch (index pinned via UNSLOTH_TORCH_INDEX_URL / _FAMILY)."
            elif _has_amd_rocm_gpu; then
                # A generation ROCm never covered is not a missing SDK (unslothai#8529).
                # Unlike the arm in get_torch_index_url, this one runs even when the arch
                # read fine, so it needs its own peer guard: an RX 5700 beside an RX 7900
                # lands here whenever the 7900's ROCm is too old, and blaming the 5700
                # would replace the upgrade advice with advice that is false for the card
                # that caused the fallback. _infer_linux_amd_gfx_arch scans every display
                # adapter, so a covered answer clears this one.
                _covered_disp_gfx=$(_infer_linux_amd_gfx_arch 2>/dev/null) || _covered_disp_gfx=""
                if [ -n "$_covered_disp_gfx" ] && _amd_arch_index_family_for_gfx "$_covered_disp_gfx" >/dev/null 2>&1; then
                    _unsup_disp_gfx=""
                else
                    _unsup_disp_gfx=$(_infer_linux_unsupported_amd_gfx_arch 2>/dev/null) || _unsup_disp_gfx=""
                fi
                if [ -n "$_unsup_disp_gfx" ]; then
                    # Scoped to the card, as above: the SDK may still help another one.
                    substep "AMD GPU detected ($_unsup_disp_gfx) -- Unsloth has no ROCm PyTorch wheels for that arch, installing CPU PyTorch." "$C_WARN"
                    substep "Installing the ROCm/HIP SDK will not give this GPU ROCm PyTorch." "$C_WARN"
                    substep "GGUF chat can still use this GPU through Vulkan: export UNSLOTH_LLAMA_CPP_BACKEND=vulkan and re-run this installer." "$C_WARN"
                    substep "That variable selects the llama.cpp bundle at install time, so setting it afterwards has no effect until you install or update again." "$C_WARN"
                else
                    substep "AMD GPU detected, but no usable ROCm/HIP install -- installing CPU-only PyTorch." "$C_WARN"
                    substep "Install the ROCm/HIP SDK and re-run this installer for GPU PyTorch." "$C_WARN"
                fi
            else
                substep "No GPU detected -- installing CPU-only PyTorch." "$C_WARN"
            fi
            if [ "$OS" = "wsl" ] && [ "$_torch_index_pinned" = false ]; then
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

# A piped/standalone install.sh has no sibling requirements tree. Bootstrap only
# the Unsloth wheel so its canonical Darwin override becomes available before
# the first with-dependencies resolution; this avoids backtracking mlx-vlm and
# then repairing it in a later phase. No-torch has no such resolve; repository,
# local, and caller-configured installs already have an override and skip this.
_bootstrap_packaged_mlx_override() {
    [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ] || return 0
    [ "$SKIP_TORCH" = false ] || return 0
    [ -f "${_OVERRIDES_FILE:-}" ] && return 0
    [ -n "${UV_OVERRIDE:-}" ] && return 0

    substep "preparing Apple Silicon model support..."
    run_install_cmd_retry "prepare Apple Silicon dependencies" \
        uv pip install --python "$_VENV_PY" --no-deps \
        --upgrade-package "$PACKAGE_NAME" -- "$PACKAGE_NAME"

    _PACKAGED_MLX_OVERRIDES=$("$_VENV_PY" -I -c "
import importlib.resources
path = importlib.resources.files('studio') / 'backend' / 'requirements' / 'single-env' / 'overrides-darwin-arm64.txt'
print(path if path.is_file() else '')
" 2>/dev/null || true)
    if [ ! -f "$_PACKAGED_MLX_OVERRIDES" ]; then
        substep "[WARN] Latest Apple Silicon model support could not be enabled. Installation will continue, but some newer models may be unavailable." "$C_WARN"
        return 0
    fi

    _MLX_OVERRIDE_TMP_ROOT=${TMPDIR:-/tmp}
    case "$_MLX_OVERRIDE_TMP_ROOT" in *[[:space:]]*) _MLX_OVERRIDE_TMP_ROOT=/tmp ;; esac
    _UV_OVERRIDE_TMPDIR=$(mktemp -d "$_MLX_OVERRIDE_TMP_ROOT/unsloth_uv.XXXXXX" 2>/dev/null) \
        || _UV_OVERRIDE_TMPDIR=""
    if [ -z "$_UV_OVERRIDE_TMPDIR" ] \
       || ! cp "$_PACKAGED_MLX_OVERRIDES" "$_UV_OVERRIDE_TMPDIR/overrides-darwin-arm64.txt"; then
        [ -n "$_UV_OVERRIDE_TMPDIR" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
        _UV_OVERRIDE_TMPDIR=""
        substep "[WARN] Latest Apple Silicon model support could not be enabled. Installation will continue, but some newer models may be unavailable." "$C_WARN"
        return 0
    fi
    _OVERRIDES_FILE="$_UV_OVERRIDE_TMPDIR/overrides-darwin-arm64.txt"
    export UV_OVERRIDE="$_OVERRIDES_FILE"
}

_bootstrap_packaged_mlx_override

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
            # --overrides replaces any UV_OVERRIDE env file, so fold its pins in; awk drops inherited torch-trio lines and newline-terminates.
            for _ov_file in ${UV_OVERRIDE:-}; do
                [ -f "$_ov_file" ] && awk '!/^[[:space:]]*torch(vision|audio)?([[:space:]<>=!~;@[]|$)/' "$_ov_file" >> "$_UNSLOTH_TORCH_OVERRIDES"
            done
            ;;
    esac
}

_unsloth_desktop_install_spec=""
if [ -n "${UNSLOTH_DESKTOP_BACKEND_VERSION:-}" ]; then
    _unsloth_desktop_install_spec="unsloth>=${UNSLOTH_DESKTOP_BACKEND_VERSION}"
fi
_unsloth_release_install_spec="${_unsloth_desktop_install_spec:-unsloth>=2026.9.2}"

if [ "$_MIGRATED" = true ]; then
    # Migrated env: force-reinstall unsloth+unsloth-zoo, preserving existing torch/CUDA unless the ROCm repair below fires.
    _gfx906_bnb_snapshot
    substep "upgrading unsloth in migrated environment..."
    if [ "$SKIP_TORCH" = true ]; then
        # No-torch: --no-deps installs (PyPI metadata still hard-deps torch), then torch-free runtime deps --no-deps.
        run_install_cmd_retry "install unsloth (migrated no-torch)" uv pip install --python "$_VENV_PY" --no-deps \
            --reinstall-package unsloth --reinstall-package unsloth-zoo \
            "$_unsloth_release_install_spec" "unsloth-zoo>=2026.9.1"
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
        _build_unsloth_torch_overrides
        run_install_cmd_retry "install unsloth (migrated)" uv pip install --python "$_VENV_PY" \
            ${_UNSLOTH_TORCH_OVERRIDES:+--overrides "$_UNSLOTH_TORCH_OVERRIDES"} \
            --reinstall-package unsloth --reinstall-package unsloth-zoo \
            "$_unsloth_release_install_spec" "unsloth-zoo>=2026.9.1"
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
    # AMD ROCm: install bitsandbytes in migrated envs too, without a fresh reinstall.
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = true ]; then
        if _is_gfx906_bnb_skip; then
            substep "gfx906: skipping prebuilt bitsandbytes (no gfx906 kernels); build from source for 4-bit QLoRA -- https://docs.unsloth.ai/get-started/install-and-update/amd" "$C_WARN"
        else
            _install_bnb_rocm "install bitsandbytes (AMD)" "$_VENV_PY"
        fi
        # Repair ROCm torch if overwritten during migrated install
        _has_hip=$("$_VENV_PY" -c "import torch; print(getattr(torch.version,'hip','') or '')" 2>/dev/null || true)
        if [ -z "$_has_hip" ]; then
            substep "repairing ROCm torch (overwritten by dependency resolution)..."
            _install_torch_default_index --force-reinstall
        fi
        _gfx906_bnb_prune
    fi
elif [ -n "$TORCH_INDEX_URL" ]; then
    # Fresh: Step 1 - install torch from explicit index (skip when --no-torch or Intel Mac)
    if [ "$SKIP_TORCH" = true ]; then
        substep "skipping PyTorch (--no-torch or Intel Mac x86_64)." "$C_WARN"
    elif [ "$_amd_gpu_radeon" = true ]; then
        _radeon_url=$(get_radeon_wheel_url "$_torch_index_leaf")
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
                # Independent highest-version picks can assemble a mismatched trio (repo publishes multiple generations); downpair to the highest common minor.
                _torch_whl=$(_pick_radeon_wheel "torch"       2>/dev/null) || _torch_whl=""
                _tv_whl=$(_pick_radeon_wheel    "torchvision" 2>/dev/null) || _tv_whl=""
                _ta_whl=$(_pick_radeon_wheel    "torchaudio"  2>/dev/null) || _ta_whl=""
                _tri_whl=$(_pick_radeon_wheel   "triton"      2>/dev/null) || _tri_whl=""

                # Verify the X.Y pairing (torchvision = torch.minor + 15); URL-decode %2B -> + first (Radeon hrefs are percent-encoded).
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
                # Kept release (_PREV_TORCH_PIN) wins here too: exact patch (else newest of its minor) + paired vision/audio; gaps fall back to the newest-trio search.
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
                                # Say so when the listing pruned the exact patch.
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

                    # Loop downwards to find the first complete matching trio (repo gaps).
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

                            # Strict X.Y validation (patch may differ) on the major.minor pairing.
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
                    # Explicit wheel URLs install the matched trio together; --find-links exposes the listing, PyPI supplies transitive deps.
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
    # AMD ROCm: install bitsandbytes once after torch (--no-torch ROCm hosts stay GGUF-only).
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = true ]; then
        if _is_gfx906_bnb_skip; then
            substep "gfx906: skipping prebuilt bitsandbytes (no gfx906 kernels); build from source for 4-bit QLoRA -- https://docs.unsloth.ai/get-started/install-and-update/amd" "$C_WARN"
        else
            _install_bnb_rocm "install bitsandbytes (AMD)" "$_VENV_PY"
        fi
    fi
    _gfx906_bnb_snapshot
    # Fresh: Step 2 - install unsloth, preserving the torch Step 1 installed
    tauri_log "STEP" "Installing Unsloth"
    substep "installing unsloth (this may take a few minutes)..."
    _build_unsloth_torch_overrides
    if [ "$SKIP_TORCH" = true ]; then
        # No-torch: install unsloth + unsloth-zoo --no-deps, then runtime deps --no-deps.
        run_install_cmd_retry "install unsloth (no-torch)" uv pip install --python "$_VENV_PY" --no-deps \
            --upgrade-package unsloth --upgrade-package unsloth-zoo \
            "$_unsloth_release_install_spec" "unsloth-zoo>=2026.9.1"
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
            --upgrade-package unsloth "$_unsloth_release_install_spec" "unsloth-zoo>=2026.9.1"
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git main..."
        run_install_cmd_retry "overlay unsloth-zoo (git main)" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
    else
        _unsloth_install_pkg="$PACKAGE_NAME"
        if [ "$PACKAGE_NAME" = "unsloth" ] && [ -n "$_unsloth_desktop_install_spec" ]; then
            _unsloth_install_pkg="$_unsloth_desktop_install_spec"
        fi
        run_install_cmd_retry "install unsloth" uv pip install --python "$_VENV_PY" \
            ${_UNSLOTH_TORCH_OVERRIDES:+--overrides "$_UNSLOTH_TORCH_OVERRIDES"} \
            --upgrade-package unsloth -- "$_unsloth_install_pkg"
    fi
    [ -n "$_UNSLOTH_TORCH_OVERRIDES" ] && rm -f "$_UNSLOTH_TORCH_OVERRIDES"
    _UNSLOTH_TORCH_OVERRIDES=""
    # AMD ROCm: repair torch if the unsloth install pulled CUDA torch from PyPI over the ROCm wheels.
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = true ]; then
        _has_hip=$("$_VENV_PY" -c "import torch; print(getattr(torch.version,'hip','') or '')" 2>/dev/null || true)
        if [ -z "$_has_hip" ]; then
            substep "repairing ROCm torch (overwritten by dependency resolution)..."
            _install_torch_default_index --force-reinstall
        fi
        _gfx906_bnb_prune
    fi
else
    # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
    tauri_log "STEP" "Installing Unsloth"
    substep "installing unsloth (this may take a few minutes)..."
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        run_install_cmd_retry "install unsloth (auto torch backend)" uv pip install --python "$_VENV_PY" "unsloth-zoo>=2026.9.1" "$_unsloth_release_install_spec" --torch-backend=auto
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git main..."
        run_install_cmd_retry "overlay unsloth-zoo (git main)" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo"
    else
        case "$PACKAGE_NAME" in
            unsloth)
                if [ -n "$_unsloth_desktop_install_spec" ]; then
                    _unsloth_install_pkg="$_unsloth_desktop_install_spec"
                else
                    _unsloth_install_pkg="$PACKAGE_NAME"
                fi
                ;;
            *) _unsloth_install_pkg="$PACKAGE_NAME" ;;
        esac
        run_install_cmd_retry "install unsloth (auto torch backend)" uv pip install --python "$_VENV_PY" --torch-backend=auto -- "$_unsloth_install_pkg"
    fi
fi

# Same probe as install.ps1: version() answers from whichever record the finder
# yields first, so a duplicate would be reported here as an ordinary version.
_installed_package_version_exit=0
if _installed_package_version=$("$_VENV_PY" -c '
import sys
try:
    from studio.install_manifest import installed_version_probe
except Exception:
    # --package installs something that does not ship studio/. Report what the
    # old probe would have, rather than claiming the version is unknown.
    from importlib.metadata import PackageNotFoundError, version
    try:
        print(version(sys.argv[1]))
    except PackageNotFoundError:
        sys.exit(1)
    sys.exit(0)
installed, conflict = installed_version_probe(sys.argv[1])
print(installed)
sys.exit(2 if conflict else (0 if installed else 1))
' "$PACKAGE_NAME" 2>/dev/null); then
    :
else
    _installed_package_version_exit=$?
    _installed_package_version=""
fi
if [ "$_installed_package_version_exit" -eq 2 ]; then
    substep "duplicate metadata found for $PACKAGE_NAME; the dependency pass will repair it"
elif [ -n "$_installed_package_version" ]; then
    step "$PACKAGE_NAME" "$_installed_package_version installed"
else
    substep "[WARN] installed $PACKAGE_NAME version could not be determined" "$C_WARN"
fi

# ── Enforce the installed torch flavor matches the detected GPU build ──
# PEP 440 ignores the +cpu/+cuXXX/+rocm local label, so uv keeps a stale torch==X+cpu against a GPU index; reinstall the right triplet, else warn loudly.
if [ "$SKIP_TORCH" = false ] && [ -n "${TORCH_INDEX_URL:-}" ]; then
    _expected_torch_tag=$(_expected_torch_flavor_tag "$TORCH_INDEX_URL")
    # Only act when a GPU build is expected (cuXXX / rocm); cpu and unknown skip.
    if [ -n "$_expected_torch_tag" ] && [ "$_expected_torch_tag" != "cpu" ]; then
        _installed_torch_ver=$(_installed_torch_version_for_tag "$_expected_torch_tag")
        _installed_torch_tag=""
        [ -n "$_installed_torch_ver" ] && _installed_torch_tag=$(_torch_flavor_tag "$_installed_torch_ver")
        # Repair only when flavor is wrong AND the index is --default-index reinstallable.
        if [ -n "$_installed_torch_tag" ] && [ "$_installed_torch_tag" != "$_expected_torch_tag" ] \
           && [ "$(_torch_index_repairable "$TORCH_INDEX_URL")" = "yes" ]; then
            substep "PyTorch flavor mismatch (installed $_installed_torch_tag, need $_expected_torch_tag) -- reinstalling correct build..."
            _install_torch_default_index \
                --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
            _installed_torch_ver=$(_installed_torch_version_for_tag "$_expected_torch_tag")
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

# ── Intel XPU: bitsandbytes with XPU kernels ──
# manylinux 0.50.0 is the first with libbitsandbytes_xpu2025.so / _xpu2026.so, and nothing else
# here touches bitsandbytes on this index (both ROCm passes are gated on that family), so a
# migrated environment would keep a pre-XPU build and lose 4-bit QLoRA.
# Out here rather than in the install branches above: those are mutually exclusive, so a copy in
# the fresh arm never runs for a migrated env (which is how the ROCm passes ended up duplicated).
# --no-deps: torch and numpy are already in. Best effort, like the Windows pass.
if [ "$SKIP_TORCH" = false ] && [ "$(_torch_index_url_leaf "${TORCH_INDEX_URL:-}")" = "xpu" ]; then
    substep "installing bitsandbytes with Intel XPU kernels..."
    run_install_cmd "install bitsandbytes (xpu)" uv pip install --python "$_VENV_PY" \
        --no-deps "$_BNB_XPU_SPEC" || \
        substep "[WARN] could not install an XPU-capable bitsandbytes; 4-bit QLoRA may be unavailable." "$C_WARN"
fi

# ── CI only: overlay a source checkout over the package just installed ──
# Not a consumer knob: no flag, absent from --help, ignored unless
# UNSLOTH_CI_SOURCE_OVERLAY names a directory holding a pyproject.toml.
#
# The clean-machine legs run THIS script from a branch but install unsloth from PyPI,
# the consumer path, so everything Python-side (studio/setup.sh, setup.ps1,
# install_python_stack.py and every requirements/constraints file they reach via
# Path(__file__)) would be the released wheel's and a branch could not be validated. An
# editable overlay re-points `import studio` at the working tree, so the
# importlib.resources lookup below finds this ref's setup.sh. NOT --local: that also
# installs `unsloth-zoo @ git+https://...`, which genuinely needs the git these legs
# remove; editable + --no-deps resolves and clones nothing, so it survives git, cmake
# and the C/C++ compilers all being gone.
if [ -n "${UNSLOTH_CI_SOURCE_OVERLAY:-}" ]; then
    if [ ! -f "$UNSLOTH_CI_SOURCE_OVERLAY/pyproject.toml" ]; then
        echo "[ERROR] UNSLOTH_CI_SOURCE_OVERLAY is set to '$UNSLOTH_CI_SOURCE_OVERLAY' but there is no pyproject.toml there." >&2
        exit 1
    fi
    substep "CI: overlaying source checkout (editable, no deps): $UNSLOTH_CI_SOURCE_OVERLAY"
    # Retry: the editable build fetches its backend from PyPI, same network risk.
    run_install_cmd_retry "overlay CI source checkout" uv pip install --python "$_VENV_PY" \
        --no-deps -e "$UNSLOTH_CI_SOURCE_OVERLAY"
fi

# ── Run studio setup ──
tauri_log "STEP" "Running Unsloth setup"
# --local uses the repo's setup.sh directly; otherwise find it in the installed package.
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
    tauri_log "ERROR" "bash is required to run studio setup"
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
# Prepend UNSLOTH_STUDIO_HOME for env-override installs without word-splitting whitespace paths.
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
    UNSLOTH_TAURI_MODE="$TAURI_MODE" \
    bash "$SETUP_SH" </dev/null || _SETUP_EXIT=$?
else
    # Reset STUDIO_LOCAL_INSTALL / STUDIO_LOCAL_REPO so a stale inherited value can't flip a normal install onto the local-dev path.
    _run_setup_with_studio_home env \
    SKIP_STUDIO_BASE="$_SKIP_BASE" \
    SKIP_STUDIO_FRONTEND="$_SKIP_FRONTEND" \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    STUDIO_LOCAL_INSTALL=0 \
    STUDIO_LOCAL_REPO= \
    UNSLOTH_NO_TORCH="$SKIP_TORCH" \
    UNSLOTH_LOCAL_LLAMA_CPP_DIR="$_WITH_LLAMA_CPP_DIR" \
    UNSLOTH_TAURI_MODE="$TAURI_MODE" \
    bash "$SETUP_SH" </dev/null || _SETUP_EXIT=$?
fi

if [ "$_SETUP_EXIT" -eq 0 ]; then
    # First: until this runs, anything that fails below reaches the exit trap, which would
    # restore the previous environment over the one just installed.
    _commit_studio_venv_replacement
    tauri_clear_install_error "studio setup completed"
fi

# ── Make 'unsloth' available via $_LOCAL_BIN (resolved earlier) ──
# Env-mode: $_LOCAL_BIN is $STUDIO_HOME/bin; skip the shell-rc PATH append.
mkdir -p "$_LOCAL_BIN"
# Refuse to delete a real directory at the shim path (could destroy user data).
_shim_path="$_LOCAL_BIN/unsloth"
if [ -d "$_shim_path" ] && [ ! -L "$_shim_path" ]; then
    echo "ERROR: $_shim_path is a directory; refusing to delete it." >&2
    echo "       Move or remove it manually, then re-run the installer." >&2
    exit 1
fi
# why: -sfn is atomic and -n prevents descent into a symlink-to-directory at
# the shim path (the directory guard above already rejects a real directory).
if ! ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path" 2>/dev/null; then
    # A reinstall rebuilds the environment at the same path, so an entry already resolving to
    # this executable is the shim we were about to write: not a failed install.
    if [ "$_shim_path" -ef "$VENV_DIR/bin/unsloth" ] 2>/dev/null; then
        substep "kept the existing shim at $_shim_path ($_LOCAL_BIN is not writable)"
    else
        echo "ERROR: could not create the shim at $_shim_path." >&2
        echo "       Make $_LOCAL_BIN writable, or run '$VENV_DIR/bin/unsloth' directly." >&2
        exit 1
    fi
fi

# Is $2 one of the colon-separated entries of $1? Field splitting also globs, so pathname
# expansion is off for the walk and restored afterwards: a directory holding *, ? or [ would
# otherwise match an unrelated entry and the persistence would be skipped.
_path_has_dir() {
    _phd_glob=on
    case $- in *f*) _phd_glob=off ;; esac
    set -f
    _phd_found=1
    _phd_old_ifs="$IFS"
    IFS=:
    for _phd_entry in $1; do
        if [ "$_phd_entry" = "$2" ]; then _phd_found=0; break; fi
    done
    IFS="$_phd_old_ifs"
    [ "$_phd_glob" = on ] && set +f
    return "$_phd_found"
}

# fish reads none of the POSIX rc files, so an `export` line is a no-op for a fish user: the
# next session resolves neither uv nor the shim. conf.d is fish's own drop-in directory and
# fish_add_path is idempotent by design. ~/.config, not XDG_CONFIG_HOME, because that is where
# astral's installer put its own fish file.
_persist_fish_path_dir() {
    _pfp_dir="$1"; _pfp_label="${2:-$1}"
    [ -n "${HOME:-}" ] || return 0
    _pfp_dir_conf="$HOME/.config/fish/conf.d"
    mkdir -p "$_pfp_dir_conf" 2>/dev/null || return 0
    _pfp_file="$_pfp_dir_conf/unsloth.fish"
    # Single-quoted: an unquoted path with a space is two arguments to fish_add_path and
    # neither exists. Inside fish single quotes only \\ and \' carry meaning.
    _pfp_quoted=$(printf '%s' "$_pfp_dir" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g")
    # The exact line we would write, not any occurrence of the directory: /opt/uv-old must not
    # pass for /opt/uv, and fish reads none of the POSIX files that would otherwise cover it.
    if ! grep -v '^[[:space:]]*#' "$_pfp_file" 2>/dev/null | grep -qxF "fish_add_path '$_pfp_quoted'"; then
        if {
            echo "# Added by Unsloth installer"
            echo "fish_add_path '$_pfp_quoted'"
        } 2>/dev/null >> "$_pfp_file"; then
            step "path" "added $_pfp_label to PATH in $_pfp_file"
        else
            step "path" "could not write $_pfp_file; add $_pfp_label to PATH yourself" "$C_WARN"
        fi
    fi
}

# A line that SETS PATH, as opposed to one that merely names the directory. The name boundary
# keeps PYTHONPATH and friends out; the three helpers are the common non-assignment spellings.
_PATH_LINE_RE='(^|[^[:alnum:]_])(PATH[[:space:]]*=|fish_add_path|pathmunge|path_helper)'

# Put a directory on the PATH of the NEXT shell, not just this process.
#   $1 the directory  $2 the rc-file literal (~/.local/bin keeps $HOME unexpanded, as it always
#   has)  $3 how to name it in the line we print  $4 the grep that says it is already there
#   $5 an explicit profile file, or empty to pick one the way this installer always has
_persist_login_path_dir() {
    _plp_dir="$1"; _plp_literal="$2"; _plp_label="$3"; _plp_pattern="$4"; _plp_file="${5:-}"
    [ -n "${HOME:-}" ] || return 0
    # fish reads none of the POSIX rc files, so an `export` line there is a no-op for a fish
    # user: the next session resolves neither uv nor the shim. conf.d is fish's own drop-in
    # directory and fish_add_path is idempotent by design.
    if [ -z "$_plp_file" ] && [ "$(basename "${SHELL:-}")" = "fish" ]; then
        _persist_fish_path_dir "$_plp_dir" "$_plp_label"
        return 0
    fi
    _SHELL_PROFILE="$_plp_file"
    if [ -n "$_SHELL_PROFILE" ]; then
        :
    elif [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
        _SHELL_PROFILE="${ZDOTDIR:-$HOME}/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        _SHELL_PROFILE="$HOME/.bashrc"
    elif [ -f "$HOME/.profile" ]; then
        _SHELL_PROFILE="$HOME/.profile"
    elif [ -w "$HOME" ]; then
        # A fresh account can have no rc file at all: astral's installer used to create one,
        # the pinned path does not. The append creates it, and every POSIX login shell reads
        # ~/.profile.
        _SHELL_PROFILE="$HOME/.profile"
    fi
    [ -n "$_SHELL_PROFILE" ] || return 0
    # Comments stripped first, then only lines that actually set PATH: a commented-out old export
    # is not an active entry, and neither is `UV_CACHE=/opt/uv` or `PYTHONPATH=/opt/uv`. The name
    # boundary is what keeps PYTHONPATH out. Taking any of them for a PATH entry leaves the next
    # shell with no uv at all.
    if ! grep -v '^[[:space:]]*#' "$_SHELL_PROFILE" 2>/dev/null \
        | grep -E "$_PATH_LINE_RE" | grep -qE "$_plp_pattern"; then
        # One redirect, so an unwritable profile leaves no half-written entry; a warning rather
        # than a failure, because under set -e an unguarded append would end the install.
        if {
            echo ''
            echo '# Added by Unsloth installer'
            echo "export PATH=\"$_plp_literal:\$PATH\""
        } 2>/dev/null >> "$_SHELL_PROFILE"; then
            step "path" "added $_plp_label to PATH in $_SHELL_PROFILE"
        else
            step "path" "could not write $_SHELL_PROFILE; add $_plp_label to PATH yourself" "$C_WARN"
        fi
    fi
}

if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_LOCAL_BIN"; then  # not on a new shell's PATH
        if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
            export PATH="$_LOCAL_BIN:$PATH"
            step "path" "exported $_LOCAL_BIN for this session (no rc-file append in env-override mode)"
        else
            _persist_login_path_dir "$_LOCAL_BIN" '$HOME/.local/bin' "~/.local/bin" '\.local/bin'
            export PATH="$_LOCAL_BIN:$PATH"
        fi
fi

# UV_INSTALL_DIR, UV_UNMANAGED_INSTALL, XDG_BIN_HOME and XDG_DATA_HOME all outrank ~/.local/bin,
# and astral's installer wrote a PATH line for whichever it picked, so replacing that installer
# means persisting its destination too. Both of astral's opt-outs are honoured. Not gated on the
# destination differing from ~/.local/bin: that IS the default, so gating there left every
# ordinary machine with the single-file write.
if [ -n "${_UNSLOTH_UV_BIN_DIR:-}" ] \
   && [ -z "${UV_NO_MODIFY_PATH:-}" ] && [ -z "${UV_UNMANAGED_INSTALL:-}" ] \
   && [ "$_STUDIO_HOME_REDIRECT" != "env" ]; then
    if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_UNSLOTH_UV_BIN_DIR"; then
        # The rc line is double-quoted, so a path holding $, ` or " would be expanded or
        # terminated by the shell that reads it. The ~/.local/bin literal is exempt: its
        # $HOME is meant to stay unexpanded.
        _uv_rc_literal=$(printf '%s' "$_UNSLOTH_UV_BIN_DIR" | sed 's/[\\"$`]/\\&/g')
        # Anchored on both sides, so /opt/uv is not satisfied by /opt/uv-old and the match has
        # to be a whole PATH entry rather than any occurrence of the text.
        _uv_grep_esc=$(printf '%s' "$_UNSLOTH_UV_BIN_DIR" | sed 's/[].[\\()*+?{}|^$\/]/\\&/g')
        # ...and the $HOME-relative spelling as well, because the shim block above writes
        # `export PATH="$HOME/.local/bin:$PATH"` unexpanded. Without this the default install
        # would add a second line for the same directory in the same file.
        case "$_UNSLOTH_UV_BIN_DIR" in
            "$HOME"/*)
                _uv_grep_esc="$_uv_grep_esc|\\\$HOME$(printf '%s' "${_UNSLOTH_UV_BIN_DIR#$HOME}" | sed 's/[].[\\()*+?{}|^$\/]/\\&/g')"
                ;;
        esac
        _uv_pattern="(^|[^[:alnum:]_.~/-])($_uv_grep_esc)([^[:alnum:]_.~/-]|\$)"
        # Every startup file astral's installer wired, because it is the installer we replaced.
        # Writing only the file for the shell that happens to be running leaves a bash user whose
        # .bash_profile does not source .bashrc, or anyone who later switches shells, without uv.
        for _uv_prof in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" \
                        "$HOME/.bash_login" "${ZDOTDIR:-$HOME}/.zshrc" "${ZDOTDIR:-$HOME}/.zshenv"; do
            # ~/.profile is created when absent, as astral does; the rest are only touched when
            # the user already has them.
            if [ "$_uv_prof" = "$HOME/.profile" ] || [ -f "$_uv_prof" ]; then
                _persist_login_path_dir "$_UNSLOTH_UV_BIN_DIR" "$_uv_rc_literal" \
                    "$_UNSLOTH_UV_BIN_DIR" "$_uv_pattern" "$_uv_prof"
            fi
        done
        _persist_fish_path_dir "$_UNSLOTH_UV_BIN_DIR"
    fi
fi
# end of the PATH persistence block

# Non-Tauri installs keep shortcuts even if setup reports failure.
if [ "$TAURI_MODE" != true ]; then
    create_studio_shortcuts "$VENV_ABS_BIN/unsloth" "$OS"
fi

# If setup.sh failed, report and exit now.
# PATH + shortcuts are already set up so the user can fix and retry.
if [ "$_SETUP_EXIT" -ne 0 ]; then
    echo ""
    if [ "$TAURI_MODE" = true ]; then
        tauri_log "ERROR_DEFAULT" "studio setup failed (exit code $_SETUP_EXIT)"
    else
        step "error" "studio setup failed (exit code $_SETUP_EXIT)" "$C_ERR"
    fi
    echo ""
    exit "$_SETUP_EXIT"
fi

# ── Tauri mode: done, skip shortcuts and auto-launch ──
if [ "$TAURI_MODE" = true ]; then
    tauri_log "DONE" ""
    exit 0
fi

# Warn if another 'unsloth' wins on PATH; canonicalize via the venv python (BSD readlink lacks -f).
_installed_bin="$VENV_DIR/bin/unsloth"
_path_unsloth=$(command -v unsloth 2>/dev/null || true)
if [ -n "$_path_unsloth" ] && [ -x "$VENV_DIR/bin/python" ]; then
    # If either side fails to resolve, skip the check rather than compare raw paths.
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

# In interactive terminals, ask before starting Unsloth unless the caller disabled the prompt; non-interactive environments (Docker, CI, cloud-init) just print instructions.
if [ "$_SKIP_AUTOSTART" != true ] && [ -t 1 ]; then
    echo ""
    # No readable answer (closed/EOF tty) defaults to no; Enter is still yes.
    # Prompt only when something can answer: `test -r` passes on the unopenable
    # /dev/tty found in containers, leaving a dangling question in the log.
    if _can_read_tty; then
        printf "  Start Unsloth Studio now? [Y/n] "
        read -r _reply </dev/tty || _reply="n"
    else
        _reply="n"
    fi
    case "${_reply:-y}" in
        [Yy]*|"")
            step "launch" "starting Unsloth Studio..."
            # Detach stdin from the piped web install's pipe: as a foreground server the
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

}

# Every byte above is parsed before this line runs, which is the point.
_unsloth_main "$@"
