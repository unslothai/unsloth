#!/bin/sh
# Unsloth Studio Installer. Usage, supported options and the web one-liner live in the README under "Unsloth Studio (web UI)" and are deliberately not repeated here: this file ships inside the Linux desktop bundle, where a header rehearsing download-and-run command lines is the first thing a generic script classifier reads. A piped install takes options as environment variables after the pipe (UNSLOTH_NO_TORCH, UNSLOTH_SKIP_AUTOSTART, UNSLOTH_INSTALL_SYSTEMD, UNSLOTH_SYSTEMD_HOST, UNSLOTH_SYSTEMD_PORT, UNSLOTH_ISOLATE_UV_CACHE, UNSLOTH_INSTALL_NO_ROLLBACK, UNSLOTH_PYTHON, UNSLOTH_STUDIO_HOME), because a bare `--no-torch` after the pipe would be read as an option to sh itself; a local run takes the equivalent flags (--no-torch, --isolated-uv-cache, --no-rollback, --python, --local). Install dir priority: UNSLOTH_STUDIO_HOME > STUDIO_HOME > $HOME/.unsloth/studio
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e
# Body is wrapped in a function so a piped sh reads the whole file before running; an early
# exit otherwise breaks curl's pipe. Do not add `exec < /dev/null` (closes the piped source).
_unsloth_main() {

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
_ISOLATE_UV_CACHE=false
_NO_ROLLBACK=false
_VENV_DISCARDED=false
_VENV_DISCARD_LEFTOVER=""
_INSTALL_SYSTEMD=false
_SYSTEMD_STARTED=false
_VERBOSE=false
_SHORTCUTS_ONLY=false
_next_is_package=false
_next_is_python=false
_next_is_llama_cpp_dir=false
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
        --isolated-uv-cache) _ISOLATE_UV_CACHE=true ;;
        --no-rollback) _NO_ROLLBACK=true ;;
        --verbose|-v) _VERBOSE=true ;;
        --shortcuts-only) _SHORTCUTS_ONLY=true ;;
        --with-llama-cpp-dir) _next_is_llama_cpp_dir=true ;;
    esac
done

case "${UNSLOTH_NO_TORCH:-}" in 1|true|TRUE|yes|YES|on|ON) _NO_TORCH_FLAG=true ;; esac
case "${UNSLOTH_SKIP_AUTOSTART:-}" in 1|true|TRUE|yes|YES|on|ON) _SKIP_AUTOSTART=true ;; esac
case "${UNSLOTH_ISOLATE_UV_CACHE:-}" in 1|true|TRUE|yes|YES|on|ON) _ISOLATE_UV_CACHE=true ;; esac
case "${UNSLOTH_INSTALL_NO_ROLLBACK:-}" in 1|true|TRUE|yes|YES|on|ON) _NO_ROLLBACK=true ;; esac
case "${UNSLOTH_INSTALL_SYSTEMD:-}" in 1|true|TRUE|yes|YES|on|ON) _INSTALL_SYSTEMD=true ;; esac
[ -z "$_USER_PYTHON" ] && [ -n "${UNSLOTH_PYTHON:-}" ] && _USER_PYTHON="$UNSLOTH_PYTHON"

if [ "$_VERBOSE" = true ]; then
    export UNSLOTH_VERBOSE=1
fi

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
        # Canonicalize both sides so CDPATH / a symlinked $HOME cannot break equality.
        if [ -d "$_tauri_override" ]; then
            _tauri_override_abs=$(CDPATH= cd -P -- "$_tauri_override" 2>/dev/null && pwd -P) \
                || _tauri_override_abs="$_tauri_override"
        else
            _tauri_override_abs="$_tauri_override"
        fi
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

_redact_install_output() {
    sed -E \
        -e 's#(https?://)[^/@[:space:]`]+@#\1<redacted>@#g' \
        -e 's#([?&][^=[:space:]&`]+)=[^&#[:space:]`]+#\1=<redacted>#g' \
        -e 's|(https?://[^[:space:]`#]+)#[^[:space:]`]+|\1#<redacted>|g' \
        "$@"
}

: "${UNSLOTH_DL_MARKER_MIN_BYTES:=52428800}"

# Markers go to stderr: the verbose path's sed redactor block-buffers and would delay them.
_uv_download_markers() {
    # Minimal images lack awk; without this the pipe closes and the child dies of SIGPIPE.
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
    if [ -z "${_UNSLOTH_MIRROR_SPARE:-}" ]; then
        _run_install_cmd_once "$@"
        return
    fi
    _run_install_cmd_once "$@" || _mirror_retry_install "$?" "$@"
}

_mirror_retry_install() {
    [ -n "${_UNSLOTH_MIRROR_SPARE:-}" ] || return "$1"
    _mri_rc=$1
    _mri_label=$2
    shift 2
    case " $* " in
        *" https://download.pytorch.org/whl"*) _mri_ran=torch ;;
        *" --index-url "*|*" --default-index "*|*" --find-links "*|*" --no-index "*|*"://"*|*" --torch-backend"*) _mri_ran="" ;;
        *" uv venv "*|*" uv python install "*) _mri_ran=python ;;
        *) _mri_ran=pypi ;;
    esac
    _mri_host=$(_mirror_failed_host "${_ric_log:-}" "$_mri_ran") || _mri_host=""
    rm -f "${_ric_log:-}"
    case "$_mri_host $* " in
        "pypi "*" --index-url "*|"pypi "*" --default-index "*) return "$_mri_rc" ;;
        "unsynced "*" --index-url "*|"unsynced "*" --default-index "*|"unsynced "*" --no-index "*) return "$_mri_rc" ;;
        "torch "*https://download.pytorch.org/whl*|"pypi "*|"python "*|"unsynced "*) ;;
        *) return "$_mri_rc" ;;
    esac
    _mirror_take "$_mri_host" || return "$_mri_rc"
    (
        for _mri_pair in $_MT_PAIRS; do export "$_mri_pair"; done
        [ "$_mri_host" != torch ] || _ric_torch_mirror=$UNSLOTH_PYTORCH_MIRROR
        _run_install_cmd_once "$_mri_label" "$@" || { _mri_rc=$?; rm -f "${_ric_log:-}"; exit "$_mri_rc"; }
    ) || return
    for _mri_pair in $_MT_PAIRS; do export "$_mri_pair"; done
    if [ "$_mri_host" = torch ]; then
        _ric_torch_mirror=$UNSLOTH_PYTORCH_MIRROR
        case "${TORCH_INDEX_URL:-}" in
            https://download.pytorch.org/whl*) TORCH_INDEX_URL=$_ric_torch_mirror${TORCH_INDEX_URL#https://download.pytorch.org/whl} ;;
        esac
    fi
}

_ric_run() {
    if "$@" 2>&1; then
        _cmd_rc=0
    else
        _cmd_rc=$?
    fi
    printf '%s' "$_cmd_rc" > "$_rcf"
}

_ric_tee() {
    if command -v tee >/dev/null 2>&1; then tee "$1"; else cat; fi
}

_run_install_cmd_once() {
    _label="$1"
    shift
    [ -z "${_ric_log:-}" ] || rm -f "$_ric_log"
    _ric_log=""
    if [ -n "${_ric_torch_mirror:-}" ]; then
        _ric_n=$#
        for _ric_arg in "$@"; do
            case "$_ric_arg" in
                https://download.pytorch.org/whl*) _ric_arg=$_ric_torch_mirror${_ric_arg#https://download.pytorch.org/whl} ;;
            esac
            set -- "$@" "$_ric_arg"
        done
        shift "$_ric_n"
    fi
    # Clear inherited uv index vars so a uv.toml cannot outrank the --default-index pin.
    case " $* " in
        *" --default-index "*) set -- env -u UV_DEFAULT_INDEX -u UV_INDEX_URL -u UV_INDEX -u UV_EXTRA_INDEX_URL -u UV_TORCH_BACKEND -u UV_FIND_LINKS -u UV_CONFIG_FILE UV_NO_CONFIG=1 "$@" ;;
    esac
    if _is_verbose; then
        _rcf=$(mktemp)
        tauri_stream_log stdout "OUTPUT_CLEAR" "$_label"
        _log=""
        if [ -n "${_UNSLOTH_MIRROR_SPARE:-}" ]; then
            _log=$(mktemp)
            _ric_run "$@" | _ric_tee "$_log" | _uv_download_markers "" "$UNSLOTH_DL_MARKER_MIN_BYTES" | _redact_install_output
        else
            _ric_run "$@" | _uv_download_markers "" "$UNSLOTH_DL_MARKER_MIN_BYTES" | _redact_install_output
        fi
        _rc=$(cat "$_rcf" 2>/dev/null || echo 1)
        rm -f "$_rcf"
        _rc=${_rc:-1}
        if [ "$_rc" -eq 0 ] 2>/dev/null; then
            [ -z "$_log" ] || rm -f "$_log"
            tauri_clear_install_error "$_label recovered"
            return 0
        fi
        _ric_log=$_log
        tauri_stream_log stdout "ERROR_OUTPUT" "$_label failed (exit code $_rc)"
        step "error" "$_label failed (exit code $_rc)" "$C_ERR" >&2
        return "$_rc"
    fi
    _log=$(mktemp)
    _rcf=$(mktemp)
    tauri_stream_log stderr "OUTPUT_CLEAR" "$_label"
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
    if [ -n "${_UNSLOTH_MIRROR_SPARE:-}" ]; then _ric_log=$_log; else rm -f "$_log"; fi
    return $_rc
}

: "${UNSLOTH_INSTALL_RETRIES:=3}"
: "${UNSLOTH_INSTALL_RETRY_DELAY:=3}"
run_install_cmd_retry() {
    _ricr_label="$1"
    # Bounds 1..100 retries and 0..3600s delay; 0?* rejects octal delays.
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
        _run_install_cmd_once "$@" && return 0
        _ricr_rc=$?
        if [ "$_ricr_attempt" -ge "$_ricr_max" ]; then
            _mirror_retry_install "$_ricr_rc" "$@" && return 0
            return $?
        fi
        substep "retrying \"$_ricr_label\" after transient failure (attempt $((_ricr_attempt + 1))/$_ricr_max, waiting ${_ricr_delay}s)..." "$C_WARN"
        sleep "$_ricr_delay" || true
        _ricr_attempt=$((_ricr_attempt + 1))
        _ricr_delay=$((_ricr_delay * 2))
    done
}

# gfx906: the prebuilt AMD bnb wheel has no gfx906 kernels and would clobber a source-built bnb.
_is_gfx906_bnb_skip() {
    [ "${_gfx906_target:-false}" = true ] && return 0
    _bnb_gfx_env=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
    _bnb_gfx_env=${_bnb_gfx_env%%:*}
    [ "$_bnb_gfx_env" = "gfx906" ] && return 0
    # A pinned index skips the reroute that sets _gfx906_target, so probe here (sole-arch rule).
    if [ -z "$_bnb_gfx_env" ] && [ "${_torch_index_pinned:-false}" = true ]; then
        _bnb_gfx_probe=$(_probe_amd_gfx_arch | awk 'NF && !seen[$0]++')
        [ "$_bnb_gfx_probe" = "gfx906" ] && return 0
    fi
    return 1
}

# pip install unsloth pulls a generic bnb; drop a freshly pulled wheel but keep a source build.
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

# bnb <= 0.49.2 NaNs at 4-bit decode on AMD (bnb #1887). Keep this floor in step with the amd extra
# in pyproject.toml and studio/install_python_stack.py.
_BNB_ROCM_PYPI_FALLBACK="bitsandbytes>=0.50.0"
# Intel XPU: 0.50.0 is the first with xpu libs; studio/setup.ps1 uses the same floor.
_BNB_XPU_SPEC="bitsandbytes>=0.50.0"
# bnb aarch64 wheels carry no ROCm binary at any version, so messages must not claim 4-bit.
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
    # uv rejects this pre-release wheel (filename/metadata version mismatch); pip accepts it.
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

case "$PACKAGE_NAME" in
    [!a-zA-Z0-9]*)
        echo "❌ ERROR: --package name must start with a letter or digit." >&2
        exit 1 ;;
    *[!a-zA-Z0-9._-]*)
        echo "❌ ERROR: --package name contains invalid characters (allowed: a-z A-Z 0-9 . _ -)" >&2
        exit 1 ;;
esac

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

PYTHON_VERSION=""

_resolve_studio_destinations() {
    _override_var=""
    _override="${UNSLOTH_STUDIO_HOME:-}"
    if [ -n "$_override" ]; then
        _override_var="UNSLOTH_STUDIO_HOME"
    else
        _override="${STUDIO_HOME:-}"
        [ -n "$_override" ] && _override_var="STUDIO_HOME"
    fi
    _override=$(printf '%s' "$_override" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
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

# uv --no-cache neither reads nor writes a cache, so nothing is recorded. Values are clap's
# BoolishValueParser set. Mirrors _uv_no_cache_requested() in unsloth_cli/commands/studio.py.
_uv_no_cache_requested() {
    case "$(printf '%s' "${UV_NO_CACHE:-}" | tr '[:upper:]' '[:lower:]')" in
        1|y|yes|t|true|on) return 0 ;;
    esac
    return 1
}

_uv_is_bucket_name() {
    case "$1" in
        *-v[0-9]*) ;;
        *) return 1 ;;
    esac
    # '' too: `##*-v` on `archive-v1-v` leaves an empty suffix; Test-StudioUvBucketName rejects it.
    case "${1##*-v}" in
        ''|*[!0-9]*) return 1 ;;
    esac
    # Every CacheBucket in uv 0.12.1 (UV_PINNED_VERSION); re-read uv-cache/src/lib.rs on a pin bump.
    case "${1%-v*}" in
        archive|binaries|builds|built-wheels|environments|flat-index) ;;
        git|interpreter|osv|python|sdists|simple|wheels) ;;
        *) return 1 ;;
    esac
    return 0
}

_absolutize_uv_cache_dir() {
    _uv_cache_path="${1-$UV_CACHE_DIR}"
    case "$_uv_cache_path" in
        /*) ;;
        *)
            _uv_cache_base="${UV_WORKING_DIR:-$PWD}"
            case "$_uv_cache_base" in
                /*) ;;
                *) _uv_cache_base="$PWD/$_uv_cache_base" ;;
            esac
            _uv_cache_path="$_uv_cache_base/$_uv_cache_path"
            ;;
    esac
    # Strip trailing slashes: the studio-vs-shared comparison below is a string compare.
    while [ "$_uv_cache_path" != / ]; do
        case "$_uv_cache_path" in
            */) _uv_cache_path="${_uv_cache_path%/}" ;;
            *) break ;;
        esac
    done
    printf '%s\n' "$_uv_cache_path"
}

# Can we create AND fill this directory? A real create, since mkdir -p exits 0 on an existing
# unwritable one and -w reads the mode, not the filesystem. mktemp, since a predictable name
# in another account's directory can be pre-created as a symlink to follow.
_uv_cache_root_is_writable() {
    mkdir -p "$1" 2>/dev/null || return 1
    _uv_root_probe=$(mktemp "$1/.unsloth-write-probe.XXXXXX" 2>/dev/null) || return 1
    # Retry the unlink and trust whether the file is gone: a held handle can make rm fail once.
    _uv_root_tries=0
    while :; do
        rm -f "$_uv_root_probe" 2>/dev/null || true
        [ -e "$_uv_root_probe" ] || break
        _uv_root_tries=$((_uv_root_tries + 1))
        if [ "$_uv_root_tries" -ge 3 ]; then
            unset _uv_root_probe _uv_root_tries
            return 1
        fi
        sleep 1
    done
    unset _uv_root_probe _uv_root_tries
    return 0
}

# Also check every bucket: a root-owned bucket from an elevated run makes uv abort.
_uv_cache_is_writable() {
    _uv_cache_root_is_writable "$1" || return 1
    _uv_w_bad=0
    # Case folding is filesystem-dependent (APFS folds, ext4 does not), so measure it.
    _uv_w_fold=0
    _uv_w_probe="$1/.unsloth-case-probe.$$-A"
    if mkdir "$_uv_w_probe" 2>/dev/null; then
        [ -d "$1/.unsloth-case-probe.$$-a" ] && _uv_w_fold=1
        rmdir "$_uv_w_probe" 2>/dev/null || true
    fi
    unset _uv_w_probe
    _uv_w_glob=on
    case $- in *f*) _uv_w_glob=off ;; esac
    set +f
    for _uv_w_dir in "$1"/*; do
        _uv_w_name="${_uv_w_dir##*/}"
        if ! _uv_is_bucket_name "$_uv_w_name"; then
            [ "$_uv_w_fold" = 1 ] || continue
            case "$_uv_w_name" in *[[:upper:]]*) ;; *) continue ;; esac
            _uv_w_lower=$(printf '%s' "$_uv_w_name" | tr '[:upper:]' '[:lower:]')
            _uv_is_bucket_name "$_uv_w_lower" || continue
        fi
        if [ ! -d "$_uv_w_dir" ]; then
            if [ -e "$_uv_w_dir" ] || [ -L "$_uv_w_dir" ]; then
                _uv_w_bad=1
            fi
            continue
        fi
        _uv_cache_root_is_writable "$_uv_w_dir" || _uv_w_bad=1
    done
    if [ "$_uv_w_glob" = off ]; then set -f; fi
    unset _uv_w_dir _uv_w_glob _uv_w_name _uv_w_lower _uv_w_fold
    if [ "$_uv_w_bad" -ne 0 ]; then
        unset _uv_w_bad
        return 1
    fi
    unset _uv_w_bad
    return 0
}

# Records which cache this install used so an update reuses it. UV_CACHE_DIR is made absolute
# because setup.sh changes directory.
_record_uv_cache_choice() {
    UV_CACHE_DIR=$(_absolutize_uv_cache_dir)
    _uv_marker_dir="$STUDIO_HOME/cache"
    _uv_marker_file="$_uv_marker_dir/uv-cache-dir"
    _uv_marker_value="$UV_CACHE_DIR"
    if [ "$_UV_MARKER_SAVED" != true ]; then
        if [ -e "$_uv_marker_file" ] || [ -L "$_uv_marker_file" ]; then
            _UV_MARKER_PREVIOUS=$(cat "$_uv_marker_file" 2>/dev/null) || return 0
            _UV_MARKER_EXISTED=true
        else
            _UV_MARKER_PREVIOUS=""
            _UV_MARKER_EXISTED=false
        fi
        _UV_MARKER_SAVED=true
    fi
    (
        mkdir -p "$_uv_marker_dir" 2>/dev/null &&
            # Unlink first: a redirection follows a symlink and truncates its target.
            rm -f "$_uv_marker_file" 2>/dev/null &&
            # And only once gone: rm can fail on a link in an undeletable directory.
            ! { [ -e "$_uv_marker_file" ] || [ -L "$_uv_marker_file" ]; } &&
            printf '%s\n' "$_uv_marker_value" > "$_uv_marker_file" 2>/dev/null
    ) || true
}

_restore_uv_cache_marker() {
    [ "${_STUDIO_INSTALL_COMMITTED:-false}" = true ] && return 0
    [ "$_UV_MARKER_SAVED" = true ] || return 0
    _uv_marker_file="$STUDIO_HOME/cache/uv-cache-dir"
    rm -f "$_uv_marker_file" 2>/dev/null || true
    if [ "$_UV_MARKER_EXISTED" = true ] \
       && ! { [ -e "$_uv_marker_file" ] || [ -L "$_uv_marker_file" ]; }; then
        printf '%s\n' "$_UV_MARKER_PREVIOUS" > "$_uv_marker_file" 2>/dev/null || true
    fi
    _UV_MARKER_SAVED=false
}

_configure_uv_cache() {
    _uv_studio_cache="$STUDIO_HOME/cache/uv"
    # Only a caller's UV_CACHE_DIR outranks selection; the flag tells it apart from our default.
    if [ "${_UV_CACHE_DIR_INSTALLER_DEFAULT:-false}" != true ]; then
        case "${UV_CACHE_DIR-}" in
            *[![:space:]]*)
                _UV_CACHE_MODE=custom
                export UV_CACHE_DIR
                if _uv_no_cache_requested; then
                    step "uv cache" "preserving custom UV_CACHE_DIR ($UV_CACHE_DIR); uv caching is off (UV_NO_CACHE), so nothing is recorded"
                else
                    _record_uv_cache_choice
                    step "uv cache" "preserving custom UV_CACHE_DIR ($UV_CACHE_DIR)"
                fi
                return 0
                ;;
        esac
    fi

    if [ "$_ISOLATE_UV_CACHE" = true ]; then
        UV_CACHE_DIR="$_uv_studio_cache"
        _UV_CACHE_MODE=isolated
        export UV_CACHE_DIR
        _uv_no_cache_requested || _record_uv_cache_choice
        step "uv cache" "forced Unsloth Studio cache isolation ($UV_CACHE_DIR); already-cached packages may download again" "$C_WARN"
        return 0
    fi

    if _uv_no_cache_requested; then
        UV_CACHE_DIR="$_uv_studio_cache"
        _UV_CACHE_MODE=studio
        export UV_CACHE_DIR
        step "uv cache" "uv caching is off (UV_NO_CACHE); nothing to select or record"
        return 0
    fi

    # Ask uv so uv.toml / UV_CONFIG_FILE count; take the last line so a notice is not the path.
    _uv_default_cache=$(env -u UV_CACHE_DIR uv cache dir 2>/dev/null \
        | sed -e 's/[[:space:]]*$//' -e '/^$/d' | tail -n 1) || _uv_default_cache=""
    if [ -z "$_uv_default_cache" ]; then
        if [ -n "${XDG_CACHE_HOME:-}" ]; then
            _uv_default_cache="${XDG_CACHE_HOME}/uv"
        elif [ -n "${HOME:-}" ]; then
            _uv_default_cache="${HOME}/.cache/uv"
        fi
    fi
    if [ -n "$_uv_default_cache" ]; then
        _uv_default_cache=$(_absolutize_uv_cache_dir "$_uv_default_cache")
    fi

    # The recorded cache outranks uv's default while warm (same as studio.py _with_studio_uv_cache).
    # Strip BOM and trailing CR: install.ps1 writes it with PowerShell 5.1 utf8 under WSL.
    _uv_recorded=$(cat "$STUDIO_HOME/cache/uv-cache-dir" 2>/dev/null) || _uv_recorded=""
    _uv_cr=$(printf '\r')
    _uv_recorded="${_uv_recorded%"$_uv_cr"}"
    unset _uv_cr
    _uv_bom=$(printf '\357\273\277')
    _uv_recorded="${_uv_recorded#"$_uv_bom"}"
    unset _uv_bom
    case "$_uv_recorded" in
        *[![:space:]]*) _uv_recorded=$(_absolutize_uv_cache_dir "$_uv_recorded") ;;
        *) _uv_recorded="" ;;
    esac

    # A pre-marker $STUDIO_HOME/cache/uv goes LAST, behind uv's default: its content may be
    # leftover backend wheels while Torch and CUDA live in the default.
    _uv_unmarked_studio=""
    if [ -z "$_uv_recorded" ] || [ ! -d "$_uv_recorded" ]; then
        _uv_unmarked_studio="$_uv_studio_cache"
    fi

    # Root and buckets must be writable; nested entries are not probed (too many files).
    _uv_scan_blocked=false
    _uv_blocked_cache=""
    _uv_warn_cache=""
    _uv_chosen_cache=""
    for _uv_candidate in "$_uv_recorded" "$_uv_default_cache" "$_uv_unmarked_studio"; do
        { [ -n "$_uv_candidate" ] && [ -z "$_uv_chosen_cache" ]; } || continue
        _uv_cand_populated=false
        _uv_cand_writable=true
        if [ -d "$_uv_candidate" ] && [ -r "$_uv_candidate" ]; then
            # The globs below are the scan, so a caller's set -f would read every cache as empty.
            _uv_glob=on
            case $- in *f*) _uv_glob=off ;; esac
            set +f

            _uv_cache_is_writable "$_uv_candidate" || _uv_cand_writable=false

            # Warm means package bytes: wheels-* holds metadata only. -L to match Get-ChildItem.
            for _uv_bucket in \
                "$_uv_candidate"/archive-* \
                "$_uv_candidate"/builds-* \
                "$_uv_candidate"/built-wheels-* \
                "$_uv_candidate"/wheels-* \
                "$_uv_candidate"/sdists-*; do
                [ -d "$_uv_bucket" ] || continue
                # Stricter than the probe: only kinds uv fills count (`archive-v0.backup` does not).
                _uv_bucket_base="${_uv_bucket##*/}"
                _uv_is_bucket_name "$_uv_bucket_base" || continue
                case "${_uv_bucket_base%-v*}" in
                    archive|builds|built-wheels|wheels|sdists) ;;
                    *) continue ;;
                esac
                if [ ! -r "$_uv_bucket" ] || [ ! -x "$_uv_bucket" ]; then
                    _uv_scan_blocked=true
                    [ -n "$_uv_blocked_cache" ] || _uv_blocked_cache="$_uv_candidate"
                    continue
                fi
                # `|| true`: head closes the pipe and find's SIGPIPE under pipefail would drop the path.
                _uv_artifact=$(find -L "$_uv_bucket" -type f \
                    ! -name CACHEDIR.TAG ! -name .git ! -name .gitignore \
                    ! -name '.unsloth-write-probe.*' \
                    ! -name '*.lock' ! -name '*.msgpack' ! -name '*.http' ! -name '*.rev' \
                    -print 2>/dev/null | head -n 1) || true
                if [ -n "$_uv_artifact" ]; then
                    _uv_cand_populated=true
                    break
                fi
            done

            if [ "$_uv_glob" = off ]; then set -f; fi
        fi
        if [ "$_uv_cand_populated" = true ] && [ "$_uv_cand_writable" = true ]; then
            _uv_chosen_cache="$_uv_candidate"
        elif [ "$_uv_cand_populated" = true ] && [ -z "$_uv_warn_cache" ]; then
            _uv_warn_cache="$_uv_candidate"
        fi
    done
    unset _uv_candidate _uv_cand_populated _uv_cand_writable _uv_unmarked_studio

    if [ -n "$_uv_chosen_cache" ]; then
        UV_CACHE_DIR="$_uv_chosen_cache"
        if [ "$_uv_chosen_cache" = "$_uv_studio_cache" ]; then
            _UV_CACHE_MODE=studio
        else
            _UV_CACHE_MODE=shared
        fi
    else
        UV_CACHE_DIR="$_uv_studio_cache"
        _UV_CACHE_MODE=studio
        # An unwritable Studio cache is no fallback: prefer the warm suspect cache, then uv's default.
        if ! _uv_cache_is_writable "$_uv_studio_cache"; then
            if [ -n "$_uv_warn_cache" ]; then
                UV_CACHE_DIR="$_uv_warn_cache"
                if [ "$_uv_warn_cache" != "$_uv_studio_cache" ]; then
                    _UV_CACHE_MODE=shared
                fi
            elif [ -n "$_uv_default_cache" ] \
                 && [ "$_uv_default_cache" != "$_uv_studio_cache" ] \
                 && _uv_cache_is_writable "$_uv_default_cache"; then
                UV_CACHE_DIR="$_uv_default_cache"
                _UV_CACHE_MODE=shared
                _uv_studio_unusable=true
            fi
        fi
    fi
    export UV_CACHE_DIR
    _record_uv_cache_choice

    case "$_UV_CACHE_MODE" in
        shared)
            step "uv cache" "reusing existing shared cache ($UV_CACHE_DIR) to avoid duplicate Torch/CUDA downloads; use --isolated-uv-cache to isolate"
            ;;
        studio)
            if [ -n "$_uv_chosen_cache" ]; then
                step "uv cache" "reusing this install's Unsloth Studio cache ($UV_CACHE_DIR)"
            elif [ "$_uv_scan_blocked" = true ] && [ "$_uv_blocked_cache" != "$UV_CACHE_DIR" ]; then
                step "uv cache" "using new Unsloth Studio-owned cache ($UV_CACHE_DIR); part of $_uv_blocked_cache could not be read, so cached packages may download again" "$C_WARN"
            elif [ -n "$_uv_warn_cache" ] && [ "$_uv_warn_cache" != "$UV_CACHE_DIR" ]; then
                step "uv cache" "using new Unsloth Studio-owned cache ($UV_CACHE_DIR); $_uv_warn_cache is populated but not writable, so cached packages may download again" "$C_WARN"
            else
                step "uv cache" "using new Unsloth Studio-owned cache ($UV_CACHE_DIR)"
            fi
            ;;
    esac
}

_prepare_studio_uv_cache_for_launch() {
    [ "${_UV_CACHE_MODE:-}" = shared ] || return 0
    # Repoint the backend only to a cache whose root AND buckets are writable.
    _uv_launch_cache="$STUDIO_HOME/cache/uv"
    _uv_cache_is_writable "$_uv_launch_cache" || return 0
    UV_CACHE_DIR="$_uv_launch_cache"
    export UV_CACHE_DIR
}
_resolve_studio_destinations
# PATH before we prepend anything, so the shim setup can tell if a new login shell finds _LOCAL_BIN.
_UNSLOTH_LOGIN_PATH="$PATH"
VENV_DIR="$STUDIO_HOME/unsloth_studio"

# Claim the root before writing into it, but only one this run may take over (env mode is a
# user workspace). A sentinel counts only as a regular non-link file outside linked dirs.
_claim_sentinel() {
    [ -f "$1" ] || return 1
    [ -L "$1" ] && return 1
    [ -n "${2:-}" ] && [ -L "$2" ] && return 1
    return 0
}

_claim_studio_root() {
    _claim_marker="$STUDIO_HOME/.unsloth-studio-owned"
    # Leave a valid marker alone: a kill between unlink and write would lose the proof.
    _claim_sentinel "$_claim_marker" && return 0
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ] \
       && ! _claim_sentinel "$VENV_DIR/.unsloth-studio-owned" "$VENV_DIR" \
       && ! _claim_sentinel "$STUDIO_HOME/share/studio.conf" "$STUDIO_HOME/share"; then
        if [ -d "$STUDIO_HOME" ]; then
            { [ -r "$STUDIO_HOME" ] && [ -x "$STUDIO_HOME" ]; } || return 0
            _claim_glob=on
            case $- in *f*) _claim_glob=off ;; esac
            set +f
            _claim_empty=true
            for _claim_entry in "$STUDIO_HOME"/* "$STUDIO_HOME"/.[!.]* "$STUDIO_HOME"/..?*; do
                if [ -e "$_claim_entry" ] || [ -L "$_claim_entry" ]; then _claim_empty=false; break; fi
            done
            [ "$_claim_glob" = off ] && set -f
            [ "$_claim_empty" = true ] || return 0
        fi
    fi
    mkdir -p "$STUDIO_HOME" 2>/dev/null || true
    # Unlink first: the redirection would follow a symlink and truncate its target.
    rm -f "$_claim_marker" 2>/dev/null || true
    if [ -e "$_claim_marker" ] || [ -L "$_claim_marker" ]; then return 0; fi
    printf '' > "$_claim_marker" 2>/dev/null || true
}
_claim_studio_root

# Keep uv's cache on the same filesystem as the venv it fills.
# uv hardlinks wheels within one filesystem and copies across a boundary, so a moved STUDIO_HOME paid double the disk and stranded the cache. An explicit UV_CACHE_DIR wins. The fallback is required, since uv aborts on a cache it cannot create; mkdir -p exits 0 for an existing unwritable directory and -w reads the mode rather than the filesystem, so probe with a real create.
# True only where the installer assigns its own default; the variable reads the same either way.
_UV_CACHE_DIR_INSTALLER_DEFAULT=false
if [ -z "${UV_CACHE_DIR:-}" ]; then
    UV_CACHE_DIR="$STUDIO_HOME/cache/uv"
    export UV_CACHE_DIR
    _UV_CACHE_DIR_INSTALLER_DEFAULT=true
    if ! _uv_cache_root_is_writable "$UV_CACHE_DIR"; then
        echo "[WARN] Cannot write to $UV_CACHE_DIR -- using uv's default cache." >&2
        echo "[WARN] Wheels will be copied into the venv rather than hardlinked, costing extra disk." >&2
        unset UV_CACHE_DIR
        _UV_CACHE_DIR_INSTALLER_DEFAULT=false
    fi
fi
_VENV_ROLLBACK_DIR=""
_VENV_ROLLBACK_TARGET="$VENV_DIR"
_VENV_ROLLBACK_ACTIVE=false
_UV_MARKER_SAVED=false
_UV_MARKER_EXISTED=false
_UV_MARKER_PREVIOUS=""
# One flag for both rollbacks so a signal cannot restore half a committed install.
_STUDIO_INSTALL_COMMITTED=false

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
    # Publish rollback state before the rename so a signal after mv still finds the old venv.
    if ! mv "$_existing_dir" "$_candidate"; then
        _VENV_ROLLBACK_ACTIVE=false
        _VENV_ROLLBACK_DIR=""
        return 1
    fi
    # --no-rollback: drop the old env now. Clear rollback state first so a signal cannot restore
    # a half-deleted backup.
    if [ "${_NO_ROLLBACK:-false}" = true ]; then
        _VENV_ROLLBACK_ACTIVE=false
        _VENV_ROLLBACK_DIR=""
        rm -rf "$_candidate" 2>/dev/null || true
        # rm -f can still fail (immutable, busy mount); report what is actually on disk.
        if [ -e "$_candidate" ] || [ -L "$_candidate" ]; then
            _VENV_DISCARD_LEFTOVER="$_candidate"
            substep "could not discard the previous environment at $_candidate" "$C_WARN"
            substep "it is no longer used for rollback; remove it by hand to reclaim the space." "$C_WARN"
        else
            _VENV_DISCARDED=true
            substep "previous environment discarded (--no-rollback); a failed install cannot be undone"
        fi
        return 0
    fi
    substep "previous environment preserved for rollback"
}

_free_space_kb() {  # path
    df -Pk "$1" 2>/dev/null | awk 'NR == 2 { print $4 }'
}

# Below 64 MiB a full disk is the cause. Fold the suffix into ERROR_DEFAULT: --tauri UI shows only that.
_set_disk_full_suffix() {
    _DISK_FULL_SUFFIX=""
    _DISK_FULL_REMEDY=""
    _DISK_FULL_MB=""
    [ -n "${STUDIO_HOME:-}" ] || return 0
    _dfs_free=$(_free_space_kb "$STUDIO_HOME")
    [ -n "$_dfs_free" ] || return 0
    [ "$_dfs_free" -lt 65536 ] 2>/dev/null || return 0
    _DISK_FULL_MB=$((_dfs_free / 1024))
    # A failed discard left a tree behind; deleting it is likely what makes a retry fit.
    if [ -n "${_VENV_DISCARD_LEFTOVER:-}" ]; then
        _DISK_FULL_REMEDY="Free some space and re-run. The previous environment could not be removed and is still at $_VENV_DISCARD_LEFTOVER; deleting it will reclaim that space."
    elif [ "${_VENV_DISCARDED:-false}" = true ]; then
        _DISK_FULL_REMEDY="Free some space and re-run. The previous environment was already discarded by --no-rollback, so the installer has nothing further of its own to reclaim."
    elif [ "${_NO_ROLLBACK:-false}" = true ]; then
        _DISK_FULL_REMEDY="Free some space and re-run."
    else
        _DISK_FULL_REMEDY="Free some space and re-run. --no-rollback (UNSLOTH_INSTALL_NO_ROLLBACK=1) drops the previous environment instead of keeping a copy of it during the install."
    fi
    _DISK_FULL_SUFFIX=": $STUDIO_HOME has only $_DISK_FULL_MB MB free, so the disk is full, which is very likely the cause. $_DISK_FULL_REMEDY"
}

# uv creates only into an absent path or empty dir; hidden entries and dangling links count.
_dir_has_entries() {  # dir
    if [ ! -d "$1" ]; then
        # -L too: mkdir(2) gives EEXIST for a dangling symlink, which -e misses.
        { [ -e "$1" ] || [ -L "$1" ]; } && return 0
        return 1
    fi
    # Unreadable dirs cannot be enumerated and would look empty; fail closed like install.ps1.
    { [ -r "$1" ] && [ -x "$1" ]; } || return 0
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

# Move the venv aside instead of rm -rf: the legacy migration has no rollback copy, so a failed
# uv venv would leave no environment. With a replacement in flight, plain removal is fine.
_discard_venv_for_recreate() {
    if [ "$_VENV_ROLLBACK_ACTIVE" != true ] && [ -d "$1" ] \
       && _start_studio_venv_replacement "$1"; then
        return 0
    fi
    rm -rf "$1"
}

_restore_studio_venv_replacement() {
    [ "${_STUDIO_INSTALL_COMMITTED:-false}" = true ] && return 0
    [ "$_VENV_ROLLBACK_ACTIVE" = true ] || return 0
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
        _studio_venv_rollback_must_be_preserved "$_stale_rollback" && continue
        if rm -rf "$_stale_rollback"; then
            substep "removed stale environment rollback ${_stale_rollback##*/}"
        else
            echo "⚠️  Could not remove stale environment rollback $_stale_rollback" >&2
        fi
    done
}

_commit_studio_venv_replacement() {
    # First and alone, because a signal can land between any two statements.
    _STUDIO_INSTALL_COMMITTED=true
    _UV_MARKER_SAVED=false
    if [ "$_VENV_ROLLBACK_ACTIVE" = true ]; then
        _rollback_to_remove="$_VENV_ROLLBACK_DIR"
        # Clear restore state before deleting so an interrupt cannot restore a half-deleted backup.
        _VENV_ROLLBACK_ACTIVE=false
        _VENV_ROLLBACK_DIR=""
        if [ -n "$_rollback_to_remove" ] \
           && { [ -e "$_rollback_to_remove" ] || [ -L "$_rollback_to_remove" ]; }; then
            if ! rm -rf "$_rollback_to_remove"; then
                echo "⚠️  Could not remove environment rollback $_rollback_to_remove" >&2
            fi
        fi
    fi
    # Prune only after success so an interrupted install keeps the last known-good env.
    _prune_stale_studio_venv_rollbacks
}

_cleanup_install_temporaries() {
    [ -n "${_UV_OVERRIDE_TMPDIR:-}" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
    [ -n "${_UV_INSTALL_NAME_TOOL_SHIM_DIR:-}" ] && rm -rf "$_UV_INSTALL_NAME_TOOL_SHIM_DIR" 2>/dev/null || true
    [ -n "${_UV_VENV_CAPTURE_DIR:-}" ] && rm -rf "$_UV_VENV_CAPTURE_DIR" 2>/dev/null || true
    [ -n "${_UNSLOTH_TORCH_OVERRIDES:-}" ] && rm -f "$_UNSLOTH_TORCH_OVERRIDES" 2>/dev/null || true
    # The pinned uv path's cleanup runs only on return; a Ctrl-C would leave staging files on PATH.
    [ -n "${_UIP_WORK:-}" ] && rm -rf "$_UIP_WORK" 2>/dev/null || true
    [ -n "${_UIP_STAGE:-}" ] && rm -f "$_UIP_STAGE" 2>/dev/null || true
    [ -n "${_UIP_STAGE2:-}" ] && rm -f "$_UIP_STAGE2" 2>/dev/null || true
    [ -n "${_ROCM_TAG_MEMO_DIR:-}" ] && rm -rf "$_ROCM_TAG_MEMO_DIR" 2>/dev/null || true
    [ -n "${_ric_log:-}" ] && rm -f "$_ric_log" 2>/dev/null || true
    # This shell holds the probe's ceiling; kill the probe on cancel or it runs on.
    if [ -n "${_UV_PROBE_TARGET:-}" ] && [ -n "${_UV_PROBE_PID:-}" ]; then
        # Two seconds so a cancel does not look ignored.
        _uv_probe_terminate "$_UV_PROBE_TARGET" "$_UV_PROBE_PID" 2
        _UV_PROBE_TARGET=""
        _UV_PROBE_PID=""
    fi
}

_on_install_exit() {
    _status=$?
    if [ "$_status" -ne 0 ]; then
        # Restore must come first and nothing may prevent it (a failed write under set -e would abort).
        # Disk is measured before the restore, which can free gigabytes; `|| true` so it cannot abort.
        if [ "${_DISK_FULL_REPORTED:-false}" != true ] && command -v _set_disk_full_suffix >/dev/null 2>&1; then
            _set_disk_full_suffix || true
        fi
        _restore_studio_venv_replacement
        _restore_uv_cache_marker
        # Written after the restore, every write `|| true`: a closed --tauri stdout must not abort the trap.
        if [ "${_DISK_FULL_REPORTED:-false}" != true ] && [ -n "${_DISK_FULL_SUFFIX:-}" ]; then
            tauri_log "ERROR_DEFAULT" "unsloth studio install failed (exit code $_status)$_DISK_FULL_SUFFIX" || true
            echo "       $STUDIO_HOME has only $_DISK_FULL_MB MB free -- the disk is full, which is very likely the cause." >&2 || true
            echo "       $_DISK_FULL_REMEDY" >&2 || true
        fi
    fi
    _cleanup_install_temporaries
    exit "$_status"
}

_on_install_signal() {
    _signal_status="$1"
    trap - EXIT
    trap '' HUP INT TERM
    _restore_studio_venv_replacement
    _restore_uv_cache_marker
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
_UV_PROBE_TARGET=""
_UV_PROBE_PID=""
trap _on_install_exit EXIT
trap '_on_install_signal 129' HUP
trap '_on_install_signal 130' INT
trap '_on_install_signal 143' TERM

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

_is_pkg_installed() {
    case "$1" in
        build-essential) command -v gcc >/dev/null 2>&1 ;;
        libcurl4-openssl-dev)
            command -v dpkg >/dev/null 2>&1 && dpkg -s "$1" >/dev/null 2>&1 ;;
        pciutils)
            command -v lspci >/dev/null 2>&1 ;;
        bubblewrap)
            command -v bwrap >/dev/null 2>&1 ;;
        *) command -v "$1" >/dev/null 2>&1 ;;
    esac
}

# Distro label for the elevation prompt (#6207), read from /etc/os-release.
_apt_distro_description() {
    # Plain ( ... ) subshell, not $(): bash 3.2 misparses case `;;` inside command substitution.
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

# `test -r` passes in containers where open() fails with ENXIO, so probe with a real open.
# Subshell required: in dash a failed redirection on `:` exits the script.
_can_read_tty() {
    ( : </dev/tty ) >/dev/null 2>&1
}

_resolve_systemd_install_script() {
    if [ "$_REPO_IS_CHECKOUT" = "1" ] && [ -f "$_REPO_ROOT/studio/systemd/install_user_service.sh" ]; then
        printf '%s\n' "$_REPO_ROOT/studio/systemd/install_user_service.sh"
        return 0
    fi
    [ -x "$VENV_DIR/bin/python" ] || return 0
    "$VENV_DIR/bin/python" -I -c \
        "import importlib.resources as r; print(r.files('studio') / 'systemd' / 'install_user_service.sh')" \
        2>/dev/null || true
}

_install_systemd_user_service() {
    case "$OS" in
        linux|wsl) ;;
        *) step "systemd" "UNSLOTH_INSTALL_SYSTEMD is Linux only; skipped" "$C_WARN"; return 0 ;;
    esac
    _sd_script=$(_resolve_systemd_install_script)
    if [ -z "$_sd_script" ] || [ ! -f "$_sd_script" ]; then
        step "systemd" "service helper not found in this install; skipped" "$C_WARN"
        return 0
    fi

    set -- --unsloth-exe "$VENV_DIR/bin/unsloth" \
        --host "${UNSLOTH_SYSTEMD_HOST:-127.0.0.1}" --port "${UNSLOTH_SYSTEMD_PORT:-8888}" --enable --start
    [ "$_STUDIO_HOME_REDIRECT" != "default" ] && set -- "$@" --studio-home "$STUDIO_HOME"

    if bash "$_sd_script" "$@" >/dev/null; then
        _SYSTEMD_STARTED=true
        step "systemd" "user service enabled (unsloth-studio.service)"
        substep "status: systemctl --user status unsloth-studio.service"
        substep "logs:   journalctl --user -u unsloth-studio.service -f"
        substep "stop:   systemctl --user stop unsloth-studio.service"
        substep "binds 127.0.0.1 by default; set UNSLOTH_SYSTEMD_HOST=0.0.0.0 before install for LAN"
        substep "for boot without a login session, run once: loginctl enable-linger \"\$USER\""
    else
        step "systemd" "user service install failed; see the error above" "$C_WARN"
    fi
}

_smart_apt_install() {
    _PKGS="$*"

    apt-get update -y </dev/null >/dev/null 2>&1 || true
    apt-get install -y $_PKGS </dev/null >/dev/null 2>&1 || true

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

    # Optional installs never elevate: no prompt or Tauri NEED_SUDO over unused build tools.
    if [ "${_SMART_APT_OPTIONAL:-false}" = true ]; then
        return 2
    fi

    if [ "$TAURI_MODE" = true ]; then
        # Rust handles elevation.
        tauri_log "NEED_SUDO" "$_STILL_MISSING"
        exit 2
    fi

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
            read -r REPLY </dev/tty || REPLY="n"
            case "$REPLY" in
                [nN]*)
                    echo ""
                    echo "    Please install these packages first, then re-run Unsloth Studio setup:"
                    echo "    sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                    exit 1
                    ;;
            esac
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
            # -n -k: only a real NOPASSWD rule passes, without prompting into a closed stdin.
            echo "    No terminal to confirm on; trying passwordless sudo."
            if sudo -n -k apt-get update -y </dev/null &&
                sudo -n -k apt-get install -y $_STILL_MISSING </dev/null; then
                echo "    Installed with passwordless sudo."
            else
                echo ""
                echo "    Could not install these packages: $_STILL_MISSING"
                echo "    Detected ${_ad_desc}."
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

# Install id is 64 lowercase hex (backend _STUDIO_INSTALL_ID_RE). It is baked into a single-quoted
# launcher assignment, so anything else is rejected. LC_ALL=C keeps character classes stable.
_css_install_id_is_valid() (
    LC_ALL=C
    export LC_ALL
    case "${1:-}" in
        "" | *[!0123456789abcdef]*) return 1 ;;
    esac
    [ "${#1}" -eq 64 ]
)

_css_read_valid_install_id() (
    LC_ALL=C
    export LC_ALL
    [ -f "$1" ] || return 0
    [ -s "$1" ] || return 0
    # Shell variables drop NULs, so detect them separately or an id with NUL reads as valid.
    if [ -n "$({ tr -dc '\000' < "$1" | tr '\000' 'N'; } 2>/dev/null)" ]; then
        return 0
    fi
    # Group the redirect so the shell's own error is silenced; a failed read must not license a rewrite.
    _cvi_id=$({ cat "$1"; } 2>/dev/null) || return 1
    _cvi_id=${_cvi_id#"${_cvi_id%%[![:space:]]*}"}
    _cvi_id=${_cvi_id%"${_cvi_id##*[![:space:]]}"}
    if _css_install_id_is_valid "$_cvi_id"; then
        printf '%s' "$_cvi_id"
    fi
)

# ── Helper: create desktop shortcuts and launcher script ──
create_studio_shortcuts() {
    _css_exe="$1"
    _css_os="$2"

    if [ ! -x "$_css_exe" ]; then
        echo "[WARN] Cannot create shortcuts: unsloth not found at $_css_exe"
        return 0
    fi

    _css_exe_dir=$(cd "$(dirname "$_css_exe")" && pwd)
    _css_exe="$_css_exe_dir/$(basename "$_css_exe")"

    _css_data_dir="$DATA_DIR"
    _css_launcher="$_css_data_dir/launch-studio.sh"
    _css_icon_png="$_css_data_dir/unsloth-studio.png"
    _css_gem_png="$_css_data_dir/unsloth-gem.png"

    mkdir -p "$_css_data_dir"

    _css_id_dir="$STUDIO_HOME/share"
    mkdir -p "$_css_id_dir"
    _css_id_file="$_css_id_dir/studio_install_id"
    # Reuse a valid id; unreadable is not malformed (another user's backend may report it).
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
        # Publish no-clobber via ln (EEXIST) since the desktop app mints the same id; $$ is unreliable
        # in subshells, so the id is in the temp name.
        _css_id_tmp="$_css_id_file.$$.$(printf '%.8s' "$_css_new_id").tmp"
        if printf '%s' "$_css_new_id" > "$_css_id_tmp"; then
            if ! ln "$_css_id_tmp" "$_css_id_file" 2>/dev/null; then
                # Replace an invalid incumbent with one rename. -d: renaming onto a directory moves inside it.
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
        _css_studio_root_id=$(_css_read_valid_install_id "$_css_id_file") || true
        unset _css_new_id _css_id_tmp _css_incumbent
    fi
    if [ -z "$_css_studio_root_id" ]; then
        echo "[WARN] Cannot create launcher: failed to read $_css_id_file" >&2
        return 1
    fi
    _css_is_env_mode=false
    [ "$_STUDIO_HOME_REDIRECT" = "env" ] && _css_is_env_mode=true

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

# A missing or malformed id makes /api/health report "", which the baked id never matches: restore ours no-clobber (the desktop app mints it too), leaving a different valid id or an unreadable file alone.
_repair_studio_install_id() (
    LC_ALL=C
    export LC_ALL
    _rid_file=${STUDIO_INSTALL_ID_FILE:-}
    [ -n "$_rid_file" ] && [ -n "$_EXPECTED_STUDIO_ROOT_ID" ] || return 0
    _rid_has_valid() {
        [ -e "$_rid_file" ] || return 1
        [ -f "$_rid_file" ] || return 0
        _rid_cur=$({ cat "$_rid_file"; } 2>/dev/null) || return 0
        _rid_cur=${_rid_cur#"${_rid_cur%%[![:space:]]*}"}
        _rid_cur=${_rid_cur%"${_rid_cur##*[![:space:]]}"}
        case "$_rid_cur" in
            "" | *[!0123456789abcdef]*) return 1 ;;
        esac
        [ "${#_rid_cur}" -eq 64 ]
    }
    _rid_has_valid && return 0
    mkdir -p "$(dirname "$_rid_file")" 2>/dev/null || return 0
    _rid_tmp=$(mktemp "$_rid_file.XXXXXX" 2>/dev/null) || return 0
    if printf '%s' "$_EXPECTED_STUDIO_ROOT_ID" > "$_rid_tmp" 2>/dev/null; then
        if ! ln "$_rid_tmp" "$_rid_file" 2>/dev/null && ! _rid_has_valid; then
            mv -f "$_rid_tmp" "$_rid_file" 2>/dev/null || true
        fi
        chmod 600 "$_rid_file" 2>/dev/null || true
    fi
    rm -f "$_rid_tmp" 2>/dev/null
    return 0
)
_repair_studio_install_id

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

    # Bake fixed placeholders FIRST so a literal @@STUDIO_ROOT_ID@@ in $DATA_DIR survives.
    sed -e "s|@@STUDIO_ROOT_ID@@|$_css_studio_root_id|g" \
        -e "s|@@INSTALLED_IS_ENV_MODE@@|$_css_is_env_mode|g" \
        "$_css_launcher" > "$_css_launcher.tmp" \
        && mv "$_css_launcher.tmp" "$_css_launcher"

    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
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

    _css_quoted_exe=$(printf '%s' "$_css_exe" | sed "s/'/'\\\\''/g")
    {
        printf '%s\n' "UNSLOTH_EXE='$_css_quoted_exe'"
        _css_quoted_id_file=$(printf '%s' "$_css_id_file" | sed "s/'/'\\\\''/g")
        printf '%s\n' "STUDIO_INSTALL_ID_FILE='$_css_quoted_id_file'"
        if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
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
            printf '%s\n' 'if [ -z "${UNSLOTH_LLAMA_CPP_PATH:-}" ]; then'
            printf '%s\n' "    export UNSLOTH_LLAMA_CPP_PATH='$_css_quoted_llama'"
            printf '%s\n' 'fi'
        fi
    } > "$_css_data_dir/studio.conf"

    _css_script_dir=""
    if [ -n "${0:-}" ] && [ -f "$0" ]; then
        _css_script_dir=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || true
    fi

    _css_found_icon=""
    _css_venv_dir=$(dirname "$(dirname "$_css_exe")")
    # `studio` is a top-level package in the wheel, and only frontend/dist ships.
    for _sp in "$_css_venv_dir"/lib/python*/site-packages/studio/frontend/dist; do
        if [ -f "$_sp/rounded-512.png" ]; then
            _css_found_icon="$_sp/rounded-512.png"
        fi
    done
    if [ -z "$_css_found_icon" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/frontend/public/rounded-512.png" ]; then
        _css_found_icon="$_css_script_dir/studio/frontend/public/rounded-512.png"
    fi

    if [ -n "$_css_found_icon" ]; then
        cp "$_css_found_icon" "$_css_icon_png" 2>/dev/null || true
        cp "$_css_found_icon" "$_css_gem_png" 2>/dev/null || true
    else
        download "https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/rounded-512.png" "$_css_icon_png" 2>/dev/null || true
        cp "$_css_icon_png" "$_css_gem_png" 2>/dev/null || true
    fi

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

    _css_tauri_icns=""
    for _sp in "$_css_venv_dir"/lib/python*/site-packages/studio/src-tauri/icons; do
        if [ -f "$_sp/icon.icns" ]; then
            _css_tauri_icns="$_sp/icon.icns"
        fi
    done
    if [ -z "$_css_tauri_icns" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/src-tauri/icons/icon.icns" ]; then
        _css_tauri_icns="$_css_script_dir/studio/src-tauri/icons/icon.icns"
    fi

    _css_tauri_png=""
    for _sp in "$_css_venv_dir"/lib/python*/site-packages/studio/src-tauri/icons; do
        if [ -f "$_sp/icon.png" ]; then
            _css_tauri_png="$_sp/icon.png"
        fi
    done
    if [ -z "$_css_tauri_png" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/src-tauri/icons/icon.png" ]; then
        _css_tauri_png="$_css_script_dir/studio/src-tauri/icons/icon.png"
    fi

    # Env-mode is workspace-scoped: skip launchers that may point at a deleted workspace.
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
        substep "wrote launcher at $_css_launcher (persistent shortcuts skipped in env-override mode)"
        return 0
    fi

    _css_created=0

    if [ "$_css_os" = "linux" ]; then
        _css_app_dir="$HOME/.local/share/applications"
        mkdir -p "$_css_app_dir"

        _css_desktop="$_css_app_dir/unsloth-studio.desktop"
        _css_exec_escaped=$(printf '%s' "$_css_launcher" | sed 's/\\/\\\\/g; s/"/\\"/g')
        # Prefer the higher-resolution Tauri icon.png, but persist it under the installed data directory so local-checkout shortcuts survive repo moves.
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

        if [ -d "$HOME/Desktop" ]; then
            cp "$_css_desktop" "$HOME/Desktop/unsloth-studio.desktop" 2>/dev/null || true
            chmod +x "$HOME/Desktop/unsloth-studio.desktop" 2>/dev/null || true
            if command -v gio >/dev/null 2>&1; then
                gio set "$HOME/Desktop/unsloth-studio.desktop" metadata::trusted true 2>/dev/null || true
            fi
        fi

        update-desktop-database "$_css_app_dir" 2>/dev/null || true
        _css_created=1

    elif [ "$_css_os" = "macos" ]; then
        _css_app="$HOME/Applications/Unsloth Studio.app"
        _css_contents="$_css_app/Contents"
        _css_macos_dir="$_css_contents/MacOS"
        _css_res_dir="$_css_contents/Resources"
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
        # Old `ln -sf` shortcuts planted a self-referential link inside the bundle.
        if [ -L "$_css_app/Unsloth Studio.app" ]; then
            rm -f "$_css_app/Unsloth Studio.app" 2>/dev/null || true
        fi
        mkdir -p "$_css_macos_dir" "$_css_res_dir"

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

        # Prefer the pre-built Tauri icon.icns; fall back to generating one from the gem PNG via sips+iconutil, then to a plain PNG copy.
        # ── AppIcon ──
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
        if [ ! -f "$_css_res_dir/AppIcon.icns" ] && [ -f "$_css_icon_png" ]; then
            cp "$_css_icon_png" "$_css_res_dir/AppIcon.icns" 2>/dev/null || true
        fi

        # Touch so Finder indexes it
        touch "$_css_app"

        # -n is required: without it a re-run creates the link inside the bundle.
        if [ -d "$HOME/Desktop" ]; then
            ln -sfn "$_css_app" "$HOME/Desktop/Unsloth Studio" 2>/dev/null || true
        fi
        _css_created=1

    elif [ "$_css_os" = "wsl" ]; then
        _css_distro="${WSL_DISTRO_NAME:-}"

        _css_wsl_args=""
        if [ -n "$_css_distro" ]; then
            _css_wsl_args="-d \"$_css_distro\" "
        fi
        _css_wsl_args="${_css_wsl_args}-- bash -l -c \"exec \\\"$_css_launcher\\\"\""

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

        _css_sc_args_ps=$(printf '%s' "$_css_sc_args" | sed "s/'/''/g")

        if [ -n "$_css_distro" ]; then
            _css_lnk_name="Unsloth Studio (WSL - ${_css_distro}).lnk"
        else
            _css_lnk_name="Unsloth Studio (WSL).lnk"
        fi
        _css_lnk_name_ps=$(printf '%s' "$_css_lnk_name" | sed "s/'/''/g")

        _css_wsl_ico_win=""
        for _sp in "$_css_venv_dir"/lib/python*/site-packages/studio/frontend/dist; do
            if [ -f "$_sp/unsloth.ico" ] && command -v wslpath >/dev/null 2>&1; then
                _css_wsl_ico_win=$(wslpath -w "$_sp/unsloth.ico" 2>/dev/null) || _css_wsl_ico_win=""
            fi
        done
        _css_wsl_ico_win_ps=$(printf '%s' "$_css_wsl_ico_win" | sed "s/'/''/g")

        # Create shortcuts via a temp PowerShell script to avoid escaping issues.
        # Written to Windows %TEMP%, not WSL /tmp: a UNC path is remote and RemoteSigned refuses it.
        _css_win_temp=""
        if command -v wslpath >/dev/null 2>&1 && command -v cmd.exe >/dev/null 2>&1; then
            # cmd /d skips AutoRun banners; values are quoted because cmd expands %TEMP% before parsing &.
            # Candidates in order; a redirected network %TEMP% is remote, so fall through to a local dir.
            _css_win_temp_list=$(cmd.exe /d /c 'echo "%TEMP%"&echo "%LOCALAPPDATA%\Temp"&echo "%SystemRoot%\Temp"' 2>/dev/null \
                | tr -d '\r') || _css_win_temp_list=""
            # Mapped network drives are the remote zone too, so exclude them like UNC paths.
            _css_net_drives=$(cmd.exe /d /c 'net use' 2>/dev/null | tr -d '\r' \
                | awk '/\\\\/ { for (i = 1; i <= NF; i++) if ($i ~ /^[A-Za-z]:$/) print substr($i, 1, 1) }' \
                | tr 'a-z' 'A-Z') || _css_net_drives=""

            _css_old_ifs=$IFS
            IFS='
'
            for _css_cand in $_css_win_temp_list; do
                IFS=$_css_old_ifs
                _css_cand=${_css_cand#\"}
                _css_cand=${_css_cand%\"}
                _css_cand=$(printf '%s' "$_css_cand" | sed 's/[[:space:]]*$//')
                case "$_css_cand" in
                    # Unexpanded variable, or a UNC path, which is the remote zone.
                    ""|'%'*'%'*|'\\'*) continue ;;
                esac
                _css_cand_letter=$(printf '%s' "$_css_cand" | cut -c1 | tr 'a-z' 'A-Z')
                _css_is_net=0
                for _css_nd in $_css_net_drives; do
                    [ "$_css_nd" = "$_css_cand_letter" ] && _css_is_net=1 && break
                done
                [ "$_css_is_net" = 1 ] && continue
                _css_cand_unix=$(wslpath -u "$_css_cand" 2>/dev/null) || continue
                [ -d "$_css_cand_unix" ] || continue
                _css_win_temp=$_css_cand_unix
                break
            done
            IFS=$_css_old_ifs
            if [ -n "$_css_win_temp" ] && [ ! -d "$_css_win_temp" ]; then
                _css_win_temp=""
            fi
        fi
        # No fallback to WSL /tmp: it is a UNC path again, and this branch is best-effort.
        _css_ps1_tmp=""
        if [ -n "$_css_win_temp" ]; then
            _css_ps1_tmp=$(mktemp "$_css_win_temp/unsloth-shortcut-XXXXXX.ps1" 2>/dev/null) || _css_ps1_tmp=""
        fi
        if [ -n "$_css_sc_target" ]; then
            _css_ps1_body=$(cat << WSLPS1_EOF
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
\$packagedIcon = '$_css_wsl_ico_win_ps'
if (-not (Test-Path -LiteralPath \$iconPath) -and \$packagedIcon -and (Test-Path -LiteralPath \$packagedIcon)) {
    try {
        New-Item -ItemType Directory -Force -Path \$iconDir | Out-Null
        Copy-Item -LiteralPath \$packagedIcon -Destination \$iconPath -Force -ErrorAction Stop
    } catch {}
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
# Per-item SHCNE_UPDATEITEM so a rewritten same-name .lnk re-reads its icon; the global broadcast
# alone does not. Called through a Windows Python's ctypes, as install.ps1 does, so this script
# defines no native types. Without one the shortcut still works, and the heavier refresh below
# still runs on a first install or an icon change.
if (\$created.Count -gt 0) {
    \$isAdmin = \$true
    try {
        \$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {}
    # Never launch a user-writable interpreter from an elevated shell.
    \$pyCandidates = @()
    if (-not \$isAdmin) {
        \$pyCandidates += Join-Path \$env:USERPROFILE '.unsloth\studio\unsloth_studio\Scripts\python.exe'
        foreach (\$name in @('python3', 'python')) {
            try {
                foreach (\$cmd in @(Get-Command \$name -All -CommandType Application -ErrorAction SilentlyContinue)) {
                    if (\$cmd -and \$cmd.Source) { \$pyCandidates += \$cmd.Source }
                }
            } catch {}
        }
    }
    # SHCNF_FLUSH (0x1000): the child exits at once and a queued notification would be lost.
    \$refreshCode = "import ctypes,os;from ctypes import wintypes as w;f=ctypes.WinDLL('shell32').SHChangeNotify;f.restype=None;f.argtypes=[w.LONG,w.UINT,w.LPCWSTR,w.LPCWSTR];[f(0x2000,0x1005,p,None) for p in os.environ['UNSLOTH_SHORTCUT_PATHS'].split('|') if p];f(0x8000000,0x1000,None,None);print('ok')"
    foreach (\$py in \$pyCandidates) {
        # The WindowsApps alias opens the Store instead of running anything.
        if (-not \$py -or \$py -like '*\Microsoft\WindowsApps\*') { continue }
        if (-not (Test-Path -LiteralPath \$py -PathType Leaf)) { continue }
        \$proc = \$null
        try {
            \$psi = New-Object System.Diagnostics.ProcessStartInfo
            \$psi.FileName = \$py
            # -I -S: no user site, no PYTHON* variables, no sitecustomize. -B: write no .pyc.
            \$psi.Arguments = '-I -S -B -c "' + \$refreshCode + '"'
            # '|' cannot appear in a Windows path, so it separates them safely.
            \$psi.EnvironmentVariables['UNSLOTH_SHORTCUT_PATHS'] = (\$created -join '|')
            \$psi.WorkingDirectory = Split-Path -Parent \$py
            \$psi.UseShellExecute = \$false
            \$psi.RedirectStandardOutput = \$true
            \$psi.RedirectStandardError = \$true
            \$psi.CreateNoWindow = \$true
            \$proc = [System.Diagnostics.Process]::Start(\$psi)
            \$out = \$proc.StandardOutput.ReadToEndAsync()
            \$null = \$proc.StandardError.ReadToEndAsync()
            if (-not \$proc.WaitForExit(10000)) {
                try { \$proc.Kill() } catch {}
                continue
            }
            if (\$proc.ExitCode -eq 0 -and \$out.Wait(2000) -and "\$(\$out.Result)".Trim() -eq 'ok') { break }
        } catch {
        } finally {
            if (\$proc) { try { \$proc.Dispose() } catch {} }
        }
    }
}
# Heavier on-disk icon-cache clear + StartMenuExperienceHost tile rebuild
# (preserve start2.bin) only on first install or a real icon change, so a no-op
# WSL reinstall does not purge caches and kill a shell process for nothing.
# See tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
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
)

            if [ -n "$_css_ps1_tmp" ]; then
                printf '%s\n' "$_css_ps1_body" > "$_css_ps1_tmp"
                _css_ps1_win=$(wslpath -w "$_css_ps1_tmp" 2>/dev/null)
                if [ -n "$_css_ps1_win" ]; then
                    powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "$_css_ps1_win" >/dev/null 2>&1 && _css_created=1
                fi
                rm -f "$_css_ps1_tmp"
            else
                # No Windows dir is reachable (automount off): pipe the script on stdin, where no execution
                # policy applies. Use our own pipe: a piped install's stdin is the download (#7548).
                printf '%s\n' "$_css_ps1_body" | powershell.exe -NoProfile -Command - >/dev/null 2>&1 && _css_created=1
            fi
        fi
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

tauri_log "STEP" "Detecting platform"
OS="linux"
if [ "$(uname)" = "Darwin" ]; then
    OS="macos"
elif grep -qi microsoft /proc/version 2>/dev/null; then
    OS="wsl"
fi
step "platform" "$OS"

if [ "$_SHORTCUTS_ONLY" = true ]; then
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
# Rosetta is tracked apart from MAC_INTEL: hardware checks must still see Apple Silicon.
_MAC_ROSETTA=false
if [ "$OS" = "macos" ] && [ "$_ARCH" = "x86_64" ]; then
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

SKIP_TORCH=false
if [ "$_NO_TORCH_FLAG" = true ] || [ "$MAC_INTEL" = true ]; then
    SKIP_TORCH=true
fi

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

# Discrete AMD cards are not in /proc/cpuinfo, so ask the Windows host via WMI.
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

_run_bounded() {
    _rb_secs=10
    if [ "${1:-}" = "--secs" ]; then _rb_secs=$2; shift 2; fi
    if command -v timeout >/dev/null 2>&1; then
        timeout "$_rb_secs" "$@"
    else
        "$@"
    fi
}

_cvd_hides_nvidia() {
    [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] || return 1
    _cvd_trim=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]')
    [ -z "$_cvd_trim" ] || [ "$_cvd_trim" = "-1" ]
}

# NVIDIA inventory via NVML / CUDA driver API when nvidia-smi is absent or hangs. Inline because
# studio/nvidia_probe.py is not on disk yet. Memoised so all callers agree.
_NVIDIA_LIBRARY_INVENTORY_STATE=""
_NVIDIA_LIBRARY_INVENTORY_VALUE=""
_nvidia_library_inventory() {
    [ "${UNSLOTH_NVIDIA_LIBRARY_PROBE:-1}" != "0" ] || return 1
    case "${_NVIDIA_LIBRARY_INVENTORY_STATE:-}" in
        found) printf '%s\n' "$_NVIDIA_LIBRARY_INVENTORY_VALUE"; return 0 ;;
        none) return 1 ;;
    esac
    if command -v python3 >/dev/null 2>&1; then _nli_py=python3
    elif [ -n "${VENV_DIR:-}" ] && [ -x "$VENV_DIR/bin/python" ]; then _nli_py="$VENV_DIR/bin/python"
    else return 1
    fi
    _NVIDIA_LIBRARY_INVENTORY_STATE="none"
    _NVIDIA_LIBRARY_INVENTORY_VALUE=""
    for _nli_reader in nvml cuda; do
        case "$_nli_reader" in nvml) _nli_secs=30 ;; *) _nli_secs=20 ;; esac
        _NVIDIA_LIBRARY_INVENTORY_VALUE=$(_run_bounded --secs "$_nli_secs" "$_nli_py" -I - "$_nli_reader" 2>/dev/null <<'PY'
import ctypes, os, sys

def load(*names):
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

def sym(lib, name):  # an older NVML exports the unversioned entry point only
    return getattr(lib, name, None) or getattr(lib, name.replace("_v2", ""))

def nvml():
    lib = load("libnvidia-ml.so.1", "libnvidia-ml.so")
    if lib is None or sym(lib, "nvmlInit_v2")() != 0:
        return None
    try:
        count, version = ctypes.c_uint(), ctypes.c_int()
        if sym(lib, "nvmlDeviceGetCount_v2")(ctypes.byref(count)) != 0 or not count.value:
            return None
        if sym(lib, "nvmlSystemGetCudaDriverVersion_v2")(ctypes.byref(version)) != 0 or version.value < 1000:
            return None
        caps = []
        for i in range(count.value):
            dev, major, minor = ctypes.c_void_p(), ctypes.c_int(), ctypes.c_int()
            if sym(lib, "nvmlDeviceGetHandleByIndex_v2")(i, ctypes.byref(dev)) != 0 or \
               lib.nvmlDeviceGetCudaComputeCapability(dev, ctypes.byref(major), ctypes.byref(minor)) != 0:
                return None  # one unreadable GPU voids the source
            caps.append(f"{major.value}.{minor.value}")
        return version.value, caps
    finally:
        lib.nvmlShutdown()

def cuda():
    # The driver API honours CUDA_VISIBLE_DEVICES; the inventory must be the physical one, so a
    # hidden pre-Turing card still caps the family (studio/nvidia_probe.py does the same).
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    lib = load("libcuda.so.1", "libcuda.so")
    if lib is None or lib.cuInit(0) != 0:
        return None
    count, version = ctypes.c_int(), ctypes.c_int()
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0 or not count.value:
        return None
    if lib.cuDriverGetVersion(ctypes.byref(version)) != 0 or version.value < 1000:
        return None
    caps = []
    for i in range(count.value):
        dev, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, _MINOR = 76.
        if lib.cuDeviceGet(ctypes.byref(dev), i) != 0 or \
           lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) != 0 or \
           lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) != 0:
            return None
        caps.append(f"{major.value}.{minor.value}")
    return version.value, caps

# argv[1] runs one reader ("nvml" / "cuda") so each gets its own deadline; none runs both.
only = {"nvml": (nvml,), "cuda": (cuda,)}.get(sys.argv[1] if len(sys.argv) > 1 else "")
found = None
for reader in only or (nvml, cuda):  # a reader that raises (a missing symbol) yields to the next
    try:
        found = reader()
    except Exception:
        found = None
    if found and found[1]:
        break
if not found or not found[1]:
    sys.exit(1)
version, caps = found
print(f"{version // 1000}.{version % 1000 // 10} {','.join(caps)}")
PY
) && [ -n "$_NVIDIA_LIBRARY_INVENTORY_VALUE" ] && break
        _NVIDIA_LIBRARY_INVENTORY_VALUE=""
    done
    [ -n "$_NVIDIA_LIBRARY_INVENTORY_VALUE" ] || return 1
    _NVIDIA_LIBRARY_INVENTORY_STATE="found"
    printf '%s\n' "$_NVIDIA_LIBRARY_INVENTORY_VALUE"
}

_nvidia_driver_cuda_version() {
    [ "${UNSLOTH_NVIDIA_LIBRARY_PROBE:-1}" != "0" ] || return 1
    if command -v python3 >/dev/null 2>&1; then _ndv_py=python3
    elif [ -n "${VENV_DIR:-}" ] && [ -x "$VENV_DIR/bin/python" ]; then _ndv_py="$VENV_DIR/bin/python"
    else _ndv_py=""
    fi
    if [ -n "$_ndv_py" ]; then
        _ndv_ver=$(_run_bounded "$_ndv_py" -I -c '
import ctypes, sys
for name in ("libcuda.so.1", "libcuda.so"):
    try:
        lib = ctypes.CDLL(name)
        break
    except OSError:
        lib = None
v = ctypes.c_int()
if lib is None or lib.cuDriverGetVersion(ctypes.byref(v)) != 0 or v.value < 1000:
    sys.exit(1)
print(f"{v.value // 1000}.{v.value % 1000 // 10}")
' 2>/dev/null </dev/null | awk 'NR == 1 { print $1 }') || _ndv_ver=""
        case "$_ndv_ver" in
            [0-9]*.[0-9]*) printf '%s\n' "$_ndv_ver"; return 0 ;;
        esac
    fi
    _ndv_drv=$(awk 'NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+\.[0-9]+(\.[0-9]+)?$/) { split($i, v, "."); print v[1]; exit } }' \
        /proc/driver/nvidia/version 2>/dev/null) || _ndv_drv=""
    case "$_ndv_drv" in ''|*[!0-9]*) return 1 ;; esac
    if [ "$_ndv_drv" -ge 580 ]; then echo "13.0"
    elif [ "$_ndv_drv" -ge 570 ]; then echo "12.8"
    elif [ "$_ndv_drv" -ge 560 ]; then echo "12.6"
    elif [ "$_ndv_drv" -ge 525 ]; then echo "12.0"
    elif [ "$_ndv_drv" -ge 450 ]; then echo "11.0"
    else return 1
    fi
}

_has_usable_nvidia_gpu() {
    _nv_smi_wedged=""
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
        # Captured rather than piped so `timeout`'s own 124 stays visible. A wedged
        # driver still detects as a GPU through /proc below, and the banner must not
        # then pay the 10s bound a second time asking a hung nvidia-smi for a name.
        _nv_l_rc=0
        _nv_l_out=$(_run_bounded "$_nvsmi" -L 2>/dev/null) || _nv_l_rc=$?
        if [ "$_nv_l_rc" = "124" ]; then
            _nv_smi_wedged=1
        fi
        if printf '%s\n' "$_nv_l_out" | awk '/^GPU[[:space:]]+[0-9]+:/{found=1} END{exit !found}'; then
            return 0
        fi
    fi
    if [ -d /proc/driver/nvidia/gpus ] && \
       [ -n "$(ls -A /proc/driver/nvidia/gpus 2>/dev/null)" ]; then
        return 0
    fi
    _nvidia_library_inventory >/dev/null 2>&1 && return 0
    return 1
}

# Row index of the GPU whose UUID starts with $1, or empty. NVIDIA allows a UUID to
# be abbreviated to any unique leading portion, hence the prefix match.
_nv_idx_from_uuid() {
    # NVIDIA accepts an abbreviation only when it is a UNIQUE leading portion, so a
    # prefix matching two cards selects NO device. Counting instead of stopping at the
    # first hit keeps the banner from naming one of them.
    _run_bounded "$_nvsmi" --query-gpu=uuid --format=csv,noheader 2>/dev/null \
        | awk -v want="$1" '
            NF { gsub(/^[[:space:]]+|[[:space:]]+$/,""); if (index($0, want) == 1) { hits++; idx = NR-1 } }
            END { if (hits == 1) print idx }' || true
}

# Bounded, and reuses the resolved $_nvsmi: detection may have succeeded without PATH nvidia-smi.
_nv_banner_fields() {
    _nv_name=""; _nv_sm=""; _nv_driver=""; _nv_row=""; _nv_cc=""; _nv_ambiguous=""
    [ -n "${_nvsmi:-}" ] || return 0
    # Detection already waited out the full bound on this binary. Asking again cannot
    # succeed and would double the stall, so the banner keeps the vendor-only wording.
    [ -z "${_nv_smi_wedged:-}" ] || return 0
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so its rows are the physical devices and
    # the mask has to be resolved against them by hand.
    _nv_idx=0
    _nv_by_ordinal=1
    _nv_vis="${CUDA_VISIBLE_DEVICES:-}"
    # Only the FIRST entry selects the device, and only IT decides the form of the mask.
    # CUDA truncates enumeration at the first invalid index, so the documented `2,-1`
    # means "device 2, then stop"; classifying the whole string would see the `-1`, call
    # it non-numeric, and send a plain ordinal down the UUID path.
    _nv_tok="${_nv_vis%%,*}"
    case "$_nv_tok" in
        '') ;;
        *[!0-9]*)
            _nv_by_ordinal=""
            case "$_nv_tok" in
                MIG-GPU-*)
                    # Pre-R470 MIG name, MIG-<GPU-UUID>/<gi>/<ci>: the parent UUID is
                    # embedded, so it still matches a --query-gpu=uuid row.
                    _nv_tok="${_nv_tok#MIG-}"; _nv_tok="${_nv_tok%%/*}"
                    _nv_idx=$(_nv_idx_from_uuid "$_nv_tok") ;;
                MIG-*)
                    # R470+ MIG UUIDs are opaque; nvidia-smi -L nests them under their GPU.
                    _nv_idx=$(_run_bounded "$_nvsmi" -L 2>/dev/null | awk -v want="$_nv_tok" '
                        /^GPU[[:space:]]+[0-9]+:/ { cur = $2 + 0 }
                        index($0, want) > 0 { print cur; exit }' || true) ;;
                *)
                    _nv_idx=$(_nv_idx_from_uuid "$_nv_tok") ;;
            esac
            # An identity mask that does not resolve means CUDA selected NO device: an
            # abbreviation short enough to match two cards, or a UUID for a card that is
            # not here. Row 0 is not a fallback for that -- it is a different card.
            case "$_nv_idx" in ''|*[!0-9]*) _nv_idx=0; _nv_ambiguous=1 ;; esac
            ;;
        *) _nv_idx="$_nv_tok" ;;
    esac
    # Canonicalise the ordinal before anything compares or subscripts with it.
    # `[` parses with strtol and ERRORS on a value wider than a long, printing a
    # shell diagnostic and skipping the range check below, so an absurd ordinal
    # would have named row 0. Clamped rather than rejected: 9999 is past any real
    # host, so it stays out of range and declines, which is the right answer.
    _nv_idx=$(printf '%s' "$_nv_idx" \
        | awk '{ n = $0 + 0; if (n < 0) n = 0; if (n > 9999) n = 9999; printf "%d", n }')
    _nv_all=$(_run_bounded "$_nvsmi" --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || true)
    _nv_row=$(printf '%s\n' "$_nv_all" \
        | awk -v idx="$_nv_idx" 'NF { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx+0] }')
    [ -n "$_nv_row" ] || return 0
    # CUDA's default FASTEST_FIRST order differs from nvidia-smi's PCI order, so an ordinal names a row
    # only under PCI_BUS_ID or identical cards. An out-of-range ordinal exposes no device.
    _nv_rowcount=$(printf '%s\n' "$_nv_all" | awk 'NF { n++ } END { print n+0 }')
    if [ -n "$_nv_by_ordinal" ] && [ "$_nv_idx" -ge "$_nv_rowcount" ]; then
        _nv_ambiguous=1
    fi
    if [ -n "$_nv_by_ordinal" ]; then
        _nv_order=$(printf '%s' "${CUDA_DEVICE_ORDER:-}" | tr '[:lower:]' '[:upper:]' | tr -d '[:space:]')
        _nv_models=$(printf '%s\n' "$_nv_all" \
            | awk -F, 'NF { k=$0; sub(/,[^,]*$/,"",k); if (!(k in s)) { s[k]; n++ } } END { print n+0 }')
        if [ "$_nv_order" != "PCI_BUS_ID" ] && [ "$_nv_models" -gt 1 ]; then
            _nv_ambiguous=1
        fi
    fi
    # Split from the right: nvidia-smi does not quote, so a comma in a device name
    # would otherwise shift every field.
    _nv_driver=$(printf '%s' "$_nv_row" | awk -F, 'NF>=3 { gsub(/^[[:space:]]+|[[:space:]]+$/,"",$NF); print $NF }')
    _nv_cc=$(printf '%s' "$_nv_row" | awk -F, 'NF>=3 { gsub(/^[[:space:]]+|[[:space:]]+$/,"",$(NF-1)); print $(NF-1) }')
    _nv_name=$(printf '%s' "$_nv_row" | awk -F, 'NF>=3 { out=$1; for(i=2;i<=NF-2;i++) out=out","$i; gsub(/^[[:space:]]+|[[:space:]]+$/,"",out); print out }')
    [ -n "$_nv_name" ] || _nv_name=$(printf '%s' "$_nv_row" | awk -F, '{ gsub(/^[[:space:]]+|[[:space:]]+$/,"",$1); print $1 }')
    # Old nvidia-smi prints placeholders for unknown fields (470 has no compute_cap).
    case "$_nv_name"   in '[N/A]'|'[Not Supported]'|'[Unknown Error]') _nv_name="" ;; esac
    case "$_nv_driver" in '[N/A]'|'[Not Supported]'|'[Unknown Error]') _nv_driver="" ;; esac
    if [ -n "$_nv_ambiguous" ]; then _nv_name=""; _nv_cc=""; fi
    case "$_nv_cc" in
        [0-9]*.[0-9]*) _nv_sm="sm_$(printf '%s' "$_nv_cc" | awk -F. '{ print ($1*10)+$2 }')" ;;
    esac
    return 0
}

# Strix Halo ROCm-on-WSL needs Ubuntu 24.04: re-run in one, else CPU. Never create a distro.
_maybe_reroute_strixhalo_to_2404() {
    [ "${OS:-}" = "wsl" ] || return 0
    _rr_pin=$(printf '%s' "${UNSLOTH_TORCH_INDEX_URL:-}${UNSLOTH_TORCH_INDEX_FAMILY:-}" | tr -d '[:space:]')
    [ -n "$_rr_pin" ] && return 0
    [ "${SKIP_TORCH:-false}" = "false" ] || return 0
    [ "${UNSLOTH_SKIP_ROCM_WSL_SETUP:-0}" = "1" ] && return 0
    [ "${UNSLOTH_WSL_REROUTED:-0}" = "1" ] && return 0
    [ -e /dev/dxg ] || return 0
    if _has_usable_nvidia_gpu; then return 0; fi
    if ! grep -qiE 'Ryzen AI Max|Radeon 80[0-9][05]S|Strix Halo' /proc/cpuinfo 2>/dev/null \
       && ! _wsl_amd_gpu_name >/dev/null 2>&1; then
        return 0
    fi
    if [ -e /opt/rocm/lib/librocdxg.so ] || [ -e /opt/rocm/lib64/librocdxg.so ]; then
        return 0
    fi
    _rr_ver=""
    [ -r /etc/os-release ] && _rr_ver=$(. /etc/os-release 2>/dev/null; printf '%s' "${VERSION_ID:-}")
    case "$_rr_ver" in 24.04) return 0 ;; esac
    command -v wsl.exe >/dev/null 2>&1 || { UNSLOTH_SKIP_ROCM_WSL_SETUP=1; return 0; }
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
    # A --local checkout cannot be replayed in another distro; tell the user instead.
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        substep "This is a --local install; re-run it from $_rr_target instead:" "$C_WARN"
        substep "  wsl -d $_rr_target -- bash -lc 'cd <your checkout> && ./install.sh --local'" "$C_WARN"
        substep "Continuing CPU-only in Ubuntu ${_rr_ver:-this distro} for now." "$C_WARN"
        UNSLOTH_SKIP_ROCM_WSL_SETUP=1
        return 0
    fi
    _rr_q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
    _rr_exports="set -o pipefail; export UNSLOTH_WSL_REROUTED=1"

    # Forward only a user override of UV_CACHE_DIR; our own default would pin the child to `custom`.
    _rr_uv_cache=""
    if [ "${_UV_CACHE_DIR_INSTALLER_DEFAULT:-false}" != true ]; then
        case "${UV_CACHE_DIR-}" in
            *[![:space:]]*) _rr_uv_cache="$UV_CACHE_DIR" ;;
        esac
    fi
    if [ -n "$_rr_uv_cache" ]; then
        _rr_exports="$_rr_exports; export UV_CACHE_DIR=$(_rr_q "$_rr_uv_cache")"
    else
        _rr_exports="$_rr_exports; unset UV_CACHE_DIR"
    fi
    unset _rr_uv_cache
    [ "$_ISOLATE_UV_CACHE" = true ] && _rr_exports="$_rr_exports; export UNSLOTH_ISOLATE_UV_CACHE=1"
    [ "$_STUDIO_HOME_REDIRECT" = "env" ] && _rr_exports="$_rr_exports; export UNSLOTH_STUDIO_HOME=$(_rr_q "$STUDIO_HOME")"
    [ "${UNSLOTH_ROCM_WSL_AUTO:-0}" = "1" ] && _rr_exports="$_rr_exports; export UNSLOTH_ROCM_WSL_AUTO=1"
    [ -n "${UNSLOTH_TORCH_INDEX_URL:-}" ] && _rr_exports="$_rr_exports; export UNSLOTH_TORCH_INDEX_URL=$(_rr_q "$UNSLOTH_TORCH_INDEX_URL")"
    [ -n "${UNSLOTH_TORCH_INDEX_FAMILY:-}" ] && _rr_exports="$_rr_exports; export UNSLOTH_TORCH_INDEX_FAMILY=$(_rr_q "$UNSLOTH_TORCH_INDEX_FAMILY")"
    [ -n "${UNSLOTH_MIRROR_FALLBACK:-}" ] && _rr_exports="$_rr_exports; export UNSLOTH_MIRROR_FALLBACK=$(_rr_q "$UNSLOTH_MIRROR_FALLBACK")"
    [ "$_SKIP_AUTOSTART" = true ] && _rr_exports="$_rr_exports; export UNSLOTH_SKIP_AUTOSTART=1"
    if [ "$_INSTALL_SYSTEMD" = true ]; then
        _rr_exports="$_rr_exports; export UNSLOTH_INSTALL_SYSTEMD=1"
        [ -n "${UNSLOTH_SYSTEMD_HOST:-}" ] && _rr_exports="$_rr_exports; export UNSLOTH_SYSTEMD_HOST=$(_rr_q "$UNSLOTH_SYSTEMD_HOST")"
        [ -n "${UNSLOTH_SYSTEMD_PORT:-}" ] && _rr_exports="$_rr_exports; export UNSLOTH_SYSTEMD_PORT=$(_rr_q "$UNSLOTH_SYSTEMD_PORT")"
    fi
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
    # pipefail so a failed download is not masked by sh exiting 0 on empty input.
    _rr_rc=0
    wsl.exe -d "$_rr_target" -- bash -lc "$_rr_exports; $_rr_cmd" || _rr_rc=$?
    if [ "$_rr_rc" -eq 0 ]; then
        exit 0
    fi
    if [ "$TAURI_MODE" = true ] && [ "$_rr_rc" -eq 2 ]; then
        exit 2
    fi
    substep "Could not auto-continue in $_rr_target; run it yourself:" "$C_WARN"
    substep "  wsl -d $_rr_target -- bash -lc 'curl -fsSL https://unsloth.ai/install.sh | sh'"
    substep "Continuing CPU-only in Ubuntu ${_rr_ver:-this distro} for now." "$C_WARN"
    UNSLOTH_SKIP_ROCM_WSL_SETUP=1
    return 0
}
_maybe_reroute_strixhalo_to_2404 || true

tauri_log "STEP" "Checking system dependencies"

# Without the Xcode CLT /usr/bin/git is a stub, so `command -v git` is not enough.
_has_working_git() {
    command -v git >/dev/null 2>&1 || return 1
    # On Apple Silicon, running the /usr/bin/git CLT shim pops a GUI dialog, so answer from the path.
    # Intel Macs may have a working /usr/bin/git and still probe it. _CLT_GIT_SHIM is a test hook.
    if [ "${OS:-}" = "macos" ] &&
       { [ "${MAC_INTEL:-false}" != true ] || [ "${_MAC_ROSETTA:-false}" = true ]; } &&
       [ "$(command -v git)" = "${_CLT_GIT_SHIM:-/usr/bin/git}" ] &&
       ! xcode-select -p >/dev/null 2>&1; then
        return 1
    fi
    git --version >/dev/null 2>&1
}

# A function so tests/sh can sed-extract it. Consumer installs need no toolchain; only --local needs git.
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
        step "deps" "no Xcode Command Line Tools (not required)" "$C_WARN"
        substep "Unsloth installs prebuilt binaries and wheels, so no compiler is needed."
        substep "Install them only for a llama.cpp source build: xcode-select --install"
    elif command -v cmake >/dev/null 2>&1; then
        step "deps" "all system dependencies found"
    else
        step "deps" "using prebuilt llama.cpp (cmake not found)" "$C_WARN"
        substep "Install cmake only if you want a source build: brew install cmake"
    fi
    return 0
}

# Only a download transport is required: llama.cpp ships prebuilts, so build tools are optional.
_check_linux_deps() {
    _transport_missing=false
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        _transport_missing=true
    fi

    # Optional: git for triton_kernels, the rest for a source build. Warn, never stop.
    _optional_missing=""
    command -v cmake       >/dev/null 2>&1 || _optional_missing="$_optional_missing cmake"
    _has_working_git                       || _optional_missing="$_optional_missing git"
    command -v gcc         >/dev/null 2>&1 || _optional_missing="$_optional_missing build-essential"
    command -v curl-config >/dev/null 2>&1 || _optional_missing="$_optional_missing libcurl4-openssl-dev"
    _optional_missing="${_optional_missing# }"

    if [ "$STUDIO_LOCAL_INSTALL" = true ] && ! _has_working_git; then
        echo ""
        step "deps" "git is required for --local installs" "$C_ERR"
        substep "--local installs unsloth-zoo from git+https://github.com/unslothai/unsloth-zoo,"
        substep "which needs git. Install it with your package manager, then re-run."
        substep "A normal (non---local) install needs no git and no compiler."
        return 1
    fi

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

    if [ -n "$_optional_missing" ] && command -v apt-get >/dev/null 2>&1; then
        step "deps" "installing optional build tools: $_optional_missing" "$C_DIM"
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

# Keep in step with os_sandbox._BWRAP_APPARMOR_FIX.
_BWRAP_APPARMOR_FIX="sudo apt-get install -y apparmor-profiles && sudo install -m 644 /usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/ && sudo apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict"

# The command that installs bubblewrap here; keep in step with os_sandbox._BWRAP_INSTALL_COMMANDS.
_bwrap_install_command() {
    if command -v apt-get >/dev/null 2>&1; then echo "sudo apt-get install -y bubblewrap"
    elif command -v dnf >/dev/null 2>&1; then echo "sudo dnf install -y bubblewrap"
    elif command -v pacman >/dev/null 2>&1; then echo "sudo pacman -S --needed bubblewrap"
    elif command -v zypper >/dev/null 2>&1; then echo "sudo zypper install -y bubblewrap"
    elif command -v apk >/dev/null 2>&1; then echo "sudo apk add bubblewrap"
    fi
}

# Optional: without bubblewrap, tool calls run with software safeguards. Never elevates.
_check_linux_tool_sandbox() {
    _bw_restrict=""
    read -r _bw_restrict <"${_BW_USERNS_SYSCTL:-/proc/sys/kernel/apparmor_restrict_unprivileged_userns}" 2>/dev/null || true
    if ! command -v bwrap >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
        ( _SMART_APT_OPTIONAL=true; _smart_apt_install bubblewrap ) || true
    fi
    if ! command -v bwrap >/dev/null 2>&1; then
        step "sandbox" "bubblewrap not installed: tool calls run with software safeguards" "$C_WARN"
        _bw_cmd="$(_bwrap_install_command)"
        # Ubuntu 23.10+ blocks bwrap via AppArmor even when installed.
        case "$_bw_restrict:$_bw_cmd" in
            1:*apt-get*) _bw_cmd="$_bw_cmd && $_BWRAP_APPARMOR_FIX" ;;
        esac
        if [ -n "$_bw_cmd" ]; then
            substep "To run them in an OS sandbox: $_bw_cmd"
        else
            substep "To run them in an OS sandbox, install bubblewrap with your package manager."
        fi
        return 0
    fi
    if bwrap --unshare-user --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup --ro-bind / / true </dev/null >/dev/null 2>&1; then
        step "sandbox" "bubblewrap works: tool calls run in an OS sandbox"
        return 0
    fi
    if [ "$_bw_restrict" = 1 ]; then
        step "sandbox" "AppArmor blocks bubblewrap: tool calls run with software safeguards" "$C_WARN"
        substep "To enable it, load Ubuntu's own bwrap profile:"
        substep "  $_BWRAP_APPARMOR_FIX"
    else
        step "sandbox" "bubblewrap cannot create a sandbox here: tool calls run with software safeguards" "$C_WARN"
        substep "Containers usually block user namespaces; outside one, check user.max_user_namespaces."
    fi
    return 0
}

case "$OS" in
    macos)
        _check_macos_deps || exit 1
        ;;
    linux|wsl)
        _check_linux_deps || exit 1
        _check_linux_tool_sandbox || true
        ;;
esac

# ── BEGIN mirror fallback (kept identical in install.sh and studio/setup.sh) ──
# Only in mainland China (or UNSLOTH_MIRROR_FALLBACK=1): swaps a default host below 1 MiB/s or unreachable for its mirror when faster; user-set sources untouched; UNSLOTH_MIRROR_FALLBACK=0 disables.
_MIRROR_CERNET="https://tuna.mirrors.cernet.edu.cn"
_MIRROR_PYPI="$_MIRROR_CERNET/pypi/web/simple"
_MIRROR_NPM="https://registry.npmmirror.com"
# GitHub-release mirrors keep only the newest Python builds, while a pinned uv asks for the builds it shipped with; npmmirror keeps every release.
_MIRROR_PYTHON="$_MIRROR_NPM/-/binary/python-build-standalone"
_MIRROR_MIN_BPS=1048576

_mirror_probe() {
    _mp_out=$(curl -sL -o /dev/null -r "0-$3" -w '%{http_code} %{speed_download}' --connect-timeout "$2" --max-time "$2" "$1" 2>/dev/null) || true
    _mp_bps=${_mp_out#* }
    _mp_bps=${_mp_bps%%.*}
    case "$_mp_bps" in ''|*[!0-9]*) _mp_bps=0 ;; esac
    _mp_code=${_mp_out%% *}
    case "$_mp_code" in [0-9][0-9][0-9]) ;; *) _mp_code=000 ;; esac
    echo "$_mp_code $_mp_bps"
}

_mirror_url() {
    case "$1" in
        pypi) echo "https://files.pythonhosted.org/packages/72/d6/207945fe69903b9794e2ef3e42608c91a59972567343a6719078d99c71f7/uv-0.12.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" ;;
        cernet-pypi) echo "$_MIRROR_CERNET/pypi/web/packages/72/d6/207945fe69903b9794e2ef3e42608c91a59972567343a6719078d99c71f7/uv-0.12.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" ;;
        torch) echo "https://download-r2.pytorch.org/whl/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" ;;
        cernet-torch) echo "$_MIRROR_CERNET/pytorch/whl/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" ;;
        node) echo "https://nodejs.org/dist/v24.18.0/node-v24.18.0-linux-x64.tar.gz" ;;
        npmmirror-node) echo "$_MIRROR_NPM/-/binary/node/v24.18.0/node-v24.18.0-linux-x64.tar.gz" ;;
        npm) echo "https://registry.npmjs.org/typescript/-/typescript-5.9.3.tgz" ;;
        npmmirror) echo "$_MIRROR_NPM/typescript/-/typescript-5.9.3.tgz" ;;
        astral) echo "https://releases.astral.sh/github/uv/releases/download/0.12.1/uv-x86_64-unknown-linux-gnu.tar.gz" ;;
        pypi-index) echo "https://pypi.org/simple/uv/" ;;
        torch-index) echo "https://download.pytorch.org/whl/cpu/torch/" ;;
        cernet-pypi-index) echo "$_MIRROR_PYPI/uv/" ;;
        cernet-torch-index) echo "$_MIRROR_CERNET/pytorch/whl/cpu/torch/" ;;
    esac
}

_mirror_default() {
    case "$1" in python|uvbin) echo astral ;; *) echo "$1" ;; esac
}

_mirror_source() {
    case "$1" in npm|python) echo npmmirror ;; node) echo npmmirror-node ;; pypi|uvbin) echo cernet-pypi ;; *) echo "cernet-$1" ;; esac
}

_mirror_index_probe() {
    for _mip_name in "$@"; do
        case "$_mip_name" in
            pypi|torch|cernet-pypi|cernet-torch)
                _mirror_probe "$(_mirror_url "$_mip_name-index")" 4 1023 > "$_mf_dir/$_mip_name-index" &
                _mf_pids="$_mf_pids $!" ;;
        esac
    done
}

_mirror_index_wait() {
    for _miw_pid in $_mf_pids; do
        wait "$_miw_pid" || true
    done
    _mf_pids=""
}

_mirror_index_ok() {
    [ -f "$_mf_dir/$1-index" ] || return 0
    read -r _mio_code _mio_bps < "$_mf_dir/$1-index"
    case "$_mio_code" in 2??) return 0 ;; *) return 1 ;; esac
}

_mirror_uv_project_config() {
    _mup_dir=$PWD
    while [ -n "$_mup_dir" ]; do
        if [ -f "$_mup_dir/uv.toml" ]; then echo "$_mup_dir/uv.toml"; return 0; fi
        if grep -Eqs '^[[:space:]]*\[+tool\.uv(\.|\])' "$_mup_dir/pyproject.toml"; then echo "$_mup_dir/pyproject.toml"; return 0; fi
        [ "$_mup_dir" = / ] && return 0
        _mup_dir=$(dirname "$_mup_dir")
    done
}

_mirror_configured() {
    case "$1" in
        uv)
            [ -n "${UV_DEFAULT_INDEX:-}${UV_INDEX_URL:-}${UV_INDEX:-}${UV_EXTRA_INDEX_URL:-}" ] && return 0
            _mic_key='\[\[(tool\.uv\.)?index\]\]|(pip\.)?(index|index-url|default-index|extra-index-url|no-index)[[:space:]]*=' ;;
        python)
            [ -n "${UV_PYTHON_INSTALL_MIRROR:-}" ] && return 0
            _mic_key='python-install-mirror[[:space:]]*=' ;;
        pip)
            [ -n "${PIP_INDEX_URL:-}${PIP_EXTRA_INDEX_URL:-}${PIP_NO_INDEX:-}" ] && return 0
            _mic_key='(index[-_]url|extra[-_]index[-_]url|no[-_]index)[[:space:]]*[=:]' ;;
    esac
    _mic_suffix=uv/uv.toml
    [ "$1" != pip ] || _mic_suffix=pip/pip.conf
    if [ "$1" = pip ]; then
        set -- "${PIP_CONFIG_FILE:-}" "${VENV_DIR:+$VENV_DIR/pip.conf}" "${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf" "$HOME/.pip/pip.conf" "$HOME/Library/Application Support/pip/pip.conf" /etc/xdg/pip/pip.conf /etc/pip.conf
    else
        set -- "${UV_CONFIG_FILE:-}" "$(_mirror_uv_project_config)" "${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml" /etc/xdg/uv/uv.toml /etc/uv/uv.toml
    fi
    _mic_xdg=${XDG_CONFIG_DIRS:-}
    while [ -n "$_mic_xdg" ]; do
        set -- "$@" "${_mic_xdg%%:*}/$_mic_suffix"
        case "$_mic_xdg" in *:*) _mic_xdg=${_mic_xdg#*:} ;; *) _mic_xdg="" ;; esac
    done
    for _mic_file in "$@"; do
        if [ -f "$_mic_file" ] && grep -Eq "^[[:space:]]*($_mic_key)" "$_mic_file" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

_mirror_probe_all() {
    _mpa_dir="$1"
    _mpa_secs="$2"
    shift 2
    _mpa_pids=""
    for _mpa_name in "$@"; do
        _mirror_probe "$(_mirror_url "$_mpa_name")" "$_mpa_secs" 1048575 > "$_mpa_dir/$_mpa_name" &
        _mpa_pids="$_mpa_pids $!"
    done
    for _mpa_pid in $_mpa_pids; do
        wait "$_mpa_pid" || true
    done
}

_mirror_vars() {
    case "$1" in
        pypi)
            [ "$_mf_uv" = false ] || echo "UV_DEFAULT_INDEX=$_MIRROR_PYPI"
            [ "$_mf_pip" = false ] || echo "PIP_INDEX_URL=$_MIRROR_PYPI" ;;
        unsynced)
            # Only for one rerun: uv's unsafe-first-match fetches every package from every index, and fails outright when one is unreachable.
            [ "$_mf_uv" = false ] || echo "UV_DEFAULT_INDEX=https://pypi.org/simple UV_INDEX=$_MIRROR_PYPI UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY:-unsafe-first-match}"
            [ "$_mf_pip" = false ] || echo "PIP_EXTRA_INDEX_URL=https://pypi.org/simple PIP_INDEX_URL=$_MIRROR_PYPI" ;;
        torch) echo "UNSLOTH_PYTORCH_MIRROR=$_MIRROR_CERNET/pytorch/whl" ;;
        node) echo "UNSLOTH_NODE_MIRROR=$_MIRROR_NPM/-/binary/node" ;;
        npm) echo "UNSLOTH_NPM_REGISTRY=$_MIRROR_NPM" ;;
        python) echo "UV_PYTHON_INSTALL_MIRROR=$_MIRROR_PYTHON" ;;
        uvbin) echo "UNSLOTH_UV_WHEEL_MIRROR=$_MIRROR_CERNET/pypi/web" ;;
    esac
}

_mirror_name() {
    case "$1" in
        pypi) echo "PyPI" ;;
        unsynced) echo "The PyPI mirror" ;;
        torch) echo "download.pytorch.org" ;;
        node) echo "nodejs.org" ;;
        npm) echo "registry.npmjs.org" ;;
        python) echo "releases.astral.sh (Python builds)" ;;
        uvbin) echo "releases.astral.sh (uv)" ;;
    esac
}

_mirror_use() {
    _mu_to=""
    for _mu_pair in $(_mirror_vars "$1"); do
        export "$_mu_pair"
        [ -n "$_mu_to" ] || _mu_to=${_mu_pair#*=}
    done
    step "mirror" "$(_mirror_name "$1") is $2 ($(($3 / 1024)) KB/s, mirror $(($4 / 1024)) KB/s); using $_mu_to" "$C_WARN"
    _mf_switched="$_mf_switched $1"
    [ "$1 $2" != "pypi slow" ] || _mf_unsynced=true
}

_mirror_take() {
    _MT_PAIRS=""
    _mt_spare=""
    for _mt_entry in ${_UNSLOTH_MIRROR_SPARE:-}; do
        if [ "${_mt_entry%%|*}" = "$1" ]; then
            _MT_PAIRS=$(printf '%s' "${_mt_entry#*|}" | tr '|' ' ')
        else
            _mt_spare="$_mt_spare $_mt_entry"
        fi
    done
    [ -n "$_MT_PAIRS" ] || return 1
    export _UNSLOTH_MIRROR_SPARE="${_mt_spare# }"
    _mt_to=${_MT_PAIRS%% *}
    step "mirror" "$(_mirror_name "$1") failed; retrying through ${_mt_to#*=}" "$C_WARN"
}

_mirror_switch() {
    _mirror_take "$1" || return 1
    for _ms_pair in $_MT_PAIRS; do export "$_ms_pair"; done
}

_mirror_failed_host() {
    if ! grep -Eqi 'error sending request|timed out|network timeout|idle timeout|connection (reset|refused|closed|aborted)|network aborted|broken pipe|dns error|failed to lookup address|name resolution|nodename nor servname|network is unreachable|error decoding response body|end of file before message length|unexpected eof|tls handshake|sslerror|certificate verify failed|server error|service unavailable|bad gateway|gateway time-?out|too many requests|max retries exceeded|remotedisconnected|incompleteread|econnreset|etimedout|eidletimeout|eai_again|enotfound|econnrefused|socket hang up' "$1" 2>/dev/null; then
        grep -Eqi 'only [^ ]+ (.* )?(is|are) available|no versions? of|not found in the package registry|could not find a version that satisfies|no matching distribution found' "$1" 2>/dev/null || return 1
        echo unsynced
        return 0
    fi
    if grep -Eq 'download(-r2)?\.pytorch\.org' "$1"; then echo torch
    elif grep -q 'python-build-standalone' "$1"; then echo python
    elif grep -q 'registry\.npmjs\.org' "$1"; then echo npm
    elif grep -Eq 'pypi\.org|pythonhosted\.org' "$1"; then echo pypi
    elif [ -n "${2:-}" ] && ! grep -Eq 'https?://' "$1"; then echo "$2"
    else return 1
    fi
}

# No network call: a mainland China time zone, or a resolver from a mainland public DNS or cloud (the addresses below).
_mirror_in_china() {
    _mcn_tz=${TZ:-}
    [ -n "$_mcn_tz" ] || _mcn_tz=$(cat /etc/timezone 2>/dev/null) || true
    [ -n "$_mcn_tz" ] || _mcn_tz=$(readlink /etc/localtime 2>/dev/null) || true
    case "${_mcn_tz#:}" in
        *Asia/Shanghai|*Asia/Chongqing|*Asia/Chungking|*Asia/Harbin|*Asia/Urumqi|*Asia/Kashgar|PRC|*/PRC) return 0 ;;
    esac
    grep -Eqs '^[[:space:]]*nameserver[[:space:]]+(223\.5\.5\.5|223\.6\.6\.6|119\.29\.29\.29|114\.114\.11[45]\.11[0459]|182\.254\.116\.116|119\.28\.28\.28|180\.76\.76\.76|1\.2\.4\.8|210\.2\.4\.8|100\.100\.2\.13[68]|183\.60\.8[23]\.(19|98))[[:space:]]*$' /etc/resolv.conf /run/systemd/resolve/resolv.conf
}

# Decided once per process; when off, a retry state inherited from a parent is dropped so nothing downstream acts on it.
_mirror_enabled() {
    if [ -z "${_mirror_on:-}" ]; then
        case "${UNSLOTH_MIRROR_FALLBACK:-}" in
            0|false|False|FALSE|no|off) _mirror_on=no ;;
            1|true|True|TRUE|yes|on) _mirror_on=yes ;;
            *) if _mirror_in_china; then _mirror_on=yes; else _mirror_on=no; fi ;;
        esac
        [ "$_mirror_on" = yes ] || unset _UNSLOTH_MIRROR_SPARE
    fi
    [ "$_mirror_on" = yes ]
}

_mirror_fallback() {
    _mirror_enabled || return 0
    [ -z "${_UNSLOTH_MIRROR_PROBED:-}" ] || return 0
    command -v curl >/dev/null 2>&1 || return 0
    [ "${1:-}" = spare ] || export _UNSLOTH_MIRROR_PROBED=1
    _mf_uv=true
    _mf_pip=true
    _mf_switched=""
    _mf_unsynced=false
    _mirror_configured uv && _mf_uv=false
    _mirror_configured pip && _mf_pip=false
    _mf_hosts=""
    if [ "$_mf_uv" = true ] || [ "$_mf_pip" = true ]; then
        _mf_hosts="pypi"
    fi
    [ -n "${UNSLOTH_PYTORCH_MIRROR:-}${UNSLOTH_TORCH_INDEX_URL:-}" ] || _mf_hosts="$_mf_hosts torch"
    [ -n "${UNSLOTH_NODE_MIRROR:-}" ] || _mf_hosts="$_mf_hosts node"
    [ -n "${UNSLOTH_NPM_REGISTRY:-}${NPM_CONFIG_REGISTRY:-}${npm_config_registry:-}" ] || _mf_hosts="$_mf_hosts npm"
    _mirror_configured python || _mf_hosts="$_mf_hosts python"
    [ -n "${UNSLOTH_UV_WHEEL_MIRROR:-}${UV_DOWNLOAD_URL:-}${INSTALLER_DOWNLOAD_URL:-}${UV_INSTALLER_GHE_BASE_URL:-}${UV_INSTALLER_GITHUB_BASE_URL:-}" ] || _mf_hosts="$_mf_hosts uvbin"
    [ -n "$_mf_hosts" ] || return 0
    # UV_OFFLINE (uv's spellings) asked for no network: arm the retries, probe nothing.
    _mf_uvo=${UV_OFFLINE:-}
    _mf_uvo=${_mf_uvo#"${_mf_uvo%%[![:space:]]*}"}
    _mf_uvo=${_mf_uvo%"${_mf_uvo##*[![:space:]]}"}
    case "${1:-}/$_mf_uvo" in
        spare/* | */1 | */[Tt] | */[Tt][Rr][Uu][Ee] | */[Yy] | */[Yy][Ee][Ss] | */[Oo][Nn]) _mirror_spare_export; return 0 ;;
    esac
    _mf_dir=$(mktemp -d 2>/dev/null) || return 0
    _mf_pids=""
    _mirror_index_probe $_mf_hosts
    for _mf_name in $(for _mf_host in $_mf_hosts; do _mirror_default "$_mf_host"; done | sort -u); do
        _mirror_probe "$(_mirror_url "$_mf_name")" 1.5 1048575 > "$_mf_dir/$_mf_name"
    done
    _mirror_index_wait
    _mf_slow=""
    for _mf_host in $_mf_hosts; do
        read -r _mf_code _mf_bps < "$_mf_dir/$(_mirror_default "$_mf_host")"
        _mirror_index_ok "$_mf_host" || _mf_code=000
        case "$_mf_code" in
            000|3??) _mf_slow="$_mf_slow $_mf_host" ;;
            2??) [ "$_mf_bps" -ge "$_MIRROR_MIN_BPS" ] || _mf_slow="$_mf_slow $_mf_host" ;;
        esac
    done
    if [ -n "$_mf_slow" ]; then
        _mirror_index_probe $_mf_slow $(for _mf_host in $_mf_slow; do echo "cernet-$_mf_host"; done)
        _mirror_probe_all "$_mf_dir" 4 $(for _mf_host in $_mf_slow; do _mirror_default "$_mf_host"; _mirror_source "$_mf_host"; done | sort -u)
        _mirror_index_wait
        for _mf_host in $_mf_slow; do
            read -r _mf_code _mf_bps < "$_mf_dir/$(_mirror_default "$_mf_host")"
            read -r _mf_mcode _mf_mbps < "$_mf_dir/$(_mirror_source "$_mf_host")"
            _mirror_index_ok "$_mf_host" || _mf_code=000
            _mirror_index_ok "cernet-$_mf_host" || _mf_mcode=000
            case "$_mf_code" in
                2??) _mf_how=slow ;;
                *) _mf_how=blocked; _mf_bps=0 ;;
            esac
            case "$_mf_mcode" in
                2??) [ "$_mf_bps" -lt "$_MIRROR_MIN_BPS" ] && [ "$_mf_mbps" -gt "$_mf_bps" ] && _mirror_use "$_mf_host" "$_mf_how" "$_mf_bps" "$_mf_mbps" ;;
            esac
        done
        if [ -n "$_mf_switched" ]; then
            substep "Set UNSLOTH_MIRROR_FALLBACK=0 to always use the default hosts."
        fi
    fi
    rm -rf "$_mf_dir"
    _mirror_spare_export
}

_mirror_spare_export() {
    _mf_spare=""
    for _mf_host in $_mf_hosts; do
        case " $_mf_switched " in *" $_mf_host "*) continue ;; esac
        _mf_entry=""
        for _mf_pair in $(_mirror_vars "$_mf_host"); do
            _mf_entry="$_mf_entry|$_mf_pair"
        done
        [ -z "$_mf_entry" ] || _mf_spare="$_mf_spare $_mf_host$_mf_entry"
    done
    if [ "$_mf_unsynced" = true ]; then
        _mf_spare="$_mf_spare unsynced"
        for _mf_pair in $(_mirror_vars unsynced); do
            _mf_spare="$_mf_spare|$_mf_pair"
        done
    fi
    export _UNSLOTH_MIRROR_SPARE="${_mf_spare# }"
}
# ── END mirror fallback ──

# ── Install uv ──
tauri_log "STEP" "Installing uv package manager"
# 0.9.3 is the first uv offering CPython 3.13.9; older ones resolve "3.13" to the broken 3.13.8.
UV_MIN_VERSION="0.9.3"
# Previous floor; offline hosts may keep a uv between the two.
UV_OFFLINE_MIN_VERSION="0.8.16"

: "${UV_COMPILE_BYTECODE_TIMEOUT:=180}"
export UV_COMPILE_BYTECODE_TIMEOUT

: "${UV_HTTP_RETRIES:=5}"
export UV_HTTP_RETRIES
: "${UV_HTTP_TIMEOUT:=180}"
export UV_HTTP_TIMEOUT

if [ "$OS" = "macos" ]; then
    : "${UV_SYSTEM_CERTS:=1}"
    : "${UV_NATIVE_TLS:=$UV_SYSTEM_CERTS}"
fi
[ -n "${UV_SYSTEM_CERTS:-}" ] && export UV_SYSTEM_CERTS
[ -n "${UV_NATIVE_TLS:-}" ] && export UV_NATIVE_TLS

_mirror_fallback

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

# 3.13.8: python/cpython#139783 breaks inspect.getsourcelines, so `import torch` fails.
PYTHON_SKIP="3.13.8"

# Skips exist only because torch cannot import; --no-torch installs ignore them.
_python_skip_applies() {
    [ "$SKIP_TORCH" != true ]
}

_python_is_skipped() {
    _python_skip_applies || return 1
    for _bad in $PYTHON_SKIP; do
        [ "$1" = "$_bad" ] && return 0
    done
    return 1
}

# Exclude skipped patches with a PEP 440 range ("!=3.13.8", not ">=3.13.9") so an offline
# host can still use a cached 3.13.7.
_python_request() {  # requested version -> what uv is asked for
    _python_skip_applies || { echo "$1"; return 0; }
    case "$1" in
        [0-9]*.[0-9]*.*|*/*|*\\*) echo "$1"; return 0 ;;
        [0-9]*.[0-9]*) ;;
        *) echo "$1"; return 0 ;;
    esac
    _req_minor=${1#*.}
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
# Mirrors Install-UvFromRelease in install.ps1; bumping the version means bumping every hash (<asset>.sha256) and every _uv_pinned_wheel entry.
UV_PINNED_VERSION="0.12.1"

# Prints the glibc minor or nothing; an unconfirmed host must fall back to astral's musl build.
_uv_glibc_minor() {
    _ugm_line=$( (ldd --version 2>/dev/null || true) | head -1 )
    case "$_ugm_line" in *[Mm]usl*) return 1 ;; esac
    _ugm_ver=$(printf '%s\n' "$_ugm_line" | awk '{print $NF}')
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

_uv_pinned_asset() {
    _upa_os=$(uname -s 2>/dev/null || echo unknown)
    _upa_arch=$(uname -m 2>/dev/null || echo unknown)
    case "$_upa_os" in
        Linux)
            [ "$(getconf LONG_BIT 2>/dev/null || echo 0)" = "64" ] || return 1
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
            # Under Rosetta, astral ships the native arm64 build; match it.
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

_uv_pinned_wheel() {
    case "$1" in
        uv-x86_64-unknown-linux-gnu.tar.gz)
            echo "packages/72/d6/207945fe69903b9794e2ef3e42608c91a59972567343a6719078d99c71f7/uv-0.12.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 27211df9b277f440dea438a4e525ba40250fb721ad39b8927eefc2d91f9aea15" ;;
        uv-aarch64-unknown-linux-gnu.tar.gz)
            echo "packages/9a/c7/29e426865c2eb8df61253dae93b953523f48c9fca1e471c7e49ff068f19a/uv-0.12.1-py3-none-manylinux_2_28_aarch64.whl b255ac23958e45f39f9c7a4cd65890df5ef46f539a3b14de03bd296bbba9cb60" ;;
        uv-x86_64-apple-darwin.tar.gz)
            echo "packages/fd/07/a417475380e901f4325d13b09938baab227b0c143547124b944c5bc71783/uv-0.12.1-py3-none-macosx_10_12_x86_64.whl 41b8fc2335f682312a1ca39a7b4abfd6af800992065c663582ca3e4d51cf9258" ;;
        uv-aarch64-apple-darwin.tar.gz)
            echo "packages/c9/68/391ff0cc3d8020e64adc43bb4e50607f744c69e792fb7623dc7c1526704b/uv-0.12.1-py3-none-macosx_11_0_arm64.whl 2e9b0b86e180abc5968b979c6e25203b32e85969abb5083ee1e8b88a5aa98a76" ;;
        *) return 1 ;;
    esac
}

_uv_unzip() {
    if command -v unzip >/dev/null 2>&1 && unzip -qo "$1" -d "$2" >/dev/null 2>&1; then return 0; fi
    case "$(tar --version 2>/dev/null)" in
        *bsdtar*) tar -xf "$1" -C "$2" 2>/dev/null && return 0 ;;
    esac
    command -v python3 >/dev/null 2>&1 &&
        python3 -m zipfile -e "$1" "$2" >/dev/null 2>&1
}

_uv_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
    fi
}

# Hang-proof liveness probe: no stdin and a ceiling (timeout -k, or a watchdog on macOS).
_uv_signal_target() {
    # bash reads a bare negative pid as a signal spec and dash refuses `--`; the sign picks the form.
    case "$2" in
        -*) kill "-$1" -- "$2" 2>/dev/null || : ;;
        *)  kill "-$1" "$2" 2>/dev/null || : ;;
    esac
}

# TERM, then KILL, as `timeout -k` does. $1 target, $2 pid to watch, $3 grace seconds.
_uv_probe_terminate() {
    _upt_grace=0
    _uv_signal_target TERM "$1"
    while [ "$_upt_grace" -lt "$3" ] && kill -0 "$2" 2>/dev/null; do
        sleep 1
        _upt_grace=$((_upt_grace + 1))
    done
    if kill -0 "$2" 2>/dev/null; then _uv_signal_target KILL "$1"; fi
    unset _upt_grace
}

_uv_probe_exec() {
    _upe_secs="${_UV_PROBE_SECONDS:-20}"
    if command -v timeout >/dev/null 2>&1 && timeout -k 1 5 true >/dev/null 2>&1; then
        timeout -k 5 "$_upe_secs" "$1" --version >/dev/null 2>&1 </dev/null
        return $?
    fi
    # Monitor mode gives the probe its own process group so signals reach its children.
    _upe_monitor=off
    case "$-" in *m*) _upe_monitor=on ;; esac
    [ "$_upe_monitor" = on ] || set -m 2>/dev/null || :
    "$1" --version >/dev/null 2>&1 </dev/null &
    _upe_pid=$!
    [ "$_upe_monitor" = on ] || set +m 2>/dev/null || :
    _upe_target="$_upe_pid"
    if command -v ps >/dev/null 2>&1; then
        _upe_pgid=$(ps -o pgid= -p "$_upe_pid" 2>/dev/null)
        _upe_self=$(ps -o pgid= -p $$ 2>/dev/null)
        _upe_pgid=${_upe_pgid##* }
        _upe_self=${_upe_self##* }
        case "$_upe_pgid$_upe_self" in
            ''|*[!0-9]*) : ;;
            *) [ "$_upe_pgid" = "$_upe_self" ] || _upe_target="-$_upe_pgid" ;;
        esac
    fi
    # Published so the HUP/INT/TERM handlers kill the probe instead of orphaning it.
    _UV_PROBE_TARGET="$_upe_target"
    _UV_PROBE_PID="$_upe_pid"
    _upe_waited=0
    while kill -0 "$_upe_pid" 2>/dev/null; do
        if [ "$_upe_waited" -ge "$_upe_secs" ]; then
            _uv_probe_terminate "$_upe_target" "$_upe_pid" 5
            wait "$_upe_pid" 2>/dev/null
            _UV_PROBE_TARGET=""
            _UV_PROBE_PID=""
            unset _upe_pid _upe_waited _upe_target _upe_pgid _upe_self
            return 124
        fi
        sleep 1
        _upe_waited=$((_upe_waited + 1))
    done
    wait "$_upe_pid"
    _upe_rc=$?
    _UV_PROBE_TARGET=""
    _UV_PROBE_PID=""
    unset _upe_pid _upe_waited _upe_target _upe_pgid _upe_self
    return $_upe_rc
}

_uv_install_pinned() {
    _UIP_UNFETCHED=false
    _uip_spec=$(_uv_pinned_asset) || return 1
    [ -n "$_uip_spec" ] || return 1
    _uip_asset=${_uip_spec%% *}
    _uip_want=${_uip_spec##* }
    command -v tar >/dev/null 2>&1 || return 1
    if [ -z "$(_uv_sha256 /dev/null)" ]; then return 1; fi

    _uip_dest=""
    for _uip_candidate in "${UV_INSTALL_DIR:-}" "${UV_UNMANAGED_INSTALL:-}" "${XDG_BIN_HOME:-}"; do
        if [ -n "$_uip_candidate" ]; then _uip_dest="$_uip_candidate"; break; fi
    done
    if [ -z "$_uip_dest" ] && [ -n "${XDG_DATA_HOME:-}" ]; then _uip_dest="$XDG_DATA_HOME/../bin"; fi
    if [ -z "$_uip_dest" ]; then
        [ -n "${HOME:-}" ] || return 1
        _uip_dest="$HOME/.local/bin"
    fi

    _uip_work=$(mktemp -d 2>/dev/null) || return 1
    _UIP_WORK="$_uip_work"
    _uip_rc=1
    # A configured mirror is exclusive, as for astral; download() has no timeout.
    if [ -n "${UV_DOWNLOAD_URL:-}" ]; then
        _uip_bases="${UV_DOWNLOAD_URL%/}"
    elif [ -n "${INSTALLER_DOWNLOAD_URL:-}" ]; then
        _uip_bases="${INSTALLER_DOWNLOAD_URL%/}"
    elif [ -n "${UV_INSTALLER_GHE_BASE_URL:-}" ]; then
        _uip_bases="${UV_INSTALLER_GHE_BASE_URL%/}/astral-sh/uv/releases/download/$UV_PINNED_VERSION"
    elif [ -n "${UV_INSTALLER_GITHUB_BASE_URL:-}" ]; then
        _uip_bases="${UV_INSTALLER_GITHUB_BASE_URL%/}/astral-sh/uv/releases/download/$UV_PINNED_VERSION"
    elif [ -n "${UNSLOTH_UV_WHEEL_MIRROR:-}" ]; then
        _uip_bases=""
        if _uip_wheel=$(_uv_pinned_wheel "$_uip_asset"); then
            _uip_path=${_uip_wheel% *}
            _uip_bases="${UNSLOTH_UV_WHEEL_MIRROR%/}/${_uip_path%/*}"
            _uip_asset=${_uip_path##*/}
            _uip_want=${_uip_wheel##* }
        fi
    else
        _uip_bases="https://releases.astral.sh/github/uv/releases/download/$UV_PINNED_VERSION
https://github.com/astral-sh/uv/releases/download/$UV_PINNED_VERSION"
    fi
    _UIP_UNFETCHED=true
    for _uip_base in $_uip_bases; do
        if ! download "$_uip_base/$_uip_asset" "$_uip_work/$_uip_asset" 2>/dev/null; then continue; fi
        _UIP_UNFETCHED=false
        _uip_got=$(_uv_sha256 "$_uip_work/$_uip_asset")
        if [ "$_uip_got" != "$_uip_want" ]; then
            # Not tauri_log: the app shows unknown [TAURI:*] markers verbatim in its progress UI.
            if _is_verbose; then
                echo "uv archive digest mismatch from $_uip_base, trying the next source" >&2
            fi
            continue
        fi
        case "$_uip_asset" in
            *.whl) _uv_unzip "$_uip_work/$_uip_asset" "$_uip_work" || continue ;;
            *) tar -xzf "$_uip_work/$_uip_asset" -C "$_uip_work" 2>/dev/null || continue ;;
        esac
        mkdir -p "$_uip_dest" 2>/dev/null || break
        _uip_placed=0
        # uv and uvx ship as a set: stage both, then publish both, or touch nothing.
        _uip_ready=1
        for _uip_exe in uv uvx; do
            if [ -d "$_uip_dest/$_uip_exe" ]; then _uip_ready=0; break; fi
            _uip_src=$(find "$_uip_work" -type f -name "$_uip_exe" 2>/dev/null | head -1)
            if [ -z "$_uip_src" ] || [ ! -f "$_uip_src" ]; then _uip_ready=0; break; fi
            # Rename, not cp: cp writes through a symlinked destination. mktemp avoids racing installers.
            _uip_stage=$(mktemp "$_uip_dest/.$_uip_exe.XXXXXX" 2>/dev/null) || { _uip_ready=0; break; }
            if [ "$_uip_exe" = "uv" ]; then _UIP_STAGE="$_uip_stage"; else _UIP_STAGE2="$_uip_stage"; fi
            if ! cp -f "$_uip_src" "$_uip_stage" 2>/dev/null; then _uip_ready=0; break; fi
            # 0755, not +x: under umask 077, +x leaves uv unusable for other accounts.
            chmod 0755 "$_uip_stage" 2>/dev/null || true
            # Validate before publishing: the rename destroys the incumbent (noexec, missing loader).
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
            _UNSLOTH_UV_BIN_DIR="$_uip_dest"
            _uip_rc=0
        fi
        break
    done
    rm -rf "$_uip_work"
    _UIP_WORK=""
    _UIP_STAGE=""
    _UIP_STAGE2=""
    # Nothing is unwound on failure: deleting could remove a working uv the host already had.
    return "$_uip_rc"
}

if ! command -v uv >/dev/null 2>&1 || ! _uv_version_ok uv; then
    # Hosts at uv 0.8.16-0.9.2 worked offline before the floor rose, so download failure is not fatal.
    _uv_present_before=false
    if command -v uv >/dev/null 2>&1; then
        # `|| _uv_prev_ver=`: without awk the pipeline exits 127 and set -e would kill the install.
        _uv_prev_ver=$(uv --version 2>/dev/null | awk '{print $2}' 2>/dev/null) \
            || _uv_prev_ver=""
        if [ -z "$_uv_prev_ver" ] || _uv_version_ok uv "$UV_OFFLINE_MIN_VERSION"; then
            _uv_present_before=true
        fi
    fi
    substep "installing uv package manager..."
    _uv_refreshed=true
    if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then
        # Pinned release first: fetch a digest-checked data file rather than download-run-delete a remote script. See tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
        if _uv_install_pinned || { [ "$_UIP_UNFETCHED" = true ] && _mirror_switch uvbin && _uv_install_pinned; }; then
            :
        else
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
    if [ -n "${_UNSLOTH_UV_BIN_DIR:-}" ] && [ "$_UNSLOTH_UV_BIN_DIR" != "$HOME/.local/bin" ]; then
        export PATH="$_UNSLOTH_UV_BIN_DIR:$PATH"
    fi
fi

_configure_uv_cache

# ── Create venv (migrate old layout if possible, otherwise fresh) ──
tauri_log "STEP" "Creating virtual environment"
mkdir -p "$STUDIO_HOME"

_MIGRATED=false
_PREV_TORCH_VER=""
_EXISTING_INSTALL=false

if [ -x "$VENV_DIR/bin/python" ] || _dir_has_entries "$VENV_DIR"; then
    # why: matching guard to the .venv branch below -- in env-mode $STUDIO_HOME is user-chosen, so refuse to nuke an existing unsloth_studio that lacks Unsloth sentinels; accept the in-VENV marker so partial-install retries are not blocked. The root marker goes through _claim_sentinel, not -f: the claim refuses to write one through a link, so reading one through a link here would undo that. The older sentinels keep -f, since bin/unsloth is legitimately a symlink into the venv.
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ] \
       && ! _claim_sentinel "$STUDIO_HOME/.unsloth-studio-owned" \
       && [ ! -f "$VENV_DIR/.unsloth-studio-owned" ] \
       && [ ! -f "$STUDIO_HOME/share/studio.conf" ] \
       && [ ! -f "$STUDIO_HOME/bin/unsloth" ]; then
        echo "ERROR: $VENV_DIR already exists but does not look like an Unsloth Studio install." >&2
        echo "       Move it aside or choose an empty UNSLOTH_STUDIO_HOME." >&2
        exit 1
    fi
    _EXISTING_INSTALL=true
    # Record the old torch before the venv moves aside. Read version.py first: `import torch` can
    # hang on a wedged Intel driver.
    _PREV_TORCH_VER=""
    for _prev_tv in "$VENV_DIR"/lib/python*/site-packages/torch/version.py; do
        [ -f "$_prev_tv" ] || continue
        _PREV_TORCH_VER=$(sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" "$_prev_tv" | head -n 1)
        break
    done
    [ -n "$_PREV_TORCH_VER" ] || _PREV_TORCH_VER=$(_run_bounded "$VENV_DIR/bin/python" -c \
        "import torch; print(torch.__version__)" 2>/dev/null | tail -n 1 || true)
    if [ "${_NO_ROLLBACK:-false}" = true ]; then
        substep "moving the existing environment aside..."
    else
        substep "preserving existing environment for rollback..."
    fi
    if ! _start_studio_venv_replacement "$VENV_DIR"; then
        echo "ERROR: could not move $VENV_DIR aside to reinstall." >&2
        echo "       Check that $STUDIO_HOME is writable, or move $VENV_DIR yourself and re-run." >&2
        exit 1
    fi
elif [ "$_STUDIO_HOME_REDIRECT" != "env" ] && [ -x "$STUDIO_HOME/.venv/bin/python" ]; then
    substep "found legacy Unsloth environment, validating..."
    _EXISTING_INSTALL=true
    for _prev_tv in "$STUDIO_HOME"/.venv/lib/python*/site-packages/torch/version.py; do
        [ -f "$_prev_tv" ] || continue
        _PREV_TORCH_VER=$(sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" "$_prev_tv" | head -n 1)
        break
    done
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
        # `mv` into an existing directory nests it (#9479), so clear the empty target first.
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
elif [ "$_STUDIO_HOME_REDIRECT" != "env" ] && _dir_has_entries "$STUDIO_HOME/.venv"; then
    _EXISTING_INSTALL=true
    for _prev_tv in "$STUDIO_HOME"/.venv/lib/python*/site-packages/torch/version.py; do
        [ -f "$_prev_tv" ] || continue
        _PREV_TORCH_VER=$(sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" "$_prev_tv" | head -n 1)
        break
    done
fi

if [ "$SKIP_TORCH" = true ] && [ "$MAC_INTEL" = true ] && [ -z "$_USER_PYTHON" ] && [ -x "$VENV_DIR/bin/python" ]; then
    _PY_MM=$("$VENV_DIR/bin/python" -c \
        "import sys; print('{}.{}'.format(*sys.version_info[:2]))" 2>/dev/null || echo "")
    if [ "$_PY_MM" != "3.12" ]; then
        echo "  Recreating Intel Mac environment with Python 3.12 (was $_PY_MM)..."
        rm -rf "$VENV_DIR"
    fi
fi

# uv runs install_name_tool on macOS, which pops the CLT dialog without dev tools
# (astral-sh/uv#14893). Do not execute it or xcrun to probe.
_macos_has_selected_install_name_tool() {
    _uvv_developer_dir=$(xcode-select -p 2>/dev/null) || return 1
    [ -n "$_uvv_developer_dir" ] && [ -d "$_uvv_developer_dir" ] || return 1

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

# Without a real install_name_tool, shim a failing one for uv only, so uv keeps its warning path.
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

# arm64 CPython so uv does not reuse a cached x86_64 (Rosetta) build; torch has no macOS x86_64
# wheels. only-managed avoids executing /usr/bin/python3 (CLT dialog); retry unflagged offline.
_uv_venv_arm64() {  # label
    _run_uv_venv "$1" "$VENV_DIR" \
        --python-preference only-managed \
        --python "cpython-${PYTHON_VERSION}-macos-aarch64-none" \
    || _run_uv_venv "$1 (system Python)" "$VENV_DIR" \
        --python "cpython-${PYTHON_VERSION}-macos-aarch64-none"
}

# Fedora sets python-downloads = "manual"; install the interpreter explicitly, then retry.
_uv_venv_requested() {  # label
    _uvvr_label="$1"
    _uvvr_req="$(_python_request "$PYTHON_VERSION")"
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

if [ -x "$VENV_DIR/bin/python" ]; then
    : > "$VENV_DIR/.unsloth-studio-owned" 2>/dev/null || true
fi

# Two independent Apple Silicon venv guards (Rosetta x86_64, the 3.13.8 torch bug); re-inspect, not elif.
if [ -z "$_USER_PYTHON" ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _inspect_venv() {
        "$VENV_DIR/bin/python" -c \
            "import platform, sys; print(platform.machine(), '{}.{}.{}'.format(*sys.version_info[:3]))" \
            2>/dev/null || echo " "
    }
    _info=$(_inspect_venv)
    _VENV_ARCH=${_info%% *}
    _PY_VER=${_info##* }
    if [ -z "$_VENV_ARCH" ] && [ -x "$VENV_DIR/bin/python" ]; then
        # file -L first: lipo is a CLT shim and running it can pop a GUI dialog.
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

# The request above only decides what a NEW venv gets. A venv from an earlier run, on any platform, can still hold a skipped interpreter, and reusing it is how the reported installs stayed broken across re-runs. Honour --python.
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

# Bump all three ceilings when the next minor is validated; ROCm floors below stay literal.
_TORCH_CEILING="2.12.0"
_TORCHVISION_CEILING="0.27.0"
_TORCHAUDIO_CEILING="2.12.0"
# Tightened for Python 3.13+ on arm64 macOS (no cp313 wheels below 2.6).
TORCH_CONSTRAINT="torch>=2.4,<${_TORCH_CEILING}"
if [ "$SKIP_TORCH" = false ] && [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    _PY_MINOR=$("$VENV_DIR/bin/python" -c \
        "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$_PY_MINOR" -ge 13 ] 2>/dev/null; then
        TORCH_CONSTRAINT="torch>=2.6,<${_TORCH_CEILING}"
    fi
fi
# Companions bounded to torch's window: torchaudio 2.11 dropped its torch pin, so it can drift.
TORCHVISION_CONSTRAINT="torchvision>=0.19,<${_TORCHVISION_CEILING}"
TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<${_TORCHAUDIO_CEILING}"
_CU130_TORCH_CEILING="2.15.0"
_CU130_NEW_INSTALL_TORCH="torch>=2.13.0,<2.14.0"

_REPO_ROOT="$(cd "$(dirname "$0" 2>/dev/null || echo ".")" && pwd)"
# Trust adjacent scripts only for an explicit --local run from the file itself; a piped install's
# _REPO_ROOT is the caller's cwd, where a planted file would run.
_REPO_IS_CHECKOUT=0
case "$0" in
    */install.sh|install.sh)
        [ "$STUDIO_LOCAL_INSTALL" = true ] && [ -r "$0" ] && _REPO_IS_CHECKOUT=1 ;;
esac

_ZOO_REF="${UNSLOTH_ZOO_REF:-main}"
_ZOO_GIT_SPEC="unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo@${_ZOO_REF}"

_find_no_torch_runtime() {
    if [ "$_REPO_IS_CHECKOUT" = "1" ] && [ -f "$_REPO_ROOT/studio/backend/requirements/no-torch-runtime.txt" ]; then
        echo "$_REPO_ROOT/studio/backend/requirements/no-torch-runtime.txt"
        return
    fi
    _rt=$(find "$VENV_DIR" -path "*/studio/backend/requirements/no-torch-runtime.txt" -print -quit 2>/dev/null || echo "")
    if [ -n "$_rt" ]; then
        echo "$_rt"
        return
    fi
}

_ensure_rocm_probe_env() {
    export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
    if ! command -v rocminfo >/dev/null 2>&1 && [ -x /opt/rocm/bin/rocminfo ]; then
        PATH="$PATH:/opt/rocm/bin"
    fi
}

# True when the run was told to install ROCm torch even with CUDA usable (#10450).
# Mirrors install_python_stack._rocm_torch_explicitly_requested; keep the two in step.
_rocm_torch_explicitly_requested() {
    case "$(printf '%s' "${UNSLOTH_FORCE_ROCM_TORCH:-}" \
            | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# The NVIDIA veto lives inside this function because tests/sh lifts probes one function at a time.
_has_amd_rocm_gpu() {
    _ensure_rocm_probe_env
    if [ "${1:-}" != "ignore-nvidia" ] && _has_usable_nvidia_gpu && \
       ! _rocm_torch_explicitly_requested; then
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
        # vendor_id 4098 (0x1002) is AMD; KFD CPU nodes report 0 and NVIDIA's open module 4318.
        return 0
    fi
    return 1
}

# Hardware-only check: _infer_linux_amd_gfx_arch returns a declared arch before probing.
_amd_hardware_corroborated() {
    _amd_gpu_present_via_pci && return 0
    # On WSL neither /dev/dxg nor librocdxg names a vendor, hence "physical" mode.
    if [ -e /dev/dxg ] || grep -qi microsoft /proc/version 2>/dev/null; then
        for _ahc_d in /opt/rocm/lib /opt/rocm/lib64 /opt/rocm-*/lib /opt/rocm-*/lib64; do
            { [ -e "$_ahc_d/librocdxg.so" ] || [ -e "$_ahc_d/librocdxg.so.1" ]; } || continue
            _has_usable_nvidia_gpu || return 0
            [ -n "$(_probe_amd_gfx_arch physical 2>/dev/null)" ] && return 0
            return 1
        done
    fi
    return 1
}

# Whether any index carries kernels for this arch (gfx1010 has none).
# Mirrors _gfx_has_a_wheel_route / _GENERIC_ROCM_WHEEL_GFX in install_python_stack.py.
_amd_gfx_has_wheel_route() {
    _amd_arch_index_family_for_gfx "$1" >/dev/null 2>&1 && return 0
    case "$1" in
        gfx900|gfx906|gfx908|gfx90a|gfx942|gfx950) : ;;
        gfx1030|gfx1100|gfx1101|gfx1102|gfx1150|gfx1151|gfx1200|gfx1201) : ;;
        *) return 1 ;;
    esac
    _amd_generic_tag_carries_gfx "$1"
}

# Whether the generic wheel for this host's ROCm version covers the arch; unreadable is NO.
# Mirrors _GENERIC_WHEEL_GFX_MIN_ROCM.
_amd_generic_tag_carries_gfx() {
    case "$1" in
        gfx950|gfx1150|gfx1151) _agtc_min_major=7; _agtc_min_minor=0 ;;
        gfx1200|gfx1201)        _agtc_min_major=6; _agtc_min_minor=4 ;;
        *) return 0 ;;
    esac
    _agtc_tag=$(_detect_rocm_version_tag 2>/dev/null) || _agtc_tag=""
    _agtc_ver=${_agtc_tag#rocm}
    _agtc_major=$(printf '%s' "$_agtc_ver" | cut -d. -f1)
    _agtc_minor=$(printf '%s' "$_agtc_ver" | cut -d. -f2)
    case "$_agtc_major$_agtc_minor" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$_agtc_major" -gt "$_agtc_min_major" ] && return 0
    [ "$_agtc_major" -eq "$_agtc_min_major" ] && [ "$_agtc_minor" -ge "$_agtc_min_minor" ] && return 0
    return 1
}

# ${VAR+x}: a set-but-empty mask hides every device. Mirrors _visible_masks_select_no_gpu.
_amd_visible_masks_select_no_gpu() {
    if [ -n "${HIP_VISIBLE_DEVICES+x}" ]; then
        _avm_hip=HIP_VISIBLE_DEVICES
    else
        _avm_hip=CUDA_VISIBLE_DEVICES
    fi
    for _avm_var in ROCR_VISIBLE_DEVICES "$_avm_hip"; do
        eval "_avm_set=\${$_avm_var+x}"
        eval "_avm_val=\${$_avm_var:-}"
        [ -n "$_avm_set" ] || continue
        case "$(printf '%s' "$_avm_val" | tr -d '[:space:]')" in
            ''|-1) return 0 ;;
        esac
    done
    return 1
}

# Survivors are the prefix of resolvable ordinals; ROCr also stops at a repeated ordinal.
_amd_mask_survivors() {
    printf '%s\n' "$1" | _ams_vis="$2" _ams_layer="${3:-}" awk '
        BEGIN { vis = ENVIRON["_ams_vis"]; layer = ENVIRON["_ams_layer"] }
        NF { vals[n++] = $0 }
        END {
            count = split(vis, want, ",")
            for (i = 1; i <= count; i++) {
                gsub(/^[ \t]+|[ \t]+$/, "", want[i])
                if (want[i] !~ /^[0-9]+$/) break
                idx = want[i] + 0
                if (idx >= n) break
                if (layer == "rocr" && (idx in seen)) break
                seen[idx] = 1
                print vals[idx]
            }
        }'
}

# ROCr filters and renumbers beneath HIP; the first survivor wins.
_amd_runtime_gfx_target() {
    _argt_list="$1"
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        _argt_list=$(_amd_mask_survivors "$_argt_list" "$ROCR_VISIBLE_DEVICES" rocr)
        [ -n "$_argt_list" ] || return 0
    fi
    _argt_hip="${HIP_VISIBLE_DEVICES:-}"
    if [ -z "${HIP_VISIBLE_DEVICES+x}" ]; then
        _argt_hip="${CUDA_VISIBLE_DEVICES:-}"
    fi
    if [ -n "$_argt_hip" ]; then
        _argt_list=$(_amd_mask_survivors "$_argt_list" "$_argt_hip")
        [ -n "$_argt_list" ] || return 0
    fi
    printf '%s\n' "$_argt_list" | awk 'NF { print; exit }'
}

# Mirror of _SHADOWING_INTEGRATED_GFX, held to it by tests/studio/install/test_rocm_arch_table_parity.py.
_amd_gfx_is_shadowing_integrated() {
    case "$1" in
        gfx90c|gfx1013|gfx1033|gfx1035|gfx1036|gfx1103|gfx1153) return 0 ;;
    esac
    return 1
}

# Prefer the discrete card over an integrated one enumerated first (#7776); never gfx906.
_amd_prefer_discrete_gfx() {
    _apdg_devs="$1"
    _apdg_sel="$2"
    if ! _amd_gfx_is_shadowing_integrated "$_apdg_sel"; then
        printf '%s' "$_apdg_sel"
        return 0
    fi
    if [ -n "${HIP_VISIBLE_DEVICES+x}" ] || [ -n "${ROCR_VISIBLE_DEVICES+x}" ] || \
       [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
        printf '%s' "$_apdg_sel"
        return 0
    fi
    _apdg_others=$(printf '%s\n' "$_apdg_devs" | awk 'NF' | while IFS= read -r _apdg_g; do
        _amd_gfx_is_shadowing_integrated "$_apdg_g" && continue
        [ "$_apdg_g" = gfx906 ] && continue
        printf '%s\n' "$_apdg_g"
    done)
    _apdg_pick=$(printf '%s\n' "$_apdg_others" | awk 'NF' | while IFS= read -r _apdg_g; do
        _amd_gfx_has_wheel_route "$_apdg_g" && printf '%s\n' "$_apdg_g"
    done | awk 'NF { print; exit }')
    if [ -z "$_apdg_pick" ] && ! _amd_gfx_has_wheel_route "$_apdg_sel"; then
        _apdg_pick=$(printf '%s\n' "$_apdg_others" | awk 'NF { print; exit }')
    fi
    [ -n "$_apdg_pick" ] || _apdg_pick="$_apdg_sel"
    printf '%s' "$_apdg_pick"
}

_amd_request_has_a_wheel_route() {
    _AMD_REQUEST_TARGET_GFX=""
    # Only a probe-resolved target may clear a safety gate; a declared one just routes.
    _AMD_REQUEST_TARGET_SOURCE=""
    _amd_visible_masks_select_no_gpu && return 1
    # A declared arch decides, but only after hardware corroborates there is an AMD GPU.
    _arwr_decl=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" \
        | tr '[:upper:]' '[:lower:]' | sed 's/:.*$//' | tr -d '[:space:]')
    if [ -n "$_arwr_decl" ]; then
        _amd_hardware_corroborated || _kfd_gfx_targets 2>/dev/null | grep -q . || return 1
        if [ -n "${HIP_VISIBLE_DEVICES+x}" ] || [ -n "${ROCR_VISIBLE_DEVICES+x}" ] || \
           [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
            _arwr_dd=$(_amd_ordered_gfx_devices 2>/dev/null | sed 's/:.*$//' \
                | tr '[:upper:]' '[:lower:]' | awk 'NF')
            [ -n "$_arwr_dd" ] || _arwr_dd=$(_kfd_gfx_targets 2>/dev/null | sed 's/:.*$//' \
                | tr '[:upper:]' '[:lower:]' | awk 'NF')
            if [ -n "$_arwr_dd" ] && [ -z "$(_amd_runtime_gfx_target "$_arwr_dd")" ]; then
                return 1
            fi
        fi
        if _amd_gfx_has_wheel_route "$_arwr_decl"; then
            _AMD_REQUEST_TARGET_GFX="$_arwr_decl"
            _AMD_REQUEST_TARGET_SOURCE=declared
            return 0
        fi
        return 1
    fi
    _arwr_all=$(_probe_amd_gfx_arch physical 2>/dev/null || true)
    [ -n "$_arwr_all" ] || _arwr_all=$(_kfd_gfx_targets 2>/dev/null || true)
    if [ -z "$_arwr_all" ]; then
        _amd_hardware_corroborated || return 1
        _arwr_all=$(_infer_linux_amd_gfx_arch 2>/dev/null || true)
    fi
    _arwr_archs=$(printf '%s\n' "$_arwr_all" | sed 's/:.*$//' \
        | tr '[:upper:]' '[:lower:]' | awk 'NF')
    [ -n "$_arwr_archs" ] || return 1
    # Physical count before masks: gfx906's rocm6.3 tag requires it to be the sole arch.
    _arwr_count=$(printf '%s\n' "$_arwr_archs" | sort -u | wc -l | tr -d ' ')
    # Fails closed: a wrong yes replaces a working CUDA stack with kernel-less wheels.
    _arwr_devs=$(_amd_ordered_gfx_devices 2>/dev/null | sed 's/:.*$//' \
        | tr '[:upper:]' '[:lower:]' | awk 'NF')
    if [ -z "$_arwr_devs" ]; then
        _arwr_kfd=$(_kfd_gfx_targets 2>/dev/null | sed 's/:.*$//' \
            | tr '[:upper:]' '[:lower:]' | awk 'NF')
        if [ -n "$_arwr_kfd" ] && \
           [ "$(printf '%s\n' "$_arwr_kfd" | awk 'NF' | wc -l | tr -d ' ')" \
             = "$(printf '%s\n' "$_arwr_archs" | awk 'NF' | wc -l | tr -d ' ')" ]; then
            _arwr_devs="$_arwr_kfd"
        fi
    fi
    if [ -n "$_arwr_devs" ]; then
        _arwr_sel=$(_amd_runtime_gfx_target "$_arwr_devs")
    elif [ "$_arwr_count" -eq 1 ]; then
        _arwr_sel=$(_amd_runtime_gfx_target "$_arwr_archs")
    else
        return 1
    fi
    [ -n "$_arwr_sel" ] || return 1
    _arwr_pref="$_arwr_devs"
    [ -n "$_arwr_pref" ] || _arwr_pref="$_arwr_archs"
    _arwr_sel=$(_amd_prefer_discrete_gfx "$_arwr_pref" "$_arwr_sel")
    [ "$_arwr_sel" = gfx906 ] && [ "$_arwr_count" -gt 1 ] && return 1
    _amd_gfx_has_wheel_route "$_arwr_sel" || return 1
    _AMD_REQUEST_TARGET_GFX="$_arwr_sel"
    _AMD_REQUEST_TARGET_SOURCE=probe
    return 0
}

# Single answer to "does NVIDIA win" so index selection and reroutes agree (#10450).
_nvidia_gpu_wins_over_amd() {
    _has_usable_nvidia_gpu || return 1
    if _rocm_torch_explicitly_requested && _amd_request_has_a_wheel_route; then
        return 1
    fi
    return 0
}
# AMD display GPU on PCI even when ROCm cannot use it; only sharpens the no-GPU hint.
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

_amd_candidate_nodes() {
    printf '%s\n' /dev/kfd /dev/dri/renderD*
}

# Unreadable is not the same answer as "not AMD". Mirrors utils/hardware/amd.py::_render_node_vendor.
_amd_render_node_vendor() {
    _arnv_file="/sys/class/drm/${1##*/}/device/vendor"
    [ -r "$_arnv_file" ] || return 1
    read -r _arnv_vendor < "$_arnv_file" 2>/dev/null || return 1
    printf '%s' "$_arnv_vendor"
}

# AMD device nodes that exist but this user cannot open (root:render 0660). AMD-owned only:
# render nodes are root:render for every vendor.
_amd_nodes_closed_to_this_user() {
    _anctu_amd_in_topology=""
    _amd_candidate_nodes | while IFS= read -r _node; do
        [ -e "$_node" ] || continue
        { [ -r "$_node" ] && [ -w "$_node" ]; } && continue
        if [ "$_node" = /dev/kfd ]; then
            # An unreadable KFD topology is not evidence of another vendor; DRM confirms instead.
            # Mirrors utils/hardware/amd.py::amd_nodes_closed_to_this_user.
            _anctu_kfd_state=0
            _kfd_topology_amd_state || _anctu_kfd_state=$?
            if [ "$_anctu_kfd_state" -eq 1 ]; then
                continue
            elif [ "$_anctu_kfd_state" -ne 0 ]; then
                _a_confirmed_amd_render_node_exists || continue
            fi
        elif _node_vendor=$(_amd_render_node_vendor "$_node"); then
            [ "$_node_vendor" = "0x1002" ] || continue
        else
            # Unknown sysfs vendor is not "not AMD"; the world-readable KFD topology answers instead.
            if [ -z "$_anctu_amd_in_topology" ]; then
                if _kfd_topology_has_an_amd_gpu; then
                    _anctu_amd_in_topology=yes
                else
                    _anctu_amd_in_topology=no
                fi
            fi
            [ "$_anctu_amd_in_topology" = yes ] || continue
        fi
        printf '%s\n' "$_node"
    done
}

_an_amd_render_node_is_open() {
    for _anro_node in /dev/dri/renderD*; do
        [ -e "$_anro_node" ] || continue
        _anro_vendor_file="/sys/class/drm/${_anro_node##*/}/device/vendor"
        [ -r "$_anro_vendor_file" ] || continue
        read -r _anro_vendor < "$_anro_vendor_file" 2>/dev/null || continue
        [ "$_anro_vendor" = "0x1002" ] || continue
        { [ -r "$_anro_node" ] && [ -w "$_anro_node" ]; } && return 0
    done
    return 1
}

# Exit 0 = KFD names an AMD GPU, 1 = read and none, 2 = unreadable (masked /sys/class/kfd).
# Mirrors utils/hardware/amd.py::_kfd_topology_amd_state.
_kfd_topology_amd_state() {
    _ktas=$(awk '
        FNR == 1 { read_one = 1 }
        /vendor_id/ && $2 == 4098 { found = 1 }
        END { print (read_one + 0) ":" (found + 0) }
    ' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null) || _ktas=""
    case "$_ktas" in
        *:1) return 0 ;;
        1:0) return 1 ;;
        *)   return 2 ;;
    esac
}

_kfd_node_is_amds() {
    _kina_state=0
    _kfd_topology_amd_state || _kina_state=$?
    [ "$_kina_state" -ne 1 ]
}

# Mirrors utils/hardware/amd.py::_amd_nodes_the_runtime_lacks; the two must agree.
_amd_silicon_behind_a_missing_kfd() {
    if _kfd_topology_has_an_amd_gpu; then return 0; fi
    _asbmk_state=0
    _kfd_topology_amd_state || _asbmk_state=$?
    [ "$_asbmk_state" -eq 2 ] || return 1
    _a_confirmed_amd_render_node_exists
}

# Strict: an unreadable vendor does not count. Mirrors amd.py::_a_confirmed_amd_render_node_exists.
_a_confirmed_amd_render_node_exists() {
    for _acarne_node in /dev/dri/renderD*; do
        [ -e "$_acarne_node" ] || continue
        _acarne_vendor=$(_amd_render_node_vendor "$_acarne_node") || continue
        [ "$_acarne_vendor" = "0x1002" ] && return 0
    done
    return 1
}

_kfd_topology_has_an_amd_gpu() {
    _kthag=0
    _kfd_topology_amd_state || _kthag=$?
    [ "$_kthag" -eq 0 ]
}

# Unknown vendor counts as present. Mirrors utils/hardware/amd.py::_amd_render_node_exists.
_amd_render_node_present() {
    _arnp_unknown=false
    for _arnp_node in /dev/dri/renderD*; do
        [ -e "$_arnp_node" ] || continue
        _arnp_vendor_file="/sys/class/drm/${_arnp_node##*/}/device/vendor"
        if [ -r "$_arnp_vendor_file" ] && \
           read -r _arnp_vendor < "$_arnp_vendor_file" 2>/dev/null; then
            [ "$_arnp_vendor" = "0x1002" ] && return 0
        else
            _arnp_unknown=true
        fi
    done
    [ "$_arnp_unknown" = true ]
}

# NSS names can hold backslashes or spaces; quote like Python shlex.quote for paste-safe commands.
_shell_quote() {
    case "$1" in
        "") printf "''" ;;
        *[!A-Za-z0-9_@%+=:,./-]*)
            printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
        *) printf '%s' "$1" ;;
    esac
}

# One line per node: join:NAME, gid:N (no group entry, usermod cannot take a GID), mode:PATH.
_amd_node_repairs() {
    for _anr_node in $1; do
        # With an ACL (ls shows "+") the group bits are the mask, so they cannot prescribe membership.
        # shellcheck disable=SC2012  # not parsing a file LIST: this reads column 11 of the
        # mode string for one named node; find -printf cannot report the ACL "+".
        case "$(ls -ld "$_anr_node" 2>/dev/null | cut -c11)" in
            +) printf 'acl:%s\n' "$_anr_node"; continue ;;
        esac
        # The group name goes last: NSS controls it and it may contain the separator.
        stat -c '%a|%g|%u|%n|%G' "$_anr_node" 2>/dev/null || true
    done | awk -F'|' -v self="$(id -u 2>/dev/null || echo -1)" \
              -v mygids=" $(id -G 2>/dev/null) " '
        /^acl:/ { print; next }
        {
            # mode|gid|uid|path|group-name. The name is last and is rejoined from every
            # remaining field, so a name carrying the separator is recovered whole rather
            # than silently truncated at its first pipe.
            gname = $5
            for (i = 6; i <= NF; i++) { gname = gname "|" $i }
            # A name that cannot be pasted on as ONE group is treated as no name at all,
            # and the node is reported by its GID instead. usermod -G takes a
            # comma-separated list, so a group genuinely named "render,sudo" is two groups
            # to it and the privileged test below, which compares whole names, walks
            # straight past it. Quoting is the wrong layer: the splitting happens inside
            # usermod, after the shell has handed it a single argument. The charset is
            # groupadd(8) portable plus the trailing $ of a Samba machine account, and
            # mirrors _GROUP_NAME_RE in utils/hardware/amd.py.
            if (gname !~ /^[A-Za-z_][A-Za-z0-9_.-]*[$]?$/) { gname = "" }
            # POSIX resolves the owner class EXCLUSIVELY once the uid matches, so on a
            # node this account owns the group bits are never consulted and no membership
            # opens it however they read. The repair there is the mode.
            if (self != -1 && $3 + 0 == self + 0) {
                # Unless the OWNER digit already grants rw: the node is known shut, so
                # the mode is not what denies it. Same external denial as the
                # already-a-member branch, by owner class. Mirrors amd.py external.
                # Padded first: stat %a drops leading zeros, so mode 060 prints "60" and
                # an owner index of 0 is not an error -- gawk and mawk return the GROUP
                # digit, busybox "", so the answer would vary by which awk the host ships.
                # The group read counts from the right, so it is unaffected.
                perm = $1
                while (length(perm) < 3) { perm = "0" perm }
                u = substr(perm, length(perm) - 2, 1) + 0
                if (u == 6 || u == 7) { print "external:" $4; next }
                print "owner:" $4; next
            }
            # A record whose GID did not come back as a number is one this cannot reason
            # about at all, so it is reported as a node no membership opens rather than
            # branched on. Unreachable while stat answers, and the direction that cannot
            # invent a repair if it ever stops.
            if ($2 !~ /^[0-9]+$/) { print "mode:" $4; next }
            # Neither the owner nor in the owning group leaves this account in the OTHER
            # class, which POSIX resolves exclusively too: if its digit already grants rw
            # the mode is not what denies a node already known shut, and no chmod or
            # usermod moves it. Mirrors the other-class branch in amd.py. Guarded on
            # membership, since a member is in the group class and the already branch
            # names the group; above the group-digit test, which would otherwise print a
            # mode repair for a node the mode is not blocking. The other digit is the
            # LAST, so unlike the owner one it needs no padding to be read.
            o = substr($1, length($1), 1) + 0
            if ((o == 6 || o == 7) && mygids !~ (" " $2 " ")) { print "external:" $4; next }
            # Group digit of the octal mode; read AND write, since HIP and the Vulkan
            # loader both open the node read-write.
            g = substr($1, length($1) - 1, 1) + 0
            if (g != 6 && g != 7) { print "mode:" $4; next }
            # Joining one of these hands over a great deal besides the GPU, so a node
            # owned by one is a udev misconfiguration to report, not a membership to
            # prescribe. gid 0 as well as the name, since a renamed root group is still
            # root. Mirrors _PRIVILEGED_GROUPS in utils/hardware/amd.py. ABOVE the unnamed
            # test: stat prints UNKNOWN for a gid the group database cannot name, so a
            # minimal container with no entry for gid 0 filed the node as an ordinary
            # unnamed GID and prescribed groupadd -g 0 plus usermod into root.
            if ($2 + 0 == 0 || gname ~ /^(root|wheel|sudo|admin|adm|disk|kmem|shadow|docker|lxd)$/) {
                pname = gname
                if (pname == "" || pname ~ /^UNKNOWN/) { pname = "root" }
                if (!pseen[pname]++) print "privileged:" pname
                next
            }
            # The node is already known shut, so a group this account is ALREADY in is
            # not what denies it: a container device cgroup or an LSM is, and usermod
            # would exit 0 and change nothing. Read from `id -G`, space-padded so 100
            # cannot match 1001. Above the unnamed branch for the same reason.
            if (mygids ~ (" " $2 " ")) {
                held = gname
                if (held == "" || held ~ /^UNKNOWN/) { held = $2 }
                if (!aseen[held]++) print "already:" held
                next
            }
            if (gname == "" || gname ~ /^UNKNOWN/) { if (!gseen[$2]++) print "gid:" $2; next }
            if (!nseen[gname]++) print "join:" gname
        }'
}

_amd_probe_arches() {
    printf '%s\n' "$1" | sed 's/:.*$//' | tr '[:upper:]' '[:lower:]' | awk 'NF' | sort -u
}

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

# setup.sh prefers UNSLOTH_ROCM_GFX_ARCH, so do not name an arbitrary family member.
_amd_sole_index_arch() {
    _sia=$(_amd_probe_arches "$1")
    [ -n "$_sia" ] || return 1
    [ "$(printf '%s\n' "$_sia" | awk 'END{print NR}')" -eq 1 ] || return 1
    _amd_arch_index_family_for_gfx "$_sia" >/dev/null 2>&1 || return 1
    printf '%s\n' "$_sia"
}

# Mirrors install.ps1 $archFamilyMap.
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

# Kept in sync with install.ps1 nameArchTable.
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
        *"RX 6950"*|*"RX 6900"*|*"RX 6850"*|*"RX 6800"*|*"RX 6750"*|*"RX 6700"*|*"PRO W6800"*|*"PRO W6900"*) echo gfx1030 ;;
        *"RX 6650"*|*"RX 6600"*|*"PRO W6600"*|*"PRO W6650"*) echo gfx1032 ;;
        *"RX 6550"*|*"RX 6500"*|*"RX 6450"*|*"RX 6400"*|*"RX 6300"*|*"PRO W6400"*|*"PRO W6500"*|*"PRO W6300"*) echo gfx1034 ;;
        *) return 1 ;;
    esac
}

# Messaging only, never routes (#8529). ORDER IS LOAD-BEARING: RDNA 1 arms before Polaris,
# since *"RX 570"* would match "RX 5700 XT". Polaris 11/12 deliberately left out.
_infer_unsupported_amd_gfx_arch_from_gpu_name() {
    case "$1" in
        *"Radeon Pro V520"*|*"Radeon Pro 5600M"*) echo gfx1011 ;;  # RDNA 1
        *"RX 5700"*|*"RX 5600"*|*"Radeon Pro 5600 XT"*|*"Radeon Pro 5700"*|*"Radeon Pro W5700"*) echo gfx1010 ;;  # RDNA 1 (Navi 10)
        *"RX 5500"*|*"RX 5300"*|*"Radeon Pro W5500"*|*"Radeon Pro W5300"*) echo gfx1012 ;;  # RDNA 1 (Navi 14)
        *"RX 470"|*"RX 470"[!0]*|*"RX 480"|*"RX 480"[!0]*|*"RX 570"|*"RX 570"[!0]*|*"RX 580"|*"RX 580"[!0]*|*"RX 590"|*"RX 590"[!0]*|*"Radeon Pro WX 7100"*|*"Radeon Pro WX 5100"*) echo gfx803 ;;  # Polaris 10/20/30
        *) return 1 ;;
    esac
}

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

# Best-effort gfx inference when ROCm tools cannot see the GPU (unslothai#7301).
_infer_linux_amd_gfx_arch() {
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        printf '%s\n' "$(printf '%s' "$UNSLOTH_ROCM_GFX_ARCH" | tr '[:upper:]' '[:lower:]')"
        return 0
    fi
    _gpu_evidence=""
    if [ -e /dev/dxg ] || grep -qi microsoft /proc/version 2>/dev/null; then
        for _d in /opt/rocm/lib /opt/rocm/lib64 /opt/rocm-*/lib /opt/rocm-*/lib64; do
            { [ -e "$_d/librocdxg.so" ] || [ -e "$_d/librocdxg.so.1" ]; } && _rocdxg=1 && break
        done
        [ -n "${_rocdxg:-}" ] || return 1
        _gpu_evidence=1
    elif _amd_gpu_present_via_pci; then
        _gpu_evidence=1
    fi
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
        # Scan every display line; vendor match is case-sensitive ("ATI" vs "CorporATIon").
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

# ROCr builds gfx<major><minor><stepping hex> (9.0.10 -> gfx90a).
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

# KFD gfx_target_version is immune to HSA_OVERRIDE_GFX_VERSION (#7331).
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

# Pair each GPU gfx id with its own marketing name. Keep in sync with studio/setup.sh.
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

_amd_smi_hip_order() {
    # POSIX awk forbids a newline in a -v value (fatal under gawk --posix), so records arrive
    # on stdin before the map, sentinel-separated. Line 1 names the index space.
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
            # All or nothing, like get_hip_id_by_gpu_index: a partial or colliding map is not 1:1.
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

# amd-smi uses KFD discovery order, masks use HIP order; `amd-smi list -e` maps them.
# Unknown arches keep their slot so later ordinals do not shift. Keep in sync with studio/setup.sh.
_gfx_arch_slots() {
    awk -F'|' '
        NF { rec[n++] = $1; if ($1 != "") any = 1 }
        END {
            if (!any) exit
            for (i = 0; i < n; i++) print (rec[i] == "" ? "unknown" : rec[i])
        }
    '
}

# One record per adapter in `GPU: N` order. Keep in sync with studio/setup.sh.
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
        # amd-smi upper-cases every key (amdsmi_logger.py _capitalize_keys), so match
        # case-folded. Two header shapes: `GPU: 0` opens a keyed block with the arch later,
        # `GPU[0] : gfx1100` IS the record. Matching only the first answered no arch at all.
        /^[[:space:]]*GPU[[:space:]]*[:\[][[:space:]]*[0-9]/ {
            flush(); started = 1
            if (match($0, /gfx[1-9][0-9a-z][0-9a-z][0-9a-z]?/)) gfx = substr($0, RSTART, RLENGTH)
            next
        }
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


# Physical arch under an HSA_OVERRIDE_GFX_VERSION spoof (#7331), or nothing. Requires corroboration
# (KFD, then unspoofed rocminfo). Kept in sync with _hsa_spoofed_physical_gfx in install_python_stack.py.
_hsa_spoofed_physical_gfx() {
    _hsp_inferred="${1:-}"
    _hsp_probed_all="${2:-}"
    [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] || return 0
    case "$_hsp_inferred" in
        gfx1151|gfx1150|gfx1152) : ;;
        *) return 0 ;;
    esac
    _hsp_n=$(printf '%s\n' "$_hsp_probed_all" | awk 'NF && !seen[$0]++ { n++ } END { print n + 0 }')
    [ "${_hsp_n:-0}" -ne 1 ] && return 0
    _hsp_probed=$(printf '%s\n' "$_hsp_probed_all" | awk 'NF { print; exit }')
    [ -n "$_hsp_probed" ] || return 0
    [ "$_hsp_probed" = "$_hsp_inferred" ] && return 0
    [ "$(_hsa_override_gfx_arch "$HSA_OVERRIDE_GFX_VERSION")" = "$_hsp_probed" ] || return 0

    echo "  [WARN] HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION is set; ROCm reports" >&2
    echo "  [WARN] $_hsp_probed but this host's product name is $_hsp_inferred. Checking for a spoof." >&2

    _hsp_kfd=$(_kfd_gfx_targets | awk 'NF')
    if [ -n "$_hsp_kfd" ]; then
        if [ "$_hsp_kfd" = "$_hsp_inferred" ]; then
            echo "  [WARN] KFD topology sysfs reports $_hsp_inferred -- $_hsp_probed is a spoof." >&2
            printf '%s\n' "$_hsp_inferred"
        else
            echo "  [WARN] The kernel does not corroborate a spoof; keeping $_hsp_probed." >&2
        fi
        return 0
    fi

    # Re-probe without the override and masks; an unchanged name means real silicon.
    _hsp_re=""
    if command -v rocminfo >/dev/null 2>&1; then
        _hsp_re=$( (unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; \
                    rocminfo 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' | awk 'NF && !seen[$0]++' || true)
    fi
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

# Prints gfx token(s) or nothing and always returns 0 (set -e). Tool probes run with masks cleared
# (#7314); "physical" also strips HSA_OVERRIDE_GFX_VERSION (unslothai#7331).
# shellcheck disable=SC2086  # $_pg_strip is a LIST of names for unset; quoting it would
# unset one variable whose name contains spaces.
_probe_amd_gfx_arch() {
    _ensure_rocm_probe_env
    case "${1:-}" in
        physical) _pg_strip="ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION" ;;
        *)        _pg_strip="ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES" ;;
    esac
    if [ "${1:-}" = "physical" ]; then
        _pg=""
    else
        _pg=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]')
    fi
    if [ -z "$_pg" ] && command -v rocminfo >/dev/null 2>&1; then
        _pg=$( (unset $_pg_strip; rocminfo 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
    fi
    if [ -z "$_pg" ] && command -v amd-smi >/dev/null 2>&1; then
        _pg=$( (unset $_pg_strip; amd-smi list 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        if [ -z "$_pg" ]; then
            _pg=$( (unset $_pg_strip; amd-smi static --asic 2>/dev/null) | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        fi
    fi
    printf '%s\n' "$_pg"
}

# One gfx per GPU in ROCr order. Twin of install_python_stack._detect_amd_gfx_codes(dedup = False).
_amd_ordered_gfx_devices() {
    _ensure_rocm_probe_env
    command -v rocminfo >/dev/null 2>&1 || return 0
    (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION; rocminfo 2>/dev/null) \
        | _rocminfo_gpu_records | sed 's/|.*$//' | awk 'NF'
}
# Physical inventory (ignores CUDA_VISIBLE_DEVICES): the wheel must support the host.
# Shared decision with install.ps1 / setup.ps1 / install_python_stack.py.
_nvidia_cu126_verdict() {
    if [ -n "${2:-}" ]; then
        _ncv_caps=$2
    else
        [ -n "$1" ] || return 0
        _ncv_caps=$(_run_bounded "$1" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null) || return 0
    fi
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

_cap_cuda_family_for_pre_turing() {
    case "$_ARCH" in
        x86_64|amd64) ;;
        *) printf '%s\n' "$1"; return ;;
    esac
    case "$1" in
        cu128|cu130) ;;
        *) printf '%s\n' "$1"; return ;;
    esac
    case "$(_nvidia_cu126_verdict "$2" "${3:-}")" in
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
# One helper per source, each returning 0 unconditionally: under set -e a failing source would kill the installer before the actionable warning at the end of the ROCm branch. Every source that execs runs through _run_bounded, since highest-wins consults all five and a single wedged probe would hang the installer; a timed-out probe just declined to answer.
_rocm_tag_from_amd_smi() {
    command -v amd-smi >/dev/null 2>&1 || return 0
    _run_bounded amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
        'NF>1{v=$2; sub(/[ \t|].*$/, "", v); if (v ~ /^[0-9]+\.[0-9]+/) {split(v,a,"."); print "rocm"a[1]"."a[2]} exit}' || return 0
}

_rocm_tag_from_version_file() {
    [ -r /opt/rocm/.info/version ] || return 0
    awk -F. '{print "rocm"$1"."$2; exit}' /opt/rocm/.info/version || return 0
}

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
    # Require status "installed": removed-but-not-purged packages still report a version.
    # `|| true`: dpkg-query exits nonzero when either package is absent.
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
    # Bounded: rpm -q can wedge on stale BerkeleyDB locks or a running dnf.
    # Query all names at once and emit every version; Fedora may lack rocm-core (unslothai#8731).
    _rt_ver=$(_run_bounded rpm -q --qf '%{VERSION}\n' rocm-core rocm-runtime rocm-hip 2>/dev/null \
        | awk '/^[0-9]/{print}') || return 0
    [ -n "$_rt_ver" ] || return 0
    printf '%s\n' "$_rt_ver" | awk -F'[.-]' 'NF{print "rocm"$1"."$2}' || return 0
}

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

# Take the highest source, not the first: split packaging ships stale components (#8402).

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
        _rt_seen=$(printf '%s\n' "$_rt_readings" \
            | grep '^rocm[1-9][0-9]*\.[0-9][0-9]*$' | sort -u | tr '\n' ' ') || _rt_seen=""
        case "$_rt_seen" in
            ""|"$_rt_best ") : ;;
            *) echo "[WARN] ROCm version sources disagree (${_rt_seen% }) -- using the highest, $_rt_best." >&2 ;;
        esac
    fi
    if [ -n "${_ROCM_TAG_MEMO:-}" ]; then
        printf '%s\n' "$_rt_best" > "$_ROCM_TAG_MEMO" 2>/dev/null || true
    fi
    printf '%s\n' "$_rt_best"
}

# ── Detect GPU and choose PyTorch index URL ──
# Mirrors Get-TorchIndexUrl in install.ps1. CPU-only hosts get the cpu index (auto picks unsloth==2024.8).
get_torch_index_url() {
    _AMD_REQUEST_TARGET_GFX=""
    _AMD_REQUEST_TARGET_SOURCE=""
    _base="${UNSLOTH_PYTORCH_MIRROR:-https://download.pytorch.org/whl}"
    _base="${_base%/}"
    # An explicit override skips ALL GPU probing: the URL is verbatim, _FAMILY is its leaf.
    _url="${UNSLOTH_TORCH_INDEX_URL:-}"
    _url="${_url#"${_url%%[![:space:]]*}"}"; _url="${_url%"${_url##*[![:space:]]}"}"
    if [ -n "$_url" ]; then
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
    case "$(uname -s)" in Darwin) echo "$_base/cpu"; return ;; esac
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
    # Here too: UNSLOTH_TORCH_BACKEND=cuda would make _ensure_rocm_torch return early (#10450).
    if [ "$_nvidia_detected" -eq 1 ] && _rocm_torch_explicitly_requested && \
       _amd_request_has_a_wheel_route; then
        echo "[INFO] UNSLOTH_FORCE_ROCM_TORCH is set and an AMD GPU is present -- selecting ROCm PyTorch over CUDA." >&2
        echo "[INFO] One torch install serves one vendor: the NVIDIA card will not be available to training until this is unset and the installer re-run." >&2
        _nvidia_detected=0
    fi
    if [ "$_nvidia_detected" -eq 0 ]; then
        case "$(uname -m)" in
            x86_64|amd64) : ;;
            *) echo "$_base/cpu"; return ;;
        esac
        if ! _has_amd_rocm_gpu; then
            echo "$_base/cpu"; return
        fi
        _amd_gfx_probe=$(_probe_amd_gfx_arch)
        if [ -z "$_amd_gfx_probe" ]; then
            if _amd_inferred_gfx=$(_infer_linux_amd_gfx_arch 2>/dev/null) && \
               [ -n "$_amd_inferred_gfx" ] && \
               _amd_arch_index_family_for_gfx "$_amd_inferred_gfx" >/dev/null 2>&1; then
                echo "[WARN] AMD GPU detected but rocminfo/amd-smi can't read its gfx arch -- inferring $_amd_inferred_gfx from hardware IDs." >&2
                echo "$_base/cpu"; return
            fi
            # Unsupported arch (#8529): advice only, same CPU index.
            if _amd_unsup_gfx=$(_infer_linux_unsupported_amd_gfx_arch 2>/dev/null); then
                echo "[WARN] AMD GPU detected ($_amd_unsup_gfx) -- Unsloth has no ROCm PyTorch wheels for that arch, installing CPU PyTorch." >&2
                echo "[WARN] This is expected on this GPU; repairing rocminfo/amd-smi or setting UNSLOTH_ROCM_GFX_ARCH will not give it ROCm PyTorch." >&2
                # `export` is load-bearing: a bare assignment never reaches the re-run.
                echo "[INFO] GGUF chat can still use this GPU through Vulkan: export UNSLOTH_LLAMA_CPP_BACKEND=vulkan and re-run this installer (it selects the llama.cpp bundle at install time)." >&2
                if _amd_therock_extra=$(_therock_device_extra_for_gfx "$_amd_unsup_gfx" 2>/dev/null); then
                    echo "[INFO] Untested: AMD's TheRock publishes nightly $_amd_unsup_gfx wheels. To try them, export both and re-run:" >&2
                    echo "[INFO]   export UNSLOTH_TORCH_INDEX_URL='$THEROCK_MIRROR'" >&2
                    echo "[INFO]   export UNSLOTH_TORCH_EXTRA=$_amd_therock_extra" >&2
                fi
                echo "$_base/cpu"; return
            fi
            echo "[WARN] AMD GPU detected but its gfx arch can't be read (rocminfo/amd-smi missing or not enumerating the GPU) -- installing CPU-only PyTorch." >&2
            echo "[WARN] For GPU PyTorch, install or repair rocminfo/amd-smi (e.g. sudo pacman -S rocm-hip-sdk) and re-run this installer." >&2
            echo "$_base/cpu"; return
        fi
        # Archs measured to compute INCORRECTLY under ROCm route to CPU instead. Not "everything AMD does not list": unsloth serves gfx906 and gfx1031-gfx1036 on purpose (#7277), while gfx1033 (Van Gogh) installs ROCm wheels and then computes wrong answers (studio/ROCM_RDNA2_APU.md). PRESENCE, not selection: picking the runtime's GPU needs the mask layering _runtime_gfx_target() implements, so a mixed host takes the cpu index and keeps the UNSLOTH_TORCH_INDEX_URL escape hatch. Inline, not a helper: harnesses extract get_torch_index_url alone. "physical" mode so no override hides the silicon; KFD first, since amdkfd writes gfx_target_version from the kernel.
        _amd_gfx_gate_probe=$(_probe_amd_gfx_arch physical 2>/dev/null || true)
        [ -n "$_amd_gfx_gate_probe" ] || _amd_gfx_gate_probe=$(_kfd_gfx_targets 2>/dev/null || true)
        if [ -z "$_amd_gfx_gate_probe" ] && [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
            # HSA_OVERRIDE_GFX_VERSION only, not UNSLOTH_ROCM_GFX_ARCH (a declared arch for #7301 hosts).
            echo "[WARN] HSA_OVERRIDE_GFX_VERSION is set and this host cannot confirm its real arch (no unspoofed rocminfo, amd-smi or KFD topology)." >&2
            echo "[WARN] Installing CPU-only PyTorch rather than trusting the spoofed name: gfx1033 (Van Gogh) computes incorrect results under ROCm (studio/ROCM_RDNA2_APU.md)." >&2
            echo "[WARN] Unset HSA_OVERRIDE_GFX_VERSION so the arch can be read, or pin UNSLOTH_TORCH_INDEX_URL to choose deliberately." >&2
            echo "$_base/cpu"; return
        fi
        [ -n "$_amd_gfx_gate_probe" ] || _amd_gfx_gate_probe="$_amd_gfx_probe"
        _amd_gfx_tokens=" $(printf '%s\n' "$_amd_gfx_gate_probe" | sed 's/:.*$//' \
            | tr '[:upper:]' '[:lower:]' | tr '\n' ' ')"
        _amd_gfx_bad_arch=false
        case "$_amd_gfx_tokens" in
            *" gfx1033 "*) _amd_gfx_bad_arch=true ;;
        esac
        case "${_AMD_REQUEST_TARGET_GFX:-}" in
            ""|gfx1033) : ;;
            *)
                # if/fi, not `[ ... ] &&`: a failing last test would abort under set -e.
                if [ "${_AMD_REQUEST_TARGET_SOURCE:-}" = probe ]; then
                    _amd_gfx_bad_arch=false
                fi
                ;;
        esac
        if [ "$_amd_gfx_bad_arch" = true ]; then
            echo "[WARN] AMD gfx1033 (Van Gogh) computes incorrect results under ROCm -- installing CPU-only PyTorch." >&2
            echo "[WARN] ROCm wheels install on it but training diverges to NaN and gradcheck fails; forward math is fine." >&2
            echo "[WARN] Details: studio/ROCM_RDNA2_APU.md. Override with UNSLOTH_TORCH_INDEX_URL if you want ROCm anyway." >&2
            echo "$_base/cpu"; return
        fi
        # end of the miscomputing-arch gate -- tests/sh/test_rocm_bad_arch_gate.sh lifts
        # the block between the header comment above and this line, so keep both exact.
        _rocm_tag=""
        _rocm_tag=$(_detect_rocm_version_tag) || _rocm_tag=""
        case "$_rocm_tag" in
            rocm[1-9]*.[0-9]*) : ;;
            *) _rocm_tag="" ;;
        esac
        if [ -n "$_rocm_tag" ]; then
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
            # 6.5+ clips to rocm6.4, 7.3+ caps to rocm7.2. Leading ( on every arm: bash 3.2 ends $(...) at a bare `)`.
            _rocm_index=$(case "$_rocm_tag" in
                (rocm6.0|rocm6.0.*) echo "$_base/rocm6.0" ;;
                (rocm6.1|rocm6.1.*) echo "$_base/rocm6.1" ;;
                (rocm6.2|rocm6.2.*) echo "$_base/rocm6.2" ;;
                (rocm6.3|rocm6.3.*) echo "$_base/rocm6.3" ;;
                (rocm6.4|rocm6.4.*) echo "$_base/rocm6.4" ;;
                (rocm7.0|rocm7.0.*) echo "$_base/rocm7.0" ;;
                (rocm7.1|rocm7.1.*) echo "$_base/rocm7.1" ;;
                (rocm7.2|rocm7.2.*) echo "$_base/rocm7.2" ;;
                (rocm6.*)
                    echo "$_base/rocm6.4" ;;
                (*)
                    echo "$_base/rocm7.2" ;;
            esac)
            _rocm_leaf=${_rocm_index##*/}
            if [ "$_rocm_tag" != "$_rocm_leaf" ]; then
                echo "[INFO] No validated PyTorch for ROCm ${_rocm_tag#rocm}; capping to the $_rocm_leaf index (its wheels bundle their own runtime, so this is expected)." >&2
            fi
            echo "$_rocm_index"
            return
        fi
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
            echo "[WARN] ${ROCM_PATH:-/opt/rocm} exists, so ROCm is likely installed but not reporting a version this installer can read." >&2
            echo "[WARN] Pin the wheels and re-run: UNSLOTH_TORCH_INDEX_FAMILY=rocm6.4   (a PyTorch wheel leaf: rocm6.0-6.4, rocm7.0-7.2)" >&2
        else
            echo "[WARN] Install the ROCm/HIP SDK, then re-run this installer:" >&2
            echo "[WARN]   $(_rocm_sdk_install_hint)" >&2
        fi
        echo "[WARN] Version sources checked: amd-smi, /opt/rocm/.info/version, hipconfig, dpkg, rpm (Debian runtime package: libhsa-runtime64-1)." >&2
        echo "$_base/cpu"; return
    fi
    _smi_rc=0
    _smi_out=$(export LC_ALL=C; _run_bounded "$_smi" 2>/dev/null) || _smi_rc=$?
    if [ "$_smi_rc" = "124" ]; then
        echo "[INFO] nvidia-smi did not answer within 10s; retrying with a 45s limit..." >&2
        _smi_rc=0
        _smi_out=$(export LC_ALL=C; _run_bounded --secs 45 "$_smi" 2>/dev/null) || _smi_rc=$?
        [ "$_smi_rc" = "124" ] && _smi=""
    fi
    _cuda_ver=$(printf '%s\n' "$_smi_out" \
        | sed -n \
            -e 's/.*CUDA UMD Version:[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
            -e 's/.*CUDA Version:[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | head -1)
    _inventory_caps=""
    _cuda_from_driver=""
    if [ -n "${UNSLOTH_PYTORCH_MIRROR:-}" ]; then _pin_hint="UNSLOTH_TORCH_INDEX_FAMILY="
    else _pin_hint="UNSLOTH_TORCH_INDEX_URL=$_base/"
    fi
    if [ -z "$_cuda_ver" ]; then
        if _inventory=$(_nvidia_library_inventory) && [ -n "$_inventory" ]; then
            _cuda_ver=${_inventory%% *}
            _inventory_caps=$(printf '%s' "${_inventory#* }" | tr ',' '\n')
        elif _cuda_ver=$(_nvidia_driver_cuda_version) && [ -n "$_cuda_ver" ]; then
            _cuda_from_driver=1
        else
            echo "[WARN] Could not determine CUDA version from nvidia-smi, defaulting to cu126" >&2
            echo "[WARN] cu126 has no kernels for Blackwell (sm_100 / sm_120). To choose the wheel yourself, re-run with" >&2
            echo "[WARN]   ${_pin_hint}cu128   (or cu130 on a driver that supports CUDA 13)" >&2
            echo "$_base/cu126"; return
        fi
    fi
    _major=${_cuda_ver%%.*}
    _minor=${_cuda_ver#*.}
    if [ "$_major" -ge 13 ]; then _cuda_tag=cu130
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 8 ]; then _cuda_tag=cu128
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 6 ]; then _cuda_tag=cu126
    elif [ "$_major" -ge 12 ]; then _cuda_tag=cu124
    elif [ "$_major" -ge 11 ]; then _cuda_tag=cu118
    else echo "$_base/cpu"; return; fi
    _cuda_tag=$(_cap_cuda_family_for_pre_turing "$_cuda_tag" "$_smi" "$_inventory_caps")
    if [ -n "$_cuda_from_driver" ]; then
        echo "[WARN] nvidia-smi and the NVIDIA driver libraries did not answer in time; the driver supports CUDA $_cuda_ver." >&2
        echo "[WARN] Selecting the $_cuda_tag PyTorch wheels from the driver version alone. If that is wrong for this GPU, re-run with" >&2
        echo "[WARN]   ${_pin_hint}cu126   (Maxwell to Hopper, sm_50-90)" >&2
        echo "[WARN]   ${_pin_hint}cu128   (Turing and newer, including Blackwell)" >&2
    fi
    echo "$_base/$_cuda_tag"
}

# ── Torch flavor helpers (to repair a stale CPU / wrong-CUDA wheel) ──
# The xpu arm is load-bearing: without it +xpu reads as cpu and reinstalls every run.
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

_torch_index_url_leaf() {
    _tl_u="${1%%\?*}"
    _tl_u="${_tl_u%%#*}"
    while [ -n "$_tl_u" ] && [ "${_tl_u%/}" != "$_tl_u" ]; do
        _tl_u="${_tl_u%/}"
    done
    printf '%s' "${_tl_u##*/}" | tr '[:upper:]' '[:lower:]'
}

_torch_index_url_is_rocm() {
    case "$(_torch_index_url_leaf "${1:-}")" in
        rocm*|gfx*) return 0 ;;
        *)          return 1 ;;
    esac
}

# HIP reads CUDA_VISIBLE_DEVICES when HIP_VISIBLE_DEVICES is unset, so "" hides the AMD card.
_warn_if_cuda_mask_hides_amd() {
    _cvd_hides_nvidia || return 0
    [ -z "${HIP_VISIBLE_DEVICES+x}" ] || return 0
    _torch_index_url_is_rocm "${1:-}" || return 0
    _amd_gpu_present_via_pci || return 0
    ( unset CUDA_VISIBLE_DEVICES; _has_usable_nvidia_gpu ) >/dev/null 2>&1 || return 0
    echo "" >&2
    echo "[WARN] CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" hid the NVIDIA GPU, so ROCm torch was selected." >&2
    echo "[WARN] ROCm reads CUDA_VISIBLE_DEVICES too: left set, it also hides the AMD GPU and" >&2
    echo "[WARN] Unsloth Studio / Desktop will run CPU-only. Unset it before launching" >&2
    echo "[WARN] (HIP_VISIBLE_DEVICES=0 picks the AMD card), and reinstall with" >&2
    echo "[WARN] UNSLOTH_FORCE_ROCM_TORCH=1 rather than the mask to ask for ROCm torch." >&2
    echo "" >&2
}

_is_pip_rocm_family_leaf() {
    case "$1" in
        gfx[0-9]*) return 0 ;;
        rocm[0-9]*)
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

_cu130_torch213_route() {
    [ "$(_cu130_torch213_platform "$1")" = "yes" ] || { echo "no"; return; }
    _pypi_unsloth_admits_torch "2.13.0"
}

_cu130_torch213_platform() {
    [ "$(_torch_index_url_leaf "$1")" = "cu130" ] || { echo "no"; return; }
    case "$OS" in linux|wsl) ;; *) echo "no"; return ;; esac
    case "$_ARCH" in x86_64|amd64) ;; *) echo "no"; return ;; esac
    _ctr_py=$("$VENV_DIR/bin/python" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || echo "")
    [ "$_ctr_py" = "3.13" ] && echo "yes" || echo "no"
}

# `studio update` runs the installed release's setup, which would downgrade a newer torch;
# any failure answers "no".
_pypi_unsloth_admits_torch() {
    _pua_off=$(printf '%s' "${UV_OFFLINE:-}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    case "$_pua_off" in 1|t|true|y|yes|on) echo "no"; return ;; esac
    if [ -n "${UV_EXCLUDE_NEWER:-}${UV_EXCLUDE_NEWER_PACKAGE:-}" ] || _mirror_configured uv; then
        echo "no"
        return
    fi
    _pua_url="${UNSLOTH_PYPI_JSON_URL:-https://pypi.org/pypi/unsloth/json}"
    _pua_out=$(_run_bounded --secs 20 "$VENV_DIR/bin/python" - "$_pua_url" "$1" 2>/dev/null <<'PY' || true
import json, re, sys, urllib.request
url, want = sys.argv[1], sys.argv[2]
def rel(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) + (0,) * (3 - len(re.findall(r"\d+", v)[:3]))
try:
    with urllib.request.urlopen(url, timeout = 15) as r:
        reqs = json.load(r)["info"].get("requires_dist") or []
    specs = [q for q in reqs if re.match(r"^torch\s*[<>=!~(]", q) and "extra ==" not in q]
    if len(specs) != 1:
        raise ValueError(specs)
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
           ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    body = specs[0].split(";", 1)[0][len("torch"):].strip().strip("()")
    ok = True
    for part in filter(None, (p.strip() for p in body.split(","))):
        m = re.fullmatch(r"(<=|>=|==|!=|<|>)\s*([0-9][0-9.]*)", part)
        if m is None:
            raise ValueError(part)
        ok = ok and ops[m.group(1)](rel(want), rel(m.group(2)))
    print("yes" if ok else "no")
except Exception:
    print("no")
PY
)
    [ "$(printf '%s' "$_pua_out" | tail -n 1)" = "yes" ] && echo "yes" || echo "no"
}

# torchaudio 2.11 is the last release (stable ABI), so newer torch minors pair with it.
_torchaudio_for_torch_minor() {
    if [ "$1" -ge 12 ] 2>/dev/null; then
        echo "torchaudio==2.11.*"
    else
        echo "torchaudio==2.$1.*"
    fi
}

_previous_torch_pin() {
    _ptp_ver="$1"
    _ptp_con="$2"
    [ -n "$_ptp_ver" ] || { echo ""; return; }
    [ "${UNSLOTH_TORCH_UPGRADE:-0}" = "1" ] && [ "${3:-}" != "keep" ] && { echo ""; return; }
    _ptp_base="${_ptp_ver%%+*}"
    case "$_ptp_base" in
        *[!0-9.]* | *..* | .* | *.) echo ""; return ;;
        [0-9]*.[0-9]*) ;;
        *) echo ""; return ;;
    esac
    [ "$(_torch_release_in_window "$_ptp_base" "$_ptp_con")" = "yes" ] || { echo ""; return; }
    echo "torch==$_ptp_base"
}

THEROCK_MIRROR="${UNSLOTH_THEROCK_MIRROR:-https://rocm.nightlies.amd.com/whl-multi-arch/}"

_therock_device_extra_for_gfx() {
    case "$1" in
        gfx1010|gfx1011|gfx1012) echo "device-$1" ;;
        *) return 1 ;;
    esac
}

_torch_spec_with_extra() {
    _tswe_spec="$1"
    [ -n "${_TORCH_EXTRA:-}" ] || { printf '%s' "$_tswe_spec"; return; }
    _tswe_name="${_tswe_spec%%[<>=~!]*}"
    _tswe_rest="${_tswe_spec#"$_tswe_name"}"
    case "$_tswe_name" in
        *"]")
            printf '%s,%s]%s' "${_tswe_name%?}" "$_TORCH_EXTRA" "$_tswe_rest"
            ;;
        *)
            printf '%s[%s]%s' "$_tswe_name" "$_TORCH_EXTRA" "$_tswe_rest"
            ;;
    esac
}

_install_torch_default_index() {
    if [ -n "$_PREV_TORCH_PIN" ]; then
        _itdi_base="${_PREV_TORCH_PIN#torch==}"
        _itdi_minor="${_itdi_base#*.}"
        _itdi_minor="${_itdi_minor%%.*}"
        _itdi_tv="torchvision"
        _itdi_ta="torchaudio"
        case "$_itdi_base" in
            2.*)
                _itdi_tv="torchvision==0.$((_itdi_minor + 15)).*"
                _itdi_ta=$(_torchaudio_for_torch_minor "$_itdi_minor")
                ;;
        esac
        if ! run_install_cmd_retry "install PyTorch (kept release)" uv pip install --python "$_VENV_PY" "$(_torch_spec_with_extra "$TORCH_CONSTRAINT")" "$(_torch_spec_with_extra "$_itdi_tv")" "$_itdi_ta" \
            --default-index "$TORCH_INDEX_URL" "$@"; then
            substep "[WARN] $_PREV_TORCH_PIN is not installable from $(_strip_index_url_credentials "$TORCH_INDEX_URL") -- installing the newest supported release instead" "$C_WARN"
            TORCH_CONSTRAINT="$_PREV_FALLBACK_CONSTRAINT"
            _PREV_TORCH_PIN=""
            run_install_cmd_retry "install PyTorch" uv pip install --python "$_VENV_PY" "$(_torch_spec_with_extra "$TORCH_CONSTRAINT")" "$(_torch_spec_with_extra "$TORCHVISION_CONSTRAINT")" "$TORCHAUDIO_CONSTRAINT" \
                --default-index "$TORCH_INDEX_URL" "$@"
        fi
    else
        run_install_cmd_retry "install PyTorch" uv pip install --python "$_VENV_PY" "$(_torch_spec_with_extra "$TORCH_CONSTRAINT")" "$(_torch_spec_with_extra "$TORCHVISION_CONSTRAINT")" "$TORCHAUDIO_CONSTRAINT" \
            --default-index "$TORCH_INDEX_URL" "$@"
    fi
}

_expected_torch_flavor_tag() {
    _leaf=$(_torch_index_url_leaf "$1")
    case "$_leaf" in
        cu[0-9]*)
            case "${_leaf#cu}" in
                *[!0-9]*) echo "" ;;
                *)        echo "$_leaf" ;;
            esac
            ;;
        cpu)          echo "cpu" ;;
        xpu)          echo "xpu" ;;
        *)
            if _is_pip_rocm_family_leaf "$_leaf"; then echo "rocm"; else echo ""; fi
            ;;
    esac
}

# xpu reads the version from disk: `import torch` can hang on a wedged Intel driver.
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

_torch_index_repairable() {
    _leaf=$(_torch_index_url_leaf "$1")
    case "$_leaf" in
        cu[0-9]*) echo "yes" ;;
        xpu)      echo "yes" ;;
        *)
            if _is_pip_rocm_family_leaf "$_leaf"; then echo "yes"; else echo "no"; fi
            ;;
    esac
}

_strip_index_url_credentials() {
    _sic_url="$1"
    case "$_sic_url" in
        *://*) ;;
        *) printf '%s' "$_sic_url"; return ;;
    esac
    _sic_scheme="${_sic_url%%://*}"
    _sic_rest="${_sic_url#*://}"
    _sic_rest="${_sic_rest%%\?*}"
    _sic_rest="${_sic_rest%%#*}"
    _sic_auth="${_sic_rest%%/*}"
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

    case "$_full_ver" in
        [1-9]*.[0-9]*.[0-9]*) : ;;
        [1-9]*.[0-9]*) : ;;
        *) echo ""; return ;;
    esac
    echo "https://repo.radeon.com/rocm/manylinux/rocm-rel-${_full_ver}/"
}

_RADEON_LISTING=""
_RADEON_PYTAG=""
_RADEON_BASE_URL=""
_RADEON_HOST_ANSWERED=false

_radeon_fetch_listing() {
    _RADEON_BASE_URL="$1"
    _RADEON_PYTAG=$("$_VENV_PY" -c "
import sys
print('cp{}{}'.format(sys.version_info.major, sys.version_info.minor))
" 2>/dev/null) || return 1
    _radeon_http=""
    _radeon_rc=0
    if command -v curl >/dev/null 2>&1; then
        _RADEON_LISTING=$(curl -fsSL --max-time 20 -w '\n%{http_code}' "$_RADEON_BASE_URL" 2>/dev/null) || _radeon_rc=$?
        _radeon_nl='
'
        _radeon_http=${_RADEON_LISTING##*"$_radeon_nl"}
        _RADEON_LISTING=${_RADEON_LISTING%"$_radeon_nl"*}
        _RADEON_LISTING=$(printf '%s' "$_RADEON_LISTING")
    elif command -v wget >/dev/null 2>&1; then
        _RADEON_LISTING=$(wget -qO- --timeout=20 "$_RADEON_BASE_URL" 2>/dev/null) || _radeon_rc=$?
    fi
    [ "$_radeon_rc" -eq 0 ] || _RADEON_LISTING=""
    # curl -f exits 22 on 429/5xx too, and wget's 8 is any error.
    case "$_radeon_http" in
        404|410) [ "$_RADEON_HOST_ANSWERED" = inconclusive ] || _RADEON_HOST_ANSWERED=true ;;
        *) [ -n "$_RADEON_LISTING" ] || _RADEON_HOST_ANSWERED=inconclusive ;;
    esac
    [ -n "$_RADEON_LISTING" ] || return 1
}

_pick_radeon_wheel() {
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

# ROCm-on-WSL bootstrap for AMD Strix Halo (gfx1151): idempotent, best-effort.
_persist_rocm_wsl_dropin() {
    [ -e /opt/rocm/lib/librocdxg.so ] || [ -e /opt/rocm/lib64/librocdxg.so ] || return 0
    _rw_rocm=/opt/rocm
    export HSA_ENABLE_DXG_DETECTION=1
    case ":${PATH}:" in
        *":${_rw_rocm}/bin:"*) ;;
        *) export PATH="${_rw_rocm}/bin:${PATH}" ;;
    esac
    export LD_LIBRARY_PATH="${_rw_rocm}/lib:${LD_LIBRARY_PATH:-}"
    [ -r /etc/profile.d/unsloth-rocm-wsl.sh ] && return 0
    _rw_dropin="$(
        printf '# >>> Unsloth ROCm-on-WSL (gfx1151) >>>\n'
        printf 'export HSA_ENABLE_DXG_DETECTION=1\n'
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
    if _has_usable_nvidia_gpu; then return 0; fi
    _ensure_rocm_probe_env
    if command -v rocminfo >/dev/null 2>&1 && \
       rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx[1-9]/ && !/generic/{found=1} END{exit !found}'; then
        _persist_rocm_wsl_dropin
        return 0
    fi
    [ -e /dev/dxg ] || return 0
    if ! grep -qiE 'Ryzen AI Max|Radeon 80[0-9][05]S|Strix Halo' /proc/cpuinfo 2>/dev/null \
       && ! _wsl_amd_gpu_name >/dev/null 2>&1; then
        return 0
    fi
    command -v bash >/dev/null 2>&1 || return 0

    if [ -e /opt/rocm/lib/librocdxg.so ] || [ -e /opt/rocm/lib64/librocdxg.so ]; then
        if [ -r /etc/profile.d/unsloth-rocm-wsl.sh ]; then
            # shellcheck disable=SC1091
            . /etc/profile.d/unsloth-rocm-wsl.sh || true
        else
            _persist_rocm_wsl_dropin
        fi
        return 0
    fi

    echo ""
    _rw_gpu="$(_wsl_amd_gpu_name 2>/dev/null || true)"; [ -n "$_rw_gpu" ] || _rw_gpu="an AMD GPU"
    substep "Detected ${_rw_gpu} in WSL with no ROCm runtime yet." "$C_WARN"
    substep "Setting up ROCm-on-WSL (ROCm 7.2 + librocdxg) automatically to enable this GPU."
    substep "One-time, uses sudo and a large download. (skip: re-run with UNSLOTH_SKIP_ROCM_WSL_SETUP=1)"

    # PINNED commit, never a branch: this runs unattended with root, so a moving ref would be
    # remote root code. Local copy only for a --local checkout run.
    _ROCM_WSL_HELPER_REF="b1d829182f8c490a326cf7690f156d2de06381f2"
    # librocdxg pin (v1.2.2); kept equal to the helper's defaults (a test enforces that).
    _rw_dxg_ref="${UNSLOTH_LIBROCDXG_REF:-}"
    _rw_dxg_sha="${UNSLOTH_LIBROCDXG_SHA:-}"
    if [ -z "$_rw_dxg_ref" ]; then
        _rw_dxg_ref="4955d12888a3ec57057f1cf8660c2485e415e74c"
        [ -n "$_rw_dxg_sha" ] || _rw_dxg_sha="$_rw_dxg_ref"
    fi
    if [ -n "$_rw_dxg_sha" ]; then
        _rw_dxg_ref="$_rw_dxg_sha"
    fi
    _rw_helper="${_REPO_ROOT:-.}/scripts/install_rocm_wsl_strixhalo.sh"
    _rw_tmp=""
    if [ "$_REPO_IS_CHECKOUT" != "1" ] || [ ! -r "$_rw_helper" ]; then
        # Never a fixed /tmp name: this runs elevated and another user could own it.
        if ! _rw_tmp="$(mktemp 2>/dev/null)" || [ -z "$_rw_tmp" ]; then
            substep "Could not create a private temp file for the ROCm-on-WSL helper; using CPU fallback." "$C_WARN"
            return 0
        fi
        if download "https://raw.githubusercontent.com/unslothai/unsloth/${_ROCM_WSL_HELPER_REF}/scripts/install_rocm_wsl_strixhalo.sh" "$_rw_tmp" 2>/dev/null; then
            _rw_helper="$_rw_tmp"
        else
            substep "Could not fetch the ROCm-on-WSL helper; using CPU fallback." "$C_WARN"
            [ -n "$_rw_tmp" ] && rm -f "$_rw_tmp"
            return 0
        fi
    fi

    # Run only a helper declaring the contract, so a missing pinned ref fails closed.
    if ! grep -q "^UNSLOTH_ROCM_WSL_HELPER_CONTRACT=2$" "$_rw_helper" 2>/dev/null; then
        substep "ROCm-on-WSL helper predates the pinned-source check; using CPU fallback." "$C_WARN"
        [ -n "$_rw_tmp" ] && rm -f "$_rw_tmp"
        return 0
    fi

    # Runs automatically by default (opt out: UNSLOTH_SKIP_ROCM_WSL_SETUP=1). Under Tauri only with
    # UNSLOTH_ROCM_WSL_AUTO=1.
    _rw_go=1
    if [ "${TAURI_MODE:-false}" = "true" ] && [ "${UNSLOTH_ROCM_WSL_AUTO:-0}" != "1" ]; then
        tauri_log "ROCM_WSL_AVAILABLE" "strixhalo"
        substep "Enable the GPU from the desktop app (or set UNSLOTH_ROCM_WSL_AUTO=1)." "$C_WARN"
        _rw_go=0
    fi

    if [ "$_rw_go" = "1" ]; then
        if UNSLOTH_WSL_SMOKE_TEST=0 \
           UNSLOTH_LIBROCDXG_REF="$_rw_dxg_ref" UNSLOTH_LIBROCDXG_SHA="$_rw_dxg_sha" \
           bash "$_rw_helper"; then
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
_torch_index_pinned=false
_ti_url_trim="${UNSLOTH_TORCH_INDEX_URL:-}"
_ti_url_trim="${_ti_url_trim#"${_ti_url_trim%%[![:space:]]*}"}"; _ti_url_trim="${_ti_url_trim%"${_ti_url_trim##*[![:space:]]}"}"
_ti_family_trim="${UNSLOTH_TORCH_INDEX_FAMILY:-}"
_ti_family_trim="${_ti_family_trim#"${_ti_family_trim%%[![:space:]]*}"}"; _ti_family_trim="${_ti_family_trim%"${_ti_family_trim##*[![:space:]]}"}"
if [ -n "$_ti_url_trim" ] || [ -n "$_ti_family_trim" ]; then
    _torch_index_pinned=true
fi
[ "$_torch_index_pinned" = true ] || _maybe_bootstrap_rocm_wsl || true

# A file outlives the command substitution. mktemp -d, never $$: a predictable /tmp path can be
# pre-created as a symlink.
_ROCM_TAG_MEMO_DIR=$(mktemp -d "${TMPDIR:-/tmp}/unsloth-rocm.XXXXXX" 2>/dev/null) \
    && _ROCM_TAG_MEMO="$_ROCM_TAG_MEMO_DIR/tag" || _ROCM_TAG_MEMO=""
_TORCH_EXTRA=""
_te_trim="${UNSLOTH_TORCH_EXTRA:-}"
_te_trim="${_te_trim#"${_te_trim%%[![:space:]]*}"}"; _te_trim="${_te_trim%"${_te_trim##*[![:space:]]}"}"
if [ -n "$_te_trim" ]; then
    if [ "$_torch_index_pinned" = true ]; then
        _TORCH_EXTRA="$_te_trim"
    else
        echo "[WARN] UNSLOTH_TORCH_EXTRA=$_te_trim ignored: it needs UNSLOTH_TORCH_INDEX_URL or _FAMILY set too." >&2
    fi
fi

if [ "$_torch_index_pinned" = false ] && [ "$SKIP_TORCH" = false ]; then
    _has_usable_nvidia_gpu >/dev/null 2>&1 || true
fi
TORCH_INDEX_URL=$(get_torch_index_url)

# Runtime-less reroute to AMD per-arch wheels when the gfx is inferable (unslothai#7301, #7314).
# A readable gfx with an unsupported ROCm version stays on cpu; an unreadable version is rerouted.

_amd_no_rocm_version_reroute=false
_amd_probed_gfx_first=""
# Read before the branches: a closed node is invisible to every probe (#10466).
# `|| true`: a diagnostic must never abort the install under set -e.
_closed_amd_nodes="$(_amd_nodes_closed_to_this_user || true)"
case "$TORCH_INDEX_URL" in
    */cpu)
        if [ "$_torch_index_pinned" = false ] && [ "$SKIP_TORCH" = false ] && \
           [ -z "${UNSLOTH_ROCM_GFX_ARCH:-}" ] && \
           ! _nvidia_gpu_wins_over_amd && _has_amd_rocm_gpu; then
            _amd_probe_out=$(_probe_amd_gfx_arch)
            _amd_spoof_inferred=$(_infer_linux_amd_gfx_arch 2>/dev/null || true)
            _amd_spoof_physical=$(_hsa_spoofed_physical_gfx "$_amd_spoof_inferred" "$_amd_probe_out")
            if [ -n "${_amd_spoof_physical:-}" ]; then
                _amd_probe_out="$_amd_spoof_physical"
            fi
            if [ -z "${_amd_spoof_physical:-}" ] && [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && \
               [ "$(_kfd_gfx_targets | awk 'NF { n++ } END { print n + 0 }')" -gt 1 ]; then
                _amd_probe_out=""
            fi
            _amd_probed_family=$(_amd_agreed_index_family "$_amd_probe_out") \
                || _amd_probed_family=""
            _amd_probed_gfx_first=$(_amd_sole_index_arch "$_amd_probe_out") \
                || _amd_probed_gfx_first=""
            _amd_reroute_target=""
            if _rocm_torch_explicitly_requested && _amd_request_has_a_wheel_route; then
                _amd_reroute_target="${_AMD_REQUEST_TARGET_GFX:-}"
            fi
            _amd_reroute_bad_arch=false
            # A measured-bad arch anywhere disqualifies the whole family (gfx1033 shares gfx103X-all).
            case " $(printf '%s\n' "$_amd_probe_out" | sed 's/:.*$//' \
                     | tr '[:upper:]' '[:lower:]' | tr '\n' ' ')" in
                *" gfx1033 "*) _amd_reroute_bad_arch=true ;;
            esac
            case "$_amd_reroute_target" in
                ""|gfx1033) : ;;
                *)
                    if [ "${_AMD_REQUEST_TARGET_SOURCE:-}" = probe ]; then
                        _amd_reroute_bad_arch=false
                    fi
                    ;;
            esac
            if [ "$_amd_reroute_bad_arch" = true ]; then
                echo "[WARN] AMD gfx1033 (Van Gogh) is in the probed inventory -- not routing torch to the shared $_amd_probed_family index (studio/ROCM_RDNA2_APU.md)." >&2
                _amd_probed_family=""
                _amd_probed_gfx_first=""
            fi
            if [ "$_amd_reroute_bad_arch" = false ] && [ -n "$_amd_reroute_target" ] && \
               [ "${_AMD_REQUEST_TARGET_SOURCE:-}" = probe ]; then
                _amd_target_family=$(_amd_arch_index_family_for_gfx "$_amd_reroute_target") \
                    || _amd_target_family=""
                if [ -n "$_amd_target_family" ]; then
                    _amd_probed_family="$_amd_target_family"
                    _amd_probed_gfx_first="$_amd_reroute_target"
                fi
            fi
            if [ -n "${_amd_probed_family:-}" ] && \
               [ -z "$(_detect_rocm_version_tag 2>/dev/null)" ]; then
                _amd_no_rocm_version_reroute=true
            fi
        fi
        ;;
esac
if [ "$_torch_index_pinned" = false ] && [ "$SKIP_TORCH" = false ] && \
   ! _nvidia_gpu_wins_over_amd && \
   { [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ] || ! _has_amd_rocm_gpu || \
     [ -z "$(_probe_amd_gfx_arch)" ] || \
     [ "${_amd_no_rocm_version_reroute:-false}" = true ]; } && \
   case "$(uname -s)" in Linux) true ;; *) false ;; esac && \
   case "$_ARCH" in x86_64|amd64) true ;; *) false ;; esac; then
    case "$TORCH_INDEX_URL" in
        */cpu)
            _linux_inferred_gfx=$(_infer_linux_amd_gfx_arch 2>/dev/null || true)
            _linux_inferred_gfx=$(_amd_sole_index_arch "$_linux_inferred_gfx") \
                || _linux_inferred_gfx=""
            if [ "${_amd_no_rocm_version_reroute:-false}" = true ]; then
                _linux_inferred_gfx="${_amd_probed_gfx_first:-}"
            fi
            # Re-check the gfx1033 gate here: this reroute can bypass it. Also unset UNSLOTH_ROCM_GFX_ARCH
            # so setup.sh does not request a HIP build. Keyed on the physical inventory.
            _amd_reroute_physical=$(_probe_amd_gfx_arch physical 2>/dev/null || true)
            [ -n "$_amd_reroute_physical" ] || _amd_reroute_physical=$(_kfd_gfx_targets 2>/dev/null || true)
            case " $(printf '%s\n' "$_amd_reroute_physical" | sed 's/:.*$//' \
                     | tr '[:upper:]' '[:lower:]' | tr '\n' ' ')" in
                *" gfx1033 "*) _linux_inferred_gfx="gfx1033" ;;
            esac
            case "${_linux_inferred_gfx%%:*}" in
                gfx1033)
                    echo "[WARN] AMD gfx1033 (Van Gogh) computes incorrect results under ROCm -- keeping CPU-only PyTorch (studio/ROCM_RDNA2_APU.md)." >&2
                    _linux_inferred_gfx=""
                    unset UNSLOTH_ROCM_GFX_ARCH ;;
            esac
            _amd_family=""
            if [ "${_amd_no_rocm_version_reroute:-false}" = true ] && \
               [ -n "$_linux_inferred_gfx" ]; then
                _amd_family="${_amd_probed_family:-}"
            elif [ -n "$_linux_inferred_gfx" ]; then
                _amd_family=$(_amd_arch_index_family_for_gfx "$_linux_inferred_gfx") || _amd_family=""
            elif [ "${_amd_no_rocm_version_reroute:-false}" = true ] && \
                 [ -z "${_amd_probed_gfx_first:-}" ]; then
                _amd_family="${_amd_probed_family:-}"
            fi
            if [ -n "$_amd_family" ]; then
                    _amd_mirror="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}"
                    while [ "${_amd_mirror%/}" != "$_amd_mirror" ]; do
                        _amd_mirror="${_amd_mirror%/}"
                    done
                    TORCH_INDEX_URL="${_amd_mirror}/${_amd_family}/"
                    if [ -n "$_linux_inferred_gfx" ]; then
                        export UNSLOTH_ROCM_GFX_ARCH="$_linux_inferred_gfx"
                    fi
                    if [ "${_amd_no_rocm_version_reroute:-false}" = true ] && \
                       [ -n "${_amd_spoof_physical:-}" ]; then
                        unset HSA_OVERRIDE_GFX_VERSION
                        echo "  [WARN] Clearing HSA_OVERRIDE_GFX_VERSION for the rest of this install:" >&2
                        echo "  [WARN] these wheels carry $_linux_inferred_gfx kernels, so the runtime has" >&2
                        echo "  [WARN] to report the real arch. Remove the export from your shell profile" >&2
                        echo "  [WARN] (~/.bashrc, ~/.profile) as well, or the next terminal restores it." >&2
                    fi
                    case "$_amd_family" in
                        gfx120X-all|gfx1151|gfx1150|gfx1152|gfx103X-all|gfx110X-all)
                            TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
                            TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
                            TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
                            ;;
                    esac
                    echo "" >&2
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

if [ "$_torch_index_pinned" = false ] && [ "$SKIP_TORCH" = false ] && \
   _rocm_torch_explicitly_requested && _has_usable_nvidia_gpu; then
    if ! _torch_index_url_is_rocm "$TORCH_INDEX_URL"; then
        _cuda_fallback_index=$(UNSLOTH_FORCE_ROCM_TORCH=0 get_torch_index_url)
        if [ -n "$_cuda_fallback_index" ] && \
           [ "$_cuda_fallback_index" != "$TORCH_INDEX_URL" ]; then
            echo "[WARN] UNSLOTH_FORCE_ROCM_TORCH is set, but no ROCm wheel index could be selected for this host." >&2
            echo "[WARN] Keeping the CUDA build rather than installing CPU PyTorch on a machine with a working NVIDIA GPU." >&2
            TORCH_INDEX_URL="$_cuda_fallback_index"
        fi
    fi
fi
if [ "$SKIP_TORCH" = false ]; then
    _warn_if_cuda_mask_hides_amd "$TORCH_INDEX_URL"
fi
# Classify on the lowercased FINAL leaf (query, fragment, trailing slashes stripped), in lockstep
# with _torch_index_url_leaf. Unknown leaves leave UNSLOTH_TORCH_BACKEND unset.
_torch_index_leaf="${TORCH_INDEX_URL%%\?*}"
_torch_index_leaf="${_torch_index_leaf%%#*}"
while [ -n "$_torch_index_leaf" ] && [ "${_torch_index_leaf%/}" != "$_torch_index_leaf" ]; do
    _torch_index_leaf="${_torch_index_leaf%/}"
done
_torch_index_leaf="${_torch_index_leaf##*/}"
_torch_index_leaf=$(printf '%s' "$_torch_index_leaf" | tr '[:upper:]' '[:lower:]')
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
    *)          unset UNSLOTH_TORCH_BACKEND ;;
esac

if [ -n "${UNSLOTH_TORCH_BACKEND:-}" ] &&
   { [ "$_torch_backend_was_stated" != true ] ||
     [ "$_torch_backend_stated_value" != "$UNSLOTH_TORCH_BACKEND" ]; }; then
    export UNSLOTH_TORCH_BACKEND_SOURCE="resolved"
else
    unset UNSLOTH_TORCH_BACKEND_SOURCE
fi

if _is_pip_rocm_family_leaf "$_torch_index_leaf"; then
    _torch_index_is_rocm_family=true
else
    _torch_index_is_rocm_family=false
fi

case "$_torch_index_leaf" in
    rocm7.2|gfx120x-all|gfx1151|gfx1150|gfx1152|gfx103x-all|gfx110x-all)
        TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
        TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
        TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
        ;;
    # Floor 2.6: unsloth raises at import for XPU below it.
    xpu)
        TORCH_CONSTRAINT="torch>=2.6,<2.11.0"
        TORCHVISION_CONSTRAINT="torchvision>=0.21,<0.26.0"
        TORCHAUDIO_CONSTRAINT="torchaudio>=2.6,<2.11.0"
        ;;
esac

_amd_gpu_radeon=false
_gfx_rocm64_target=false
_gfx_rocm64_floor_maj=""
_gfx_rocm64_floor_min=""
_amd_arch_index_routed=false
_amd_arch_index_family=""
if [ "$_torch_index_pinned" = false ]; then
case "$_torch_index_leaf" in
    rocm*)
        if _has_amd_rocm_gpu && command -v rocminfo >/dev/null 2>&1 && \
           rocminfo 2>/dev/null | grep -q 'Marketing Name:.*Radeon'; then
            _amd_gpu_radeon=true
        fi
        ;;
esac
_rocm_leaf_below() {
    case "$1" in rocm[0-9]*.[0-9]*) : ;; *) return 1 ;; esac
    _rb=${1#rocm}; _maj=${_rb%%.*}; _min=${_rb#*.}; _min=${_min%%.*}
    case "$_maj$_min" in *[!0-9]*) return 1 ;; esac
    if [ "$_maj" -lt "$2" ]; then return 0; fi
    if [ "$_maj" -eq "$2" ] && [ "$_min" -lt "$3" ]; then return 0; fi
    return 1
}
# Mirrors _installed_rocm_wheel_is_below in studio/install_python_stack.py.
_venv_torch_amd_family() {
    "$1" -c 'import re
from importlib import metadata
try:
    reqs = metadata.requires("rocm") or []
except Exception:
    reqs = []
for r in reqs:
    m = re.search(r"rocm[-_]sdk[-_]libraries[-_]([A-Za-z0-9][A-Za-z0-9._-]*)", r, re.I)
    if m:
        print(re.split(r"[=<>!~;,\[\]()\s]", m.group(1))[0].lower().replace("_", "-"))
        break' 2>/dev/null || true
}

_venv_torch_rocm_below() {
    _vtr_leaf=$("$1" -c 'import re, torch; m = re.search(r"rocm([0-9]+)\.([0-9]+)", getattr(torch, "__version__", "") or ""); print("rocm%s.%s" % m.groups() if m else "")' 2>/dev/null || true)
    [ -n "$_vtr_leaf" ] || return 0
    _rocm_leaf_below "$_vtr_leaf" "$2" "$3"
}

# ── Strix Halo / Strix Point: route to the AMD arch-specific index ───────────
# gfx1151/gfx1150 need torch 2.11+rocm7.13 from repo.amd.com; older generic indexes lack the fixes (#7264).
case "$_torch_index_leaf" in
    rocm[0-9]*)
        # Re-declared because test_rocm_support.py lifts this arm out whole (set -u).
        _gfx_rocm64_target=false
        _gfx_rocm64_floor_maj=""
        _gfx_rocm64_floor_min=""
        _amd_arch_index_routed=false
        _amd_arch_index_family=""
        _gfx_all=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
        _gfx_all=${_gfx_all%%:*}
        _gfx_probe=""
        _gfx_space=hip
        if [ -z "$_gfx_all" ] && command -v rocminfo >/dev/null 2>&1; then
            _gfx_all=$(rocminfo 2>/dev/null | _rocminfo_gpu_records | _gfx_arch_slots || true)
            [ -n "$_gfx_all" ] && { _gfx_probe=rocminfo; _gfx_space=hip; }
        fi
        if [ -z "$_gfx_all" ] && command -v amd-smi >/dev/null 2>&1; then
            _gfx_records=$(amd-smi list 2>/dev/null | _amd_smi_gpu_records || true)
            case "$_gfx_records" in *gfx*) ;; *)
                _gfx_records=$(amd-smi static --asic 2>/dev/null | _amd_smi_gpu_records || true) ;;
            esac
            if [ -n "$_gfx_records" ]; then
                _gfx_smi_out=$(amd-smi list -e 2>/dev/null | _amd_smi_hip_order "$_gfx_records" || true)
                _gfx_space=$(printf '%s\n' "$_gfx_smi_out" | sed -n 1p)
                _gfx_records=$(printf '%s\n' "$_gfx_smi_out" | tail -n +2)
                _gfx_all=$(printf '%s\n' "$_gfx_records" | _gfx_arch_slots || true)
                [ -n "$_gfx_all" ] && _gfx_probe=amd-smi
            fi
        fi
        # A mask hiding every agent still lands here; re-probe unmasked. ${VAR+x}: set-but-empty hides all.
        if [ -z "$_gfx_all" ] && [ -n "${ROCR_VISIBLE_DEVICES+x}${HIP_VISIBLE_DEVICES+x}" ]; then
            if command -v rocminfo >/dev/null 2>&1; then
                _gfx_all=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; rocminfo 2>/dev/null) | _rocminfo_gpu_records | _gfx_arch_slots || true)
                [ -n "$_gfx_all" ] && _gfx_space=hip
            fi
            if [ -z "$_gfx_all" ] && command -v amd-smi >/dev/null 2>&1; then
                _gfx_all=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; amd-smi list 2>/dev/null) | _amd_smi_gpu_records | _gfx_arch_slots || true)
                [ -z "$_gfx_all" ] && \
                    _gfx_all=$( (unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; amd-smi static --asic 2>/dev/null) | _amd_smi_gpu_records | _gfx_arch_slots || true)
                [ -n "$_gfx_all" ] && _gfx_space=discovery
            fi
        fi
        # Correct an HSA_OVERRIDE_GFX_VERSION spoof back to the physical arch first (#7331).
        _spoof_physical=""
        if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && [ -n "$_gfx_all" ]; then
            _spoof_inferred=$(_infer_linux_amd_gfx_arch 2>/dev/null || true)
            _spoof_physical=$(_hsa_spoofed_physical_gfx "$_spoof_inferred" "$_gfx_all")
            [ -n "$_spoof_physical" ] && _gfx_all="$_spoof_physical"
        fi
        _runtime_gfx=""
        _rocr_unresolved=""
        if [ -n "$_gfx_all" ]; then
            # first-set-wins, mirroring _pick_visible_index (and _HIP_LAYER_MASKS) in
            # studio/install_python_stack.py. rocminfo output is ALREADY ROCr-filtered, so
            # indexing by ROCR again shadows CUDA, its HIP alias: ROCR=2,1 + CUDA=1 is survivor 2.
            _vis_masks="HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES"
            if [ "$_gfx_probe" != rocminfo ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ "$ROCR_VISIBLE_DEVICES" != "-1" ]; then
                _rocr_kept=$(printf '%s\n' "$_gfx_all" | awk -v m="$ROCR_VISIBLE_DEVICES" '
                    NF { v[n++] = $0 }
                    END { k = split(m, t, ","); for (i = 1; i <= k; i++) { gsub(/[[:space:]]/, "", t[i]); if (t[i] !~ /^[0-9]+$/) continue; x = t[i] + 0; if (x >= n || (x in s)) break; s[x] = 1; print v[x] } }')
                [ -n "$_rocr_kept" ] && _gfx_all="$_rocr_kept"
                _rocr_unresolved=$(printf '%s' "$ROCR_VISIBLE_DEVICES" | tr -d '0-9, \t')
            fi
            _vis_var=""
            _vis=""
            for _vis_m in $_vis_masks; do
                eval "_vis_m_set=\${$_vis_m+x}"
                if [ -n "$_vis_m_set" ]; then
                    _vis_var="$_vis_m"
                    eval "_vis=\$$_vis_m"
                    break
                fi
            done
            _idx=0
            if [ -n "$_vis" ] && [ "$_vis" != "-1" ]; then
                _first=${_vis%%,*}
                case "$_first" in
                    ''|*[!0-9]*) _idx=0 ;;
                    *) _idx=$_first ;;
                esac
            fi
            _runtime_gfx=$(printf '%s\n' "$_gfx_all" | awk -v idx="$_idx" '
                NF { vals[n++] = $0 }
                END {
                    if (idx < 0 || idx >= n) idx = 0
                    if (n > 0) print vals[idx]
                }')
            if [ "$_gfx_space" != hip ] && \
               [ "$(printf '%s\n' "$_gfx_all" | awk 'NF && !seen[$0]++ { n++ } END { print n + 0 }')" -gt 1 ]; then
                echo "" >&2
                echo "  [WARN] amd-smi lists unlike adapters in discovery order and \`amd-smi list -e\`" >&2
                echo "  [WARN] returned no HIP_ID map, so no ordinal here names a known device." >&2
                echo "  [WARN] Skipping arch-specific torch routing. Set UNSLOTH_ROCM_GFX_ARCH to" >&2
                echo "  [WARN] name the target explicitly." >&2
                echo "" >&2
                _runtime_gfx=""
            elif [ -n "${_rocr_unresolved:-}" ] && \
               [ "$(printf '%s\n' "$_gfx_all" | awk 'NF && !seen[$0]++ { n++ } END { print n + 0 }')" -gt 1 ]; then
                echo "" >&2
                echo "  [WARN] ROCR_VISIBLE_DEVICES selects a GPU by UUID, which amd-smi output cannot map to a" >&2
                echo "  [WARN] position, and the adapters differ. Skipping arch-specific torch routing." >&2
                echo "  [WARN] Set UNSLOTH_ROCM_GFX_ARCH to name the target explicitly." >&2
                echo "" >&2
                _runtime_gfx=""
            fi
        fi
        _gfx906_env=$(printf '%s' "${UNSLOTH_ROCM_GFX_ARCH:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
        _gfx906_env=${_gfx906_env%%:*}
        _strix_gfx=""
        if [ "$_gfx906_env" != "gfx906" ]; then
            case "$_runtime_gfx" in
                gfx1151|gfx1150|gfx1152) _strix_gfx="$_runtime_gfx" ;;
            esac
        fi
        if [ -n "$_strix_gfx" ] && _rocm_leaf_below "$_torch_index_leaf" 7 13; then
            echo "" >&2
            echo "  [WARN] $_strix_gfx (Strix) detected -- routing to the AMD arch-specific index" >&2
            echo "  [WARN] torch 2.11+rocm7.13 has AMD's real gfx1150/gfx1151 fixes (the ROCm 7.1" >&2
            echo "  [WARN] _grouped_mm segfault, moe_utils.py:167, and later Strix kernel bugs)," >&2
            echo "  [WARN] and is more reliable than the rocm7.2 index or an offline Radeon repo." >&2
            echo "" >&2
            _amd_strix_base="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}"
            while [ "${_amd_strix_base%/}" != "$_amd_strix_base" ]; do
                _amd_strix_base="${_amd_strix_base%/}"
            done
            TORCH_INDEX_URL="${_amd_strix_base}/${_strix_gfx}/"
            _amd_arch_index_routed=true
            _amd_arch_index_family="$_strix_gfx"
            TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
            # Pin companions to 2.11 (per-gfx index publishes them independently).
            TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
            TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
            _amd_gpu_radeon=false
            # Clear the spoof only when native wheels are really going in.
            # Mirrors _clear_confirmed_hsa_spoof in studio/install_python_stack.py.
            if [ -n "$_spoof_physical" ] && [ "$SKIP_TORCH" = false ]; then
                unset HSA_OVERRIDE_GFX_VERSION
                echo "  [WARN] Clearing HSA_OVERRIDE_GFX_VERSION for the rest of this install:" >&2
                echo "  [WARN] the $_strix_gfx wheels carry $_strix_gfx kernels, so the runtime has" >&2
                echo "  [WARN] to report the real arch. Remove the export from your shell profile" >&2
                echo "  [WARN] (~/.bashrc, ~/.profile) as well, or the next terminal restores it." >&2
            fi
        fi
        # RDNA 4 generic wheels below 7.13 have a null HIP _grouped_mm (TheRock #5284).
        _rdna4_gfx=""
        if [ "$_gfx906_env" != "gfx906" ]; then
            case "$_runtime_gfx" in
                gfx1200|gfx1201) _rdna4_gfx="$_runtime_gfx" ;;
            esac
        fi
        # gfx120X-all publishes cp310+ only.
        if [ -n "$_rdna4_gfx" ] && [ "$("${VENV_DIR:-}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)" = "3.9" ]; then
            _rdna4_gfx=""
        fi
        if [ -n "$_rdna4_gfx" ] && _rocm_leaf_below "$_torch_index_leaf" 7 13; then
            echo "" >&2
            echo "  [WARN] $_rdna4_gfx (RDNA 4) detected -- routing to the AMD arch-specific index" >&2
            echo "  [WARN] torch 2.11+rocm7.13 fixes the RDNA 4 _grouped_mm kernel that the" >&2
            echo "  [WARN] $_torch_index_leaf wheels lack, so training does not fall back to a slow path." >&2
            echo "" >&2
            _amd_rdna4_base="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}"
            while [ "${_amd_rdna4_base%/}" != "$_amd_rdna4_base" ]; do
                _amd_rdna4_base="${_amd_rdna4_base%/}"
            done
            # Literal, not _amd_arch_index_family_for_gfx: tests lift this arm out whole.
            TORCH_INDEX_URL="${_amd_rdna4_base}/gfx120X-all/"
            _amd_arch_index_routed=true
            _amd_arch_index_family="gfx120x-all"
            TORCH_CONSTRAINT="torch>=2.11.0,<2.12.0"
            TORCHVISION_CONSTRAINT="torchvision>=0.26.0,<0.27.0"
            TORCHAUDIO_CONSTRAINT="torchaudio>=2.11.0,<2.12.0"
            _amd_gpu_radeon=false
            _torch_index_leaf="gfx120x-all"
        fi
        # Navi 33 (gfx1102) floors at rocm6.3; gfx1200/gfx1201 at 6.4, matching
        # _GENERIC_WHEEL_GFX_MIN_ROCM in studio/install_python_stack.py (keep the two tables in step).
        case "$_runtime_gfx" in
            gfx1102)          _gfx_rocm64_floor_maj=6; _gfx_rocm64_floor_min=3 ;;
            gfx1200|gfx1201)  _gfx_rocm64_floor_maj=6; _gfx_rocm64_floor_min=4 ;;
        esac
        if [ -n "$_gfx_rocm64_floor_maj" ]; then
            _gfx_rocm64_target=true
            _amd_gpu_radeon=false
            _gfx_rocm64_tag="rocm${_gfx_rocm64_floor_maj}.${_gfx_rocm64_floor_min}"
            if _rocm_leaf_below "$_torch_index_leaf" "$_gfx_rocm64_floor_maj" "$_gfx_rocm64_floor_min"; then
                echo "" >&2
                echo "  [WARN] $_runtime_gfx detected -- routing torch to $_gfx_rocm64_tag because older" >&2
                echo "  [WARN] generic ROCm wheel families do not ship $_runtime_gfx kernels." >&2
                echo "" >&2
                _amd_rocm64_base="${UNSLOTH_PYTORCH_MIRROR:-https://download.pytorch.org/whl}"
                while [ "${_amd_rocm64_base%/}" != "$_amd_rocm64_base" ]; do
                    _amd_rocm64_base="${_amd_rocm64_base%/}"
                done
                TORCH_INDEX_URL="${_amd_rocm64_base}/${_gfx_rocm64_tag}"
                _torch_index_leaf="$_gfx_rocm64_tag"
            fi
        fi
        # ── MI50 / Radeon VII (gfx906, Vega 20): legacy community-supported path ──
        # Newer rocm wheel families bundle ROCm libraries whose Tensile kernels dropped gfx906 (rocBLAS "TensileLibrary.dat ... not read for gfx906", ROCm/TheRock#1844), so a rocm6.4+/7.x index installs a torch that fails at the first BLAS call. The rocm6.3 index is the last one whose wheels run on gfx906 (torch 2.7.0 verified on MI50 32GB; up to 2.9 in community use), so reroute any newer picked index and leave rocm6.0-6.3 alone. Target resolution: an explicit UNSLOTH_ROCM_GFX_ARCH wins, letting a host whose rocminfo/amd-smi emit no gfx token still opt in. Otherwise only treat gfx906 as the target when it is the SOLE distinct arch present: _gfx_all is de-duplicated by visible index, which loses per-device ordinals on a mixed host, so a non-gfx906 selection must never be downgraded to rocm6.3.
        _gfx906_target=false
        if [ -n "$_gfx906_env" ]; then
            [ "$_gfx906_env" = "gfx906" ] && _gfx906_target=true
        elif [ -n "$_gfx_all" ]; then
            _gfx906_uniq=$(printf '%s\n' "$_gfx_all" | awk 'NF && !seen[$0]++')
            [ "$_gfx906_uniq" = "gfx906" ] && _gfx906_target=true
        fi
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
            TORCH_CONSTRAINT="torch>=2.4,<2.11.0"
            TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.26.0"
            TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<2.11.0"
        fi
        ;;
esac
fi  # _torch_index_pinned guard (Radeon + Strix reroute)
# New installs on this route get 2.13; preservation keeps 2.4-2.14 installs on their release.
_PRESERVE_TORCH_CONSTRAINT="$TORCH_CONSTRAINT"
_CU130_NEW_INSTALL_ROUTE=false
if [ "$SKIP_TORCH" = false ] && [ "$(_cu130_torch213_platform "$TORCH_INDEX_URL")" = "yes" ]; then
    _PRESERVE_TORCH_CONSTRAINT="torch>=2.4,<${_CU130_TORCH_CEILING}"
    # An existing home never takes 2.13 unasked.
    if { [ "$_EXISTING_INSTALL" = false ] || [ "${UNSLOTH_TORCH_UPGRADE:-0}" = "1" ]; } \
       && [ "$(_pypi_unsloth_admits_torch "2.13.0")" = "yes" ]; then
        _CU130_NEW_INSTALL_ROUTE=true
    fi
fi
_PREV_TORCH_PIN=""
_PREV_FALLBACK_CONSTRAINT="$TORCH_CONSTRAINT"
if [ "$SKIP_TORCH" = false ]; then
    _prev_pin=$(_previous_torch_pin "$_PREV_TORCH_VER" "$_PRESERVE_TORCH_CONSTRAINT")
    if [ -z "$_prev_pin" ] && [ "${UNSLOTH_TORCH_UPGRADE:-0}" = "1" ] \
       && [ "$_CU130_NEW_INSTALL_ROUTE" = false ] && [ -n "$_PREV_TORCH_VER" ] \
       && [ "$(_torch_release_in_window "${_PREV_TORCH_VER%%+*}" "$TORCH_CONSTRAINT")" != "yes" ]; then
        _prev_pin=$(_previous_torch_pin "$_PREV_TORCH_VER" "$_PRESERVE_TORCH_CONSTRAINT" keep)
    fi
    if [ -n "$_prev_pin" ]; then
        _PREV_TORCH_PIN="$_prev_pin"
        TORCH_CONSTRAINT="$_prev_pin"
        substep "existing install has torch $_PREV_TORCH_VER -- keeping it (set UNSLOTH_TORCH_UPGRADE=1 to get the newest release)"
    fi
fi
if [ -z "$_PREV_TORCH_PIN" ] && [ "$_CU130_NEW_INSTALL_ROUTE" = true ]; then
    TORCH_CONSTRAINT="$_CU130_NEW_INSTALL_TORCH"
    TORCHVISION_CONSTRAINT="torchvision>=0.28.0,<0.29.0"
fi

_TAURI_TORCH_INDEX_FAMILY=$(_tauri_torch_index_family "$TORCH_INDEX_URL")
if [ "$_amd_gpu_radeon" = true ] && [ "$SKIP_TORCH" = false ]; then
    _TAURI_TORCH_INDEX_FAMILY="radeon"
fi
_TAURI_GPU_BRANCH=$(_tauri_gpu_branch "$_TAURI_TORCH_INDEX_FAMILY" "$_amd_gpu_radeon")
tauri_diag_marker "$_TAURI_GPU_BRANCH" "$_TAURI_TORCH_INDEX_FAMILY"


# ── GPU detection summary (mirrors install.ps1 step "gpu" block) ──
# Asked of the resolved index; a pin names a wheel family, not a card.
if _has_usable_nvidia_gpu && \
   { [ "$_torch_index_pinned" = true ] || ! _torch_index_url_is_rocm "$TORCH_INDEX_URL"; }; then
    _nv_banner_fields
    if [ -n "$_nv_name" ] && [ -n "$_nv_sm" ]; then
        step "gpu" "$_nv_name ($_nv_sm)"
    elif [ -n "$_nv_name" ]; then
        step "gpu" "$_nv_name"
    else
        step "gpu" "NVIDIA GPU detected"
    fi
    # An `if`, not `[ ... ] && substep ...`: the AND-list form leaves a non-zero status
    # behind on the common path where there is no driver string to print.
    if [ -n "$_nv_driver" ]; then substep "Driver: $_nv_driver"; fi
elif _torch_index_url_is_rocm "$TORCH_INDEX_URL"; then
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
            _gpu_disp_smi_space=$(printf '%s\n' "$_gpu_disp_smi_out" | sed -n 1p)
            _gpu_disp_smi_records=$(printf '%s\n' "$_gpu_disp_smi_out" | tail -n +2)
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
            'NF { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx+0] }')
        _gpu_disp_gfx=${_gpu_disp_record%%|*}
        _gpu_disp_mkt=${_gpu_disp_record#*|}
    fi
    # Only pre-TARGET_GRAPHICS_VERSION amd-smi lands here: names but no arch in the record.
    if [ -z "$_gpu_disp_gfx" ]; then
        _gpu_disp_gfx=$(printf '%s\n' "$_gpu_disp_gfx_all" | awk -v idx="$_gpu_vis_idx" \
            'NF && !seen[$0]++ { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx+0] }')
    fi
    # UNSLOTH_ROCM_GFX_ARCH env override (mirrors install.ps1)
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        _gpu_disp_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
        substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $_gpu_disp_gfx"
    elif [ -z "$_gpu_disp_gfx" ] && [ -n "$_gpu_disp_mkt" ]; then
        # In sync with install.ps1 nameArchTable; gfx1102 before gfx1100 ("RX 7700S").
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
            *"RX 6950"*|*"RX 6900"*|*"RX 6850"*|*"RX 6800"*|*"RX 6750"*|*"RX 6700"*|*"PRO W6800"*|*"PRO W6900"*) _gpu_disp_gfx="gfx1030" ;;  # RDNA 2 (Navi 21)
            *"RX 6650"*|*"RX 6600"*|*"PRO W6600"*|*"PRO W6650"*)                                            _gpu_disp_gfx="gfx1032" ;;  # RDNA 2 (Navi 23)
            *"RX 6550"*|*"RX 6500"*|*"RX 6450"*|*"RX 6400"*|*"RX 6300"*|*"PRO W6400"*|*"PRO W6500"*|*"PRO W6300"*)                    _gpu_disp_gfx="gfx1034" ;;  # RDNA 2 (Navi 24)
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
    if [ -n "$_gpu_disp_mkt" ] && [ -n "$_gpu_disp_gfx" ]; then
        step "gpu" "$_gpu_disp_mkt ($_gpu_disp_gfx)"
    elif [ -n "$_gpu_disp_mkt" ]; then
        step "gpu" "$_gpu_disp_mkt"
    elif [ -n "$_gpu_disp_gfx" ]; then
        step "gpu" "AMD ROCm ($_gpu_disp_gfx)"
    else
        step "gpu" "AMD ROCm"
    fi
    _rocm_root="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
    if [ -d "$_rocm_root" ]; then
        substep "ROCm: $_rocm_root"
    else
        substep "ROCm: runtime detected (no SDK tree at $_rocm_root)"
    fi
    [ -n "$_gpu_rocm_ver" ] && substep "hipconfig: $_gpu_rocm_ver"
elif [ "$OS" = "macos" ] && [ "$_ARCH" = "arm64" ]; then
    step "gpu" "Apple Silicon (Metal, unified memory)"
elif _has_amd_rocm_gpu; then
    if [ "$_torch_index_pinned" = true ]; then
        step "gpu" "AMD GPU (torch index pinned: $_torch_index_leaf)" "$C_WARN"
    else
        step "gpu" "AMD GPU (no usable ROCm -- CPU fallback)" "$C_WARN"
    fi
else
    step "gpu" "none (CPU-only)" "$C_WARN"
fi

case "$TORCH_INDEX_URL" in
    */cpu)
        if [ "$SKIP_TORCH" = false ] && [ "$OS" != "macos" ]; then
            if [ "$_torch_index_pinned" = true ]; then
                substep "CPU-only PyTorch (index pinned via UNSLOTH_TORCH_INDEX_URL / _FAMILY)."
            elif _has_amd_rocm_gpu; then
                _covered_disp_gfx=$(_infer_linux_amd_gfx_arch 2>/dev/null) || _covered_disp_gfx=""
                if [ -n "$_covered_disp_gfx" ] && _amd_arch_index_family_for_gfx "$_covered_disp_gfx" >/dev/null 2>&1; then
                    _unsup_disp_gfx=""
                else
                    _unsup_disp_gfx=$(_infer_linux_unsupported_amd_gfx_arch 2>/dev/null) || _unsup_disp_gfx=""
                fi
                if [ -n "$_unsup_disp_gfx" ]; then
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
# Host properties, gated on the route so hybrid NVIDIA hosts are not told to install ROCm.
# Two sibling arms (absent vs closed /dev/kfd); harnesses lift this block by walking back to the `if`.

# Whether the TORCH this run installs can open an AMD device node, which only a ROCm wheel does.
_torch_opens_amd_nodes() {
    [ "$SKIP_TORCH" = true ] && return 1
    _toan_leaf=$(_torch_index_url_leaf "${TORCH_INDEX_URL:-}")
    case "$_toan_leaf" in
        rocm-rel-*[!0-9.]*) return 1 ;;
        rocm-rel-[0-9]*) return 0 ;;
    esac
    _is_pip_rocm_family_leaf "$_toan_leaf" && return 0
    return 1
}

_auto_bundle_opens_amd_nodes() {
    case "$(_requested_llama_backend)" in
        cpu|cuda|rocm|hip|vulkan) return 0 ;;
    esac
    if [ -z "${_amd_auto_nvidia_cached:-}" ]; then
        if _has_usable_nvidia_gpu; then
            _amd_auto_nvidia_cached=yes
        else
            _amd_auto_nvidia_cached=no
        fi
    fi
    [ "$_amd_auto_nvidia_cached" = yes ] && return 1
    return 0
}

# Resolved like llama_backend.py::environment_backend_override: a recognised
# UNSLOTH_LLAMA_CPP_BACKEND wins, else legacy UNSLOTH_FORCE_VULKAN applies.
_requested_llama_backend() {
    _rlb=$(printf '%s' "${UNSLOTH_LLAMA_CPP_BACKEND:-}" | awk '{$1=$1; print tolower($0)}')
    case "$_rlb" in
        cpu|cuda|rocm|hip|vulkan|auto) printf '%s\n' "$_rlb"; return 0 ;;
    esac
    case "$(printf '%s' "${UNSLOTH_FORCE_VULKAN:-}" | awk '{$1=$1; print tolower($0)}')" in
        1|true|yes|on) printf '%s\n' vulkan; return 0 ;;
    esac
    printf '%s\n' "$_rlb"
}

_run_may_open_kfd() {
    _torch_opens_amd_nodes && return 0
    case "$(_requested_llama_backend)" in
        vulkan|cpu|cuda) return 1 ;;
    esac
    _auto_bundle_opens_amd_nodes || return 1
    return 0
}

_run_may_open_a_gpu_node() {
    _torch_opens_amd_nodes && return 0
    case "$(_requested_llama_backend)" in
        cpu|cuda) return 1 ;;
    esac
    _auto_bundle_opens_amd_nodes || return 1
    return 0
}

# Leaf through _is_pip_rocm_family_leaf (rejects custom pins); repo.radeon.com handled apart.
# Guarded: a diagnostic may not abort under set -e.
_amd_node_diag_leaf=$(_torch_index_url_leaf "$TORCH_INDEX_URL" || true)
case "$_amd_node_diag_leaf" in
    rocm-rel-*[!0-9.]*) _amd_node_diag_route=false ;;
    cpu|rocm-rel-[0-9]*) _amd_node_diag_route=true ;;
    *)
        if _is_pip_rocm_family_leaf "$_amd_node_diag_leaf"; then
            _amd_node_diag_route=true
        else
            _amd_node_diag_route=false
        fi
        ;;
esac
if [ "$SKIP_TORCH" = true ]; then
    if _run_may_open_a_gpu_node; then
        _amd_node_diag_route=true
    else
        _amd_node_diag_route=false
    fi
fi
case "$(_requested_llama_backend)" in
    rocm|hip|vulkan) _amd_node_diag_route=true ;;
esac
# Separate branches: amd-smi succeeds with only /dev/dri mapped, where HIP has no /dev/kfd.
if [ "$_amd_node_diag_route" = true ] && \
   _run_may_open_kfd && [ "$OS" != "macos" ] && \
   [ ! -e /dev/kfd ] && _amd_silicon_behind_a_missing_kfd; then
    substep "An AMD GPU is in the KFD topology but /dev/kfd is not present, so the" "$C_WARN"
    substep "  driver is loaded and reinstalling ROCm changes nothing: the node itself"
    substep "  is missing. Under Docker, recreate the container with --device /dev/kfd"
    substep "  --device /dev/dri; on a bare host it is a udev or devtmpfs problem."
elif [ "$_amd_node_diag_route" = true ] && \
   _run_may_open_kfd && [ "$OS" != "macos" ] && \
   ! printf '%s\n' "$_closed_amd_nodes" | grep -qx /dev/kfd && \
   ! _has_amd_rocm_gpu ignore-nvidia && _amd_gpu_present_via_pci && \
   [ -e /dev/kfd ] && _kfd_node_is_amds; then
        substep "An AMD GPU is on the PCI bus and /dev/kfd is present and openable, so" "$C_WARN"
        substep "  the kernel stack is already loaded and reinstalling it changes nothing."
        substep "  What is missing is the ROCm userspace that reads the card: install"
        substep "  rocminfo and amd-smi (rocminfo, rocm-smi-lib) and re-run. Strix Halo"
        substep "  (gfx1151/gfx1150) also needs a recent kernel (6.11+) and ROCm 7.x."
elif [ "$_amd_node_diag_route" = true ] && \
   _run_may_open_kfd && [ "$OS" != "macos" ] && \
   ! printf '%s\n' "$_closed_amd_nodes" | grep -qx /dev/kfd && \
   ! _has_amd_rocm_gpu ignore-nvidia && _amd_gpu_present_via_pci && \
   { [ ! -e /dev/kfd ] || ! _kfd_node_is_amds; }; then
        substep "An AMD GPU is on the PCI bus but ROCm cannot see it (no /dev/kfd," "$C_WARN"
        substep "  rocminfo, or amd-smi). Install the ROCm kernel stack so /dev/kfd exists;"
        substep "  Strix Halo (gfx1151/gfx1150) needs a recent kernel (6.11+) and ROCm 7.x."
fi
if ! _run_may_open_kfd; then
    _closed_amd_nodes=$(printf '%s\n' "$_closed_amd_nodes" | grep -vx /dev/kfd || true)
fi
# Only group membership repairs this (#10466). /dev/kfd stops ROCm; a render node also stops Vulkan.
if [ "$_amd_node_diag_route" = true ] && _run_may_open_a_gpu_node && \
   [ -n "$_closed_amd_nodes" ]; then
    substep "An AMD GPU is present but this account cannot open its device nodes:" "$C_WARN"
    printf '%s\n' "$_closed_amd_nodes" | while IFS= read -r _n; do
        substep "  $_n"
    done
    if printf '%s\n' "$_closed_amd_nodes" | grep -qv '^/dev/kfd$'; then
        if _an_amd_render_node_is_open; then
            substep "  Every backend needs them, ROCm and Vulkan alike, but another AMD"
            substep "  render node on this host is open, so what they block is the card"
            substep "  behind them rather than all GPU work."
        else
            substep "  Every backend needs them, ROCm and Vulkan alike."
        fi
    else
        substep "  ROCm needs it; Vulkan does not."
    fi
    _closed_amd_repairs=$(_amd_node_repairs "$_closed_amd_nodes" || true)
    _closed_amd_groups=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^join://p' \
        | tr '\n' ',' | sed 's/,*$//')
    _closed_amd_gids=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^gid://p' \
        | tr '\n' ',' | sed 's/,*$//')
    _closed_amd_modes=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^mode://p' \
        | tr '\n' ',' | sed 's/,*$//')
    _closed_amd_acls=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^acl://p' \
        | tr '\n' ',' | sed 's/,*$//')
    _closed_amd_owned=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^owner://p' \
        | tr '\n' ',' | sed 's/,*$//')
    _closed_amd_priv=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^privileged://p' \
        | tr '\n' ',' | sed 's/,*$//')
    _closed_amd_already=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^already://p' \
        | tr '\n' ',' | sed 's/,*$//')
    _closed_amd_external=$(printf '%s\n' "$_closed_amd_repairs" | sed -n 's/^external://p' \
        | tr '\n' ',' | sed 's/,*$//')
    # $USER may be stale in containers; `id -un` fails for a uid with no passwd entry, where GNU id
    # still prints the uid, so use its status, not stdout.
    _amd_repair_user=$(id -un 2>/dev/null) || _amd_repair_user=''
    if [ -z "$_closed_amd_groups" ] && [ -z "$_closed_amd_gids" ] && \
       [ -z "$_closed_amd_modes" ] && [ -z "$_closed_amd_acls" ] && \
       [ -z "$_closed_amd_owned" ] && [ -z "$_closed_amd_priv" ] && \
       [ -z "$_closed_amd_already" ] && [ -z "$_closed_amd_external" ]; then
        _closed_amd_groups="render,video"
    fi
    if [ -n "$_closed_amd_groups" ] && [ -z "$_amd_repair_user" ]; then
        _closed_amd_group_adds=$(printf '%s\n' "$_closed_amd_groups" | tr ',' '\n' \
            | while IFS= read -r _cga_name; do
                  [ -n "$_cga_name" ] || continue
                  printf -- '--group-add %s ' "$(_shell_quote "$_cga_name")"
              done | sed 's/ *$//')
        substep "  This uid has no passwd entry, so usermod has no account to name:" "$C_WARN"
        substep "  recreate the container passing $_closed_amd_group_adds, or run it"
        substep "  as an account this system knows."
    elif [ -n "$_closed_amd_groups" ]; then
        case "$_closed_amd_groups" in
            *,*) substep "  Add yourself to the $_closed_amd_groups groups, then log out" ;;
            *)   substep "  Add yourself to the $_closed_amd_groups group, then log out" ;;
        esac
        substep "  and back in:"
        substep "  sudo usermod -a -G $(_shell_quote "$_closed_amd_groups") $(_shell_quote "$_amd_repair_user")"
    fi
    if [ -n "$_closed_amd_gids" ]; then
        _closed_amd_gid_adds=$(printf '%s' "$_closed_amd_gids" | tr ',' '\n' \
            | sed 's/^/--group-add /' | tr '\n' ' ' | sed 's/ *$//')
        case "$_closed_amd_gids" in
            *,*) substep "  Some of those nodes belong to GIDs $_closed_amd_gids, which have no" "$C_WARN"
                 substep "  group entry here, so usermod cannot name them: create a group for each" ;;
            *)   substep "  Some of those nodes belong to GID $_closed_amd_gids, which has no" "$C_WARN"
                 substep "  group entry here, so usermod cannot name it: create a group for it" ;;
        esac
        if [ -n "$_amd_repair_user" ]; then
            substep "  and add yourself to every one of them, then log out and back in:"
            for _amd_gid in $(printf '%s' "$_closed_amd_gids" | tr ',' ' '); do
                # Generated name, not a <name> placeholder: angle brackets are redirections.
                _amd_gid_name="amdgpu$_amd_gid"
                # Chained with && so a failed groupadd cannot be followed by a usermod on the wrong group.
                substep "  sudo groupadd -g $_amd_gid $_amd_gid_name && \\"
                substep "    sudo usermod -a -G $_amd_gid_name $(_shell_quote "$_amd_repair_user")"
            done
            substep "  or recreate the container passing $_closed_amd_gid_adds."
        else
            substep "  and this uid has no passwd entry either, so neither groupadd nor"
            substep "  usermod has anything to name: recreate the container passing"
            substep "  $_closed_amd_gid_adds."
        fi
    fi
    if [ -n "$_closed_amd_modes" ]; then
        substep "  $_closed_amd_modes does not grant its own group read and write, so no" "$C_WARN"
        substep "  membership opens it: fix the udev rule or the node's permissions."
    fi
    if [ -n "$_closed_amd_already" ]; then
        case "$_closed_amd_already" in
            *,*) substep "  This account is already in the $_closed_amd_already groups that own" "$C_WARN" ;;
            *)   substep "  This account is already in the $_closed_amd_already group that owns" "$C_WARN" ;;
        esac
        substep "  those nodes, so usermod would change nothing: something outside the"
        substep "  file mode is denying them, typically a container device cgroup or an"
        substep "  LSM such as SELinux or AppArmor."
    fi
    if [ -n "$_closed_amd_owned" ]; then
        substep "  $_closed_amd_owned is owned by this account, and POSIX stops at the" "$C_WARN"
        substep "  owner bits once the uid matches, so no group membership opens it"
        substep "  however its group bits read: fix the mode, or the udev rule behind it."
    fi
    if [ -n "$_closed_amd_external" ]; then
        substep "  $_closed_amd_external is already granted read and write by the" "$C_WARN"
        substep "  permission bits that apply to this account, so the mode is not what"
        substep "  is shutting it:"
        substep "  something outside the file mode is denying it, typically a container"
        substep "  device cgroup or an LSM such as SELinux or AppArmor."
    fi
    if [ -n "$_closed_amd_priv" ]; then
        substep "  Those nodes belong to the $_closed_amd_priv group, which grants a" "$C_WARN"
        substep "  great deal besides the GPU, so joining it is not the repair: fix the"
        substep "  udev rule so the node is owned by render or video instead."
    fi
    if [ -n "$_closed_amd_acls" ]; then
        substep "  $_closed_amd_acls carries a POSIX ACL, so the group permissions cannot" "$C_WARN"
        substep "  be read from its mode: check the real grant with getfacl before"
        substep "  changing group membership."
    fi
    if ! _amd_render_node_present; then
        if _run_may_open_kfd; then
            _amd_map_devices="--device /dev/kfd --device /dev/dri"
        else
            _amd_map_devices="--device /dev/dri"
        fi
        substep "  No AMD render node (/dev/dri/renderD*) is present either, and ROCm and" "$C_WARN"
        substep "  Vulkan both open one, so the device mapping needs fixing too; under"
        substep "  Docker that is $_amd_map_devices."
    fi
elif [ "$_amd_node_diag_route" = true ] && _run_may_open_a_gpu_node && \
     [ "$OS" != "macos" ] && \
     ! _amd_render_node_present && _kfd_topology_has_an_amd_gpu; then
    if _run_may_open_kfd; then
        _amd_map_devices="--device /dev/kfd --device /dev/dri"
    else
        _amd_map_devices="--device /dev/dri"
    fi
    substep "An AMD GPU is in the KFD topology but no AMD render node" "$C_WARN"
    substep "  (/dev/dri/renderD*) is present, and ROCm and Vulkan both open one, so the"
    substep "  device mapping needs fixing; under Docker that is $_amd_map_devices."
fi

tauri_log "STEP" "Installing PyTorch"
_VENV_PY="$VENV_DIR/bin/python"

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

# A released unsloth wheel can pin an older torch; freeze the installed trio via uv --overrides.
# Every with-deps unsloth install must call this before resolving and rm it after.
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
            # uv resolves an override's -r includes relative to that file, so merge beside it. Globbing off:
            # uv reads literal names.
            _ov_glob=on
            case $- in *f*) _ov_glob=off ;; esac
            set -f
            _ov_dir=""
            _ov_dir_ok=1
            for _ov_file in ${UV_OVERRIDE:-}; do
                [ -f "$_ov_file" ] || continue
                _ov_this=$(CDPATH= cd -- "$(dirname -- "$_ov_file")" 2>/dev/null && pwd) || _ov_this=""
                if [ -z "$_ov_this" ]; then
                    _ov_dir_ok=0
                    break
                elif [ -z "$_ov_dir" ]; then
                    _ov_dir=$_ov_this
                elif [ "$_ov_dir" != "$_ov_this" ]; then
                    _ov_dir_ok=0
                    break
                fi
            done
            _UNSLOTH_TORCH_OVERRIDES=""
            if [ "$_ov_dir_ok" = 1 ] && [ -n "$_ov_dir" ] && [ -w "$_ov_dir" ]; then
                _UNSLOTH_TORCH_OVERRIDES="$_ov_dir/.unsloth-torch-overrides.$$.txt"
                # 0600: inherited requirements can hold authenticated URLs; `: >` keeps an old file's mode.
                if (umask 077; : > "$_UNSLOTH_TORCH_OVERRIDES") 2>/dev/null; then
                    chmod 600 "$_UNSLOTH_TORCH_OVERRIDES" 2>/dev/null || true
                else
                    _UNSLOTH_TORCH_OVERRIDES=""
                fi
            fi
            [ -n "$_UNSLOTH_TORCH_OVERRIDES" ] || _UNSLOTH_TORCH_OVERRIDES=$(mktemp)
            printf '%s\n' "$_torch_trio_pins" > "$_UNSLOTH_TORCH_OVERRIDES"
            # --overrides replaces UV_OVERRIDE, so fold its pins in. tolower: uv would reject "Torch<2.11".
            for _ov_file in ${UV_OVERRIDE:-}; do
                [ -f "$_ov_file" ] && awk '!(tolower($0) ~ /^[[:space:]]*torch(vision|audio)?([[:space:]<>=!~;@[]|$)/)' "$_ov_file" >> "$_UNSLOTH_TORCH_OVERRIDES"
            done
            # `if`, not `&&`: a false last test would make the function fail under set -e.
            if [ "$_ov_glob" = on ]; then set +f; fi
            ;;
    esac
}

_unsloth_desktop_install_spec=""
if [ -n "${UNSLOTH_DESKTOP_BACKEND_VERSION:-}" ]; then
    _unsloth_desktop_install_spec="unsloth>=${UNSLOTH_DESKTOP_BACKEND_VERSION}"
fi
_unsloth_release_install_spec="${_unsloth_desktop_install_spec:-unsloth>=2026.10.1}"

if [ "$_MIGRATED" = true ]; then
    _gfx906_bnb_snapshot
    substep "upgrading unsloth in migrated environment..."
    if [ "$SKIP_TORCH" = true ]; then
        # --no-deps: this spec IS the zoo floor; keep it equal to pyproject.toml
        # (tests/test_installer_zoo_floor_parity.py).
        run_install_cmd_retry "install unsloth (migrated no-torch)" uv pip install --python "$_VENV_PY" --no-deps \
            --reinstall-package unsloth --reinstall-package unsloth-zoo \
            "$_unsloth_release_install_spec" "unsloth-zoo>=2026.10.1"
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
            "$_unsloth_release_install_spec" "unsloth-zoo>=2026.10.1"
        [ -n "$_UNSLOTH_TORCH_OVERRIDES" ] && rm -f "$_UNSLOTH_TORCH_OVERRIDES"
        _UNSLOTH_TORCH_OVERRIDES=""
    fi
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git ${_ZOO_REF}..."
        run_install_cmd_retry "overlay unsloth-zoo (git ${_ZOO_REF})" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "$_ZOO_GIT_SPEC"
    fi
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
        elif [ "$_gfx_rocm64_target" = true ] && \
             _venv_torch_rocm_below "$_VENV_PY" "$_gfx_rocm64_floor_maj" "$_gfx_rocm64_floor_min"; then
            substep "reinstalling torch from $_torch_index_leaf (the migrated wheels have no kernels for this GPU)..."
            _install_torch_default_index --force-reinstall
        elif [ "${_amd_arch_index_routed:-false}" = true ] && {
                 _venv_torch_rocm_below "$_VENV_PY" 7 13 ||
                 { _vfam=$(_venv_torch_amd_family "$_VENV_PY")
                   [ -n "$_vfam" ] && [ "$_vfam" != "${_amd_arch_index_family:-}" ]; }; }; then
            substep "reinstalling torch from the AMD per-arch index (the migrated wheels do not match it)..."
            _install_torch_default_index --force-reinstall
        fi
        _gfx906_bnb_prune
    fi
    # The ROCm repair above cannot reach an extras pin: whl-multi-arch is not a pip ROCm family
    # leaf, so _torch_index_is_rocm_family is false and the migrated install silently resolves
    # torch from PyPI. The pin is the only signal a specific build was wanted, so repair on it.
    # Inert by default: every default path leaves _TORCH_EXTRA empty.
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = false ] && [ -n "${_TORCH_EXTRA:-}" ]; then
        substep "reinstalling torch from the pinned index (UNSLOTH_TORCH_EXTRA=$_TORCH_EXTRA)..."
        _install_torch_default_index --force-reinstall
    fi
elif [ -n "$TORCH_INDEX_URL" ]; then
    if [ "$SKIP_TORCH" = true ]; then
        substep "skipping PyTorch (--no-torch or Intel Mac x86_64)." "$C_WARN"
    elif [ "$_amd_gpu_radeon" = true ]; then
        _radeon_url=$(get_radeon_wheel_url "$_torch_index_leaf")
        if [ -n "$_radeon_url" ]; then
            _radeon_listing_ok=false
            if _radeon_fetch_listing "$_radeon_url" 2>/dev/null; then
                _radeon_listing_ok=true
            else
                _radeon_url_short=$(printf '%s\n' "$_radeon_url" \
                    | sed 's|rocm-rel-\([0-9]*\)\.\([0-9]*\)\.[0-9]*/|rocm-rel-\1.\2/|')
                if [ "$_radeon_url_short" != "$_radeon_url" ] && \
                   _radeon_fetch_listing "$_radeon_url_short" 2>/dev/null; then
                    _radeon_listing_ok=true
                fi
            fi

            if [ "$_radeon_listing_ok" = true ]; then
                _torch_whl=$(_pick_radeon_wheel "torch"       2>/dev/null) || _torch_whl=""
                _tv_whl=$(_pick_radeon_wheel    "torchvision" 2>/dev/null) || _tv_whl=""
                _ta_whl=$(_pick_radeon_wheel    "torchaudio"  2>/dev/null) || _ta_whl=""
                _tri_whl=$(_pick_radeon_wheel   "triton"      2>/dev/null) || _tri_whl=""

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
                            _c_torch_ver=$(_extract_version "$_curr_torch" "torch")
                            _c_tv_ver=$(_extract_version "$_curr_tv" "torchvision")
                            _c_ta_ver=$(_extract_version "$_curr_ta" "torchaudio")

                            _c_torch_major=${_c_torch_ver%%.*}
                            _c_torch_minor=${_c_torch_ver#*.}
                            _c_ta_major=${_c_ta_ver%%.*}
                            _c_ta_minor=${_c_ta_ver#*.}
                            _c_tv_major=${_c_tv_ver%%.*}
                            _c_tv_minor=${_c_tv_ver#*.}

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
            elif [ "$_RADEON_HOST_ANSWERED" = true ]; then
                _radeon_rel=${_radeon_url%/}
                _radeon_rel=${_radeon_rel##*/}
                substep "repo.radeon.com has no $_radeon_rel wheels; using $(_strip_index_url_credentials "$TORCH_INDEX_URL")"
                _install_torch_default_index
            else
                substep "[WARN] Radeon repo unreachable; falling back to ROCm index ($(_strip_index_url_credentials "$TORCH_INDEX_URL"))" "$C_WARN"
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
    if [ "$SKIP_TORCH" = false ] && [ "$_torch_index_is_rocm_family" = true ]; then
        if _is_gfx906_bnb_skip; then
            substep "gfx906: skipping prebuilt bitsandbytes (no gfx906 kernels); build from source for 4-bit QLoRA -- https://docs.unsloth.ai/get-started/install-and-update/amd" "$C_WARN"
        else
            _install_bnb_rocm "install bitsandbytes (AMD)" "$_VENV_PY"
        fi
    fi
    _gfx906_bnb_snapshot
    tauri_log "STEP" "Installing Unsloth"
    substep "installing unsloth (this may take a few minutes)..."
    _build_unsloth_torch_overrides
    if [ "$SKIP_TORCH" = true ]; then
        # --no-deps: this spec IS the zoo floor here. Kept equal to pyproject.toml's.
        run_install_cmd_retry "install unsloth (no-torch)" uv pip install --python "$_VENV_PY" --no-deps \
            --upgrade-package unsloth --upgrade-package unsloth-zoo \
            "$_unsloth_release_install_spec" "unsloth-zoo>=2026.10.1"
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
            substep "overlaying unsloth-zoo from git ${_ZOO_REF}..."
            run_install_cmd_retry "overlay unsloth-zoo (git ${_ZOO_REF})" uv pip install --python "$_VENV_PY" \
                --no-deps --reinstall-package unsloth-zoo \
                "$_ZOO_GIT_SPEC"
        fi
    elif [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        run_install_cmd_retry "install unsloth (local)" uv pip install --python "$_VENV_PY" \
            ${_UNSLOTH_TORCH_OVERRIDES:+--overrides "$_UNSLOTH_TORCH_OVERRIDES"} \
            --upgrade-package unsloth "$_unsloth_release_install_spec" "unsloth-zoo>=2026.10.1"
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git ${_ZOO_REF}..."
        run_install_cmd_retry "overlay unsloth-zoo (git ${_ZOO_REF})" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "$_ZOO_GIT_SPEC"
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
        run_install_cmd_retry "install unsloth (auto torch backend)" uv pip install --python "$_VENV_PY" "unsloth-zoo>=2026.10.1" "$_unsloth_release_install_spec" --torch-backend=auto
        substep "overlaying local repo (editable)..."
        run_install_cmd "overlay local repo" uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
        substep "overlaying unsloth-zoo from git ${_ZOO_REF}..."
        run_install_cmd_retry "overlay unsloth-zoo (git ${_ZOO_REF})" uv pip install --python "$_VENV_PY" \
            --no-deps --reinstall-package unsloth-zoo \
            "$_ZOO_GIT_SPEC"
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

_installed_package_version_exit=0
if _installed_package_version=$("$_VENV_PY" -I -c '
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
# PEP 440 ignores the +cpu/+cuXXX local label, so uv keeps a stale torch+cpu against a GPU index.
if [ "$SKIP_TORCH" = false ] && [ -n "${TORCH_INDEX_URL:-}" ]; then
    _expected_torch_tag=$(_expected_torch_flavor_tag "$TORCH_INDEX_URL")
    if [ -n "$_expected_torch_tag" ] && [ "$_expected_torch_tag" != "cpu" ]; then
        _installed_torch_ver=$(_installed_torch_version_for_tag "$_expected_torch_tag")
        _installed_torch_tag=""
        [ -n "$_installed_torch_ver" ] && _installed_torch_tag=$(_torch_flavor_tag "$_installed_torch_ver")
        if [ -n "$_installed_torch_tag" ] && [ "$_installed_torch_tag" != "$_expected_torch_tag" ] \
           && [ "$(_torch_index_repairable "$TORCH_INDEX_URL")" = "yes" ]; then
            substep "PyTorch flavor mismatch (installed $_installed_torch_tag, need $_expected_torch_tag) -- reinstalling correct build..."
            _install_torch_default_index \
                --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
            _installed_torch_ver=$(_installed_torch_version_for_tag "$_expected_torch_tag")
            _installed_torch_tag=""
            [ -n "$_installed_torch_ver" ] && _installed_torch_tag=$(_torch_flavor_tag "$_installed_torch_ver")
        fi
        if [ "$_installed_torch_tag" = "cpu" ]; then
            substep "[WARN] PyTorch is CPU-only but a $_expected_torch_tag GPU build was expected for this machine." "$C_WARN"
            substep "[WARN] Training and GPU inference will run on CPU until this is fixed." "$C_WARN"
            substep "[WARN] Re-run this installer, or reinstall the GPU build manually:" "$C_WARN"
            substep "[WARN]   uv pip install --python \"$_VENV_PY\" \"$(_torch_spec_with_extra "$TORCH_CONSTRAINT")\" \"$(_torch_spec_with_extra "$TORCHVISION_CONSTRAINT")\" \"$TORCHAUDIO_CONSTRAINT\" --default-index $(_strip_index_url_credentials "$TORCH_INDEX_URL") --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio" "$C_WARN"
        fi
    fi
fi

if [ "$SKIP_TORCH" = false ] && ! _cvd_hides_nvidia; then
    case "${_expected_torch_tag:-}" in
        cu[0-9]*)
            _arch_check=$(_run_bounded --secs 120 "$_VENV_PY" -c '
import ctypes, sys

def load(*names):
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

def nvml_caps():
    lib = load("libnvidia-ml.so.1", "libnvidia-ml.so")
    if lib is None or lib.nvmlInit_v2() != 0:
        return None
    try:
        count, caps = ctypes.c_uint(), set()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or not count.value:
            return None
        for i in range(count.value):
            dev, major, minor = ctypes.c_void_p(), ctypes.c_int(), ctypes.c_int()
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(dev)) != 0 or \
               lib.nvmlDeviceGetCudaComputeCapability(dev, ctypes.byref(major), ctypes.byref(minor)) != 0:
                return None
            caps.add((major.value, minor.value))
        return sorted(caps)
    finally:
        lib.nvmlShutdown()

try:
    import torch
    if not torch.version.cuda or getattr(torch.version, "hip", None):
        sys.exit(0)
    try:
        archs = torch._C._cuda_getArchFlags().split()
    except Exception:
        archs = torch.cuda.get_arch_list()
    try:
        caps = nvml_caps()
    except Exception:
        caps = None
    if caps is None:
        if not torch.cuda.is_available():
            sys.exit(0)
        caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
except Exception:
    sys.exit(0)

def runs(arch, cap):
    kind, _, rest = arch.partition("_")
    n = len(rest) - len(rest.lstrip("0123456789"))
    digits, suffix = rest[:n], rest[n:]
    if n < 2:
        return False
    built = (int(digits[:-1]), int(digits[-1]))
    if suffix == "a":  # arch-specific cubin / PTX: that exact GPU only
        return built == cap
    if kind == "sm":  # a cubin runs on its own major at the same or a newer minor
        return built[0] == cap[0] and built[1] <= cap[1]
    return kind == "compute" and built <= cap  # PTX is JIT-compiled forward

missing = [c for c in caps if not any(runs(a, c) for a in archs)]
if not archs or not caps or not missing:
    sys.exit(0)
driver = ctypes.c_int()
cuda = load("libcuda.so.1", "libcuda.so")  # cuDriverGetVersion needs no cuInit
if cuda is None or cuda.cuDriverGetVersion(ctypes.byref(driver)) != 0:
    driver.value = 0
family = "cu126" if min(missing) < (7, 5) else ("cu130" if driver.value >= 13000 else "cu128")
status = "none" if len(missing) == len(caps) else "some"
# No wheel to point at (pre-Maxwell, or the family already installed): warn, never fail the install.
if min(missing) < (5, 0) or family == "cu" + torch.version.cuda.replace(".", ""):
    status = "nofix"
fmt = lambda cs: ",".join(f"{a}.{b}" for a, b in cs)
print("UNSLOTH_ARCH_CHECK=%s|%s|%s|%s|%s" % (status, fmt(missing), torch.__version__, " ".join(archs), family))
' 2>/dev/null | sed -n 's/^UNSLOTH_ARCH_CHECK=//p' | tail -n 1 || true)
            if [ -n "$_arch_check" ]; then
                IFS='|' read -r _ac_status _ac_caps _ac_torch _ac_archs _ac_family <<EOF_ARCH
$_arch_check
EOF_ARCH
                if [ -n "${UNSLOTH_PYTORCH_MIRROR:-}" ]; then _ac_pin="UNSLOTH_TORCH_INDEX_FAMILY=$_ac_family"
                else _ac_pin="UNSLOTH_TORCH_INDEX_URL=https://download.pytorch.org/whl/$_ac_family"
                fi
                if [ "$_ac_status" = "none" ] && [ "$_torch_index_pinned" = false ]; then
                    tauri_log "ERROR" "PyTorch $_ac_torch has no kernels for this GPU (compute capability $_ac_caps)"
                    substep "[ERROR] PyTorch $_ac_torch has no kernels for this GPU (compute capability $_ac_caps)." "$C_ERR"
                    substep "[ERROR] It was built for: $_ac_archs" "$C_ERR"
                    substep "[ERROR] Training would fail with \"no kernel image is available for execution on the device\"." "$C_ERR"
                    substep "[ERROR] Re-run this installer with the matching PyTorch wheels:" "$C_ERR"
                    substep "[ERROR]   $_ac_pin" "$C_ERR"
                    exit 1
                fi
                substep "[WARN] PyTorch $_ac_torch has no kernels for the GPUs with compute capability $_ac_caps." "$C_WARN"
                if [ "$_ac_status" = "nofix" ]; then
                    substep "[WARN] It was built for: $_ac_archs. Those GPUs will not be usable for training." "$C_WARN"
                else
                    substep "[WARN] It was built for: $_ac_archs. Those GPUs will not be usable; for them, re-run with $_ac_pin" "$C_WARN"
                fi
            fi
            ;;
    esac
fi

# An extras pin lands on a leaf the flavor enforcement above does not recognise, so it skips
# the pin, yet whether the build works IS the reason to pin an extra. Ask torch rather than
# read the version label: these indexes need not carry a +rocm local tag, which
# _torch_flavor_tag would read as "cpu". Bounded: a half-working HIP runtime can hang it.
if [ "$SKIP_TORCH" = false ] && [ -n "${_TORCH_EXTRA:-}" ]; then
    _extra_probe=$(_run_bounded "$_VENV_PY" -c \
        "import torch; print('UNSLOTH_CUDA_OK=%s' % torch.cuda.is_available())" 2>/dev/null \
        | sed -n 's/^UNSLOTH_CUDA_OK=//p' | tail -n 1 || true)
    if [ "$_extra_probe" = "True" ]; then
        substep "torch reports the GPU is usable (UNSLOTH_TORCH_EXTRA=$_TORCH_EXTRA)."
    else
        [ -n "$_extra_probe" ] || _extra_probe="unavailable (torch did not import)"
        substep "[WARN] Installed with UNSLOTH_TORCH_EXTRA=$_TORCH_EXTRA, but torch.cuda.is_available() is $_extra_probe." "$C_WARN"
        substep "[WARN] Training and GPU inference will run on CPU. These wheels are not tested by Unsloth." "$C_WARN"
        substep "[WARN] Please report the result either way: https://github.com/unslothai/unsloth/issues" "$C_WARN"
    fi
fi

# 0.50.0 is the first bnb with XPU libs; outside the install branches so migrated envs get it too.
# ── Intel XPU: bitsandbytes with XPU kernels ──
if [ "$SKIP_TORCH" = false ] && [ "$(_torch_index_url_leaf "${TORCH_INDEX_URL:-}")" = "xpu" ]; then
    substep "installing bitsandbytes with Intel XPU kernels..."
    run_install_cmd "install bitsandbytes (xpu)" uv pip install --python "$_VENV_PY" \
        --no-deps "$_BNB_XPU_SPEC" || \
        substep "[WARN] could not install an XPU-capable bitsandbytes; 4-bit QLoRA may be unavailable." "$C_WARN"
fi

# ── CI only: overlay a source checkout over the package just installed ──
# Not a consumer knob. Editable + --no-deps so CI legs validate this ref without git.
if [ -n "${UNSLOTH_CI_SOURCE_OVERLAY:-}" ]; then
    if [ ! -f "$UNSLOTH_CI_SOURCE_OVERLAY/pyproject.toml" ]; then
        echo "[ERROR] UNSLOTH_CI_SOURCE_OVERLAY is set to '$UNSLOTH_CI_SOURCE_OVERLAY' but there is no pyproject.toml there." >&2
        exit 1
    fi
    substep "CI: overlaying source checkout (editable, no deps): $UNSLOTH_CI_SOURCE_OVERLAY"
    run_install_cmd_retry "overlay CI source checkout" uv pip install --python "$_VENV_PY" \
        --no-deps -e "$UNSLOTH_CI_SOURCE_OVERLAY"
fi

tauri_log "STEP" "Running Unsloth setup"
SETUP_SH=""
if [ "$STUDIO_LOCAL_INSTALL" = true ] && [ -f "$_REPO_ROOT/studio/setup.sh" ]; then
    SETUP_SH="$_REPO_ROOT/studio/setup.sh"
fi

if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    # -I: otherwise the caller's cwd shadows `studio` with a checkout's setup.sh.
    SETUP_SH=$("$VENV_DIR/bin/python" -I -c "
import importlib.resources
print(importlib.resources.files('studio') / 'setup.sh')
" 2>/dev/null || echo "")
fi

if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    SETUP_SH=$(find "$VENV_DIR" -path "*/studio/setup.sh" -print -quit 2>/dev/null || echo "")
fi

if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    tauri_log "ERROR" "Could not find studio/setup.sh in the installed package"
    echo "❌ ERROR: Could not find studio/setup.sh in the installed package."
    exit 1
fi

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
_SKIP_FRONTEND=0
if [ "$TAURI_MODE" = true ]; then
    _SKIP_FRONTEND=1
fi
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
    # Reset STUDIO_LOCAL_* so a stale value cannot flip a normal install onto the local-dev path.
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
    # First: until this runs, a failure below would restore the old environment over the new one.
    _commit_studio_venv_replacement
    tauri_clear_install_error "studio setup completed"
fi

mkdir -p "$_LOCAL_BIN"
_shim_path="$_LOCAL_BIN/unsloth"
if [ -d "$_shim_path" ] && [ ! -L "$_shim_path" ]; then
    echo "ERROR: $_shim_path is a directory; refusing to delete it." >&2
    echo "       Move or remove it manually, then re-run the installer." >&2
    exit 1
fi
if ! ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path" 2>/dev/null; then
    if [ "$_shim_path" -ef "$VENV_DIR/bin/unsloth" ] 2>/dev/null; then
        substep "kept the existing shim at $_shim_path ($_LOCAL_BIN is not writable)"
    else
        echo "ERROR: could not create the shim at $_shim_path." >&2
        echo "       Make $_LOCAL_BIN writable, or run '$VENV_DIR/bin/unsloth' directly." >&2
        exit 1
    fi
fi

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

# Rewrite in place only lines this installer wrote (prepend vs append under conda, #5871),
# copying back into the original so a symlinked rc keeps link, mode and owner.
_unsloth_repoint_rc_line() {
    [ -f "$1" ] || return 1
    # OURS, not merely matching: a hand-written `export PATH="$HOME/.local/bin:$PATH"` is a
    # line users have too, and demoting theirs would move that whole directory behind the
    # rest of PATH for good. The `# Added by Unsloth` marker above the line is the ownership
    # record, so a line without one is left alone.
    _URRL_OLD="$2" awk '
        $0 == ENVIRON["_URRL_OLD"] && prev ~ /^# Added by Unsloth/ { found = 1 }
        { prev = $0 }
        END { exit(found ? 0 : 1) }
    ' "$1" 2>/dev/null || return 1
    # Staged and renamed, because `cat tmp > file` truncates the profile first and an
    # interrupt leaves wreckage. Onto the RESOLVED path: renaming over a chezmoi or stow
    # symlink would replace it with a regular file. `readlink -f` is GNU-only, hence the walk.
    _urrl_real="$1"
    _urrl_hops=0
    while [ -L "$_urrl_real" ] && [ "$_urrl_hops" -lt 40 ]; do
        _urrl_hops=$((_urrl_hops + 1))
        _urrl_target="$(readlink -- "$_urrl_real" 2>/dev/null)" || break
        [ -n "$_urrl_target" ] || break
        case "$_urrl_target" in
            /*) _urrl_real="$_urrl_target" ;;
            *) _urrl_real="$(dirname -- "$_urrl_real")/$_urrl_target" ;;
        esac
    done
    [ -f "$_urrl_real" ] || return 1
    _urrl_tmp="$_urrl_real.unsloth-tmp.$$"
    # `cp -p` keeps the original's mode; without it the umask masks it and a 0644 .bashrc
    # comes back 0600. ENVIRON, not `-v`: POSIX awk decodes backslash escapes in `-v`, so an
    # escaped path arrived as something else and renamed an unchanged file.
    if { cp -p -- "$_urrl_real" "$_urrl_tmp" 2>/dev/null \
        || cp -- "$_urrl_real" "$_urrl_tmp" 2>/dev/null; } \
        && _URRL_OLD="$2" _URRL_NEW="$3" awk '
            $0 == ENVIRON["_URRL_OLD"] && prev ~ /^# Added by Unsloth/ { print ENVIRON["_URRL_NEW"]; prev = $0; next }
            { print; prev = $0 }
        ' "$_urrl_real" > "$_urrl_tmp" 2>/dev/null \
        && mv -f -- "$_urrl_tmp" "$_urrl_real" 2>/dev/null; then
        return 0
    fi
    # The original is untouched on every failure above; only the staged copy needs removing.
    rm -f -- "$_urrl_tmp" 2>/dev/null
    return 1
}

_unsloth_conda_env_active() {
    [ -n "${CONDA_PREFIX:-}" ] || [ -n "${CONDA_DEFAULT_ENV:-}" ]
}

_persist_fish_path_dir() {
    _pfp_dir="$1"; _pfp_label="${2:-$1}"; _pfp_mode="${3:-}"
    [ -n "${HOME:-}" ] || return 0
    _pfp_dir_conf="$HOME/.config/fish/conf.d"
    mkdir -p "$_pfp_dir_conf" 2>/dev/null || return 0
    _pfp_file="$_pfp_dir_conf/unsloth.fish"
    _pfp_quoted=$(printf '%s' "$_pfp_dir" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g")
    # Under conda append, since fish_add_path prepends (#5871). -a -P -m are all required:
    # -P appends to PATH, -m moves an existing entry. See fishshell.com fish_add_path docs.
    _pfp_line="fish_add_path '$_pfp_quoted'"
    if _unsloth_conda_env_active; then
        _pfp_line="fish_add_path -a -P -m '$_pfp_quoted'"
        for _pfp_stale in "fish_add_path '$_pfp_quoted'" "fish_add_path -a '$_pfp_quoted'" \
                          "fish_add_path -a -P '$_pfp_quoted'"; do
            if _unsloth_repoint_rc_line "$_pfp_file" "$_pfp_stale" "$_pfp_line"; then
                step "path" "moved $_pfp_label after the active conda environment in $_pfp_file"
            fi
        done
    fi
    [ "$_pfp_mode" = "repoint" ] && return 0
    if ! grep -v '^[[:space:]]*#' "$_pfp_file" 2>/dev/null \
        | grep -qxF -e "fish_add_path '$_pfp_quoted'" -e "fish_add_path -a '$_pfp_quoted'" -e "fish_add_path -a -P '$_pfp_quoted'" -e "fish_add_path -a -P -m '$_pfp_quoted'"; then
        # Single redirect; failure only warns. 2>/dev/null first so the shell's own error is silenced.
        if {
            echo "# Added by Unsloth installer"
            echo "$_pfp_line"
        } 2>/dev/null >> "$_pfp_file"; then
            step "path" "added $_pfp_label to PATH in $_pfp_file"
        else
            step "path" "could not write $_pfp_file; add $_pfp_label to PATH yourself" "$C_WARN"
            substep "Unsloth is installed and works; only the PATH line is missing."
            substep "Add this to your fish config to get 'unsloth' in new shells:"
            substep "  $_pfp_line"
        fi
    fi
}

_PATH_LINE_RE='(^|[^[:alnum:]_])(PATH[[:space:]]*=|fish_add_path|pathmunge|path_helper)'

# Persist $1 on the next shell's PATH. $6 "repoint" only fixes a line a previous run wrote.
_persist_login_path_dir() {
    _plp_dir="$1"; _plp_literal="$2"; _plp_label="$3"; _plp_pattern="$4"; _plp_file="${5:-}"
    _plp_mode="${6:-}"
    [ -n "${HOME:-}" ] || return 0
    if [ -z "$_plp_file" ] && [ "$(basename "${SHELL:-}")" = "fish" ]; then
        _persist_fish_path_dir "$_plp_dir" "$_plp_label" "$_plp_mode"
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
        # A fresh account can have no rc file at all; every POSIX login shell reads ~/.profile.
        _SHELL_PROFILE="$HOME/.profile"
    fi
    [ -n "$_SHELL_PROFILE" ] || return 0
    # A persisted PREPEND outlives the activation and leaves conda resolving out of our
    # directory in every later shell (#5871). Inside one, write the same line as an APPEND;
    # the grep below matches either spelling, so no second line is added. install.ps1 makes
    # the same choice for the Windows registry.
    _plp_line="export PATH=\"$_plp_literal:\$PATH\""
    if _unsloth_conda_env_active; then
        _plp_prepend="$_plp_line"
        _plp_line="export PATH=\"\$PATH:$_plp_literal\""
        if grep -qxF "$_plp_prepend" "$_SHELL_PROFILE" 2>/dev/null; then
            if _unsloth_repoint_rc_line "$_SHELL_PROFILE" "$_plp_prepend" "$_plp_line"; then
                step "path" "moved $_plp_label after the active conda environment in $_SHELL_PROFILE"
            else
                step "path" "could not reposition $_plp_label in $_SHELL_PROFILE" "$C_WARN"
                substep "It is listed before conda, so conda resolves binaries out of it."
                substep "Replace that line with:"
                substep "  $_plp_line"
            fi
        fi
    fi
    [ "$_plp_mode" = "repoint" ] && return 0
    if ! grep -v '^[[:space:]]*#' "$_SHELL_PROFILE" 2>/dev/null \
        | grep -E "$_PATH_LINE_RE" | grep -qE "$_plp_pattern"; then
        # One redirect so a dying write cannot leave a dangling marker. Failure only warns: read-only rc
        # files (NixOS, home-manager) are supported.
        if {
            echo ''
            echo '# Added by Unsloth installer'
            echo "$_plp_line"
        } 2>/dev/null >> "$_SHELL_PROFILE"; then
            step "path" "added $_plp_label to PATH in $_SHELL_PROFILE"
        else
            step "path" "could not write $_SHELL_PROFILE; add $_plp_label to PATH yourself" "$C_WARN"
            substep "Unsloth is installed and works; only the PATH line is missing."
            substep "Add this to your shell config to get 'unsloth' in new shells:"
            substep "  $_plp_line"
        fi
    fi
}

if _unsloth_conda_env_active && [ "$_STUDIO_HOME_REDIRECT" != "env" ]; then
    _persist_login_path_dir "$_LOCAL_BIN" '$HOME/.local/bin' "~/.local/bin" '\.local/bin' "" repoint
fi

if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_LOCAL_BIN"; then  # not on a new shell's PATH
        if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
            export PATH="$_LOCAL_BIN:$PATH"
            step "path" "exported $_LOCAL_BIN for this session (no rc-file append in env-override mode)"
        else
            _persist_login_path_dir "$_LOCAL_BIN" '$HOME/.local/bin' "~/.local/bin" '\.local/bin'
            export PATH="$_LOCAL_BIN:$PATH"
        fi
fi

# Persist uv's destination too, as astral's installer did; honour both of its opt-outs.
if [ -n "${_UNSLOTH_UV_BIN_DIR:-}" ] \
   && [ -z "${UV_NO_MODIFY_PATH:-}" ] && [ -z "${UV_UNMANAGED_INSTALL:-}" ] \
   && [ "$_STUDIO_HOME_REDIRECT" != "env" ]; then
    if _unsloth_conda_env_active; then
        _uv_repoint_literal=$(printf '%s' "$_UNSLOTH_UV_BIN_DIR" | sed 's/[\\"$`]/\\&/g')
        # Also match the $HOME-relative spelling the shim block writes, across every astral startup file.
        _uv_repoint_home_literal=""
        case "$_UNSLOTH_UV_BIN_DIR" in
            "$HOME"/*)
                _uv_repoint_home_literal='$HOME'$(printf '%s' "${_UNSLOTH_UV_BIN_DIR#$HOME}" | sed 's/[\\"$`]/\\&/g')
                ;;
        esac
        for _uv_prof in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" \
                        "$HOME/.bash_login" "${ZDOTDIR:-$HOME}/.zshrc" "${ZDOTDIR:-$HOME}/.zshenv"; do
            [ -f "$_uv_prof" ] || continue
            _persist_login_path_dir "$_UNSLOTH_UV_BIN_DIR" "$_uv_repoint_literal" \
                "$_UNSLOTH_UV_BIN_DIR" "" "$_uv_prof" repoint
            if [ -n "$_uv_repoint_home_literal" ]; then
                _persist_login_path_dir "$_UNSLOTH_UV_BIN_DIR" "$_uv_repoint_home_literal" \
                    "$_UNSLOTH_UV_BIN_DIR" "" "$_uv_prof" repoint
            fi
        done
        _persist_fish_path_dir "$_UNSLOTH_UV_BIN_DIR" "" repoint
    fi
    if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_UNSLOTH_UV_BIN_DIR"; then
        _uv_rc_literal=$(printf '%s' "$_UNSLOTH_UV_BIN_DIR" | sed 's/[\\"$`]/\\&/g')
        _uv_grep_esc=$(printf '%s' "$_UNSLOTH_UV_BIN_DIR" | sed 's/[].[\\()*+?{}|^$\/]/\\&/g')
        case "$_UNSLOTH_UV_BIN_DIR" in
            "$HOME"/*)
                _uv_grep_esc="$_uv_grep_esc|\\\$HOME$(printf '%s' "${_UNSLOTH_UV_BIN_DIR#$HOME}" | sed 's/[].[\\()*+?{}|^$\/]/\\&/g')"
                ;;
        esac
        _uv_pattern="(^|[^[:alnum:]_.~/-])($_uv_grep_esc)([^[:alnum:]_.~/-]|\$)"
        for _uv_prof in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" \
                        "$HOME/.bash_login" "${ZDOTDIR:-$HOME}/.zshrc" "${ZDOTDIR:-$HOME}/.zshenv"; do
            if [ "$_uv_prof" = "$HOME/.profile" ] || [ -f "$_uv_prof" ]; then
                _persist_login_path_dir "$_UNSLOTH_UV_BIN_DIR" "$_uv_rc_literal" \
                    "$_UNSLOTH_UV_BIN_DIR" "$_uv_pattern" "$_uv_prof"
            fi
        done
        _persist_fish_path_dir "$_UNSLOTH_UV_BIN_DIR"
    fi
fi
# end of the PATH persistence block

if [ "$TAURI_MODE" != true ]; then
    create_studio_shortcuts "$VENV_ABS_BIN/unsloth" "$OS"
fi

# If setup.sh failed, report and exit now.
if [ "$_SETUP_EXIT" -ne 0 ]; then
    echo ""
    # Below 64 MiB the full disk is the cause. Folded into ERROR_DEFAULT: --tauri shows only that.
    _set_disk_full_suffix
    if [ "$TAURI_MODE" = true ]; then
        tauri_log "ERROR_DEFAULT" "studio setup failed (exit code $_SETUP_EXIT)$_DISK_FULL_SUFFIX"
    else
        step "error" "studio setup failed (exit code $_SETUP_EXIT)" "$C_ERR"
    fi
    if [ -n "$_DISK_FULL_SUFFIX" ]; then
        echo "       $STUDIO_HOME has only $_DISK_FULL_MB MB free -- the disk is full, which is very likely the cause." >&2 || true
        echo "       $_DISK_FULL_REMEDY" >&2 || true
    fi
    _DISK_FULL_REPORTED=true
    echo ""
    exit "$_SETUP_EXIT"
fi

# ── Tauri mode: done, skip shortcuts and auto-launch ──
if [ "$TAURI_MODE" = true ]; then
    tauri_log "DONE" ""
    exit 0
fi

_installed_bin="$VENV_DIR/bin/unsloth"
_path_unsloth=$(command -v unsloth 2>/dev/null || true)
if [ -n "$_path_unsloth" ] && [ -x "$VENV_DIR/bin/python" ]; then
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

if [ "$_INSTALL_SYSTEMD" = true ]; then
    _install_systemd_user_service
    if [ "$_SYSTEMD_STARTED" = true ]; then
        _SKIP_AUTOSTART=true
    fi
fi
if [ "$_SKIP_AUTOSTART" != true ] && [ -t 1 ]; then
    echo ""
    if _can_read_tty; then
        printf "  Start Unsloth Studio now? [Y/n] "
        read -r _reply </dev/tty || _reply="n"
    else
        _reply="n"
    fi
    case "${_reply:-y}" in
        [Yy]*|"")
            step "launch" "starting Unsloth Studio..."

            _prepare_studio_uv_cache_for_launch
            # Detach stdin so the server cannot drain the rest of a piped script. trap '' INT waits for
            # studio's shutdown; the subshell resets INT so the child still gets Ctrl+C.
            trap '' INT
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
    _li_shim_q="'$(printf '%s' "${_LOCAL_BIN}/unsloth" | sed "s/'/'\\\\''/g")'"
    _li_act_q="'$(printf '%s' "${VENV_DIR}/bin/activate" | sed "s/'/'\\\\''/g")'"
    if [ "$_STUDIO_HOME_REDIRECT" = "env" ]; then
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
