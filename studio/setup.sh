#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The deps pass can replace this file while bash reads the old one (_setup_rerun_if_replaced).
_SETUP_SELF="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
_SETUP_SELF_SUM=$(cksum < "$_SETUP_SELF" 2>/dev/null || true)
_SETUP_ARGV=("$@")
_SETUP_START_PWD=$PWD
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RULE=$(printf '\342\224\200%.0s' {1..52})

# ── Parse flags ──
# --local: overlay unsloth editable and unsloth-zoo from git main (mirrors install.sh --local).
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

# Maintainer-editable defaults; user env vars override.
# Prefer _DEFAULT_LLAMA_TAG=latest: master bypasses the prebuilt resolver and forces a source build.
_DEFAULT_LLAMA_PR_FORCE=""
_DEFAULT_LLAMA_SOURCE="https://github.com/ggml-org/llama.cpp"
_DEFAULT_LLAMA_TAG="latest"
_DEFAULT_LLAMA_FORCE_COMPILE_REF="master"

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

# Usage: step <label> <message> [color]; substep <message> [color]
step()    { printf "  ${C_DIM}%-15.15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  %-15s${2:-$C_DIM}%s${C_RST}\n" "" "$1"; }

setup_fail() {
    local exit_code=$1
    shift
    [ "$exit_code" -ne 0 ] || exit_code=1
    local message
    message=$(printf '%s' "$*" | tr '\r\n' '  ')
    # Mirrors setup.ps1 (update.rs promotes this line). Test each variable separately: one
    # joined subject lets a comma alias the other arm.
    local tauri_marker=0
    case "${UNSLOTH_TAURI_MODE:-0}" in 1|true) tauri_marker=1 ;; esac
    case "${UNSLOTH_TAURI_UPDATE:-0}" in 1|true) tauri_marker=1 ;; esac
    if [ "$tauri_marker" -eq 1 ]; then printf '[TAURI:ERROR] %s\n' "$message"; fi
    exit "$exit_code"
}

# `test -r` passes in containers where open() fails with ENXIO; probe with a real open.
# Mirrors install.sh's _can_read_tty (setup.sh runs as its own process).
_can_read_tty() {
    ( : </dev/tty ) >/dev/null 2>&1
}

_is_verbose() {
    [ "${UNSLOTH_VERBOSE:-0}" = "1" ]
}

_filter_download_output() {
    if _is_verbose; then
        cat
        return
    fi
    local line
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            "Downloading "*": "*"% ("*") at "*"/s"|"Downloading "*": "*" downloaded at "*"/s")
                printf '%s\n' "$line"
                ;;
        esac
    done
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

_SETUP_PROBE_TARGET=""
_SETUP_PROBE_PID=""
_SETUP_PROBE_PREV_TRAP=""

# Bash needs `--` before a negative process-group ID; dash rejects it.
_setup_probe_signal_target() {
    case "$2" in
        -*) kill "-$1" -- "$2" 2>/dev/null || : ;;
        *)  kill "-$1" "$2" 2>/dev/null || : ;;
    esac
}

_setup_probe_terminate() {
    _supt_grace=0
    _setup_probe_signal_target TERM "$1"
    while [ "$_supt_grace" -lt "$3" ] && kill -0 "$2" 2>/dev/null; do
        sleep 1
        _supt_grace=$((_supt_grace + 1))
    done
    if kill -0 "$2" 2>/dev/null; then _setup_probe_signal_target KILL "$1"; fi
    unset _supt_grace
}

_setup_probe_restore_trap() {
    _SETUP_PROBE_TARGET=""
    _SETUP_PROBE_PID=""
    if [ -n "${_SETUP_PROBE_PREV_TRAP:-}" ]; then
        eval "$_SETUP_PROBE_PREV_TRAP"
    else
        trap - HUP INT TERM
    fi
    _SETUP_PROBE_PREV_TRAP=""
}

_setup_probe_on_signal() {
    if [ -n "${_SETUP_PROBE_TARGET:-}" ] && [ -n "${_SETUP_PROBE_PID:-}" ]; then
        _setup_probe_terminate "$_SETUP_PROBE_TARGET" "$_SETUP_PROBE_PID" 2
        wait "$_SETUP_PROBE_PID" 2>/dev/null || :
    fi
    _setup_probe_restore_trap
    kill -s "$1" "$$" 2>/dev/null || :
}

# Bounded --version probe; output goes to a file so lingering children cannot hold a pipe.
_setup_probe_version() {
    _supe_secs="${_SETUP_PROBE_SECONDS:-20}"
    _supe_out="${2:-/dev/null}"
    if command -v timeout >/dev/null 2>&1 && timeout -k 1 5 true >/dev/null 2>&1; then
        timeout -k 5 "$_supe_secs" "$1" --version >"$_supe_out" 2>/dev/null </dev/null
        _supe_rc=$?
        if [ "$_supe_rc" -eq 137 ]; then _supe_rc=124; fi
        return $_supe_rc
    fi
    _supe_monitor=off
    case "$-" in *m*) _supe_monitor=on ;; esac
    [ "$_supe_monitor" = on ] || set -m 2>/dev/null || :
    "$1" --version >"$_supe_out" 2>/dev/null </dev/null &
    _supe_pid=$!
    [ "$_supe_monitor" = on ] || set +m 2>/dev/null || :
    # Signal a group only if it differs from setup's own group.
    _supe_target="$_supe_pid"
    if command -v ps >/dev/null 2>&1; then
        _supe_pgid=$(ps -o pgid= -p "$_supe_pid" 2>/dev/null)
        _supe_self=$(ps -o pgid= -p $$ 2>/dev/null)
        _supe_pgid=${_supe_pgid##* }
        _supe_self=${_supe_self##* }
        case "$_supe_pgid$_supe_self" in
            ''|*[!0-9]*) : ;;
            *) [ "$_supe_pgid" = "$_supe_self" ] || _supe_target="-$_supe_pgid" ;;
        esac
    fi
    _SETUP_PROBE_TARGET="$_supe_target"
    _SETUP_PROBE_PID="$_supe_pid"
    _SETUP_PROBE_PREV_TRAP=$(trap -p HUP INT TERM 2>/dev/null) || _SETUP_PROBE_PREV_TRAP=""
    trap '_setup_probe_on_signal HUP' HUP
    trap '_setup_probe_on_signal INT' INT
    trap '_setup_probe_on_signal TERM' TERM
    _supe_waited=0
    while kill -0 "$_supe_pid" 2>/dev/null; do
        if [ "$_supe_waited" -ge "$_supe_secs" ]; then
            _setup_probe_terminate "$_supe_target" "$_supe_pid" 5
            wait "$_supe_pid" 2>/dev/null
            _setup_probe_restore_trap
            unset _supe_pid _supe_waited _supe_target _supe_pgid _supe_self
            return 124
        fi
        sleep 1
        _supe_waited=$((_supe_waited + 1))
    done
    wait "$_supe_pid"
    _supe_rc=$?
    _setup_probe_restore_trap
    unset _supe_pid _supe_waited _supe_target _supe_pgid _supe_self
    return $_supe_rc
}

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

# studio/frontend/.npmrc pins the registry (supply-chain lock); UNSLOTH_NPM_REGISTRY opts into
# --registry for npm/bun installs. Empty array expands to nothing under set -u.
_NPM_REGISTRY_ARGS=()
if [ -n "${UNSLOTH_NPM_REGISTRY:-}" ]; then
    _NPM_REGISTRY_ARGS=(--registry "$UNSLOTH_NPM_REGISTRY")
fi
_CAPTURE_LOG=""

_npm_mirror_retry() {
    [ "$(_mirror_failed_host "${_CAPTURE_LOG:-}" npm)" = npm ] && _mirror_take npm || return 1
    run_quiet_no_exit "$1" npm "${_NPM_INSTALL:-install}" --no-fund --no-audit --loglevel=error --registry "${_MT_PAIRS#*=}" || return
    export "$_MT_PAIRS"
    _NPM_REGISTRY_ARGS=(--registry "$UNSLOTH_NPM_REGISTRY")
}

# Match local errno on npm's code line only, before the network check (FetchError names the
# registry even on local errors). Keep in sync with setup.ps1.
_NPM_LOCAL_FAILURE_RE='npm (error|ERR!) code (EACCES|EPERM|EBUSY|ENOSPC|ENFILE|EMFILE)|operation was rejected by your operating system'

_suggest_npm_local_failure() {
    printf '\n' >&2
    if [ "${1:-}" = socket ]; then
        step "frontend" "the OS refused node's connection to the npm registry" "$C_WARN" >&2
        substep "Allow $(command -v node 2>/dev/null || echo node) in your firewall/antivirus, or use another Node install." >&2
        return 0
    fi
    step "frontend" "npm hit a local file error (permission, lock or disk full)" "$C_WARN" >&2
    substep "Try: npm cache clean --force, or check the npm cache is writable." >&2
    return 0
}

# Guidance only when the registry lock is the likely cause; the mirror fallback does any switching.
_suggest_npm_registry() {
    local _log="${1:-}"
    local _plain=""
    if [ -n "$_log" ] && [ -s "$_log" ]; then _plain="$(sed "s/$(printf '\033')\[[0-9;]*m//g" "$_log")"; fi
    if [ -n "$_plain" ] && grep -Eq "$_NPM_LOCAL_FAILURE_RE" <<<"$_plain"; then
        if grep -q 'FetchError' <<<"$_plain" && ! grep -Eq 'npm (error|ERR!) path ' <<<"$_plain"; then
            _suggest_npm_local_failure socket
        else
            _suggest_npm_local_failure
        fi
        return 0
    fi
    [ -n "${UNSLOTH_NPM_REGISTRY:-}" ] && return 0
    if [ -n "$_log" ] && [ -s "$_log" ] \
        && ! grep -Eqi '40[13]|ENOTFOUND|ECONNREFUSED|ECONNRESET|ETIMEDOUT|EAI_AGAIN|ConnectionRefused|failed to resolve|registry\.npmjs\.org|getaddrinfo|tunneling socket|network|proxy|self.?signed|unable to (get|verify)' "$_log"; then
        return 0
    fi
    # Read npm config from / so the frontend's pinned registry does not mask the user's mirror.
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
            setup_fail "$exit_code" "$label failed (exit code $exit_code)"
        else
            return "$exit_code"
        fi
    fi

    local tmplog
    tmplog=$(mktemp) || {
        step "error" "Failed to create temporary file" "$C_ERR" >&2
        if [ "$on_fail" = "exit" ]; then
            setup_fail 1 "Failed to create temporary file for $label"
        fi
        return 1
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
            setup_fail "$exit_code" "$label failed (exit code $exit_code)"
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
    # Echo ok|too_old|unknown then X.Y. llama.cpp needs CUDA toolkit >= 12.4.
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

# Empty capability list means build CPU, not a PTX-only binary; a partial list yields nothing.
_probe_compute_caps() {
    [ -f "$SCRIPT_DIR/nvidia_probe.py" ] && command -v python3 >/dev/null 2>&1 || return 0
    _setup_run_smi python3 -I "$SCRIPT_DIR/nvidia_probe.py" 2>/dev/null | awk '
        /^GPU [0-9]+:/ {
            if (match($0, /\(compute [0-9]+\.[0-9]+\)$/)) caps = caps substr($0, RSTART + 9, RLENGTH - 10) "\n"
            else bad = 1
        }
        END { if (!bad) printf "%s", caps }' || true
}

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

# The build dir is renamed into place, so CMake's build-tree RUNPATH dies; use $ORIGIN.
# USE_LINK_PATH keeps toolchain dirs (ROCm, CUDA, Nix).
_llama_relocatable_rpath_args() {
    case "$(uname -s 2>/dev/null)" in
        Linux) printf '%s' '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=$ORIGIN -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON' ;;
        *) printf '' ;;
    esac
}

# MiB for the OS and per compile job; 2048 is above measured nvcc peaks to cover MSVC, hipcc, link.
_LLAMA_BUILD_RESERVE_MB=2048
_LLAMA_BUILD_MB_PER_JOB=2048

# Pure (args: cores, RAM MiB) so tests can drive it. UNSLOTH_LLAMA_BUILD_JOBS wins.
_llama_jobs_for() {
    local _cores=$1 _mem_mb=$2 _jobs
    if [[ "${UNSLOTH_LLAMA_BUILD_JOBS:-}" =~ ^[0-9]+$ ]] && [ "$UNSLOTH_LLAMA_BUILD_JOBS" -ge 1 ]; then
        printf '%s' "$UNSLOTH_LLAMA_BUILD_JOBS"
        return 0
    fi
    if ! [[ "$_cores" =~ ^[0-9]+$ ]] || [ "$_cores" -lt 1 ]; then _cores=4; fi
    if ! [[ "$_mem_mb" =~ ^[0-9]+$ ]]; then
        printf '%s' "$_cores"
        return 0
    fi
    _jobs=$(( (_mem_mb - _LLAMA_BUILD_RESERVE_MB) / _LLAMA_BUILD_MB_PER_JOB ))
    [ "$_jobs" -lt 1 ] && _jobs=1
    [ "$_jobs" -gt "$_cores" ] && _jobs=$_cores
    printf '%s' "$_jobs"
}

# Builtin read, not `head | tr`: a failing pipeline under set -euo pipefail aborted installs.
_cg_read() {
    local _v=""
    [ -f "$1" ] && [ -r "$1" ] || return 0
    IFS= read -r _v < "$1" 2>/dev/null || true
    printf '%s' "${_v//[[:space:]]/}"
    return 0
}

# v2 writes "max", v1 a near-2^63 sentinel. Zero is a real limit (memory.high).
_cg_limit() {
    [[ "$1" =~ ^[0-9]+$ ]] || return 0
    [ "$1" -lt 4611686018427387904 ] && printf '%s' "$1"
    return 0
}

# NUL-delimited: a mount point may contain a newline.
_cg_dirs() {
    local _root=$1 _cur
    if [ -n "${2:-}" ] && [ "$2" != "/" ]; then
        _cur="$_root/${2#/}"
        while [ "$_cur" != "$_root" ] && [ "$_cur" != "/" ]; do
            printf '%s\0' "$_cur"
            case "$_cur" in
                */*) _cur=${_cur%/*}; [ -n "$_cur" ] || _cur="/" ;;
                *) break ;;
            esac
        done
    fi
    printf '%s\0' "$_root"
}

# Shared awk decoder for mountinfo escapes; no strtonum so it runs under mawk and BSD awk.
_cg_unesc_prog() {
    cat <<'_CG_AWK_UNESC'
        function unesc(s,   out, i, c, o, v) {
            out = ""; i = 1
            while (i <= length(s)) {
                c = substr(s, i, 1)
                if (c == "\\" && substr(s, i + 1, 3) ~ /^[0-7][0-7][0-7]$/) {
                    o = substr(s, i + 1, 3)
                    v = (substr(o, 1, 1) + 0) * 64 + (substr(o, 2, 1) + 0) * 8 + (substr(o, 3, 1) + 0)
                    out = out sprintf("%c", v); i += 4
                } else { out = out c; i++ }
            }
            return out
        }
_CG_AWK_UNESC
}

# Decode late: \011 and \012 decode to the delimiters of the records below.
_cg_unesc() {
    [ -n "$1" ] || return 0
    printf '%s\n' "$1" | awk "$(_cg_unesc_prog)"'{ printf "%s", unesc($0); exit }' || true
    return 0
}

# v1 hierarchies are found by controller name in super options, not assumed at <root>/<name>.
_cg_mounts() {
    [ -r "$1" ] || return 0
    awk -v want="$2" '
        {
            for (i = 1; i <= NF; i++) if ($i == "-") break
            if (i + 3 > NF) next
            if (want == "cgroup2") {
                if ($(i + 1) == "cgroup2") print $4 "\t" $5
                next
            }
            if ($(i + 1) != "cgroup") next
            n = split($(i + 3), opts, ",")
            for (j = 1; j <= n; j++) if (opts[j] == want) { print $4 "\t" $5; next }
        }' "$1" 2>/dev/null || true
    return 0
}

# Map the process cgroup path under a bind-mounted subtree root.
_cg_rel() {
    local _root=$1 _rel=$2
    [ -n "$_rel" ] || return 0
    [ "$_root" = "/" ] && { printf '%s' "$_rel"; return 0; }
    case "$_rel" in
        "$_root") printf '%s' "/" ;;
        "$_root"/*) printf '%s' "${_rel#"$_root"}" ;;
        *) : ;;
    esac
    return 0
}

# Every containing mount is inspected and the smallest allowance wins (nested rootless podman).
_cg_pick_mounts() {
    local _rel=$1 _root _point _droot _any="" _firstroot="" _firstpoint=""
    while IFS=$'\t' read -r _root _point; do
        [ -n "$_point" ] || continue
        if [ -z "$_firstpoint" ]; then _firstroot=$_root; _firstpoint=$_point; fi
        _droot=$(_cg_unesc "$_root")
        [ -n "$(_cg_rel "$_droot" "$_rel")" ] || continue
        printf '%s\t%s\n' "$_root" "$_point"
        _any=1
    done
    [ -n "$_any" ] || [ -z "$_firstpoint" ] || printf '%s\t%s\n' "$_firstroot" "$_firstpoint"
    return 0
}

# Free MiB under the binding cgroup limit. Mirrors dataset_num_proc.py: each limit pairs with
# its own directory's usage, and the smallest remaining allowance wins.
_cgroup_free_mb() {
    local _root=$1 _proc=$2 _mnt=${3:-} _rel _dir _used _limit _free _name _min=""
    local _v2rel _v1rel _v2mnts _v1mnts _mroot _mpoint
    _cg_consider() {
        [ -n "$1" ] || return 0
        if [[ "$2" =~ ^[0-9]+$ ]]; then _free=$(( $1 - $2 )); else _free=$1; fi
        [ "$_free" -lt 0 ] && _free=0
        if [ -z "$_min" ] || [ "$_free" -lt "$_min" ]; then _min=$_free; fi
        return 0
    }
    # Only the first two colons delimit: a systemd unit name may contain one.
    _v2rel=$(awk '/^0::/ { print substr($0, 4); exit }' "$_proc" 2>/dev/null || true)
    _v1rel=$(awk '
        {
            a = index($0, ":"); if (a == 0) next
            rest = substr($0, a + 1)
            b = index(rest, ":"); if (b == 0) next
            if (substr(rest, 1, b - 1) ~ /(^|,)memory(,|$)/) { print substr(rest, b + 1); exit }
        }' "$_proc" 2>/dev/null || true)
    _v2mnts=$(_cg_mounts "$_mnt" cgroup2 | _cg_pick_mounts "$_v2rel")
    _v1mnts=$(_cg_mounts "$_mnt" memory | _cg_pick_mounts "$_v1rel")
    [ -n "$_v2mnts" ] || _v2mnts=$(printf '/\t%s' "$_root")
    [ -n "$_v1mnts" ] || _v1mnts=$(printf '/\t%s' "$_root/memory")

    while IFS=$'\t' read -r _mroot _mpoint; do
        [ -n "$_mpoint" ] || continue
        # The sentinel keeps $() from eating a trailing newline.
        _mroot=$(_cg_unesc "$_mroot"; printf X); _mroot=${_mroot%X}
        _mpoint=$(_cg_unesc "$_mpoint"; printf X); _mpoint=${_mpoint%X}
        [ -d "$_mpoint" ] || continue
        _rel=$(_cg_rel "$_mroot" "$_v2rel")
        while IFS= read -r -d '' _dir; do
            _used=$(_cg_read "$_dir/memory.current")
            for _name in memory.max memory.high; do
                _limit=$(_cg_limit "$(_cg_read "$_dir/$_name")")
                _cg_consider "$_limit" "$_used"
            done
        done < <(_cg_dirs "$_mpoint" "$_rel")
    done <<< "$_v2mnts"

    while IFS=$'\t' read -r _mroot _mpoint; do
        [ -n "$_mpoint" ] || continue
        _mroot=$(_cg_unesc "$_mroot"; printf X); _mroot=${_mroot%X}
        _mpoint=$(_cg_unesc "$_mpoint"; printf X); _mpoint=${_mpoint%X}
        [ -d "$_mpoint" ] || continue
        _rel=$(_cg_rel "$_mroot" "$_v1rel")
        while IFS= read -r -d '' _dir; do
            _used=$(_cg_read "$_dir/memory.usage_in_bytes")
            _limit=$(_cg_limit "$(_cg_read "$_dir/memory.limit_in_bytes")")
            _cg_consider "$_limit" "$_used"
        done < <(_cg_dirs "$_mpoint" "$_rel")
    done <<< "$_v1mnts"

    [ -n "$_min" ] && printf '%d' "$(( _min / 1048576 ))"
    return 0
}

# free + inactive, page size from the header (not 4096 on Apple Silicon); speculative and
# purgeable are already counted, so adding them double-counts.
_vm_stat_avail_mb() {
    awk '
        /page size of/ {
            for (i = 1; i < NF; i++) if ($i == "of") { ps = $(i + 1) + 0; break }
        }
        /^Pages (free|inactive)/ {
            gsub(/\./, "", $NF); pages += $NF
        }
        # A zero page count is a reading; only a missing page size is a failure.
        END { if (ps > 0) printf "%d", pages * ps / 1048576 }' || true
    return 0
}

# MemAvailable, not MemTotal; a lower cgroup allowance wins.
_usable_ram_mb() {
    local _meminfo=${1:-/proc/meminfo} _bytes _mb="" _free _avail
    if [ -r "$_meminfo" ]; then
        # Absent before Linux 3.14. `|| true`: errexit applies to a failing assignment in POSIX mode.
        _mb=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024; exit }' "$_meminfo") || true
        [ -n "$_mb" ] || _mb=$(awk '/^MemTotal:/ { printf "%d", $2 / 1024; exit }' "$_meminfo") || true
    elif _bytes=$(sysctl -n hw.memsize 2>/dev/null); then
        [[ "$_bytes" =~ ^[0-9]+$ ]] && _mb=$(( _bytes / 1048576 ))
        # macOS: vm_stat, falling back to hw.memsize. Zero is kept so a busy Mac builds at 1 job.
        _avail=$(vm_stat 2>/dev/null | _vm_stat_avail_mb || true)
        if [[ "$_avail" =~ ^[0-9]+$ ]]; then _mb=$_avail; fi
    fi
    _free=$(_cgroup_free_mb /sys/fs/cgroup /proc/self/cgroup /proc/self/mountinfo)
    if [[ "$_free" =~ ^[0-9]+$ ]]; then
        if [ -z "$_mb" ] || [ "$_free" -lt "$_mb" ]; then _mb=$_free; fi
    fi
    printf '%s' "$_mb"
    return 0
}

_llama_build_jobs() {
    _llama_jobs_for \
        "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)" \
        "$(_usable_ram_mb)"
}

# Opt-in: llama-server's first GPU pass JIT-compiles CUDA kernels and stalls installs.
_staged_validation_enabled() {
    local _raw="${UNSLOTH_LLAMA_STAGED_VALIDATION:-}"
    # Match install_llama_prebuilt.py staged_validation_enabled(): strip + lowercase.
    _raw="$(printf '%s' "$_raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]')"
    case "$_raw" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

_source_smoke_install_kind() {
    if [ "${_TRY_METAL_CPU_FALLBACK:-false}" = true ]; then
        printf '%s' "macos-arm64"
        return 0
    fi
    case "${GPU_BACKEND:-}" in
        cuda)
            case "$(uname -m 2>/dev/null || true)" in
                aarch64|arm64) printf '%s' "linux-arm64-cuda" ;;
                *) printf '%s' "linux-cuda" ;;
            esac
            ;;
        rocm) printf '%s' "linux-rocm" ;;
        *) printf '%s' "" ;;
    esac
}

# 10s timeout so a wedged NVIDIA driver cannot hang setup.
_setup_run_smi() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 10 "$@"
    else
        "$@"
    fi
}

# CUDA_VISIBLE_DEVICES "" or "-1" hides every NVIDIA device; nvidia-smi ignores it.
_setup_cvd_hides_nvidia() {
    [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] || return 1
    _setup_cvd_trim=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]')
    [ -z "$_setup_cvd_trim" ] || [ "$_setup_cvd_trim" = "-1" ]
}

# Present regardless of mask; falls back to /proc/driver/nvidia/gpus. Mirrors install.sh.
_setup_has_physical_nvidia_gpu() {
    _setup_nv_smi_wedged=""
    _setup_nvsmi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        _setup_nvsmi="nvidia-smi"
    elif [ -x "/usr/bin/nvidia-smi" ]; then
        _setup_nvsmi="/usr/bin/nvidia-smi"
    fi
    if [ -n "$_setup_nvsmi" ]; then
        # Captured, not piped, so timeout's 124 stays visible.
        _setup_nv_l_rc=0
        _setup_nv_l_out=$(_setup_run_smi "$_setup_nvsmi" -L 2>/dev/null) || _setup_nv_l_rc=$?
        if [ "$_setup_nv_l_rc" = "124" ]; then
            _setup_nv_smi_wedged=1
        fi
        if printf '%s\n' "$_setup_nv_l_out" \
           | awk '/^GPU[[:space:]]+[0-9]+:/{found=1} END{exit !found}'; then
            return 0
        fi
    fi
    if [ -d /proc/driver/nvidia/gpus ] && \
       [ -n "$(ls -A /proc/driver/nvidia/gpus 2>/dev/null)" ]; then
        return 0
    fi
    # Last: NVML / the CUDA driver API, which ship with the driver and not with nvidia-smi.
    if [ "${UNSLOTH_NVIDIA_LIBRARY_PROBE:-1}" != "0" ] && [ -f "$SCRIPT_DIR/nvidia_probe.py" ] \
            && command -v python3 >/dev/null 2>&1; then
        _setup_run_smi python3 -I "$SCRIPT_DIR/nvidia_probe.py" >/dev/null 2>&1 && return 0
    fi
    return 1
}

# NVIDIA allows a UUID abbreviated to any unique leading portion.
_setup_nv_idx_from_uuid() {
    # A prefix matching two cards selects no device, so count matches.
    _setup_run_smi "$_setup_nvsmi" --query-gpu=uuid --format=csv,noheader 2>/dev/null \
        | awk -v want="$1" '
            NF { gsub(/^[[:space:]]+|[[:space:]]+$/,""); if (index($0, want) == 1) { hits++; idx = NR-1 } }
            END { if (hits == 1) print idx }' || true
}

# Bounded, and uses the nvidia-smi already resolved: a wedged driver blocks it indefinitely.
_setup_nv_banner_fields() {
    _setup_nv_name=""; _setup_nv_sm=""; _setup_nv_driver=""
    _setup_nv_row=""; _setup_nv_cc=""; _setup_nv_ambiguous=""
    [ -n "${_setup_nvsmi:-}" ] || return 0
    [ -z "${_setup_nv_smi_wedged:-}" ] || return 0
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so resolve the mask against its rows.
    _setup_nv_idx=0
    _setup_nv_by_ordinal=1
    _setup_nv_vis="${CUDA_VISIBLE_DEVICES:-}"
    # Only the first entry decides: CUDA stops at the first invalid index (`2,-1` = device 2).
    _setup_nv_tok="${_setup_nv_vis%%,*}"
    case "$_setup_nv_tok" in
        '') ;;
        *[!0-9]*)
            _setup_nv_by_ordinal=""
            case "$_setup_nv_tok" in
                MIG-GPU-*)
                    # Pre-R470 MIG name embeds the parent UUID.
                    _setup_nv_tok="${_setup_nv_tok#MIG-}"; _setup_nv_tok="${_setup_nv_tok%%/*}"
                    _setup_nv_idx=$(_setup_nv_idx_from_uuid "$_setup_nv_tok") ;;
                MIG-*)
                    # R470+ MIG UUIDs carry nothing of the parent; `nvidia-smi -L` nests them.
                    _setup_nv_idx=$(_setup_run_smi "$_setup_nvsmi" -L 2>/dev/null | awk -v want="$_setup_nv_tok" '
                        /^GPU[[:space:]]+[0-9]+:/ { cur = $2 + 0 }
                        index($0, want) > 0 { print cur; exit }' || true) ;;
                *)
                    _setup_nv_idx=$(_setup_nv_idx_from_uuid "$_setup_nv_tok") ;;
            esac
            # An unresolved identity mask selects no device; row 0 is a different card.
            case "$_setup_nv_idx" in ''|*[!0-9]*) _setup_nv_idx=0; _setup_nv_ambiguous=1 ;; esac
            ;;
        *) _setup_nv_idx="$_setup_nv_tok" ;;
    esac
    # Clamp: `[` errors on values wider than a long and skips the range check.
    _setup_nv_idx=$(printf '%s' "$_setup_nv_idx" \
        | awk '{ n = $0 + 0; if (n < 0) n = 0; if (n > 9999) n = 9999; printf "%d", n }')
    _setup_nv_all=$(_setup_run_smi "$_setup_nvsmi" --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || true)
    _setup_nv_row=$(printf '%s\n' "$_setup_nv_all" \
        | awk -v idx="$_setup_nv_idx" 'NF { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx+0] }')
    [ -n "$_setup_nv_row" ] || return 0
    # CUDA defaults to FASTEST_FIRST but nvidia-smi lists in PCI order, so an ordinal maps to a row
    # only under PCI_BUS_ID or identical cards. An ordinal past the last row exposes no device.
    _setup_nv_rowcount=$(printf '%s\n' "$_setup_nv_all" | awk 'NF { n++ } END { print n+0 }')
    if [ -n "$_setup_nv_by_ordinal" ] && [ "$_setup_nv_idx" -ge "$_setup_nv_rowcount" ]; then
        _setup_nv_ambiguous=1
    fi
    if [ -n "$_setup_nv_by_ordinal" ]; then
        _setup_nv_order=$(printf '%s' "${CUDA_DEVICE_ORDER:-}" | tr '[:lower:]' '[:upper:]' | tr -d '[:space:]')
        _setup_nv_models=$(printf '%s\n' "$_setup_nv_all" \
            | awk -F, 'NF { k=$0; sub(/,[^,]*$/,"",k); if (!(k in s)) { s[k]; n++ } } END { print n+0 }')
        if [ "$_setup_nv_order" != "PCI_BUS_ID" ] && [ "$_setup_nv_models" -gt 1 ]; then
            _setup_nv_ambiguous=1
        fi
    fi
    # Split from the right: nvidia-smi does not quote commas in names.
    _setup_nv_driver=$(printf '%s' "$_setup_nv_row" | awk -F, 'NF>=3 { gsub(/^[[:space:]]+|[[:space:]]+$/,"",$NF); print $NF }')
    _setup_nv_cc=$(printf '%s' "$_setup_nv_row" | awk -F, 'NF>=3 { gsub(/^[[:space:]]+|[[:space:]]+$/,"",$(NF-1)); print $(NF-1) }')
    _setup_nv_name=$(printf '%s' "$_setup_nv_row" | awk -F, 'NF>=3 { out=$1; for(i=2;i<=NF-2;i++) out=out","$i; gsub(/^[[:space:]]+|[[:space:]]+$/,"",out); print out }')
    [ -n "$_setup_nv_name" ] || _setup_nv_name=$(printf '%s' "$_setup_nv_row" | awk -F, '{ gsub(/^[[:space:]]+|[[:space:]]+$/,"",$1); print $1 }')
    # Old nvidia-smi answers unsupported fields with a placeholder.
    case "$_setup_nv_name"   in '[N/A]'|'[Not Supported]'|'[Unknown Error]') _setup_nv_name="" ;; esac
    case "$_setup_nv_driver" in '[N/A]'|'[Not Supported]'|'[Unknown Error]') _setup_nv_driver="" ;; esac
    if [ -n "$_setup_nv_ambiguous" ]; then _setup_nv_name=""; _setup_nv_cc=""; fi
    case "$_setup_nv_cc" in
        [0-9]*.[0-9]*) _setup_nv_sm="sm_$(printf '%s' "$_setup_nv_cc" | awk -F. '{ print ($1*10)+$2 }')" ;;
    esac
    return 0
}

_setup_has_usable_nvidia_gpu() {
    if _setup_cvd_hides_nvidia; then
        return 1
    fi
    _setup_has_physical_nvidia_gpu
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

# ggml's -compress-mode=size (toolkit >= 12.8) does not load on a driver below 12.4 (#12842).
_cuda_driver_needs_uncompressed_fatbin() {
    _cuda_version_gt "12.4" "${1:-}"
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
import re
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
_backend_raw = payload.get("backend")
# Absent before #8520, and null when backend_for_install_kind() had no answer. str() on a
# non-string would diverge from the setup.ps1 twin (Python "[1, 2]" vs PowerShell "1 2").
backend = _backend_raw.strip() if isinstance(_backend_raw, str) else ""
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
# Name the backend: a Vulkan and a ROCm bundle print an identical line without it. The
# shape check keeps the line single-line and matches the setup.ps1 twin byte for byte.
if re.fullmatch(r"[A-Za-z0-9._+-]{1,32}", backend):
    message += f" -- {backend} backend"
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
rm -rf "$REPO_ROOT/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/backend/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/tmp/unsloth_compiled_cache"

# Cache-only: storage, cookies, settings, models and the database are untouched.
_clear_webview_caches() {
    # No HOME: bail rather than let `set -u` abort or "" expand to /Library and /.cache.
    [ -n "${HOME:-}" ] || return 0
    _wvc_bid="ai.unsloth.studio"
    _wvc_paths=()
    _wvc_root=""
    case "$(uname -s 2>/dev/null)" in
        Darwin)
            # Library/WebKit/<bid> is user storage and is left alone.
            _wvc_paths=("$HOME/Library/Caches/$_wvc_bid")
            _wvc_root="$HOME/Library/Application Support/$_wvc_bid"
            ;;
        Linux)
            # A relative XDG_DATA_HOME is invalid per XDG and dropped, matching Tauri.
            _wvc_data="${XDG_DATA_HOME:-$HOME/.local/share}"
            case "$_wvc_data" in /*) ;; *) _wvc_data="$HOME/.local/share" ;; esac
            _wvc_data="$_wvc_data/$_wvc_bid"
            _wvc_paths=(
                "$_wvc_data/WebKitCache"
                "$_wvc_data/CacheStorage"
                "$_wvc_data/serviceworkers"
            )
            _wvc_root="$_wvc_data"
            ;;
        *) return 0 ;;
    esac
    # Drop the version stamp first so the app retries the clear. `if`, not `&&`, under set -e.
    if [ -n "$_wvc_root" ]; then
        rm -f "$_wvc_root/.webview-cache-cleared" 2>/dev/null || true
    fi
    _wvc_cleared=false
    for _wvc_p in "${_wvc_paths[@]}"; do
        # -L too: a dangling symlink still occupies the path.
        [ -e "$_wvc_p" ] || [ -L "$_wvc_p" ] || continue
        rm -rf "$_wvc_p" 2>/dev/null && _wvc_cleared=true || true
    done
    if [ "$_wvc_cleared" = true ]; then
        substep "cleared stale WebView caches ($_wvc_bid); settings and data kept"
    fi
    return 0
}
# Not called here: a typo'd STUDIO_HOME override must be validated before clearing.

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
    # A typo in the override must fail fast instead of creating an empty dir. Mirrors setup.ps1.
    if [ ! -d "$_studio_override" ]; then
        echo "ERROR: $_studio_override_var=$_studio_override does not exist." >&2
        echo "       Run install.sh to create the install root before 'unsloth studio update'." >&2
        setup_fail 1 "$_studio_override_var=$_studio_override does not exist"
    fi
    if [ ! -w "$_studio_override" ]; then
        echo "ERROR: $_studio_override_var=$_studio_override is not writable." >&2
        setup_fail 1 "$_studio_override_var=$_studio_override is not writable"
    fi
    STUDIO_HOME="$(CDPATH= cd -P -- "$_studio_override" && pwd -P)" ||
        setup_fail 1 "Could not resolve $_studio_override_var=$_studio_override"
else
    STUDIO_HOME="$HOME/.unsloth/studio"
fi

STAGE_ROOT="${UNSLOTH_STUDIO_STAGE_ROOT:-}"
RUNTIME_ROOT="${STAGE_ROOT:-$STUDIO_HOME}"
VENV_DIR="$RUNTIME_ROOT/unsloth_studio"
if _mirror_enabled; then
    _mirror_spare_pwd=$PWD
    cd "$SCRIPT_DIR"
    _mirror_fallback spare
    cd "$_mirror_spare_pwd" 2>/dev/null || :
fi

# Same uv cache install.sh chose: `unsloth studio update` runs this script alone, and a
# second cache would copy every wheel across filesystems. Reads the LIVE marker; never writes.
_uv_no_cache_requested() {
    # --no-cache outranks --cache-dir. Same spelling as install.sh's _uv_no_cache_requested.
    case "$(printf '%s' "${UV_NO_CACHE:-}" | tr '[:upper:]' '[:lower:]')" in
        1|y|yes|t|true|on) return 0 ;;
    esac
    return 1
}

_uv_is_bucket_name() {
    # install.sh's _uv_is_bucket_name, verbatim: the two have to call the same directories uv's.
    case "$1" in
        *-v[0-9]*) ;;
        *) return 1 ;;
    esac
    case "${1##*-v}" in
        ''|*[!0-9]*) return 1 ;;
    esac
    case "${1%-v*}" in
        archive|binaries|builds|built-wheels|environments|flat-index) ;;
        git|interpreter|osv|python|sdists|simple|wheels) ;;
        *) return 1 ;;
    esac
    return 0
}

_uv_cache_probe_writable() {
    # mktemp (O_EXCL), not $$: a predictable name can be pre-created as a symlink.
    _uv_cache_probe=""
    if ! mkdir -p "$1" 2>/dev/null \
       || ! _uv_cache_probe=$(mktemp "$1/.unsloth-write-probe.XXXXXX" 2>/dev/null); then
        [ -z "$_uv_cache_probe" ] || rm -f "$_uv_cache_probe" 2>/dev/null || true
        unset _uv_cache_probe
        return 1
    fi
    # An ACL may allow create but deny unlink; retry, then trust whether the file is gone.
    _uv_cache_tries=0
    while :; do
        rm -f "$_uv_cache_probe" 2>/dev/null || true
        [ -e "$_uv_cache_probe" ] || break
        _uv_cache_tries=$((_uv_cache_tries + 1))
        if [ "$_uv_cache_tries" -ge 3 ]; then
            unset _uv_cache_probe _uv_cache_tries
            return 1
        fi
        sleep 1
    done
    unset _uv_cache_probe _uv_cache_tries
    return 0
}

_uv_cache_folds_case() {  # <dir>
    # Measured, not assumed: a Mac can mount APFS (folds) or ext4.
    _uvf_probe="$1/.unsloth-case-probe.$$-A"
    mkdir "$_uvf_probe" 2>/dev/null || { unset _uvf_probe; return 1; }
    if [ -d "$1/.unsloth-case-probe.$$-a" ]; then
        rmdir "$_uvf_probe" 2>/dev/null || true
        unset _uvf_probe
        return 0
    fi
    rmdir "$_uvf_probe" 2>/dev/null || true
    unset _uvf_probe
    return 1
}

_uv_store_key() {  # <entry name> <folds:1|0>  -> the name uv opens it as, or nonzero
    if _uv_is_bucket_name "$1"; then
        printf '%s' "$1"
        return 0
    fi
    [ "$2" = 1 ] || return 1
    case "$1" in *[[:upper:]]*) ;; *) return 1 ;; esac
    _uvk_lower=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    if _uv_is_bucket_name "$_uvk_lower"; then
        printf '%s' "$_uvk_lower"
        unset _uvk_lower
        return 0
    fi
    unset _uvk_lower
    return 1
}

_uv_cache_usable() {
    # Only uv's stores: a 0555 archive-* or interpreter-v4 aborts uv.
    _uv_cache_probe_writable "$1" || return 1
    _uv_cache_folds_case "$1" && _uvu_fold=1 || _uvu_fold=0
    for _uvu_bucket in "$1"/*; do
        _uvu_name=$(_uv_store_key "${_uvu_bucket##*/}" "$_uvu_fold") || continue
        # Only stores `uv pip install` creates. Probed even if a trivial install tolerates read-only:
        # a later source build or new interpreter aborts. Re-measure on a uv pin bump.
        case "${_uvu_name%-v*}" in
            archive|builds|built-wheels|git|interpreter|sdists|simple|wheels) ;;
            *) continue ;;
        esac
        if [ ! -d "$_uvu_bucket" ]; then
            # A file or symlink at a store path makes uv's mkdir fail.
            if [ -e "$_uvu_bucket" ] || [ -L "$_uvu_bucket" ]; then
                unset _uvu_bucket
                return 1
            fi
            continue
        fi
        if ! _uv_cache_probe_writable "$_uvu_bucket"; then
            unset _uvu_bucket _uvu_name _uvu_fold _uvu_ctl
            return 1
        fi
        # sdists-* .git is the one measured to abort when read-only. Index stores are probed one level
        # deep: uv rewrites that metadata on every resolve (0.12.1 and 0.10.7).
        case "$_uvu_name" in
            simple-* | wheels-*)
                for _uvu_shard in "$_uvu_bucket"/* "$_uvu_bucket"/index/*; do
                    # index/<hash> is used with --index-url (torch wheels) and must be writable too.
                    [ "${_uvu_shard#"$_uvu_bucket"/index/}" != "*" ] || continue
                    if [ ! -d "$_uvu_shard" ]; then
                        # A file or dangling symlink at a shard also aborts uv; a `*/` glob would skip it.
                        { [ -e "$_uvu_shard" ] || [ -L "$_uvu_shard" ]; } || continue
                        unset _uvu_bucket _uvu_name _uvu_fold _uvu_ctl _uvu_shard
                        return 1
                    fi
                    if ! _uv_cache_probe_writable "$_uvu_shard"; then
                        unset _uvu_bucket _uvu_name _uvu_fold _uvu_ctl _uvu_shard
                        return 1
                    fi
                done
                unset _uvu_shard
                ;;
        esac
        case "$_uvu_name" in sdists-*) _uvu_ctl=.git ;; *) _uvu_ctl="" ;; esac
        if [ -n "$_uvu_ctl" ] && ! _uv_control_files_writable "$_uvu_bucket" "$_uvu_ctl"; then
            unset _uvu_bucket _uvu_name _uvu_fold _uvu_ctl
            return 1
        fi
    done
    unset _uvu_bucket _uvu_name _uvu_fold _uvu_ctl _uvu_shard
    _uv_control_files_writable "$1" .lock || return 1
    return 0
}

_uv_control_files_writable() {  # <dir> <name>...
    # Only names uv needs writable (root .lock, sdists-v9/.git); rejecting more loses the warm cache.
    # Re-measure on a pin bump.
    _uvc_dir=$1
    shift
    for _uvc_name in "$@"; do
        _uvc_path="$_uvc_dir/$_uvc_name"
        { [ -e "$_uvc_path" ] || [ -L "$_uvc_path" ]; } || continue
        # A .lock directory also breaks uv; -f alone missed it.
        if [ ! -f "$_uvc_path" ] || [ ! -r "$_uvc_path" ] || [ ! -w "$_uvc_path" ]; then
            unset _uvc_dir _uvc_name _uvc_path
            return 1
        fi
    done
    unset _uvc_dir _uvc_name _uvc_path
    return 0
}

_uv_cache_warm() {
    # Package bytes, not metadata. Mirrors install.sh and unsloth_cli's _uv_cache_has_packages.
    [ -n "${1:-}" ] && [ -d "$1" ] && [ -r "$1" ] || return 1
    # On a case-folding filesystem Archive-V0 is archive-v0, and a glob does not fold.
    _uv_cache_folds_case "$1" && _uvw_fold=1 || _uvw_fold=0
    for _uvw_bucket in "$1"/*; do
        [ -d "$_uvw_bucket" ] || continue
        # Stricter than the probe, as install.sh: archive-v0.backup is not reusable.
        _uvw_base=$(_uv_store_key "${_uvw_bucket##*/}" "$_uvw_fold") || continue
        case "${_uvw_base%-v*}" in
            archive|builds|built-wheels|wheels|sdists) ;;
            *) continue ;;
        esac
        [ -r "$_uvw_bucket" ] && [ -x "$_uvw_bucket" ] || continue
        # -print -quit, not `| head`: under pipefail SIGPIPE on a large bucket reads as cold.
        _uvw_hit=$(find -L "$_uvw_bucket" -type f \
            ! -name CACHEDIR.TAG ! -name .git ! -name .gitignore \
            ! -name '*.lock' ! -name '*.msgpack' ! -name '*.http' ! -name '*.rev' \
            -print -quit 2>/dev/null) || true
        # find exits nonzero after an unreadable leaf even after printing the hit.
        if [ -n "$_uvw_hit" ]; then
            unset _uvw_bucket _uvw_hit _uvw_base _uvw_fold
            return 0
        fi
    done
    unset _uvw_bucket _uvw_hit _uvw_base _uvw_fold
    return 1
}

_recorded_uv_cache() {
    # One absolute path plus newline; tolerates a BOM and CR. The sentinel preserves trailing newlines.
    _ruc_raw=$(cat "$STUDIO_HOME/cache/uv-cache-dir" 2>/dev/null && printf x) || return 1
    _ruc_raw=${_ruc_raw%x}
    _ruc_raw=${_ruc_raw%"$_UV_MARKER_LF"}
    _ruc_raw=${_ruc_raw#"$_UV_MARKER_BOM"}
    _ruc_raw=${_ruc_raw%"$_UV_MARKER_CR"}
    case "$_ruc_raw" in
        # Absolute only: setup.sh has already changed directory.
        "") unset _ruc_raw; return 1 ;;
        /*) ;;
        *) unset _ruc_raw; return 1 ;;
    esac
    printf '%s' "$_ruc_raw"
    unset _ruc_raw
    return 0
}
_UV_MARKER_BOM=$(printf '\357\273\277')
_UV_MARKER_CR=$(printf '\r')
_UV_MARKER_LF=$(printf '\n.')
_UV_MARKER_LF=${_UV_MARKER_LF%.}

# Tiers: caller value, --no-cache, then the recorded marker or the Studio cache.
_uv_caller_value=false
# All-whitespace is not a caller's choice; must agree with install.sh's test.
case "${UV_CACHE_DIR-}" in
    *[![:space:]]*) _uv_caller_value=true ;;
esac
if [ "$_uv_caller_value" = true ]; then
    :
elif _uv_no_cache_requested; then
    # Unset, not left alone: uv parses an exported EMPTY value as `--cache-dir ''` and fails.
    unset UV_CACHE_DIR
else
    _uv_recorded=$(_recorded_uv_cache && printf x) || _uv_recorded=""
    _uv_recorded=${_uv_recorded%x}
    if [ -n "$_uv_recorded" ] && _uv_cache_warm "$_uv_recorded" \
       && _uv_cache_usable "$_uv_recorded"; then
        # Only while it holds packages and is writable: uv aborts on a read-only cache.
        UV_CACHE_DIR="$_uv_recorded"
        export UV_CACHE_DIR
    else
        UV_CACHE_DIR="$STUDIO_HOME/cache/uv"
        export UV_CACHE_DIR
        # Check the whole cache: a read-only archive-v0 passes a root probe and aborts uv.
        if ! _uv_cache_usable "$UV_CACHE_DIR"; then
            echo "[WARN] Cannot write to $UV_CACHE_DIR -- using uv's default cache." >&2
            unset UV_CACHE_DIR
        fi
    fi
    unset _uv_recorded
fi
unset _uv_caller_value
VENV_T5_530_DIR="$RUNTIME_ROOT/.venv_t5_530"
VENV_T5_550_DIR="$RUNTIME_ROOT/.venv_t5_550"
VENV_T5_510_DIR="$RUNTIME_ROOT/.venv_t5_510"

# Venv-gated: an empty override aborts at the venv check below, so do not clear first.
if [ -z "$STAGE_ROOT" ] && [ -x "$VENV_DIR/bin/python" ]; then
    _clear_webview_caches
fi

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
# Runtimes sit beside studio/ under UNSLOTH_HOME; captured before section 7 reassigns it.
# Stripped before anything else, like _studio_override above: " " counts as unset.
_MASTER_ROOT=$(printf '%s' "${UNSLOTH_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
if [ -n "$_MASTER_ROOT" ]; then
    case "$_MASTER_ROOT" in
        "~") _MASTER_ROOT="$HOME" ;;
        "~/"*) _MASTER_ROOT="$HOME/${_MASTER_ROOT#'~/'}" ;;
    esac
    # Absolutised against the caller's cwd: node and llama.cpp are chosen from different cwds.
    case "$_MASTER_ROOT" in
        /*) ;;
        *) _MASTER_ROOT="$PWD/$_MASTER_ROOT" ;;
    esac
    if [ -d "$_MASTER_ROOT" ]; then
        # Keep the expanded value when it cannot be canonicalised, as the Python resolver does.
        _master_root_canon=$(CDPATH= cd -P -- "$_MASTER_ROOT" 2>/dev/null && pwd -P) || _master_root_canon=""
        [ -z "$_master_root_canon" ] || _MASTER_ROOT="$_master_root_canon"
        unset _master_root_canon
    fi
fi

# "false" licenses installers to replace/rm unmarked runtime trees, so track it separately.
_RUNTIME_ROOT_IS_CUSTOM="$_STUDIO_HOME_IS_CUSTOM"
# Keyed on where runtimes land; staging excluded because STAGE_ROOT takes precedence.
if [ -z "${STAGE_ROOT:-}" ] && [ -n "$_MASTER_ROOT" ]; then
    # Canonicalised like _MASTER_ROOT, or a symlinked $HOME reads as custom.
    _rrc_legacy="$HOME/.unsloth"
    if [ -d "$_rrc_legacy" ]; then
        _rrc_canon=$(CDPATH= cd -P -- "$_rrc_legacy" 2>/dev/null && pwd -P) || _rrc_canon=""
        [ -z "$_rrc_canon" ] || _rrc_legacy="$_rrc_canon"
        unset _rrc_canon
    fi
    if [ "$_MASTER_ROOT" = "$_rrc_legacy" ]; then
        _RUNTIME_ROOT_IS_CUSTOM=false
    else
        _RUNTIME_ROOT_IS_CUSTOM=true
    fi
    unset _rrc_legacy
fi
# Directory-local evidence Unsloth created "$1": only prebuilt-installer metadata counts,
# since this runs right before an rm -rf.
_studio_owned_adoptable() {
    [ -f "$1/UNSLOTH_PREBUILT_INFO.json" ] && return 0
    [ -f "$1/UNSLOTH_NODE_PREBUILT_INFO.json" ] && return 0
    [ -f "$1/UNSLOTH_WHISPER_PREBUILT_INFO.json" ] && return 0
    return 1
}
# Marker probes need search (+x), not read (+r): in an unsearchable dir every probe reports absent, so our install looks foreign.
_studio_dir_unsearchable() {
    [ -d "$1" ] || return 1
    ( cd -- "$1" ) 2>/dev/null && return 1
    return 0
}

# Also needs +r for callers that list or replace the tree: mode 111 is searchable but still fails install_llama_prebuilt.py.
_studio_dir_unreadable() {
    [ -d "$1" ] || return 1
    _studio_dir_unsearchable "$1" && return 0
    ls -A -- "$1" >/dev/null 2>&1 && return 1
    return 0
}

# Mirrors Exit-PathAccessDenied in setup.ps1. owner-unverified: do not claim or advise deleting.
_path_access_denied() {
    _pad_dir="$1"
    _pad_label="$2"
    _pad_mode="${3:-}"
    step "permissions" "$_pad_label at $_pad_dir cannot be read: permission denied" "$C_ERR"
    if [ "$_pad_mode" = "owner-unverified" ]; then
        substep "Unsloth cannot confirm this folder is its own install while it is unreadable, so it will not tell you to remove it" "$C_WARN"
        substep "Restore access, or move the folder aside, then re-run setup:" "$C_WARN"
    else
        substep "This folder lives outside the app, so reinstalling Unsloth Studio reuses it and fails the same way" "$C_WARN"
        substep "Simplest fix: delete or rename $_pad_dir, then re-run setup (it is a managed cache and gets reinstalled)" "$C_WARN"
        substep "If deleting is denied too, it belongs to another user; restore access with:" "$C_WARN"
    fi
    substep "ls -ld \"$_pad_dir\"" "$C_WARN"
    substep "chmod -R u+rwX \"$_pad_dir\"" "$C_WARN"
    if [ "$_pad_mode" = "owner-unverified" ]; then
        setup_fail 1 "Permission denied reading $_pad_label at $_pad_dir. Unsloth cannot confirm that folder is its own install while it is unreadable: restore access, or move it aside, then re-run setup."
    fi
    setup_fail 1 "Permission denied reading the existing $_pad_label at $_pad_dir. Delete or rename that folder (Unsloth reinstalls it) or restore access, then re-run setup. Reinstalling the app does not reset it."
}

# POSIX follows a final symlink when the path ends in /, so "link/" is never -L. Strip it, but never past the root.
_studio_rstrip_slash() {
    _srs_path="$1"
    while [ "$_srs_path" != "/" ] && [ "${_srs_path%/}" != "$_srs_path" ]; do
        _srs_path="${_srs_path%/}"
    done
    printf '%s' "$_srs_path"
}

# An unsearchable ancestor makes a real path read as missing; report the deepest stat-able one.
_report_denied_ancestor() {
    _rda_probe="$(_studio_rstrip_slash "$1")"
    _rda_hops=0
    while [ ! -e "$_rda_probe" ] && [ "$_rda_probe" != "/" ] && [ "$_rda_probe" != "." ]; do
        # The hop cap breaks symlink cycles.
        if [ -L "$_rda_probe" ] && [ "$_rda_hops" -lt 40 ]; then
            _rda_hops=$((_rda_hops + 1))
            _rda_target="$(readlink -- "$_rda_probe")" || break
            case "$_rda_target" in
                /*) _rda_probe="$_rda_target" ;;
                *) _rda_probe="$(dirname -- "$_rda_probe")/$_rda_target" ;;
            esac
            _rda_probe="$(_studio_rstrip_slash "$_rda_probe")"
            continue
        fi
        _rda_probe="$(dirname -- "$_rda_probe")"
    done
    if _studio_dir_unsearchable "$_rda_probe"; then
        _path_access_denied "$_rda_probe" "$2" owner-unverified
    fi
}

_studio_path_shape() {
    if [ -L "$1" ]; then
        if [ -e "$1" ]; then printf 'a symlink'; else printf 'a dangling symlink'; fi
    elif [ -f "$1" ]; then printf 'a regular file'
    elif [ -d "$1" ]; then printf 'a directory'
    else printf 'an existing path'
    fi
}

_assert_studio_owned_or_absent() {
    _aso_dir="$1"
    _aso_label="$2"
    _aso_custom="${3:-$_STUDIO_HOME_IS_CUSTOM}"
    # -d alone treated a dangling symlink or file as absent and the caller rm -rf'd it.
    if [ ! -d "$_aso_dir" ] && [ ! -e "$_aso_dir" ] && [ ! -L "$_aso_dir" ]; then
        return 0
    fi
    if [ "$_aso_custom" = true ] && [ ! -f "$_aso_dir/$_STUDIO_OWNED_MARKER" ]; then
        if [ ! -d "$_aso_dir" ]; then
            echo "ERROR: $_aso_dir already exists and is not an Unsloth-owned $_aso_label." >&2
            echo "       It is $(_studio_path_shape "$_aso_dir"). Move it aside before re-running." >&2
            setup_fail 1 "$_aso_label path is not an Unsloth-owned install: $_aso_dir"
        fi
        if _studio_owned_adoptable "$_aso_dir"; then
            : > "$_aso_dir/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
            return 0
        fi
        # An unsearchable tree hides its own marker: report permissions, not ownership.
        if _studio_dir_unsearchable "$_aso_dir"; then
            _path_access_denied "$_aso_dir" "$_aso_label" owner-unverified
        fi
        echo "ERROR: $_aso_dir already exists and is not marked as an Unsloth-owned $_aso_label." >&2
        echo "       Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." >&2
        setup_fail 1 "$_aso_label path is not an Unsloth-owned install: $_aso_dir"
    fi
}


_packaged_frontend_available() {
    # Mode 0 trusts the wheel's dist (extraction mtimes are meaningless), unless a pyproject.toml
    # beside studio/ shows a source checkout (editable overlay).
    [ "${STUDIO_LOCAL_INSTALL:-}" = "0" ] &&
        [ ! -f "$REPO_ROOT/pyproject.toml" ] &&
        [ -f "$SCRIPT_DIR/frontend/dist/index.html" ]
}

if [ "$_LLAMA_ONLY" != "1" ]; then
# ── Detect whether frontend needs building ──
# Tauri owns its bundle; PyPI installs use the wheel's dist; only source installs rebuild by mtime.
if [ "${SKIP_STUDIO_FRONTEND:-0}" = "1" ]; then
    _NEED_FRONTEND_BUILD=false
    step "frontend" "bundled (Tauri)"
elif _packaged_frontend_available; then
    _NEED_FRONTEND_BUILD=false
    step "frontend" "bundled (pip install)"
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
fi  # end packaged/Tauri guard

# OXC validator runtime needs node/npm whenever its dir exists, regardless of dist staleness.
_OXC_DIR="$SCRIPT_DIR/backend/core/data_recipe/oxc-validator"
if [ "$_NEED_FRONTEND_BUILD" = false ] && [ ! -d "$_OXC_DIR" ]; then
    step "frontend" "up to date"
    verbose_substep "frontend dist is newer than source inputs"
else

# Vite 8 needs Node ^20.19 || >=22.12 || >=23 and npm >= 11.
# decide_node_source -> system | bundled | skip (unit-tested in tests/sh/test_node_decision.sh).
decide_node_source() {
    _dns_node="${1#v}"
    _dns_npm="$2"
    _dns_skip="$3"
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
if [ -n "$STAGE_ROOT" ]; then
    _NODE_PARENT="$RUNTIME_ROOT"
elif [ -n "$_MASTER_ROOT" ]; then
    _NODE_PARENT="$_MASTER_ROOT"
elif [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
    _NODE_PARENT="$STUDIO_HOME"
else
    _NODE_PARENT="$HOME/.unsloth"
fi
NODE_DIR="$_NODE_PARENT/node"

# Bound system node/npm probes so a broken binary on PATH cannot stall setup.
_probe_system_node_tool() {
    _PROBED_VER=""
    command -v "$1" >/dev/null 2>&1 || return 0
    _pnt_out="$(mktemp)"
    _pnt_rc=0
    _setup_probe_version "$1" "$_pnt_out" || _pnt_rc=$?
    if [ "$_pnt_rc" -eq 0 ]; then
        _PROBED_VER="$(head -n 1 "$_pnt_out")"
    elif [ "$_pnt_rc" -eq 124 ]; then
        substep "system $1 ($(command -v "$1")) did not answer --version within ${_SETUP_PROBE_SECONDS:-20}s; not using it" "$C_WARN"
    fi
    rm -f "$_pnt_out"
}
_probe_system_node_tool node
_SYS_NODE_VER="$_PROBED_VER"
_SYS_NPM_VER=""
if [ -n "$_SYS_NODE_VER" ]; then
    _probe_system_node_tool npm
    _SYS_NPM_VER="$_PROBED_VER"
fi
NODE_SOURCE="$(decide_node_source "$_SYS_NODE_VER" "$_SYS_NPM_VER" "${UNSLOTH_SKIP_NODE_INSTALL:-0}")"
_FRONTEND_SKIP=false

if [ "$NODE_SOURCE" = system ]; then
    step "node" "$_SYS_NODE_VER | npm $_SYS_NPM_VER (system)"
elif [ "$NODE_SOURCE" = bundled ]; then
    mkdir -p "$_NODE_PARENT"
    # install_node_prebuilt.py uses os.replace(); never displace a user-owned node dir.
    if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$NODE_DIR" "Node install" "$_RUNTIME_ROOT_IS_CUSTOM"
    fi
    substep "installing isolated Node (system Node/npm left untouched)..."
    # Runs before venv activation, so bare `python` may be absent.
    if [ -x "$VENV_DIR/bin/python" ]; then
        _NODE_PY="$VENV_DIR/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        _NODE_PY="python3"
    else
        _NODE_PY="python"
    fi
    _NODE_LOG="$(mktemp)"
    set +e
    for _node_try in default mirror; do
        if _is_verbose || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "1" ] || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "true" ]; then
            "$_NODE_PY" "$SCRIPT_DIR/install_node_prebuilt.py" --install-dir "$NODE_DIR" 2>&1 | tee -a "$_NODE_LOG" | _filter_download_output
            _NODE_STATUS=${PIPESTATUS[0]}
        else
            "$_NODE_PY" "$SCRIPT_DIR/install_node_prebuilt.py" --install-dir "$NODE_DIR" >>"$_NODE_LOG" 2>&1
            _NODE_STATUS=$?
        fi
        # Exit 3 (lock held) and 4 (permission denied) are not network failures.
        [ "$_NODE_STATUS" -ne 0 ] && [ "$_NODE_STATUS" -ne 3 ] && [ "$_NODE_STATUS" -ne 4 ] && [ "$_node_try" = default ] || break
        _mirror_switch node || break
    done
    set -e
    if [ "$_NODE_STATUS" -eq 3 ]; then
        step "node" "install blocked by another active Unsloth install" "$C_ERR"
        sed 's/^/   | /' "$_NODE_LOG" >&2; rm -f "$_NODE_LOG"
        substep "close other Unsloth installs and retry"
        setup_fail 3 "Node install is blocked by another active Unsloth install"
    elif [ "$_NODE_STATUS" -ne 0 ]; then
        step "node" "isolated Node install failed" "$C_ERR"
        sed 's/^/   | /' "$_NODE_LOG" >&2; rm -f "$_NODE_LOG"
        substep "install Node >= 20.19 (with npm >= 11) yourself and re-run, or check your network"
        setup_fail 1 "Could not install an isolated Node runtime"
    elif grep -Fq "keeping existing isolated Node" "$_NODE_LOG"; then
        if grep -Fq 'takeown /F' "$_NODE_LOG"; then
            sed 's/^/   | /' "$_NODE_LOG" >&2
        fi
        step "node" "update not applied, existing isolated Node kept" "$C_WARN"
    fi
    grep -Fq "already matches" "$_NODE_LOG" && verbose_substep "isolated Node already up to date"
    rm -f "$_NODE_LOG"
    if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ] && [ -d "$NODE_DIR" ]; then
        : > "$NODE_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    fi
    export PATH="$NODE_DIR/bin:$PATH"
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
    step "frontend" "up to date"
    verbose_substep "frontend dist is newer than source inputs"
else

# Install bun only into the isolated Node prefix; never globally on a system Node.
if command -v bun &>/dev/null; then
    substep "bun already installed ($(bun --version))"
elif [ -f "$SCRIPT_DIR/frontend/package-lock.json" ]; then
    verbose_substep "skipping global bun install (package-lock.json installs with npm ci)"
elif [ "$NODE_SOURCE" = bundled ]; then
    substep "installing bun..."
    # --allow-scripts=bun: npm >=11.16 gates install scripts and bun's postinstall fetches its binary.
    if run_maybe_quiet npm install -g bun --allow-scripts=bun "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" && command -v bun &>/dev/null; then
        substep "bun installed ($(bun --version))"
    else
        substep "bun install skipped (npm will be used instead)"
    fi
else
    verbose_substep "skipping global bun install on system Node (npm will be used)"
fi

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

# package-lock.json wins (`npm ci`); bun only without one. Not run_quiet: it exits on failure.
# bun's cache can corrupt and exit 0 with missing binaries, so verify, clear, and retry once.
_try_bun_install() {
    local _log _exit_code=0
    _log=$(mktemp)
    bun install --frozen-lockfile "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" >"$_log" 2>&1 || _exit_code=$?

    # bun may create .exe shims on Windows (Git Bash / MSYS2)
    if [ "$_exit_code" -eq 0 ] \
        && { [ -x node_modules/.bin/tsc ] || [ -f node_modules/.bin/tsc.exe ] || [ -f node_modules/.bin/tsc.bunx ]; } \
        && { [ -x node_modules/.bin/vite ] || [ -f node_modules/.bin/vite.exe ] || [ -f node_modules/.bin/vite.bunx ]; }; then
        rm -f "$_log"
        return 0
    fi

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

_FRONTEND_INSTALL_LOG=$(mktemp)
_CAPTURE_LOG="$_FRONTEND_INSTALL_LOG"
_bun_install_ok=false
_NPM_INSTALL=install
[ -f package-lock.json ] && _NPM_INSTALL=ci
if [ ! -f package-lock.json ] && [ -f bun.lock ] && command -v bun &>/dev/null; then
    substep "using bun for package install (faster)"
    if _try_bun_install; then
        _bun_install_ok=true
    else
        echo "   Clearing bun cache and retrying..."
        run_maybe_quiet bun pm cache rm || true
        if _try_bun_install; then
            _bun_install_ok=true
        fi
    fi
fi
if [ "$_bun_install_ok" = false ]; then
    # `|| rc=$?` keeps this off set -e's exit path so the hint branch is reachable.
    _npm_install_rc=0
    run_quiet_no_exit "npm $_NPM_INSTALL" npm "$_NPM_INSTALL" --no-fund --no-audit --loglevel=error "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" || _npm_install_rc=$?
    if [ "$_npm_install_rc" -ne 0 ] && _npm_mirror_retry "npm $_NPM_INSTALL"; then
        _npm_install_rc=0
    fi
    if [ "$_npm_install_rc" -ne 0 ]; then
        _suggest_npm_registry "$_FRONTEND_INSTALL_LOG"
        rm -f "$_FRONTEND_INSTALL_LOG"
        setup_fail "$_npm_install_rc" "Frontend dependency installation failed (exit code $_npm_install_rc)"
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

# Skip when NODE_SOURCE=skip: there is no suitable Node.
if [ -d "$_OXC_DIR" ] && [ "${NODE_SOURCE:-}" != skip ] && command -v npm &>/dev/null; then
    cd "$_OXC_DIR"
    _OXC_INSTALL_LOG=$(mktemp)
    _CAPTURE_LOG="$_OXC_INSTALL_LOG"
    _oxc_install_rc=0
    _NPM_INSTALL=install
    [ -f package-lock.json ] && _NPM_INSTALL=ci
    run_quiet_no_exit "npm $_NPM_INSTALL (oxc validator runtime)" npm "$_NPM_INSTALL" --no-fund --no-audit --loglevel=error "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" || _oxc_install_rc=$?
    if [ "$_oxc_install_rc" -ne 0 ] && _npm_mirror_retry "npm $_NPM_INSTALL (oxc validator runtime)"; then
        _oxc_install_rc=0
    fi
    _CAPTURE_LOG=""
    if [ "$_oxc_install_rc" -ne 0 ]; then
        _suggest_npm_registry "$_OXC_INSTALL_LOG"
        rm -f "$_OXC_INSTALL_LOG"
        setup_fail "$_oxc_install_rc" "OXC validator dependency installation failed (exit code $_oxc_install_rc)"
    fi
    rm -f "$_OXC_INSTALL_LOG"
    cd "$SCRIPT_DIR"
elif [ -d "$_OXC_DIR" ] && [ "${NODE_SOURCE:-}" != skip ]; then
    # No npm: skip rather than abort; the validator degrades. Mirrors setup.ps1.
    substep "OXC validator runtime skipped (no npm found); code validation degrades until Node is available" "$C_WARN"
fi

_remove_agent_instruction_files \
    "$SCRIPT_DIR/frontend/node_modules" \
    "$_OXC_DIR/node_modules"

[ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
[ -d "$REPO_ROOT/.venv_overlay" ] && rm -rf "$REPO_ROOT/.venv_overlay"
[ -d "$REPO_ROOT/.venv_t5" ] && rm -rf "$REPO_ROOT/.venv_t5"
[ -d "$REPO_ROOT/.venv_t5_530" ] && rm -rf "$REPO_ROOT/.venv_t5_530"
[ -d "$REPO_ROOT/.venv_t5_550" ] && rm -rf "$REPO_ROOT/.venv_t5_550"
# Do NOT delete $STUDIO_HOME/.venv here: install.sh handles migration.

_COLAB_NO_VENV=false
if [ ! -x "$VENV_DIR/bin/python" ]; then
    if [ "$IS_COLAB" = true ]; then
        # Colab: strip version constraints so pip keeps pre-installed packages.
        substep "Colab detected, installing Unsloth backend dependencies..."
        _COLAB_REQS_TMP="$(mktemp)"
        sed 's/[><=!~;].*//' "$SCRIPT_DIR/backend/requirements/studio.txt" \
            | grep -v '^#' | grep -v '^$' > "$_COLAB_REQS_TMP"
        if [ -s "$_COLAB_REQS_TMP" ]; then
            if ! run_quiet_no_exit "install Colab backend deps" pip install -q -r "$_COLAB_REQS_TMP"; then
                rm -f "$_COLAB_REQS_TMP"
                step "python" "Colab backend dependency install failed" "$C_ERR"
                setup_fail 1 "Colab backend dependency installation failed"
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
        setup_fail 1 "Virtual environment not found at $VENV_DIR"
    fi
elif [ -n "$STAGE_ROOT" ]; then
    VIRTUAL_ENV="$VENV_DIR"
    PATH="$VENV_DIR/bin:$PATH"
    export VIRTUAL_ENV PATH
    unset PYTHONHOME
    hash -r 2>/dev/null || true
else
    source "$VENV_DIR/bin/activate"
fi
# A PYTHONPATH torch would answer the probes below instead of the venv's (#11980); Colab has no venv.
[ "$_COLAB_NO_VENV" = true ] || unset PYTHONPATH

install_python_stack() {
    [ "${STUDIO_LOCAL_INSTALL:-0}" = 1 ] && [ -x "$VENV_DIR/bin/python" ] || _mirror_fallback
    python "$SCRIPT_DIR/install_python_stack.py"
}

# Phases a release adds below would be skipped by the update installing it; exec keeps the CLI's PID.
_setup_rerun_if_replaced() {
    if [ "${UNSLOTH_SETUP_RERUN:-}" = 1 ] || [ -z "$_SETUP_SELF_SUM" ]; then
        return 0
    fi
    local _now
    _now=$(cksum < "$_SETUP_SELF" 2>/dev/null) || return 0
    if [ -z "$_now" ] || [ "$_now" = "$_SETUP_SELF_SUM" ]; then
        return 0
    fi
    step "setup" "the update replaced this setup script; finishing with the new version"
    export UNSLOTH_SETUP_RERUN=1
    unset UNSLOTH_STUDIO_FULL_DEPS
    cd "$_SETUP_START_PWD" 2>/dev/null || :
    # execfail alone is not enough: under set -e a failed exec still ends the shell.
    shopt -s execfail
    set +e
    exec "${BASH:-bash}" "$_SETUP_SELF" ${_SETUP_ARGV[@]+"${_SETUP_ARGV[@]}"}
    set -e
    shopt -u execfail
    unset UNSLOTH_SETUP_RERUN
    cd "$SCRIPT_DIR"
    substep "could not start the updated setup script; continuing with this one" "$C_WARN"
}

# ── HTTP GET to stdout (supports curl and wget) ──
# install.sh takes either transport everywhere, so a wget-only box installs fine
# and then stalled here, where curl was the only way to fetch anything.
_setup_http_get() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$1"
    else
        return 1
    fi
}

# wget --timeout is per operation with 20 retries; --tries=1 plus outer timeout gives 5s.
_setup_http_get_timed() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --max-time 5 "$1"
    elif command -v wget >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            timeout 5 wget -qO- --timeout=5 --tries=1 "$1"
        else
            wget -qO- --timeout=5 --tries=1 "$1"
        fi
    else
        return 1
    fi
}

# Pinned uv release with SHA-256 instead of piping a remote script into a shell. Mirrors install.sh.
# See tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
# Bumping the version means bumping every hash, and every _setup_uv_pinned_wheel entry.
#   curl -sL https://github.com/astral-sh/uv/releases/download/<ver>/<asset>.sha256
_SETUP_UV_PINNED_VERSION="0.12.1"
# sha256 of astral's versioned install.sh for that release; the fallback below runs only those exact bytes.
_SETUP_UV_INSTALLER_SH_SHA256="d3f5412d38c99f9d024901843bf98206f0d2c6dbe64df40d0b740e2751ca62c1"

# Mirrors _uv_glibc_minor in install.sh: astral uses musl-static below its glibc floor.
_setup_uv_glibc_minor() {
    _sugm_line=$( (ldd --version 2>/dev/null || true) | head -1 )
    case "$_sugm_line" in *[Mm]usl*) return 1 ;; esac
    _sugm_ver=$(printf '%s\n' "$_sugm_line" | awk '{print $NF}')
    case "$_sugm_ver" in
        2.[0-9]*) : ;;
        *) _sugm_ver=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $NF}') ;;
    esac
    case "$_sugm_ver" in 2.[0-9]*) : ;; *) return 1 ;; esac
    _sugm_minor=${_sugm_ver#2.}
    _sugm_minor=${_sugm_minor%%.*}
    case "$_sugm_minor" in "" | *[!0-9]*) return 1 ;; esac
    echo "$_sugm_minor"
    return 0
}

_setup_uv_pinned_asset() {
    _supa_os=$(uname -s 2>/dev/null || echo unknown)
    _supa_arch=$(uname -m 2>/dev/null || echo unknown)
    case "$_supa_os" in
        Linux)
            # A 32-bit userland on a 64-bit kernel still reports x86_64 from uname.
            [ "$(getconf LONG_BIT 2>/dev/null || echo 0)" = "64" ] || return 1
            _supa_glibc=$(_setup_uv_glibc_minor) || return 1
            case "$_supa_arch" in
                x86_64|amd64)
                    [ "$_supa_glibc" -ge 17 ] 2>/dev/null || return 1
                    echo "uv-x86_64-unknown-linux-gnu.tar.gz 90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb" ;;
                aarch64|arm64)
                    [ "$_supa_glibc" -ge 28 ] 2>/dev/null || return 1
                    echo "uv-aarch64-unknown-linux-gnu.tar.gz 769d373e146692c639b5fbaae33b331c297a32e03d30448772051902df52bbf4" ;;
                *) return 1 ;;
            esac
            ;;
        Darwin)
            # Rosetta 2 reports x86_64 from a translated shell; astral reads the same sysctl.
            if [ "$_supa_arch" = "x86_64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
                _supa_arch=arm64
            fi
            case "$_supa_arch" in
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

_setup_uv_pinned_wheel() {
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

# GNU tar cannot read a wheel and minimal images lack unzip.
_setup_uv_unzip() {
    if command -v unzip >/dev/null 2>&1 && unzip -qo "$1" -d "$2" >/dev/null 2>&1; then return 0; fi
    case "$(tar --version 2>/dev/null)" in
        *bsdtar*) tar -xf "$1" -C "$2" 2>/dev/null && return 0 ;;
    esac
    command -v python3 >/dev/null 2>&1 &&
        python3 -m zipfile -e "$1" "$2" >/dev/null 2>&1
}

_setup_uv_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
    fi
}

# Cleanup on interrupt: no other trap is active here.
_setup_uv_cleanup_temporaries() {
    [ -n "${_SIUP_WORK:-}" ] && rm -rf "$_SIUP_WORK" 2>/dev/null || true
    [ -n "${_SIUP_STAGE:-}" ] && rm -f "$_SIUP_STAGE" 2>/dev/null || true
    [ -n "${_SIUP_STAGE2:-}" ] && rm -f "$_SIUP_STAGE2" 2>/dev/null || true
}

_setup_uv_on_signal() {
    trap - EXIT HUP INT TERM
    _setup_uv_cleanup_temporaries
    exit "$1"
}

_SETUP_LOGIN_PATH="$PATH"
_SIUP_WORK=""
_SIUP_STAGE=""
_SIUP_STAGE2=""

_setup_install_uv_pinned() {
    _SIUP_UNFETCHED=false
    _siup_spec=$(_setup_uv_pinned_asset) || return 1
    [ -n "$_siup_spec" ] || return 1
    _siup_asset=${_siup_spec%% *}
    _siup_want=${_siup_spec##* }
    command -v tar >/dev/null 2>&1 || return 1
    [ -n "$(_setup_uv_sha256 /dev/null)" ] || return 1
    [ -n "${HOME:-}" ] || return 1

    # astral's full destination priority, including XDG_DATA_HOME.
    _siup_dest="${UV_INSTALL_DIR:-${UV_UNMANAGED_INSTALL:-${XDG_BIN_HOME:-}}}"
    if [ -z "$_siup_dest" ] && [ -n "${XDG_DATA_HOME:-}" ]; then _siup_dest="$XDG_DATA_HOME/../bin"; fi
    [ -n "$_siup_dest" ] || _siup_dest="$HOME/.local/bin"
    _siup_work=$(mktemp -d 2>/dev/null) || return 1
    _SIUP_WORK="$_siup_work"
    trap _setup_uv_cleanup_temporaries EXIT
    trap '_setup_uv_on_signal 129' HUP
    trap '_setup_uv_on_signal 130' INT
    trap '_setup_uv_on_signal 143' TERM
    _siup_rc=1
    # A configured mirror is exclusive, matching astral's installer.
    if [ -n "${UV_DOWNLOAD_URL:-}" ]; then
        _siup_bases="${UV_DOWNLOAD_URL%/}"
    elif [ -n "${INSTALLER_DOWNLOAD_URL:-}" ]; then
        _siup_bases="${INSTALLER_DOWNLOAD_URL%/}"
    elif [ -n "${UV_INSTALLER_GHE_BASE_URL:-}" ]; then
        _siup_bases="${UV_INSTALLER_GHE_BASE_URL%/}/astral-sh/uv/releases/download/$_SETUP_UV_PINNED_VERSION"
    elif [ -n "${UV_INSTALLER_GITHUB_BASE_URL:-}" ]; then
        _siup_bases="${UV_INSTALLER_GITHUB_BASE_URL%/}/astral-sh/uv/releases/download/$_SETUP_UV_PINNED_VERSION"
    elif [ -n "${UNSLOTH_UV_WHEEL_MIRROR:-}" ]; then
        _siup_bases=""
        if _siup_wheel=$(_setup_uv_pinned_wheel "$_siup_asset"); then
            _siup_path=${_siup_wheel% *}
            _siup_bases="${UNSLOTH_UV_WHEEL_MIRROR%/}/${_siup_path%/*}"
            _siup_asset=${_siup_path##*/}
            _siup_want=${_siup_wheel##* }
        fi
    else
        _siup_bases="https://releases.astral.sh/github/uv/releases/download/$_SETUP_UV_PINNED_VERSION
https://github.com/astral-sh/uv/releases/download/$_SETUP_UV_PINNED_VERSION"
    fi
    _SIUP_UNFETCHED=true
    for _siup_base in $_siup_bases; do
        _setup_http_get "$_siup_base/$_siup_asset" > "$_siup_work/$_siup_asset" 2>/dev/null || continue
        [ -s "$_siup_work/$_siup_asset" ] || continue
        _SIUP_UNFETCHED=false
        [ "$(_setup_uv_sha256 "$_siup_work/$_siup_asset")" = "$_siup_want" ] || continue
        case "$_siup_asset" in
            *.whl) _setup_uv_unzip "$_siup_work/$_siup_asset" "$_siup_work" || continue ;;
            *) tar -xzf "$_siup_work/$_siup_asset" -C "$_siup_work" 2>/dev/null || continue ;;
        esac
        mkdir -p "$_siup_dest" 2>/dev/null || break
        _siup_ready=1
        for _siup_exe in uv uvx; do
            # `mv f d` moves into a directory named uv and reports success.
            if [ -d "$_siup_dest/$_siup_exe" ]; then _siup_ready=0; break; fi
            _siup_src=$(find "$_siup_work" -type f -name "$_siup_exe" 2>/dev/null | head -1)
            if [ -z "$_siup_src" ]; then _siup_ready=0; break; fi
            # Per-process staging name so racing installers cannot publish each other's file.
            _siup_stage=$(mktemp "$_siup_dest/.$_siup_exe.XXXXXX" 2>/dev/null) || { _siup_ready=0; break; }
            if [ "$_siup_exe" = "uv" ]; then _SIUP_STAGE="$_siup_stage"; else _SIUP_STAGE2="$_siup_stage"; fi
            if ! cp -f "$_siup_src" "$_siup_stage" 2>/dev/null; then _siup_ready=0; break; fi
            # 0755, not +x: umask 077 would leave uv unusable for other accounts.
            chmod 0755 "$_siup_stage" 2>/dev/null || true
            # Validate before publishing: the rename destroys the incumbent.
            if [ "$_siup_exe" = "uv" ] && ! _setup_probe_version "$_siup_stage"; then _siup_ready=0; break; fi
        done
        if [ "$_siup_ready" = "1" ] &&
           mv -f "$_SIUP_STAGE" "$_siup_dest/uv" 2>/dev/null &&
           mv -f "$_SIUP_STAGE2" "$_siup_dest/uvx" 2>/dev/null; then
            _siup_rc=0
        fi
        rm -f "$_SIUP_STAGE" "$_SIUP_STAGE2" 2>/dev/null || true
        _SIUP_STAGE=""
        _SIUP_STAGE2=""
        break
    done
    rm -rf "$_siup_work"
    _SIUP_WORK=""
    _SIUP_STAGE=""
    _SIUP_STAGE2=""
    trap - EXIT HUP INT TERM
    [ -x "$_siup_dest/uv" ] || _siup_rc=1
    if [ "$_siup_rc" = "0" ]; then
        export PATH="$_siup_dest:$PATH"
        _setup_persist_uv_path "$_siup_dest"
    fi
    return "$_siup_rc"
}

# Unpinned hosts: astral's versioned installer, run only if it is the exact pinned script (a host with no sha256 tool
# runs it as before). Non-zero when it is not run or fails.
_setup_uv_fallback_run() {
    _suf_tmp=$(mktemp) || return 1
    if ! _setup_http_get "https://astral.sh/uv/$_SETUP_UV_PINNED_VERSION/install.sh" > "$_suf_tmp"; then
        rm -f "$_suf_tmp"
        return 1
    fi
    _suf_sum=$(_setup_uv_sha256 "$_suf_tmp" 2>/dev/null) || _suf_sum=""
    if [ -n "$_suf_sum" ] && [ "$_suf_sum" != "$_SETUP_UV_INSTALLER_SH_SHA256" ]; then
        echo "uv installer script failed its sha256 check; not running it" >&2
        rm -f "$_suf_tmp"
        return 1
    fi
    if _is_verbose; then
        sh "$_suf_tmp" </dev/null
    else
        sh "$_suf_tmp" </dev/null > /dev/null 2>&1
    fi
    _suf_rc=$?
    rm -f "$_suf_tmp"
    return "$_suf_rc"
}

# astral's installer wrote a profile line for whichever destination it chose. This replaces that
# installer, so setup.sh run directly (local or Colab) has to do the same or the export above
# dies with this shell and every later run reinstalls uv. Both of astral's opt-outs apply, and
# fish is handled on its own terms since it reads none of the POSIX rc files.
# Is $2 one of the colon-separated entries of $1? Field splitting also globs, so pathname
# expansion is off for the walk and restored afterwards.
_setup_path_has_dir() {
    _sphd_glob=on
    case $- in *f*) _sphd_glob=off ;; esac
    set -f
    _sphd_found=1
    _sphd_old_ifs="$IFS"
    IFS=:
    for _sphd_entry in $1; do
        if [ "$_sphd_entry" = "$2" ]; then _sphd_found=0; break; fi
    done
    IFS="$_sphd_old_ifs"
    [ "$_sphd_glob" = on ] && set +f
    return "$_sphd_found"
}

# Repoint a persisted prepend inside conda (#5871). Exact whole-line match only; content is
# copied back into the original file so a symlinked rc keeps its link, mode and owner.
_unsloth_repoint_rc_line() {
    [ -f "$1" ] || return 1
    # Only lines under our `# Added by Unsloth` marker; never demote a user's own export.
    _URRL_OLD="$2" awk '
        $0 == ENVIRON["_URRL_OLD"] && prev ~ /^# Added by Unsloth/ { found = 1 }
        { prev = $0 }
        END { exit(found ? 0 : 1) }
    ' "$1" 2>/dev/null || return 1
    # Staged and renamed onto the resolved path; `readlink -f` is GNU-only, hence the walk.
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
    # `cp -p` keeps the mode. ENVIRON, not `-v`: awk decodes backslash escapes in `-v`.
    if { cp -p -- "$_urrl_real" "$_urrl_tmp" 2>/dev/null \
        || cp -- "$_urrl_real" "$_urrl_tmp" 2>/dev/null; } \
        && _URRL_OLD="$2" _URRL_NEW="$3" awk '
            $0 == ENVIRON["_URRL_OLD"] && prev ~ /^# Added by Unsloth/ { print ENVIRON["_URRL_NEW"]; prev = $0; next }
            { print; prev = $0 }
        ' "$_urrl_real" > "$_urrl_tmp" 2>/dev/null \
        && mv -f -- "$_urrl_tmp" "$_urrl_real" 2>/dev/null; then
        return 0
    fi
    rm -f -- "$_urrl_tmp" 2>/dev/null
    return 1
}

_unsloth_conda_env_active() {
    [ -n "${CONDA_PREFIX:-}" ] || [ -n "${CONDA_DEFAULT_ENV:-}" ]
}

_setup_persist_uv_path() {
    _supp_dir="$1"
    [ -n "$_supp_dir" ] || return 0
    [ -n "${HOME:-}" ] || return 0
    [ -z "${UV_NO_MODIFY_PATH:-}" ] || return 0
    [ -z "${UV_UNMANAGED_INSTALL:-}" ] || return 0
    # Compare against the login PATH entry by entry (globs in case patterns). Repoint before
    # the early return; _setup_repoint_only makes the rest a no-op.
    _setup_repoint_only=false
    if _setup_path_has_dir "${_SETUP_LOGIN_PATH:-$PATH}" "$_supp_dir"; then
        _unsloth_conda_env_active || return 0
        _setup_repoint_only=true
    fi
    # ~/.config, not XDG_CONFIG_HOME: where astral's installer put its fish file.
    _supp_fish_dir="$HOME/.config/fish/conf.d"
    if mkdir -p "$_supp_fish_dir" 2>/dev/null; then
        _supp_fish="$_supp_fish_dir/unsloth.fish"
        _supp_quoted=$(printf '%s' "$_supp_dir" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g")
        # fish_add_path prepends; the conda arm appends. -a -P makes it a PATH append, -m moves an
        # existing universal entry. https://fishshell.com/docs/current/cmds/fish_add_path.html
        _supp_fish_line="fish_add_path '$_supp_quoted'"
        if _unsloth_conda_env_active; then
            _supp_fish_line="fish_add_path -a -P -m '$_supp_quoted'"
            # Earlier spellings all prepend, so they are repointed rather than accepted.
            for _supp_stale in "fish_add_path '$_supp_quoted'" "fish_add_path -a '$_supp_quoted'" \
                               "fish_add_path -a -P '$_supp_quoted'"; do
                _unsloth_repoint_rc_line "$_supp_fish" "$_supp_stale" "$_supp_fish_line" || true
            done
        fi
        # Exact line, any spelling counts as present, so no duplicate line is added.
        if [ "$_setup_repoint_only" != true ] && ! grep -v '^[[:space:]]*#' "$_supp_fish" 2>/dev/null \
            | grep -qxF -e "fish_add_path '$_supp_quoted'" -e "fish_add_path -a '$_supp_quoted'" \
                        -e "fish_add_path -a -P '$_supp_quoted'" \
                        -e "fish_add_path -a -P -m '$_supp_quoted'"; then
            echo "# Added by Unsloth setup" >> "$_supp_fish"
            echo "$_supp_fish_line" >> "$_supp_fish"
        fi
    fi
    # Must be active, whole and on a line that sets PATH (not PYTHONPATH or a comment).
    _supp_path_line='(^|[^[:alnum:]_])(PATH[[:space:]]*=|fish_add_path|pathmunge|path_helper)'
    _supp_grep=$(printf '%s' "$_supp_dir" | sed 's/[].[\\()*+?{}|^$\/]/\\&/g')
    # Escaped: the line is double-quoted in the profile.
    _supp_literal=$(printf '%s' "$_supp_dir" | sed 's/[\\"$`]/\\&/g')
    # Inside conda, write an APPEND so conda's ordering survives (#5871).
    _supp_export_line="export PATH=\"$_supp_literal:\$PATH\""
    _supp_export_prepend="$_supp_export_line"
    # Also match install.sh's $HOME-relative prepend; $HOME stays unexpanded.
    _supp_export_home_prepend=""
    case "$_supp_dir" in
        "$HOME"/*)
            _supp_home_literal='$HOME'$(printf '%s' "${_supp_dir#$HOME}" | sed 's/[\\"$`]/\\&/g')
            _supp_export_home_prepend="export PATH=\"$_supp_home_literal:\$PATH\""
            ;;
    esac
    if _unsloth_conda_env_active; then
        _supp_export_line="export PATH=\"\$PATH:$_supp_literal\""
    fi
    if [ "$_setup_repoint_only" = true ]; then
        for _supp_profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" \
                             "$HOME/.bash_login" "${ZDOTDIR:-$HOME}/.zshrc" "${ZDOTDIR:-$HOME}/.zshenv"; do
            [ -f "$_supp_profile" ] || continue
            _unsloth_repoint_rc_line "$_supp_profile" "$_supp_export_prepend" \
                "$_supp_export_line" || true
            if [ -n "$_supp_export_home_prepend" ]; then
                _unsloth_repoint_rc_line "$_supp_profile" "$_supp_export_home_prepend" \
                    "$_supp_export_line" || true
            fi
        done
        return 0
    fi
    # Every startup file astral's installer wired: ~/.profile, each existing bash file, zsh under ZDOTDIR.
    for _supp_profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" \
                         "$HOME/.bash_login" "${ZDOTDIR:-$HOME}/.zshrc" "${ZDOTDIR:-$HOME}/.zshenv"; do
        if [ "$_supp_profile" != "$HOME/.profile" ] && [ ! -f "$_supp_profile" ]; then continue; fi
        # Repoint before the presence check: the check cannot see the $HOME-relative prepend.
        if _unsloth_conda_env_active; then
            _unsloth_repoint_rc_line "$_supp_profile" "$_supp_export_prepend" \
                "$_supp_export_line" || true
            if [ -n "$_supp_export_home_prepend" ]; then
                _unsloth_repoint_rc_line "$_supp_profile" "$_supp_export_home_prepend" \
                    "$_supp_export_line" || true
            fi
        fi
        if grep -v '^[[:space:]]*#' "$_supp_profile" 2>/dev/null \
            | grep -E "$_supp_path_line" \
            | grep -qE "(^|[^[:alnum:]_.~/-])$_supp_grep([^[:alnum:]_.~/-]|\$)"; then
            continue
        fi
        echo '' >> "$_supp_profile"
        echo '# Added by Unsloth setup' >> "$_supp_profile"
        echo "$_supp_export_line" >> "$_supp_profile"
    done
}

_SETUP_UV_PROBE_MISS=""
_SETUP_UV_LOOKED=""
_SETUP_UV_DIR=""
_SETUP_UV_TOO_OLD=""
# install.sh's UV_MIN_VERSION: older uv's Python manifest tops out below what torch needs.
_SETUP_UV_MIN_VERSION="0.9.3"

_setup_uv_version_at_least() {
    [ -n "$1" ] || return 1
    printf '%s\n' "$1" | awk -v floor="$2" '
        NR == 1 {
            # It has to be uv saying it. Another binary that runs and prints a version of its
            # own ("curl 8.9.1") would otherwise clear a floor of 0.9.3 on the strength of
            # being curl 8.
            if ($1 != "uv") { exit 1 }
            # A prerelease is the version it precedes minus something, so it is compared as that
            # version and refused when that lands exactly on the floor, as install.sh does.
            core = $2
            pre = (sub(/[-+].*$/, "", core) > 0)
            split(core, have, ".")
            if (have[1] !~ /^[0-9]+$/) { exit 1 }
            split(floor, want, ".")
            for (i = 1; i <= 3; i++) {
                h = (have[i] ~ /^[0-9]+$/) ? have[i] + 0 : 0
                w = (want[i] ~ /^[0-9]+$/) ? want[i] + 0 : 0
                if (h > w) { exit 0 }
                if (h < w) { exit 1 }
            }
            if (pre) { exit 1 }
            # Braced, like every other exit in this program: setup.sh is allowed exactly two
            # exits of its own (tests/sh/test_tauri_retry_failure_context.sh counts the lines),
            # and an awk exit indented on a line of its own reads as a third.
            { exit 0 }
        }
        { exit 1 }
    '
}

# Answers in _SETUP_UV_DIR: diagnostics would die with a command-substitution subshell.
_setup_find_installed_uv() {
    # Find a uv a previous run installed but this PATH lacks; it must run, not merely exist.
    # Cleared on entry so a second search does not report the first one's misses.
    _SETUP_UV_PROBE_MISS=""
    _SETUP_UV_LOOKED=""
    _SETUP_UV_DIR=""
    _SETUP_UV_TOO_OLD=""
    _sfu_seen=""
    # A file, not $(): a descendant holding the pipe open is the hang the ceiling prevents.
    _sfu_ver_file=""
    if command -v mktemp >/dev/null 2>&1; then
        _sfu_ver_file=$(mktemp 2>/dev/null) || _sfu_ver_file=""
    fi
    for _sfu_dir in "${UV_INSTALL_DIR:-}" "${UV_UNMANAGED_INSTALL:-}" "${XDG_BIN_HOME:-}" \
        "${XDG_DATA_HOME:+$XDG_DATA_HOME/../bin}" "${HOME:+$HOME/.local/bin}"; do
        [ -n "$_sfu_dir" ] || continue
        # Once per directory, so a hanging candidate costs the ceiling once.
        case "$_sfu_seen" in *"|$_sfu_dir|"*) continue ;; esac
        _sfu_seen="$_sfu_seen|$_sfu_dir|"
        _SETUP_UV_LOOKED="${_SETUP_UV_LOOKED:+$_SETUP_UV_LOOKED, }$_sfu_dir/uv"
        [ -x "$_sfu_dir/uv" ] || continue
        # Asked twice: one transient miss would downgrade uv via the pinned download.
        if _setup_probe_version "$_sfu_dir/uv" "${_sfu_ver_file:-/dev/null}" ||
           { sleep 2; _setup_probe_version "$_sfu_dir/uv" "${_sfu_ver_file:-/dev/null}"; }; then
            # `read`, not `cat`: must work on a bare PATH and not trip set -e on an empty file.
            _sfu_ver=""
            if [ -n "$_sfu_ver_file" ]; then
                read -r _sfu_ver < "$_sfu_ver_file" 2>/dev/null || _sfu_ver=""
            fi
            if _setup_uv_version_at_least "$_sfu_ver" "$_SETUP_UV_MIN_VERSION"; then
                _SETUP_UV_DIR="$_sfu_dir"
                if [ -n "$_sfu_ver_file" ]; then rm -f "$_sfu_ver_file" 2>/dev/null || :; fi
                unset _sfu_dir _sfu_ver _sfu_seen _sfu_ver_file
                return 0
            fi
            _SETUP_UV_TOO_OLD="$_sfu_dir/uv"
            continue
        fi
        _SETUP_UV_PROBE_MISS="$_sfu_dir/uv"
    done
    if [ -n "$_sfu_ver_file" ]; then rm -f "$_sfu_ver_file" 2>/dev/null || :; fi
    unset _sfu_dir _sfu_ver _sfu_seen _sfu_ver_file
    return 1
}

USE_UV=false
if command -v uv &>/dev/null; then
    USE_UV=true
elif _setup_find_installed_uv; then
    _setup_uv_dir="$_SETUP_UV_DIR"
    # Appended: a python beside uv must not shadow the staged $VENV_DIR/bin/python.
    export PATH="$PATH:$_setup_uv_dir"
    step "uv" "reusing the uv installed at $_setup_uv_dir (it was not on PATH)"
    USE_UV=true
    unset _setup_uv_dir
elif [ -n "$STAGE_ROOT" ]; then
    step "uv" "using pip inside the staged environment"
elif {
    if [ -n "${_SETUP_UV_TOO_OLD:-}" ]; then
        step "uv" "the uv at $_SETUP_UV_TOO_OLD is older than $_SETUP_UV_MIN_VERSION; installing the pinned release"
    elif [ -n "${_SETUP_UV_PROBE_MISS:-}" ]; then
        step "uv" "the uv at $_SETUP_UV_PROBE_MISS did not answer --version twice; installing the pinned release"
    elif [ -n "${_SETUP_UV_LOOKED:-}" ]; then
        step "uv" "no installed uv at $_SETUP_UV_LOOKED; installing the pinned release"
    fi
    _SETUP_UV_PINNED_OK=false
    if _setup_install_uv_pinned || { [ "$_SIUP_UNFETCHED" = true ] && _mirror_switch uvbin && _setup_install_uv_pinned; }; then
        _SETUP_UV_PINNED_OK=true
    else
        _setup_uv_fallback_run
    fi
}; then
    # Only for astral's installer; prepending after the pinned path could shadow the verified uv.
    [ "$_SETUP_UV_PINNED_OK" = true ] || export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null && USE_UV=true
fi

fast_install() {
    if [ "$USE_UV" = true ]; then
        uv pip install --python "$(command -v python)" "$@" && return 0
    fi
    python -m pip install "$@"
}

fast_install_sidecar() (
    unset UV_OVERRIDE
    fast_install "$@" && return 0
    _fis_rc=$?
    # A pin the PyPI mirror has not synced yet: one rerun with pypi.org behind it.
    for _fis_entry in ${_UNSLOTH_MIRROR_SPARE:-}; do
        [ "${_fis_entry%%|*}" = unsynced ] || continue
        for _fis_pair in $(printf '%s' "${_fis_entry#*|}" | tr '|' ' '); do export "$_fis_pair"; done
        fast_install "$@"
        return
    done
    return "$_fis_rc"
)

cd "$SCRIPT_DIR"

if [ "$_COLAB_NO_VENV" = true ]; then
    step "python" "backend deps installed into system Python"
    substep "continuing to llama.cpp install for GGUF inference support"
fi

# Fast update path: skip Python deps when the installed version matches PyPI.
_setup_install_is_verified() {
    # Single definition of a complete install, shared by the incomplete-install guard and offline rule.
    "$VENV_DIR/bin/python" -c "
import os, sys
sys.path.insert(0, sys.argv[1])
try:
    import install_manifest
except Exception:
    # Present but unimportable is damage, not an old release, and this is the
    # one file whose damage silences every check below. Absent keeps the old
    # escape: separating it from an old tree needs a RECORD walk here, and the
    # CLI already reports studio_install_manifest_missing.
    sys.exit(1 if os.path.isfile(os.path.join(sys.argv[1], 'install_manifest.py')) else 0)
import inspect
# Only skip the payload scan on a tree too old to offer it. Catching TypeError instead
# also swallowed one raised inside verify_install, and retried shallow on real damage.
deep = {'deep': True} if 'deep' in inspect.signature(install_manifest.verify_install).parameters else {}
sys.exit(0 if install_manifest.verify_install(**deep)['ok'] else 1)
" "$SCRIPT_DIR" 2>/dev/null
}

_uv_offline_requested() {
    # uv's boolish UV_OFFLINE spellings (uv 0.10.7): y, yes, t, true, on, 1.
    _uvo=${UV_OFFLINE:-}
    _uvo=${_uvo#"${_uvo%%[![:space:]]*}"}
    _uvo=${_uvo%"${_uvo##*[![:space:]]}"}
    case "$_uvo" in
        1 | [Tt] | [Tt][Rr][Uu][Ee] | [Yy] | [Yy][Ee][Ss] | [Oo][Nn]) unset _uvo; return 0 ;;
    esac
    unset _uvo
    return 1
}

_fast_path_escapes() {

    # Values are install_python_stack.py's ("1", "true", "yes", "on"), not _uv_offline_requested's.
    _fpe_full=${UNSLOTH_STUDIO_FULL_DEPS:-}
    _fpe_full=${_fpe_full#"${_fpe_full%%[![:space:]]*}"}
    _fpe_full=${_fpe_full%"${_fpe_full##*[![:space:]]}"}
    case "$_fpe_full" in
        1 | [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss] | [Oo][Nn])
            substep "UNSLOTH_STUDIO_FULL_DEPS is set -- forcing dependency pass..."
            _SKIP_PYTHON_DEPS=false
            ;;
    esac
    unset _fpe_full

    # A pre-#6483 install stuck on anyio>=4.14 would never reach the anyio repair (#6797).
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
    # A pre-pin tokenizers breaks `import transformers`; ask the metadata, not the import.
    if "$VENV_DIR/bin/python" -c "
import sys
from importlib.metadata import PackageNotFoundError, requires, version
try:
    from packaging.requirements import Requirement
    installed = version('tokenizers')
    windows = [
        req.specifier
        for req in (Requirement(raw) for raw in (requires('transformers') or []))
        if req.name == 'tokenizers' and req.marker is None
    ]
except (PackageNotFoundError, ImportError, ValueError, IndexError):
    sys.exit(1)
sys.exit(0 if windows and installed not in windows[0] else 1)
" 2>/dev/null; then
        substep "installed transformers rejects the installed tokenizers -- forcing dependency pass to repair..."
        _SKIP_PYTHON_DEPS=false
    fi
    # Failures and timeouts keep the fast path.
    _fpe_missing_torch=false
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 5 180 "$VENV_DIR/bin/python" \
            "$SCRIPT_DIR/install_python_stack.py" --missing-torch-needs-dependency-pass \
            >/dev/null 2>&1 && _fpe_missing_torch=true
    elif "$VENV_DIR/bin/python" "$SCRIPT_DIR/install_python_stack.py" \
            --missing-torch-needs-dependency-pass >/dev/null 2>&1; then
        _fpe_missing_torch=true
    fi
    if [ "$_fpe_missing_torch" = true ]; then
        # Offline the pass can only fail, and failing it loses the verified install.
        if [ "${_OFFLINE_FAST_PATH:-false}" = true ] || _uv_offline_requested; then
            if [ "$_SKIP_PYTHON_DEPS" = true ]; then
                substep "PyTorch is not installed but UV_OFFLINE is set -- left for the next online update"
            fi
        else
            substep "PyTorch is not installed -- forcing dependency pass to reinstall it..."
            _SKIP_PYTHON_DEPS=false
        fi
    fi
    unset _fpe_missing_torch
    # The pinned Diffusers main build is installed only by the pass.
    _fpe_diffusers=false
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 5 180 "$VENV_DIR/bin/python" \
            "$SCRIPT_DIR/install_python_stack.py" --diffusers-main-needs-dependency-pass \
            >/dev/null 2>&1 && _fpe_diffusers=true
    elif "$VENV_DIR/bin/python" "$SCRIPT_DIR/install_python_stack.py" \
            --diffusers-main-needs-dependency-pass >/dev/null 2>&1; then
        _fpe_diffusers=true
    fi
    if [ "$_fpe_diffusers" = true ] && [ "$_SKIP_PYTHON_DEPS" = true ]; then
        if [ "${_OFFLINE_FAST_PATH:-false}" = true ] || _uv_offline_requested; then
            substep "pinned Diffusers build is not installed but UV_OFFLINE is set -- left for the next online update"
        else
            substep "pinned Diffusers build is not installed -- forcing dependency pass..."
            _SKIP_PYTHON_DEPS=false
        fi
    fi
    unset _fpe_diffusers
    if [ -n "${UNSLOTH_DESKTOP_BACKEND_VERSION:-}" ]; then
        if ! "$VENV_DIR/bin/python" -c "
import re, sys
try:
    from packaging.version import parse as parse_v
except ImportError:
    def parse_v(v):
        match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', (v or '').strip())
        return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None
installed = parse_v(sys.argv[1])
required = parse_v(sys.argv[2])
sys.exit(0 if installed is not None and required is not None and installed >= required else 1)
" "$INSTALLED_VER" "$UNSLOTH_DESKTOP_BACKEND_VERSION" 2>/dev/null; then
            substep "$_PKG_NAME $INSTALLED_VER < $UNSLOTH_DESKTOP_BACKEND_VERSION (required by desktop app) -- forcing dependency pass to update..."
            _SKIP_PYTHON_DEPS=false
        fi
    fi
    # Only the dependency pass acts on an XPU pin. Mirrors setup.ps1.
    _setup_pin="${UNSLOTH_TORCH_INDEX_URL:-${UNSLOTH_TORCH_INDEX_FAMILY:-}}"
    # Strip query/fragment first: an authenticated mirror (…/whl/xpu?token=...) is a supported pin.
    _setup_pin="${_setup_pin%%\#*}"
    _setup_pin="${_setup_pin%%\?*}"
    # ALL trailing slashes, like the shared leaf parsers: a single %/ leaves "…/xpu/" behind.
    while [ "${_setup_pin%/}" != "$_setup_pin" ]; do _setup_pin="${_setup_pin%/}"; done
    # Exact, lowercased leaf: a suffix match (…/private-xpu) would force a pass _ensure_xpu_torch declines.
    _setup_pin_leaf=$(printf '%s' "${_setup_pin##*/}" | tr '[:upper:]' '[:lower:]')
    # Read from disk: a wedged Intel driver can hang `import torch`.
    _setup_pin_ok=false
    _setup_pin_is_xpu=false
    for _setup_pin_tv in "$VENV_DIR"/lib/python*/site-packages/torch/version.py; do
        [ -f "$_setup_pin_tv" ] || continue
        _setup_pin_ver=$(sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" "$_setup_pin_tv" | head -1)
        case "$_setup_pin_ver" in
            *+xpu)
                _setup_pin_is_xpu=true
                _setup_pin_maj=${_setup_pin_ver%%.*}
                _setup_pin_rest=${_setup_pin_ver#*.}
                _setup_pin_min=${_setup_pin_rest%%.*}
                case "$_setup_pin_maj$_setup_pin_min" in
                    *[!0-9]*) ;;
                    *) [ "$_setup_pin_maj" -eq 2 ] && [ "$_setup_pin_min" -ge 6 ] && \
                       [ "$_setup_pin_min" -lt 11 ] && _setup_pin_ok=true ;;
                esac
                ;;
        esac
        break
    done
    # A leftover generic triton shadows the XPU build. Non-XPU leaves matched exactly.
    _setup_known_nonxpu_leaf() {
        case "$1" in
            cpu|gfx[0-9]*) return 0 ;;
            cu[0-9]*) case "${1#cu}" in *[!0-9]*) return 1 ;; esac ;;
            rocm[0-9]*)
                # Both parts non-empty all-digits: rocm7., rocm7.2.1 are custom pins.
                _setup_rocm_rest="${1#rocm}"
                case "$_setup_rocm_rest" in
                    *.*.*) return 1 ;;
                    *.*)
                        case "${_setup_rocm_rest%%.*}" in *[!0-9]*) return 1 ;; esac
                        case "${_setup_rocm_rest#*.}" in "" | *[!0-9]*) return 1 ;; esac
                        ;;
                    *[!0-9]*) return 1 ;;
                esac
                ;;
            *) return 1 ;;
        esac
        return 0
    }
    _setup_pin_known_nonxpu=false
    _setup_known_nonxpu_leaf "$_setup_pin_leaf" && _setup_pin_known_nonxpu=true
    _setup_generic_triton=false
    if [ "$_setup_pin_is_xpu" = true ] || [ "$_setup_pin_leaf" = "xpu" ]; then
        for _setup_tri in "$VENV_DIR"/lib/python*/site-packages/triton-*.dist-info; do
            [ -d "$_setup_tri" ] && _setup_generic_triton=true && break
        done
    fi
    if [ "$_setup_pin_leaf" = "xpu" ] && [ "$_setup_pin_ok" = false ]; then
        substep "XPU index pinned but torch does not match -- forcing dependency pass to repair..."
        _SKIP_PYTHON_DEPS=false
    elif [ "$_setup_pin_is_xpu" = true ] && [ "$_setup_generic_triton" = true ]; then
        substep "generic triton shadows the XPU build -- forcing dependency pass to repair..."
        _SKIP_PYTHON_DEPS=false
    elif [ "$_setup_pin_is_xpu" = true ] && [ "$_setup_pin_known_nonxpu" = true ]; then
        substep "$_setup_pin_leaf pinned over an XPU wheel -- forcing dependency pass to migrate..."
        _SKIP_PYTHON_DEPS=false
    fi
    # Explicit cu*/rocm*/cpu pin over a torch labelled with another family; untagged torch is left alone.
    _setup_pin_have_family=""
    case "${_setup_pin_ver:-}" in
        *+cu[0-9]*) _setup_pin_have_family=cu ;;
        *+rocm*) _setup_pin_have_family=rocm ;;
        *+cpu) _setup_pin_have_family=cpu ;;
        *+xpu) _setup_pin_have_family=xpu ;;
    esac
    _setup_pin_want_family=""
    if [ "$_setup_pin_known_nonxpu" = true ]; then
        case "$_setup_pin_leaf" in
            cu[0-9]*) _setup_pin_want_family=cu ;;
            rocm[0-9]* | gfx[0-9]*) _setup_pin_want_family=rocm ;;
            cpu) _setup_pin_want_family=cpu ;;
        esac
    fi
    if [ "$_SKIP_PYTHON_DEPS" = true ] && [ -n "$_setup_pin_want_family" ] \
        && [ -n "$_setup_pin_have_family" ] && [ "$_setup_pin_have_family" != xpu ] \
        && [ "$_setup_pin_have_family" != "$_setup_pin_want_family" ]; then
        substep "$_setup_pin_leaf pinned over a +$_setup_pin_have_family torch wheel -- forcing dependency pass to reinstall torch from the pin..."
        _SKIP_PYTHON_DEPS=false
    fi
    # Explicit under `set -e`: the decision is in _SKIP_PYTHON_DEPS.
    return 0
}

_SKIP_PYTHON_DEPS=false
_SKIP_VERSION_CHECK=false
# Set here so an exported value from the caller cannot reach it.
_OFFLINE_FAST_PATH=false
if [ "$_COLAB_NO_VENV" = true ]; then
    _SKIP_VERSION_CHECK=true
fi
_PKG_NAME="${STUDIO_PACKAGE_NAME:-unsloth}"
if [ "$_SKIP_VERSION_CHECK" != true ] && [ "${SKIP_STUDIO_BASE:-0}" != "1" ] && [ "${STUDIO_LOCAL_INSTALL:-0}" != "1" ]; then
    _INSTALLED_VERSION_PROBE_EXIT=0
    if INSTALLED_VER=$("$VENV_DIR/bin/python" -c "
import sys
sys.path.insert(0, sys.argv[2])
import install_manifest
version, conflict = install_manifest.installed_version_probe(sys.argv[1], ('unsloth-zoo',))
print(version)
sys.exit(2 if conflict else (0 if version else 1))
" "$_PKG_NAME" "$SCRIPT_DIR" 2>/dev/null); then
        :
    else
        _INSTALLED_VERSION_PROBE_EXIT=$?
        INSTALLED_VER=""
    fi

    LATEST_VER=$(_setup_http_get_timed "https://pypi.org/pypi/$_PKG_NAME/json" 2>/dev/null \
        | "$VENV_DIR/bin/python" -c "import sys,json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null \
        || echo "")

    if [ "$_INSTALLED_VERSION_PROBE_EXIT" -eq 2 ]; then
        substep "duplicate metadata found for a core package -- forcing package repair..."
    elif [ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ] && [ "$INSTALLED_VER" = "$LATEST_VER" ]; then
        step "python" "$_PKG_NAME $INSTALLED_VER is up to date"
        _SKIP_PYTHON_DEPS=true
        # An interrupted install leaves the package current while studio.txt never finished.
        if ! _setup_install_is_verified; then
            substep "studio install incomplete -- forcing dependency pass to repair..."
            _SKIP_PYTHON_DEPS=false
        fi
        _fast_path_escapes
    elif [ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ]; then
        substep "$_PKG_NAME $INSTALLED_VER -> $LATEST_VER available, updating..."
    elif [ -z "$LATEST_VER" ]; then
        # UV_OFFLINE is not a blip: keep a verified tree, still held to the fast-path escapes.
        if [ -n "$INSTALLED_VER" ] && _uv_offline_requested && _setup_install_is_verified; then
            substep "PyPI is unreachable and UV_OFFLINE is set -- keeping the verified install"
            _SKIP_PYTHON_DEPS=true
            _OFFLINE_FAST_PATH=true
            _fast_path_escapes
        else
            substep "could not reach PyPI, updating to be safe..."
        fi
    fi
fi

# The fast path skips ROCm repair. Exit 0 forces the pass; failures keep the fast path.
if [ "$_SKIP_PYTHON_DEPS" = true ] && [ -x "$VENV_DIR/bin/python" ]; then
    _setup_amd_torch_stale=false
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 5 180 "$VENV_DIR/bin/python" \
            "$SCRIPT_DIR/install_python_stack.py" --amd-torch-needs-dependency-pass \
            >/dev/null 2>&1 && _setup_amd_torch_stale=true
    elif "$VENV_DIR/bin/python" "$SCRIPT_DIR/install_python_stack.py" \
            --amd-torch-needs-dependency-pass >/dev/null 2>&1; then
        _setup_amd_torch_stale=true
    fi
    if [ "$_setup_amd_torch_stale" = true ]; then
        substep "installed PyTorch is not a ROCm build on this AMD host -- forcing dependency pass to repair..."
        substep "   (set UNSLOTH_TORCH_BACKEND=cpu to keep a deliberate CPU install)"
        _SKIP_PYTHON_DEPS=false
    fi
fi

# Same for an NVIDIA host left on a CPU wheel. setup.ps1 heals this at its stale-venv check.
if [ "$_SKIP_PYTHON_DEPS" = true ] && [ -x "$VENV_DIR/bin/python" ]; then
    _setup_cuda_torch_stale=false
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 5 180 "$VENV_DIR/bin/python" \
            "$SCRIPT_DIR/install_python_stack.py" --cuda-torch-needs-dependency-pass \
            >/dev/null 2>&1 && _setup_cuda_torch_stale=true
    elif "$VENV_DIR/bin/python" "$SCRIPT_DIR/install_python_stack.py" \
            --cuda-torch-needs-dependency-pass >/dev/null 2>&1; then
        _setup_cuda_torch_stale=true
    fi
    if [ "$_setup_cuda_torch_stale" = true ]; then
        # Offline the pass can only fail, and failing it loses the verified install.
        if [ "${_OFFLINE_FAST_PATH:-false}" = true ] || _uv_offline_requested; then
            substep "installed PyTorch cannot use this NVIDIA GPU but UV_OFFLINE is set -- left for the next online update"
        else
            substep "installed PyTorch cannot use this NVIDIA GPU -- forcing dependency pass to repair..."
            substep "   (set UNSLOTH_TORCH_BACKEND=cpu to keep a deliberate CPU install)"
            _SKIP_PYTHON_DEPS=false
        fi
    fi
fi

# Same for an install that chose XPU torch for its Intel GPU (install.sh's auto route, recorded
# in the manifest) but now holds another wheel: _ensure_xpu_torch runs only inside the pass.
if [ "$_SKIP_PYTHON_DEPS" = true ] && [ -x "$VENV_DIR/bin/python" ]; then
    _setup_xpu_torch_stale=false
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 5 180 "$VENV_DIR/bin/python" \
            "$SCRIPT_DIR/install_python_stack.py" --xpu-torch-needs-dependency-pass \
            >/dev/null 2>&1 && _setup_xpu_torch_stale=true
    elif "$VENV_DIR/bin/python" "$SCRIPT_DIR/install_python_stack.py" \
            --xpu-torch-needs-dependency-pass >/dev/null 2>&1; then
        _setup_xpu_torch_stale=true
    fi
    if [ "$_setup_xpu_torch_stale" = true ]; then
        if [ "${_OFFLINE_FAST_PATH:-false}" = true ] || _uv_offline_requested; then
            substep "installed PyTorch is not the XPU build this install chose but UV_OFFLINE is set -- left for the next online update"
        else
            substep "installed PyTorch is not the XPU build this install chose -- forcing dependency pass to repair..."
            _SKIP_PYTHON_DEPS=false
        fi
    fi
fi

if [ "$_SKIP_PYTHON_DEPS" = false ]; then
    install_python_stack
    _setup_rerun_if_replaced
else
    step "python" "dependencies up to date"
    verbose_substep "python deps check: installed=$_PKG_NAME@${INSTALLED_VER:-unknown} latest=${LATEST_VER:-unknown}"
fi

# ── 6b. Pre-install transformers 5.x into .venv_t5_530/, .venv_t5_550/, and .venv_t5_510/ ──
# Separate dirs avoid runtime pip overhead; training prepends the right one to sys.path.
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
# Audited AND installed from here: a name the audit cannot reach on disk reads stale forever.
_SIDECAR_COMMON_PINS="huggingface_hub==1.8.0 hf_xet==1.4.2"

# Remnants of a failed tiktoken install shadow the ambient copy and nothing else clears them.
_sidecar_drop_tiktoken() {
    for _sdt_entry in "$1"/tiktoken "$1"/tiktoken_ext "$1"/tiktoken.libs "$1"/tiktoken-*.dist-info; do
        [ -e "$_sdt_entry" ] && rm -rf "$_sdt_entry" 2>/dev/null
    done
    _sdt_left=0
    for _sdt_entry in "$1"/tiktoken "$1"/tiktoken_ext "$1"/tiktoken.libs "$1"/tiktoken-*.dist-info; do
        [ -e "$_sdt_entry" ] && _sdt_left=1
    done
    unset _sdt_entry
    if [ "$_sdt_left" = 1 ]; then
        unset _sdt_left
        return 1
    fi
    unset _sdt_left
    return 0
}

_sidecar_retire_after_failed_tiktoken() {
    rm -rf "$1" 2>/dev/null || true
    substep "the $2 sidecar kept part of a failed tiktoken install; retired, rebuilt on the next update"
}

_sidecar_top_up_tiktoken() {
    _stt_dir="$1"
    _stt_label="$2"
    # Offline: fast_install's pip fallback would reach the network.
    [ "${_OFFLINE_FAST_PATH:-false}" = true ] && return 0
    _uv_offline_requested && return 0
    # RECORD is written last; a recordless dist-info goes first since uv cannot uninstall it.
    for _stt_info in "$_stt_dir"/tiktoken-*.dist-info; do
        if [ -d "$_stt_info" ] && [ ! -f "$_stt_info/RECORD" ]; then
            rm -rf "$_stt_info" || true   # `|| true`: set -e; the reinstall below handles a leftover
        fi
    done
    unset _stt_info
    for _stt_meta in "$_stt_dir"/tiktoken-*.dist-info/METADATA; do
        if [ -f "$_stt_meta" ] && [ -f "${_stt_meta%METADATA}RECORD" ] && [ -f "$_stt_dir/tiktoken/__init__.py" ]; then
            unset _stt_meta
            return 0
        fi
    done
    unset _stt_meta
    # Dropping the metadata makes uv reinstall instead of calling the pin satisfied.
    for _stt_info in "$_stt_dir"/tiktoken-*.dist-info; do
        [ -d "$_stt_info" ] && { rm -rf "$_stt_info" || true; }
    done
    unset _stt_info
    _mirror_fallback
    if ! fast_install_sidecar --target "$_stt_dir" --no-deps --upgrade "tiktoken" >/dev/null 2>&1; then
        if _sidecar_drop_tiktoken "$_stt_dir"; then
            substep "could not install tiktoken into the $_stt_label sidecar -- Qwen tokenizers may fail"
        else
            _sidecar_retire_after_failed_tiktoken "$_stt_dir" "$_stt_label"
        fi
    fi
    return 0
}

_sidecar_current() {
    _sc_dir="$1"
    _sc_ver="$2"
    [ -d "$_sc_dir" ] || return 1
    # No shim: answer from the grep, as setup.ps1 does.
    if [ ! -f "$SCRIPT_DIR/install_manifest.py" ]; then
        _target_has_pkg_version "$_sc_dir" "transformers" "$_sc_ver"
        return $?
    fi
    _sc_python="$VENV_DIR/bin/python"
    if [ ! -x "$_sc_python" ]; then
        _sc_python=$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)
    fi
    if [ -z "$_sc_python" ]; then
        unset _sc_python
        _target_has_pkg_version "$_sc_dir" "transformers" "$_sc_ver"
        return $?
    fi
    # Bounded where timeout exists: the shim cannot interrupt a stalled read.
    # shellcheck disable=SC2086  # the pins are a deliberate word-split list
    if command -v timeout >/dev/null 2>&1; then
        _sc_out=$(timeout -k 5 60 "$_sc_python" "$SCRIPT_DIR/install_manifest.py" sidecar "$_sc_dir" \
            "transformers==$_sc_ver" $_SIDECAR_COMMON_PINS 2>/dev/null)
        _sc_rc=$?
    else
        _sc_out=$("$_sc_python" "$SCRIPT_DIR/install_manifest.py" sidecar "$_sc_dir" \
            "transformers==$_sc_ver" $_SIDECAR_COMMON_PINS 2>/dev/null)
        _sc_rc=$?
    fi
    # 124 is timeout's TERM, 137 its KILL.
    if [ "$_sc_rc" -eq 124 ] || [ "$_sc_rc" -eq 137 ]; then
        _sc_out="sidecar: audit did not answer within 60 seconds"
    fi
    unset _sc_python
    case "$_sc_out" in
        "sidecar: current")
            unset _sc_out _sc_rc
            return 0
            ;;
        sidecar:*)
            verbose_substep "sidecar $_sc_dir: ${_sc_out#sidecar: }"
            unset _sc_out _sc_rc
            return 1
            ;;
    esac
    unset _sc_out
    if [ "$_sc_rc" -ne 0 ]; then
        verbose_substep "sidecar $_sc_dir: audit failed (exit $_sc_rc)"
        unset _sc_rc
        return 1
    fi
    unset _sc_rc
    # An old shim exits 0 silently: fall back to the grep.
    _target_has_pkg_version "$_sc_dir" "transformers" "$_sc_ver"
}

_install_sidecar() {
    _is_dir="$1"
    _is_ver="$2"
    _is_label="$3"
    _mirror_fallback
    _assert_studio_owned_or_absent "$_is_dir" "transformers $_is_label sidecar venv"
    [ -d "$_is_dir" ] && rm -rf "$_is_dir"
    mkdir -p "$_is_dir"
    : > "$_is_dir/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    run_quiet "install transformers $_is_ver" fast_install_sidecar --target "$_is_dir" --no-deps "transformers==$_is_ver"
    # From $_SIDECAR_COMMON_PINS, not a copy: a pin demanded but never installed reads stale forever.
    for _is_pin in $_SIDECAR_COMMON_PINS; do
        run_quiet "install ${_is_pin%%==*} for $_is_label" fast_install_sidecar --target "$_is_dir" --no-deps "$_is_pin"
    done
    unset _is_pin
    # Optional, as in setup.ps1; _sidecar_top_up_tiktoken retries it later.
    if ! run_quiet_no_exit "install tiktoken for $_is_label" fast_install_sidecar --target "$_is_dir" --no-deps "tiktoken"; then
        if _sidecar_drop_tiktoken "$_is_dir"; then
            substep "could not install tiktoken into the $_is_label sidecar -- Qwen tokenizers may fail"
        else
            _sidecar_retire_after_failed_tiktoken "$_is_dir" "$_is_label"
        fi
    fi
    step "transformers" "$_is_ver pre-installed"
}

_NEED_T5_530=false
_NEED_T5_550=false
_NEED_T5_510=false
# Under UV_OFFLINE the legacy-tree migration (wipe + rebuilds) waits for an online update.
if [ -d "$STUDIO_HOME/.venv_t5" ] && { [ "${_OFFLINE_FAST_PATH:-false}" = true ] || _uv_offline_requested; }; then
    substep "legacy transformers sidecar left in place -- UV_OFFLINE is set, migration waits for the next online update"
elif [ -d "$STUDIO_HOME/.venv_t5" ]; then
    # A staged run's venvs may never be activated, so only the live update migrates.
    if [ -z "$STAGE_ROOT" ]; then
        _assert_studio_owned_or_absent "$STUDIO_HOME/.venv_t5" "legacy transformers sidecar venv"
        rm -rf "$STUDIO_HOME/.venv_t5"
    fi
    _NEED_T5_530=true
    _NEED_T5_550=true
    _NEED_T5_510=true
fi
_sidecar_current "$VENV_T5_530_DIR" "5.3.0" || _NEED_T5_530=true
_sidecar_current "$VENV_T5_550_DIR" "5.5.0" || _NEED_T5_550=true
_sidecar_current "$VENV_T5_510_DIR" "5.10.2" || _NEED_T5_510=true
# Sidecar rebuilds reach the network; defer under offline keep. Deferred tiers are not reported current.
_DEFER_T5_530=false
_DEFER_T5_550=false
_DEFER_T5_510=false
# The pip fallback does not read UV_OFFLINE.
if [ "${_OFFLINE_FAST_PATH:-false}" != true ] && _uv_offline_requested; then
    for _ofp in "530 5.3.0" "550 5.5.0" "510 5.10.2"; do
        # Not `set --`: at top-level scope that overwrites the script's own "$@".
        _ofp_key=${_ofp%% *}; _ofp_ver=${_ofp#* }
        if eval "[ \"\$_NEED_T5_$_ofp_key\" = true ]"; then
            substep "transformers $_ofp_ver sidecar is stale or missing but UV_OFFLINE is set -- left for the next online update"
            eval "_NEED_T5_$_ofp_key=false"
            eval "_DEFER_T5_$_ofp_key=true"
        fi
    done
    unset _ofp _ofp_key _ofp_ver
fi
if [ "${_OFFLINE_FAST_PATH:-false}" = true ]; then
    for _ofp in "530 5.3.0" "550 5.5.0" "510 5.10.2"; do
        _ofp_key=${_ofp%% *}; _ofp_ver=${_ofp#* }
        if eval "[ \"\$_NEED_T5_$_ofp_key\" = true ]"; then
            substep "transformers $_ofp_ver sidecar is stale but UV_OFFLINE is set -- left for the next online update"
            eval "_NEED_T5_$_ofp_key=false"
            eval "_DEFER_T5_$_ofp_key=true"
        fi
    done
    unset _ofp _ofp_key _ofp_ver
fi

if [ "$_NEED_T5_530" = true ]; then
    _install_sidecar "$VENV_T5_530_DIR" "5.3.0" "5.3"
elif [ "$_DEFER_T5_530" = true ]; then
    step "transformers" "5.3.0 sidecar stale -- left for the next online update"
else
    step "transformers" "5.3.0 sidecar current"
    _sidecar_top_up_tiktoken "$VENV_T5_530_DIR" "5.3"
fi
if [ "$_NEED_T5_550" = true ]; then
    _install_sidecar "$VENV_T5_550_DIR" "5.5.0" "5.5"
elif [ "$_DEFER_T5_550" = true ]; then
    step "transformers" "5.5.0 sidecar stale -- left for the next online update"
else
    step "transformers" "5.5.0 sidecar current"
    _sidecar_top_up_tiktoken "$VENV_T5_550_DIR" "5.5"
fi
if [ "$_NEED_T5_510" = true ]; then
    _install_sidecar "$VENV_T5_510_DIR" "5.10.2" "5.10"
elif [ "$_DEFER_T5_510" = true ]; then
    step "transformers" "5.10.2 sidecar stale -- left for the next online update"
else
    step "transformers" "5.10.2 sidecar current"
    _sidecar_top_up_tiktoken "$VENV_T5_510_DIR" "5.10"
fi
fi

# ── GPU detection summary (mirrors setup.ps1 step "gpu" block) ──
# WSL2 ROCDXG: rocminfo sees the GPU over /dev/dxg only with HSA_ENABLE_DXG_DETECTION=1,
# and /opt/rocm/bin may be off PATH.
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
if ! command -v rocminfo >/dev/null 2>&1 && [ -x /opt/rocm/bin/rocminfo ]; then
    PATH="$PATH:/opt/rocm/bin"
fi
_setup_amd_detected=false
_setup_nvidia_usable=false
_setup_nvidia_physical=false
_setup_gfx_all=""
_setup_gfx=""
_setup_hip_map_missing=0
_setup_amd_probe=""
_setup_rocr_uuid_declined=0
_setup_mkt=""
_setup_amd_records=""

# Pair each rocminfo gfx id with its marketing name. Keep in sync with install.sh.
_setup_rocminfo_gpu_records() {
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

# amd-smi lists in discovery order but HIP/ROCR masks index HIP order; map via `amd-smi list -e`
# (HIP_ID). Keep in sync with install.sh.
_setup_amd_smi_hip_order() {
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

# One `gfx|name` per adapter in `GPU: N` order. Keep in sync with install.sh.
_setup_amd_smi_gpu_records() {
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

# Intel XPU. There is no vendor probe here like nvidia-smi / rocminfo -- Linux Intel support is
# an explicit index pin, not autodetection -- so the installed runtime IS the signal. The local
# label is read off disk first so a CPU-only host never pays for an `import torch`.
_setup_torch_is_xpu=false
_setup_xpu_ready=false
for _setup_tv in "$VENV_DIR"/lib/python*/site-packages/torch/version.py; do
    [ -f "$_setup_tv" ] || continue
    grep -q "^__version__ = '[^']*+xpu" "$_setup_tv" 2>/dev/null || continue
    _setup_torch_is_xpu=true
    # Bounded: a stalled Intel driver wedges `import torch`. SIGALRM covers hosts without timeout.
    _setup_xpu_probe='import signal; signal.alarm(60); import torch,sys; sys.exit(0 if torch.xpu.is_available() else 1)'
    if command -v timeout >/dev/null 2>&1; then
        timeout 60 "$VENV_DIR/bin/python" -c "$_setup_xpu_probe" >/dev/null 2>&1 && _setup_xpu_ready=true
    elif "$VENV_DIR/bin/python" -c "$_setup_xpu_probe" >/dev/null 2>&1; then
        _setup_xpu_ready=true
    fi
    break
done

# bitsandbytes has XPU kernels only from 0.50.0; `studio update` runs this file, not install.sh.
if [ "$_setup_torch_is_xpu" = true ]; then
    # run_quiet_no_exit, NOT run_quiet: the latter aborts the update on a best-effort step.
    run_quiet_no_exit "install bitsandbytes (xpu)" fast_install --no-deps "bitsandbytes>=0.50.0" || \
        substep "[WARN] could not install an XPU-capable bitsandbytes; 4-bit QLoRA may be unavailable."
fi
# Kept in sync with install.sh (and the PS nameArchTable).
# gfx1102 before gfx1100 so the spaceless "RX 7700S" lands on gfx1102 (case has no lookahead).
_setup_supported_gfx_from_name() {
    _sup_gfx_in="$1"
    _sup_gfx_out=""
    case "$_sup_gfx_in" in
        *9070*|*9080*|*"R9700"*)                                                                       _sup_gfx_out="gfx1201" ;;  # RDNA 4 (Navi 48: RX 9070 / 9080, Radeon AI PRO R9700)
        *9060*)                                                                                        _sup_gfx_out="gfx1200" ;;  # RDNA 4 (Navi 44)
        *"8065S"*|*"8060S"*|*"8050S"*|*"8040S"*|*"Strix Halo"*|*"Ryzen AI Max"*|*"AI Max"*) _sup_gfx_out="gfx1151" ;;  # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
        *"890M"*|*"880M"*|*"Strix Point"*|*"HX 37"*|*"AI 9 HX"*|*"AI 9 36"*) _sup_gfx_out="gfx1150" ;;  # RDNA 3.5 (Strix Point: Radeon 890M/880M, Ryzen AI 9 HX 370/375)
        *"860M"*|*"840M"*|*"Krackan"*|*"AI 7 35"*|*"AI 5 34"*|*"AI 7 PRO 35"*|*"AI 5 33"*) _sup_gfx_out="gfx1152" ;;  # RDNA 3.5 (Krackan Point: Radeon 860M/840M, Ryzen AI 7 350 / AI 5 340)
        *"RX 7600"*|*"RX 7700S"*|*"RX 7650"*|*"PRO W7600"*|*"PRO W7500"*)                              _sup_gfx_out="gfx1102" ;;  # RDNA 3 (Navi 33)
        *"RX 7800"*|*"RX 7700"*|*"PRO W7700"*|*"PRO V710"*)                                            _sup_gfx_out="gfx1101" ;;  # RDNA 3 (Navi 32)
        *"RX 7900"*|*"PRO W7900"*|*"PRO W7800"*)                                                       _sup_gfx_out="gfx1100" ;;  # RDNA 3 desktop / workstation (Navi 31)
        *"780M"*|*"760M"*|*"740M"*|*"Phoenix"*|*"Hawk Point"*|*"Z1 Extreme"*|*"Z2 Extreme"*)            _sup_gfx_out="gfx1103" ;;  # RDNA 3 iGPU (Phoenix / Hawk Point)
        *"RX 6950"*|*"RX 6900"*|*"RX 6850"*|*"RX 6800"*|*"RX 6750"*|*"RX 6700"*|*"PRO W6800"*|*"PRO W6900"*) _sup_gfx_out="gfx1030" ;;  # RDNA 2 (Navi 21)
        *"RX 6650"*|*"RX 6600"*|*"PRO W6600"*|*"PRO W6650"*)                                            _sup_gfx_out="gfx1032" ;;  # RDNA 2 (Navi 23)
        *"RX 6550"*|*"RX 6500"*|*"RX 6450"*|*"RX 6400"*|*"RX 6300"*|*"PRO W6400"*|*"PRO W6500"*|*"PRO W6300"*)                    _sup_gfx_out="gfx1034" ;;  # RDNA 2 (Navi 24)
    esac
    [ -n "$_sup_gfx_out" ] || return 1
    printf '%s\n' "$_sup_gfx_out"
}

# Held to install.sh by tests/studio/install/test_rocm_arch_table_parity.py, which looks
# helpers up by install.sh's names: keep the _amd_* prefix.
_amd_gfx_is_shadowing_integrated() {
    case "$1" in
        gfx90c|gfx1013|gfx1033|gfx1035|gfx1036|gfx1103|gfx1153) return 0 ;;
    esac
    return 1
}

# gfx906 is never a candidate. No torch-wheel-route filter here, unlike install.sh: do not "restore" it.
_amd_prefer_discrete_gfx() {
    _apdg_devs="$1"
    _apdg_sel="$2"
    if [ -z "$_apdg_sel" ] || ! _amd_gfx_is_shadowing_integrated "$_apdg_sel"; then
        printf '%s' "$_apdg_sel"
        return 0
    fi
    # A mask the user SET is their own device choice: `+x`, not `-n`, so empty still counts.
    if [ -n "${HIP_VISIBLE_DEVICES+x}" ] || [ -n "${ROCR_VISIBLE_DEVICES+x}" ] || \
       [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
        printf '%s' "$_apdg_sel"
        return 0
    fi
    # Here-doc, not a pipe: a piped `while` is a subshell, and an early break SIGPIPEs under pipefail.
    _apdg_pick=""
    while IFS= read -r _apdg_g; do
        if [ -n "$_apdg_g" ] && [ "$_apdg_g" != gfx906 ] && \
           ! _amd_gfx_is_shadowing_integrated "$_apdg_g"; then
            _apdg_pick="$_apdg_g"
            break
        fi
    done <<EOF
$_apdg_devs
EOF
    [ -n "$_apdg_pick" ] || _apdg_pick="$_apdg_sel"
    printf '%s' "$_apdg_pick"
}

# Skip AMD probes on a usable-NVIDIA host (mirrors _has_rocm_gpu in install_python_stack.py).
if _setup_has_physical_nvidia_gpu; then
    _setup_nvidia_physical=true
fi
if _setup_has_usable_nvidia_gpu; then
    _setup_nvidia_usable=true
fi
if [ "$_setup_nvidia_usable" != true ]; then
    if command -v rocminfo >/dev/null 2>&1; then
        _setup_amd_records=$(_setup_run_smi rocminfo 2>/dev/null | _setup_rocminfo_gpu_records || true)
        _setup_gfx_all=$(printf '%s\n' "$_setup_amd_records" | awk -F'|' '$1 != "" { print $1 }')
    fi
    if [ -n "$_setup_gfx_all" ]; then
        _setup_amd_detected=true
        _setup_amd_probe=rocminfo
    elif command -v amd-smi >/dev/null 2>&1 && \
         _setup_run_smi amd-smi list 2>/dev/null | awk '/^GPU[[:space:]]*[:\[][[:space:]]*[0-9]/{ found=1 } END{ exit !found }'; then
        _setup_amd_detected=true
        _setup_amd_records=$(_setup_run_smi amd-smi static --asic 2>/dev/null | _setup_amd_smi_gpu_records || true)
        if [ -n "$_setup_amd_records" ]; then
            _setup_amd_smi_out=$(_setup_run_smi amd-smi list -e 2>/dev/null \
                | _setup_amd_smi_hip_order "$_setup_amd_records" || true)
            # Expansion, not `| head -n 1`: head exiting early SIGPIPEs printf under pipefail.
            _setup_amd_space=${_setup_amd_smi_out%%$'\n'*}
            _setup_amd_records=$(printf '%s\n' "$_setup_amd_smi_out" | tail -n +2)
            # No HIP map and unlike adapters: any ordinal is a guess, so decline rather than forward --rocm-gfx.
            if [ "$_setup_amd_space" != hip ] && \
               [ "$(printf '%s\n' "$_setup_amd_records" | awk -F'|' \
                    'NF { k = ($1 != "" ? $1 : "name:" $2); if (!(k in seen)) { seen[k]; n++ } }
                     END { print n + 0 }')" -gt 1 ]; then
                _setup_amd_records=""
                _setup_gfx_all=""
                _setup_hip_map_missing=1
            fi
        fi
        _setup_gfx_all=$(_setup_run_smi amd-smi list 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        [ -z "$_setup_gfx_all" ] && \
            _setup_gfx_all=$(printf '%s\n' "$_setup_amd_records" | awk -F'|' '$1 != "" { print $1 }')
    elif [ -e /dev/kfd ] && \
         awk '/vendor_id/ && $2 == 4098 { found = 1 } END { exit !found }' \
             /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null; then
        # KFD sysfs fallback, AMD vendor_id 4098 only (mirrors install.sh _has_amd_rocm_gpu).
        _setup_amd_detected=true
        _setup_amd_records=""
    fi
fi

if [ "$_setup_nvidia_usable" = true ]; then
    _setup_nv_banner_fields
    if [ -n "$_setup_nv_name" ] && [ -n "$_setup_nv_sm" ]; then
        step "gpu" "$_setup_nv_name ($_setup_nv_sm)"
    elif [ -n "$_setup_nv_name" ]; then
        step "gpu" "$_setup_nv_name"
    else
        step "gpu" "NVIDIA GPU detected"
    fi
    # `if`, not `&&`: the AND-list leaves a non-zero status behind.
    if [ -n "$_setup_nv_driver" ]; then substep "Driver: $_setup_nv_driver"; fi
elif [ "$_setup_amd_detected" = true ]; then
    # As install.sh: ROCr picks survivors, then the first set HIP-layer mask indexes them.
    if [ "$_setup_amd_probe" != rocminfo ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ "$ROCR_VISIBLE_DEVICES" != "-1" ]; then
        _setup_rocr_keep() {
            _setup_kept=$(printf '%s\n' "$1" | awk -v m="$ROCR_VISIBLE_DEVICES" '
                NF { v[n++] = $0 }
                END { k = split(m, t, ","); for (i = 1; i <= k; i++) { gsub(/[[:space:]]/, "", t[i]); if (t[i] !~ /^[0-9]+$/) continue; x = t[i] + 0; if (x >= n || (x in s)) break; s[x] = 1; print v[x] } }')
            if [ -n "$_setup_kept" ]; then printf '%s\n' "$_setup_kept"; else printf '%s\n' "$1"; fi
        }
        if [ -n "$(printf '%s' "$ROCR_VISIBLE_DEVICES" | tr -d '0-9, \t')" ] && \
           [ "$(printf '%s\n' "${_setup_amd_records:-$_setup_gfx_all}" | awk -F'|' \
                'NF { k = ($1 != "" ? $1 : "name:" $2); if (!(k in seen)) { seen[k]; n++ } } END { print n + 0 }')" -gt 1 ]; then
            _setup_amd_records=""
            _setup_gfx_all=""
            _setup_rocr_uuid_declined=1
        fi
        [ -n "$_setup_amd_records" ] && _setup_amd_records=$(_setup_rocr_keep "$_setup_amd_records")
        [ -n "$_setup_gfx_all" ] && _setup_gfx_all=$(_setup_rocr_keep "$_setup_gfx_all")
    fi
    if [ -n "${HIP_VISIBLE_DEVICES+x}" ]; then
        _setup_vis="$HIP_VISIBLE_DEVICES"
    else
        _setup_vis="${CUDA_VISIBLE_DEVICES:-}"
    fi
    _setup_vis_idx=0
    if [ -n "$_setup_vis" ] && [ "$_setup_vis" != "-1" ]; then
        _setup_first="${_setup_vis%%,*}"
        case "$_setup_first" in ''|*[!0-9]*) ;; *) _setup_vis_idx=$_setup_first ;; esac
    fi
    if [ -n "$_setup_amd_records" ]; then
        _setup_amd_record=$(printf '%s\n' "$_setup_amd_records" | awk -v idx="$_setup_vis_idx" \
            'NF { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx+0] }')
        _setup_gfx=${_setup_amd_record%%|*}
        _setup_mkt=${_setup_amd_record#*|}
    fi
    if [ -z "$_setup_gfx" ]; then
        _setup_gfx=$(printf '%s\n' "$_setup_gfx_all" | awk -v idx="$_setup_vis_idx" \
            'NF && !seen[$0]++ { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx+0] }')
    fi
    # An iGPU enumerated first would take the ROCm bundle (#7776). Skipped under UNSLOTH_ROCM_GFX_ARCH.
    _setup_gfx_pref=""
    if [ -z "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        _setup_gfx_cands="$_setup_gfx_all"
        if [ -n "$_setup_amd_records" ]; then
            _setup_gfx_cands=$(printf '%s\n' "$_setup_amd_records" | awk -F'|' '$1 != "" { print $1 }')
        fi
        _setup_gfx_pref=$(_amd_prefer_discrete_gfx "$_setup_gfx_cands" "$_setup_gfx")
    fi
    if [ -n "$_setup_gfx_pref" ] && [ "$_setup_gfx_pref" != "$_setup_gfx" ]; then
        substep "Integrated $_setup_gfx enumerated first; installing for discrete $_setup_gfx_pref"
        substep "Set UNSLOTH_ROCM_GFX_ARCH=$_setup_gfx to target the integrated GPU instead."
        _setup_gfx="$_setup_gfx_pref"
        _setup_mkt=$(printf '%s\n' "$_setup_amd_records" | awk -F'|' -v gfx="$_setup_gfx" \
            '$1 == gfx { print $2; exit }')
    fi
    # UNSLOTH_ROCM_GFX_ARCH env override (mirrors setup.ps1)
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        _setup_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
        substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $_setup_gfx"
    # Name-based arch inference when tools don't report gfx (mirrors setup.ps1 nameArchTable)
    elif [ -z "$_setup_gfx" ] && [ -n "$_setup_mkt" ]; then
        _setup_gfx=$(_setup_supported_gfx_from_name "$_setup_mkt") || _setup_gfx=""
        if [ -n "$_setup_gfx" ]; then
            substep "gfx arch inferred from GPU name: $_setup_gfx"
            substep "Tip: set UNSLOTH_ROCM_GFX_ARCH=$_setup_gfx to skip inference next time"
        fi
    fi
    if [ -z "$_setup_gfx" ] && [ "$_setup_hip_map_missing" = 1 ]; then
        substep "Unlike AMD adapters and no HIP id map (amd-smi list -e needs ROCm 6.4+):"
        substep "cannot tell which one this session selects. Set UNSLOTH_ROCM_GFX_ARCH to pick."
    fi
    if [ -z "$_setup_gfx" ] && [ "$_setup_rocr_uuid_declined" = 1 ]; then
        substep "ROCR_VISIBLE_DEVICES names a GPU by UUID, which amd-smi cannot place, and the"
        substep "adapters differ. Set UNSLOTH_ROCM_GFX_ARCH to pick."
    fi
    _setup_rocm_ver=""
    if command -v hipconfig >/dev/null 2>&1; then
        _setup_rocm_ver=$(hipconfig --version 2>/dev/null | awk 'NR==1 && /^[0-9]/{print; exit}' || true)
    fi
    if [ -z "$_setup_rocm_ver" ] && command -v amd-smi >/dev/null 2>&1; then
        _setup_rocm_ver=$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
            'NF>1{gsub(/[[:space:]]/,"", $2); print $2; exit}' || true)
    fi
    # Report-only table for arches Unsloth's ROCm wheels do not cover; never selects a wheel or prebuilt.
    # Order matters: RDNA 1 arms precede Polaris, or *"RX 570"* would swallow "RX 5700 XT".
    _setup_unsupported_gfx_from_name() {
        case "$1" in
            *"Radeon Pro V520"*|*"Radeon Pro 5600M"*) echo gfx1011 ;;  # RDNA 1
            *"RX 5700"*|*"RX 5600"*|*"Radeon Pro 5600 XT"*|*"Radeon Pro 5700"*|*"Radeon Pro W5700"*) echo gfx1010 ;;  # RDNA 1 (Navi 10)
            *"RX 5500"*|*"RX 5300"*|*"Radeon Pro W5500"*|*"Radeon Pro W5300"*) echo gfx1012 ;;  # RDNA 1 (Navi 14)
            *"RX 470"|*"RX 470"[!0]*|*"RX 480"|*"RX 480"[!0]*|*"RX 570"|*"RX 570"[!0]*|*"RX 580"|*"RX 580"[!0]*|*"RX 590"|*"RX 590"[!0]*|*"Radeon Pro WX 7100"*|*"Radeon Pro WX 5100"*) echo gfx803 ;;  # Polaris 10/20/30
            *) return 1 ;;
        esac
    }
    # Not written back into _setup_mkt: that would feed --rocm-gfx on the KFD path.
    _setup_unsupported_gfx_any() {
        # Peer guard first: amd-smi reports only the first device's name.
        _setup_unsup_pci=""
        if command -v lspci >/dev/null 2>&1; then
            _setup_unsup_pci=$(lspci -nn 2>/dev/null | grep -E 'VGA compatible controller|3D controller|Display controller' | grep -E 'AMD|ATI' || true)
            while IFS= read -r _setup_unsup_ln; do
                [ -n "$_setup_unsup_ln" ] || continue
                if _setup_supported_gfx_from_name "$_setup_unsup_ln" >/dev/null 2>&1; then
                    return 1
                fi
            done <<EOF
$_setup_unsup_pci
EOF
        fi
        if [ -n "$1" ] && _setup_unsup_named=$(_setup_unsupported_gfx_from_name "$1"); then
            echo "$_setup_unsup_named"
            return 0
        fi
        [ -n "$_setup_unsup_pci" ] || return 1
        while IFS= read -r _setup_unsup_ln; do
            [ -n "$_setup_unsup_ln" ] || continue
            if _setup_unsup_hit=$(_setup_unsupported_gfx_from_name "$_setup_unsup_ln"); then
                echo "$_setup_unsup_hit"
                return 0
            fi
        done <<EOF
$_setup_unsup_pci
EOF
        return 1
    }
    if [ -n "$_setup_gfx" ] && [ -n "$_setup_mkt" ]; then
        step "gpu" "$_setup_mkt ($_setup_gfx)"
    elif [ -n "$_setup_gfx" ]; then
        step "gpu" "AMD ROCm ($_setup_gfx)"
    elif _setup_unsup_gfx=$(_setup_unsupported_gfx_any "$_setup_mkt"); then
        step "gpu" "AMD GPU detected ($_setup_unsup_gfx) -- no ROCm PyTorch wheels Unsloth installs"
        # unsloth raises at import without CUDA/XPU (unsloth/device_type.py). An explicit index pin
        # changes that. Trimmed like get_torch_index_url; distinct name from the XPU block's _setup_pin.
        _setup_unsup_pin="${UNSLOTH_TORCH_INDEX_URL:-}${UNSLOTH_TORCH_INDEX_FAMILY:-}"
        _setup_unsup_pin=$(printf '%s' "$_setup_unsup_pin" | tr -d '[:space:]')
        if [ -n "$_setup_unsup_pin" ]; then
            substep "The torch index you pinned is used as given, so torch is whatever it publishes."
        else
            substep "torch stays CPU-only: Unsloth training and GPU inference are unavailable."
            substep "No HIP SDK install and no UNSLOTH_ROCM_GFX_ARCH value gives this GPU one."
        fi
        substep "GGUF chat can still use this GPU through Vulkan: export UNSLOTH_LLAMA_CPP_BACKEND=vulkan,"
        substep "then re-run the installer. It picks the llama.cpp bundle at install time, so setting"
        substep "it afterwards has no effect until you install or update again."
    elif [ -n "$_setup_mkt" ]; then
        # Deliberately below the unsupported arm, so the training warning is not dropped.
        step "gpu" "$_setup_mkt"
    else
        step "gpu" "AMD ROCm"
    fi
    # /opt/rocm is a fallback, not a detection: only claim a path that exists, matching install.sh.
    _setup_rocm_root="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
    if [ -d "$_setup_rocm_root" ]; then
        substep "ROCm: $_setup_rocm_root"
    else
        substep "ROCm: runtime detected (no SDK tree at $_setup_rocm_root)"
    fi
    [ -n "$_setup_rocm_ver" ] && substep "hipconfig: $_setup_rocm_ver"
elif [ "$_setup_xpu_ready" = true ]; then
    step "gpu" "Intel GPU detected (XPU runtime)"
    substep "PyTorch XPU (SYCL) provides training and GPU inference on this GPU."
elif [ "$_setup_torch_is_xpu" = true ]; then
    step "gpu" "Intel GPU (XPU runtime unavailable)" "$C_WARN"
    substep "PyTorch has the XPU build but cannot initialise it -- update the Intel GPU compute driver."
    substep "Until then training and GPU inference are unavailable; chat and GGUF still work."
elif [ "$(uname -s 2>/dev/null)" = "Darwin" ] && [ "$(uname -m 2>/dev/null)" = "arm64" ]; then
    step "gpu" "Apple Silicon (Metal, unified memory)"
else
    step "gpu" "none (chat-only / GGUF)" "$C_WARN"
    substep "Training and GPU inference require an NVIDIA or AMD ROCm GPU."
fi

# Legacy default keeps ~/.unsloth/llama.cpp so pre-PR builds are still discovered.
if [ -n "$STAGE_ROOT" ]; then
    UNSLOTH_HOME="$RUNTIME_ROOT"
elif [ -n "$_MASTER_ROOT" ]; then
    UNSLOTH_HOME="$_MASTER_ROOT"
elif [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
    UNSLOTH_HOME="$STUDIO_HOME"
else
    UNSLOTH_HOME="$HOME/.unsloth"
fi
mkdir -p "$UNSLOTH_HOME"
# Record the master root inside the Studio tree, for the uninstaller, only when readers honour it.
_master_root_note_is_honoured() {
    # Re-canonicalised: the root may have been created after the earlier resolution.
    _mrn_root=$(CDPATH= cd -P -- "$1" 2>/dev/null && pwd -P) || _mrn_root=""
    [ -n "$_mrn_root" ] || return 1
    # The legacy default root is refused by both uninstallers; writing it would enable portable mode.
    _mrn_legacy=$(CDPATH= cd -P -- "${HOME:-}/.unsloth" 2>/dev/null && pwd -P) || _mrn_legacy="${HOME:-}/.unsloth"
    [ "$_mrn_root" != "$_mrn_legacy" ] || return 1
    # Readers decline any note in ~/.unsloth/studio, so do not write one there.
    _mrn_studio=$(CDPATH= cd -P -- "$STUDIO_HOME" 2>/dev/null && pwd -P) || _mrn_studio="$STUDIO_HOME"
    _mrn_legacy_studio=$(CDPATH= cd -P -- "${HOME:-}/.unsloth/studio" 2>/dev/null && pwd -P) \
        || _mrn_legacy_studio="${HOME:-}/.unsloth/studio"
    if [ "$_mrn_studio" = "$_mrn_legacy_studio" ]; then
        echo "NOTE: the managed runtimes were installed under $_mrn_root, but Studio itself is the" >&2
        echo "      default install at $_mrn_studio, where no reader honours a recorded root. That" >&2
        echo "      root cannot be recorded, so a later launch or uninstall will not find them." >&2
        echo "      Set UNSLOTH_STUDIO_HOME=$_mrn_root/studio, or re-export UNSLOTH_HOME whenever" >&2
        echo "      you run Unsloth." >&2
        return 1
    fi
    # All readers require the Studio dir inside the named root; otherwise warn instead of writing.
    case "$STUDIO_HOME" in
        "$_mrn_root"|"$_mrn_root"/*) return 0 ;;
    esac
    echo "NOTE: the managed runtimes were installed under $_mrn_root, but Studio itself lives at" >&2
    echo "      $STUDIO_HOME, which is outside it. That root cannot be recorded, so a later launch" >&2
    echo "      or uninstall will not find them. Set UNSLOTH_STUDIO_HOME=$_mrn_root/studio, or" >&2
    echo "      re-export UNSLOTH_HOME whenever you run Unsloth." >&2
    return 1
}
if [ -n "$_MASTER_ROOT" ] && [ -z "$STAGE_ROOT" ] && _master_root_note_is_honoured "$UNSLOTH_HOME"; then
    if mkdir -p "$STUDIO_HOME/share" 2>/dev/null; then
        # Staged then renamed via mktemp: this note licenses deletions.
        if ! _mrn_tmp=$(mktemp "$STUDIO_HOME/share/.unsloth-master-root.XXXXXX" 2>/dev/null); then
            _mrn_tmp=""
        fi
        if [ -n "$_mrn_tmp" ] && printf '%s\n' "$UNSLOTH_HOME" > "$_mrn_tmp" 2>/dev/null; then
            mv -f "$_mrn_tmp" "$STUDIO_HOME/share/.unsloth-master-root" 2>/dev/null \
                || rm -f "$_mrn_tmp" 2>/dev/null || true
        elif [ -n "$_mrn_tmp" ]; then
            rm -f "$_mrn_tmp" 2>/dev/null || true
        fi
        unset _mrn_tmp
    fi
fi
# Clear a stale note naming the legacy default root; it would keep portable mode on forever.
if [ -f "$STUDIO_HOME/share/.unsloth-master-root" ]; then
    _mrn_old=$(head -n 1 "$STUDIO_HOME/share/.unsloth-master-root" 2>/dev/null \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//') || _mrn_old=""
    if [ -n "$_mrn_old" ]; then
        case "$_mrn_old" in
            "~") _mrn_old="${HOME:-}" ;;
            "~/"*) _mrn_old="${HOME:-}/${_mrn_old#'~/'}" ;;
        esac
        _mrn_old_canon=$(CDPATH= cd -P -- "$_mrn_old" 2>/dev/null && pwd -P) || _mrn_old_canon=""
        [ -n "$_mrn_old_canon" ] && _mrn_old="$_mrn_old_canon"
        _mrn_legacy=$(CDPATH= cd -P -- "${HOME:-}/.unsloth" 2>/dev/null && pwd -P) || _mrn_legacy="${HOME:-}/.unsloth"
        if [ "$_mrn_old" = "$_mrn_legacy" ]; then
            rm -f "$STUDIO_HOME/share/.unsloth-master-root" 2>/dev/null || true
        fi
    fi
    unset _mrn_old _mrn_old_canon
fi
LLAMA_CPP_DIR="$UNSLOTH_HOME/llama.cpp"
LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
_NEED_LLAMA_SOURCE_BUILD=false
_LLAMA_CPP_DEGRADED=false
_LLAMA_CPP_NO_SPACE=false
_LLAMA_KEEP_PREBUILT_ACTIVE=false
_LLAMA_KEPT_GPU_PREBUILT=""
_LLAMA_UPDATE_FAIL_REASON=""
_LLAMA_CPU_ONLY_ON_GPU_HOST=false
_LLAMA_FORCE_COMPILE="${UNSLOTH_LLAMA_FORCE_COMPILE:-0}"
_REQUESTED_LLAMA_TAG="${UNSLOTH_LLAMA_TAG:-${_DEFAULT_LLAMA_TAG}}"
_HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
_HOST_MACHINE="$(uname -m 2>/dev/null || true)"
_source_backend_choice="$(printf '%s' "${UNSLOTH_LLAMA_CPP_BACKEND:-}" | awk '{$1=$1; print tolower($0)}')"
_source_legacy_force_vulkan="$(printf '%s' "${UNSLOTH_FORCE_VULKAN:-}" | awk '{$1=$1; print tolower($0)}')"
_explicit_llama_source_backend=""
if [ "$_HOST_SYSTEM" != "Darwin" ]; then
    case "$_source_backend_choice" in
        hip) _explicit_llama_source_backend="rocm" ;;
        cpu|cuda|rocm|vulkan) _explicit_llama_source_backend="$_source_backend_choice" ;;
        auto) ;;
        *)
            case "$_source_legacy_force_vulkan" in
                1|true|yes|on) _explicit_llama_source_backend="vulkan" ;;
            esac
            ;;
    esac
fi

# Every supported host pulls its llama.cpp prebuilt from the unslothai fork.
_HELPER_RELEASE_REPO="unslothai/llama.cpp"
# UNSLOTH_ROCM_GFX_ARCH may be set where no probe fired; honour it for --rocm-gfx.
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

if [ -z "$_LLAMA_PR" ] && [ -n "$_LLAMA_PR_FORCE" ] && \
   [[ "$_LLAMA_PR_FORCE" =~ ^[0-9]+$ ]] && [ "$_LLAMA_PR_FORCE" -gt 0 ]; then
    _LLAMA_PR="$_LLAMA_PR_FORCE"
    step "llama.cpp" "baked-in PR_FORCE=$_LLAMA_PR_FORCE" "$C_WARN"
fi

if [ -n "$_LLAMA_PR" ]; then
    if ! [[ "$_LLAMA_PR" =~ ^[0-9]+$ ]] || [ "$_LLAMA_PR" -le 0 ]; then
        step "llama.cpp" "UNSLOTH_LLAMA_PR=$_LLAMA_PR is not a valid PR number" "$C_ERR"
        setup_fail 1 "UNSLOTH_LLAMA_PR=$_LLAMA_PR is not a valid PR number"
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

# check_llama_cpp() looks for a root llama-quantize shim. Best-effort: tree may be read-only.
_link_local_llama_quantize_shim() {
    if [ -x "$1/build/bin/llama-quantize" ] && [ ! -e "$1/llama-quantize" ]; then
        ln -sf build/bin/llama-quantize "$1/llama-quantize" 2>/dev/null || \
            substep "could not create llama-quantize shim in linked dir (read-only?); GGUF export may be unavailable"
    fi
}

# Accept any layout LlamaCppBackend._layout_candidates() resolves.
_has_local_llama_server() {
    [ -x "$1/llama-server" ] || [ -x "$1/build/bin/llama-server" ]
}

# $4 = refs: installer alias and commit-prefix matching; a release tag compares exactly.
_installed_prebuilt_ref_matches() {
    [ -f "$1/UNSLOTH_PREBUILT_INFO.json" ] || return 1
    python - "$SCRIPT_DIR/install_llama_prebuilt.py" "$1/UNSLOTH_PREBUILT_INFO.json" "$2" "$3" "${4:-exact}" <<'PY' 2>/dev/null
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("installer", sys.argv[1])
installer = importlib.util.module_from_spec(spec)
sys.modules["installer"] = installer  # the module's dataclasses resolve through sys.modules
spec.loader.exec_module(installer)
try:
    marker = json.load(open(sys.argv[2], encoding="utf-8"))
except Exception:
    marker = {}
values = [marker.get(f) for f in sys.argv[3].split(",")] if isinstance(marker, dict) else []
values = [v.strip() for v in values if isinstance(v, str) and v.strip()]
same = any(
    v == sys.argv[4] or (sys.argv[5] == "refs" and installer.refs_match(v, sys.argv[4])) for v in values
)
sys.exit(0 if same else 1)
PY
}

_installed_prebuilt_backend() {
    [ -f "$1/UNSLOTH_PREBUILT_INFO.json" ] || return 0
    python - "$1/UNSLOTH_PREBUILT_INFO.json" <<'PY' 2>/dev/null || true
import json
import sys

KNOWN = ("cuda", "rocm", "vulkan", "cpu")
try:
    marker = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    marker = {}
if not isinstance(marker, dict):
    marker = {}
def _field(key):
    value = marker.get(key)
    value = value.strip().lower() if isinstance(value, str) else ""
    return "rocm" if value == "hip" else value

# A recorded backend is final, known to this script or not; llama_backend was the request.
answer = _field("backend")
if not answer and _field("llama_backend") in KNOWN:
    answer = _field("llama_backend")
if not answer:
    asset = marker.get("asset")
    asset = asset.lower() if isinstance(asset, str) else ""
    for value in KNOWN:
        if f"-{value}" in asset or (value == "rocm" and "-hip" in asset):
            answer = value
            break
print(answer)
PY
}

_llama_update_fail_reason() {
    if grep -qiE "429|rate limit" "$1" 2>/dev/null; then
        echo "GitHub rate limit"
    elif grep -qiE "timed out|timeout|connection|resol|network|unreachable|50[234]" "$1" 2>/dev/null; then
        echo "network error"
    else
        echo "download failed"
    fi
}

_installed_prebuilt_runs() {
    python "$SCRIPT_DIR/install_llama_prebuilt.py" --check-installed "$1" >/dev/null 2>&1
}

_setup_has_intel_gpu() {
    grep -qs -i "^0x8086" /sys/class/drm/card*/device/vendor 2>/dev/null
}

# A CPU-only source build must not replace a working GPU prebuilt while the GPU is there (#9255).
_gpu_prebuilt_to_keep_over_cpu_build() {
    local install_dir=$1 backend
    [ "$_LLAMA_FORCE_COMPILE" != "1" ] || return 1
    [ -z "$_LLAMA_PR" ] || return 1
    if [ -n "${UNSLOTH_LLAMA_RELEASE_TAG:-}" ]; then
        _installed_prebuilt_ref_matches "$install_dir" release_tag "$UNSLOTH_LLAMA_RELEASE_TAG" || return 1
    fi
    case "${UNSLOTH_LLAMA_TAG:-}" in
        ""|latest|master) ;;
        *) _installed_prebuilt_ref_matches "$install_dir" \
               tag,requested_source_ref,resolved_source_ref,source_commit "$UNSLOTH_LLAMA_TAG" refs || return 1 ;;
    esac
    _has_local_llama_server "$install_dir" || return 1
    backend="$(_installed_prebuilt_backend "$install_dir")"
    case "$backend" in
        cuda) [ "$_setup_nvidia_physical" = true ] || return 1 ;;
        rocm) [ "$_setup_amd_detected" = true ] || return 1 ;;
        vulkan)
            [ "$_setup_amd_detected" = true ] || [ "$_setup_nvidia_physical" = true ] \
                || _setup_has_intel_gpu || return 1 ;;
        *) return 1 ;;
    esac
    _installed_prebuilt_runs "$install_dir" || return 1
    printf '%s' "$backend"
}

# UNSLOTH_LLAMA_KEEP_PREBUILT=1 (Docker build sees no GPU and would install CPU over CUDA).
_keep_installed_gpu_prebuilt() {
    local install_dir=$1 requested_tag=$2 repo=$3 release_pin=${4:-}
    case "$(printf '%s' "${UNSLOTH_LLAMA_KEEP_PREBUILT:-}" | awk '{$1=$1; print tolower($0)}')" in
        1|true|yes|on) ;;
        *) return 1 ;;
    esac
    # An explicitly requested backend must still be installed, or fail loudly, never kept over.
    [ -z "${_explicit_llama_source_backend:-}" ] || return 1
    _has_local_llama_server "$install_dir" || return 1
    [ -f "$install_dir/UNSLOTH_PREBUILT_INFO.json" ] || return 1
    python - "$install_dir/UNSLOTH_PREBUILT_INFO.json" "$requested_tag" "$repo" "$release_pin" "$SCRIPT_DIR" <<'PY' 2>/dev/null
import json
import re
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if not isinstance(payload, dict):
    raise SystemExit(1)

requested = sys.argv[2].strip()
repo = sys.argv[3].strip()
release_pin = sys.argv[4].strip() if len(sys.argv) > 4 else ""
GPU_TOKENS = ("cuda", "rocm", "hip", "vulkan", "metal")
# Fork bundles record only "platform"; install_llama_prebuilt.py also records "backend". Accept either.
backend = str(payload.get("backend") or "").strip().lower()
platform_kind = str(payload.get("platform") or payload.get("install_kind") or "").strip().lower()
if payload.get("force_cpu") is True:
    raise SystemExit(1)
if backend not in GPU_TOKENS and not any(token in platform_kind for token in GPU_TOKENS):
    raise SystemExit(1)
if repo and str(payload.get("published_repo") or "").strip() != repo:
    raise SystemExit(1)


def base_build(tag: str) -> str:
    """b10840 out of b10840-mix-d5c17a0, so the marker's normalized "tag" still matches."""
    match = re.match(r"b(\d+)", tag.strip())
    return f"b{match.group(1)}" if match else tag.strip()


recorded = {str(payload.get(key) or "").strip() for key in ("release_tag", "tag", "upstream_tag")}
recorded.discard("")
if not recorded:
    raise SystemExit(1)
# UNSLOTH_LLAMA_RELEASE_TAG names one published release, so only that exact release_tag can be kept.
if release_pin and str(payload.get("release_tag") or "").strip() != release_pin:
    raise SystemExit(1)
if requested and requested.lower() != "latest":
    if re.fullmatch(r"b\d+", requested):
        # A bare base build pin is satisfied by any mix release cut from that build.
        if requested not in {base_build(t) for t in recorded}:
            raise SystemExit(1)
    elif requested not in recorded:
        # b10840-mix-new and b10840-mix-old share a base build but are different bundles.
        raise SystemExit(1)
# Keep the Docker shortcut consistent with desktop preflight without probing a GPU
# or executing the CUDA binaries on the GPU-less image build host.
sys.path.insert(0, sys.argv[5])
from install_llama_prebuilt import installed_runtime_health

health = installed_runtime_health(Path(sys.argv[1]).parent)
raise SystemExit(1 if health is not None and not health[0] else 0)
PY
}

_LOCAL_LLAMA_CPP_LINKED=false
if [ -n "${UNSLOTH_LOCAL_LLAMA_CPP_DIR:-}" ]; then
    if [ ! -d "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" ]; then
        _report_denied_ancestor "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" "UNSLOTH_LOCAL_LLAMA_CPP_DIR"
        step "llama.cpp" "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $UNSLOTH_LOCAL_LLAMA_CPP_DIR" "$C_ERR"
        setup_fail 1 "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $UNSLOTH_LOCAL_LLAMA_CPP_DIR"
    fi
    # In an if condition so a denied dir reports instead of tripping errexit.
    if ! _RESOLVED_LOCAL="$(CDPATH= cd -P -- "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" 2>/dev/null && pwd -P)"; then
        # owner-unverified: this is the user's own tree, never advise deleting it.
        _path_access_denied "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" "UNSLOTH_LOCAL_LLAMA_CPP_DIR" owner-unverified
    fi
    # Canonicalise before comparing, or a symlinked $HOME makes the rm -rf below wipe the tree.
    _CANON_LLAMA_CPP_DIR="$LLAMA_CPP_DIR"
    _LLAMA_CPP_PARENT="$(dirname "$LLAMA_CPP_DIR")"
    if [ -d "$_LLAMA_CPP_PARENT" ]; then
        if _canon_parent="$(CDPATH= cd -P -- "$_LLAMA_CPP_PARENT" 2>/dev/null && pwd -P)"; then
            _CANON_LLAMA_CPP_DIR="$_canon_parent/$(basename "$LLAMA_CPP_DIR")"
        else
            _path_access_denied "$_LLAMA_CPP_PARENT" "Unsloth install directory" owner-unverified
        fi
    fi
    if [ "$_RESOLVED_LOCAL" = "$_CANON_LLAMA_CPP_DIR" ]; then
        # Flag points at the install itself: reuse a build there, never delete-then-link onto itself.
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
        # Reuse skips both prebuilt and source build, so require a runnable llama-server.
        if ! _has_local_llama_server "$_RESOLVED_LOCAL"; then
            step "llama.cpp" "no llama-server under $_RESOLVED_LOCAL (looked for ./llama-server and ./build/bin/llama-server) -- build llama.cpp there first, or drop --with-llama-cpp-dir" "$C_ERR"
            setup_fail 1 "No llama-server was found under $_RESOLVED_LOCAL"
        fi
        # Drop a stale link from a previous run before the ownership check.
        [ -L "$LLAMA_CPP_DIR" ] && rm -f "$LLAMA_CPP_DIR"
        if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ]; then
            _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install" "$_RUNTIME_ROOT_IS_CUSTOM"
        fi
        rm -rf "$LLAMA_CPP_DIR" || true
        if [ -e "$LLAMA_CPP_DIR" ]; then
            # Unreadable, not just unsearchable: mode 111 defeats the rm above.
            if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
                _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
            fi
            step "llama.cpp" "the existing install could not be replaced with a link" "$C_ERR"
            setup_fail 3 "$LLAMA_CPP_DIR could not be replaced with a link to $_RESOLVED_LOCAL."
        fi
        ln -sfn "$_RESOLVED_LOCAL" "$LLAMA_CPP_DIR"
        _link_local_llama_quantize_shim "$LLAMA_CPP_DIR"
        step "llama.cpp" "linked local directory: $_RESOLVED_LOCAL"
        _LOCAL_LLAMA_CPP_LINKED=true
        _NEED_LLAMA_SOURCE_BUILD=false
        _SKIP_PREBUILT_INSTALL=true
    fi
fi

# Check here: the source-build swap only reaches its guards after the whole build.
if [ "$_LOCAL_LLAMA_CPP_LINKED" != true ]; then
    if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install" "$_RUNTIME_ROOT_IS_CUSTOM"
    fi
    if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
        _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
    fi
fi

if [ "$_LOCAL_LLAMA_CPP_LINKED" = true ]; then
    : # local directory linked above; skip prebuilt install
elif [ -n "$_explicit_llama_source_backend" ] && [ "$_NEED_LLAMA_SOURCE_BUILD" = true ]; then
    step "llama.cpp" "$_explicit_llama_source_backend was explicitly requested, but this installation requires a source build" "$C_ERR"
    substep "Explicit backend selection requires a matching prebuilt bundle; allow prebuilts or unset UNSLOTH_LLAMA_CPP_BACKEND"
    setup_fail 1 "$_explicit_llama_source_backend was explicitly requested, but this installation requires a source build. Explicit backend selection requires a matching prebuilt bundle."
elif [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
    step "llama.cpp" "UNSLOTH_LLAMA_FORCE_COMPILE=1 -- skipping prebuilt" "$C_WARN"
    _NEED_LLAMA_SOURCE_BUILD=true
elif [ "${_SKIP_PREBUILT_INSTALL:-false}" = true ]; then
    substep "prebuilt install skipped -- falling back to source build"
elif _keep_installed_gpu_prebuilt "$LLAMA_CPP_DIR" "$_REQUESTED_LLAMA_TAG" "$_HELPER_RELEASE_REPO" "${UNSLOTH_LLAMA_RELEASE_TAG:-}"; then
    step "llama.cpp" "keeping the installed GPU prebuilt (UNSLOTH_LLAMA_KEEP_PREBUILT=1)"
    print_installed_llama_prebuilt_release "$LLAMA_CPP_DIR"
    _LLAMA_KEEP_PREBUILT_ACTIVE=true
else
    substep "installing prebuilt llama.cpp..."
    if [ -d "$LLAMA_CPP_DIR" ]; then
        substep "existing install detected -- validating update"
    fi
    # install_llama_prebuilt.py uses os.replace(), which would displace an unrelated tree.
    if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install" "$_RUNTIME_ROOT_IS_CUSTOM"
    fi
    if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
        _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
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
    # Forward the gfx arch so the per-gfx ROCm prebuilt is picked (implies --has-rocm).
    if [ -n "${_setup_gfx:-}" ]; then
        _PREBUILT_CMD+=(--rocm-gfx "$_setup_gfx")
    elif [ "$_setup_amd_detected" = true ] && \
         { command -v hipcc >/dev/null 2>&1 || [ -x /opt/rocm/bin/hipcc ] || \
           ls /opt/rocm-*/bin/hipcc >/dev/null 2>&1; }; then
        # No gfx: forward --has-rocm only when hipcc can build, else fall through to CPU prebuilt.
        _PREBUILT_CMD+=(--has-rocm)
    fi
    # Reporting only: the installer reads UNSLOTH_LLAMA_CPP_BACKEND itself.
    case "$_source_backend_choice" in
        cpu)
            if [ "$_HOST_SYSTEM" = "Darwin" ]; then
                step "llama.cpp" "UNSLOTH_LLAMA_CPP_BACKEND=cpu has no effect on macOS (universal build; use -ngl 0 at runtime for CPU-only)" "$C_WARN" >&2
            fi
            ;;
        vulkan)
            if [ "$_HOST_SYSTEM" = "Darwin" ]; then
                step "llama.cpp" "Vulkan has no effect on macOS; the universal build uses Metal" "$C_WARN" >&2
            else
                step "llama.cpp" "Vulkan selected for GGUF inference; the PyTorch training backend is unchanged" "$C_OK"
            fi
            ;;
        ""|auto|cuda|hip|rocm) ;;
        *) step "llama.cpp" "Ignoring UNSLOTH_LLAMA_CPP_BACKEND='$_source_backend_choice' (expected 'auto', 'cpu', 'cuda', 'vulkan', 'hip', or 'rocm')" "$C_WARN" >&2 ;;
    esac
    _PREBUILT_LOG="$(mktemp)"
    set +e
    if _is_verbose || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "1" ] || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "true" ]; then
        "${_PREBUILT_CMD[@]}" 2>&1 | tee "$_PREBUILT_LOG" | _filter_download_output
        _PREBUILT_STATUS=${PIPESTATUS[0]}
    else
        "${_PREBUILT_CMD[@]}" >"$_PREBUILT_LOG" 2>&1
        _PREBUILT_STATUS=$?
    fi
    set -e

    if [ "$_PREBUILT_STATUS" -eq 0 ]; then
        if grep -Fq "already matches" "$_PREBUILT_LOG"; then
            step "llama.cpp" "prebuilt up to date and validated"
        elif grep -Fq "keeping the existing complete install" "$_PREBUILT_LOG"; then
            # Exit 0 can also mean the existing tree was kept after a transient failure.
            step "llama.cpp" "update unavailable, existing prebuilt kept" "$C_WARN"
        else
            step "llama.cpp" "prebuilt installed and validated"
        fi
        if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ] && [ -d "$LLAMA_CPP_DIR" ]; then
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
        setup_fail 3 "llama.cpp install is blocked by an active llama.cpp process"
    elif [ "$_PREBUILT_STATUS" -eq 4 ]; then
        step "llama.cpp" "not enough disk space to install llama.cpp" "$C_WARN"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        substep "free up disk or move UNSLOTH_STUDIO_HOME/TMPDIR to a larger volume, then re-run"
        _LLAMA_CPP_NO_SPACE=true
        _has_local_llama_server "$LLAMA_CPP_DIR" || _LLAMA_CPP_DEGRADED=true
        # A preserved server may not satisfy an explicit backend request; never report success.
        if [ -n "$_explicit_llama_source_backend" ]; then
            step "llama.cpp" "$_explicit_llama_source_backend was explicitly requested, so the installer will not keep an unverified existing backend" "$C_ERR"
            setup_fail 1 "$_explicit_llama_source_backend was explicitly requested, so the installer will not keep an unverified existing llama.cpp backend."
        fi
    elif [ "$_PREBUILT_STATUS" -eq 5 ]; then
        step "llama.cpp" "selected backend could not be installed" "$C_ERR"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "prebuilt update failed; existing install restored"
        fi
        substep "check the error above, choose another backend, or retry"
        setup_fail 1 "The selected llama.cpp backend could not be installed, so the installer will not substitute a different source backend."
    elif [ "$_PREBUILT_STATUS" -eq 2 ]; then
        step "llama.cpp" "prebuilt install failed" "$C_WARN"
        print_llama_error_log "$_PREBUILT_LOG"
        _LLAMA_UPDATE_FAIL_REASON="$(_llama_update_fail_reason "$_PREBUILT_LOG")"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "prebuilt update failed; existing install restored"
        fi
        # A working GPU prebuilt beats any source build: keep it and retry next time.
        if _LLAMA_KEPT_GPU_PREBUILT="$(_gpu_prebuilt_to_keep_over_cpu_build "$LLAMA_CPP_DIR")"; then
            step "llama.cpp" "update failed ($_LLAMA_UPDATE_FAIL_REASON); keeping the installed $_LLAMA_KEPT_GPU_PREBUILT prebuilt, the next update will retry" "$C_WARN"
        else
            substep "falling back to source build"
            _NEED_LLAMA_SOURCE_BUILD=true
        fi
    else
        step "llama.cpp" "prebuilt helper failed unexpectedly" "$C_ERR"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "existing install was restored or left unchanged"
        fi
        substep "source build was not started because it cannot repair an unexpected helper or permissions error"
        setup_fail 1 "llama.cpp prebuilt helper failed unexpectedly (exit code $_PREBUILT_STATUS). Check the error above and retry setup."
    fi
fi

# Reuse a complete source build; a tree with a prebuilt marker must also pass the completeness check.
_LLAMA_REUSE_EXISTING=true
if [ "$_NEED_LLAMA_SOURCE_BUILD" = true ] && [ -d "$LLAMA_CPP_DIR" ]; then
    python "$SCRIPT_DIR/install_llama_prebuilt.py" \
        --check-existing-install "$LLAMA_CPP_DIR" >/dev/null 2>&1 \
        || _LLAMA_REUSE_EXISTING=false
fi

if [ "$_NEED_LLAMA_SOURCE_BUILD" = true ] && \
   [ "$_LLAMA_FORCE_COMPILE" != "1" ] && \
   [ -z "$_LLAMA_PR" ] && \
   [ "$_LLAMA_REUSE_EXISTING" = true ] && \
   [ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ] && \
   [ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    step "llama.cpp" "existing source build found; skipping rebuild"
    ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
    if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ]; then
        : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    fi
    _NEED_LLAMA_SOURCE_BUILD=false
fi

if [ -n "$STAGE_ROOT" ] && [ "$_NEED_LLAMA_SOURCE_BUILD" = true ]; then
    setup_fail 1 "Background staging cannot install system build tools for llama.cpp; retry with the foreground updater."
fi

# WSL: sudo needs a password unavailable during GGUF export, so install build deps here.
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
        if _can_read_tty; then
            printf "  %-15s" ""
            printf "accept? [Y/n] "
            # The device opened, so a failed read is EOF, not consent: decline.
            read -r REPLY </dev/tty || REPLY="n"
            case "$REPLY" in
                [nN]*)
                    substep "skipped -- run manually:"
                    substep "sudo apt-get install -y $_STILL_MISSING"
                    _SKIP_GGUF_BUILD=true
                    ;;
                *)
                    # Missing GGUF build deps are recoverable; do not let set -e abort on apt.
                    if sudo apt-get update -y </dev/null &&
                        sudo apt-get install -y $_STILL_MISSING </dev/null; then
                        step "gguf deps" "installed"
                    else
                        step "gguf deps" "install failed -- run manually:" "$C_WARN"
                        substep "sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                        _SKIP_GGUF_BUILD=true
                    fi
                    ;;
            esac
        else
            # -n refuses to prompt, -k ignores a cached timestamp: only NOPASSWD gets through.
            if sudo -n -k apt-get update -y </dev/null &&
                sudo -n -k apt-get install -y $_STILL_MISSING </dev/null; then
                step "gguf deps" "installed (non-interactive sudo)"
            else
                step "gguf deps" "needs sudo, no terminal -- run manually:" "$C_WARN"
                substep "sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                _SKIP_GGUF_BUILD=true
            fi
        fi
    else
        step "gguf deps" "missing (no sudo) -- install manually:" "$C_WARN"
        substep "apt-get install -y $_STILL_MISSING"
        _SKIP_GGUF_BUILD=true
    fi
fi

# Source build fallback at ~/.unsloth/llama.cpp, shared by inference and GGUF export.
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
            CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=ON $(_llama_relocatable_rpath_args)"
            # --depth 1 makes llama.cpp stamp build 1; Studio needs the tag's number (#12798).
            if [ -z "$_LLAMA_PR" ] && [ "$_RESOLVED_SOURCE_REF_KIND" != "commit" ] \
                && [[ "$_RESOLVED_SOURCE_REF" =~ ^b([0-9]+)$ ]]; then
                CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_NUMBER=${BASH_REMATCH[1]}"
            fi
            _TRY_METAL_CPU_FALLBACK=false
            _HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
            _HOST_MACHINE="$(uname -m 2>/dev/null || true)"
            _IS_MACOS_ARM64=false
            if [ "$_HOST_SYSTEM" = "Darwin" ] && { [ "$_HOST_MACHINE" = "arm64" ] || [ "$_HOST_MACHINE" = "aarch64" ]; }; then
                _IS_MACOS_ARM64=true
            fi

            # macOS: pin a low deployment target before CPU_FALLBACK_CMAKE_ARGS copies CMAKE_ARGS.
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
            # A CUDA toolkit alone is not proof of a GPU; callers gate on a real card first.
            _select_nvcc() {
                if command -v nvcc &>/dev/null; then
                    NVCC_PATH="$(command -v nvcc)"
                    GPU_BACKEND="cuda"
                elif [ -x /usr/local/cuda/bin/nvcc ]; then
                    NVCC_PATH="/usr/local/cuda/bin/nvcc"
                    export PATH="/usr/local/cuda/bin:$PATH"
                    GPU_BACKEND="cuda"
                elif ls /usr/local/cuda-*/bin/nvcc &>/dev/null 2>&1; then
                    NVCC_PATH="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
                    export PATH="$(dirname "$NVCC_PATH"):$PATH"
                    GPU_BACKEND="cuda"
                fi
            }

            if [ "$_setup_nvidia_usable" = true ]; then
                _select_nvcc
            fi

            # ROCm only when an AMD GPU was detected: hipcc alone must not select a HIP build.
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

            # Retry CUDA for a masked NVIDIA card after ROCm, so a mixed host prefers the visible AMD GPU.
            if [ -z "$GPU_BACKEND" ] && [ "$_setup_nvidia_physical" = true ]; then
                _select_nvcc
            fi

            _BUILD_DESC="building"
            if [ "$_IS_MACOS_ARM64" = true ]; then
                _BUILD_DESC="building (Metal)"
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_USE_BF16=ON -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
                CPU_FALLBACK_CMAKE_ARGS="$CPU_FALLBACK_CMAKE_ARGS -DGGML_METAL=OFF"
                _TRY_METAL_CPU_FALLBACK=true
            elif [ -n "$NVCC_PATH" ]; then
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
                        # An empty arch list means CPU instead of a PTX-only binary.
                        _raw_caps=""
                        # Resolve nvidia-smi off PATH too, or a CUDA host drops to CPU.
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
                        if [ -z "$CUDA_ARCHS" ] && [ "${UNSLOTH_NVIDIA_LIBRARY_PROBE:-1}" != "0" ]; then
                            CUDA_ARCHS="$(_resolve_cuda_archs "$(_probe_compute_caps)" "")"
                        fi

                        if [ -n "$CUDA_ARCHS" ]; then
                            CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
                            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS=--threads=0"
                            if _cuda_driver_needs_uncompressed_fatbin "$_DRIVER_MAX_CUDA"; then
                                CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA_COMPRESSION_MODE=none"
                                substep "driver CUDA $_DRIVER_MAX_CUDA predates 12.4; building uncompressed CUDA kernels it can load." "$C_WARN"
                            fi
                            _BUILD_DESC="building (CUDA, sm_${CUDA_ARCHS//;/+sm_})"

                            # Allow a host gcc newer than nvcc's whitelist; via env to avoid word-splitting.
                            export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }-allow-unsupported-compiler"
                        else
                            substep "could not detect a CUDA compute capability; building CPU llama.cpp instead of a PTX-only binary (set UNSLOTH_LLAMA_CUDA_ARCHS, e.g. \"120\", to force a CUDA build)." "$C_WARN"
                            GPU_BACKEND=""
                            _BUILD_DESC="building (CPU, CUDA arch undetectable)"
                        fi
                    fi
                fi
            elif [ "$GPU_BACKEND" = "rocm" ]; then
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

                if command -v hipconfig &>/dev/null; then
                    _HIP_CLANG_DIR="$(hipconfig -l 2>/dev/null || true)"
                    [ -n "$_HIP_CLANG_DIR" ] && export HIPCXX="$_HIP_CLANG_DIR/clang"
                fi

                GPU_TARGETS=""
                if command -v rocminfo &>/dev/null; then
                    _gfx_list=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]{2,4}[a-z]?' | sort -u || true)
                    _valid_gfx=""
                    for _gfx in $_gfx_list; do
                        if [[ "$_gfx" =~ ^gfx[0-9]{2,4}[a-z]?$ ]]; then
                            # Drop bare family ids (gfx11) from rocminfo's generic lines; clang accepts only gfxNNN.
                            # No real AMD GPU has a 2-digit gfx id.
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

            if [ -z "$GPU_BACKEND" ] && [ "$_IS_MACOS_ARM64" != true ] \
                    && _LLAMA_KEPT_GPU_PREBUILT="$(_gpu_prebuilt_to_keep_over_cpu_build "$LLAMA_CPP_DIR")"; then
                step "llama.cpp" "keeping the installed $_LLAMA_KEPT_GPU_PREBUILT prebuilt: the source fallback could only build for the CPU" "$C_WARN"
                BUILD_OK=false
            else
                substep "$_BUILD_DESC..."
            fi

            NCPU=$(_llama_build_jobs)
            verbose_substep "parallel jobs: $NCPU (RAM-capped; UNSLOTH_LLAMA_BUILD_JOBS overrides)"
            CMAKE_GENERATOR_ARGS=""
            if command -v ninja &>/dev/null; then
                CMAKE_GENERATOR_ARGS="-G Ninja"
            fi

            _gpu_fallback_label() {
                if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                    echo "Metal"
                elif [ -n "$GPU_BACKEND" ]; then
                    printf '%s' "$GPU_BACKEND" | tr '[:lower:]' '[:upper:]'
                fi
            }

            if [ "$BUILD_OK" = true ] && ! run_quiet_no_exit "cmake llama.cpp" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CMAKE_ARGS; then
                _FB_LABEL="$(_gpu_fallback_label)"
                if [ -n "$_FB_LABEL" ]; then
                    _TRY_METAL_CPU_FALLBACK=false
                    substep "$_FB_LABEL configure failed; retrying CPU build..." "$C_WARN"
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback after $_FB_LABEL configure failed)"
                        # Clear so a later build failure won't re-enter fallback.
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
            # Best-effort: the target exists only on llama.cpp trees carrying the diffusion example.
            run_quiet_no_exit "build diffusion visual server" cmake --build "$_BUILD_TMP/build" --config Release --target llama-diffusion-gemma-visual-server -j"$NCPU" || true
        fi

        # Opt-in smoke test (CUDA JIT stalls); runs before the install swap.
        if [ "$BUILD_OK" = true ] && _staged_validation_enabled; then
            _FB_LABEL="$(_gpu_fallback_label)"
            _SMOKE_KIND="$(_source_smoke_install_kind)"
            if [ -n "$_FB_LABEL" ]; then
                _SMOKE_CMD=(
                    python "$SCRIPT_DIR/install_llama_prebuilt.py"
                    --validate-install "$_BUILD_TMP"
                )
                [ -n "$_SMOKE_KIND" ] && _SMOKE_CMD+=(--install-kind "$_SMOKE_KIND")
                _SMOKE_RC=0
                run_quiet_no_exit "validate source llama.cpp" "${_SMOKE_CMD[@]}" || _SMOKE_RC=$?
                # Exit 4 is a full disk: a CPU rebuild needs more space, so keep what we have.
                if [ "$_SMOKE_RC" -eq 4 ]; then
                    substep "not enough disk space to validate the $_FB_LABEL build; keeping it" "$C_WARN"
                    _LLAMA_CPP_NO_SPACE=true
                elif [ "$_SMOKE_RC" -ne 0 ]; then
                    substep "$_FB_LABEL source build failed smoke test; retrying CPU build..." "$C_WARN"
                    _TRY_METAL_CPU_FALLBACK=false
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback after $_FB_LABEL smoke failed)"
                        GPU_BACKEND=""
                        run_quiet_no_exit "build llama-server (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU" || BUILD_OK=false
                        if [ "$BUILD_OK" = true ]; then
                            run_quiet_no_exit "build llama-quantize (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-quantize -j"$NCPU" || true
                            run_quiet_no_exit "build diffusion visual server (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-diffusion-gemma-visual-server -j"$NCPU" || true
                        fi
                    else
                        BUILD_OK=false
                    fi
                fi
            fi
        fi

        # A GPU build that fell back to CPU on the way is caught here, after the fact.
        if [ "$BUILD_OK" = true ] && [ -z "$GPU_BACKEND" ] && [ "$_TRY_METAL_CPU_FALLBACK" != true ]; then
            if _LLAMA_KEPT_GPU_PREBUILT="$(_gpu_prebuilt_to_keep_over_cpu_build "$LLAMA_CPP_DIR")"; then
                step "llama.cpp" "keeping the installed $_LLAMA_KEPT_GPU_PREBUILT prebuilt: the source build fell back to the CPU" "$C_WARN"
                BUILD_OK=false
            elif [ "$_setup_nvidia_physical" = true ] || [ "$_setup_amd_detected" = true ] \
                    || _setup_has_intel_gpu; then
                _LLAMA_CPU_ONLY_ON_GPU_HOST=true
            fi
        fi
        # Swap only after build succeeds -- preserves existing install on failure
        if [ "$BUILD_OK" = true ]; then
            _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install" "$_RUNTIME_ROOT_IS_CUSTOM"
            # || true so errexit does not strand the build; rm's stderr names the subpath.
            rm -rf "$LLAMA_CPP_DIR" || true
            if [ -e "$LLAMA_CPP_DIR" ]; then
                if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
                    _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
                fi
                step "llama.cpp" "built, but the existing install could not be replaced" "$C_ERR"
                setup_fail 3 "The llama.cpp build succeeded but $LLAMA_CPP_DIR could not be replaced. The new build is at $_BUILD_TMP."
            fi
            mv "$_BUILD_TMP" "$LLAMA_CPP_DIR"
            : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
            # Symlink to llama.cpp root -- check_llama_cpp() looks for the binary there
            QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
            if [ -f "$QUANTIZE_BIN" ]; then
                ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
            fi
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
        elif [ -n "$_LLAMA_KEPT_GPU_PREBUILT" ]; then
            print_installed_llama_prebuilt_release "$LLAMA_CPP_DIR"
        else
            step "llama.cpp" "build failed" "$C_ERR"
            [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
        fi
    fi
}
fi  # end _SKIP_GGUF_BUILD check

# ── arm64 Linux GPU: CPU prebuilt as a last resort ──
# Source build produced nothing: install the CPU prebuilt. Skipped on a full disk.
if [ "$_LLAMA_CPP_DEGRADED" = true ] \
        && [ "$_LLAMA_CPP_NO_SPACE" != true ] \
        && [ "$_LLAMA_KEEP_PREBUILT_ACTIVE" != true ] \
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
    if run_quiet_no_exit "arm64 CPU prebuilt" "${_ARM64_CPU_CMD[@]}"; then
        step "llama.cpp" "arm64 CPU prebuilt installed (GPU build unavailable)" "$C_WARN"
        _LLAMA_CPP_DEGRADED=false
        if [ "$_setup_nvidia_physical" = true ] || [ "$_setup_amd_detected" = true ]; then
            _LLAMA_CPU_ONLY_ON_GPU_HOST=true
        fi
        print_installed_llama_prebuilt_release "$LLAMA_CPP_DIR"
    fi
fi

if [ ! -L "$LLAMA_CPP_DIR" ] && {
    [ "$_RUNTIME_ROOT_IS_CUSTOM" != true ] ||
        [ -f "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" ] ||
        _studio_owned_adoptable "$LLAMA_CPP_DIR"
}; then
    _remove_agent_instruction_files "$LLAMA_CPP_DIR"
fi

# whisper.cpp: optional, fail-open; installs beside llama.cpp as _managed_whisper_cpp_dir() expects.
WHISPER_CPP_DIR="$UNSLOTH_HOME/whisper.cpp"
if [ -n "${WHISPER_SERVER_PATH:-}" ] || [ -n "${UNSLOTH_WHISPER_CPP_PATH:-}" ]; then
    verbose_substep "whisper.cpp: using a user-configured binary/dir; skipping managed install"
elif [ "${UNSLOTH_SKIP_WHISPER_INSTALL:-0}" = "1" ]; then
    verbose_substep "whisper.cpp: install skipped (UNSLOTH_SKIP_WHISPER_INSTALL=1)"
else
    if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$WHISPER_CPP_DIR" "whisper.cpp install" "$_RUNTIME_ROOT_IS_CUSTOM"
    fi
    _WHISPER_CMD=(python "$SCRIPT_DIR/install_whisper_prebuilt.py" --install-dir "$WHISPER_CPP_DIR")
    if [ -n "${UNSLOTH_WHISPER_RELEASE_TAG:-}" ]; then
        _WHISPER_CMD+=(--published-release-tag "$UNSLOTH_WHISPER_RELEASE_TAG")
    fi
    if [ -n "${_setup_gfx:-}" ]; then
        _WHISPER_CMD+=(--rocm-gfx "$_setup_gfx")
    elif [ "$_setup_amd_detected" = true ]; then
        _WHISPER_CMD+=(--has-rocm)
    fi
    _WHISPER_LOG="$(mktemp)"
    set +e
    if _is_verbose || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "1" ] || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "true" ]; then
        "${_WHISPER_CMD[@]}" 2>&1 | tee "$_WHISPER_LOG" | _filter_download_output
        _WHISPER_STATUS=${PIPESTATUS[0]}
    else
        "${_WHISPER_CMD[@]}" >"$_WHISPER_LOG" 2>&1
        _WHISPER_STATUS=$?
    fi
    set -e
    if [ "$_WHISPER_STATUS" -eq 0 ]; then
        if grep -Fq "already matches" "$_WHISPER_LOG"; then
            step "whisper.cpp" "prebuilt up to date"
        elif grep -Fq "keeping the existing complete install" "$_WHISPER_LOG"; then
            step "whisper.cpp" "update unavailable, existing prebuilt kept" "$C_WARN"
        else
            step "whisper.cpp" "prebuilt installed"
        fi
        if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ] && [ -d "$WHISPER_CPP_DIR" ]; then
            : > "$WHISPER_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
        fi
        rm -f "$_WHISPER_LOG"
    elif [ "$_WHISPER_STATUS" -eq 3 ]; then
        step "whisper.cpp" "install busy; keeping existing runtime" "$C_WARN"
        rm -f "$_WHISPER_LOG"
    else
        _WHISPER_RECOVERED=false
        _WHISPER_BUILD="$SCRIPT_DIR/../scripts/build_whisper_cpp.sh"
        if [ "${UNSLOTH_WHISPER_FORCE_COMPILE:-0}" = "1" ] && [ -f "$_WHISPER_BUILD" ] \
                && command -v cmake >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
            substep "whisper.cpp prebuilt unavailable; building from source (UNSLOTH_WHISPER_FORCE_COMPILE=1)..."
            # A stale marker would report "already matches" over the source binary.
            rm -f "$WHISPER_CPP_DIR/UNSLOTH_WHISPER_PREBUILT_INFO.json" 2>/dev/null || true
            if run_quiet_no_exit "whisper.cpp source build" \
                    env UNSLOTH_HOME="$UNSLOTH_HOME" sh "$_WHISPER_BUILD"; then
                _WHISPER_RECOVERED=true
                step "whisper.cpp" "source build installed"
                if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ] && [ -d "$WHISPER_CPP_DIR" ]; then
                    : > "$WHISPER_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
                fi
            else
                :
            fi
        fi
        if [ "$_WHISPER_RECOVERED" != true ]; then
            if [ "$_WHISPER_STATUS" -eq 2 ]; then
                _WHISPER_REQUIRED_TAG="$(sed -n 's/.*slim bundle requires llama\.cpp \([^; ]*\).*/\1/p' "$_WHISPER_LOG" | tail -n 1)"
                _WHISPER_INSTALLED_TAG="$(python - "$UNSLOTH_HOME/llama.cpp/UNSLOTH_PREBUILT_INFO.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("release_tag", ""))
except Exception:
    pass
PY
)"
                _WHISPER_PAIRING="installed llama.cpp ${_WHISPER_INSTALLED_TAG:-unknown}; whisper requires ${_WHISPER_REQUIRED_TAG:-unknown}"
                step "whisper.cpp" "no compatible prebuilt ($_WHISPER_PAIRING); curated whisper.cpp dictation is unavailable; publish the paired releases in llama.cpp then whisper.cpp order; browser and Transformers dictation remain available" "$C_WARN"
            else
                step "whisper.cpp" "prebuilt install failed; curated whisper.cpp dictation is unavailable; retry setup or inspect verbose output; browser and Transformers dictation remain available" "$C_WARN"
            fi
        fi
        rm -f "$_WHISPER_LOG"
    fi
fi

# audio.cpp vendors its own ggml, so it does not pair with llama. Fail-open.
AUDIO_CPP_DIR="$UNSLOTH_HOME/audio.cpp"
if [ -n "${AUDIOCPP_SERVER_PATH:-}" ] || [ -n "${UNSLOTH_AUDIO_CPP_PATH:-}" ]; then
    verbose_substep "audio.cpp: using a user-configured binary/dir; skipping managed install"
elif [ "${UNSLOTH_SKIP_AUDIO_CPP_INSTALL:-0}" = "1" ]; then
    verbose_substep "audio.cpp: install skipped (UNSLOTH_SKIP_AUDIO_CPP_INSTALL=1)"
elif [ -f "$SCRIPT_DIR/install_audio_cpp_prebuilt.py" ]; then
    if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$AUDIO_CPP_DIR" "audio.cpp install" "$_RUNTIME_ROOT_IS_CUSTOM"
    fi
    _AUDIO_CPP_CMD=(python "$SCRIPT_DIR/install_audio_cpp_prebuilt.py" --install-dir "$AUDIO_CPP_DIR")
    if [ -n "${UNSLOTH_AUDIO_CPP_ACCELERATOR:-}" ]; then
        _AUDIO_CPP_CMD+=(--accelerator "$UNSLOTH_AUDIO_CPP_ACCELERATOR")
    fi
    _AUDIO_CPP_LOG="$(mktemp)"
    set +e
    if _is_verbose || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "1" ] || [ "${UNSLOTH_TAURI_UPDATE:-0}" = "true" ]; then
        "${_AUDIO_CPP_CMD[@]}" 2>&1 | tee "$_AUDIO_CPP_LOG" | _filter_download_output
        _AUDIO_CPP_STATUS=${PIPESTATUS[0]}
    else
        "${_AUDIO_CPP_CMD[@]}" >"$_AUDIO_CPP_LOG" 2>&1
        _AUDIO_CPP_STATUS=$?
    fi
    set -e
    if [ "$_AUDIO_CPP_STATUS" -eq 0 ]; then
        if grep -Fq "already matches" "$_AUDIO_CPP_LOG"; then
            step "audio.cpp" "prebuilt up to date"
        elif grep -Fq "keeping the existing complete install" "$_AUDIO_CPP_LOG"; then
            step "audio.cpp" "update unavailable, existing prebuilt kept" "$C_WARN"
        else
            step "audio.cpp" "prebuilt installed"
        fi
        if [ "$_RUNTIME_ROOT_IS_CUSTOM" = true ] && [ -d "$AUDIO_CPP_DIR" ]; then
            : > "$AUDIO_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
        fi
    elif [ "$_AUDIO_CPP_STATUS" -eq 3 ]; then
        step "audio.cpp" "install busy; keeping existing runtime" "$C_WARN"
    else
        step "audio.cpp" "prebuilt install failed; audio.cpp models are unavailable; retry setup or inspect verbose output; other audio engines remain available" "$C_WARN"
    fi
    rm -f "$_AUDIO_CPP_LOG"
fi

# Named in the footer: every path to a lost GPU exits 0.
_print_llama_gpu_notes() {
    if [ -n "$_LLAMA_KEPT_GPU_PREBUILT" ]; then
        printf "  ${C_WARN}%-15s%s${C_RST}\n" "llama.cpp" "update failed (${_LLAMA_UPDATE_FAIL_REASON:-the source fallback could only build for the CPU}); the installed $_LLAMA_KEPT_GPU_PREBUILT prebuilt was kept and the next update will retry"
    fi
    if [ "$_LLAMA_CPU_ONLY_ON_GPU_HOST" = true ]; then
        printf "  ${C_WARN}%-15s%s${C_RST}\n" "warning" "GPU acceleration is unavailable: the prebuilt install failed and the source fallback could only build for the CPU (no GPU toolkit, or the GPU build failed above), so GGUF inference will run on the CPU"
        printf "  ${C_WARN}%-15s%s${C_RST}\n" "" "fix the cause named above, then re-run this installer to restore the GPU"
    fi
}

if [ "$_LLAMA_ONLY" = "1" ]; then
    echo ""
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "llama.cpp update finished (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "llama.cpp update finished"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    _print_llama_gpu_notes
elif [ "$IS_COLAB" = true ]; then
    echo ""
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "Unsloth Studio Setup Complete (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "Unsloth Studio Setup Complete"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    _print_llama_gpu_notes
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
    _print_llama_gpu_notes
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_DIM}%-15s${C_WARN}%s${C_RST}\n" "launch" "unsloth studio -p 8888"
    else
        printf "  ${C_DIM}%-15s${C_OK}%s${C_RST}\n" "launch" "unsloth studio -p 8888"
    fi
    printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
    printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
fi
echo ""

# From install.sh exit non-zero so it can report GGUF failure; `studio update` stays successful.
if [ "$_LLAMA_CPP_DEGRADED" = true ] && [ "${SKIP_STUDIO_BASE:-0}" = "1" ]; then
    # In Tauri mode a non-zero exit is not "report", it is "abort": install.rs turns the
    # error into "Installation failed", so one transient prebuilt download failure (a
    # single HTTP 403 rate limit will do it) fails the whole first-launch install of an
    # app whose own footer just said Installed. Everything except GGUF inference works,
    # and whisper.cpp in this same script already degrades rather than failing for
    # exactly this case. Match it, and say what is missing and how to get it back.
    #
    # PROGRESS, not STEP: install.rs maps [TAURI:STEP] to the install-step event, and
    # use-tauri-backend.ts counts those against the seven-entry INSTALL_STEPS list that
    # install.sh already emits in full, so an eighth marker renders "Step 8 of 7" and
    # discards the payload. [TAURI:PROGRESS] becomes install-progress-detail, which
    # InstallingContent renders verbatim, so the user actually reads the limitation.
    #
    # DIAG as well: progress detail is cleared by the next install-step and is gone
    # once the install screen closes, so it cannot answer "why is GGUF missing"
    # afterwards. record_diag_marker keeps this in the support report.
    case "${UNSLOTH_TAURI_MODE:-0}" in
        1|true)
            printf '[TAURI:PROGRESS] %s\n' \
                "llama.cpp unavailable; GGUF inference is disabled until 'unsloth studio update' succeeds"
            printf '[TAURI:DIAG] %s\n' "llama_cpp=unavailable"
            ;;
        *)
            setup_fail 1 "llama.cpp setup did not produce a usable server"
            ;;
    esac
fi

# A desktop repair runs update.rs, which sets UNSLOTH_TAURI_UPDATE alone; record the DIAG
# marker. Exit stays successful.
if [ "$_LLAMA_CPP_DEGRADED" = true ] && [ "${SKIP_STUDIO_BASE:-0}" != "1" ]; then
    case "${UNSLOTH_TAURI_UPDATE:-0}" in
        1|true) printf '[TAURI:DIAG] %s\n' "llama_cpp=unavailable" ;;
    esac
fi
