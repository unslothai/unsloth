#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Writes (and with --enable/--start, enables) the optional systemd user unit for Unsloth Studio.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/unsloth-studio.service.in"
UNIT_NAME="unsloth-studio.service"

_UNSLOTH_EXE=""
_STUDIO_HOME=""
_HOST="${UNSLOTH_SYSTEMD_HOST:-127.0.0.1}"
_PORT="${UNSLOTH_SYSTEMD_PORT:-8888}"
_DO_ENABLE=false
_DO_START=false

_usage() {
    cat <<'EOF'
Usage: install_user_service.sh --unsloth-exe PATH [options]

Options:
  --unsloth-exe PATH   Absolute path to the unsloth CLI (required)
  --studio-home PATH   Sets UNSLOTH_STUDIO_HOME in the unit (custom installs)
  --host HOST          Bind address (default: 127.0.0.1; UNSLOTH_SYSTEMD_HOST overrides;
                       use 0.0.0.0 for LAN / all-interfaces)
  --port PORT          Listen port (default: 8888; UNSLOTH_SYSTEMD_PORT overrides)
  --enable             Write unit, daemon-reload, and systemctl --user enable
  --start              Also systemctl --user start (implies --enable)
  -h, --help           Show this help
EOF
}

# Inside a double-quoted unit value: backslash and quote escape, % is a specifier.
_unit_quote() {
    local v="$1"
    v="${v//\\/\\\\}"
    v="${v//\"/\\\"}"
    v="${v//%/%%}"
    printf '"%s"' "$v"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --unsloth-exe) _UNSLOTH_EXE="${2:-}"; shift 2 ;;
        --studio-home) _STUDIO_HOME="${2:-}"; shift 2 ;;
        --host) _HOST="${2:-}"; shift 2 ;;
        --port) _PORT="${2:-}"; shift 2 ;;
        --enable) _DO_ENABLE=true; shift ;;
        --start) _DO_ENABLE=true; _DO_START=true; shift ;;
        -h|--help) _usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            _usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$_UNSLOTH_EXE" ]; then
    echo "ERROR: --unsloth-exe is required." >&2
    exit 2
fi
if [ ! -x "$_UNSLOTH_EXE" ]; then
    echo "ERROR: unsloth executable not found or not executable: $_UNSLOTH_EXE" >&2
    exit 1
fi
case "$_PORT" in
    ''|*[!0-9]*|0) echo "ERROR: --port must be a positive integer." >&2; exit 2 ;;
esac
if [ -z "$_HOST" ]; then
    echo "ERROR: --host must not be empty." >&2
    exit 2
fi
for _v in "$_HOST" "$_STUDIO_HOME" "$_UNSLOTH_EXE"; do
    case "$_v" in
        *[[:cntrl:]]*) echo "ERROR: control characters are not allowed in unit values: $_v" >&2; exit 2 ;;
    esac
done
case "$_UNSLOTH_EXE" in
    *[\"\'\\]*) echo "ERROR: systemd cannot run an executable whose path contains quotes or backslashes: $_UNSLOTH_EXE" >&2; exit 2 ;;
esac
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: service template missing: $TEMPLATE" >&2
    exit 1
fi
if [ "$(uname -s 2>/dev/null || true)" != "Linux" ]; then
    echo "ERROR: systemd user service install is supported on Linux only." >&2
    exit 1
fi
if [ "$_DO_ENABLE" = true ]; then
    if ! command -v systemctl >/dev/null 2>&1; then
        echo "ERROR: systemctl not found; systemd is required." >&2
        exit 1
    fi
    if ! systemctl --user show-environment >/dev/null 2>&1; then
        echo "ERROR: systemd user session is unavailable (is the user bus running?)." >&2
        exit 1
    fi
fi

_UNSLOTH_EXE="$(CDPATH= cd -P -- "$(dirname "$_UNSLOTH_EXE")" && pwd -P)/$(basename "$_UNSLOTH_EXE")"
if [ -n "$_STUDIO_HOME" ] && [ -d "$_STUDIO_HOME" ]; then
    _STUDIO_HOME="$(CDPATH= cd -P -- "$_STUDIO_HOME" && pwd -P)"
fi

# No ExecStop: `unsloth studio stop` would also stop every other Studio on this home.
_exec_start="$(_unit_quote "$_UNSLOTH_EXE") studio -H $(_unit_quote "$_HOST") -p $_PORT"
_env_line=""
[ -n "$_STUDIO_HOME" ] && _env_line="Environment=$(_unit_quote "UNSLOTH_STUDIO_HOME=$_STUDIO_HOME")"

# Enabling: the unit goes where the user manager looks, which follows its own HOME, not ours.
_cfg_home="$HOME"
_cfg_xdg="${XDG_CONFIG_HOME:-}"
if [ "$_DO_ENABLE" = true ]; then
    _mgr_env=$(systemctl --user show-environment 2>/dev/null || true)
    _mgr_home=$(printf '%s\n' "$_mgr_env" | sed -n 's/^HOME=//p' | head -n 1)
    case "$_mgr_home" in /*) _cfg_home="$_mgr_home"; _cfg_xdg=$(printf '%s\n' "$_mgr_env" | sed -n 's/^XDG_CONFIG_HOME=//p' | head -n 1) ;; esac
fi
case "$_cfg_xdg" in
    /*) _unit_dir="$_cfg_xdg/systemd/user" ;;
    *) _unit_dir="$_cfg_home/.config/systemd/user" ;;
esac
mkdir -p "$_unit_dir"
_unit_path="$_unit_dir/$UNIT_NAME"
# Never replace a unit the user wrote themselves.
if [ -e "$_unit_path" ] && [ "$(head -n 1 "$_unit_path" 2>/dev/null)" != "# unsloth-studio-managed-systemd" ]; then
    echo "ERROR: $_unit_path exists and was not written by Unsloth; leaving it alone." >&2
    exit 1
fi
_tmp="$(mktemp "$_unit_dir/.$UNIT_NAME.XXXXXX")"
trap 'rm -f "$_tmp"' EXIT
while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
        @@ENVIRONMENT_LINES@@) [ -z "$_env_line" ] || printf '%s\n' "$_env_line" ;;
        ExecStart=@@EXEC_START@@) printf 'ExecStart=%s\n' "$_exec_start" ;;
        *) printf '%s\n' "$_line" ;;
    esac
done < "$TEMPLATE" > "$_tmp"
chmod 0644 "$_tmp"
mv "$_tmp" "$_unit_path"

if [ "$_DO_ENABLE" != true ]; then
    printf '%s\n' "$_unit_path"
    exit 0
fi

systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
if [ "$_DO_START" = true ]; then
    systemctl --user restart "$UNIT_NAME"
fi
printf '%s\n' "$_unit_path"
