#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Install or refresh the optional systemd user unit for Unsloth Studio (#9258).
# Called from install.sh; may also be invoked directly after setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/unsloth-studio.service.in"
UNIT_NAME="unsloth-studio.service"

_UNSLOTH_EXE=""
_STUDIO_HOME=""
# Match `unsloth studio` (127.0.0.1). Opt into LAN with --host / UNSLOTH_SYSTEMD_HOST=0.0.0.0.
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

_systemd_escape() {
    # systemd unit escaping for paths embedded in quoted strings.
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --unsloth-exe)
            _UNSLOTH_EXE="${2:-}"
            shift 2
            ;;
        --studio-home)
            _STUDIO_HOME="${2:-}"
            shift 2
            ;;
        --host)
            _HOST="${2:-}"
            shift 2
            ;;
        --port)
            _PORT="${2:-}"
            shift 2
            ;;
        --enable)
            _DO_ENABLE=true
            shift
            ;;
        --start)
            _DO_ENABLE=true
            _DO_START=true
            shift
            ;;
        -h|--help)
            _usage
            exit 0
            ;;
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
    ''|*[!0-9]*) echo "ERROR: --port must be a positive integer." >&2; exit 2 ;;
    0) echo "ERROR: --port must be greater than zero." >&2; exit 2 ;;
esac
if [ -z "$_HOST" ]; then
    echo "ERROR: --host must not be empty." >&2
    exit 2
fi
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
if [ -n "$_STUDIO_HOME" ]; then
    if [ -d "$_STUDIO_HOME" ]; then
        _STUDIO_HOME="$(CDPATH= cd -P -- "$_STUDIO_HOME" && pwd -P)"
    fi
fi

_exe_q=$(_systemd_escape "$_UNSLOTH_EXE")
_exec_start="\"$_exe_q\" studio -H \"$_HOST\" -p $_PORT"
_exec_stop="\"$_exe_q\" studio stop"

_env_lines=""
if [ -n "$_STUDIO_HOME" ]; then
    _home_q=$(_systemd_escape "$_STUDIO_HOME")
    _env_lines="Environment=\"UNSLOTH_STUDIO_HOME=$_home_q\""
fi

_unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$_unit_dir"
_unit_path="$_unit_dir/$UNIT_NAME"
_tmp="$(mktemp)"
sed \
    -e "s|@@ENVIRONMENT_LINES@@|${_env_lines}|" \
    -e "s|@@EXEC_START@@|${_exec_start}|" \
    -e "s|@@EXEC_STOP@@|${_exec_stop}|" \
    "$TEMPLATE" > "$_tmp"
mv "$_tmp" "$_unit_path"
chmod 0644 "$_unit_path"

if [ "$_DO_ENABLE" != true ]; then
    printf '%s\n' "$_unit_path"
    exit 0
fi

systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
if [ "$_DO_START" = true ]; then
    systemctl --user restart "$UNIT_NAME" || systemctl --user start "$UNIT_NAME"
fi
printf '%s\n' "$_unit_path"
