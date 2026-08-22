#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=== test_tauri_cli_path_precedence ==="

_fn_start=$(grep -n '^_path_has_dir() {' "$INSTALL_SH" | head -1 | cut -d: -f1)
_guard_end=$(grep -n '^# end of the PATH persistence block$' "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ -z "$_fn_start" ] || [ -z "$_guard_end" ]; then
    echo "  FAIL: install.sh PATH persistence block not found"
    exit 1
fi
sed -n "${_fn_start},${_guard_end}p" "$INSTALL_SH" > "$WORK/path_guard.sh"

_run_guard() {
    _home="$1"
    _tauri="$2"
    mkdir -p "$_home/.local/bin" "$_home/foreign"
    printf '#!/bin/sh\n' > "$_home/.local/bin/unsloth"
    printf '#!/bin/sh\n' > "$_home/foreign/unsloth"
    chmod +x "$_home/.local/bin/unsloth" "$_home/foreign/unsloth"
    (
        set +e
        step() { :; }
        HOME="$_home"; export HOME
        SHELL="/bin/bash"; export SHELL
        TAURI_MODE="$_tauri"; export TAURI_MODE
        PATH="$_home/foreign:$_home/.local/bin:/usr/bin:/bin"; export PATH
        _LOCAL_BIN="$HOME/.local/bin"
        _STUDIO_HOME_REDIRECT=default
        _UNSLOTH_LOGIN_PATH="$PATH"
        _UNSLOTH_UV_BIN_DIR=""
        # shellcheck disable=SC1090
        . "$WORK/path_guard.sh"
    ) >/dev/null 2>&1
}

_tauri_home="$WORK/tauri"
mkdir -p "$_tauri_home"
printf 'export PATH="%s/foreign:$HOME/.local/bin:$PATH"\n' "$_tauri_home" > "$_tauri_home/.bashrc"
_run_guard "$_tauri_home" true
_run_guard "$_tauri_home" true

_exact_line='export PATH="$HOME/.local/bin:$PATH"'
_exact_count=$(grep -cF "$_exact_line" "$_tauri_home/.bashrc" || true)
if [ "$_exact_count" = 1 ]; then
    echo "  PASS: Tauri adds one idempotent managed CLI prepend"
else
    echo "  FAIL: Tauri adds one idempotent managed CLI prepend (count=$_exact_count)"
    exit 1
fi

_resolved=$(HOME="$_tauri_home" PATH="/usr/bin:/bin" bash --noprofile --norc -c 'source "$HOME/.bashrc"; command -v unsloth')
if [ "$_resolved" = "$_tauri_home/.local/bin/unsloth" ]; then
    echo "  PASS: a fresh shell resolves the managed CLI before the foreign launcher"
else
    echo "  FAIL: a fresh shell resolves the managed CLI before the foreign launcher (got=$_resolved)"
    exit 1
fi

_normal_home="$WORK/normal"
mkdir -p "$_normal_home"
printf 'export PATH="%s/foreign:$HOME/.local/bin:$PATH"\n' "$_normal_home" > "$_normal_home/.bashrc"
_run_guard "$_normal_home" false
if grep -qF "$_exact_line" "$_normal_home/.bashrc"; then
    echo "  FAIL: non-Tauri install rewrites an existing PATH entry"
    exit 1
else
    echo "  PASS: non-Tauri install keeps existing PATH behavior"
fi

_tauri_marker=$(grep -n 'Tauri mode: done, skip shortcuts and auto-launch' "$INSTALL_SH" | head -1 | cut -d: -f1)
_tauri_exit=$(awk -v start="$_tauri_marker" 'NR > start && /exit 0/ { print NR; exit }' "$INSTALL_SH")
_shadow_probe=$(grep -n 'PATH="\$_UNSLOTH_LOGIN_PATH" command -v unsloth' "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ -n "$_tauri_marker" ] && [ -n "$_tauri_exit" ] && [ -n "$_shadow_probe" ] && [ "$_shadow_probe" -lt "$_tauri_exit" ]; then
    echo "  PASS: Tauri checks the inherited PATH before its success exit"
else
    echo "  FAIL: Tauri shadow check is not before its success exit"
    exit 1
fi
