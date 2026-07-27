#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_FUNC_FILE=$(mktemp)
{
    sed -n '/^run_install_cmd()/,/^}/p' "$INSTALL_SH"
    sed -n '/^run_install_cmd_retry()/,/^}/p' "$INSTALL_SH"
    sed -n '/^tauri_log()/,/^}/p' "$INSTALL_SH"
    sed -n '/^tauri_clear_install_error()/,/^}/p' "$INSTALL_SH"
} > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

substep() {
    :
}

step() {
    :
}

sleep() {
    :
}

_is_verbose() {
    return 1
}

_redact_install_output() {
    cat "$1"
}

echo "=== run_install_cmd_retry Tauri failure context ==="

TAURI_MODE=true
_test_attempt=0
_test_command() {
    _test_attempt=$((_test_attempt + 1))
    [ "$_test_attempt" -eq 2 ]
}

UNSLOTH_INSTALL_RETRIES=3
UNSLOTH_INSTALL_RETRY_DELAY=0
_stdout_file=$(mktemp)
_stderr_file=$(mktemp)
trap 'rm -f "$_stdout_file" "$_stderr_file"' EXIT
run_install_cmd_retry "install PyTorch" _test_command >"$_stdout_file" 2>"$_stderr_file"
_stdout_clear_count=$(grep -c '^\[TAURI:ERROR_CLEAR\] install PyTorch recovered$' "$_stdout_file")
_stderr_clear_count=$(grep -c '^\[TAURI:ERROR_CLEAR\] install PyTorch recovered$' "$_stderr_file")
if [ "$_stdout_clear_count" -ne 1 ] || [ "$_stderr_clear_count" -ne 1 ]; then
    echo "  FAIL: recovered retry emitted $_stdout_clear_count stdout and $_stderr_clear_count stderr clear markers"
    exit 1
fi
echo "  PASS: recovered retry clears stale context on both streams"

_test_command() {
    return 9
}

if run_install_cmd_retry "install PyTorch" _test_command >"$_stdout_file" 2>"$_stderr_file"; then
    echo "  FAIL: permanent failure returned success"
    exit 1
else
    _exit_code=$?
fi
if [ "$_exit_code" -ne 9 ]; then
    echo "  FAIL: permanent failure returned exit code $_exit_code"
    exit 1
fi
if grep -q '^\[TAURI:ERROR_CLEAR\]' "$_stdout_file" ||
    grep -q '^\[TAURI:ERROR_CLEAR\]' "$_stderr_file"; then
    echo "  FAIL: permanent failure cleared its failure context"
    exit 1
fi
echo "  PASS: permanent failure retains its failure and exit code"

UNSLOTH_INSTALL_RETRIES=1
if run_install_cmd_retry "preferred PyTorch build" _test_command >"$_stdout_file" 2>"$_stderr_file"; then
    echo "  FAIL: failed preferred build returned success"
    exit 1
fi
_test_command() {
    return 0
}
run_install_cmd_retry "fallback PyTorch build" _test_command >"$_stdout_file" 2>"$_stderr_file"
if ! grep -q '^\[TAURI:ERROR_CLEAR\] fallback PyTorch build recovered$' "$_stdout_file" ||
    ! grep -q '^\[TAURI:ERROR_CLEAR\] fallback PyTorch build recovered$' "$_stderr_file"; then
    echo "  FAIL: successful fallback retained the preferred build failure"
    exit 1
fi
echo "  PASS: successful fallback clears an exhausted preferred failure"
