#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"

_FUNC_FILE=$(mktemp)
{
    sed -n '/^run_install_cmd()/,/^}/p' "$INSTALL_SH"
    sed -n '/^run_install_cmd_retry()/,/^}/p' "$INSTALL_SH"
    sed -n '/^tauri_log()/,/^}/p' "$INSTALL_SH"
    sed -n '/^tauri_stream_log()/,/^}/p' "$INSTALL_SH"
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
    cat "$@"
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
    printf '%s\n' "resolver error: no space left on device"
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
if ! grep -qxF '[TAURI:OUTPUT_CLEAR] install PyTorch' "$_stderr_file" ||
    ! grep -qxF 'resolver error: no space left on device' "$_stderr_file" ||
    ! tail -n 1 "$_stderr_file" |
        grep -qxF '[TAURI:ERROR_OUTPUT] install PyTorch failed (exit code 9)'; then
    echo "  FAIL: quiet failure did not bind its command output to the structured error"
    exit 1
fi
echo "  PASS: permanent failure retains its command output and exit code"

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

_test_command() {
    return 0
}
run_install_cmd "successful unstructured fallback" _test_command >"$_stdout_file" 2>"$_stderr_file"
if ! grep -q '^\[TAURI:ERROR_CLEAR\] successful unstructured fallback recovered$' "$_stdout_file" ||
    ! grep -q '^\[TAURI:ERROR_CLEAR\] successful unstructured fallback recovered$' "$_stderr_file"; then
    echo "  FAIL: initial success did not clear unstructured failure context"
    exit 1
fi
echo "  PASS: every successful wrapped command clears unstructured failure context"

_is_verbose() {
    return 0
}
_test_command() {
    printf '%s\n' "resolver error: proxy authentication required"
    return 7
}
set +e
(
    set -e
    run_install_cmd "verbose failure" _test_command
) >"$_stdout_file" 2>"$_stderr_file"
_exit_code=$?
set -e
if [ "$_exit_code" -ne 7 ]; then
    echo "  FAIL: verbose failure returned exit code $_exit_code instead of 7"
    exit 1
fi
if ! grep -qxF '[TAURI:OUTPUT_CLEAR] verbose failure' "$_stdout_file" ||
    ! grep -qxF 'resolver error: proxy authentication required' "$_stdout_file" ||
    ! tail -n 1 "$_stdout_file" |
        grep -qxF '[TAURI:ERROR_OUTPUT] verbose failure failed (exit code 7)'; then
    echo "  FAIL: verbose failure did not bind its command output and exit code"
    exit 1
fi
echo "  PASS: verbose failure retains its command output and exit code under set -e"

_test_command() {
    _cmd_rc=9
    return 0
}
run_install_cmd "verbose clobbering success" _test_command >"$_stdout_file" 2>"$_stderr_file"
if ! grep -q '^\[TAURI:ERROR_CLEAR\] verbose clobbering success recovered$' "$_stdout_file"; then
    echo "  FAIL: verbose success inherited a status variable written by the wrapped function"
    exit 1
fi
echo "  PASS: verbose success records its status after the wrapped function returns"

_test_command() {
    return 7
}
_missing_status_parent=$(mktemp -d)
rmdir "$_missing_status_parent"
_missing_status_path="$_missing_status_parent/status"
set +e
(
    set -e
    mktemp() {
        printf '%s\n' "$_missing_status_path"
    }
    run_install_cmd "missing status file" _test_command
) >"$_stdout_file" 2>"$_stderr_file"
_exit_code=$?
set -e
if [ "$_exit_code" -ne 1 ] ||
    ! grep -q '^\[TAURI:ERROR_OUTPUT\] missing status file failed (exit code 1)$' "$_stdout_file"; then
    echo "  FAIL: missing verbose status defaulted to an empty or invalid exit code"
    exit 1
fi
echo "  PASS: missing verbose status defaults before reporting the failure"

_SETUP_FUNC_FILE=$(mktemp)
sed -n '/^setup_fail()/,/^}/p' "$SETUP_SH" > "$_SETUP_FUNC_FILE"
# shellcheck disable=SC1090
. "$_SETUP_FUNC_FILE"
rm -f "$_SETUP_FUNC_FILE"

set +e
(
    UNSLOTH_TAURI_MODE=1
    setup_fail 7 "specific setup failure"
) >"$_stdout_file" 2>"$_stderr_file"
_exit_code=$?
set -e
if [ "$_exit_code" -ne 7 ] ||
    ! grep -qxF '[TAURI:ERROR] specific setup failure' "$_stdout_file"; then
    echo "  FAIL: Tauri setup failure did not emit its explicit error and exit code"
    exit 1
fi

set +e
(
    UNSLOTH_TAURI_MODE=0
    setup_fail 7 "specific setup failure"
) >"$_stdout_file" 2>"$_stderr_file"
_exit_code=$?
set -e
if [ "$_exit_code" -ne 7 ] || [ -s "$_stdout_file" ] || [ -s "$_stderr_file" ]; then
    echo "  FAIL: non-Tauri setup failure emitted desktop protocol output"
    exit 1
fi
echo "  PASS: setup failures emit explicit context only in Tauri mode"

_setup_mode_count=$(grep -c 'UNSLOTH_TAURI_MODE="$TAURI_MODE"' "$INSTALL_SH")
if [ "$_setup_mode_count" -ne 2 ]; then
    echo "  FAIL: Unix installer does not pass Tauri mode to both setup invocations"
    exit 1
fi

# setup.sh has exactly two exits. setup_fail is the only failure exit: every failure path goes
# through it so the desktop gets a [TAURI:ERROR] line, not a bare exit code. The pinned-uv signal
# handler is not a failure, it re-raises as 128+signal like install.sh's _on_install_signal,
# because install.rs cancels an install by SIGTERMing the process group with intentional_stop and
# a cancel must not raise a failure banner. A third exit, or a second inside either, is unrouted.
# `|| true`: grep -c reports 0 with status 1, which under set -e would abort with no explanation.
_setup_fail_exits=$(sed -n '/^setup_fail()/,/^}/p' "$SETUP_SH" |
    grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)
_setup_signal_exits=$(sed -n '/^_setup_uv_on_signal()/,/^}/p' "$SETUP_SH" |
    grep -Ec '^[[:space:]]*exit[[:space:]]+' || true)
_setup_exit_count=$(grep -Ec '^[[:space:]]*exit[[:space:]]+' "$SETUP_SH" || true)
if [ "$_setup_fail_exits" -ne 1 ] ||
    [ "$_setup_signal_exits" -ne 1 ] ||
    [ "$_setup_exit_count" -ne $((_setup_fail_exits + _setup_signal_exits)) ] ||
    ! grep -q '^[[:space:]]*exit "\$exit_code"$' "$SETUP_SH" ||
    ! grep -q '^[[:space:]]*exit "\$1"$' "$SETUP_SH"; then
    echo "  FAIL: Unix setup has explicit exits outside setup_fail"
    exit 1
fi
# Trap-only: called from ordinary control flow the handler is the unrouted failure exit again.
_setup_signal_refs=$(grep -c '_setup_uv_on_signal' "$SETUP_SH" || true)
if [ "$_setup_signal_refs" -ne 4 ] ||
    ! grep -qF "trap '_setup_uv_on_signal 129' HUP" "$SETUP_SH" ||
    ! grep -qF "trap '_setup_uv_on_signal 130' INT" "$SETUP_SH" ||
    ! grep -qF "trap '_setup_uv_on_signal 143' TERM" "$SETUP_SH"; then
    echo "  FAIL: Unix setup signal handler is reachable outside its HUP/INT/TERM traps"
    exit 1
fi
echo "  PASS: Unix setup routes explicit exits through setup_fail"

# Prove the exception behaves as claimed: a stubbed pinned-uv install, interrupted for real in
# Tauri mode, must report 128+signal, leave no temporaries behind, and print no cancel failure.
_signal_dir=$(mktemp -d)
trap 'rm -f "$_stdout_file" "$_stderr_file"; rm -rf "$_signal_dir"' EXIT
{
    sed -n '/^_setup_uv_cleanup_temporaries()/,/^}/p' "$SETUP_SH"
    sed -n '/^_setup_uv_on_signal()/,/^}/p' "$SETUP_SH"
} > "$_signal_dir/uv_signal_fns.sh"
cat > "$_signal_dir/interrupted_install.sh" <<'SIGNAL_STUB'
# shellcheck disable=SC1090
. "$1/uv_signal_fns.sh"
_SIUP_WORK="$1/work"
_SIUP_STAGE="$1/dest/.uv.stage"
_SIUP_STAGE2="$1/dest/.uvx.stage"
trap _setup_uv_cleanup_temporaries EXIT
trap '_setup_uv_on_signal 129' HUP
trap '_setup_uv_on_signal 130' INT
trap '_setup_uv_on_signal 143' TERM
: > "$1/ready"
# Short sleeps: a trap runs only once the foreground command returns.
while :; do sleep 0.1; done
SIGNAL_STUB
mkdir -p "$_signal_dir/work/uv-x86_64-unknown-linux-gnu" "$_signal_dir/dest"
: > "$_signal_dir/work/uv-x86_64-unknown-linux-gnu/uv"
: > "$_signal_dir/dest/.uv.stage"
: > "$_signal_dir/dest/.uvx.stage"
set +e
UNSLOTH_TAURI_MODE=1 bash "$_signal_dir/interrupted_install.sh" "$_signal_dir" \
    >"$_stdout_file" 2>"$_stderr_file" &
_signal_pid=$!
_signal_waited=0
while [ ! -e "$_signal_dir/ready" ] && [ "$_signal_waited" -lt 100 ]; do
    command sleep 0.1
    _signal_waited=$((_signal_waited + 1))
done
kill -TERM "$_signal_pid" 2>/dev/null
wait "$_signal_pid"
_exit_code=$?
set -e
if [ "$_exit_code" -ne 143 ]; then
    echo "  FAIL: interrupted setup reported exit code $_exit_code instead of 143"
    exit 1
fi
if [ -d "$_signal_dir/work" ] ||
    [ -e "$_signal_dir/dest/.uv.stage" ] ||
    [ -e "$_signal_dir/dest/.uvx.stage" ]; then
    echo "  FAIL: interrupted setup left the pinned uv temporaries behind"
    exit 1
fi
if grep -q '^\[TAURI:' "$_stdout_file" || grep -q '^\[TAURI:' "$_stderr_file"; then
    echo "  FAIL: interrupted setup reported a cancel as an installer failure"
    exit 1
fi
echo "  PASS: interrupted setup cleans up and re-raises the signal without failure context"

_rollback_block=$(sed -n \
    '/^_restore_studio_venv_replacement()/,/^}/p' \
    "$INSTALL_SH")
_rollback_progress_count=$(printf '%s\n' "$_rollback_block" |
    grep -c 'rollback_substep')
if [ "$_rollback_progress_count" -ne 2 ]; then
    echo "  FAIL: successful Unix rollback output can replace failure context"
    exit 1
fi
echo "  PASS: successful Unix rollback remains structured progress"

_setup_success_block=$(sed -n \
    '/^if \[ "$_SETUP_EXIT" -eq 0 \]; then$/,/^mkdir -p "\$_LOCAL_BIN"$/p' \
    "$INSTALL_SH")
if ! printf '%s\n' "$_setup_success_block" |
    grep -q 'tauri_clear_install_error "studio setup completed"'; then
    echo "  FAIL: successful studio setup does not clear recovered setup errors before post-setup work"
    exit 1
fi

_setup_failure_block=$(sed -n \
    '/^# If setup.sh failed, report and exit now\.$/,/^fi$/p' \
    "$INSTALL_SH")
if ! printf '%s\n' "$_setup_failure_block" |
    grep -q 'tauri_log "ERROR_DEFAULT" "studio setup failed'; then
    echo "  FAIL: failed studio setup does not preserve output before its generic fallback"
    exit 1
fi
echo "  PASS: studio setup success clears recovered errors and failure preserves specific output"

_ps_setup_block=$(sed -n \
    '/if (\$setupExit -ne 0) {/,/# ── Expose `unsloth` via a shim dir/p' \
    "$INSTALL_PS1")
if ! printf '%s\n' "$_ps_setup_block" | grep -q 'Exit-InstallFailure' ||
    ! printf '%s\n' "$_ps_setup_block" |
        grep -q 'Clear-TauriInstallError "studio setup completed"'; then
    echo "  FAIL: Windows setup does not preserve failed output and clear successful output"
    exit 1
fi
echo "  PASS: Windows setup uses the same failure-context boundaries"

if ! grep -q '\$env:UNSLOTH_TAURI_MODE = if (\$TauriMode)' "$INSTALL_PS1"; then
    echo "  FAIL: Windows installer does not pass Tauri mode to setup"
    exit 1
fi

_ps_setup_exit_count=$(grep -Ec '^[[:space:]]*exit[[:space:]]+' "$SETUP_PS1")
if [ "$_ps_setup_exit_count" -ne 1 ] ||
    ! grep -q '^[[:space:]]*exit \$Code$' "$SETUP_PS1"; then
    echo "  FAIL: Windows setup has explicit exits outside Exit-SetupFailure"
    exit 1
fi
echo "  PASS: Windows setup routes explicit exits through Exit-SetupFailure"

_ps_command_block=$(sed -n \
    '/function Invoke-InstallCommand {/,/function New-StudioShortcuts {/p' \
    "$INSTALL_PS1")
if ! printf '%s\n' "$_ps_command_block" |
    grep -q 'Write-TauriLog "ERROR_OUTPUT" "$Label failed' ||
    ! printf '%s\n' "$_ps_command_block" |
        grep -q 'Write-TauriLog "OUTPUT_CLEAR" \$Label' ||
    ! printf '%s\n' "$_ps_command_block" |
        grep -q 'Clear-TauriInstallError "$Label recovered"' ||
    ! printf '%s\n' "$_ps_command_block" |
        grep -q 'Invoke-InstallCommand -Command \$Command -Label \$Label'; then
    echo "  FAIL: Windows command output is not attributed and cleared at the command boundary"
    exit 1
fi

_ps_exit_block=$(sed -n \
    '/function Exit-InstallFailure {/,/# ── Parse flags/p' \
    "$INSTALL_PS1")
if ! printf '%s\n' "$_ps_exit_block" |
    grep -q 'Write-TauriLog "ERROR_DEFAULT" \$Message'; then
    echo "  FAIL: Windows finalization can overwrite producer-owned failure context"
    exit 1
fi
echo "  PASS: Windows command failures preserve output through retries and finalization"
