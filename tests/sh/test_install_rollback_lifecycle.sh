#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Exercises install.sh's real rollback helpers without downloading the Studio stack.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
PASS=0
FAIL=0

ok()  { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

ROLLBACK_BLOCK=$(sed -n '/^_VENV_ROLLBACK_DIR=""/,/^trap '\''_on_install_signal 143'\'' TERM$/p' "$INSTALL_SH")
if ! printf '%s\n' "$ROLLBACK_BLOCK" | grep -q '^_on_install_signal() {'; then
    echo "  FAIL: could not extract rollback lifecycle block from install.sh"
    exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

run_signal_case() {
    _signal="$1"
    _expected_status="$2"
    _case_dir="$WORK/signal-$_signal"
    mkdir -p "$_case_dir/unsloth_studio"
    printf 'old\n' > "$_case_dir/unsloth_studio/generation"
    _harness="$_case_dir/harness.sh"
    {
        printf '%s\n' 'set -e'
        printf '%s\n' 'substep() { :; }'
        printf '%s\n' 'C_WARN=""'
        printf "STUDIO_HOME='%s'\n" "$_case_dir"
        printf "VENV_DIR='%s/unsloth_studio'\n" "$_case_dir"
        printf '%s\n' "$ROLLBACK_BLOCK"
        printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
        printf '%s\n' 'mkdir -p "$VENV_DIR"'
        printf '%s\n' 'printf "partial\n" > "$VENV_DIR/generation"'
        printf 'kill -%s $$\n' "$_signal"
        printf '%s\n' 'exit 99'
    } > "$_harness"

    set +e
    dash "$_harness" >/dev/null 2>&1
    _status=$?
    set -e
    if [ "$_status" = "$_expected_status" ]; then
        ok "dash $_signal exits with $_expected_status"
    else
        bad "dash $_signal exits with $_expected_status (got $_status)"
    fi
    if [ "$(cat "$_case_dir/unsloth_studio/generation" 2>/dev/null)" = "old" ]; then
        ok "dash $_signal restores the previous environment"
    else
        bad "dash $_signal did not restore the previous environment"
    fi
    if ! find "$_case_dir" -maxdepth 1 -name 'unsloth_studio.rollback.*' -print -quit | grep -q .; then
        ok "dash $_signal leaves no rollback copy"
    else
        bad "dash $_signal left a rollback copy"
    fi
}

echo "=== install.sh signal rollback ==="
run_signal_case INT 130
run_signal_case TERM 143
run_signal_case HUP 129

echo "=== install.sh transition boundaries ==="
START_BOUNDARY_DIR="$WORK/start-boundary"
mkdir -p "$START_BOUNDARY_DIR/unsloth_studio"
printf 'old\n' > "$START_BOUNDARY_DIR/unsloth_studio/generation"
START_BOUNDARY_HARNESS="$START_BOUNDARY_DIR/harness.sh"
{
    printf '%s\n' 'set -e'
    printf '%s\n' 'substep() { :; }'
    printf '%s\n' 'C_WARN=""'
    printf "STUDIO_HOME='%s'\n" "$START_BOUNDARY_DIR"
    printf "VENV_DIR='%s/unsloth_studio'\n" "$START_BOUNDARY_DIR"
    printf '%s\n' "$ROLLBACK_BLOCK"
    printf '%s\n' 'mv() { command mv "$@"; kill -TERM $$; }'
    printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
} > "$START_BOUNDARY_HARNESS"
set +e
dash "$START_BOUNDARY_HARNESS" >/dev/null 2>&1
_start_boundary_status=$?
set -e
if [ "$_start_boundary_status" -eq 143 ] \
   && [ "$(cat "$START_BOUNDARY_DIR/unsloth_studio/generation" 2>/dev/null)" = "old" ]; then
    ok "signal immediately after rollback rename restores the old environment"
else
    bad "rollback state was not published before rename"
fi

COMMIT_BOUNDARY_DIR="$WORK/commit-boundary"
mkdir -p "$COMMIT_BOUNDARY_DIR/unsloth_studio"
printf 'old\n' > "$COMMIT_BOUNDARY_DIR/unsloth_studio/generation"
COMMIT_BOUNDARY_HARNESS="$COMMIT_BOUNDARY_DIR/harness.sh"
{
    printf '%s\n' 'set -e'
    printf '%s\n' 'substep() { :; }'
    printf '%s\n' 'C_WARN=""'
    printf "STUDIO_HOME='%s'\n" "$COMMIT_BOUNDARY_DIR"
    printf "VENV_DIR='%s/unsloth_studio'\n" "$COMMIT_BOUNDARY_DIR"
    printf '%s\n' "$ROLLBACK_BLOCK"
    printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
    printf '%s\n' 'mkdir -p "$VENV_DIR"'
    printf '%s\n' 'printf "new\n" > "$VENV_DIR/generation"'
    printf '%s\n' 'rm() { kill -TERM $$; }'
    printf '%s\n' '_commit_studio_venv_replacement'
} > "$COMMIT_BOUNDARY_HARNESS"
set +e
dash "$COMMIT_BOUNDARY_HARNESS" >/dev/null 2>&1
_commit_boundary_status=$?
set -e
if [ "$_commit_boundary_status" -eq 143 ] \
   && [ "$(cat "$COMMIT_BOUNDARY_DIR/unsloth_studio/generation" 2>/dev/null)" = "new" ]; then
    ok "signal during committed-backup deletion keeps the new environment"
else
    bad "signal during committed-backup deletion restored a partial backup"
fi

echo "=== install.sh successful cleanup ==="
PRUNE_DIR="$WORK/prune"
mkdir -p "$PRUNE_DIR/unsloth_studio"
printf 'old\n' > "$PRUNE_DIR/unsloth_studio/generation"
PRUNE_HARNESS="$PRUNE_DIR/harness.sh"
{
    printf '%s\n' 'set -e'
    printf '%s\n' 'substep() { :; }'
    printf '%s\n' 'C_WARN=""'
    printf "STUDIO_HOME='%s'\n" "$PRUNE_DIR"
    printf "VENV_DIR='%s/unsloth_studio'\n" "$PRUNE_DIR"
    printf '%s\n' "$ROLLBACK_BLOCK"
    printf '%s\n' '_start_studio_venv_replacement "$VENV_DIR"'
    printf '%s\n' 'mkdir -p "$VENV_DIR"'
    printf '%s\n' 'printf "new\n" > "$VENV_DIR/generation"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/unsloth_studio.rollback.20000101000000.999999999"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/unsloth_studio.rollback.20000101000001.$$"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/unsloth_studio.rollback.user-data"'
    printf '%s\n' 'mkdir "$STUDIO_HOME/outside"'
    printf '%s\n' 'ln -s "$STUDIO_HOME/outside" "$STUDIO_HOME/unsloth_studio.rollback.20000101000002.999999998"'
    printf '%s\n' '_commit_studio_venv_replacement'
} > "$PRUNE_HARNESS"

sh "$PRUNE_HARNESS" >/dev/null 2>&1
if [ "$(cat "$PRUNE_DIR/unsloth_studio/generation" 2>/dev/null)" = "new" ]; then
    ok "successful replacement keeps the new environment"
else
    bad "successful replacement lost the new environment"
fi
if [ ! -d "$PRUNE_DIR/unsloth_studio.rollback.20000101000000.999999999" ]; then
    ok "successful install removes an orphan from a dead PID"
else
    bad "successful install left an orphan from a dead PID"
fi
_active_count=$(find "$PRUNE_DIR" -maxdepth 1 -type d -name 'unsloth_studio.rollback.20000101000001.*' | wc -l)
if [ "$_active_count" -eq 1 ]; then
    ok "successful install preserves a concurrent installer's rollback"
else
    bad "successful install removed a concurrent installer's rollback"
fi
if [ -d "$PRUNE_DIR/unsloth_studio.rollback.user-data" ]; then
    ok "stale cleanup preserves names outside the generated format"
else
    bad "stale cleanup removed a non-generated rollback name"
fi
if [ -L "$PRUNE_DIR/unsloth_studio.rollback.20000101000002.999999998" ] \
   && [ -d "$PRUNE_DIR/outside" ]; then
    ok "stale cleanup does not follow rollback symlinks"
else
    bad "stale cleanup mutated a rollback symlink target"
fi

echo "=== install.ps1 rollback wiring ==="
if grep -q '^    function Remove-StaleStudioVenvRollbacks {' "$INSTALL_PS1" \
   && grep -q '^    Remove-StaleStudioVenvRollbacks$' "$INSTALL_PS1"; then
    ok "Windows installer prunes stale rollbacks after success"
else
    bad "Windows installer does not wire stale rollback cleanup"
fi
if grep -q '^    } finally {$' "$INSTALL_PS1" \
   && grep -A3 '^    } finally {$' "$INSTALL_PS1" | grep -q 'Restore-StudioVenvRollback'; then
    ok "Windows replacement is protected by finally"
else
    bad "Windows replacement lacks finally rollback"
fi
if grep -A18 '^    function Remove-StudioVenvTreeWithRetry {' "$INSTALL_PS1" \
   | grep -q 'ErrorAction Stop'; then
    ok "Windows rollback deletion failures are observable and retried"
else
    bad "Windows rollback deletion still hides failures"
fi

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "ALL PASSED"
