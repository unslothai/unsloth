#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# setup.sh's final llama.cpp verdict: report in Tauri mode, still fail elsewhere.
#
# The trailing block is top-level code, not a function, so it is extracted and run with
# setup_fail stubbed. That keeps the contract pinned without executing an install.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_BLOCK=$(mktemp)
sed -n '/^if \[ "\$_LLAMA_CPP_DEGRADED" = true \] && \[ "\${SKIP_STUDIO_BASE:-0}" = "1" \]; then/,/^fi$/p' \
    "$SETUP_SH" > "$_BLOCK"
if [ ! -s "$_BLOCK" ]; then
    echo "FAIL: could not find the llama.cpp degraded block in setup.sh"
    exit 1
fi

# degraded, skip_base, tauri_mode -> expected rc, expected substring ("" means no output)
run_case() {
    _label="$1"; _degraded="$2"; _skip="$3"; _tauri="$4"; _rc_want="$5"; _want="$6"
    set +e
    _out=$(
        _LLAMA_CPP_DEGRADED="$_degraded" \
        SKIP_STUDIO_BASE="$_skip" \
        UNSLOTH_TAURI_MODE="$_tauri" \
        bash -c 'setup_fail() { printf "SETUP_FAIL: %s\n" "$*"; exit "$1"; }; . "$1"' _ "$_BLOCK" 2>&1
    )
    _rc=$?
    set -e
    if [ "$_rc" != "$_rc_want" ]; then
        echo "  FAIL: $_label (expected rc $_rc_want, got $_rc; output: $_out)"; FAIL=$((FAIL + 1)); return
    fi
    case "$_want" in
        "") [ -z "$_out" ] || { echo "  FAIL: $_label (expected no output, got: $_out)"; FAIL=$((FAIL + 1)); return; } ;;
        *)  case "$_out" in *"$_want"*) ;; *) echo "  FAIL: $_label (expected '$_want' in: $_out)"; FAIL=$((FAIL + 1)); return ;; esac ;;
    esac
    echo "  PASS: $_label"; PASS=$((PASS + 1))
}

echo "=== llama.cpp degraded verdict ==="
# The fix: a transient prebuilt failure must not abort the desktop first-launch install.
run_case "tauri mode reports instead of aborting" true 1 1 0 "[TAURI:PROGRESS]"
run_case "tauri mode names the recovery command" true 1 true 0 "unsloth studio update"
# Progress detail is cleared by the next install-step and discarded when the
# install screen closes, so the verdict also has to reach the support report.
run_case "tauri mode records the verdict"        true 1 1 0 "[TAURI:DIAG] llama_cpp=unavailable"
# The half that must not regress: install.sh still needs the non-zero exit.
run_case "shell install still fails"             true 1 0 1 "SETUP_FAIL"
run_case "unset tauri mode still fails"          true 1 "" 1 "SETUP_FAIL"
# Untouched paths.
run_case "direct 'studio update' stays silent"   true 0 1 0 ""
run_case "a healthy llama.cpp says nothing"      false 1 1 0 ""

# [TAURI:STEP] would be counted by use-tauri-backend.ts against the seven-entry
# INSTALL_STEPS list install.sh already fills, rendering "Step 8 of 7" and throwing
# the text away. The notice has to ride the progress-detail channel instead.
_step_out=$(
    _LLAMA_CPP_DEGRADED=true SKIP_STUDIO_BASE=1 UNSLOTH_TAURI_MODE=1 \
    bash -c 'setup_fail() { exit "$1"; }; . "$1"' _ "$_BLOCK" 2>&1
)
case "$_step_out" in
    *"[TAURI:STEP]"*)
        echo "  FAIL: notice must not use [TAURI:STEP] (got: $_step_out)"; FAIL=$((FAIL + 1)) ;;
    *)
        echo "  PASS: notice does not consume an install step"; PASS=$((PASS + 1)) ;;
esac

# ── Windows parity: setup.ps1 must degrade the same way (static check) ──
# install.ps1 turns any non-zero setup.ps1 status into Exit-InstallFailure, which
# install.rs reports as "Installation failed", so an unconditional
# Exit-SetupFailure here aborts the Windows first-launch install for the same
# transient download failure this test pins as survivable on Unix.
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
_PS_BLOCK=$(sed -n '/^if (\$script:LlamaCppDegraded -and \$env:SKIP_STUDIO_BASE -eq "1") {/,/^}$/p' "$SETUP_PS1")
if [ -z "$_PS_BLOCK" ]; then
    echo "  FAIL: could not find the llama.cpp degraded block in setup.ps1"; FAIL=$((FAIL + 1))
else
    case "$_PS_BLOCK" in
        *'UNSLOTH_TAURI_MODE'*'[TAURI:PROGRESS]'*'Exit-SetupFailure'*)
            echo "  PASS: setup.ps1 degrades in Tauri mode and still fails elsewhere"; PASS=$((PASS + 1)) ;;
        *)
            echo "  FAIL: setup.ps1 does not mirror the Tauri degraded handling"; FAIL=$((FAIL + 1)) ;;
    esac
    case "$_PS_BLOCK" in
        *'[TAURI:DIAG] llama_cpp=unavailable'*)
            echo "  PASS: setup.ps1 records the verdict too"; PASS=$((PASS + 1)) ;;
        *)
            echo "  FAIL: setup.ps1 does not record the degraded verdict"; FAIL=$((FAIL + 1)) ;;
    esac
fi

rm -f "$_BLOCK"
echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
