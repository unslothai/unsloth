#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _llama_jobs_for() from studio/setup.sh: the cmake -j count.
#
# A 20-thread 16 GB box used to build llama.cpp at -j20. Each nvcc job peaks near
# 2 GB, so that oversubscribed RAM until the machine stopped responding. The job
# count is now the smaller of the core count and what RAM can hold.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
PASS=0
FAIL=0

# Extract the constants and the helper (same sed range as the other function tests).
_FUNC_FILE=$(mktemp)
sed -n '/^_LLAMA_BUILD_RESERVE_MB=/,/^_LLAMA_BUILD_MB_PER_JOB=/p' "$SETUP_SH" > "$_FUNC_FILE"
sed -n '/^_llama_jobs_for()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"
if [ ! -s "$_FUNC_FILE" ]; then
    echo "FAIL: could not extract _llama_jobs_for from setup.sh"
    exit 1
fi

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

# $1 = cores, $2 = total RAM MiB, $3 = UNSLOTH_LLAMA_BUILD_JOBS
run_jobs() {
    UNSLOTH_LLAMA_BUILD_JOBS="${3:-}" \
        bash -c ". '$_FUNC_FILE'; _llama_jobs_for \"\$1\" \"\$2\"" _ "$1" "$2"
}

echo "=== test_llama_build_jobs ==="

# 1) The reported case: 20 threads, 16 GB. (16384 - 2048) / 2048 = 7.
assert_eq "20 cores / 16 GB caps at 7" "7" "$(run_jobs 20 16384 "")"

# 2) RAM is only a ceiling, never a floor: a 4-core 64 GB box still gets 4.
assert_eq "cores win when RAM is ample" "4" "$(run_jobs 4 65536 "")"

# 3) Small-RAM machines still make progress rather than dividing to zero.
assert_eq "4 GB floors at 1" "1" "$(run_jobs 8 4096 "")"
assert_eq "2 GB floors at 1" "1" "$(run_jobs 8 2048 "")"

# 4) Unreadable RAM keeps the previous behaviour instead of guessing low.
assert_eq "empty RAM keeps cores" "20" "$(run_jobs 20 "" "")"
assert_eq "garbage RAM keeps cores" "20" "$(run_jobs 20 "unknown" "")"

# 5) Unreadable core count falls back to 4, as the old nproc chain did.
assert_eq "garbage cores default to 4" "4" "$(run_jobs "" 65536 "")"

# 6) The override wins over both, so a large-RAM builder can opt out.
assert_eq "override wins over cap" "32" "$(run_jobs 20 16384 32)"
assert_eq "override wins over cores" "64" "$(run_jobs 4 4096 64)"

# 7) A junk or zero override is ignored rather than producing -j0.
assert_eq "zero override ignored" "7" "$(run_jobs 20 16384 0)"
assert_eq "junk override ignored" "7" "$(run_jobs 20 16384 "lots")"

# ── Windows parity (static check) ──
# setup.ps1 carries its own copy because it cannot source setup.sh. The user who
# reported this was on Windows, so a cap that only lands on Unix fixes nothing.
_check_ps1() {
    _label="$1"; _pattern="$2"
    if grep -qE "$_pattern" "$SETUP_PS1"; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (no match for '$_pattern' in setup.ps1)"; FAIL=$((FAIL + 1))
    fi
}

echo "=== setup.ps1 parity ==="
_check_ps1 "setup.ps1 caps the job count" '^[[:space:]]*\$NumCpu = Get-LlamaBuildJobs$'
_check_ps1 "setup.ps1 honours the override" 'UNSLOTH_LLAMA_BUILD_JOBS'
_check_ps1 "setup.ps1 reserve matches setup.sh" '^\$LlamaBuildReserveMb = 2048$'
_check_ps1 "setup.ps1 per-job budget matches setup.sh" '^\$LlamaBuildMbPerJob = 2048$'

# The bare ProcessorCount is what caused the hang; it must not come back.
if grep -qE '^\s*\$NumCpu = \[Environment\]::ProcessorCount' "$SETUP_PS1"; then
    echo "  FAIL: setup.ps1 still sets -j from the raw core count"; FAIL=$((FAIL + 1))
else
    echo "  PASS: setup.ps1 no longer sets -j from the raw core count"; PASS=$((PASS + 1))
fi

rm -f "$_FUNC_FILE"
echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
