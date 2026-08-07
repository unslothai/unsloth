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

# ── Container limits ──
# /proc/meminfo is not namespaced, so it reports the host's RAM inside a
# container. Without this a 4 GB container on a big host budgets from the host
# and the build gets OOM-killed.
_CG_FILE=$(mktemp)
sed -n '/^_cgroup_limit_mb()/,/^}/p' "$SETUP_SH" > "$_CG_FILE"
if [ ! -s "$_CG_FILE" ]; then
    echo "FAIL: could not extract _cgroup_limit_mb from setup.sh"
    exit 1
fi

_TMPD=$(mktemp -d)
run_cgroup() {
    bash -c ". '$_CG_FILE'; _cgroup_limit_mb \"\$@\"" _ "$@"
}

echo "=== cgroup limit ==="

# 1) cgroup v2 in a 4 GB container.
printf '4294967296' > "$_TMPD/v2"
assert_eq "v2 limit read as MiB" "4096" "$(run_cgroup "$_TMPD/v2")"

# 2) v2 on an unconstrained host writes the literal "max".
printf 'max' > "$_TMPD/v2max"
assert_eq "v2 'max' is not a limit" "" "$(run_cgroup "$_TMPD/v2max")"

# 3) v1 in a 2 GB container, reached only when v2 is absent.
printf '2147483648' > "$_TMPD/v1"
assert_eq "v1 limit read as MiB" "2048" "$(run_cgroup "$_TMPD/missing" "$_TMPD/v1")"

# 4) v1's unlimited sentinel is enormous, so it loses the min() below.
printf '9223372036854771712' > "$_TMPD/v1max"
assert_eq "v1 sentinel is astronomically large" "8796093022207" "$(run_cgroup "$_TMPD/v1max")"

# 5) No cgroup files at all: nothing to report.
assert_eq "absent files report nothing" "" "$(run_cgroup "$_TMPD/missing" "$_TMPD/also-missing")"

# 6) v2 wins when both exist, since it is passed first.
assert_eq "v2 wins over v1" "4096" "$(run_cgroup "$_TMPD/v2" "$_TMPD/v1")"

# The min() that ties it together. _total_ram_mb hardcodes the real cgroup paths,
# so stub the reader to prove the wiring rather than just the parts.
_RAM_FILE=$(mktemp)
sed -n '/^_total_ram_mb()/,/^}/p' "$SETUP_SH" > "$_RAM_FILE"
# The stub ignores the real paths _total_ram_mb passes it and echoes $FAKE_LIMIT.
run_total_ram() {
    FAKE_LIMIT="$1" bash -c \
        '. "$1"; _cgroup_limit_mb() { printf "%s" "$FAKE_LIMIT"; }; _total_ram_mb' _ "$_RAM_FILE"
}
_HOST_MB=$(run_total_ram "")

# A limit below host RAM wins; one above it, including v1's sentinel, does not.
assert_eq "a lower cgroup limit wins" "512" "$(run_total_ram 512)"
assert_eq "a higher limit is ignored" "$_HOST_MB" "$(run_total_ram 8796093022207)"
assert_eq "no limit leaves host RAM" "$_HOST_MB" "$(run_total_ram "")"
assert_eq "a zero limit is ignored" "$_HOST_MB" "$(run_total_ram 0)"

# End to end: a 4 GB container on this host builds at -j1, not -j(cores).
assert_eq "4 GB container floors at 1 job" "1" "$(run_jobs 20 "$(run_total_ram 4096)" "")"

rm -rf "$_TMPD" "$_CG_FILE" "$_RAM_FILE"

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
