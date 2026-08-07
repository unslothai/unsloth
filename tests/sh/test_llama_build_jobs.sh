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

# ── Container and slice limits ──
# /proc/meminfo is not namespaced, so it reports the host inside a container.
# Reading only the hierarchy root works in a container with a private cgroup
# namespace, but under Slurm, systemd or --cgroupns=host the binding limit is
# the process's own path or an ancestor's, and a root-only read sees "max".
_CG_FILE=$(mktemp)
for _fn in _cg_read _cg_limit _cg_dirs _cg_mount _cgroup_free_mb; do
    sed -n "/^${_fn}()/,/^}/p" "$SETUP_SH" >> "$_CG_FILE"
done
if [ ! -s "$_CG_FILE" ]; then
    echo "FAIL: could not extract the cgroup readers from setup.sh"
    exit 1
fi

_TMPD=$(mktemp -d)
# $1 = fallback root, $2 = /proc/self/cgroup stand-in, $3 = mountinfo stand-in.
# The mountinfo argument is always passed, and defaults to a path that does not
# exist, so a Linux runner's real mounts cannot leak into these fixtures.
run_cgroup() {
    bash -c ". '$_CG_FILE'; _cgroup_free_mb \"\$1\" \"\$2\" \"\$3\"" \
        _ "$1" "$2" "${3:-$_TMPD/no-mountinfo}"
}

echo "=== cgroup limits ==="

# 1) v2, limit on the process's own leaf: the container-with-cgroupns shape.
mkdir -p "$_TMPD/leaf/a"
printf 'max\n'        > "$_TMPD/leaf/memory.max"
printf '4294967296\n' > "$_TMPD/leaf/a/memory.max"
printf '0::/a\n'      > "$_TMPD/leaf.proc"
assert_eq "v2 leaf limit" "4096" "$(run_cgroup "$_TMPD/leaf" "$_TMPD/leaf.proc")"

# 2) v2, limit on an ancestor: the systemd slice and Slurm job-step shape. A
#    root-only reader saw "max" here and budgeted from the host.
mkdir -p "$_TMPD/anc/a/b"
printf 'max\n'        > "$_TMPD/anc/memory.max"
printf '4294967296\n' > "$_TMPD/anc/a/memory.max"
printf 'max\n'        > "$_TMPD/anc/a/b/memory.max"
printf '0::/a/b\n'    > "$_TMPD/anc.proc"
assert_eq "v2 ancestor limit" "4096" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"

# 3) Usage is subtracted, paired with the directory that set the limit.
printf '1073741824\n' > "$_TMPD/anc/a/memory.current"
assert_eq "usage subtracted from its own limit" "3072" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"

# 4) Over-usage floors at zero rather than going negative.
printf '8589934592\n' > "$_TMPD/anc/a/memory.current"
assert_eq "over-usage floors at 0" "0" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"
rm -f "$_TMPD/anc/a/memory.current"

# 5) memory.high binds as much as memory.max, and the smaller wins.
printf '2147483648\n' > "$_TMPD/anc/a/memory.high"
assert_eq "memory.high counts, smallest wins" "2048" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"
rm -f "$_TMPD/anc/a/memory.high"

# 6) v2 "max" everywhere is not a limit.
mkdir -p "$_TMPD/unl"
printf 'max\n'   > "$_TMPD/unl/memory.max"
printf '0::/\n'  > "$_TMPD/unl.proc"
assert_eq "v2 'max' is not a limit" "" "$(run_cgroup "$_TMPD/unl" "$_TMPD/unl.proc")"

# 7) v1 under <root>/memory, with its own controller column.
mkdir -p "$_TMPD/v1/memory/slice"
printf '2147483648\n' > "$_TMPD/v1/memory/slice/memory.limit_in_bytes"
printf '2:cpu,memory:/slice\n' > "$_TMPD/v1.proc"
assert_eq "v1 limit via its controller column" "2048" "$(run_cgroup "$_TMPD/v1" "$_TMPD/v1.proc")"

# 8) v1's unlimited sentinel is not a limit.
printf '9223372036854771712\n' > "$_TMPD/v1/memory/slice/memory.limit_in_bytes"
assert_eq "v1 sentinel is not a limit" "" "$(run_cgroup "$_TMPD/v1" "$_TMPD/v1.proc")"

# 9) No cgroupfs at all: nothing to report, and no error.
assert_eq "absent hierarchy reports nothing" "" "$(run_cgroup "$_TMPD/missing" "$_TMPD/missing.proc")"

# 10) v1 mounted somewhere other than <root>/memory. Without reading mountinfo
#     this branch was skipped and the budget silently fell back to host memory.
mkdir -p "$_TMPD/odd/mem-controller/slice"
printf '2147483648\n' > "$_TMPD/odd/mem-controller/slice/memory.limit_in_bytes"
printf '2:memory:/slice\n' > "$_TMPD/odd.proc"
printf '%s\n' "40 30 0:35 / $_TMPD/odd/mem-controller rw - cgroup cgroup rw,memory" > "$_TMPD/odd.mnt"
assert_eq "v1 found at a relocated mount" "2048" \
    "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc" "$_TMPD/odd.mnt")"
assert_eq "and missed without mountinfo" "" "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc")"

# 11) v1 co-mounted with other controllers: matched by super option, not name.
printf '%s\n' "41 30 0:36 / $_TMPD/odd/mem-controller rw - cgroup cgroup rw,cpu,memory,cpuacct" \
    > "$_TMPD/odd.co"
assert_eq "v1 found on a co-mounted hierarchy" "2048" \
    "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc" "$_TMPD/odd.co")"

# 12) A controller that is not memory must not be mistaken for it.
printf '%s\n' "42 30 0:37 / $_TMPD/odd/mem-controller rw - cgroup cgroup rw,cpu,cpuacct" \
    > "$_TMPD/odd.other"
assert_eq "a non-memory hierarchy is not used" "" \
    "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc" "$_TMPD/odd.other")"

# 13) v2 unified mounted away from the root, as under systemd hybrid mode.
mkdir -p "$_TMPD/hyb/unified/a"
printf 'max\n'        > "$_TMPD/hyb/unified/memory.max"
printf '1073741824\n' > "$_TMPD/hyb/unified/a/memory.max"
printf '0::/a\n'      > "$_TMPD/hyb.proc"
printf '%s\n' "43 30 0:38 / $_TMPD/hyb/unified rw - cgroup2 cgroup2 rw" > "$_TMPD/hyb.mnt"
assert_eq "v2 found at a relocated unified mount" "1024" \
    "$(run_cgroup "$_TMPD/hyb" "$_TMPD/hyb.proc" "$_TMPD/hyb.mnt")"

# The min() that ties it together. _usable_ram_mb hardcodes the real paths, so
# stub the reader to prove the wiring rather than just the parts.
_RAM_FILE=$(mktemp)
sed -n '/^_usable_ram_mb()/,/^}/p' "$SETUP_SH" > "$_RAM_FILE"
run_usable_ram() {
    FAKE_FREE="$1" bash -c \
        '. "$1"; _cgroup_free_mb() { printf "%s" "$FAKE_FREE"; }; _usable_ram_mb' _ "$_RAM_FILE"
}
_HOST_MB=$(run_usable_ram "")

assert_eq "a lower cgroup allowance wins" "512" "$(run_usable_ram 512)"
assert_eq "a higher allowance is ignored" "$_HOST_MB" "$(run_usable_ram 8796093022207)"
assert_eq "no cgroup leaves host memory" "$_HOST_MB" "$(run_usable_ram "")"

# End to end: a 4 GB allowance builds at -j1, not -j(cores).
assert_eq "4 GB allowance floors at 1 job" "1" "$(run_jobs 20 "$(run_usable_ram 4096)" "")"

# Host memory comes from what is actually available, not what is installed: a
# 16 GiB box with 8 GiB resident must not be handed a 14 GiB compile budget.
# Match the awk field pattern, not the bare word, so prose about MemAvailable
# cannot satisfy this on its own.
if grep -q '/\^MemAvailable:/' "$SETUP_SH"; then
    echo "  PASS: host memory reads MemAvailable"; PASS=$((PASS + 1))
else
    echo "  FAIL: host memory still reads MemTotal only"; FAIL=$((FAIL + 1))
fi
# MemTotal stays as the pre-3.14 fallback, and must come after MemAvailable.
_AVAIL_LINE=$(grep -n '/\^MemAvailable:/' "$SETUP_SH" | head -1 | cut -d: -f1)
_TOTAL_LINE=$(grep -n '/\^MemTotal:/' "$SETUP_SH" | head -1 | cut -d: -f1)
if [ -n "$_AVAIL_LINE" ] && [ -n "$_TOTAL_LINE" ] && [ "$_AVAIL_LINE" -lt "$_TOTAL_LINE" ]; then
    echo "  PASS: MemTotal kept as the fallback, after MemAvailable"; PASS=$((PASS + 1))
else
    echo "  FAIL: MemTotal must remain, and only as the fallback"; FAIL=$((FAIL + 1))
fi

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
# Windows has no cgroups, but it has the same installed-versus-available gap.
# AvailableMBytes counts the standby list; the Free counters do not.
_check_ps1 "setup.ps1 budgets from available memory" 'AvailableMBytes'
_check_ps1 "setup.ps1 feeds it to the job count" 'Get-LlamaJobsFor .*-TotalMb \(Get-UsableMemoryMb\)'
_check_ps1 "setup.ps1 keeps installed RAM as the fallback" 'TotalPhysicalMemory'

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
