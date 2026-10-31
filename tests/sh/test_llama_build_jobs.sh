#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _llama_jobs_for() from studio/setup.sh: the cmake -j count is min(cores, what RAM holds);
# see _LLAMA_BUILD_* in setup.sh for the per-job budget.
set -e

# A developer or CI shell may export the override; only the override assertions may see it.
unset UNSLOTH_LLAMA_BUILD_JOBS

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
_FUNC_FILE=$(mktemp)
sed -n '/^_LLAMA_BUILD_RESERVE_MB=/,/^_LLAMA_BUILD_MB_PER_JOB=/p' "$SETUP_SH" > "$_FUNC_FILE"
sed -n '/^_llama_jobs_for()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"
if [ ! -s "$_FUNC_FILE" ]; then
    echo "FAIL: could not extract _llama_jobs_for from setup.sh"
    exit 1
fi

# $1 = cores, $2 = total RAM MiB, $3 = UNSLOTH_LLAMA_BUILD_JOBS
run_jobs() {
    UNSLOTH_LLAMA_BUILD_JOBS="${3:-}" \
        bash -c ". '$_FUNC_FILE'; _llama_jobs_for \"\$1\" \"\$2\"" _ "$1" "$2"
}

echo "=== test_llama_build_jobs ==="

# (16384 - 2048) / 2048 = 7.
assert_eq "20 cores / 16 GB caps at 7" "7" "$(run_jobs 20 16384 "")"

assert_eq "cores win when RAM is ample" "4" "$(run_jobs 4 65536 "")"

assert_eq "4 GB floors at 1" "1" "$(run_jobs 8 4096 "")"
assert_eq "2 GB floors at 1" "1" "$(run_jobs 8 2048 "")"

assert_eq "empty RAM keeps cores" "20" "$(run_jobs 20 "" "")"
assert_eq "garbage RAM keeps cores" "20" "$(run_jobs 20 "unknown" "")"

assert_eq "garbage cores default to 4" "4" "$(run_jobs "" 65536 "")"

assert_eq "override wins over cap" "32" "$(run_jobs 20 16384 32)"
assert_eq "override wins over cores" "64" "$(run_jobs 4 4096 64)"

assert_eq "zero override ignored" "7" "$(run_jobs 20 16384 0)"
assert_eq "junk override ignored" "7" "$(run_jobs 20 16384 "lots")"

# /proc/meminfo is not namespaced. Under Slurm, systemd or --cgroupns=host the binding limit
# is on the process's own path or an ancestor, so a root-only read sees "max".
_CG_FILE=$(mktemp)
for _fn in _cg_read _cg_limit _cg_dirs _cg_unesc_prog _cg_unesc _cg_mounts _cg_rel \
           _cg_pick_mounts _cgroup_free_mb; do
    sed -n "/^${_fn}()/,/^}/p" "$SETUP_SH" >> "$_CG_FILE"
done
if [ ! -s "$_CG_FILE" ]; then
    echo "FAIL: could not extract the cgroup readers from setup.sh"
    exit 1
fi

_TMPD=$(mktemp -d)
# $1 fallback root, $2 /proc/self/cgroup stand-in, $3 mountinfo stand-in (default a missing
# path so a runner's real mounts cannot leak in).
run_cgroup() {
    bash -c ". '$_CG_FILE'; _cgroup_free_mb \"\$1\" \"\$2\" \"\$3\"" \
        _ "$1" "$2" "${3:-$_TMPD/no-mountinfo}"
}

echo "=== cgroup limits ==="

mkdir -p "$_TMPD/leaf/a"
printf 'max\n'        > "$_TMPD/leaf/memory.max"
printf '4294967296\n' > "$_TMPD/leaf/a/memory.max"
printf '0::/a\n'      > "$_TMPD/leaf.proc"
assert_eq "v2 leaf limit" "4096" "$(run_cgroup "$_TMPD/leaf" "$_TMPD/leaf.proc")"

mkdir -p "$_TMPD/anc/a/b"
printf 'max\n'        > "$_TMPD/anc/memory.max"
printf '4294967296\n' > "$_TMPD/anc/a/memory.max"
printf 'max\n'        > "$_TMPD/anc/a/b/memory.max"
printf '0::/a/b\n'    > "$_TMPD/anc.proc"
assert_eq "v2 ancestor limit" "4096" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"

printf '1073741824\n' > "$_TMPD/anc/a/memory.current"
assert_eq "usage subtracted from its own limit" "3072" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"

printf '8589934592\n' > "$_TMPD/anc/a/memory.current"
assert_eq "over-usage floors at 0" "0" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"
rm -f "$_TMPD/anc/a/memory.current"

printf '2147483648\n' > "$_TMPD/anc/a/memory.high"
assert_eq "memory.high counts, smallest wins" "2048" "$(run_cgroup "$_TMPD/anc" "$_TMPD/anc.proc")"
rm -f "$_TMPD/anc/a/memory.high"

# A zero memory.high is a limit (it throttles, not kills). memory.max of 0 OOM-kills, so
# only memory.high is exercised.
mkdir -p "$_TMPD/zh/leaf"
printf 'max\n' > "$_TMPD/zh/memory.max"
printf 'max\n' > "$_TMPD/zh/leaf/memory.max"
printf '0\n'   > "$_TMPD/zh/leaf/memory.high"
printf '0::/leaf\n' > "$_TMPD/zh.proc"
assert_eq "a zero memory.high is a limit, not the absence of one" "0" \
    "$(run_cgroup "$_TMPD/zh" "$_TMPD/zh.proc")"
assert_eq "and a zero allowance builds at 1 job, not \$(nproc)" "1" \
    "$(run_jobs 20 "$(run_cgroup "$_TMPD/zh" "$_TMPD/zh.proc")" "")"

mkdir -p "$_TMPD/unl"
printf 'max\n'   > "$_TMPD/unl/memory.max"
printf '0::/\n'  > "$_TMPD/unl.proc"
assert_eq "v2 'max' is not a limit" "" "$(run_cgroup "$_TMPD/unl" "$_TMPD/unl.proc")"

mkdir -p "$_TMPD/v1/memory/slice"
printf '2147483648\n' > "$_TMPD/v1/memory/slice/memory.limit_in_bytes"
printf '2:cpu,memory:/slice\n' > "$_TMPD/v1.proc"
assert_eq "v1 limit via its controller column" "2048" "$(run_cgroup "$_TMPD/v1" "$_TMPD/v1.proc")"

printf '9223372036854771712\n' > "$_TMPD/v1/memory/slice/memory.limit_in_bytes"
assert_eq "v1 sentinel is not a limit" "" "$(run_cgroup "$_TMPD/v1" "$_TMPD/v1.proc")"

assert_eq "absent hierarchy reports nothing" "" "$(run_cgroup "$_TMPD/missing" "$_TMPD/missing.proc")"

# v1 mounted away from <root>/memory needs mountinfo, else host memory is used.
mkdir -p "$_TMPD/odd/mem-controller/slice"
printf '2147483648\n' > "$_TMPD/odd/mem-controller/slice/memory.limit_in_bytes"
printf '2:memory:/slice\n' > "$_TMPD/odd.proc"
printf '%s\n' "40 30 0:35 / $_TMPD/odd/mem-controller rw - cgroup cgroup rw,memory" > "$_TMPD/odd.mnt"
assert_eq "v1 found at a relocated mount" "2048" \
    "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc" "$_TMPD/odd.mnt")"
assert_eq "and missed without mountinfo" "" "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc")"

printf '%s\n' "41 30 0:36 / $_TMPD/odd/mem-controller rw - cgroup cgroup rw,cpu,memory,cpuacct" \
    > "$_TMPD/odd.co"
assert_eq "v1 found on a co-mounted hierarchy" "2048" \
    "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc" "$_TMPD/odd.co")"

printf '%s\n' "42 30 0:37 / $_TMPD/odd/mem-controller rw - cgroup cgroup rw,cpu,cpuacct" \
    > "$_TMPD/odd.other"
assert_eq "a non-memory hierarchy is not used" "" \
    "$(run_cgroup "$_TMPD/odd" "$_TMPD/odd.proc" "$_TMPD/odd.other")"

mkdir -p "$_TMPD/hyb/unified/a"
printf 'max\n'        > "$_TMPD/hyb/unified/memory.max"
printf '1073741824\n' > "$_TMPD/hyb/unified/a/memory.max"
printf '0::/a\n'      > "$_TMPD/hyb.proc"
printf '%s\n' "43 30 0:38 / $_TMPD/hyb/unified rw - cgroup2 cgroup2 rw" > "$_TMPD/hyb.mnt"
assert_eq "v2 found at a relocated unified mount" "1024" \
    "$(run_cgroup "$_TMPD/hyb" "$_TMPD/hyb.proc" "$_TMPD/hyb.mnt")"

# Bind-mounted subtree: /proc/self/cgroup is host-absolute, so it must be mapped through the
# mount root to find the job's limit, not the outer slice's.
mkdir -p "$_TMPD/bind/cg/job"
printf '8589934592\n' > "$_TMPD/bind/cg/memory.max"
printf '1073741824\n' > "$_TMPD/bind/cg/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/bind.proc"
printf '%s\n' "50 30 0:40 /slice $_TMPD/bind/cg rw - cgroup2 cgroup2 rw" > "$_TMPD/bind.mnt"
assert_eq "bind mount finds the innermost limit" "1024" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bind.proc" "$_TMPD/bind.mnt")"

printf '%s\n' "50 30 0:40 / $_TMPD/bind/cg rw - cgroup2 cgroup2 rw" > "$_TMPD/bind.slash"
assert_eq "an unmapped join settles for the outer limit" "8192" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bind.proc" "$_TMPD/bind.slash")"

printf '0::/elsewhere/job\n' > "$_TMPD/bind.out"
assert_eq "a path outside the mount root yields the mount's own limit" "8192" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bind.out" "$_TMPD/bind.mnt")"

mkdir -p "$_TMPD/bind/v1/task"
printf '4294967296\n' > "$_TMPD/bind/v1/memory.limit_in_bytes"
printf '2147483648\n' > "$_TMPD/bind/v1/task/memory.limit_in_bytes"
printf '3:memory:/docker/abc/task\n' > "$_TMPD/bindv1.proc"
printf '%s\n' "51 30 0:41 /docker/abc $_TMPD/bind/v1 rw - cgroup cgroup rw,memory" \
    > "$_TMPD/bindv1.mnt"
assert_eq "v1 bind mount finds the innermost limit" "2048" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bindv1.proc" "$_TMPD/bindv1.mnt")"

mkdir -p "$_TMPD/multi/unrelated" "$_TMPD/multi/real/job"
printf '8589934592\n' > "$_TMPD/multi/unrelated/memory.max"
printf '8589934592\n' > "$_TMPD/multi/real/memory.max"
printf '1073741824\n' > "$_TMPD/multi/real/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/multi.proc"
{
    printf '%s\n' "60 30 0:50 /unrelated $_TMPD/multi/unrelated rw - cgroup2 cgroup2 rw"
    printf '%s\n' "61 30 0:51 /slice $_TMPD/multi/real rw - cgroup2 cgroup2 rw"
} > "$_TMPD/multi.mnt"
assert_eq "the containing mount wins over an earlier one" "1024" \
    "$(run_cgroup "$_TMPD/multi" "$_TMPD/multi.proc" "$_TMPD/multi.mnt")"

{
    printf '%s\n' "62 30 0:52 / $_TMPD/multi/unrelated rw - cgroup2 cgroup2 rw"
    printf '%s\n' "63 30 0:53 /slice $_TMPD/multi/real rw - cgroup2 cgroup2 rw"
} > "$_TMPD/multi.spec"
assert_eq "the most specific containing root wins" "1024" \
    "$(run_cgroup "$_TMPD/multi" "$_TMPD/multi.proc" "$_TMPD/multi.spec")"

printf '0::/nowhere\n' > "$_TMPD/multi.none"
assert_eq "no containing mount falls back to the first" "8192" \
    "$(run_cgroup "$_TMPD/multi" "$_TMPD/multi.none" "$_TMPD/multi.mnt")"

mkdir -p "$_TMPD/multi/v1a" "$_TMPD/multi/v1b/task"
printf '8589934592\n' > "$_TMPD/multi/v1a/memory.limit_in_bytes"
printf '4294967296\n' > "$_TMPD/multi/v1b/memory.limit_in_bytes"
printf '2147483648\n' > "$_TMPD/multi/v1b/task/memory.limit_in_bytes"
printf '3:memory:/docker/abc/task\n' > "$_TMPD/multiv1.proc"
{
    printf '%s\n' "64 30 0:54 /other $_TMPD/multi/v1a rw - cgroup cgroup rw,memory"
    printf '%s\n' "65 30 0:55 /docker/abc $_TMPD/multi/v1b rw - cgroup cgroup rw,memory"
} > "$_TMPD/multiv1.mnt"
assert_eq "v1 picks the containing mount too" "2048" \
    "$(run_cgroup "$_TMPD/multi" "$_TMPD/multiv1.proc" "$_TMPD/multiv1.mnt")"

# A limit above the narrower mount's root is only visible through the broader one, so every
# containing mount is inspected and the smallest allowance wins.
mkdir -p "$_TMPD/anc2/broad/slice/job/task" "$_TMPD/anc2/narrow/task"
printf 'max\n'        > "$_TMPD/anc2/broad/memory.max"
printf '1073741824\n' > "$_TMPD/anc2/broad/slice/memory.max"
printf 'max\n'        > "$_TMPD/anc2/broad/slice/job/memory.max"
printf 'max\n'        > "$_TMPD/anc2/broad/slice/job/task/memory.max"
printf '8589934592\n' > "$_TMPD/anc2/narrow/memory.max"
printf 'max\n'        > "$_TMPD/anc2/narrow/task/memory.max"
printf '0::/slice/job/task\n' > "$_TMPD/anc2.proc"
{
    printf '%s\n' "80 30 0:70 / $_TMPD/anc2/broad rw - cgroup2 cgroup2 rw"
    printf '%s\n' "81 30 0:70 /slice/job $_TMPD/anc2/narrow rw - cgroup2 cgroup2 rw"
} > "$_TMPD/anc2.mnt"
assert_eq "an ancestor limit above the narrower mount is still seen" "1024" \
    "$(run_cgroup "$_TMPD/anc2" "$_TMPD/anc2.proc" "$_TMPD/anc2.mnt")"
{
    printf '%s\n' "81 30 0:70 /slice/job $_TMPD/anc2/narrow rw - cgroup2 cgroup2 rw"
    printf '%s\n' "80 30 0:70 / $_TMPD/anc2/broad rw - cgroup2 cgroup2 rw"
} > "$_TMPD/anc2rev.mnt"
assert_eq "and the mount order does not change the answer" "1024" \
    "$(run_cgroup "$_TMPD/anc2" "$_TMPD/anc2.proc" "$_TMPD/anc2rev.mnt")"
printf '536870912\n' > "$_TMPD/anc2/narrow/task/memory.max"
assert_eq "the narrower mount still contributes its own limit" "512" \
    "$(run_cgroup "$_TMPD/anc2" "$_TMPD/anc2.proc" "$_TMPD/anc2.mnt")"
printf 'max\n' > "$_TMPD/anc2/narrow/task/memory.max"

# mountinfo escapes space, tab, newline and backslash as octal.
mkdir -p "$_TMPD/esc/a b/job"
printf '8589934592\n' > "$_TMPD/esc/a b/memory.max"
printf '1073741824\n' > "$_TMPD/esc/a b/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/esc.proc"
printf '%s\n' "70 30 0:60 /slice $_TMPD/esc/a\\040b rw - cgroup2 cgroup2 rw" > "$_TMPD/esc.mnt"
assert_eq "an escaped mount point is decoded" "1024" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/esc.proc" "$_TMPD/esc.mnt")"

mkdir -p "$_TMPD/esc/plain/job"
printf '2147483648\n' > "$_TMPD/esc/plain/job/memory.max"
printf '0::/a b/job\n' > "$_TMPD/escroot.proc"
printf '%s\n' "71 30 0:61 /a\\040b $_TMPD/esc/plain rw - cgroup2 cgroup2 rw" > "$_TMPD/escroot.mnt"
assert_eq "an escaped mount root is decoded" "2048" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/escroot.proc" "$_TMPD/escroot.mnt")"

# A colon is legal in a systemd unit name.
mkdir -p "$_TMPD/colon/cg/slice:tenant/job"
printf '8589934592\n' > "$_TMPD/colon/cg/slice:tenant/memory.max"
printf '1073741824\n' > "$_TMPD/colon/cg/slice:tenant/job/memory.max"
printf '0::/slice:tenant/job\n' > "$_TMPD/colon.proc"
assert_eq "v2 keeps a colon in the path" "1024" "$(run_cgroup "$_TMPD/colon/cg" "$_TMPD/colon.proc")"

mkdir -p "$_TMPD/colon/v1/memory/slice:tenant/task"
printf '8589934592\n' > "$_TMPD/colon/v1/memory/slice:tenant/memory.limit_in_bytes"
printf '2147483648\n' > "$_TMPD/colon/v1/memory/slice:tenant/task/memory.limit_in_bytes"
printf '4:memory:/slice:tenant/task\n' > "$_TMPD/colonv1.proc"
assert_eq "v1 keeps a colon in the path" "2048" "$(run_cgroup "$_TMPD/colon/v1" "$_TMPD/colonv1.proc")"

printf '4:cpu,memoryfoo:/slice:tenant/task\n' > "$_TMPD/colonv1.bad"
assert_eq "a lookalike controller is not matched" "" \
    "$(run_cgroup "$_TMPD/colon/v1" "$_TMPD/colonv1.bad")"

# \012 must travel escaped and be decoded once, or it splits the reader's line output.
_NL=$'\n'
mkdir -p "$_TMPD/esc/two${_NL}lines/job"
printf '8589934592\n' > "$_TMPD/esc/two${_NL}lines/memory.max"
printf '3221225472\n' > "$_TMPD/esc/two${_NL}lines/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/escnl.proc"
printf '%s\n' "72 30 0:62 /slice $_TMPD/esc/two\\012lines rw - cgroup2 cgroup2 rw" \
    > "$_TMPD/escnl.mnt"
assert_eq "a newline in the mount point survives" "3072" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/escnl.proc" "$_TMPD/escnl.mnt")"

mkdir -p "$_TMPD/esc/two${_NL}lines/outer/inner"
printf '2147483648\n' > "$_TMPD/esc/two${_NL}lines/outer/memory.max"
printf 'max\n'        > "$_TMPD/esc/two${_NL}lines/outer/inner/memory.max"
printf '0::/slice/outer/inner\n' > "$_TMPD/escnl2.proc"
assert_eq "a newline survives the ancestor walk" "2048" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/escnl2.proc" "$_TMPD/escnl.mnt")"

# $() eats a trailing newline, so the decode carries a sentinel.
mkdir -p "$_TMPD/esc/trail${_NL}/job"
printf '1073741824\n' > "$_TMPD/esc/trail${_NL}/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/esctrail.proc"
printf '%s\n' "74 30 0:64 /slice $_TMPD/esc/trail\\012 rw - cgroup2 cgroup2 rw" \
    > "$_TMPD/esctrail.mnt"
assert_eq "a trailing newline is not eaten" "1024" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/esctrail.proc" "$_TMPD/esctrail.mnt")"

# setup.sh runs `set -euo pipefail` and NCPU=$(_llama_build_jobs) is on the critical path,
# so a non-zero helper aborts the install. These use the real options.
_STRICT_FILE=$(mktemp)
for _fn in _cg_read _cg_limit _cg_dirs _cg_unesc_prog _cg_unesc _cg_mounts \
           _cg_rel _cg_pick_mounts _cgroup_free_mb _vm_stat_avail_mb _usable_ram_mb; do
    sed -n "/^${_fn}()/,/^}/p" "$SETUP_SH" >> "$_STRICT_FILE"
done
sed -n '/^_LLAMA_BUILD_RESERVE_MB=/,/^_LLAMA_BUILD_MB_PER_JOB=/p' "$SETUP_SH" >> "$_STRICT_FILE"
sed -n '/^_llama_jobs_for()/,/^}/p' "$SETUP_SH" >> "$_STRICT_FILE"

# _usable_ram_mb hardcodes /sys/fs/cgroup; stub it so a memory-limited container does not
# leak its allowance into host-memory assertions.
_NO_CGROUP='_cgroup_free_mb() { :; }; '

run_strict_rc() {
    bash -c 'set -euo pipefail; . "$1"; shift; eval "$@" >/dev/null 2>&1' \
        _ "$_STRICT_FILE" "$1" >/dev/null 2>&1
    printf '%s' "$?"
}

mkdir -p "$_TMPD/strict/leaf"
printf '4294967296\n' > "$_TMPD/strict/leaf/memory.max"
printf '0::/leaf\n'   > "$_TMPD/strict.proc"
printf '%s\n' "1 2 0:1 / $_TMPD/strict rw - cgroup2 cgroup2 rw" > "$_TMPD/strict.mnt"

# A directory passes `[ -r ]`, the shape that took the install down.
assert_eq "strict: _cg_read on a directory does not abort" "0" \
    "$(run_strict_rc '_cg_read "'"$_TMPD"'/strict"')"
assert_eq "strict: _cg_read on a missing file does not abort" "0" \
    "$(run_strict_rc '_cg_read "'"$_TMPD"'/strict/nope"')"
assert_eq "strict: _cg_mounts on an unreadable file does not abort" "0" \
    "$(run_strict_rc '_cg_mounts "'"$_TMPD"'/strict" cgroup2')"
assert_eq "strict: _cg_unesc does not abort" "0" \
    "$(run_strict_rc '_cg_unesc /a/b')"

# Reading a FIFO blocks forever; the -f guard rules that out for any path.
if mkfifo "$_TMPD/fifo" 2>/dev/null && command -v timeout >/dev/null 2>&1; then
    _fifo_rc=$(timeout 5 bash -c 'set -euo pipefail; . "$1"; _cg_read "$2" >/dev/null' \
                   _ "$_STRICT_FILE" "$_TMPD/fifo" >/dev/null 2>&1; printf '%s' "$?")
    assert_eq "strict: _cg_read on a FIFO returns instead of blocking" "0" "$_fifo_rc"
else
    # Stock macOS has no GNU timeout, and an unbounded read would hang the suite.
    echo "  SKIP: mkfifo or timeout unavailable"
fi

# A padded value must still compare as a number.
mkdir -p "$_TMPD/strict/pad"
printf '  4294967296  \n' > "$_TMPD/strict/pad/memory.max"
printf '0::/pad\n' > "$_TMPD/strictpad.proc"
assert_eq "a padded limit value is still parsed" "4096" \
    "$(run_cgroup "$_TMPD/strict" "$_TMPD/strictpad.proc" "$_TMPD/strict.mnt")"

# A failing awk (missing, or busybox without it) must drop the cgroup allowance, not the install.
mkdir -p "$_TMPD/noawk"
printf '#!/bin/sh\nexit 1\n' > "$_TMPD/noawk/awk"
chmod +x "$_TMPD/noawk/awk"
_noawk_rc=$(PATH="$_TMPD/noawk:$PATH" bash -c \
    'set -euo pipefail; . "$1"; _cgroup_free_mb "$2" "$3" "$4" >/dev/null' \
    _ "$_STRICT_FILE" "$_TMPD/strict" "$_TMPD/strict.proc" "$_TMPD/strict.mnt" >/dev/null 2>&1
    printf '%s' "$?")
assert_eq "strict: a failing awk does not abort the install" "0" "$_noawk_rc"
printf 'MemTotal:       16777216 kB\nMemAvailable:   12582912 kB\n' > "$_TMPD/meminfo-strict"
_noawk_jobs=$(PATH="$_TMPD/noawk:$PATH" bash -c \
    'set -euo pipefail; . "$1"; eval "$3"; _llama_jobs_for 20 "$(_usable_ram_mb "$2")"' \
    _ "$_STRICT_FILE" "$_TMPD/meminfo-strict" "$_NO_CGROUP" 2>/dev/null)
assert_eq "a failing awk falls back to the core count" "20" "$_noawk_jobs"
assert_eq "strict: _cgroup_free_mb on a real tree does not abort" "0" \
    "$(run_strict_rc '_cgroup_free_mb "'"$_TMPD"'/strict" "'"$_TMPD"'/strict.proc" "'"$_TMPD"'/strict.mnt"')"
assert_eq "strict: _cgroup_free_mb on nothing does not abort" "0" \
    "$(run_strict_rc '_cgroup_free_mb /nx /nx /nx')"
assert_eq "strict: _usable_ram_mb does not abort" "0" \
    "$(run_strict_rc '_usable_ram_mb')"
# The two AND-lists in _llama_jobs_for are a `set -e` footgun; cover each false branch.
assert_eq "strict: _llama_jobs_for with the cap binding does not abort" "0" \
    "$(run_strict_rc '_llama_jobs_for 20 16384')"
assert_eq "strict: _llama_jobs_for with cores binding does not abort" "0" \
    "$(run_strict_rc '_llama_jobs_for 4 65536')"
assert_eq "strict: _llama_jobs_for with the floor binding does not abort" "0" \
    "$(run_strict_rc '_llama_jobs_for 8 0')"
assert_eq "strict: the job count still comes out" "7" \
    "$(bash -c 'set -euo pipefail; . "$1"; _llama_jobs_for 20 16384' _ "$_STRICT_FILE")"

# POSIX mode applies errexit to a failing assignment, which pins the `|| true` on each read.
# $1 PATH prefix, $2 expression to assign from, $3 prelude.
run_posix_assign() {
    PATH="$1:$PATH" bash --posix -c \
        'set -euo pipefail; . "$1"; eval "${3:-}"; v=$(eval "$2"); printf "SURVIVED[%s]" "$v"' \
        _ "$_STRICT_FILE" "$2" "${3:-}" 2>/dev/null
}
assert_eq "POSIX mode: a failing awk does not abort _usable_ram_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_usable_ram_mb '"$_TMPD"'/meminfo-strict')"
assert_eq "POSIX mode: a failing awk does not abort _cgroup_free_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_cgroup_free_mb '"$_TMPD"'/strict '"$_TMPD"'/strict.proc '"$_TMPD"'/strict.mnt')"
assert_eq "POSIX mode: a failing awk does not abort _cg_unesc" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_cg_unesc /a/b')"
assert_eq "POSIX mode: a failing awk does not abort _cg_mounts" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_cg_mounts '"$_TMPD"'/strict.mnt cgroup2')"

# vm_stat does not exist on Linux, and _vm_stat_avail_mb is a pipeline.
assert_eq "POSIX mode: a missing vm_stat does not abort _vm_stat_avail_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_vm_stat_avail_mb </dev/null')"
assert_eq "POSIX mode: a failing awk does not abort _vm_stat_avail_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" 'printf "" | _vm_stat_avail_mb')"
assert_eq "POSIX mode: the macOS branch survives an unparseable vm_stat" "SURVIVED[16384]" \
    "$(PATH="$_TMPD/noawk:$PATH" bash --posix -c \
        'set -euo pipefail; . "$1"
         sysctl() { printf "17179869184"; }
         v=$(_usable_ram_mb "$2"); printf "SURVIVED[%s]" "$v"' \
        _ "$_STRICT_FILE" "$_TMPD/nope" 2>/dev/null)"
# (12288 - 2048) / 2048 = 5.
assert_eq "POSIX mode: the normal path still produces the capped count" "5" \
    "$(bash --posix -c 'set -euo pipefail; . "$1"; eval "$3"; _llama_jobs_for 20 "$(_usable_ram_mb "$2")"' \
        _ "$_STRICT_FILE" "$_TMPD/meminfo-strict" "$_NO_CGROUP" 2>/dev/null)"
assert_eq "POSIX mode: an unreadable meminfo keeps the core count" "20" \
    "$(bash --posix -c 'set -euo pipefail; . "$1"; eval "$3"; _llama_jobs_for 20 "$(_usable_ram_mb "$2")"' \
        _ "$_STRICT_FILE" "$_TMPD/nope" "$_NO_CGROUP" 2>/dev/null)"

rm -f "$_STRICT_FILE"

# Pinned meminfo: MemAvailable moves between two live reads and would race.
_RAM_FILE=$(mktemp)
sed -n '/^_vm_stat_avail_mb()/,/^}/p;/^_usable_ram_mb()/,/^}/p' "$SETUP_SH" > "$_RAM_FILE"
printf 'MemTotal:       16777216 kB\nMemAvailable:   12582912 kB\n' > "$_TMPD/meminfo"
run_usable_ram() {
    FAKE_FREE="$1" bash -c \
        '. "$1"; _cgroup_free_mb() { printf "%s" "$FAKE_FREE"; }; _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/meminfo"
}

assert_eq "a lower cgroup allowance wins" "512" "$(run_usable_ram 512)"
assert_eq "a higher allowance is ignored" "12288" "$(run_usable_ram 8796093022207)"
assert_eq "no cgroup leaves host memory" "12288" "$(run_usable_ram "")"

# MemTotal is the pre-3.14 fallback when MemAvailable is absent.
printf 'MemTotal:       16777216 kB\n' > "$_TMPD/meminfo-old"
assert_eq "pre-3.14 kernels fall back to MemTotal" "16384" \
    "$(bash -c '. "$1"; _cgroup_free_mb() { :; }; _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/meminfo-old")"

# free 1024 + inactive 1024 at the 16 KiB page size is 32 MiB. speculative and purgeable are
# deliberately NOT summed, so adding either back shows as a wrong number.
_VM_SAMPLE=$(printf '%s\n' \
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)" \
    "Pages free:                                    1024." \
    "Pages active:                                999999." \
    "Pages inactive:                                1024." \
    "Pages speculative:                              512." \
    "Pages wired down:                            999999." \
    "Pages purgeable:                                512.")
run_vm_stat() { bash -c ". '$_RAM_FILE'; _vm_stat_avail_mb" _; }
assert_eq "vm_stat sums the disjoint reclaimable queues" "32" "$(printf '%s\n' "$_VM_SAMPLE" | run_vm_stat)"
assert_eq "vm_stat ignores active and wired" "32" \
    "$(printf '%s\n' "$_VM_SAMPLE" | sed 's/999999/1/' | run_vm_stat)"
# speculative is a subset of free per xnu vm_statistics.h; adding it double-counts.
assert_eq "a larger speculative count does not change the answer" "32" \
    "$(printf '%s\n' "$_VM_SAMPLE" | sed 's/^Pages speculative:.*/Pages speculative:  99999./' | run_vm_stat)"
# purgeable is a page attribute, not a queue; those pages are already counted.
assert_eq "a larger purgeable count does not change the answer" "32" \
    "$(printf '%s\n' "$_VM_SAMPLE" | sed 's/^Pages purgeable:.*/Pages purgeable:  99999./' | run_vm_stat)"
assert_eq "free moves the answer" "48" \
    "$(printf '%s\n' "$_VM_SAMPLE" | sed 's/^Pages free:.*/Pages free:  2048./' | run_vm_stat)"
assert_eq "inactive moves the answer" "48" \
    "$(printf '%s\n' "$_VM_SAMPLE" | sed 's/^Pages inactive:.*/Pages inactive:  2048./' | run_vm_stat)"
assert_eq "vm_stat gibberish yields nothing" "" "$(printf 'no stats here\n' | run_vm_stat)"
assert_eq "vm_stat reports a genuine zero" "0" \
    "$(printf '%s\n' "Mach Virtual Memory Statistics: (page size of 16384 bytes)" \
        "Pages active: 999999." | run_vm_stat)"

# Linux sysctl has no hw.memsize, so stubbing sysctl makes this branch run on Linux CI.
assert_eq "_usable_ram_mb prefers vm_stat over hw.memsize" "777" \
    "$(bash -c '. "$1"
                sysctl() { printf "17179869184"; }
                _cgroup_free_mb() { :; }
                _vm_stat_avail_mb() { printf "777"; }
                _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/no-meminfo")"
# Zero is a reading and is kept, not replaced by installed RAM.
assert_eq "_usable_ram_mb keeps a zero reading" "0" \
    "$(bash -c '. "$1"
                sysctl() { printf "17179869184"; }
                _cgroup_free_mb() { :; }
                _vm_stat_avail_mb() { printf "0"; }
                _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/no-meminfo")"
assert_eq "zero usable RAM still builds at 1 job" "1" "$(run_jobs 20 0 "")"
assert_eq "_usable_ram_mb falls back to hw.memsize" "16384" \
    "$(bash -c '. "$1"
                sysctl() { printf "17179869184"; }
                _cgroup_free_mb() { :; }
                _vm_stat_avail_mb() { :; }
                _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/no-meminfo")"

assert_eq "4 GB allowance floors at 1 job" "1" "$(run_jobs 20 "$(run_usable_ram 4096)" "")"

# Budget from available, not installed, memory. Match the awk pattern so prose cannot satisfy it.
if grep -q '/\^MemAvailable:/' "$SETUP_SH"; then
    echo "  PASS: host memory reads MemAvailable"; PASS=$((PASS + 1))
else
    echo "  FAIL: host memory still reads MemTotal only"; FAIL=$((FAIL + 1))
fi
_AVAIL_LINE=$(grep -n '/\^MemAvailable:/' "$SETUP_SH" | head -1 | cut -d: -f1)
_TOTAL_LINE=$(grep -n '/\^MemTotal:/' "$SETUP_SH" | head -1 | cut -d: -f1)
if [ -n "$_AVAIL_LINE" ] && [ -n "$_TOTAL_LINE" ] && [ "$_AVAIL_LINE" -lt "$_TOTAL_LINE" ]; then
    echo "  PASS: MemTotal kept as the fallback, after MemAvailable"; PASS=$((PASS + 1))
else
    echo "  FAIL: MemTotal must remain, and only as the fallback"; FAIL=$((FAIL + 1))
fi

rm -rf "$_TMPD" "$_CG_FILE" "$_RAM_FILE"

# setup.ps1 carries its own copy because it cannot source setup.sh.
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
# AvailableMBytes counts the standby list; the Free counters do not.
_check_ps1 "setup.ps1 budgets from available memory" 'AvailableMBytes'
_check_ps1 "setup.ps1 feeds it to the job count" 'Get-LlamaJobsFor .*-TotalMb \(Get-UsableMemoryMb\)'
_check_ps1 "setup.ps1 keeps installed RAM as the fallback" 'TotalPhysicalMemory'
# Zero available is a reading; only a negative value means "could not read".
_check_ps1 "setup.ps1 keeps a zero reading" '^[[:space:]]*if \(\$null -ne \$avail\) \{ return \[long\]\$avail \}$'
_check_ps1 "setup.ps1 signals unreadable memory as -1" '^[[:space:]]*return -1$'
_check_ps1 "setup.ps1 treats only a negative as unreadable" '^[[:space:]]*if \(\$TotalMb -lt 0\) \{ return \$Cores \}$'
if grep -qE '^\s*if \(\$TotalMb -le 0\) \{ return \$Cores \}' "$SETUP_PS1"; then
    echo "  FAIL: setup.ps1 still treats zero available memory as unreadable"; FAIL=$((FAIL + 1))
else
    echo "  PASS: setup.ps1 no longer treats zero as unreadable"; PASS=$((PASS + 1))
fi

if grep -qE '^\s*\$NumCpu = \[Environment\]::ProcessorCount' "$SETUP_PS1"; then
    echo "  FAIL: setup.ps1 still sets -j from the raw core count"; FAIL=$((FAIL + 1))
else
    echo "  PASS: setup.ps1 no longer sets -j from the raw core count"; PASS=$((PASS + 1))
fi

rm -f "$_FUNC_FILE"
echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
