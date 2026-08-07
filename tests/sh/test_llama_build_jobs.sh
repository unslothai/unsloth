#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _llama_jobs_for() from studio/setup.sh: the cmake -j count.
#
# A 20-thread 16 GB box used to build llama.cpp at -j20 and stopped responding.
# A full CUDA build at -j20 measures ~8.2 GiB in aggregate, from ~30 concurrent
# compiler processes, which a 16 GB machine with a desktop on it does not have.
# The job count is now the smaller of the core count and what RAM can hold; see
# _LLAMA_BUILD_* in setup.sh for where the per-job budget comes from.
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
for _fn in _cg_read _cg_limit _cg_dirs _cg_unesc_prog _cg_unesc _cg_mounts _cg_rel \
           _cg_pick_mount _cgroup_free_mb; do
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

# 14) A bind-mounted subtree: mount root /slice at the mount point, while
#     /proc/self/cgroup still reports the host-absolute /slice/job. The files
#     are at <mountpoint>/job, so joining the two unmapped walks a path that
#     does not exist and settles on the outer slice limit instead of the job's.
mkdir -p "$_TMPD/bind/cg/job"
printf '8589934592\n' > "$_TMPD/bind/cg/memory.max"
printf '1073741824\n' > "$_TMPD/bind/cg/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/bind.proc"
printf '%s\n' "50 30 0:40 /slice $_TMPD/bind/cg rw - cgroup2 cgroup2 rw" > "$_TMPD/bind.mnt"
assert_eq "bind mount finds the innermost limit" "1024" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bind.proc" "$_TMPD/bind.mnt")"

# 15) The same fixture with the mount root claimed as "/" picks the outer limit,
#     which is what the unmapped join used to do.
printf '%s\n' "50 30 0:40 / $_TMPD/bind/cg rw - cgroup2 cgroup2 rw" > "$_TMPD/bind.slash"
assert_eq "an unmapped join settles for the outer limit" "8192" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bind.proc" "$_TMPD/bind.slash")"

# 16) A process outside the mount's root gets nothing from it rather than a
#     limit belonging to some other part of the tree.
printf '0::/elsewhere/job\n' > "$_TMPD/bind.out"
assert_eq "a path outside the mount root yields the mount's own limit" "8192" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bind.out" "$_TMPD/bind.mnt")"

# 17) v1 bind mounts map the same way, via the controller's own path column.
mkdir -p "$_TMPD/bind/v1/task"
printf '4294967296\n' > "$_TMPD/bind/v1/memory.limit_in_bytes"
printf '2147483648\n' > "$_TMPD/bind/v1/task/memory.limit_in_bytes"
printf '3:memory:/docker/abc/task\n' > "$_TMPD/bindv1.proc"
printf '%s\n' "51 30 0:41 /docker/abc $_TMPD/bind/v1 rw - cgroup cgroup rw,memory" \
    > "$_TMPD/bindv1.mnt"
assert_eq "v1 bind mount finds the innermost limit" "2048" \
    "$(run_cgroup "$_TMPD/bind" "$_TMPD/bindv1.proc" "$_TMPD/bindv1.mnt")"

# 18) A hierarchy mounted more than once: an unrelated subtree listed first
#     must not shadow the mount that actually contains the process.
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

# 19) Most specific wins when several roots contain the path.
{
    printf '%s\n' "62 30 0:52 / $_TMPD/multi/unrelated rw - cgroup2 cgroup2 rw"
    printf '%s\n' "63 30 0:53 /slice $_TMPD/multi/real rw - cgroup2 cgroup2 rw"
} > "$_TMPD/multi.spec"
assert_eq "the most specific containing root wins" "1024" \
    "$(run_cgroup "$_TMPD/multi" "$_TMPD/multi.proc" "$_TMPD/multi.spec")"

# 20) When no mount contains the path, the first is still used rather than
#     dropping the hierarchy entirely.
printf '0::/nowhere\n' > "$_TMPD/multi.none"
assert_eq "no containing mount falls back to the first" "8192" \
    "$(run_cgroup "$_TMPD/multi" "$_TMPD/multi.none" "$_TMPD/multi.mnt")"

# 21) The same, for a v1 controller listed twice.
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

# 22) mountinfo escapes space, tab, newline and backslash as octal. Returning
#     the fields verbatim gave a path that does not exist.
mkdir -p "$_TMPD/esc/a b/job"
printf '8589934592\n' > "$_TMPD/esc/a b/memory.max"
printf '1073741824\n' > "$_TMPD/esc/a b/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/esc.proc"
printf '%s\n' "70 30 0:60 /slice $_TMPD/esc/a\\040b rw - cgroup2 cgroup2 rw" > "$_TMPD/esc.mnt"
assert_eq "an escaped mount point is decoded" "1024" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/esc.proc" "$_TMPD/esc.mnt")"

# 23) The mount root is escaped the same way, and has to match the process path.
mkdir -p "$_TMPD/esc/plain/job"
printf '2147483648\n' > "$_TMPD/esc/plain/job/memory.max"
printf '0::/a b/job\n' > "$_TMPD/escroot.proc"
printf '%s\n' "71 30 0:61 /a\\040b $_TMPD/esc/plain rw - cgroup2 cgroup2 rw" > "$_TMPD/escroot.mnt"
assert_eq "an escaped mount root is decoded" "2048" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/escroot.proc" "$_TMPD/escroot.mnt")"

# 24) A colon is legal in a systemd unit name, and -F: with $3 truncated there.
mkdir -p "$_TMPD/colon/cg/slice:tenant/job"
printf '8589934592\n' > "$_TMPD/colon/cg/slice:tenant/memory.max"
printf '1073741824\n' > "$_TMPD/colon/cg/slice:tenant/job/memory.max"
printf '0::/slice:tenant/job\n' > "$_TMPD/colon.proc"
assert_eq "v2 keeps a colon in the path" "1024" "$(run_cgroup "$_TMPD/colon/cg" "$_TMPD/colon.proc")"

# 25) Same for v1, where the path is the third field rather than the remainder.
mkdir -p "$_TMPD/colon/v1/memory/slice:tenant/task"
printf '8589934592\n' > "$_TMPD/colon/v1/memory/slice:tenant/memory.limit_in_bytes"
printf '2147483648\n' > "$_TMPD/colon/v1/memory/slice:tenant/task/memory.limit_in_bytes"
printf '4:memory:/slice:tenant/task\n' > "$_TMPD/colonv1.proc"
assert_eq "v1 keeps a colon in the path" "2048" "$(run_cgroup "$_TMPD/colon/v1" "$_TMPD/colonv1.proc")"

# 26) The controller column is still matched exactly, not by substring.
printf '4:cpu,memoryfoo:/slice:tenant/task\n' > "$_TMPD/colonv1.bad"
assert_eq "a lookalike controller is not matched" "" \
    "$(run_cgroup "$_TMPD/colon/v1" "$_TMPD/colonv1.bad")"

# 27) \012 is a legal escape, and decoding it where the fields are read put a
#     newline into the reader's own line-oriented output, splitting one mount
#     record into two. The record has to travel escaped and be decoded once.
_NL=$'\n'
mkdir -p "$_TMPD/esc/two${_NL}lines/job"
printf '8589934592\n' > "$_TMPD/esc/two${_NL}lines/memory.max"
printf '3221225472\n' > "$_TMPD/esc/two${_NL}lines/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/escnl.proc"
printf '%s\n' "72 30 0:62 /slice $_TMPD/esc/two\\012lines rw - cgroup2 cgroup2 rw" \
    > "$_TMPD/escnl.mnt"
assert_eq "a newline in the mount point survives" "3072" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/escnl.proc" "$_TMPD/escnl.mnt")"

# 28) And the same mount point one level down, so the ancestor walk has to
#     carry the newline too rather than only the leaf lookup.
mkdir -p "$_TMPD/esc/two${_NL}lines/outer/inner"
printf '2147483648\n' > "$_TMPD/esc/two${_NL}lines/outer/memory.max"
printf 'max\n'        > "$_TMPD/esc/two${_NL}lines/outer/inner/memory.max"
printf '0::/slice/outer/inner\n' > "$_TMPD/escnl2.proc"
assert_eq "a newline survives the ancestor walk" "2048" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/escnl2.proc" "$_TMPD/escnl.mnt")"

# 29) A trailing newline is the one $() silently eats, so the decode carries a
#     sentinel. Without it the path loses its last character and does not exist.
mkdir -p "$_TMPD/esc/trail${_NL}/job"
printf '1073741824\n' > "$_TMPD/esc/trail${_NL}/job/memory.max"
printf '0::/slice/job\n' > "$_TMPD/esctrail.proc"
printf '%s\n' "74 30 0:64 /slice $_TMPD/esc/trail\\012 rw - cgroup2 cgroup2 rw" \
    > "$_TMPD/esctrail.mnt"
assert_eq "a trailing newline is not eaten" "1024" \
    "$(run_cgroup "$_TMPD/esc" "$_TMPD/esctrail.proc" "$_TMPD/esctrail.mnt")"

# ── The shell options the real script runs under ──
# Everything above sources into a plain `bash -c`, but studio/setup.sh:5 is
# `set -euo pipefail`, and NCPU=$(_llama_build_jobs) sits on the install's
# critical path. A helper returning non-zero there does not degrade the job
# count, it aborts the install at the build step. `head | tr` inside _cg_read
# did exactly that for any input where the pipeline failed, which -r alone does
# not rule out: a directory passes it, and a cgroup can be torn down between the
# test and the open. These drive the helpers with the real options on.
_STRICT_FILE=$(mktemp)
for _fn in _cg_read _cg_limit _cg_dirs _cg_unesc_prog _cg_unesc _cg_mounts \
           _cg_rel _cg_pick_mount _cgroup_free_mb _vm_stat_avail_mb _usable_ram_mb; do
    sed -n "/^${_fn}()/,/^}/p" "$SETUP_SH" >> "$_STRICT_FILE"
done
sed -n '/^_LLAMA_BUILD_RESERVE_MB=/,/^_LLAMA_BUILD_MB_PER_JOB=/p' "$SETUP_SH" >> "$_STRICT_FILE"
sed -n '/^_llama_jobs_for()/,/^}/p' "$SETUP_SH" >> "$_STRICT_FILE"

# Echoes the exit status of running $1 under the real options.
run_strict_rc() {
    bash -c 'set -euo pipefail; . "$1"; shift; eval "$@" >/dev/null 2>&1' \
        _ "$_STRICT_FILE" "$1" >/dev/null 2>&1
    printf '%s' "$?"
}

mkdir -p "$_TMPD/strict/leaf"
printf '4294967296\n' > "$_TMPD/strict/leaf/memory.max"
printf '0::/leaf\n'   > "$_TMPD/strict.proc"
printf '%s\n' "1 2 0:1 / $_TMPD/strict rw - cgroup2 cgroup2 rw" > "$_TMPD/strict.mnt"

# A directory passes `[ -r ]`, so this is the shape that took the install down.
assert_eq "strict: _cg_read on a directory does not abort" "0" \
    "$(run_strict_rc '_cg_read "'"$_TMPD"'/strict"')"
assert_eq "strict: _cg_read on a missing file does not abort" "0" \
    "$(run_strict_rc '_cg_read "'"$_TMPD"'/strict/nope"')"
assert_eq "strict: _cg_mounts on an unreadable file does not abort" "0" \
    "$(run_strict_rc '_cg_mounts "'"$_TMPD"'/strict" cgroup2')"
assert_eq "strict: _cg_unesc does not abort" "0" \
    "$(run_strict_rc '_cg_unesc /a/b')"

# The -f guard is not only about failing: reading a FIFO blocks forever, and an
# install that hangs is worse than one that stops. Nothing in cgroupfs is a
# FIFO, but the guard is what makes that true of any path handed to the reader.
if mkfifo "$_TMPD/fifo" 2>/dev/null; then
    _fifo_rc=$(timeout 5 bash -c 'set -euo pipefail; . "$1"; _cg_read "$2" >/dev/null' \
                   _ "$_STRICT_FILE" "$_TMPD/fifo" >/dev/null 2>&1; printf '%s' "$?")
    assert_eq "strict: _cg_read on a FIFO returns instead of blocking" "0" "$_fifo_rc"
else
    echo "  SKIP: mkfifo unavailable"
fi

# Whitespace really is stripped, rather than the trailing newline happening to
# be eaten by read. A value with padding must still compare as a number.
mkdir -p "$_TMPD/strict/pad"
printf '  4294967296  \n' > "$_TMPD/strict/pad/memory.max"
printf '0::/pad\n' > "$_TMPD/strictpad.proc"
assert_eq "a padded limit value is still parsed" "4096" \
    "$(run_cgroup "$_TMPD/strict" "$_TMPD/strictpad.proc" "$_TMPD/strict.mnt")"

# If awk itself fails -- missing, or a busybox build without the applet -- the
# reader must give up on the cgroup allowance, not take the install with it.
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
    'set -euo pipefail; . "$1"; _llama_jobs_for 20 "$(_usable_ram_mb "$2")"' \
    _ "$_STRICT_FILE" "$_TMPD/meminfo-strict" 2>/dev/null)
assert_eq "a failing awk falls back to the core count" "20" "$_noawk_jobs"
assert_eq "strict: _cgroup_free_mb on a real tree does not abort" "0" \
    "$(run_strict_rc '_cgroup_free_mb "'"$_TMPD"'/strict" "'"$_TMPD"'/strict.proc" "'"$_TMPD"'/strict.mnt"')"
assert_eq "strict: _cgroup_free_mb on nothing does not abort" "0" \
    "$(run_strict_rc '_cgroup_free_mb /nx /nx /nx')"
assert_eq "strict: _usable_ram_mb does not abort" "0" \
    "$(run_strict_rc '_usable_ram_mb')"
# The two AND-lists in _llama_jobs_for are the classic `set -e` footgun; cover
# the branch where each of them is false.
assert_eq "strict: _llama_jobs_for with the cap binding does not abort" "0" \
    "$(run_strict_rc '_llama_jobs_for 20 16384')"
assert_eq "strict: _llama_jobs_for with cores binding does not abort" "0" \
    "$(run_strict_rc '_llama_jobs_for 4 65536')"
assert_eq "strict: _llama_jobs_for with the floor binding does not abort" "0" \
    "$(run_strict_rc '_llama_jobs_for 8 0')"
# And the value still arrives, rather than the function exiting early with none.
assert_eq "strict: the job count still comes out" "7" \
    "$(bash -c 'set -euo pipefail; . "$1"; _llama_jobs_for 20 16384' _ "$_STRICT_FILE")"

# POSIX mode is the strict end of the range: bash applies errexit to a failing
# assignment there, which it does not do by default. A user with POSIXLY_CORRECT
# exported, or bash invoked as sh, gets those semantics, and an unreadable file
# must cost the cap rather than the install. This is what pins the `|| true` on
# each read; without them the shell exits before the job count is ever printed.
# The distinguishing form is the assignment: `v=$(reader)` propagates the
# reader's failure, while passing it as an argument discards the status. The
# real call site is NCPU=$(_llama_build_jobs), so pin the assignment form.
run_posix_assign() {  # $1 = a PATH prefix, $2 = the expression to assign from
    PATH="$1:$PATH" bash --posix -c \
        'set -euo pipefail; . "$1"; shift; v=$(eval "$@"); printf "SURVIVED[%s]" "$v"' \
        _ "$_STRICT_FILE" "$2" 2>/dev/null
}
assert_eq "POSIX mode: a failing awk does not abort _usable_ram_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_usable_ram_mb '"$_TMPD"'/meminfo-strict')"
assert_eq "POSIX mode: a failing awk does not abort _cgroup_free_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_cgroup_free_mb '"$_TMPD"'/strict '"$_TMPD"'/strict.proc '"$_TMPD"'/strict.mnt')"
assert_eq "POSIX mode: a failing awk does not abort _cg_unesc" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_cg_unesc /a/b')"
assert_eq "POSIX mode: a failing awk does not abort _cg_mounts" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_cg_mounts '"$_TMPD"'/strict.mnt cgroup2')"

# The macOS branch: vm_stat does not exist on Linux, and _vm_stat_avail_mb is a
# pipeline, so it has to survive both a missing binary and a failing awk.
assert_eq "POSIX mode: a missing vm_stat does not abort _vm_stat_avail_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" '_vm_stat_avail_mb </dev/null')"
assert_eq "POSIX mode: a failing awk does not abort _vm_stat_avail_mb" "SURVIVED[]" \
    "$(run_posix_assign "$_TMPD/noawk" 'printf "" | _vm_stat_avail_mb')"
# The macOS path of _usable_ram_mb, reached by making the meminfo unreadable and
# stubbing sysctl, must also survive a vm_stat that is absent or unparseable.
assert_eq "POSIX mode: the macOS branch survives an unparseable vm_stat" "SURVIVED[16384]" \
    "$(PATH="$_TMPD/noawk:$PATH" bash --posix -c \
        'set -euo pipefail; . "$1"
         sysctl() { printf "17179869184"; }
         v=$(_usable_ram_mb "$2"); printf "SURVIVED[%s]" "$v"' \
        _ "$_STRICT_FILE" "$_TMPD/nope" 2>/dev/null)"
# And with a working awk the numbers are still right under POSIX rules.
# 12288 MiB available -> (12288 - 2048) / 2048 = 5.
assert_eq "POSIX mode: the normal path still produces the capped count" "5" \
    "$(bash --posix -c 'set -euo pipefail; . "$1"; _llama_jobs_for 20 "$(_usable_ram_mb "$2")"' \
        _ "$_STRICT_FILE" "$_TMPD/meminfo-strict" 2>/dev/null)"
assert_eq "POSIX mode: an unreadable meminfo keeps the core count" "20" \
    "$(bash --posix -c 'set -euo pipefail; . "$1"; _llama_jobs_for 20 "$(_usable_ram_mb "$2")"' \
        _ "$_STRICT_FILE" "$_TMPD/nope" 2>/dev/null)"

rm -f "$_STRICT_FILE"

# The min() that ties it together. Drive _usable_ram_mb from a pinned meminfo
# rather than the live one: MemAvailable moves between two reads, so comparing
# a cached host figure against a second read is a race on any Linux runner.
_RAM_FILE=$(mktemp)
sed -n '/^_vm_stat_avail_mb()/,/^}/p;/^_usable_ram_mb()/,/^}/p' "$SETUP_SH" > "$_RAM_FILE"
printf 'MemTotal:       16777216 kB\nMemAvailable:   12582912 kB\n' > "$_TMPD/meminfo"
# $1 = the cgroup allowance the stubbed reader returns.
run_usable_ram() {
    FAKE_FREE="$1" bash -c \
        '. "$1"; _cgroup_free_mb() { printf "%s" "$FAKE_FREE"; }; _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/meminfo"
}

assert_eq "a lower cgroup allowance wins" "512" "$(run_usable_ram 512)"
assert_eq "a higher allowance is ignored" "12288" "$(run_usable_ram 8796093022207)"
# Host memory is what is available, not what is fitted: 16 GiB with 12 GiB free
# budgets from 12.
assert_eq "no cgroup leaves host memory" "12288" "$(run_usable_ram "")"

# MemTotal is still the pre-3.14 fallback when MemAvailable is absent.
printf 'MemTotal:       16777216 kB\n' > "$_TMPD/meminfo-old"
assert_eq "pre-3.14 kernels fall back to MemTotal" "16384" \
    "$(bash -c '. "$1"; _cgroup_free_mb() { :; }; _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/meminfo-old")"

# macOS has no MemAvailable, so the reclaim-aware figure comes from vm_stat.
# Fed synthetically: 1024 + 1024 + 512 + 512 pages at the 16 KiB Apple Silicon
# page size is 48 MiB, and the page size is read rather than assumed to be 4096.
_VM_SAMPLE=$(printf '%s\n' \
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)" \
    "Pages free:                                    1024." \
    "Pages active:                                999999." \
    "Pages inactive:                                1024." \
    "Pages speculative:                              512." \
    "Pages wired down:                            999999." \
    "Pages purgeable:                                512.")
run_vm_stat() { bash -c ". '$_RAM_FILE'; _vm_stat_avail_mb" _; }
assert_eq "vm_stat sums the reclaimable classes" "48" "$(printf '%s\n' "$_VM_SAMPLE" | run_vm_stat)"
# active and wired are resident, not reclaimable, and must not be counted.
assert_eq "vm_stat ignores active and wired" "48" \
    "$(printf '%s\n' "$_VM_SAMPLE" | sed 's/999999/1/' | run_vm_stat)"
assert_eq "vm_stat gibberish yields nothing" "" "$(printf 'no stats here\n' | run_vm_stat)"
# A header with no reclaimable pages is a zero reading, not a parse failure.
assert_eq "vm_stat reports a genuine zero" "0" \
    "$(printf '%s\n' "Mach Virtual Memory Statistics: (page size of 16384 bytes)" \
        "Pages active: 999999." | run_vm_stat)"

# And _usable_ram_mb must consult it. An unreadable meminfo path forces the
# branch that reads hw.memsize -- but that branch is an `elif` on sysctl
# succeeding, and sysctl has no hw.memsize on Linux, so the branch was never
# entered and this asserted nothing off a Mac. Stubbing sysctl is what makes it
# run on a Linux runner, which is where CI actually is.
assert_eq "_usable_ram_mb prefers vm_stat over hw.memsize" "777" \
    "$(bash -c '. "$1"
                sysctl() { printf "17179869184"; }
                _cgroup_free_mb() { :; }
                _vm_stat_avail_mb() { printf "777"; }
                _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/no-meminfo")"
# Zero is a reading and is kept. Falling back to 16 GiB of installed RAM on a
# Mac with nothing reclaimable is exactly the oversubscription this PR removes.
assert_eq "_usable_ram_mb keeps a zero reading" "0" \
    "$(bash -c '. "$1"
                sysctl() { printf "17179869184"; }
                _cgroup_free_mb() { :; }
                _vm_stat_avail_mb() { printf "0"; }
                _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/no-meminfo")"
# And zero usable still builds, at one job.
assert_eq "zero usable RAM still builds at 1 job" "1" "$(run_jobs 20 0 "")"
# Only unparseable output falls back to installed RAM.
assert_eq "_usable_ram_mb falls back to hw.memsize" "16384" \
    "$(bash -c '. "$1"
                sysctl() { printf "17179869184"; }
                _cgroup_free_mb() { :; }
                _vm_stat_avail_mb() { :; }
                _usable_ram_mb "$2"' \
        _ "$_RAM_FILE" "$_TMPD/no-meminfo")"

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
# Zero available memory is a reading, not a failure. Treating it as unreadable
# fell back to installed RAM and handed a machine with nothing left its full
# core count, so only a negative value now means "could not read".
_check_ps1 "setup.ps1 keeps a zero reading" '^[[:space:]]*if \(\$null -ne \$avail\) \{ return \[long\]\$avail \}$'
_check_ps1 "setup.ps1 signals unreadable memory as -1" '^[[:space:]]*return -1$'
_check_ps1 "setup.ps1 treats only a negative as unreadable" '^[[:space:]]*if \(\$TotalMb -lt 0\) \{ return \$Cores \}$'
if grep -qE '^\s*if \(\$TotalMb -le 0\) \{ return \$Cores \}' "$SETUP_PS1"; then
    echo "  FAIL: setup.ps1 still treats zero available memory as unreadable"; FAIL=$((FAIL + 1))
else
    echo "  PASS: setup.ps1 no longer treats zero as unreadable"; PASS=$((PASS + 1))
fi

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
