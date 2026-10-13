#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# setup.sh runs the frontend build in the background on a source install (#8818).
# These run the real helpers (and the real setup_fail) against a stand-in _build_frontend:
# the join reports the build's output and exit code, a setup that fails or is interrupted
# first takes the whole build with it, puts back what it hid and leaves the tree marked for
# a rebuild, a new build waits for one still being stopped, and the gate keeps Desktop,
# staged updates, verbose, opted-out and small machines serial.
set -u

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
SETUP_SH="$ROOT/studio/setup.sh"
. "$ROOT/tests/sh/_harness.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

HELPERS="$WORK/helpers.sh"
FNS="setup_fail _remove_agent_instruction_files _frontend_can_build_in_background
    _frontend_build_cleanup _frontend_proc_id _frontend_build_begin _frontend_bg_stop _frontend_bg_start _frontend_bg_cancel
    _frontend_bg_join _frontend_bg_reap"
{
    printf '_FRONTEND_BG_PID=""\n_FRONTEND_BG_LOG=""\n'
    grep '^_FRONTEND_BUILD_MARK=' "$SETUP_SH"
    for fn in $FNS; do
        sed -n "/^$fn() {/,/^}/p" "$SETUP_SH"
    done
} > "$HELPERS"
for fn in $FNS; do
    grep -q "^$fn() {" "$HELPERS" || { echo "FAIL: $fn not found in setup.sh"; exit 1; }
done

STUBS='
set -euo pipefail
step() { echo "step $*"; }
substep() { echo "substep $*"; }
_is_verbose() { [ "${UNSLOTH_VERBOSE:-0}" = "1" ]; }
_install_oxc_runtime() { echo "oxc-installed"; [ -z "${OXC_FAIL:-}" ] || exit "$OXC_FAIL"; }
_OXC_DIR="$SCRIPT_DIR/backend/oxc"
'

FE="$WORK/outer/studio/frontend"
MARK="$FE/.unsloth-build.local"
run_case() {
    # $1 = body appended after the helpers; prints the script output and "rc=N".
    local out rc=0
    out=$(SCRIPT_DIR="$WORK/outer/studio" STAGE_ROOT="" bash -c "$STUBS
. '$HELPERS'
$1" 2>&1) || rc=$?
    printf '%s\nrc=%s\n' "$out" "$rc"
}
wait_gone() {
    # Wait up to $2 s for pid $1 to disappear; 0 when it did.
    local _i=0
    while kill -0 "$1" 2>/dev/null && [ "$_i" -lt "$(( $2 * 5 ))" ]; do sleep 0.2; _i=$((_i + 1)); done
    ! kill -0 "$1" 2>/dev/null
}
mkdir -p "$FE/node_modules/pkg" "$FE/src"

echo "== join reports a successful build after the parent's own steps =="
out=$(run_case '
_build_frontend() { sleep 1; echo "built-line"; }
_frontend_bg_start
echo "parent-step"
_frontend_bg_join
echo "after-join"')
assert_contains "build output is replayed" "$out" "built-line"
assert_eq "parent step runs while the build is still going" "parent-step" \
    "$(printf '%s\n' "$out" | grep -E '^(parent-step|built-line)$' | head -1)"
assert_contains "parent continues after a good build" "$out" "after-join"
assert_contains "exit 0" "$out" "rc=0"

echo "== the OXC runtime installs inside the job, after the build =="
assert_eq "build line then OXC line" "built-line oxc-installed" \
    "$(printf '%s\n' "$out" | grep -E '^(built-line|oxc-installed)$' | tr '\n' ' ' | sed 's/ $//')"
out=$(OXC_FAIL=6 run_case '
_build_frontend() { echo "built-line"; }
_frontend_bg_start
_frontend_bg_join
echo "not-reached"')
assert_contains "an OXC failure in the job fails setup with its code" "$out" "rc=6"
assert_not_contains "setup stops" "$out" "not-reached"

echo "== a failed build fails setup with its exit code =="
out=$(run_case '
_build_frontend() { echo "npm error"; exit 7; }
_frontend_bg_start
_frontend_bg_join
echo "not-reached"')
assert_contains "build output is replayed" "$out" "npm error"
assert_not_contains "setup stops" "$out" "not-reached"
assert_contains "exit 7" "$out" "rc=7"

echo "== errexit inside the build still fails it =="
out=$(run_case '
_build_frontend() { false; echo "kept-going"; }
_frontend_bg_start
_frontend_bg_join')
assert_not_contains "a failing command stops the build" "$out" "kept-going"
assert_contains "and fails setup" "$out" "rc=1"

echo "== reap is non-blocking and joins a finished build =="
out=$(run_case '
_build_frontend() { sleep 2; echo "late"; }
_frontend_bg_start
_frontend_bg_reap
echo "pid-after-first-reap=${_FRONTEND_BG_PID:+set}"
sleep 3
_frontend_bg_reap
echo "pid-after-second-reap=${_FRONTEND_BG_PID:-empty}"')
assert_contains "running build is left alone" "$out" "pid-after-first-reap=set"
assert_contains "finished build is joined" "$out" "pid-after-second-reap=empty"
assert_contains "and its output shown" "$out" "late"

echo "== reap surfaces a failed build at the next boundary =="
out=$(run_case '
_build_frontend() { exit 5; }
_frontend_bg_start
sleep 1
_frontend_bg_reap
echo "not-reached"')
assert_contains "exit 5" "$out" "rc=5"
assert_not_contains "setup stops at the boundary" "$out" "not-reached"

# A build that marks the tree, hides an ancestor .gitignore, writes dist and ignores TERM.
STUBBORN="_build_frontend() {
    _frontend_build_begin
    mv '$WORK/outer/.gitignore' '$WORK/outer/.gitignore._twbuild'
    mkdir -p '$FE/dist'
    python3 -c 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)' &
    echo \$! > '$WORK/grandchild'
    wait
}"
prime() {
    printf '*\n' > "$WORK/outer/.gitignore"
    rm -rf "$WORK/outer/.gitignore._twbuild" "$FE/dist" "$WORK/grandchild" "$MARK"
}
check_aftermath() {
    local gc
    gc=$(cat "$WORK/grandchild" 2>/dev/null || echo 0)
    if wait_gone "$gc" 10; then ok "$1: TERM-resistant child was killed"; else bad "$1: child $gc still running"; kill -KILL "$gc" 2>/dev/null; fi
    assert_eq "$1: ancestor .gitignore is back" "*" "$(cat "$WORK/outer/.gitignore" 2>/dev/null)"
    if [ -e "$MARK" ]; then ok "$1: tree stays marked, so the next run rebuilds"; else bad "$1: build mark gone"; fi
}

echo "== setup_fail stops the build before it returns =="
prime
out=$(run_case "$STUBBORN
_frontend_bg_start
sleep 1
echo \"log=\$_FRONTEND_BG_LOG\"
setup_fail 3 boom")
assert_contains "setup exits with its own code" "$out" "rc=3"
gc=$(cat "$WORK/grandchild" 2>/dev/null || echo 0)
if kill -0 "$gc" 2>/dev/null; then bad "build still running when setup_fail returned"; else ok "nothing of the build survives setup_fail"; fi
log=$(printf '%s\n' "$out" | sed -n 's/^log=//p')
if [ -n "$log" ] && [ ! -e "$log" ]; then ok "build log removed"; else bad "build log '$log' left behind"; fi
check_aftermath "setup_fail"

echo "== a setup killed by errexit is cleaned up by the watcher =="
prime
out=$(run_case "$STUBBORN
_frontend_bg_start
sleep 1
false")
assert_contains "errexit exit code" "$out" "rc=1"
check_aftermath "errexit"

echo "== Ctrl-C to setup's whole process group =="
prime
python3 - "$WORK/outer/studio" "$HELPERS" "$STUBS" "$STUBBORN" <<'PY'
import os, signal, subprocess, sys, time
script_dir, helpers, stubs, stubborn = sys.argv[1:]
body = f"{stubs}\n. '{helpers}'\n{stubborn}\n_frontend_bg_start\nsleep 300\n"
p = subprocess.Popen(["bash", "-c", body], env=dict(os.environ, SCRIPT_DIR=script_dir, STAGE_ROOT=""),
                     start_new_session=True, stdin=subprocess.DEVNULL)
time.sleep(1.5)
os.killpg(p.pid, signal.SIGINT)
p.wait()
PY
check_aftermath "Ctrl-C"

echo "== a build that exits first leaves nothing running after the join =="
out=$(run_case "
_build_frontend() {
    python3 -c 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)' &
    echo \$! > '$WORK/grandchild'
}
_frontend_bg_start
sleep 1
_frontend_bg_join
sleep 0.5
if kill -0 \$(cat '$WORK/grandchild') 2>/dev/null; then echo child-alive; else echo child-gone; fi")
assert_contains "the join itself killed the build's leftover child" "$out" "child-gone"
kill -KILL "$(cat "$WORK/grandchild" 2>/dev/null || echo 0)" 2>/dev/null

echo "== nothing is signalled once the build is joined =="
python3 -c 'import subprocess as s; print(s.Popen(["sleep", "60"], start_new_session=True, stdout=s.DEVNULL, stderr=s.DEVNULL).pid)' > "$WORK/other"
other=$(cat "$WORK/other")
out=$(run_case "
_build_frontend() { :; }
_frontend_bg_start
_frontend_bg_join
# Pretend the old id now names another group; a later failure must leave it alone.
_FRONTEND_BG_LOG=/nonexistent
setup_fail 4 later")
assert_contains "setup_fail exit" "$out" "rc=4"
if kill -0 "$other" 2>/dev/null; then ok "unrelated process group left alone"; kill "$other"; else bad "unrelated process group $other was killed"; fi

echo "== a new build waits for one still being stopped, then takes over =="
prime
( sleep 3 ) & owner=$!
ps -o pid= -o lstart= -p "$owner" | sed 's/^ *//' > "$MARK"
mv "$WORK/outer/.gitignore" "$WORK/outer/.gitignore._twbuild"
start=$(date +%s)
out=$(run_case '_frontend_build_begin; echo "self=$$"')
waited=$(( $(date +%s) - start ))
if [ "$waited" -ge 2 ]; then ok "waited for the live owner (${waited}s)"; else bad "did not wait for the live owner (${waited}s)"; fi
assert_eq "hidden .gitignore put back before the new build" "*" "$(cat "$WORK/outer/.gitignore" 2>/dev/null)"
assert_eq "marker now names the new build" "$(printf '%s\n' "$out" | sed -n 's/^self=//p')" "$(cut -d' ' -f1 "$MARK")"
wait "$owner" 2>/dev/null
start=$(date +%s)
run_case '_frontend_build_begin' >/dev/null
if [ $(( $(date +%s) - start )) -le 1 ]; then ok "a dead owner is not waited for"; else bad "waited for a dead owner"; fi
rm -f "$MARK"

echo "== background build cleans agent guide files out of node_modules =="
touch "$FE/node_modules/pkg/AGENTS.md"
out=$(run_case '
_build_frontend() { :; }
_frontend_bg_start
_frontend_bg_join')
if [ -e "$FE/node_modules/pkg/AGENTS.md" ]; then bad "AGENTS.md left in node_modules"; else ok "AGENTS.md removed"; fi

echo "== cleanup never replaces a .gitignore that exists =="
printf 'mine\n' > "$WORK/outer/.gitignore"
printf '*\n' > "$WORK/outer/.gitignore._twbuild"
run_case '_frontend_build_cleanup' >/dev/null
assert_eq "existing .gitignore kept" "mine" "$(cat "$WORK/outer/.gitignore")"
rm -f "$WORK/outer/.gitignore._twbuild"

echo "== gate =="
gate() {
    SCRIPT_DIR="$WORK/outer/studio" bash -c "$STUBS
. '$HELPERS'
nproc() { [ -n \"\${FAKE_CPUS-16}\" ] && echo \"\${FAKE_CPUS-16}\"; }
sysctl() { return 1; }
_usable_ram_mb() { printf '%s' \"\${FAKE_MB-65536}\"; }
if _frontend_can_build_in_background; then echo yes; else echo no; fi" 2>/dev/null
}
assert_eq "16 CPUs, 64 GiB usable overlaps" "yes" "$(STAGE_ROOT= gate)"
assert_eq "4 CPUs, 12 GiB usable overlaps" "yes" "$(FAKE_CPUS=4 FAKE_MB=12288 STAGE_ROOT= gate)"
assert_eq "2 CPUs stays serial" "no" "$(FAKE_CPUS=2 STAGE_ROOT= gate)"
assert_eq "unreadable CPU count stays serial" "no" "$(FAKE_CPUS= STAGE_ROOT= gate)"
assert_eq "8 GiB usable (cgroup or free) stays serial" "no" "$(FAKE_MB=8192 STAGE_ROOT= gate)"
assert_eq "unreadable memory stays serial" "no" "$(FAKE_MB= STAGE_ROOT= gate)"
assert_eq "opt-out stays serial" "no" "$(UNSLOTH_STUDIO_SERIAL_SETUP=1 STAGE_ROOT= gate)"
assert_eq "opt-out (true) stays serial" "no" "$(UNSLOTH_STUDIO_SERIAL_SETUP=true STAGE_ROOT= gate)"
assert_eq "verbose stays serial" "no" "$(UNSLOTH_VERBOSE=1 STAGE_ROOT= gate)"
assert_eq "Desktop install stays serial" "no" "$(UNSLOTH_TAURI_MODE=1 STAGE_ROOT= gate)"
assert_eq "Desktop update stays serial" "no" "$(UNSLOTH_TAURI_UPDATE=true STAGE_ROOT= gate)"
assert_eq "staged update stays serial" "no" "$(STAGE_ROOT=/tmp/stage gate)"
assert_eq "no usable ps stays serial" "no" "$(SCRIPT_DIR="$WORK/outer/studio" STAGE_ROOT= bash -c "$STUBS
. '$HELPERS'
nproc() { echo 16; }
_usable_ram_mb() { printf 65536; }
ps() { return 1; }
if _frontend_can_build_in_background; then echo yes; else echo no; fi" 2>/dev/null)"

summary
