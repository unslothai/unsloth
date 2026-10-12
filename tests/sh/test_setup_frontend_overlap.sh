#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# setup.sh runs the frontend build in the background on a source install (#8818).
# These run the real helpers against a stand-in _build_frontend: the join reports
# the build's output and exit code, a setup that dies first takes the build with it,
# and the gate keeps Desktop, staged updates, verbose and opted-out runs serial.
set -u

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
SETUP_SH="$ROOT/studio/setup.sh"
. "$ROOT/tests/sh/_harness.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

HELPERS="$WORK/helpers.sh"
{
    printf '_FRONTEND_BG_PID=""\n_FRONTEND_BG_WATCH=""\n_FRONTEND_BG_LOG=""\n'
    for fn in _remove_agent_instruction_files _frontend_can_build_in_background \
        _frontend_bg_start _frontend_bg_join _frontend_bg_reap _frontend_build_exit; do
        sed -n "/^$fn() {/,/^}/p" "$SETUP_SH"
    done
} > "$HELPERS"
for fn in _frontend_can_build_in_background _frontend_bg_start _frontend_bg_join \
    _frontend_bg_reap _frontend_build_exit; do
    grep -q "^$fn() {" "$HELPERS" || { echo "FAIL: $fn not found in setup.sh"; exit 1; }
done

# Stand-ins for setup.sh's output helpers; setup_fail exits like the real one.
STUBS='
set -euo pipefail
step() { echo "step $*"; }
substep() { echo "substep $*"; }
setup_fail() { echo "setup_fail $1"; exit "$1"; }
_is_verbose() { [ "${UNSLOTH_VERBOSE:-0}" = "1" ]; }
'

run_case() {
    # $1 = body appended after the helpers; prints the script output and "rc=N".
    local out rc=0
    out=$(SCRIPT_DIR="$WORK/studio" STAGE_ROOT="" bash -c "$STUBS
. '$HELPERS'
$1" 2>&1) || rc=$?
    printf '%s\nrc=%s\n' "$out" "$rc"
}

mkdir -p "$WORK/studio/frontend/node_modules/pkg"

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

echo "== a failed build fails setup with its exit code =="
out=$(run_case '
_build_frontend() { echo "npm error"; exit 7; }
_frontend_bg_start
_frontend_bg_join
echo "not-reached"')
assert_contains "build output is replayed" "$out" "npm error"
assert_contains "setup_fail gets the build exit code" "$out" "setup_fail 7"
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

echo "== a setup that dies first stops the build and its children =="
MARK="$WORK/mark"
rm -f "$MARK"
out=$(run_case "
_build_frontend() {
    trap 'echo cleaned > \"$MARK\"' EXIT
    sleep 300 &
    echo \$! > '$WORK/grandchild'
    wait
}
_frontend_bg_start
sleep 1
setup_fail 3")
assert_contains "setup exits with its own code" "$out" "rc=3"
for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -f "$MARK" ] && break
    sleep 1
done
assert_eq "build ran its EXIT trap" "cleaned" "$(cat "$MARK" 2>/dev/null)"
gc=$(cat "$WORK/grandchild" 2>/dev/null || echo 0)
for _ in 1 2 3 4 5; do kill -0 "$gc" 2>/dev/null || break; sleep 1; done
if kill -0 "$gc" 2>/dev/null; then
    bad "grandchild $gc still running"
    kill "$gc" 2>/dev/null
else
    ok "grandchild process was stopped"
fi

echo "== a SIGINT to setup stops the build too =="
rm -f "$MARK"
SCRIPT_DIR="$WORK/studio" STAGE_ROOT="" bash -c "$STUBS
. '$HELPERS'
_build_frontend() { trap 'echo cleaned > \"$MARK\"' EXIT; sleep 300; }
_frontend_bg_start
sleep 300" >/dev/null 2>&1 &
parent=$!
sleep 1
kill -INT "$parent" 2>/dev/null; kill -TERM "$parent" 2>/dev/null
for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -f "$MARK" ] && break
    sleep 1
done
assert_eq "interrupted setup leaves no build behind" "cleaned" "$(cat "$MARK" 2>/dev/null)"
wait "$parent" 2>/dev/null

echo "== background build cleans agent guide files out of node_modules =="
touch "$WORK/studio/frontend/node_modules/pkg/AGENTS.md"
out=$(run_case '
_build_frontend() { :; }
_frontend_bg_start
_frontend_bg_join')
if [ -e "$WORK/studio/frontend/node_modules/pkg/AGENTS.md" ]; then
    bad "AGENTS.md left in node_modules"
else
    ok "AGENTS.md removed"
fi

echo "== a build that dies part way marks dist stale =="
mkdir -p "$WORK/studio/frontend/dist" "$WORK/studio/frontend/src"
touch "$WORK/studio/frontend/src/app.tsx"
touch "$WORK/studio/frontend/dist"
out=$(run_case '
_HIDDEN_GITIGNORES=()
_restore_gitignores() { :; }
_frontend_build_exit')
newer=$(find "$WORK/studio/frontend/src" -type f -newer "$WORK/studio/frontend/dist" -print -quit)
assert_eq "sources are newer than dist, so the next run rebuilds" \
    "$WORK/studio/frontend/src/app.tsx" "$newer"

echo "== gate =="
gate() {
    SCRIPT_DIR="$WORK/studio" bash -c "$STUBS
. '$HELPERS'
getconf() { echo \"\${FAKE_CPUS-16}\"; }
if _frontend_can_build_in_background; then echo yes; else echo no; fi" 2>/dev/null
}
# The memory probe reads this host; the CPU count is stubbed.
if [ -r /proc/meminfo ] && [ "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" -ge 8388608 ]; then
    assert_eq "big box overlaps" "yes" "$(STAGE_ROOT= gate)"
fi
assert_eq "2 CPUs stays serial" "no" "$(FAKE_CPUS=2 STAGE_ROOT= gate)"
assert_eq "unreadable CPU count stays serial" "no" "$(FAKE_CPUS= STAGE_ROOT= gate)"
assert_eq "opt-out stays serial" "no" "$(UNSLOTH_STUDIO_SERIAL_SETUP=1 STAGE_ROOT= gate)"
assert_eq "verbose stays serial" "no" "$(UNSLOTH_VERBOSE=1 STAGE_ROOT= gate)"
assert_eq "Desktop install stays serial" "no" "$(UNSLOTH_TAURI_MODE=1 STAGE_ROOT= gate)"
assert_eq "Desktop update stays serial" "no" "$(UNSLOTH_TAURI_UPDATE=true STAGE_ROOT= gate)"
assert_eq "staged update stays serial" "no" "$(STAGE_ROOT=/tmp/stage gate)"

summary
