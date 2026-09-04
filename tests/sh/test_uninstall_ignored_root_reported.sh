#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# When UNSLOTH_HOME is set it takes the precedence chain alone and UNSLOTH_STUDIO_HOME /
# STUDIO_HOME are ignored, deliberately: see tests/sh/test_uninstall_home_precedence.sh for
# why an install found there is treated as somebody else's and spared.
#
# This test pins the OTHER half of that decision -- that the user is TOLD. Every build that
# shipped before UNSLOTH_HOME existed read UNSLOTH_STUDIO_HOME as "remove this install", and
# for an env-mode install re-exporting it is the only way to reach the tree: install.sh puts
# DATA_DIR at $STUDIO_HOME/share (install.sh:585), so there is no breadcrumb under $HOME for
# a bare run to find. A user upgrading from such a build, with UNSLOTH_HOME left over from a
# portable install or exported for anything else, would otherwise read "Unsloth Studio
# uninstalled" and "No studio.db was found" while that tree and its chat history sat on disk,
# with nothing on stdout naming the variable that had been ignored.
#
# The uninstaller runs for real against a fixture HOME and the assertions read its OUTPUT plus
# the removal outcome, so this cannot pass on a script that merely mentions the variable.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

# Same reason as tests/sh/test_uninstall_home_precedence.sh: on WSL the uninstall body
# reaches Windows-side state and sudo, neither contained by a fixture HOME.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

assert_grep() {
    if grep -qF "$3" "$2"; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (not in output: $3)"; FAIL=$((FAIL + 1))
    fi
}

assert_no_grep() {
    if grep -qF "$3" "$2"; then
        echo "  FAIL: $1 (unexpectedly in output: $3)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS: $1"; PASS=$((PASS + 1))
    fi
}

assert_content() {
    if [ -f "$2" ] && [ "$(cat "$2")" = "$3" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (missing or rewritten: $2)"; FAIL=$((FAIL + 1))
    fi
}

# An env-mode custom install at $1, the shape install.sh writes for UNSLOTH_STUDIO_HOME:
# the venv owner marker, share/studio.conf (DATA_DIR lives INSIDE the root, so nothing
# under $HOME points at it), and a studio.db holding $2.
make_flat_install() {
    mkdir -p "$1/unsloth_studio" "$1/share" "$1/bin"
    : > "$1/unsloth_studio/.unsloth-studio-owned"
    printf '%s\n' "$2" > "$1/studio.db"
    printf "UNSLOTH_EXE='%s'\n" "$1/unsloth_studio/bin/unsloth" > "$1/share/studio.conf"
}

# A portable master root at $1 holding its Studio install one level down.
make_portable_install() {
    mkdir -p "$1/studio/unsloth_studio" "$1/bin"
    : > "$1/studio/unsloth_studio/.unsloth-studio-owned"
    printf '%s\n' "$1" > "$1/.unsloth-portable-root"
    printf '%s\n' "$2" > "$1/studio/studio.db"
}

# env -i: the caller's own UNSLOTH_* / XDG_* would send removals outside the fixture.
run_uninstall() { # $1 fixture dir, $2 log name, then env assignments
    _ru_dir="$1"; _ru_log="$2"; shift 2
    mkdir -p "$_ru_dir/home/.local/share" "$_ru_dir/home/.local/bin" "$_ru_dir/home/Desktop"
    env -i PATH="$PATH" HOME="$_ru_dir/home" TMPDIR="$_ru_dir" \
        XDG_RUNTIME_DIR="$_ru_dir/run" XDG_DATA_HOME="$_ru_dir/home/.local/share" \
        XDG_CACHE_HOME="$_ru_dir/home/.cache" XDG_CONFIG_HOME="$_ru_dir/home/.config" \
        XDG_STATE_HOME="$_ru_dir/home/.local/state" "$@" \
        sh "$UNINSTALL_SH" > "$_ru_dir/$_ru_log" 2>&1
}

# Case 1: a stale UNSLOTH_HOME naming no install at all, and a real env-mode install named
# by UNSLOTH_STUDIO_HOME. The install is spared (the deliberate policy), so the run has to
# say which tree it left and why.
CASE1=$(mktemp -d)
trap 'rm -rf "$CASE1"' EXIT
mkdir -p "$CASE1/portable"
make_flat_install "$CASE1/other/studio" second-install-history
run_uninstall "$CASE1" uninstall.log \
    UNSLOTH_HOME="$CASE1/portable" UNSLOTH_STUDIO_HOME="$CASE1/other/studio"
assert_content "the ignored install is still on disk" \
    "$CASE1/other/studio/studio.db" second-install-history
assert_grep "the run reports that UNSLOTH_STUDIO_HOME was ignored" \
    "$CASE1/uninstall.log" "UNSLOTH_STUDIO_HOME / STUDIO_HOME were ignored"
assert_grep "and names the tree it left behind" \
    "$CASE1/uninstall.log" "$CASE1/other/studio"
assert_grep "and says the chat history was kept" \
    "$CASE1/uninstall.log" "chat history was NOT removed"
assert_grep "and gives a command that reaches it" \
    "$CASE1/uninstall.log" "UNSLOTH_HOME= UNSLOTH_STUDIO_HOME="
# The suggestion has to run where the user is, and BSD env on macOS is not somewhere to
# send them for a -u. A blank value takes the same branch as an unset one because
# _custom_studio_roots trims before testing -n, which the blank case below pins.
assert_no_grep "and does not send the user to env -u" \
    "$CASE1/uninstall.log" "env -u"

# Case 2: the STUDIO_HOME alias is reported the same way.
CASE2=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2"' EXIT
mkdir -p "$CASE2/portable"
make_flat_install "$CASE2/other/studio" second-install-history
run_uninstall "$CASE2" uninstall.log \
    UNSLOTH_HOME="$CASE2/portable" STUDIO_HOME="$CASE2/other/studio"
assert_grep "a STUDIO_HOME install left behind is reported too" \
    "$CASE2/uninstall.log" "$CASE2/other/studio"

# Case 3: both variables describe the SAME portable install (UNSLOTH_STUDIO_HOME is the
# root's own studio/ child). The master root's removal took the child with it, so there is
# nothing left behind and no note may claim otherwise.
CASE3=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3"' EXIT
make_portable_install "$CASE3/portable" portable-history
run_uninstall "$CASE3" uninstall.log \
    UNSLOTH_HOME="$CASE3/portable" UNSLOTH_STUDIO_HOME="$CASE3/portable/studio"
assert_no_grep "no note when both variables name the one install" \
    "$CASE3/uninstall.log" "were ignored"

# Case 4: UNSLOTH_HOME alone. Nothing was ignored, so nothing is reported.
CASE4=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3" "$CASE4"' EXIT
make_portable_install "$CASE4/portable" portable-history
run_uninstall "$CASE4" uninstall.log UNSLOTH_HOME="$CASE4/portable"
assert_no_grep "no note when only UNSLOTH_HOME is set" \
    "$CASE4/uninstall.log" "were ignored"

# Case 5: UNSLOTH_STUDIO_HOME alone -- the pre-UNSLOTH_HOME contract. The install is
# removed, so there is nothing to report.
CASE5=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3" "$CASE4" "$CASE5"' EXIT
make_flat_install "$CASE5/other/studio" second-install-history
run_uninstall "$CASE5" uninstall.log UNSLOTH_STUDIO_HOME="$CASE5/other/studio"
assert_no_grep "no note when UNSLOTH_HOME is unset" \
    "$CASE5/uninstall.log" "were ignored"

# Case 6: a whitespace-only UNSLOTH_HOME is what an unset variable looks like to install.sh
# and storage_roots.py, so it claims nothing and the install is REMOVED -- no note.
CASE6=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3" "$CASE4" "$CASE5" "$CASE6"' EXIT
make_flat_install "$CASE6/other/studio" second-install-history
run_uninstall "$CASE6" uninstall.log \
    UNSLOTH_HOME="   " UNSLOTH_STUDIO_HOME="$CASE6/other/studio"
assert_no_grep "a blank UNSLOTH_HOME reports nothing ignored" \
    "$CASE6/uninstall.log" "were ignored"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
