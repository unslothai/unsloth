#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# UNSLOTH_HOME, UNSLOTH_STUDIO_HOME and STUDIO_HOME each name a WHOLE install, so the
# uninstaller takes the first one that is set and ignores the rest. install.sh does the
# same: UNSLOTH_HOME seeds _UNSLOTH_ROOT, that turns portable mode on, and
# _resolve_studio_destinations then derives STUDIO_HOME from the root without ever reading
# UNSLOTH_STUDIO_HOME. So with both exported nothing was installed at UNSLOTH_STUDIO_HOME
# by that environment, and a real install found there is somebody else's -- which the
# uninstaller used to rm -rf, chat history included, while removing the portable root.
#
# The uninstaller runs for real against a fixture HOME, so every case asserts the removal
# OUTCOME rather than the presence of a line in the script. The last four cases pin the
# single-variable behaviours, which this precedence must not change.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

# Same reason as tests/sh/test_uninstall_prebuilt_artifacts.sh: on WSL the uninstall
# body reaches Windows-side state and sudo, neither contained by a fixture HOME.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

assert_gone() {
    if [ -e "$2" ] || [ -L "$2" ]; then
        echo "  FAIL: $1 (still present: $2)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS: $1"; PASS=$((PASS + 1))
    fi
}

assert_present() {
    if [ -e "$2" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (removed: $2)"; FAIL=$((FAIL + 1))
    fi
}

assert_content() {
    if [ -f "$2" ] && [ "$(cat "$2")" = "$3" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (missing or rewritten: $2)"; FAIL=$((FAIL + 1))
    fi
}

# A complete custom install at $1: the venv owner marker install.sh writes, share/studio.conf,
# and a studio.db holding $2 so a deletion is visible in the content, not just the path.
make_flat_install() {
    mkdir -p "$1/unsloth_studio" "$1/share" "$1/bin"
    : > "$1/unsloth_studio/.unsloth-studio-owned"
    printf '%s\n' "$2" > "$1/studio.db"
    printf "UNSLOTH_EXE='%s'\n" "$1/unsloth_studio/bin/unsloth" > "$1/share/studio.conf"
}

# A portable master root at $1 holding its Studio install one level down, the layout
# install.sh --root writes. $3 = "conf" also writes <root>/share/studio.conf; without it
# this is an install that failed before create_studio_shortcuts, the state a user reaches
# for the uninstaller in (see tests/sh/test_uninstall_portable_root_scope.sh).
make_portable_install() {
    mkdir -p "$1/studio/unsloth_studio" "$1/bin"
    : > "$1/studio/unsloth_studio/.unsloth-studio-owned"
    printf '%s\n' "$1" > "$1/.unsloth-portable-root"
    printf '%s\n' "$2" > "$1/studio/studio.db"
    if [ "${3:-}" = conf ]; then
        mkdir -p "$1/share"
        printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
            "$1/studio/unsloth_studio/bin/unsloth" "$1" > "$1/share/studio.conf"
    fi
}

# env -i: the caller's own UNSLOTH_* / XDG_* would send removals outside the fixture. The
# variables under test are appended by the caller as NAME=VALUE arguments.
run_uninstall() { # $1 fixture dir, $2 log name, then env assignments
    _ru_dir="$1"; _ru_log="$2"; shift 2
    mkdir -p "$_ru_dir/home/.local/share" "$_ru_dir/home/.local/bin" "$_ru_dir/home/Desktop"
    env -i PATH="$PATH" HOME="$_ru_dir/home" TMPDIR="$_ru_dir" \
        XDG_RUNTIME_DIR="$_ru_dir/run" XDG_DATA_HOME="$_ru_dir/home/.local/share" \
        XDG_CACHE_HOME="$_ru_dir/home/.cache" XDG_CONFIG_HOME="$_ru_dir/home/.config" \
        XDG_STATE_HOME="$_ru_dir/home/.local/state" "$@" \
        sh "$UNINSTALL_SH" > "$_ru_dir/$_ru_log" 2>&1
}

# Case 1: a portable root to remove, and a SEPARATE install still named by a stale
# UNSLOTH_STUDIO_HOME. Only the portable root may go.
CASE1=$(mktemp -d)
trap 'rm -rf "$CASE1"' EXIT
make_portable_install "$CASE1/portable" portable-history
make_flat_install "$CASE1/other/studio" second-install-history
run_uninstall "$CASE1" uninstall.log \
    UNSLOTH_HOME="$CASE1/portable" UNSLOTH_STUDIO_HOME="$CASE1/other/studio"
assert_gone "the portable root named by UNSLOTH_HOME is removed" "$CASE1/portable"
assert_present "a second install named by a stale UNSLOTH_STUDIO_HOME is kept" \
    "$CASE1/other/studio"
assert_content "and its chat history is untouched" \
    "$CASE1/other/studio/studio.db" second-install-history

# Case 2: the same, with a stale STUDIO_HOME instead of UNSLOTH_STUDIO_HOME.
CASE2=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2"' EXIT
make_portable_install "$CASE2/portable" portable-history
make_flat_install "$CASE2/other/studio" second-install-history
run_uninstall "$CASE2" uninstall.log \
    UNSLOTH_HOME="$CASE2/portable" STUDIO_HOME="$CASE2/other/studio"
assert_gone "the portable root is removed with a stale STUDIO_HOME set" "$CASE2/portable"
assert_content "a second install named by a stale STUDIO_HOME keeps its history" \
    "$CASE2/other/studio/studio.db" second-install-history

# Case 3: UNSLOTH_HOME names a directory that is not an Unsloth install at all (the user
# already deleted the portable root by hand, per install.sh's closing `rm -rf` hint, and
# left the variable exported). The refusal must not become a licence to fall through to
# UNSLOTH_STUDIO_HOME and delete the install found there instead.
CASE3=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3"' EXIT
mkdir -p "$CASE3/portable"
make_flat_install "$CASE3/other/studio" second-install-history
run_uninstall "$CASE3" uninstall.log \
    UNSLOTH_HOME="$CASE3/portable" UNSLOTH_STUDIO_HOME="$CASE3/other/studio"
assert_content "a stale UNSLOTH_HOME does not fall through to another install" \
    "$CASE3/other/studio/studio.db" second-install-history

# Case 4: both variables describe the SAME portable install (UNSLOTH_STUDIO_HOME is the
# root's own studio/ child). Ignoring it must strand nothing: the master root's removal
# takes the child with it.
CASE4=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3" "$CASE4"' EXIT
make_portable_install "$CASE4/portable" portable-history conf
run_uninstall "$CASE4" uninstall.log \
    UNSLOTH_HOME="$CASE4/portable" UNSLOTH_STUDIO_HOME="$CASE4/portable/studio"
assert_gone "a self-contained portable root is removed whole" "$CASE4/portable"

# Case 5: UNSLOTH_HOME alone, the behaviour tests/sh/test_uninstall_portable_root_scope.sh
# pins. Repeated here so a regression shows up beside the precedence it belongs to.
CASE5=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3" "$CASE4" "$CASE5"' EXIT
make_portable_install "$CASE5/portable" portable-history conf
run_uninstall "$CASE5" uninstall.log UNSLOTH_HOME="$CASE5/portable"
assert_gone "UNSLOTH_HOME alone still removes the portable root" "$CASE5/portable"

# Case 6: UNSLOTH_STUDIO_HOME alone still removes its custom root.
CASE6=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3" "$CASE4" "$CASE5" "$CASE6"' EXIT
make_flat_install "$CASE6/custom" custom-history
run_uninstall "$CASE6" uninstall.log UNSLOTH_STUDIO_HOME="$CASE6/custom"
assert_gone "UNSLOTH_STUDIO_HOME alone still removes its custom root" "$CASE6/custom"

# Case 7: STUDIO_HOME alone still removes its custom root, and is still ignored when
# UNSLOTH_STUDIO_HOME is set beside it.
CASE7=$(mktemp -d)
trap 'rm -rf "$CASE1" "$CASE2" "$CASE3" "$CASE4" "$CASE5" "$CASE6" "$CASE7"' EXIT
make_flat_install "$CASE7/custom" custom-history
run_uninstall "$CASE7" alias.log STUDIO_HOME="$CASE7/custom"
assert_gone "STUDIO_HOME alone still removes its custom root" "$CASE7/custom"

make_flat_install "$CASE7/primary" primary-history
make_flat_install "$CASE7/alias" alias-history
run_uninstall "$CASE7" both.log \
    UNSLOTH_STUDIO_HOME="$CASE7/primary" STUDIO_HOME="$CASE7/alias"
assert_gone "UNSLOTH_STUDIO_HOME still wins over STUDIO_HOME" "$CASE7/primary"
assert_content "and the STUDIO_HOME install keeps its history" \
    "$CASE7/alias/studio.db" alias-history

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
