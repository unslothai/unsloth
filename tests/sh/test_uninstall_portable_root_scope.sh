#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# `UNSLOTH_HOME=<root> scripts/uninstall.sh` is ADDITIVE: it removes that root on top
# of the default install it always removes ($HOME/.unsloth/studio, the launcher data
# dir, the shortcuts). The uninstaller documents that in --help, so the behaviour is
# asserted here rather than changed; what must not happen is install.sh's portable
# closing message offering that command as the way to remove the selected root, which
# sent a user with both installs into deleting the default one's studio.db.
#
# The uninstaller runs for real against a fixture HOME, so the first half asserts the
# removal OUTCOME rather than the presence of a line in the script.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# Same reason as tests/sh/test_uninstall_prebuilt_artifacts.sh: on WSL the uninstall
# body reaches Windows-side state and sudo, neither contained by a fixture HOME.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

assert_gone() {
    if [ -e "$2" ] || [ -L "$2" ]; then
        echo "  FAIL: $1 (still present: $2)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS: $1"; PASS=$((PASS + 1))
    fi
}

check_block() { # label expectation(yes|no) haystack needle
    case "$3" in
        *"$4"*) _got=yes ;;
        *) _got=no ;;
    esac
    if [ "$_got" = "$2" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1"; FAIL=$((FAIL + 1))
    fi
}

# A default install and a portable root side by side, the layout the printed command hit.
FAKE_HOME="$_TMP_ROOT/home"
PORTABLE_ROOT="$_TMP_ROOT/portable"
mkdir -p "$FAKE_HOME/.unsloth/studio/unsloth_studio" "$FAKE_HOME/.local/share/unsloth" \
    "$FAKE_HOME/.local/bin" "$FAKE_HOME/Desktop" \
    "$PORTABLE_ROOT/studio/unsloth_studio" "$PORTABLE_ROOT/share" "$PORTABLE_ROOT/bin"
echo "default" > "$FAKE_HOME/.unsloth/studio/studio.db"
: > "$FAKE_HOME/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
printf "UNSLOTH_EXE='%s'\n" \
    "$FAKE_HOME/.unsloth/studio/unsloth_studio/bin/unsloth" \
    > "$FAKE_HOME/.local/share/unsloth/studio.conf"
echo "portable" > "$PORTABLE_ROOT/studio/studio.db"
: > "$PORTABLE_ROOT/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$PORTABLE_ROOT" > "$PORTABLE_ROOT/.unsloth-portable-root"
# The in-root master root record install.sh writes for a nested layout. It needs no removal
# rule of its own -- it lives INSIDE the Studio root, which goes wholesale -- and this asserts
# that rather than assuming it, so a uninstaller that ever stops taking the tree whole cannot
# leave a file behind that still reads as a portable install.
printf '%s\n' "$PORTABLE_ROOT" > "$PORTABLE_ROOT/studio/.unsloth-master-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE_ROOT/studio/unsloth_studio/bin/unsloth" "$PORTABLE_ROOT" \
    > "$PORTABLE_ROOT/share/studio.conf"

# env -i: the caller's own UNSLOTH_* / XDG_* would send removals outside the fixture.
env -i PATH="$PATH" HOME="$FAKE_HOME" TMPDIR="$_TMP_ROOT" \
    XDG_RUNTIME_DIR="$_TMP_ROOT/run" XDG_DATA_HOME="$FAKE_HOME/.local/share" \
    XDG_CACHE_HOME="$FAKE_HOME/.cache" XDG_CONFIG_HOME="$FAKE_HOME/.config" \
    XDG_STATE_HOME="$FAKE_HOME/.local/state" UNSLOTH_HOME="$PORTABLE_ROOT" \
    sh "$UNINSTALL_SH" > "$_TMP_ROOT/uninstall.log" 2>&1

assert_gone "UNSLOTH_HOME removes the portable root" "$PORTABLE_ROOT"
assert_gone "the master root record goes with the tree" \
    "$PORTABLE_ROOT/studio/.unsloth-master-root"
assert_gone "UNSLOTH_HOME also removes the default install" "$FAKE_HOME/.unsloth/studio"
assert_gone "UNSLOTH_HOME also removes the default chat history" \
    "$FAKE_HOME/.unsloth/studio/studio.db"
assert_gone "UNSLOTH_HOME also removes the default launcher data" \
    "$FAKE_HOME/.local/share/unsloth"

# Second fixture, portable root only: with no default install to supply one, the
# nested database at <root>/studio/studio.db is the only one there is. The removal
# always worked; the summary read it at <root>/studio.db, found nothing, and told the
# user no chat history had been removed right after removing it.
ONLY_HOME="$_TMP_ROOT/home-only"
ONLY_ROOT="$_TMP_ROOT/portable-only"
mkdir -p "$ONLY_HOME/.local/share" "$ONLY_HOME/.local/bin" \
    "$ONLY_ROOT/studio/unsloth_studio" "$ONLY_ROOT/share" "$ONLY_ROOT/bin"
echo "portable" > "$ONLY_ROOT/studio/studio.db"
: > "$ONLY_ROOT/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$ONLY_ROOT" > "$ONLY_ROOT/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$ONLY_ROOT/studio/unsloth_studio/bin/unsloth" "$ONLY_ROOT" \
    > "$ONLY_ROOT/share/studio.conf"

env -i PATH="$PATH" HOME="$ONLY_HOME" TMPDIR="$_TMP_ROOT" \
    XDG_RUNTIME_DIR="$_TMP_ROOT/run-only" XDG_DATA_HOME="$ONLY_HOME/.local/share" \
    XDG_CACHE_HOME="$ONLY_HOME/.cache" XDG_CONFIG_HOME="$ONLY_HOME/.config" \
    XDG_STATE_HOME="$ONLY_HOME/.local/state" UNSLOTH_HOME="$ONLY_ROOT" \
    sh "$UNINSTALL_SH" > "$_TMP_ROOT/uninstall-only.log" 2>&1

assert_gone "a portable-only run removes the nested root" "$ONLY_ROOT"
only_log="$(cat "$_TMP_ROOT/uninstall-only.log")"
check_block "the summary reports the nested chat history as removed" \
    yes "$only_log" "the chat history in the install(s) removed"
check_block "the summary does not claim no studio.db was found" \
    no "$only_log" "No studio.db was found"

assert_present() {
    if [ -e "$2" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (removed: $2)"; FAIL=$((FAIL + 1))
    fi
}

# Third fixture, a portable install that failed before create_studio_shortcuts wrote
# <root>/share/studio.conf. The uninstaller is exactly what a user reaches for then, and
# the venv marker it looks for sits one level down in a nested portable root. A directory
# carrying the portable marker and nothing else is still somebody else's and must survive.
PARTIAL_HOME="$_TMP_ROOT/home-partial"
PARTIAL_ROOT="$_TMP_ROOT/portable-partial"
BARE_ROOT="$_TMP_ROOT/portable-bare"
mkdir -p "$PARTIAL_HOME/.local/share" "$PARTIAL_HOME/.local/bin" \
    "$PARTIAL_ROOT/studio/unsloth_studio" "$PARTIAL_ROOT/bin" "$BARE_ROOT"
: > "$PARTIAL_ROOT/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$PARTIAL_ROOT" > "$PARTIAL_ROOT/.unsloth-portable-root"
printf '%s\n' "$BARE_ROOT" > "$BARE_ROOT/.unsloth-portable-root"
echo "not ours" > "$BARE_ROOT/notes.txt"
# Same rule for the record: it is an ownership signal for the RESOLVERS, which only ever read
# one inside a root they already resolved. It is not evidence that a directory handed to the
# uninstaller is ours, and _is_studio_root deliberately does not consult it.
RECORD_ONLY_ROOT="$_TMP_ROOT/record-only"
mkdir -p "$RECORD_ONLY_ROOT/studio"
printf '%s\n' "$RECORD_ONLY_ROOT" > "$RECORD_ONLY_ROOT/studio/.unsloth-master-root"
echo "not ours" > "$RECORD_ONLY_ROOT/notes.txt"

uninstall_root() { # root logname
    env -i PATH="$PATH" HOME="$PARTIAL_HOME" TMPDIR="$_TMP_ROOT" \
        XDG_RUNTIME_DIR="$_TMP_ROOT/run-$2" XDG_DATA_HOME="$PARTIAL_HOME/.local/share" \
        XDG_CACHE_HOME="$PARTIAL_HOME/.cache" XDG_CONFIG_HOME="$PARTIAL_HOME/.config" \
        XDG_STATE_HOME="$PARTIAL_HOME/.local/state" UNSLOTH_HOME="$1" \
        sh "$UNINSTALL_SH" > "$_TMP_ROOT/uninstall-$2.log" 2>&1
}

uninstall_root "$PARTIAL_ROOT" partial
assert_gone "a partial portable install is removed without studio.conf" "$PARTIAL_ROOT"

uninstall_root "$BARE_ROOT" bare
assert_present "a directory with the marker alone is refused" "$BARE_ROOT"
assert_present "and its contents are kept" "$BARE_ROOT/notes.txt"
check_block "the refusal is reported" yes "$(cat "$_TMP_ROOT/uninstall-bare.log")" \
    "refusing to remove non-Unsloth path"

uninstall_root "$RECORD_ONLY_ROOT" record
assert_present "a directory with the master root record alone is refused" "$RECORD_ONLY_ROOT"
assert_present "and its contents are kept" "$RECORD_ONLY_ROOT/notes.txt"

# So the portable closing message must not print that command as the removal.
done_block="$(sed -n '/portable install; everything lives in:/,/were left untouched/p' "$INSTALL_SH")"
if [ -z "$done_block" ]; then
    echo "  FAIL: could not extract install.sh's portable closing message"; FAIL=$((FAIL + 1))
fi
check_block "closing message removes the root with rm -rf" yes "$done_block" "rm -rf '"
check_block "closing message prints no UNSLOTH_HOME= uninstall command" \
    no "$done_block" "UNSLOTH_HOME='"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
