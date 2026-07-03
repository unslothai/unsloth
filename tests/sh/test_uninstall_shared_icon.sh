#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for _drop_shared_icon_if_unused() from scripts/uninstall.sh.
#
# The WSL uninstall shares %LOCALAPPDATA%\Unsloth Studio\unsloth.ico with the native
# install and every other WSL distro's shortcut. Removing one side must KEEP the icon
# while any "Unsloth Studio*.lnk" still references it, and drop it (plus the dir, if
# empty) only once the last shortcut is gone. Reciprocal of uninstall.ps1's
# _RemoveDataDirKeepingWslIcon. Tested hermetically: the function is extracted from
# uninstall.sh and run against per-test fixture dirs.
#
# Follows the extract-via-sed pattern of test_strixhalo_wsl_reroute.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

# All fixtures/temp files live under one root removed on exit, so a set -e abort
# can't leak dirs into $TMPDIR.
_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

assert_file()   { _l="$1"; [ -f "$2" ] && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (missing $2)"; FAIL=$((FAIL+1)); }; }
assert_nofile() { _l="$1"; [ -f "$2" ] && { echo "  FAIL: $_l (unexpected $2)"; FAIL=$((FAIL+1)); } || { echo "  PASS: $_l"; PASS=$((PASS+1)); }; }
assert_dir()    { _l="$1"; [ -d "$2" ] && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (missing dir $2)"; FAIL=$((FAIL+1)); }; }
assert_nodir()  { _l="$1"; [ -d "$2" ] && { echo "  FAIL: $_l (unexpected dir $2)"; FAIL=$((FAIL+1)); } || { echo "  PASS: $_l"; PASS=$((PASS+1)); }; }

# Extract just the function definition (12-space indented in the WSL branch).
FUNC_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/_drop_shared_icon_if_unused() {/,/^            }/p' "$UNINSTALL_SH" > "$FUNC_FILE"
# shellcheck disable=SC1090
. "$FUNC_FILE"

# make_user [shortcut_relpath] : a fresh fake Windows user dir with unsloth.ico,
# optionally placing an "Unsloth Studio*.lnk" at the given relative path.
make_user() {
    _u=$(mktemp -d -p "$_TMP_ROOT")
    mkdir -p "$_u/AppData/Local/Unsloth Studio"
    : > "$_u/AppData/Local/Unsloth Studio/unsloth.ico"
    if [ -n "${1:-}" ]; then
        mkdir -p "$_u/$(dirname "$1")"
        : > "$_u/$1"
    fi
    echo "$_u"
}
ICO='AppData/Local/Unsloth Studio/unsloth.ico'
DIR='AppData/Local/Unsloth Studio'
SM='AppData/Roaming/Microsoft/Windows/Start Menu/Programs'

# 1. A surviving NATIVE shortcut (Start Menu) -> icon and dir kept.
u=$(make_user "$SM/Unsloth Studio.lnk")
_drop_shared_icon_if_unused "$u"
assert_file "native Start Menu shortcut -> icon kept" "$u/$ICO"
assert_dir  "native Start Menu shortcut -> dir kept"  "$u/$DIR"

# 2. A surviving native shortcut on the Desktop -> icon kept.
u=$(make_user "Desktop/Unsloth Studio.lnk")
_drop_shared_icon_if_unused "$u"
assert_file "native Desktop shortcut -> icon kept" "$u/$ICO"

# 3. Another WSL distro's shortcut survives -> icon kept (shared by all distros).
u=$(make_user "Desktop/Unsloth Studio (WSL - OtherDistro).lnk")
_drop_shared_icon_if_unused "$u"
assert_file "other WSL distro shortcut -> icon kept" "$u/$ICO"

# 4. A shortcut under OneDrive\Desktop is also honored.
u=$(make_user "OneDrive/Desktop/Unsloth Studio.lnk")
_drop_shared_icon_if_unused "$u"
assert_file "OneDrive Desktop shortcut -> icon kept" "$u/$ICO"

# 5. No shortcut anywhere -> icon removed and the (now empty) dir removed.
u=$(make_user)
_drop_shared_icon_if_unused "$u"
assert_nofile "no shortcut -> icon removed" "$u/$ICO"
assert_nodir  "no shortcut -> empty dir removed" "$u/$DIR"

# 6. No shortcut, but the data dir has another file -> icon removed, dir KEPT.
u=$(make_user)
: > "$u/$DIR/launch-studio.ps1"
_drop_shared_icon_if_unused "$u"
assert_nofile "no shortcut -> icon removed (non-empty dir)" "$u/$ICO"
assert_dir    "non-empty dir kept" "$u/$DIR"

# 7. Missing data dir is a safe no-op (no error, exit 0).
u=$(mktemp -d -p "$_TMP_ROOT")
if _drop_shared_icon_if_unused "$u"; then
    echo "  PASS: missing data dir -> no-op"; PASS=$((PASS+1))
else
    echo "  FAIL: missing data dir errored"; FAIL=$((FAIL+1))
fi

# 8. A non-Unsloth .lnk must NOT keep the icon alive.
u=$(make_user "Desktop/Some Other App.lnk")
_drop_shared_icon_if_unused "$u"
assert_nofile "unrelated shortcut -> icon still removed" "$u/$ICO"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
