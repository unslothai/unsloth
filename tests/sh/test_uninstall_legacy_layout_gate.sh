#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _is_studio_root decides whether ~/.unsloth/studio is recursively deleted.
#
# It exists because the default root used to be deleted with no ownership check at all, so
# the documented one-liner destroyed whatever happened to sit there. The first version of
# the gate then went too far the other way: it only knew the CURRENT layout, and refused
# genuinely old installs, leaving the user's tree and studio.db on disk while printing
# their own install as a non-Unsloth path. Before the unsloth_studio rename the venv was
# <root>/.venv, and before .unsloth-studio-owned neither name carried a marker, so an old
# install can reach the gate carrying none of the original three sentinels. install.sh
# still migrates that layout, so it is not hypothetical.
#
# Both directions matter and both are checked here: everything Unsloth put there must be
# removable, and everything else must be refused. The POSIX twin of
# tests/studio/test_uninstall_legacy_layout_gate.ps1.
#
# The uninstaller body kills processes and deletes trees, so it is not executed: the gate
# is extracted with sed and run against per-test fixtures, following
# test_uninstall_shared_icon.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT
# Keep $HOME clear of the fixture trees: the gate's siblings consult it.
HOME="$_TMP_ROOT/home"
mkdir -p "$HOME"

_fn=$(sed -n '/^_is_studio_root() {/,/^}/p' "$UNINSTALL_SH")
if [ -z "$_fn" ]; then
    echo "  FAIL: could not extract _is_studio_root from $UNINSTALL_SH"
    exit 1
fi
eval "$_fn"

# name, expected (own|foreign), then paths. A trailing / makes a directory.
check() {
    _name="$1"; _want="$2"; shift 2
    _root="$_TMP_ROOT/$(printf '%s' "$_name" | tr -c 'a-zA-Z0-9' '_')"
    mkdir -p "$_root"
    for _rel in "$@"; do
        case "$_rel" in
            */) mkdir -p "$_root/$_rel" ;;
            *)  mkdir -p "$_root/$(dirname "$_rel")"; printf 'x' > "$_root/$_rel" ;;
        esac
    done
    if _is_studio_root "$_root"; then _got=own; else _got=foreign; fi
    if [ "$_got" = "$_want" ]; then
        echo "  PASS: $_name"; PASS=$((PASS+1))
    else
        echo "  FAIL: $_name (got $_got, want $_want)"; FAIL=$((FAIL+1))
    fi
}

echo "Layouts Unsloth created, which must stay removable:"
check "current: share/studio.conf" own "share/studio.conf"
check "current: unsloth_studio owner marker" own "unsloth_studio/.unsloth-studio-owned"
check "legacy .venv carrying the owner marker" own ".venv/.unsloth-studio-owned"
check "pre-marker unsloth_studio venv" own "unsloth_studio/bin/unsloth" "unsloth_studio/bin/python"
check "pre-marker legacy .venv" own ".venv/bin/unsloth" ".venv/bin/python"

echo
echo "Directories that are not ours, which must stay refused:"
# The case the gate exists for: indistinguishable from a user's own project venv.
check "a bare project venv" foreign ".venv/bin/python"
check "a venv merely NAMED unsloth_studio" foreign "unsloth_studio/bin/python"
# bin/unsloth is the console script pip generates for the unsloth distribution, so the
# check has to be that name and not "a venv with any console script in it".
check "a venv with an unrelated console script" foreign ".venv/bin/black" ".venv/bin/python"
check "a hand-made scratch directory" foreign "notes.txt"
check "an empty directory" foreign

echo
echo "Edge cases:"
if _is_studio_root ""; then
    echo "  FAIL: an empty path was claimed"; FAIL=$((FAIL+1))
else
    echo "  PASS: an empty path is refused"; PASS=$((PASS+1))
fi
if _is_studio_root "$_TMP_ROOT/nothing-here"; then
    echo "  FAIL: a missing path was claimed"; FAIL=$((FAIL+1))
else
    echo "  PASS: a missing path is refused"; PASS=$((PASS+1))
fi
_spaced="$_TMP_ROOT/a dir with spaces/studio"
mkdir -p "$_spaced/unsloth_studio"
printf 'x' > "$_spaced/unsloth_studio/.unsloth-studio-owned"
if _is_studio_root "$_spaced"; then
    echo "  PASS: a path containing spaces is handled"; PASS=$((PASS+1))
else
    echo "  FAIL: a path containing spaces was refused"; FAIL=$((FAIL+1))
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
