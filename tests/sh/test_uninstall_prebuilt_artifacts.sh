#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# scripts/uninstall.sh must remove the prebuilt siblings of ~/.unsloth/studio,
# then prune ~/.unsloth itself.
#
# Each prebuilt serializes on <parent>/.<name>.install.lock (prebuilt_core.py
# install_lock_path), so llama.cpp, node and whisper.cpp each leave one behind.
# A stray lock is a zero-byte file that looks harmless, but the final
# `rmdir "$HOME/.unsloth"` refuses a non-empty directory, so one missed lock
# keeps the whole tree on disk. whisper.cpp only installs when a prebuilt
# matching the pinned llama.cpp build exists, so a live install often skips it
# and never exercises that path; the fixture below always creates it.
#
# The uninstaller runs for real against a fixture HOME, so this asserts the
# removal OUTCOME rather than the presence of a line in the script.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

# This suite runs the REAL uninstaller, and overriding HOME does not contain it
# on WSL: the body detects WSL from /proc/version and then reaches host state
# outside the fixture -- powershell.exe deletes Windows-side "Unsloth Studio*.lnk"
# shortcuts and the shared unsloth.ico (uninstall.sh WSL branch), the interop-off
# fallback scans /mnt/{c,d,e}/Users, and `sudo rm -f /etc/profile.d/unsloth-rocm-wsl.sh`
# touches the system. Skip there, exactly like tests/sh/test_uninstall_arg_guard.sh.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

# Isolate every environment-controlled removal path.
unset UNSLOTH_STUDIO_HOME STUDIO_HOME UNSLOTH_UNINSTALL_ROCM
XDG_RUNTIME_DIR="$_TMP_ROOT/run"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"

assert_gone() {
    if [ -e "$2" ] || [ -L "$2" ]; then
        echo "  FAIL: $1 (still present: $2)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS: $1"; PASS=$((PASS + 1))
    fi
}

assert_kept() {
    if [ -e "$2" ] || [ -L "$2" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (missing: $2)"; FAIL=$((FAIL + 1))
    fi
}

# A default-mode install plus every sibling artifact the prebuilt installers
# can leave under ~/.unsloth. Explicit mktemp template: BSD mktemp with no
# template implies -t and would land outside _TMP_ROOT on macOS.
make_home() {
    FIXTURE_HOME=$(mktemp -d "$_TMP_ROOT/home.XXXXXX")
    mkdir -p "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin" \
             "$FIXTURE_HOME/.unsloth/llama.cpp/build/bin" \
             "$FIXTURE_HOME/.unsloth/node/bin" \
             "$FIXTURE_HOME/.unsloth/whisper.cpp/build/bin" \
             "$FIXTURE_HOME/.unsloth/.cache" \
             "$FIXTURE_HOME/.unsloth/.staging" \
             "$FIXTURE_HOME/.local/share/unsloth" \
             "$FIXTURE_HOME/.local/bin"
    : > "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin/unsloth"
    : > "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
    : > "$FIXTURE_HOME/.unsloth/llama.cpp/build/bin/llama-server"
    : > "$FIXTURE_HOME/.unsloth/node/bin/node"
    : > "$FIXTURE_HOME/.unsloth/whisper.cpp/build/bin/whisper-server"
    : > "$FIXTURE_HOME/.unsloth/.llama.cpp.install.lock"
    : > "$FIXTURE_HOME/.unsloth/.node.install.lock"
    : > "$FIXTURE_HOME/.unsloth/.whisper.cpp.install.lock"
    # Taking over an abandoned lock renames it rather than deleting it
    # (install_node_prebuilt.py), so these accumulate across interrupted runs.
    : > "$FIXTURE_HOME/.unsloth/.node.install.lock.stale.12345"
    : > "$FIXTURE_HOME/.unsloth/.llama.cpp.install.lock.stale.6789"
    ln -s "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin/unsloth" \
          "$FIXTURE_HOME/.local/bin/unsloth"
}

echo "=== every prebuilt artifact is removed, then ~/.unsloth is pruned ==="

make_home
HOME="$FIXTURE_HOME" sh "$UNINSTALL_SH" > "$_TMP_ROOT/out.log" 2>&1 || {
    echo "  FAIL: uninstall exited $?"; FAIL=$((FAIL + 1)); cat "$_TMP_ROOT/out.log"; }

assert_gone "studio install dir"        "$FIXTURE_HOME/.unsloth/studio"
assert_gone "llama.cpp prebuilt"        "$FIXTURE_HOME/.unsloth/llama.cpp"
assert_gone "node runtime"              "$FIXTURE_HOME/.unsloth/node"
assert_gone "whisper.cpp prebuilt"      "$FIXTURE_HOME/.unsloth/whisper.cpp"
assert_gone "prebuilt cache"            "$FIXTURE_HOME/.unsloth/.cache"
assert_gone "staging dir"               "$FIXTURE_HOME/.unsloth/.staging"
assert_gone "llama.cpp install lock"    "$FIXTURE_HOME/.unsloth/.llama.cpp.install.lock"
assert_gone "node install lock"         "$FIXTURE_HOME/.unsloth/.node.install.lock"
assert_gone "whisper.cpp install lock"  "$FIXTURE_HOME/.unsloth/.whisper.cpp.install.lock"
assert_gone "stale node lock"           "$FIXTURE_HOME/.unsloth/.node.install.lock.stale.12345"
assert_gone "stale llama.cpp lock"      "$FIXTURE_HOME/.unsloth/.llama.cpp.install.lock.stale.6789"
assert_gone "~/.unsloth is pruned"      "$FIXTURE_HOME/.unsloth"
assert_gone "CLI shim"                  "$FIXTURE_HOME/.local/bin/unsloth"

echo "=== user content under ~/.unsloth is never removed ==="

# rmdir refuses a non-empty directory, which is what protects user files. The
# uninstaller must still clear its own artifacts around them.
make_home
mkdir -p "$FIXTURE_HOME/.unsloth/my-own-models"
: > "$FIXTURE_HOME/.unsloth/my-own-models/keep-me.gguf"
: > "$FIXTURE_HOME/.unsloth/notes.txt"
HOME="$FIXTURE_HOME" sh "$UNINSTALL_SH" > "$_TMP_ROOT/out2.log" 2>&1 || {
    echo "  FAIL: uninstall exited $?"; FAIL=$((FAIL + 1)); cat "$_TMP_ROOT/out2.log"; }

assert_kept "a user directory survives"        "$FIXTURE_HOME/.unsloth/my-own-models/keep-me.gguf"
assert_kept "a user file survives"             "$FIXTURE_HOME/.unsloth/notes.txt"
assert_kept "~/.unsloth kept for user content" "$FIXTURE_HOME/.unsloth"
assert_gone "artifacts still cleared"          "$FIXTURE_HOME/.unsloth/whisper.cpp"
assert_gone "locks still cleared"              "$FIXTURE_HOME/.unsloth/.node.install.lock"

echo "=== a missing ~/.unsloth is a clean no-op ==="

FIXTURE_HOME=$(mktemp -d "$_TMP_ROOT/empty.XXXXXX")
if HOME="$FIXTURE_HOME" sh "$UNINSTALL_SH" > "$_TMP_ROOT/out3.log" 2>&1; then
    echo "  PASS: uninstalling a machine with no install exits 0"; PASS=$((PASS + 1))
else
    echo "  FAIL: uninstalling a machine with no install exited $?"; FAIL=$((FAIL + 1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
