#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Uninstall must leave a shared uv cache on disk and name the commands that
# free it. Install can reuse ~/.cache/uv (#10204); wiping it here would drop
# wheels other uv projects still use.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

unset UNSLOTH_STUDIO_HOME STUDIO_HOME UNSLOTH_UNINSTALL_ROCM
XDG_RUNTIME_DIR="$_TMP_ROOT/run"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"

ok()   { echo "  PASS: $1"; PASS=$((PASS + 1)); }
nope() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

FIXTURE_HOME=$(mktemp -d "$_TMP_ROOT/home.XXXXXX")
mkdir -p "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin" \
         "$FIXTURE_HOME/.cache/uv/wheels-v0"
: > "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin/unsloth"
: > "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$FIXTURE_HOME/.cache/uv/wheels-v0/keep-me.whl"

HOME="$FIXTURE_HOME" sh "$UNINSTALL_SH" > "$_TMP_ROOT/out.log" 2>&1 || {
    echo "  FAIL: uninstall exited $?"; cat "$_TMP_ROOT/out.log"; exit 1
}

if [ -f "$FIXTURE_HOME/.cache/uv/wheels-v0/keep-me.whl" ]; then
    ok "shared uv cache survives uninstall"
else
    nope "shared uv cache was deleted"
fi

OUT=$(cat "$_TMP_ROOT/out.log")
case "$OUT" in
    *"uv cache prune"*) ok "output names uv cache prune" ;;
    *) nope "output missing uv cache prune" ;;
esac
case "$OUT" in
    *"uv cache clean"*) ok "output names uv cache clean" ;;
    *) nope "output missing uv cache clean" ;;
esac
case "$OUT" in
    *"removed: $FIXTURE_HOME/.cache/uv"*) nope "output claims to have removed the uv cache" ;;
    *) ok "output does not claim to remove the uv cache" ;;
esac

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
