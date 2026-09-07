#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Uninstall must leave a shared uv cache on disk and only name it when the
# install recorded one outside the removed tree. A Studio-owned cache under
# the install root goes with the rm; telling the user to `uv cache clean`
# after that would wipe an unrelated cache.
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

make_studio() {
    FIXTURE_HOME=$(mktemp -d "$_TMP_ROOT/home.XXXXXX")
    mkdir -p "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin"
    : > "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin/unsloth"
    : > "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
}

run_uninstall() {
    HOME="$FIXTURE_HOME" sh "$UNINSTALL_SH" > "$_TMP_ROOT/out.log" 2>&1 || {
        echo "  FAIL: uninstall exited $?"; cat "$_TMP_ROOT/out.log"; exit 1
    }
    OUT=$(cat "$_TMP_ROOT/out.log")
}

echo "=== shared uv cache recorded outside the install root ==="

make_studio
mkdir -p "$FIXTURE_HOME/.cache/uv/wheels-v0" \
         "$FIXTURE_HOME/.unsloth/studio/cache"
: > "$FIXTURE_HOME/.cache/uv/wheels-v0/keep-me.whl"
printf '%s\n' "$FIXTURE_HOME/.cache/uv" > "$FIXTURE_HOME/.unsloth/studio/cache/uv-cache-dir"
run_uninstall

if [ -f "$FIXTURE_HOME/.cache/uv/wheels-v0/keep-me.whl" ]; then
    ok "shared uv cache survives uninstall"
else
    nope "shared uv cache was deleted"
fi
case "$OUT" in
    *"the uv package cache at $FIXTURE_HOME/.cache/uv was left in place"*)
        ok "output names the recorded shared cache"
        ;;
    *) nope "output missing the recorded shared cache path" ;;
esac
case "$OUT" in
    *"uv cache clean"*) ok "shared-cache note names uv cache clean" ;;
    *) nope "shared-cache note missing uv cache clean" ;;
esac
case "$OUT" in
    *"uv cache prune"*) nope "shared-cache note still leads with prune" ;;
    *) ok "shared-cache note does not lead with prune" ;;
esac

echo "=== Studio-owned uv cache under the install root ==="

make_studio
mkdir -p "$FIXTURE_HOME/.unsloth/studio/cache/uv/wheels-v0"
: > "$FIXTURE_HOME/.unsloth/studio/cache/uv/wheels-v0/x.whl"
printf '%s\n' "$FIXTURE_HOME/.unsloth/studio/cache/uv" \
    > "$FIXTURE_HOME/.unsloth/studio/cache/uv-cache-dir"
run_uninstall

if [ -e "$FIXTURE_HOME/.unsloth/studio" ]; then
    nope "studio install dir still present"
else
    ok "studio install dir (and its uv cache) is gone"
fi
case "$OUT" in
    *"uv package cache"*) nope "studio-mode still claims a uv cache was left" ;;
    *) ok "studio-mode is silent about a leftover uv cache" ;;
esac
case "$OUT" in
    *"uv cache clean"*) nope "studio-mode still points at uv cache clean" ;;
    *) ok "studio-mode does not point at uv cache clean" ;;
esac

echo "=== no uv-cache-dir marker (pre-#10204 install) ==="

make_studio
run_uninstall
case "$OUT" in
    *"if install reused a shared uv cache"*) ok "missing marker hedges instead of claiming a leftover" ;;
    *) nope "missing marker did not hedge" ;;
esac

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
