#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# uv splits UV_OVERRIDE on whitespace, so a repo cloned under a path with a space
# truncates it and aborts every later uv call (issue #6503). install.sh must hand
# uv a space-free copy. Exercises the real install.sh hardening block.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

ok()  { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

# Extract the UV_OVERRIDE hardening block (outer case ... esac plus the export)
# and run it directly, so the test tracks install.sh rather than a copy of it.
BLOCK=$(awk '
    /case "[$]_OVERRIDES_FILE" in/ { grab = 1 }
    grab { print }
    grab && /export UV_OVERRIDE="[$]_OVERRIDES_FILE"/ { exit }
' "$INSTALL_SH")
if ! printf '%s' "$BLOCK" | grep -q 'export UV_OVERRIDE'; then
    echo "  FAIL: could not extract UV_OVERRIDE block from install.sh"
    exit 1
fi

run_block() {
    _OVERRIDES_FILE="$1"
    _UV_OVERRIDE_TMPDIR=""
    unset UV_OVERRIDE
    eval "$BLOCK"
}

echo "=== test_install_uv_override_space ==="

# 1. Spaced path -> space-free copy with identical contents, temp dir tracked.
WORK=$(mktemp -d)
mkdir -p "$WORK/Open Source"
SRC="$WORK/Open Source/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$SRC"
run_block "$SRC"
case "$UV_OVERRIDE" in
    *[[:space:]]*) bad "spaced path: UV_OVERRIDE still contains whitespace ($UV_OVERRIDE)" ;;
    *)             ok  "spaced path: UV_OVERRIDE is whitespace-free" ;;
esac
[ "$UV_OVERRIDE" != "$SRC" ] && ok "spaced path: points at a copy" || bad "spaced path: not copied"
[ "$(cat "$UV_OVERRIDE" 2>/dev/null)" = "transformers>=4.57.6" ] \
    && ok "spaced path: copy contents identical" || bad "spaced path: contents differ"
{ [ -n "$_UV_OVERRIDE_TMPDIR" ] && [ -d "$_UV_OVERRIDE_TMPDIR" ]; } \
    && ok "spaced path: temp dir tracked for cleanup" || bad "spaced path: temp dir not tracked"
# The exit-trap cleanup (_on_install_exit) must then remove it.
[ -n "$_UV_OVERRIDE_TMPDIR" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
[ ! -d "$_UV_OVERRIDE_TMPDIR" ] && ok "spaced path: temp dir removable" || bad "spaced path: temp dir lingers"
rm -rf "$WORK"

# 2. No-space path -> passthrough, no temp dir.
PLAIN=$(mktemp -d)
PSRC="$PLAIN/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$PSRC"
run_block "$PSRC"
[ "$UV_OVERRIDE" = "$PSRC" ] && ok "no-space path: UV_OVERRIDE unchanged" || bad "no-space path: changed ($UV_OVERRIDE)"
[ -z "$_UV_OVERRIDE_TMPDIR" ] && ok "no-space path: no temp dir created" || bad "no-space path: temp dir created"
rm -rf "$PLAIN"

# 3. TMPDIR itself contains a space -> fall back to the original path, no leak.
WORK2=$(mktemp -d)
mkdir -p "$WORK2/Open Source" "$WORK2/tmp dir"
SRC2="$WORK2/Open Source/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$SRC2"
RES=$( TMPDIR="$WORK2/tmp dir"; export TMPDIR; run_block "$SRC2"
       printf 'UV_OVERRIDE=%s\nTMPDIR_VAR=%s\n' "$UV_OVERRIDE" "$_UV_OVERRIDE_TMPDIR" )
echo "$RES" | grep -qx "UV_OVERRIDE=$SRC2" \
    && ok "spaced TMPDIR: falls back to original path" || bad "spaced TMPDIR: did not fall back ($RES)"
echo "$RES" | grep -qx "TMPDIR_VAR=" \
    && ok "spaced TMPDIR: no temp dir tracked" || bad "spaced TMPDIR: temp dir tracked"
# mktemp may have created a dir under the spaced TMPDIR; it must not be leaked.
_leftover=$(find "$WORK2/tmp dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n1)
[ -z "$_leftover" ] && ok "spaced TMPDIR: no leaked temp dir" || bad "spaced TMPDIR: leaked $_leftover"
rm -rf "$WORK2"

# 4. A tab in the path is whitespace uv also splits on -> copied like a space.
WORK3=$(mktemp -d)
TABDIR=$(printf 'Open\tSource')
mkdir -p "$WORK3/$TABDIR"
SRC3="$WORK3/$TABDIR/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$SRC3"
run_block "$SRC3"
case "$UV_OVERRIDE" in
    *[[:space:]]*) bad "tab path: UV_OVERRIDE still contains whitespace" ;;
    *)             ok  "tab path: UV_OVERRIDE is whitespace-free" ;;
esac
[ -n "$_UV_OVERRIDE_TMPDIR" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
rm -rf "$WORK3"

# 5. install.sh must clear _UV_OVERRIDE_TMPDIR before registering the exit trap,
# so an inherited value can never reach the trap's rm -rf.
_init_line=$(grep -n '^_UV_OVERRIDE_TMPDIR=""' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_trap_line=$(grep -n '^trap _on_install_exit EXIT' "$INSTALL_SH" | head -n1 | cut -d: -f1)
{ [ -n "$_init_line" ] && [ -n "$_trap_line" ] && [ "$_init_line" -lt "$_trap_line" ]; } \
    && ok "init: _UV_OVERRIDE_TMPDIR cleared before exit trap" \
    || bad "init: _UV_OVERRIDE_TMPDIR not cleared before exit trap (init=$_init_line trap=$_trap_line)"

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
