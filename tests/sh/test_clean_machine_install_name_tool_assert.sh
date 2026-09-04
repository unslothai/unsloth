#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Keep the clean-machine allow-list narrow: uv may self-ID one managed libpython on a
# CLT-present control leg, while CLT-absent legs must not reach install_name_tool at all.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSERT_SH="$SCRIPT_DIR/../../.github/scripts/clean-machine-assert.sh"

INSTALL_NAME_TOOL_HELPER="$SCRIPT_DIR/../../.github/scripts/clean-machine-install-name-tool.sh"
ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT
PASS=0
FAIL=0

expect_rc() {
    _label="$1"; _expected="$2"; _check="$3"; _trace_content="$4"
    printf '%b' "$_trace_content" > "$ROOT/trace.log"
    set +e
    UV_PYTHON_INSTALL_DIR="$UV_ROOT" UNSLOTH_TOOL_TRACE="$ROOT/trace.log" \
        INSTALL_LOG="$ROOT/install.log" bash "$ASSERT_SH" "$_check" \
        > "$ROOT/out.log" 2>&1
    _actual=$?
    set -e
    if [ "$_actual" -eq "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected rc $_expected, got $_actual)"
        cat "$ROOT/out.log"
        FAIL=$((FAIL + 1))
    fi
}

: > "$ROOT/install.log"
UV_ROOT="$ROOT/uv python"
_dylib="$UV_ROOT/cpython-3.12-macos-aarch64-none/lib/libpython3.12.dylib"
mkdir -p "$(dirname "$_dylib")" "$UV_ROOT/x" "$ROOT/outside/lib"
: > "$_dylib"
_traversal="$UV_ROOT/x/../../outside/lib/libpython3.12.dylib"
: > "$ROOT/outside/lib/libpython3.12.dylib"
trace_arg() {
    printf 'h'
    printf '%s' "$1" | od -An -v -tx1 | tr -d ' \n'
}
trace_record() {
    _argc="$1"; shift
    printf 'install_name_tool\\t%s' "$_argc"
    for _arg in "$@"; do printf '\\t%s' "$(trace_arg "$_arg")"; done
    printf '\\n'
}
_valid=$(trace_record 3 -id "$_dylib" "$_dylib")

echo "=== shared wrapper and sentinel helper ==="
mkdir -p "$ROOT/bin"
bash "$INSTALL_NAME_TOOL_HELPER" write sentinel "$ROOT/bin/install_name_tool"
: > "$ROOT/trace.log"
if PATH="$ROOT/bin:$PATH" UNSLOTH_TOOL_TRACE="$ROOT/trace.log" \
    bash "$INSTALL_NAME_TOOL_HELPER" verify-sentinel "$ROOT/trace.log" focused; then
    echo "  PASS: shared helper verifies and clears the CLT-absent sentinel"
    PASS=$((PASS + 1))
else
    echo "  FAIL: shared helper could not verify its CLT-absent sentinel"
    FAIL=$((FAIL + 1))
fi
if [ ! -s "$ROOT/trace.log" ]; then
    echo "  PASS: sentinel self-test trace is cleared before installer assertions"
    PASS=$((PASS + 1))
else
    echo "  FAIL: sentinel self-test trace was not cleared"
    FAIL=$((FAIL + 1))
fi

cat > "$ROOT/real-install-name-tool" <<'REAL_TOOL'
#!/bin/sh
printf '%s\n' "$*" >> "$REAL_TOOL_LOG"
REAL_TOOL
chmod +x "$ROOT/real-install-name-tool"
bash "$INSTALL_NAME_TOOL_HELPER" write passthrough \
    "$ROOT/bin/install_name_tool" "$ROOT/real-install-name-tool"
: > "$ROOT/trace.log"
: > "$ROOT/real-tool.log"
UNSLOTH_TOOL_TRACE="$ROOT/trace.log" REAL_TOOL_LOG="$ROOT/real-tool.log" \
    "$ROOT/bin/install_name_tool" -id "$_dylib" "$_dylib"
_expected_trace=$(printf '%b' "$_valid")

_decoded_dylib=$(bash "$INSTALL_NAME_TOOL_HELPER" decode "$(trace_arg "$_dylib")")
if [ "$_decoded_dylib" = "$_dylib" ]; then
    echo "  PASS: shared helper decodes workflow trace operands"
    PASS=$((PASS + 1))
else
    echo "  FAIL: shared helper changed a decoded workflow trace operand"
    FAIL=$((FAIL + 1))
fi
if [ "$(cat "$ROOT/trace.log")" = "$_expected_trace" ] \
   && [ "$(cat "$ROOT/real-tool.log")" = "-id $_dylib $_dylib" ]; then
    echo "  PASS: shared passthrough encodes boundaries and executes the real tool"
    PASS=$((PASS + 1))
else
    echo "  FAIL: shared passthrough trace or real-tool argv changed"
    FAIL=$((FAIL + 1))
fi

echo "=== CLT-absent assertion ==="
expect_rc "empty trace proves no Apple shim escape" 0 nodylibtool ""
expect_rc "any install_name_tool hit fails the absent leg" 1 nodylibtool "$_valid"

echo "=== CLT-present exact allow-list ==="
expect_rc "exact libpython self-ID patch is accepted, including spaces" 0 dylibpatch "$_valid"
expect_rc "control must actually observe a patch" 1 dylibpatch ""
expect_rc "wrong argc is rejected" 1 dylibpatch "$(trace_record 2 -id "$_dylib" "$_dylib")"
expect_rc "wrong operation is rejected" 1 dylibpatch "$(trace_record 3 -change "$_dylib" "$_dylib")"
expect_rc "different source and destination are rejected" 1 dylibpatch \
    "$(trace_record 3 -id "$_dylib" /tmp/libpython3.12.dylib)"
expect_rc "non-libpython dylib is rejected" 1 dylibpatch \
    "$(trace_record 3 -id /tmp/lib/libtorch.dylib /tmp/lib/libtorch.dylib)"
expect_rc "absolute libpython outside uv's managed-Python root is rejected" 1 dylibpatch \
    "$(trace_record 3 -id /tmp/lib/libpython3.12.dylib /tmp/lib/libpython3.12.dylib)"
expect_rc "path traversal outside uv's managed-Python root is rejected" 1 dylibpatch \
    "$(trace_record 3 -id "$_traversal" "$_traversal")"
expect_rc "relative libpython path is rejected" 1 dylibpatch \
    "$(trace_record 3 -id lib/libpython3.12.dylib lib/libpython3.12.dylib)"
expect_rc "empty extra argument remains visible and is rejected" 1 dylibpatch \
    "$(trace_record 4 -id "$_dylib" "$_dylib" "")"
expect_rc "tab inside an operand cannot forge argument boundaries" 1 dylibpatch \
    "$(trace_record 3 -id "$_dylib" "$_dylib"$'\tignored')"
expect_rc "legacy unescaped records are rejected" 1 dylibpatch \
    "install_name_tool\t3\t-id\t$_dylib\t$_dylib\n"
expect_rc "mixed valid and invalid records still fail" 1 dylibpatch \
    "${_valid}$(trace_record 3 -delete_rpath "$_dylib" "$_dylib")"

echo "=== general no-toolchain assertion ==="
expect_rc "notools permits the exact optional uv patch" 0 notools "$_valid"
expect_rc "notools rejects arbitrary install_name_tool use" 1 notools \
    "$(trace_record 3 -add_rpath /tmp/a /tmp/a)"
expect_rc "notools still rejects git" 1 notools "git\t--version\n"
expect_rc "notools still permits the xcode-select availability probe" 0 notools \
    "xcode-select\t-p\n"

echo ""
echo "Passed: $PASS, Failed: $FAIL"
[ "$FAIL" -eq 0 ]
