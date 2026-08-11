#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _uv_venv_arm64: asks uv for a managed CPython only, so uv does not execute every
# python on PATH (which is the Xcode CLT dialog on a Mac without the tools), and
# falls back to the unflagged request so an offline host keeps its system Python.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

_FN=$(mktemp)
sed -n '/^_uv_venv_arm64()/,/^}/p' "$INSTALL_SH" > "$_FN"
[ -s "$_FN" ] || { echo "  FAIL: _uv_venv_arm64 not found in install.sh"; exit 1; }

# $1 = shell, $2 = exit code the only-managed attempt returns. The stub logs the
# flags it was handed so ordering and fallback are both visible in one trace.
_run() {
    "$1" -c '
        . "'"$_FN"'"
        VENV_DIR=/tmp/venv; PYTHON_VERSION=3.12
        run_install_cmd() {
            shift
            case " $* " in
                *" only-managed "*) echo "managed"; return '"$2"' ;;
            esac
            echo "unflagged"; return 0
        }
        _uv_venv_arm64 "create venv" && echo "rc=0" || echo "rc=$?"
    ' 2>&1 | tr '\n' ' '
}

for _sh in sh bash; do
    echo "=== _uv_venv_arm64 under $_sh ==="

    _out=$(_run "$_sh" 0)
    assert_eq "managed request succeeds, no fallback" "managed rc=0 " "$_out"

    _out=$(_run "$_sh" 2)
    assert_eq "managed request fails, falls back unflagged" "managed unflagged rc=0 " "$_out"

    # Both attempts failing must still surface non-zero: the caller runs under
    # set -e and the rollback trap has to fire.
    _out=$("$_sh" -c '
        . "'"$_FN"'"
        VENV_DIR=/tmp/venv; PYTHON_VERSION=3.12
        run_install_cmd() { return 2; }
        _uv_venv_arm64 "create venv" && echo "rc=0" || echo "rc=$?"
    ' 2>&1)
    assert_eq "both attempts fail, non-zero propagates" "rc=2" "$_out"
done

echo "=== install.sh call sites ==="

# PYTHON_VERSION is re-assigned to 3.12 before the last call site, so the helper
# has to read it at call time rather than capture it.
assert_contains "helper expands PYTHON_VERSION at call time" \
    "$(cat "$_FN")" 'cpython-${PYTHON_VERSION}-macos-aarch64-none'

_direct=$(grep -c 'uv venv .*cpython-\${PYTHON_VERSION}-macos-aarch64-none' "$INSTALL_SH" || true)
assert_eq "arm64 sites go through the helper only" "0" "$_direct"

_calls=$(grep -c '^ *_uv_venv_arm64 ' "$INSTALL_SH" || true)
assert_eq "all three arm64 venv sites routed" "3" "$_calls"

# The non-arm64 branch takes _python_request, which carries a user --python or an
# explicit interpreter path. only-managed would ignore an interpreter they asked for.
_out=$(grep -A 1 '_python_request "\$PYTHON_VERSION"' "$INSTALL_SH" || true)
if echo "$_out" | grep -q 'only-managed'; then
    echo "  FAIL: only-managed leaked onto a _python_request call site"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: _python_request call sites left unflagged"
    PASS=$((PASS + 1))
fi

rm -f "$_FN"
echo ""
echo "Passed: $PASS, Failed: $FAIL"
[ "$FAIL" -eq 0 ]
