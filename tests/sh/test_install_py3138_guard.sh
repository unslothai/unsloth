#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Behavioural test for the Python 3.13.8 venv guard in install.sh (#7803).
#
# 3.13.8 carries a CPython regression (gh-139783) that breaks `import torch`.
# The guard that recreates such a venv used to sit behind an `OS = macos` /
# `_ARCH = arm64` gate, so Linux and WSL -- which default to "3.13" and let uv
# pick the patch -- never reached it and silently degraded Studio to CPU.
#
# The guard block is extracted from install.sh and executed against stubs, so
# these assertions cover behaviour rather than the presence of source text.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _got="$2"; _want="$3"
    if [ "$_got" = "$_want" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (want '$_want', got '$_got')"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF -- "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle' in '$_haystack')"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF -- "$_needle"; then
        echo "  FAIL: $_label (found '$_needle' but should not)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    fi
}

# Extract the guard: from the 3.13.8 constants down to the column-0 `fi` that
# closes `if [ -z "$_USER_PYTHON" ]`. Every `fi` inside the block is indented.
GUARD=$(awk '
    /^_PY_TORCH_BROKEN=/ { found = 1 }
    found                { print }
    found && /^fi$/      { exit }
' "$INSTALL_SH")

if [ -z "$GUARD" ]; then
    echo "  FAIL: could not extract the 3.13.8 guard from install.sh"
    exit 1
fi
echo "=== extracted guard ($(echo "$GUARD" | wc -l | tr -d ' ') lines) ==="

# Runs the guard once in a scratch HOME.
#   $1 OS            $2 _ARCH        $3 venv's reported "arch version"
#   $4 _USER_PYTHON  $5 "ok" if the stub uv can build 3.13.9, else "no-3139"
# Echoes: "<resulting PYTHON_VERSION>|<newline-free log of uv --python specs>"
run_guard() {
    _t=$(mktemp -d)
    (
        set -e
        OS="$1"; _ARCH="$2"; _reported="$3"; _USER_PYTHON="$4"; _uv_mode="$5"
        VENV_DIR="$_t/venv"
        PYTHON_VERSION="3.13"
        CALLS="$_t/calls"
        : > "$CALLS"

        mkdir -p "$VENV_DIR/bin"
        _make_python() {
            printf '#!/bin/sh\necho "%s"\n' "$1" > "$VENV_DIR/bin/python"
            chmod +x "$VENV_DIR/bin/python"
        }
        _make_python "$_reported"

        # Stub for the real installer helper. Records the --python spec it was
        # asked for, refuses 3.13.9 when the scenario says uv is too old, and
        # otherwise "creates" a venv reporting the requested version.
        run_install_cmd() {
            shift  # label
            _spec=""
            while [ $# -gt 0 ]; do
                if [ "$1" = "--python" ]; then _spec="$2"; fi
                shift
            done
            echo "$_spec" >> "$CALLS"
            if [ "$_uv_mode" = "no-3139" ] && case "$_spec" in *3.13.9*) true ;; *) false ;; esac; then
                return 1
            fi
            mkdir -p "$VENV_DIR/bin"
            case "$_spec" in
                *3.13.9*) _make_python "x86_64 3.13.9" ;;
                *3.12*)   _make_python "x86_64 3.12.10" ;;
                *)        _make_python "x86_64 3.13.8" ;;
            esac
            return 0
        }

        # The guard narrates to stdout; only the summary line below is parsed.
        # shellcheck disable=SC2034
        eval "$GUARD" > /dev/null

        printf '%s|%s\n' "$PYTHON_VERSION" "$(tr '\n' ' ' < "$CALLS")"
    )
    _rc=$?
    rm -rf "$_t"
    return $_rc
}

echo ""
echo "=== Linux/WSL on 3.13.8: the guard must fire (the #7803 regression) ==="
out=$(run_guard linux x86_64 "x86_64 3.13.8" "" ok)
ver=${out%%|*}; calls=${out#*|}
assert_eq        "linux: venv is rebuilt on 3.13.9, not 3.12" "$ver" "3.13.9"
assert_contains  "linux: uv asked for a plain 3.13.9 spec"    "$calls" "3.13.9"
assert_not_contains \
    "linux: no macOS-only interpreter spec leaks onto Linux" "$calls" "macos-aarch64"
assert_not_contains "linux: does not fall back to 3.12 when 3.13.9 exists" "$calls" "3.12"

echo ""
echo "=== Linux/WSL on 3.13.8 with a uv too old for 3.13.9 ==="
out=$(run_guard linux x86_64 "x86_64 3.13.8" "" no-3139)
ver=${out%%|*}; calls=${out#*|}
assert_eq       "stale uv: install still completes on the 3.12 fallback" "$ver" "3.12"
assert_contains "stale uv: 3.13.9 is attempted first" "$calls" "3.13.9"
assert_contains "stale uv: 3.12 is attempted second" "$calls" "3.12"

echo ""
echo "=== Apple Silicon keeps its arch-explicit spec ==="
out=$(run_guard macos arm64 "arm64 3.13.8" "" ok)
ver=${out%%|*}; calls=${out#*|}
assert_eq       "macos arm64: rebuilt on 3.13.9" "$ver" "3.13.9"
assert_contains "macos arm64: spec stays arch-pinned" "$calls" "cpython-3.13.9-macos-aarch64-none"

echo ""
echo "=== The guard stays out of the way when it should ==="
out=$(run_guard linux x86_64 "x86_64 3.13.9" "" ok)
ver=${out%%|*}; calls=${out#*|}
assert_eq "healthy 3.13.9 venv: left alone" "$ver" "3.13"
assert_eq "healthy 3.13.9 venv: uv not called" "$(echo "$calls" | tr -d ' ')" ""

out=$(run_guard linux x86_64 "x86_64 3.13.8" "3.13.8" ok)
ver=${out%%|*}; calls=${out#*|}
assert_eq "--python override: user's choice is honoured, guard skipped" \
    "$(echo "$calls" | tr -d ' ')" ""

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "ALL PASSED"
