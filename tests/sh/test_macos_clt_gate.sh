#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the macOS system-dependency gate in install.sh.
#
# History: the gate was inline top-level code running
#   xcode-select -p || { xcode-select --install; exit 1; }
# so a brand-new Mac could not install at all, and being inline rather than a function
# it was out of reach of the tests/sh sed-extraction convention that would have caught
# it.
#
# The contract now: a consumer install must SUCCEED with no Xcode Command Line Tools
# (uv, CPython, llama.cpp/whisper.cpp/Node are all prebuilt, triton is skipped on
# macOS), while `--local` must still fail loudly: unsloth-zoo comes from a git+https
# URL.
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

assert_not_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  FAIL: $_label (found '$_needle' but should not)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    fi
}

# ── Extract the functions under test ──
_FN_FILE=$(mktemp)
sed -n '/^_has_working_git()/,/^}/p'   "$INSTALL_SH" >  "$_FN_FILE"
sed -n '/^_check_macos_deps()/,/^}/p'  "$INSTALL_SH" >> "$_FN_FILE"

if ! grep -q '_check_macos_deps()' "$_FN_FILE"; then
    echo "FAIL: could not extract _check_macos_deps from install.sh"
    echo "      (the gate must stay a top-level function so this test can reach it)"
    exit 1
fi

# Minimal harness: the output helpers install.sh would otherwise provide.
_HARNESS=$(mktemp)
cat > "$_HARNESS" <<'HARNESS'
C_WARN=''; C_ERR=''; C_OK=''; C_DIM=''; C_RST=''
step()    { echo "STEP $1 $2"; }
substep() { echo "SUBSTEP $1"; }
tauri_log() { echo "[TAURI:$1] $2"; }
HARNESS

_BIN=$(mktemp -d)

# Each tool is absent, a working stub, or a broken stub mimicking the Xcode CLT shim
# (exists, exits non-zero).
_mk() { printf '#!/bin/sh\n%s\n' "$2" > "$_BIN/$1"; chmod +x "$_BIN/$1"; }

# PATH is the sandbox and ONLY the sandbox, so unstocked tools are genuinely absent and
# the host's /usr/bin/git cannot leak in. bash must therefore be invoked absolutely.
_SH="${BASH:-/bin/bash}"

_run_gate() {
    # $1 = STUDIO_LOCAL_INSTALL
    ( PATH="$_BIN"; export PATH
      "$_SH" -c ". '$_HARNESS'; . '$_FN_FILE'; STUDIO_LOCAL_INSTALL=$1; _check_macos_deps; echo \"RC=\$?\"" 2>&1 )
}

echo "=== clean Mac: no CLT at all (xcode-select missing) ==="
rm -f "$_BIN"/*
_out="$(_run_gate false)"
assert_contains "does not exit 1"                    "$_out" "RC=0"
assert_contains "says CLT are not required"          "$_out" "not required"
assert_not_contains "never claims CLT are required"  "$_out" "are required"

echo "=== clean Mac: CLT stubs present but non-functional (the real virgin-Mac shape) ==="
# With no CLT, /usr/bin/git EXISTS and fails when run, so `command -v git` succeeds.
# The gate must not be fooled by that.
rm -f "$_BIN"/*
_mk xcode-select 'exit 1'
_mk git 'echo "xcrun: error: invalid active developer path" >&2; exit 1'
_out="$(_run_gate false)"
assert_contains "consumer install proceeds"          "$_out" "RC=0"
assert_contains "reports CLT absent but optional"    "$_out" "not required"

echo "=== --local with a non-functional git: must fail loudly ==="
_out="$(_run_gate true)"
assert_contains "fails"                              "$_out" "RC=1"
assert_contains "explains why git is needed"         "$_out" "unsloth-zoo"
assert_contains "names the remedy"                   "$_out" "xcode-select --install"
assert_contains "emits a machine-readable marker"    "$_out" "[TAURI:NEED_XCODE_CLT]"
assert_contains "says a normal install needs none"   "$_out" "non---local"

echo "=== --local with a working git: proceeds ==="
rm -f "$_BIN"/*
_mk xcode-select 'exit 1'
_mk git 'echo "git version 2.50.0"; exit 0'
_out="$(_run_gate true)"
assert_contains "--local proceeds when git works"    "$_out" "RC=0"

echo "=== CLT installed + cmake present ==="
rm -f "$_BIN"/*
_mk xcode-select 'echo /Library/Developer/CommandLineTools; exit 0'
_mk git 'echo "git version 2.50.0"; exit 0'
_mk cmake 'echo "cmake version 3.30.0"; exit 0'
_out="$(_run_gate false)"
assert_contains "all deps found"                     "$_out" "all system dependencies found"
assert_contains "rc 0"                               "$_out" "RC=0"

echo "=== CLT installed, cmake missing: prebuilt path, not fatal ==="
rm -f "$_BIN"/*
_mk xcode-select 'echo /Library/Developer/CommandLineTools; exit 0'
_mk git 'echo "git version 2.50.0"; exit 0'
_out="$(_run_gate false)"
assert_contains "uses prebuilt llama.cpp"            "$_out" "using prebuilt llama.cpp"
assert_contains "rc 0"                               "$_out" "RC=0"

echo "=== the gate never fires the GUI installer on the consumer path ==="
# The dialog needs a GUI session a curl-piped or Tauri-spawned install does not have.
rm -f "$_BIN"/*
_mk xcode-select 'if [ "$1" = "--install" ]; then echo "GUI-DIALOG-FIRED"; fi; exit 1'
_out="$(_run_gate false)"
assert_not_contains "no GUI dialog on consumer path" "$_out" "GUI-DIALOG-FIRED"

echo "=== _has_working_git distinguishes present-but-broken from working ==="
rm -f "$_BIN"/*
_mk git 'exit 1'
_r="$(PATH="$_BIN" "$_SH" -c ". '$_FN_FILE'; _has_working_git && echo yes || echo no")"
assert_eq "broken git stub -> no" "no" "$_r"
_mk git 'echo ok; exit 0'
_r="$(PATH="$_BIN" "$_SH" -c ". '$_FN_FILE'; _has_working_git && echo yes || echo no")"
assert_eq "working git -> yes" "yes" "$_r"
rm -f "$_BIN"/git
_r="$(PATH="$_BIN" "$_SH" -c ". '$_FN_FILE'; _has_working_git && echo yes || echo no")"
assert_eq "absent git -> no" "no" "$_r"

rm -rf "$_BIN" "$_FN_FILE" "$_HARNESS"

echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
