#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A consumer install must succeed with no Xcode Command Line Tools, while `--local` must still
# fail loudly: unsloth-zoo comes from a git+https URL.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
_FN_FILE=$(mktemp)
sed -n '/^_has_working_git()/,/^}/p'   "$INSTALL_SH" >  "$_FN_FILE"
sed -n '/^_check_macos_deps()/,/^}/p'  "$INSTALL_SH" >> "$_FN_FILE"

if ! grep -q '_check_macos_deps()' "$_FN_FILE"; then
    echo "FAIL: could not extract _check_macos_deps from install.sh"
    echo "      (the gate must stay a top-level function so this test can reach it)"
    exit 1
fi

_HARNESS=$(mktemp)
cat > "$_HARNESS" <<'HARNESS'
C_WARN=''; C_ERR=''; C_OK=''; C_DIM=''; C_RST=''
step()    { echo "STEP $1 $2"; }
substep() { echo "SUBSTEP $1"; }
tauri_log() { echo "[TAURI:$1] $2"; }
HARNESS

_BIN=$(mktemp -d)

# Each tool is absent, a working stub, or a broken stub like the CLT shim (exists, exits non-zero).
_mk() { printf '#!/bin/sh\n%s\n' "$2" > "$_BIN/$1"; chmod +x "$_BIN/$1"; }

# PATH is ONLY the sandbox so host /usr/bin/git cannot leak in; bash is invoked absolutely.
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
# With no CLT, /usr/bin/git exists and fails when run, so `command -v git` succeeds.
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
# The dialog needs a GUI session a piped or Tauri-spawned install lacks.
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

# On macOS the probe must never execute /usr/bin/git: the CLT shim raises a GUI dialog.
# The stub records execution, so an empty marker file proves it stayed unrun.
echo "=== macOS: the CLT git shim is never executed ==="
rm -f "$_BIN"/*
_RAN="$(mktemp -u)"
rm -f "$_RAN"
_mk git "echo ran >> '$_RAN'; exit 1"
_mk xcode-select 'exit 1'          # no toolchain selected == a clean Mac
_r="$(PATH="$_BIN" OS=macos _CLT_GIT_SHIM="$_BIN/git" "$_SH" -c \
    ". '$_FN_FILE'; _has_working_git && echo yes || echo no")"
assert_eq "clean Mac + shim git -> no" "no" "$_r"
if [ -f "$_RAN" ]; then
    assert_eq "shim git was NOT executed (no GUI dialog)" "not-executed" "executed"
else
    assert_eq "shim git was NOT executed (no GUI dialog)" "not-executed" "not-executed"
fi

# Intel runners can have a real /usr/bin/git without CLT selected, so real Intel probes it.
rm -f "$_RAN"
_mk git "echo ran >> '$_RAN'; exit 0"
_r="$(PATH="$_BIN" OS=macos MAC_INTEL=true _CLT_GIT_SHIM="$_BIN/git" "$_SH" -c \
    ". '$_FN_FILE'; _has_working_git && echo yes || echo no")"
assert_eq "Intel Mac + working system git -> yes" "yes" "$_r"
if [ -f "$_RAN" ]; then
    assert_eq "Intel system git WAS executed" "executed" "executed"
else
    assert_eq "Intel system git WAS executed" "executed" "not-executed"
fi

# Under Rosetta MAC_INTEL is set but /usr/bin/git is still the shim: _MAC_ROSETTA wins.
rm -f "$_RAN"
_mk git "echo ran >> '$_RAN'; exit 1"
_r="$(PATH="$_BIN" OS=macos MAC_INTEL=true _MAC_ROSETTA=true _CLT_GIT_SHIM="$_BIN/git" "$_SH" -c \
    ". '$_FN_FILE'; _has_working_git && echo yes || echo no")"
assert_eq "Rosetta on Apple Silicon + shim git -> no" "no" "$_r"
if [ -f "$_RAN" ]; then
    assert_eq "Rosetta shim git was NOT executed (no GUI dialog)" "not-executed" "executed"
else
    assert_eq "Rosetta shim git was NOT executed (no GUI dialog)" "not-executed" "not-executed"
fi

# Pin the _MAC_ROSETTA assignment: a rename would silently fall back to executing the shim.
assert_contains "install.sh sets _MAC_ROSETTA when sysctl reports arm64 hardware" \
    "$(sed -n '/^MAC_INTEL=false$/,/^fi$/p' "$INSTALL_SH")" "_MAC_ROSETTA=true"

# A real git elsewhere on PATH (Homebrew) is still probed by executing it.
rm -f "$_RAN"
_mk git "echo ran >> '$_RAN'; exit 1"
_r="$(PATH="$_BIN" OS=macos _CLT_GIT_SHIM=/usr/bin/git "$_SH" -c \
    ". '$_FN_FILE'; _has_working_git && echo yes || echo no")"
assert_eq "clean Mac + non-shim working git -> probed" "no" "$_r"
if [ -f "$_RAN" ]; then
    assert_eq "non-shim git WAS executed" "executed" "executed"
else
    assert_eq "non-shim git WAS executed" "executed" "not-executed"
fi

rm -rf "$_BIN" "$_FN_FILE" "$_HARNESS"

echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
