#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the Linux/WSL system-dependency gate in install.sh.
#
# History: the gate hard-required cmake, git, gcc and libcurl4-openssl-dev "to build
# the GGUF inference engine", auto-installing them on apt distros and `exit 1`-ing
# everywhere else with a Fedora/Arch/openSUSE instruction. Nothing on the consumer
# path builds anything: unslothai/llama.cpp publishes linux-x64 and linux-arm64
# prebuilts covering cpu, cuda12, cuda13, rocm and vulkan. So the gate stranded every
# non-apt distro over tooling that was never used -- the same defect the macOS Xcode
# CLT gate had.
#
# The contract now: only a download transport (curl or wget) is fatal. Build tooling
# is a warning. git is required for --local only, because that mode installs
# unsloth-zoo from a git+https URL.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        echo "  ---- output ----"; echo "$_haystack" | sed 's/^/  | /'
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
sed -n '/^_check_linux_deps()/,/^}/p'  "$INSTALL_SH" >> "$_FN_FILE"

if ! grep -q '_check_linux_deps()' "$_FN_FILE"; then
    echo "FAIL: could not extract _check_linux_deps from install.sh"
    echo "      (the gate must stay a top-level function so this test can reach it)"
    exit 1
fi

_HARNESS=$(mktemp)
cat > "$_HARNESS" <<'HARNESS'
C_WARN=''; C_ERR=''; C_OK=''; C_DIM=''; C_RST=''
step()    { echo "STEP $1 $2"; }
substep() { echo "SUBSTEP $1"; }
tauri_log() { echo "[TAURI:$1] $2"; }
# Stand-in for the real apt path. Records that it was reached and with what, so a
# test can tell "asked apt for curl" from "asked apt for the whole build toolchain".
_smart_apt_install() { echo "APT_CALLED: $*"; }
HARNESS

_BIN=$(mktemp -d)
_mk() { printf '#!/bin/sh\n%s\n' "$2" > "$_BIN/$1"; chmod +x "$_BIN/$1"; }

# PATH is the sandbox and ONLY the sandbox, so an unstocked tool is genuinely absent
# and the host's /usr/bin/cmake cannot turn an "absent" case into a "present" one.
# bash therefore has to be invoked by absolute path.
_SH="${BASH:-/bin/bash}"

_run_gate() {
    # $1 = STUDIO_LOCAL_INSTALL
    ( PATH="$_BIN"; export PATH
      "$_SH" -c ". '$_HARNESS'; . '$_FN_FILE'; STUDIO_LOCAL_INSTALL=$1; _check_linux_deps; echo \"RC=\$?\"" 2>&1 )
}

echo "=== Fedora/Arch/openSUSE shape: curl present, no build tooling, no apt ==="
# This is the case that used to exit 1 with "Automatic system package installation is
# supported on apt-based Linux distributions only".
rm -f "$_BIN"/*
_mk curl 'exit 0'
_out="$(_run_gate false)"
assert_contains "install proceeds"                       "$_out" "RC=0"
assert_contains "says the prebuilt is used"              "$_out" "using prebuilt llama.cpp"
assert_contains "names what is missing"                  "$_out" "cmake"
assert_contains "says it is not required"                "$_out" "Not required"
assert_not_contains "does not demand a package manager"  "$_out" "apt-based"
assert_not_contains "does not reach apt for build tools" "$_out" "APT_CALLED"

echo "=== wget instead of curl is an acceptable transport ==="
rm -f "$_BIN"/*
_mk wget 'exit 0'
_out="$(_run_gate false)"
assert_contains "install proceeds"                       "$_out" "RC=0"
assert_not_contains "does not ask apt for curl"          "$_out" "APT_CALLED"

echo "=== no transport at all, no apt: the one genuinely fatal case ==="
rm -f "$_BIN"/*
_out="$(_run_gate false)"
assert_contains "fails"                                  "$_out" "RC=1"
assert_contains "names the missing transport"            "$_out" "curl"
assert_contains "explains what it is needed for"         "$_out" "download"
assert_contains "gives a non-apt remedy"                 "$_out" "dnf install curl"

echo "=== no transport, apt available: auto-install curl and ONLY curl ==="
rm -f "$_BIN"/*
_mk apt-get 'exit 0'
_out="$(_run_gate false)"
assert_contains "install proceeds"                       "$_out" "RC=0"
assert_contains "asks apt for curl"                      "$_out" "APT_CALLED: curl"
assert_not_contains "does not ask apt for cmake"         "$_out" "APT_CALLED: curl cmake"
# Only the transport goes to apt. The build tooling is still reported as missing in
# the warning line, so match the apt call itself rather than the package names.
assert_contains    "apt asked for exactly curl"          "$_out" "APT_CALLED: curl
"
assert_contains    "build tooling only warned about"     "$_out" "using prebuilt llama.cpp"

echo "=== fully equipped machine: no warnings ==="
rm -f "$_BIN"/*
for t in curl cmake gcc curl-config git; do _mk "$t" 'exit 0'; done
_out="$(_run_gate false)"
assert_contains "install proceeds"                       "$_out" "RC=0"
assert_contains "reports everything found"               "$_out" "all system dependencies found"
assert_not_contains "no prebuilt fallback warning"       "$_out" "using prebuilt llama.cpp"

echo "=== --local without git: must fail loudly (matches macOS) ==="
rm -f "$_BIN"/*
_mk curl 'exit 0'
_out="$(_run_gate true)"
assert_contains "fails"                                  "$_out" "RC=1"
assert_contains "explains why git is needed"             "$_out" "unsloth-zoo"
assert_contains "says a normal install needs none"       "$_out" "non---local"

echo "=== --local with a git that exists but does not work ==="
# Mirrors the macOS CLT-stub shape: `command -v git` succeeds, running it fails.
rm -f "$_BIN"/*
_mk curl 'exit 0'
_mk git 'echo "broken" >&2; exit 1'
_out="$(_run_gate true)"
assert_contains "still fails"                            "$_out" "RC=1"

echo "=== --local with a working git proceeds ==="
rm -f "$_BIN"/*
_mk curl 'exit 0'
_mk git 'exit 0'
_out="$(_run_gate true)"
assert_contains "install proceeds"                       "$_out" "RC=0"

rm -rf "$_BIN" "$_FN_FILE" "$_HARNESS"
echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
