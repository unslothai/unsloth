#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the Linux/WSL system-dependency gate in install.sh.
#
# History: the gate hard-required cmake, git, gcc and libcurl4-openssl-dev, installing
# them on apt distros and `exit 1`-ing everywhere else. Nothing on the consumer path
# builds anything, so it stranded every non-apt distro over unused tooling.
#
# The contract now: only a download transport (curl or wget) is fatal, build tooling
# is a warning, and git is required for --local only (unsloth-zoo git+https URL).
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
# Records its args so a test can tell "asked apt for curl" from "asked for everything".
_smart_apt_install() { echo "APT_CALLED: $*"; }
HARNESS

_BIN=$(mktemp -d)
_mk() { printf '#!/bin/sh\n%s\n' "$2" > "$_BIN/$1"; chmod +x "$_BIN/$1"; }

# PATH is the sandbox and ONLY the sandbox, so unstocked tools are genuinely absent and
# the host's /usr/bin/cmake cannot leak in. bash must therefore be invoked absolutely.
_SH="${BASH:-/bin/bash}"

_run_gate() {
    # $1 = STUDIO_LOCAL_INSTALL
    ( PATH="$_BIN"; export PATH
      "$_SH" -c ". '$_HARNESS'; . '$_FN_FILE'; STUDIO_LOCAL_INSTALL=$1; _check_linux_deps; echo \"RC=\$?\"" 2>&1 )
}

echo "=== Fedora/Arch/openSUSE shape: curl present, no build tooling, no apt ==="
# Used to exit 1 with "supported on apt-based Linux distributions only".
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
# Build tooling still appears in the warning line, so match the apt call, not names.
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

echo "=== apt present: git is auto-installed, because triton_kernels needs it ==="
# Regression: making git optional without this failed at "6/14 triton kernels", whose
# requirement is a git+https URL.
rm -f "$_BIN"/*
_mk curl 'exit 0'
_mk apt-get 'exit 0'
_out="$(_run_gate false)"
assert_contains "install proceeds"                       "$_out" "RC=0"
assert_contains "apt is asked for git"                   "$_out" "git"
assert_contains "apt is actually called"                 "$_out" "APT_CALLED"

echo "=== no apt and no git: warn about the triton skip, do not fail ==="
rm -f "$_BIN"/*
_mk curl 'exit 0'
_out="$(_run_gate false)"
assert_contains "install proceeds"                       "$_out" "RC=0"
assert_contains "names the consequence of no git"        "$_out" "triton kernels"
assert_not_contains "does not call it required to run"   "$_out" "is required"

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

echo "=== optional apt packages never ask for elevation, in any mode ==="
# Regression: the optional bypass sat inside the TAURI_MODE branch, so a plain
# `curl | sh` on a non-root Debian box still hit the sudo prompt (default yes) and
# installed cmake, GCC and dev headers that nothing on the consumer path uses.
_APT_FN=$(mktemp)
{
    sed -n '/^_is_pkg_installed()/,/^}$/p'       "$INSTALL_SH"
    sed -n '/^_apt_distro_description()/,/^}$/p' "$INSTALL_SH"
    sed -n '/^_can_read_tty()/,/^}$/p'           "$INSTALL_SH"
    sed -n '/^_smart_apt_install()/,/^}$/p'      "$INSTALL_SH"
} > "$_APT_FN"

_run_apt() {
    # $1 = TAURI_MODE, $2 = _SMART_APT_OPTIONAL. apt-get always fails, as it does
    # for a non-root user, so the function reaches its escalation decision.
    rm -f "$_BIN"/*
    _mk apt-get 'exit 100'
    _mk sudo 'echo "ELEVATION_ATTEMPTED: $*"; exit 1'
    ln -sf "$(command -v sed)" "$_BIN/sed"   # the function trims its list with sed
    # _APT_FN after _HARNESS so the real function replaces the recording stub.
    ( PATH="$_BIN"; export PATH
      "$_SH" -c ". '$_HARNESS'; . '$_APT_FN'; TAURI_MODE=$1; _SMART_APT_OPTIONAL=$2
                ( _smart_apt_install unsloth_absent_pkg ); echo \"RC=\$?\"" 2>&1 )
}

_out="$(_run_apt false true)"
assert_contains "optional: returns 2 so the caller can continue" "$_out" "RC=2"
assert_not_contains "optional: no sudo prompt"                   "$_out" "elevated permissions"
assert_not_contains "optional: sudo never invoked"               "$_out" "ELEVATION_ATTEMPTED"

_out="$(_run_apt true true)"
assert_contains "optional in Tauri: returns 2"                   "$_out" "RC=2"
assert_not_contains "optional in Tauri: no NEED_SUDO dialog"     "$_out" "NEED_SUDO"

_out="$(_run_apt false false)"
assert_contains "required: still escalates"                      "$_out" "ELEVATION_ATTEMPTED"

_out="$(_run_apt true false)"
assert_contains "required in Tauri: still asks Rust to elevate"  "$_out" "NEED_SUDO"

rm -f "$_APT_FN"
rm -rf "$_BIN" "$_FN_FILE" "$_HARNESS"
echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
