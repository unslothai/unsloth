#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the Linux tool-sandbox step in install.sh: bubblewrap is optional, so it is installed
# only when that needs no elevation, and otherwise the one command that enables it is printed.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_contains() {
    if echo "$2" | grep -qF "$3"; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (expected to find '$3')"
        echo "$2" | sed 's/^/  | /'; FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    if echo "$2" | grep -qF "$3"; then
        echo "  FAIL: $1 (found '$3' but should not)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS: $1"; PASS=$((PASS + 1))
    fi
}

_FN_FILE=$(mktemp)
grep '^_BWRAP_APPARMOR_FIX=' "$INSTALL_SH"                  >  "$_FN_FILE"
sed -n '/^_bwrap_install_command()/,/^}/p'   "$INSTALL_SH" >> "$_FN_FILE"
sed -n '/^_check_linux_tool_sandbox()/,/^}/p' "$INSTALL_SH" >> "$_FN_FILE"
if ! grep -q '_check_linux_tool_sandbox()' "$_FN_FILE"; then
    echo "FAIL: could not extract _check_linux_tool_sandbox from install.sh"
    exit 1
fi

_BIN=$(mktemp -d)
_SYSCTL=$(mktemp)
_HARNESS=$(mktemp)
cat > "$_HARNESS" <<HARNESS
C_WARN=''; C_ERR=''; C_OK=''; C_DIM=''; C_RST=''
step()    { echo "STEP \$1 \$2"; }
substep() { echo "SUBSTEP \$1"; }
# Stands in for a root apt: creates bwrap when APT_WORKS=1, as an unprivileged apt-get would not.
_smart_apt_install() {
    echo "APT_CALLED: \$* optional=\${_SMART_APT_OPTIONAL:-false}"
    if [ "\${APT_WORKS:-0}" = 1 ]; then printf '#!/bin/sh\nexit 0\n' > "$_BIN/bwrap"; /bin/chmod +x "$_BIN/bwrap"; fi
    return 2
}
HARNESS

_mk() { printf '#!/bin/sh\n%s\n' "$2" > "$_BIN/$1"; chmod +x "$_BIN/$1"; }
_SH="${BASH:-/bin/bash}"

_run() {
    ( PATH="$_BIN"; export PATH
      "$_SH" -c "_BW_USERNS_SYSCTL='$_SYSCTL'; APT_WORKS=${1:-0}; . '$_HARNESS'; . '$_FN_FILE'; _check_linux_tool_sandbox; echo \"RC=\$?\"" 2>&1 )
}

echo "=== apt host, installer already root: bubblewrap is installed, optionally ==="
rm -f "$_BIN"/*; : > "$_SYSCTL"
_mk apt-get 'exit 0'
_out="$(_run 1)"
assert_contains "asks apt for bubblewrap"           "$_out" "APT_CALLED: bubblewrap"
assert_contains "never as a required package"       "$_out" "optional=true"
assert_contains "then reports the sandbox works"    "$_out" "bubblewrap works"
assert_contains "never fails the install"           "$_out" "RC=0"

echo "=== apt host, ordinary user: no prompt, the exact command is printed ==="
rm -f "$_BIN"/*
_mk apt-get 'exit 0'
_out="$(_run 0)"
assert_contains "warns about software safeguards"   "$_out" "bubblewrap not installed"
assert_contains "names the apt command"             "$_out" "sudo apt-get install -y bubblewrap"
assert_contains "never fails the install"           "$_out" "RC=0"

echo "=== dnf host: no apt call, the dnf command is printed ==="
rm -f "$_BIN"/*
_mk dnf 'exit 0'
_out="$(_run 0)"
assert_not_contains "does not reach for apt"        "$_out" "APT_CALLED"
assert_contains "names the dnf command"             "$_out" "sudo dnf install -y bubblewrap"

echo "=== pacman host ==="
rm -f "$_BIN"/*
_mk pacman 'exit 0'
_out="$(_run 0)"
assert_contains "names the pacman command"          "$_out" "sudo pacman -S --needed bubblewrap"

echo "=== unknown package manager ==="
rm -f "$_BIN"/*
_out="$(_run 0)"
assert_contains "falls back to generic advice"      "$_out" "install bubblewrap with your package manager"

echo "=== bubblewrap present and working: nothing to install ==="
rm -f "$_BIN"/*
_mk apt-get 'exit 0'
_mk bwrap 'exit 0'
_out="$(_run 0)"
assert_not_contains "no apt call"                   "$_out" "APT_CALLED"
assert_contains "reports it works"                  "$_out" "bubblewrap works"

echo "=== bubblewrap present, AppArmor restricts user namespaces ==="
rm -f "$_BIN"/*
_mk bwrap 'echo "bwrap: setting up uid map: Permission denied" >&2; exit 1'
echo 1 > "$_SYSCTL"
_out="$(_run 0)"
assert_contains "names AppArmor"                    "$_out" "AppArmor blocks bubblewrap"
assert_contains "gives Ubuntu's own profile"        "$_out" "extra-profiles/bwrap-userns-restrict /etc/apparmor.d/"
assert_contains "and loads it"                      "$_out" "apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict"
assert_contains "never fails the install"           "$_out" "RC=0"

echo "=== bubblewrap present, blocked without AppArmor (a container) ==="
: > "$_SYSCTL"
_out="$(_run 0)"
assert_contains "says why"                          "$_out" "cannot create a sandbox here"
assert_not_contains "does not blame AppArmor"       "$_out" "AppArmor blocks"

echo "=== the Linux branch runs the step without letting it fail the install ==="
_case="$(sed -n '/^case "\$OS" in/,/^esac/p' "$INSTALL_SH")"
assert_contains "called after the deps gate"        "$_case" "_check_linux_tool_sandbox || true"

rm -rf "$_BIN" "$_FN_FILE" "$_HARNESS" "$_SYSCTL"
echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
