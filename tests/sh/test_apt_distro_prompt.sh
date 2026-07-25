#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's _apt_distro_description helper (#6207).
# The sudo Accept? prompt should name the detected distro and say packages come
# from official apt repos. Hermetic: extract the helper and rewrite
# /etc/os-release to per-test fixtures (same pattern as test_strixhalo_wsl_reroute.sh).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    _label="$1"; _hay="$2"; _needle="$3"
    case "$_hay" in
        *"$_needle"*) echo "  PASS: $_label"; PASS=$((PASS + 1)) ;;
        *) echo "  FAIL: $_label (missing '$_needle' in: $_hay)"; FAIL=$((FAIL + 1)) ;;
    esac
}

# Extract helper with /etc/os-release rewritten to $1.
build_func() {
    _fix="$1"
    _f=$(mktemp -p "$_TMP_ROOT")
    sed -n '/^_apt_distro_description()/,/^}/p' "$INSTALL_SH" \
        | sed -e "s#/etc/os-release#$_fix/os-release#g" \
        > "$_f"
    echo "$_f"
}

run_desc() {
    _os="$1"
    _d=$(mktemp -d -p "$_TMP_ROOT")
    printf '%s\n' "$_os" > "$_d/os-release"
    _f=$(build_func "$_d")
    # shellcheck disable=SC1090
    . "$_f"
    _apt_distro_description
}

echo "=== _apt_distro_description ==="

assert_eq "ubuntu name+version debian-like" \
    "Ubuntu 24.04 (debian-like)" \
    "$(run_desc "$(printf 'NAME=\"Ubuntu\"\nVERSION_ID=\"24.04\"\nID=ubuntu\nID_LIKE=debian\n')")"

assert_eq "debian name+version debian-like" \
    "Debian GNU/Linux 12 (debian-like)" \
    "$(run_desc "$(printf 'NAME=\"Debian GNU/Linux\"\nVERSION_ID=\"12\"\nID=debian\n')")"

assert_eq "pretty_name fallback when name/version missing" \
    "Linux Mint 22 (debian-like)" \
    "$(run_desc "$(printf 'PRETTY_NAME=\"Linux Mint 22\"\nID=linuxmint\nID_LIKE=\"ubuntu debian\"\n')")"

# NAME alone (no VERSION_ID) — still prefer NAME over PRETTY_NAME.
assert_eq "name only" \
    "Pop!_OS (debian-like)" \
    "$(run_desc "$(printf 'NAME=\"Pop!_OS\"\nID=pop\nID_LIKE=\"ubuntu debian\"\n')")"

assert_eq "missing os-release file" \
    "a debian-like system" \
    "$(
        _d=$(mktemp -d -p "$_TMP_ROOT")
        _f=$(build_func "$_d")
        # shellcheck disable=SC1090
        . "$_f"
        _apt_distro_description
    )"

echo "=== _smart_apt_install prompt contract ==="
_smart=$(sed -n '/^_smart_apt_install()/,/^}/p' "$INSTALL_SH")
assert_contains "calls distro helper" "$_smart" '_apt_distro_description'
assert_contains "names detected distro" "$_smart" 'Detected ${_ad_desc}'
assert_contains "mentions apt-get" "$_smart" 'sudo apt-get'
assert_contains "mentions official repos" "$_smart" "official repositories"
assert_contains "rejects tarball worry" "$_smart" "not a third-party tarball"

# ── No-TTY sudo escalation (#7307 Problem 7) ────────────────────────
# The old code assumed consent when /dev/tty was unreadable, then ran sudo with
# stdin closed, so a password-requiring host died on a raw sudo error instead of
# printing the actionable "install these first" message. Drive the real function
# with /dev/tty rewritten to a fixture path, the same trick used for
# /etc/os-release above, so both TTY states are reachable hermetically.
echo "=== _smart_apt_install no-TTY escalation ==="

# $1 tty: "tty" | "notty"   $2 sudo: "nopasswd" | "needspasswd" | "absent"
run_smart() {
    _tty_mode="$1"; _sudo_mode="$2"
    _d=$(mktemp -d -p "$_TMP_ROOT")
    [ "$_tty_mode" = tty ] && printf 'y\n' > "$_d/tty"

    _f=$(mktemp -p "$_TMP_ROOT")
    sed -n '/^_smart_apt_install()/,/^}/p' "$INSTALL_SH" \
        | sed -e "s#/dev/tty#$_d/tty#g" > "$_f"

    (
        TAURI_MODE=false
        _apt_distro_description() { echo "TestOS 1.0 (debian-like)"; }
        _is_pkg_installed() { return 1; }          # nothing ever installs
        apt-get() { return 1; }                    # unprivileged attempt fails
        command() {
            if [ "$1" = -v ] && [ "$2" = sudo ]; then
                [ "$_sudo_mode" != absent ]; return $?
            fi
            builtin command "$@"
        }
        sudo() {
            if [ "$1" = -n ]; then
                [ "$_sudo_mode" = nopasswd ]; return $?
            fi
            echo "SUDO_RAN: $*"
        }
        # shellcheck disable=SC1090
        . "$_f"
        _smart_apt_install cmake 2>&1
        echo "EXIT:$?"
    ) || true
}

_out=$(run_smart notty needspasswd)
assert_contains "no tty + password sudo: says it cannot run unattended" \
    "$_out" "cannot be done unattended"
assert_contains "no tty + password sudo: gives the manual command" \
    "$_out" "sudo apt-get update -y && sudo apt-get install -y cmake"
assert_contains "no tty + password sudo: names the distro" \
    "$_out" "TestOS 1.0 (debian-like)"
case "$_out" in
    *SUDO_RAN*) echo "  FAIL: no tty + password sudo must not invoke sudo"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: no tty + password sudo does not invoke sudo"; PASS=$((PASS + 1)) ;;
esac
case "$_out" in
    *"Accept? [Y/n]"*) echo "  FAIL: must not print an unanswerable prompt"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: no dangling Accept? prompt without a tty"; PASS=$((PASS + 1)) ;;
esac

# Passwordless sudo is the one case where unattended escalation is legitimate.
_out=$(run_smart notty nopasswd)
assert_contains "no tty + passwordless sudo: still installs" "$_out" "SUDO_RAN: apt-get install -y cmake"
assert_contains "no tty + passwordless sudo: says why it proceeded" \
    "$_out" "proceeding with passwordless sudo"

# A readable tty must behave exactly as before: prompt, then honour the answer.
_out=$(run_smart tty needspasswd)
assert_contains "tty present: still prompts" "$_out" "Accept? [Y/n]"
assert_contains "tty present: accepts and installs" "$_out" "SUDO_RAN: apt-get install -y cmake"

# No sudo at all keeps its own message.
_out=$(run_smart notty absent)
assert_contains "no sudo binary: unchanged message" "$_out" "sudo is not available on this system"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
