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

# A unix socket is the closest portable stand-in for the /dev/tty found inside
# containers and systemd units: the mode bits satisfy `test -r`, but open()
# fails with ENXIO. Callers must verify the shape before relying on it.
make_unopenable() {
    python3 -c 'import socket,sys; socket.socket(socket.AF_UNIX).bind(sys.argv[1])' \
        "$1" 2>/dev/null
}

# $1 tty:  "tty" | "notty" | "unopenable"
# $2 sudo: "nopasswd" | "needspasswd" | "aptneedspasswd" | "absent"
run_smart() {
    _tty_mode="$1"; _sudo_mode="$2"
    _d=$(mktemp -d -p "$_TMP_ROOT")
    case "$_tty_mode" in
        tty)        printf 'y\n' > "$_d/tty" ;;
        unopenable) make_unopenable "$_d/tty" ;;
    esac

    _f=$(mktemp -p "$_TMP_ROOT")
    sed -n -e '/^_can_read_tty()/,/^}/p' \
           -e '/^_sudo_runs_unattended()/,/^}/p' \
           -e '/^_smart_apt_install()/,/^}/p' "$INSTALL_SH" \
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
                case "$_sudo_mode" in
                    nopasswd) return 0 ;;
                    # Trivial commands are NOPASSWD but apt-get is not, which is
                    # exactly what a `sudo -n true` probe reads as unattended.
                    aptneedspasswd)
                        case " $* " in
                            *" apt-get "*) return 1 ;;
                            *) return 0 ;;
                        esac
                        ;;
                    *) return 1 ;;
                esac
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

# A /dev/tty that passes `test -r` but cannot be opened must count as no tty.
# Only assert when this platform can actually produce that shape.
_probe=$(mktemp -d -p "$_TMP_ROOT")
if make_unopenable "$_probe/tty" && [ -r "$_probe/tty" ] && ! ( : <"$_probe/tty" ) 2>/dev/null; then
    _out=$(run_smart unopenable needspasswd)
    assert_contains "unopenable tty: treated as no tty" "$_out" "cannot be done unattended"
    case "$_out" in
        *"Accept? [Y/n]"*) echo "  FAIL: unopenable tty must not print a prompt"; FAIL=$((FAIL + 1)) ;;
        *) echo "  PASS: unopenable tty prints no prompt"; PASS=$((PASS + 1)) ;;
    esac
else
    echo "  SKIP: this platform cannot fake a readable-but-unopenable /dev/tty"
fi

# NOPASSWD on trivial commands but not on apt-get: the case `sudo -n true` got
# wrong. Must fall back to the actionable message, not blind escalation.
_out=$(run_smart notty aptneedspasswd)
assert_contains "apt-get needs a password: says it cannot run unattended" \
    "$_out" "cannot be done unattended"
case "$_out" in
    *SUDO_RAN*) echo "  FAIL: apt-get needing a password must not invoke sudo"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: apt-get needing a password does not invoke sudo"; PASS=$((PASS + 1)) ;;
esac

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
