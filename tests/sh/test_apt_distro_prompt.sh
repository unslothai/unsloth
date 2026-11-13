#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's _apt_distro_description. Hermetic: the helper is extracted
# with /etc/os-release rewritten to per-test fixtures.
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

# NAME alone (no VERSION_ID): still prefer NAME over PRETTY_NAME.
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

# No-TTY escalation: /dev/tty is rewritten to a fixture so every TTY state is hermetic.
echo "=== _smart_apt_install no-TTY escalation ==="

# Mimics the container/systemd /dev/tty: passes `test -r` but open() fails with ENXIO.
make_unopenable() {
    python3 -c 'import socket,sys; socket.socket(socket.AF_UNIX).bind(sys.argv[1])' \
        "$1" 2>/dev/null
}

# $1 tty:  "tty" | "notty" | "unopenable"
# $2 sudo: "nopasswd" | "needspasswd" | "aptneedspasswd" | "cached" | "absent"
run_smart() {
    _tty_mode="$1"; _sudo_mode="$2"
    _d=$(mktemp -d -p "$_TMP_ROOT")
    case "$_tty_mode" in
        tty)        printf 'y\n' > "$_d/tty" ;;
        # Opens but reads EOF straight away: openable is not answerable.
        eof)        : > "$_d/tty" ;;
        unopenable) make_unopenable "$_d/tty" ;;
    esac

    _f=$(mktemp -p "$_TMP_ROOT")
    sed -n -e '/^_can_read_tty()/,/^}/p' \
           -e '/^_smart_apt_install()/,/^}/p' "$INSTALL_SH" \
        | sed -e "s#/dev/tty#$_d/tty#g" > "$_f"

    (
        TAURI_MODE=false
        _apt_distro_description() { echo "TestOS 1.0 (debian-like)"; }
        _is_pkg_installed() { return 1; }
        apt-get() { return 1; }
        command() {
            if [ "$1" = -v ] && [ "$2" = sudo ]; then
                [ "$_sudo_mode" != absent ]; return $?
            fi
            builtin command "$@"
        }
        # -n refuses when a password is needed; -k ignores cached timestamps (see the man page).
        sudo() {
            _noninteractive=false
            _ignore_cache=false
            while :; do
                case "$1" in
                    -n) _noninteractive=true; shift ;;
                    -k) _ignore_cache=true; shift ;;
                    *)  break ;;
                esac
            done
            if [ "$_noninteractive" = true ]; then
                case "$_sudo_mode" in
                    nopasswd) ;;
                    cached) [ "$_ignore_cache" = true ] && return 1 ;;
                    # Authorized, but NOPASSWD only on trivial commands: list mode says yes,
                    # execution still needs a password.
                    aptneedspasswd)
                        case " $* " in
                            *" apt-get "*) return 1 ;;
                        esac
                        ;;
                    *) return 1 ;;
                esac
            fi
            if [ "$_sudo_mode" = denied ]; then
                echo "sudo: user is not allowed to execute that" >&2
                return 1
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
    *SUDO_RAN*) echo "  FAIL: no tty + password sudo must not run apt-get as root"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: no tty + password sudo runs nothing as root"; PASS=$((PASS + 1)) ;;
esac
case "$_out" in
    *"Accept? [Y/n]"*) echo "  FAIL: must not print an unanswerable prompt"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: no dangling Accept? prompt without a tty"; PASS=$((PASS + 1)) ;;
esac

# Passwordless escalation is the one case where an unattended install is legitimate.
_out=$(run_smart notty nopasswd)
assert_contains "no tty + passwordless sudo: still installs" "$_out" "SUDO_RAN: apt-get install -y cmake"
assert_contains "no tty + passwordless sudo: says why it proceeded" \
    "$_out" "passwordless sudo"

_out=$(run_smart tty needspasswd)
assert_contains "tty present: still prompts" "$_out" "Accept? [Y/n]"
assert_contains "tty present: accepts and installs" "$_out" "SUDO_RAN: apt-get install -y cmake"

# Consent given but elevated apt-get fails: must print the manual command, not die on the error.
_out=$(run_smart tty denied)
assert_contains "tty + denied sudo: gives the manual command" \
    "$_out" "sudo apt-get update -y && sudo apt-get install -y cmake"

_out=$(run_smart notty absent)
assert_contains "no sudo binary: unchanged message" "$_out" "sudo is not available on this system"

# Only assert where the platform can produce the unopenable-tty shape.
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

# A tty that yields EOF must decline: a failed read is nobody answering.
_out=$(run_smart eof needspasswd)
assert_contains "eof tty: declines instead of escalating" \
    "$_out" "Please install these packages first"
case "$_out" in
    *SUDO_RAN*) echo "  FAIL: eof tty must not escalate"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: eof tty runs nothing as root"; PASS=$((PASS + 1)) ;;
esac

# A cached timestamp from an unrelated escalation must not count as passwordless (-k).
_out=$(run_smart notty cached)
assert_contains "cached credentials: says it cannot run unattended" \
    "$_out" "cannot be done unattended"
case "$_out" in
    *SUDO_RAN*) echo "  FAIL: a cached timestamp must not authorise unattended install"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: cached credentials run nothing as root"; PASS=$((PASS + 1)) ;;
esac

# The exit status of the elevated command passes through, so a failure may be apt's, not a password.
assert_contains "failure message does not blame a password exclusively" \
    "$_out" "or apt-get itself"

# Both `-n true` and `-n -l` answer authorization, not authentication; only running the
# command with -n is truthful.
_out=$(run_smart notty aptneedspasswd)
assert_contains "apt-get needs a password: says it cannot run unattended" \
    "$_out" "cannot be done unattended"
case "$_out" in
    *SUDO_RAN*) echo "  FAIL: apt-get needing a password must not run as root"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: apt-get needing a password runs nothing as root"; PASS=$((PASS + 1)) ;;
esac

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
