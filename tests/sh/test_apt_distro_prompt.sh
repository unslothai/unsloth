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
# stdin closed, so a password-requiring host died on a raw sudo error. Drive the
# real function with /dev/tty rewritten to a fixture, the same trick used for
# /etc/os-release above, so every TTY state is reachable hermetically.
echo "=== _smart_apt_install no-TTY escalation ==="

# Closest portable stand-in for the /dev/tty inside containers and systemd
# units: the mode bits satisfy `test -r`, but open() fails with ENXIO. Callers
# must verify the shape before relying on it.
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
        # Opens fine but reads EOF straight away (drained/half-closed
        # terminal): openable is not the same as answerable.
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
        _is_pkg_installed() { return 1; }          # nothing ever installs
        apt-get() { return 1; }                    # unprivileged attempt fails
        command() {
            if [ "$1" = -v ] && [ "$2" = sudo ]; then
                [ "$_sudo_mode" != absent ]; return $?
            fi
            builtin command "$@"
        }
        # Models real sudo: -n refuses (exit 1, nothing runs) when a password
        # would be needed. -k ignores any cached timestamp for this invocation
        # (sudo(8)), so only a real NOPASSWD rule counts as passwordless.
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
                    # A valid timestamp from an earlier, unrelated sudo. Without
                    # -k this looks passwordless; with -k it must not.
                    cached) [ "$_ignore_cache" = true ] && return 1 ;;
                    # Authorized for everything, NOPASSWD only on trivial
                    # commands: `sudo -l` says yes while execution still needs
                    # a password. Authorization is not the question to ask.
                    aptneedspasswd)
                        case " $* " in
                            *" apt-get "*) return 1 ;;
                        esac
                        ;;
                    *) return 1 ;;
                esac
            fi
            # Sudoers refuses the command outright, with or without -n.
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

# Passwordless sudo is the one case where unattended escalation is legitimate.
_out=$(run_smart notty nopasswd)
assert_contains "no tty + passwordless sudo: still installs" "$_out" "SUDO_RAN: apt-get install -y cmake"
assert_contains "no tty + passwordless sudo: says why it proceeded" \
    "$_out" "passwordless sudo"

# A readable tty must behave exactly as before: prompt, then honour the answer.
_out=$(run_smart tty needspasswd)
assert_contains "tty present: still prompts" "$_out" "Accept? [Y/n]"
assert_contains "tty present: accepts and installs" "$_out" "SUDO_RAN: apt-get install -y cmake"

# Consent given at a real tty, but the elevated apt-get fails anyway (sudoers
# denial, wrong password, apt error). The interactive branch must say what to
# run by hand, like the headless branch does, not die on the bare sudo error.
_out=$(run_smart tty denied)
assert_contains "tty + denied sudo: gives the manual command" \
    "$_out" "sudo apt-get update -y && sudo apt-get install -y cmake"

# No sudo at all keeps its own message.
_out=$(run_smart notty absent)
assert_contains "no sudo binary: unchanged message" "$_out" "sudo is not available on this system"

# A /dev/tty that passes `test -r` but cannot be opened counts as no tty.
# Only assert where the platform can actually produce that shape.
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

# A tty that opens but yields EOF must decline: a failed read is nobody
# answering, and calling that "yes" escalates through the branch that does
# have a terminal.
_out=$(run_smart eof needspasswd)
assert_contains "eof tty: declines instead of escalating" \
    "$_out" "Please install these packages first"
case "$_out" in
    *SUDO_RAN*) echo "  FAIL: eof tty must not escalate"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: eof tty runs nothing as root"; PASS=$((PASS + 1)) ;;
esac

# A cached timestamp from an earlier, unrelated sudo must not count as
# passwordless: nobody answered this run's prompt and the apt-get rule still
# carries PASSWD. Asserts the -k is present and effective.
_out=$(run_smart notty cached)
assert_contains "cached credentials: says it cannot run unattended" \
    "$_out" "cannot be done unattended"
case "$_out" in
    *SUDO_RAN*) echo "  FAIL: a cached timestamp must not authorise unattended install"; FAIL=$((FAIL + 1)) ;;
    *) echo "  PASS: cached credentials run nothing as root"; PASS=$((PASS + 1)) ;;
esac

# The failure message must not blame a password when apt itself failed: sudo
# passes the command's own exit status through when the command runs.
assert_contains "failure message does not blame a password exclusively" \
    "$_out" "or apt-get itself"

# Authorized for apt-get but not NOPASSWD on it. Both `sudo -n true` and
# `sudo -n -l -- apt-get ...` read this as unattended, since list mode answers
# authorization, not authentication. Only running it with -n is truthful.
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
