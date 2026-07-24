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

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
