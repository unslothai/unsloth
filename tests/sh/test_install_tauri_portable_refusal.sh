#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: --tauri refusing a portable install must describe the install, not blame
# flags the user did not type.
#
# The refusal predates these PRs and is right: the desktop app resolves ~/.unsloth/studio in
# Rust, so installing over a portable tree would leave it launching a Studio that is not
# there. What changed is how you reach it. Portable mode is now also ADOPTED from markers on
# disk, so a desktop repair -- which passes --tauri and nothing else -- started failing with
# "--portable and --root are not supported with --tauri" on a machine where neither flag was
# supplied.
#
# That is reachable by following the documented instructions: `install.sh --portable` with no
# `--root` selects $HOME/.unsloth, the very directory the desktop app uses. The desktop
# installer also strips UNSLOTH_PORTABLE before spawning the script, so the way out is a
# terminal, and the message has to say that rather than name a flag.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

blockA="$(awk '
    /^# ── Parse flags ──$/ {g = 1}
    g {print}
    /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {s = 1}
    s && /^fi$/ {exit}
' "$INSTALL")"
blockT="$(awk '
    /^# Custom Unsloth roots are unsupported with --tauri unless override == legacy default\.$/ {g = 1}
    g {print}
    g && /^    exit 1$/ {print "fi"; exit}
' "$INSTALL")"
case "$blockA" in *'_ROOT_FROM_FLAG=false'*) : ;; *) echo "FAIL: parser extraction broke"; exit 1 ;; esac
case "$blockT" in
    *'if [ "$_ROOT_FROM_FLAG" = true ]; then'*) : ;;
    *) echo "FAIL: the tauri refusal no longer tells typed from adopted"; exit 1 ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
printf '%s\n' 'substep() { :; }' "$blockA" "$blockT" 'echo REACHED-INSTALL' > "$SNIP"

run() { # home [args...]
    _h="$1"; shift
    env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" sh "$SNIP" "$@" \
        > "$T/out" 2>"$T/err"
    printf '%s' "$?"
}
# -- so a pattern starting with "--" is not read as a grep option. Without it every one of
# these checks answered "no" by erroring out, including the one asserting a message is ABSENT.
said() { grep -qF -- "$1" "$T/err" && printf yes || printf no; }

# A portable install occupying the default root, which is what `install.sh --portable` with no
# --root produces, and exactly the directory the desktop app wants.
H="$T/home"
mkdir -p "$H/.unsloth/studio/unsloth_studio"
printf '%s\n' "$H/.unsloth" > "$H/.unsloth/.unsloth-portable-root"
printf '%s\n' "$H/.unsloth" > "$H/.unsloth/studio/.unsloth-master-root"

check "a desktop repair over a portable root is refused" "1" "$(run "$H" --tauri)"
check "and does not blame --portable"      no  "$(said "--portable and --root are not supported")"
check "it names the install instead"       yes "$(said "is portable, and the desktop")"
check "says no flag asked for it"          yes "$(said "No flag asked for this")"
check "and gives the way out"              yes "$(said "UNSLOTH_PORTABLE=0 sh install.sh")"
check "naming the root it found"           yes "$(said "$H/.unsloth")"

# A user who DID type the flag gets the original message, which was accurate for them.
check "a typed --portable with --tauri is still refused" "1" "$(run "$H" --tauri --portable)"
check "and still names the flags"          yes "$(said "--portable and --root are not supported")"
check "flat too"                           "1" "$(run "$H" --tauri --root "$T/elsewhere")"
check "also naming the flags"              yes "$(said "--portable and --root are not supported")"

# The ordinary case has to keep working: a normal install and --tauri must reach the install.
H2="$T/home2"; mkdir -p "$H2/.unsloth/studio/unsloth_studio"
check "a normal install still installs with --tauri" "0" "$(run "$H2" --tauri)"
check "and got past the guard" "REACHED-INSTALL" "$(cat "$T/out")"

# And --tauri on a machine with no Unsloth at all.
H3="$T/home3"; mkdir -p "$H3"
check "a fresh machine still installs with --tauri" "0" "$(run "$H3" --tauri)"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
