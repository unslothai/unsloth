#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: uninstalling a portable install rooted at ~/.unsloth must not take the rest
# of ~/.unsloth with it.
#
# `--portable` with no `--root` selects $HOME/.unsloth, which is not a directory the user
# chose: it has been there since their first install, and whatever they put beside Unsloth's
# own children is theirs. Treating it as a master root made the custom-root loop rm -rf the
# whole tree, where the baseline uninstaller removed Unsloth's children by name and finished
# with an rmdir that refuses a non-empty directory.
#
# Removing a root the user NAMED stays wholesale. That is the promise of --root, it predates
# these PRs for a custom UNSLOTH_STUDIO_HOME, and the second case below holds it in place.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
UNINSTALL="$HERE/../../scripts/uninstall.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}
state() { # path
    if [ -e "$1" ]; then printf present; else printf gone; fi
}

# Self-validate: the exemption has to still be in the script being tested.
grep -q '_custom_default_root' "$UNINSTALL" \
    || { echo "FAIL: uninstall.sh has no default-root exemption"; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

# A portable install converted in place at ~/.unsloth, with the user's own files beside it.
H="$T/home"
mkdir -p "$H/.unsloth/studio/unsloth_studio/bin" "$H/.unsloth/share" "$H/.unsloth/cache/uv" \
         "$H/.unsloth/bin" "$H/.unsloth/personal" "$H/.local/bin" "$H/.local/share"
: > "$H/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$H/.unsloth/studio/studio.db"
printf '%s\n' "$H/.unsloth" > "$H/.unsloth/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\n" "$H/.unsloth/studio/unsloth_studio/bin/unsloth" \
    > "$H/.unsloth/share/studio.conf"
printf 'my notes\n' > "$H/.unsloth/personal/keep.txt"
printf 'backup\n'   > "$H/.unsloth/important.txt"

env -i HOME="$H" PATH="$PATH" UNSLOTH_HOME="$H/.unsloth" sh "$UNINSTALL" >/dev/null 2>&1 || true

check "a user file beside the install survives"   present "$(state "$H/.unsloth/personal/keep.txt")"
check "and so does one at the top of the root"    present "$(state "$H/.unsloth/important.txt")"
# Everything Unsloth put there still has to go, or the exemption has just broken uninstall.
check "the Studio root is removed"                gone    "$(state "$H/.unsloth/studio")"
check "the portable cache directory is removed"   gone    "$(state "$H/.unsloth/cache")"
check "the portable bin directory is removed"     gone    "$(state "$H/.unsloth/bin")"
check "share/ is removed"                         gone    "$(state "$H/.unsloth/share")"
check "the portable marker is removed"            gone    "$(state "$H/.unsloth/.unsloth-portable-root")"

# Same install, nothing of the user's in it: the directory itself should still disappear, via
# the default block's rmdir.
H2="$T/home2"
mkdir -p "$H2/.unsloth/studio/unsloth_studio/bin" "$H2/.unsloth/share" "$H2/.unsloth/cache" \
         "$H2/.unsloth/bin" "$H2/.local/bin" "$H2/.local/share"
: > "$H2/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$H2/.unsloth" > "$H2/.unsloth/.unsloth-portable-root"
env -i HOME="$H2" PATH="$PATH" UNSLOTH_HOME="$H2/.unsloth" sh "$UNINSTALL" >/dev/null 2>&1 || true
check "an Unsloth-only default root still goes entirely" gone "$(state "$H2/.unsloth")"

# A root the user named keeps the old, promised behaviour: it goes in one piece.
H3="$T/home3"; R3="$T/dedicated"
mkdir -p "$H3/.local/bin" "$H3/.local/share" "$R3/studio/unsloth_studio/bin" "$R3/share" \
         "$R3/cache/uv" "$R3/bin"
: > "$R3/studio/unsloth_studio/.unsloth-studio-owned"
: > "$R3/studio/studio.db"
printf '%s\n' "$R3" > "$R3/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\n" "$R3/studio/unsloth_studio/bin/unsloth" > "$R3/share/studio.conf"
env -i HOME="$H3" PATH="$PATH" UNSLOTH_HOME="$R3" sh "$UNINSTALL" >/dev/null 2>&1 || true
check "a root the user named is still removed whole" gone "$(state "$R3")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
