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
# ...and the flat-root database branch has to still resolve a relocated database before it
# reports on one, the way _remove_root_recording_db does for every other layout.
grep -q '_custom_db_data' "$UNINSTALL" \
    || { echo "FAIL: the flat-root database branch no longer tracks the symlink target"; exit 1; }

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

# A FLAT install at the default root puts the venv and the database AT the root, and the
# default block only knows the nested shape. The first version of this exception removed the
# markers and left both behind: gigabytes retained with their metadata stripped, and the run
# still reporting success.
H4="$T/home4"
mkdir -p "$H4/.unsloth/unsloth_studio/bin" "$H4/.unsloth/share" "$H4/.unsloth/cache" \
         "$H4/.unsloth/bin" "$H4/.local/bin" "$H4/.local/share"
: > "$H4/.unsloth/unsloth_studio/.unsloth-studio-owned"
: > "$H4/.unsloth/studio.db"
printf '%s\n' "$H4/.unsloth" > "$H4/.unsloth/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\n" "$H4/.unsloth/unsloth_studio/bin/unsloth" > "$H4/.unsloth/share/studio.conf"
printf 'notes\n' > "$H4/.unsloth/my-notes.txt"
out4="$(env -i HOME="$H4" PATH="$PATH" UNSLOTH_HOME="$H4/.unsloth" sh "$UNINSTALL" 2>/dev/null || true)"
check "a flat install's owned venv is removed"  gone    "$(state "$H4/.unsloth/unsloth_studio")"
check "and its database goes with it"           gone    "$(state "$H4/.unsloth/studio.db")"
check "while the user's own file still stays"   present "$(state "$H4/.unsloth/my-notes.txt")"
# ...and the run has to SAY it took the chat history. A flat root's studio.db sits beside the
# venv, not inside it, so the recording helper -- which is handed the venv -- probed the wrong
# path and found nothing; the database was then deleted by a plain removal that records
# nothing, and the run closed by telling the user no studio.db was found and that their
# history must be somewhere else. Deleting the data and denying it is worse than either alone.
says() { case "$out4" in *"$1"*) printf yes ;; *) printf no ;; esac; }
check "the flat database is reported as removed"     yes "$(says 'studio.db it found')"
check "and not reported as never having been found"  no  "$(says 'No studio.db was found')"

# Same flat root, but studio.db is a symlink to a database on another volume -- the relocation
# _remove_root_recording_db already allows for on every other layout. `-f` reads the database
# through the link, but `rm -rf` unlinks the LINK and never the target, so the lexical path
# reads as absent afterwards and the before/after pair concluded the database was destroyed.
# The run then closed by telling the user their chat history was gone while every byte of it
# was still on the other disk: the exact inverse of the H4 case above, and a false statement
# either way. Nothing may be deleted off the far volume here either.
H4b="$T/home4b"; EXT="$T/elsewhere"
mkdir -p "$H4b/.unsloth/unsloth_studio/bin" "$H4b/.unsloth/share" "$H4b/.unsloth/cache" \
         "$H4b/.unsloth/bin" "$H4b/.local/bin" "$H4b/.local/share" "$EXT"
: > "$H4b/.unsloth/unsloth_studio/.unsloth-studio-owned"
printf 'the real chat history\n' > "$EXT/studio.db"
ln -s "$EXT/studio.db" "$H4b/.unsloth/studio.db"
printf '%s\n' "$H4b/.unsloth" > "$H4b/.unsloth/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\n" "$H4b/.unsloth/unsloth_studio/bin/unsloth" > "$H4b/.unsloth/share/studio.conf"
out4b="$(env -i HOME="$H4b" PATH="$PATH" UNSLOTH_HOME="$H4b/.unsloth" sh "$UNINSTALL" 2>/dev/null || true)"
says4b() { case "$out4b" in *"$1"*) printf yes ;; *) printf no ;; esac; }
# The control: without it, a "reports the survivor" pass could just be a fixture whose link
# was never followed in the first place.
check "the link inside the root is unlinked"      gone    "$(state "$H4b/.unsloth/studio.db")"
check "the database on the other volume survives" present "$(state "$EXT/studio.db")"
check "so the run does NOT claim the history is gone" no  "$(says4b 'the chat history in the install(s) removed')"
check "it says the history may still be on disk"     yes "$(says4b 'chat history may still be on disk')"

# The owner marker is what licences that removal. Without it the directory is the user's.
H5="$T/home5"
mkdir -p "$H5/.unsloth/unsloth_studio" "$H5/.local/bin" "$H5/.local/share"
printf 'mine\n' > "$H5/.unsloth/unsloth_studio/mine.txt"
printf '%s\n' "$H5/.unsloth" > "$H5/.unsloth/.unsloth-portable-root"
env -i HOME="$H5" PATH="$PATH" UNSLOTH_HOME="$H5/.unsloth" sh "$UNINSTALL" >/dev/null 2>&1 || true
check "an unowned unsloth_studio dir is left alone" present "$(state "$H5/.unsloth/unsloth_studio/mine.txt")"

# The advertised no-argument uninstall, from a fresh shell, on the default portable root.
# `--portable` without `--root` selects $HOME/.unsloth and puts DATA_DIR at <root>/share, so
# nothing is ever written to $HOME/.local/share/unsloth and a fresh shell carries no
# UNSLOTH_HOME: the enumerator emitted nothing at all, the default block removed studio/ by
# name, and bin/, share/ and the multi-gigabyte cache/ stayed on disk under a closing
# "Unsloth Studio uninstalled." The surviving .unsloth-portable-root is also what a later
# plain `curl | sh` adopts, so the root the user believed was gone quietly came back portable.
# NO root variables here on purpose -- passing one is what hid this.
H6="$T/home6"
mkdir -p "$H6/.unsloth/studio/unsloth_studio/bin" "$H6/.unsloth/share" "$H6/.unsloth/cache/uv" \
         "$H6/.unsloth/bin" "$H6/.unsloth/personal" "$H6/.local/bin" "$H6/.local/share"
: > "$H6/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$H6/.unsloth/studio/studio.db"
printf 'wheel bytes\n' > "$H6/.unsloth/cache/uv/big.bin"
printf '%s\n' "$H6/.unsloth" > "$H6/.unsloth/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\n" "$H6/.unsloth/studio/unsloth_studio/bin/unsloth" \
    > "$H6/.unsloth/share/studio.conf"
printf 'my notes\n' > "$H6/.unsloth/personal/keep.txt"
env -i HOME="$H6" PATH="$PATH" sh "$UNINSTALL" >/dev/null 2>&1 || true
check "flagless: the Studio root is removed"          gone    "$(state "$H6/.unsloth/studio")"
check "flagless: the cache is removed"                gone    "$(state "$H6/.unsloth/cache")"
check "flagless: bin/ is removed"                     gone    "$(state "$H6/.unsloth/bin")"
check "flagless: share/ is removed"                   gone    "$(state "$H6/.unsloth/share")"
check "flagless: the portable marker is removed"      gone    "$(state "$H6/.unsloth/.unsloth-portable-root")"
check "flagless: the user's own file survives"        present "$(state "$H6/.unsloth/personal/keep.txt")"

# Same run, nothing of the user's: the root itself should go.
H7="$T/home7"
mkdir -p "$H7/.unsloth/studio/unsloth_studio/bin" "$H7/.unsloth/share" "$H7/.unsloth/cache" \
         "$H7/.unsloth/bin" "$H7/.local/bin" "$H7/.local/share"
: > "$H7/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$H7/.unsloth" > "$H7/.unsloth/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\n" "$H7/.unsloth/studio/unsloth_studio/bin/unsloth" \
    > "$H7/.unsloth/share/studio.conf"
env -i HOME="$H7" PATH="$PATH" sh "$UNINSTALL" >/dev/null 2>&1 || true
check "flagless: an Unsloth-only default root goes entirely" gone "$(state "$H7/.unsloth")"

# A plain ~/.unsloth with no portable marker must NOT be adopted by the new discovery: it is a
# normal install, and the default block owns it.
H8="$T/home8"
mkdir -p "$H8/.unsloth/personal" "$H8/.local/bin" "$H8/.local/share"
printf 'mine\n' > "$H8/.unsloth/personal/keep.txt"
env -i HOME="$H8" PATH="$PATH" sh "$UNINSTALL" >/dev/null 2>&1 || true
check "flagless: an unmarked ~/.unsloth is left alone" present "$(state "$H8/.unsloth/personal/keep.txt")"

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
