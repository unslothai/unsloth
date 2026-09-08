#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: the pre-#5190 arm of the env-mode venv guard must survive a symlinked HOME.
#
# tests/sh/test_install_venv_guard_symlinked_home.sh pins this for the IN-ROOT config. The same
# defect sat one directory over. An install made before #5190 shipped neither the owner marker
# nor a custom root, so its config went to the fixed ~/.local/share/unsloth/studio.conf and it
# has none of the three newer sentinels; that arm is the only thing that can vouch for it, and
# it matched the recorded UNSLOTH_EXE byte-for-byte. Those installs recorded the LEXICAL
# "$HOME/..." spelling (create_studio_shortcuts derives it with a logical cd/pwd), while
# env-mode canonicalizes STUDIO_HOME with cd -P. On a managed or NFS home the two spellings
# name one file and differ as strings, so `install.sh --portable` on such a tree refused its
# own valid environment outright: "does not look like an Unsloth Studio install", exit 1.
#
# The equality against the legacy root was already canonicalized on both sides. The record was
# not. The fix reuses _venv_guard_conf_names_venv, which is the filesystem proof the in-root
# config already gets; the negatives at the bottom hold the arm as narrow as it was.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

blockH="$(awk '/^_venv_guard_conf_names_venv\(\) \{$/ {g=1} g {print} g && /^\}$/ {exit}' "$INSTALL")"
# Latched on the first occurrence and closed on the outer `fi`: "# Narrow on purpose." is not
# unique in install.sh, and a sed range on it restarts and swallows a hundred unrelated lines.
# The enclosing `if [ "$_venv_guard_owned" != true ]; then` is supplied below, so the 8-space
# `fi` this ends on is the one that closes it.
blockL="$(awk '
    !g && /^            _venv_guard_legacy="\$HOME\/\.unsloth\/studio"$/ {g = 1}
    g {print}
    g && /^        fi$/ {exit}
' "$INSTALL")"

# Self-validating: a stale range must read as a broken test, never as a pass.
case "$blockH" in *'cd -P -- "$VENV_DIR/bin"'*) : ;;
    *) echo "FAIL: the canonical ownership comparison is gone"; exit 1 ;; esac
case "$blockH" in *'_vgc_conf='*) : ;;
    *) echo "FAIL: the helper no longer takes a config path"; exit 1 ;; esac
case "$blockL" in *'.local/share/unsloth/studio.conf'*) : ;;
    *) echo "FAIL: the legacy arm extraction broke"; exit 1 ;; esac
case "$blockL" in *'_venv_guard_conf_names_venv "$HOME'*) : ;;
    *) echo "FAIL: the legacy arm does not canonicalize the recorded launcher"; exit 1 ;; esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
{
    printf '%s\n' "$blockH"
    # The same escape the writers use, lifted from install.sh so the two cannot drift.
    grep -F '_venv_guard_exe=$(printf' "$INSTALL" | head -n 1 | sed 's/^ *//'
    printf '%s\n' '_venv_guard_owned=false'
    printf '%s\n' 'if [ "$_venv_guard_owned" != true ]; then'
    printf '%s\n' "$blockL"
    printf '%s\n' 'if [ "$_venv_guard_owned" = true ]; then echo ACCEPTED; else echo REFUSED; fi'
} > "$SNIP"
grep -q '_venv_guard_exe=' "$SNIP" || { echo "FAIL: could not lift the exe escape"; exit 1; }

verdict() { # home studio_home venv_dir
    env -i PATH="$PATH" HOME="$1" STUDIO_HOME="$2" VENV_DIR="$3" sh "$SNIP"
}

# ── A pre-#5190 default install reached through a symlinked HOME ──
# None of the newer sentinels: no .unsloth-studio-owned, no <root>/share/studio.conf, no
# <root>/bin/unsloth. Only the legacy config, recording the lexical spelling.
mkdir -p "$T/physical/.local/share/unsloth"
ln -s "$T/physical" "$T/home"
LEXHOME="$T/home"
LEX="$LEXHOME/.unsloth/studio"
mkdir -p "$LEX/unsloth_studio/bin"
printf "UNSLOTH_EXE='%s'\n" "$LEX/unsloth_studio/bin/unsloth" \
    > "$LEXHOME/.local/share/unsloth/studio.conf"
CANON="$(CDPATH= cd -P -- "$LEX" && pwd -P)"
check "the two spellings really do differ" "differ" \
    "$([ "$LEX" = "$CANON" ] && echo same || echo differ)"
check "our own pre-#5190 install is accepted" "ACCEPTED" \
    "$(verdict "$LEXHOME" "$CANON" "$CANON/unsloth_studio")"

# The control: with a real (non-symlinked) HOME the exact match already worked, so the case
# above is the symlink and not the fixture.
mkdir -p "$T/plain/.local/share/unsloth" "$T/plain/.unsloth/studio/unsloth_studio/bin"
printf "UNSLOTH_EXE='%s'\n" "$T/plain/.unsloth/studio/unsloth_studio/bin/unsloth" \
    > "$T/plain/.local/share/unsloth/studio.conf"
check "an unsymlinked home is accepted too" "ACCEPTED" \
    "$(verdict "$T/plain" "$T/plain/.unsloth/studio" "$T/plain/.unsloth/studio/unsloth_studio")"

# ── The narrowness this must not undo ──

# The arm only fires when STUDIO_HOME *is* the legacy default root. A user-named directory
# cannot be vouched for by the legacy config, however well it matches.
mkdir -p "$T/physical/named/unsloth_studio/bin"
printf "UNSLOTH_EXE='%s'\n" "$T/physical/named/unsloth_studio/bin/unsloth" \
    > "$LEXHOME/.local/share/unsloth/studio.conf"
check "a user-named root is still refused" "REFUSED" \
    "$(verdict "$LEXHOME" "$T/physical/named" "$T/physical/named/unsloth_studio")"

# A legacy config naming a DIFFERENT venv must not vouch for this one.
mkdir -p "$T/physical/elsewhere/bin"
printf "UNSLOTH_EXE='%s'\n" "$T/physical/elsewhere/bin/unsloth" \
    > "$LEXHOME/.local/share/unsloth/studio.conf"
check "a config naming another venv is refused" "REFUSED" \
    "$(verdict "$LEXHOME" "$CANON" "$CANON/unsloth_studio")"

# Nor one naming a different PROGRAM in the right directory.
printf "UNSLOTH_EXE='%s'\n" "$LEX/unsloth_studio/bin/python" \
    > "$LEXHOME/.local/share/unsloth/studio.conf"
check "a config naming another program is refused" "REFUSED" \
    "$(verdict "$LEXHOME" "$CANON" "$CANON/unsloth_studio")"

# A relative, a vanished and a missing recording all decline rather than comparing two
# unresolved strings.
printf "UNSLOTH_EXE='%s'\n" "unsloth_studio/bin/unsloth" \
    > "$LEXHOME/.local/share/unsloth/studio.conf"
check "a relative recording is refused" "REFUSED" \
    "$(verdict "$LEXHOME" "$CANON" "$CANON/unsloth_studio")"
printf "UNSLOTH_EXE='%s'\n" "$T/gone/bin/unsloth" \
    > "$LEXHOME/.local/share/unsloth/studio.conf"
check "a vanished recording is refused" "REFUSED" \
    "$(verdict "$LEXHOME" "$CANON" "$CANON/unsloth_studio")"
rm -f "$LEXHOME/.local/share/unsloth/studio.conf"
check "no legacy config at all is refused" "REFUSED" \
    "$(verdict "$LEXHOME" "$CANON" "$CANON/unsloth_studio")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
