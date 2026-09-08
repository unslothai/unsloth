#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: upgrading our own pre-#5190 install must not be refused because HOME is a
# symlink.
#
# The env-mode venv guard was tightened so that a sentinel OUTSIDE the venv has to NAME the
# venv it vouches for, rather than merely exist. That is right: a user-chosen workspace
# holding their own bin/unsloth and their own unsloth_studio virtualenv is exactly the shape
# the guard is there to refuse, and existence alone let it through and then moved that
# environment aside for good.
#
# But installers before #5190 wrote UNSLOTH_EXE with the lexical "$HOME/..." spelling, while
# env-mode canonicalizes STUDIO_HOME with cd -P. On a machine where HOME is a symlink -- a
# managed or NFS home, or /home -> /System/Volumes/Data/home on macOS -- the two spellings
# name one file and differ as strings, so our own config stopped vouching for our own venv
# and the upgrade died with "does not look like an Unsloth Studio install".
#
# main accepted any existing share/studio.conf, so this refusal came in with the tightening
# rather than being inherited. The fix canonicalizes the comparison; it does not weaken it,
# which is what the negative cases at the bottom are here to hold.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

blockH="$(awk '
    /^_venv_guard_conf_names_venv\(\) \{$/ {g = 1}
    g {print}
    g && /^\}$/ {exit}
' "$INSTALL")"
case "$blockH" in
    *'cd -P -- "$VENV_DIR/bin"'*) : ;;
    *) echo "FAIL: the canonical ownership comparison is gone"; exit 1 ;;
esac
# It has to be wired into the guard, not merely defined.
grep -q 'elif _venv_guard_conf_names_venv; then' "$INSTALL" \
    || { echo "FAIL: the helper is defined but the guard does not call it"; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
printf '%s\n' "$blockH" \
    'if _venv_guard_conf_names_venv; then echo OWNED; else echo FOREIGN; fi' > "$SNIP"

owned() { # studio_home venv_dir
    env -i PATH="$PATH" STUDIO_HOME="$1" VENV_DIR="$2" sh "$SNIP"
}

# A pre-#5190 install reached through a symlinked HOME. The config records the path as the
# user's shell spelled it; STUDIO_HOME arrives canonicalized, as env-mode makes it.
mkdir -p "$T/physical"
ln -s "$T/physical" "$T/home"
LEX="$T/home/.unsloth/studio"
mkdir -p "$LEX/unsloth_studio/bin" "$LEX/share"
printf "UNSLOTH_EXE='%s'\n" "$LEX/unsloth_studio/bin/unsloth" > "$LEX/share/studio.conf"
CANON="$(CDPATH= cd -P -- "$LEX" && pwd -P)"
check "the two spellings really do differ" "differ" \
    "$([ "$LEX" = "$CANON" ] && echo same || echo differ)"
check "our own legacy install is recognised" "OWNED" \
    "$(owned "$CANON" "$CANON/unsloth_studio")"

# The modern spelling still works. Its own directory, so the lexical case above cannot be
# what makes this one pass.
M="$T/modern"; mkdir -p "$M/unsloth_studio/bin" "$M/share"
printf "UNSLOTH_EXE='%s'\n" "$M/unsloth_studio/bin/unsloth" > "$M/share/studio.conf"
check "a config written canonically is recognised too" "OWNED" \
    "$(owned "$M" "$M/unsloth_studio")"

# ── The tightening this must not undo ──

# A config naming a DIFFERENT venv must not vouch for this one. This is the case the whole
# guard exists for: somebody's own workspace, with their own unsloth_studio in it.
F1="$T/workspace"
mkdir -p "$F1/unsloth_studio/bin" "$F1/share" "$F1/elsewhere/bin"
printf "UNSLOTH_EXE='%s'\n" "$F1/elsewhere/bin/unsloth" > "$F1/share/studio.conf"
check "a config naming another venv does not vouch" "FOREIGN" \
    "$(owned "$F1" "$F1/unsloth_studio")"

# A config naming a different PROGRAM in the right directory must not either.
F2="$T/ws2"; mkdir -p "$F2/unsloth_studio/bin" "$F2/share"
printf "UNSLOTH_EXE='%s'\n" "$F2/unsloth_studio/bin/python" > "$F2/share/studio.conf"
check "a config naming another program does not vouch" "FOREIGN" \
    "$(owned "$F2" "$F2/unsloth_studio")"

# No config at all, an empty one, and a relative recording: all decline.
F3="$T/ws3"; mkdir -p "$F3/unsloth_studio/bin" "$F3/share"
check "no config declines" "FOREIGN" "$(owned "$F3" "$F3/unsloth_studio")"
: > "$F3/share/studio.conf"
check "an empty config declines" "FOREIGN" "$(owned "$F3" "$F3/unsloth_studio")"
printf "UNSLOTH_EXE='%s'\n" "unsloth_studio/bin/unsloth" > "$F3/share/studio.conf"
check "a relative recording declines" "FOREIGN" "$(owned "$F3" "$F3/unsloth_studio")"

# A recorded directory that is not there cannot be canonicalized, so it declines rather than
# comparing two unresolved strings.
printf "UNSLOTH_EXE='%s'\n" "$T/gone/bin/unsloth" > "$F3/share/studio.conf"
check "a vanished recording declines" "FOREIGN" "$(owned "$F3" "$F3/unsloth_studio")"

# An escaped quote is left to the exact-match test rather than unescaped here.
F4="$T/ws4"; mkdir -p "$F4/unsloth_studio/bin" "$F4/share"
printf "UNSLOTH_EXE='%s'\n" "$F4/unsloth_studio/bin/uns'\\''loth" > "$F4/share/studio.conf"
check "an escaped quote declines here" "FOREIGN" "$(owned "$F4" "$F4/unsloth_studio")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
