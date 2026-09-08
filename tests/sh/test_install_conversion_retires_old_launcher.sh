#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: converting a nested portable install back must retire the portable
# configuration too, not just the wrapper.
#
# A nested `--portable` run writes its share/studio.conf and launch-studio.sh at <root>/share,
# one level ABOVE the tree it owns, and that config unconditionally exports UNSLOTH_HOME and
# UNSLOTH_PORTABLE=1 before naming the venv. Converting back writes a NEW pair into
# <root>/studio/share and retired only <root>/bin/unsloth, so the old config and launcher
# stayed: two otherwise-valid launchers over one install, and whichever path the user kept
# decided where the HF caches and the projects root went. The tree read as converted and was
# not.
#
# Moved aside rather than deleted, like the wrapper, so a conversion that fails hands back a
# tree that launches the way it did. The negative cases below are the point: a config that is
# not ours, or one naming a different install, is somebody else's launcher.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}
state() { if [ -e "$1" ]; then printf present; else printf gone; fi; }

blockC="$(awk '
    /^_clear_stale_portable_marker\(\) \{$/ {g = 1}
    g {print}
    g && /^\}$/ {exit}
' "$INSTALL")"
case "$blockC" in
    *'_PORTABLE_OLDCONF_PATH="$_spm_share/studio.conf"'*) : ;;
    *) echo "FAIL: the conversion no longer retires the portable configuration"; exit 1 ;;
esac
# Committing and rolling back both have to know about the new slots, or a successful run
# leaves dotfiles in share/ and a failed one cannot put the pair back.
grep -q '_restore_portable_shim_slot "$_PORTABLE_OLDCONF_PATH"' "$INSTALL" \
    || { echo "FAIL: the retired config is never restored on rollback"; exit 1; }
grep -q 'rm -f "$_PORTABLE_OLDCONF_BACKUP"' "$INSTALL" \
    || { echo "FAIL: the retired config's copy is never committed"; exit 1; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
printf '%s\n' 'substep() { :; }' 'C_WARN=""' \
    '_PORTABLE_MARKER_PATH_2=""' '_PORTABLE_MARKER_PRIOR_2=""' \
    '_PORTABLE_SHIM_PATH=""' '_PORTABLE_SHIM_BACKUP=""' \
    '_PORTABLE_OLDCONF_PATH=""' '_PORTABLE_OLDCONF_BACKUP=""' \
    '_PORTABLE_OLDLAUNCH_PATH=""' '_PORTABLE_OLDLAUNCH_BACKUP=""' \
    "$blockC" \
    'STUDIO_HOME="$FIXTURE_STUDIO"' 'DATA_DIR="$FIXTURE_DATA"' \
    '_PORTABLE_MODE=false' \
    '_clear_stale_portable_marker' \
    'printf "CONF=%s\n" "$_PORTABLE_OLDCONF_PATH"' > "$SNIP"

convert() { # studio_home data_dir
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" \
        FIXTURE_STUDIO="$1" FIXTURE_DATA="$2" sh "$SNIP" > "$T/out" 2>&1
}
mkdir -p "$T/home"

# A nested portable install: venv under <root>/studio, config and launcher at <root>/share.
make_nested() { # root
    mkdir -p "$1/studio/unsloth_studio/bin" "$1/studio/share" "$1/share" "$1/bin"
    printf '%s\n' "$1" > "$1/.unsloth-portable-root"
    printf '%s\n' "$1" > "$1/studio/.unsloth-master-root"
    {
        printf "export UNSLOTH_HOME='%s'\n" "$1"
        printf 'export UNSLOTH_PORTABLE=1\n'
        printf "UNSLOTH_EXE='%s'\n" "$1/studio/unsloth_studio/bin/unsloth"
    } > "$1/share/studio.conf"
    printf '#!/bin/sh\n. "%s/share/studio.conf"\n' "$1" > "$1/share/launch-studio.sh"
    chmod +x "$1/share/launch-studio.sh"
}

R="$T/R"; make_nested "$R"
convert "$R/studio" "$R/studio/share"
check "the portable config is retired"    gone "$(state "$R/share/studio.conf")"
check "and its launcher with it"          gone "$(state "$R/share/launch-studio.sh")"
# Moved aside, not deleted, so a failed conversion can hand it back.
check "the config was kept for rollback"  "1" \
    "$(ls -a "$R/share" | grep -c 'unsloth-portable-conf\.')"
check "and so was the launcher"           "1" \
    "$(ls -a "$R/share" | grep -c 'unsloth-portable-launch\.')"
check "the marker went too"               gone "$(state "$R/.unsloth-portable-root")"

# ── What must be left alone ──

# A config that is not ours: no UNSLOTH_PORTABLE=1 line.
R2="$T/R2"; make_nested "$R2"
printf "UNSLOTH_EXE='%s'\n" "$R2/studio/unsloth_studio/bin/unsloth" > "$R2/share/studio.conf"
convert "$R2/studio" "$R2/studio/share"
check "a non-portable config is left alone" present "$(state "$R2/share/studio.conf")"

# A portable config naming a DIFFERENT install's venv.
R3="$T/R3"; make_nested "$R3"
{
    printf 'export UNSLOTH_PORTABLE=1\n'
    printf "UNSLOTH_EXE='%s'\n" "$T/somewhere/else/bin/unsloth"
} > "$R3/share/studio.conf"
convert "$R3/studio" "$R3/studio/share"
check "a config naming another install is left alone" present "$(state "$R3/share/studio.conf")"

# The FLAT layout, where <parent>/share IS the DATA_DIR this run is about to write. Retiring
# there would delete the launchers of the install that is staying.
R4="$T/R4"; make_nested "$R4"
convert "$R4/studio" "$R4/share"
check "a flat layout keeps its own share/" present "$(state "$R4/share/studio.conf")"
check "and its launcher"                   present "$(state "$R4/share/launch-studio.sh")"

# A launcher with no config beside it is inert, so nothing is retired and nothing is claimed.
R5="$T/R5"; make_nested "$R5"
rm -f "$R5/share/studio.conf"
convert "$R5/studio" "$R5/studio/share"
check "a lone launcher is not touched" present "$(state "$R5/share/launch-studio.sh")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
