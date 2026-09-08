#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: converting a pre-#5190 default install to --portable must not be refused.
#
# The env-mode ownership guard wants one of three sentinels, and all three are written under
# STUDIO_HOME. #5190 (7be10852c, 2026-05-05) shipped the in-venv .unsloth-studio-owned marker
# and custom-root support in the same commit, so a default install last refreshed before that
# date has none of them: it never had a custom root, and its studio.conf went to the fixed
# default ~/.local/share/unsloth, where the UNSLOTH_EXE line has lived since #4568.
#
# --portable always sets _STUDIO_HOME_REDIRECT=env, which is what arms the guard, so such a
# tree was accepted by a plain upgrade and refused by the very same upgrade plus --portable.
#
# The guard must stay strict everywhere else: the legacy evidence is only good for the legacy
# default root, never for a directory the user named.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

# The guard block, lifted whole. Its opener is deliberately a two-line `if` in install.sh
# precisely so this extraction is unambiguous.
block="$(awk '
    /^    if \[ "\$_STUDIO_HOME_REDIRECT" = "env" \] \\$/ {g = 1}
    g {print}
    g && /^    fi$/ {exit}
' "$INSTALL")"

case "$block" in *'_venv_guard_owned'*) : ;; *) echo "FAIL: guard extraction broke"; exit 1 ;; esac
case "$block" in
    *'does not look like an Unsloth Studio install'*) : ;;
    *) echo "FAIL: guard lost its refusal"; exit 1 ;;
esac
case "$block" in
    *'.local/share/unsloth/studio.conf'*) : ;;
    *) echo "FAIL: guard lost the pre-#5190 evidence"; exit 1 ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

SNIP='
STUDIO_HOME="$FIXTURE_STUDIO"
VENV_DIR="$STUDIO_HOME/unsloth_studio"
_STUDIO_HOME_REDIRECT="$FIXTURE_MODE"
'"$block"'
echo ACCEPTED'

guard() { # home studio mode
    env -i HOME="$1" PATH="$PATH" FIXTURE_STUDIO="$2" FIXTURE_MODE="$3" \
        sh -c "$SNIP" _ 2>/dev/null | tail -n 1
}

# A pre-#5190 default install: venv on disk, no owner marker, no studio.conf under the studio
# root, and the real one at the fixed legacy location.
premarker() { # home
    mkdir -p "$1/.unsloth/studio/unsloth_studio/bin" "$1/.local/share/unsloth"
    : > "$1/.unsloth/studio/unsloth_studio/bin/unsloth"
    printf "UNSLOTH_EXE='%s'\n" "$1/.unsloth/studio/unsloth_studio/bin/unsloth" \
        > "$1/.local/share/unsloth/studio.conf"
}

H1="$T/h1"; mkdir -p "$H1"; premarker "$H1"
check "a plain upgrade of a pre-marker install is accepted" "ACCEPTED" \
    "$(guard "$H1" "$H1/.unsloth/studio" default)"
check "and so is the same tree with --portable"             "ACCEPTED" \
    "$(guard "$H1" "$H1/.unsloth/studio" env)"

# A current install still passes on the marker alone.
H2="$T/h2"; mkdir -p "$H2/.unsloth/studio/unsloth_studio"
: > "$H2/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
check "a marked install still passes"                       "ACCEPTED" \
    "$(guard "$H2" "$H2/.unsloth/studio" env)"

# The point of the guard: someone else's directory is still refused, even with the legacy
# studio.conf present, because STUDIO_HOME is not the legacy default root.
H3="$T/h3"; mkdir -p "$H3/mystuff/unsloth_studio/bin" "$H3/.local/share/unsloth"
: > "$H3/mystuff/unsloth_studio/bin/unsloth"
printf "UNSLOTH_EXE='%s'\n" "$H3/mystuff/unsloth_studio/bin/unsloth" \
    > "$H3/.local/share/unsloth/studio.conf"
check "an unrelated custom root is still refused" "" \
    "$(guard "$H3" "$H3/mystuff" env | grep -x ACCEPTED || printf '')"

# And a legacy default root with no evidence at all is still refused: the new arm needs the
# recorded UNSLOTH_EXE line, not merely the right path.
H4="$T/h4"; mkdir -p "$H4/.unsloth/studio/unsloth_studio/bin"
: > "$H4/.unsloth/studio/unsloth_studio/bin/unsloth"
check "a legacy root with no recorded exe is still refused" "" \
    "$(guard "$H4" "$H4/.unsloth/studio" env | grep -x ACCEPTED || printf '')"

# A studio.conf naming a DIFFERENT venv must not vouch for this one.
H5="$T/h5"; mkdir -p "$H5/.unsloth/studio/unsloth_studio/bin" "$H5/.local/share/unsloth"
: > "$H5/.unsloth/studio/unsloth_studio/bin/unsloth"
printf "UNSLOTH_EXE='%s'\n" "$H5/somewhere/else/bin/unsloth" \
    > "$H5/.local/share/unsloth/studio.conf"
check "a studio.conf naming another venv does not vouch" "" \
    "$(guard "$H5" "$H5/.unsloth/studio" env | grep -x ACCEPTED || printf '')"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
