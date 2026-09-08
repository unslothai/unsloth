#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: routine updating must not quietly convert a portable install back.
#
# The documented update is `curl -fsSL https://unsloth.ai/install.sh | sh`, with no arguments
# and no exports, because the shim's environment does not reach a fresh shell. That read as a
# request for a normal install: _PORTABLE_MODE defaulted to false, _clear_stale_portable_marker
# removed the master record and the portable marker, and every cache moved back under $HOME.
# The user was told only by a substep.
#
# Converting back is legitimate, so it stays available -- but as something asked for
# (UNSLOTH_PORTABLE=0), not as the side effect of an update.
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

case "$blockA" in *'--portable) _PORTABLE_MODE=true ;;'*) : ;; *) echo "FAIL: extraction broke"; exit 1 ;; esac
case "$blockA" in
    *'.unsloth-portable-root'*) : ;;
    *) echo "FAIL: the parser no longer adopts the on-disk layout"; exit 1 ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
printf '%s\n' 'substep() { :; }' "$blockA" \
    'printf "PORTABLE=%s ROOT=%s\n" "$_PORTABLE_MODE" "$_UNSLOTH_ROOT"' > "$SNIP"

parse() { # home [env assignments...] -- passed through env
    _h="$1"; shift
    env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" "$@" sh "$SNIP"
}
mode() { printf '%s' "$1" | sed -n 's/^PORTABLE=\([^ ]*\).*/\1/p'; }
root() { printf '%s' "$1" | sed -n 's/.*ROOT=//p'; }

# 1. A nested portable install at the default location, recorded by a previous run.
H1="$T/h1"; mkdir -p "$H1/.unsloth/studio"
printf '%s\n' "$H1/.unsloth" > "$H1/.unsloth/studio/.unsloth-master-root"
o="$(parse "$H1")"
check "a plain re-run keeps a nested portable install" "true"        "$(mode "$o")"
check "and keeps its root"                             "$H1/.unsloth" "$(root "$o")"

# 2. The parent marker alone is enough; a flat install at ~/.unsloth has only that.
H2="$T/h2"; mkdir -p "$H2/.unsloth/unsloth_studio"
printf '%s\n' "$H2/.unsloth" > "$H2/.unsloth/.unsloth-portable-root"
check "the parent marker alone is enough" "true" "$(mode "$(parse "$H2")")"

# 3. Converting back is still possible, but has to be asked for.
check "UNSLOTH_PORTABLE=0 still converts back" "false" \
    "$(mode "$(parse "$H1" UNSLOTH_PORTABLE=0)")"
check "and so does UNSLOTH_PORTABLE=off"       "false" \
    "$(mode "$(parse "$H1" UNSLOTH_PORTABLE=off)")"

# 4. A normal install stays normal: no marker, no adoption.
H4="$T/h4"; mkdir -p "$H4/.unsloth/studio/unsloth_studio"
check "a normal install is not made portable" "false" "$(mode "$(parse "$H4")")"

# 5. A named Studio root is spoken for only by its own marker, never by ~/.unsloth's.
H5="$T/h5"; mkdir -p "$H5/.unsloth/studio" "$H5/named"
printf '%s\n' "$H5/.unsloth" > "$H5/.unsloth/studio/.unsloth-master-root"
check "a named root does not inherit the default root's marker" "false" \
    "$(mode "$(parse "$H5" UNSLOTH_STUDIO_HOME="$H5/named")")"
printf '%s\n' "$H5/named" > "$H5/named/.unsloth-portable-root"
o5="$(parse "$H5" UNSLOTH_STUDIO_HOME="$H5/named")"
check "but its own marker is honoured"       "true"       "$(mode "$o5")"
check "and it becomes the root"              "$H5/named"  "$(root "$o5")"

# 6. An explicit --root still wins over whatever is on disk.
H6="$T/h6"; mkdir -p "$H6/.unsloth/studio" "$T/elsewhere"
printf '%s\n' "$H6/.unsloth" > "$H6/.unsloth/studio/.unsloth-master-root"
o6="$(env -i HOME="$H6" PATH="$PATH" USER="${USER:-tester}" sh "$SNIP" --root "$T/elsewhere")"
check "--root still wins" "$T/elsewhere" "$(root "$o6")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
