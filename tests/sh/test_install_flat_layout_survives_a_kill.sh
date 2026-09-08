#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: an interrupted reinstall must not change a portable root's layout.
#
# _start_studio_venv_replacement renames <root>/unsloth_studio to
# unsloth_studio.rollback.<ts>.<pid> before building the new one. The flat-layout selector used
# to require that directory to exist, so a run killed inside that window left a flat root
# looking un-flat, and the retry resolved <root>/studio instead: a different data root, with the
# real studio.db and every cache stranded at <root>, and rollback pruning aimed at the new path.
# To the user that reads as losing the entire chat history to a crashed update.
#
# HUP/INT/TERM are trapped and roll back, so the window only matters for SIGKILL or power loss.
# The test does not need to kill anything: it recreates the on-disk state such a kill leaves.
#
# The inverse matters just as much. A NESTED root mid-reinstall has its own venv renamed too,
# and must still resolve nested; the sentinels name the venv path exactly, which is what keeps
# the two apart.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

blockT="$(grep '^_trim_ws() ' "$INSTALL")"
blockB="$(awk '/^_resolve_studio_destinations\(\) \{$/ {g=1} g {print} g && /^\}$/ {exit}' "$INSTALL")"
case "$blockT" in *_trim_ws*) : ;; *) echo "FAIL: _trim_ws extraction broke"; exit 1 ;; esac
case "$blockB" in *'_PORTABLE_FLAT=true'*) : ;; *) echo "FAIL: selector extraction broke"; exit 1 ;; esac
# The point of the fix: the live venv directory must not be what decides the layout.
case "$blockB" in
    *'if [ -d "$_rsd_flat_venv" ] && [ ! -d'*)
        echo "FAIL: the flat selector gates on the venv directory again"; exit 1 ;;
    *) : ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
printf '%s\n' 'substep() { :; }' "$blockT" "$blockB" \
    '_PORTABLE_MODE=true' '_PORTABLE_FLAT=false' '_UNSLOTH_ROOT="$FIXTURE_ROOT"' \
    '_resolve_studio_destinations' \
    'printf "FLAT=%s|%s\n" "$_PORTABLE_FLAT" "$STUDIO_HOME"' > "$SNIP"

resolve() { env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" FIXTURE_ROOT="$1" sh "$SNIP"; }
flat() { printf '%s' "$1" | sed -n 's/^FLAT=\([^|]*\).*/\1/p'; }
home_of() { printf '%s' "$1" | sed -n 's/.*|//p'; }

mkdir -p "$T/home"

# A real flat portable install: venv, its owner marker, studio.conf naming it, and the database.
R="$T/flat"
mkdir -p "$R/unsloth_studio/bin" "$R/share" "$R/bin"
: > "$R/unsloth_studio/.unsloth-studio-owned"
: > "$R/studio.db"
printf '%s\n' "$R" > "$R/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\n" "$R/unsloth_studio/bin/unsloth" > "$R/share/studio.conf"

o="$(resolve "$R")"
check "a healthy flat root resolves flat" "true" "$(flat "$o")"
check "and keeps its data root"           "$R"   "$(home_of "$o")"

# Exactly what a SIGKILL between the rename and the new venv leaves behind.
mv "$R/unsloth_studio" "$R/unsloth_studio.rollback.20260908120000.4242"
o2="$(resolve "$R")"
check "the retry after a kill is still flat"     "true" "$(flat "$o2")"
check "so the existing studio.db is still ours"  "$R"   "$(home_of "$o2")"
check "and the database was never moved"         "yes"  \
    "$([ -f "$R/studio.db" ] && echo yes || echo no)"

# Only studio.conf survives (no bin/ shim at all): still enough on its own.
R2="$T/confonly"
mkdir -p "$R2/share"
printf "UNSLOTH_EXE='%s'\n" "$R2/unsloth_studio/bin/unsloth" > "$R2/share/studio.conf"
check "studio.conf alone identifies the flat root" "true" "$(flat "$(resolve "$R2")")"

# The inverse: a nested root whose own venv is renamed aside must stay nested. Its studio.conf
# names <root>/studio/unsloth_studio, so the flat sentinels cannot match it.
N="$T/nested"
mkdir -p "$N/studio/share" "$N/bin"
printf "UNSLOTH_EXE='%s'\n" "$N/studio/unsloth_studio/bin/unsloth" > "$N/studio/share/studio.conf"
printf '%s\n' "$N" > "$N/.unsloth-portable-root"
printf '%s\n' "$N" > "$N/studio/.unsloth-master-root"
on="$(resolve "$N")"
check "a nested root mid-reinstall stays nested" "false"     "$(flat "$on")"
check "and keeps its studio/ data root"          "$N/studio" "$(home_of "$on")"

# A directory with no Unsloth evidence at all is a fresh install, not somebody's flat root.
E="$T/empty"; mkdir -p "$E"
check "an empty root is a normal nested install" "false" "$(flat "$(resolve "$E")")"

# Nor may an unrelated folder that merely contains a directory called unsloth_studio.
U="$T/unrelated"; mkdir -p "$U/unsloth_studio"
check "an unowned unsloth_studio dir does not claim the root" "false" "$(flat "$(resolve "$U")")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
