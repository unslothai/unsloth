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

case "$blockA" in *'--portable) _PORTABLE_MODE=true'*) : ;; *) echo "FAIL: extraction broke"; exit 1 ;; esac
case "$blockA" in
    *'_ROOT_FROM_FLAG=false'*) : ;;
    *) echo "FAIL: the escape hatch's typed-vs-inherited distinction is not in range"; exit 1 ;;
esac
case "$blockA" in
    *'.unsloth-portable-root'*) : ;;
    *) echo "FAIL: the parser no longer adopts the on-disk layout"; exit 1 ;;
esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
printf '%s\n' 'substep() { :; }' "$blockA" \
    'printf "PORTABLE=%s ROOT=%s STUDIO=%s\n" "$_PORTABLE_MODE" "$_UNSLOTH_ROOT" "${UNSLOTH_STUDIO_HOME:-}"' \
    'printf "CHILDHOME=%s\n" "$(sh -c '"'"'printf %s "${UNSLOTH_HOME:-}"'"'"')"' > "$SNIP"

parse() { # home [env assignments...] -- passed through env
    _h="$1"; shift
    env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" "$@" sh "$SNIP"
}
mode() { printf '%s' "$1" | sed -n 's/^PORTABLE=\([^ ]*\).*/\1/p'; }
root() { printf '%s' "$1" | sed -n 's/.* ROOT=\(.*\) STUDIO=.*/\1/p'; }
studio_of() { printf '%s' "$1" | sed -n 's/.* STUDIO=//p'; }
childhome() { printf '%s' "$1" | sed -n 's/^CHILDHOME=//p'; }

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

# 5b. A NESTED portable install names its master root in a record at the Studio root, not in a
# flat marker. Checking only the flat one left `--root R` plus a rerun carrying
# UNSLOTH_STUDIO_HOME=R/studio -- which is exactly what share/studio.conf exports -- reading as
# normal with both markers on disk, and _clear_stale_portable_marker then removed them.
H5b="$T/h5b"; R5b="$T/nestedroot"
mkdir -p "$H5b" "$R5b/studio/unsloth_studio" "$R5b/bin"
printf '%s\n' "$R5b" > "$R5b/.unsloth-portable-root"
printf '%s\n' "$R5b" > "$R5b/studio/.unsloth-master-root"
o5b="$(parse "$H5b" UNSLOTH_STUDIO_HOME="$R5b/studio")"
check "a nested named root is adopted from its master record" "true"  "$(mode "$o5b")"
check "and the record names the root"                         "$R5b"  "$(root "$o5b")"
check "it still converts back on request"                     "false" \
    "$(mode "$(parse "$H5b" UNSLOTH_PORTABLE=0 UNSLOTH_STUDIO_HOME="$R5b/studio")")"

# The record is trusted only as far as it is usable. A relative or empty one must not become a
# root resolved against the caller's cwd.
H5c="$T/h5c"; mkdir -p "$H5c" "$T/rel/studio" "$T/empty/studio"
printf 'relative/path\n' > "$T/rel/studio/.unsloth-master-root"
: > "$T/empty/studio/.unsloth-master-root"
check "a relative master record is ignored" "false" \
    "$(mode "$(parse "$H5c" UNSLOTH_STUDIO_HOME="$T/rel/studio")")"
check "an empty master record is ignored"   "false" \
    "$(mode "$(parse "$H5c" UNSLOTH_STUDIO_HOME="$T/empty/studio")")"

# 7. The documented escape has to work from where a user would actually run it: a shell opened
# from the portable install, where the shim and share/studio.conf have already exported
# UNSLOTH_HOME. "A root is what makes it portable" used to outrank the off value they typed, so
# `UNSLOTH_PORTABLE=0 curl ... | sh` there did nothing at all.
H7="$T/h7"; R7="$T/escroot"
mkdir -p "$H7" "$R7/studio/unsloth_studio"
o7="$(parse "$H7" UNSLOTH_HOME="$R7" UNSLOTH_PORTABLE=0)"
check "an inherited root does not beat a typed off value" "false" "$(mode "$o7")"
# ...and it converts THAT install, not a fresh one at the default location.
check "and it targets the install the root names" "$R7/studio" "$(studio_of "$o7")"

# A flat root converts in place too.
H7b="$T/h7b"; R7b="$T/escflat"
mkdir -p "$H7b" "$R7b/unsloth_studio"
check "a flat root converts in place" "$R7b" \
    "$(studio_of "$(parse "$H7b" UNSLOTH_HOME="$R7b" UNSLOTH_PORTABLE=0)")"

# A flat root that merely HAS a `studio` directory beside its venv still converts itself. A
# bare `[ -d <root>/studio ]` called that nested and aimed the conversion one level down -- the
# parent-marker guard then recognised the real flat install and kept it portable, so the
# documented way back did nothing while the run built a second environment at the wrong path
# and still reported the conversion. Ownership decides, using the same four sentinels as
# _resolve_studio_destinations and that guard, so all three agree on what "flat" means.
# An empty or leftover <root>/studio is the reachable shape: install.sh's own nested branch
# mkdir -p's it, so a nested run that was interrupted leaves one behind.
H7c="$T/h7c"; R7c="$T/escflat-neighbour"
mkdir -p "$H7c" "$R7c/unsloth_studio/bin" "$R7c/studio"
: > "$R7c/unsloth_studio/.unsloth-studio-owned"
check "a flat root with a leftover studio dir still converts itself" "$R7c" \
    "$(studio_of "$(parse "$H7c" UNSLOTH_HOME="$R7c" UNSLOTH_PORTABLE=0)")"

# But a root that is genuinely NESTED is excluded first, even when the parent also owns a venv:
# that is the documented two-install shape (a flat parent with a separate normal install at
# <parent>/studio), and the layout selector resolves it the same way.
H7cc="$T/h7cc"; R7cc="$T/escflat-real-neighbour"
mkdir -p "$H7cc" "$R7cc/unsloth_studio/bin" "$R7cc/studio/unsloth_studio"
: > "$R7cc/unsloth_studio/.unsloth-studio-owned"
check "a real nested install beside a flat parent still wins" "$R7cc/studio" \
    "$(studio_of "$(parse "$H7cc" UNSLOTH_HOME="$R7cc" UNSLOTH_PORTABLE=0)")"

# Each of the three sentinels that live outside the venv proves it on its own.
for _s in conf shim link; do
    _d="$T/escflat-$_s"; mkdir -p "$_d/unsloth_studio/bin" "$_d/studio" "$_d/share" "$_d/bin"
    case "$_s" in
        conf) printf "UNSLOTH_EXE='%s'\n" "$_d/unsloth_studio/bin/unsloth" > "$_d/share/studio.conf" ;;
        shim) printf "exec '%s' \"\$@\"\n" "$_d/unsloth_studio/bin/unsloth" > "$_d/bin/unsloth" ;;
        link) : > "$_d/unsloth_studio/bin/unsloth"; ln -s "$_d/unsloth_studio/bin/unsloth" "$_d/bin/unsloth" ;;
    esac
    check "the $_s sentinel alone proves the flat layout" "$_d" \
        "$(studio_of "$(parse "$H7c" UNSLOTH_HOME="$_d" UNSLOTH_PORTABLE=0)")"
done

# The negative that keeps it honest: an UNOWNED unsloth_studio directory beside a real nested
# install must not promote the root to flat. `unsloth_studio` is an ordinary directory name.
H7d="$T/h7d"; R7d="$T/escnested"
mkdir -p "$H7d" "$R7d/unsloth_studio" "$R7d/studio/unsloth_studio"
check "an unowned venv directory does not flatten a nested root" "$R7d/studio" \
    "$(studio_of "$(parse "$H7d" UNSLOTH_HOME="$R7d" UNSLOTH_PORTABLE=0)")"

# The conversion also has to stop handing the OLD master root to studio/setup.sh. Clearing the
# private copy left UNSLOTH_HOME exported, and setup.sh takes it ahead of a custom Studio root
# for node, llama.cpp and whisper.cpp: the run updated those three at <master>/* while the
# normal share/studio.conf it wrote resolves them under <studio-home>/*, so the conversion
# reported success and came up with no llama-server and no managed Node.
check "the conversion does not pass the old root to its children" "" \
    "$(childhome "$(parse "$H7" UNSLOTH_HOME="$R7" UNSLOTH_PORTABLE=0)")"
# Including from the shell a user actually converts in, which carries BOTH variables and so
# skips the branch above entirely.
check "and not when a studio home was inherited too" "" \
    "$(childhome "$(parse "$H7" UNSLOTH_HOME="$R7" UNSLOTH_STUDIO_HOME="$R7/studio" UNSLOTH_PORTABLE=0)")"
# A portable run must still export it, or the unset has leaked into the path that needs it.
check "a portable run still hands the root down" "$R7" \
    "$(childhome "$(parse "$H7" UNSLOTH_HOME="$R7")")"

# An inherited root with no off value is still portable: this must not become a way to lose it.
check "an inherited root alone is still portable" "true" \
    "$(mode "$(parse "$H7" UNSLOTH_HOME="$R7")")"

# A root TYPED on this command line together with an off value is a contradiction, not an
# inheritance. Refusing beats guessing which half they meant.
contradicts() { # extra args
    env -i HOME="$H7" PATH="$PATH" USER="${USER:-tester}" UNSLOTH_PORTABLE=0 \
        sh "$SNIP" "$@" >/dev/null 2>&1; printf '%s' "$?"
}
check "--root plus an off value is refused"     "1" "$(contradicts --root "$R7")"
check "--portable plus an off value is refused" "1" "$(contradicts --portable)"

# 6. An explicit --root still wins over whatever is on disk.
H6="$T/h6"; mkdir -p "$H6/.unsloth/studio" "$T/elsewhere"
printf '%s\n' "$H6/.unsloth" > "$H6/.unsloth/studio/.unsloth-master-root"
o6="$(env -i HOME="$H6" PATH="$PATH" USER="${USER:-tester}" sh "$SNIP" --root "$T/elsewhere")"
check "--root still wins" "$T/elsewhere" "$(root "$o6")"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
