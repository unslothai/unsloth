#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: `bash studio/setup.sh` run directly must not rebuild the native runtimes
# at the wrong level of a portable install.
#
# A NESTED portable install (`install.sh --root /data/unsloth`) keeps node/, llama.cpp/ and
# whisper.cpp/ beside studio/, under the master root, and every derivation of those reads
# UNSLOTH_HOME. Running setup.sh directly carries none, so the fallback chain fell through to
# its custom-Studio-root arm and built all three under <root>/studio instead: a second
# multi-GB copy of llama.cpp, Node and whisper.cpp, with the ones already at <root> orphaned
# and still counted against the disk. Measured, not assumed -- the "the record is what
# decides" case below is the same resolution with the fix removed.
#
# Nothing escapes the root here, so this is a disk-space and correctness defect rather than a
# containment one. `unsloth studio update` never showed it, because the CLI fills UNSLOTH_HOME
# in before calling this script. The record install.sh leaves at the Studio root is what
# closes the gap for every other caller.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

blockT="$(grep '^_setup_trim_ws() ' "$SETUP")"
blockP="$(awk '/^_setup_abs_path\(\) \{$/ {g=1} g {print} g && /^\}$/ {exit}' "$SETUP")"
blockR="$(awk '
    /^if \[ -z "\$UNSLOTH_HOME" \] && \[ -f "\$STUDIO_HOME\/\.unsloth-master-root" \]; then$/ {g=1}
    g {print}
    g && /^fi$/ {exit}
' "$SETUP")"
blockF="$(awk '
    /^_PORTABLE_ROOT="\$\{UNSLOTH_HOME:-\}"$/ {g=1}
    g {print}
    g && /^LLAMA_CPP_DIR=/ {exit}
' "$SETUP")"

case "$blockT" in *_setup_trim_ws*) : ;; *) echo "FAIL: trim extraction broke"; exit 1 ;; esac
case "$blockP" in *_sap_path*)      : ;; *) echo "FAIL: abs_path extraction broke"; exit 1 ;; esac
case "$blockR" in
    *'.unsloth-master-root'*) : ;;
    *) echo "FAIL: setup.sh no longer recovers the master root from disk"; exit 1 ;;
esac
case "$blockF" in *'LLAMA_CPP_DIR='*) : ;; *) echo "FAIL: fallback extraction broke"; exit 1 ;; esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
SNIP="$T/snip.sh"
# mkdir -p in the lifted fallback would create the very directory the escape case is about,
# and a real run makes it too -- but here it would hide which arm was taken from the check
# below, so it is stubbed out rather than allowed to succeed silently.
printf '%s\n' 'set -u' "$blockT" "$blockP" \
    'UNSLOTH_HOME=$(_setup_trim_ws "${UNSLOTH_HOME:-}")' \
    '[ -z "$UNSLOTH_HOME" ] || UNSLOTH_HOME=$(_setup_abs_path "$UNSLOTH_HOME")' \
    'STUDIO_HOME="$FIXTURE_STUDIO"' 'STAGE_ROOT=""' \
    '_STUDIO_HOME_IS_CUSTOM="$FIXTURE_CUSTOM"' 'RUNTIME_ROOT="$STUDIO_HOME"' \
    'mkdir() { :; }' \
    "$blockR" "$blockF" \
    'printf "LLAMA=%s\n" "$LLAMA_CPP_DIR"' > "$SNIP"

run() { # studio_home is_custom [env...]
    _s="$1"; _c="$2"; shift 2
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" \
        FIXTURE_STUDIO="$_s" FIXTURE_CUSTOM="$_c" "$@" bash "$SNIP" \
        | sed -n 's/^LLAMA=//p'
}
mkdir -p "$T/home"

# A nested portable install, exactly as install.sh leaves it: the master record sits at the
# Studio root and names the level above.
R="$T/data/unsloth"
mkdir -p "$R/studio/unsloth_studio" "$R/llama.cpp"
printf '%s\n' "$R" > "$R/.unsloth-portable-root"
printf '%s\n' "$R" > "$R/studio/.unsloth-master-root"

check "a direct run finds the llama.cpp already in the root" \
    "$R/llama.cpp" "$(run "$R/studio" true)"
# Nothing lands beside the venv, which is where the duplicate build used to go.
check "and builds no second copy under studio/" "no" \
    "$([ -e "$R/studio/llama.cpp" ] && echo yes || echo no)"

# An environment that names the root still outranks the file, and agrees with it.
check "an explicit UNSLOTH_HOME still wins" "$R/llama.cpp" \
    "$(run "$R/studio" true UNSLOTH_HOME="$R")"

# A record naming somewhere else is still obeyed -- it is the installer's own writing, and
# the same evidence install.sh re-adopts from.
R2="$T/other"; mkdir -p "$R2"
printf '%s\n' "$R2" > "$T/data/unsloth/studio/.unsloth-master-root"
check "the record is what decides, not the layout" "$R2/llama.cpp" "$(run "$R/studio" true)"
printf '%s\n' "$R" > "$T/data/unsloth/studio/.unsloth-master-root"

# Untrustworthy records fall back to the old behaviour rather than resolving against the
# caller's cwd or a path that is not there.
S3="$T/rel/studio"; mkdir -p "$S3"
printf 'relative/path\n' > "$S3/.unsloth-master-root"
check "a relative record is ignored"  "$S3/llama.cpp" "$(run "$S3" true)"
: > "$S3/.unsloth-master-root"
check "an empty record is ignored"    "$S3/llama.cpp" "$(run "$S3" true)"
printf '%s\n' "$T/does-not-exist" > "$S3/.unsloth-master-root"
check "a record naming nothing on disk is ignored" "$S3/llama.cpp" "$(run "$S3" true)"

# And a normal install, which has no record at all, keeps the legacy default so that
# pre-existing ~/.unsloth/llama.cpp builds are still discovered.
S4="$T/home/.unsloth/studio"; mkdir -p "$S4"
check "a normal install still uses the legacy default" \
    "$T/home/.unsloth/llama.cpp" "$(run "$S4" false)"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
