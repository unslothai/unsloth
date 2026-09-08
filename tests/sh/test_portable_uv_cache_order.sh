#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: in portable mode the installer must fill the SAME uv cache its launchers
# read, and must not report its own seed back to the user as a cache they chose.
#
# install.sh seeds UV_CACHE_DIR from STUDIO_HOME at top level, and only calls
# _export_portable_roots afterwards. For a NESTED portable install (--root DIR) STUDIO_HOME is
# <root>/studio, so the install filled <root>/studio/cache/uv while the generated bin/unsloth
# shim, share/studio.conf and storage_roots' portable defaults all resolve <root>/cache/uv:
# every later `unsloth studio update` started from a cold cache and re-downloaded the Torch
# and CUDA wheels the install had just fetched, into a second multi-GB tree under the same
# root. And because _epr_default only tests "is this non-blank", it read that seed as an
# explicit caller value in BOTH layouts, so a user who had set nothing was told "kept your own
# cache locations: UV_CACHE_DIR" and warned it might survive `rm -rf <root>`.
#
# tests/sh/test_portable_keeps_explicit_caches.sh cannot see either half: it lifts
# _resolve_studio_destinations and _export_portable_roots WITHOUT the top-level uv block
# between them, so it exercises an ordering the real script never has. This file lifts all
# three, in install.sh's own order.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
INSTALL="$HERE/../../install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/home" "$T/mine"

# ── Lift the real code, in the real order ────────────────────────────
blockT="$(grep '^_trim_ws() ' "$INSTALL")"
blockR="$(awk '/^_resolve_studio_destinations\(\) \{$/ {g=1} g {print} g && /^\}$/ {exit}' "$INSTALL")"
blockU="$(awk '/^# Keep uv.s cache on the same filesystem as the venv it fills\.$/,/^fi$/' "$INSTALL")"
blockE="$(awk '/^_export_portable_roots\(\) \{$/ {g=1} g {print} g && /^\}$/ {exit}' "$INSTALL")"

# Self-validating extraction: a stale awk range must read as a broken test, never as a pass.
case "$blockT" in *'_trim_ws'*) : ;; *) echo "FAIL: _trim_ws extraction broke"; exit 1 ;; esac
case "$blockR" in *'STUDIO_HOME="$UNSLOTH_ROOT/studio"'*) : ;;
    *) echo "FAIL: _resolve_studio_destinations extraction broke"; exit 1 ;; esac
case "$blockU" in *'if [ -z "${UV_CACHE_DIR:-}" ]; then'*) : ;;
    *) echo "FAIL: uv cache block extraction broke"; exit 1 ;; esac
case "$blockU" in *'unset _uv_cache_probe'*) : ;;
    *) echo "FAIL: uv cache block was truncated before its end"; exit 1 ;; esac
case "$blockE" in *'_epr_default UV_CACHE_DIR'*) : ;;
    *) echo "FAIL: _export_portable_roots extraction broke"; exit 1 ;; esac

# The ordering under test is a property of install.sh, not of this snippet: assert it holds in
# the file too, or the snippet below would keep testing a sequence the script no longer has.
uv_line=$(grep -n '^# Keep uv.s cache on the same filesystem as the venv it fills\.$' "$INSTALL" | head -1 | cut -d: -f1)
epr_line=$(grep -n '^_export_portable_roots$' "$INSTALL" | head -1 | cut -d: -f1)
res_line=$(grep -n '^_resolve_studio_destinations$' "$INSTALL" | head -1 | cut -d: -f1)
check "install.sh still resolves, then seeds the cache, then exports the portable roots" "yes" \
    "$([ -n "$uv_line" ] && [ -n "$epr_line" ] && [ -n "$res_line" ] \
       && [ "$res_line" -lt "$uv_line" ] && [ "$uv_line" -lt "$epr_line" ] && echo yes || echo no)"

SNIP='set -e
substep() { printf "SUBSTEP %s\n" "$1"; }
'"$blockT"'
'"$blockR"'
'"$blockE"'
_PORTABLE_MODE="$FIXTURE_PORTABLE"
_PORTABLE_FLAT="$FIXTURE_FLAT"
_UNSLOTH_ROOT="$FIXTURE_ROOT"
UNSLOTH_STUDIO_HOME="$FIXTURE_ROOT"
_resolve_studio_destinations
'"$blockU"'
[ "$_PORTABLE_MODE" = true ] && _export_portable_roots
printf "STUDIO_HOME=%s\n" "$STUDIO_HOME"
printf "UV_CACHE_DIR=%s\n" "${UV_CACHE_DIR:-<unset>}"'

run() { # portable, flat, root, [extra env assignments]
    # shellcheck disable=SC2086
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" \
        FIXTURE_PORTABLE="$1" FIXTURE_FLAT="$2" FIXTURE_ROOT="$3" $4 \
        sh -c "$SNIP" _ 2>&1
}
field() { printf '%s\n' "$1" | sed -n "s/^$2=//p"; }
kept_lines() { printf '%s\n' "$1" | grep -c 'kept your own cache locations' | tr -d ' '; }

# ── 1. nested portable: the install fills the cache the launchers read ──
R="$T/nested"
out="$(run true false "$R" "")"
check "nested portable resolves the child Studio home" "$R/studio" "$(field "$out" STUDIO_HOME)"
check "nested portable fills the root cache, not the Studio one" "$R/cache/uv" \
    "$(field "$out" UV_CACHE_DIR)"
check "and the install-time cache is the directory that was created" "yes" \
    "$([ -d "$R/cache/uv" ] && echo yes || echo no)"
check "no stray Studio-level uv cache is created" "no" \
    "$([ -d "$R/studio/cache/uv" ] && echo yes || echo no)"
check "a user who set nothing is not told their caches were kept" "0" "$(kept_lines "$out")"

# ── 2. flat portable: same answer, and no false warning either ──
F="$T/flat"
outf="$(run true true "$F" "")"
check "flat portable keeps the root as its Studio home" "$F" "$(field "$outf" STUDIO_HOME)"
check "flat portable also lands on the root cache" "$F/cache/uv" "$(field "$outf" UV_CACHE_DIR)"
check "flat portable does not claim to have kept a cache" "0" "$(kept_lines "$outf")"

# ── 3. an explicit value is still preserved AND still reported ──
# The fix must not overcorrect: a cache the caller really did name is left alone, and the
# warning that it may sit outside the root still fires.
E="$T/explicit"
oute="$(run true false "$E" "UV_CACHE_DIR=$T/mine/uv")"
check "an explicit UV_CACHE_DIR survives the whole sequence" "$T/mine/uv" \
    "$(field "$oute" UV_CACHE_DIR)"
check "and is still reported as kept" "1" "$(kept_lines "$oute")"
check "the kept warning names UV_CACHE_DIR" "yes" \
    "$(printf '%s\n' "$oute" | grep -q 'kept your own cache locations:.*UV_CACHE_DIR' && echo yes || echo no)"
# A blank inherited value counts as unset here as everywhere else, so it cannot pin the cache
# to the Studio path by defeating the seed's own -z test.
outb="$(run true false "$T/blank" "UV_CACHE_DIR=")"
check "a blank UV_CACHE_DIR still selects the root cache" "$T/blank/cache/uv" \
    "$(field "$outb" UV_CACHE_DIR)"

# ── 4. a normal install is unchanged: the cache follows STUDIO_HOME ──
N="$T/normal"
outn="$(run false false "$N" "")"
check "a non-portable install still co-locates with STUDIO_HOME" "$N/cache/uv" \
    "$(field "$outn" UV_CACHE_DIR)"

# ── 5. structural: the installer, the shim and studio.conf must agree ──
# Read out of install.sh's own writer lines, so moving either writer off <root>/cache/uv
# breaks this test rather than silently splitting the cache again.
shim_default="$(grep -F "export UV_CACHE_DIR='\$_shim_root/cache/uv'" "$INSTALL" | head -1)"
conf_default="$(grep -F "export UV_CACHE_DIR='\$_css_quoted_root/cache/uv'" "$INSTALL" | head -1)"
check "the shim still defaults the uv cache to <root>/cache/uv" "yes" \
    "$([ -n "$shim_default" ] && echo yes || echo no)"
check "share/studio.conf still defaults the uv cache to <root>/cache/uv" "yes" \
    "$([ -n "$conf_default" ] && echo yes || echo no)"
# Both writers are guarded, so an explicit value at launch still wins.
check "the shim default is guarded, not forced" "yes" \
    "$(printf '%s\n' "$shim_default" | grep -q '\[ -n .*UV_CACHE_DIR' && echo yes || echo no)"

if [ "$fails" -eq 0 ]; then echo "All checks passed"; else echo "$fails check(s) failed"; fi
[ "$fails" -eq 0 ]
