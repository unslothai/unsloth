#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Guards uv cache co-location: uv hardlinks into the venv only on the same filesystem and copies
# otherwise, so the cache follows STUDIO_HOME unless the caller set UV_CACHE_DIR.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
_FN_FILE=$(mktemp)
_TMP=$(mktemp -d)
trap 'rm -rf "$_FN_FILE" "$_TMP"' EXIT
# The helper first: without it `if !` reads exit 127 as "not writable" and every case passes.
awk '/^_uv_cache_root_is_writable\(\) \{$/,/^\}$/' "$INSTALL_SH" > "$_FN_FILE"
awk '/^# Keep uv.s cache on the same filesystem as the venv it fills\.$/,/^fi$/' \
    "$INSTALL_SH" >> "$_FN_FILE"

if ! grep -q 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$_FN_FILE"; then
    echo "FAIL: could not extract the UV_CACHE_DIR block from install.sh"
    exit 1
fi
if ! grep -q '^_uv_cache_root_is_writable() {' "$_FN_FILE"; then
    echo "FAIL: could not extract _uv_cache_root_is_writable from install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
_run() {
    "$_SH" -c "
        STUDIO_HOME='$1'
        if [ -n '$2' ]; then UV_CACHE_DIR='$2'; export UV_CACHE_DIR; else unset UV_CACHE_DIR; fi
        . '$_FN_FILE'
        printf '%s' \"\${UV_CACHE_DIR:-<unset>}\"
    "
}

echo "=== unset: defaults under STUDIO_HOME ==="
assert_eq "default home"    "$_TMP/studio/cache/uv" "$(_run "$_TMP/studio" '')"
assert_eq "redirected home" "$_TMP/sdcard/unsloth/cache/uv" "$(_run "$_TMP/sdcard/unsloth" '')"

echo "=== the directory is actually created (uv would too, but not before we log) ==="
_run "$_TMP/mk" '' >/dev/null
assert_eq "cache dir created" "yes" "$([ -d "$_TMP/mk/cache/uv" ] && echo yes || echo no)"

echo "=== an uncreatable cache falls back to uv's default, it does not stay exported ==="
# uv aborts on a cache path it cannot create, so a failed mkdir must drop the export.
: > "$_TMP/blocked"
assert_eq "uncreatable cache is dropped" "<unset>" "$(_run "$_TMP/blocked" '')"
mkdir -p "$_TMP/rofile" && : > "$_TMP/rofile/cache"
assert_eq "cache-as-file is dropped"     "<unset>" "$(_run "$_TMP/rofile" '')"
# mkdir -p succeeds on an existing unwritable dir, so writability must be probed.
mkdir -p "$_TMP/leftover/cache/uv" && chmod 500 "$_TMP/leftover/cache/uv"
if [ "$(id -u)" = "0" ]; then
    echo "  SKIP: unwritable-cache case (root writes through the mode bits)"
else
    assert_eq "unwritable existing cache is dropped" "<unset>" "$(_run "$_TMP/leftover" '')"
fi
chmod 700 "$_TMP/leftover/cache/uv"
_run "$_TMP/probe" '' >/dev/null
assert_eq "write probe cleaned up" "" "$(ls -A "$_TMP/probe/cache/uv")"
if command -v uv >/dev/null 2>&1; then
    _uv_rc=$("$_SH" -c "
        STUDIO_HOME='$_TMP/rofile'
        unset UV_CACHE_DIR
        . '$_FN_FILE'
        uv venv '$_TMP/rofile/venv' >/dev/null 2>&1
        printf '%s' \"\$?\"
    ")
    assert_eq "uv still runs after the fallback" "0" "$_uv_rc"
fi

echo "=== a caller-set UV_CACHE_DIR is never overridden ==="
assert_eq "explicit value kept" "/custom/uvcache" "$(_run "$_TMP/studio" '/custom/uvcache')"

echo "=== it is exported, not just assigned (uv runs in child processes) ==="
_exported=$("$_SH" -c "
    STUDIO_HOME='$_TMP/studio'
    unset UV_CACHE_DIR
    . '$_FN_FILE'
    sh -c 'printf %s \"\${UV_CACHE_DIR:-<unset>}\"'
")
assert_eq "visible to child processes" "$_TMP/studio/cache/uv" "$_exported"

echo "=== structural: set before uv is first invoked ==="
_set_line=$(grep -n 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$INSTALL_SH" | head -1 | cut -d: -f1)
_uv_line=$(grep -n 'installing uv package manager' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "precedes uv bootstrap" "yes" \
    "$([ -n "$_set_line" ] && [ -n "$_uv_line" ] && [ "$_set_line" -lt "$_uv_line" ] && echo yes || echo no)"
# Match the call that creates the venv, not its label; comment lines and case globs are dropped.
_venv_line=$(grep -nE '(^|[^[:alnum:]_"`])uv venv([[:space:]]|$)' "$INSTALL_SH" \
    | grep -vE '^[0-9]+:[[:space:]]*#' | grep -v '\*" uv venv "\*' | head -1 | cut -d: -f1)
assert_eq "found the venv creation call" "yes" "$([ -n "$_venv_line" ] && echo yes || echo no)"
assert_eq "precedes venv creation" "yes" \
    "$([ -n "$_set_line" ] && [ -n "$_venv_line" ] && [ "$_set_line" -lt "$_venv_line" ] && echo yes || echo no)"

echo "=== studio/setup.sh sets the SAME cache (it is the standalone update entry point) ==="
# `unsloth studio update` runs studio/setup.sh directly, so it needs the same block.
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
_SETUP_FN=$(mktemp)
trap 'rm -rf "$_FN_FILE" "$_TMP" "$_SETUP_FN"' EXIT
awk '/^# Same uv cache install\.sh chose/,/^fi$/' "$SETUP_SH" > "$_SETUP_FN"
if ! grep -q 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$_SETUP_FN"; then
    echo "  FAIL: could not extract the UV_CACHE_DIR block from studio/setup.sh"
    FAIL=$((FAIL + 1))
else
    _run_setup() {
        "$_SH" -c "
            STUDIO_HOME='$1'
            if [ -n '$2' ]; then UV_CACHE_DIR='$2'; export UV_CACHE_DIR; else unset UV_CACHE_DIR; fi
            . '$_SETUP_FN'
            printf '%s' \"\${UV_CACHE_DIR:-<unset>}\"
        "
    }
    assert_eq "setup.sh defaults under STUDIO_HOME" \
        "$_TMP/upd/cache/uv" "$(_run_setup "$_TMP/upd" '')"
    assert_eq "setup.sh matches install.sh for a redirected home" \
        "$_TMP/sdcard/unsloth/cache/uv" "$(_run_setup "$_TMP/sdcard/unsloth" '')"
    assert_eq "setup.sh keeps a caller-set value" \
        "/custom/uvcache" "$(_run_setup "$_TMP/upd" '/custom/uvcache')"
    mkdir -p "$_TMP/updro" && : > "$_TMP/updro/cache"
    assert_eq "setup.sh drops an unusable cache" "<unset>" "$(_run_setup "$_TMP/updro" '')"
fi

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
