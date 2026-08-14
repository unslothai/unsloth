#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the uv cache co-location in install.sh.
#
# History: uv's cache defaulted to $HOME/.cache/uv while STUDIO_HOME can be pointed
# anywhere with UNSLOTH_STUDIO_HOME. uv hardlinks wheels out of its cache into the venv
# when both are on one filesystem and COPIES when they are not, so every redirected
# install paid twice the disk cost and left several GB of cache on the drive the user had
# deliberately moved the install off -- an SD card or second disk being the usual reason
# to redirect at all.
#
# Measured with torch 2.11.0+cpu: co-located, a 749 MB cache and a 748 MB venv occupy
# 755 MB between them (st_nlink 2 on the shared objects); across a filesystem boundary
# the same files are duplicated (st_nlink 1).
#
# The contract:
#   * unset UV_CACHE_DIR  -> $STUDIO_HOME/cache/uv, exported, directory created
#   * caller-set          -> left exactly as-is
#   * follows STUDIO_HOME wherever UNSLOTH_STUDIO_HOME puts it
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

# Lift the block out of install.sh so the real code is what runs here.
_FN_FILE=$(mktemp)
_TMP=$(mktemp -d)
trap 'rm -rf "$_FN_FILE" "$_TMP"' EXIT
awk '/^# Keep uv.s cache on the same filesystem as the venv it fills\.$/,/^fi$/' \
    "$INSTALL_SH" > "$_FN_FILE"

if ! grep -q 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$_FN_FILE"; then
    echo "FAIL: could not extract the UV_CACHE_DIR block from install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
_run() {  # $1 = STUDIO_HOME, $2 = preset UV_CACHE_DIR ("" for unset)
    "$_SH" -c "
        STUDIO_HOME='$1'
        if [ -n '$2' ]; then UV_CACHE_DIR='$2'; export UV_CACHE_DIR; else unset UV_CACHE_DIR; fi
        . '$_FN_FILE'
        printf '%s' \"\${UV_CACHE_DIR:-<unset>}\"
    "
}

echo "=== unset: defaults under STUDIO_HOME ==="
assert_eq "default home"    "$_TMP/studio/cache/uv" "$(_run "$_TMP/studio" '')"
# The whole point: a redirected STUDIO_HOME takes the cache with it.
assert_eq "redirected home" "$_TMP/sdcard/unsloth/cache/uv" "$(_run "$_TMP/sdcard/unsloth" '')"

echo "=== the directory is actually created (uv would too, but not before we log) ==="
_run "$_TMP/mk" '' >/dev/null
assert_eq "cache dir created" "yes" "$([ -d "$_TMP/mk/cache/uv" ] && echo yes || echo no)"

echo "=== an uncreatable cache falls back to uv's default, it does not stay exported ==="
# uv aborts with "Failed to initialize cache at ..." on a cache path it cannot create, so
# keeping the export after a failed mkdir turns a disk optimisation into a hard install
# failure on a host where uv's own default cache would have worked. Reachable with a
# writable STUDIO_HOME whose "cache" entry is not a usable directory.
: > "$_TMP/blocked"          # STUDIO_HOME is a FILE -> $STUDIO_HOME/cache/uv cannot exist
assert_eq "uncreatable cache is dropped" "<unset>" "$(_run "$_TMP/blocked" '')"
mkdir -p "$_TMP/rofile" && : > "$_TMP/rofile/cache"   # writable home, "cache" is a file
assert_eq "cache-as-file is dropped"     "<unset>" "$(_run "$_TMP/rofile" '')"
# ... and the fallback really is usable, unlike the path we just refused.
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
# Match the call that creates the venv, not the label it happens to carry. The
# literal 'run_install_cmd "create venv" uv venv' stopped existing when #8479
# moved venv creation behind _run_uv_venv, and a label this file cannot find
# reads as "the cache is set too late" rather than "the grep is stale". Comment
# lines are dropped so the prose above the helper does not answer first.
_venv_line=$(grep -nE '(^|[^[:alnum:]_"`])uv venv([[:space:]]|$)' "$INSTALL_SH" \
    | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1)
assert_eq "found the venv creation call" "yes" "$([ -n "$_venv_line" ] && echo yes || echo no)"
assert_eq "precedes venv creation" "yes" \
    "$([ -n "$_set_line" ] && [ -n "$_venv_line" ] && [ "$_set_line" -lt "$_venv_line" ] && echo yes || echo no)"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
