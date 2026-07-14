#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's torch-index MARKER helpers (_normalize_index_url,
# _write_torch_index_marker), which record the resolved wheel --index-url so a later
# update detects a ROCm/gfx pin change by exact compare. Helpers extracted and sourced.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

_TORCH_INDEX_MARKER_NAME=".unsloth-torch-index"

# Extract the marker helpers from install.sh and source them.
_FUNC_FILE=$(mktemp)
{
    sed -n '/^_strip_index_url_credentials()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_torch_index_url_leaf()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_is_pip_rocm_family_leaf()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_normalize_family_leaf()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_normalize_index_url()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_write_torch_index_marker()/,/^}/p' "$INSTALL_SH"
} > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

echo "=== _normalize_index_url ==="
assert_eq "trailing slashes stripped + leaf lowered" \
    "https://repo.amd.com/rocm/whl/gfx120x-all" \
    "$(_normalize_index_url 'https://repo.amd.com/rocm/whl/gfx120X-all///')"
assert_eq "whitespace trimmed" \
    "https://download.pytorch.org/whl/cu128" \
    "$(_normalize_index_url '  https://download.pytorch.org/whl/cu128  ')"
assert_eq "host + custom (unknown-family) leaf case preserved" \
    "https://Mirror.Local/Simple" \
    "$(_normalize_index_url 'https://Mirror.Local/Simple/')"
# gfx120X-all (capital X) and AMD's lowercase pip leaf normalise equal.
assert_eq "gfx120X-all == gfx120x-all after normalize" \
    "$(_normalize_index_url 'https://repo.amd.com/rocm/whl/gfx120x-all')" \
    "$(_normalize_index_url 'https://repo.amd.com/rocm/whl/gfx120X-all')"
assert_eq "empty -> empty" "" "$(_normalize_index_url '   ')"
# rocm7.2 KNOWN-2.11 leaf normalises to itself.
assert_eq "rocm7.2 unchanged" \
    "https://download.pytorch.org/whl/rocm7.2" \
    "$(_normalize_index_url 'https://download.pytorch.org/whl/rocm7.2/')"
# A suffixed rocm7.2-Private mirror is a CUSTOM pin: its case must survive normalization
# (a case-only change is a real mismatch), unlike the exact rocm7.2 family.
assert_eq "suffixed rocm7.2-Private case preserved" \
    "https://co.internal/whl/rocm7.2-Private" \
    "$(_normalize_index_url 'https://co.internal/whl/rocm7.2-Private/')"

echo "=== _strip_index_url_credentials ==="
assert_eq "user:token@ stripped" \
    "https://mirror.local/simple" \
    "$(_strip_index_url_credentials 'https://user:tok@mirror.local/simple')"
assert_eq "credential-free url unchanged" \
    "https://mirror.local/simple" \
    "$(_strip_index_url_credentials 'https://mirror.local/simple')"
assert_eq "@ in path preserved" \
    "https://h/pa@th" \
    "$(_strip_index_url_credentials 'https://u:p@h/pa@th')"
# A token can also ride in the query/fragment; both are dropped so it never lands in
# the marker or logged output.
assert_eq "query token stripped" \
    "https://mirror.local/simple" \
    "$(_strip_index_url_credentials 'https://mirror.local/simple?token=SECRET')"
assert_eq "fragment stripped" \
    "https://mirror.local/simple" \
    "$(_strip_index_url_credentials 'https://mirror.local/simple#tok')"
assert_eq "userinfo and query both stripped" \
    "https://mirror.local/simple" \
    "$(_strip_index_url_credentials 'https://u:p@mirror.local/simple?token=SECRET')"
assert_eq "query stripped on host-only url" \
    "https://mirror.local" \
    "$(_strip_index_url_credentials 'https://mirror.local?token=SECRET')"
# Backward compatibility: an OLD marker that recorded credentials compares equal to the
# same pin with or without them (normalization strips both sides).
assert_eq "normalize: creds on either side compare equal" \
    "$(_normalize_index_url 'https://x/cu128')" \
    "$(_normalize_index_url 'https://user:tok@x/cu128/')"

echo "=== _torch_index_url_leaf ==="
assert_eq "query dropped before classification" \
    "cu128" "$(_torch_index_url_leaf 'https://m/whl/cu128?token=x')"
assert_eq "fragment dropped before classification" \
    "cu128" "$(_torch_index_url_leaf 'https://m/whl/cu128#frag')"
assert_eq "plain leaf lowercased" \
    "gfx120x-all" "$(_torch_index_url_leaf 'https://m/whl/gfx120X-all/')"

echo "=== _write_torch_index_marker ==="
_VD=$(mktemp -d)
_write_torch_index_marker "$_VD" "https://download.pytorch.org/whl/rocm7.2"
assert_eq "marker written verbatim (single line)" \
    "https://download.pytorch.org/whl/rocm7.2" \
    "$(cat "$_VD/$_TORCH_INDEX_MARKER_NAME" 2>/dev/null)"

# Overwrite (per-arch switch gfx1151 -> gfx120X-all): the marker is replaced.
_write_torch_index_marker "$_VD" "https://repo.amd.com/rocm/whl/gfx120X-all"
assert_eq "marker overwritten on re-install" \
    "https://repo.amd.com/rocm/whl/gfx120X-all" \
    "$(cat "$_VD/$_TORCH_INDEX_MARKER_NAME" 2>/dev/null)"

# Blank URL is ignored (nothing meaningful to record) -- existing marker kept.
_write_torch_index_marker "$_VD" "   "
assert_eq "blank url leaves prior marker intact" \
    "https://repo.amd.com/rocm/whl/gfx120X-all" \
    "$(cat "$_VD/$_TORCH_INDEX_MARKER_NAME" 2>/dev/null)"

# No stray temp files left behind by the atomic write.
_leftover=$(find "$_VD" -maxdepth 1 -name "$_TORCH_INDEX_MARKER_NAME.*.tmp" 2>/dev/null | wc -l | tr -d ' ')
assert_eq "no stray temp file left" "0" "$_leftover"

# Credentials never persist in the marker file.
_write_torch_index_marker "$_VD" "https://user:sekrit@mirror.local/simple"
assert_eq "marker stores credential-free url" \
    "https://mirror.local/simple" \
    "$(cat "$_VD/$_TORCH_INDEX_MARKER_NAME" 2>/dev/null)"
assert_eq "marker body has no secret" \
    "0" "$(grep -c sekrit "$_VD/$_TORCH_INDEX_MARKER_NAME" 2>/dev/null)"

# A query-carried token must not persist in the marker either.
_write_torch_index_marker "$_VD" "https://mirror.local/simple?token=sekrit2"
assert_eq "marker drops query token" \
    "https://mirror.local/simple" \
    "$(cat "$_VD/$_TORCH_INDEX_MARKER_NAME" 2>/dev/null)"
assert_eq "marker body has no query secret" \
    "0" "$(grep -c sekrit2 "$_VD/$_TORCH_INDEX_MARKER_NAME" 2>/dev/null)"

# Missing venv dir -> no-op, no error, no file created.
_MISSING="$_VD/does_not_exist_dir"
_write_torch_index_marker "$_MISSING" "https://x/cu128"
assert_eq "missing venv dir -> no marker" \
    "absent" \
    "$([ -e "$_MISSING/$_TORCH_INDEX_MARKER_NAME" ] && echo present || echo absent)"

rm -rf "$_VD"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
