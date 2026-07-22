#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's torch-flavor helpers (_torch_flavor_tag,
# _expected_torch_flavor_tag, _torch_index_repairable) that drive the
# stale-CPU-PyTorch repair. Helpers are extracted from install.sh and sourced.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# Extract the helper functions from install.sh and source them
# (_torch_index_url_leaf is the shared leaf extractor the classifiers call).
_FUNC_FILE=$(mktemp)
{
    sed -n '/^_torch_flavor_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_torch_index_url_leaf()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_is_pip_rocm_family_leaf()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_expected_torch_flavor_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_torch_index_repairable()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_tauri_torch_index_family()/,/^}/p' "$INSTALL_SH"
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

echo "=== _torch_flavor_tag ==="
assert_eq "cu130 wheel"        "cu130" "$(_torch_flavor_tag '2.10.0+cu130')"
assert_eq "cu128 wheel"        "cu128" "$(_torch_flavor_tag '2.8.0+cu128')"
assert_eq "cu124 wheel"        "cu124" "$(_torch_flavor_tag '2.5.1+cu124')"
assert_eq "cu118 wheel"        "cu118" "$(_torch_flavor_tag '2.4.0+cu118')"
assert_eq "cpu wheel"          "cpu"   "$(_torch_flavor_tag '2.10.0+cpu')"
assert_eq "untagged -> cpu"    "cpu"   "$(_torch_flavor_tag '2.10.0')"
assert_eq "rocm wheel"         "rocm"  "$(_torch_flavor_tag '2.11.0+rocm7.1')"
assert_eq "nightly cu130"      "cu130" "$(_torch_flavor_tag '2.10.0.dev20250601+cu130')"
assert_eq "cu130 with suffix"  "cu130" "$(_torch_flavor_tag '2.10.0+cu130.post1')"
assert_eq "empty -> empty"     ""      "$(_torch_flavor_tag '')"
assert_eq "garbage -> cpu"     "cpu"   "$(_torch_flavor_tag 'not-a-version')"

echo "=== _expected_torch_flavor_tag ==="
assert_eq "cu130 index"        "cu130" "$(_expected_torch_flavor_tag 'https://download.pytorch.org/whl/cu130')"
assert_eq "cu130 trailing /"   "cu130" "$(_expected_torch_flavor_tag 'https://download.pytorch.org/whl/cu130/')"
assert_eq "cu128 index"        "cu128" "$(_expected_torch_flavor_tag 'https://download.pytorch.org/whl/cu128')"
assert_eq "cpu index"          "cpu"   "$(_expected_torch_flavor_tag 'https://download.pytorch.org/whl/cpu')"
assert_eq "rocm index"         "rocm"  "$(_expected_torch_flavor_tag 'https://download.pytorch.org/whl/rocm7.2')"
assert_eq "amd gfx index"      "rocm"  "$(_expected_torch_flavor_tag 'https://repo.amd.com/rocm/whl/gfx120X-all/')"
assert_eq "mirror cu130 leaf"  "cu130" "$(_expected_torch_flavor_tag 'https://my.mirror/pytorch/whl/cu130')"
assert_eq "unrecognized leaf"  ""      "$(_expected_torch_flavor_tag 'https://my.mirror/whl/simple')"
assert_eq "empty url"          ""      "$(_expected_torch_flavor_tag '')"
# Query/fragment dropped before classification: .../cu128?token=x classifies as cu128,
# not an opaque leaf that reinstalls every run.
assert_eq "query-bearing cu128" "cu128" "$(_expected_torch_flavor_tag 'https://m/whl/cu128?token=x')"
assert_eq "fragment-bearing cpu" "cpu"  "$(_expected_torch_flavor_tag 'https://m/whl/cpu#frag')"
# A cu-suffixed CUSTOM leaf (cu128-private, cu128x) is NOT the cu128 family (exact
# cu+digits only). Mirrors Python re.fullmatch(cu[0-9]+) / PowerShell.
assert_eq "cu-suffix custom leaf" ""    "$(_expected_torch_flavor_tag 'https://m/whl/cu128-private')"
assert_eq "cu-alnum custom leaf"  ""    "$(_expected_torch_flavor_tag 'https://m/whl/cu128x')"
assert_eq "bare cu digits stays"  "cu126" "$(_expected_torch_flavor_tag 'https://m/whl/cu126')"
# A custom leaf merely STARTING with rocm (rocm-current, rocm-rel-7.2.1) is NOT a pip
# rocm family -> "" (custom); real families (rocm7.2) and gfx indexes stay "rocm".
assert_eq "custom rocm-current"   ""    "$(_expected_torch_flavor_tag 'https://mirror/whl/rocm-current')"
assert_eq "radeon rocm-rel leaf"  ""    "$(_expected_torch_flavor_tag 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1')"
assert_eq "real rocm7.2 stays"    "rocm" "$(_expected_torch_flavor_tag 'https://download.pytorch.org/whl/rocm7.2')"
# A rocm<digit>-SUFFIX private mirror (rocm7.2-private, rocm7-current) is a custom pin ->
# "" (custom); match the family exactly, not the prefix.
assert_eq "suffixed rocm7.2-private" "" "$(_expected_torch_flavor_tag 'https://co.internal/whl/rocm7.2-private')"
assert_eq "suffixed rocm7-current"   "" "$(_expected_torch_flavor_tag 'https://co.internal/whl/rocm7-current')"
assert_eq "two-dot rocm7.2.1"        "" "$(_expected_torch_flavor_tag 'https://co.internal/whl/rocm7.2.1')"
assert_eq "bare rocm7 stays"       "rocm" "$(_expected_torch_flavor_tag 'https://download.pytorch.org/whl/rocm7')"

echo "=== _torch_index_repairable ==="
assert_eq "cu130 repairable"   "yes"   "$(_torch_index_repairable 'https://download.pytorch.org/whl/cu130')"
assert_eq "rocm7.2 repairable" "yes"   "$(_torch_index_repairable 'https://download.pytorch.org/whl/rocm7.2')"
assert_eq "gfx repairable"     "yes"   "$(_torch_index_repairable 'https://repo.amd.com/rocm/whl/gfx120X-all/')"
assert_eq "gfx1151 repairable" "yes"   "$(_torch_index_repairable 'https://repo.amd.com/rocm/whl/gfx1151/')"
assert_eq "cpu NOT repairable" "no"    "$(_torch_index_repairable 'https://download.pytorch.org/whl/cpu')"
assert_eq "unknown NOT repair" "no"    "$(_torch_index_repairable 'https://my.mirror/whl/simple')"
# A suffixed rocm leaf is a verbatim pin, not a --default-index repairable family.
assert_eq "rocm-private NOT repair" "no" "$(_torch_index_repairable 'https://co.internal/whl/rocm7.2-private')"

echo "=== _is_pip_rocm_family_leaf ==="
assert_family() {
    _label="$1"; _expected="$2"; _leaf="$3"
    if _is_pip_rocm_family_leaf "$_leaf"; then _actual="yes"; else _actual="no"; fi
    assert_eq "$_label" "$_expected" "$_actual"
}
assert_family "rocm7.2 family"        "yes" "rocm7.2"
assert_family "rocm6.4 family"        "yes" "rocm6.4"
assert_family "bare rocm7 family"     "yes" "rocm7"
assert_family "gfx120x-all family"    "yes" "gfx120x-all"
assert_family "gfx1151 family"        "yes" "gfx1151"
assert_family "rocm7.2-private custom" "no" "rocm7.2-private"
assert_family "rocm7-current custom"  "no"  "rocm7-current"
assert_family "rocm-current custom"   "no"  "rocm-current"
assert_family "rocm-rel-7.2.1 custom" "no"  "rocm-rel-7.2.1"
assert_family "rocm7.2.1 custom"      "no"  "rocm7.2.1"
# A trailing dot (rocm7.) or leading/double dot is NOT a family: both major and minor must
# be non-empty all-digits, matching Python re.fullmatch(rocm\d+(?:\.\d+)?). Bash previously
# accepted rocm7. via a bare %/-style trim while Python rejected it (validator asymmetry).
assert_family "rocm7. trailing-dot custom" "no" "rocm7."
assert_family "rocm.7 leading-dot custom"  "no" "rocm.7"
assert_family "rocm7..2 double-dot custom" "no" "rocm7..2"
assert_family "cpu not rocm"          "no"  "cpu"
assert_family "cu128 not rocm"        "no"  "cu128"
assert_family "simple not rocm"       "no"  "simple"

echo "=== _torch_index_url_leaf (ALL trailing slashes stripped -> non-empty leaf) ==="
# A double (or triple) trailing slash must yield the real leaf, not an empty string that
# fails every classifier arm. Python .rstrip("/") drops them all; bash must match (a bare
# %/ left .../cu128// classifying as "").
assert_eq "double slash cu128 leaf"   "cu128"   "$(_torch_index_url_leaf 'https://m/whl/cu128//')"
assert_eq "triple slash rocm7.2 leaf" "rocm7.2" "$(_torch_index_url_leaf 'https://m/whl/rocm7.2///')"
assert_eq "double slash + token leaf" "cu128"   "$(_torch_index_url_leaf 'https://m/whl/cu128//?token=x')"
assert_eq "single slash cu128 leaf"   "cu128"   "$(_torch_index_url_leaf 'https://m/whl/cu128/')"
# The classifier that consumes the leaf must therefore still tag a double-slash index.
assert_eq "double-slash cu128 tag"    "cu128"   "$(_expected_torch_flavor_tag 'https://m/whl/cu128//')"
assert_eq "double-slash rocm7.2 tag"  "rocm"    "$(_expected_torch_flavor_tag 'https://m/whl/rocm7.2//')"

echo "=== _tauri_torch_index_family (credential redaction) ==="
# A token/fragment must be stripped BEFORE classification so it never reaches the
# [TAURI:DIAG] line (the family is the last path segment, which else carries the query).
SKIP_TORCH=false
assert_eq "token stripped from rocm"  "rocm7.2" "$(_tauri_torch_index_family 'https://mirror/whl/rocm7.2?token=SECRET')"
assert_eq "token-bearing cu classifies" "cu128" "$(_tauri_torch_index_family 'https://m/whl/cu128?token=x')"
assert_eq "fragment stripped cpu"     "cpu"     "$(_tauri_torch_index_family 'https://m/whl/cpu#frag')"
assert_eq "plain rocm7.2 unchanged"   "rocm7.2" "$(_tauri_torch_index_family 'https://download.pytorch.org/whl/rocm7.2')"
# A trailing slash must be stripped too, or the */cu128 and */cpu arms miss .../cu128/
# and it falls through to "auto".
assert_eq "trailing slash cu128"      "cu128"   "$(_tauri_torch_index_family 'https://download.pytorch.org/whl/cu128/')"
assert_eq "slash + token cu128"       "cu128"   "$(_tauri_torch_index_family 'https://m/whl/cu128/?token=x')"
assert_eq "trailing slash cpu"        "cpu"     "$(_tauri_torch_index_family 'https://m/whl/cpu/')"
# Regression guard: no secret token substring may survive in any classification.
_leak=$(_tauri_torch_index_family 'https://mirror/whl/rocm7.2?token=SECRET')
case "$_leak" in
    *SECRET*|*token*) assert_eq "no token leak in family" "clean" "leaked:$_leak" ;;
    *)                assert_eq "no token leak in family" "clean" "clean" ;;
esac

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
