#!/bin/bash
# Unit tests for install.sh's torch-flavor helpers (_torch_flavor_tag,
# _expected_torch_flavor_tag, _torch_index_repairable) that drive the
# stale-CPU-PyTorch repair. Helpers are extracted from install.sh and sourced.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# Extract the three helper functions from install.sh and source them.
_FUNC_FILE=$(mktemp)
{
    sed -n '/^_torch_flavor_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_expected_torch_flavor_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_torch_index_repairable()/,/^}/p' "$INSTALL_SH"
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

echo "=== _torch_index_repairable ==="
assert_eq "cu130 repairable"   "yes"   "$(_torch_index_repairable 'https://download.pytorch.org/whl/cu130')"
assert_eq "rocm7.2 repairable" "yes"   "$(_torch_index_repairable 'https://download.pytorch.org/whl/rocm7.2')"
assert_eq "gfx repairable"     "yes"   "$(_torch_index_repairable 'https://repo.amd.com/rocm/whl/gfx120X-all/')"
assert_eq "gfx1151 repairable" "yes"   "$(_torch_index_repairable 'https://repo.amd.com/rocm/whl/gfx1151/')"
assert_eq "cpu NOT repairable" "no"    "$(_torch_index_repairable 'https://download.pytorch.org/whl/cpu')"
assert_eq "unknown NOT repair" "no"    "$(_torch_index_repairable 'https://my.mirror/whl/simple')"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
