#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests UNSLOTH_TORCH_EXTRA plumbing in install.sh: the spec rewriter, the TheRock gfx map,
# and the pin guard. Functions are lifted from install.sh rather than restated, so the test
# fails if the real implementation drifts.
# Follows the same assertion pattern as test_torch_constraint.sh.
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

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle' in: $_haystack)"
        FAIL=$((FAIL + 1))
    fi
}

# Lift a shell function body verbatim: from `name() {` to the first column-0 `}`.
lift_fn() {
    awk -v fn="$1" '$0 ~ "^"fn"\\(\\) \\{" {p=1} p {print} p && /^\}/ {exit}' "$INSTALL_SH"
}

FN_SPEC=$(lift_fn _torch_spec_with_extra)
FN_GFX=$(lift_fn _therock_device_extra_for_gfx)

[ -n "$FN_SPEC" ] || { echo "FAIL: _torch_spec_with_extra not found in install.sh"; exit 1; }
[ -n "$FN_GFX" ] || { echo "FAIL: _therock_device_extra_for_gfx not found in install.sh"; exit 1; }

echo "=== _torch_spec_with_extra: unset extra is a no-op ==="
run_spec() {
    _TORCH_EXTRA="$1" bash -c "$FN_SPEC"'
        _torch_spec_with_extra "$1"' _ "$2"
}

# Every default path leaves _TORCH_EXTRA empty; the spec must come back byte-identical.
for spec in 'torch>=2.4,<2.11.0' 'torch>=2.11.0,<2.12.0' 'torch==2.9.0' \
            'torchvision>=0.19,<0.26.0' 'torchvision==0.24.*' 'torchvision' 'torch~=2.8'; do
    assert_eq "unset: $spec unchanged" "$spec" "$(run_spec "" "$spec")"
done

echo ""
echo "=== _torch_spec_with_extra: extra lands before the version operator ==="
assert_eq "range spec"   'torch[device-gfx1010]>=2.4,<2.11.0'      "$(run_spec device-gfx1010 'torch>=2.4,<2.11.0')"
assert_eq "exact pin"    'torch[device-gfx1010]==2.9.0'            "$(run_spec device-gfx1010 'torch==2.9.0')"
assert_eq "wildcard pin" 'torchvision[device-gfx1010]==0.24.*'     "$(run_spec device-gfx1010 'torchvision==0.24.*')"
assert_eq "bare name"    'torchvision[device-gfx1010]'             "$(run_spec device-gfx1010 'torchvision')"
assert_eq "compatible"   'torch[device-gfx1012]~=2.8'              "$(run_spec device-gfx1012 'torch~=2.8')"

echo ""
echo "=== _therock_device_extra_for_gfx: only arches TheRock builds ==="
run_gfx() {
    bash -c "$FN_GFX"'
        _therock_device_extra_for_gfx "$1" 2>/dev/null || echo "__none__"' _ "$1"
}
assert_eq "gfx1010 (RX 5700)"  "device-gfx1010" "$(run_gfx gfx1010)"
assert_eq "gfx1011 (PRO V520)" "device-gfx1011" "$(run_gfx gfx1011)"
assert_eq "gfx1012 (RX 5500)"  "device-gfx1012" "$(run_gfx gfx1012)"
# Polaris keeps the message it already has: TheRock has no gfx803 target, so pointing a
# user at wheels that do not exist would be worse than saying nothing (#8529, #8458).
assert_eq "gfx803 (Polaris)"   "__none__"       "$(run_gfx gfx803)"
# Arches Unsloth's own indexes already cover must never reach this path.
assert_eq "gfx1030 covered"    "__none__"       "$(run_gfx gfx1030)"
assert_eq "gfx1100 covered"    "__none__"       "$(run_gfx gfx1100)"
assert_eq "empty arg"          "__none__"       "$(run_gfx '')"

echo ""
echo "=== UNSLOTH_TORCH_EXTRA is ignored without a pinned index ==="
# The guard, restated from install.sh's block: no index this script picks by itself
# publishes extras, so an unpinned extra could only turn a working resolve into an error.
run_guard() {
    UNSLOTH_TORCH_EXTRA="$1" UNSLOTH_TORCH_INDEX_URL="$2" UNSLOTH_TORCH_INDEX_FAMILY="$3" bash -c '
        _torch_index_pinned=false
        _ti_url_trim="${UNSLOTH_TORCH_INDEX_URL:-}"
        _ti_url_trim="${_ti_url_trim#"${_ti_url_trim%%[![:space:]]*}"}"; _ti_url_trim="${_ti_url_trim%"${_ti_url_trim##*[![:space:]]}"}"
        _ti_family_trim="${UNSLOTH_TORCH_INDEX_FAMILY:-}"
        _ti_family_trim="${_ti_family_trim#"${_ti_family_trim%%[![:space:]]*}"}"; _ti_family_trim="${_ti_family_trim%"${_ti_family_trim##*[![:space:]]}"}"
        if [ -n "$_ti_url_trim" ] || [ -n "$_ti_family_trim" ]; then
            _torch_index_pinned=true
        fi
        _TORCH_EXTRA=""
        _te_trim="${UNSLOTH_TORCH_EXTRA:-}"
        _te_trim="${_te_trim#"${_te_trim%%[![:space:]]*}"}"; _te_trim="${_te_trim%"${_te_trim##*[![:space:]]}"}"
        if [ -n "$_te_trim" ] && [ "$_torch_index_pinned" = true ]; then
            _TORCH_EXTRA="$_te_trim"
        fi
        printf "%s" "${_TORCH_EXTRA:-__empty__}"'
}
assert_eq "extra + URL pin honoured"     "device-gfx1010" "$(run_guard device-gfx1010 'https://rocm.nightlies.amd.com/whl-multi-arch/' '')"
assert_eq "extra + family pin honoured"  "device-gfx1010" "$(run_guard device-gfx1010 '' 'rocm7.2')"
assert_eq "extra alone ignored"          "__empty__"      "$(run_guard device-gfx1010 '' '')"
assert_eq "whitespace extra ignored"     "__empty__"      "$(run_guard '   ' 'https://example.invalid/whl/' '')"
# A whitespace-only pin is unset in get_torch_index_url, so it must not arm the extra either.
assert_eq "whitespace pin does not arm"  "__empty__"      "$(run_guard device-gfx1010 '   ' '')"
assert_eq "no extra set"                 "__empty__"      "$(run_guard '' 'https://rocm.nightlies.amd.com/whl-multi-arch/' '')"

echo ""
echo "=== torchaudio never carries the extra ==="
# TheRock's documented invocation leaves torchaudio bare; it reaches the right build
# through torch's own rocm[libraries] dependency.
assert_eq "audio bare in _install_torch_default_index" "0" \
    "$(awk '/^_install_torch_default_index\(\) \{/{p=1} p{print} p&&/^\}/{exit}' "$INSTALL_SH" \
        | grep -c '_torch_spec_with_extra "\$TORCHAUDIO_CONSTRAINT"')"
# ...and torch/torchvision always do: three uv invocations, two rewritten specs each.
assert_eq "torch+vision rewritten on every uv line" "6" \
    "$(awk '/^_install_torch_default_index\(\) \{/{p=1} p{print} p&&/^\}/{exit}' "$INSTALL_SH" \
        | grep -o '_torch_spec_with_extra' | wc -l | tr -d ' ')"

echo ""
echo "=== the extras install reports back instead of failing silently ==="
# The flavor enforcement above it is gated on a recognised leaf, which an extras index is
# not, so without this block a TheRock install that lands on CPU looks exactly like one
# that worked. Run the real block against a stub interpreter.
run_probe_block() {
    _stub_out="$1"
    _tmp=$(mktemp -d)
    # A stand-in for the venv python: prints what a probe of torch.cuda.is_available() would.
    printf '#!/bin/sh\n%s\n' "$_stub_out" > "$_tmp/py"
    chmod +x "$_tmp/py"
    awk '/^# An extras pin lands on a leaf/{p=1} p{print} p&&/^fi$/{exit}' "$INSTALL_SH" > "$_tmp/block.sh"
    bash -c '
        set -euo pipefail
        C_WARN=""; C_DIM=""; C_RST=""
        substep() { printf "  %s\n" "$1"; }
        _run_bounded() { "$@"; }
        SKIP_TORCH=false
        _VENV_PY="$1/py"
        _TORCH_EXTRA="device-gfx1010"
        . "$1/block.sh"' _ "$_tmp" 2>&1
    rm -rf "$_tmp"
}
assert_contains "usable GPU is confirmed, not silent" \
    "$(run_probe_block 'echo True')" "torch reports the GPU is usable"
assert_contains "CPU landing warns" \
    "$(run_probe_block 'echo False')" "torch.cuda.is_available() is False"
assert_contains "CPU landing asks for a report" \
    "$(run_probe_block 'echo False')" "Please report the result"
# A torch that cannot import at all prints nothing; the message must still be readable.
assert_contains "unimportable torch is labelled" \
    "$(run_probe_block 'exit 1')" "torch did not import"
# The block must never fire on a default (no extra) run: that is every existing install.
assert_eq "no extra: block is inert" "" \
    "$(bash -c '
        set -euo pipefail
        substep() { printf "  %s\n" "$1"; }
        _run_bounded() { echo False; }
        SKIP_TORCH=false; _VENV_PY=/nonexistent; _TORCH_EXTRA=""
        '"$(awk '/^# An extras pin lands on a leaf/{p=1} p{print} p&&/^fi$/{exit}' "$INSTALL_SH")"'
    ' 2>&1)"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
