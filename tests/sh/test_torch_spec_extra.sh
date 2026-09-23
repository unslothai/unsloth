#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests UNSLOTH_TORCH_EXTRA plumbing in install.sh: spec rewriter, TheRock gfx map, pin guard.
# Functions are lifted from install.sh, not restated, so drift fails the test.
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
# TheRock has no gfx803 target, so Polaris must not be pointed at wheels that do not exist (#8529, #8458).
assert_eq "gfx803 (Polaris)"   "__none__"       "$(run_gfx gfx803)"
# Arches Unsloth's own indexes already cover must never reach this path.
assert_eq "gfx1030 covered"    "__none__"       "$(run_gfx gfx1030)"
assert_eq "gfx1100 covered"    "__none__"       "$(run_gfx gfx1100)"
assert_eq "empty arg"          "__none__"       "$(run_gfx '')"

echo ""
echo "=== UNSLOTH_TORCH_EXTRA is ignored without a pinned index ==="
# Restated from install.sh: no auto-picked index publishes extras, so an unpinned extra could only break a working resolve.
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
# TheRock leaves torchaudio bare; it reaches the right build through torch's rocm[libraries] dependency.
assert_eq "audio bare in _install_torch_default_index" "0" \
    "$(awk '/^_install_torch_default_index\(\) \{/{p=1} p{print} p&&/^\}/{exit}' "$INSTALL_SH" \
        | grep -c '_torch_spec_with_extra "\$TORCHAUDIO_CONSTRAINT"')"
# ...and torch/torchvision always do: three uv invocations, two rewritten specs each.
assert_eq "torch+vision rewritten on every uv line" "6" \
    "$(awk '/^_install_torch_default_index\(\) \{/{p=1} p{print} p&&/^\}/{exit}' "$INSTALL_SH" \
        | grep -o '_torch_spec_with_extra' | wc -l | tr -d ' ')"

echo ""
echo "=== the extras install reports back instead of failing silently ==="
# The flavor enforcement is gated on a recognised leaf, which an extras index is not, so
# without this block a TheRock install landing on CPU looks exactly like one that worked.
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
# The stubs speak the probe's sentinel, not a bare boolean: the block filters stdout to
# UNSLOTH_CUDA_OK= lines, so a bare "True" is dropped exactly as a banner would be.
assert_contains "usable GPU is confirmed, not silent" \
    "$(run_probe_block 'echo UNSLOTH_CUDA_OK=True')" "torch reports the GPU is usable"
# Why the sentinel exists: reading all of stdout gave "BANNER\nTrue", so a working GPU was
# reported as landing on CPU.
assert_contains "a startup banner does not hide the answer" \
    "$(run_probe_block 'echo "sitecustomize: hello"; echo UNSLOTH_CUDA_OK=True')" \
    "torch reports the GPU is usable"
assert_contains "CPU landing warns" \
    "$(run_probe_block 'echo UNSLOTH_CUDA_OK=False')" "torch.cuda.is_available() is False"
assert_contains "CPU landing asks for a report" \
    "$(run_probe_block 'echo UNSLOTH_CUDA_OK=False')" "Please report the result"
# A torch that cannot import at all prints nothing; the message must still be readable.
assert_contains "unimportable torch is labelled" \
    "$(run_probe_block 'exit 1')" "torch did not import"
# The block must never fire on a default (no extra) run: that is every existing install.
assert_eq "no extra: block is inert" "" \
    "$(bash -c '
        set -euo pipefail
        substep() { printf "  %s\n" "$1"; }
        _run_bounded() { echo UNSLOTH_CUDA_OK=False; }
        SKIP_TORCH=false; _VENV_PY=/nonexistent; _TORCH_EXTRA=""
        '"$(awk '/^# An extras pin lands on a leaf/{p=1} p{print} p&&/^fi$/{exit}' "$INSTALL_SH")"'
    ' 2>&1)"

echo ""
echo "=== an existing extra is merged, not stacked ==="
# torch[a][b] is not a PEP 508 requirement and uv rejects it; torch[a,b] is.
_stack=$(bash -c '
    _TORCH_EXTRA="device-gfx1010"
    '"$(sed -n '/^_torch_spec_with_extra()/,/^}/p' "$INSTALL_SH")"'
    _torch_spec_with_extra "torch[rocm]>=2.4,<2.11.0"')
assert_eq "existing extra joins the same bracket" "torch[rocm,device-gfx1010]>=2.4,<2.11.0" "$_stack"

echo ""
echo "=== the spec rewriter is POSIX, not bashism ==="
# install.sh runs under /bin/sh on hosts where that is dash.
for _sh in dash sh bash; do
    command -v "$_sh" >/dev/null 2>&1 || continue
    _posix=$("$_sh" -c '
        _TORCH_EXTRA="device-gfx1010"
        '"$(sed -n '/^_torch_spec_with_extra()/,/^}/p' "$INSTALL_SH")"'
        _torch_spec_with_extra "torch>=2.4,<2.11.0"' 2>&1)
    assert_eq "$_sh rewrites the spec identically" "torch[device-gfx1010]>=2.4,<2.11.0" "$_posix"
    _inert=$("$_sh" -c '
        _TORCH_EXTRA=""
        '"$(sed -n '/^_torch_spec_with_extra()/,/^}/p' "$INSTALL_SH")"'
        _torch_spec_with_extra "torch>=2.4,<2.11.0"' 2>&1)
    assert_eq "$_sh leaves a default run untouched" "torch>=2.4,<2.11.0" "$_inert"
done

echo ""
echo "=== a migrated venv still reaches the pinned index ==="
# The repair beside it is gated on _torch_index_is_rocm_family, which whl-multi-arch is not,
# so without this arm the documented two exports install PyPI torch and stop.
_migrated_block=$(awk '/^    # The ROCm repair above cannot reach an extras pin/{p=1} p{print} p&&/^    fi$/{exit}' "$INSTALL_SH")
assert_contains "the migrated arm exists" "$_migrated_block" "_install_torch_default_index --force-reinstall"
run_migrated() {
    bash -c '
        set -euo pipefail
        C_WARN=""
        substep() { printf "  %s\n" "$1"; }
        _install_torch_default_index() { printf "  REINSTALL %s\n" "$*"; }
        SKIP_TORCH='"$1"'; _torch_index_is_rocm_family='"$2"'; _TORCH_EXTRA="'"$3"'"
        '"$_migrated_block"'
    ' 2>&1
}
assert_contains "extras pin repairs a migrated venv" \
    "$(run_migrated false false device-gfx1010)" "REINSTALL --force-reinstall"
assert_eq "no extra: migrated arm is inert" "" "$(run_migrated false false '')"
assert_eq "rocm family: the existing repair owns it" "" \
    "$(run_migrated false true device-gfx1010)"
assert_eq "--no-torch: nothing is installed" "" "$(run_migrated true false device-gfx1010)"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
