#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The Radeon repo route must not install a torch that cannot run torch.compile on
# the venv's Python. repo.radeon.com rocm-rel-6.4 carries torch 2.5.1 as its only
# cp313 build; a fresh Linux install creates a Python 3.13 venv, so a Radeon card
# floored to rocm6.4 got a venv that imports but fails "Dynamo is not supported on
# Python 3.13+" on the first training step. The ROCm index has 2.9.1 for cp313.
# Helpers are extracted from install.sh and sourced.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"

_FUNC_FILE=$(mktemp)
{
    sed -n '/^_pick_radeon_wheel()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_radeon_torch_compiles_for_pytag()/,/^}/p' "$INSTALL_SH"
} > "$_FUNC_FILE"
grep -q '^_radeon_torch_compiles_for_pytag()' "$_FUNC_FILE" || { echo "FATAL: helper not extracted"; exit 1; }
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

_can() { if _radeon_torch_compiles_for_pytag "$1" "$2"; then echo yes; else echo no; fi; }

echo "=== _radeon_torch_compiles_for_pytag: Dynamo per Python ==="
assert_eq "2.5 on cp313 cannot compile"  "no"  "$(_can 2.5 cp313)"
assert_eq "2.6 on cp313 compiles"        "yes" "$(_can 2.6 cp313)"
assert_eq "2.9 on cp313 compiles"        "yes" "$(_can 2.9 cp313)"
assert_eq "2.9 on cp314 cannot compile"  "no"  "$(_can 2.9 cp314)"
assert_eq "2.10 on cp314 compiles"       "yes" "$(_can 2.10 cp314)"
assert_eq "3.0 on cp314 compiles"        "yes" "$(_can 3.0 cp314)"
assert_eq "2.5 on cp312 is unconstrained" "yes" "$(_can 2.5 cp312)"
assert_eq "unparsable version on cp313 is refused" "no" "$(_can abc cp313)"

echo "=== the real rocm-rel-6.4 listing ==="
# Wheel names as published under https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/.
_RADEON_BASE_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/"
_RADEON_LISTING=$(for w in \
    torch-2.5.1%2Brocm6.4.0.git640334b6-cp312-cp312-linux_x86_64.whl \
    torch-2.5.1%2Brocm6.4.0.git640334b6-cp313-cp313-linux_x86_64.whl \
    torch-2.6.0%2Brocm6.4.0.git2fb0ac2b-cp312-cp312-linux_x86_64.whl \
    torchvision-0.20.1%2Brocm6.4.0.git04d8fc4a-cp313-cp313-linux_x86_64.whl \
    torchvision-0.21.0%2Brocm6.4.0.git4040d51f-cp312-cp312-linux_x86_64.whl \
    torchaudio-2.5.0%2Brocm6.4.0.git56bc006d-cp313-cp313-linux_x86_64.whl \
    torchaudio-2.6.0%2Brocm6.4.0.gitd8831425-cp312-cp312-linux_x86_64.whl; do
    printf '<a href="%s">%s</a>\n' "$w" "$w"; done)
_ver_of() { printf '%s' "${1##*/}" | sed 's/%2[Bb]/+/g' | sed -n 's|^torch-\([0-9]*\.[0-9]*\).*|\1|p'; }

_RADEON_PYTAG=cp313
_w=$(_pick_radeon_wheel torch)
assert_eq "cp313 newest torch in rocm-rel-6.4 is 2.5" "2.5" "$(_ver_of "$_w")"
assert_eq "and the route rejects it" "no" "$(_can "$(_ver_of "$_w")" "$_RADEON_PYTAG")"
_RADEON_PYTAG=cp312
_w=$(_pick_radeon_wheel torch)
assert_eq "cp312 newest torch in rocm-rel-6.4 is 2.6" "2.6" "$(_ver_of "$_w")"
assert_eq "and the route keeps it" "yes" "$(_can "$(_ver_of "$_w")" "$_RADEON_PYTAG")"

echo "=== the check gates the Radeon install ==="
_loop=$(grep -n 'Loop downwards to find the first complete matching trio' "$INSTALL_SH" | head -1 | cut -d: -f1)
_guard=$(grep -n '! _radeon_torch_compiles_for_pytag "\$_sel_torch_ver" "\$_RADEON_PYTAG"' "$INSTALL_SH" | head -1 | cut -d: -f1)
_fallback=$(grep -n 'Radeon repo lacks a compatible wheel set for this Python' "$INSTALL_SH" | head -1 | cut -d: -f1)
_install=$(grep -n 'installing PyTorch from Radeon repo' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "check runs after the trio search, before the fallback decision" "yes" \
    "$([ -n "$_loop" ] && [ -n "$_guard" ] && [ -n "$_fallback" ] && [ "$_loop" -lt "$_guard" ] && [ "$_guard" -lt "$_fallback" ] && [ "$_fallback" -lt "$_install" ] && echo yes)"
assert_eq "a rejected set clears the match flag the fallback reads" "yes" \
    "$(sed -n "${_guard},$((_guard + 3))p" "$INSTALL_SH" | grep -q '_radeon_versions_match=false' && echo yes)"

echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "$FAIL check(s) FAILED"
    exit 1
fi
echo "All $PASS checks passed"
