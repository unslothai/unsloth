#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests for the torch-trio --overrides guard on the Step-2 unsloth installs in
# install.sh. Released unsloth wheels can pin an older torch (unsloth 2026.7.2
# declares torch<2.11.0); without the overrides file a with-deps resolve from
# PyPI silently downgrades the torch trio Step 1 just installed, and the flavor
# guard cannot see it (PyPI's torch 2.10 default is itself cu128-flavored).
# Follows the same assertion pattern as test_torch_constraint.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

assert_true() {
    _label="$1"; _ok="$2"
    if [ "$_ok" = "0" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== test_unsloth_torch_override ==="

# 1. Both with-deps unsloth installs carry the overrides expansion. The no-torch
#    path installs --no-deps and needs no guard.
_local_block=$(grep -A2 '"install unsloth (local)"' "$INSTALL_SH")
printf '%s' "$_local_block" | grep -q -- '--overrides "\$_UNSLOTH_TORCH_OVERRIDES"'
assert_true "local (with-deps) unsloth install passes --overrides" "$?"

_generic_block=$(grep -A2 '"install unsloth" uv pip install' "$INSTALL_SH")
printf '%s' "$_generic_block" | grep -q -- '--overrides "\$_UNSLOTH_TORCH_OVERRIDES"'
assert_true "generic (with-deps) unsloth install passes --overrides" "$?"

_no_torch_block=$(grep -A2 '"install unsloth (no-torch)"' "$INSTALL_SH")
if printf '%s' "$_no_torch_block" | grep -q -- '--overrides'; then _rc=1; else _rc=0; fi
assert_true "no-torch (--no-deps) unsloth install has no overrides" "$_rc"

# 2. The overrides file is only built when SKIP_TORCH=false.
grep -B2 '_torch_trio_pins=\$(' "$INSTALL_SH" | grep -q 'SKIP_TORCH" = false'
assert_true "overrides file build is gated on SKIP_TORCH=false" "$?"

# 3. The pin-collection snippet emits exact ==pins for the installed trio (run
#    the embedded python against this test's own interpreter).
_snippet=$(sed -n '/_torch_trio_pins=\$("\$_VENV_PY" -c "/,/^" 2>\/dev\/null)/p' "$INSTALL_SH" \
    | sed '1s/.*-c "//' | sed '$d')
_out=$(python3 -c "$_snippet" 2>&1) || true
# torch may or may not be importable on the test host; the snippet must not
# crash and every line it does emit must be an exact pkg==version pin.
if [ -n "$_out" ]; then
    printf '%s\n' "$_out" | grep -vqE '^(torch|torchvision|torchaudio)==.+$' && _rc=1 || _rc=0
else
    _rc=0
fi
assert_true "pin snippet emits only exact trio ==pins (or nothing)" "$_rc"

# 4. The temp overrides file is cleaned up after Step 2.
grep -q 'rm -f "\$_UNSLOTH_TORCH_OVERRIDES"' "$INSTALL_SH"
assert_true "overrides temp file is removed after the unsloth installs" "$?"

# 5. Any UV_OVERRIDE env file is folded into the temp overrides file (the CLI
#    --overrides flag would otherwise replace it, dropping e.g. the macOS arm64
#    darwin overrides on the generic install path).
grep -q 'for _ov_file in \${UV_OVERRIDE:-}' "$INSTALL_SH"
assert_true "UV_OVERRIDE env files are merged into the overrides file" "$?"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
