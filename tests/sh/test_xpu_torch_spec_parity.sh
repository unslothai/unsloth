#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The XPU torch trio must match across install.sh, install_python_stack.py and install.ps1; the
# 2.6 floor matters since _utils.py raises below it. Win-arm64 drops torchaudio (no wheel).
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/../.."
INSTALL_SH="$ROOT/install.sh"
STACK_PY="$ROOT/studio/install_python_stack.py"
INSTALL_PS1="$ROOT/install.ps1"

PASS=0
FAIL=0
check() {
    if [ "$2" = "$3" ]; then
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %-38s got=%s want=%s\n' "$1" "$2" "$3"
        FAIL=$((FAIL + 1))
    fi
}

TORCH='torch>=2.6,<2.11.0'
VISION='torchvision>=0.21,<0.26.0'
AUDIO='torchaudio>=2.6,<2.11.0'

sh_has() { grep -qF "\"$1\"" "$INSTALL_SH" && echo yes || echo no; }
check "install.sh torch floor"    "$(sh_has "$TORCH")"  yes
check "install.sh torchvision"    "$(sh_has "$VISION")" yes
check "install.sh torchaudio"     "$(sh_has "$AUDIO")"  yes

# Read the tuple, not the whole file: a stray match elsewhere would hide drift.
spec=$(awk '/^_XPU_TORCH_PKG_SPEC/, /^\)/' "$STACK_PY")
[ -n "$spec" ] || { echo "FATAL: _XPU_TORCH_PKG_SPEC not found in $STACK_PY" >&2; exit 1; }
py_has() { printf '%s' "$spec" | grep -qF "\"$1\"" && echo yes || echo no; }
check "install_python_stack torch"       "$(py_has "$TORCH")"  yes
check "install_python_stack torchvision" "$(py_has "$VISION")" yes
check "install_python_stack torchaudio"  "$(py_has "$AUDIO")"  yes

ps_has() { grep -qF "\"$1\"" "$INSTALL_PS1" && echo yes || echo no; }
check "install.ps1 torch floor"   "$(ps_has "$TORCH")"  yes
check "install.ps1 torchvision"   "$(ps_has "$VISION")" yes
check "install.ps1 torchaudio"    "$(ps_has "$AUDIO")"  yes

check "stack classifies the xpu leaf" \
    "$(grep -q '_TORCH_BACKEND = "xpu"' "$STACK_PY" && echo yes || echo no)" yes
check "stack repairs an xpu pin" \
    "$(grep -q 'def _ensure_xpu_torch' "$STACK_PY" && echo yes || echo no)" yes
check "repair runs at both call sites" \
    "$(grep -c '^        if _ensure_xpu_torch() is False:' "$STACK_PY")" 2
check "rocm helper skips xpu" \
    "$(grep -q '_TORCH_BACKEND in ("cuda", "cpu", "xpu")' "$STACK_PY" && echo yes || echo no)" yes

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
