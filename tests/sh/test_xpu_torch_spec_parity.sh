#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The Intel XPU torch trio must be identical in every place that installs it: install.sh,
# studio/install_python_stack.py (the `unsloth studio update` route) and install.ps1. A drifted
# floor gives a different torch depending on which command the user ran, and the 2.6 floor is
# not cosmetic -- unsloth/models/_utils.py raises at import for an XPU device below it.
#
# Windows on ARM legitimately drops torchaudio (no win_arm64 wheel), so install.ps1 is checked
# for the floors rather than for an identical trio.
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

# install.sh sets these in the xpu arm of its index-leaf case.
sh_has() { grep -qF "\"$1\"" "$INSTALL_SH" && echo yes || echo no; }
check "install.sh torch floor"    "$(sh_has "$TORCH")"  yes
check "install.sh torchvision"    "$(sh_has "$VISION")" yes
check "install.sh torchaudio"     "$(sh_has "$AUDIO")"  yes

# The shared stack keeps them in _XPU_TORCH_PKG_SPEC. Read the tuple rather than grepping the
# whole file: a stray match elsewhere would hide a drifted spec.
spec=$(awk '/^_XPU_TORCH_PKG_SPEC/, /^\)/' "$STACK_PY")
[ -n "$spec" ] || { echo "FATAL: _XPU_TORCH_PKG_SPEC not found in $STACK_PY" >&2; exit 1; }
py_has() { printf '%s' "$spec" | grep -qF "\"$1\"" && echo yes || echo no; }
check "install_python_stack torch"       "$(py_has "$TORCH")"  yes
check "install_python_stack torchvision" "$(py_has "$VISION")" yes
check "install_python_stack torchaudio"  "$(py_has "$AUDIO")"  yes

# install.ps1 builds the trio in $_xpuSpecs and drops torchaudio only on win-arm64.
ps_has() { grep -qF "\"$1\"" "$INSTALL_PS1" && echo yes || echo no; }
check "install.ps1 torch floor"   "$(ps_has "$TORCH")"  yes
check "install.ps1 torchvision"   "$(ps_has "$VISION")" yes
check "install.ps1 torchaudio"    "$(ps_has "$AUDIO")"  yes

# The pin must be acted on from the update route: an xpu leaf names no family the cuda/rocm
# helpers know, so without this the CPU wheel survives forever.
check "stack classifies the xpu leaf" \
    "$(grep -q '_TORCH_BACKEND = "xpu"' "$STACK_PY" && echo yes || echo no)" yes
check "stack repairs an xpu pin" \
    "$(grep -q 'def _ensure_xpu_torch' "$STACK_PY" && echo yes || echo no)" yes
# Wired in at BOTH repair points, or the final pass silently undoes the first.
check "repair runs at both call sites" \
    "$(grep -c '^        _ensure_xpu_torch()' "$STACK_PY")" 2
# The ROCm helper must skip an xpu backend, or it treats the pin as an AMD host.
check "rocm helper skips xpu" \
    "$(grep -q '_TORCH_BACKEND in ("cuda", "cpu", "xpu")' "$STACK_PY" && echo yes || echo no)" yes

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
