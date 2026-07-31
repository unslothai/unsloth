#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Static analysis: the --cpu / --vulkan llama.cpp backend flags must be wired
# consistently across both installers (install.sh / install.ps1) so they reach
# setup.sh / setup.ps1 via UNSLOTH_LLAMA_CPP_BACKEND.
#
# Why the flag exists (#7213): `UNSLOTH_LLAMA_CPP_BACKEND=cpu curl ... | sh` binds
# the env var to curl, not the piped sh, so the installer never sees it and picks
# the auto-detected backend (Vulkan on an Intel/other iGPU, which crashed the
# reporter). The flag survives `| sh -s -- --cpu`, and the installer exports the
# env var setup.sh already reads. This pins that wiring so a refactor can't drop it.
#
# Shape/wiring test: greps the committed scripts. No Python, GPU, or network.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
ENV_VAR="UNSLOTH_LLAMA_CPP_BACKEND"
PASS=0
FAIL=0

assert_contains() {
    _label="$1"; _file="$2"; _needle="$3"
    if grep -qF -- "$_needle" "$_file"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle' in $(basename "$_file"))"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "=== install.sh: parses --cpu/--vulkan and exports $ENV_VAR ==="
assert_contains "install.sh: accepts --cpu flag"     "$INSTALL_SH" "--cpu) _LLAMA_CPP_BACKEND_FLAG=cpu"
assert_contains "install.sh: accepts --vulkan flag"  "$INSTALL_SH" "--vulkan) _LLAMA_CPP_BACKEND_FLAG=vulkan"
assert_contains "install.sh: exports $ENV_VAR"       "$INSTALL_SH" "export UNSLOTH_LLAMA_CPP_BACKEND=\"\$_LLAMA_CPP_BACKEND_FLAG\""
assert_contains "install.sh: forwards $ENV_VAR across the WSL reroute" \
    "$INSTALL_SH" "export UNSLOTH_LLAMA_CPP_BACKEND=\$(_rr_q \"\$UNSLOTH_LLAMA_CPP_BACKEND\")"
assert_contains "install.sh: documents the correct (post-pipe) env form" \
    "$INSTALL_SH" "| UNSLOTH_LLAMA_CPP_BACKEND=cpu sh"

echo ""
echo "=== install.ps1: parses --cpu/--vulkan and sets $ENV_VAR ==="
assert_contains "install.ps1: accepts --cpu flag"    "$INSTALL_PS1" '$LlamaCppBackendFlag = "cpu"'
assert_contains "install.ps1: accepts --vulkan flag" "$INSTALL_PS1" '$LlamaCppBackendFlag = "vulkan"'
assert_contains "install.ps1: sets $ENV_VAR for the setup run" "$INSTALL_PS1" '$env:UNSLOTH_LLAMA_CPP_BACKEND = $LlamaCppBackendFlag'
# The iex path runs in the caller's session, so the backend must be restored/cleared
# after setup rather than leaking (mirrors UNSLOTH_STUDIO_HOME / _LOCAL_LLAMA_CPP_DIR).
assert_contains "install.ps1: restores/clears $ENV_VAR after setup (no iex leak)" \
    "$INSTALL_PS1" "Remove-Item Env:UNSLOTH_LLAMA_CPP_BACKEND"

echo ""
echo "=== setup scripts already consume $ENV_VAR (the flag's target) ==="
assert_contains "setup.sh: reads $ENV_VAR"           "$SETUP_SH"  "UNSLOTH_LLAMA_CPP_BACKEND"
assert_contains "setup.sh: maps cpu backend to --force-cpu" "$SETUP_SH" "--force-cpu"
assert_contains "setup.ps1: reads $ENV_VAR"          "$SETUP_PS1" "UNSLOTH_LLAMA_CPP_BACKEND"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
