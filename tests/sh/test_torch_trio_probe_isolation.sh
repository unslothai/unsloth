#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test for #11980. The Step-2 --overrides file freezes the venv's torch trio so a
# released unsloth wheel cannot downgrade it, and the pins come from a probe run with the venv's
# interpreter. Unisolated, PYTHONPATH and the user site dir sit on sys.path AHEAD of the venv's
# own site-packages, so on a DGX/NGC host -- which exports PYTHONPATH at the system torch -- the
# probe reported that torch instead of the venv's. The frozen pin then named a version that was
# neither on an index nor installed in the venv, and uv attributes an override to the package
# that declared the requirement, so it read as though the wheel carried the pin:
#   Because there is no version of torch==2.9.0a0+50eac811a6.nv25.9 and unsloth>=2026.9.11
#   depends on torch==2.9.0a0+50eac811a6.nv25.9, we can conclude that ... cannot be used.
# A pin read from the venv is always satisfiable there, whatever its local label, so an NGC or
# source-built torch that really IS resident stays frozen as intended -- the point is that the
# probe must describe the environment uv resolves into.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"

# _harness.sh gives ok/bad/assert_eq/summary; the grep assertions want a rc-shaped one.
assert_true() { if [ "$2" = "0" ]; then ok "$1"; else bad "$1"; fi; }

echo "=== both pin-minting probes are isolated in install.sh ==="
grep -q '_torch_trio_pins=\$("\$_VENV_PY" -I -c "' "$INSTALL_SH"
assert_true "the trio probe that builds the overrides file runs python -I" "$?"
grep -q '_PREV_TORCH_VER=\$(_run_bounded "\$VENV_DIR/bin/python" -I -c' "$INSTALL_SH"
assert_true "the kept-release fallback probe runs python -I" "$?"

# The embedded snippet, extracted from install.sh and run for real against a venv whose torch is
# shadowed on PYTHONPATH. Needs an interpreter that can make a venv; nothing else.
_PY=$(command -v python3 || true)
if [ -z "$_PY" ]; then
    echo "  SKIP: no python3 to build a venv with"
    summary
fi

_snippet=$(sed -n '/_torch_trio_pins=\$("\$_VENV_PY" -I -c "/,/^" 2>\/dev\/null)/p' "$INSTALL_SH" \
    | sed '1s/.*-I -c "//' | sed '$d')
[ -n "$_snippet" ]
assert_true "the trio-probe snippet was extracted from install.sh" "$?"

_LAB=$(mktemp -d)
trap 'rm -rf "$_LAB"' EXIT
"$_PY" -m venv "$_LAB/venv" >/dev/null 2>&1 || {
    echo "  SKIP: venv creation unavailable on this host"
    summary
}
_VENV_PY="$_LAB/venv/bin/python"
[ -x "$_VENV_PY" ] || _VENV_PY="$_LAB/venv/Scripts/python.exe"

# The venv's own torch: a dist-info is all importlib.metadata needs.
_site=$("$_VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
mkdir -p "$_site/torch-2.11.0+cu130.dist-info"
printf 'Metadata-Version: 2.1\nName: torch\nVersion: 2.11.0+cu130\n' \
    > "$_site/torch-2.11.0+cu130.dist-info/METADATA"
# An NGC container's torch, reachable only through PYTHONPATH -- what DGX OS exports.
mkdir -p "$_LAB/ngc/torch-2.9.0a0+50eac811a6.nv25.9.dist-info"
printf 'Metadata-Version: 2.1\nName: torch\nVersion: 2.9.0a0+50eac811a6.nv25.9\n' \
    > "$_LAB/ngc/torch-2.9.0a0+50eac811a6.nv25.9.dist-info/METADATA"

echo "=== the probe reports the venv's torch, not PYTHONPATH's (#11980) ==="
_leaked=$(PYTHONPATH="$_LAB/ngc" "$_VENV_PY" -c "$_snippet" 2>/dev/null | head -n 1)
assert_eq "unisolated, PYTHONPATH's NGC torch wins (the bug)" \
    "torch==2.9.0a0+50eac811a6.nv25.9" "$_leaked"
_isolated=$(PYTHONPATH="$_LAB/ngc" "$_VENV_PY" -I -c "$_snippet" 2>/dev/null | head -n 1)
assert_eq "isolated, the venv's own torch is what gets frozen" \
    "torch==2.11.0+cu130" "$_isolated"

# -I must not cost us the venv itself: an isolated probe that cannot see site-packages would
# report nothing and silently drop the downgrade guard the overrides file exists to provide.
_clean=$("$_VENV_PY" -I -c "$_snippet" 2>/dev/null | head -n 1)
assert_eq "isolation keeps the venv's site-packages visible" "torch==2.11.0+cu130" "$_clean"

# PYTHONNOUSERSITE is not enough on its own, which is why -I (not -E or -s alone) is the flag.
_user=$(PYTHONPATH="$_LAB/ngc" PYTHONNOUSERSITE=1 "$_VENV_PY" -c "$_snippet" 2>/dev/null | head -n 1)
assert_eq "suppressing only the user site still leaks PYTHONPATH" \
    "torch==2.9.0a0+50eac811a6.nv25.9" "$_user"

# The -I fix makes the RESOLVE correct, but `import torch` is not isolated, so the shadowing copy
# is still what the backend would import. Turning a loud failure into a silent one would be worse
# than the bug, so the installer must say so -- once, with the version on each side.
echo "=== a shadowed torch is reported to the user (#11980) ==="
_FN=$(mktemp)
{
    sed -n '/^_warn_if_torch_shadowed()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_build_unsloth_torch_overrides()/,/^}/p' "$INSTALL_SH"
} > "$_FN"

_drive() {
    (
        . "$_FN"
        SKIP_TORCH=false; C_WARN=""; C_DIM=""; C_RST=""; _TORCH_SHADOW_WARNED=false
        substep() { echo "WARN: $1"; }
        TAURI_MODE=true
        tauri_log() { echo "LOG[$1]: $2"; }
        _VENV_PY="$1"
        unset UV_OVERRIDE
        _build_unsloth_torch_overrides
        printf 'PINS:%s\n' "$(tr '\n' ' ' < "$_UNSLOTH_TORCH_OVERRIDES" 2>/dev/null)"
        rm -f "$_UNSLOTH_TORCH_OVERRIDES" 2>/dev/null || true
    )
}

_clean_out=$(PYTHONPATH= _drive "$_VENV_PY")
assert_contains "no leak: the venv's torch is frozen" "$_clean_out" "PINS:torch==2.11.0+cu130"
assert_not_contains "no leak: nothing is warned about" "$_clean_out" "WARN:"

_leak_out=$(PYTHONPATH="$_LAB/ngc" _drive "$_VENV_PY")
assert_contains "leak: the venv's torch is still what gets frozen" "$_leak_out" "PINS:torch==2.11.0+cu130"
assert_contains "leak: the shadowing version is named" "$_leak_out" "2.9.0a0+50eac811a6.nv25.9"
assert_contains "leak: the venv's version is named too" "$_leak_out" "2.11.0+cu130"
assert_contains "leak: the user is told what to do" "$_leak_out" "unset PYTHONPATH"
_warn_count=$(printf '%s\n' "$_leak_out" | grep -c "shadows this environment" || true)
assert_eq "leak: the shadow is reported once, not per call" "1" "$_warn_count"
assert_contains "leak: the shadow reaches the diagnostics log" "$_leak_out" \
    "LOG[DIAG]: torch_shadow=1 ambient=2.9.0a0+50eac811a6.nv25.9 venv=2.11.0+cu130"
assert_not_contains "no leak: no DIAG line is emitted" "$_clean_out" "torch_shadow=1"
rm -f "$_FN"

summary
