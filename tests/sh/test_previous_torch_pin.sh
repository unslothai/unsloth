#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _previous_torch_pin keeps the previous torch RELEASE on a re-run regardless of flavor tag:
# the pin installs from the freshly chosen index. Per-leaf windows still win.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"
_FUNC_FILE=$(mktemp)
{
    sed -n '/^_torch_release_in_window()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_previous_torch_pin()/,/^}/p' "$INSTALL_SH"
} > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

unset UNSLOTH_TORCH_UPGRADE

echo "=== _previous_torch_pin: in-window releases are kept, any flavor ==="
assert_eq "cu126 wheel"                  "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cu126' 'torch>=2.4,<2.12.0')"
assert_eq "cu130 wheel"                  "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cu130' 'torch>=2.4,<2.12.0')"
assert_eq "cpu wheel"                    "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cpu' 'torch>=2.4,<2.12.0')"
assert_eq "PyPI bare version (CUDA build on Linux)" "torch==2.10.0" "$(_previous_torch_pin '2.10.0' 'torch>=2.4,<2.12.0')"
assert_eq "rocm wheel"                   "torch==2.10.0" "$(_previous_torch_pin '2.10.0+rocm6.4' 'torch>=2.4,<2.11.0')"
assert_eq "rocm three-component tag"     "torch==2.9.1"  "$(_previous_torch_pin '2.9.1+rocm7.2.1' 'torch>=2.4,<2.12.0')"
assert_eq "Intel xpu wheel"              "torch==2.9.0"  "$(_previous_torch_pin '2.9.0+xpu' 'torch>=2.4,<2.12.0')"
assert_eq "local suffix stripped"        "torch==2.9.1"  "$(_previous_torch_pin '2.9.1+cu128' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: raised floors reject older releases ==="
# rocm7.2 / Strix leaves raise TORCH_CONSTRAINT before the pin is evaluated.
assert_eq "old 2.10 vs rocm7.2 floor"    "" "$(_previous_torch_pin '2.10.0+rocm7.1' 'torch>=2.11.0,<2.12.0')"
assert_eq "2.11 passes the rocm7.2 floor" "torch==2.11.0" "$(_previous_torch_pin '2.11.0+rocm7.2' 'torch>=2.11.0,<2.12.0')"

echo "=== _previous_torch_pin: probe noise never becomes a pin ==="
assert_eq "empty version"                "" "$(_previous_torch_pin '' 'torch>=2.4,<2.12.0')"
assert_eq "garbage version"              "" "$(_previous_torch_pin 'not-a-version' 'torch>=2.4,<2.12.0')"
assert_eq "traceback fragment"           "" "$(_previous_torch_pin "ModuleNotFoundError: No module named 'torch'" 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: nightly / dev / source builds never pin ==="
# No stable index carries nightlies, so pinning one would burn a doomed resolve.
assert_eq "nightly dev build"            "" "$(_previous_torch_pin '2.11.0.dev20250704+cu128' 'torch>=2.4,<2.12.0')"
assert_eq "source build a0 tag"          "" "$(_previous_torch_pin '2.9.0a0+gitabc1234' 'torch>=2.4,<2.12.0')"
assert_eq "release candidate"            "" "$(_previous_torch_pin '2.11.0rc1+cu130' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: out-of-window releases never pin ==="
assert_eq "2.3.x below the cu floor"     "" "$(_previous_torch_pin '2.3.1+cu118' 'torch>=2.4,<2.12.0')"
assert_eq "2.12.x above the cu ceiling"  "" "$(_previous_torch_pin '2.12.0+cu130' 'torch>=2.4,<2.12.0')"
assert_eq "floor boundary 2.4.0 kept"    "torch==2.4.0"  "$(_previous_torch_pin '2.4.0+cu126' 'torch>=2.4,<2.12.0')"
assert_eq "ceiling-adjacent 2.11.x kept" "torch==2.11.1" "$(_previous_torch_pin '2.11.1+cu130' 'torch>=2.4,<2.12.0')"
assert_eq "cpu window excludes 2.11.x"   "" "$(_previous_torch_pin '2.11.0+cpu' 'torch>=2.4,<2.11.0')"
assert_eq "mac floor excludes 2.5.x"     "" "$(_previous_torch_pin '2.5.1' 'torch>=2.6,<2.11.0')"
assert_eq "malformed window never pins"  "" "$(_previous_torch_pin '2.10.0+cu126' 'torch')"
assert_eq "empty window never pins"      "" "$(_previous_torch_pin '2.10.0+cu126' '')"

echo "=== _torch_release_in_window ==="
assert_eq "in window"            "yes" "$(_torch_release_in_window '2.10.0' 'torch>=2.4,<2.12.0')"
assert_eq "at floor"             "yes" "$(_torch_release_in_window '2.4.0' 'torch>=2.4,<2.12.0')"
assert_eq "below floor"          "no"  "$(_torch_release_in_window '2.3.1' 'torch>=2.4,<2.12.0')"
assert_eq "at ceiling"           "no"  "$(_torch_release_in_window '2.12.0' 'torch>=2.4,<2.12.0')"
assert_eq "next major"           "no"  "$(_torch_release_in_window '3.0.0' 'torch>=2.4,<2.12.0')"
assert_eq "patch-level floor"    "yes" "$(_torch_release_in_window '2.11.5' 'torch>=2.11.0,<2.12.0')"
assert_eq "no ceiling -> no"     "no"  "$(_torch_release_in_window '2.10.0' 'torch>=2.4')"
assert_eq "garbage minor -> no"  "no"  "$(_torch_release_in_window '2.x' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: UNSLOTH_TORCH_UPGRADE=1 opts out ==="
assert_eq "upgrade env set"    "" "$(UNSLOTH_TORCH_UPGRADE=1 _previous_torch_pin '2.10.0+cu126' 'torch>=2.4,<2.12.0')"
assert_eq "upgrade env 0"      "torch==2.10.0" "$(UNSLOTH_TORCH_UPGRADE=0 _previous_torch_pin '2.10.0+cu126' 'torch>=2.4,<2.12.0')"

echo "=== the preservation probe reads off disk, not through the interpreter ==="
# `import torch` can block forever on a wedged Intel driver. Executed, not grepped: the
# stub interpreter records being called.
_PREVBLK=$(mktemp)
{
    sed -n '/^_run_bounded()/,/^}/p' "$INSTALL_SH"
    awk '/^    _PREV_TORCH_VER=""$/{on=1} on{print} on && /tail -n 1 \|\| true\)$/{exit}' \
        "$INSTALL_SH"
} > "$_PREVBLK"
grep -q 'version.py' "$_PREVBLK" || { echo "FATAL: probe block not extracted"; exit 1; }
grep -q 'import torch' "$_PREVBLK" || { echo "FATAL: extraction lost the fallback"; exit 1; }
grep -q '^_run_bounded()' "$_PREVBLK" || { echo "FATAL: could not extract _run_bounded"; exit 1; }

_prev_probe() {  # $1 = torch label to put on disk ("" for no version.py)
    _d=$(mktemp -d)
    mkdir -p "$_d/bin"
    printf '#!/bin/sh\ntouch "%s/CALLED"\necho 9.9.9+fromimport\n' "$_d" > "$_d/bin/python"
    chmod +x "$_d/bin/python"
    if [ -n "$1" ]; then
        mkdir -p "$_d/lib/python3.12/site-packages/torch"
        printf "__version__ = '%s'\n" "$1" > "$_d/lib/python3.12/site-packages/torch/version.py"
    fi
    ( VENV_DIR="$_d"; . "$_PREVBLK"; printf '%s|%s' "$_PREV_TORCH_VER" "$([ -f "$_d/CALLED" ] && echo called || echo not-called)" )
    rm -rf "$_d"
}
assert_eq "xpu wheel read off disk"      "2.9.1+xpu|not-called"      "$(_prev_probe '2.9.1+xpu')"
assert_eq "cuda wheel read off disk"     "2.9.1+cu128|not-called"    "$(_prev_probe '2.9.1+cu128')"
assert_eq "untagged wheel read off disk" "2.9.1|not-called"          "$(_prev_probe '2.9.1')"
assert_eq "falls back with no version.py" "9.9.9+fromimport|called"  "$(_prev_probe '')"
rm -f "$_PREVBLK"

echo "=== install.sh wiring ==="
# The probe must run against the OLD venv, before it is moved aside for rollback.
_probe_line=$(grep -n '_PREV_TORCH_VER=\$(' "$INSTALL_SH" | head -1 | cut -d: -f1)
_move_line=$(grep -n '_start_studio_venv_replacement "\$VENV_DIR"' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "probe exists"                  "yes" "$([ -n "$_probe_line" ] && echo yes)"
assert_eq "probe before venv replacement" "yes" "$([ -n "$_probe_line" ] && [ -n "$_move_line" ] && [ "$_probe_line" -lt "$_move_line" ] && echo yes)"
# The pin must be evaluated after the last index/constraint decision (Strix raises the floor).
_pin_line=$(grep -n '_prev_pin=\$(_previous_torch_pin' "$INSTALL_SH" | head -1 | cut -d: -f1)
_strix_line=$(grep -n 'Strix Halo / Strix Point:' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "pin evaluated after the Strix reroute" "yes" "$([ -n "$_pin_line" ] && [ -n "$_strix_line" ] && [ "$_pin_line" -gt "$_strix_line" ] && echo yes)"
assert_eq "resolve-failure fallback wired" "yes" "$(grep -q 'TORCH_CONSTRAINT="\$_PREV_FALLBACK_CONSTRAINT"' "$INSTALL_SH" && echo yes)"
assert_eq "pin gated on SKIP_TORCH"        "yes" "$(grep -q 'if \[ "\$SKIP_TORCH" = false \]; then' "$INSTALL_SH" && echo yes)"
# Every --default-index torch install must go through the kept-release helper, so a pinned
# release missing from the index never aborts a rerun.
_helper_uses=$(grep -c '_install_torch_default_index' "$INSTALL_SH")
assert_eq "kept-release helper used by all default-index paths" "yes" "$([ "$_helper_uses" -ge 8 ] && echo yes)"
_repair_uses=$(grep -c '_install_torch_default_index --force-reinstall' "$INSTALL_SH")
assert_eq "ROCm repairs routed through the kept-release helper" "yes" "$([ "$_repair_uses" -ge 2 ] && echo yes)"
# The flavor repair runs under set -e, so a direct uv call with a bad pin would abort.
assert_eq "flavor repair routed through the kept-release helper" "yes" "$(grep -q '_install_torch_default_index \\' "$INSTALL_SH" && grep -q -- '--reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio' "$INSTALL_SH" && echo yes)"
# torchaudio no longer exact-pins torch, so unconstrained it resolves a mismatched build.
assert_eq "kept-release install pairs torchvision/torchaudio to the kept minor" "yes" "$(grep -q '_itdi_ta=\$(_torchaudio_for_torch_minor "\$_itdi_minor")' "$INSTALL_SH" && grep -q 'torchvision==0.\$((_itdi_minor + 15)).\*' "$INSTALL_SH" && echo yes)"
# Radeon direct wheels: the exact-first kept-trio attempt runs before the newest-trio search,
# which only runs when that attempt found nothing.
_radeon_kept_line=$(grep -n '_kept_torch=\$(_pick_radeon_wheel "torch" *"\${_prev_kept_base}"' "$INSTALL_SH" | head -1 | cut -d: -f1)
_radeon_loop_line=$(grep -n 'Loop downwards to find the first complete matching trio' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "Radeon kept-trio attempt before the newest-trio search" "yes" "$([ -n "$_radeon_kept_line" ] && [ -n "$_radeon_loop_line" ] && [ "$_radeon_kept_line" -lt "$_radeon_loop_line" ] && echo yes)"
assert_eq "Radeon newest-trio search gated on no kept match" "yes" "$(grep -q 'if \[ "\$_radeon_versions_match" != true \] &&' "$INSTALL_SH" && echo yes)"
assert_eq "Radeon kept-trio gap falls back with a warning" "yes" "$(grep -q 'lacks a complete wheel set for kept' "$INSTALL_SH" && echo yes)"

echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "$FAIL check(s) FAILED"
    exit 1
fi
echo "All $PASS checks passed"
