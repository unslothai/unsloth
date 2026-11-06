#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# #11980: the installer's venv torch probes must ignore a torch on PYTHONPATH (NGC 2.9.0a0+...nv25.9).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

echo "=== test_torch_probe_isolation ==="

WORK=$(mktemp -d)
trap 'rm -rf "${WORK:?}"' EXIT
python3 -m venv --without-pip "$WORK/venv"
_VENV_PY="$WORK/venv/bin/python"
SP=$("$_VENV_PY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
fake_dist() {
    mkdir -p "$1/$2-$3.dist-info"
    printf 'Metadata-Version: 2.1\nName: %s\nVersion: %s\n' "$2" "$3" > "$1/$2-$3.dist-info/METADATA"
}
fake_dist "$SP" torch 2.11.0+cu130
fake_dist "$SP" torchvision 0.26.0+cu130
fake_dist "$SP" torchaudio 2.11.0+cu130
mkdir -p "$WORK/shadow"
fake_dist "$WORK/shadow" torch 2.9.0a0+50eac811a6.nv25.9

FUNCS=$(sed -n '/^_run_bounded() {/,/^}/p; /^_TORCH_SHADOW_WARNED=false$/p; /^_warn_torch_shadowed() {/,/^}/p; /^_build_unsloth_torch_overrides() {/,/^}/p' "$INSTALL_SH")
case "$FUNCS" in
    *_warn_torch_shadowed*_build_unsloth_torch_overrides*) ok "extracted the probe functions from install.sh" ;;
    *) bad "could not extract the probe functions from install.sh"; echo "Results: $PASS passed, $FAIL failed"; exit 1 ;;
esac

# Real builder, cwd = shadow, PYTHONPATH=$1; called twice since the warning is once per run.
run_builder() {
    (
        cd "$WORK/shadow"
        export PYTHONPATH="$1"
        SKIP_TORCH=false
        C_WARN=""
        unset UV_OVERRIDE
        substep() { echo "SUBSTEP: $1"; }
        eval "$FUNCS"
        _build_unsloth_torch_overrides
        cat "$_UNSLOTH_TORCH_OVERRIDES"; rm -f "$_UNSLOTH_TORCH_OVERRIDES"
        _build_unsloth_torch_overrides
        rm -f "$_UNSLOTH_TORCH_OVERRIDES"
    )
}

OUT=$(run_builder "$WORK/shadow")
assert_contains "shadowed: venv torch is frozen" "$OUT" "torch==2.11.0+cu130"
assert_not_contains "shadowed: PYTHONPATH torch is not frozen" "$OUT" "torch==2.9.0a0"
assert_contains "shadowed: torchvision still frozen" "$OUT" "torchvision==0.26.0+cu130"
assert_contains "shadowed: user is told which torch shadows the venv" "$OUT" "PYTHONPATH exposes torch 2.9.0a0+50eac811a6.nv25.9"
assert_eq "shadowed: the warning prints once per run" "2" "$(printf '%s\n' "$OUT" | grep -c '^SUBSTEP:')"

OUT=$(run_builder "")
assert_contains "cwd shadow only: venv torch is frozen" "$OUT" "torch==2.11.0+cu130"
assert_eq "cwd shadow only: no PYTHONPATH warning" "" "$(printf '%s\n' "$OUT" | sed '/^SUBSTEP:/!d')"

_unisolated=$(grep -nE '(_VENV_PY"|VENV_DIR/bin/python") -c' "$INSTALL_SH" \
    | grep -E "import torch|version\(_p\)" || true)
assert_eq "no venv torch probe in install.sh runs without -I" "" "$_unisolated"
for _probe in '_torch_trio_pins=$("$_VENV_PY" -I -c' \
              '"$_VENV_PY" -I -c "import torch; print(torch.__version__)"' \
              '_PREV_TORCH_VER=$(_run_bounded "$VENV_DIR/bin/python" -I -c'; do
    assert_contains "install.sh still carries: $_probe" "$(cat "$INSTALL_SH")" "$_probe"
done

SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
assert_contains "setup.sh drops PYTHONPATH once the venv is active (Colab no-venv keeps it)" \
    "$(cat "$SETUP_SH")" '[ "$_COLAB_NO_VENV" = true ] || unset PYTHONPATH'
_unset_line=$(grep -n '|| unset PYTHONPATH$' "$SETUP_SH" | head -n 1 | cut -d: -f1)
_stack_line=$(grep -n '^    python "$SCRIPT_DIR/install_python_stack.py"$' "$SETUP_SH" | head -n 1 | cut -d: -f1)
if [ -n "$_unset_line" ] && [ -n "$_stack_line" ] && [ "$_unset_line" -lt "$_stack_line" ]; then
    ok "setup.sh drops PYTHONPATH before install_python_stack.py runs"
else
    bad "setup.sh drops PYTHONPATH before install_python_stack.py runs (unset=$_unset_line stack=$_stack_line)"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
