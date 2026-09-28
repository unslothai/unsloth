#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# install.sh gives new Linux x86_64 cu130 Python 3.13 installs torch 2.13 (the only route the
# prebuilt-wheels-cu13 release covers) while an existing 2.4-2.14 install keeps its release.
# Helpers are extracted from install.sh and sourced; the venv python is a stub.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"
_FUNC_FILE=$(mktemp)
{
    for fn in _run_bounded _pypi_unsloth_admits_torch _torch_index_url_leaf _cu130_torch213_route _cu130_torch213_platform _torchaudio_for_torch_minor _torch_release_in_window _previous_torch_pin; do
        sed -n "/^${fn}()/,/^}/p" "$INSTALL_SH"
        echo ""
    done
} > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"
unset UNSLOTH_TORCH_UPGRADE

VENV_DIR=$(mktemp -d)
mkdir -p "$VENV_DIR/bin"
# Answers the version probe (-c) with $1 and runs anything else (the PyPI gate) on a real python.
_stub_python() {
    printf '#!/bin/sh\nif [ "$1" = "-c" ]; then echo %s; else exec python3 "$@"; fi\n' "$1" > "$VENV_DIR/bin/python"
    chmod +x "$VENV_DIR/bin/python"
}
CU130="https://download.pytorch.org/whl/cu130"
_pypi_fixture() {
    printf '{"info": {"requires_dist": ["numpy", "%s", "torchvision"]}}' "$1" > "$VENV_DIR/pypi.json"
    export UNSLOTH_PYPI_JSON_URL="file://$VENV_DIR/pypi.json"
}
_pypi_fixture "torch<2.15.0,>=2.4.0"

echo "=== route: only cu130 + Linux/WSL + x86_64 + Python 3.13 ==="
_stub_python 3.13
OS=linux; _ARCH=x86_64
assert_eq "linux x86_64 cu130 py3.13"      "yes" "$(_cu130_torch213_route "$CU130")"
assert_eq "trailing slash and query"       "yes" "$(_cu130_torch213_route "$CU130/?token=x")"
OS=wsl
assert_eq "wsl counts as linux"            "yes" "$(_cu130_torch213_route "$CU130")"
OS=linux
for leaf in cu128 cu126 cu124 cu118 cpu rocm7.2 xpu gfx1151; do
    assert_eq "leaf $leaf stays on the old window" "no" "$(_cu130_torch213_route "https://download.pytorch.org/whl/$leaf")"
done
OS=macos
assert_eq "macOS never"                    "no" "$(_cu130_torch213_route "$CU130")"
OS=linux; _ARCH=aarch64
assert_eq "aarch64 never (no wheels)"      "no" "$(_cu130_torch213_route "$CU130")"
_ARCH=x86_64
for py in 3.11 3.12 3.14; do
    _stub_python "$py"
    assert_eq "Python $py never (cp313 wheels only)" "no" "$(_cu130_torch213_route "$CU130")"
done
rm -f "$VENV_DIR/bin/python"
assert_eq "no venv python never"           "no" "$(_cu130_torch213_route "$CU130")"

echo "=== route: only once the PyPI release admits torch 2.13 ==="
_stub_python 3.13
assert_eq "release admits 2.13"            "yes" "$(_cu130_torch213_route "$CU130")"
_pypi_fixture "torch<2.13.0,>=2.4.0"
assert_eq "release still capped below 2.13" "no" "$(_cu130_torch213_route "$CU130")"
_pypi_fixture "torch>=2.4.0"
assert_eq "uncapped release"               "yes" "$(_cu130_torch213_route "$CU130")"
_pypi_fixture "torch<2.15.0,>=2.4.0 ; extra == \\\"x\\\""
assert_eq "extra-only torch is no evidence" "no" "$(_cu130_torch213_route "$CU130")"
export UNSLOTH_PYPI_JSON_URL="http://127.0.0.1:9/unreachable"
assert_eq "unreachable index keeps the old window" "no" "$(_cu130_torch213_route "$CU130")"
_pypi_fixture "torch<2.15.0,>=2.4.0"
assert_eq "UV_EXCLUDE_NEWER keeps the old window" "no" "$(UV_EXCLUDE_NEWER=2026-06-01T00:00:00Z _cu130_torch213_route "$CU130")"
assert_eq "UV_EXCLUDE_NEWER_PACKAGE keeps the old window" "no" "$(UV_EXCLUDE_NEWER_PACKAGE=unsloth=2026-06-01T00:00:00Z _cu130_torch213_route "$CU130")"
assert_eq "empty cutoff is no cutoff" "yes" "$(UV_EXCLUDE_NEWER= _cu130_torch213_route "$CU130")"

echo "=== preservation never waits on PyPI ==="
# Runs install.sh's own block: an existing 2.13 install re-run while PyPI is unreachable or
# still capped keeps the wide window (only the new-install default waits on the gate).
_PRESERVE_BLOCK=$(sed -n '/^_PRESERVE_TORCH_CONSTRAINT="\$TORCH_CONSTRAINT"$/,/^fi$/p' "$INSTALL_SH")
assert_eq "block found in install.sh" "yes" "$([ -n "$_PRESERVE_BLOCK" ] && echo yes)"
_CU130_TORCH_CEILING="2.15.0"; _CU130_NEW_INSTALL_TORCH="torch>=2.13.0,<2.14.0"; SKIP_TORCH=false; TORCH_INDEX_URL="$CU130"
for gate in unreachable capped open; do
    case "$gate" in
        unreachable) export UNSLOTH_PYPI_JSON_URL="http://127.0.0.1:9/unreachable" ;;
        capped) _pypi_fixture "torch<2.13.0,>=2.4.0" ;;
        open) _pypi_fixture "torch<2.15.0,>=2.4.0" ;;
    esac
    TORCH_CONSTRAINT="torch>=2.4,<2.12.0"; TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.27.0"
    eval "$_PRESERVE_BLOCK"
    assert_eq "PyPI $gate: 2.13 install kept" "torch==2.13.0" "$(_previous_torch_pin '2.13.0+cu130' "$_PRESERVE_TORCH_CONSTRAINT")"
    want="torch>=2.4,<2.12.0"; [ "$gate" = open ] && want="torch>=2.13.0,<2.14.0"
    assert_eq "PyPI $gate: new-install default" "$want" "$TORCH_CONSTRAINT"
done
_stub_python 3.12
TORCH_CONSTRAINT="torch>=2.4,<2.12.0"; eval "$_PRESERVE_BLOCK"
assert_eq "Python 3.12 keeps the old window" "torch>=2.4,<2.12.0" "$_PRESERVE_TORCH_CONSTRAINT"
_stub_python 3.13

echo "=== preservation window keeps every existing 2.4-2.14 release ==="
PRESERVE='torch>=2.4,<2.15.0'
for v in 2.4.0+cu121 2.10.0+cu128 2.11.0+cu130 2.12.1+cu130 2.13.0+cu130 2.14.0+cu130 2.11.0; do
    assert_eq "keeps $v" "torch==${v%%+*}" "$(_previous_torch_pin "$v" "$PRESERVE")"
done
assert_eq "2.15 is outside the window"     "" "$(_previous_torch_pin '2.15.0+cu130' "$PRESERVE")"
assert_eq "nightly never pins"             "" "$(_previous_torch_pin '2.14.0.dev20260801+cu130' "$PRESERVE")"
UNSLOTH_TORCH_UPGRADE=1
assert_eq "UNSLOTH_TORCH_UPGRADE=1 opts out" "" "$(_previous_torch_pin '2.11.0+cu130' "$PRESERVE")"
unset UNSLOTH_TORCH_UPGRADE
# The new-install spec must never be the preservation window: that would upgrade 2.11 users.
assert_eq "new-install spec alone would drop a kept 2.11" "" "$(_previous_torch_pin '2.11.0+cu130' 'torch>=2.13.0,<2.14.0')"
assert_eq "kept pin is evaluated against the preservation window" "yes" \
    "$(grep -q '_prev_pin=$(_previous_torch_pin "$_PREV_TORCH_VER" "$_PRESERVE_TORCH_CONSTRAINT")' "$INSTALL_SH" && echo yes)"
assert_eq "new-install spec is torch 2.13" "yes" \
    "$(grep -q '^_CU130_NEW_INSTALL_TORCH="torch>=2.13.0,<2.14.0"$' "$INSTALL_SH" && echo yes)"

echo "=== torchaudio pairing for a kept minor ==="
assert_eq "2.10 -> 2.10"  "torchaudio==2.10.*" "$(_torchaudio_for_torch_minor 10)"
assert_eq "2.11 -> 2.11"  "torchaudio==2.11.*" "$(_torchaudio_for_torch_minor 11)"
assert_eq "2.12 -> 2.11"  "torchaudio==2.11.*" "$(_torchaudio_for_torch_minor 12)"
assert_eq "2.13 -> 2.11"  "torchaudio==2.11.*" "$(_torchaudio_for_torch_minor 13)"
assert_eq "2.14 -> 2.11"  "torchaudio==2.11.*" "$(_torchaudio_for_torch_minor 14)"
assert_eq "2.9 -> 2.9"    "torchaudio==2.9.*"  "$(_torchaudio_for_torch_minor 9)"

rm -rf "${VENV_DIR:?}"
summary
