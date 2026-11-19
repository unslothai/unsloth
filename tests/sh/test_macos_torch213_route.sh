#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# New Apple Silicon installs on macOS 14+ get torch 2.13 (fused MPS attention); existing 2.4-2.14 installs keep theirs.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"
_FUNC_FILE=$(mktemp)
{
    for fn in _mirror_uv_project_config _mirror_configured _run_bounded _pypi_unsloth_admits_torch _torch_index_url_leaf _cu130_torch213_platform _cu130_torch213_host_ok _macos_torch213_platform _macos_torch213_host_ok _torch_release_in_window _previous_torch_pin _dir_has_entries; do
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
# Answers the GIL probe with $2 (default gil-enabled), the version probe (-c) with $1, and runs
# anything else (the PyPI gate) on a real python.
_stub_python() {
    printf '#!/bin/sh\ncase "$2" in *Py_GIL_DISABLED*) echo %s; exit 0 ;; esac\nif [ "$1" = "-c" ]; then echo %s; else exec python3 "$@"; fi\n' "${2:-gil-enabled}" "$1" > "$VENV_DIR/bin/python"
    chmod +x "$VENV_DIR/bin/python"
}
MACOS_MAJOR=15
_macos_product_major() { echo "$MACOS_MAJOR"; }
_uv_glibc_minor() { return 1; }
CPU="https://download.pytorch.org/whl/cpu"
_pypi_fixture() {
    printf '{"info": {"requires_dist": ["numpy", "%s", "torchvision"]}}' "$1" > "$VENV_DIR/pypi.json"
    export UNSLOTH_PYPI_JSON_URL="file://$VENV_DIR/pypi.json"
}
_pypi_fixture "torch<2.15.0,>=2.4.0"

echo "=== platform: only the /cpu leaf on arm64 macOS with Python 3.10 - 3.14 ==="
_stub_python 3.13
OS=macos; _ARCH=arm64
assert_eq "macOS arm64 cpu py3.13"         "yes" "$(_macos_torch213_platform "$CPU")"
assert_eq "trailing slash and query"       "yes" "$(_macos_torch213_platform "$CPU/?token=x")"
for leaf in cu130 cu128 rocm7.2 xpu; do
    assert_eq "leaf $leaf never"           "no" "$(_macos_torch213_platform "https://download.pytorch.org/whl/$leaf")"
done
for os in linux wsl windows; do
    OS=$os
    assert_eq "$os never"                  "no" "$(_macos_torch213_platform "$CPU")"
done
OS=macos; _ARCH=x86_64
assert_eq "Intel / Rosetta never"          "no" "$(_macos_torch213_platform "$CPU")"
_ARCH=arm64
for py in 3.10 3.11 3.12 3.14; do
    _stub_python "$py"
    assert_eq "Python $py"                 "yes" "$(_macos_torch213_platform "$CPU")"
done
for py in 3.9 3.15 ""; do
    _stub_python "$py"
    assert_eq "Python '$py' never (no torch 2.13 wheel)" "no" "$(_macos_torch213_platform "$CPU")"
done
rm -f "$VENV_DIR/bin/python"
assert_eq "no venv python never"           "no" "$(_macos_torch213_platform "$CPU")"
_stub_python 3.13
OS=macos; _ARCH=arm64
assert_eq "cu130 platform still never on macOS" "no" "$(_cu130_torch213_platform "$CPU")"

echo "=== host: macOS 14+ (macosx_14_0 wheels), GIL-enabled Python ==="
for m in 14 15 26 27; do
    MACOS_MAJOR=$m
    assert_eq "macOS $m"                   "yes" "$(_macos_torch213_host_ok)"
done
for m in 11 12 13 ""; do
    MACOS_MAJOR=$m
    assert_eq "macOS '$m' never"           "no" "$(_macos_torch213_host_ok)"
done
MACOS_MAJOR=15
_stub_python 3.13 gil-disabled
assert_eq "free-threaded 3.13 never (no cp313t torch 2.13)" "no" "$(_macos_torch213_host_ok)"
assert_eq "free-threaded 3.13 keeps the preservation window" "yes" "$(_macos_torch213_platform "$CPU")"
_stub_python 3.13

echo "=== preservation never waits on PyPI; only a fresh home takes 2.13 ==="
_PRESERVE_BLOCK=$(sed -n '/^_PRESERVE_TORCH_CONSTRAINT="\$TORCH_CONSTRAINT"$/,/^if \[ -z "\$_PREV_TORCH_PIN" \] && \[ "\$_MACOS_NEW_INSTALL_ROUTE" = true \]; then$/p' "$INSTALL_SH"
    printf '    TORCH_CONSTRAINT="$_MACOS_NEW_INSTALL_TORCH"\n    TORCHVISION_CONSTRAINT="torchvision>=0.28.0,<0.29.0"\nfi\n')
assert_eq "block found in install.sh" "yes" "$(printf '%s' "$_PRESERVE_BLOCK" | grep -q _MACOS_NEW_INSTALL_ROUTE=true && echo yes)"
assert_eq "install.sh new-install body matches the replayed one" "yes" \
    "$(grep -A2 '^if \[ -z "\$_PREV_TORCH_PIN" \] && \[ "\$_MACOS_NEW_INSTALL_ROUTE" = true \]; then$' "$INSTALL_SH" | tail -2 \
        | tr -d '\n' | grep -qF 'TORCH_CONSTRAINT="$_MACOS_NEW_INSTALL_TORCH"    TORCHVISION_CONSTRAINT="torchvision>=0.28.0,<0.29.0"' && echo yes)"
substep() { :; }
eval "$(grep -E '^_(CU130|MACOS)_(TORCH_CEILING|NEW_INSTALL_TORCH)=' "$INSTALL_SH")"
assert_eq "macOS ceiling" "2.15.0" "$_MACOS_TORCH_CEILING"
assert_eq "macOS new-install spec" "torch>=2.13.0,<2.14.0" "$_MACOS_NEW_INSTALL_TORCH"
SKIP_TORCH=false; TORCH_INDEX_URL="$CPU"; PACKAGE_NAME="unsloth"
_run_block() {  # $1 existing-install flag, $2 previous torch version, $3 default floor
    _EXISTING_INSTALL="$1"; _PREV_TORCH_VER="$2"
    TORCH_CONSTRAINT="torch>=${3:-2.6},<2.12.0"; TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.27.0"
    eval "$_PRESERVE_BLOCK"
}
for gate in unreachable capped open; do
    case "$gate" in
        unreachable) export UNSLOTH_PYPI_JSON_URL="http://127.0.0.1:9/unreachable" ;;
        capped) _pypi_fixture "torch<2.13.0,>=2.4.0" ;;
        open) _pypi_fixture "torch<2.15.0,>=2.4.0" ;;
    esac
    _run_block true "2.13.0"
    assert_eq "PyPI $gate: existing 2.13 kept" "torch==2.13.0" "$TORCH_CONSTRAINT"
    _run_block true "2.14.1"
    assert_eq "PyPI $gate: existing 2.14 kept" "torch==2.14.1" "$TORCH_CONSTRAINT"
    _run_block true "2.11.0"
    assert_eq "PyPI $gate: existing 2.11 kept" "torch==2.11.0" "$TORCH_CONSTRAINT"
    assert_eq "PyPI $gate: kept 2.11 keeps its torchvision window" "torchvision>=0.19,<0.27.0" "$TORCHVISION_CONSTRAINT"
    assert_eq "PyPI $gate: kept pin falls back to the old default" "torch>=2.6,<2.12.0" "$_PREV_FALLBACK_CONSTRAINT"
    _run_block true ""
    assert_eq "PyPI $gate: existing home with unreadable torch stays on the old default" "torch>=2.6,<2.12.0" "$TORCH_CONSTRAINT"
    _run_block false ""
    want="torch>=2.6,<2.12.0"; tv="torchvision>=0.19,<0.27.0"
    [ "$gate" = open ] && want="torch>=2.13.0,<2.14.0" && tv="torchvision>=0.28.0,<0.29.0"
    assert_eq "PyPI $gate: fresh home default" "$want" "$TORCH_CONSTRAINT"
    assert_eq "PyPI $gate: fresh home torchvision" "$tv" "$TORCHVISION_CONSTRAINT"
done
assert_eq "preservation keeps the Python-dependent floor" "torch>=2.6,<2.15.0" "$_PRESERVE_TORCH_CONSTRAINT"
_run_block true "2.11.0" 2.4
assert_eq "preservation keeps the 2.4 floor below Python 3.13" "torch>=2.4,<2.15.0" "$_PRESERVE_TORCH_CONSTRAINT"
_run_block true "2.15.0"
assert_eq "2.15 is outside the window: old default" "torch>=2.6,<2.12.0" "$TORCH_CONSTRAINT"
UNSLOTH_TORCH_UPGRADE=1; _run_block true "2.11.0"; unset UNSLOTH_TORCH_UPGRADE
assert_eq "UNSLOTH_TORCH_UPGRADE=1 moves an existing home to 2.13" "torch>=2.13.0,<2.14.0" "$TORCH_CONSTRAINT"
UNSLOTH_TORCH_UPGRADE=1; _run_block true "2.13.0"; unset UNSLOTH_TORCH_UPGRADE
assert_eq "UNSLOTH_TORCH_UPGRADE=1 on a resident 2.13 takes the 2.13 window" "torch>=2.13.0,<2.14.0" "$TORCH_CONSTRAINT"
PACKAGE_NAME="unsloth-fork"; _run_block false ""; PACKAGE_NAME="unsloth"
assert_eq "--package other than unsloth: fresh home keeps the old default" "torch>=2.6,<2.12.0" "$TORCH_CONSTRAINT"
MACOS_MAJOR=13; _run_block false ""; MACOS_MAJOR=15
assert_eq "macOS 13 fresh home keeps the old default" "torch>=2.6,<2.12.0" "$TORCH_CONSTRAINT"
assert_eq "macOS 13 still preserves a resident release" "torch>=2.6,<2.15.0" "$_PRESERVE_TORCH_CONSTRAINT"
SKIP_TORCH=true; _run_block false ""; SKIP_TORCH=false
assert_eq "--no-torch never takes the route" "torch>=2.6,<2.12.0" "$TORCH_CONSTRAINT"
_stub_python 3.9; _run_block false "" 2.4; _stub_python 3.13
assert_eq "Python 3.9 keeps the old window" "torch>=2.4,<2.12.0" "$_PRESERVE_TORCH_CONSTRAINT"
OS=linux; TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"; _run_block false "" 2.4
assert_eq "Linux cu128 is untouched" "torch>=2.4,<2.12.0" "$TORCH_CONSTRAINT"
assert_eq "Linux cu128 window untouched" "torch>=2.4,<2.12.0" "$_PRESERVE_TORCH_CONSTRAINT"
TORCH_INDEX_URL="$CPU"; _run_block false "" 2.4
assert_eq "Linux CPU is untouched" "torch>=2.4,<2.12.0" "$TORCH_CONSTRAINT"
OS=macos
assert_eq "the generic ceiling is unchanged" "yes" "$(grep -q '^_TORCH_CEILING="2.12.0"$' "$INSTALL_SH" && echo yes)"

rm -rf "${VENV_DIR:?}"
summary
