#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _llama_relocatable_rpath_args() from studio/setup.sh: $ORIGIN RUNPATH on Linux only.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
_FUNC_FILE=$(mktemp)
sed -n '/^_llama_relocatable_rpath_args()/,/^}/p' "$SETUP_SH" > "$_FUNC_FILE"
if [ ! -s "$_FUNC_FILE" ]; then
    echo "FAIL: could not extract _llama_relocatable_rpath_args from setup.sh"
    exit 1
fi

run_args() {
    UNAME_S="$1" bash -c ". '$_FUNC_FILE'; uname() { printf '%s' \"\$UNAME_S\"; }; _llama_relocatable_rpath_args"
}

echo "=== test_llama_relocatable_rpath ==="

LINUX="$(run_args Linux)"
# $ORIGIN must stay literal: the loader expands it, not the shell.
assert_contains "Linux bakes an \$ORIGIN install RUNPATH" "$LINUX" '-DCMAKE_INSTALL_RPATH=$ORIGIN'
assert_contains "Linux builds with the install RUNPATH" "$LINUX" '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON'
assert_contains "Linux keeps the link path" "$LINUX" '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON'
assert_not_contains "no build-tree path is named" "$LINUX" 'llama.cpp.build'

# macOS gets @loader_path in the Metal branch instead.
assert_eq "Darwin adds nothing" "" "$(run_args Darwin)"
assert_eq "unknown host adds nothing" "" "$(run_args "")"

# CPU_FALLBACK_CMAKE_ARGS must be copied after the base line to inherit the flags.
BASE_LINE="$(grep -n 'CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release' "$SETUP_SH" | head -1)"
assert_contains "configure line calls the helper" "$BASE_LINE" '$(_llama_relocatable_rpath_args)'
FALLBACK_LINE="$(grep -n 'CPU_FALLBACK_CMAKE_ARGS="\$CMAKE_ARGS"' "$SETUP_SH" | head -1 | cut -d: -f1)"
BASE_NO="$(echo "$BASE_LINE" | cut -d: -f1)"
if [ -n "$FALLBACK_LINE" ] && [ "$FALLBACK_LINE" -gt "$BASE_NO" ]; then
    ok "CPU fallback args are copied after the RUNPATH is set"
else
    bad "CPU fallback args are copied after the RUNPATH is set (base=$BASE_NO fallback=$FALLBACK_LINE)"
fi

rm -f "$_FUNC_FILE"
summary
