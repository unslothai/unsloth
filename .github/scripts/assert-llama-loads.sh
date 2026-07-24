#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Assert Unsloth installed a llama.cpp that loads and runs on THIS macOS. Tests
# the contract that matters (binaries load and their minimum-OS is <= this host)
# instead of the old "did install.sh fall back to a source build?" grep, since a
# source build with a correct deployment target is a valid outcome.
set -uo pipefail

UNSLOTH_HOME="${STUDIO_HOME:-$HOME/.unsloth}"
LLAMA_DIR="${LLAMA_CPP_DIR:-$UNSLOTH_HOME/llama.cpp}"
BIN_DIR="$LLAMA_DIR/build/bin"

fail() {
  echo "::error::$*"
  if [ -f logs/install.log ]; then
    echo "---- install.log (llama.cpp lines) ----"
    grep -E "llama-prebuilt|llama\.cpp|macos prebuilt|falling back" logs/install.log | tail -80 || true
  fi
  exit 1
}

SERVER="$(find "$LLAMA_DIR" -type f -name 'llama-server' 2>/dev/null | head -1)"
QUANT="$(find "$LLAMA_DIR" -type f -name 'llama-quantize' 2>/dev/null | head -1)"
[ -n "$SERVER" ] || fail "llama-server not found under $LLAMA_DIR after install"
[ -n "$QUANT" ]  || fail "llama-quantize not found under $LLAMA_DIR after install"

HOST_VER="$(sw_vers -productVersion 2>/dev/null || echo '0')"
HOST_MAJOR="${HOST_VER%%.*}"

# Static minimum-OS check on every Mach-O we ship. vtool ships with the Xcode
# command line tools, which GitHub macOS runners always have; if it is somehow
# missing we skip the static check and rely on the runtime launch below.
if command -v vtool >/dev/null 2>&1; then
  while IFS= read -r macho; do
    [ -n "$macho" ] || continue
    minos="$(vtool -show-build "$macho" 2>/dev/null | awk '/minos/{print $2; exit}')"
    [ -n "$minos" ] || continue
    min_major="${minos%%.*}"
    if [ "$min_major" -gt "$HOST_MAJOR" ] 2>/dev/null; then
      fail "$(basename "$macho") is built for macOS $minos but this runner is macOS $HOST_VER (prebuilt is newer than the host)"
    fi
  done < <(find "$BIN_DIR" -type f \( -name '*.dylib' -o -name 'llama-server' -o -name 'llama-quantize' \) 2>/dev/null)
fi

# Runtime launch: --version forces dyld to load every linked dylib (including
# libggml-metal.dylib). A missing Metal symbol or too-new binary fails here.
if ! "$SERVER" --version >/tmp/llama-server-version.txt 2>&1; then
  echo "---- llama-server --version output ----"
  cat /tmp/llama-server-version.txt || true
  fail "llama-server failed to launch on macOS $HOST_VER (dyld load / symbol error)"
fi

echo "llama.cpp load validation passed on macOS $HOST_VER"
echo "  server: $SERVER"
sed -n '1,4p' /tmp/llama-server-version.txt 2>/dev/null || true
