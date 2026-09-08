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

# The launch above uses this shell's environment, not the one Unsloth builds for
# its child. That one was Linux-shaped on macOS (LD_LIBRARY_PATH, which dyld
# ignores) while the installer's own validation set DYLD_LIBRARY_PATH, so the
# defect could not show up at install time (#8566). A unit test with a
# monkeypatched sys.platform cannot prove the real thing; this can.
# Resolve the interpreter from STUDIO_HOME first, not from PATH. The
# clean-machine lane scrubs PATH down to system directories and puts the shim
# under its own UNSLOTH_STUDIO_HOME, so `command -v unsloth` is empty there and
# a PATH-only lookup would skip this assertion in the one lane whose whole
# point is a clean install, while CI still reported success. The tauri delivery
# nests its venv one level deeper (clean-machine-install-ci.yml checks
# $HOME_DIR/studio/unsloth_studio), so both layouts are candidates.
STUDIO_PY=""
for candidate in \
  "$UNSLOTH_HOME/unsloth_studio/bin/python" \
  "$UNSLOTH_HOME/studio/unsloth_studio/bin/python" \
  "$UNSLOTH_HOME/.venv/bin/python" \
  "$HOME/.unsloth/unsloth_studio/bin/python"; do
  [ -x "$candidate" ] && { STUDIO_PY="$candidate"; break; }
done
if [ -z "$STUDIO_PY" ]; then
  for shim in "$UNSLOTH_HOME/bin/unsloth" "$(command -v unsloth || true)"; do
    [ -n "$shim" ] && [ -x "$shim" ] || continue
    candidate="$(head -1 "$shim" | sed 's/^#!//' | awk '{print $1}')"
    [ -n "$candidate" ] && [ -x "$candidate" ] && { STUDIO_PY="$candidate"; break; }
  done
fi
# Fail rather than skip: an install that produced a llama-server but no
# reachable interpreter is itself a broken install, and a skip here is
# indistinguishable from a pass.
[ -n "$STUDIO_PY" ] || fail "no Unsloth interpreter found under $UNSLOTH_HOME or on PATH; cannot check the launch environment"
if [ -n "$STUDIO_PY" ]; then
  if ! PYTHONPATH=studio/backend "$STUDIO_PY" - "$SERVER" <<'PY'
import os, sys
from core.inference.llama_cpp import LlamaCppBackend, _llama_lib_dir

binary = sys.argv[1]
lib_dir = str(_llama_lib_dir(binary))
env = LlamaCppBackend._llama_server_env_for_binary(binary)
got = env.get("DYLD_LIBRARY_PATH", "")
print(f"DYLD_LIBRARY_PATH: {got or '<unset>'}")
if not got:
    sys.exit("Unsloth would launch llama-server with no DYLD_LIBRARY_PATH; dyld ignores LD_LIBRARY_PATH")
if got.split(os.pathsep)[0] != lib_dir:
    sys.exit(f"expected {lib_dir} first on DYLD_LIBRARY_PATH, got {got}")
print("child launch environment is correct for dyld")
PY
  then
    fail "Unsloth's llama-server launch environment is wrong for macOS (see above)"
  fi
fi

echo "llama.cpp load validation passed on macOS $HOST_VER"
echo "  server: $SERVER"
sed -n '1,4p' /tmp/llama-server-version.txt 2>/dev/null || true
