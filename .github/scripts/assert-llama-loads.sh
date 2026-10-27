#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Assert the installed llama.cpp loads and runs on this macOS (min-OS <= host).
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

# vtool comes with the CLT; skip the static check if missing.
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

# --version forces dyld to load every linked dylib.
if ! "$SERVER" --version >/tmp/llama-server-version.txt 2>&1; then
  echo "---- llama-server --version output ----"
  cat /tmp/llama-server-version.txt || true
  fail "llama-server failed to launch on macOS $HOST_VER (dyld load / symbol error)"
fi

# Also check the env Unsloth builds for its child (it must use DYLD_LIBRARY_PATH).
# Resolve the interpreter from STUDIO_HOME first; the clean-machine lane scrubs PATH.
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
# Fail rather than skip: a skip is indistinguishable from a pass.
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
