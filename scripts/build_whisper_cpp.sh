#!/bin/sh
# Build whisper.cpp's whisper-server for Studio's GGUF dictation engine.
#
# Installs into the managed Studio home so the backend's binary discovery
# (core/inference/stt_ggml_sidecar.py::find_whisper_server_binary) picks it up:
#   <UNSLOTH_STUDIO_HOME>/whisper.cpp/build/bin/whisper-server   (custom home)
#   ~/.unsloth/whisper.cpp/build/bin/whisper-server              (default)
#
# Usage:
#   ./scripts/build_whisper_cpp.sh              # build the pinned tag
#   WHISPER_CPP_TAG=v1.9.0 ./scripts/build_whisper_cpp.sh
#
# Requires: git, cmake, a C/C++ toolchain (the same prerequisites as a
# llama.cpp source build). GPU backends are auto-detected by whisper.cpp's
# CMake (Metal on macOS; set GGML_CUDA=1 to force a CUDA build on Linux).

set -eu

WHISPER_CPP_SOURCE="${WHISPER_CPP_SOURCE:-https://github.com/ggml-org/whisper.cpp}"
WHISPER_CPP_TAG="${WHISPER_CPP_TAG:-v1.9.1}"

STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-${STUDIO_HOME:-}}"
if [ -n "$STUDIO_HOME" ]; then
    INSTALL_DIR="$STUDIO_HOME/whisper.cpp"
else
    INSTALL_DIR="$HOME/.unsloth/whisper.cpp"
fi

command -v git >/dev/null 2>&1 || { echo "ERROR: git is required" >&2; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake is required" >&2; exit 1; }

echo "==> Building whisper.cpp ($WHISPER_CPP_TAG) into $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

if [ ! -d "$INSTALL_DIR/src/.git" ]; then
    rm -rf "$INSTALL_DIR/src"
    git clone --depth 1 --branch "$WHISPER_CPP_TAG" "$WHISPER_CPP_SOURCE" "$INSTALL_DIR/src"
else
    git -C "$INSTALL_DIR/src" fetch --depth 1 origin "$WHISPER_CPP_TAG"
    git -C "$INSTALL_DIR/src" checkout FETCH_HEAD
fi

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF"
if [ "${GGML_CUDA:-0}" = "1" ]; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON"
fi

# shellcheck disable=SC2086
cmake -S "$INSTALL_DIR/src" -B "$INSTALL_DIR/src/build" $CMAKE_FLAGS
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
cmake --build "$INSTALL_DIR/src/build" --config Release --target whisper-server -j"$NCPU"

mkdir -p "$INSTALL_DIR/build/bin"
cp "$INSTALL_DIR/src/build/bin/whisper-server" "$INSTALL_DIR/build/bin/whisper-server"

echo "==> Installed $INSTALL_DIR/build/bin/whisper-server"
"$INSTALL_DIR/build/bin/whisper-server" --help >/dev/null 2>&1 && echo "==> Binary runs OK"
