#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -euo pipefail

usage() {
  echo "Usage: $0 APPIMAGE|--appdir APPDIR" >&2
  exit 2
}

if [[ $# -eq 2 && "$1" == "--appdir" ]]; then
  appdir="$2"
  cleanup=""
elif [[ $# -eq 1 && -f "$1" ]]; then
  work_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/unsloth-appimage-verify.XXXXXX")"
  cleanup="$work_root"
  trap 'rm -rf -- "$cleanup"' EXIT
  appimage="$(realpath "$1")"
  (
    cd "$work_root"
    "$appimage" --appimage-extract >/dev/null
  )
  appdir="$work_root/squashfs-root"
else
  usage
fi

[[ -d "$appdir" ]] || { echo "AppDir does not exist: $appdir" >&2; exit 1; }
[[ -x "$appdir/AppRun" ]] || { echo "Complete AppImage has no executable AppRun" >&2; exit 1; }

if ! grep -Rqs 'GIO_MODULE_DIR=' \
  "$appdir/AppRun" "$appdir/apprun-hooks" "$appdir/usr/lib" 2>/dev/null; then
  echo "Complete AppImage does not isolate bundled GIO modules" >&2
  exit 1
fi

if ! grep -Rqs 'unset[[:space:]]\+GIO_EXTRA_MODULES' \
  "$appdir/AppRun" "$appdir/apprun-hooks" "$appdir/usr/lib" 2>/dev/null; then
  echo "Complete AppImage does not reject host GIO_EXTRA_MODULES" >&2
  exit 1
fi
binary="$(find "$appdir/usr/bin" -maxdepth 1 -type f -name 'unsloth*' -perm -111 -print -quit 2>/dev/null)"
[[ -n "$binary" ]] || { echo "Complete AppImage has no Unsloth executable" >&2; exit 1; }

machine="$(readelf -h "$binary" | sed -n 's/^[[:space:]]*Machine:[[:space:]]*//p')"
[[ "$machine" == "Advanced Micro Devices X86-64" ]] || {
  echo "Complete AppImage has the wrong architecture: ${machine:-unknown}" >&2
  exit 1
}

require_basename() {
  local pattern="$1"
  if ! find "$appdir" \( -type f -o -type l \) -name "$pattern" -print -quit | grep -q .; then
    echo "Complete AppImage is missing required runtime component: $pattern" >&2
    return 1
  fi
}

for component in \
  'libglib-2.0.so*' 'libgobject-2.0.so*' 'libgio-2.0.so*' \
  'libgtk-3.so*' 'libgdk-3.so*' 'libgdk_pixbuf-2.0.so*' \
  'libwebkit2gtk-4.1.so*' 'libjavascriptcoregtk-4.1.so*' 'libsoup-3.0.so*' \
  'libappindicator3.so*' 'WebKitNetworkProcess' 'WebKitWebProcess' \
  'libwebkit2gtkinjectedbundle.so'; do
  require_basename "$component"
done

# The bundle owns one coherent userspace web runtime, but glibc, the loader,
# Wayland client ABI, and graphics drivers must remain host-owned so an Ubuntu
# 22.04 build can use a newer distribution's coherent EGL/Mesa/Wayland stack.
for forbidden in \
  'libc.so*' 'libpthread.so*' 'libm.so*' 'libdl.so*' 'librt.so*' \
  'ld-linux*.so*' 'libGL.so*' 'libGLX*.so*' 'libEGL.so*' \
  'libwayland-client.so*' \
  'libnghttp2.so*' 'libcurl*.so*' 'libstdc++.so*' 'libgcc_s.so*' \
  'libdrm.so*' 'libgbm.so*' 'libvulkan.so*' 'libcuda.so*'; do
  if found="$(find "$appdir" \( -type f -o -type l \) -name "$forbidden" -print -quit)" && [[ -n "$found" ]]; then
    echo "Complete AppImage must leave host runtime component unbundled: $found" >&2
    exit 1
  fi
done

max_glibc="$({ readelf --version-info "$binary" 2>/dev/null || true; } |
  grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/GLIBC_//' | sort -Vu | tail -1)"
[[ -n "$max_glibc" ]] || { echo "Could not determine the executable glibc floor" >&2; exit 1; }
if [[ "$(printf '%s\n%s\n' 2.35 "$max_glibc" | sort -V | tail -1)" != "2.35" ]]; then
  echo "Executable requires GLIBC_$max_glibc, newer than the Ubuntu 22.04 floor GLIBC_2.35" >&2
  exit 1
fi

printf 'Verified complete x86_64 AppImage runtime (GLIBC_%s): %s\n' "$max_glibc" "$appdir"
