#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "Usage: $0 APPDIR" >&2
  exit 2
fi

appdir="$(realpath "$1")"
libdir="$appdir/usr/lib"
[[ -d "$libdir" ]] || { echo "AppDir has no usr/lib: $appdir" >&2; exit 1; }
command -v patchelf >/dev/null || { echo "patchelf is required" >&2; exit 1; }

# These libraries cross into a running host service, driver, or a host module
# loaded later by WebKit. They must remain one coherent host-side closure. This
# is intentionally a final sweep rather than a linuxdeploy discovery exclusion:
# input plugins can add dependencies after linuxdeploy's initial closure pass.
host_patterns=(
  'ld-linux*.so*' 'libc.so*' 'libm.so*' 'libdl.so*' 'libpthread.so*'
  'librt.so*' 'libresolv.so*' 'libnss_*.so*' 'libutil.so*' 'libanl.so*'
  'libGL*.so*' 'libEGL*.so*' 'libGLES*.so*' 'libOpenGL*.so*'
  'libgbm.so*' 'libdrm.so*' 'libglapi.so*' 'libvulkan.so*' 'libcuda.so*'
  'libX11.so*' 'libX11-xcb.so*' 'libxcb*.so*' 'libXext.so*'
  'libXrandr.so*' 'libXi.so*' 'libXcursor.so*' 'libXfixes.so*'
  'libXrender.so*' 'libXcomposite.so*' 'libXdamage.so*'
  'libXinerama.so*' 'libXau.so*' 'libXdmcp.so*' 'libxshmfence.so*'
  'libwayland-*.so*' 'libasound.so*' 'libpulse*.so*'
  'libstdc++.so*' 'libgcc_s.so*' 'libnghttp2.so*' 'libcurl*.so*'
)
for pattern in "${host_patterns[@]}"; do
  while IFS= read -r -d '' bundled; do
    rm -f -- "$bundled"
  done < <(find "$appdir" \( -type f -o -type l \) -name "$pattern" -print0)
done

# AppRun deliberately exports no AppDir-wide loader path. Give every dynamic ELF
# a path to the one private library directory relative to its own location. A
# dlopened host object then resolves through its own metadata and ld.so.cache,
# instead of inheriting a global Ubuntu-build-host directory (#7953).
patched=0
while IFS= read -r -d '' object; do
  [[ "$(head -c 4 "$object" 2>/dev/null || true)" == $'\177ELF' ]] || continue
  readelf -d "$object" 2>/dev/null | grep -q 'Dynamic section' || continue
  relative="$(realpath --relative-to="$(dirname -- "$object")" "$libdir")"
  if [[ "$relative" == "." ]]; then
    runpath='$ORIGIN'
  else
    runpath="\$ORIGIN/$relative"
  fi
  patchelf --set-rpath "$runpath" "$object"
  ((patched += 1))
done < <(find "$appdir" -type f -print0)

((patched > 0)) || { echo "AppDir contains no dynamic ELF objects" >&2; exit 1; }
printf 'Finalized AppImage runtime with %d $ORIGIN RUNPATHs: %s\n' "$patched" "$appdir"
