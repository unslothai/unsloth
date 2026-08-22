#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -euo pipefail

# The ELF sweep below parses readelf output, which is translated.
export LC_ALL=C

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "Usage: $0 APPDIR" >&2
  exit 2
fi

appdir="$(realpath "$1")"
libdir="$appdir/usr/lib"
[[ -d "$libdir" ]] || { echo "AppDir has no usr/lib: $appdir" >&2; exit 1; }
command -v patchelf >/dev/null || { echo "patchelf is required" >&2; exit 1; }

asset_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
for asset in UnslothSafeEmoji.ttf UnslothSafeEmoji.LICENSE unsloth-appimage-fonts.conf; do
  [[ -f "$asset_dir/$asset" ]] || {
    echo "Complete AppImage asset is missing: $asset_dir/$asset" >&2
    exit 1
  }
done
install -D -m 644 \
  "$asset_dir/UnslothSafeEmoji.ttf" \
  "$appdir/usr/share/unsloth/fonts/UnslothSafeEmoji.ttf"
install -D -m 644 \
  "$asset_dir/UnslothSafeEmoji.LICENSE" \
  "$appdir/usr/share/doc/unsloth-safe-emoji/copyright"
install -D -m 644 \
  "$asset_dir/unsloth-appimage-fonts.conf" \
  "$appdir/usr/etc/fonts/unsloth-appimage.conf"

# Remove host-coupled libraries after all input plugins deploy their dependencies.
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
  # Keep libva and libvdpau so libgstlibav can load without host VA-API.
  'libnvidia-*.so*'
  'libstdc++.so*' 'libgcc_s.so*' 'libnghttp2.so*' 'libcurl*.so*'
)
for pattern in "${host_patterns[@]}"; do
  while IFS= read -r -d '' bundled; do
    rm -f -- "$bundled"
  done < <(find "$appdir" \( -type f -o -type l \) -name "$pattern" -print0)
done

# Use $ORIGIN RUNPATHs so host objects do not inherit bundled libraries (#7953).
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
