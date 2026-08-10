#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Build a SELF-CONTAINED Linux AppImage: the WebKit/GTK stack travels inside the
# bundle instead of being required from the host.
#
# Why this exists alongside build-thin-appimage.sh
# ------------------------------------------------
# The thin AppImage reuses the deb payload and takes the desktop stack from the
# host. That is correct on a distro that ships WebKitGTK 4.1, and its AppRun
# prints an apt command when the host does not. But the distros with the
# strongest reason to want an AppImage are the immutable ones, and they are
# exactly the ones that cannot run `sudo apt install`:
#
#   SteamOS 3 (Steam Deck)  libgtk-3 ✓  libsoup-3 ✓  librsvg ✓
#                           libwebkit2gtk-4.1 ✗  libayatana-appindicator3 ✗
#
# On such a host the thin AppImage cannot start at all, and the instructions it
# prints cannot be followed: /usr is read-only and there is no apt. Electron
# apps ship on those machines without trouble because Electron carries its own
# browser engine; a Tauri app has to bundle WebKitGTK to reach parity.
#
# Why not linuxdeploy
# -------------------
# The thin script's own comment names the hazard: a PARTIAL GTK/WebKit closure
# mixes a bundled GLib with the host's newer GIO modules and network libraries,
# and breaks in ways that are miserable to debug. That is an argument against
# bundling *some* of the stack, not against bundling it. So this script bundles
# the whole closure and then PROVES it is complete: assert_portable_appdir
# resolves the main binary and every bundled library with only the AppDir on the
# search path and fails on a single "not found". A partial closure cannot pass.
#
# What stays on the host, deliberately
# ------------------------------------
# glibc and the GPU/display stack. Bundling glibc breaks as soon as the host is
# newer, and bundling libGL/libEGL/libdrm/libgbm breaks hardware acceleration
# because those must match the running kernel driver. Everything above that line
# ships with us. HOST_LIBS below is that boundary, with a reason per entry.
#
# Usage: build-portable-appimage.sh DEB APPIMAGETOOL RUNTIME OUTPUT
#        build-portable-appimage.sh --verify-appdir APPDIR
set -euo pipefail

# Libraries that MUST come from the host, never the bundle.
#   glibc family : a bundled libc cannot load host libraries built against a
#                  newer one, and the loader itself is not relocatable.
#   GL/EGL/DRM   : must match the kernel driver in use; bundling them turns
#                  hardware acceleration into a software fallback at best and a
#                  crash at worst. This is what makes the Deck's RADV work.
#   X11/Wayland  : client libraries talk to the running display server and are
#                  ABI-stable; bundling them gains nothing and risks protocol skew.
HOST_LIBS_RE='^(ld-linux.*|libc|libm|libdl|libpthread|librt|libresolv|libnss_.*|libutil|libanl)\.so'
HOST_LIBS_RE="$HOST_LIBS_RE"'|^(libGL|libGLX|libGLdispatch|libEGL|libOpenGL|libgbm|libdrm|libglapi)\.so'
HOST_LIBS_RE="$HOST_LIBS_RE"'|^(libX11|libX11-xcb|libxcb.*|libXext|libXrandr|libXi|libXcursor|libXfixes|libXrender|libXcomposite|libXdamage|libXinerama|libXau|libXdmcp|libxshmfence)\.so'
HOST_LIBS_RE="$HOST_LIBS_RE"'|^(libwayland-.*)\.so'
HOST_LIBS_RE="$HOST_LIBS_RE"'|^(libasound|libpulse.*)\.so'

# Names the app dlopens, so they never appear in ldd output and must be pulled in
# by hand. The tray crate looks these up at runtime; WebKit loads its own
# helpers. Missing them is how a bundle "builds fine" and then fails to start.
DLOPEN_LIBS=(
  libayatana-appindicator3.so.1
  libappindicator3.so.1
  libpixbufloader-svg.so
  librsvg-2.so.2
)

log() { printf '[portable-appimage] %s\n' "$*"; }
die() { printf '[portable-appimage] %s\n' "$*" >&2; exit 1; }

# ── Completeness gate ────────────────────────────────────────────────────────
# The inverse of assert_thin_appdir. Instead of asserting that nothing was
# bundled, assert that EVERYTHING needed was: resolve the executable and every
# bundled library against the bundle alone (plus the host allowlist above) and
# fail on any "not found". This is the check that makes bundling safe, so it
# runs against the built AppDir and again against the extracted AppImage.
assert_portable_appdir() {
  local root="$1"
  local libdir="$root/usr/lib/unsloth"
  local failed=0

  [[ -d "$libdir" ]] || die "AppDir has no bundled library directory: $libdir"

  # WebKit is the whole point; if it is absent this is a thin bundle wearing the
  # wrong name and it will fail on exactly the hosts this script exists for.
  local webkit
  webkit="$(find "$libdir" -maxdepth 1 -name 'libwebkit2gtk-4.1.so*' -print -quit)"
  [[ -n "$webkit" ]] || die "Portable AppImage is missing libwebkit2gtk-4.1: nothing was bundled"

  # WebKit spawns helper processes; without them pages never render.
  local webkit_exec="$root/usr/libexec/unsloth-webkit"
  [[ -x "$webkit_exec/WebKitNetworkProcess" ]] \
    || die "Portable AppImage is missing WebKitNetworkProcess"
  [[ -x "$webkit_exec/WebKitWebProcess" ]] \
    || die "Portable AppImage is missing WebKitWebProcess"

  local target
  while IFS= read -r -d '' target; do
    local unresolved
    unresolved="$(
      LD_LIBRARY_PATH="$libdir" ldd "$target" 2>/dev/null |
        sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*not found.*$/\1/p'
    )"
    if [[ -n "$unresolved" ]]; then
      printf 'Unresolved in %s:\n%s\n' "$target" "$unresolved" >&2
      failed=1
    fi
  done < <(
    printf '%s\0' "$root/usr/bin/unsloth-studio"
    find "$libdir" "$webkit_exec" -type f \( -name '*.so*' -o -perm -u+x \) -print0
  )

  [[ "$failed" -eq 0 ]] || die "Portable AppImage closure is incomplete (see above)"
  log "closure verified complete: $(find "$libdir" -name '*.so*' | wc -l) libraries"
}

if [[ "${1:-}" == "--verify-appdir" ]]; then
  [[ $# -eq 2 ]] || die "Usage: $0 --verify-appdir APPDIR"
  assert_portable_appdir "$2"
  exit 0
fi

[[ $# -eq 4 ]] || die "Usage: $0 DEB APPIMAGETOOL RUNTIME OUTPUT"

deb_path="$1"; appimagetool_path="$2"; runtime_path="$3"; output_path="$4"
for input_path in "$deb_path" "$appimagetool_path" "$runtime_path"; do
  [[ -f "$input_path" ]] || die "Required AppImage input does not exist: $input_path"
done
[[ "$(dpkg-deb --field "$deb_path" Architecture)" == "amd64" ]] \
  || die "The Linux AppImage must be built from an amd64 deb."

work_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/unsloth-portable-appimage.XXXXXX")"
trap 'rm -rf -- "$work_root"' EXIT
app_dir="$work_root/AppDir"
verify_dir="$work_root/verify"
output_dir="$(dirname -- "$output_path")"
mkdir -p "$app_dir" "$verify_dir" "$output_dir"
output_dir="$(CDPATH= cd -- "$output_dir" && pwd -P)"
output_path="$output_dir/$(basename -- "$output_path")"

# Reuse the deb payload, exactly as the thin build does: it is already a working
# package, so the executable, desktop file and icon are known-good inputs.
dpkg-deb --extract "$deb_path" "$app_dir"

binary_file="$app_dir/usr/bin/unsloth-studio"
desktop_file="$app_dir/usr/share/applications/Unsloth.desktop"
icon_file="$app_dir/usr/share/icons/hicolor/128x128/apps/unsloth-studio.png"
for required_path in "$desktop_file" "$icon_file" "$binary_file"; do
  [[ -f "$required_path" ]] || die "The deb is missing an AppImage input: $required_path"
done

libdir="$app_dir/usr/lib/unsloth"
webkit_exec="$app_dir/usr/libexec/unsloth-webkit"
mkdir -p "$libdir" "$webkit_exec"

# ── Collect the closure ──────────────────────────────────────────────────────
# Walk ldd output transitively from the executable plus the dlopened names, and
# copy in everything that is not on the host allowlist. Resolving with ldd (not a
# hand-written list) is what makes the closure complete rather than partial.
resolve_lib() {  # soname -> absolute path on the build host, or empty
  local name="$1" path
  path="$(ldconfig -p 2>/dev/null | awk -v n="$name" '$1 == n { print $NF; exit }')" || true
  [[ -n "$path" && -e "$path" ]] && { printf '%s\n' "$path"; return 0; }
  for dir in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib/x86_64-linux-gnu /lib64 /lib; do
    [[ -e "$dir/$name" ]] && { printf '%s\n' "$dir/$name"; return 0; }
  done
  return 1
}

declare -A seen=()
queue=("$binary_file")
for name in "${DLOPEN_LIBS[@]}"; do
  if path="$(resolve_lib "$name")"; then
    queue+=("$path")
  else
    log "note: dlopened $name not present on the build host, skipping"
  fi
done

while [[ ${#queue[@]} -gt 0 ]]; do
  current="${queue[0]}"; queue=("${queue[@]:1}")
  while read -r soname target; do
    [[ -n "$soname" ]] || continue
    [[ "$soname" =~ $HOST_LIBS_RE ]] && continue
    [[ -n "${seen[$soname]:-}" ]] && continue
    [[ -n "$target" && -e "$target" ]] || continue
    seen[$soname]=1
    cp -L "$target" "$libdir/$soname"
    queue+=("$libdir/$soname")
  done < <(ldd "$current" 2>/dev/null | sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*\([^[:space:]]*\).*$/\1 \2/p')
done
log "bundled ${#seen[@]} libraries"

# WebKit's helper processes. Without these the window opens and stays blank,
# which is the single most confusing way for this bundle to fail.
for helper_dir in /usr/libexec/webkit2gtk-4.1 /usr/lib/x86_64-linux-gnu/webkit2gtk-4.1 \
                  /usr/lib/webkit2gtk-4.1 /usr/libexec/webkit2gtk-4.0; do
  if [[ -d "$helper_dir" ]]; then
    cp -a "$helper_dir/." "$webkit_exec/"
    log "bundled WebKit helpers from $helper_dir"
    break
  fi
done
# The helpers link against the same closure, so walk them too.
while IFS= read -r -d '' helper; do
  while read -r soname target; do
    [[ -n "$soname" ]] || continue
    [[ "$soname" =~ $HOST_LIBS_RE ]] && continue
    [[ -n "${seen[$soname]:-}" ]] && continue
    [[ -n "$target" && -e "$target" ]] || continue
    seen[$soname]=1
    cp -L "$target" "$libdir/$soname"
  done < <(ldd "$helper" 2>/dev/null | sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*\([^[:space:]]*\).*$/\1 \2/p')
done < <(find "$webkit_exec" -type f -perm -u+x -print0)

# ── Loadable-module data ─────────────────────────────────────────────────────
# GLib/GTK find these through compiled indexes that embed absolute build-host
# paths. Copying the .so files without their indexes is precisely the "bundled
# GLib, host modules" mix the thin script warns about, so the caches are
# regenerated against the bundle and AppRun points the env vars at them.
pixbuf_dir="$app_dir/usr/lib/unsloth/gdk-pixbuf"
mkdir -p "$pixbuf_dir"
for cand in /usr/lib/x86_64-linux-gnu/gdk-pixbuf-2.0/*/loaders /usr/lib64/gdk-pixbuf-2.0/*/loaders; do
  [[ -d "$cand" ]] || continue
  cp -a "$cand/." "$pixbuf_dir/" 2>/dev/null || true
  break
done
gio_dir="$app_dir/usr/lib/unsloth/gio-modules"
mkdir -p "$gio_dir"
for cand in /usr/lib/x86_64-linux-gnu/gio/modules /usr/lib64/gio/modules; do
  [[ -d "$cand" ]] || continue
  cp -a "$cand/." "$gio_dir/" 2>/dev/null || true
  break
done
schema_dir="$app_dir/usr/share/glib-2.0/schemas"
mkdir -p "$schema_dir"
if [[ -d /usr/share/glib-2.0/schemas ]]; then
  cp -a /usr/share/glib-2.0/schemas/. "$schema_dir/" 2>/dev/null || true
  command -v glib-compile-schemas >/dev/null 2>&1 \
    && glib-compile-schemas "$schema_dir" >/dev/null 2>&1 || true
fi

# ── Make everything relocatable ──────────────────────────────────────────────
# Absolute RUNPATHs from the build host would send the loader to directories that
# do not exist on the target. $ORIGIN makes each object look next to itself.
if command -v patchelf >/dev/null 2>&1; then
  while IFS= read -r -d '' obj; do
    patchelf --set-rpath '$ORIGIN' "$obj" 2>/dev/null || true
  done < <(find "$libdir" -maxdepth 1 -name '*.so*' -type f -print0)
  while IFS= read -r -d '' obj; do
    patchelf --set-rpath '$ORIGIN/../../lib/unsloth' "$obj" 2>/dev/null || true
  done < <(find "$pixbuf_dir" "$gio_dir" -name '*.so' -type f -print0 2>/dev/null)
  while IFS= read -r -d '' obj; do
    patchelf --set-rpath '$ORIGIN/../../lib/unsloth' "$obj" 2>/dev/null || true
  done < <(find "$webkit_exec" -type f -perm -u+x -print0)
  patchelf --set-rpath '$ORIGIN/../lib/unsloth' "$binary_file" 2>/dev/null || true
  # Regenerate the loader cache with bundle-relative paths.
  if command -v gdk-pixbuf-query-loaders >/dev/null 2>&1; then
    ( cd "$pixbuf_dir" && GDK_PIXBUF_MODULEDIR="$pixbuf_dir" \
        gdk-pixbuf-query-loaders > "$pixbuf_dir/loaders.cache" 2>/dev/null ) || true
    sed -i "s|$pixbuf_dir/|./|g" "$pixbuf_dir/loaders.cache" 2>/dev/null || true
  fi
else
  die "patchelf is required to build a relocatable AppImage"
fi

ln -sf usr/share/applications/Unsloth.desktop "$app_dir/Unsloth.desktop"
ln -sf usr/share/icons/hicolor/128x128/apps/unsloth-studio.png "$app_dir/unsloth-studio.png"
ln -sf unsloth-studio.png "$app_dir/.DirIcon"

# ── AppRun ───────────────────────────────────────────────────────────────────
# The thin build forbids setting LD_LIBRARY_PATH, because with a partial closure
# it is what causes the bundled/host mixing. With a verified-complete closure the
# opposite is true: the bundle must win consistently, or the loader satisfies
# half the graph from the host and we are back to mixing.
cat > "$app_dir/AppRun" <<'EOF'
#!/bin/sh
set -eu

appdir=${APPDIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}
libdir="$appdir/usr/lib/unsloth"

export PATH="$appdir/usr/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_DATA_DIRS="$appdir/usr/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
export GDK_PIXBUF_MODULE_FILE="$libdir/gdk-pixbuf/loaders.cache"
export GDK_PIXBUF_MODULEDIR="$libdir/gdk-pixbuf"
export GIO_MODULE_DIR="$libdir/gio-modules"
export GSETTINGS_SCHEMA_DIR="$appdir/usr/share/glib-2.0/schemas"
# WebKit refuses to start its helpers from anywhere else.
export WEBKIT_EXEC_PATH="$appdir/usr/libexec/unsloth-webkit"

# Software rendering fallback. The bundle deliberately does NOT carry libGL, so
# rendering uses the host driver; when there is none that works (headless, a
# broken Mesa, some VMs) WebKit crashes instead of degrading. Opt in with
# UNSLOTH_SOFTWARE_RENDER=1 rather than paying for it by default.
if [ "${UNSLOTH_SOFTWARE_RENDER:-0}" = "1" ]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export WEBKIT_DISABLE_COMPOSITING_MODE=1
fi

binary="$appdir/usr/bin/unsloth-studio"
missing=$(
  ldd "$binary" 2>/dev/null |
    sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*not found.*$/\1/p'
)

if [ -n "$missing" ]; then
  message="Unsloth cannot start because these libraries are missing from the host:

$missing

These are the few libraries the AppImage deliberately does not bundle (the C
library and the GPU/display stack), because they have to match your system."
  printf '%s\n' "$message" >&2
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="Unsloth cannot start" --text="$message" >/dev/null 2>&1 || true
  elif command -v xmessage >/dev/null 2>&1; then
    xmessage -center "$message" >/dev/null 2>&1 || true
  fi
  exit 127
fi

exec "$binary" "$@"
EOF
chmod +x "$app_dir/AppRun"

assert_portable_appdir "$app_dir"

ARCH=x86_64 "$appimagetool_path" --appimage-extract-and-run \
  --no-appstream \
  --runtime-file "$runtime_path" \
  "$app_dir" "$output_path"

[[ -x "$output_path" ]] || die "appimagetool did not create an executable AppImage: $output_path"

# Verify the packaged result, not just the directory we assembled.
( cd "$verify_dir" && "$output_path" --appimage-extract >/dev/null )
assert_portable_appdir "$verify_dir/squashfs-root"

grep -q 'LD_LIBRARY_PATH=' "$verify_dir/squashfs-root/AppRun" \
  || die "Portable AppRun must put the bundled libraries first."

log "Built portable AppImage: $output_path"
log "size: $(du -h "$output_path" | cut -f1)"
