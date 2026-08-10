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
  librsvg-2.so.2
)

stamp_appimage_bundle_type() {
  local binary_path="$1"

  python3 - "$binary_path" <<'PY'
import pathlib
import sys

binary_path = pathlib.Path(sys.argv[1])
deb_marker = b"__TAURI_BUNDLE_TYPE_VAR_DEB"
appimage_marker = b"__TAURI_BUNDLE_TYPE_VAR_APP"
binary = binary_path.read_bytes()

deb_count = binary.count(deb_marker)
appimage_count = binary.count(appimage_marker)
if deb_count != 1 or appimage_count != 0:
    sys.exit(
        f"Expected exactly one Tauri deb bundle marker and no AppImage marker in "
        f"{binary_path}; found deb={deb_count}, appimage={appimage_count}"
    )

binary_path.write_bytes(binary.replace(deb_marker, appimage_marker, 1))

stamped = binary_path.read_bytes()
if stamped.count(deb_marker) != 0 or stamped.count(appimage_marker) != 1:
    sys.exit(f"Failed to stamp the Tauri AppImage bundle marker in {binary_path}")
PY
}

# Dependency pairs ("soname path") for one ELF.
#
# ldd renders a normal dependency as `name => /path (0x..)`, but an ABSOLUTE DT_NEEDED
# as `/path (0x..)` with no arrow at all. Matching only the arrow form silently skips
# those, which is how an absolute libsqlite3.so reference reached a shipped bundle: the
# file was never copied, and on the target the loader went looking for a build-host
# directory. Both forms are parsed here so every caller sees the same thing.
ldd_pairs() {  # elf -> "soname target" per line
  ldd "$1" 2>/dev/null | sed -n \
    -e 's|^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*\([^[:space:]]*\).*$|\1 \2|p' \
    -e 's|^[[:space:]]*\(/[^[:space:]]*\)[[:space:]]*(0x.*$|\1 \1|p'
}

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
    # Names on the host allowlist are supposed to come from the target system,
    # so their absence on THIS machine says nothing about the bundle. Everything
    # else unresolved is a genuine hole in the closure.
    unresolved="$(
      LD_LIBRARY_PATH="$libdir" ldd "$target" 2>/dev/null |
        sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*not found.*$/\1/p' |
        while IFS= read -r miss; do
          [[ "$miss" =~ $HOST_LIBS_RE ]] || printf '%s\n' "$miss"
        done
    )"
    if [[ -n "$unresolved" ]]; then
      printf 'Unresolved in %s:\n%s\n' "$target" "$unresolved" >&2
      failed=1
    fi
  done < <(
    printf '%s\0' "$root/usr/bin/unsloth-studio"
    find "$libdir" "$webkit_exec" -type f \( -name '*.so*' -o -perm -u+x \) -print0
  )

  # ldd resolves an absolute DT_NEEDED happily ON THE BUILD HOST, because the path is
  # right there -- so the closure check above cannot see this class of fault at all.
  # Check the entries themselves: any '/' means the loader will bypass RUNPATH and go
  # looking for a build-host directory on the user's machine.
  local absneeded=0 obj dep
  while IFS= read -r -d '' obj; do
    for dep in $(patchelf --print-needed "$obj" 2>/dev/null || true); do
      case "$dep" in
        */*) printf 'Absolute DT_NEEDED in %s: %s\n' "$obj" "$dep" >&2
             absneeded=$((absneeded + 1)) ;;
      esac
    done
  done < <(
    printf '%s\0' "$root/usr/bin/unsloth-studio"
    find "$libdir" "$webkit_exec" -type f \( -name '*.so*' -o -perm -u+x \) -print0
  )
  [[ "$absneeded" -eq 0 ]] || die "$absneeded absolute DT_NEEDED entry(ies) -- these bypass RUNPATH"

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

# tauri-bundler stamps the executable copied into a deb so the updater selects the
# deb installer. This one is going into an AppImage, so stamp the equal-length
# AppImage marker. Without it the app logs "APPDIR ... but this application was not
# detected as an AppImage" and the updater rejects its own download as an invalid deb.
stamp_appimage_bundle_type "$binary_file"

libdir="$app_dir/usr/lib/unsloth"
webkit_exec="$app_dir/usr/libexec/unsloth-webkit"
mkdir -p "$libdir" "$webkit_exec"

# ── Collect the closure ──────────────────────────────────────────────────────
# Walk ldd output transitively from the executable plus the dlopened names, and
# copy in everything that is not on the host allowlist. Resolving with ldd (not a
# hand-written list) is what makes the closure complete rather than partial.
# Extra directories to search for dlopened names, colon-separated. Needed on build
# hosts that are not FHS-laid-out (nix, guix), where ldconfig knows nothing and the
# libraries live outside /usr/lib. Empty and harmless on a normal distro.
EXTRA_LIB_DIRS="${PORTABLE_APPIMAGE_LIB_DIRS:-}"

resolve_lib() {  # soname -> absolute path on the build host, or empty
  local name="$1" path dir
  path="$(ldconfig -p 2>/dev/null | awk -v n="$name" '$1 == n { print $NF; exit }')" || true
  [[ -n "$path" && -e "$path" ]] && { printf '%s\n' "$path"; return 0; }
  local IFS=:
  for dir in $EXTRA_LIB_DIRS ${LD_LIBRARY_PATH:-} ; do
    [[ -n "$dir" && -e "$dir/$name" ]] && { printf '%s\n' "$dir/$name"; return 0; }
  done
  IFS=$' \t\n'
  for dir in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib/x86_64-linux-gnu /lib64 /lib; do
    [[ -e "$dir/$name" ]] && { printf '%s\n' "$dir/$name"; return 0; }
  done
  return 1
}

declare -A seen=()
queue=("$binary_file")
for name in "${DLOPEN_LIBS[@]}"; do
  if path="$(resolve_lib "$name")"; then
    # Copy the library ITSELF, not just walk its dependencies. Queuing alone
    # bundled everything the tray library needs and not the tray library, so the
    # app started and then panicked in libappindicator-sys' dlopen.
    seen[$name]=1
    cp -L "$path" "$libdir/$name"
    queue+=("$libdir/$name")
  else
    log "note: dlopened $name not present on the build host, skipping"
  fi
done

while [[ ${#queue[@]} -gt 0 ]]; do
  current="${queue[0]}"; queue=("${queue[@]:1}")
  while read -r soname target; do
    [[ -n "$soname" ]] || continue
    # ldd prints the dynamic loader with an absolute path in the name column, and
    # some hosts print absolute paths generally, so reduce to a bare soname before
    # matching the host allowlist or using it as a destination filename.
    soname="$(basename -- "$soname")"
    [[ "$soname" =~ $HOST_LIBS_RE ]] && continue
    [[ -n "${seen[$soname]:-}" ]] && continue
    [[ -n "$target" && -e "$target" ]] || continue
    seen[$soname]=1
    cp -L "$target" "$libdir/$soname"
    queue+=("$libdir/$soname")
  done < <(ldd_pairs "$current")
done
log "bundled ${#seen[@]} libraries"

# WebKit's helper processes. Without these the window opens and stays blank,
# which is the single most confusing way for this bundle to fail.
# Ask pkg-config where WebKit actually lives before guessing at FHS paths: the
# helpers sit under $exec_prefix/libexec on any layout, which is the only thing
# that holds on a non-FHS build host (nix, guix) as well as on Debian.
webkit_search=()
if webkit_prefix="$(pkg-config --variable=exec_prefix webkit2gtk-4.1 2>/dev/null)" \
   && [[ -n "$webkit_prefix" ]]; then
  webkit_search+=("$webkit_prefix/libexec/webkit2gtk-4.1")
fi
[[ -n "${PORTABLE_APPIMAGE_WEBKIT_EXEC_DIR:-}" ]] \
  && webkit_search+=("$PORTABLE_APPIMAGE_WEBKIT_EXEC_DIR")
webkit_search+=(
  /usr/libexec/webkit2gtk-4.1
  /usr/lib/x86_64-linux-gnu/webkit2gtk-4.1
  /usr/lib/webkit2gtk-4.1
  /usr/libexec/webkit2gtk-4.0
)
WEBKIT_HELPER_SRC=""
for helper_dir in "${webkit_search[@]}"; do
  if [[ -d "$helper_dir" && -e "$helper_dir/WebKitNetworkProcess" ]]; then
    cp -a "$helper_dir/." "$webkit_exec/"
    chmod -R u+w "$webkit_exec"
    # Remember where they came from: that directory IS the string compiled into
    # libwebkit2gtk, and patching the literal beats guessing at the layout.
    WEBKIT_HELPER_SRC="$helper_dir"
    log "bundled WebKit helpers from $helper_dir"
    break
  fi
done

# WebKit also dlopens an "injected bundle" -- the library it loads into every web
# process to run the API's JS glue -- from a second compiled-in absolute path, under
# $prefix/lib/webkit2gtk-4.1/injected-bundle. Without it the window opens but every
# page logs "Error loading the injected bundle" and the app's JS bridge is dead.
# Bundle it under the same fixed link as the helpers so one symlink serves both.
WEBKIT_INJECTED_SRC=""
for injected_dir in \
  "${webkit_prefix:-}/lib/webkit2gtk-4.1/injected-bundle" \
  "${WEBKIT_HELPER_SRC:-}/injected-bundle" \
  /usr/lib/x86_64-linux-gnu/webkit2gtk-4.1/injected-bundle \
  /usr/lib/webkit2gtk-4.1/injected-bundle \
  /usr/lib64/webkit2gtk-4.1/injected-bundle; do
  if [[ -d "$injected_dir" ]]; then
    mkdir -p "$webkit_exec/injected-bundle"
    cp -a "$injected_dir/." "$webkit_exec/injected-bundle/"
    chmod -R u+w "$webkit_exec/injected-bundle"
    WEBKIT_INJECTED_SRC="$injected_dir"
    log "bundled WebKit injected bundle from $injected_dir"
    break
  fi
done
# Anything we copy in can pull further dependencies: the WebKit helpers, and the
# GIO modules / pixbuf loaders copied below. Walking only the main binary leaves
# exactly the partial closure this design exists to avoid, so walk every one.
walk_closure() {  # dir -> copies missing deps of every ELF under it into $libdir
  local root="$1" obj soname target
  [[ -d "$root" ]] || return 0
  while IFS= read -r -d '' obj; do
    while read -r soname target; do
      [[ -n "$soname" ]] || continue
      soname="$(basename -- "$soname")"
      [[ "$soname" =~ $HOST_LIBS_RE ]] && continue
      [[ -n "${seen[$soname]:-}" ]] && continue
      [[ -n "$target" && -e "$target" ]] || continue
      seen[$soname]=1
      cp -L "$target" "$libdir/$soname"
      # Newly copied library may itself pull more in.
      walk_one "$libdir/$soname"
    done < <(ldd_pairs "$obj")
  done < <(find "$root" -type f \( -name '*.so*' -o -perm -u+x \) -print0)
}

walk_one() {  # single ELF -> copies its missing deps, transitively
  local obj="$1" soname target
  while read -r soname target; do
    [[ -n "$soname" ]] || continue
    soname="$(basename -- "$soname")"
    [[ "$soname" =~ $HOST_LIBS_RE ]] && continue
    [[ -n "${seen[$soname]:-}" ]] && continue
    [[ -n "$target" && -e "$target" ]] || continue
    seen[$soname]=1
    cp -L "$target" "$libdir/$soname"
    walk_one "$libdir/$soname"
  done < <(ldd_pairs "$obj")
}

# MiniBrowser is WebKit's own test harness, not something the app ever launches.
# It drags in the whole X11/Wayland/GBM client stack, so drop it rather than
# bundle dependencies nothing uses.
rm -f "$webkit_exec/MiniBrowser"

walk_closure "$webkit_exec"

# ── Loadable-module data ─────────────────────────────────────────────────────
# GLib/GTK find these through compiled indexes that embed absolute build-host
# paths. Copying the .so files without their indexes is precisely the "bundled
# GLib, host modules" mix the thin script warns about, so the caches are
# regenerated against the bundle and AppRun points the env vars at them.
# Ask pkg-config for the module directories that belong to the SAME gio and
# gdk-pixbuf we just bundled. Guessing at /usr/lib*/gio/modules copies the HOST's
# modules next to our GLib, which is the bundled-GLib-with-host-modules mix the
# thin build warns about -- and on a build host whose GLib is not the host's, the
# modules link against libraries that are not in the bundle at all.
pixbuf_dir="$app_dir/usr/lib/unsloth/gdk-pixbuf"
gio_dir="$app_dir/usr/lib/unsloth/gio-modules"
mkdir -p "$pixbuf_dir" "$gio_dir"

src_pixbuf="$(pkg-config --variable=gdk_pixbuf_moduledir gdk-pixbuf-2.0 2>/dev/null || true)"
[[ -d "${src_pixbuf:-}" ]] || src_pixbuf="$(
  ls -d /usr/lib/x86_64-linux-gnu/gdk-pixbuf-2.0/*/loaders /usr/lib64/gdk-pixbuf-2.0/*/loaders 2>/dev/null | head -1
)"
if [[ -d "${src_pixbuf:-}" ]]; then
  cp -a "$src_pixbuf/." "$pixbuf_dir/" 2>/dev/null || true
  chmod -R u+w "$pixbuf_dir"
  log "pixbuf loaders from $src_pixbuf"
fi

src_gio="$(pkg-config --variable=giomoduledir gio-2.0 2>/dev/null || true)"
[[ -d "${src_gio:-}" ]] || src_gio="$(
  ls -d /usr/lib/x86_64-linux-gnu/gio/modules /usr/lib64/gio/modules 2>/dev/null | head -1
)"
if [[ -d "${src_gio:-}" ]]; then
  cp -a "$src_gio/." "$gio_dir/" 2>/dev/null || true
  chmod -R u+w "$gio_dir"
  log "gio modules from $src_gio"
fi

schema_dir="$app_dir/usr/share/glib-2.0/schemas"
mkdir -p "$schema_dir"
src_schema="$(pkg-config --variable=schemasdir gsettings-desktop-schemas 2>/dev/null || true)"
[[ -d "${src_schema:-}" ]] || src_schema=/usr/share/glib-2.0/schemas
if [[ -d "$src_schema" ]]; then
  cp -a "$src_schema/." "$schema_dir/" 2>/dev/null || true
  chmod -R u+w "$schema_dir"
  command -v glib-compile-schemas >/dev/null 2>&1 \
    && glib-compile-schemas "$schema_dir" >/dev/null 2>&1 || true
fi

# The modules pull their own libraries (libproxy, gnutls, rsvg, jxl); without
# this they would resolve back to the build host at runtime.
walk_closure "$pixbuf_dir"
walk_closure "$gio_dir"

# ── Make everything relocatable ──────────────────────────────────────────────
# Absolute RUNPATHs from the build host would send the loader to directories that
# do not exist on the target. $ORIGIN makes each object look next to itself.
if command -v patchelf >/dev/null 2>&1; then
  # Copies inherit the source mode, and libraries on a read-only prefix (nix,
  # guix, or anything served from a store) arrive without write permission --
  # patchelf then fails on every object and the bundle ships with absolute
  # build-host RUNPATHs that resolve nowhere on the target. Make them writable
  # first; the `|| true` on each patchelf call would otherwise hide it.
  chmod -R u+w "$libdir" "$webkit_exec" "$pixbuf_dir" "$gio_dir" 2>/dev/null || true
  chmod u+w "$binary_file" 2>/dev/null || true
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

  # Every executable must use the HOST's dynamic loader.
  #
  # A build host with a non-standard prefix (nix, guix, a relocated toolchain)
  # stamps its own absolute interpreter path into PT_INTERP. An AppImage built
  # that way cannot start on any machine that lacks that exact path -- and on the
  # build machine itself it starts but then runs under a loader whose
  # ld.so.cache does not know the target's library directories, so precisely the
  # libraries we deliberately did NOT bundle (wayland, GL, EGL) come up missing.
  # Observed exactly that: libwayland-client.so.0 not found on a host that has it
  # in /usr/lib64. Normalise to the FHS path the runtime guarantees.
  normalise_interpreter() {
    local obj="$1" interp
    interp="$(patchelf --print-interpreter "$obj" 2>/dev/null || true)"
    [[ -n "$interp" ]] || return 0            # static or not an executable
    case "$interp" in
      /lib64/ld-linux-x86-64.so.2|/lib/ld-linux-x86-64.so.2|/lib/ld64.so.*) return 0 ;;
    esac
    patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 "$obj" 2>/dev/null \
      && log "interpreter normalised: $(basename -- "$obj") (was $interp)"
  }
  normalise_interpreter "$binary_file"
  while IFS= read -r -d '' obj; do
    normalise_interpreter "$obj"
  done < <(find "$webkit_exec" -type f -perm -u+x -print0)
  # Regenerate the loader cache with bundle-relative paths.
  if command -v gdk-pixbuf-query-loaders >/dev/null 2>&1; then
    ( cd "$pixbuf_dir" && GDK_PIXBUF_MODULEDIR="$pixbuf_dir" \
        gdk-pixbuf-query-loaders > "$pixbuf_dir/loaders.cache" 2>/dev/null ) || true
    sed -i "s|$pixbuf_dir/|./|g" "$pixbuf_dir/loaders.cache" 2>/dev/null || true
  fi
  # DT_NEEDED entries that are ABSOLUTE PATHS.
  #
  # A dependency is normally recorded as a bare soname and resolved through
  # RUNPATH/ld.so.cache, but a linker invoked with a full path can record that path
  # instead. The loader then ignores RUNPATH entirely for that entry and opens the
  # absolute path, which on the target does not exist -- so the bundle fails to start
  # with a missing-library error naming a build-host directory. Seen with libsoup,
  # libtinysparql and libwebkit2gtk all recording an absolute libsqlite3.so.
  #
  # The file is already bundled under its basename by the closure walk, so rewriting
  # the entry to that basename makes it resolve through $ORIGIN like everything else.
  while IFS= read -r -d '' obj; do
    for dep in $(patchelf --print-needed "$obj" 2>/dev/null || true); do
      case "$dep" in
        */*)
          dep_base="$(basename -- "$dep")"
          if [[ -e "$libdir/$dep_base" ]]; then
            patchelf --replace-needed "$dep" "$dep_base" "$obj" 2>/dev/null \
              && log "rewrote absolute DT_NEEDED in $(basename -- "$obj"): $dep_base"
          else
            die "absolute DT_NEEDED $dep in $obj, and $dep_base is not bundled"
          fi
          ;;
      esac
    done
  done < <(
    printf '%s\0' "$binary_file"
    find "$libdir" "$webkit_exec" -type f \( -name '*.so*' -o -perm -u+x \) -print0
  )

  # Prove the rewrite happened. Every `patchelf ... || true` above is a place a
  # read-only file or an unsupported object silently keeps its build-host RUNPATH,
  # which produces a bundle that works only on the machine that built it.
  leaked=0
  while IFS= read -r -d '' obj; do
    case "$(patchelf --print-rpath "$obj" 2>/dev/null || true)" in
      */nix/store/*|/usr/*|/opt/*)
        printf 'RUNPATH still absolute: %s -> %s\n' \
          "$obj" "$(patchelf --print-rpath "$obj" 2>/dev/null)" >&2
        leaked=$((leaked + 1))
        ;;
    esac
  done < <(
    printf '%s\0' "$binary_file"
    find "$libdir" "$webkit_exec" -type f \( -name '*.so*' -o -perm -u+x \) -print0
  )
  [[ "$leaked" -eq 0 ]] || die "$leaked object(s) kept a build-host RUNPATH"
  log "RUNPATHs rewritten to \$ORIGIN"

  bad_interp=0
  while IFS= read -r -d '' obj; do
    case "$(patchelf --print-interpreter "$obj" 2>/dev/null || true)" in
      ''|/lib64/ld-linux-x86-64.so.2|/lib/ld-linux-x86-64.so.2) ;;
      *) printf 'Non-standard interpreter: %s -> %s\n' \
           "$obj" "$(patchelf --print-interpreter "$obj" 2>/dev/null)" >&2
         bad_interp=$((bad_interp + 1)) ;;
    esac
  done < <(
    printf '%s\0' "$binary_file"
    find "$webkit_exec" -type f -perm -u+x -print0
  )
  [[ "$bad_interp" -eq 0 ]] || die "$bad_interp executable(s) point at a build-host loader"
  log "interpreters point at the host loader"
else
  die "patchelf is required to build a relocatable AppImage"
fi

# ── Redirect WebKit's compiled-in helper directory ───────────────────────────
# WebKitGTK dropped WEBKIT_EXEC_PATH; the path to WebKitNetworkProcess /
# WebKitWebProcess is baked into libwebkit2gtk at build time as an absolute
# string. Inside an AppImage that path points at the BUILD host, so the helpers
# either fail to spawn or, worse, spawn the build machine's copies -- which is
# what happened here: they launched from the build prefix and then could not find
# libX11 under that prefix's loader.
#
# The mount point is random (/tmp/.mount_XXXXXX), so it cannot be patched in at
# build time. Rewrite the string to a short fixed path instead and have AppRun
# point that at the bundle. Replacement must be no longer than the original; the
# remainder is NUL-padded, which keeps every offset in the ELF intact.
#
# The compiled-in paths are NOT at a fixed layout. Debian/Ubuntu put both the
# helpers and the injected bundle under the multiarch libdir
# (/usr/lib/x86_64-linux-gnu/webkit2gtk-4.1), while nix and Fedora use
# $prefix/libexec and /usr/lib64. An earlier version of this matched the literals
# '/libexec/webkit2gtk-4.X' and '/lib/webkit2gtk-4.X/injected-bundle', which score
# zero hits on Ubuntu -- and because a no-op looked identical to success, the build
# passed and shipped an AppImage with exactly the fault this patch exists to fix.
# So: patch the directories we actually copied from, fall back to a layout-agnostic
# pattern, and treat "nothing matched" as a build failure.
WEBKIT_LINK_PATH="/tmp/.unsloth-wk-$(id -u 2>/dev/null || echo 0)"
python3 - "$libdir" "$WEBKIT_LINK_PATH" \
         "${WEBKIT_HELPER_SRC:-}" "${WEBKIT_INJECTED_SRC:-}" <<'PYEOF'
import pathlib, re, sys

libdir, link = pathlib.Path(sys.argv[1]), sys.argv[2].encode()
helper_src, injected_src = sys.argv[3].encode(), sys.argv[4].encode()

def literal(path):
    """Match an exact directory string, with or without a trailing slash."""
    return re.compile(re.escape(path.rstrip(b'/')) + rb'/?\x00')

# Injected bundle first: its string ends in '/injected-bundle', so the helper
# patterns (which anchor the NUL straight after the version) cannot swallow it.
patterns = []
if injected_src:
    patterns.append(('injected', literal(injected_src), link + b'/injected-bundle'))
patterns.append(('injected',
                 re.compile(rb'[\x20-\x7e]*/webkit2gtk-4\.[01]/injected-bundle/?\x00'),
                 link + b'/injected-bundle'))
if helper_src:
    patterns.append(('helper', literal(helper_src), link))
patterns.append(('helper',
                 re.compile(rb'[\x20-\x7e]*/webkit2gtk-4\.[01]/?\x00'), link))

found = {'helper': 0, 'injected': 0}
for so in sorted(libdir.glob('libwebkit2gtk-*.so*')):
    if so.is_symlink():
        continue
    blob = bytearray(so.read_bytes())
    hits = 0
    for kind, pat, replacement in patterns:
        for m in list(pat.finditer(blob)):
            original = m.group()[:-1]
            if len(replacement) > len(original):
                print(f"  cannot patch: {replacement!r} longer than {original!r}",
                      file=sys.stderr)
                sys.exit(1)
            blob[m.start():m.start() + len(m.group())] = (
                replacement + b'\x00' * (len(m.group()) - len(replacement)))
            found[kind] += 1
            hits += 1
    if hits:
        so.write_bytes(bytes(blob))
        print(f"  redirected {hits} compiled-in path(s) in {so.name} -> {link.decode()}")

missing = [k for k, n in found.items() if n == 0]
if missing:
    print(f"  ERROR: found no compiled-in WebKit {'/'.join(missing)} path to redirect.",
          file=sys.stderr)
    print("  The bundle would fall back to the build host's paths at runtime, so this",
          file=sys.stderr)
    print("  is a hard failure rather than a warning. Check the layout of:", file=sys.stderr)
    print(f"    helpers  = {helper_src.decode() or '<not found>'}", file=sys.stderr)
    print(f"    injected = {injected_src.decode() or '<not found>'}", file=sys.stderr)
    sys.exit(1)
PYEOF

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
# WebKitGTK has its helper directory compiled in (WEBKIT_EXEC_PATH is gone), and
# the build rewrote that string to the fixed path below. Point it at this mount.
# Re-created every launch: the mount point changes each run, and a stale link
# from a previous version would silently spawn the wrong helpers.
WEBKIT_LINK="__WEBKIT_LINK_PATH__"
ln -sfn "$appdir/usr/libexec/unsloth-webkit" "$WEBKIT_LINK" 2>/dev/null || true

# Software rendering fallback. The bundle deliberately does NOT carry libGL, so
# rendering uses the host driver; when there is none that works (headless, a
# broken Mesa, some VMs) WebKit crashes instead of degrading. Opt in with
# UNSLOTH_SOFTWARE_RENDER=1 rather than paying for it by default.
if [ "${UNSLOTH_SOFTWARE_RENDER:-0}" = "1" ]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export WEBKIT_DISABLE_COMPOSITING_MODE=1
fi

# Libraries carry their DATA directories as absolute paths fixed at build time, and
# unlike RUNPATHs those cannot be rewritten. On a build host whose prefix is not FHS
# (nix, guix) the bundled libxkbcommon looks for keymaps under a prefix the target does
# not have, and GTK then fails to set up a keymap at all. Point the standard overrides
# at the host's copies when the caller has not, which is a no-op on a Debian-built
# bundle because the values already match.
for _xkb in /usr/share/X11/xkb /usr/local/share/X11/xkb; do
  [ -n "${XKB_CONFIG_ROOT:-}" ] && break
  [ -d "$_xkb" ] && { XKB_CONFIG_ROOT="$_xkb"; export XKB_CONFIG_ROOT; break; }
done
for _xloc in /usr/share/X11/locale /usr/local/share/X11/locale; do
  [ -n "${XLOCALEDIR:-}" ] && break
  [ -d "$_xloc" ] && { XLOCALEDIR="$_xloc"; export XLOCALEDIR; break; }
done

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
sed -i "s|__WEBKIT_LINK_PATH__|$WEBKIT_LINK_PATH|" "$app_dir/AppRun"
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
