#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -euo pipefail

# Every check below parses readelf output, which is translated.
export LC_ALL=C

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

require_basename() {
  local pattern="$1"
  if ! find "$appdir" \( -type f -o -type l \) -name "$pattern" -print -quit | grep -q .; then
    echo "Complete AppImage is missing required runtime component: $pattern" >&2
    return 1
  fi
}

# FIRST, before the launcher-hygiene checks below. Those ask whether a bundle that
# ships a runtime scopes it correctly; this asks whether it ships one at all, and a
# bundle that ships none fails them too -- for a reason that reads like an env-var
# bug. A pre-#9113 thin AppImage, which resolves webkit2gtk from the host, reported
# "does not clear an inherited LD_LIBRARY_PATH" and sent the reader after the wrong
# defect; the honest answer is the soname list below.
#
# Report every miss rather than exiting on the first: the AppRun host-preflight
# names all four missing sonames at once, and a verifier that names one per CI run
# turns one diagnosis into as many red runs as there are absent components.
missing_components=()
required_components=()
for component in \
  'libglib-2.0.so*' 'libgobject-2.0.so*' 'libgio-2.0.so*' \
  'libgtk-3.so*' 'libgdk-3.so*' 'libgdk_pixbuf-2.0.so*' \
  'libwebkit2gtk-4.1.so*' 'libjavascriptcoregtk-4.1.so*' 'libsoup-3.0.so*' \
  'libappindicator3.so*' 'WebKitNetworkProcess' 'WebKitWebProcess' \
  'libwebkit2gtkinjectedbundle.so' \
  'libgiognutls.so' \
  'libgstcoreelements.so' 'libgstplayback.so' 'libgstpulseaudio.so' \
  'libgstisomp4.so' 'libgstvideoparsersbad.so' 'libgstlibav.so' \
  'gst-plugin-scanner'; do
  required_components+=("$component")
  require_basename "$component" || missing_components+=("$component")
done
if ((${#missing_components[@]} > 0)); then
  echo "Complete AppImage is missing ${#missing_components[@]} of ${#required_components[@]} required runtime components; it resolves them from the host" >&2
  exit 1
fi

launchers=("$appdir/AppRun")
[[ ! -e "$appdir/AppRun.wrapped" ]] || launchers+=("$appdir/AppRun.wrapped")
if grep -aEq '^[[:space:]]*(export[[:space:]]+)?LD_LIBRARY_PATH=' "${launchers[@]}" \
  "$appdir/apprun-hooks"/* 2>/dev/null || \
  grep -aFq 'LD_LIBRARY_PATH=%s/usr/lib/' "${launchers[@]}" 2>/dev/null; then
  echo "Complete AppImage must not set a global AppDir LD_LIBRARY_PATH (#7953)" >&2
  exit 1
fi

# An inherited LD_LIBRARY_PATH outranks the $ORIGIN RUNPATHs below.
if ! grep -aEq '^[[:space:]]*unset[[:space:]]+LD_LIBRARY_PATH([[:space:]]|$)' \
  "${launchers[@]}" 2>/dev/null; then
  echo "Complete AppImage does not clear an inherited LD_LIBRARY_PATH" >&2
  exit 1
fi

safe_emoji_font="$appdir/usr/share/unsloth/fonts/UnslothSafeEmoji.ttf"
safe_emoji_license="$appdir/usr/share/doc/unsloth-safe-emoji/copyright"
fontconfig_file="$appdir/usr/etc/fonts/unsloth-appimage.conf"
for required_file in "$safe_emoji_font" "$safe_emoji_license" "$fontconfig_file"; do
  [[ -f "$required_file" ]] || {
    echo "Complete AppImage is missing safe emoji runtime file: $required_file" >&2
    exit 1
  }
done
for table in CBDT CBLC; do
  grep -aFq "$table" "$safe_emoji_font" || {
    echo "Complete AppImage safe emoji font has no $table bitmap table" >&2
    exit 1
  }
done
if grep -aFq 'COLR' "$safe_emoji_font"; then
  echo "Complete AppImage safe emoji font unexpectedly contains a COLR table" >&2
  exit 1
fi
grep -Fq 'Unsloth Safe Emoji' "$fontconfig_file" || {
  echo "Complete AppImage fontconfig does not prefer its safe emoji family" >&2
  exit 1
}
grep -Fq '@APPDIR@/usr/share/unsloth/fonts' "$fontconfig_file" || {
  echo "Complete AppImage fontconfig does not name its font directory absolutely" >&2
  exit 1
}
grep -Fq 'sed "s|@APPDIR@|$unsloth_fonts_appdir|g" "$unsloth_fonts_template"' "${launchers[@]}" || {
  echo "Complete AppImage does not pin fontconfig to its safe emoji policy" >&2
  exit 1
}
# A mount path carries the AppImage's own file name, so it reaches the policy encoded.
grep -Fq "s,&,\\&amp;,g" "${launchers[@]}" || {
  echo "Complete AppImage does not encode its mount path for the fontconfig policy" >&2
  exit 1
}

# Exercise the policy with the host Fontconfig, including version 2.13.
if command -v fc-match >/dev/null && command -v fc-query >/dev/null &&
  fc-query "$safe_emoji_font" >/dev/null 2>&1; then
  fc_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/unsloth-appimage-fc.XXXXXX")"
  # Encode the AppDir exactly as AppRun does, so this exercises the shipped policy.
  fc_appdir="$(printf '%s' "$appdir" | sed \
    -e 's,&,\&amp;,g' -e 's,<,\&lt;,g' -e 's,>,\&gt;,g' \
    -e 's,\\,\\\\,g' -e 's,&,\\&,g' -e 's,|,\\|,g')"
  sed "s|@APPDIR@|$fc_appdir|g" "$fontconfig_file" >"$fc_root/fonts.conf"
  selected="$(FONTCONFIG_FILE="$fc_root/fonts.conf" XDG_CACHE_HOME="$fc_root" \
    fc-match -f '%{file}' 'sans-serif:charset=1f680')"
  if [[ "$selected" != "$safe_emoji_font" ]]; then
    echo "Complete AppImage fontconfig picks ${selected:-nothing} for U+1F680," \
      "not its bundled safe emoji font" >&2
    rm -rf -- "$fc_root"
    exit 1
  fi
  # Emoji coverage must not cost the host's text fonts their own requests.
  text_font="$(FONTCONFIG_FILE="$fc_root/fonts.conf" XDG_CACHE_HOME="$fc_root" \
    fc-match -f '%{file}' 'sans-serif:charset=41')"
  if [[ -n "$text_font" && "$text_font" != "$safe_emoji_font" ]]; then
    for family in sans-serif serif monospace; do
      chosen="$(FONTCONFIG_FILE="$fc_root/fonts.conf" XDG_CACHE_HOME="$fc_root" \
        fc-match -f '%{file}' "$family")"
      [[ "$chosen" != "$safe_emoji_font" ]] || {
        echo "Complete AppImage fontconfig hands $family to the emoji font" >&2
        rm -rf -- "$fc_root"
        exit 1
      }
    done
  fi
  rm -rf -- "$fc_root"
fi

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

# Require module search paths to stay inside the AppDir.
for module_path in GTK_PATH GIO_MODULE_DIR; do
  value="$(grep -hs "^export ${module_path}=" "$appdir/apprun-hooks"/* 2>/dev/null |
    tail -1 | sed "s/^export ${module_path}=//; s/^\"//; s/\"$//")"
  if [[ -z "$value" ]]; then
    echo "Complete AppImage does not pin $module_path to the bundle" >&2
    exit 1
  fi
  IFS=':' read -ra entries <<<"$value"
  for entry in "${entries[@]}"; do
    case "$entry" in
      '$APPDIR/'*) ;;
      *)
        echo "Bundled $module_path reaches a host module directory: $entry" >&2
        exit 1
        ;;
    esac
  done
done
# Reject Tauri's absolute build-machine .DirIcon link.
[[ -e "$appdir/.DirIcon" ]] || { echo "Complete AppImage has no resolvable .DirIcon" >&2; exit 1; }
case "$(readlink "$appdir/.DirIcon" 2>/dev/null)" in
  /*) echo "Complete AppImage pins .DirIcon to a build-machine path" >&2; exit 1 ;;
esac

binary="$(find "$appdir/usr/bin" -maxdepth 1 -type f -name 'unsloth*' -perm -111 -print -quit 2>/dev/null)"
[[ -n "$binary" ]] || { echo "Complete AppImage has no Unsloth executable" >&2; exit 1; }

machine="$(readelf -h "$binary" | sed -n 's/^[[:space:]]*Machine:[[:space:]]*//p')"
[[ "$machine" == "Advanced Micro Devices X86-64" ]] || {
  echo "Complete AppImage has the wrong architecture: ${machine:-unknown}" >&2
  exit 1
}

# Require the media plugins that must match the bundled GStreamer core.
gst_plugin_count="$(find "$appdir/usr/lib/gstreamer-1.0" -maxdepth 1 -type f -name '*.so' 2>/dev/null | wc -l)"
if [[ "$gst_plugin_count" -lt 50 ]]; then
  echo "Complete AppImage bundles only $gst_plugin_count GStreamer plugins" >&2
  exit 1
fi

# Reject libraries that must remain part of the host runtime.
for forbidden in \
  'ld-linux*.so*' 'libc.so*' 'libpthread.so*' 'libm.so*' 'libdl.so*' 'librt.so*' \
  'libresolv.so*' 'libnss_*.so*' 'libutil.so*' 'libanl.so*' \
  'libGL*.so*' 'libEGL*.so*' 'libGLES*.so*' 'libOpenGL*.so*' \
  'libdrm.so*' 'libgbm.so*' 'libglapi.so*' 'libvulkan.so*' 'libcuda.so*' \
  'libX11.so*' 'libX11-xcb.so*' 'libxcb*.so*' 'libXext.so*' \
  'libXrandr.so*' 'libXi.so*' 'libXcursor.so*' 'libXfixes.so*' \
  'libXrender.so*' 'libXcomposite.so*' 'libXdamage.so*' \
  'libXinerama.so*' 'libXau.so*' 'libXdmcp.so*' 'libxshmfence.so*' \
  'libwayland-*.so*' 'libasound.so*' 'libpulse*.so*' \
  'libnvidia-*.so*' \
  'libnghttp2.so*' 'libcurl*.so*' 'libstdc++.so*' 'libgcc_s.so*'; do
  if found="$(find "$appdir" \( -type f -o -type l \) -name "$forbidden" -print -quit)" && [[ -n "$found" ]]; then
    echo "Complete AppImage must leave host runtime component unbundled: $found" >&2
    exit 1
  fi
done
dynamic_count=0
runpath_failures=0
while IFS= read -r -d '' object; do
  [[ "$(head -c 4 "$object" 2>/dev/null || true)" == $'\177ELF' ]] || continue
  # Reject foreign-architecture objects copied from multilib hosts.
  object_machine="$(readelf -h "$object" 2>/dev/null |
    sed -n 's/^[[:space:]]*Machine:[[:space:]]*//p')"
  if [[ "$object_machine" != "Advanced Micro Devices X86-64" ]]; then
    echo "Bundled object has the wrong architecture (${object_machine:-unknown}): $object" >&2
    ((runpath_failures += 1))
    continue
  fi
  dynamic="$(readelf -d "$object" 2>/dev/null || true)"
  grep -q 'Dynamic section' <<<"$dynamic" || continue
  ((dynamic_count += 1))
  runpath="$(sed -n 's/.*\(RPATH\|RUNPATH\).*Library .*path: \[\(.*\)\].*/\2/p' <<<"$dynamic")"
  case "$runpath" in
    '$ORIGIN'|'$ORIGIN/'*) ;;
    *)
      echo "Dynamic AppImage object lacks an \$ORIGIN-relative RUNPATH: $object" >&2
      ((runpath_failures += 1))
      ;;
  esac
  if grep -Eq '\(NEEDED\).*Shared library: \[/' <<<"$dynamic"; then
    echo "Dynamic AppImage object has an absolute DT_NEEDED entry: $object" >&2
    ((runpath_failures += 1))
  fi
done < <(find "$appdir" -type f -print0)
((dynamic_count > 0)) || { echo "Complete AppImage contains no dynamic ELF objects" >&2; exit 1; }
((runpath_failures == 0)) || exit 1



max_glibc="$({ readelf --version-info "$binary" 2>/dev/null || true; } |
  grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/GLIBC_//' | sort -Vu | tail -1)"
[[ -n "$max_glibc" ]] || { echo "Could not determine the executable glibc floor" >&2; exit 1; }
if [[ "$(printf '%s\n%s\n' 2.35 "$max_glibc" | sort -V | tail -1)" != "2.35" ]]; then
  echo "Executable requires GLIBC_$max_glibc, newer than the Ubuntu 22.04 floor GLIBC_2.35" >&2
  exit 1
fi

printf 'Verified complete x86_64 AppImage runtime (GLIBC_%s): %s\n' "$max_glibc" "$appdir"
