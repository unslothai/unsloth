#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -eu

# The hooks ran with the runtime's APPDIR; keep it to undo their exports in host mode.
unsloth_hook_appdir="${APPDIR:-}"

# Bundled ELF objects use $ORIGIN RUNPATHs; display and driver libraries stay host-owned.
APPDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PATH="$APPDIR/usr/bin:$APPDIR/usr/sbin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export APPDIR PATH

# Host mode runs a copy of the executable whose $ORIGIN/../lib RUNPATH reaches nothing, so the
# whole GTK/WebKit stack comes from the host. Undo the gtk and gstreamer hook exports that pin
# GTK, GIO, GSettings and GStreamer to the bundle; their inherited values are already gone.
unsloth_host_env() {
  unset GTK_DATA_PREFIX GTK_EXE_PREFIX GTK_PATH GTK_IM_MODULE_FILE GDK_PIXBUF_MODULE_FILE \
    GSETTINGS_SCHEMA_DIR GIO_MODULE_DIR GIO_EXTRA_MODULES GST_REGISTRY_REUSE_PLUGIN_SCANNER \
    GST_PLUGIN_SYSTEM_PATH_1_0 GST_PLUGIN_PATH_1_0 GST_PLUGIN_SCANNER_1_0 GST_PTP_HELPER_1_0
  # The hook forces Adwaita unless the user asked for a theme through APPIMAGE_GTK_THEME.
  [ "${APPIMAGE_GTK_THEME+x}" = x ] || unset GTK_THEME
  # A bundled-mode parent relaunching (an in-app update) passes its Jammy font policy back in.
  case "${FONTCONFIG_FILE:-}" in
    */unsloth-studio/fonts-*.conf | */usr/etc/fonts/unsloth-appimage.conf) unset FONTCONFIG_FILE ;;
  esac
  [ "${XDG_DATA_DIRS+x}" = x ] || return 0
  unsloth_hooked="${unsloth_hook_appdir:-$APPDIR}"
  # Each pass through the hooks (a relaunch) prepends the same prefix again.
  for unsloth_root in "$unsloth_hooked" "$APPDIR"; do
    while :; do
      case "$XDG_DATA_DIRS" in
        "$unsloth_root/usr/share:/usr/share:"*)
          XDG_DATA_DIRS="${XDG_DATA_DIRS#"$unsloth_root/usr/share:/usr/share:"}" ;;
        *) break ;;
      esac
    done
  done
  unsloth_rest="$XDG_DATA_DIRS"
  unsloth_kept=
  unsloth_first=1
  while :; do
    unsloth_entry="${unsloth_rest%%:*}"
    case "$unsloth_entry" in
      "$APPDIR" | "$APPDIR"/* | "$unsloth_hooked" | "$unsloth_hooked"/*) ;;
      *)
        if [ -n "$unsloth_first" ]; then
          unsloth_kept="$unsloth_entry"
          unsloth_first=
        else
          unsloth_kept="$unsloth_kept:$unsloth_entry"
        fi
        ;;
    esac
    case "$unsloth_rest" in
      *:*) unsloth_rest="${unsloth_rest#*:}" ;;
      *) break ;;
    esac
  done
  # Empty or unset both give GLib's default /usr/local/share:/usr/share.
  if [ -n "$unsloth_kept" ]; then
    XDG_DATA_DIRS="$unsloth_kept"
  else
    unset XDG_DATA_DIRS
  fi
  unset unsloth_hooked unsloth_root unsloth_rest unsloth_kept unsloth_first unsloth_entry
}

# UNSLOTH_APPIMAGE_WEBKIT=auto (default) uses a host WebKitGTK >= 2.54 when the probe proves the
# host stack and its appindicator load; host forces it; bundled, or anything else, keeps the bundle.
unsloth_host_exe="$APPDIR/usr/bin/host/unsloth-studio"
unsloth_webkit_request="${UNSLOTH_APPIMAGE_WEBKIT:-auto}"
unsloth_webkit_mode=bundled
unsloth_webkit_detail=
if [ ! -f "$unsloth_host_exe" ] || [ ! -x "$unsloth_host_exe" ]; then
  unsloth_webkit_detail="reason: no host copy"
else
  case "$unsloth_webkit_request" in
    host)
      unsloth_webkit_mode=host
      unsloth_webkit_detail="override, UNSLOTH_APPIMAGE_WEBKIT=host"
      ;;
    bundled)
      unsloth_webkit_detail="reason: override"
      ;;
    auto)
      if ! command -v timeout >/dev/null 2>&1; then
        unsloth_webkit_detail="reason: no timeout"
      else
        unsloth_probe_rc=0
        unsloth_probe_out=$(
          unsloth_host_env
          unset LD_LIBRARY_PATH
          cd / && exec timeout -k 1 5 "$unsloth_host_exe" --unsloth-webkit-probe </dev/null 2>/dev/null
        ) || unsloth_probe_rc=$?
        unsloth_probe_version=
        case "$unsloth_probe_out" in
          *'
'*) ;;
          *)
            unsloth_probe_version="$(printf '%s\n' "$unsloth_probe_out" |
              sed -n 's/^webkit=\([0-9]\{1,4\}\)\.\([0-9]\{1,4\}\)\.\([0-9]\{1,4\}\)$/\1.\2.\3/p')"
            ;;
        esac
        if [ "$unsloth_probe_rc" -ne 0 ]; then
          unsloth_webkit_detail="reason: probe exit $unsloth_probe_rc"
        elif [ -z "$unsloth_probe_version" ]; then
          unsloth_webkit_detail="reason: unrecognized probe output"
        else
          unsloth_probe_major="${unsloth_probe_version%%.*}"
          unsloth_probe_minor="${unsloth_probe_version#*.}"
          unsloth_probe_minor="${unsloth_probe_minor%%.*}"
          if [ "$unsloth_probe_major" -gt 2 ] ||
            { [ "$unsloth_probe_major" -eq 2 ] && [ "$unsloth_probe_minor" -ge 54 ]; }; then
            unsloth_webkit_mode=host
            unsloth_webkit_detail="host WebKitGTK $unsloth_probe_version, UNSLOTH_APPIMAGE_WEBKIT=auto"
          else
            unsloth_webkit_detail="reason: version $unsloth_probe_version < 2.54"
          fi
          unset unsloth_probe_major unsloth_probe_minor
        fi
        unset unsloth_probe_rc unsloth_probe_out unsloth_probe_version
      fi
      ;;
    *)
      unsloth_webkit_detail="reason: invalid override '$(printf '%s' "$unsloth_webkit_request" |
        tr -c 'A-Za-z0-9._-' '?' | cut -c 1-32)'"
      ;;
  esac
fi

if [ "$unsloth_webkit_mode" = bundled ]; then
  # Keep Jammy's Skia off host COLRv1 fonts. Fontconfig 2.13 misresolves relative paths, so write
  # this mount's absolute path (XML/sed-encoded; the AppImage name is in it) into user state.
  unsloth_fonts_template="$APPDIR/usr/etc/fonts/unsloth-appimage.conf"
  unsloth_fonts_state="${XDG_RUNTIME_DIR:-${XDG_CACHE_HOME:-${HOME:-}/.cache}}/unsloth-studio"
  FONTCONFIG_FILE="$unsloth_fonts_template"
  if [ -r "$unsloth_fonts_template" ] &&
    mkdir -p "$unsloth_fonts_state" 2>/dev/null &&
    unsloth_fonts_appdir="$(printf '%s' "$APPDIR" | sed \
      -e 's,&,\&amp;,g' -e 's,<,\&lt;,g' -e 's,>,\&gt;,g' \
      -e 's,\\,\\\\,g' -e 's,&,\\&,g' -e 's,|,\\|,g')" &&
    sed "s|@APPDIR@|$unsloth_fonts_appdir|g" "$unsloth_fonts_template" \
      >"$unsloth_fonts_state/fonts-${APPDIR##*/}.conf" 2>/dev/null; then
    FONTCONFIG_FILE="$unsloth_fonts_state/fonts-${APPDIR##*/}.conf"
    for unsloth_stale in "$unsloth_fonts_state"/fonts-*.conf; do
      [ -f "$unsloth_stale" ] || continue
      [ "$unsloth_stale" != "$FONTCONFIG_FILE" ] || continue
      unsloth_stale_mount="$(sed -n 's|^[[:space:]]*<dir>\(.*\)/usr/share/unsloth/fonts</dir>.*|\1|p' \
        "$unsloth_stale" 2>/dev/null | head -1 |
        sed -e 's,&lt;,<,g' -e 's,&gt;,>,g' -e 's,&amp;,\&,g')"
      [ -n "$unsloth_stale_mount" ] && [ -d "$unsloth_stale_mount" ] || rm -f "$unsloth_stale"
    done
    unset unsloth_stale unsloth_stale_mount
  fi
  export FONTCONFIG_FILE
  unset unsloth_fonts_appdir unsloth_fonts_template unsloth_fonts_state
fi

# An inherited LD_LIBRARY_PATH would put host GLib/GTK/WebKit before the bundle's RUNPATHs.
# Managed children still get the saved value; the guard keeps a restart from erasing it.
if [ "${LD_LIBRARY_PATH+x}" = x ]; then
  UNSLOTH_HOST_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
  export UNSLOTH_HOST_LD_LIBRARY_PATH
  unset LD_LIBRARY_PATH
fi

# Overwrite any inherited value so a relaunch re-decides; the app reads it to pick its plan.
UNSLOTH_APPIMAGE_WEBKIT_MODE="$unsloth_webkit_mode"
export UNSLOTH_APPIMAGE_WEBKIT_MODE
printf 'unsloth-studio: AppImage WebKitGTK runtime=%s (%s)\n' \
  "$unsloth_webkit_mode" "$unsloth_webkit_detail" >&2

# WebKitGTK resolves its helpers relative to its /usr prefix.
cd "$APPDIR/usr"

if [ "$unsloth_webkit_mode" = host ]; then
  unsloth_host_env
  unset unsloth_hook_appdir unsloth_host_exe unsloth_webkit_request unsloth_webkit_mode \
    unsloth_webkit_detail
  exec "$APPDIR/usr/bin/host/unsloth-studio" "$@"
fi
unset unsloth_hook_appdir unsloth_host_exe unsloth_webkit_request unsloth_webkit_mode \
  unsloth_webkit_detail
exec "$APPDIR/usr/bin/unsloth-studio" "$@"
