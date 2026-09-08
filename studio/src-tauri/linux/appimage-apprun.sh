#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -eu

# Bundled ELF objects use $ORIGIN RUNPATHs; display and driver libraries stay host-owned.
APPDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PATH="$APPDIR/usr/bin:$APPDIR/usr/sbin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export APPDIR PATH

# Keep Jammy's Skia off host COLRv1 fonts. Fontconfig 2.13 misresolves relative
# font paths, so materialize this mount's absolute path in user-owned state.
# The path carries the AppImage's own file name, so encode it for XML and for
# sed on the way in; the cleanup below decodes it on the way back out.
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
  # Preserve policies for live mounts and remove only departed ones.
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

# The loader reads LD_LIBRARY_PATH before those RUNPATHs, so an inherited value would put
# host GLib, GTK, WebKit or GStreamer in front of the bundle. Managed children still get it;
# the guard keeps a restart of an already-cleared process from erasing the saved value.
if [ "${LD_LIBRARY_PATH+x}" = x ]; then
  UNSLOTH_HOST_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
  export UNSLOTH_HOST_LD_LIBRARY_PATH
  unset LD_LIBRARY_PATH
fi

# WebKitGTK resolves its helpers relative to its /usr prefix.
cd "$APPDIR/usr"


exec "$APPDIR/usr/bin/unsloth-studio" "$@"
