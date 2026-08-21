#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -eu

# Bundled ELF objects use $ORIGIN RUNPATHs; display and driver libraries stay host-owned.
APPDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PATH="$APPDIR/usr/bin:$APPDIR/usr/sbin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export APPDIR PATH

# Keep WebKitGTK on the bundled CBDT emoji font. Jammy's Skia can abort when a
# newer host FreeType selects a COLRv1 font, while host text fonts remain usable.
# The policy carries this mount's AppDir because Fontconfig has no portable way
# to name a directory relative to the config file: prefix="relative" arrived in
# 2.14, and 2.13 hosts such as Ubuntu 22.04 anchor such a path at the host root.
# Materialize it in a directory this user owns; a shared /tmp path would let
# anyone else on the machine choose what this process treats as a font policy.
unsloth_fonts_template="$APPDIR/usr/etc/fonts/unsloth-appimage.conf"
unsloth_fonts_state="${XDG_RUNTIME_DIR:-${XDG_CACHE_HOME:-${HOME:-}/.cache}}/unsloth-studio"
FONTCONFIG_FILE="$unsloth_fonts_template"
if [ -r "$unsloth_fonts_template" ] &&
  mkdir -p "$unsloth_fonts_state" 2>/dev/null &&
  sed "s|@APPDIR@|$APPDIR|g" "$unsloth_fonts_template" \
    >"$unsloth_fonts_state/fonts-${APPDIR##*/}.conf" 2>/dev/null; then
  FONTCONFIG_FILE="$unsloth_fonts_state/fonts-${APPDIR##*/}.conf"
  # Each mount gets its own name, so a second instance cannot repoint the first
  # one's policy mid-run. Drop only the copies whose mount is gone: an age-based
  # sweep would delete a live instance's copy out from under it, and its next
  # WebKit helper would start with an unreadable FONTCONFIG_FILE, back on the
  # host's COLRv1 font and back to the crash this file exists to prevent.
  for unsloth_stale in "$unsloth_fonts_state"/fonts-*.conf; do
    [ -f "$unsloth_stale" ] || continue
    [ "$unsloth_stale" != "$FONTCONFIG_FILE" ] || continue
    unsloth_stale_mount="$(sed -n 's|^[[:space:]]*<dir>\(.*\)/usr/share/unsloth/fonts</dir>.*|\1|p' \
      "$unsloth_stale" 2>/dev/null | head -1)"
    [ -n "$unsloth_stale_mount" ] && [ -d "$unsloth_stale_mount" ] || rm -f "$unsloth_stale"
  done
  unset unsloth_stale unsloth_stale_mount
fi
export FONTCONFIG_FILE
unset unsloth_fonts_template unsloth_fonts_state

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
