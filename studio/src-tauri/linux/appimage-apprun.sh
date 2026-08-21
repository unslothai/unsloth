#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -eu

# Bundled ELF objects use $ORIGIN RUNPATHs; display and driver libraries stay host-owned.
APPDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PATH="$APPDIR/usr/bin:$APPDIR/usr/sbin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export APPDIR PATH

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
