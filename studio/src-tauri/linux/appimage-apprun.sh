#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -eu

# linuxdeploy wraps this launcher with its generated GUI hook runner. Resolve the
# AppDir without adding a process-wide library search path: bundled ELF objects
# use $ORIGIN RUNPATHs, while host display and driver libraries stay host-owned.
APPDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PATH="$APPDIR/usr/bin:$APPDIR/usr/sbin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export APPDIR PATH

# Jammy's WebKitGTK helper prefix is relative to its /usr prefix. Preserve
# that working-directory contract without a global loader path.
cd "$APPDIR/usr"


exec "$APPDIR/usr/bin/unsloth-studio" "$@"
