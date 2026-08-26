#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Print the absolute path of the `unsloth` CLI belonging to ONE Unsloth home.
#
# Usage:
#   parity-find-unsloth.sh <studio-home>
#
# Why this is not `command -v unsloth`. install.sh writes a shim into
# $HOME/.local/bin, which is a single name shared by every install on the
# machine; the second of two installs overwrites the first. A job that runs two
# builds side by side and then asks PATH which one to launch gets the same build
# twice, wearing two labels, and the comparison reports "no difference" -- which
# is exactly the shape of failure the parity job exists to detect elsewhere.
#
# The layout is `$STUDIO_HOME/unsloth_studio/bin/unsloth` (install.sh sets
# VENV_DIR="$STUDIO_HOME/unsloth_studio"). The other candidates are the legacy
# layouts `runtime/lifecycle._find_unsloth_bin` still accepts; they are checked
# so this script and that function cannot disagree about where an Unsloth lives.
# Nothing is guessed: if no candidate exists this fails loudly rather than
# falling back to whatever PATH offers.

set -euo pipefail

HOME_DIR="${1:?parity-find-unsloth.sh: <studio-home> is required}"

for candidate in \
  "$HOME_DIR/unsloth_studio/bin/unsloth" \
  "$HOME_DIR/bin/unsloth" \
  "$HOME_DIR"/.venv*/bin/unsloth
do
  if [ -x "$candidate" ]; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done

{
  echo "parity-find-unsloth.sh: no unsloth CLI under $HOME_DIR"
  echo "looked for unsloth_studio/bin/unsloth, bin/unsloth and .venv*/bin/unsloth"
  echo "contents:"
  ls -la "$HOME_DIR" 2>&1 || true
} >&2
exit 1
