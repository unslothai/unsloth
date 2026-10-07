#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Print the absolute path of the `unsloth` CLI belonging to ONE Unsloth home.
# Usage: parity-find-unsloth.sh <studio-home>
# Not `command -v unsloth`: the shared ~/.local/bin shim points at the last install.
# Candidates must match runtime/lifecycle._find_unsloth_bin.

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
