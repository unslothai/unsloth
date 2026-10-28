#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Install one side of a UI-parity comparison via the tree's own `install.sh --local --no-torch`.
# Usage: parity-install-side.sh <tree> <studio-home> <log-path>
# Not the install-unsloth-local action: it only handles the root checkout and the default home.

set -uo pipefail

TREE="${1:?parity-install-side.sh: <tree> is required}"
HOME_DIR="${2:?parity-install-side.sh: <studio-home> is required}"
LOG="${3:-logs/install.log}"

[ -f "$TREE/install.sh" ] || {
  echo "parity-install-side.sh: no install.sh in $TREE" >&2
  exit 2
}

mkdir -p "$(dirname "$LOG")" "$HOME_DIR"

echo "[parity] installing $TREE into $HOME_DIR"
echo "[parity] commit $(git -C "$TREE" rev-parse HEAD)"

set -o pipefail
(
  cd "$TREE" || exit 2
  UNSLOTH_STUDIO_HOME="$HOME_DIR" bash install.sh --local --no-torch 2>&1
) | tee "$LOG" | while IFS= read -r line || [ -n "$line" ]; do
  printf '[%4ds] %s\n' "$SECONDS" "$line"
done
