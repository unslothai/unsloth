#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Install ONE side of a UI-parity comparison: run the checked-out tree's own
# `install.sh --local --no-torch` into a home of its own.
#
# Usage:
#   parity-install-side.sh <tree> <studio-home> <log-path>
#
# NOT `.github/actions/install-unsloth-local`. That action is ROOT-CHECKOUT ONLY
# and says so: `uses: ./...` resolves from GITHUB_WORKSPACE and takes no
# expressions, so it cannot be pointed at a second tree, and it installs into
# the default home, so calling it twice would put the second build on top of the
# first. A parity job needs exactly the two things it cannot give: another tree
# and another home.
#
# What IS carried over from it, because it is the part that matters:
#
#  * `set -o pipefail` before the tee. Without it the step reports the exit
#    status of `tee`, not of install.sh, and a failed install passes.
#  * The elapsed-seconds prefix on the STEP LOG only. This is the largest step
#    in the job and install.sh's output carries no timestamps anywhere, so which
#    phase spent the time cannot be read off a CI log. Guessing has already been
#    misleading in this repo. Downstream of `tee` deliberately: the log file
#    keeps byte-for-byte what install.sh wrote.
#  * UV_CACHE_DIR is inherited from the environment rather than set here, so one
#    restored cache serves both sides.
#
# `--no-torch` because studiobench drives an external provider through its own
# pacer and loads no model. A parity digest of the DOM needs the frontend and
# the backend, and nothing that torch provides.

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
