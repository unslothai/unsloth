#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"

CASK="$SCRIPT_DIR/../../packaging/homebrew/Casks/unsloth.rb"


assert_match() {
  if grep -qE "$2" "$CASK"; then
    ok "$1"
  else
    bad "$1"
  fi
}

[ -f "$CASK" ] || { bad "packaging/homebrew/Casks/unsloth.rb exists"; summary; exit 1; }
ok "packaging/homebrew/Casks/unsloth.rb exists"

assert_match "Linux cask links unsloth into PATH" 'binary.*Unsloth\.AppImage.*target: "unsloth"'
assert_match "Linux cask writes a desktop entry" 'unsloth-studio\.desktop'
assert_match "Linux cask registers the AppImage" 'app_image "Unsloth-Desktop-Linux\.AppImage"'

summary
