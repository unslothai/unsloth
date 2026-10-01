#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# A piped install's _REPO_ROOT is the caller's cwd: only a --local checkout run may read files there.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_tmp="$(mktemp -d)"
trap 'rm -rf "$_tmp"' EXIT

_gate="$(sed -n '/^_REPO_IS_CHECKOUT=0$/,/^esac$/p' "$INSTALL_SH")"
_lookup="$(sed -n '/^_find_no_torch_runtime() {$/,/^}$/p' "$INSTALL_SH")"
[ -n "$_gate" ] || bad "could not extract the _REPO_IS_CHECKOUT gate"
[ -n "$_lookup" ] || bad "could not extract _find_no_torch_runtime"

mkdir -p "$_tmp/cwd/studio/backend/requirements" "$_tmp/venv/lib/studio/backend/requirements"
echo "planted" > "$_tmp/cwd/studio/backend/requirements/no-torch-runtime.txt"
echo "official" > "$_tmp/venv/lib/studio/backend/requirements/no-torch-runtime.txt"
cp "$INSTALL_SH" "$_tmp/cwd/install.sh"

_resolve() {
    # $1 = the $0 the installer sees, $2 = STUDIO_LOCAL_INSTALL
    (
        _zero="$1"
        STUDIO_LOCAL_INSTALL="$2"
        cd "$_tmp/cwd"
        _REPO_ROOT="$_tmp/cwd"
        VENV_DIR="$_tmp/venv"
        eval "$(printf '%s\n' "$_gate" | sed "s|\"\$0\"|\"$_zero\"|g; s|\\[ -r \"\$0\" \\]|[ -r \"$_zero\" ]|g")"
        eval "$_lookup"
        _find_no_torch_runtime
    )
}

echo "=== no-torch runtime requirements source ==="
assert_eq "piped install ignores a file planted in the cwd" \
    "$_tmp/venv/lib/studio/backend/requirements/no-torch-runtime.txt" "$(_resolve sh "")"
assert_eq "piped install with STUDIO_LOCAL_INSTALL still ignores the cwd" \
    "$_tmp/venv/lib/studio/backend/requirements/no-torch-runtime.txt" "$(_resolve sh true)"
assert_eq "checkout run without --local uses the installed copy" \
    "$_tmp/venv/lib/studio/backend/requirements/no-torch-runtime.txt" "$(_resolve ./install.sh "")"
assert_eq "--local checkout run uses the checkout's copy" \
    "$_tmp/cwd/studio/backend/requirements/no-torch-runtime.txt" "$(_resolve ./install.sh true)"

echo "=== ROCm-on-WSL helper staging ==="
assert_not_contains "no fixed /tmp fallback for the sudo helper" "$(cat "$INSTALL_SH")" "/tmp/_unsloth_rocm_wsl.sh"

summary
