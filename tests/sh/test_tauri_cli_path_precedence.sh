#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# #9184: a desktop install left an older `unsloth` ahead of the ~/.local/bin shim, so
# `unsloth start` never reached the managed CLI. The PATH persistence block is extracted from
# install.sh and run under /bin/sh, as the installer runs.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=== test_tauri_cli_path_precedence ==="

_fn_start=$(grep -n '^_path_has_dir() {' "$INSTALL_SH" | head -1 | cut -d: -f1)
_guard_end=$(grep -n '^# end of the PATH persistence block$' "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ -z "$_fn_start" ] || [ -z "$_guard_end" ]; then
    echo "  FAIL: install.sh PATH persistence block not found"
    exit 1
fi
sed -n "${_fn_start},${_guard_end}p" "$INSTALL_SH" > "$WORK/path_guard.sh"

EXACT_LINE='export PATH="$HOME/.local/bin:$PATH"'

# $1 home  $2 TAURI_MODE  $3 login PATH  $4 _STUDIO_HOME_REDIRECT
_run_guard() {
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV HOME="$1" SHELL=/bin/bash TAURI_MODE="$2" \
        LOGIN_PATH="$3" REDIRECT="${4:-default}" GUARD="$WORK/path_guard.sh" sh -c '
        set -e
        step() { echo "$*"; }
        substep() { :; }
        C_WARN=""
        PATH="$LOGIN_PATH"
        _LOCAL_BIN="$HOME/.local/bin"
        _STUDIO_HOME_REDIRECT="$REDIRECT"
        _UNSLOTH_LOGIN_PATH="$PATH"
        _UNSLOTH_UV_BIN_DIR=""
        . "$GUARD"
        echo guard-finished
    '
}

_mk_home() {
    mkdir -p "$1/.local/bin" "$1/foreign"
    printf '#!/bin/sh\n' > "$1/foreign/unsloth"
    printf '#!/bin/sh\n' > "$1/venv_unsloth"
    chmod +x "$1/foreign/unsloth" "$1/venv_unsloth"
    ln -sfn "$1/venv_unsloth" "$1/.local/bin/unsloth"
    printf 'export PATH="%s/foreign:$HOME/.local/bin:$PATH"\n' "$1" > "$1/.bashrc"
}

_fail() { echo "  FAIL: $*"; exit 1; }

# 1. Shadowed desktop install: one prepend, idempotent, and a fresh shell resolves the shim.
H="$WORK/tauri"; _mk_home "$H"
_run_guard "$H" true "$H/foreign:$H/.local/bin:/usr/bin:/bin" > /dev/null
_run_guard "$H" true "$H/foreign:$H/.local/bin:/usr/bin:/bin" > /dev/null
[ "$(grep -cxF "$EXACT_LINE" "$H/.bashrc")" = 1 ] || _fail "shadowed desktop install did not add exactly one prepend"
_resolved=$(HOME="$H" PATH="/usr/bin:/bin" bash --noprofile --norc -c 'source "$HOME/.bashrc"; command -v unsloth')
[ "$_resolved" = "$H/.local/bin/unsloth" ] || _fail "fresh shell resolves $_resolved"
echo "  PASS: shadowed desktop install prepends once and the shim wins"

# 2. Nothing shadows the shim: the rc file is left alone (directory first, or the same binary).
H="$WORK/first"; _mk_home "$H"; cp "$H/.bashrc" "$WORK/first.orig"
_run_guard "$H" true "$H/.local/bin:$H/foreign:/usr/bin:/bin" > /dev/null
cmp -s "$H/.bashrc" "$WORK/first.orig" || _fail "unshadowed desktop install edited the rc file"
H="$WORK/same"; _mk_home "$H"; cp "$H/.bashrc" "$WORK/same.orig"
mkdir -p "$H/venvbin"; ln -sfn "$H/venv_unsloth" "$H/venvbin/unsloth"
_run_guard "$H" true "$H/venvbin:$H/.local/bin:/usr/bin:/bin" > /dev/null
cmp -s "$H/.bashrc" "$WORK/same.orig" || _fail "same binary earlier on PATH was treated as a shadow"
echo "  PASS: unshadowed desktop install leaves the rc file untouched"

# 3. Non-desktop install, custom Studio root and an active conda env keep their behaviour.
H="$WORK/normal"; _mk_home "$H"
_run_guard "$H" false "$H/foreign:$H/.local/bin:/usr/bin:/bin" > /dev/null
! grep -qxF "$EXACT_LINE" "$H/.bashrc" || _fail "non-desktop install added a prepend"
H="$WORK/custom"; _mk_home "$H"
_run_guard "$H" true "$H/foreign:$H/.local/bin:/usr/bin:/bin" env > /dev/null
! grep -qxF "$EXACT_LINE" "$H/.bashrc" || _fail "custom Studio root wrote the rc file"
H="$WORK/conda"; _mk_home "$H"
env HOME="$H" SHELL=/bin/bash CONDA_PREFIX=/opt/conda LOGIN_PATH="$H/foreign:$H/.local/bin:/usr/bin:/bin" \
    GUARD="$WORK/path_guard.sh" sh -c '
    set -e; step() { :; }; substep() { :; }; C_WARN=""; TAURI_MODE=true
    PATH="$LOGIN_PATH"; _LOCAL_BIN="$HOME/.local/bin"; _STUDIO_HOME_REDIRECT=default
    _UNSLOTH_LOGIN_PATH="$PATH"; _UNSLOTH_UV_BIN_DIR=""; . "$GUARD"' > /dev/null
! grep -qxF "$EXACT_LINE" "$H/.bashrc" || _fail "desktop install inside conda added a prepend (#5871)"
echo "  PASS: non-desktop, custom-root and conda installs unchanged"

# 4. An rc file that cannot be written warns and the install carries on.
H="$WORK/rofail"; _mk_home "$H"; rm -f "${H:?}/.bashrc"; mkdir "$H/.bashrc" "$H/.profile"
_out=$(_run_guard "$H" true "$H/foreign:$H/.local/bin:/usr/bin:/bin")
case "$_out" in *"could not write"*guard-finished*) ;; *) _fail "failed append aborted the install: $_out" ;; esac
echo "  PASS: failed profile append warns and the install continues"
