#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _is_studio_root decides whether ~/.unsloth/studio, or a UNSLOTH_STUDIO_HOME root, is recursively
# deleted, so both directions are asserted in both modes: a gate that accepts everything, anywhere,
# passes a one-sided test. The uninstaller body deletes trees, so the gate is sed'd out of it.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT
# Keep $HOME clear of the fixture trees: the gate's siblings consult it.
HOME="$_TMP_ROOT/home"
mkdir -p "$HOME"

# _is_studio_root calls both helpers, so all three come across or the suite is vacuous.
for _name in _is_venv_dir _is_installer_leftover_name _is_studio_root; do
    _fn=$(sed -n "/^$_name() {/,/^}/p" "$UNINSTALL_SH")
    if [ -z "$_fn" ]; then
        echo "  FAIL: could not extract $_name from $UNINSTALL_SH"
        exit 1
    fi
    eval "$_fn"
done

# name, expected (own|foreign), mode (managed|custom), then paths; a trailing / makes a dir.
# "managed" is the uninstaller's second argument, for $HOME/.unsloth/studio and nothing else.
check() {
    _name="$1"; _want="$2"; _mode="$3"; shift 3
    _root="$_TMP_ROOT/$(printf '%s' "$_name" | tr -c 'a-zA-Z0-9' '_')"
    mkdir -p "$_root"
    for _rel in "$@"; do
        case "$_rel" in
            */) mkdir -p "$_root/$_rel" ;;
            *)  mkdir -p "$_root/$(dirname "$_rel")"; printf 'x' > "$_root/$_rel" ;;
        esac
    done
    case "$_mode" in
        managed) _is_studio_root "$_root" managed ;;
        *)       _is_studio_root "$_root" ;;
    esac && _got=own || _got=foreign
    if [ "$_got" = "$_want" ]; then
        echo "  PASS: $_name"; PASS=$((PASS+1))
    else
        echo "  FAIL: $_name (got $_got, want $_want)"; FAIL=$((FAIL+1))
    fi
}

echo "Layouts Unsloth created at the managed root, which must stay removable:"
check "partial install: the root marker alone" own managed ".unsloth-studio-owned"
check "current: share/studio.conf" own managed "share/studio.conf"
check "current: unsloth_studio owner marker" own managed "unsloth_studio/.unsloth-studio-owned"
check "legacy .venv carrying the owner marker" own managed ".venv/.unsloth-studio-owned"
check "pre-marker unsloth_studio venv" own managed "unsloth_studio/bin/unsloth" "unsloth_studio/bin/python"
check "pre-marker legacy .venv" own managed ".venv/bin/unsloth" ".venv/bin/python"
check "pre-marker venv proved by pyvenv.cfg alone" own managed ".venv/bin/unsloth" ".venv/pyvenv.cfg"
# An install that died before the marker was written. The root is ours and holds nothing else.
check "partial install: rollback copy only" own managed "unsloth_studio.rollback.20260908120000.4242/pyvenv.cfg"
check "partial install: invalid legacy venv only" own managed ".venv.invalid.20260908120000.4242/pyvenv.cfg"

echo
echo "A custom root proves itself with a marker, wherever the user put it:"
check "custom: the root marker alone" own custom ".unsloth-studio-owned"
check "custom: share/studio.conf" own custom "share/studio.conf"
check "custom: unsloth_studio owner marker" own custom "unsloth_studio/.unsloth-studio-owned"
check "custom: legacy .venv owner marker" own custom ".venv/.unsloth-studio-owned"

echo
echo "Directories that are not ours, which must stay refused:"
check "a bare project venv" foreign managed ".venv/bin/python"
check "a venv merely NAMED unsloth_studio" foreign managed "unsloth_studio/bin/python"
# bin/unsloth is pip's console script for the unsloth distribution, not just any console script.
check "a venv with an unrelated console script" foreign managed ".venv/bin/black" ".venv/bin/python"
check "a hand-made scratch directory" foreign managed "notes.txt"
check "an empty directory" foreign managed
# Why the split exists: pip writes bin/unsloth into ANY venv with the wheel.
check "a project venv with unsloth pip-installed, as a custom root" foreign custom \
    ".venv/bin/unsloth" ".venv/bin/python" "pyproject.toml" "src/main.py"
check "a project whose venv is NAMED unsloth_studio, as a custom root" foreign custom \
    "unsloth_studio/bin/unsloth" "unsloth_studio/bin/python"
check "a partial-install leftover at a custom root" foreign custom \
    ".venv.invalid.20260908120000.4242/pyvenv.cfg"
# The literal glob must not match itself when the directory holds nothing.
check "a directory named like the glob is not conjured" foreign managed "notes.txt"
# The name alone is not proof: only a renamed venv carries the shape.
check "a FILE named like a leftover" foreign managed ".venv.invalid.20260908120000.4242"
check "an empty directory named like a leftover" foreign managed "unsloth_studio.rollback.20260908120000.4242/"
check "a leftover-named directory holding the user's own files" foreign managed \
    ".venv.invalid.20260908120000.4242/notes.txt"
# bin/unsloth is only pip's console script when it is inside a venv; on its own it is a file.
check "a console script with no venv around it" foreign managed ".venv/bin/unsloth"
check "the same under a directory named unsloth_studio" foreign managed "unsloth_studio/bin/unsloth"
# install.sh preserves any rollback outside <stamp>.<pid>[.<n>] as user data, so neither may the
# uninstaller read one as ownership: "unsloth_studio.rollback.notes" is somebody's directory.
check "a rollback-named venv outside the installer's format" foreign managed \
    "unsloth_studio.rollback.user-data/pyvenv.cfg"
check "an invalid-venv name outside the installer's format" foreign managed \
    ".venv.invalid.backup/pyvenv.cfg"
check "a rollback name with a non-numeric pid" foreign managed \
    "unsloth_studio.rollback.20260908120000.mine/pyvenv.cfg"
check "a rollback name with a short stamp" foreign managed \
    "unsloth_studio.rollback.2026.4242/pyvenv.cfg"
# install.sh refuses to prune a rollback symlink and only ever renames a directory into place,
# so a link with the right name points at a venv it did not put there.
_linked="$_TMP_ROOT/linked_leftover"
mkdir -p "$_linked" "$_TMP_ROOT/somebodys_venv"
: > "$_TMP_ROOT/somebodys_venv/pyvenv.cfg"
: > "$_linked/keepme.txt"
ln -s "$_TMP_ROOT/somebodys_venv" "$_linked/unsloth_studio.rollback.20260908120000.4242"
if _is_studio_root "$_linked" managed; then
    echo "  FAIL: a symlinked leftover was claimed"; FAIL=$((FAIL+1))
else
    echo "  PASS: a symlinked leftover is refused"; PASS=$((PASS+1))
fi
# ... and the shapes the installers really write, including install.sh's date fallback.
check "partial install: rollback with a numeric suffix" own managed \
    "unsloth_studio.rollback.20260908120000.4242.2/pyvenv.cfg"
check "partial install: install.sh's 'time' date fallback" own managed \
    ".venv.invalid.time.4242/pyvenv.cfg"
# install.sh claims the root before it creates the uv cache, so the cache is not a sentinel:
# a hand-made ~/.unsloth/studio/cache/uv must not authorize deleting the rest of the root.
check "a uv cache with no root marker" foreign managed "cache/uv/" "notes.txt"
check "the same at a custom root" foreign custom "cache/uv/"

echo
echo "Edge cases:"
if _is_studio_root "" managed; then
    echo "  FAIL: an empty path was claimed"; FAIL=$((FAIL+1))
else
    echo "  PASS: an empty path is refused"; PASS=$((PASS+1))
fi
if _is_studio_root "$_TMP_ROOT/nothing-here" managed; then
    echo "  FAIL: a missing path was claimed"; FAIL=$((FAIL+1))
else
    echo "  PASS: a missing path is refused"; PASS=$((PASS+1))
fi
_spaced="$_TMP_ROOT/a dir with spaces/studio"
mkdir -p "$_spaced/unsloth_studio"
printf 'x' > "$_spaced/unsloth_studio/.unsloth-studio-owned"
if _is_studio_root "$_spaced"; then
    echo "  PASS: a path containing spaces is handled"; PASS=$((PASS+1))
else
    echo "  FAIL: a path containing spaces was refused"; FAIL=$((FAIL+1))
fi

# The other half of the same question: install.sh decides when to WRITE the marker this gate
# reads. In env mode $STUDIO_HOME is a user-chosen workspace, so claiming one the installer is
# about to refuse would hand somebody's project to the uninstaller.
echo
echo "Who the installer is allowed to claim:"
_fn=$(sed -n '/^_claim_studio_root() {/,/^}/p' "$INSTALL_SH")
if [ -z "$_fn" ]; then
    echo "  FAIL: could not extract _claim_studio_root from $INSTALL_SH"; FAIL=$((FAIL+1))
else
    eval "$_fn"
    # name, expected (claimed|left), redirect mode, then paths.
    claim_check() {
        _cname="$1"; _cwant="$2"; _STUDIO_HOME_REDIRECT="$3"; shift 3
        STUDIO_HOME="$_TMP_ROOT/claim_$(printf '%s' "$_cname" | tr -c 'a-zA-Z0-9' '_')"
        VENV_DIR="$STUDIO_HOME/unsloth_studio"
        mkdir -p "$STUDIO_HOME"
        for _crel in "$@"; do
            mkdir -p "$STUDIO_HOME/$(dirname "$_crel")"
            : > "$STUDIO_HOME/$_crel"
        done
        _claim_studio_root
        if [ -f "$STUDIO_HOME/.unsloth-studio-owned" ]; then _cgot=claimed; else _cgot=left; fi
        if [ "$_cgot" = "$_cwant" ]; then
            echo "  PASS: $_cname"; PASS=$((PASS+1))
        else
            echo "  FAIL: $_cname (got $_cgot, want $_cwant)"; FAIL=$((FAIL+1))
        fi
    }
    claim_check "the default root, always" claimed default
    claim_check "an empty custom root" claimed env
    claim_check "a custom root already carrying our marker" claimed env "unsloth_studio/.unsloth-studio-owned"
    claim_check "a custom root with share/studio.conf" claimed env "share/studio.conf"
    # The case: the venv-step guard refuses this root, so the claim must not run ahead of it.
    claim_check "somebody's workspace" left env "pyproject.toml" "src/main.py"
    claim_check "somebody's workspace with a venv of their own" left env "unsloth_studio/pyvenv.cfg"

    # A symlink at the marker path: the redirection would follow it and truncate the target.
    STUDIO_HOME="$_TMP_ROOT/claim_symlink"
    # shellcheck disable=SC2034  # read by the extracted _claim_studio_root
    VENV_DIR="$STUDIO_HOME/unsloth_studio"
    _STUDIO_HOME_REDIRECT=default
    mkdir -p "$STUDIO_HOME"
    printf 'precious' > "$_TMP_ROOT/claim_symlink_target"
    ln -s "$_TMP_ROOT/claim_symlink_target" "$STUDIO_HOME/.unsloth-studio-owned"
    _claim_studio_root
    if [ "$(cat "$_TMP_ROOT/claim_symlink_target")" = "precious" ]; then
        echo "  PASS: a symlinked marker path does not truncate its target"; PASS=$((PASS+1))
    else
        echo "  FAIL: a symlinked marker path truncated its target"; FAIL=$((FAIL+1))
    fi
    if [ -f "$STUDIO_HOME/.unsloth-studio-owned" ] && [ ! -L "$STUDIO_HOME/.unsloth-studio-owned" ]; then
        echo "  PASS: and the marker is a regular file afterwards"; PASS=$((PASS+1))
    else
        echo "  FAIL: the marker is still a link"; FAIL=$((FAIL+1))
    fi
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
