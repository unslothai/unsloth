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
# install.sh guarantees a regular file there, so a link planted in somebody's workspace is not
# proof: -f follows it, and the custom-root loop deletes what it accepts.
_linkmark="$_TMP_ROOT/linked_root_marker"
mkdir -p "$_linkmark"
: > "$_TMP_ROOT/linked_root_marker_target"
: > "$_linkmark/keepme.txt"
ln -s "$_TMP_ROOT/linked_root_marker_target" "$_linkmark/.unsloth-studio-owned"
if _is_studio_root "$_linkmark" managed; then
    echo "  FAIL: a symlinked root marker was claimed"; FAIL=$((FAIL+1))
else
    echo "  PASS: a symlinked root marker is refused"; PASS=$((PASS+1))
fi
if _is_studio_root "$_linkmark"; then
    echo "  FAIL: ... and at a custom root too"; FAIL=$((FAIL+1))
else
    echo "  PASS: and at a custom root too"; PASS=$((PASS+1))
fi
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
# Only the rollback name has a collision counter; .venv.invalid is written once per run.
check "an invalid-venv name with a rollback-style suffix" foreign managed \
    ".venv.invalid.20260908120000.4242.2/pyvenv.cfg"
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
# _claim_studio_root calls _claim_sentinel, so both come across or every claim silently takes
# the wrong branch on a command-not-found and the block below asserts nothing.
_fn=""
for _cname in _claim_sentinel _claim_studio_root; do
    _cfn=$(sed -n "/^$_cname() {/,/^}/p" "$INSTALL_SH")
    if [ -z "$_cfn" ]; then
        echo "  FAIL: could not extract $_cname from $INSTALL_SH"; FAIL=$((FAIL+1)); _fn=""; break
    fi
    _fn="$_fn
$_cfn"
done
if [ -z "$_fn" ]; then
    :
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
    # A file called bin/unsloth is any file called bin/unsloth. The uninstaller reads it only as
    # a symlink into the venv, and this list authorizes a delete, so it is not on it.
    claim_check "a workspace holding a plain bin/unsloth" left env "bin/unsloth" "notes.txt"

    # A link named like our marker: -f follows it, which would skip the emptiness test and get
    # the link replaced by a real marker on somebody's workspace.
    STUDIO_HOME="$_TMP_ROOT/claim_linked_marker"
    # shellcheck disable=SC2034  # read by the extracted _claim_studio_root
    VENV_DIR="$STUDIO_HOME/unsloth_studio"
    _STUDIO_HOME_REDIRECT="env"
    mkdir -p "$STUDIO_HOME"
    : > "$_TMP_ROOT/claim_linked_marker_target"
    : > "$STUDIO_HOME/notes.txt"
    ln -s "$_TMP_ROOT/claim_linked_marker_target" "$STUDIO_HOME/.unsloth-studio-owned"
    _claim_studio_root
    if [ -L "$STUDIO_HOME/.unsloth-studio-owned" ]; then
        echo "  PASS: a linked marker is not read as proof, and is left as it was"; PASS=$((PASS+1))
    else
        echo "  FAIL: a linked marker was replaced with a real one"; FAIL=$((FAIL+1))
    fi

    # sh -f: the globs are the whole emptiness test, so without expansion every workspace would
    # read as empty. Driven with globbing off, which is what `sh -f install.sh` gives.
    STUDIO_HOME="$_TMP_ROOT/claim_noglob"
    VENV_DIR="$STUDIO_HOME/unsloth_studio"
    _STUDIO_HOME_REDIRECT="env"
    mkdir -p "$STUDIO_HOME"
    : > "$STUDIO_HOME/notes.txt"
    set -f
    _claim_studio_root
    set +f
    if [ -f "$STUDIO_HOME/.unsloth-studio-owned" ]; then
        echo "  FAIL: with globbing off, a workspace read as empty and was claimed"; FAIL=$((FAIL+1))
    else
        echo "  PASS: globbing off does not make a workspace look empty"; PASS=$((PASS+1))
    fi
    case $- in *f*) echo "  FAIL: the scan left globbing disabled"; FAIL=$((FAIL+1)) ;;
                *) echo "  PASS: and the caller's globbing setting is restored"; PASS=$((PASS+1)) ;;
    esac

    # Called twice per install: the second call must not unlink a marker the first one wrote.
    STUDIO_HOME="$_TMP_ROOT/claim_twice"
    VENV_DIR="$STUDIO_HOME/unsloth_studio"
    _STUDIO_HOME_REDIRECT="default"
    mkdir -p "$STUDIO_HOME"
    _claim_studio_root
    printf 'first' > "$STUDIO_HOME/.unsloth-studio-owned"
    _claim_studio_root
    if [ "$(cat "$STUDIO_HOME/.unsloth-studio-owned")" = "first" ]; then
        echo "  PASS: a second claim leaves a valid marker alone"; PASS=$((PASS+1))
    else
        echo "  FAIL: a second claim rewrote a valid marker"; FAIL=$((FAIL+1))
    fi

    # A root we cannot write, holding a link to a target we can: rm fails, and writing anyway
    # would truncate the target. No marker is the correct outcome, not a truncated file.
    STUDIO_HOME="$_TMP_ROOT/claim_ro_root"
    # shellcheck disable=SC2034  # read by the extracted _claim_studio_root
    VENV_DIR="$STUDIO_HOME/unsloth_studio"
    _STUDIO_HOME_REDIRECT="default"
    mkdir -p "$STUDIO_HOME"
    printf 'precious' > "$_TMP_ROOT/claim_ro_target"
    ln -s "$_TMP_ROOT/claim_ro_target" "$STUDIO_HOME/.unsloth-studio-owned"
    chmod 500 "$STUDIO_HOME"
    _claim_studio_root
    chmod 700 "$STUDIO_HOME"
    if [ "$(cat "$_TMP_ROOT/claim_ro_target")" = "precious" ]; then
        echo "  PASS: an unremovable link is not written through"; PASS=$((PASS+1))
    else
        echo "  FAIL: an unremovable link had its target truncated"; FAIL=$((FAIL+1))
    fi

    # A symlink at the marker path: the redirection would follow it and truncate the target.
    STUDIO_HOME="$_TMP_ROOT/claim_symlink"
    # shellcheck disable=SC2034  # read by the extracted _claim_studio_root
    VENV_DIR="$STUDIO_HOME/unsloth_studio"
    _STUDIO_HOME_REDIRECT="default"
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

# The claim's other consequence: a first install that dies after creating unsloth_studio leaves
# an occupied venv with no in-venv marker, and the retry guard would refuse to replace it unless
# it reads the root marker the same run wrote. Structural, because the guard is a condition in
# the middle of the install and running one here is not practical.
_guard=$(sed -n '/why: matching guard to the .venv branch below/,/Move it aside or choose an empty/p' "$INSTALL_SH")
# Built, not written literally, so the dollar stays a character rather than reading as an
# expansion here: the guard's SOURCE text is what is being searched.
_guard_pat="[\$]STUDIO_HOME/[.]unsloth-studio-owned"
if [ -z "$_guard" ]; then
    echo "  FAIL: could not find install.sh's env-mode replacement guard"; FAIL=$((FAIL+1))
elif printf '%s' "$_guard" | grep -q "$_guard_pat"; then
    echo "  PASS: the retry guard reads the root marker"; PASS=$((PASS+1))
else
    echo "  FAIL: the retry guard ignores the root marker install.sh just wrote"; FAIL=$((FAIL+1))
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
