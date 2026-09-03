#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test: install.sh must not switch `--root DIR` from the documented nested layout
# to the flat one just because SOMETHING called DIR/unsloth_studio is on disk.
#
# --root accepts any writable directory, and _resolve_studio_destinations used a bare
# `[ -d "$root/unsloth_studio" ]` to decide the layout. Two ways that goes wrong on a shared or
# reused directory:
#   * an EMPTY leftover of that name made STUDIO_HOME the root itself, so Studio state (venv,
#     cache, assets, studio.db) landed directly in DIR instead of DIR/studio;
#   * a POPULATED unrelated one -- somebody's dev venv called unsloth_studio -- did the same,
#     and the venv-replacement ownership guard further down then refused the whole install,
#     costing a nested install that would have been perfectly valid at DIR/studio.
#
# The layout is now chosen from the same three sentinels that guard accepts, which gives the
# invariant this file pins behaviourally: whenever the flat layout is selected, the ownership
# guard passes. Both halves are driven by extracting the REAL blocks from install.sh and
# running them against per-case fixtures, so a rewrite that drops the rule fails here.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

check() { # label expected actual
    if [ "$2" = "$3" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (expected [$2], got [$3])"; FAIL=$((FAIL + 1))
    fi
}

# ── The real blocks, lifted out of install.sh ─────────────────────────────────────────────
blk() { awk "$1" "$INSTALL"; }
TRIM="$(grep '^_trim_ws() ' "$INSTALL")"
RESOLVE="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
# The venv-replacement ownership guard. Two real pieces of install.sh, composed here the way
# the script nests them: the OUTER "is there already something at VENV_DIR" test, and the inner
# ownership `if` it wraps. Both extracted, never retyped -- the inner block on its own refuses
# every empty VENV_DIR and would make each verdict below meaningless.
DIRENT="$(blk '/^_dir_has_entries\(\) \{/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
OUTER="$(grep -F 'if [ -x "$VENV_DIR/bin/python" ] || _dir_has_entries "$VENV_DIR"; then' "$INSTALL")"
INNER="$(sed -n '/^    if \[ "\$_STUDIO_HOME_REDIRECT" = "env" \] \\$/,/^    fi$/p' "$INSTALL")"
GUARD="$OUTER
$INNER
fi"

# A silently empty or truncated extraction is how this kind of test goes vacuous.
[ -n "$TRIM" ]    || { echo "FAIL: extracted an empty _trim_ws"; exit 1; }
[ -n "$RESOLVE" ] || { echo "FAIL: extracted an empty _resolve_studio_destinations"; exit 1; }
[ -n "$DIRENT" ]  || { echo "FAIL: extracted an empty _dir_has_entries"; exit 1; }
[ -n "$OUTER" ]   || { echo "FAIL: extracted an empty occupied-venv test"; exit 1; }
[ -n "$INNER" ]   || { echo "FAIL: extracted an empty ownership guard"; exit 1; }
case "$RESOLVE" in *'_PORTABLE_FLAT=true'*) : ;; *) echo "FAIL: the resolver lost its flat-layout branch"; exit 1 ;; esac
case "$GUARD" in *'does not look like an Unsloth Studio install'*) : ;;
    *) echo "FAIL: the extracted guard is not the ownership guard"; exit 1 ;; esac
case "$GUARD" in *'exit 1'*) : ;; *) echo "FAIL: the extracted guard cannot refuse anything"; exit 1 ;; esac

# Print "<flat> <studio_home>" for a --root install pointed at $1.
layout() { # root
    env -i HOME="$_TMP_ROOT/home" PATH="$PATH" USER="${USER:-tester}" _RSD_ROOT="$1" sh -c '
        set -e
        '"$TRIM"'
        '"$RESOLVE"'
        substep() { :; }
        _PORTABLE_MODE=true
        _PORTABLE_FLAT=false
        _UNSLOTH_ROOT="$_RSD_ROOT"
        _resolve_studio_destinations
        printf "%s %s\n" "$_PORTABLE_FLAT" "$STUDIO_HOME"
    '
}

# "ok" when the ownership guard would let this layout proceed, "refused" when it exits 1.
guard_verdict() { # root
    _gv_out=$(env -i HOME="$_TMP_ROOT/home" PATH="$PATH" USER="${USER:-tester}" _RSD_ROOT="$1" sh -c '
        set -e
        '"$TRIM"'
        '"$RESOLVE"'
        '"$DIRENT"'
        substep() { :; }
        _PORTABLE_MODE=true
        _PORTABLE_FLAT=false
        _UNSLOTH_ROOT="$_RSD_ROOT"
        _resolve_studio_destinations
        VENV_DIR="$STUDIO_HOME/unsloth_studio"
        '"$GUARD"'
        printf "ok\n"
    ' 2>/dev/null) || _gv_out=refused
    case "$_gv_out" in *ok*) printf 'ok\n' ;; *) printf 'refused\n' ;; esac
}

mkdir -p "$_TMP_ROOT/home"
new_root() { mktemp -d "$_TMP_ROOT/root.XXXXXX"; }

# ── 1. Negative: nothing about the directory says Unsloth ─────────────────────────────────
R="$(new_root)"
check "an empty --root nests" "false $R/studio" "$(layout "$R")"

R="$(new_root)"; mkdir -p "$R/unsloth_studio"
check "an EMPTY leftover named unsloth_studio does not select flat" "false $R/studio" "$(layout "$R")"
check "and Studio state stays out of the shared directory" ok "$(guard_verdict "$R")"

# Somebody's own virtualenv that happens to be called unsloth_studio. This is the case that
# used to select flat and then be refused outright by the ownership guard.
R="$(new_root)"; mkdir -p "$R/unsloth_studio/bin" "$R/unsloth_studio/lib"
: > "$R/unsloth_studio/pyvenv.cfg"; : > "$R/unsloth_studio/bin/python"
check "an unrelated dev venv named unsloth_studio does not select flat" "false $R/studio" "$(layout "$R")"
check "so the nested install it asked for is no longer refused" ok "$(guard_verdict "$R")"

# A real NESTED install with a stray directory of that name beside its studio/ child. Reading
# it as flat would relocate the existing install.
R="$(new_root)"
mkdir -p "$R/studio/unsloth_studio" "$R/unsloth_studio" "$R/share" "$R/bin"
: > "$R/studio/unsloth_studio/.unsloth-studio-owned"
: > "$R/share/studio.conf"
: > "$R/bin/unsloth"
check "a stray dir cannot flatten an existing nested root" "false $R/studio" "$(layout "$R")"

# ── 2. Positive: the ownership requirement must not collapse into "never flat" ─────────────
R="$(new_root)"
mkdir -p "$R/unsloth_studio/bin" "$R/share" "$R/bin"
: > "$R/unsloth_studio/.unsloth-studio-owned"
: > "$R/share/studio.conf"
: > "$R/bin/unsloth"
check "a complete flat install is still flat" "true $R" "$(layout "$R")"
check "and the ownership guard agrees it is ours" ok "$(guard_verdict "$R")"

# Each sentinel on its own, since a real install can be missing one: the in-venv owner marker
# is written best-effort, and share/studio.conf only appears once shortcuts are created.
for _sent in owner conf shim; do
    R="$(new_root)"; mkdir -p "$R/unsloth_studio/bin"
    case "$_sent" in
        owner) : > "$R/unsloth_studio/.unsloth-studio-owned" ;;
        conf)  mkdir -p "$R/share"; : > "$R/share/studio.conf" ;;
        shim)  mkdir -p "$R/bin";   : > "$R/bin/unsloth" ;;
    esac
    check "the $_sent sentinel alone selects flat" "true $R" "$(layout "$R")"
    check "the $_sent sentinel alone also satisfies the guard" ok "$(guard_verdict "$R")"
done

# ── 3. The invariant, stated as a pair: selecting flat implies the guard passes ────────────
# A populated flat venv WITHOUT any sentinel is the one shape that must be nested-and-allowed
# rather than flat-and-refused; a bare `-d` rule gives the opposite on exactly this fixture.
R="$(new_root)"; mkdir -p "$R/unsloth_studio/lib"; : > "$R/unsloth_studio/marker"
check "an unowned populated venv nests"       "false $R/studio" "$(layout "$R")"
check "an unowned populated venv is accepted" ok                "$(guard_verdict "$R")"

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
