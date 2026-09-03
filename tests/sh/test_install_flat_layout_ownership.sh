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
#
# The second round of the same bug: "the sentinel exists" is not "the sentinel is ours".
# `unsloth` is an ordinary word, so a reused DIR can hold the user's own bin/unsloth helper
# beside their own unsloth_studio virtualenv, and BOTH the selector and the guard used to
# accept that bare file. The install then took the flat layout, moved the unrelated
# environment aside as a rollback copy, and deleted that copy on success -- the user's venv
# destroyed and the nested install they asked for never created. So the two sentinels that
# live outside the venv (share/studio.conf, bin/unsloth) must now NAME the candidate venv,
# in the exact shape their writer emits, and the cases below feed real generated content
# rather than `: >` placeholders.
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
# That opener is unique only because of the line continuation: the one-line spelling of the
# same test appears twice in create_studio_shortcuts, EARLIER in the file, so a guard rewritten
# onto one line would make the range above extract an unrelated block and quietly hand every
# case below the wrong verdict.
[ "$(grep -c '^    if \[ "\$_STUDIO_HOME_REDIRECT" = "env" \] \\$' "$INSTALL")" -eq 1 ] \
    || { echo "FAIL: the ownership guard's opener is no longer a unique anchor"; exit 1; }
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
# Structural, on top of the behaviour below: both halves must still MATCH the recorded venv
# rather than test a path for existence. Dropping either one is the regression this file
# exists for, and a fixture set can always be weakened by accident.
for _half in RESOLVE GUARD; do
    eval "_hb=\$$_half"
    case "$_hb" in *"UNSLOTH_EXE='"*) : ;;
        *) echo "FAIL: $_half no longer matches studio.conf against the venv"; exit 1 ;; esac
    case "$_hb" in *"exec '"*) : ;;
        *) echo "FAIL: $_half no longer matches the generated shim against the venv"; exit 1 ;; esac
    case "$_hb" in *'-ef'*) : ;;
        *) echo "FAIL: $_half no longer resolves a bin/unsloth symlink to the venv"; exit 1 ;; esac
done

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

# The guard on its own, in plain env-mode (UNSLOTH_STUDIO_HOME=DIR, no --portable), where
# STUDIO_HOME is DIR and there is no layout question at all. Needed for non-vacuity: once the
# selector nests a suspicious root, VENV_DIR moves to <root>/studio/unsloth_studio, which is
# empty, so the occupancy test short-circuits and guard_verdict() below can no longer tell a
# hardened guard from the bare-existence one. This entry point reaches it directly, and it is
# also the shape the guard was originally written for.
guard_env_verdict() { # dir
    _ge_out=$(env -i HOME="$_TMP_ROOT/home" PATH="$PATH" USER="${USER:-tester}" _GE_DIR="$1" sh -c '
        set -e
        '"$DIRENT"'
        _STUDIO_HOME_REDIRECT=env
        STUDIO_HOME="$_GE_DIR"
        VENV_DIR="$STUDIO_HOME/unsloth_studio"
        '"$OUTER"'
        '"$INNER"'
        fi
        printf "ok\n"
    ' 2>/dev/null) || _ge_out=refused
    case "$_ge_out" in *ok*) printf 'ok\n' ;; *) printf 'refused\n' ;; esac
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

# ── The sentinels as their writers actually emit them ─────────────────────────────────────
# Byte-for-byte the lines install.sh produces, so "exists" and "is ours" can be told apart.
# gen_* take (root, venv) separately on purpose: pointing a sentinel at a venv OTHER than the
# candidate is the negative case, and it has to be expressible.
# Both writers single-quote the paths they record and escape `'` as `'\''` first; reproduce
# that here, or the apostrophe cases below would be testing the fixture instead of install.sh.
sq() { printf '%s' "$1" | sed "s/'/'\\\\''/g"; }
gen_conf() {  # root venv -- create_studio_shortcuts' share/studio.conf
    mkdir -p "$1/share"
    {
        printf "UNSLOTH_EXE='%s/bin/unsloth'\n" "$(sq "$2")"
        printf 'export UNSLOTH_PORTABLE=1\n'
    } > "$1/share/studio.conf"
}
gen_portable_shim() {  # root venv -- the generated --portable wrapper
    mkdir -p "$1/bin"
    {
        printf '%s\n' '#!/bin/sh'
        printf '%s\n' '# Generated by install.sh --portable. Keeps every Unsloth path inside'
        printf "export UNSLOTH_HOME='%s'\n" "$(sq "$1")"
        printf 'export UNSLOTH_PORTABLE=1\n'
        printf "exec '%s/bin/unsloth' \"\$@\"\n" "$(sq "$2")"
    } > "$1/bin/unsloth"
    chmod +x "$1/bin/unsloth"
}
gen_symlink_shim() {  # root venv -- the non-portable env-mode `ln -sfn` shape
    mkdir -p "$1/bin" "$2/bin"
    : > "$2/bin/unsloth"
    ln -sfn "$2/bin/unsloth" "$1/bin/unsloth"
}

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
mkdir -p "$R/studio/unsloth_studio" "$R/unsloth_studio"
: > "$R/studio/unsloth_studio/.unsloth-studio-owned"
gen_conf "$R" "$R/studio/unsloth_studio"
gen_portable_shim "$R" "$R/studio/unsloth_studio"
check "a stray dir cannot flatten an existing nested root" "false $R/studio" "$(layout "$R")"

# ── 1b. Negative: a sentinel of the right NAME that is not ours ────────────────────────────
# The reported case. `unsloth` is an ordinary word, so a reused --root can hold the user's own
# helper script called bin/unsloth beside their own virtualenv called unsloth_studio. Bare
# existence read that as a flat install of ours, the guard agreed, and the environment was
# moved aside and its rollback copy deleted once the install succeeded.
R="$(new_root)"; mkdir -p "$R/unsloth_studio/lib" "$R/bin"
: > "$R/unsloth_studio/pyvenv.cfg"
printf '#!/bin/sh\n# my own helper\necho hi\n' > "$R/bin/unsloth"; chmod +x "$R/bin/unsloth"
check "an unrelated bin/unsloth script does not select flat" "false $R/studio" "$(layout "$R")"
check "and the guard refuses to adopt the venv beside it"    ok "$(guard_verdict "$R")"

# Same shape via the other outside sentinel.
R="$(new_root)"; mkdir -p "$R/unsloth_studio/lib" "$R/share"
: > "$R/unsloth_studio/pyvenv.cfg"
printf 'some other tool wrote this\n' > "$R/share/studio.conf"
check "an unrelated share/studio.conf does not select flat" "false $R/studio" "$(layout "$R")"
check "and the guard refuses to adopt the venv beside it"   ok "$(guard_verdict "$R")"

# Our own generated shapes, but naming a DIFFERENT venv: evidence for some other install is
# not evidence for this one.
R="$(new_root)"; mkdir -p "$R/unsloth_studio/lib"; : > "$R/unsloth_studio/pyvenv.cfg"
gen_conf "$R" "$_TMP_ROOT/elsewhere/unsloth_studio"
gen_portable_shim "$R" "$_TMP_ROOT/elsewhere/unsloth_studio"
check "sentinels naming another venv do not select flat" "false $R/studio" "$(layout "$R")"
check "and the guard is not satisfied by them either"    ok "$(guard_verdict "$R")"

# A symlink at bin/unsloth that resolves somewhere else entirely.
R="$(new_root)"; mkdir -p "$R/unsloth_studio/lib" "$R/bin" "$_TMP_ROOT/other/bin"
: > "$R/unsloth_studio/pyvenv.cfg"; : > "$_TMP_ROOT/other/bin/unsloth"
ln -sfn "$_TMP_ROOT/other/bin/unsloth" "$R/bin/unsloth"
check "a bin/unsloth symlink to another tree does not select flat" "false $R/studio" "$(layout "$R")"
check "and the guard is not satisfied by it either"               ok "$(guard_verdict "$R")"

# ── 2. Positive: the ownership requirement must not collapse into "never flat" ─────────────
R="$(new_root)"
mkdir -p "$R/unsloth_studio/bin"
: > "$R/unsloth_studio/.unsloth-studio-owned"
gen_conf "$R" "$R/unsloth_studio"
gen_portable_shim "$R" "$R/unsloth_studio"
check "a complete flat install is still flat" "true $R" "$(layout "$R")"
check "and the ownership guard agrees it is ours" ok "$(guard_verdict "$R")"

# Each sentinel on its own, since a real install can be missing one: the in-venv owner marker
# is written best-effort, and share/studio.conf only appears once shortcuts are created.
# `shim` and `link` are the two shapes bin/unsloth really takes -- the generated --portable
# wrapper, and the `ln -sfn` a non-portable env-mode install leaves at the same path.
for _sent in owner conf shim link; do
    R="$(new_root)"; mkdir -p "$R/unsloth_studio/bin"
    case "$_sent" in
        owner) : > "$R/unsloth_studio/.unsloth-studio-owned" ;;
        conf)  gen_conf "$R" "$R/unsloth_studio" ;;
        shim)  gen_portable_shim "$R" "$R/unsloth_studio" ;;
        link)  gen_symlink_shim "$R" "$R/unsloth_studio" ;;
    esac
    check "the $_sent sentinel alone selects flat" "true $R" "$(layout "$R")"
    check "the $_sent sentinel alone also satisfies the guard" ok "$(guard_verdict "$R")"
done

# The writers single-quote every path they record and escape `'` as `'\''`; the readers above
# rebuild that same escaping. A root with an apostrophe in it is where the two would diverge,
# and divergence here means a genuine install stops being recognised as its own.
R="$(mktemp -d "$_TMP_ROOT/o'brien.XXXXXX")"; mkdir -p "$R/unsloth_studio/bin"
gen_conf "$R" "$R/unsloth_studio"
check "an apostrophe in the root keeps studio.conf readable" "true $R" "$(layout "$R")"
check "and the guard still recognises it"                    ok "$(guard_verdict "$R")"
R="$(mktemp -d "$_TMP_ROOT/o'brien.XXXXXX")"; mkdir -p "$R/unsloth_studio/bin"
gen_portable_shim "$R" "$R/unsloth_studio"
check "an apostrophe in the root keeps the shim readable" "true $R" "$(layout "$R")"
check "and the guard still recognises it"                 ok "$(guard_verdict "$R")"

# ── 2b. The guard on its own, in plain env-mode ───────────────────────────────────────────
# UNSLOTH_STUDIO_HOME=DIR with no --portable: DIR is the Studio root outright, so the guard is
# the ONLY thing standing between an unrelated DIR/unsloth_studio and `mv` + `rm -rf`. Every
# case here is an occupied venv, so the guard always runs.
G="$(new_root)"; mkdir -p "$G/unsloth_studio/lib" "$G/bin"
: > "$G/unsloth_studio/pyvenv.cfg"
printf '#!/bin/sh\necho my own helper\n' > "$G/bin/unsloth"
check "env-mode: an unrelated bin/unsloth does not make the venv ours" refused "$(guard_env_verdict "$G")"

G="$(new_root)"; mkdir -p "$G/unsloth_studio/lib" "$G/share"
: > "$G/unsloth_studio/pyvenv.cfg"
printf 'written by something else\n' > "$G/share/studio.conf"
check "env-mode: an unrelated share/studio.conf does not either" refused "$(guard_env_verdict "$G")"

G="$(new_root)"; mkdir -p "$G/unsloth_studio/lib"; : > "$G/unsloth_studio/pyvenv.cfg"
gen_conf "$G" "$_TMP_ROOT/elsewhere/unsloth_studio"
gen_portable_shim "$G" "$_TMP_ROOT/elsewhere/unsloth_studio"
check "env-mode: sentinels naming another venv do not either" refused "$(guard_env_verdict "$G")"

# ...and the four things that DO make it ours, so the guard cannot collapse into refusing
# every reinstall.
for _sent in owner conf shim link; do
    G="$(new_root)"; mkdir -p "$G/unsloth_studio/lib"; : > "$G/unsloth_studio/pyvenv.cfg"
    case "$_sent" in
        owner) : > "$G/unsloth_studio/.unsloth-studio-owned" ;;
        conf)  gen_conf "$G" "$G/unsloth_studio" ;;
        shim)  gen_portable_shim "$G" "$G/unsloth_studio" ;;
        link)  gen_symlink_shim "$G" "$G/unsloth_studio" ;;
    esac
    check "env-mode: the $_sent sentinel lets the reinstall proceed" ok "$(guard_env_verdict "$G")"
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
