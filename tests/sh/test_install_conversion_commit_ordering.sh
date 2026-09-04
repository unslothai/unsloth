#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test for the ordering of everything a portable conversion publishes or commits.
#
# The rollback machinery around --portable has one invariant: an artifact is snapshotted before
# it is written, restored by both handlers, and released at the point past which its loss is
# worse than its retention. Three separate ways of breaking it have shipped, and all three are
# the same mistake -- state that goes out, or stays out, on the wrong side of a boundary:
#
#   1  the marker at <root>/.unsloth-portable-root was published, and then the master root
#      record failed, with the EXIT trap not yet armed. The run exited having installed
#      nothing and left the tree reading as portable.
#   2  the venv was committed at the setup gate while the marker waited three hundred lines,
#      so a fatal shim step in between kept a built portable environment and deleted the
#      marker that makes it portable -- or, converting the other way, kept a normal venv and
#      restored the portable markers and launcher in front of it.
#   3  create_studio_shortcuts rewrote share/studio.conf and share/launch-studio.sh BEFORE the
#      gate that reports setup.sh's failure, and nothing put them back, so a failed flat
#      conversion left the restored install exporting UNSLOTH_PORTABLE=1 through its launcher.
#
# Driven by extracting the real blocks from install.sh and running them against fixtures.
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTALL="$ROOT/install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi
}

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/home"

blk() { awk "$1" "$INSTALL"; }
blockT="$(grep '^_trim_ws() ' "$INSTALL")"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockM="$(blk '/^_PORTABLE_MARKER_PATH_1=""$/ {grab=1} grab {print} /^_PORTABLE_SHIM_BACKUP=""$/ {exit}')"
blockE="$(blk '/^_export_portable_roots\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockR="$(blk '/^_VENV_ROLLBACK_DIR=""$/ {grab=1} grab {print} /^trap ._on_install_signal 143. TERM$/ {exit}')"
TAIL="$(sed -n '/^if \[ "\$_SETUP_EXIT" -eq 0 \]; then$/,/^_commit_portable_marker$/p' "$INSTALL")"

# Self-validate every extraction, or the assertions below are about "".
case "$blockT" in *_trim_ws*) : ;; *) echo "FAIL: blockT extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockM" in *'_PORTABLE_CONF_BACKUP'*) : ;; *) echo "FAIL: blockM lost the launcher slots"; exit 1 ;; esac
case "$blockE" in *'.unsloth-master-root'*) : ;; *) echo "FAIL: blockE extraction broke"; exit 1 ;; esac
case "$blockR" in *'_snapshot_portable_launcher() {'*) : ;; *) echo "FAIL: blockR lost the launcher snapshot"; exit 1 ;; esac
case "$blockR" in *'_restore_portable_launcher() {'*) : ;; *) echo "FAIL: blockR lost the launcher restore"; exit 1 ;; esac
case "$TAIL" in *'_snapshot_portable_launcher'*) : ;; *) echo "FAIL: the tail does not snapshot the launcher"; exit 1 ;; esac
case "$TAIL" in *'_commit_portable_marker'*) : ;; *) echo "FAIL: TAIL extraction broke"; exit 1 ;; esac

echo "[1] nothing is published or retired before the handlers that unwind it are armed"
_trap_at=$(grep -n '^trap _on_install_exit EXIT$' "$INSTALL" | head -n1 | cut -d: -f1)
_export_at=$(grep -n '^_export_portable_roots$' "$INSTALL" | head -n1 | cut -d: -f1)
_clear_at=$(grep -n '^_clear_stale_portable_marker$' "$INSTALL" | head -n1 | cut -d: -f1)
check "1 the EXIT trap is armed somewhere" yes "$([ -n "$_trap_at" ] && echo yes || echo no)"
check "1 the portable publish runs after it" yes \
    "$([ -n "$_export_at" ] && [ "$_trap_at" -lt "$_export_at" ] && echo yes || echo no)"
check "1 the stale-marker reset runs after it too" yes \
    "$([ -n "$_clear_at" ] && [ "$_trap_at" -lt "$_clear_at" ] && echo yes || echo no)"
# Both still have to precede the uv bootstrap, or the cache roots are exported after the
# caches they are meant to move have already been filled.
_uv_at=$(grep -n 'astral.sh/uv' "$INSTALL" | head -n1 | cut -d: -f1)
check "1 the uv bootstrap is where we think it is" yes "$([ -n "$_uv_at" ] && echo yes || echo no)"
check "1 and both still precede it" yes \
    "$([ "$_export_at" -lt "$_uv_at" ] && [ "$_clear_at" -lt "$_uv_at" ] && echo yes || echo no)"

echo "[2] a failed master root record does not leave the portable marker behind"
# blockE run on its own, with no traps at all: the publish has to unwind its own window, so
# that this holds however the call site is ordered.
SNIP='
set -e
'"$blockT"'
'"$blockB"'
'"$blockM"'
'"$blockE"'
substep() { :; }
_PORTABLE_MODE=true
_PORTABLE_FLAT=false
_UNSLOTH_ROOT="$_FIXTURE_ROOT"
_resolve_studio_destinations
# From a trap, not after the call: _export_portable_roots exits on these fixtures, so a
# straight-line printf here never runs and the assertion below would pass on empty output.
_report_slot() { printf "PUBLISHED %s\n" "$_PORTABLE_MARKER_PATH_1"; }
trap _report_slot EXIT
_export_portable_roots
'
publish() { # root
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" _FIXTURE_ROOT="$1" \
        sh -c "$SNIP" _ > "$T/out" 2> "$T/err" && printf '0\n' || printf '%s\n' "$?"
}
R2="$(mktemp -d "$T/root.XXXXXX")"
mkdir -p "$R2/studio/.unsloth-master-root/leftover"
check "2 the install fails" 1 "$(publish "$R2")"
check "2 the record error is reported" yes \
    "$(grep -qF "could not write the master root record" "$T/err" && echo yes || echo no)"
check "2 no portable marker is left at the root" no \
    "$([ -f "$R2/.unsloth-portable-root" ] && echo yes || echo no)"
check "2 the marker slot is released" "PUBLISHED " "$(grep '^PUBLISHED' "$T/out" || printf 'PUBLISHED \n')"
check "2 the directory in the way is untouched" yes \
    "$([ -d "$R2/studio/.unsloth-master-root/leftover" ] && echo yes || echo no)"

echo "[3] ...and a marker that was already there comes back byte for byte"
R3="$(mktemp -d "$T/root.XXXXXX")"
mkdir -p "$R3/studio/.unsloth-master-root"
printf '/somewhere/else\n' > "$R3/.unsloth-portable-root"
check "3 the install fails" 1 "$(publish "$R3")"
check "3 the previous marker is restored" "/somewhere/else" \
    "$(cat "$R3/.unsloth-portable-root" 2>/dev/null || echo MISSING)"

echo "[4] the venv and the portable identity commit together"
_gate_at=$(grep -n '^if \[ "\$_SETUP_EXIT" -eq 0 \]; then$' "$INSTALL" | head -n1 | cut -d: -f1)
_venv_commit_at=$(grep -n '^    _commit_studio_venv_replacement$' "$INSTALL" | head -n1 | cut -d: -f1)
# the call site inside the gate, not the one inside _commit_portable_marker far above it
_marker_commit_at=$(awk -v g="$_gate_at" 'NR>g && $0=="    _commit_portable_marker identity" {print NR; exit}' "$INSTALL")
_shim_at=$(grep -n '^mkdir -p "\$_LOCAL_BIN"$' "$INSTALL" | head -n1 | cut -d: -f1)
check "4 the portable identity commits inside the setup-succeeded gate" yes \
    "$([ -n "$_marker_commit_at" ] && [ "$_gate_at" -lt "$_marker_commit_at" ] \
       && [ "$_marker_commit_at" -lt "$_shim_at" ] && echo yes || echo no)"
check "4 immediately after the venv commit, nothing fatal between them" yes \
    "$([ -n "$_venv_commit_at" ] && [ "$_venv_commit_at" -lt "$_marker_commit_at" ] \
       && [ "$((_marker_commit_at - _venv_commit_at))" -le 30 ] && echo yes || echo no)"

# Behaviourally: setup.sh succeeded, the new portable venv is on disk, and the shim step is
# fatal because <root>/bin/unsloth is a directory. The venv is permanent by then, so the
# marker has to be too.
write_tail() { # dir, extra lines file
    {
        printf '%s\n' 'set -e'
        printf '%s\n' 'substep() { :; }'
        printf '%s\n' 'rollback_substep() { printf "ROLLBACK %s\n" "$1"; }'
        printf '%s\n' 'step() { :; }'
        printf '%s\n' 'tauri_log() { :; }'
        printf '%s\n' 'tauri_clear_install_error() { :; }'
        printf '%s\n' 'TAURI_MODE=false; OS=linux; C_WARN=""; C_ERR=""'
        printf '%s\n' 'SHELL=/bin/sh; export SHELL'
        printf '%s\n' 'unset ZDOTDIR ZSH_VERSION UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL'
        printf '%s\n' '_UNSLOTH_LOGIN_PATH="/usr/bin:/bin"'
        printf '%s\n' '_UNSLOTH_UV_BIN_DIR=""'
        printf '%s\n' '_STUDIO_HOME_REDIRECT=env'
        printf '%s\n' "$blockM"
        printf '%s\n' "$blockR"
        cat "$2"
        printf '%s\n' "$TAIL"
        printf '%s\n' 'echo REACHED-END'
    } > "$1/harness.sh"
}

B="$T/itemB"; mkdir -p "$B/studio" "$B/bin/unsloth/occupied" "$B/home"
cat > "$T/itemB.pre" <<PRE
STUDIO_HOME='$B/studio'
UNSLOTH_ROOT='$B'
DATA_DIR='$B/share'
VENV_DIR='$B/studio/unsloth_studio'
_LOCAL_BIN='$B/bin'
HOME='$B/home'; export HOME
_PORTABLE_MODE=true
mkdir -p "\$VENV_DIR/bin"
printf 'new\n' > "\$VENV_DIR/generation"
printf '#!/bin/sh\n' > "\$VENV_DIR/bin/unsloth"; chmod +x "\$VENV_DIR/bin/unsloth"
VENV_ABS_BIN="\$VENV_DIR/bin"
mkdir -p "\$STUDIO_HOME/unsloth_studio.old"
_VENV_ROLLBACK_DIR="\$STUDIO_HOME/unsloth_studio.old"
_VENV_ROLLBACK_TARGET="\$VENV_DIR"
_VENV_ROLLBACK_ACTIVE=true
printf '%s\n' "\$UNSLOTH_ROOT" > "\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PATH_1="\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PRIOR_1=n
create_studio_shortcuts() { return 0; }
_SETUP_EXIT=0
PRE
write_tail "$B" "$T/itemB.pre"
set +e
sh "$B/harness.sh" > "$B/out" 2>"$B/err"; _rcB=$?
set -e
check "4 the fatal shim step still fails the install" 1 "$_rcB"
check "4 the new environment stays" new "$(cat "$B/studio/unsloth_studio/generation" 2>/dev/null || echo MISSING)"
check "4 and so does the marker that makes it portable" yes \
    "$([ -f "$B/.unsloth-portable-root" ] && echo yes || echo no)"

echo "[5] a failed conversion hands back the launcher it rewrote"
# create_studio_shortcuts is stubbed to do the one thing this is about: overwrite studio.conf
# and launch-studio.sh the way a --portable run does. setup.sh has failed, so the whole
# conversion must be off the disk by the time the installer returns.
C="$T/itemC"; mkdir -p "$C/share" "$C/bin" "$C/unsloth_studio/bin" "$C/home"
printf '#!/bin/sh\n' > "$C/unsloth_studio/bin/unsloth"; chmod +x "$C/unsloth_studio/bin/unsloth"
printf "UNSLOTH_EXE='%s'\n" "$C/unsloth_studio/bin/unsloth" > "$C/share/studio.conf"
printf '#!/usr/bin/env bash\n# the launcher the normal install wrote\n' > "$C/share/launch-studio.sh"
chmod +x "$C/share/launch-studio.sh"
ln -sfn "$C/unsloth_studio/bin/unsloth" "$C/bin/unsloth"
cat > "$T/itemC.pre" <<PRE
STUDIO_HOME='$C'
UNSLOTH_ROOT='$C'
DATA_DIR='$C/share'
VENV_DIR='$C/unsloth_studio'
_LOCAL_BIN='$C/bin'
HOME='$C/home'; export HOME
_PORTABLE_MODE=true
_PORTABLE_FLAT=true
VENV_ABS_BIN="\$VENV_DIR/bin"
printf '%s\n' "\$UNSLOTH_ROOT" > "\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PATH_1="\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PRIOR_1=n
create_studio_shortcuts() {
    {
        printf "UNSLOTH_EXE='%s'\n" "\$VENV_DIR/bin/unsloth"
        printf '%s\n' "export UNSLOTH_HOME='\$UNSLOTH_ROOT'"
        printf '%s\n' "export UNSLOTH_PORTABLE=1"
    } > "\$DATA_DIR/studio.conf"
    printf '#!/usr/bin/env bash\n# regenerated by the conversion\n' > "\$DATA_DIR/launch-studio.sh"
    return 0
}
_SETUP_EXIT=3
PRE
write_tail "$C" "$T/itemC.pre"
set +e
sh "$C/harness.sh" > "$C/out" 2>"$C/err"; _rcC=$?
set -e
check "5 the conversion reports setup.sh's failure" 3 "$_rcC"
check "5 studio.conf is back to the normal install's" no \
    "$(grep -q '^export UNSLOTH_PORTABLE=1$' "$C/share/studio.conf" && echo yes || echo no)"
check "5 with the exe line it had" yes \
    "$(grep -qF "UNSLOTH_EXE='$C/unsloth_studio/bin/unsloth'" "$C/share/studio.conf" && echo yes || echo no)"
check "5 launch-studio.sh is back too" yes \
    "$(grep -qF 'the launcher the normal install wrote' "$C/share/launch-studio.sh" && echo yes || echo no)"
check "5 it kept its executable bit" yes "$([ -x "$C/share/launch-studio.sh" ] && echo yes || echo no)"
check "5 and no copy is left behind" 0 \
    "$(find "$C/share" -maxdepth 1 \( -name '.unsloth-studio-conf.*' -o -name '.unsloth-launch-studio.*' \) | wc -l | tr -d ' ')"

echo "[6] the reverse conversion is covered by the same slot"
# portable -> normal over the same flat root: the rewrite STRIPS the portable exports, and a
# failed run has to put them back or the restored portable install stops containing itself.
V="$T/itemV"; mkdir -p "$V/share" "$V/bin" "$V/unsloth_studio/bin" "$V/home"
printf '#!/bin/sh\n' > "$V/unsloth_studio/bin/unsloth"; chmod +x "$V/unsloth_studio/bin/unsloth"
{
    printf "UNSLOTH_EXE='%s'\n" "$V/unsloth_studio/bin/unsloth"
    printf '%s\n' "export UNSLOTH_PORTABLE=1"
} > "$V/share/studio.conf"
printf '#!/usr/bin/env bash\n# the portable launcher\n' > "$V/share/launch-studio.sh"
chmod +x "$V/share/launch-studio.sh"
cat > "$T/itemV.pre" <<PRE
STUDIO_HOME='$V'
DATA_DIR='$V/share'
VENV_DIR='$V/unsloth_studio'
_LOCAL_BIN='$V/bin'
HOME='$V/home'; export HOME
_PORTABLE_MODE=false
VENV_ABS_BIN="\$VENV_DIR/bin"
_PORTABLE_MARKER_PATH_1='$V/.unsloth-portable-root'
_PORTABLE_MARKER_PRIOR_1='y$V'
create_studio_shortcuts() {
    printf "UNSLOTH_EXE='%s'\n" "\$VENV_DIR/bin/unsloth" > "\$DATA_DIR/studio.conf"
    printf '#!/usr/bin/env bash\n# regenerated by the reverse conversion\n' > "\$DATA_DIR/launch-studio.sh"
    return 0
}
_SETUP_EXIT=3
PRE
write_tail "$V" "$T/itemV.pre"
set +e
sh "$V/harness.sh" > "$V/out" 2>"$V/err"; _rcV=$?
set -e
check "6 the conversion reports the failure" 3 "$_rcV"
check "6 the portable exports come back" yes \
    "$(grep -q '^export UNSLOTH_PORTABLE=1$' "$V/share/studio.conf" && echo yes || echo no)"
check "6 the portable launcher comes back" yes \
    "$(grep -qF 'the portable launcher' "$V/share/launch-studio.sh" && echo yes || echo no)"
check "6 and no copy is left behind" 0 \
    "$(find "$V/share" -maxdepth 1 \( -name '.unsloth-studio-conf.*' -o -name '.unsloth-launch-studio.*' \) | wc -l | tr -d ' ')"

echo "[7] an ordinary install that converts nothing is not touched by any of this"
N="$T/itemN"; mkdir -p "$N/share" "$N/bin" "$N/unsloth_studio/bin" "$N/home"
printf '#!/bin/sh\n' > "$N/unsloth_studio/bin/unsloth"; chmod +x "$N/unsloth_studio/bin/unsloth"
printf 'stale\n' > "$N/share/studio.conf"
cat > "$T/itemN.pre" <<PRE
STUDIO_HOME='$N'
DATA_DIR='$N/share'
VENV_DIR='$N/unsloth_studio'
_LOCAL_BIN='$N/bin'
HOME='$N/home'; export HOME
_PORTABLE_MODE=false
VENV_ABS_BIN="\$VENV_DIR/bin"
create_studio_shortcuts() {
    printf 'rewritten\n' > "\$DATA_DIR/studio.conf"
    return 0
}
_SETUP_EXIT=3
PRE
write_tail "$N" "$T/itemN.pre"
set +e
sh "$N/harness.sh" > "$N/out" 2>"$N/err"; _rcN=$?
set -e
check "7 the failure is still reported" 3 "$_rcN"
check "7 no snapshot was taken" 0 \
    "$(find "$N/share" -maxdepth 1 -name '.unsloth-*' | wc -l | tr -d ' ')"
check "7 and the shortcuts written on a failed run are kept, as they always were" rewritten \
    "$(cat "$N/share/studio.conf")"

echo "[8] the ordering, exercised: a silent set -e abort in the publish window unwinds too"
# The two exits inside the publish are not the whole window. `_PRIOR="y$(cat -- "$path")"`
# takes its exit status from the substitution, so a record that exists and cannot be read --
# mode 640 in the shared root these markers are written for -- ends the install under set -e
# from a line that prints nothing. Whether that leaves a marker behind is decided purely by
# where the CALLS sit relative to the traps, so this runs install.sh's own region verbatim,
# from the publish definition down to the traps, and lets the file order itself.
SEQ="$(sed -n '/^_export_portable_roots() {$/,/^# ── Helper: download a URL/p' "$INSTALL" | sed '$d')"
_seq_calls=$(printf '%s\n' "$SEQ" | grep -c '^_export_portable_roots$\|^_clear_stale_portable_marker$' || true)
check "8 the region carries both call sites" 2 "$_seq_calls"
case "$SEQ" in *'trap _on_install_exit EXIT'*) : ;; *) echo "FAIL: SEQ extraction lost the traps"; exit 1 ;; esac

R8="$(mktemp -d "$T/root.XXXXXX")"
mkdir -p "$R8/studio"
printf '/an/older/root\n' > "$R8/studio/.unsloth-master-root"
chmod 000 "$R8/studio/.unsloth-master-root" 2>/dev/null || true
if cat -- "$R8/studio/.unsloth-master-root" >/dev/null 2>&1; then
    echo "  SKIP  8 this user can read a mode-000 file (running as root?); the abort cannot be staged"
else
    ORDER='
set -e
substep() { :; }
rollback_substep() { printf "ROLLBACK %s\n" "$1"; }
C_WARN=""
'"$blockT"'
'"$blockB"'
_PORTABLE_MODE=true
_PORTABLE_FLAT=false
_SHORTCUTS_ONLY=false
_UNSLOTH_ROOT="$_FIXTURE_ROOT"
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockM"'
'"$SEQ"'
echo REACHED-END
'
    set +e
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" _FIXTURE_ROOT="$R8" \
        sh -c "$ORDER" _ > "$T/out8" 2> "$T/err8"
    _rc8=$?
    set -e
    check "8 the unreadable record still ends the install" yes \
        "$([ "$_rc8" -ne 0 ] && echo yes || echo no)"
    check "8 and the marker it had already published is gone" no \
        "$([ -f "$R8/.unsloth-portable-root" ] && echo yes || echo no)"
fi
chmod 644 "$R8/studio/.unsloth-master-root" 2>/dev/null || true

echo "[9] a conversion that SUCCEEDS keeps the rewrite and leaves no copies behind"
S="$T/itemS"; mkdir -p "$S/share" "$S/bin" "$S/unsloth_studio/bin" "$S/home"
printf '#!/bin/sh\n' > "$S/unsloth_studio/bin/unsloth"; chmod +x "$S/unsloth_studio/bin/unsloth"
printf "UNSLOTH_EXE='old'\n" > "$S/share/studio.conf"
printf '#!/usr/bin/env bash\n# the old launcher\n' > "$S/share/launch-studio.sh"
chmod +x "$S/share/launch-studio.sh"
# a launcher already on the shim path, so the portable displacement actually arms and the
# "no copy left" assertion below has something it could catch
printf '#!/bin/sh\n# the flat install we are converting\n' > "$S/bin/unsloth"
chmod +x "$S/bin/unsloth"
cat > "$T/itemS.pre" <<PRE
STUDIO_HOME='$S'
UNSLOTH_ROOT='$S'
DATA_DIR='$S/share'
VENV_DIR='$S/unsloth_studio'
_LOCAL_BIN='$S/bin'
HOME='$S/home'; export HOME
_PORTABLE_MODE=true
_PORTABLE_FLAT=true
VENV_ABS_BIN="\$VENV_DIR/bin"
printf '%s\n' "\$UNSLOTH_ROOT" > "\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PATH_1="\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PRIOR_1=n
create_studio_shortcuts() {
    printf '%s\n' "export UNSLOTH_PORTABLE=1" > "\$DATA_DIR/studio.conf"
    printf '#!/usr/bin/env bash\n# the portable launcher\n' > "\$DATA_DIR/launch-studio.sh"
    return 0
}
_SETUP_EXIT=0
PRE
write_tail "$S" "$T/itemS.pre"
set +e
sh "$S/harness.sh" > "$S/out" 2>"$S/err"; _rcS=$?
set -e
check "9 the install succeeds" 0 "$_rcS"
check "9 it ran to the end" yes "$(grep -qx 'REACHED-END' "$S/out" && echo yes || echo no)"
check "9 the conversion's studio.conf stands" yes \
    "$(grep -q '^export UNSLOTH_PORTABLE=1$' "$S/share/studio.conf" && echo yes || echo no)"
# ...and nothing was snapshotted to begin with: the identity committed at the gate, so this
# rewrite describes the tree that now exists and has nothing to be handed back to.
check "9 the committed conversion snapshots nothing" yes \
    "$(grep -q 'ROLLBACK restored' "$S/out" && echo no || echo yes)"
check "9 the displaced launcher was replaced, not restored" yes \
    "$(grep -qF 'Generated by install.sh --portable' "$S/bin/unsloth" && echo yes || echo no)"
check "9 the marker stands" yes "$([ -f "$S/.unsloth-portable-root" ] && echo yes || echo no)"
check "9 and the copies are released, not left in \$DATA_DIR" 0 \
    "$(find "$S/share" -maxdepth 1 \( -name '.unsloth-studio-conf.*' -o -name '.unsloth-launch-studio.*' \) | wc -l | tr -d ' ')"
check "9 no shim copy is left either" 0 \
    "$(find "$S/bin" -maxdepth 1 \( -name '.unsloth-*' -o -name '.unsloth.shim.*' \) | wc -l | tr -d ' ')"

echo "[10] the same invariant converting the other way: portable -> normal"
# Nobody reported this half, and it is the worse one. setup.sh succeeded, so the new NORMAL
# venv is permanent; a fatal shim step afterwards used to restore the portable marker and the
# <root>/bin/unsloth wrapper this run had retired, in front of that venv, so the converted tree
# read as portable again through a launcher exporting UNSLOTH_PORTABLE=1. The marker and the
# retirement belong to the venv, so they commit with it.
P="$T/itemP"; mkdir -p "$P/studio" "$P/bin" "$P/home/.local/bin/unsloth/occupied"
printf '%s\n' "$P" > "$P/.unsloth-portable-root"
printf '#!/bin/sh\nexport UNSLOTH_PORTABLE=1\n' > "$P/bin/.unsloth-portable-shim.retired"
cat > "$T/itemP.pre" <<PRE
STUDIO_HOME='$P/studio'
DATA_DIR='$P/home/.local/share/unsloth'
VENV_DIR='$P/studio/unsloth_studio'
_LOCAL_BIN='$P/home/.local/bin'
HOME='$P/home'; export HOME
_PORTABLE_MODE=false
mkdir -p "\$VENV_DIR/bin" "\$DATA_DIR"
printf '#!/bin/sh\n' > "\$VENV_DIR/bin/unsloth"; chmod +x "\$VENV_DIR/bin/unsloth"
VENV_ABS_BIN="\$VENV_DIR/bin"
# what _clear_stale_portable_marker leaves armed after a portable -> normal conversion
rm -f '$P/.unsloth-portable-root'
_PORTABLE_MARKER_PATH_1='$P/.unsloth-portable-root'
_PORTABLE_MARKER_PRIOR_1='y$P'
_PORTABLE_SHIM_PATH='$P/bin/unsloth'
_PORTABLE_SHIM_BACKUP='$P/bin/.unsloth-portable-shim.retired'
create_studio_shortcuts() { return 0; }
_SETUP_EXIT=0
PRE
write_tail "$P" "$T/itemP.pre"
set +e
sh "$P/harness.sh" > "$P/out" 2>"$P/err"; _rcP=$?
set -e
check "10 the fatal shim step still fails the install" 1 "$_rcP"
check "10 the new normal environment stays" yes \
    "$([ -x "$P/studio/unsloth_studio/bin/unsloth" ] && echo yes || echo no)"
check "10 the portable marker stays removed" no \
    "$([ -f "$P/.unsloth-portable-root" ] && echo yes || echo no)"
# ...but the launcher is handed BACK, not kept retired. Its replacement is written at
# $_LOCAL_BIN, which is the step that just failed, so releasing this copy with the venv would
# answer a failed install by deleting the user's only launcher and putting nothing in its
# place. The identity commits with the venv; the launchers wait for the launcher that
# supersedes them.
check "10 the launcher it retired is handed back, not deleted" yes \
    "$([ -f "$P/bin/unsloth" ] && echo yes || echo no)"
check "10 and no copy is orphaned in <root>/bin" 0 \
    "$(find "$P/bin" -maxdepth 1 -name '.unsloth-portable-shim.*' | wc -l | tr -d ' ')"

echo "[11] a COMMITTED conversion whose wiring fails afterwards keeps its wiring"
# The trap that catches this is the same one, so the snapshot has to be gated on whether the
# conversion is still uncommitted rather than on portable mode: past the setup gate the venv
# and the marker stand, and handing studio.conf back to the install that is no longer there
# would leave the launcher naming an exe this run replaced.
W="$T/itemW"; mkdir -p "$W/share" "$W/bin" "$W/unsloth_studio/bin" "$W/home"
printf '#!/bin/sh\n' > "$W/unsloth_studio/bin/unsloth"; chmod +x "$W/unsloth_studio/bin/unsloth"
printf "UNSLOTH_EXE='the-old-normal-install'\n" > "$W/share/studio.conf"
# the fatal step has to be one that runs AFTER the rewrite, so it is create_studio_shortcuts
# itself failing partway: the exact case where the pre-conversion studio.conf used to come back
# in front of a venv and a marker that had already committed.
cat > "$T/itemW.pre" <<PRE
STUDIO_HOME='$W'
UNSLOTH_ROOT='$W'
DATA_DIR='$W/share'
VENV_DIR='$W/unsloth_studio'
_LOCAL_BIN='$W/bin'
HOME='$W/home'; export HOME
_PORTABLE_MODE=true
_PORTABLE_FLAT=true
VENV_ABS_BIN="\$VENV_DIR/bin"
printf '%s\n' "\$UNSLOTH_ROOT" > "\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PATH_1="\$UNSLOTH_ROOT/.unsloth-portable-root"
_PORTABLE_MARKER_PRIOR_1=n
create_studio_shortcuts() {
    printf "UNSLOTH_EXE='%s'\n" "\$VENV_DIR/bin/unsloth" > "\$DATA_DIR/studio.conf"
    return 1
}
_SETUP_EXIT=0
PRE
write_tail "$W" "$T/itemW.pre"
set +e
sh "$W/harness.sh" > "$W/out" 2>"$W/err"; _rcW=$?
set -e
check "11 the later step still fails the install" 1 "$_rcW"
check "11 the marker stands" yes "$([ -f "$W/.unsloth-portable-root" ] && echo yes || echo no)"
check "11 and studio.conf still names the environment that was built" yes \
    "$(grep -qF "UNSLOTH_EXE='$W/unsloth_studio/bin/unsloth'" "$W/share/studio.conf" && echo yes || echo no)"
check "11 not the install that is no longer there" no \
    "$(grep -qF 'the-old-normal-install' "$W/share/studio.conf" && echo yes || echo no)"
check "11 and nothing was copied aside" 0 \
    "$(find "$W/share" -maxdepth 1 \( -name '.unsloth-studio-conf.*' -o -name '.unsloth-launch-studio.*' \) | wc -l | tr -d ' ')"

echo ""
if [ "$fails" -eq 0 ]; then echo "  all checks passed"; else echo "  $fails failed"; fi
[ "$fails" -eq 0 ]
