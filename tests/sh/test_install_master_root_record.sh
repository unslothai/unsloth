#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test: a NESTED portable install records its master root INSIDE the Studio root.
#
# .unsloth-portable-root sits at $UNSLOTH_ROOT, one level ABOVE the tree the install owns, and
# honouring it hands that directory to the backend as UNSLOTH_HOME -- where the managed
# llama.cpp, node and whisper.cpp are resolved from and then executed. So every reader first
# proves our installer wrote it (storage_roots._parent_marker_is_trustworthy). Under the
# `umask 002` that is standard on multi-user boxes and CI images the selected root is
# group-writable, that proof fails for an install that SUCCEEDED, and the
# `source <root>/studio/unsloth_studio/bin/activate` path the summary prints resolves back to
# $HOME/.unsloth. .unsloth-master-root is the same association written where only the operator
# can write it, so it needs no such proof.
#
# Driven by extracting the REAL blocks from install.sh -- the slot declarations, the publish,
# the reset, and the whole venv-rollback + trap block -- and running them against fixtures. The
# last section runs the REAL storage_roots resolver over the tree those blocks produced, so
# this fails if either half stops agreeing.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
BACKEND="$ROOT/studio/backend"
RECORD=".unsloth-master-root"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blk() { awk "$1" "$INSTALL"; }
blockA="$(blk '/^# ── Parse flags ──$/ {grab=1} grab {print} /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen=1} seen && /^fi$/ {exit}')"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockM="$(blk '/^_PORTABLE_MARKER_PATH_1=""$/ {grab=1} grab {print} /^_PORTABLE_SHIM_BACKUP=""$/ {exit}')"
blockE="$(blk '/^_export_portable_roots\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockD="$(blk '/^_clear_stale_portable_marker\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockR="$(blk '/^_VENV_ROLLBACK_DIR=""$/ {grab=1} grab {print} /^trap ._on_install_signal 143. TERM$/ {exit}')"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockM" in *'_PORTABLE_MARKER_PRIOR_3'*) : ;; *) echo "FAIL: blockM lost the record slot"; exit 1 ;; esac
# blockM is delimited by the LAST slot declaration; a new slot appended after it would make
# every blockM in tests/sh swallow the rest of install.sh instead.
case "$blockM" in *_export_portable_roots*) echo "FAIL: blockM ran past the slot declarations"; exit 1 ;; *) : ;; esac
case "$blockE" in *"$RECORD"*) : ;; *) echo "FAIL: the publish no longer writes the record"; exit 1 ;; esac
case "$blockE" in *'_PORTABLE_MARKER_PRIOR_3'*) : ;; *) echo "FAIL: blockE stopped recording the publish"; exit 1 ;; esac
case "$blockD" in *"$RECORD"*) : ;; *) echo "FAIL: the reset no longer clears the record"; exit 1 ;; esac
case "$blockD" in *'_PORTABLE_MARKER_PRIOR_3'*) : ;; *) echo "FAIL: blockD stopped recording the reset"; exit 1 ;; esac
case "$blockR" in *'_PORTABLE_MARKER_PATH_3'*) : ;; *) echo "FAIL: the record slot never rolls back"; exit 1 ;; esac
# Several tests lift these blocks out and source them standalone, where a helper defined
# elsewhere dies as "command not found" inside a condition and the guard goes silently inert.
case "$blockE" in *'_portable_record'*) echo "FAIL: blockE calls a helper it does not define"; exit 1 ;; *) : ;; esac
case "$blockD" in *'_portable_record'*) echo "FAIL: blockD calls a helper it does not define"; exit 1 ;; *) : ;; esac
# The commit has to release the slot, or a post-install failure would delete the record of the
# install that just succeeded.
_commit_block="$(blk '/^_commit_portable_marker\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
case "$_commit_block" in *'_PORTABLE_MARKER_PRIOR_3'*) : ;; *) echo "FAIL: the commit does not release the record slot"; exit 1 ;; esac

# set -e is on in install.sh, so a block ending in a bare `[ cond ] && action` kills the run.
SNIP='set -e
C_WARN=""
substep() { printf "  . %s\n" "$1"; }
rollback_substep() { printf "  R %s\n" "$1"; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockM"'
'"$blockE"'
_export_portable_roots
'"$blockD"'
_clear_stale_portable_marker
_UNSLOTH_LOGIN_PATH="$PATH"
VENV_DIR="$STUDIO_HOME/unsloth_studio"
'"$blockR"'
# ── everything above is install.sh verbatim ──
printf "reached|%s\n" "$STUDIO_HOME"
printf "SLOT3 %s\n" "$_PORTABLE_MARKER_PATH_3"
printf "PRIOR3 %s\n" "$_PORTABLE_MARKER_PRIOR_3"
if [ -d "$VENV_DIR" ]; then _start_studio_venv_replacement "$VENV_DIR"; fi
case "${FAIL_MODE:-ok}" in
    ok)
        _commit_studio_venv_replacement
        _commit_portable_marker
        ;;
    *)
        exit 7
        ;;
esac'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
new_home() { mktemp -d "$T/home.XXXXXX"; }

# env -i: the caller's own UNSLOTH_* would mask the branches under test.
run_install() { # fakehome [env assignments, "--" separates] [args]
    _home="$1"; shift
    _env=""
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do _env="$_env $1"; shift; done
    [ "$#" -eq 0 ] || shift
    # shellcheck disable=SC2086
    env -i HOME="$_home" PATH="$PATH" USER="${USER:-tester}" FAIL_MODE="${FAIL_MODE:-ok}" $_env \
        bash -c "$SNIP" _ "$@" > "$T/out" 2>"$T/err"
    printf '%s\n' "$?"
}
expect_ok() { # fakehome [env...] -- [args]
    _rc="$(run_install "$@")"
    if [ "$_rc" -ne 0 ] || ! grep -q '^reached|' "$T/out"; then
        printf '  FAIL  installer snippet aborted (rc=%s)\n%s\n%s\n' "$_rc" "$(cat "$T/out")" "$(cat "$T/err")"
        fails=$((fails + 1))
    fi
}
record_state() { # dir
    if [ -f "$1/$RECORD" ]; then printf present; else printf gone; fi
}
record_body() { # dir
    if [ -f "$1/$RECORD" ]; then cat -- "$1/$RECORD"; else printf '<absent>'; fi
}

echo "[1] a nested portable install records its master root inside the Studio root"
H1="$(new_home)"; R1="$H1/vol"
expect_ok "$H1" -- --root "$R1"
check "1 the record is written" present "$(record_state "$R1/studio")"
check "1 naming the master root" "$R1" "$(record_body "$R1/studio")"
check "1 and the parent marker is still published" present \
    "$([ -f "$R1/.unsloth-portable-root" ] && printf present || printf gone)"
check "1 with no prior record snapshotted" "PRIOR3 n" "$(grep '^PRIOR3' "$T/out")"

echo "[2] a group-writable root is accepted and recorded just the same"
# The reported failure: `umask 002` makes the root group-writable, the install completes, and
# without this record the activated venv can no longer find it. Section 6 proves the runtime half.
H2="$(new_home)"; R2="$H2/shared"
mkdir -p "$R2"; chmod 0775 "$R2"
expect_ok "$H2" -- --root "$R2"
check "2 the record is written under a group-writable root" present "$(record_state "$R2/studio")"
check "2 naming the master root" "$R2" "$(record_body "$R2/studio")"

echo "[3] the FLAT layout writes none: its marker is already at the Studio root"
H3="$(new_home)"; R3="$H3/flat"
mkdir -p "$R3"
expect_ok "$H3" "UNSLOTH_STUDIO_HOME=$R3" -- --portable
check "3 the flat root is the Studio root" "reached|$R3" "$(grep '^reached|' "$T/out")"
check "3 no record beside the marker" gone "$(record_state "$R3")"
check "3 and the slot stays empty" "SLOT3 " "$(grep '^SLOT3' "$T/out")"

echo "[3b] ...and it clears one an earlier NESTED install left in that directory"
# The record names the level ABOVE $STUDIO_HOME and outranks the marker a flat run writes AT
# it, so leaving it would send this install's caches into the parent tree.
H3b="$(new_home)"; R3b="$H3b/vol"
mkdir -p "$R3b/studio"
printf '%s\n' "$R3b" > "$R3b/studio/$RECORD"
expect_ok "$H3b" "UNSLOTH_STUDIO_HOME=$R3b/studio" -- --portable
check "3b the stale nested record is removed" gone "$(record_state "$R3b/studio")"
check "3b and the flat marker is published in its place" present \
    "$([ -f "$R3b/studio/.unsloth-portable-root" ] && printf present || printf gone)"

echo "[4] a NORMAL reinstall of the same Studio root drops the record"
# Without this the tree still reads as portable, ahead of both markers, and UNSLOTH_PORTABLE=0
# cannot turn it off.
H4="$(new_home)"; R4="$H4/vol"
mkdir -p "$R4/studio/unsloth_studio"
printf '%s\n' "$R4" > "$R4/studio/$RECORD"
printf '%s\n' "$R4" > "$R4/.unsloth-portable-root"
expect_ok "$H4" "UNSLOTH_STUDIO_HOME=$R4/studio" --
check "4 the record is removed" gone "$(record_state "$R4/studio")"
check "4 the parent marker goes with it" gone \
    "$([ -f "$R4/.unsloth-portable-root" ] && printf present || printf gone)"

echo "[5] ...and only the one inside the tree being reinstalled"
# A record belongs to whichever Studio root holds it. Reinstalling a SIBLING must not touch it.
H5="$(new_home)"; R5="$H5/vol"
mkdir -p "$R5/studio/unsloth_studio" "$R5/other"
printf '%s\n' "$R5" > "$R5/studio/$RECORD"
expect_ok "$H5" "UNSLOTH_STUDIO_HOME=$R5/other" --
check "5 a sibling install keeps the neighbour's record" present "$(record_state "$R5/studio")"

echo "[6] --shortcuts-only installs nothing, so it must not convert a tree back"
H6="$(new_home)"; R6="$H6/vol"
mkdir -p "$R6/studio/unsloth_studio"
printf '%s\n' "$R6" > "$R6/studio/$RECORD"
expect_ok "$H6" "UNSLOTH_STUDIO_HOME=$R6/studio" -- --shortcuts-only
check "6 the record survives" present "$(record_state "$R6/studio")"

echo "[7] a failed portable run rolls the record back with the venv"
H7="$(new_home)"; R7="$H7/vol"
FAIL_MODE=fail run_install "$H7" -- --root "$R7" > /dev/null
check "7 a record this run published is removed again" gone "$(record_state "$R7/studio")"
# A run over an EXISTING record snapshots the previous bytes first. It does not put them back
# while its own record is still on disk -- the same rule slot 1 applies to the marker, since a
# file that is back already belongs to a concurrent install of the same tree.
H8="$(new_home)"; R8="$H8/vol"
mkdir -p "$R8/studio/unsloth_studio"
printf '%s\n' "/somewhere/else" > "$R8/studio/$RECORD"
FAIL_MODE=fail run_install "$H8" -- --root "$R8" > /dev/null
check "7 the previous record is snapshotted" "PRIOR3 y/somewhere/else" "$(grep '^PRIOR3' "$T/out")"
# A run that REMOVED one restores it, which is the shape where the snapshot is load-bearing:
# a flat conversion that dies must not leave the nested tree with no record at all.
H8b="$(new_home)"; R8b="$H8b/vol"
mkdir -p "$R8b/studio"
printf '%s\n' "$R8b" > "$R8b/studio/$RECORD"
FAIL_MODE=fail run_install "$H8b" "UNSLOTH_STUDIO_HOME=$R8b/studio" -- --portable > /dev/null
check "7 a record a failed flat conversion removed is restored" "$R8b" "$(record_body "$R8b/studio")"
# The same for the normal-mode reset in _clear_stale_portable_marker.
H8c="$(new_home)"; R8c="$H8c/vol"
mkdir -p "$R8c/studio"
printf '%s\n' "$R8c" > "$R8c/studio/$RECORD"
FAIL_MODE=fail run_install "$H8c" "UNSLOTH_STUDIO_HOME=$R8c/studio" -- > /dev/null
check "7 a record a failed normal reinstall removed is restored" "$R8c" "$(record_body "$R8c/studio")"

echo "[8] a successful run keeps what it wrote"
H9="$(new_home)"; R9="$H9/vol"
expect_ok "$H9" -- --root "$R9"
check "8 the record stands after the commit" "$R9" "$(record_body "$R9/studio")"

echo "[9] a record that cannot be written is fatal, not a silent non-portable install"
H10="$(new_home)"; R10="$H10/vol"
mkdir -p "$R10/studio/$RECORD/leftover"
_rc="$(run_install "$H10" -- --root "$R10")"
check "9 the install fails instead of reporting success" 1 "$_rc"
check "9 the error names the record path" yes \
    "$(grep -qF -- "$R10/studio/$RECORD" "$T/out" "$T/err" && printf yes || printf no)"
check "9 and says a directory is in the way" yes \
    "$(grep -qF -- "A directory is in its place" "$T/out" "$T/err" && printf yes || printf no)"
check "9 the directory is left exactly as it was" yes \
    "$([ -d "$R10/studio/$RECORD/leftover" ] && printf yes || printf no)"

echo "[10] the runtime half, through the real resolver"
if command -v python3 > /dev/null 2>&1; then
    PROBE="$T/probe.py"
    cat > "$PROBE" <<'PYEOF'
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
home = sr.unsloth_home()
print("__JSON__" + json.dumps({
    "portable": sr.portable_mode(),
    "studio_root": str(sr.studio_root()),
    "unsloth_home": str(home) if home else None,
}))
PYEOF
    probe() { # fakehome studio_home key
        _pout=$(env -i HOME="$1" PATH="$PATH" _BACKEND="$BACKEND" \
            _PREFIX="$2/unsloth_studio" python3 "$PROBE" 2>"$T/perr")
        _pjson=$(printf '%s\n' "$_pout" | sed -n 's/^__JSON__//p')
        if [ -z "$_pjson" ]; then
            printf '  FAIL  storage_roots probe produced no output\n%s\n' "$(cat "$T/perr")"
            fails=$((fails + 1)); printf 'probe-failed'; return 0
        fi
        printf '%s' "$_pjson" | _K="$3" python3 -c \
            'import json,os,sys; print(json.load(sys.stdin)[os.environ["_K"]])'
    }
    # The tree section 2 built: group-writable root, so the parent marker alone cannot be
    # believed and only the record keeps this install findable from an activated venv.
    mkdir -p "$R2/studio/unsloth_studio"
    check "10 a group-writable portable root reads as portable" True "$(probe "$H2" "$R2/studio" portable)"
    check "10 and names the root the user selected" "$R2" "$(probe "$H2" "$R2/studio" unsloth_home)"
    check "10 and keeps its Studio root" "$R2/studio" "$(probe "$H2" "$R2/studio" studio_root)"
    # Section 4's tree, after the normal reinstall: back to a plain install.
    check "10 after a normal reinstall the runtime reads as non-portable" False \
        "$(probe "$H4" "$R4/studio" portable)"
else
    printf '  SKIP  storage_roots probe (no python3)\n'
fi

echo "[11] a root whose path contains a newline is refused, not recorded truncated"
# The record holds one path per file and every reader takes the first line of it, so a root
# with a newline in its name is recorded as a TRUNCATED PREFIX: the install reports success
# and the activated venv then resolves whatever unrelated directory sits at that prefix, or
# loses portable mode entirely and writes the caches, the projects root and studio.db back
# under $HOME. Refused at the installer instead, which is what keeps the one-line record and
# every reader of it in agreement without an escape format either side could get wrong.
H11="$(new_home)"
R11="$H11/vol
evil"
mkdir -p "$R11"
_rc="$(run_install "$H11" -- --root "$R11")"
check "11 the install fails instead of recording a prefix" 1 "$_rc"
check "11 the error says why" yes \
    "$(grep -qF -- "containing a newline" "$T/out" "$T/err" && printf yes || printf no)"
check "11 nothing is recorded under the newline root" gone "$(record_state "$R11/studio")"
check "11 and no parent marker is published either" gone \
    "$([ -f "$R11/.unsloth-portable-root" ] && printf present || printf gone)"
# The same gate on the env-var route, which reaches the identical resolver.
H11b="$(new_home)"
R11b="$H11b/vol
evil"
mkdir -p "$R11b"
# Not through run_install: it word-splits its env assignments, so a value with a newline in
# it would be split into a stray command before the installer ever saw it.
_rc=0
env -i HOME="$H11b" PATH="$PATH" USER="${USER:-tester}" FAIL_MODE=ok \
    UNSLOTH_STUDIO_HOME="$R11b" bash -c "$SNIP" _ --portable > "$T/out" 2>"$T/err" || _rc=$?
check "11 UNSLOTH_STUDIO_HOME is refused the same way" 1 "$_rc"
check "11 that error says why too" yes \
    "$(grep -qF -- "containing a newline" "$T/out" "$T/err" && printf yes || printf no)"
# One character short of a newline, and reachable without one: the argument trim runs before
# `pwd -P`, so a trailing slash hides a trailing space from it and the resolved root ends in
# one. Both readers strip the line they read, so that root is recorded and read back as a
# DIFFERENT directory -- `/vol/x ` written, `/vol/x` resolved.
H11e="$(new_home)"; R11e="$H11e/x "
mkdir -p "$R11e"
_rc="$(run_install "$H11e" -- --root "$R11e/")"
check "11 a root ending in whitespace is refused" 1 "$_rc"
check "11 that error says why as well" yes \
    "$(grep -qF -- "starts or ends with whitespace" "$T/out" "$T/err" && printf yes || printf no)"
check "11 and records nothing in it" gone "$(record_state "$R11e/studio")"
# Pinned so the gate cannot collapse into refusing every root. A space is the near miss --
# legal, common on macOS, and untouched by anything here.
H11c="$(new_home)"; R11c="$H11c/my vol"
expect_ok "$H11c" -- --root "$R11c"
check "11 an ordinary path with a space still installs" "$R11c" "$(record_body "$R11c/studio")"
H11d="$(new_home)"; R11d="$H11d/plain"
expect_ok "$H11d" -- --root "$R11d"
check "11 and a plain path still installs" "$R11d" "$(record_body "$R11d/studio")"
expect_ok "$H11d" "UNSLOTH_STUDIO_HOME=$H11d/normal" --
check "11 a normal custom-root install is untouched" "reached|$H11d/normal" \
    "$(grep '^reached|' "$T/out")"

if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
