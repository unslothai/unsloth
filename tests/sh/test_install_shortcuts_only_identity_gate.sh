#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test: --shortcuts-only must not MINT a portable identity for a tree that has none.
#
# The EXIT handler unwinds the portable marker and the master root record when a run FAILS, and
# rightly leaves them alone when it succeeds. --shortcuts-only is the path that breaks that
# symmetry: it rewrites the launchers of an install that already exists, builds nothing, and
# exits 0 several hundred lines PAST the publish. Combined with --root or UNSLOTH_PORTABLE=1
# over an ordinary custom install at <root>/studio it therefore performed a permanent
# conversion -- storage_roots.unsloth_home() went from None to <root>, and the managed
# llama.cpp, node and whisper.cpp that install keeps under its own Studio root started
# resolving as portable siblings one level up, where nothing had ever been built.
#
# So the gate is on the TREE, not on the flags: a --shortcuts-only run over an install that IS
# portable is the whole point of `unsloth studio update` there and has to keep working, and it
# reaches the gate in portable mode because the shim exports UNSLOTH_HOME. Either on-disk
# signal passes, so a moved or half-repaired root is repaired by the publish rather than
# refused. The last section runs the REAL storage_roots resolver over the refused tree, so this
# fails if either half stops agreeing.
#
# Driven by extracting the REAL blocks from install.sh -- the flag parse, the destination
# resolver, the slot declarations, the gate, the publish and the reset -- and running them
# against fixtures.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
BACKEND="$ROOT/studio/backend"
MARKER=".unsloth-portable-root"
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
# The gate itself. Anchored on its own opening line, which is the only `--shortcuts-only AND
# portable` test in the file, and closed by the first `fi` at column 0 after it.
GATE_ANCHOR='if [ "$_SHORTCUTS_ONLY" = true ] && [ "$_PORTABLE_MODE" = true ]; then'
blockG="$(blk '/^if \[ "\$_SHORTCUTS_ONLY" = true \] && \[ "\$_PORTABLE_MODE" = true \]; then$/ {grab=1} grab {print} grab && /^fi$/ {exit}')"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--shortcuts-only) _SHORTCUTS_ONLY=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_STUDIO_HOME_REDIRECT=env'*) : ;; *) echo "FAIL: blockB extraction broke"; exit 1 ;; esac
case "$blockM" in *'_PORTABLE_MARKER_PRIOR_3'*) : ;; *) echo "FAIL: blockM lost the record slot"; exit 1 ;; esac
case "$blockM" in *_export_portable_roots*) echo "FAIL: blockM ran past the slot declarations"; exit 1 ;; *) : ;; esac
case "$blockE" in *"$RECORD"*) : ;; *) echo "FAIL: the publish no longer writes the record"; exit 1 ;; esac
case "$blockD" in *"$MARKER"*) : ;; *) echo "FAIL: blockD extraction broke"; exit 1 ;; esac
# The anchor has to be UNIQUE, or the range above lifts some other block and every assertion
# below stays green while testing nothing. Checked against install.sh itself, not the extract.
_anchor_hits=$(grep -cxF "$GATE_ANCHOR" "$INSTALL")
[ "$_anchor_hits" = 1 ] || { echo "FAIL: the gate anchor matches $_anchor_hits lines, not 1"; exit 1; }
case "$blockG" in *"$MARKER"*) : ;; *) echo "FAIL: the gate no longer tests the portable marker"; exit 1 ;; esac
case "$blockG" in *"$RECORD"*) : ;; *) echo "FAIL: the gate no longer tests the master root record"; exit 1 ;; esac
case "$blockG" in *'exit 1'*) : ;; *) echo "FAIL: the gate no longer refuses anything"; exit 1 ;; esac
# Both roots have to be the RESOLVED ones, so the gate cannot run before blockB.
case "$blockG" in *'$UNSLOTH_ROOT/'*) : ;; *) echo "FAIL: the gate stopped testing \$UNSLOTH_ROOT"; exit 1 ;; esac
case "$blockG" in *'$STUDIO_HOME/'*) : ;; *) echo "FAIL: the gate stopped testing \$STUDIO_HOME"; exit 1 ;; esac
# A helper called from a lifted block dies as "command not found" inside a condition and the
# guard goes silently inert while everything below still passes.
case "$blockG" in *substep*|*_trim_ws*|*_canon*) echo "FAIL: the gate calls a helper it does not define"; exit 1 ;; *) : ;; esac
# set -e: at top level a false `[ -f x ] && flag=true` is a failing command and ends the script,
# so the gate must spell its tests out as `if`.
case "$blockG" in *'] && _so_portable_tree=true'*) echo "FAIL: the gate uses an AND-list that set -e kills"; exit 1 ;; *) : ;; esac

# And it has to sit between the resolver and the publish in the REAL file: after blockB, so the
# roots are resolved, and before the call, so nothing is on disk when it refuses.
_resolve_line=$(grep -n '^_resolve_studio_destinations$' "$INSTALL" | head -n1 | cut -d: -f1)
_gate_line=$(grep -nxF "$GATE_ANCHOR" "$INSTALL" | head -n1 | cut -d: -f1)
_publish_line=$(grep -n '^_export_portable_roots$' "$INSTALL" | head -n1 | cut -d: -f1)
_reset_line=$(grep -n '^_clear_stale_portable_marker$' "$INSTALL" | head -n1 | cut -d: -f1)
_trap_line=$(grep -n '^trap _on_install_exit EXIT$' "$INSTALL" | head -n1 | cut -d: -f1)
for _v in _resolve_line _gate_line _publish_line _reset_line _trap_line; do
    eval "_val=\$$_v"
    [ -n "$_val" ] || { echo "FAIL: could not locate $_v in install.sh"; exit 1; }
done
[ "$_gate_line" -gt "$_resolve_line" ] || { echo "FAIL: the gate runs before the roots are resolved"; exit 1; }
[ "$_gate_line" -lt "$_publish_line" ] || { echo "FAIL: the gate runs after the publish"; exit 1; }
[ "$_gate_line" -lt "$_reset_line" ] || { echo "FAIL: the gate runs after the reset"; exit 1; }
# The traps stay armed first; wave two moved the publish below them and that must not regress.
[ "$_trap_line" -lt "$_publish_line" ] || { echo "FAIL: the publish moved back above the traps"; exit 1; }

# The OTHER path that exits 0 having built nothing in this distro: the Strix Halo WSL reroute.
# The install happened in the target distro, so a --root run that rerouted must not leave this
# distro carrying a marker and a record for a tree with no venv in it -- and over an existing
# normal install at <root>/studio that would be the same conversion the gate above refuses. The
# reroute does not forward --root, so those records would not even describe the child. Static,
# because reaching that branch needs wsl.exe and a second distro: assert the unwind is there and
# comes BEFORE the exit, which is the only ordering that can be wrong.
_rr_tail="$(blk '/^    wsl\.exe -d "\$_rr_target" -- bash -lc "\$_rr_exports; \$_rr_cmd" \|\| _rr_rc=\$\?$/ {grab=1} grab {print} grab && /^        exit 0$/ {exit}')"
case "$_rr_tail" in
    *'exit 0'*) : ;;
    *) echo "FAIL: could not locate the WSL reroute success exit"; exit 1 ;;
esac
case "$_rr_tail" in
    *'_restore_portable_marker'*) : ;;
    *) echo "FAIL: the WSL reroute exits 0 without unwinding what the publish put here"; exit 1 ;;
esac

SNIP='set -e
C_WARN=""
substep() { :; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockM"'
'"$blockE"'
'"$blockD"'
'"$blockG"'
_export_portable_roots
_clear_stale_portable_marker
# ── everything above is install.sh verbatim ──
printf "reached|%s\n" "$STUDIO_HOME"'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
new_home() { mktemp -d "$T/home.XXXXXX"; }

# env -i: the caller's own UNSLOTH_* would mask the branches under test.
run_install() { # fakehome [env assignments] -- [args]
    _home="$1"; shift
    _env=""
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do _env="$_env $1"; shift; done
    [ "$#" -eq 0 ] || shift
    # shellcheck disable=SC2086
    env -i HOME="$_home" PATH="$PATH" USER="${USER:-tester}" $_env \
        bash -c "$SNIP" _ "$@" > "$T/out" 2>"$T/err"
    _rc=$?
    return "$_rc"
}
expect_ok() { # fakehome ...
    if run_install "$@" && grep -q '^reached|' "$T/out"; then printf '  PASS  ran to the publish\n'
    else printf '  FAIL  snippet aborted (rc=%s)\n%s\n%s\n' "$?" "$(cat "$T/out")" "$(cat "$T/err")"; fails=$((fails+1)); fi
}
expect_refused() { # name fakehome ...
    _name="$1"; shift
    if run_install "$@"; then printf '  FAIL  %s : the run was allowed\n' "$_name"; fails=$((fails+1))
    elif grep -q 'cannot convert' "$T/err"; then printf '  PASS  %s\n' "$_name"
    else printf '  FAIL  %s : refused for the wrong reason\n%s\n' "$_name" "$(cat "$T/err")"; fails=$((fails+1)); fi
}
state() { if [ -f "$1" ]; then printf present; else printf absent; fi; }

# A finished NORMAL custom install at <root>/studio: a venv, its own share/ and bin/, and one
# managed helper under the Studio root, which is where a non-portable install keeps them.
mk_normal() { # root
    mkdir -p "$1/studio/unsloth_studio/bin" "$1/studio/share" "$1/studio/bin" "$1/studio/node"
    printf '#!/bin/sh\nexit 0\n' > "$1/studio/unsloth_studio/bin/unsloth"
    chmod +x "$1/studio/unsloth_studio/bin/unsloth"
    : > "$1/studio/unsloth_studio/.unsloth-studio-owned"
}
mk_portable_nested() { # root
    mk_normal "$1"
    printf '%s\n' "$1" > "$1/$MARKER"
    printf '%s\n' "$1" > "$1/studio/$RECORD"
}

echo "[1] --shortcuts-only --root over a NORMAL install must not mint an identity"
H1="$(new_home)"; R1="$H1/vol"; mk_normal "$R1"
expect_refused "--root is refused on a non-portable tree" "$H1" -- --shortcuts-only --root "$R1"
check "no marker was published" absent "$(state "$R1/$MARKER")"
check "no record was published" absent "$(state "$R1/studio/$RECORD")"

echo "[2] the same combination seeded from the environment"
H2="$(new_home)"; R2="$H2/vol"; mk_normal "$R2"
expect_refused "UNSLOTH_PORTABLE=1 is refused on a non-portable tree" \
    "$H2" UNSLOTH_PORTABLE=1 "UNSLOTH_STUDIO_HOME=$R2/studio" -- --shortcuts-only
check "no marker from the env path" absent "$(state "$R2/$MARKER")"
check "no record from the env path" absent "$(state "$R2/studio/$RECORD")"

echo "[3] a GENUINE portable install must still regenerate its launchers"
H3="$(new_home)"; R3="$H3/vol"; mk_portable_nested "$R3"
expect_ok "$H3" -- --shortcuts-only --root "$R3"
check "the marker survives" present "$(state "$R3/$MARKER")"
check "the record survives" present "$(state "$R3/studio/$RECORD")"
check "the record still names the root" "$R3" "$(cat "$R3/studio/$RECORD")"

echo "[4] and via UNSLOTH_HOME, which is how \`unsloth studio update\` gets here"
H4="$(new_home)"; R4="$H4/vol"; mk_portable_nested "$R4"
expect_ok "$H4" "UNSLOTH_HOME=$R4" UNSLOTH_PORTABLE=1 -- --shortcuts-only
check "the marker survives the env path" present "$(state "$R4/$MARKER")"

echo "[5] a FLAT portable root, where the marker is the only signal there can be"
H5="$(new_home)"; R5="$H5/flat"
mkdir -p "$R5/unsloth_studio/bin" "$R5/share" "$R5/bin"
printf '#!/bin/sh\nexit 0\n' > "$R5/unsloth_studio/bin/unsloth"; chmod +x "$R5/unsloth_studio/bin/unsloth"
: > "$R5/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$R5" > "$R5/$MARKER"
expect_ok "$H5" -- --shortcuts-only --root "$R5"
check "the flat marker survives" present "$(state "$R5/$MARKER")"

echo "[6] EITHER signal passes, so a half-lost pair is repaired rather than refused"
H6="$(new_home)"; R6="$H6/vol"; mk_portable_nested "$R6"; rm -f "$R6/studio/$RECORD"
expect_ok "$H6" -- --shortcuts-only --root "$R6"
check "the marker alone admits the run" present "$(state "$R6/$MARKER")"
check "and the publish mints the missing record" "$R6" "$(cat "$R6/studio/$RECORD" 2>/dev/null)"
H6b="$(new_home)"; R6b="$H6b/vol"; mk_portable_nested "$R6b"; rm -f "$R6b/$MARKER"
expect_ok "$H6b" -- --shortcuts-only --root "$R6b"
check "the record alone admits the run" present "$(state "$R6b/$MARKER")"

echo "[7] a root that MOVED: the marker names its old path, and the run rewrites it"
H7="$(new_home)"; R7="$H7/vol"; mk_portable_nested "$R7"
printf '%s\n' "$H7/somewhere-else" > "$R7/$MARKER"
expect_ok "$H7" -- --shortcuts-only --root "$R7"
check "the marker is repaired to the new root" "$R7" "$(cat "$R7/$MARKER")"

echo "[8] the gate must not fire on runs it has no business touching"
H8="$(new_home)"; R8="$H8/vol"; mk_normal "$R8"
expect_ok "$H8" "UNSLOTH_STUDIO_HOME=$R8/studio" -- --shortcuts-only
check "plain --shortcuts-only still gains no marker" absent "$(state "$R8/$MARKER")"
check "plain --shortcuts-only still gains no record" absent "$(state "$R8/studio/$RECORD")"
H8b="$(new_home)"; R8b="$H8b/vol"; mk_normal "$R8b"
expect_ok "$H8b" -- --root "$R8b"
check "a REAL portable install still publishes its marker" present "$(state "$R8b/$MARKER")"
check "a REAL portable install still publishes its record" present "$(state "$R8b/studio/$RECORD")"

# ── The runtime half: the refused tree must still resolve as a normal install. ──
echo "[9] storage_roots over the refused tree"
if command -v python3 >/dev/null 2>&1 && [ -d "$BACKEND/utils/paths" ]; then
    _py_out=$(cd "$BACKEND" && env -i HOME="$H1" PATH="$PATH" \
        UNSLOTH_STUDIO_HOME="$R1/studio" PYTHONPATH="$BACKEND" python3 - <<'PY' 2>&1
import sys
from utils.paths.storage_roots import studio_root, unsloth_home, portable_mode
sys.path.insert(0, ".")
from utils.node_runtime import managed_node_dir
print("home=%s" % unsloth_home())
print("portable=%s" % portable_mode())
print("node=%s" % managed_node_dir())
PY
)
    check "unsloth_home stays unset" "home=None" "$(printf '%s\n' "$_py_out" | grep '^home=')"
    check "portable_mode stays off" "portable=False" "$(printf '%s\n' "$_py_out" | grep '^portable=')"
    check "the managed node dir stays under the Studio root" "node=$R1/studio/node" \
        "$(printf '%s\n' "$_py_out" | grep '^node=')"
else
    printf '  SKIP  python3 or the backend tree is unavailable\n'
fi

if [ "$fails" -eq 0 ]; then echo "ALL PASS"; else echo "$fails FAILED"; exit 1; fi
