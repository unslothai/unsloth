#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests that setup.sh finishes with the copy the update installed. The dependency pass upgrades the
# package that ships setup.sh, and bash keeps reading the old file through its open descriptor, so
# without the handoff every phase a release adds after that pass (the audio.cpp prebuilt, first) was
# skipped by the update that installed it. The real start block and _setup_rerun_if_replaced are
# sliced out of setup.sh into fake old/new/third scripts whose "dependency step" replaces the file
# the way uv does.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
# shellcheck source=_harness.sh
source "$SCRIPT_DIR/_harness.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

drift() {
    echo "FATAL: the self-rerun extraction no longer matches $SETUP_SH -- $1" >&2
    echo "       Fix the extraction in $0 (or the block in setup.sh), do not silence it." >&2
    exit 1
}

START_FIRST='_SETUP_SELF="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"'
START_LAST='_SETUP_START_PWD=$PWD'
FUNC_FIRST='_setup_rerun_if_replaced() {'

[ "$(grep -cxF -- "$START_FIRST" "$SETUP_SH")" = 1 ] || drift "expected exactly one '$START_FIRST' line"
[ "$(grep -cxF -- "$START_LAST" "$SETUP_SH")" = 1 ] || drift "expected exactly one '$START_LAST' line"
[ "$(grep -cxF -- "$FUNC_FIRST" "$SETUP_SH")" = 1 ] || drift "expected exactly one '$FUNC_FIRST' line"

awk -v A="$START_FIRST" -v B="$START_LAST" '$0 == A { on = 1 } on { print } on && $0 == B { exit }' \
    "$SETUP_SH" > "$WORK/start_blk.sh"
awk -v A="$FUNC_FIRST" '$0 == A { on = 1 } on { print } on && /^}$/ { exit }' \
    "$SETUP_SH" > "$WORK/func_blk.sh"
[ "$(wc -l < "$WORK/start_blk.sh")" -ge 4 ] || drift "the start block is shorter than four lines"
[ "$(tail -n 1 "$WORK/func_blk.sh")" = "}" ] || drift "_setup_rerun_if_replaced has no closing brace"
grep -q 'exec ' "$WORK/func_blk.sh" || drift "_setup_rerun_if_replaced no longer execs the new copy"

# The handoff only helps if it sits right behind the one call that runs the full dependency pass.
_calls=$(grep -cx '    install_python_stack' "$SETUP_SH" || true)
assert_eq "setup.sh runs the full dependency pass from exactly one place" "1" "$_calls"
_after=$(awk '$0 == "    install_python_stack" { getline; print; exit }' "$SETUP_SH")
assert_eq "the rerun check directly follows that dependency pass" "    _setup_rerun_if_replaced" "$_after"

# One fake setup.sh per variant. Its dependency step is driven by DEPS_<VARIANT>:
#   none | same | unlink:<src> | rename:<src> | unlink-fail:<src>
# "same" writes identical bytes into a new file, the strongest form of "nothing changed".
make_variant() {
    _v="$1"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        echo 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
        cat "$WORK/start_blk.sh"
        echo 'step()    { printf "STEP %s: %s\n" "$1" "$2"; }'
        echo 'substep() { printf "SUB %s\n" "$1"; }'
        echo 'C_WARN='
        cat "$WORK/func_blk.sh"
        cat <<EOF
echo "START $_v pid=\$\$ pwd=\$PWD guard=\${UNSLOTH_SETUP_RERUN:-}"
echo "FULLDEPS $_v [\${UNSLOTH_STUDIO_FULL_DEPS-<unset>}]"
for _a in "\$@"; do echo "ARG $_v [\$_a]"; done
cd "\$SCRIPT_DIR"
_action=\${DEPS_$_v:-none}
case "\$_action" in
    none) ;;
    same) cp "\$_SETUP_SELF" "\$SCRIPT_DIR/.same"; rm -f "\$_SETUP_SELF"; cp "\$SCRIPT_DIR/.same" "\$_SETUP_SELF" ;;
    unlink:*) rm -f "\$_SETUP_SELF"; cp "\${_action#unlink:}" "\$_SETUP_SELF" ;;
    rename:*) cp "\${_action#rename:}" "\$SCRIPT_DIR/.next"; mv -f "\$SCRIPT_DIR/.next" "\$_SETUP_SELF" ;;
    unlink-fail:*) rm -f "\$_SETUP_SELF"; cp "\${_action#unlink-fail:}" "\$_SETUP_SELF"; false ;;
esac
[ -z "\${FORCE_BAD_BASH:-}" ] || BASH=/nonexistent/bash
_setup_rerun_if_replaced
echo "PHASE $_v after-deps guard=\${UNSLOTH_SETUP_RERUN:-}"
exit \${EXIT_$_v:-0}
EOF
    } > "$WORK/src_$_v.sh"
}
for _v in OLD NEW THIRD; do make_variant "$_v"; done
mkdir -p "$WORK/launch dir"

# Installs OLD as the package's setup.sh and runs it from a launch dir the way the CLI does.
OUT=""
RC=0
run_setup() {
    _interp="$1"; shift
    rm -rf "$WORK/pkg"; mkdir -p "$WORK/pkg"
    cp "$WORK/src_OLD.sh" "$WORK/pkg/setup.sh"
    RC=0
    OUT=$(cd "$WORK/launch dir" && env "$@" "$_interp" "$WORK/pkg/setup.sh" --local "a b" 2>&1) || RC=$?
}
count() { printf '%s\n' "$OUT" | grep -cF -- "$1" || true; }
HANDOFF="STEP setup: the update replaced this setup script; finishing with the new version"

INTERPRETERS=(bash)
if [ "$(uname -s)" = Darwin ] && [ -x /bin/bash ]; then
    # The CLI's `bash` on a stock Mac is 3.2, older than every array idiom we lean on.
    INTERPRETERS+=(/bin/bash)
fi
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) RENAME_OK=0 ;; # rename over an open file is refused on Windows
    *) RENAME_OK=1 ;;
esac

for INTERP in "${INTERPRETERS[@]}"; do
    echo "== $INTERP =="

    run_setup "$INTERP" DEPS_OLD="unlink:$WORK/src_NEW.sh"
    assert_eq "[$INTERP] replaced (unlink+create): exits 0" "0" "$RC"
    assert_eq "[$INTERP] replaced: hands off once" "1" "$(count "$HANDOFF")"
    assert_eq "[$INTERP] replaced: the new copy runs once" "1" "$(count "START NEW")"
    assert_eq "[$INTERP] replaced: the new copy's later phases run" "1" "$(count "PHASE NEW after-deps guard=1")"
    assert_eq "[$INTERP] replaced: the old copy's later phases do not" "0" "$(count "PHASE OLD")"
    _old_pid=$(printf '%s\n' "$OUT" | sed -n 's/^START OLD pid=\([0-9]*\).*/\1/p')
    assert_contains "[$INTERP] replaced: exec keeps the PID the CLI waits on" "$OUT" "START NEW pid=$_old_pid "
    assert_contains "[$INTERP] replaced: the new copy starts in the launch dir" "$OUT" "pwd=$WORK/launch dir guard=1"
    assert_contains "[$INTERP] replaced: --local reaches the new copy" "$OUT" "ARG NEW [--local]"
    assert_contains "[$INTERP] replaced: an argument with a space stays whole" "$OUT" "ARG NEW [a b]"

    if [ "$RENAME_OK" = 1 ]; then
        run_setup "$INTERP" DEPS_OLD="rename:$WORK/src_NEW.sh"
        assert_eq "[$INTERP] replaced (rename): hands off once" "1" "$(count "$HANDOFF")"
        assert_eq "[$INTERP] replaced (rename): the old copy's later phases do not run" "0" "$(count "PHASE OLD")"
        assert_eq "[$INTERP] replaced (rename): the new copy's later phases run" "1" "$(count "PHASE NEW")"
    fi

    run_setup "$INTERP" UNSLOTH_STUDIO_FULL_DEPS=1 DEPS_OLD="unlink:$WORK/src_NEW.sh"
    assert_contains "[$INTERP] full deps: the first copy sees the override" "$OUT" "FULLDEPS OLD [1]"
    assert_contains "[$INTERP] full deps: the rerun does not force a second pass" "$OUT" "FULLDEPS NEW [<unset>]"

    run_setup "$INTERP" DEPS_OLD=none
    assert_eq "[$INTERP] untouched: no handoff" "0" "$(count "$HANDOFF")"
    assert_eq "[$INTERP] untouched: the old copy finishes" "1" "$(count "PHASE OLD after-deps guard=")"

    run_setup "$INTERP" DEPS_OLD=same
    assert_eq "[$INTERP] same bytes rewritten: no handoff" "0" "$(count "$HANDOFF")"
    assert_eq "[$INTERP] same bytes rewritten: the old copy finishes" "1" "$(count "PHASE OLD")"

    run_setup "$INTERP" DEPS_OLD="unlink-fail:$WORK/src_NEW.sh"
    assert_eq "[$INTERP] failed dependency pass: fails" "1" "$([ "$RC" -ne 0 ] && echo 1 || echo 0)"
    assert_eq "[$INTERP] failed dependency pass: no handoff" "0" "$(count "$HANDOFF")"
    assert_eq "[$INTERP] failed dependency pass: the new copy never starts" "0" "$(count "START NEW")"

    run_setup "$INTERP" DEPS_OLD="unlink:$WORK/src_NEW.sh" DEPS_NEW="unlink:$WORK/src_THIRD.sh"
    assert_eq "[$INTERP] replaced again by the rerun: still one handoff" "1" "$(count "$HANDOFF")"
    assert_eq "[$INTERP] replaced again by the rerun: a third copy never starts" "0" "$(count "START THIRD")"
    assert_eq "[$INTERP] replaced again by the rerun: the rerun finishes" "1" "$(count "PHASE NEW")"

    run_setup "$INTERP" UNSLOTH_SETUP_RERUN=1 DEPS_OLD="unlink:$WORK/src_NEW.sh"
    assert_eq "[$INTERP] guard already set: no handoff" "0" "$(count "$HANDOFF")"
    assert_eq "[$INTERP] guard already set: the running copy finishes" "1" "$(count "PHASE OLD")"

    run_setup "$INTERP" DEPS_OLD="unlink:$WORK/src_NEW.sh" EXIT_NEW=5
    assert_eq "[$INTERP] the new copy's exit status is the run's" "5" "$RC"

    run_setup "$INTERP" DEPS_OLD="unlink:$WORK/src_NEW.sh" FORCE_BAD_BASH=1
    assert_eq "[$INTERP] exec fails: the run still exits 0" "0" "$RC"
    assert_contains "[$INTERP] exec fails: says so" "$OUT" "SUB could not start the updated setup script; continuing with this one"
    assert_eq "[$INTERP] exec fails: the old copy finishes without the guard" "1" "$(count "PHASE OLD after-deps guard=")"
done

summary
