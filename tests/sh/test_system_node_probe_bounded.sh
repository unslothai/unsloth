#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Verify hung probes and lingering children cannot stall setup (#11709),
# with GNU timeout and the watchdog used on stock macOS.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"

WORK=$(mktemp -d)
# Unique, so cleanup reaches only the children these fakes started.
SLEEP_MARK=$((300000 + $$ % 10000))
cleanup() { pkill -f "sleep $SLEEP_MARK" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

HELPER=$(awk '
    /^_setup_probe_signal_target\(\) \{/ { grab = 1 }
    /^_setup_probe_terminate\(\) \{/ { grab = 1 }
    /^_setup_probe_restore_trap\(\) \{/ { grab = 1 }
    /^_setup_probe_on_signal\(\) \{/ { grab = 1 }
    /^_setup_probe_version\(\) \{/ { grab = 1 }
    /^_probe_system_node_tool\(\) \{/ { grab = 1 }
    grab { print }
    grab && /^}/ { grab = 0 }
' "$SETUP_SH")
for _fn in _setup_probe_signal_target _setup_probe_terminate _setup_probe_restore_trap \
           _setup_probe_on_signal _setup_probe_version _probe_system_node_tool; do
    printf '%s\n' "$HELPER" | grep -q "^$_fn() {" || {
        echo "FATAL: could not extract $_fn from setup.sh" >&2; exit 1; }
done
printf '%s\n' "$HELPER" > "$WORK/helper.sh"

mkdir -p "$WORK/hang" "$WORK/stubborn" "$WORK/leak" "$WORK/good" "$WORK/notimeout" "$WORK/tools"
printf '#!/bin/sh\nexec sleep %s\n' "$SLEEP_MARK" > "$WORK/hang/node"
# Ignores TERM, so only the KILL after the grace period stops it (GNU timeout then exits 137).
printf '#!/bin/sh\ntrap "" TERM\nsleep %s\n' "$SLEEP_MARK" > "$WORK/stubborn/node"
printf '#!/bin/sh\n( sleep %s ) &\necho v24.0.0\n' "$SLEEP_MARK" > "$WORK/leak/node"
printf '#!/bin/sh\necho v22.17.1\n' > "$WORK/good/node"
# Stock macOS has no GNU timeout; a failing stand-in sends the probe down its watchdog branch.
printf '#!/bin/sh\nexit 1\n' > "$WORK/notimeout/timeout"
chmod +x "$WORK"/*/*
# Exclude the host's Node. Include an external `true` for the GNU timeout check;
# `type -P` resolves its binary rather than the shell builtin.
for _t in bash sh mktemp head rm sleep ps cat timeout true; do
    _src=$(type -P "$_t") && ln -s "$_src" "$WORK/tools/$_t"
done

# probe <fake dir> <with|without timeout>: prints "<seconds> <version>|<output>"
probe() {
    _p="$WORK/$1:$WORK/tools"
    [ "$2" = without ] && _p="$WORK/notimeout:$_p"
    _start=$(date +%s)
    # Catch children that keep the caller's output pipe open.
    _res=$(env PATH="$_p" _SETUP_PROBE_SECONDS=2 bash -c '
        substep() { printf "%s\n" "$1"; }
        C_WARN=
        . "$1/helper.sh"
        _probe_system_node_tool node > "$1/said" 2>&1
        printf "%s|%s" "$_PROBED_VER" "$(cat "$1/said")"
    ' _ "$WORK" | cat)
    echo "$(( $(date +%s) - _start )) $_res"
}

for _mode in with without; do
    echo "$_mode GNU timeout"

    _out=$(probe hang "$_mode")
    _secs=${_out%% *}; _rest=${_out#* }
    [ "$_secs" -le 10 ] && ok "hung node: probe returns ($_secs s)" || bad "hung node: probe returns ($_secs s)"
    assert_eq "hung node: counts as no system Node" "" "${_rest%%|*}"
    assert_contains "hung node: says which node it gave up on" "${_rest#*|}" "$WORK/hang/node) did not answer --version within 2s"

    _out=$(probe stubborn "$_mode")
    _secs=${_out%% *}; _rest=${_out#* }
    [ "$_secs" -le 12 ] && ok "node ignoring TERM: probe returns ($_secs s)" || bad "node ignoring TERM: probe returns ($_secs s)"
    assert_eq "node ignoring TERM: counts as no system Node" "" "${_rest%%|*}"
    assert_contains "node ignoring TERM: still says which node it gave up on" "${_rest#*|}" "$WORK/stubborn/node) did not answer --version within 2s"

    _out=$(probe leak "$_mode")
    _secs=${_out%% *}; _rest=${_out#* }
    [ "$_secs" -le 3 ] && ok "node leaving a child running: probe returns ($_secs s)" \
        || bad "node leaving a child running: probe returns ($_secs s)"
    assert_eq "node leaving a child running: its answer is kept" "v24.0.0" "${_rest%%|*}"

    _out=$(probe good "$_mode"); _rest=${_out#* }
    assert_eq "healthy node: version read" "v22.17.1" "${_rest%%|*}"
    assert_eq "healthy node: nothing printed" "" "${_rest#*|}"

    _out=$(probe missing "$_mode"); _rest=${_out#* }
    assert_eq "no node on PATH: empty, silently" "|" "$_rest"
done

summary
