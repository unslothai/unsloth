#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The POSIX hardware summary must recognise a working Intel XPU runtime, and `studio update`
# must raise the bitsandbytes floor for it. The summary only tested NVIDIA / AMD / Apple, and
# `studio update` runs setup.sh, never install.sh, where the XPU bnb floor used to live.
#
# Unlike the Windows half this runs for real: the venv tree, torch/version.py and the
# interpreter are built here, so the disk read and the runtime probe are executed, not mocked.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="${1:-$SCRIPT_DIR/../../studio/setup.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The detection + floor block verbatim, from the Intel banner to the end of the bnb install.
awk '/^# Intel XPU\. There is no vendor probe here/, /^fi$/' "$SETUP_SH" > "$WORK/blk.sh"
[ -s "$WORK/blk.sh" ] || { echo "FATAL: XPU detection block not found in $SETUP_SH" >&2; exit 1; }
# An extraction that lost either half would make the cases below pass vacuously.
grep -q '_setup_xpu_ready=true' "$WORK/blk.sh" || { echo "FATAL: extraction lost the runtime probe" >&2; exit 1; }
grep -q 'bitsandbytes>=0.50.0' "$WORK/blk.sh" || { echo "FATAL: extraction lost the bnb floor" >&2; exit 1; }

PASS=0
FAIL=0
check() {
    if [ "$2" = "$3" ]; then
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %-34s got=%s want=%s\n' "$1" "$2" "$3"
        FAIL=$((FAIL + 1))
    fi
}

# Builds a venv whose torch reports $1 and whose xpu runtime answers $2 (ok|broken|none).
make_venv() {
    _v="$WORK/venv_$3"
    rm -rf "$_v"
    mkdir -p "$_v/bin"
    if [ -n "$1" ]; then
        mkdir -p "$_v/lib/python3.12/site-packages/torch"
        printf "from typing import Optional\n__version__ = '%s'\ndebug = False\n" "$1" \
            > "$_v/lib/python3.12/site-packages/torch/version.py"
    fi
    # Stands in for the interpreter: exit 0 only when the runtime is meant to initialise.
    case "$2" in
        ok)     printf '#!/bin/sh\nexit 0\n'  > "$_v/bin/python" ;;
        broken) printf '#!/bin/sh\nexit 1\n'  > "$_v/bin/python" ;;
        none)   : ;;
    esac
    [ -f "$_v/bin/python" ] && chmod +x "$_v/bin/python"
    printf '%s' "$_v"
}

# Runs the block against a venv and echoes "<is_xpu> <ready> <bnb_fired>".
run_case() {
    (
        VENV_DIR="$1"
        # shellcheck disable=SC2317
        fast_install() { echo "BNB_FIRED" >> "$WORK/fired"; return 0; }
        # shellcheck disable=SC2317
        run_quiet() { shift; "$@"; }
        # shellcheck disable=SC2317
        run_quiet_no_exit() { shift; "$@"; }
        # shellcheck disable=SC2317
        timeout() { shift; "$@"; }
        # shellcheck disable=SC2317
        substep() { :; }
        : > "$WORK/fired"
        # shellcheck disable=SC1091
        . "$WORK/blk.sh"
        _fired=no
        [ -s "$WORK/fired" ] && _fired=yes
        echo "$_setup_torch_is_xpu $_setup_xpu_ready $_fired"
    )
}

echo "a working XPU runtime is recognised, and only a real one"
check "xpu wheel + runtime ok"   "$(run_case "$(make_venv '2.9.1+xpu' ok a)")"     "true true yes"
# A +xpu wheel installs fine on a host whose driver never initialises: it must raise the bnb
# floor but must NOT claim a GPU in the summary.
check "xpu wheel + runtime dead" "$(run_case "$(make_venv '2.9.1+xpu' broken b)")" "true false yes"
check "cuda wheel"               "$(run_case "$(make_venv '2.9.1+cu128' ok c)")"   "false false no"
check "rocm wheel"               "$(run_case "$(make_venv '2.9.1+rocm6.4' ok d)")" "false false no"
check "untagged wheel"           "$(run_case "$(make_venv '2.9.1' ok e)")"         "false false no"
check "no torch installed"       "$(run_case "$(make_venv '' ok f)")"              "false false no"
check "no venv at all"           "$(run_case "$WORK/nope")"                        "false false no"
# The interpreter is missing but the wheel is there: no crash, no GPU claim, floor still raised.
check "xpu wheel, no interpreter" "$(run_case "$(make_venv '2.9.1+xpu' none g)")"  "true false yes"

echo "the summary has an XPU arm, ranked below NVIDIA and AMD"
# Anchored at column 0: both flags are re-tested later inside indented blocks, and matching one
# of those would compare the arm against the wrong line.
_arm=$(grep -n '^elif \[ "\$_setup_xpu_ready" = true \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
_nv=$(grep -n '^if \[ "\$_setup_nvidia_usable" = true \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
_amd=$(grep -n '^elif \[ "\$_setup_amd_detected" = true \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
_none=$(grep -n 'none (chat-only / GGUF)' "$SETUP_SH" | head -1 | cut -d: -f1)
check "arm exists"               "$([ -n "$_arm" ] && echo yes || echo no)" "yes"
check "ranked after NVIDIA"      "$([ -n "$_arm" ] && [ -n "$_nv" ] && [ "$_arm" -gt "$_nv" ] && echo yes || echo no)" "yes"
check "ranked after AMD"         "$([ -n "$_arm" ] && [ -n "$_amd" ] && [ "$_arm" -gt "$_amd" ] && echo yes || echo no)" "yes"
check "ranked before the CPU arm" "$([ -n "$_arm" ] && [ -n "$_none" ] && [ "$_arm" -lt "$_none" ] && echo yes || echo no)" "yes"
# The unavailable-runtime arm must not promise CPU training: with neither CUDA nor XPU,
# unsloth/device_type.py raises NotImplementedError, so importing unsloth fails outright.
# Anchored on the ELIF at column 0 and stopped at the next arm: a plain awk range on the flag
# name matches the bitsandbytes `if` block, whose own "unavailable" warning would pass on any
# wording. Comment lines are dropped, since the arm quotes the wording it must NOT use.
_warn_arm=$(awk '/^elif \[ "\$_setup_torch_is_xpu" = true \]; then$/{on=1;next} on && /^elif|^else/{exit} on' "$SETUP_SH" | grep -v '^[[:space:]]*#')
check "warn arm was extracted" "$(printf '%s' "$_warn_arm" | grep -ci 'XPU runtime unavailable')" "1"
check "no CPU-training promise" "$(printf '%s' "$_warn_arm" | grep -ci 'run on CPU\|runs on CPU')" "0"
check "says training is unavailable" "$(printf '%s' "$_warn_arm" | grep -ci 'are unavailable')" "1"

# The floor must run before the summary; the other order would report on a stale venv.
_bnb=$(grep -n 'install bitsandbytes (xpu)' "$SETUP_SH" | head -1 | cut -d: -f1)
check "floor precedes the summary" "$([ -n "$_bnb" ] && [ -n "$_none" ] && [ "$_bnb" -lt "$_none" ] && echo yes || echo no)" "yes"
# run_quiet routes failure to setup_fail and EXITS, which would abort an otherwise fine
# `studio update` over a best-effort step and make the warning below unreachable.
check "floor uses the nonfatal wrapper" \
    "$(grep -q 'run_quiet_no_exit "install bitsandbytes (xpu)"' "$SETUP_SH" && echo yes || echo no)" "yes"
# An unbounded `import torch` hangs forever on a stalled Intel driver, the host this classifies.
check "runtime probe is bounded" \
    "$(grep -q 'timeout 60 "\$VENV_DIR/bin/python" -c "\$_setup_xpu_probe"' "$SETUP_SH" && echo yes || echo no)" "yes"
# ...and `timeout` is not everywhere (base macOS, minimal Linux images), so the fallback arm
# ran the very probe this bounds with no deadline at all.
_probe=$(sed -n "s/^ *_setup_xpu_probe='\(.*\)'$/\1/p" "$SETUP_SH" | head -1)
check "probe was extracted" "$([ -n "$_probe" ] && echo yes || echo no)" "yes"
check "probe carries its own deadline" \
    "$(printf '%s' "$_probe" | grep -q 'signal.alarm(' && echo yes || echo no)" "yes"
# Both arms must run the SAME string, or only one is bounded. Comment lines are excluded: the
# summary arm below names the same call. So does the #8473 probe, which asks it inside its OWN
# bounded probe -- excluded by name, not by raising the count, so a copy-paste still fails here.
check "no second probe literal" \
    "$(grep -v '^ *#' "$SETUP_SH" | grep -v "^ *_setup_torch_probe='" | grep -c "torch.xpu.is_available()")" "1"
check "fallback arm reuses the probe" \
    "$(grep -q 'elif "\$VENV_DIR/bin/python" -c "\$_setup_xpu_probe"' "$SETUP_SH" && echo yes || echo no)" "yes"
# Execute it against a torch that never returns, alarm shortened: python installs no SIGALRM
# handler, so the default action kills the process even while a stalled driver blocks in C.
if command -v python3 >/dev/null 2>&1; then
    mkdir -p "$WORK/fakemod"
    printf 'import ctypes\nctypes.CDLL(None).sleep(60)\n' > "$WORK/fakemod/torch.py"
    _fast_probe=$(printf '%s' "$_probe" | sed 's/signal\.alarm([0-9]*)/signal.alarm(2)/')
    # Own watchdog, not GNU timeout: macOS ships none, and a missing one exits 127 instantly,
    # which reads as "the probe was killed" and passed both checks below without python running.
    _t0=$(date +%s)
    ( cd "$WORK/fakemod" && PYTHONPATH="$WORK/fakemod" python3 -c "$_fast_probe" ) >/dev/null 2>&1 &
    _probe_pid=$!
    ( sleep 30; kill -9 "$_probe_pid" ) >/dev/null 2>&1 &
    _watchdog_pid=$!
    # Silenced: the shell reports "Alarm clock" for a signal-killed job here.
    { wait "$_probe_pid"; _rc=$?; } 2>/dev/null
    kill "$_watchdog_pid" >/dev/null 2>&1
    _elapsed=$(( $(date +%s) - _t0 ))
    check "a wedged driver is killed, not waited on" "$([ "$_rc" -ne 0 ] && echo yes || echo no)" "yes"
    # Lower bound too: the alarm is 2s, so an instant return means the probe never ran.
    check "killed at the deadline, not after" \
        "$([ "$_elapsed" -ge 1 ] && [ "$_elapsed" -lt 10 ] && echo yes || echo no)" "yes"
fi

# The dependency pass is the ONLY thing that acts on an XPU pin, so a CPU install switched to
# the xpu family would keep its CPU wheel forever once the fast path skips that pass.
_esc=$(grep -n 'XPU index pinned but torch does not match' "$SETUP_SH" | head -1 | cut -d: -f1)
_gate=$(grep -n '^if \[ "\$_SKIP_PYTHON_DEPS" = false \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
check "fast path has an XPU escape" "$([ -n "$_esc" ] && echo yes || echo no)" "yes"
check "escape precedes the skip gate" \
    "$([ -n "$_esc" ] && [ -n "$_gate" ] && [ "$_esc" -lt "$_gate" ] && echo yes || echo no)" "yes"
# It must clear the flag, not merely warn.
check "escape forces the dependency pass" \
    "$(awk -v a="$_esc" 'NR>=a && NR<=a+2 && /_SKIP_PYTHON_DEPS=false/{f=1} END{print (f?"yes":"no")}' "$SETUP_SH")" "yes"

# An authenticated or fragmented mirror is a supported pin shape; a raw suffix test reads it as
# "no XPU pin" and skips the repair.
check "pin match strips the query" \
    "$(grep -q '_setup_pin="\${_setup_pin%%\\?\*}"' "$SETUP_SH" && echo yes || echo no)" "yes"
check "pin match strips the fragment" \
    "$(grep -q '_setup_pin="\${_setup_pin%%\\#\*}"' "$SETUP_SH" && echo yes || echo no)" "yes"
# The escape must not launch an interpreter: a wedged driver hangs inside `import torch`, and
# this runs before the bounded probes. Bounded by the acting chain, not a line offset, so a new
# check inside cannot move the window. What it DECIDES is test_setup_xpu_fastpath_escape.sh.
_blk=$(awk '/_setup_pin="\$\{UNSLOTH_TORCH_INDEX_URL/{on=1} on{print} on && /_SKIP_PYTHON_DEPS=false$/{n++} n==2{exit}' "$SETUP_SH")
check "escape block was found" "$([ -n "$_blk" ] && echo yes || echo no)" "yes"
check "escape reads the flavour off disk" \
    "$(printf '%s' "$_blk" | grep -q 'site-packages/torch/version.py' && echo yes || echo no)" "yes"
check "escape launches no interpreter" \
    "$(printf '%s' "$_blk" | grep -q '\$VENV_DIR/bin/python' && echo no || echo yes)" "yes"
# One %/ leaves a slash on ".../xpu//", which reads as "no XPU pin".
check "pin match strips every trailing slash" \
    "$(printf '%s' "$_blk" | grep -q 'while \[ "\${_setup_pin%/}" != "\$_setup_pin" \]' && echo yes || echo no)" "yes"

echo "an installed +xpu wheel whose runtime is dead gets its own arm"
# is_available() false on a real +xpu install means the compute DRIVER is missing or too old;
# falling through to "none (chat-only / GGUF)" hides the one action that fixes it.
_dead=$(grep -n '^elif \[ "\$_setup_torch_is_xpu" = true \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
check "dead-runtime arm exists" "$([ -n "$_dead" ] && echo yes || echo no)" "yes"
check "ranked after the ready arm" \
    "$([ -n "$_dead" ] && [ -n "$_arm" ] && [ "$_dead" -gt "$_arm" ] && echo yes || echo no)" "yes"
check "ranked before the CPU arm" \
    "$([ -n "$_dead" ] && [ -n "$_none" ] && [ "$_dead" -lt "$_none" ] && echo yes || echo no)" "yes"
# Matched on the substep ARGUMENT, not any line in the window: the comment above the arm says
# "compute driver" too, so a looser match stayed green with the message itself gutted.
check "arm names the driver fix" \
    "$(awk -v a="$_dead" 'NR>a && NR<a+8 && /substep "[^"]*compute driver/{f=1} END{print (f?"yes":"no")}' "$SETUP_SH")" "yes"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
