#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The POSIX hardware summary must recognise a working Intel XPU runtime, and `studio update`
# must raise the bitsandbytes floor for it.
#
# Two gaps this covers. The summary only tested NVIDIA, AMD and Apple Silicon, so a Linux host
# running the +xpu wheel install.sh had just installed was told "none (chat-only / GGUF)" and
# that training needs NVIDIA or AMD. And `unsloth studio update` runs setup.sh, never
# install.sh, so the XPU bitsandbytes floor there was unreachable on the one route an existing
# XPU user actually takes -- unsloth's own floor is 0.45.5, which a pre-XPU wheel satisfies
# forever.
#
# Unlike the Windows half this runs for real: the venv tree, torch/version.py and the
# interpreter are all built here, so the disk read and the runtime probe are genuinely
# executed rather than mocked.
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
# A +xpu wheel installs fine on a host whose driver never initialises: the wheel must raise
# the bnb floor but must NOT claim a GPU in the summary.
check "xpu wheel + runtime dead" "$(run_case "$(make_venv '2.9.1+xpu' broken b)")" "true false yes"
check "cuda wheel"               "$(run_case "$(make_venv '2.9.1+cu128' ok c)")"   "false false no"
check "rocm wheel"               "$(run_case "$(make_venv '2.9.1+rocm6.4' ok d)")" "false false no"
check "untagged wheel"           "$(run_case "$(make_venv '2.9.1' ok e)")"         "false false no"
check "no torch installed"       "$(run_case "$(make_venv '' ok f)")"              "false false no"
check "no venv at all"           "$(run_case "$WORK/nope")"                        "false false no"
# The interpreter is missing but the wheel is there: no crash, no GPU claim, floor still raised.
check "xpu wheel, no interpreter" "$(run_case "$(make_venv '2.9.1+xpu' none g)")"  "true false yes"

echo "the summary has an XPU arm, ranked below NVIDIA and AMD"
# Anchored at column 0: both flags are re-tested later inside indented blocks, and matching
# one of those would compare the arm against the wrong line and pass or fail by accident.
_arm=$(grep -n '^elif \[ "\$_setup_xpu_ready" = true \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
_nv=$(grep -n '^if \[ "\$_setup_nvidia_usable" = true \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
_amd=$(grep -n '^elif \[ "\$_setup_amd_detected" = true \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
_none=$(grep -n 'none (chat-only / GGUF)' "$SETUP_SH" | head -1 | cut -d: -f1)
check "arm exists"               "$([ -n "$_arm" ] && echo yes || echo no)" "yes"
check "ranked after NVIDIA"      "$([ -n "$_arm" ] && [ -n "$_nv" ] && [ "$_arm" -gt "$_nv" ] && echo yes || echo no)" "yes"
check "ranked after AMD"         "$([ -n "$_arm" ] && [ -n "$_amd" ] && [ "$_arm" -gt "$_amd" ] && echo yes || echo no)" "yes"
check "ranked before the CPU arm" "$([ -n "$_arm" ] && [ -n "$_none" ] && [ "$_arm" -lt "$_none" ] && echo yes || echo no)" "yes"
# The floor must run before the summary; the other order would report on a stale venv.
_bnb=$(grep -n 'install bitsandbytes (xpu)' "$SETUP_SH" | head -1 | cut -d: -f1)
check "floor precedes the summary" "$([ -n "$_bnb" ] && [ -n "$_none" ] && [ "$_bnb" -lt "$_none" ] && echo yes || echo no)" "yes"
# run_quiet routes failure to setup_fail and EXITS, so using it here would abort an otherwise
# fine `studio update` over a best-effort step and make the warning below unreachable.
check "floor uses the nonfatal wrapper" \
    "$(grep -q 'run_quiet_no_exit "install bitsandbytes (xpu)"' "$SETUP_SH" && echo yes || echo no)" "yes"
# An unbounded `import torch` hangs forever on a stalled Intel driver, which is precisely the
# host this probe exists to classify.
check "runtime probe is bounded" \
    "$(grep -q 'timeout 60 "\$VENV_DIR/bin/python" -c "\$_setup_xpu_probe"' "$SETUP_SH" && echo yes || echo no)" "yes"

# The manifest fast path skips install_python_stack entirely when the package version is
# current, and that pass is the ONLY thing that acts on an XPU pin, so without an escape a CPU
# install switched to the xpu family keeps its CPU wheel forever.
_esc=$(grep -n 'XPU index pinned but torch does not match' "$SETUP_SH" | head -1 | cut -d: -f1)
_gate=$(grep -n '^if \[ "\$_SKIP_PYTHON_DEPS" = false \]; then' "$SETUP_SH" | head -1 | cut -d: -f1)
check "fast path has an XPU escape" "$([ -n "$_esc" ] && echo yes || echo no)" "yes"
check "escape precedes the skip gate" \
    "$([ -n "$_esc" ] && [ -n "$_gate" ] && [ "$_esc" -lt "$_gate" ] && echo yes || echo no)" "yes"
# It must clear the flag, not merely warn.
check "escape forces the dependency pass" \
    "$(awk -v a="$_esc" 'NR>=a && NR<=a+2 && /_SKIP_PYTHON_DEPS=false/{f=1} END{print (f?"yes":"no")}' "$SETUP_SH")" "yes"

# An authenticated or fragmented mirror is a supported pin shape; a raw suffix test reads it
# as "no XPU pin" and skips the repair entirely.
check "pin match strips the query" \
    "$(grep -q '_setup_pin="\${_setup_pin%%\\?\*}"' "$SETUP_SH" && echo yes || echo no)" "yes"
check "pin match strips the fragment" \
    "$(grep -q '_setup_pin="\${_setup_pin%%\\#\*}"' "$SETUP_SH" && echo yes || echo no)" "yes"
# The escape must not launch an interpreter: a wedged Intel driver hangs inside `import torch`,
# and this runs before the bounded probes.
check "escape reads the flavour off disk" \
    "$(awk -v a="$_esc" 'NR>=a-24 && NR<=a && /site-packages\/torch\/version.py/{f=1} END{print (f?"yes":"no")}' "$SETUP_SH")" "yes"
check "escape launches no interpreter" \
    "$(awk -v a="$_esc" 'NR>=a-24 && NR<=a && /\$VENV_DIR\/bin\/python/{f=1} END{print (f?"no":"yes")}' "$SETUP_SH")" "yes"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
