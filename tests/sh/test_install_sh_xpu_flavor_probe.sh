#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# install.sh's flavor guard must not launch an interpreter on the XPU path.
#
# Adding "xpu" to _expected_torch_flavor_tag is what made this reachable: the leaf used to
# fall through to "" (custom), so the guard skipped and never ran its `import torch` probe on
# an Intel pin. Now it does -- and `import torch` loads the SYCL runtime, which blocks
# indefinitely on a wedged Intel compute driver. That is precisely the host this guard runs
# on, and it runs BEFORE setup.sh's bounded probes, so install.sh would hang with no timeout
# anywhere to save it.
#
# The function is extracted and executed for real against built venv trees, so the glob, the
# sed and the interpreter/disk split are exercised rather than read.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

awk '/^_installed_torch_version_for_tag\(\) \{/, /^\}$/' "$INSTALL_SH" > "$WORK/fn.sh"
[ -s "$WORK/fn.sh" ] || { echo "FATAL: _installed_torch_version_for_tag not found in $INSTALL_SH" >&2; exit 1; }
# An extraction that lost either arm would make every case below pass vacuously.
grep -q 'torch/version.py' "$WORK/fn.sh" || { echo "FATAL: extraction lost the disk read" >&2; exit 1; }
grep -q 'import torch' "$WORK/fn.sh" || { echo "FATAL: extraction lost the interpreter read" >&2; exit 1; }

PASS=0
FAIL=0
check() {
    if [ "$2" = "$3" ]; then
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %-42s got=[%s] want=[%s]\n' "$1" "$2" "$3"
        FAIL=$((FAIL + 1))
    fi
}

# A venv whose torch/version.py (if $1 is non-empty) reports $1, under python$2.
make_venv() {
    _v="$WORK/venv_$3"
    rm -rf "$_v"
    mkdir -p "$_v/bin"
    if [ -n "$1" ]; then
        mkdir -p "$_v/lib/python$2/site-packages/torch"
        printf "from typing import Optional\n__version__ = '%s'\ndebug = False\n" "$1" \
            > "$_v/lib/python$2/site-packages/torch/version.py"
    fi
    # Stands in for the venv interpreter. Any use of it on the XPU path is a bug, so it
    # reports a value nothing else could produce.
    printf '#!/bin/sh\necho INTERPRETER_WAS_LAUNCHED\n' > "$_v/bin/python"
    chmod +x "$_v/bin/python"
    printf '%s' "$_v"
}

probe() {
    (
        VENV_DIR="$1"
        _VENV_PY="$1/bin/python"
        # shellcheck disable=SC1091
        . "$WORK/fn.sh"
        _installed_torch_version_for_tag "$2"
    )
}

echo "the xpu path reads the label off disk, never from the interpreter"
check "xpu wheel"            "$(probe "$(make_venv '2.9.1+xpu' 3.12 a)" xpu)"    "2.9.1+xpu"
# A migrated venv can still hold a CPU wheel under an xpu pin: that is the mismatch the
# guard exists to repair, so the label has to come back accurately, not empty.
check "stale cpu wheel"      "$(probe "$(make_venv '2.9.1+cpu' 3.12 b)" xpu)"    "2.9.1+cpu"
check "untagged wheel"       "$(probe "$(make_venv '2.9.1' 3.12 c)" xpu)"        "2.9.1"
check "no torch installed"   "$(probe "$(make_venv '' 3.12 d)" xpu)"             ""
check "no venv at all"       "$(probe "$WORK/nope" xpu)"                         ""
# 3.10 sorts before 3.12 in a glob but only one tree exists, so this catches a hardcoded
# python3.12 path rather than the glob the code actually uses.
check "any python minor"     "$(probe "$(make_venv '2.9.1+xpu' 3.10 e)" xpu)"    "2.9.1+xpu"
# The whole point: the interpreter stub prints a token no disk read can produce.
check "xpu launches nothing" "$(probe "$(make_venv '2.9.1+xpu' 3.12 f)" xpu | grep -c INTERPRETER)" "0"
check "missing torch launches nothing" \
    "$(probe "$(make_venv '' 3.12 g)" xpu | grep -c INTERPRETER)" "0"

echo "every other family keeps the interpreter read it has always used"
# Non-Intel behaviour must be byte-identical to before this PR: these hosts have no SYCL
# runtime to wedge on, and torch/version.py is not where a source install reports from.
for _tag in cu128 cu118 rocm ""; do
    check "tag '${_tag:-<empty>}' still asks python" \
        "$(probe "$(make_venv '2.9.1+cu128' 3.12 h)" "$_tag")" "INTERPRETER_WAS_LAUNCHED"
done

echo "the guard is wired to the helper at BOTH reads"
# The second read happens after the repair reinstall; leaving it as a raw `import torch`
# would reintroduce the hang one line later.
check "no raw import torch left in the guard" \
    "$(awk '/^# ── Enforce the installed torch flavor matches/, /^fi$/' "$INSTALL_SH" \
        | grep -c 'import torch; print')" "0"
check "helper used twice in the guard" \
    "$(awk '/^# ── Enforce the installed torch flavor matches/, /^fi$/' "$INSTALL_SH" \
        | grep -c '_installed_torch_version_for_tag')" "2"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
