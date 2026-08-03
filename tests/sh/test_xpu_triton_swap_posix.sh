#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Generic Triton must not shadow torch's XPU Triton on a Linux Intel install.
#
# Both distributions own the top-level `triton` package. Resolving unsloth against a pinned
# +xpu torch pulls BOTH (uv reports pytorch-triton-xpu 3.5.0 alongside triton 3.7.1), and
# install.sh installs unsloth after torch, so the CUDA-oriented build lands last and
# torch.compile loads the wrong library on an Intel GPU.
#
# The order is the fragile part and is what this asserts by execution: fetch, THEN uninstall,
# THEN install. Uninstalling last would delete the shared paths the XPU build had just
# written, because those paths are in generic triton's own RECORD.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

awk '/^# ── Intel XPU: replace generic Triton with torch.s XPU build ──$/, /^fi$/' \
    "$INSTALL_SH" > "$WORK/blk.sh"
[ -s "$WORK/blk.sh" ] || { echo "FATAL: XPU triton block not found in $INSTALL_SH" >&2; exit 1; }
# An extraction that lost the payload would make every case below pass vacuously.
grep -q 'pip download' "$WORK/blk.sh" || { echo "FATAL: extraction lost the pre-fetch" >&2; exit 1; }

awk '/^_torch_index_url_leaf\(\) \{/,/^\}/' "$INSTALL_SH" > "$WORK/leaf.sh"
[ -s "$WORK/leaf.sh" ] || { echo "FATAL: could not extract _torch_index_url_leaf" >&2; exit 1; }

PASS=0
FAIL=0
check() {
    if [ "$2" = "$3" ]; then
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %-40s got=%s want=%s\n' "$1" "$2" "$3"
        FAIL=$((FAIL + 1))
    fi
}

# Fake interpreter: answers the two metadata probes from env, and logs a pip download/uninstall.
make_py() {
    _py="$WORK/py_$3"
    cat > "$_py" <<EOF
#!/bin/sh
case "\$*" in
    *"requires('torch')"*) echo '$1' ;;
    *"d.metadata\['Name'\]"*|*distributions*) echo '$2' ;;
    *"pip download"*)
        echo "DOWNLOAD \$*" >> "$WORK/log"
        [ "\$UNSLOTH_TEST_DL_OK" = 1 ] || exit 1
        for a in \$*; do case "\$prev" in -d) mkdir -p "\$a"; : > "\$a/triton_xpu-3.5.0-py3-none-any.whl" ;; esac; prev="\$a"; done
        ;;
    *"pip uninstall"*) echo "PIPUNINSTALL" >> "$WORK/log" ;;
esac
exit 0
EOF
    chmod +x "$_py"
    printf '%s' "$_py"
}

# Runs the block and echoes the ordered action log, space separated.
run_case() {
    (
        _VENV_PY="$1"
        SKIP_TORCH="${2:-false}"
        # ${3-...}, not ${3:-...}: the empty-index case passes "" deliberately, and the
        # colon form would substitute the default and silently test the wrong thing.
        TORCH_INDEX_URL="${3-https://download.pytorch.org/whl/xpu}"
        C_WARN=""
        export UNSLOTH_TEST_DL_OK="${4:-1}"
        : > "$WORK/log"
        # shellcheck disable=SC2317
        substep() { case "$1" in *"[WARN]"*) echo "WARN" >> "$WORK/log" ;; esac; }
        # shellcheck disable=SC2317
        run_install_cmd() { shift; echo "INSTALL $*" >> "$WORK/log"; return 0; }
        # shellcheck disable=SC2317
        uv() {
            case "$*" in
                *uninstall*) echo "UNINSTALL" >> "$WORK/log" ;;
                *install*)   echo "INSTALL $*" >> "$WORK/log" ;;
            esac
            return 0
        }
        # shellcheck disable=SC1091
        . "$WORK/leaf.sh"; . "$WORK/blk.sh"
        awk '{printf "%s ", $1} END{print ""}' "$WORK/log" | sed 's/ *$//'
    )
}

XPUREQ='pytorch-triton-xpu==3.5.0'
NEWREQ='triton-xpu==3.6.0'
CUREQ='triton==3.7.1'

echo "the swap fires only when generic triton is shadowing an XPU triton"
# The whole point: fetch first, uninstall second, install third.
check "orders fetch, uninstall, install" \
    "$(run_case "$(make_py "$XPUREQ" '3.7.1' a)")" "DOWNLOAD UNINSTALL INSTALL"
# torch 2.10 renamed the distribution; the spec is read from torch, never hardcoded.
check "handles the triton-xpu rename" \
    "$(run_case "$(make_py "$NEWREQ" '3.7.1' b)")" "DOWNLOAD UNINSTALL INSTALL"
check "no generic triton -> no swap" \
    "$(run_case "$(make_py "$XPUREQ" '' c)")" ""
# torch asking for CUDA triton means this is not the +xpu wheel the branch assumes.
check "torch wants generic triton -> no swap" \
    "$(run_case "$(make_py "$CUREQ" '3.7.1' d)")" ""
check "torch declares no triton -> no swap" \
    "$(run_case "$(make_py '' '3.7.1' e)")" ""
check "non-xpu index -> no swap" \
    "$(run_case "$(make_py "$XPUREQ" '3.7.1' f)" false https://download.pytorch.org/whl/cu128)" ""
check "no-torch mode -> no swap" \
    "$(run_case "$(make_py "$XPUREQ" '3.7.1' g)" true)" ""
check "empty index -> no swap" \
    "$(run_case "$(make_py "$XPUREQ" '3.7.1' h)" false '')" ""
# A dead mirror must warn and leave the venv working, never uninstall with nothing to install.
check "fetch fails -> warn, nothing removed" \
    "$(run_case "$(make_py "$XPUREQ" '3.7.1' i)" false https://download.pytorch.org/whl/xpu 0)" "DOWNLOAD WARN"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
