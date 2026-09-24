#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The macOS trace leg's `notools` lets git fetch the pinned git+ requirements and nothing else.
#
# A default install with a working git clones the pinned Diffusers main build through uv, so the
# trace records git. `notools` rejected every git line, and the leg went red on the first
# installer PR after the build became the default. The allowance is structural: each git line
# may name only a remote from the requirement files in UNSLOTH_ALLOW_GIT_FROM, and a remoteless
# line must be one of uv's own cache operations, counted only when an allowed remote was
# fetched. The first trace below is the one that leg recorded.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
ASSERT_SH="$REPO_ROOT/.github/scripts/clean-machine-assert.sh"
PIN="$REPO_ROOT/studio/backend/requirements/diffusers-main.txt"
ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT
PASS=0
FAIL=0

expect_rc() {
    _label="$1"; _expected="$2"; _allow="$3"; _trace_content="$4"
    printf '%b' "$_trace_content" > "$ROOT/trace.log"
    set +e
    UV_CACHE_DIR="$UV_CACHE" UNSLOTH_ALLOW_GIT_FROM="$_allow" UNSLOTH_TOOL_TRACE="$ROOT/trace.log" \
        INSTALL_LOG="$ROOT/install.log" bash "$ASSERT_SH" notools \
        > "$ROOT/out.log" 2>&1
    _actual=$?
    set -e
    if [ "$_actual" -eq "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected rc $_expected, got $_actual)"
        cat "$ROOT/out.log"
        FAIL=$((FAIL + 1))
    fi
}

: > "$ROOT/install.log"
REMOTE=$(sed -n 's/^[^#]*git+\(https:\/\/[^@]*\)@.*/\1/p' "$PIN")
SHA=$(sed -n 's/^[^#]*git+https:\/\/[^@]*@\([0-9a-f]*\).*/\1/p' "$PIN")
[ -n "$REMOTE" ] && [ -n "$SHA" ] || { echo "no git+ pin in $PIN"; exit 1; }
UV_CACHE=/Users/runner/work/unsloth/unsloth/.clean-machine/uv-cache
CACHE=$UV_CACHE/git-v0

# The macOS trace / file leg's record, with the pin read from the file it came from.
UV_CLONE="git\t--version
git\t-c remote.origin.url=$REMOTE submodule update --init
git\tclone --local $CACHE/db/76e25d04238765dd $CACHE/checkouts/76e25d04238765dd/${SHA:0:9}
git\tfetch --tags --force --update-head-ok $REMOTE +refs/heads/*:refs/remotes/origin/* +HEAD:refs/remotes/origin/HEAD
git\tinit
git\treset --hard $SHA
git\trev-parse
git\trev-parse --short $SHA
git\trev-parse $SHA^0
git\trev-parse HEAD
git\tsubmodule update --recursive --init
xcode-select\t-p
"

echo "=== pinned git+ remotes ==="
expect_rc "uv cloning the pinned Diffusers build passes" 0 "$PIN" "$UV_CLONE"
expect_rc "the same trace fails with no allowance, as before" 1 "" "$UV_CLONE"
expect_rc "an allowance file that does not exist allows nothing" 1 "$ROOT/missing.txt" "$UV_CLONE"
expect_rc "the installer's git --version probe alone passes" 0 "$PIN" "git\t--version\n"
expect_rc "a remote with the .git suffix dropped still matches" 0 "$PIN" \
    "git\tfetch ${REMOTE%.git} +HEAD:refs/remotes/origin/HEAD\n"

echo "=== everything else is still a hit ==="
expect_rc "cloning any other remote fails" 1 "$PIN" \
    "${UV_CLONE}git\tclone https://github.com/someone/else.git /tmp/else\n"
expect_rc "an scp-style remote that is not pinned fails" 1 "$PIN" \
    "${UV_CLONE}git\tfetch git@github.com:someone/else.git\n"
expect_rc "a config-injected remote that is not pinned fails" 1 "$PIN" \
    "${UV_CLONE}git\t-c remote.origin.url=https://example.com/x.git submodule update\n"
expect_rc "a lookalike host that only starts with the pin fails" 1 "$PIN" \
    "git\tfetch ${REMOTE%.git}-fork.git +HEAD:refs/remotes/origin/HEAD\n"
expect_rc "remoteless git with no allowed fetch fails" 1 "$PIN" \
    "git\tinit\ngit\trev-parse HEAD\n"
expect_rc "a remoteless fetch beside the allowed one fails (its URL is in config)" 1 "$PIN" \
    "${UV_CLONE}git\tfetch origin\n"
expect_rc "a local clone outside uv's cache fails" 1 "$PIN" \
    "${UV_CLONE}git\tclone /tmp/unrelated /tmp/out\n"
expect_rc "a local clone that climbs out of uv's cache fails" 1 "$PIN" \
    "${UV_CLONE}git\tclone --local $CACHE/db/x/../../../x $CACHE/checkouts/x/y\n"
expect_rc "a reset to a commit nothing pins fails" 1 "$PIN" \
    "${UV_CLONE}git\treset --hard 0000000000000000000000000000000000000000\n"
expect_rc "any other remoteless subcommand fails" 1 "$PIN" \
    "${UV_CLONE}git\tpull\n"
expect_rc "an allowed remote with an extra -c option fails" 1 "$PIN" \
    "${UV_CLONE}git\t-c core.sshCommand=evil fetch $REMOTE\n"
expect_rc "pushing to the allowed remote fails" 1 "$PIN" \
    "${UV_CLONE}git\tpush $REMOTE HEAD:refs/heads/oops\n"
expect_rc "cloning the allowed remote outside uv's fetch fails" 1 "$PIN" \
    "${UV_CLONE}git\tclone $REMOTE /tmp/out\n"
expect_rc "a -c after the subcommand fails too" 1 "$PIN" \
    "${UV_CLONE}git\tfetch -c core.sshCommand=evil $REMOTE\n"
expect_rc "a compiler next to the allowed clone still fails" 1 "$PIN" \
    "${UV_CLONE}clang\t-c foo.c\n"
expect_rc "brew next to the allowed clone still fails" 1 "$PIN" \
    "${UV_CLONE}brew\tinstall cmake\n"

echo ""
echo "Passed: $PASS, Failed: $FAIL"
[ "$FAIL" -eq 0 ]
