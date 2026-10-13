#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
SYSTEM_RM=$(command -v rm)
trap '"$SYSTEM_RM" -rf "$WORK"' EXIT

sed -n '/^_setup_anchor_path() {/,/^}/p' "$SETUP_SH" > "$WORK/anchor_path.sh"
if ! grep -q '^_setup_anchor_path() {' "$WORK/anchor_path.sh"; then
    echo 'FAIL: setup.sh has no _setup_anchor_path function' >&2
    exit 1
fi
if ! grep -Eq '^_setup_anchor_path([[:space:]]|$)' "$SETUP_SH"; then
    echo 'FAIL: setup.sh does not call _setup_anchor_path' >&2
    exit 1
fi

# shellcheck disable=SC1090,SC1091
. "$WORK/anchor_path.sh"

mkdir -p "$WORK/start/bin" "$WORK/start/relative/bin" "$WORK/other"
cat > "$WORK/start/bin/rm" <<'EOF'
#!/bin/sh
printf '%s\n' 'fake rm'
EOF
chmod +x "$WORK/start/bin/rm"

(
    cd "$WORK/start"
    _SETUP_START_PWD=$PWD
    PATH='./bin::relative/bin:/usr/bin'
    hash -p ./bin/rm rm
    _setup_anchor_path

    expected="$WORK/start/bin:$WORK/start:$WORK/start/relative/bin:/usr/bin"
    if [ "$PATH" != "$expected" ]; then
        printf 'FAIL: PATH after anchoring\nexpected: %s\nactual:   %s\n' "$expected" "$PATH" >&2
        exit 1
    fi

    cd "$WORK/other"
    actual=$(rm)
    if [ "$actual" != 'fake rm' ]; then
        printf 'FAIL: rm after cd returned %s\n' "$actual" >&2
        exit 1
    fi
)

echo 'PASS: relative and empty PATH entries stay anchored after cd'
