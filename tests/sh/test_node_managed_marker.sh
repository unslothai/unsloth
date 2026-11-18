#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Installer metadata beside the PATH Node routes it through the pin check when NODE_DIR differs.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="${SETUP_SH:-$HERE/../../studio/setup.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
source "$HERE/_harness.sh"

sed -n '/^decide_node_source()/,/^}/p' "$SETUP_SH" > "$WORK/decision"
sed -n '/^_probe_system_node_tool node$/,/^_FRONTEND_SKIP=false$/p' "$SETUP_SH" > "$WORK/selection"
for part in decision selection; do
    [ -s "$WORK/$part" ] || { echo "Cannot extract $part from setup.sh"; exit 1; }
done

for who in live external; do
    mkdir -p "$WORK/$who/node/bin"
    printf '#!/bin/sh\necho v24.18.0\n' > "$WORK/$who/node/bin/node"
    printf '#!/bin/sh\necho 11.0.0\n' > "$WORK/$who/node/bin/npm"
    chmod +x "$WORK/$who/node/bin/node" "$WORK/$who/node/bin/npm"
done
echo '{"version": "24.18.0"}' > "$WORK/live/node/UNSLOTH_NODE_PREBUILT_INFO.json"

source_for() { # node dir on PATH, NODE_DIR, UNSLOTH_SKIP_NODE_INSTALL
    PATH="$1/bin:$PATH" UNSLOTH_SKIP_NODE_INSTALL="$3" bash -c "
        set -euo pipefail
        step() { :; }; substep() { :; }
        _probe_system_node_tool() { _PROBED_VER=\$(\"\$1\" -v); }
        $(cat "$WORK/decision")
        NODE_DIR='$2'
        $(cat "$WORK/selection")
        echo \"\$NODE_SOURCE\""
}

assert_eq 'staged update: the live managed Node gets the pin check' bundled \
    "$(source_for "$WORK/live/node" "$WORK/stage/node" 0)"
assert_eq 'staged update: a compatible external Node stays read-only' system \
    "$(source_for "$WORK/external/node" "$WORK/stage/node" 0)"
assert_eq 'staged update: the skip flag still keeps the managed Node' system \
    "$(source_for "$WORK/live/node" "$WORK/stage/node" 1)"
assert_eq 'same dir: the managed Node gets the pin check' bundled \
    "$(source_for "$WORK/live/node" "$WORK/live/node" 0)"
# A user's own Node linked in at the managed location has no installer metadata: never replaced.
ln -s "$WORK/external/node" "$WORK/linked"
assert_eq 'a user Node linked in at NODE_DIR stays read-only' system \
    "$(source_for "$WORK/linked" "$WORK/linked" 0)"

summary
