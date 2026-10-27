#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Exercise the real setup selection and re-exec across a pin-only package update.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="${SETUP_SH:-$HERE/../../studio/setup.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
source "$HERE/_harness.sh"

sed -n '/^_SETUP_SELF=/,/^_SETUP_START_PWD=/p' "$SETUP_SH" > "$WORK/start"
sed -n '/^decide_node_source()/,/^}/p' "$SETUP_SH" > "$WORK/decision"
sed -n '/^_probe_system_node_tool node$/,/^_FRONTEND_SKIP=false$/p' "$SETUP_SH" > "$WORK/selection"
sed -n '/^_setup_rerun_if_replaced()/,/^}/p' "$SETUP_SH" > "$WORK/rerun"
for part in start decision selection rerun; do
    [ -s "$WORK/$part" ] || { echo "Cannot extract $part from setup.sh"; exit 1; }
done

{
    cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EOF
    cat "$WORK/start" "$WORK/decision" "$WORK/rerun"
    cat <<'EOF'
NODE_DIR="$SCRIPT_DIR/node"
step() { echo "RERUN"; }
substep() { :; }
_frontend_bg_join() { :; }
_probe_system_node_tool() { _PROBED_VER=$("$1" -v); }
EOF
    cat "$WORK/selection"
    cat <<'EOF'
echo "SOURCE $NODE_SOURCE"
# Stand-in for archive I/O: record replacements, not just installer invocations.
# The installer itself is covered by test_install_node_prebuilt_logic.py.
if [ "$NODE_SOURCE" = bundled ]; then
    if ! cmp -s "$SCRIPT_DIR/node_prebuilt_pins.json" "$NODE_DIR/installed-pin"; then
        cp "$SCRIPT_DIR/node_prebuilt_pins.json" "$NODE_DIR/installed-pin"
        echo "DOWNLOAD $(cat "$NODE_DIR/installed-pin")"
    fi
fi
# Simulate uv replacing the manifest while setup.sh stays byte-identical.
if [ "${UNSLOTH_SETUP_RERUN:-}" != 1 ]; then
    cp "$SCRIPT_DIR/next-pin" "$SCRIPT_DIR/node_prebuilt_pins.json"
fi
_setup_rerun_if_replaced
EOF
} > "$WORK/template.sh"

run_case() {
    local name=$1 target=$2 source=$3 skip=$4
    local root="$WORK/$name"
    mkdir -p "$root/node/bin" "$root/external/bin"
    cp "$WORK/template.sh" "$root/setup.sh"
    printf '%s\n' old > "$root/node_prebuilt_pins.json"
    printf '%s\n' old > "$root/node/installed-pin"
    echo '{}' > "$root/node/UNSLOTH_NODE_PREBUILT_INFO.json" # the installer always writes it
    printf '%s\n' "$target" > "$root/next-pin"
    for bin in "$root/node/bin" "$root/external/bin"; do
        printf '#!/bin/sh\necho v24.18.0\n' > "$bin/node"
        printf '#!/bin/sh\necho 11.0.0\n' > "$bin/npm"
        chmod +x "$bin/node" "$bin/npm"
    done
    local bin="$root/$source/bin"
    if [ "$source" = alias ]; then
        ln -s "$root/node" "$root/alias"
    fi
    OUT=$(env -u UNSLOTH_SETUP_RERUN PATH="$bin:$PATH" \
        UNSLOTH_SKIP_NODE_INSTALL="$skip" bash "$root/setup.sh")
    INSTALLED=$(cat "$root/node/installed-pin")
}

run_case managed new node 0
assert_eq 'managed Node gets the new pin on the first update' new "$INSTALLED"
assert_eq 'a pin-only change reruns setup once' 1 "$(printf '%s\n' "$OUT" | grep -c '^RERUN' || true)"
assert_eq 'only the new archive is downloaded' 1 "$(printf '%s\n' "$OUT" | grep -c '^DOWNLOAD new' || true)"

run_case unchanged old node 0
assert_eq 'unchanged pin keeps Node' old "$INSTALLED"
assert_eq 'unchanged pin never downloads' 0 "$(printf '%s\n' "$OUT" | grep -c '^DOWNLOAD' || true)"
assert_eq 'unchanged setup and pin do not rerun' 0 "$(printf '%s\n' "$OUT" | grep -c '^RERUN' || true)"

run_case symlink new alias 0
assert_eq 'managed Node through a directory symlink gets the new pin' new "$INSTALLED"

run_case external new external 0
assert_eq 'compatible external Node does not replace the managed cache' old "$INSTALLED"
assert_eq 'external runtime remains the source in both passes' 2 "$(printf '%s\n' "$OUT" | grep -c '^SOURCE system' || true)"
assert_eq 'external Node causes no managed download' 0 "$(printf '%s\n' "$OUT" | grep -c '^DOWNLOAD' || true)"

run_case opted_out new node 1
assert_eq 'skip flag keeps the old managed runtime' old "$INSTALLED"
assert_eq 'skip flag prevents downloads in both passes' 0 "$(printf '%s\n' "$OUT" | grep -c '^DOWNLOAD' || true)"

summary
