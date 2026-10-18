#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A package index that lags PyPI must not make every install unsatisfiable (#14031): install.sh lowers its
# unsloth / unsloth-zoo floors to the newest release that index has, and only then. uv is a stub.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"
_FUNC_FILE=$(mktemp)
for fn in _mirror_uv_project_config _mirror_configured _relax_unsloth_floors_to_index; do
    sed -n "/^${fn}()/,/^}/p" "$INSTALL_SH" >> "$_FUNC_FILE"
done
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

VENV_DIR=$(mktemp -d)
_VENV_PY="$VENV_DIR/bin/python"
_UNSLOTH_FEED_CEILING_MARKER="$VENV_DIR/unsloth_feed_ceiling.json"
CALLS="$VENV_DIR/calls"
SUBSTEPS="$VENV_DIR/substeps"
substep() { echo "$1" >> "$SUBSTEPS"; }
export XDG_CONFIG_HOME="$VENV_DIR/xdg" UV_CONFIG_FILE="" XDG_CONFIG_DIRS=""
unset UV_DEFAULT_INDEX UV_INDEX_URL UV_INDEX UV_EXTRA_INDEX_URL UV_EXCLUDE_NEWER UV_EXCLUDE_NEWER_PACKAGE
cd "$VENV_DIR"

# The index serves unsloth $FEED_UNSLOTH and unsloth-zoo $FEED_ZOO; a floor above either is unsatisfiable,
# and FEED_DOWN=1 fails every resolve (an unreachable index).
uv() {
    echo "$*" >> "$CALLS"
    _req=$(cat)
    [ "${FEED_DOWN:-0}" = 1 ] && return 2
    case "$_req" in *"unsloth>=$NEED_UNSLOTH"*|*"unsloth-zoo>=$NEED_ZOO"*)
        [ "$FEED_UNSLOTH" = "$NEED_UNSLOTH" ] && [ "$FEED_ZOO" = "$NEED_ZOO" ] || return 1 ;;
    esac
    printf 'unsloth==%s\nunsloth-zoo==%s\n' "$FEED_UNSLOTH" "$FEED_ZOO"
}
NEED_UNSLOTH=2026.10.6; NEED_ZOO=2026.10.6

_run() {
    : > "$CALLS"; : > "$SUBSTEPS"
    _unsloth_desktop_install_spec=""
    [ -n "${UNSLOTH_DESKTOP_BACKEND_VERSION:-}" ] && _unsloth_desktop_install_spec="unsloth>=$UNSLOTH_DESKTOP_BACKEND_VERSION"
    _unsloth_release_install_spec="${_unsloth_desktop_install_spec:-unsloth>=$NEED_UNSLOTH}"
    _unsloth_zoo_install_spec="unsloth-zoo>=$NEED_ZOO"
    _relax_unsloth_floors_to_index
}

echo "=== no custom index: no probe, floors kept ==="
FEED_UNSLOTH=2026.10.3; FEED_ZOO=2026.10.3
echo stale > "$_UNSLOTH_FEED_CEILING_MARKER"
_run
assert_eq "release floor kept" "unsloth>=2026.10.6" "$_unsloth_release_install_spec"
assert_eq "zoo floor kept" "unsloth-zoo>=2026.10.6" "$_unsloth_zoo_install_spec"
assert_eq "uv never asked" "0" "$(wc -l < "$CALLS" | tr -d ' ')"
assert_eq "a leftover marker is dropped" "no" "$([ -f "$_UNSLOTH_FEED_CEILING_MARKER" ] && echo yes || echo no)"

echo "=== index that has the release: floors kept, one probe ==="
export UV_DEFAULT_INDEX=https://feed.example/simple
FEED_UNSLOTH=2026.10.6; FEED_ZOO=2026.10.6
_run
assert_eq "release floor kept" "unsloth>=2026.10.6" "$_unsloth_release_install_spec"
assert_eq "one probe" "1" "$(wc -l < "$CALLS" | tr -d ' ')"
assert_contains "probe ignores dependencies" "$(cat "$CALLS")" "pip compile --python $_VENV_PY --no-deps"
assert_eq "nothing printed" "" "$(cat "$SUBSTEPS")"

echo "=== index a week behind: floors lowered to what it has ==="
FEED_UNSLOTH=2026.10.3; FEED_ZOO=2026.10.2
_run
assert_eq "release floor lowered" "unsloth>=2026.10.3" "$_unsloth_release_install_spec"
assert_eq "zoo floor lowered" "unsloth-zoo>=2026.10.2" "$_unsloth_zoo_install_spec"
assert_contains "says why" "$(cat "$SUBSTEPS")" "does not have unsloth>=2026.10.6 yet; installing unsloth 2026.10.3"
assert_eq "no marker without Desktop" "no" "$([ -f "$_UNSLOTH_FEED_CEILING_MARKER" ] && echo yes || echo no)"

echo "=== Desktop on that index: its floor too, and the marker it reads ==="
UNSLOTH_DESKTOP_BACKEND_VERSION=2026.10.6
_run
assert_eq "desktop floor lowered" "unsloth>=2026.10.3" "$_unsloth_desktop_install_spec"
assert_eq "release floor follows" "unsloth>=2026.10.3" "$_unsloth_release_install_spec"
assert_eq "marker" '{"requested": "2026.10.6", "available": "2026.10.3"}' "$(cat "$_UNSLOTH_FEED_CEILING_MARKER")"
_marker_ok=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["requested"], d["available"])' "$_UNSLOTH_FEED_CEILING_MARKER")
assert_eq "marker is JSON" "2026.10.6 2026.10.3" "$_marker_ok"

echo "=== feed caught up: marker cleared on the next run ==="
FEED_UNSLOTH=2026.10.6; FEED_ZOO=2026.10.6
_run
assert_eq "desktop floor kept" "unsloth>=2026.10.6" "$_unsloth_desktop_install_spec"
assert_eq "marker gone" "no" "$([ -f "$_UNSLOTH_FEED_CEILING_MARKER" ] && echo yes || echo no)"
unset UNSLOTH_DESKTOP_BACKEND_VERSION

echo "=== unreachable index: floors kept, the install reports the real error ==="
FEED_DOWN=1
_run
assert_eq "release floor kept" "unsloth>=2026.10.6" "$_unsloth_release_install_spec"
assert_eq "zoo floor kept" "unsloth-zoo>=2026.10.6" "$_unsloth_zoo_install_spec"
FEED_DOWN=0

echo "=== upload cutoff alone also counts ==="
unset UV_DEFAULT_INDEX
export UV_EXCLUDE_NEWER=2026-10-09T00:00:00Z
FEED_UNSLOTH=2026.10.3; FEED_ZOO=2026.10.3
_run
assert_eq "release floor lowered" "unsloth>=2026.10.3" "$_unsloth_release_install_spec"
unset UV_EXCLUDE_NEWER

echo "=== index from uv.toml ==="
mkdir -p "$XDG_CONFIG_HOME/uv"
printf '[[index]]\nurl = "https://feed.example/simple"\ndefault = true\n' > "$XDG_CONFIG_HOME/uv/uv.toml"
_run
assert_eq "release floor lowered" "unsloth>=2026.10.3" "$_unsloth_release_install_spec"

cd /
rm -rf "${VENV_DIR:?}"
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
