#!/usr/bin/env bash
# The PyPI wheel already contains the release-built frontend. Wheel extraction
# mtimes can make its source files appear newer than dist, so packaged installs
# must not fall through to the source-build freshness check.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
SETUP_SH="$ROOT/studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

sed -n '/^_packaged_frontend_available() {/,/^}/p' "$SETUP_SH" > "$WORK/helper.sh"
grep -q '^_packaged_frontend_available() {' "$WORK/helper.sh" || {
    echo "FAIL: packaged frontend helper not found"
    exit 1
}
# shellcheck disable=SC1090
. "$WORK/helper.sh"

SCRIPT_DIR="$WORK/studio"
mkdir -p "$SCRIPT_DIR/frontend/dist"
printf '<!doctype html>\n' > "$SCRIPT_DIR/frontend/dist/index.html"

STUDIO_LOCAL_INSTALL=0
_packaged_frontend_available || {
    echo "FAIL: PyPI install with packaged index should skip the build"
    exit 1
}

STUDIO_LOCAL_INSTALL=1
if _packaged_frontend_available; then
    echo "FAIL: local/source install must retain frontend rebuilds"
    exit 1
fi

unset STUDIO_LOCAL_INSTALL
if _packaged_frontend_available; then
    echo "FAIL: unspecified setup mode must retain frontend rebuilds"
    exit 1
fi

STUDIO_LOCAL_INSTALL=0
rm -f "$SCRIPT_DIR/frontend/dist/index.html"
if _packaged_frontend_available; then
    echo "FAIL: missing packaged index must fall back to a frontend build"
    exit 1
fi

echo "All packaged frontend checks passed"
