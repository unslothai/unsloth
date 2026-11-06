#!/usr/bin/env bash
# Wheel extraction mtimes can make sources look newer than dist, so packaged installs skip
# the freshness check. A wheel ships no top-level files, so pyproject.toml beside studio/
# marks an editable source tree that keeps the mtime rebuild.
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

SCRIPT_DIR="$WORK/site-packages/studio"
REPO_ROOT="$WORK/site-packages"
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
printf '<!doctype html>\n' > "$SCRIPT_DIR/frontend/dist/index.html"

# Editable overlay: PyPI mode, but $SCRIPT_DIR is a checkout; the mtime check owns it.
SCRIPT_DIR="$WORK/checkout/studio"
REPO_ROOT="$WORK/checkout"
mkdir -p "$SCRIPT_DIR/frontend/dist"
printf '<!doctype html>\n' > "$SCRIPT_DIR/frontend/dist/index.html"
printf '[project]\nname = "unsloth"\n' > "$REPO_ROOT/pyproject.toml"

STUDIO_LOCAL_INSTALL=0
if _packaged_frontend_available; then
    echo "FAIL: source checkout in PyPI mode must retain frontend rebuilds"
    exit 1
fi

rm -f "$REPO_ROOT/pyproject.toml"
if ! _packaged_frontend_available; then
    echo "FAIL: packaged layout should skip once no source marker remains"
    exit 1
fi

echo "All packaged frontend checks passed"
