#!/usr/bin/env bash
# The PyPI wheel already contains the release-built frontend. Wheel extraction
# mtimes can make its source files appear newer than dist, so packaged installs
# must not fall through to the source-build freshness check.
#
# The setup mode alone does not identify a packaged tree: an editable overlay
# (UNSLOTH_CI_SOURCE_OVERLAY, or a venv left editable by an earlier --local
# run) keeps the mode at 0 while setup.sh runs out of a checkout. A wheel ships
# no top-level files, so pyproject.toml beside studio/ marks the tree as source
# and has to keep the mtime rebuild.
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

# The layout a wheel produces: studio/ under site-packages, nothing above it.
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

# Editable overlay: PyPI mode, but $SCRIPT_DIR is a checkout carrying a dist
# left by some earlier build. The mtime check owns this tree, not the skip.
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

# The same tree without the marker is indistinguishable from a packaged one,
# which is what makes the marker load-bearing rather than incidental.
rm -f "$REPO_ROOT/pyproject.toml"
if ! _packaged_frontend_available; then
    echo "FAIL: packaged layout should skip once no source marker remains"
    exit 1
fi

echo "All packaged frontend checks passed"
