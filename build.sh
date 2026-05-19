#!/usr/bin/env bash

set -euo pipefail

# PyPI/Studio release publishing must use `./build.sh publish` (or an
# equivalent stamp -> build -> verify-dist -> upload flow) so packaged Studio
# artifacts include the display-only Studio release version.

# 1. Build frontend (Vite outputs to dist/)
cd studio/frontend

# Clean stale dist to force a full rebuild
rm -rf dist

# Tailwind v4's oxide scanner respects .gitignore in parent directories.
# Python venvs create a .gitignore with "*" (ignore everything), which
# prevents Tailwind from scanning .tsx source files for class names.
# Temporarily hide any such .gitignore during the build, then restore it.
_HIDDEN_GITIGNORES=()
_dir="$(pwd)"
while [ "$_dir" != "/" ]; do
    _dir="$(dirname "$_dir")"
    if [ -f "$_dir/.gitignore" ] && grep -qx '\*' "$_dir/.gitignore" 2>/dev/null; then
        mv "$_dir/.gitignore" "$_dir/.gitignore._twbuild"
        _HIDDEN_GITIGNORES+=("$_dir/.gitignore")
    fi
done

_restore_gitignores() {
    for _gi in "${_HIDDEN_GITIGNORES[@]+"${_HIDDEN_GITIGNORES[@]}"}"; do
        mv "${_gi}._twbuild" "$_gi" 2>/dev/null || true
    done
}
trap _restore_gitignores EXIT

# Frontend install: Bun --frozen-lockfile first (faster) if a
# committed bun.lock is present, npm ci as the always-available
# fallback. Both run lockfile-strict so the install is byte-
# reproducible from whichever lockfile the chosen package manager
# understands. Build always runs through Node -- avoids bun runtime
# quirks on some platforms. This is build.sh (release wheel build,
# typically CI); we do NOT auto-install bun here -- the calling
# environment should provide it explicitly. The cache-corruption
# recovery ladder mirrors studio/setup.sh.
_install_ok=false
if [ -f bun.lock ] && command -v bun &>/dev/null; then
    _attempts=0
    while [ "$_attempts" -lt 2 ] && [ "$_install_ok" != "true" ]; do
        _attempts=$((_attempts + 1))
        if bun install --frozen-lockfile --no-progress \
            && { [ -x node_modules/.bin/tsc ] || [ -f node_modules/.bin/tsc.exe ]; } \
            && { [ -x node_modules/.bin/vite ] || [ -f node_modules/.bin/vite.exe ]; }; then
            _install_ok=true
            break
        fi
        echo "⚠ bun --frozen-lockfile incomplete (try $_attempts); clearing cache + retrying"
        rm -rf node_modules
        bun pm cache rm >/dev/null 2>&1 || true
    done
fi
if [ "$_install_ok" != "true" ]; then
    echo "→ falling back to npm ci"
    rm -rf node_modules
    if ! npm ci --no-fund --no-audit; then
        echo "❌ ERROR: npm ci failed" >&2
        exit 1
    fi
fi
npm run build       # outputs to studio/frontend/dist/

_restore_gitignores
trap - EXIT

# Validate CSS output -- catch truncated Tailwind builds before packaging
MAX_CSS_SIZE=$(find dist/assets -name '*.css' -exec wc -c {} + 2>/dev/null | sort -n | tail -1 | awk '{print $1}')
if [ -z "$MAX_CSS_SIZE" ]; then
    echo "❌ ERROR: No CSS files were emitted into dist/assets."
    echo "   The frontend build may have failed silently."
    exit 1
fi
if [ "$MAX_CSS_SIZE" -lt 100000 ]; then
    echo "❌ ERROR: Largest CSS file is only $((MAX_CSS_SIZE / 1024))KB (expected >100KB)."
    echo "   Tailwind may not have scanned all source files."
    echo "   Check for .gitignore files blocking the Tailwind oxide scanner."
    exit 1
fi
echo "✅ Frontend CSS validated (${MAX_CSS_SIZE} bytes)"

cd ../..

# 2. Clean old artifacts
rm -rf build dist *.egg-info

# 3. Stamp display-only Studio release metadata for packaged builds.
_STUDIO_BUILD_INFO="studio/backend/utils/_studio_release_build.py"
_STUDIO_BUILD_INFO_BACKUP="$(mktemp)"
cp "$_STUDIO_BUILD_INFO" "$_STUDIO_BUILD_INFO_BACKUP"
_restore_studio_build_info() {
    cp "$_STUDIO_BUILD_INFO_BACKUP" "$_STUDIO_BUILD_INFO" 2>/dev/null || true
    rm -f "$_STUDIO_BUILD_INFO_BACKUP"
}
trap _restore_studio_build_info EXIT

if [ "${1:-}" = "publish" ]; then
    STUDIO_STAMPED_VERSION="$(python scripts/stamp_studio_release.py --require-release)"
else
    STUDIO_STAMPED_VERSION="$(python scripts/stamp_studio_release.py)"
fi

# 4. Build wheel/sdist
python -m build

if [ "${1:-}" = "publish" ]; then
    python scripts/stamp_studio_release.py --verify-dist dist --expected "$STUDIO_STAMPED_VERSION"
fi

_restore_studio_build_info
trap - EXIT

# 5. Optionally publish
if [ "${1:-}" = "publish" ]; then
    python -m twine upload dist/*
fi
