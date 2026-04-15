#!/usr/bin/env bash

set -euo pipefail

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

# Use bun for install if available (faster), fall back to npm.
_install_ok=false
if command -v bun &>/dev/null; then
    if bun install; then
        _install_ok=true
    else
        echo "⚠ bun install failed, falling back to npm"
        rm -rf node_modules
    fi
fi
if [ "$_install_ok" != "true" ]; then
    if ! npm install; then
        echo "❌ ERROR: package install failed" >&2
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

# 3. Build wheel
python -m build

# 4. Optionally publish
if [ "${1:-}" = "publish" ]; then
    python -m twine upload dist/*
fi
