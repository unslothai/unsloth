#!/usr/bin/env sh
# Unsloth Studio uninstaller (macOS / Linux / WSL).
# Stops running servers and removes install dir, launcher data,
# desktop shortcut, .app bundle, and Launch Services entry.
#
# Usage: curl -fsSL https://unsloth.ai/uninstall.sh | sh

set -e

_pkill_studio() {
    if command -v pkill >/dev/null 2>&1; then
        pkill -TERM -f "unsloth studio -p [0-9]" 2>/dev/null || true
        sleep 0.5
        pkill -KILL -f "unsloth studio -p [0-9]" 2>/dev/null || true
    fi
}

_remove_path() {
    _p="$1"
    if [ -e "$_p" ] || [ -L "$_p" ]; then
        rm -rf "$_p" 2>/dev/null && echo "  removed: $_p" || echo "  could not remove: $_p" >&2
    fi
}

_uid=$(id -u 2>/dev/null || echo 0)
_os=$(uname 2>/dev/null || echo unknown)

echo "Stopping any running Unsloth Studio servers..."
_pkill_studio

echo "Removing data and install directories..."
_remove_path "$HOME/.unsloth/studio"
_remove_path "$HOME/.local/share/unsloth"

echo "Removing desktop shortcut and launcher lock..."
_remove_path "$HOME/Desktop/Unsloth Studio"
_remove_path "$HOME/Desktop/unsloth-studio.desktop"
# Locks are namespaced per-uid; env-mode adds an extra suffix.
_lock_glob="${XDG_RUNTIME_DIR:-/tmp}/unsloth-studio-launcher-${_uid}"
for _lock in "$_lock_glob".lock "$_lock_glob"-*.lock; do
    [ -e "$_lock" ] && _remove_path "$_lock"
done

case "$_os" in
    Darwin)
        echo "Removing macOS .app bundle and Launch Services entry..."
        _remove_path "$HOME/Applications/Unsloth Studio.app"
        _lsr="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
        if [ -x "$_lsr" ]; then
            "$_lsr" -u "$HOME/Applications/Unsloth Studio.app" 2>/dev/null || true
        fi
        ;;
    Linux)
        echo "Removing Linux .desktop entry..."
        _remove_path "$HOME/.local/share/applications/unsloth-studio.desktop"
        if command -v update-desktop-database >/dev/null 2>&1; then
            update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
        fi
        ;;
esac

echo ""
echo "Unsloth Studio uninstalled."
echo "Note: Hugging Face model cache at ~/.cache/huggingface was left in place."
echo "Remove it manually with 'rm -rf ~/.cache/huggingface/hub' if desired."
