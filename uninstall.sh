#!/usr/bin/env sh
# Unsloth Studio uninstaller (macOS / Linux / WSL).
# Stops running servers and removes install dir, launcher data,
# CLI shim, desktop shortcut, .app bundle, and Launch Services entry.
# Honors custom roots set via UNSLOTH_STUDIO_HOME / STUDIO_HOME at
# install time (read back from studio.conf).
#
# Usage: curl -fsSL https://unsloth.ai/uninstall.sh | sh

set -e

_pkill_studio() {
    if command -v pkill >/dev/null 2>&1; then
        # Cover both `-p PORT` and `--port PORT` invocation styles.
        for _pat in "unsloth studio.*-p[ =][0-9]" "unsloth studio.*--port[ =][0-9]"; do
            pkill -TERM -f "$_pat" 2>/dev/null || true
        done
        sleep 0.5
        for _pat in "unsloth studio.*-p[ =][0-9]" "unsloth studio.*--port[ =][0-9]"; do
            pkill -KILL -f "$_pat" 2>/dev/null || true
        done
    fi
}

_remove_path() {
    _p="$1"
    if [ -e "$_p" ] || [ -L "$_p" ]; then
        rm -rf "$_p" 2>/dev/null && echo "  removed: $_p" || echo "  could not remove: $_p" >&2
    fi
}

# Read custom STUDIO_HOME from studio.conf if present. install.sh writes
# UNSLOTH_EXE='<root>/unsloth_studio/bin/unsloth', so the install root is
# three dirnames up. Used only to remove non-default-rooted installs.
_custom_studio_home() {
    _conf="$HOME/.local/share/unsloth/studio.conf"
    [ -f "$_conf" ] || return 0
    _exe=$(sed -n "s/^UNSLOTH_EXE='\(.*\)'$/\1/p" "$_conf" | head -n1)
    [ -n "$_exe" ] || return 0
    _root=$(dirname "$(dirname "$(dirname "$_exe")")")
    case "$_root" in
        "$HOME/.unsloth/studio"|"") ;;
        *) printf '%s\n' "$_root" ;;
    esac
}

_uid=$(id -u 2>/dev/null || echo 0)
_os=$(uname 2>/dev/null || echo unknown)
_is_wsl=0
[ "$_os" = "Linux" ] && grep -qi microsoft /proc/version 2>/dev/null && _is_wsl=1

echo "Stopping any running Unsloth Studio servers..."
_pkill_studio

echo "Removing data and install directories..."
_custom_root=$(_custom_studio_home || true)
if [ -n "$_custom_root" ]; then
    _remove_path "$_custom_root"
fi
_remove_path "$HOME/.unsloth/studio"
_remove_path "$HOME/.local/share/unsloth"
# CLI shim (broken symlink once the venv is gone).
_remove_path "$HOME/.local/bin/unsloth"

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
        if [ "$_is_wsl" = "1" ]; then
            echo "Removing WSL Windows-side shortcuts..."
            # install.sh creates 'Unsloth Studio.lnk' on the Windows Desktop and
            # Start Menu Programs folder via powershell.exe; mirror that path.
            if command -v powershell.exe >/dev/null 2>&1; then
                # shellcheck disable=SC2016
                # $env:APPDATA is a PowerShell expansion; intentionally literal at shell level.
                powershell.exe -NoProfile -Command '
                    $names = @("Desktop","StartMenu");
                    $dirs = @(
                        [Environment]::GetFolderPath("Desktop"),
                        (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs")
                    );
                    foreach ($d in $dirs) {
                        if (-not $d) { continue }
                        $p = Join-Path $d "Unsloth Studio.lnk";
                        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
                    }' >/dev/null 2>&1 || true
            fi
        fi
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
