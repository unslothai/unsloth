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

# Resolve a custom install root from any of:
#   1. UNSLOTH_STUDIO_HOME / STUDIO_HOME env vars at uninstall time
#   2. Default-mode studio.conf at $HOME/.local/share/unsloth/studio.conf
#   3. Env-mode studio.conf at $<root>/share/studio.conf (discovered via 1)
# install.sh writes UNSLOTH_EXE='<root>/unsloth_studio/bin/unsloth', so
# the install root is three dirnames up. Prints each discovered non-default
# root on its own line; the caller iterates and de-duplicates.
_custom_studio_roots() {
    _seen=""
    _emit() {
        _r="$1"
        [ -z "$_r" ] && return 0
        case "$_r" in "$HOME/.unsloth/studio"|/|"") return 0 ;; esac
        case ":$_seen:" in *":$_r:"*) return 0 ;; esac
        _seen="$_seen:$_r"
        printf '%s\n' "$_r"
    }
    _from_conf() {
        [ -f "$1" ] || return 0
        # Tolerate paths containing apostrophes (install.sh emits '\'' for them).
        _exe=$(sed -n "s/^UNSLOTH_EXE='\(.*\)'\$/\1/p" "$1" | head -n1)
        _exe=$(printf '%s' "$_exe" | sed "s/'\\\\''/'/g")
        [ -n "$_exe" ] || return 0
        _emit "$(dirname "$(dirname "$(dirname "$_exe")")")"
    }
    # 1. env vars (UNSLOTH_STUDIO_HOME wins over STUDIO_HOME, matching install.sh).
    if [ -n "${UNSLOTH_STUDIO_HOME:-}" ]; then
        _emit "$UNSLOTH_STUDIO_HOME"
        _from_conf "$UNSLOTH_STUDIO_HOME/share/studio.conf"
    fi
    if [ -n "${STUDIO_HOME:-}" ]; then
        _emit "$STUDIO_HOME"
        _from_conf "$STUDIO_HOME/share/studio.conf"
    fi
    # 2. default-mode conf.
    _from_conf "$HOME/.local/share/unsloth/studio.conf"
}

# Remove $HOME/.local/bin/unsloth only if it's a Studio-managed symlink.
# Studio's install.sh writes this as a symlink into the studio venv
# (install.sh: `ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path"`). A
# pip-installed `unsloth` CLI is a regular file — leave it alone to avoid
# wiping an unrelated install.
_remove_cli_shim() {
    _shim="$HOME/.local/bin/unsloth"
    [ -L "$_shim" ] || return 0
    _target=$(readlink "$_shim" 2>/dev/null || true)
    case "$_target" in
        */unsloth_studio/bin/unsloth) _remove_path "$_shim" ;;
        *) ;;
    esac
}

_uid=$(id -u 2>/dev/null || echo 0)
_os=$(uname 2>/dev/null || echo unknown)
_is_wsl=0
[ "$_os" = "Linux" ] && grep -qi microsoft /proc/version 2>/dev/null && _is_wsl=1

echo "Stopping any running Unsloth Studio servers..."
_pkill_studio

echo "Removing data and install directories..."
_custom_studio_roots | while IFS= read -r _custom_root; do
    [ -n "$_custom_root" ] && _remove_path "$_custom_root"
done
_remove_path "$HOME/.unsloth/studio"
_remove_path "$HOME/.local/share/unsloth"
# CLI shim: only the symlink Studio created, never a pip-installed file.
_remove_cli_shim

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
# Env-mode installs leave no breadcrumb in $HOME, so a custom root can
# only be located if the user re-exports the variable. Print a hint when
# neither var is set so the bare `curl | sh` flow doesn't silently miss.
if [ -z "${UNSLOTH_STUDIO_HOME:-}" ] && [ -z "${STUDIO_HOME:-}" ]; then
    echo ""
    echo "If you installed Unsloth Studio with UNSLOTH_STUDIO_HOME or STUDIO_HOME"
    echo "pointing at a custom directory, re-run this script with the same variable"
    echo "set to also remove that install tree, e.g.:"
    echo "  UNSLOTH_STUDIO_HOME=/your/path sh uninstall.sh"
fi
