#!/bin/sh
set -e

STUDIO_HOME="$HOME/.unsloth/studio"
DATA_DIR="$HOME/.local/share/unsloth"
LOCAL_BIN="$HOME/.local/bin"
UNSYMBOL="$LOCAL_BIN/unsloth"
UNSYMBOL_STUDIO="$LOCAL_BIN/unsloth-studio"

printf "This will remove:\n"
printf "  • Launcher scripts and shortcuts\n"
printf "  • CLI symlinks\n"
printf "  • ~/.unsloth directory (including virtual environment)\n"
printf "  • ~/unsloth_compiled_cache\n\n"
printf "This will NOT remove:\n"
printf "  • uv package manager\n"
printf "  • System packages (gcc, cmake, git, etc.)\n"
printf "  • Other Python environments\n\n"
printf "Continue with uninstall? [y/N] "
if [ -r /dev/tty ]; then
    read -r REPLY </dev/tty
else
    REPLY="n"
fi

case "$REPLY" in
    [yY][eE][sS]|[yY])
        ;;
    *)
        printf "Uninstall cancelled.\n"
        exit 0
        ;;
esac

# Kill running unsloth studio processes (be specific to avoid killing this script)
if command -v pgrep >/dev/null 2>&1 && command -v pkill >/dev/null 2>&1; then
    # Only kill python processes running unsloth studio, not scripts with unsloth in name
    for pid in $(pgrep -f "python.*unsloth.*studio" 2>/dev/null || true); do
        kill "$pid" 2>/dev/null || true
    done
fi
sleep 2

if [ -d "$DATA_DIR" ]; then
    rm -rf "$DATA_DIR"
fi

if [ -f "$HOME/.local/share/applications/unsloth-studio.desktop" ]; then
    rm -f "$HOME/.local/share/applications/unsloth-studio.desktop"
fi

if [ -f "$HOME/Desktop/unsloth-studio.desktop" ]; then
    rm -f "$HOME/Desktop/unsloth-studio.desktop"
fi

if [ -d "$HOME/Applications/Unsloth Studio.app" ]; then
    rm -rf "$HOME/Applications/Unsloth Studio.app"
fi

if [ -L "$HOME/Desktop/Unsloth Studio" ]; then
    rm -f "$HOME/Desktop/Unsloth Studio"
fi

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
fi

if [ -L "$UNSYMBOL" ] || [ -f "$UNSYMBOL" ]; then
    rm -f "$UNSYMBOL"
fi

if [ -L "$UNSYMBOL_STUDIO" ] || [ -f "$UNSYMBOL_STUDIO" ]; then
    rm -f "$UNSYMBOL_STUDIO"
fi

if [ -d "$HOME/.unsloth" ]; then
    rm -rf "$HOME/.unsloth"
fi

if [ -d "$HOME/unsloth_compiled_cache" ]; then
    rm -rf "$HOME/unsloth_compiled_cache"
fi

printf "Uninstall complete.\n"
