#!/bin/sh
set -e

STUDIO_HOME="$HOME/.unsloth/studio"
DATA_DIR="$HOME/.local/share/unsloth"
LOCAL_BIN="$HOME/.local/bin"
UNSYMBOL="$LOCAL_BIN/unsloth"

printf "This will remove launcher scripts, shortcuts, and CLI symlink.\n"
printf "Python virtual environment and packages will be PRESERVED.\n\n"
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

if command -v pkill >/dev/null 2>&1; then
    pkill -f "unsloth.*studio" 2>/dev/null || true
fi
if command -v killall >/dev/null 2>&1; then
    killall -q unsloth 2>/dev/null || true
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

printf "Uninstall complete.\n"
printf "Python environment preserved at: %s/unsloth_studio\n" "$STUDIO_HOME"
printf "To use manually: source %s/unsloth_studio/bin/activate\n" "$STUDIO_HOME"
