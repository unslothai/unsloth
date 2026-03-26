#!/bin/sh
# Unsloth Studio Installer
# Usage (curl):  curl -fsSL https://unsloth.ai/install.sh | sh
# Usage (wget):  wget -qO- https://unsloth.ai/install.sh | sh
# Usage (local): ./install.sh --local   (install from local repo instead of PyPI)
# Usage (test):  ./install.sh --package roland-sloth  (install a different package name)
set -e

# ── Ensure common tool paths are on PATH ──
# macOS .app bundles inherit a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin).
# Homebrew, MacPorts, and ~/.local/bin are not included by default.
for _extra_path in /opt/homebrew/bin /usr/local/bin "$HOME/.local/bin"; do
    case ":$PATH:" in
        *":$_extra_path:"*) ;;
        *) [ -d "$_extra_path" ] && export PATH="$_extra_path:$PATH" ;;
    esac
done

# ── Parse flags ──
STUDIO_LOCAL_INSTALL=false
PACKAGE_NAME="unsloth"
TAURI_MODE=false
_next_is_package=false
for arg in "$@"; do
    if [ "$_next_is_package" = true ]; then
        PACKAGE_NAME="$arg"
        _next_is_package=false
        continue
    fi
    case "$arg" in
        --local) STUDIO_LOCAL_INSTALL=true ;;
        --package) _next_is_package=true ;;
        --tauri) TAURI_MODE=true ;;
    esac
done

if [ "$_next_is_package" = true ]; then
    echo "❌ ERROR: --package requires an argument." >&2
    exit 1
fi

# ── Tauri structured output ──
tauri_log() {
    if [ "$TAURI_MODE" = true ]; then
        echo "[TAURI:$1] $2"
    fi
}

PYTHON_VERSION="3.13"
STUDIO_HOME="$HOME/.unsloth/studio"
VENV_DIR="$STUDIO_HOME/unsloth_studio"

# ── Helper: download a URL to a file (supports curl and wget) ──
download() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$2" "$1"
    else
        echo "Error: neither curl nor wget found. Install one and re-run."
        exit 1
    fi
}

# ── Helper: check if a single package is available on the system ──
_is_pkg_installed() {
    case "$1" in
        build-essential) command -v gcc >/dev/null 2>&1 ;;
        libcurl4-openssl-dev)
            command -v dpkg >/dev/null 2>&1 && dpkg -s "$1" >/dev/null 2>&1 ;;
        pciutils)
            command -v lspci >/dev/null 2>&1 ;;
        *) command -v "$1" >/dev/null 2>&1 ;;
    esac
}

# ── Helper: install packages via apt, escalating to sudo only if needed ──
# Usage: _smart_apt_install pkg1 pkg2 pkg3 ...
_smart_apt_install() {
    _PKGS="$*"

    # Step 1: Try installing without sudo (works when already root)
    apt-get update -y </dev/null >/dev/null 2>&1 || true
    apt-get install -y $_PKGS </dev/null >/dev/null 2>&1 || true

    # Step 2: Check which packages are still missing
    _STILL_MISSING=""
    for _pkg in $_PKGS; do
        if ! _is_pkg_installed "$_pkg"; then
            _STILL_MISSING="$_STILL_MISSING $_pkg"
        fi
    done
    _STILL_MISSING=$(echo "$_STILL_MISSING" | sed 's/^ *//')

    if [ -z "$_STILL_MISSING" ]; then
        return 0
    fi

    # In Tauri mode, report needed packages and exit — Rust handles elevation
    if [ "$TAURI_MODE" = true ]; then
        tauri_log "NEED_SUDO" "$_STILL_MISSING"
        exit 2
    fi

    # Step 3: Escalate -- need elevated permissions for remaining packages
    if command -v sudo >/dev/null 2>&1; then
        echo ""
        echo "    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "    WARNING: We require sudo elevated permissions to install:"
        echo "    $_STILL_MISSING"
        echo "    If you accept, we'll run sudo now, and it'll prompt your password."
        echo "    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
        printf "    Accept? [Y/n] "
        if [ -r /dev/tty ]; then
            read -r REPLY </dev/tty || REPLY="y"
        else
            REPLY="y"
        fi
        case "$REPLY" in
            [nN]*)
                echo ""
                echo "    Please install these packages first, then re-run Unsloth Studio setup:"
                echo "    sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                exit 1
                ;;
            *)
                sudo apt-get update -y </dev/null
                sudo apt-get install -y $_STILL_MISSING </dev/null
                ;;
        esac
    else
        echo ""
        echo "    sudo is not available on this system."
        echo "    Please install these packages as root, then re-run Unsloth Studio setup:"
        echo "    apt-get update -y && apt-get install -y $_STILL_MISSING"
        exit 1
    fi
}

# ── Helper: create desktop shortcuts and launcher script ──
# Usage: create_studio_shortcuts <unsloth_exe> <os>
# Creates ~/.local/share/unsloth/launch-studio.sh (shared launcher),
# plus platform-specific shortcuts (Linux .desktop / macOS .app bundle).
# Skipped on WSL (no native desktop).
create_studio_shortcuts() {
    _css_exe="$1"
    _css_os="$2"

    # Skip on WSL -- no native desktop environment
    if [ "$_css_os" = "wsl" ]; then
        return 0
    fi

    # Validate exe
    if [ ! -x "$_css_exe" ]; then
        echo "[WARN] Cannot create shortcuts: unsloth not found at $_css_exe"
        return 0
    fi

    # Resolve absolute path
    _css_exe_dir=$(cd "$(dirname "$_css_exe")" && pwd)
    _css_exe="$_css_exe_dir/$(basename "$_css_exe")"

    _css_data_dir="$HOME/.local/share/unsloth"
    _css_launcher="$_css_data_dir/launch-studio.sh"
    _css_icon_png="$_css_data_dir/unsloth-studio.png"
    _css_gem_png="$_css_data_dir/unsloth-gem.png"

    mkdir -p "$_css_data_dir"

    # ── Write launcher script ──
    # The launcher is Bash (not POSIX sh).
    # We write it with a placeholder and substitute the exe path via sed.
    cat > "$_css_launcher" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
# Unsloth Studio Launcher
# Auto-generated by install.sh -- do not edit manually.
set -euo pipefail

DATA_DIR="$HOME/.local/share/unsloth"

# Read exe path from config written at install time.
# Sourcing is safe: the config file is written by install.sh, not user input.
if [ -f "$DATA_DIR/studio.conf" ]; then
    . "$DATA_DIR/studio.conf"
fi
if [ -z "${UNSLOTH_EXE:-}" ] || [ ! -x "${UNSLOTH_EXE:-}" ]; then
    echo "Error: UNSLOTH_EXE not set or not executable. Re-run the installer." >&2
    exit 1
fi

BASE_PORT=8888
MAX_PORT_OFFSET=20
TIMEOUT_SEC=60
POLL_INTERVAL_SEC=1
LOG_FILE="$DATA_DIR/studio.log"
LOCK_DIR="${XDG_RUNTIME_DIR:-/tmp}/unsloth-studio-launcher-$(id -u).lock"

# ── HTTP GET helper (supports curl and wget) ──
_http_get() {
    _url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsS --max-time 1 "$_url" 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- --timeout=1 "$_url" 2>/dev/null
    else
        return 1
    fi
}

# ── Health check ──
_check_health() {
    _port=$1
    _resp=$(_http_get "http://127.0.0.1:$_port/api/health") || return 1
    case "$_resp" in
        *'"status"'*'"healthy"'*'"service"'*'"Unsloth UI Backend"'*) return 0 ;;
        *'"service"'*'"Unsloth UI Backend"'*'"status"'*'"healthy"'*) return 0 ;;
    esac
    return 1
}

# ── Port scanning ──
_candidate_ports() {
    echo "$BASE_PORT"
    _max_port=$((BASE_PORT + MAX_PORT_OFFSET))
    if command -v ss >/dev/null 2>&1; then
        ss -tlnH 2>/dev/null | awk '{print $4}' | grep -oE '[0-9]+$' | \
            awk -v lo="$BASE_PORT" -v hi="$_max_port" '$1 >= lo && $1 <= hi && $1 != lo {print}' || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP -sTCP:LISTEN -nP 2>/dev/null | awk '{print $9}' | grep -oE '[0-9]+$' | \
            awk -v lo="$BASE_PORT" -v hi="$_max_port" '$1 >= lo && $1 <= hi && $1 != lo {print}' || true
    else
        _offset=1
        while [ "$_offset" -le "$MAX_PORT_OFFSET" ]; do
            echo $((BASE_PORT + _offset))
            _offset=$((_offset + 1))
        done
    fi
}

_find_healthy_port() {
    for _p in $(_candidate_ports | sort -un); do
        if _check_health "$_p"; then
            echo "$_p"
            return 0
        fi
    done
    return 1
}

# ── Check if a port is busy ──
_is_port_busy() {
    _port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -tlnH 2>/dev/null | awk '{print $4}' | grep -qE "[.:]$_port$"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$_port" -sTCP:LISTEN -nP >/dev/null 2>&1
    else
        return 1
    fi
}

# ── Find a free port in range ──
_find_launch_port() {
    _offset=0
    while [ "$_offset" -le "$MAX_PORT_OFFSET" ]; do
        _candidate=$((BASE_PORT + _offset))
        if ! _is_port_busy "$_candidate"; then
            echo "$_candidate"
            return 0
        fi
        _offset=$((_offset + 1))
    done
    return 1
}

# ── Open browser ──
_open_browser() {
    _url="$1"
    if [ "$(uname)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
        open "$_url"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$_url" >/dev/null 2>&1 &
    else
        echo "Open in your browser: $_url" >&2
    fi
}

# ── Spawn terminal with studio command ──
_spawn_terminal() {
    _cmd="$1"
    _os=$(uname)
    if [ "$_os" = "Darwin" ]; then
        # Escape backslashes and double-quotes for AppleScript string
        _cmd_escaped=$(printf '%s' "$_cmd" | sed 's/\\/\\\\/g; s/"/\\"/g')
        osascript -e "tell application \"Terminal\" to do script \"$_cmd_escaped\"" >/dev/null 2>&1 && return 0
    else
        for _term in gnome-terminal konsole xfce4-terminal mate-terminal lxterminal xterm; do
            if command -v "$_term" >/dev/null 2>&1; then
                case "$_term" in
                    gnome-terminal) "$_term" -- sh -c "$_cmd" & return 0 ;;
                    konsole)        "$_term" -e sh -c "$_cmd" & return 0 ;;
                    xterm)          "$_term" -e sh -c "$_cmd" & return 0 ;;
                    *)              "$_term" -e sh -c "$_cmd" & return 0 ;;
                esac
            fi
        done
    fi
    # Fallback: background with log
    echo "No terminal emulator found; running in background. Logs: $LOG_FILE" >&2
    nohup sh -c "$_cmd" >> "$LOG_FILE" 2>&1 &
    return 0
}

# ── Atomic directory-based single-instance guard ──
_acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "$$" > "$LOCK_DIR/pid"
        return 0
    fi

    # Lock dir exists -- check if owner is still alive
    _old_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
    if [ -n "$_old_pid" ] && kill -0 "$_old_pid" 2>/dev/null; then
        # Another launcher is running; wait for it to bring Studio up
        _deadline=$(($(date +%s) + TIMEOUT_SEC))
        while [ "$(date +%s)" -lt "$_deadline" ]; do
            _port=$(_find_healthy_port) && {
                _open_browser "http://localhost:$_port"
                exit 0
            }
            sleep "$POLL_INTERVAL_SEC"
        done
        echo "Timed out waiting for other launcher (PID $_old_pid)" >&2
        exit 0
    fi

    # Stale lock -- reclaim
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null || return 1
    echo "$$" > "$LOCK_DIR/pid"
}

_release_lock() {
    rm -rf "$LOCK_DIR"
}

# ── Main ──
# Fast path: already healthy
_port=$(_find_healthy_port) && {
    _open_browser "http://localhost:$_port"
    exit 0
}

_acquire_lock
trap '_release_lock' EXIT INT TERM

# Post-lock re-check (handles race with another launcher)
_port=$(_find_healthy_port) && {
    _open_browser "http://localhost:$_port"
    exit 0
}

# Find a free port in range
_launch_port=$(_find_launch_port) || {
    echo "No free port found in range ${BASE_PORT}-$((BASE_PORT + MAX_PORT_OFFSET))" >&2
    exit 1
}

# Launch studio in a terminal
_launch_cmd=$(printf '%q ' "$UNSLOTH_EXE" studio -H 0.0.0.0 -p "$_launch_port")
_launch_cmd=${_launch_cmd% }
_spawn_terminal "$_launch_cmd"

# Poll for health
_deadline=$(($(date +%s) + TIMEOUT_SEC))
while [ "$(date +%s)" -lt "$_deadline" ]; do
    _port=$(_find_healthy_port) && {
        _open_browser "http://localhost:$_port"
        exit 0
    }
    sleep "$POLL_INTERVAL_SEC"
done

echo "Unsloth Studio did not become healthy within ${TIMEOUT_SEC}s." >&2
echo "Check logs at: $LOG_FILE" >&2
exit 1
LAUNCHER_EOF

    chmod +x "$_css_launcher"

    # Write the exe path to a separate conf file sourced by the launcher.
    # Using single-quote wrapping with the standard '\'' escape for any
    # embedded apostrophes. This avoids all sed metacharacter issues.
    _css_quoted_exe=$(printf '%s' "$_css_exe" | sed "s/'/'\\\\''/g")
    printf '%s\n' "UNSLOTH_EXE='$_css_quoted_exe'" > "$_css_data_dir/studio.conf"

    # ── Icon: try bundled, then download ──
    # favicon.png (small, for Linux) and unsloth-gem.png (large, for macOS icns)
    _css_script_dir=""
    if [ -n "${0:-}" ] && [ -f "$0" ]; then
        _css_script_dir=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || true
    fi

    # Try to find favicon.png from installed package (site-packages) or local repo
    _css_found_favicon=""
    _css_found_gem=""
    _css_venv_dir=$(dirname "$(dirname "$_css_exe")")
    # Check site-packages
    for _sp in "$_css_venv_dir"/lib/python*/site-packages/unsloth/studio/frontend/public; do
        if [ -f "$_sp/favicon.png" ]; then
            _css_found_favicon="$_sp/favicon.png"
        fi
        if [ -f "$_sp/unsloth-gem.png" ]; then
            _css_found_gem="$_sp/unsloth-gem.png"
        fi
    done
    # Check local repo (when running from clone)
    if [ -z "$_css_found_favicon" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/frontend/public/favicon.png" ]; then
        _css_found_favicon="$_css_script_dir/studio/frontend/public/favicon.png"
    fi
    if [ -z "$_css_found_gem" ] && [ -n "$_css_script_dir" ] && [ -f "$_css_script_dir/studio/frontend/public/unsloth-gem.png" ]; then
        _css_found_gem="$_css_script_dir/studio/frontend/public/unsloth-gem.png"
    fi

    # Copy or download favicon.png
    if [ -n "$_css_found_favicon" ]; then
        cp "$_css_found_favicon" "$_css_icon_png" 2>/dev/null || true
    elif [ ! -f "$_css_icon_png" ]; then
        download "https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/favicon.png" "$_css_icon_png" 2>/dev/null || true
    fi
    # Copy or download unsloth-gem.png (for macOS icns)
    if [ -n "$_css_found_gem" ]; then
        cp "$_css_found_gem" "$_css_gem_png" 2>/dev/null || true
    elif [ ! -f "$_css_gem_png" ]; then
        download "https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/unsloth-gem.png" "$_css_gem_png" 2>/dev/null || true
    fi

    # Validate PNG header (first 4 bytes: \x89PNG)
    _css_validate_png() {
        [ -f "$1" ] || return 1
        _hdr=$(od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' ')
        [ "$_hdr" = "89504e47" ]
    }
    if [ -f "$_css_icon_png" ] && ! _css_validate_png "$_css_icon_png"; then
        rm -f "$_css_icon_png"
    fi
    if [ -f "$_css_gem_png" ] && ! _css_validate_png "$_css_gem_png"; then
        rm -f "$_css_gem_png"
    fi

    # ── Platform-specific shortcuts ──
    _css_created=0

    if [ "$_css_os" = "linux" ]; then
        # ── Linux: .desktop file ──
        _css_app_dir="$HOME/.local/share/applications"
        mkdir -p "$_css_app_dir"

        _css_desktop="$_css_app_dir/unsloth-studio.desktop"
        # Escape backslashes and double-quotes for .desktop Exec= field
        _css_exec_escaped=$(printf '%s' "$_css_launcher" | sed 's/\\/\\\\/g; s/"/\\"/g')
        _css_icon_escaped=$(printf '%s' "$_css_icon_png" | sed 's/\\/\\\\/g; s/"/\\"/g')
        cat > "$_css_desktop" << DESKTOP_EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Unsloth Studio
Comment=Launch Unsloth Studio
Exec="$_css_exec_escaped"
Icon=$_css_icon_escaped
Terminal=false
StartupNotify=true
Categories=Development;Science;
DESKTOP_EOF
        chmod +x "$_css_desktop"

        # Copy to ~/Desktop if it exists
        if [ -d "$HOME/Desktop" ]; then
            cp "$_css_desktop" "$HOME/Desktop/unsloth-studio.desktop" 2>/dev/null || true
            chmod +x "$HOME/Desktop/unsloth-studio.desktop" 2>/dev/null || true
            # Mark as trusted so GNOME/Nautilus allows launching via double-click
            if command -v gio >/dev/null 2>&1; then
                gio set "$HOME/Desktop/unsloth-studio.desktop" metadata::trusted true 2>/dev/null || true
            fi
        fi

        # Best-effort update database
        update-desktop-database "$_css_app_dir" 2>/dev/null || true
        _css_created=1

    elif [ "$_css_os" = "macos" ]; then
        # ── macOS: .app bundle ──
        _css_app="$HOME/Applications/Unsloth Studio.app"
        _css_contents="$_css_app/Contents"
        _css_macos_dir="$_css_contents/MacOS"
        _css_res_dir="$_css_contents/Resources"
        mkdir -p "$_css_macos_dir" "$_css_res_dir"

        # Info.plist
        cat > "$_css_contents/Info.plist" << 'PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>ai.unsloth.studio</string>
    <key>CFBundleName</key>
    <string>Unsloth Studio</string>
    <key>CFBundleDisplayName</key>
    <string>Unsloth Studio</string>
    <key>CFBundleExecutable</key>
    <string>launch-studio</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST_EOF

        # Executable stub
        cat > "$_css_macos_dir/launch-studio" << STUB_EOF
#!/bin/sh
exec "$HOME/.local/share/unsloth/launch-studio.sh" "\$@"
STUB_EOF
        chmod +x "$_css_macos_dir/launch-studio"

        # Build AppIcon.icns from unsloth-gem.png (2240x2240)
        if [ -f "$_css_gem_png" ] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
            _css_tmpdir=$(mktemp -d 2>/dev/null)
            if [ -d "$_css_tmpdir" ]; then
                _css_iconset="$_css_tmpdir/AppIcon.iconset"
                mkdir -p "$_css_iconset"
                _css_icon_ok=true
                for _sz in 16 32 128 256 512; do
                    _sz2=$((_sz * 2))
                    sips -z "$_sz" "$_sz" "$_css_gem_png" --out "$_css_iconset/icon_${_sz}x${_sz}.png" >/dev/null 2>&1 || _css_icon_ok=false
                    sips -z "$_sz2" "$_sz2" "$_css_gem_png" --out "$_css_iconset/icon_${_sz}x${_sz}@2x.png" >/dev/null 2>&1 || _css_icon_ok=false
                done
                if [ "$_css_icon_ok" = "true" ]; then
                    iconutil -c icns "$_css_iconset" -o "$_css_res_dir/AppIcon.icns" 2>/dev/null || true
                fi
                rm -rf "$_css_tmpdir"
            fi
        fi
        # Fallback: copy PNG as icon
        if [ ! -f "$_css_res_dir/AppIcon.icns" ] && [ -f "$_css_icon_png" ]; then
            cp "$_css_icon_png" "$_css_res_dir/AppIcon.icns" 2>/dev/null || true
        fi

        # Touch so Finder indexes it
        touch "$_css_app"

        # Symlink on Desktop
        if [ -d "$HOME/Desktop" ]; then
            ln -sf "$_css_app" "$HOME/Desktop/Unsloth Studio" 2>/dev/null || true
        fi
        _css_created=1
    fi

    if [ "$_css_created" -eq 1 ]; then
        echo "[OK] Created Unsloth Studio shortcut(s)"
    fi
}

echo ""
echo "========================================="
echo "   Unsloth Studio Installer"
echo "========================================="
echo ""

# ── Detect platform ──
tauri_log "STEP" "Detecting platform"
OS="linux"
if [ "$(uname)" = "Darwin" ]; then
    OS="macos"
elif grep -qi microsoft /proc/version 2>/dev/null; then
    OS="wsl"
fi
echo "==> Platform: $OS"

# ── Check system dependencies ──
# cmake and git are needed by unsloth studio setup to build the GGUF inference
# engine (llama.cpp). build-essential and libcurl-dev are also needed on Linux.
tauri_log "STEP" "Checking system dependencies"
MISSING=""

command -v cmake >/dev/null 2>&1 || MISSING="$MISSING cmake"
command -v git   >/dev/null 2>&1 || MISSING="$MISSING git"

case "$OS" in
    macos)
        # Xcode Command Line Tools provide the C/C++ compiler
        if ! xcode-select -p >/dev/null 2>&1; then
            echo ""
            echo "==> Xcode Command Line Tools are required."
            echo "    Installing (a system dialog will appear)..."
            xcode-select --install </dev/null 2>/dev/null || true
            echo "    After the installation completes, please re-run this script."
            exit 1
        fi
        ;;
    linux|wsl)
        # curl or wget is needed for downloads; check both
        if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
            MISSING="$MISSING curl"
        fi
        command -v gcc  >/dev/null 2>&1 || MISSING="$MISSING build-essential"
        # libcurl dev headers for llama.cpp HTTPS support
        if command -v dpkg >/dev/null 2>&1; then
            dpkg -s libcurl4-openssl-dev >/dev/null 2>&1 || MISSING="$MISSING libcurl4-openssl-dev"
        fi
        ;;
esac

MISSING=$(echo "$MISSING" | sed 's/^ *//')

if [ -n "$MISSING" ]; then
    echo ""
    echo "==> Unsloth Studio needs these packages: $MISSING"
    echo "    These are needed to build the GGUF inference engine."

    case "$OS" in
        macos)
            if ! command -v brew >/dev/null 2>&1; then
                echo ""
                echo "    Homebrew is required to install them."
                echo "    Install Homebrew from https://brew.sh then re-run this script."
                exit 1
            fi
            brew install $MISSING </dev/null
            ;;
        linux|wsl)
            if command -v apt-get >/dev/null 2>&1; then
                _smart_apt_install $MISSING
            else
                echo "    apt-get is not available. Please install with your package manager:"
                echo "    $MISSING"
                echo "    Then re-run Unsloth Studio setup."
                exit 1
            fi
            ;;
    esac
    echo ""
else
    echo "==> All system dependencies found."
fi

# ── Install uv ──
tauri_log "STEP" "Installing uv package manager"
UV_MIN_VERSION="0.7.14"

version_ge() {
    # returns 0 if $1 >= $2
    _a=$1
    _b=$2

    while [ -n "$_a" ] || [ -n "$_b" ]; do
        _a_part=${_a%%.*}
        _b_part=${_b%%.*}

        [ "$_a" = "$_a_part" ] && _a="" || _a=${_a#*.}
        [ "$_b" = "$_b_part" ] && _b="" || _b=${_b#*.}

        [ -z "$_a_part" ] && _a_part=0
        [ -z "$_b_part" ] && _b_part=0

        if [ "$_a_part" -gt "$_b_part" ]; then
            return 0
        fi
        if [ "$_a_part" -lt "$_b_part" ]; then
            return 1
        fi
    done

    return 0
}

_uv_version_ok() {
    _raw=$("$1" --version 2>/dev/null | awk '{print $2}') || return 1
    [ -n "$_raw" ] || return 1
    _ver=${_raw%%[-+]*}
    case "$_ver" in
        ''|*[!0-9.]*) return 1 ;;
    esac
    version_ge "$_ver" "$UV_MIN_VERSION" || return 1
    # Prerelease of the exact minimum (e.g. 0.7.14-rc1) is still below stable 0.7.14
    [ "$_ver" = "$UV_MIN_VERSION" ] && [ "$_raw" != "$_ver" ] && return 1
    return 0
}

if ! command -v uv >/dev/null 2>&1 || ! _uv_version_ok uv; then
    echo "==> Installing uv package manager..."
    _uv_tmp=$(mktemp)
    download "https://astral.sh/uv/install.sh" "$_uv_tmp"
    sh "$_uv_tmp" </dev/null
    rm -f "$_uv_tmp"
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Create venv (migrate old layout if possible, otherwise fresh) ──
tauri_log "STEP" "Creating virtual environment"
mkdir -p "$STUDIO_HOME"

_MIGRATED=false

if [ -x "$VENV_DIR/bin/python" ]; then
    # New layout already exists — nuke for fresh install
    rm -rf "$VENV_DIR"
elif [ -x "$STUDIO_HOME/.venv/bin/python" ]; then
    # Old layout exists — validate before migrating
    echo "==> Found legacy Studio environment, validating..."
    if "$STUDIO_HOME/.venv/bin/python" -c "
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
A = torch.ones((10, 10), device=device)
B = torch.ones((10, 10), device=device)
C = torch.ones((10, 10), device=device)
D = A + B
E = D @ C
torch.testing.assert_close(torch.unique(E), torch.tensor((20,), device=E.device, dtype=E.dtype))
" >/dev/null 2>&1; then
        echo "✅ Legacy environment is healthy — migrating..."
        mv "$STUDIO_HOME/.venv" "$VENV_DIR"
        echo "   Moved ~/.unsloth/studio/.venv → $VENV_DIR"
        _MIGRATED=true
    else
        echo "⚠️  Legacy environment failed validation — creating fresh environment"
        rm -rf "$STUDIO_HOME/.venv"
    fi
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "==> Creating Python ${PYTHON_VERSION} virtual environment (${VENV_DIR})..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
    echo "==> Using migrated environment at ${VENV_DIR}"
fi

# ── Resolve repo root (for --local installs) ──
_REPO_ROOT="$(cd "$(dirname "$0" 2>/dev/null || echo ".")" && pwd)"

# ── Detect GPU and choose PyTorch index URL ──
# Mirrors Get-TorchIndexUrl in install.ps1.
# On CPU-only machines this returns the cpu index, avoiding the solver
# dead-end where --torch-backend=auto resolves to unsloth==2024.8.
get_torch_index_url() {
    _base="https://download.pytorch.org/whl"
    # macOS: always CPU (no CUDA support)
    case "$(uname -s)" in Darwin) echo "$_base/cpu"; return ;; esac
    # Try nvidia-smi
    _smi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        _smi="nvidia-smi"
    elif [ -x "/usr/bin/nvidia-smi" ]; then
        _smi="/usr/bin/nvidia-smi"
    fi
    if [ -z "$_smi" ]; then echo "$_base/cpu"; return; fi
    # Parse CUDA version from nvidia-smi output (POSIX-safe, no grep -P)
    _cuda_ver=$(LC_ALL=C $_smi 2>/dev/null \
        | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | head -1)
    if [ -z "$_cuda_ver" ]; then
        echo "[WARN] Could not determine CUDA version from nvidia-smi, defaulting to cu126" >&2
        echo "$_base/cu126"; return
    fi
    _major=${_cuda_ver%%.*}
    _minor=${_cuda_ver#*.}
    if [ "$_major" -ge 13 ]; then echo "$_base/cu130"
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 8 ]; then echo "$_base/cu128"
    elif [ "$_major" -eq 12 ] && [ "$_minor" -ge 6 ]; then echo "$_base/cu126"
    elif [ "$_major" -ge 12 ]; then echo "$_base/cu124"
    elif [ "$_major" -ge 11 ]; then echo "$_base/cu118"
    else echo "$_base/cpu"; fi
}
TORCH_INDEX_URL=$(get_torch_index_url)

# ── Install unsloth directly into the venv (no activation needed) ──
tauri_log "STEP" "Installing PyTorch"
_VENV_PY="$VENV_DIR/bin/python"
if [ "$_MIGRATED" = true ]; then
    # Migrated env: force-reinstall unsloth+unsloth-zoo to ensure clean state
    # in the new venv location, while preserving existing torch/CUDA
    echo "==> Upgrading unsloth in migrated environment..."
    uv pip install --python "$_VENV_PY" \
        --reinstall-package unsloth --reinstall-package unsloth-zoo \
        "unsloth>=2026.3.14" unsloth-zoo
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        echo "==> Overlaying local repo (editable)..."
        uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
    fi
elif [ -n "$TORCH_INDEX_URL" ]; then
    # Fresh: Step 1 - install torch from explicit index
    echo "==> Installing PyTorch ($TORCH_INDEX_URL)..."
    uv pip install --python "$_VENV_PY" "torch>=2.4,<2.11.0" torchvision torchaudio \
        --index-url "$TORCH_INDEX_URL"
    # Fresh: Step 2 - install unsloth, preserving pre-installed torch
    tauri_log "STEP" "Installing Unsloth"
    echo "==> Installing unsloth (this may take a few minutes)..."
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        uv pip install --python "$_VENV_PY" \
            --upgrade-package unsloth "unsloth>=2026.3.14" unsloth-zoo
        echo "==> Overlaying local repo (editable)..."
        uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
    else
        uv pip install --python "$_VENV_PY" \
            --upgrade-package unsloth "$PACKAGE_NAME"
    fi
else
    # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
    tauri_log "STEP" "Installing Unsloth"
    echo "==> Installing unsloth (this may take a few minutes)..."
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        uv pip install --python "$_VENV_PY" unsloth-zoo "unsloth>=2026.3.14" --torch-backend=auto
        echo "==> Overlaying local repo (editable)..."
        uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
    else
        uv pip install --python "$_VENV_PY" "$PACKAGE_NAME" --torch-backend=auto
    fi
fi

# ── Run studio setup ──
tauri_log "STEP" "Running Studio setup"
# When --local, use the repo's own setup.sh directly.
# Otherwise, find it inside the installed package.
SETUP_SH=""
if [ "$STUDIO_LOCAL_INSTALL" = true ] && [ -f "$_REPO_ROOT/studio/setup.sh" ]; then
    SETUP_SH="$_REPO_ROOT/studio/setup.sh"
fi

if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    SETUP_SH=$("$VENV_DIR/bin/python" -c "
import importlib.resources
print(importlib.resources.files('studio') / 'setup.sh')
" 2>/dev/null || echo "")
fi

# Fallback: search site-packages
if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    SETUP_SH=$(find "$VENV_DIR" -path "*/studio/setup.sh" -print -quit 2>/dev/null || echo "")
fi

if [ -z "$SETUP_SH" ] || [ ! -f "$SETUP_SH" ]; then
    echo "❌ ERROR: Could not find studio/setup.sh in the installed package."
    exit 1
fi

# Ensure the venv's Python is on PATH so setup.sh can find it.
VENV_ABS_BIN="$(cd "$VENV_DIR/bin" && pwd)"
if [ -n "$VENV_ABS_BIN" ]; then
    export PATH="$VENV_ABS_BIN:$PATH"
fi

echo "==> Running unsloth setup..."
# Tauri desktop app bundles its own frontend — skip Node/npm/frontend build
_SKIP_FRONTEND=0
if [ "$TAURI_MODE" = true ]; then
    _SKIP_FRONTEND=1
fi
if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
    SKIP_STUDIO_BASE=1 \
    SKIP_STUDIO_FRONTEND="$_SKIP_FRONTEND" \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    STUDIO_LOCAL_INSTALL=1 \
    STUDIO_LOCAL_REPO="$_REPO_ROOT" \
    bash "$SETUP_SH" </dev/null
else
    SKIP_STUDIO_BASE=1 \
    SKIP_STUDIO_FRONTEND="$_SKIP_FRONTEND" \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    bash "$SETUP_SH" </dev/null
fi

# ── Make 'unsloth' available globally via ~/.local/bin ──
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/unsloth" "$HOME/.local/bin/unsloth"

_LOCAL_BIN="$HOME/.local/bin"
case ":$PATH:" in
    *":$_LOCAL_BIN:"*) ;;  # already on PATH
    *)
        _SHELL_PROFILE=""
        if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
            _SHELL_PROFILE="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            _SHELL_PROFILE="$HOME/.bashrc"
        elif [ -f "$HOME/.profile" ]; then
            _SHELL_PROFILE="$HOME/.profile"
        fi

        if [ -n "$_SHELL_PROFILE" ]; then
            if ! grep -q '\.local/bin' "$_SHELL_PROFILE" 2>/dev/null; then
                echo '' >> "$_SHELL_PROFILE"
                echo '# Added by Unsloth installer' >> "$_SHELL_PROFILE"
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$_SHELL_PROFILE"
                echo "==> Added ~/.local/bin to PATH in $_SHELL_PROFILE"
            fi
        fi
        export PATH="$_LOCAL_BIN:$PATH"
        ;;
esac

# ── Tauri mode: done, skip shortcuts and auto-launch ──
if [ "$TAURI_MODE" = true ]; then
    tauri_log "DONE" ""
    exit 0
fi

create_studio_shortcuts "$VENV_ABS_BIN/unsloth" "$OS"

echo ""
echo "========================================="
echo "   Unsloth Studio installed!"
echo "========================================="
echo ""

# Launch studio automatically in interactive terminals;
# in non-interactive environments (Docker, CI, cloud-init) just print instructions.
if [ -t 1 ]; then
    echo "==> Launching Unsloth Studio..."
    echo ""
    "$VENV_DIR/bin/unsloth" studio -H 0.0.0.0 -p 8888
    _LAUNCH_EXIT=$?
    if [ "$_LAUNCH_EXIT" -ne 0 ] && [ "$_MIGRATED" = true ]; then
        echo ""
        echo "⚠️  Unsloth Studio failed to start after migration."
        echo "   Your migrated environment may be incompatible."
        echo "   To fix, remove the environment and reinstall:"
        echo ""
        echo "   rm -rf $VENV_DIR"
        echo "   curl -fsSL https://unsloth.ai/install.sh | sh"
        echo ""
    fi
    exit "$_LAUNCH_EXIT"
else
    echo "  To launch, run:"
    echo ""
    echo "    unsloth studio -H 0.0.0.0 -p 8888"
    echo ""
    echo "  Or activate the environment first:"
    echo ""
    echo "    source ${VENV_DIR}/bin/activate"
    echo "    unsloth studio -H 0.0.0.0 -p 8888"
    echo ""
fi
