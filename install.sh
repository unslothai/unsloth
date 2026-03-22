#!/bin/sh
# Unsloth Studio Installer
# Usage (curl): curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/install.sh | sh
# Usage (wget): wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/install.sh | sh
set -e

VENV_NAME="unsloth_studio"
PYTHON_VERSION="3.13"

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

echo ""
echo "========================================="
echo "   Unsloth Studio Installer"
echo "========================================="
echo ""

# ── Detect platform ──
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

        if [ "$_a_part" -gt "$_b_part" ] 2>/dev/null; then
            return 0
        fi
        if [ "$_a_part" -lt "$_b_part" ] 2>/dev/null; then
            return 1
        fi
    done

    return 0
}

_uv_version_ok() {
    _ver=$("$1" --version 2>/dev/null | awk '{print $2}') || return 1
    [ -n "$_ver" ] || return 1
    # Strip pre-release/build suffixes
    case "$_ver" in
        ''|*[!0-9.]*) _ver=${_ver%%[-+]*} ;;
    esac
    case "$_ver" in
        ''|*[!0-9.]*) return 1 ;;
    esac
    version_ge "$_ver" "$UV_MIN_VERSION"
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

# ── Create venv (skip if it already exists and has a valid interpreter) ──
if [ ! -x "$VENV_NAME/bin/python" ]; then
    [ -e "$VENV_NAME" ] && rm -rf "$VENV_NAME"
    echo "==> Creating Python ${PYTHON_VERSION} virtual environment (${VENV_NAME})..."
    uv venv "$VENV_NAME" --python "$PYTHON_VERSION"
else
    echo "==> Virtual environment ${VENV_NAME} already exists, skipping creation."
fi

# ── Install unsloth directly into the venv (no activation needed) ──
echo "==> Installing unsloth (this may take a few minutes)..."
uv pip install --python "$VENV_NAME/bin/python" unsloth --torch-backend=auto

# ── Run studio setup ──
# Ensure the venv's Python is on PATH for setup.sh's Python discovery.
# On macOS the system Python may be outside the 3.11-3.13 range that
# setup.sh requires, but uv already installed a compatible interpreter
# inside the venv.
VENV_ABS_BIN="$(cd "$VENV_NAME/bin" && pwd)"
if [ -n "$VENV_ABS_BIN" ]; then
    export PATH="$VENV_ABS_BIN:$PATH"
fi

echo "==> Running unsloth studio setup..."
REQUESTED_PYTHON_VERSION="$(cd "$VENV_NAME/bin" && pwd)/python" \
"$VENV_NAME/bin/unsloth" studio setup </dev/null

echo ""
echo "========================================="
echo "   Unsloth Studio installed!"
echo "========================================="
echo ""

find_open_port() {
    "$VENV_NAME/bin/python" - "$1" "$2" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])
end_port = int(sys.argv[2])

for port in range(start_port, end_port + 1):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            continue
    print(port)
    raise SystemExit(0)

raise SystemExit(1)
PY
}

STUDIO_HOST="0.0.0.0"
START_PORT=8888
END_PORT=8898
# Launch studio automatically in interactive terminals;
# in non-interactive environments (Docker, CI, cloud-init) just print instructions.
if [ -t 1 ] && [ -t 2 ]; then  # stdout+stderr only; intentionally skips stdin for curl|sh
    STUDIO_PORT=$(find_open_port "$START_PORT" "$END_PORT") || STUDIO_PORT=""
    if [ -n "$STUDIO_PORT" ]; then
        echo "  To launch, run:"
        echo ""
        echo "    source ${VENV_NAME}/bin/activate"
        echo "    unsloth studio -H ${STUDIO_HOST} -p ${STUDIO_PORT}"
        echo ""
        echo "==> Auto Launching Unsloth Studio..."
        echo ""
        exec "$VENV_NAME/bin/unsloth" studio -H "$STUDIO_HOST" -p "$STUDIO_PORT" </dev/tty
    else
        echo "Note: all ports ${START_PORT}-${END_PORT} are in use."
        echo "  To launch manually, free a port and run:"
        echo ""
        echo "    source ${VENV_NAME}/bin/activate"
        echo "    unsloth studio -H ${STUDIO_HOST} -p 8888"
        echo ""
    fi
else
    echo "  To launch, run:"
    echo ""
    echo "    source ${VENV_NAME}/bin/activate"
    echo "    unsloth studio -H ${STUDIO_HOST} -p 8888"
    echo ""
fi
