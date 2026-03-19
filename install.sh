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
            xcode-select --install 2>/dev/null || true
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
            printf "    Install via Homebrew? [Y/n] "
            read -r REPLY </dev/tty 2>/dev/null || REPLY="y"
            case "$REPLY" in
                [nN]*) echo "    Skipping -- GGUF inference may not be available." ;;
                *)     brew install $MISSING ;;
            esac
            ;;
        linux|wsl)
            if command -v apt-get >/dev/null 2>&1; then
                echo "    We need elevated permissions (sudo) to install them."
                printf "    Allow? [Y/n] "
                read -r REPLY </dev/tty 2>/dev/null || REPLY="y"
                case "$REPLY" in
                    [nN]*) echo "    Skipping -- GGUF inference may not be available." ;;
                    *)
                        sudo apt-get update -y
                        sudo apt-get install -y $MISSING
                        ;;
                esac
            else
                echo "    Please install them with your package manager, then re-run."
            fi
            ;;
    esac
    echo ""
else
    echo "==> All system dependencies found."
fi

# ── Install uv ──
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv package manager..."
    _uv_tmp=$(mktemp)
    download "https://astral.sh/uv/install.sh" "$_uv_tmp"
    sh "$_uv_tmp"
    rm -f "$_uv_tmp"
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Create venv (skip if it already exists) ──
if [ ! -d "$VENV_NAME" ]; then
    echo "==> Creating Python ${PYTHON_VERSION} virtual environment (${VENV_NAME})..."
    uv venv "$VENV_NAME" --python "$PYTHON_VERSION"
else
    echo "==> Virtual environment ${VENV_NAME} already exists, skipping creation."
fi

# ── Install unsloth directly into the venv (no activation needed) ──
echo "==> Installing unsloth (this may take a few minutes)..."
uv pip install --python "$VENV_NAME/bin/python" unsloth --torch-backend=auto

# ── Run studio setup ──
echo "==> Running unsloth studio setup..."
"$VENV_NAME/bin/unsloth" studio setup </dev/null

echo ""
echo "========================================="
echo "   Unsloth Studio installed!"
echo "========================================="
echo ""
echo "  To launch, run:"
echo ""
echo "    source ${VENV_NAME}/bin/activate"
echo "    unsloth studio -H 0.0.0.0 -p 8888"
echo ""
