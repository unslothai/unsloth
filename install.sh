#!/bin/sh
# Unsloth Studio Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/install.sh | sh
set -e

VENV_NAME="unsloth_studio"
PYTHON_VERSION="3.13"

echo "==> Installing Unsloth Studio"

# 1. Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source uv into current shell
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create venv
echo "==> Creating Python ${PYTHON_VERSION} virtual environment (${VENV_NAME})..."
uv venv "$VENV_NAME" --python "$PYTHON_VERSION"

# 3. Activate venv
if [ -f "$VENV_NAME/bin/activate" ]; then
    . "$VENV_NAME/bin/activate"
elif [ -f "$VENV_NAME/Scripts/activate" ]; then
    . "$VENV_NAME/Scripts/activate"
else
    echo "Error: Could not find activation script in $VENV_NAME"
    exit 1
fi

# 4. Install unsloth
echo "==> Installing unsloth..."
uv pip install unsloth --torch-backend=auto

# 5. Run studio setup
echo "==> Running unsloth studio setup..."
unsloth studio setup

echo ""
echo "==> Unsloth Studio is ready!"
echo "    To launch, run:"
echo "      source ${VENV_NAME}/bin/activate"
echo "      unsloth studio -H 0.0.0.0 -p 8888"
