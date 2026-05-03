#!/bin/sh
# Strictly local source install for Unsloth Core + Studio.
# Creates local environments and keeps installer/runtime caches inside this checkout.
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_VERSION="3.13"
INSTALL_EXTRA="base"
INSTALL_STUDIO=true
NO_TORCH=false

usage() {
    cat <<'EOF'
Usage: ./install-source.sh [options]

Options:
  --python VERSION   Python version for uv venv creation (default: 3.13)
  --venv DIR        Virtual environment directory (default: ./.venv)
  --extra NAME      pyproject optional dependency extra (default: base)
  --no-torch        Skip PyTorch/training deps where possible
  --core-only       Install only the editable Python package, not Studio
  -h, --help        Show this help

This installer only writes inside the current checkout:
  .venv/       Python environment
  .studio/     Studio environment, database, models, outputs, and caches
  .uv-cache/   uv cache
  .uv-python/  uv-managed Python downloads
  .pip-cache/  pip cache
  .npm-cache/  npm cache
  studio/frontend/node_modules/ and studio/frontend/dist/
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python)
            [ "$#" -ge 2 ] || { echo "ERROR: --python requires a value" >&2; exit 1; }
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --venv)
            [ "$#" -ge 2 ] || { echo "ERROR: --venv requires a value" >&2; exit 1; }
            case "$2" in
                /*) VENV_DIR="$2" ;;
                *) VENV_DIR="$ROOT_DIR/$2" ;;
            esac
            shift 2
            ;;
        --extra)
            [ "$#" -ge 2 ] || { echo "ERROR: --extra requires a value" >&2; exit 1; }
            INSTALL_EXTRA="$2"
            shift 2
            ;;
        --no-torch)
            INSTALL_EXTRA=""
            NO_TORCH=true
            shift
            ;;
        --core-only)
            INSTALL_STUDIO=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[ -f "$ROOT_DIR/pyproject.toml" ] || {
    echo "ERROR: run this from the Unsloth repository root." >&2
    exit 1
}

case "$INSTALL_EXTRA" in
    *[!a-zA-Z0-9._-]*)
        echo "ERROR: --extra may only contain letters, numbers, dot, underscore, and dash." >&2
        exit 1
        ;;
esac

export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.uv-python"
export PIP_CACHE_DIR="$ROOT_DIR/.pip-cache"
export NPM_CONFIG_CACHE="$ROOT_DIR/.npm-cache"
export UNSLOTH_STUDIO_HOME="$ROOT_DIR/.studio"
export UNSLOTH_CACHE_DIR="$ROOT_DIR/.studio/cache"

case "$VENV_DIR" in
    "$ROOT_DIR"/*) ;;
    *)
        echo "ERROR: --venv must stay inside this checkout: $ROOT_DIR" >&2
        exit 1
        ;;
esac

echo "Installing Unsloth from source into: $VENV_DIR"
echo "Keeping install caches inside: $ROOT_DIR"

write_local_activation_env() {
    _activate="$VENV_DIR/bin/activate"
    [ -f "$_activate" ] || return 0
    if ! grep -q "Unsloth local source install" "$_activate"; then
        cat >>"$_activate" <<EOF

# Unsloth local source install: keep package-manager state inside this checkout.
export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.uv-python"
export PIP_CACHE_DIR="$ROOT_DIR/.pip-cache"
export NPM_CONFIG_CACHE="$ROOT_DIR/.npm-cache"
export UNSLOTH_STUDIO_HOME="$ROOT_DIR/.studio"
export UNSLOTH_CACHE_DIR="$ROOT_DIR/.studio/cache"
EOF
    fi
}

if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed
    write_local_activation_env
    VENV_PY="$VENV_DIR/bin/python"
    if [ -n "$INSTALL_EXTRA" ]; then
        uv pip install --python "$VENV_PY" -e "$ROOT_DIR[$INSTALL_EXTRA]" --torch-backend=auto
    else
        uv pip install --python "$VENV_PY" -e "$ROOT_DIR"
    fi
else
    echo "uv was not found; falling back to python venv + pip."
    echo "Tip: install uv for automatic PyTorch backend selection."
    PYTHON_BIN=${PYTHON_BIN:-python3}
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    write_local_activation_env
    VENV_PY="$VENV_DIR/bin/python"
    "$VENV_PY" -m pip install --upgrade pip setuptools wheel
    if [ -n "$INSTALL_EXTRA" ]; then
        "$VENV_PY" -m pip install -e "$ROOT_DIR[$INSTALL_EXTRA]"
    else
        "$VENV_PY" -m pip install -e "$ROOT_DIR"
    fi
fi

if [ "$INSTALL_STUDIO" = true ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: Studio source install requires uv on PATH." >&2
        echo "Install uv first, then re-run ./install-source.sh." >&2
        exit 1
    fi
    echo "Installing Unsloth Studio locally into: $UNSLOTH_STUDIO_HOME"
    if [ "$NO_TORCH" = true ]; then
        "$ROOT_DIR/install.sh" --local --no-torch
    else
        "$ROOT_DIR/install.sh" --local
    fi
fi

"$VENV_PY" - "$ROOT_DIR" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
if root not in (prefix, *prefix.parents):
    raise SystemExit(f"ERROR: environment escaped the checkout: {prefix}")
print(f"Python: {sys.executable}")
PY

cat <<EOF

Done.
Activate with:
  source "$VENV_DIR/bin/activate"
Run Studio with:
  unsloth studio -H 127.0.0.1 -p 8888
EOF
