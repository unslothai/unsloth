#!/bin/sh
# Unsloth Studio Installer
# Usage (curl):  curl -fsSL https://unsloth.ai/install.sh | sh
# Usage (wget):  wget -qO- https://unsloth.ai/install.sh | sh
# Usage (local): ./install.sh --local   (install from local repo instead of PyPI)
# Usage (test):  ./install.sh --package roland-sloth  (install a different package name)
set -e

# ── Parse flags ──
STUDIO_LOCAL_INSTALL=false
PACKAGE_NAME="unsloth"
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
    esac
done

if [ "$_next_is_package" = true ]; then
    echo "❌ ERROR: --package requires an argument." >&2
    exit 1
fi

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
    # Parse CUDA version from nvidia-smi output
    _cuda_ver=$($_smi 2>/dev/null | grep -oP 'CUDA Version:\s+\K[0-9]+\.[0-9]+' | head -1)
    if [ -z "$_cuda_ver" ]; then echo "$_base/cu126"; return; fi  # unparseable -> default
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
_VENV_PY="$VENV_DIR/bin/python"
if [ "$_MIGRATED" = true ]; then
    # Migrated env: force-reinstall unsloth+unsloth-zoo to ensure clean state
    # in the new venv location, while preserving existing torch/CUDA
    echo "==> Upgrading unsloth in migrated environment..."
    uv pip install --python "$_VENV_PY" \
        --reinstall-package unsloth --reinstall-package unsloth-zoo \
        "unsloth>=2026.3.11" unsloth-zoo
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        echo "==> Overlaying local repo (editable)..."
        uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
    fi
elif [ -n "$TORCH_INDEX_URL" ]; then
    # Fresh: Step 1 - install torch from explicit index
    echo "==> Installing PyTorch ($TORCH_INDEX_URL)..."
    uv pip install --python "$_VENV_PY" torch torchvision torchaudio \
        --index-url "$TORCH_INDEX_URL"
    # Fresh: Step 2 - install unsloth, preserving pre-installed torch
    echo "==> Installing unsloth (this may take a few minutes)..."
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        uv pip install --python "$_VENV_PY" \
            --upgrade-package unsloth "unsloth>=2026.3.11" unsloth-zoo
        echo "==> Overlaying local repo (editable)..."
        uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
    else
        uv pip install --python "$_VENV_PY" \
            --upgrade-package unsloth "$PACKAGE_NAME"
    fi
else
    # Fallback: GPU detection failed to produce a URL -- let uv resolve torch
    echo "==> Installing unsloth (this may take a few minutes)..."
    if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
        uv pip install --python "$_VENV_PY" unsloth-zoo "unsloth>=2026.3.11" --torch-backend=auto
        echo "==> Overlaying local repo (editable)..."
        uv pip install --python "$_VENV_PY" -e "$_REPO_ROOT" --no-deps
    else
        uv pip install --python "$_VENV_PY" "$PACKAGE_NAME" --torch-backend=auto
    fi
fi

# ── Run studio setup ──
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
if [ "$STUDIO_LOCAL_INSTALL" = true ]; then
    SKIP_STUDIO_BASE=1 \
    STUDIO_PACKAGE_NAME="$PACKAGE_NAME" \
    STUDIO_LOCAL_INSTALL=1 \
    STUDIO_LOCAL_REPO="$_REPO_ROOT" \
    bash "$SETUP_SH" </dev/null
else
    SKIP_STUDIO_BASE=1 \
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
