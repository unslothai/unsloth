#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Helper: run command quietly, show output only on failure ──
run_quiet() {
    local label="$1"
    shift
    local tmplog
    tmplog=$(mktemp)
    if "$@" > "$tmplog" 2>&1; then
        rm -f "$tmplog"
    else
        local exit_code=$?
        echo "❌ $label failed (exit code $exit_code):"
        cat "$tmplog"
        rm -f "$tmplog"
        exit $exit_code
    fi
}

echo "╔══════════════════════════════════════╗"
echo "║     Unsloth Studio Setup Script      ║"
echo "╚══════════════════════════════════════╝"

# ── Clean up stale Unsloth compiled caches ──
rm -rf "$REPO_ROOT/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/backend/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/tmp/unsloth_compiled_cache"

# ── Detect Colab (like unsloth does) ──
IS_COLAB=false
keynames=$'\n'$(printenv | cut -d= -f1)
if [[ "$keynames" == *$'\nCOLAB_'* ]]; then
    IS_COLAB=true
fi

# ── Detect whether frontend needs building ──
# Only skip when BOTH conditions are true:
#   1. We're inside site-packages (PyPI / pip install, not editable)
#   2. dist/ already exists (pre-built in the wheel)
# Otherwise always (re)build — handles upgrades, editable installs, and
# pip-from-source where dist/ was never built.
if [[ "$SCRIPT_DIR" == */site-packages/* ]] && [ -d "$SCRIPT_DIR/frontend/dist" ]; then
    echo "✅ Frontend pre-built (PyPI) — skipping Node/npm check."
else
NEED_NODE=true
if command -v node &>/dev/null && command -v npm &>/dev/null; then
    NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
    NPM_MAJOR=$(npm -v | cut -d. -f1)
    if [ "$NODE_MAJOR" -ge 20 ] && [ "$NPM_MAJOR" -ge 11 ]; then
        echo "✅ Node $(node -v) and npm $(npm -v) already meet requirements. Skipping nvm install."
        NEED_NODE=false
    else
        if [ "$IS_COLAB" = true ]; then
            echo "✅ Node $(node -v) and npm $(npm -v) detected in Colab."
            # In Colab, just upgrade npm directly - nvm doesn't work well
            if [ "$NPM_MAJOR" -lt 11 ]; then
                echo "   Upgrading npm to latest..."
                npm install -g npm@latest > /dev/null 2>&1
            fi
            NEED_NODE=false
        else
            echo "⚠️  Node $(node -v) / npm $(npm -v) too old. Installing via nvm..."
        fi
    fi
else
    echo "⚠️  Node/npm not found. Installing via nvm..."
fi

if [ "$NEED_NODE" = true ]; then
    # ── 2. Install nvm ──
    export NODE_OPTIONS=--dns-result-order=ipv4first # or else fails on colab.
    echo "Installing nvm..."
    curl -so- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash > /dev/null 2>&1

    # Load nvm (source ~/.bashrc won't work inside a script)
    export NVM_DIR="$HOME/.nvm"
    set +u
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # ── Fix npmrc conflict with nvm ──
    # System npm (apt, conda, etc.) may have written `prefix` or `globalconfig`
    # to ~/.npmrc, which is incompatible with nvm and causes "nvm use" to fail
    # with: "has a `globalconfig` and/or a `prefix` setting, which are
    # incompatible with nvm."
    if [ -f "$HOME/.npmrc" ]; then
        if grep -qE '^\s*(prefix|globalconfig)\s*=' "$HOME/.npmrc"; then
            echo "   Removing incompatible prefix/globalconfig from ~/.npmrc for nvm..."
            sed -i.bak '/^\s*\(prefix\|globalconfig\)\s*=/d' "$HOME/.npmrc"
        fi
    fi

    # ── 3. Install Node LTS ──
    echo "Installing Node LTS..."
    run_quiet "nvm install" nvm install --lts
    nvm use --lts > /dev/null 2>&1
    set -u
    # ── 4. Verify versions ──
    NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
    NPM_MAJOR=$(npm -v | cut -d. -f1)

    if [ "$NODE_MAJOR" -lt 20 ]; then
        echo "❌ ERROR: Node version must be >= 20 (got $(node -v))"
        exit 1
    fi
    if [ "$NPM_MAJOR" -lt 11 ]; then
        echo "⚠️  npm version is $(npm -v), expected >= 11. Updating..."
        run_quiet "npm update" npm install -g npm@latest
    fi
fi

echo "✅ Node $(node -v) | npm $(npm -v)"

# ── 5. Build frontend ──
echo ""
echo "Building frontend..."
cd "$SCRIPT_DIR/frontend"

# Tailwind v4's oxide scanner respects .gitignore in parent directories.
# Python venvs create a .gitignore with "*" (ignore everything), which
# prevents Tailwind from scanning .tsx source files for class names.
# Temporarily hide any such .gitignore during the build, then restore it.
_HIDDEN_GITIGNORES=()
_dir="$(pwd)"
while [ "$_dir" != "/" ]; do
    _dir="$(dirname "$_dir")"
    if [ -f "$_dir/.gitignore" ] && grep -qx '\*' "$_dir/.gitignore" 2>/dev/null; then
        mv "$_dir/.gitignore" "$_dir/.gitignore._twbuild"
        _HIDDEN_GITIGNORES+=("$_dir/.gitignore")
    fi
done

_restore_gitignores() {
    for _gi in "${_HIDDEN_GITIGNORES[@]}"; do
        mv "${_gi}._twbuild" "$_gi" 2>/dev/null || true
    done
}
trap _restore_gitignores EXIT

run_quiet "npm install" npm install
run_quiet "npm run build" npm run build

_restore_gitignores
trap - EXIT
cd "$SCRIPT_DIR/backend/core/data_recipe/oxc-validator"
run_quiet "npm install (oxc validator runtime)" npm install
cd "$SCRIPT_DIR"
echo "✅ Frontend built to frontend/dist"

fi  # end frontend dist check

# ── 6. Python venv + deps ──
echo ""
echo "Setting up Python environment..."

# ── 6a. Discover best Python >= 3.11 and < 3.14 (i.e. 3.11.x, 3.12.x, or 3.13.x) ──
MIN_PY_MINOR=11   # minimum minor version (>= 3.11)
MAX_PY_MINOR=13   # maximum minor version (< 3.14)
BEST_PY=""
BEST_MAJOR=0
BEST_MINOR=0

# Collect candidate python3 binaries (python3, python3.9, python3.10, …)
for candidate in $(compgen -c python3 2>/dev/null | grep -E '^python3(\.[0-9]+)?$' | sort -u); do
    if ! command -v "$candidate" &>/dev/null; then
        continue
    fi
    # Get version string, e.g. "Python 3.12.5"
    ver_str=$("$candidate" --version 2>&1 | awk '{print $2}')
    py_major=$(echo "$ver_str" | cut -d. -f1)
    py_minor=$(echo "$ver_str" | cut -d. -f2)

    # Skip anything that isn't Python 3
    if [ "$py_major" -ne 3 ] 2>/dev/null; then
        continue
    fi

    # Skip versions below 3.12 (require > 3.11)
    if [ "$py_minor" -lt "$MIN_PY_MINOR" ] 2>/dev/null; then
        continue
    fi

    # Skip versions above 3.13 (require < 3.14)
    if [ "$py_minor" -gt "$MAX_PY_MINOR" ] 2>/dev/null; then
        continue
    fi

    # Keep the highest qualifying version
    if [ "$py_minor" -gt "$BEST_MINOR" ]; then
        BEST_PY="$candidate"
        BEST_MAJOR="$py_major"
        BEST_MINOR="$py_minor"
    fi
done

if [ -z "$BEST_PY" ]; then
    echo "❌ ERROR: No Python version between 3.${MIN_PY_MINOR} and 3.${MAX_PY_MINOR} found on this system."
    echo "   Detected Python 3 installations:"
    for candidate in $(compgen -c python3 2>/dev/null | grep -E '^python3(\.[0-9]+)?$' | sort -u); do
        if command -v "$candidate" &>/dev/null; then
            echo "     - $candidate ($($candidate --version 2>&1))"
        fi
    done
    echo ""
    echo "   Please install Python 3.${MIN_PY_MINOR} or 3.${MAX_PY_MINOR}."
    echo "   For example:  sudo apt install python3.12 python3.12-venv"
    exit 1
fi

BEST_VER=$("$BEST_PY" --version 2>&1 | awk '{print $2}')
echo "✅ Using $BEST_PY ($BEST_VER) — compatible (3.${MIN_PY_MINOR}.x – 3.${MAX_PY_MINOR}.x)"

REQ_ROOT="$SCRIPT_DIR/backend/requirements"
SINGLE_ENV_CONSTRAINTS="$REQ_ROOT/single-env/constraints.txt"
SINGLE_ENV_DATA_DESIGNER="$REQ_ROOT/single-env/data-designer.txt"
SINGLE_ENV_DATA_DESIGNER_DEPS="$REQ_ROOT/single-env/data-designer-deps.txt"
SINGLE_ENV_PATCH="$REQ_ROOT/single-env/patch_metadata.py"

install_python_stack() {
    python "$SCRIPT_DIR/install_python_stack.py"
}

if [ "$IS_COLAB" = true ]; then
    # Colab: install packages directly without venv
    install_python_stack
else
    # Local: create venv under ~/.unsloth/studio/ (shared location, not in repo)
    STUDIO_HOME="$HOME/.unsloth/studio"
    VENV_DIR="$STUDIO_HOME/.venv"
    VENV_T5_DIR="$STUDIO_HOME/.venv_t5"
    mkdir -p "$STUDIO_HOME"

    # Clean up legacy in-repo venvs if they exist
    [ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
    [ -d "$REPO_ROOT/.venv_overlay" ] && rm -rf "$REPO_ROOT/.venv_overlay"
    [ -d "$REPO_ROOT/.venv_t5" ] && rm -rf "$REPO_ROOT/.venv_t5"

    rm -rf "$VENV_DIR"
    rm -rf "$VENV_T5_DIR"
    "$BEST_PY" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    cd "$SCRIPT_DIR"
    install_python_stack

    # ── 6b. Pre-install transformers 5.x into .venv_t5/ ──
    # Models like GLM-4.7-Flash need transformers>=5.2.0. Instead of pip-installing
    # at runtime (slow, ~10-15s), we pre-install into a separate directory.
    # The training subprocess just prepends .venv_t5/ to sys.path — instant switch.
    echo ""
    echo "   Pre-installing transformers 5.x for newer model support..."
    mkdir -p "$VENV_T5_DIR"
    run_quiet "pip install transformers 5.x" pip install --target "$VENV_T5_DIR" --no-deps "transformers==5.2.0"
    run_quiet "pip install huggingface_hub for t5" pip install --target "$VENV_T5_DIR" --no-deps "huggingface_hub==1.3.0"
    echo "✅ Transformers 5.x pre-installed to $VENV_T5_DIR/"

    # ── 7. WSL: pre-install GGUF build dependencies ──
    # On WSL, sudo requires a password and can't be entered during GGUF export
    # (runs in a non-interactive subprocess). Install build deps here instead.
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo ""
        echo "⚠️  WSL detected — installing build dependencies for GGUF export..."
        echo "   You may be prompted for your password."
        sudo apt-get update -y
        sudo apt-get install -y build-essential cmake curl git libcurl4-openssl-dev
        echo "✅ GGUF build dependencies installed"
    fi
fi

# ── 8. Build llama.cpp binaries for GGUF inference + export ──
# Builds at ~/.unsloth/llama.cpp — a single shared location under the user's
# home directory. This is used by both the inference server and the GGUF
# export pipeline (unsloth-zoo).
#   - llama-server: for GGUF model inference
#   - llama-quantize: for GGUF export quantization (symlinked to root for check_llama_cpp())
UNSLOTH_HOME="$HOME/.unsloth"
mkdir -p "$UNSLOTH_HOME"
LLAMA_CPP_DIR="$UNSLOTH_HOME/llama.cpp"
LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
rm -rf "$LLAMA_CPP_DIR"
{
    # Check prerequisites
    if ! command -v cmake &>/dev/null; then
        echo ""
        echo "⚠️  cmake not found — skipping llama-server build (GGUF inference won't be available)"
        echo "   Install cmake and re-run setup.sh to enable GGUF inference."
    elif ! command -v git &>/dev/null; then
        echo ""
        echo "⚠️  git not found — skipping llama-server build (GGUF inference won't be available)"
    else
        echo ""
        echo "Building llama-server for GGUF inference..."

        BUILD_OK=true
        run_quiet "clone llama.cpp" git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR" || BUILD_OK=false

        if [ "$BUILD_OK" = true ]; then
            CMAKE_ARGS=""
            # Detect CUDA: check nvcc on PATH, then common install locations
            NVCC_PATH=""
            if command -v nvcc &>/dev/null; then
                NVCC_PATH="$(command -v nvcc)"
            elif [ -x /usr/local/cuda/bin/nvcc ]; then
                NVCC_PATH="/usr/local/cuda/bin/nvcc"
                export PATH="/usr/local/cuda/bin:$PATH"
            elif ls /usr/local/cuda-*/bin/nvcc &>/dev/null 2>&1; then
                # Pick the newest cuda-XX.X directory
                NVCC_PATH="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
                export PATH="$(dirname "$NVCC_PATH"):$PATH"
            fi

            if [ -n "$NVCC_PATH" ]; then
                echo "   Building with CUDA support (nvcc: $NVCC_PATH)..."
                CMAKE_ARGS="-DGGML_CUDA=ON"

                # Detect GPU compute capability and limit CUDA architectures
                # Without this, cmake builds for ALL default archs (very slow)
                CUDA_ARCHS=""
                if command -v nvidia-smi &>/dev/null; then
                    # Read all GPUs, deduplicate (handles mixed-GPU hosts)
                    _raw_caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
                    while IFS= read -r _cap; do
                        _cap=$(echo "$_cap" | tr -d '[:space:]')
                        if [[ "$_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
                            _arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
                            # Append if not already present
                            case ";$CUDA_ARCHS;" in
                                *";$_arch;"*) ;;
                                *) CUDA_ARCHS="${CUDA_ARCHS:+$CUDA_ARCHS;}$_arch" ;;
                            esac
                        fi
                    done <<< "$_raw_caps"
                fi

                if [ -n "$CUDA_ARCHS" ]; then
                    echo "   GPU compute capabilities: ${CUDA_ARCHS//;/, } -- limiting build to detected archs"
                    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
                else
                    echo "   Could not detect GPU arch -- building for all default CUDA architectures (slower)"
                fi

                # Multi-threaded nvcc compilation (uses all CPU cores per .cu file)
                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS=--threads=0"
            elif [ -d /usr/local/cuda ] || nvidia-smi &>/dev/null; then
                echo "   CUDA driver detected but nvcc not found — building CPU-only"
                echo "   To enable GPU: install cuda-toolkit or add nvcc to PATH"
            else
                echo "   Building CPU-only (no CUDA detected)..."
            fi

            NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

            # Use Ninja if available (faster parallel builds than Make)
            CMAKE_GENERATOR_ARGS=""
            if command -v ninja &>/dev/null; then
                CMAKE_GENERATOR_ARGS="-G Ninja"
            fi

            run_quiet "cmake llama.cpp" cmake $CMAKE_GENERATOR_ARGS -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" $CMAKE_ARGS || BUILD_OK=false
        fi

        if [ "$BUILD_OK" = true ]; then
            run_quiet "build llama-server" cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server -j"$NCPU" || BUILD_OK=false
        fi

        # Also build llama-quantize (needed by unsloth-zoo's GGUF export pipeline)
        if [ "$BUILD_OK" = true ]; then
            run_quiet "build llama-quantize" cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-quantize -j"$NCPU" || true
            # Symlink to llama.cpp root — check_llama_cpp() looks for the binary there
            QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
            if [ -f "$QUANTIZE_BIN" ]; then
                ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
            fi
        fi

        if [ "$BUILD_OK" = true ]; then
            if [ -f "$LLAMA_SERVER_BIN" ]; then
                echo "✅ llama-server built at $LLAMA_SERVER_BIN"
            else
                echo "⚠️  llama-server binary not found after build — GGUF inference won't be available"
            fi
            if [ -f "$LLAMA_CPP_DIR/llama-quantize" ]; then
                echo "✅ llama-quantize available for GGUF export"
            fi
        else
            echo "⚠️  llama-server build failed — GGUF inference won't be available, but everything else works"
        fi
    fi
}

echo ""
if [ "$IS_COLAB" = true ]; then
    echo "╔══════════════════════════════════════╗"
    echo "║           Setup Complete!            ║"
    echo "╠══════════════════════════════════════╣"
    echo "║ Unsloth Studio is ready to start    ║"
    echo "║ in your Colab notebook!              ║"
    echo "╚══════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════╗"
    echo "║           Setup Complete!            ║"
    echo "╠══════════════════════════════════════╣"
    echo "║ Launch with:                         ║"
    echo "║                                      ║"
    echo "║ unsloth studio -H 0.0.0.0 -p 8000   ║"
    echo "╚══════════════════════════════════════╝"
fi
