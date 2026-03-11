#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

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

# ── 1. Check existing Node/npm versions ──
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

# ── 4b. Check / install FFmpeg (required for audio model support) ──
if command -v ffmpeg &>/dev/null; then
    echo "✅ FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "⚠️  FFmpeg not found — installing..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -y > /dev/null 2>&1
        sudo apt-get install -y ffmpeg > /dev/null 2>&1
    elif command -v yum &>/dev/null; then
        sudo yum install -y ffmpeg > /dev/null 2>&1
    elif command -v brew &>/dev/null; then
        brew install ffmpeg > /dev/null 2>&1
    fi
    if command -v ffmpeg &>/dev/null; then
        echo "✅ FFmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
    else
        echo "⚠️  Could not install FFmpeg automatically."
        echo "   Audio model support requires FFmpeg. Install it manually:"
        echo "   Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "   macOS: brew install ffmpeg"
    fi
fi

# ── 5. Build frontend ──
echo ""
echo "Building frontend..."
cd "$SCRIPT_DIR/frontend"
run_quiet "npm install" npm install
run_quiet "npm run build" npm run build
cd "$SCRIPT_DIR/backend/core/data_recipe/oxc-validator"
run_quiet "npm install (oxc validator runtime)" npm install
cd "$SCRIPT_DIR"
echo "✅ Frontend built to frontend/dist"

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
    # Local: create venv (always start fresh to preserve correct install order)
    cd "$REPO_ROOT"
    rm -rf .venv
    rm -rf .venv_overlay  # Remove legacy overlay (no longer used)
    rm -rf .venv_t5       # Will be rebuilt below
    "$BEST_PY" -m venv .venv
    source .venv/bin/activate
    cd "$SCRIPT_DIR"
    install_python_stack

    # ── 6b. Pre-install transformers 5.x into .venv_t5/ ──
    # Models like GLM-4.7-Flash need transformers>=5.2.0. Instead of pip-installing
    # at runtime (slow, ~10-15s), we pre-install into a separate directory.
    # The training subprocess just prepends .venv_t5/ to sys.path — instant switch.
    echo ""
    echo "   Pre-installing transformers 5.x for newer model support..."
    VENV_T5_DIR="$REPO_ROOT/.venv_t5"
    mkdir -p "$VENV_T5_DIR"
    run_quiet "pip install transformers 5.x" pip install --target "$VENV_T5_DIR" --no-deps "transformers==5.2.0"
    run_quiet "pip install huggingface_hub for t5" pip install --target "$VENV_T5_DIR" --no-deps "huggingface_hub==1.3.0"
    echo "✅ Transformers 5.x pre-installed to .venv_t5/"

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
        if [ -f "$LLAMA_SERVER_BIN" ]; then
            echo "✅ llama-server already exists at $LLAMA_SERVER_BIN"
        else
            echo "Building llama-server for GGUF inference..."

            BUILD_OK=true
            if [ -d "$LLAMA_CPP_DIR/.git" ]; then
                echo "   llama.cpp repo already cloned, pulling latest..."
                run_quiet "pull llama.cpp" git -C "$LLAMA_CPP_DIR" pull || echo "   ⚠️  git pull failed — using existing source"
            elif [ -e "$LLAMA_CPP_DIR" ]; then
                echo "   Removing non-git llama.cpp dir..."
                rm -rf "$LLAMA_CPP_DIR"
                run_quiet "clone llama.cpp" git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR" || BUILD_OK=false
            else
                run_quiet "clone llama.cpp" git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR" || BUILD_OK=false
            fi

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
                elif [ -d /usr/local/cuda ] || nvidia-smi &>/dev/null; then
                    echo "   CUDA driver detected but nvcc not found — building CPU-only"
                    echo "   To enable GPU: install cuda-toolkit or add nvcc to PATH"
                else
                    echo "   Building CPU-only (no CUDA detected)..."
                fi

                NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

                run_quiet "cmake llama.cpp" cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" $CMAKE_ARGS || BUILD_OK=false
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
    fi
}

# ── 9. Add shell alias (skip in Colab) ──
# Note: venv activation does NOT persist across terminal sessions.
# This alias hardcodes the venv python path so users don't need to activate.
if [ "$IS_COLAB" = false ]; then
echo ""
REPO_DIR="$REPO_ROOT"

# Detect the user's default shell and pick the right rc file
USER_SHELL="$(basename "${SHELL:-/bin/bash}")"
case "$USER_SHELL" in
    zsh)
        SHELL_RC="$HOME/.zshrc"
        ALIAS_BLOCK="alias unsloth-studio='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'
alias unsloth-ui='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'"
        ;;
    fish)
        SHELL_RC="$HOME/.config/fish/config.fish"
        ALIAS_BLOCK="alias unsloth-studio '${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'
alias unsloth-ui '${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'"
        ;;
    ksh)
        SHELL_RC="$HOME/.kshrc"
        ALIAS_BLOCK="alias unsloth-studio='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'
alias unsloth-ui='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'"
        ;;
    *)
        SHELL_RC="$HOME/.bashrc"
        ALIAS_BLOCK="alias unsloth-studio='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'
alias unsloth-ui='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${SCRIPT_DIR}/frontend/dist'"
        ;;
esac

echo "   Detected shell: $USER_SHELL → $SHELL_RC"

ALIAS_ADDED=false
if ! grep -qF "unsloth-studio" "$SHELL_RC" 2>/dev/null; then
    mkdir -p "$(dirname "$SHELL_RC")"   # needed for fish's nested config path
    cat >> "$SHELL_RC" <<UNSLOTH_EOF

# Unsloth Studio launcher
$ALIAS_BLOCK
UNSLOTH_EOF
    echo "✅ Aliases 'unsloth-studio' and 'unsloth-ui' added to $SHELL_RC"
    ALIAS_ADDED=true
else
    echo "✅ Aliases 'unsloth-studio' and 'unsloth-ui' already exist in $SHELL_RC"
fi

fi  # End of "if not Colab" for shell alias setup

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
    if [ "$ALIAS_ADDED" = true ]; then
        echo "║ Run 'source $SHELL_RC'"
        echo "║ or open a new terminal, then:       ║"
    else
        echo "║ Launch with:                         ║"
    fi
    echo "║                                      ║"
    echo "║ unsloth-studio -H 0.0.0.0 -p 8000   ║"
    echo "╚══════════════════════════════════════╝"
fi
