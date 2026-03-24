#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RULE=$(printf '\342\224\200%.0s' {1..52})

# ── Colors (same palette as startup_banner / install_python_stack) ──
if [ -n "${NO_COLOR:-}" ]; then
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
elif [ -t 1 ] || [ -n "${FORCE_COLOR:-}" ]; then
    C_TITLE=$'\033[38;5;150m'
    C_DIM=$'\033[38;5;245m'
    C_OK=$'\033[38;5;108m'
    C_WARN=$'\033[38;5;136m'
    C_ERR=$'\033[91m'
    C_RST=$'\033[0m'
else
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
fi

# ── Output helpers ──
# Consistent column layout: 2-space indent, 15-char label (fits llama-quantize), then value.
# Usage: step <label> <message> [color]   (color defaults to C_OK)
step()    { printf "  ${C_DIM}%-15.15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "$1"; }

# ── Helper: run command quietly, show output only on failure ──
_run_quiet() {
    local on_fail=$1
    local label=$2
    shift 2

    local tmplog
    tmplog=$(mktemp) || {
        step "error" "Failed to create temporary file" "$C_ERR"
        [ "$on_fail" = "exit" ] && exit 1 || return 1
    }

    if "$@" >"$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        step "error" "$label failed (exit code $exit_code)" "$C_ERR"
        cat "$tmplog" >&2
        rm -f "$tmplog"

        if [ "$on_fail" = "exit" ]; then
            exit "$exit_code"
        else
            return "$exit_code"
        fi
    fi
}

run_quiet() {
    _run_quiet exit "$@"
}

run_quiet_no_exit() {
    _run_quiet return "$@"
}

# Like run_quiet_no_exit but only prints output on failure when UNSLOTH_VERBOSE=1.
# Used for truly optional steps (e.g. llama-quantize) where failure is expected.
try_quiet() {
    local label="$1"; shift
    local tmplog; tmplog=$(mktemp)
    if "$@" > "$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        if [ "${UNSLOTH_VERBOSE:-0}" = "1" ]; then
            step "error" "$label failed (exit code $exit_code)" "$C_ERR"
            cat "$tmplog" >&2
        fi
        rm -f "$tmplog"
        return $exit_code
    fi
}

# ── Banner ──
echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "🦥 Unsloth Studio Setup"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"

# ── Clean up stale caches ──
rm -rf "$REPO_ROOT/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/backend/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/tmp/unsloth_compiled_cache"

# ── Detect Colab ──
IS_COLAB=false
keynames=$'\n'$(printenv | cut -d= -f1)
if [[ "$keynames" == *$'\nCOLAB_'* ]]; then
    IS_COLAB=true
fi

# ── Frontend ──
_NEED_FRONTEND_BUILD=true
if [ -d "$SCRIPT_DIR/frontend/dist" ]; then
    _changed=$(find "$SCRIPT_DIR/frontend" -maxdepth 1 -type f \
        -newer "$SCRIPT_DIR/frontend/dist" -print -quit 2>/dev/null)
    if [ -z "$_changed" ]; then
        _changed=$(find "$SCRIPT_DIR/frontend/src" "$SCRIPT_DIR/frontend/public" \
            -type f -newer "$SCRIPT_DIR/frontend/dist" -print -quit 2>/dev/null) || true
    fi
    [ -z "$_changed" ] && _NEED_FRONTEND_BUILD=false
fi

if [ "$_NEED_FRONTEND_BUILD" = false ]; then
    step "frontend" "up to date"
else

# ── Node ──
NEED_NODE=true
if command -v node &>/dev/null && command -v npm &>/dev/null; then
    NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
    NPM_MAJOR=$(npm -v | cut -d. -f1)
    if [ "$NODE_MAJOR" -ge 20 ] && [ "$NPM_MAJOR" -ge 11 ]; then
        NEED_NODE=false
    else
        if [ "$IS_COLAB" = true ]; then
            if [ "$NPM_MAJOR" -lt 11 ]; then
                substep "upgrading npm..."
                npm install -g npm@latest > /dev/null 2>&1
            fi
            NEED_NODE=false
        fi
    fi
fi

if [ "$NEED_NODE" = true ]; then
    substep "installing nvm..."
    export NODE_OPTIONS=--dns-result-order=ipv4first
    curl -so- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash > /dev/null 2>&1

    export NVM_DIR="$HOME/.nvm"
    set +u
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    if [ -f "$HOME/.npmrc" ]; then
        if grep -qE '^\s*(prefix|globalconfig)\s*=' "$HOME/.npmrc"; then
            sed -i.bak '/^\s*\(prefix\|globalconfig\)\s*=/d' "$HOME/.npmrc"
        fi
    fi

    substep "installing Node LTS..."
    run_quiet "nvm install" nvm install --lts
    nvm use --lts > /dev/null 2>&1
    set -u

    NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
    NPM_MAJOR=$(npm -v | cut -d. -f1)

    if [ "$NODE_MAJOR" -lt 20 ]; then
        step "node" "FAILED — version must be >= 20 (got $(node -v))" "$C_ERR"
        exit 1
    fi
    if [ "$NPM_MAJOR" -lt 11 ]; then
        substep "upgrading npm..."
        run_quiet "npm update" npm install -g npm@latest
    fi
fi

step "node" "$(node -v) | npm $(npm -v)"

# ── Build frontend ──
substep "building frontend..."
cd "$SCRIPT_DIR/frontend"

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
    for _gi in "${_HIDDEN_GITIGNORES[@]+"${_HIDDEN_GITIGNORES[@]}"}"; do
        mv "${_gi}._twbuild" "$_gi" 2>/dev/null || true
    done
}
trap _restore_gitignores EXIT

run_quiet "npm install" npm install
run_quiet "npm run build" npm run build

_restore_gitignores
trap - EXIT

_MAX_CSS=$(find "$SCRIPT_DIR/frontend/dist/assets" -name '*.css' -exec wc -c {} + 2>/dev/null | sort -n | tail -1 | awk '{print $1}')
if [ -z "$_MAX_CSS" ]; then
    step "frontend" "built (warning: no CSS emitted)" "$C_WARN"
elif [ "$_MAX_CSS" -lt 100000 ]; then
    step "frontend" "built (warning: CSS may be truncated)" "$C_WARN"
else
    step "frontend" "built"
fi

cd "$SCRIPT_DIR"

fi  # end frontend build check

# ── oxc-validator runtime ──
if [ -d "$SCRIPT_DIR/backend/core/data_recipe/oxc-validator" ] && command -v npm &>/dev/null; then
    cd "$SCRIPT_DIR/backend/core/data_recipe/oxc-validator"
    run_quiet "npm install (oxc validator runtime)" npm install
    cd "$SCRIPT_DIR"
fi

# ── Python ──
MIN_PY_MINOR=11
MAX_PY_MINOR=13
BEST_PY=""
BEST_MINOR=0

# If the caller (e.g. install.sh) already chose a Python, use it directly.
if [ -n "${REQUESTED_PYTHON_VERSION:-}" ] && [ -x "$REQUESTED_PYTHON_VERSION" ]; then
    _req_ver=$("$REQUESTED_PYTHON_VERSION" --version 2>&1 | awk '{print $2}')
    _req_major=$(echo "$_req_ver" | cut -d. -f1)
    _req_minor=$(echo "$_req_ver" | cut -d. -f2)
    if [ "$_req_major" -eq 3 ] 2>/dev/null && \
       [ "$_req_minor" -ge "$MIN_PY_MINOR" ] 2>/dev/null && \
       [ "$_req_minor" -le "$MAX_PY_MINOR" ] 2>/dev/null; then
        BEST_PY="$REQUESTED_PYTHON_VERSION"
        substep "using requested Python: $BEST_PY"
    else
        substep "ignoring requested Python $REQUESTED_PYTHON_VERSION ($_req_ver) -- outside range"
    fi
fi

if [ -z "$BEST_PY" ]; then
for candidate in $(compgen -c python3 2>/dev/null | grep -E '^python3(\.[0-9]+)?$' | sort -u); do
    if ! command -v "$candidate" &>/dev/null; then continue; fi
    ver_str=$("$candidate" --version 2>&1) || continue
    ver_str=$(echo "$ver_str" | awk '{print $2}')
    py_major=$(echo "$ver_str" | cut -d. -f1)
    py_minor=$(echo "$ver_str" | cut -d. -f2)
    [ "$py_major" -ne 3 ] 2>/dev/null && continue
    [ "$py_minor" -lt "$MIN_PY_MINOR" ] 2>/dev/null && continue
    [ "$py_minor" -gt "$MAX_PY_MINOR" ] 2>/dev/null && continue
    if [ "$py_minor" -gt "$BEST_MINOR" ]; then
        BEST_PY="$candidate"
        BEST_MINOR="$py_minor"
    fi
done
fi

if [ -z "$BEST_PY" ]; then
    step "python" "no compatible version found (need 3.${MIN_PY_MINOR}–3.${MAX_PY_MINOR})" "$C_ERR"
    substep "detected:"
    for candidate in $(compgen -c python3 2>/dev/null | grep -E '^python3(\.[0-9]+)?$' | sort -u); do
        command -v "$candidate" &>/dev/null && substep "  $candidate ($($candidate --version 2>&1))"
    done
    substep "install with: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

BEST_VER=$("$BEST_PY" --version 2>&1 | awk '{print $2}')
step "python" "$BEST_VER (3.${MIN_PY_MINOR}.x – 3.${MAX_PY_MINOR}.x)"

# ── Venv + Python deps ──
REQ_ROOT="$SCRIPT_DIR/backend/requirements"
SINGLE_ENV_CONSTRAINTS="$REQ_ROOT/single-env/constraints.txt"
SINGLE_ENV_DATA_DESIGNER="$REQ_ROOT/single-env/data-designer.txt"
SINGLE_ENV_DATA_DESIGNER_DEPS="$REQ_ROOT/single-env/data-designer-deps.txt"
SINGLE_ENV_PATCH="$REQ_ROOT/single-env/patch_metadata.py"

install_python_stack() {
    python "$SCRIPT_DIR/install_python_stack.py"
}

STUDIO_HOME="$HOME/.unsloth/studio"
VENV_DIR="$STUDIO_HOME/.venv"
VENV_T5_DIR="$STUDIO_HOME/.venv_t5"
mkdir -p "$STUDIO_HOME"

[ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
[ -d "$REPO_ROOT/.venv_overlay" ] && rm -rf "$REPO_ROOT/.venv_overlay"
[ -d "$REPO_ROOT/.venv_t5" ] && rm -rf "$REPO_ROOT/.venv_t5"

rm -rf "$VENV_DIR"
rm -rf "$VENV_T5_DIR"
if ! "$BEST_PY" -m venv "$VENV_DIR" 2>/dev/null; then
    "$BEST_PY" -m venv --without-pip "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python > /dev/null
else
    source "$VENV_DIR/bin/activate"
fi

USE_UV=false
if command -v uv &>/dev/null; then
    USE_UV=true
elif curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null && USE_UV=true
fi

fast_install() {
    if [ "$USE_UV" = true ]; then
        uv pip install --python "$(command -v python)" "$@" && return 0
    fi
    python -m pip install "$@"
}

cd "$SCRIPT_DIR"
install_python_stack

# ── Transformers 5.x ──
mkdir -p "$VENV_T5_DIR"
run_quiet "install transformers 5.x" fast_install --target "$VENV_T5_DIR" --no-deps "transformers==5.3.0"
run_quiet "install huggingface_hub for t5" fast_install --target "$VENV_T5_DIR" --no-deps "huggingface_hub==1.7.1"
run_quiet "install hf_xet for t5" fast_install --target "$VENV_T5_DIR" --no-deps "hf_xet==1.4.2"
run_quiet "install tiktoken for t5" fast_install --target "$VENV_T5_DIR" "tiktoken"
step "transformers" "5.x pre-installed"

# ── WSL: GGUF build dependencies ──
if grep -qi microsoft /proc/version 2>/dev/null; then
    _GGUF_DEPS="pciutils build-essential cmake curl git libcurl4-openssl-dev"
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y $_GGUF_DEPS >/dev/null 2>&1 || true

    _STILL_MISSING=""
    for _pkg in $_GGUF_DEPS; do
        case "$_pkg" in
            build-essential) command -v gcc >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            pciutils) command -v lspci >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            libcurl4-openssl-dev) dpkg -s "$_pkg" >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            *) command -v "$_pkg" >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
        esac
    done
    _STILL_MISSING=$(echo "$_STILL_MISSING" | sed 's/^ *//')

    if [ -z "$_STILL_MISSING" ]; then
        step "gguf deps" "installed"
    elif command -v sudo >/dev/null 2>&1; then
        step "gguf deps" "sudo required for: $_STILL_MISSING" "$C_WARN"
        printf "  %-15s" ""
        printf "accept? [Y/n] "
        if [ -r /dev/tty ]; then
            read -r REPLY </dev/tty || REPLY="y"
        else
            REPLY="y"
        fi
        case "$REPLY" in
            [nN]*)
                substep "skipped — run manually:"
                substep "sudo apt-get install -y $_STILL_MISSING"
                _SKIP_GGUF_BUILD=true
                ;;
            *)
                sudo apt-get update -y
                sudo apt-get install -y $_STILL_MISSING
                step "gguf deps" "installed"
                ;;
        esac
    else
        step "gguf deps" "missing (no sudo) — install manually:" "$C_WARN"
        substep "apt-get install -y $_STILL_MISSING"
        _SKIP_GGUF_BUILD=true
    fi
fi

# ── llama.cpp ──
UNSLOTH_HOME="$HOME/.unsloth"
mkdir -p "$UNSLOTH_HOME"
LLAMA_CPP_DIR="$UNSLOTH_HOME/llama.cpp"
LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
if [ "${_SKIP_GGUF_BUILD:-}" = true ]; then
    step "llama.cpp" "skipped (missing build deps)" "$C_WARN"
else
rm -rf "$LLAMA_CPP_DIR"
{
    if ! command -v cmake &>/dev/null; then
        step "llama.cpp" "skipped (cmake not found)" "$C_WARN"
    elif ! command -v git &>/dev/null; then
        step "llama.cpp" "skipped (git not found)" "$C_WARN"
    else
        BUILD_OK=true
        run_quiet_no_exit "clone llama.cpp" git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR" || BUILD_OK=false

        if [ "$BUILD_OK" = true ]; then
            CMAKE_ARGS="-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=ON"

            if command -v ccache &>/dev/null; then
                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
            fi

            NVCC_PATH=""
            if command -v nvcc &>/dev/null; then
                NVCC_PATH="$(command -v nvcc)"
            elif [ -x /usr/local/cuda/bin/nvcc ]; then
                NVCC_PATH="/usr/local/cuda/bin/nvcc"
                export PATH="/usr/local/cuda/bin:$PATH"
            elif ls /usr/local/cuda-*/bin/nvcc &>/dev/null 2>&1; then
                NVCC_PATH="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
                export PATH="$(dirname "$NVCC_PATH"):$PATH"
            fi

            _BUILD_DESC="building"
            if [ -n "$NVCC_PATH" ]; then
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"

                CUDA_ARCHS=""
                if command -v nvidia-smi &>/dev/null; then
                    _raw_caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
                    while IFS= read -r _cap; do
                        _cap=$(echo "$_cap" | tr -d '[:space:]')
                        if [[ "$_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
                            _arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
                            case ";$CUDA_ARCHS;" in
                                *";$_arch;"*) ;;
                                *) CUDA_ARCHS="${CUDA_ARCHS:+$CUDA_ARCHS;}$_arch" ;;
                            esac
                        fi
                    done <<< "$_raw_caps"
                fi

                if [ -n "$CUDA_ARCHS" ]; then
                    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
                    _BUILD_DESC="building (CUDA, sm_${CUDA_ARCHS//;/+sm_})"
                else
                    _BUILD_DESC="building (CUDA)"
                fi

                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS=--threads=0"
            else
                _BUILD_DESC="building (CPU)"
            fi

            substep "$_BUILD_DESC..."

            NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
            CMAKE_GENERATOR_ARGS=""
            if command -v ninja &>/dev/null; then
                CMAKE_GENERATOR_ARGS="-G Ninja"
            fi

            run_quiet_no_exit "cmake llama.cpp" cmake $CMAKE_GENERATOR_ARGS -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" $CMAKE_ARGS || BUILD_OK=false
        fi

        if [ "$BUILD_OK" = true ]; then
            run_quiet_no_exit "build llama-server" cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server -j"$NCPU" || BUILD_OK=false
        fi

        if [ "$BUILD_OK" = true ]; then
            run_quiet_no_exit "build llama-quantize" cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-quantize -j"$NCPU" || true
            # Symlink to llama.cpp root -- check_llama_cpp() looks for the binary there
            QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
            if [ -f "$QUANTIZE_BIN" ]; then
                ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
            fi
        fi

        if [ "$BUILD_OK" = true ] && [ -f "$LLAMA_SERVER_BIN" ]; then
            step "llama.cpp" "built"
            [ -f "$LLAMA_CPP_DIR/llama-quantize" ] && step "llama-quantize" "built"
        elif [ "$BUILD_OK" = true ]; then
            step "llama.cpp" "binary not found after build" "$C_WARN"
        else
            step "llama.cpp" "build failed" "$C_ERR"
        fi
    fi
}
fi  # end _SKIP_GGUF_BUILD check

# ── Footer ──
# Colab + launch example: match unslothai/unsloth (upstream main) studio/setup.sh
if [ "$IS_COLAB" = true ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║           Setup Complete!            ║"
    echo "╠══════════════════════════════════════╣"
    echo "║ Unsloth Studio is ready to start     ║"
    echo "║ in your Colab notebook!              ║"
    echo "║                                      ║"
    echo "║ from colab import start              ║"
    echo "║ start()                              ║"
    echo "╚══════════════════════════════════════╝"
else
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    printf "  ${C_TITLE}%s${C_RST}\n" "Unsloth Studio Installed"
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    printf "  ${C_DIM}%-15s${C_OK}%s${C_RST}\n" "launch" "unsloth studio -H 0.0.0.0 -p 8888"
fi
echo ""
