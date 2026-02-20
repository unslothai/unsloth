#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # ── 3. Install Node LTS ──
    echo "Installing Node LTS..."
    run_quiet "nvm install" nvm install --lts
    nvm use --lts > /dev/null 2>&1

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
cd "$SCRIPT_DIR/studio/frontend"
run_quiet "npm install" npm install
run_quiet "npm run build" npm run build
cd "$SCRIPT_DIR"
echo "✅ Frontend built to studio/frontend/dist"

# ── 6. Python venv + deps ──
echo ""
echo "Setting up Python environment..."

# ── 6a. Discover best Python <= 3.12.x ──
BEST_PY=""
BEST_MAJOR=0
BEST_MINOR=0

# Collect candidate python3 binaries (python3, python3.9, python3.10, …)
for candidate in $(compgen -c python3 2>/dev/null | grep -E '^python3(\.[0-9]+)?$' | sort -u); do
    if ! command -v "$candidate" &>/dev/null; then
        continue
    fi
    # Get version string, e.g. "Python 3.11.5"
    ver_str=$("$candidate" --version 2>&1 | awk '{print $2}')
    py_major=$(echo "$ver_str" | cut -d. -f1)
    py_minor=$(echo "$ver_str" | cut -d. -f2)

    # Skip anything that isn't Python 3
    if [ "$py_major" -ne 3 ] 2>/dev/null; then
        continue
    fi

    # Skip versions above 3.12
    if [ "$py_minor" -gt 12 ] 2>/dev/null; then
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
    echo "❌ ERROR: No Python version <= 3.12.x found on this system."
    echo "   Detected Python 3 installations:"
    for candidate in $(compgen -c python3 2>/dev/null | grep -E '^python3(\.[0-9]+)?$' | sort -u); do
        if command -v "$candidate" &>/dev/null; then
            echo "     - $candidate ($($candidate --version 2>&1))"
        fi
    done
    echo ""
    echo "   Please install Python <= 3.12.x for maximum compatibility."
    echo "   For example:  sudo apt install python3.12 python3.12-venv"
    exit 1
fi

BEST_VER=$("$BEST_PY" --version 2>&1 | awk '{print $2}')
echo "✅ Using $BEST_PY ($BEST_VER) — compatible (≤ 3.12.x)"

if [ "$IS_COLAB" = true ]; then
    # Colab: install packages directly without venv
    run_quiet "pip upgrade" pip install --upgrade pip
    echo "   Installing unsloth-zoo + unsloth..."
    run_quiet "pip install unsloth" pip install -r "$SCRIPT_DIR/studio/backend/requirements/base.txt"
    echo "   Installing additional unsloth dependencies..."
    run_quiet "pip install extras" pip install --no-cache-dir -r "$SCRIPT_DIR/studio/backend/requirements/extras.txt"
    run_quiet "pip install extras" pip install --no-deps --no-cache-dir -r "$SCRIPT_DIR/studio/backend/requirements/extras-no-deps.txt"
    run_quiet "pip install torchao+transformers" pip install --force-reinstall --no-cache-dir -r "$SCRIPT_DIR/studio/backend/requirements/overrides.txt"
    run_quiet "pip install triton_kernels" pip install --no-deps -r "$SCRIPT_DIR/studio/backend/requirements/triton-kernels.txt"
    # Patch: override llama_cpp.py with fix from unsloth-zoo branch
    LLAMA_CPP_DST="$(pip show unsloth-zoo | grep -i '^Location:' | awk '{print $2}')/unsloth_zoo/llama_cpp.py"
    curl -sSL "https://raw.githubusercontent.com/unslothai/unsloth-zoo/refs/heads/main/unsloth_zoo/llama_cpp.py" \
        -o "$LLAMA_CPP_DST"
    echo "   Installing studio dependencies..."
    run_quiet "pip install studio" pip install -r "$SCRIPT_DIR/studio/backend/requirements/studio.txt"
    echo "✅ Python dependencies installed"
else
    # Local: create venv (always start fresh to preserve correct install order)
    rm -rf .venv
    "$BEST_PY" -m venv .venv
    source .venv/bin/activate
    run_quiet "pip upgrade" pip install --upgrade pip
    echo "   Installing unsloth-zoo + unsloth..."
    run_quiet "pip install unsloth" pip install -r "$SCRIPT_DIR/studio/backend/requirements/base.txt"
    echo "   Installing additional unsloth dependencies..."
    run_quiet "pip install extras" pip install --no-cache-dir -r "$SCRIPT_DIR/studio/backend/requirements/extras.txt"
    run_quiet "pip install extras" pip install --no-deps --no-cache-dir -r "$SCRIPT_DIR/studio/backend/requirements/extras-no-deps.txt"
    run_quiet "pip install torchao+transformers" pip install --force-reinstall --no-cache-dir -r "$SCRIPT_DIR/studio/backend/requirements/overrides.txt"
    run_quiet "pip install triton_kernels" pip install --no-deps -r "$SCRIPT_DIR/studio/backend/requirements/triton-kernels.txt"
    # Patch: override llama_cpp.py with fix from unsloth-zoo branch
    LLAMA_CPP_DST="$(pip show unsloth-zoo | grep -i '^Location:' | awk '{print $2}')/unsloth_zoo/llama_cpp.py"
    curl -sSL "https://raw.githubusercontent.com/unslothai/unsloth-zoo/refs/heads/main/unsloth_zoo/llama_cpp.py" \
        -o "$LLAMA_CPP_DST"
    echo "   Installing studio dependencies..."
    run_quiet "pip install studio" pip install -r "$SCRIPT_DIR/studio/backend/requirements/studio.txt"
    echo "✅ Python dependencies installed"
    
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

# ── 8. Add shell alias (skip in Colab) ──
# Note: venv activation does NOT persist across terminal sessions.
# This alias hardcodes the venv python path so users don't need to activate.
if [ "$IS_COLAB" = false ]; then
echo ""
REPO_DIR="$SCRIPT_DIR"

# Detect the user's default shell and pick the right rc file
USER_SHELL="$(basename "${SHELL:-/bin/bash}")"
case "$USER_SHELL" in
    zsh)
        SHELL_RC="$HOME/.zshrc"
        ALIAS_BLOCK="alias unsloth-studio='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'
alias unsloth-ui='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'"
        ;;
    fish)
        SHELL_RC="$HOME/.config/fish/config.fish"
        # fish uses 'abbr' or 'function'; a simple alias works via 'alias' in config.fish
        ALIAS_BLOCK="alias unsloth-studio '${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'
alias unsloth-ui '${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'"
        ;;
    ksh)
        SHELL_RC="$HOME/.kshrc"
        ALIAS_BLOCK="alias unsloth-studio='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'
alias unsloth-ui='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'"
        ;;
    *)
        # Default to bash for bash and any other POSIX-compatible shell
        SHELL_RC="$HOME/.bashrc"
        ALIAS_BLOCK="alias unsloth-studio='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'
alias unsloth-ui='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py studio -f ${REPO_DIR}/studio/frontend/dist'"
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
