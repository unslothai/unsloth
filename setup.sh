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

# ── 1. Check existing Node/npm versions ──
NEED_NODE=true
if command -v node &>/dev/null && command -v npm &>/dev/null; then
    NODE_MAJOR=$(node -v | sed 's/v//' | cut -d. -f1)
    NPM_MAJOR=$(npm -v | cut -d. -f1)
    if [ "$NODE_MAJOR" -ge 20 ] && [ "$NPM_MAJOR" -ge 11 ]; then
        echo "✅ Node $(node -v) and npm $(npm -v) already meet requirements. Skipping nvm install."
        NEED_NODE=false
    else
        echo "⚠️  Node $(node -v) / npm $(npm -v) too old. Installing via nvm..."
    fi
else
    echo "⚠️  Node/npm not found. Installing via nvm..."
fi

if [ "$NEED_NODE" = true ]; then
    # ── 2. Install nvm ──
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
DESIRED_PY="${UNSLOTH_PYTHON:-3.12}"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    VENV_VER="$("$SCRIPT_DIR/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if ! "$SCRIPT_DIR/.venv/bin/python" -c 'import pip' >/dev/null 2>&1; then
        VENV_VER="${VENV_VER:-unknown}"
        mv "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/.venv.bak-nopip-py${VENV_VER}-$(date +%Y%m%d%H%M%S)"
    elif [ "$VENV_VER" != "$DESIRED_PY" ]; then
        mv "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/.venv.bak-py${VENV_VER:-unknown}-$(date +%Y%m%d%H%M%S)"
    fi
fi

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    if command -v uv >/dev/null 2>&1; then
        run_quiet "uv python install" uv python install "$DESIRED_PY"
        run_quiet "uv venv" uv venv --seed -p "$DESIRED_PY" "$SCRIPT_DIR/.venv"
    else
        python3 -m venv "$SCRIPT_DIR/.venv"
    fi
fi

# Avoid shell-specific activation; call venv python directly.
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
run_quiet "pip upgrade" "$VENV_PY" -m pip install --upgrade pip
echo "   Installing unsloth-zoo + unsloth..."
run_quiet "pip install unsloth" "$VENV_PY" -m pip install unsloth-zoo unsloth
echo "   Installing studio dependencies..."
run_quiet "pip install extras" "$VENV_PY" -m pip install typer fastapi uvicorn pydantic matplotlib pandas nest_asyncio "datasets==4.3.0" pyjwt easydict addict
echo "✅ Python dependencies installed"

# ── 7. Add shell alias ──
# Note: venv activation does NOT persist across terminal sessions.
# This alias hardcodes the venv python path so users don't need to activate.
echo ""
REPO_DIR="$SCRIPT_DIR"
ALIAS_CMD="${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py ui -f ${REPO_DIR}/studio/frontend/dist"

SHELL_RC_HINT="source ~/.bashrc"
if [[ "${SHELL:-}" == *fish ]]; then
    FISH_RC="$HOME/.config/fish/config.fish"
    mkdir -p "$(dirname "$FISH_RC")"
    if ! grep -qF "unsloth-ui" "$FISH_RC" 2>/dev/null; then
        cat >> "$FISH_RC" <<UNSLOTH_EOF

# Unsloth Studio launcher
alias unsloth-ui='${ALIAS_CMD}'
UNSLOTH_EOF
        echo "✅ Alias 'unsloth-ui' added to $FISH_RC"
    else
        echo "✅ Alias 'unsloth-ui' already exists in $FISH_RC"
    fi
    SHELL_RC_HINT="source $FISH_RC"
elif [[ "${SHELL:-}" == *zsh ]]; then
    ZSH_RC="$HOME/.zshrc"
    if ! grep -qF "unsloth-ui" "$ZSH_RC" 2>/dev/null; then
        cat >> "$ZSH_RC" <<UNSLOTH_EOF

# Unsloth Studio launcher
alias unsloth-ui='${ALIAS_CMD}'
UNSLOTH_EOF
        echo "✅ Alias 'unsloth-ui' added to $ZSH_RC"
    else
        echo "✅ Alias 'unsloth-ui' already exists in $ZSH_RC"
    fi
    SHELL_RC_HINT="source $ZSH_RC"
else
    BASH_RC="$HOME/.bashrc"
    if ! grep -qF "unsloth-ui" "$BASH_RC" 2>/dev/null; then
        cat >> "$BASH_RC" <<UNSLOTH_EOF

# Unsloth Studio launcher
alias unsloth-ui='${ALIAS_CMD}'
UNSLOTH_EOF
        echo "✅ Alias 'unsloth-ui' added to $BASH_RC"
    else
        echo "✅ Alias 'unsloth-ui' already exists in $BASH_RC"
    fi
    SHELL_RC_HINT="source $BASH_RC"
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║           Setup Complete!            ║"
echo "╠══════════════════════════════════════╣"
echo "║ Run '$SHELL_RC_HINT' or open a     ║"
echo "║ new terminal, then launch with:     ║"
echo "║                                      ║"
echo "║ unsloth-ui -H 0.0.0.0 -p 8000       ║"
echo "╚══════════════════════════════════════╝"
