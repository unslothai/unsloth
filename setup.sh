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
python3 -m venv .venv
source .venv/bin/activate
run_quiet "pip upgrade" pip install --upgrade pip
echo "   Installing unsloth-zoo + unsloth..."
run_quiet "pip install unsloth" pip install unsloth-zoo unsloth
echo "   Installing studio dependencies..."
run_quiet "pip install extras" pip install typer fastapi uvicorn pydantic matplotlib pandas nest_asyncio "datasets==4.3.0"
echo "✅ Python dependencies installed"

# ── 7. Add shell alias ──
# Note: venv activation does NOT persist across terminal sessions.
# This alias hardcodes the venv python path so users don't need to activate.
echo ""
REPO_DIR="$SCRIPT_DIR"
ALIAS_LINE="alias unsloth-ui='${REPO_DIR}/.venv/bin/python ${REPO_DIR}/cli.py ui'"

if ! grep -qF "unsloth-ui" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Unsloth Studio launcher" >> ~/.bashrc
    echo "$ALIAS_LINE" >> ~/.bashrc
    echo "✅ Alias 'unsloth-ui' added to ~/.bashrc"
else
    echo "✅ Alias 'unsloth-ui' already exists in ~/.bashrc"
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║           Setup Complete!            ║"
echo "╠══════════════════════════════════════╣"
echo "║ Run 'source ~/.bashrc' or open a    ║"
echo "║ new terminal, then launch with:     ║"
echo "║                                      ║"
echo "║ unsloth-ui -H 0.0.0.0 -p 8000 \    ║"
echo "║   -f studio/frontend/dist            ║"
echo "╚══════════════════════════════════════╝"
