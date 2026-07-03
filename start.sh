#!/usr/bin/env bash
# Build the studio frontend from local source, then serve it via the REPO's
# studio backend (studio/backend/run.py) run under the installed studio venv,
# so local BACKEND edits take effect (plain `unsloth studio` runs the installed
# package's backend instead). Uses the self-contained Node in
# installer_files/node so the system Node is never touched.
set -euo pipefail

PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/studio/frontend"
NODE_BIN="$SCRIPT_DIR/installer_files/node/bin"

if [ ! -x "$NODE_BIN/node" ]; then
  echo "ERROR: local Node not found at $NODE_BIN" >&2
  echo "Install it once, e.g.:" >&2
  echo "  curl -fL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C /tmp" >&2
  echo "  mv /tmp/node-v22.12.0-linux-x64 $SCRIPT_DIR/installer_files/node" >&2
  exit 1
fi

export PATH="$NODE_BIN:$PATH"
echo "==> Using $(node -v) / npm $(npm -v)"

cd "$FRONTEND_DIR"

if [ ! -d node_modules ] || ! cmp -s package-lock.json node_modules/.lockfile-stamp; then
  echo "==> Installing frontend dependencies"
  npm ci
  cp package-lock.json node_modules/.lockfile-stamp
fi

echo "==> Building frontend ($FRONTEND_DIR)"
npm run build

STUDIO_VENV_PY="${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth/studio}/unsloth_studio/bin/python"
RUN_PY="$SCRIPT_DIR/studio/backend/run.py"

if [ ! -x "$STUDIO_VENV_PY" ]; then
  echo "ERROR: studio venv python not found at $STUDIO_VENV_PY" >&2
  echo "Run 'unsloth studio' once to create the venv, or set UNSLOTH_STUDIO_HOME." >&2
  exit 1
fi

echo "==> Starting repo backend on $HOST:$PORT (frontend: $FRONTEND_DIR/dist)"
echo "    backend: $RUN_PY"
echo "    python:  $STUDIO_VENV_PY"
cd "$SCRIPT_DIR/studio/backend"
exec "$STUDIO_VENV_PY" "$RUN_PY" --host "$HOST" --port "$PORT" --frontend "$FRONTEND_DIR/dist"
