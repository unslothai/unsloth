#!/usr/bin/env bash

set -euo pipefail

# 1. Build frontend (Vite outputs to dist/)
cd studio/frontend
npm install
npm run build       # outputs to studio/frontend/dist/
cd ../..

# 2. Clean old artifacts
rm -rf build dist *.egg-info

# 3. Build wheel
python -m build

# 4. Optionally publish
if [ "${1:-}" = "publish" ]; then
    python -m twine upload dist/*
fi
