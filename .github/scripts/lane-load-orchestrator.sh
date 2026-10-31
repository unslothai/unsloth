#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Load-orchestrator freeze suite; also called by studio-load-orchestrator-ci.yml.
# Takes its venv dir as $1: concurrent installs into one site-packages race.
set -euo pipefail

venv_dir="${1:-}"

if [ -n "$venv_dir" ]; then
    python3 -m venv "$venv_dir"
    # shellcheck disable=SC1091
    . "$venv_dir/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install \
    'pytest>=8' \
    'httpx>=0.27,<1' \
    'fastapi>=0.110,<1' \
    'uvicorn>=0.30,<1' \
    'anyio>=4'

python -m pytest -v --tb=short tests/studio/load_freeze/
