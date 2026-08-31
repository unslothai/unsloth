#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# The load-orchestrator freeze suite, in ONE definition.
#
# This used to be a whole workflow holding its own runner slot for ~33s. It now runs as a
# background lane inside Lint CI, which was going to occupy a runner on every commit
# anyway. studio-load-orchestrator-ci.yml still exists as a manual escape hatch and calls
# this same script, so the two cannot drift apart.
#
# Deliberately installs no torch and no unsloth: that is what keeps it cheap enough to run
# on every commit rather than only on the four paths that used to trigger it.
#
# Takes its venv directory as $1 so the caller decides the isolation. Inside Lint CI that
# is a private venv, because the lane installs packages while the foreground lint steps are
# using the job's interpreter, and two concurrent installs into one site-packages race.
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
