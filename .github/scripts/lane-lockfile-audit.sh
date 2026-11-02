#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Lockfile supply-chain audit; also called by lockfile-audit.yml. Stdlib only, no venv.
set -euo pipefail

python3 -c "import ast; ast.parse(open('scripts/lockfile_supply_chain_audit.py').read())"
python3 scripts/lockfile_supply_chain_audit.py
