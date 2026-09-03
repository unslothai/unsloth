#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# The lockfile supply-chain audit, in ONE definition.
#
# Ran as its own workflow on its own runner slot for ~6s of work. It now runs as a
# background lane inside Lint CI. lockfile-audit.yml keeps its nightly `schedule` and calls
# this same script, so the scheduled audit is unchanged and the two cannot drift apart.
#
# Needs no dependencies at all, which is why it needs no venv: it is stdlib Python over
# files already in the checkout.
set -euo pipefail

# Parses before it runs, so a syntax error in the auditor is reported as a syntax error
# rather than as an audit finding.
python3 -c "import ast; ast.parse(open('scripts/lockfile_supply_chain_audit.py').read())"
python3 scripts/lockfile_supply_chain_audit.py
