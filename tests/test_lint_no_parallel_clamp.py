# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression suite for scripts/lint_no_parallel_clamp.py.

The lint is what stops #7717 coming back: a rule that flags `max(1, n)` gets
disabled, and one that misses `n_parallel = 1` protects nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "lint_no_parallel_clamp.py"

_spec = importlib.util.spec_from_file_location("lint_no_parallel_clamp", _SCRIPT)
lint = importlib.util.module_from_spec(_spec)
sys.modules["lint_no_parallel_clamp"] = lint
_spec.loader.exec_module(lint)


CLAMPS = (
    "def load():\n    n_parallel = 1\n",
    "def load():\n    n_parallel = _mtp_clamped_slots\n",
    "async def load():\n    n_parallel = 1\n",
    "def load():\n    if mtp:\n        n_parallel = 1\n",
)

ALLOWED = (
    # The two shapes that survive: a real capability limit, and a real resource limit.
    "def load():\n    n_parallel = 1  # allow-slot-clamp: no --kv-unified\n",
    "def load(fit):\n    n_parallel = fit.slots\n",
    # Structurally distinct, so no marker is needed for any of these.
    "def load(n_parallel: int = 1):\n    return n_parallel\n",
    "class A:\n    n_parallel: int = 1\n",
    "def load():\n    self._requested_n_parallel = 1\n",
    "def load(x):\n    n_parallel = max(1, x)\n",
    "def load(s):\n    n_parallel = getattr(s, 'llama_parallel_slots', 1)\n",
    "def load(r):\n    n_parallel = r.n_parallel\n",
    "n_parallel = 1\n",
)


@pytest.mark.parametrize("source", CLAMPS)
def test_a_silent_downgrade_is_flagged(source):
    assert lint.scan_source(source, "<test>")


@pytest.mark.parametrize("source", ALLOWED)
def test_a_legitimate_slot_count_is_not_flagged(source):
    assert lint.scan_source(source, "<test>") == []


def test_the_scripts_own_self_test_passes():
    assert lint._self_test() == 0


def test_the_studio_backend_is_clean():
    found, scanned = lint.scan_paths(lint.DEFAULT_SCAN_DIR)
    assert scanned, "no backend source files were scanned"
    assert found == [], f"silent parallel-slot downgrade(s): {found}"
