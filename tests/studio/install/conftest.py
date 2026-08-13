# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pytest config for studio/install tests: add studio/ to sys.path so `backend` imports work from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# <repo-root>/studio  →  makes `backend` importable as a package
_STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))


@pytest.fixture(autouse = True)
def _isolate_rocm_repair_ledger(tmp_path, monkeypatch):
    """Keep the ROCm repair ledger out of the interpreter's own prefix.

    _ensure_rocm_torch records an attempt under venv_root() before installing, so a
    test that drives it would otherwise write into sys.prefix and make the NEXT run
    of an unrelated test see a repair as already attempted.
    """
    import install_manifest

    ledger = tmp_path / "ledger"
    ledger.mkdir()
    monkeypatch.setattr(
        install_manifest,
        "rocm_repair_marker_path",
        lambda root = None: (root or ledger) / install_manifest.ROCM_REPAIR_MARKER,
    )
