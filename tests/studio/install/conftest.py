# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pytest config for studio/install tests: add studio/ to sys.path so `backend` imports work from the repo root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# <repo-root>/studio  →  makes `backend` importable as a package
_STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))

# These two cannot go in the per-test fixture below: install_python_stack reads them once
# at module scope, into NO_TORCH and _TORCH_BACKEND, and for a test module that import
# happens during collection, before any fixture runs. A shell that exports either --
# install.sh exports both -- would otherwise return every routing plan early and fail the
# suite on the environment rather than on the code. Recompute the constants too, since a
# whole-tree run collects other directories first and may already have imported the module.
os.environ.pop("UNSLOTH_NO_TORCH", None)
os.environ.pop("UNSLOTH_TORCH_BACKEND", None)
_ips = sys.modules.get("install_python_stack")
if _ips is not None and hasattr(_ips, "_infer_no_torch"):
    _ips.NO_TORCH = _ips._infer_no_torch()
if _ips is not None and hasattr(_ips, "_infer_torch_backend"):
    # Not simply "": UNSLOTH_TORCH_INDEX_URL / _FAMILY still feed the derivation, and those
    # are deliberately left alone (see _isolate_torch_routing_env).
    _ips._TORCH_BACKEND = _ips._infer_torch_backend()


# Every variable here steers torch routing, so a developer or a CI runner that exports
# one -- which is exactly what an AMD host does -- silently changes what these tests
# assert. Cleared per test rather than per session so a test that wants one can still
# set it: monkeypatch.setenv in the test body runs after this fixture and wins.
_ROUTING_ENV_VARS = (
    "UNSLOTH_ROCM_GFX_ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "UNSLOTH_ROCM_TORCH_INSTALLED",
    "ROCM_PATH",
    # install.sh exports this, so a suite run from an installer shell would assert the
    # fresh-install branch of the recorded-pin guards instead of the update one.
    "UNSLOTH_INSTALLER_RESOLVED_TORCH_INDEX",
)


@pytest.fixture(autouse = True)
def _isolate_torch_routing_env(monkeypatch):
    """Run every install test against an unset routing environment.

    Without this the suite reads the HOST's AMD configuration. Measured one variable
    at a time over the two pure-Python routing files, against 604 passing on a clean
    environment: UNSLOTH_ROCM_GFX_ARCH=gfx906 gave 47 failures,
    UNSLOTH_ROCM_TORCH_INSTALLED 27, ROCM_PATH 19, HSA_OVERRIDE_GFX_VERSION 3 and
    CUDA_VISIBLE_DEVICES 1. All six are clean with this fixture in place.

    UNSLOTH_TORCH_INDEX_URL / _FAMILY are deliberately NOT cleared here: their
    sensitivity predates this file (29 failures at the merge base) and reaches the
    tests through paths a per-test fixture does not cover, so silencing them here
    would hide a gap rather than close it.
    """
    for name in _ROUTING_ENV_VARS:
        monkeypatch.delenv(name, raising = False)


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
