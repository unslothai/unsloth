# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression lock for the precondition the whole Xet fallback rests on.

The HTTP fallback only works if ``HF_HUB_DISABLE_XET=1`` actually disables Xet in
the installed ``huggingface_hub``. The var is read at import time, so this asserts
it in a FRESH interpreter (huggingface/huggingface_hub#3266 was a version where it
was ignored). No GPU, no network.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _has_hf() -> bool:
    try:
        import huggingface_hub  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_hf(), reason = "huggingface_hub not installed")
def test_disable_xet_constant_set_in_fresh_interpreter():
    code = (
        "from huggingface_hub import constants as c; "
        "import sys; "
        "sys.exit(0 if c.HF_HUB_DISABLE_XET is True else 17)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env = {"HF_HUB_DISABLE_XET": "1", "PATH": _safe_path()},
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, (
        f"HF_HUB_DISABLE_XET=1 did not set constants.HF_HUB_DISABLE_XET=True "
        f"(rc={proc.returncode}): {proc.stderr}"
    )


@pytest.mark.skipif(not _has_hf(), reason = "huggingface_hub not installed")
def test_default_leaves_xet_enabled():
    code = (
        "from huggingface_hub import constants as c; "
        "import sys; "
        "sys.exit(0 if c.HF_HUB_DISABLE_XET is False else 17)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env = {"PATH": _safe_path()},  # no HF_HUB_DISABLE_XET
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, (
        f"without the env var, constants.HF_HUB_DISABLE_XET was not False "
        f"(rc={proc.returncode}): {proc.stderr}"
    )


def _safe_path() -> str:
    import os

    return os.environ.get("PATH", "")
