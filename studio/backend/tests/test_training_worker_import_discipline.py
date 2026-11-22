# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The worker must not import transformers before the sidecar is activated, or 4.57 gets pinned."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent

# Mirrors run_training_process's imports before _activate_transformers_version
# (worker.py); keep in sync. Must not pull in transformers.
_PREFLIGHT_SNIPPET = r"""
import sys

# worker.py: from utils.hf_xet_fallback import child_should_disable_xet  (+ call it)
from utils.hf_xet_fallback import child_should_disable_xet
child_should_disable_xet({})

# worker.py: from loggers.config import LogConfig
from loggers.config import LogConfig  # noqa: F401

# worker.py: from utils.hardware import hardware  (imports torch, not transformers)
try:
    from utils.hardware import hardware as _hw  # noqa: F401
except Exception:
    pass  # torch may be absent in a no-torch shard; the invariant below still applies

# worker.py: from .training import is_apple_silicon_training_platform, should_use_mlx_training_backend
# (the MLX-dispatch preflight; must also stay clear of transformers). Guarded because it may pull
# unsloth/trl, absent in a minimal shard -- but a partial import that leaked transformers would still
# be caught by the assertion below.
try:
    from core.training.training import (  # noqa: F401
        is_apple_silicon_training_platform as _is_apple,
        should_use_mlx_training_backend as _use_mlx,
    )
except Exception:
    pass

leaked_tf = sorted(m for m in sys.modules if m == "transformers" or m.startswith("transformers."))
leaked_zoo = sorted(m for m in sys.modules if m == "unsloth_zoo" or m.startswith("unsloth_zoo."))
assert not leaked_tf, f"transformers imported during worker preflight (before sidecar activation): {leaked_tf}"
assert not leaked_zoo, f"unsloth_zoo imported during worker preflight (before sidecar activation): {leaked_zoo}"
print("PREFLIGHT_CLEAN")
"""


def test_worker_preflight_does_not_import_transformers():
    """A fresh interpreter running the worker's pre-activation imports must leave ``transformers``
    (and ``unsloth_zoo``) unimported, so the 5.x sidecar prepend is not defeated by a stale module."""
    result = subprocess.run(
        [sys.executable, "-c", _PREFLIGHT_SNIPPET],
        cwd = str(_BACKEND_DIR),
        capture_output = True,
        text = True,
    )
    assert result.returncode == 0, (
        "Worker preflight imported transformers before sidecar activation.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "PREFLIGHT_CLEAN" in result.stdout, result.stdout
