# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: the training worker must not import ``transformers`` before it activates the
transformers sidecar.

``core/training/worker.py:run_training_process`` runs a preflight (Xet decision, logging, hardware
detection) and only THEN calls ``_activate_transformers_version`` -> ``activate_transformers_for_subprocess``,
which prepends the correct ``.venv_t5_*`` (5.x) sidecar to ``sys.path``. Because activation only edits
``sys.path``, it is a no-op for any module already cached in ``sys.modules``. So if the preflight imports
``transformers`` (directly or transitively via ``unsloth_zoo``), the default 4.57.x gets pinned before
the sidecar is on the path -- and 5.x models (Qwen3.5, GLM-4.7, gemma-4) then fail to load their
tokenizer/config ("Tokenizer class TokenizersBackend does not exist").

This regression shipped once when ``utils/hf_xet_fallback.py`` eagerly imported ``unsloth_zoo`` (which
imports ``transformers``) at module load; the worker imports that shim during preflight to decide the
Xet env flip (see issue #6951). This test locks the invariant in a fresh interpreter. It is CPU-only,
needs no network/GPU/weights/sidecars, so it runs in the standard ``studio-backend-ci`` matrix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend

# Mirrors run_training_process's imports that run BEFORE _activate_transformers_version (worker.py);
# keep in sync. torch-dependent imports are optional (a no-torch CI shard skips them) but must still
# not drag in transformers.
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
