# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: ``import main`` must not import torch, and the warm that replaces
it must be safe.

uvicorn binds the listening socket only after ``import main`` and the lifespan
have both finished, so anything either of them imports is time the login screen
does not exist. torch (plus the sympy/scipy/pandas/sklearn it drags in through
transformers) was about 5s of that on a GPU host, for data no request needs
before it is asked for. Four eager edges caused it:

  utils/models/model_config.py  _build_detection_sets() at module scope
  routes/models.py              from core.inference import get_inference_backend
  core/inference/orchestrator.py  from utils.hf_xet_fallback import DownloadStallError
  utils/datasets/raw_text.py    from datasets import Dataset  (annotation only)

All four are lazy now and hardware detection moved off the lifespan onto
utils/torch_warmup.py. These tests lock that in: a fresh interpreter for the
import invariant (importing in-process would measure an already-warm
sys.modules), plus unit coverage of the warm's idempotency and of the
single-detection guarantee the endpoints rely on.

CPU-only, no network, no GPU, no weights -- runs in the standard
studio-backend-ci matrix.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend

_HEAVY = ("torch", "transformers", "unsloth_zoo", "scipy", "sklearn", "sympy")

_IMPORT_MAIN_SNIPPET = r"""
import sys

# Never start the warm here: it would import the very modules under test, and
# the assertion below would race it.
import os
os.environ["UNSLOTH_STUDIO_DISABLE_TORCH_WARM"] = "1"

import main  # noqa: F401

HEAVY = %(heavy)r
leaked = sorted(
    m for m in sys.modules
    if any(m == h or m.startswith(h + ".") for h in HEAVY)
)
assert not leaked, "import main pulled heavy ML modules: %%s" %% (leaked,)
print("IMPORT_MAIN_CLEAN")
""" % {"heavy": list(_HEAVY)}


def _run(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd = str(_BACKEND_DIR),
        capture_output = True,
        text = True,
        timeout = 900,
    )


def test_import_main_does_not_import_torch():
    """`import main` must leave torch and its scientific stack unimported."""
    proc = _run(_IMPORT_MAIN_SNIPPET)
    assert proc.returncode == 0, (
        f"import main was not clean\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    )
    assert "IMPORT_MAIN_CLEAN" in proc.stdout


@pytest.mark.parametrize(
    "module_path",
    [
        "utils.models.model_config",
        "utils.datasets.raw_text",
        "core.inference.orchestrator",
        "routes.models",
    ],
)
def test_module_import_does_not_pull_torch(module_path: str):
    """Each module that used to force torch must import clean on its own."""
    snippet = (
        "import sys, importlib\n"
        f"importlib.import_module({module_path!r})\n"
        f"HEAVY = {list(_HEAVY)!r}\n"
        "leaked = sorted(m for m in sys.modules"
        " if any(m == h or m.startswith(h + '.') for h in HEAVY))\n"
        "assert not leaked, leaked\n"
        "print('CLEAN')\n"
    )
    proc = _run(snippet)
    assert proc.returncode == 0, f"{module_path}: {proc.stderr[-3000:]}"
    assert "CLEAN" in proc.stdout


def test_detection_sets_still_resolve_under_their_old_names():
    """The PEP 562 shim must keep `from ... import _VLM_MODEL_TYPES` working."""
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")
    from utils.models.model_config import (
        _AUDIO_ONLY_MODEL_TYPES,
        _VISION_CHECK_INLINE_HELPERS,
        _VISION_CHECK_SCRIPT,
        _VLM_CLASS_NAMES,
        _VLM_MODEL_TYPES,
    )

    assert "llava" in _VLM_MODEL_TYPES
    assert {"csm", "whisper"} <= _AUDIO_ONLY_MODEL_TYPES
    assert _VLM_CLASS_NAMES
    assert "_VLM_MODEL_TYPES = " in _VISION_CHECK_INLINE_HELPERS
    assert _VISION_CHECK_INLINE_HELPERS in _VISION_CHECK_SCRIPT


def test_detection_sets_are_built_once_under_concurrency():
    """Two requests racing the warm must not each read the registry."""
    from utils.models import model_config as mc

    calls = []
    original = mc._build_detection_sets
    saved = mc._DETECTION_SETS
    mc._DETECTION_SETS = None
    try:
        def counted():
            calls.append(1)
            return (frozenset({"x"}), frozenset(), frozenset())

        mc._build_detection_sets = counted
        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(mc._detection_sets())

        threads = [threading.Thread(target = worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(calls) == 1, f"built {len(calls)} times"
        assert all(r is results[0] for r in results)
    finally:
        mc._build_detection_sets = original
        mc._DETECTION_SETS = saved


def test_hardware_is_detected_once_under_concurrency():
    """ensure_hardware_detected() collapses the warm thread and any request
    that arrives mid-detection into a single run."""
    from utils.hardware import hardware as hw

    calls = []
    saved_device, saved_impl = hw.DEVICE, hw._detect_hardware_locked
    hw.DEVICE = None
    try:
        def counted():
            calls.append(1)
            hw.DEVICE = hw.DeviceType.CPU
            return hw.DEVICE

        hw._detect_hardware_locked = counted
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            hw.ensure_hardware_detected()

        threads = [threading.Thread(target = worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(calls) == 1, f"detected {len(calls)} times"
    finally:
        hw._detect_hardware_locked = saved_impl
        hw.DEVICE = saved_device


def test_warm_starts_once_and_honours_the_kill_switch(monkeypatch):
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_thread", None, raising = False)
    monkeypatch.setenv(torch_warmup.DISABLE_ENV_VAR, "1")
    assert torch_warmup.start_background_warm() is False
    assert torch_warmup.warm_status()["started"] is False

    monkeypatch.delenv(torch_warmup.DISABLE_ENV_VAR)
    monkeypatch.setattr(torch_warmup, "_STAGES", (("noop", lambda: None),))
    assert torch_warmup.start_background_warm() is True
    # Second call is a no-op: one warm thread per process.
    assert torch_warmup.start_background_warm() is False
    assert torch_warmup.join_background_warm(60) is True
    status = torch_warmup.warm_status()
    assert status["finished"] is True
    assert status["stages"]["noop"]["ok"] is True


def test_a_failing_warm_stage_is_reported_not_swallowed(monkeypatch, capsys):
    """A broken stage must be visible in the log and must not kill the warm."""
    from utils import torch_warmup

    def boom():
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(torch_warmup, "_thread", None, raising = False)
    monkeypatch.delenv(torch_warmup.DISABLE_ENV_VAR, raising = False)
    monkeypatch.setattr(
        torch_warmup, "_STAGES", (("boom", boom), ("after", lambda: None))
    )
    assert torch_warmup.start_background_warm() is True
    assert torch_warmup.join_background_warm(60) is True

    status = torch_warmup.warm_status()
    assert status["stages"]["boom"]["ok"] is False
    assert "stage exploded" in status["stages"]["boom"]["error"]
    # The stage after the failure still ran, and the process is still alive.
    assert status["stages"]["after"]["ok"] is True
    assert status["finished"] is True
    # structlog renders to stdout, not through the stdlib root handler caplog
    # attaches to, so assert on what the operator would actually see.
    assert "stage exploded" in capsys.readouterr().out
