# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def test_subprocess_crash_message_includes_signal_and_oom_hint():
    spec = importlib.util.spec_from_file_location(
        "inference_orchestrator_under_test",
        Path(__file__).resolve().parent.parent / "core/inference/orchestrator.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    orchestrator._proc = SimpleNamespace(pid = 1234, exitcode = -9)

    msg = orchestrator._subprocess_crash_message("wait")

    assert msg.startswith(
        "The inference worker stopped unexpectedly while loading the model."
    )
    assert "memory pressure" in msg
    assert "smaller model" in msg
    assert "Details:" in msg
    assert "signal=SIGKILL" in msg
    assert "exitcode=-9" in msg
