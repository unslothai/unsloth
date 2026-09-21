# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default Chat model metadata must not block on remote Hugging Face discovery."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.orchestrator import InferenceOrchestrator  # noqa: E402


def test_default_models_returns_static_defaults_before_top_fetch(monkeypatch):
    sleep_seconds = 2.0

    def _slow_fetch(self: InferenceOrchestrator) -> None:
        time.sleep(sleep_seconds)
        self._top_gguf_cache = ["unsloth/slow-GGUF"]
        self._top_models_ready.set()

    monkeypatch.setattr(InferenceOrchestrator, "_fetch_top_models", _slow_fetch)

    orchestrator = InferenceOrchestrator()
    started = time.monotonic()
    defaults = orchestrator.default_models
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"default_models blocked for {elapsed:.2f}s"
    assert defaults == orchestrator._static_models
    assert "unsloth/slow-GGUF" not in defaults

    deadline = time.monotonic() + sleep_seconds + 5
    while not orchestrator._top_models_ready.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert "unsloth/slow-GGUF" in orchestrator.default_models
