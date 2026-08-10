# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_model_defaults announces a resolution once, then drops to debug.

GET /api/inference/status resolves the defaults on every poll and the UI polls it
every 5s for as long as a tab is open, so the unconditional info line repeated
forever. The first resolution still logs at info; repeats stay visible at debug.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.models import model_config  # noqa: E402
from utils.models.model_config import load_model_defaults  # noqa: E402


class _RecordingLogger:
    """Stand-in for the module's structlog logger; caplog cannot see structlog."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def _record(self, level: str):
        def log(msg, *args, **kwargs):
            self.calls.append((level, str(msg)))
        return log

    def __getattr__(self, name: str):
        if name in ("debug", "info", "warning", "error"):
            return self._record(name)
        raise AttributeError(name)

    def at(self, level: str, needle: str) -> list[str]:
        return [m for lvl, m in self.calls if lvl == level and needle in m]


def _reset(monkeypatch) -> _RecordingLogger:
    model_config._ANNOUNCED_MODEL_DEFAULTS.clear()
    rec = _RecordingLogger()
    monkeypatch.setattr(model_config, "logger", rec)
    return rec


def test_repeat_resolution_logs_once_at_info(monkeypatch):
    rec = _reset(monkeypatch)
    for _ in range(5):
        load_model_defaults("definitely-not-a-real-model-xyz")
    assert len(rec.at("info", "defaults from")) == 1, rec.calls


def test_repeats_still_visible_at_debug(monkeypatch):
    rec = _reset(monkeypatch)
    for _ in range(3):
        load_model_defaults("definitely-not-a-real-model-xyz")
    # First call is the info line, the other two are debug: nothing is lost.
    assert len(rec.at("debug", "defaults from")) == 2, rec.calls


def test_a_different_model_gets_its_own_announcement(monkeypatch):
    rec = _reset(monkeypatch)
    load_model_defaults("definitely-not-a-real-model-xyz")
    load_model_defaults("also-not-a-real-model-abc")
    assert len(rec.at("info", "defaults from")) == 2, rec.calls


def test_announced_set_is_bounded(monkeypatch):
    _reset(monkeypatch)
    for i in range(model_config._ANNOUNCED_MODEL_DEFAULTS_MAX + 10):
        model_config._log_model_defaults(f"msg {i}", f"key-{i}")
    assert (
        len(model_config._ANNOUNCED_MODEL_DEFAULTS)
        <= model_config._ANNOUNCED_MODEL_DEFAULTS_MAX
    )
    model_config._ANNOUNCED_MODEL_DEFAULTS.clear()


def test_returned_config_is_unchanged_by_the_dedup(monkeypatch):
    _reset(monkeypatch)
    first = load_model_defaults("definitely-not-a-real-model-xyz")
    second = load_model_defaults("definitely-not-a-real-model-xyz")
    assert isinstance(first, dict)
    assert first == second
