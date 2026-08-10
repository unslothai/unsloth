# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF's own stdout progress reporting must not be teed into the server log.

The training subprocess has no terminal: its stdout goes into the server log. HF
writes a tqdm bar there (ProgressCallback) or, with disable_tqdm, a raw dict per
step (PrinterCallback). Over one 58 minute session that was 1095 bar lines and 266
raw step dicts, and because tqdm and the structlog JSON writer share the stream with
no line discipline, 152 records ended up unparseable.

Everything those lines carry is already published twice: the throttled
`training_progress` event from #7087 and the per-step SSE stream the UI charts.
`unsloth studio --verbose` restores both.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from core.training import trainer as tmod  # noqa: E402

_VERBOSE_ENV = (
    "UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS",
    "UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS",
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in _VERBOSE_ENV:
        monkeypatch.delenv(name, raising = False)


class _FakeTrainer:
    def __init__(self):
        self.removed = []

    def remove_callback(self, cls):
        self.removed.append(cls)


def test_bars_are_disabled_by_default():
    assert tmod._hf_stdout_progress_disabled() is True


def test_verbose_restores_the_bars(monkeypatch):
    for name in _VERBOSE_ENV:
        monkeypatch.setenv(name, "0")
    assert tmod._verbose_logging_requested() is True
    assert tmod._hf_stdout_progress_disabled() is False


def test_only_zeroing_both_windows_counts_as_verbose(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", "0")
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", "10000")
    assert tmod._verbose_logging_requested() is False


def test_unparseable_env_is_not_verbose(monkeypatch):
    for name in _VERBOSE_ENV:
        monkeypatch.setenv(name, "not-a-number")
    assert tmod._verbose_logging_requested() is False


def test_both_stdout_callbacks_are_removed():
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    fake = _FakeTrainer()
    tmod._drop_hf_stdout_callbacks(fake)
    assert set(fake.removed) == {PrinterCallback, ProgressCallback}


def test_verbose_keeps_the_callbacks(monkeypatch):
    for name in _VERBOSE_ENV:
        monkeypatch.setenv(name, "0")
    fake = _FakeTrainer()
    tmod._drop_hf_stdout_callbacks(fake)
    assert fake.removed == []


def test_a_trainer_that_rejects_removal_does_not_raise():
    class _Hostile:
        def remove_callback(self, cls):
            raise RuntimeError("no callbacks here")

    tmod._drop_hf_stdout_callbacks(_Hostile())  # must not propagate


def test_a_trainer_without_remove_callback_does_not_raise():
    tmod._drop_hf_stdout_callbacks(object())
