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

import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Register a stub module for a dep the backend pytest job does not install.

    Same helper, and the same reason, as in test_training_preflight.py and
    test_training_progress_callback.py: core.training.trainer imports unsloth (and through it
    unsloth_zoo) and trl at module scope, while the pytest matrix in studio-backend-ci.yml
    installs studio.txt plus torch and transformers and stops there. The heavier
    repo-cpu-tests job beside it is the one that installs unsloth_zoo, and it runs the
    REPO-ROOT tests/, not this tree -- so nothing here can rely on those packages being
    present. Unstubbed, this module fails COLLECTION, which fails the whole job rather than
    one test. Real installs are left alone, so a developer box still exercises the genuine
    import. __spec__ = None keeps the trainer's own _ensure_real_packages namespace-shadow
    guard a no-op on the stub."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - any import failure means "not usable here", so stub it
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as tmod  # noqa: E402

# Drop the stubs now that tmod is bound, because they outlive this module otherwise and the rest
# of the suite then runs against them. utils.hardware.hardware._shared_policy branches on
# `"unsloth" in sys.modules` and then reaches for unsloth.dataset_num_proc, which a spec-less
# non-package stub cannot provide, so it returns None and every shared-policy case in
# test_dataset_map_num_proc.py skips instead of running. A real install stubs nothing, so this is
# a no-op there.
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

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
