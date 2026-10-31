# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#15583: a failed training model load logs its traceback, so a bare "[WinError 6]" names the failing call."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Stubs a dep the backend pytest job lacks, so collection succeeds; real installs are left alone."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - unusable here either way, so stub it
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


pytest.importorskip("torch")
pytest.importorskip("transformers")

_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as trainer_module  # noqa: E402
from core.training.trainer import UnslothTrainer  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


class _RecordingLogger:
    def __init__(self):
        self.errors = []

    def error(self, message, *args, **kwargs):
        self.errors.append((message, kwargs, sys.exc_info()[1]))

    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.mark.parametrize(
    "error",
    [OSError(6, "The handle is invalid", None, 6), RuntimeError("XPU kernel build failed")],
    ids = ["oserror", "other"],
)
def test_model_load_error_is_logged_with_its_traceback(monkeypatch, error):
    import utils.cache_cleanup

    def fail(*_a, **_k):
        raise error

    recorder = _RecordingLogger()
    monkeypatch.setattr(trainer_module, "logger", recorder)
    monkeypatch.setattr(trainer_module, "clear_gpu_cache", lambda *a, **k: None)
    monkeypatch.setattr(utils.cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: None)
    monkeypatch.setattr(trainer_module, "detect_audio_type_checked", lambda *a, **k: (None, True))
    monkeypatch.setattr(trainer_module, "is_vision_model", fail)
    monkeypatch.setattr(trainer_module, "_env_offline", lambda: False)
    trainer = object.__new__(UnslothTrainer)
    trainer.model = trainer.tokenizer = trainer.trainer = None
    progress_errors = []
    trainer._update_progress = (
        lambda **kw: progress_errors.append(kw["error"]) if kw.get("error") else None
    )
    trainer._cleanup_audio_artifacts = lambda: None

    assert trainer.load_model("org/model", hf_token = None) is False

    message, kwargs, active = recorder.errors[-1]
    assert message.startswith("Error loading model:")
    assert kwargs.get("exc_info") is True
    # Logged inside the handler, so exc_info=True resolves to the load's own exception.
    assert active is error
    # The UI still gets the short message, not the traceback.
    assert progress_errors[-1] == str(error)
