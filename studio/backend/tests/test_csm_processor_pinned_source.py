# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CSM dataset step loaded its processor by the bare repo id, so transformers fetched and imported
whatever processor code the repo's main named by then, past the snapshot the consent gate scanned and the
tokenizer loaded. It must load the same pinned source."""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import transformers

_STUBBED: list = []


def _stub_if_missing(name, attrs):
    """Stubs a dep the backend pytest job lacks, so collection succeeds; real installs are left alone."""
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

# Drop the stubs once tmod is bound, so later modules see the real import state.
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


class _Stop(Exception):
    pass


@pytest.fixture(autouse = True)
def _cuda_trainer(monkeypatch):
    # On Apple Silicon UnslothTrainer() hands back the MLX adapter, which has no CSM path.
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda: False)


def test_csm_processor_loads_the_pinned_snapshot(monkeypatch):
    seen = {}

    def from_pretrained(source, **kwargs):
        seen.update(source = source, **kwargs)
        raise _Stop

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    t = tmod.UnslothTrainer()
    t.model_name = "mallory/voice"
    t.trust_remote_code = True
    t._remember_model_source("/cache/models--mallory--voice/snapshots/abc", "abc", True)
    with pytest.raises(_Stop):
        t._preprocess_csm_dataset(None)
    assert seen["source"] == "/cache/models--mallory--voice/snapshots/abc"
    assert seen["revision"] == "abc"
    assert seen["local_files_only"] is True


def test_csm_processor_without_a_recorded_source_uses_the_model_name(monkeypatch):
    seen = {}

    def from_pretrained(source, **kwargs):
        seen.update(source = source, **kwargs)
        raise _Stop

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    t = tmod.UnslothTrainer()
    t.model_name = "org/csm"
    with pytest.raises(_Stop):
        t._preprocess_csm_dataset(None)
    assert seen["source"] == "org/csm" and seen["revision"] is None
