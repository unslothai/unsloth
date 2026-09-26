# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training's gate check names a repo ModelScope does not host instead of asking for a token."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Register a stub module for a dep the backend pytest job does not install.

    Same helper and reason as test_trainer_stdout_quiet.py: core.training.trainer imports
    unsloth (and through it unsloth_zoo) and trl at module scope, while the pytest matrix in
    studio-backend-ci.yml installs studio.txt plus torch and transformers and deliberately
    stops there, because the repo-cpu-tests job beside it is the one that installs
    unsloth_zoo, for the REPO-ROOT tests/ tree. Unstubbed, this module fails COLLECTION and
    takes the whole job down. A real install is left alone. __spec__ = None keeps the
    trainer's own _ensure_real_packages namespace-shadow guard a no-op on the stub.
    """
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


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as trainer_module  # noqa: E402
from core.training.trainer import UnslothTrainer  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


def _missing_on_modelscope(repo):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError
    from hub.utils.hf_errors import not_on_modelscope

    return RepositoryNotFoundError(
        not_on_modelscope(repo),
        response = httpx.Response(
            404, request = httpx.Request("GET", "http://127.0.0.1:1/api/models/" + repo)
        ),
    )


@pytest.mark.parametrize(
    "gone, expected", [(True, "not on ModelScope"), (False, "gated or private")]
)
def test_gate_check_reports_a_repo_missing_on_modelscope(monkeypatch, gone, expected):
    import huggingface_hub
    import utils.cache_cleanup
    from huggingface_hub.errors import RepositoryNotFoundError
    import httpx

    def model_info(*_a, **_k):
        if gone:
            raise _missing_on_modelscope("org/model")
        raise RepositoryNotFoundError(
            "Repository Not Found",
            response = httpx.Response(404, request = httpx.Request("GET", "https://huggingface.co")),
        )

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    monkeypatch.setattr(trainer_module, "clear_gpu_cache", lambda *a, **k: None)
    monkeypatch.setattr(utils.cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: None)
    monkeypatch.setattr(trainer_module, "detect_audio_type_checked", lambda *a, **k: (None, True))
    monkeypatch.setattr(trainer_module, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(trainer_module, "_env_offline", lambda: False)
    trainer = object.__new__(UnslothTrainer)
    trainer.model = trainer.tokenizer = trainer.trainer = None
    errors = []
    trainer._update_progress = lambda **kw: errors.append(kw["error"]) if kw.get("error") else None
    trainer._cleanup_audio_artifacts = lambda: None
    assert trainer.load_model("org/model", hf_token = None) is False
    assert expected in errors[-1]
