# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regressions for a fresh review pass on the local STT dictation feature:

1. Curated GGUF dictation repos (unslothai/whisper-*-GGUF) must be hidden from
   chat pickers, not just their Transformers safetensors companions.
2. The GGUF sidecar's loaded_model/device status accessors must be lock-free so
   they never block behind an in-flight transcription (which holds self._lock).
3. A "gguf" unload on a host without whisper-server must target the Transformers
   fallback that actually served it, and unload-all must attempt both backends
   even if one raises.
4. free_stt_model_for_training must free the GGUF sidecar even when the
   Transformers unload raises (independent exception boundaries).
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# 1. Hidden-model GGUF companions ------------------------------------------------
def test_curated_gguf_dictation_repos_are_hidden():
    from utils.hidden_models import _HIDDEN_STT_REPO_IDS, is_hidden_model
    for repo in (
        "unslothai/whisper-tiny-GGUF",
        "unslothai/whisper-base-GGUF",
        "unslothai/whisper-small-GGUF",
        "unslothai/whisper-large-v3-turbo-GGUF",
        "unslothai/whisper-large-v3-GGUF",
    ):
        assert repo in _HIDDEN_STT_REPO_IDS
        assert is_hidden_model(repo) is True
        # Case-insensitive, matching how the cache stores the repo id.
        assert is_hidden_model(repo.lower()) is True

    # A same-prefix but genuinely different repo is NOT hidden.
    assert is_hidden_model("unslothai/whisper-large-v3-GGUF-finetune") is False


# 2. GGUF status accessors are lock-free ----------------------------------------
def test_gguf_status_accessors_do_not_block_on_the_inference_lock():
    from core.inference.stt_ggml_sidecar import GgmlSttSidecar

    sidecar = GgmlSttSidecar()

    class _AliveProc:
        pid = 4321

        def poll(self):
            return None  # still running

    sidecar._process = _AliveProc()
    sidecar._model_id = "small"

    holder_has_lock = threading.Event()
    release = threading.Event()

    def _hold_inference_lock():
        # Mimic transcribe() holding self._lock across the whole HTTP call.
        with sidecar._lock:
            holder_has_lock.set()
            release.wait(timeout = 5)

    holder = threading.Thread(target = _hold_inference_lock)
    holder.start()
    assert holder_has_lock.wait(timeout = 5)

    result: dict = {}

    def _read_status():
        result["model"] = sidecar.loaded_model
        result["device"] = sidecar.device

    reader = threading.Thread(target = _read_status)
    reader.start()
    reader.join(timeout = 2)
    blocked = reader.is_alive()

    release.set()
    holder.join(timeout = 5)
    reader.join(timeout = 5)

    assert not blocked, "loaded_model/device blocked on self._lock (should be lock-free)"
    assert result == {"model": "small", "device": "whisper.cpp"}


# 3. Unload resolves through the serving engine + attempts every backend ---------
def test_gguf_unload_targets_transformers_fallback_without_whisper_server(monkeypatch):
    import core.inference.stt_ggml_sidecar as ggml_module
    import routes.inference as ri

    monkeypatch.setattr(ggml_module, "is_available", lambda: False)  # no whisper-server

    calls: list = []

    class _Sidecar:
        def __init__(self, name):
            self.name = name

        def unload(self):
            calls.append(self.name)

    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda name: _Sidecar(name))

    resp = asyncio.run(ri.stt_unload(engine = "gguf", current_subject = "tester"))
    assert resp.status_code == 200
    # gguf is served by the Transformers fallback here, so that is what unloads.
    assert calls == ["transformers"]


def test_unload_all_attempts_both_backends_even_when_one_fails(monkeypatch):
    import routes.inference as ri

    attempted: list = []

    class _Sidecar:
        def __init__(self, name):
            self.name = name

        def unload(self):
            attempted.append(self.name)
            if self.name == "transformers":
                raise RuntimeError("boom")

    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda name: _Sidecar(name))

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(ri.stt_unload(engine = None, current_subject = "tester"))

    assert excinfo.value.status_code == 500
    # gguf is still attempted after the transformers unload raised.
    assert attempted == ["transformers", "gguf"]


# 4. free_stt_model_for_training isolates the two backends -----------------------
def test_free_stt_frees_gguf_even_when_transformers_unload_raises(monkeypatch):
    import routes.training_vram as tv

    class _TransformersSidecar:
        def is_loading(self):
            return False

        @property
        def loaded_model(self):
            return "whisper-small"

        def unload(self):
            raise RuntimeError("transformers unload failed")

    class _GgmlSidecar:
        def __init__(self):
            self.unloaded = False

        def is_loading(self):
            return False

        @property
        def loaded_model(self):
            return None if self.unloaded else "small"

        def unload(self):
            self.unloaded = True

    ggml = _GgmlSidecar()
    monkeypatch.setattr(
        "core.inference.stt_sidecar.get_stt_sidecar", lambda: _TransformersSidecar()
    )
    monkeypatch.setattr("core.inference.stt_ggml_sidecar.get_ggml_stt_sidecar", lambda: ggml)

    freed = tv.free_stt_model_for_training("test")

    # The Transformers failure must not skip GGUF eviction.
    assert ggml.unloaded is True
    assert any("small" in entry for entry in freed)
