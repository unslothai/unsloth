# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dictation lifecycle goes through one place, shared with the orchestrator."""

import pytest

from core.inference import stt_registry


class _Sidecar:
    def __init__(self, name, model = None, loading = False, fail = False):
        self.name = name
        self.loaded_model = model
        self.device = f"{name}-device" if model else None
        self._loading = loading
        self._fail = fail
        self.unloaded = False
        self.loaded_with = None

    def is_loading(self):
        return self._loading

    def load(self, model):
        self.loaded_with = model

    def unload(self):
        if self._fail:
            raise RuntimeError("boom")
        self.unloaded = True


def test_unload_attempts_every_engine_even_after_one_raises(monkeypatch):
    made = {}

    def make(name):
        made.setdefault(name, _Sidecar(name, fail = name == "transformers"))
        return made[name]

    monkeypatch.setattr(stt_registry, "sidecar_for", make)
    failed = stt_registry.unload()

    assert list(made) == list(stt_registry.STT_ENGINES)
    assert failed == ["transformers"]
    # The engines after the failure still released.
    assert made["gguf"].unloaded and made["mtmd"].unloaded


def test_load_delegates_to_the_engines_sidecar(monkeypatch):
    sidecar = _Sidecar("mtmd")
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecar)

    stt_registry.load("qwen3-asr-0.6b", "mtmd")
    assert sidecar.loaded_with == "qwen3-asr-0.6b"


def test_resident_reports_whichever_engine_holds_a_model(monkeypatch):
    sidecars = {
        "transformers": _Sidecar("transformers"),
        "gguf": _Sidecar("gguf"),
        "mtmd": _Sidecar("mtmd", model = "qwen3-asr-0.6b"),
    }
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    assert stt_registry.resident() == {
        "model": "qwen3-asr-0.6b",
        "engine": "mtmd",
        "device": "mtmd-device",
        "loading": False,
    }


def test_resident_reports_an_in_flight_load(monkeypatch):
    sidecars = {
        "transformers": _Sidecar("transformers"),
        "gguf": _Sidecar("gguf", loading = True),
        "mtmd": _Sidecar("mtmd"),
    }
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: sidecars[name])

    resident = stt_registry.resident()
    assert resident["loading"] is True and resident["engine"] == "gguf"


def test_resident_reports_nothing_when_every_engine_is_idle(monkeypatch):
    monkeypatch.setattr(stt_registry, "sidecar_for", lambda name: _Sidecar(name))
    assert stt_registry.resident()["model"] is None


def test_an_unimportable_engine_never_takes_the_status_down(monkeypatch):
    def make(name):
        if name == "gguf":
            raise ImportError("whisper.cpp bindings missing")
        return _Sidecar(name, model = "small" if name == "mtmd" else None)

    monkeypatch.setattr(stt_registry, "sidecar_for", make)
    # gguf raising must not hide the model mtmd is holding.
    assert stt_registry.resident()["model"] == "small"


def test_the_orchestrator_exposes_the_same_lifecycle(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator

    for name in ("load_stt_model", "unload_stt_model", "resident_stt_model"):
        assert callable(getattr(InferenceOrchestrator, name, None)), name


def test_the_route_loads_through_the_orchestrator_when_one_exists(monkeypatch):
    import routes.inference as ri
    from core.inference import orchestrator as orch

    class _Backend:
        load_stt_model = "orchestrator-load"
        unload_stt_model = "orchestrator-unload"

    monkeypatch.setattr(orch, "peek_inference_backend", lambda: _Backend())
    assert ri._stt_lifecycle() == ("orchestrator-load", "orchestrator-unload")


def test_a_cold_process_loads_without_building_an_orchestrator(monkeypatch):
    import routes.inference as ri
    from core.inference import orchestrator as orch

    def _never():
        raise AssertionError("dictation must not construct the chat orchestrator")

    monkeypatch.setattr(orch, "peek_inference_backend", lambda: None)
    monkeypatch.setattr(orch, "get_inference_backend", _never)
    # Same functions the orchestrator's methods forward to, so neither path
    # can drift from the other.
    assert ri._stt_lifecycle() == (stt_registry.load, stt_registry.unload)
