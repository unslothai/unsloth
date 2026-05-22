# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# If httpx is installed, keep the real module in sys.modules before later
# lightweight llama.cpp tests get a chance to install a partial stub.
try:  # pragma: no cover - only matters in dependency-light environments
    import httpx as _httpx  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from models.inference import LoadRequest
from utils.models.model_config import (
    ModelConfig,
    detect_audio_type,
    is_vision_model,
    load_model_config,
)


def test_model_config_direct_local_gguf_does_not_probe_transformers(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    monkeypatch.setattr(
        "utils.models.model_config._is_vision_model_uncached",
        Mock(side_effect = AssertionError("vision probe should not run")),
    )
    monkeypatch.setattr(
        "utils.models.model_config._detect_audio_from_tokenizer",
        Mock(side_effect = AssertionError("audio probe should not run")),
    )

    cfg = ModelConfig.from_identifier(str(gguf))

    assert cfg is not None
    assert cfg.is_gguf is True
    assert cfg.gguf_file == str(gguf.absolute())
    assert cfg.is_lora is False
    assert cfg.is_vision is False


def test_missing_local_gguf_returns_none_without_autoconfig(monkeypatch, tmp_path):
    missing = tmp_path / "missing.gguf"

    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        Mock(side_effect = AssertionError("AutoConfig should not run")),
    )

    assert ModelConfig.from_identifier(str(missing)) is None


def test_local_gguf_helpers_short_circuit_model_probes(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    load_probe = Mock(side_effect = AssertionError("load_model_config should not run"))
    audio_probe = Mock(side_effect = AssertionError("tokenizer probe should not run"))
    monkeypatch.setattr("utils.models.model_config.load_model_config", load_probe)
    monkeypatch.setattr(
        "utils.models.model_config._detect_audio_from_tokenizer", audio_probe
    )

    assert is_vision_model(str(gguf)) is False
    assert detect_audio_type(str(gguf)) is None
    load_probe.assert_not_called()
    audio_probe.assert_not_called()


def test_load_model_config_rejects_local_gguf(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    with pytest.raises(ValueError, match = "llama.cpp backend"):
        load_model_config(str(gguf))


class _StubLlamaBackend:
    def __init__(self):
        self.calls = []
        self.is_loaded = False
        self.model_identifier = None
        self.hf_variant = None
        self._gguf_path = None
        self._is_vision = False
        self._is_audio = False
        self._audio_type = None
        self._native_display_label = None
        self._native_grant_backed = False
        self.extra_args = []
        self.extra_args_source = None
        self.chat_template_override = None
        self.requested_spec_mode = "auto"
        self.spec_draft_n_max = None
        self.supports_reasoning = False
        self.reasoning_style = "enable_thinking"
        self.reasoning_always_on = False
        self.supports_preserve_thinking = False
        self.supports_tools = False
        self.chat_template = None
        self.context_length = None
        self.max_context_length = None
        self.native_context_length = None
        self.requested_n_ctx = None
        self._cache_type_kv = None

    @property
    def cache_type_kv(self):
        return self._cache_type_kv

    @property
    def is_vision(self):
        return self._is_vision

    def load_model(self, **kwargs):
        self.calls.append(kwargs)
        self.is_loaded = True
        self.model_identifier = kwargs["model_identifier"]
        self._gguf_path = kwargs.get("gguf_path")
        self._is_vision = bool(kwargs.get("is_vision"))
        self.hf_variant = kwargs.get("hf_variant")
        self.requested_n_ctx = kwargs.get("n_ctx")
        self.context_length = kwargs.get("n_ctx")
        self.max_context_length = 262144
        self.native_context_length = 262144
        self._cache_type_kv = kwargs.get("cache_type_kv")
        self.requested_spec_mode = kwargs.get("speculative_type") or "auto"
        self.spec_draft_n_max = kwargs.get("spec_draft_n_max")
        self.chat_template_override = kwargs.get("chat_template_override")
        self.extra_args = list(kwargs.get("extra_args") or [])
        self.extra_args_source = (self.model_identifier, self.hf_variant)
        return True

    def detect_audio_type(self):
        return None

    def unload_model(self):
        self.is_loaded = False


class _StubUnslothBackend:
    active_model_name = None

    def __init__(self):
        self.calls = []
        self.models = {}

    def load_model(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("standard backend should not load GGUF")

    def unload_model(self, *_args, **_kwargs):
        self.active_model_name = None


def _request_obj():
    return SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1)))


def test_load_route_direct_local_gguf_uses_llama_backend(monkeypatch, tmp_path):
    import routes.inference as route

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    llama = _StubLlamaBackend()
    unsloth = _StubUnslothBackend()

    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(route, "get_inference_backend", lambda: unsloth)
    monkeypatch.setattr(route, "load_inference_config", lambda _model: {})

    resp = asyncio.run(
        route.load_model(
            LoadRequest(model_path = str(gguf), max_seq_length = 71168),
            _request_obj(),
            current_subject = "test",
        )
    )

    assert resp.status == "loaded"
    assert resp.is_gguf is True
    assert len(llama.calls) == 1
    assert llama.calls[0]["gguf_path"] == str(gguf.absolute())
    assert llama.calls[0]["n_ctx"] == 71168
    assert unsloth.calls == []


def test_load_route_direct_local_gguf_reloads_for_context_and_kv(monkeypatch, tmp_path):
    import routes.inference as route

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    llama = _StubLlamaBackend()
    unsloth = _StubUnslothBackend()

    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(route, "get_inference_backend", lambda: unsloth)
    monkeypatch.setattr(route, "load_inference_config", lambda _model: {})

    asyncio.run(
        route.load_model(
            LoadRequest(model_path = str(gguf), max_seq_length = 71168),
            _request_obj(),
            current_subject = "test",
        )
    )
    resp_ctx = asyncio.run(
        route.load_model(
            LoadRequest(model_path = str(gguf), max_seq_length = 75000),
            _request_obj(),
            current_subject = "test",
        )
    )
    resp_q8 = asyncio.run(
        route.load_model(
            LoadRequest(
                model_path = str(gguf),
                max_seq_length = 75000,
                cache_type_kv = "q8_0",
            ),
            _request_obj(),
            current_subject = "test",
        )
    )

    assert resp_ctx.status == "loaded"
    assert resp_q8.status == "loaded"
    assert [call["n_ctx"] for call in llama.calls] == [71168, 75000, 75000]
    assert llama.calls[-1]["cache_type_kv"] == "q8_0"
    assert unsloth.calls == []


def test_orchestrator_rejects_gguf_before_spawning(caplog, tmp_path):
    from core.inference.orchestrator import InferenceOrchestrator

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    config = SimpleNamespace(identifier = str(gguf), is_gguf = True)

    with pytest.raises(ValueError, match = "llama.cpp backend"):
        orchestrator.load_model(config)

    assert "Spawning fresh inference subprocess" not in caplog.text


def test_inference_backend_rejects_gguf_before_fast_model(monkeypatch, tmp_path):
    from core.inference.inference import FastLanguageModel, InferenceBackend

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(
        FastLanguageModel,
        "from_pretrained",
        Mock(side_effect = AssertionError("FastLanguageModel should not run")),
    )
    backend = InferenceBackend.__new__(InferenceBackend)
    config = SimpleNamespace(identifier = str(gguf), is_gguf = True)

    with pytest.raises(ValueError, match = "llama.cpp backend"):
        backend.load_model(config)

    FastLanguageModel.from_pretrained.assert_not_called()


def test_worker_build_model_config_rejects_gguf(tmp_path):
    from core.inference.worker import _build_model_config

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    with pytest.raises(ValueError, match = "llama.cpp backend"):
        _build_model_config({"model_name": str(gguf)})


class _FakeProc:
    def __init__(self):
        self.pid = 12345
        self.stdout = []

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass


def _capture_llama_command(monkeypatch, tmp_path, *, cache_type_kv = None, free_mib = 200000):
    import core.inference.llama_cpp as llama_cpp
    from core.inference.llama_cpp import LlamaCppBackend

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    captured = {}

    backend = LlamaCppBackend()
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        classmethod(lambda cls: "/fake/llama-server"),
    )
    monkeypatch.setattr(backend, "_kill_process", lambda *a, **k: None)
    monkeypatch.setattr(
        backend,
        "_read_gguf_metadata",
        lambda _path: setattr(backend, "_context_length", 262144),
    )
    monkeypatch.setattr(backend, "_get_gguf_size_bytes", lambda _path: 16 * 1024**3)
    monkeypatch.setattr(backend, "_get_gpu_free_memory", lambda: [(0, free_mib)])
    monkeypatch.setattr(backend, "_can_estimate_kv", lambda: False)
    monkeypatch.setattr(backend, "_wait_for_health", lambda *a, **k: True)
    monkeypatch.setattr(backend, "_classify_gpu_offload", lambda *a, **k: True)
    monkeypatch.setattr(
        backend,
        "_build_speculative_flags",
        lambda **kwargs: [],
    )

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr(llama_cpp.subprocess, "Popen", fake_popen)

    assert backend.load_model(
        gguf_path = str(gguf),
        model_identifier = str(gguf),
        n_ctx = 75000,
        cache_type_kv = cache_type_kv,
        speculative_type = "off",
    )
    return captured["cmd"]


def test_llama_command_q8_cache_flags(monkeypatch, tmp_path):
    cmd = _capture_llama_command(monkeypatch, tmp_path, cache_type_kv = "q8_0")

    assert "--cache-type-k" in cmd
    assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
    assert "--cache-type-v" in cmd
    assert cmd[cmd.index("--cache-type-v") + 1] == "q8_0"
    assert "-c" in cmd
    assert cmd[cmd.index("-c") + 1] == "75000"


def test_llama_command_low_memory_stays_llama_with_fit(monkeypatch, tmp_path):
    cmd = _capture_llama_command(monkeypatch, tmp_path, free_mib = 1024)

    assert cmd[0] == "/fake/llama-server"
    assert "--fit" in cmd
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert "-c" in cmd
    assert cmd[cmd.index("-c") + 1] == "75000"
