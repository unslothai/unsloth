# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib
import sys
import types as _types
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture()
def llama_cpp_module(monkeypatch):
    module_name = "core.inference.llama_cpp"
    was_loaded = module_name in sys.modules
    backend_dir = str(Path(__file__).resolve().parent.parent)
    monkeypatch.syspath_prepend(backend_dir)

    loggers_stub = _types.ModuleType("loggers")
    loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    monkeypatch.setitem(sys.modules, "loggers", loggers_stub)
    monkeypatch.setitem(sys.modules, "structlog", _types.ModuleType("structlog"))

    module = importlib.import_module(module_name)
    yield module

    if not was_loaded:
        sys.modules.pop(module_name, None)
        parent = sys.modules.get("core.inference")
        if parent is not None and getattr(parent, "llama_cpp", None) is module:
            delattr(parent, "llama_cpp")


@pytest.fixture()
def backend_cls(llama_cpp_module):
    return llama_cpp_module.LlamaCppBackend


def _make_backend(backend_cls):
    backend = backend_cls.__new__(backend_cls)
    backend._context_length = 4096
    backend._stdout_lines = []
    backend._kill_process = mock.Mock()
    backend._find_free_port = mock.Mock(return_value = 12345)
    backend._find_diffusion_assets = mock.Mock(
        return_value = (["python", "-m", "unsloth_zoo.diffusion_studio.shim"], "/tmp/visual", None)
    )
    backend._wait_for_health = mock.Mock(return_value = True)
    backend._drain_stdout = mock.Mock()
    return backend


def test_diffusion_server_pins_requested_gpu_as_visible_ordinal_zero(
    monkeypatch, llama_cpp_module, backend_cls
):
    captured = {}

    class DummyProcess:
        stdout = None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return DummyProcess()

    monkeypatch.setattr(llama_cpp_module.subprocess, "Popen", fake_popen)

    backend = _make_backend(backend_cls)
    assert backend._start_diffusion_server(
        model_path = "/models/diffusiongemma.gguf",
        gguf_path = "/models/diffusiongemma.gguf",
        hf_repo = None,
        hf_variant = None,
        model_identifier = "local/diffusiongemma",
        n_ctx = 0,
        extra_args = None,
        gpu_ids = [1],
    )

    assert captured["cmd"][captured["cmd"].index("--gpu") + 1] == "0"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["env"]["HIP_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in captured["env"]
    assert captured["env"]["DG_GPU"] == "0"
    assert backend.tensor_parallel is False


def test_diffusion_server_rejects_multi_gpu_pins(backend_cls):
    backend = _make_backend(backend_cls)

    with pytest.raises(ValueError, match = "support one gpu_id"):
        backend._start_diffusion_server(
            model_path = "/models/diffusiongemma.gguf",
            gguf_path = "/models/diffusiongemma.gguf",
            hf_repo = None,
            hf_variant = None,
            model_identifier = "local/diffusiongemma",
            n_ctx = 0,
            extra_args = None,
            gpu_ids = [0, 1],
        )

    backend._find_diffusion_assets.assert_not_called()


def test_child_gpu_pin_clears_rocr_when_hip_is_forced(backend_cls):
    env = {"ROCR_VISIBLE_DEVICES": "1"}

    backend_cls._pin_child_gpu_env(env, "1", force_hip = True)

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_requested_gpu_filter_rejects_partial_visibility(backend_cls):
    assert backend_cls._filter_requested_gpus([(0, 1024), (1, 2048)], [1]) == [(1, 2048)]

    with pytest.raises(ValueError, match = r"Requested GPU IDs \[2\]"):
        backend_cls._filter_requested_gpus([(0, 1024), (1, 2048)], [1, 2])


def test_tensor_parallel_launch_uses_planned_gpu_order_for_tensor_split(backend_cls):
    assert backend_cls._launch_gpu_indices(
        gpu_ids = [1, 0],
        gpu_indices = [0, 1],
        tensor_parallel = True,
    ) == [0, 1]

    assert backend_cls._launch_gpu_indices(
        gpu_ids = [1, 0],
        gpu_indices = [0, 1],
        tensor_parallel = False,
    ) == [1, 0]


def test_already_in_target_state_mismatches_changed_gpu_ids(backend_cls):
    backend = _make_backend(backend_cls)
    backend._process = object()
    backend._healthy = True
    backend._model_identifier = "local/model"
    backend._gguf_path = "/models/model.gguf"
    backend._hf_variant = None
    backend._requested_n_ctx = 0
    backend._gpu_ids = [0]
    backend._cache_type_kv = None
    backend._tensor_parallel = False
    backend._requested_spec_mode = "auto"
    backend._speculative_type = None
    backend._chat_template_override = None
    backend._mtp_draft_path = None

    assert not backend._already_in_target_state(
        model_identifier = "local/model",
        gguf_path = "/models/model.gguf",
        hf_variant = None,
        n_ctx = 0,
        cache_type_kv = None,
        speculative_type = None,
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gpu_ids = [1],
    )
