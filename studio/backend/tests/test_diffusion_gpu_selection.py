# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import types as _types
from pathlib import Path
from unittest import mock

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


def _make_backend() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
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


def test_diffusion_server_pins_requested_gpu_as_visible_ordinal_zero(monkeypatch):
    captured = {}

    class DummyProcess:
        stdout = None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return DummyProcess()

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.Popen", fake_popen)

    backend = _make_backend()
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


def test_child_gpu_pin_clears_rocr_when_hip_is_forced():
    env = {"ROCR_VISIBLE_DEVICES": "1"}

    LlamaCppBackend._pin_child_gpu_env(env, "1", force_hip = True)

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_tensor_parallel_launch_uses_planned_gpu_order_for_tensor_split():
    assert LlamaCppBackend._launch_gpu_indices(
        gpu_ids = [1, 0],
        gpu_indices = [0, 1],
        tensor_parallel = True,
    ) == [0, 1]

    assert LlamaCppBackend._launch_gpu_indices(
        gpu_ids = [1, 0],
        gpu_indices = [0, 1],
        tensor_parallel = False,
    ) == [1, 0]


def test_already_in_target_state_mismatches_changed_gpu_ids():
    backend = _make_backend()
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
