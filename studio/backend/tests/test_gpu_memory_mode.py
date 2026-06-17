# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the GPU Memory mode dropdown.

The dropdown threads a single ``gpu_memory_mode`` ("auto" | "fit") from the
chat UI through the load request. "fit" hands all memory management to
llama.cpp's ``--fit on``: no CUDA/HIP device masking, no context auto-reduce,
no gpu-layer or tensor-split planning. These tests pin:

  * the pydantic request/response/status contract (snake_case key,
    default "auto", invalid values rejected),
  * the backend ``gpu_memory_mode`` property and its reset on unload,
  * the ``_already_in_target_state`` reload-detection branch, and
  * that the fit branch in ``load_model`` empties the probed GPU set and
    drops tensor parallelism so the selection below no-ops.
"""

from __future__ import annotations

import inspect
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Same external-dep stubs as the other llama_cpp unit tests so importing
# the backend doesn't drag in structlog / httpx / loggers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {
        "__init__": lambda s, **kw: None,
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend
from models.inference import (
    InferenceStatusResponse,
    LoadRequest,
    LoadResponse,
)


# ── Pydantic contract (snake_case key, default "auto") ───────────────


def test_load_request_defaults_gpu_memory_mode_auto():
    assert LoadRequest(model_path = "owner/repo").gpu_memory_mode == "auto"


def test_load_request_accepts_fit():
    req = LoadRequest(model_path = "owner/repo", gpu_memory_mode = "fit")
    assert req.gpu_memory_mode == "fit"


def test_load_request_round_trips_json_key():
    req = LoadRequest.model_validate({"model_path": "owner/repo", "gpu_memory_mode": "fit"})
    assert req.gpu_memory_mode == "fit"
    assert req.model_dump()["gpu_memory_mode"] == "fit"


def test_load_request_rejects_unknown_mode():
    with pytest.raises(ValueError):
        LoadRequest(model_path = "owner/repo", gpu_memory_mode = "bogus")


@pytest.mark.parametrize("model_cls", [LoadResponse, InferenceStatusResponse])
def test_response_models_emit_gpu_memory_mode(model_cls):
    if model_cls is LoadResponse:
        default = model_cls(
            status = "loaded",
            model = "owner/repo",
            display_name = "repo",
            inference = {},
        )
        fit = model_cls(
            status = "loaded",
            model = "owner/repo",
            display_name = "repo",
            inference = {},
            gpu_memory_mode = "fit",
        )
    else:
        default = model_cls()
        fit = model_cls(gpu_memory_mode = "fit")
    assert default.model_dump()["gpu_memory_mode"] == "auto"
    assert fit.model_dump()["gpu_memory_mode"] == "fit"


# ── Backend property + reset ─────────────────────────────────────────


class _FakeProcess:
    """Stand-in for subprocess.Popen so _kill_process is a no-op."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def test_gpu_memory_mode_property_defaults_auto():
    assert LlamaCppBackend().gpu_memory_mode == "auto"


def test_gpu_memory_mode_property_reflects_field():
    backend = LlamaCppBackend()
    backend._gpu_memory_mode = "fit"
    assert backend.gpu_memory_mode == "fit"


def test_unload_resets_gpu_memory_mode():
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._gpu_memory_mode = "fit"
    backend.unload_model()
    assert backend.gpu_memory_mode == "auto"


# ── _already_in_target_state reload-detection branch ─────────────────


def _loaded_backend(gpu_memory_mode: str) -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()  # is_loaded only checks "is not None"
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._gguf_path = None
    backend._gpu_memory_mode = gpu_memory_mode
    return backend


def _target_state(backend: LlamaCppBackend, gpu_memory_mode: str) -> bool:
    return backend._already_in_target_state(
        gguf_path = None,
        model_identifier = "owner/repo",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gpu_memory_mode = gpu_memory_mode,
    )


@pytest.mark.parametrize("mode", ["auto", "fit"])
def test_already_in_target_state_matches_same_mode(mode):
    assert _target_state(_loaded_backend(mode), mode) is True


@pytest.mark.parametrize("loaded,requested", [("auto", "fit"), ("fit", "auto")])
def test_already_in_target_state_reloads_on_mode_change(loaded, requested):
    # Flipping the dropdown either direction must force a reload so the
    # command is rebuilt with/without --fit on (and the GPU masking).
    assert _target_state(_loaded_backend(loaded), requested) is False


# ── load_model fit branch bypasses Unsloth GPU management ────────────


def _load_model_source() -> str:
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def test_fit_mode_empties_gpus_and_drops_tensor_parallel():
    src = _load_model_source()
    gate = src.find('if gpu_memory_mode == "fit":')
    assert gate != -1, "load_model must branch on gpu_memory_mode == 'fit'"
    # Emptying the probed set makes the selection / TP planning below no-op:
    # gpu_indices stays None (no masking), use_fit True (--fit on), and
    # tp_tensor_split None (no --tensor-split).
    block = src[gate : gate + 800]
    assert "gpus = []" in block, "fit branch must empty the probed GPU set"
    assert "tensor_parallel = False" in block, "fit branch must drop tensor parallelism"
    # --fit aborts under --split-mode tensor, so a raw-extras split-mode is stripped.
    assert "strip_split_mode_only(extra_args)" in block
    # Auto (0) lets --fit size context; an explicit context is honored so --fit
    # optimizes gpu-layers around it.
    assert "requested_ctx if requested_ctx > 0 else 0" in block
    # The fit branch sits before GPU selection assigns gpu_indices.
    assert gate < src.find("gpu_indices, use_fit = None, True")
    # --fit on is the use_fit emission the fit branch relies on.
    assert 'cmd.extend(["--fit", "on"])' in src


def test_fit_mode_never_sends_ctx_size_zero():
    # Sending "-c 0" sets fit_params_min_ctx = UINT32_MAX in llama.cpp, pinning
    # the full native context and disabling --fit's reduction. So the base cmd
    # must never carry -c, "-c 0" is emitted only outside fit mode, and a
    # positive context is passed through (which --fit optimizes layers around).
    src = _load_model_source()
    base_start = src.find("cmd = [")
    base_end = src.find("\n                ]", base_start)
    base_block = src[base_start:base_end]
    assert '"-c"' not in base_block, "-c must be conditional, not in the base cmd list"
    assert 'cmd.extend(["-c", str(effective_ctx)])' in src, "positive ctx must pass -c"
    zero = src.find('cmd.extend(["-c", "0"])')
    assert zero != -1, '"-c 0" emission must exist for non-fit mode'
    guard = src.rfind('elif gpu_memory_mode != "fit":', 0, zero)
    assert guard != -1 and zero - guard < 120, '"-c 0" must sit under non-fit guard'


# ── Manual mode (--gpu-layers + --fit off + --cpu-moe) ───────────────


def test_load_request_accepts_manual():
    req = LoadRequest(
        model_path = "owner/repo",
        gpu_memory_mode = "manual",
        gpu_layers = 20,
        cpu_moe = True,
    )
    assert req.gpu_memory_mode == "manual"
    assert req.gpu_layers == 20
    assert req.cpu_moe is True


def test_load_request_manual_defaults():
    req = LoadRequest(model_path = "owner/repo")
    assert req.gpu_layers == -1
    assert req.cpu_moe is False


@pytest.mark.parametrize("model_cls", [LoadResponse, InferenceStatusResponse])
def test_response_models_emit_manual_fields(model_cls):
    if model_cls is LoadResponse:
        obj = model_cls(
            status = "loaded",
            model = "owner/repo",
            display_name = "repo",
            inference = {},
            gpu_memory_mode = "manual",
            gpu_layers = 20,
            cpu_moe = True,
            n_layers = 32,
        )
    else:
        obj = model_cls(
            gpu_memory_mode = "manual", gpu_layers = 20, cpu_moe = True, n_layers = 32
        )
    dumped = obj.model_dump()
    assert dumped["gpu_memory_mode"] == "manual"
    assert dumped["gpu_layers"] == 20
    assert dumped["cpu_moe"] is True
    assert dumped["n_layers"] == 32


def test_manual_properties_default_and_reflect_and_reset():
    backend = LlamaCppBackend()
    assert backend.gpu_layers == -1 and backend.cpu_moe is False
    backend._gpu_layers = 20
    backend._cpu_moe = True
    assert backend.gpu_layers == 20 and backend.cpu_moe is True
    backend._process = _FakeProcess()
    backend.unload_model()
    assert backend.gpu_layers == -1 and backend.cpu_moe is False


def _target_state_manual(backend, *, gpu_layers, cpu_moe):
    return backend._already_in_target_state(
        gguf_path = None,
        model_identifier = "owner/repo",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gpu_memory_mode = "manual",
        gpu_layers = gpu_layers,
        cpu_moe = cpu_moe,
    )


def test_manual_reloads_on_gpu_layers_or_cpu_moe_change():
    backend = _loaded_backend("manual")
    backend._gpu_layers = 20
    backend._cpu_moe = False
    # Same knobs -> no reload.
    assert _target_state_manual(backend, gpu_layers = 20, cpu_moe = False) is True
    # Changed layer count -> reload.
    assert _target_state_manual(backend, gpu_layers = 16, cpu_moe = False) is False
    # Toggled MoE offload -> reload.
    assert _target_state_manual(backend, gpu_layers = 20, cpu_moe = True) is False


def test_manual_emits_gpu_layers_fit_off_and_cpu_moe():
    src = _load_model_source()
    gate = src.find('elif gpu_memory_mode == "manual":')
    assert gate != -1, "load_model must branch on manual mode"
    block = src[gate : gate + 700]
    assert "gpus = []" in block and "tensor_parallel = False" in block
    # The cmd emits an explicit layer count with fit disabled, plus cpu-moe.
    assert 'cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])' in src
    assert 'cmd.append("--cpu-moe")' in src
    # Manual forces use_fit False so --fit-ctx is never added under --fit off.
    emit = src.find('cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])')
    assert "use_fit = False" in src[src.rfind("\n", 0, emit) - 200 : emit + 80]


def test_fit_auto_floors_fit_ctx_at_8192():
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend._ctx_integrity_flags)
    assert '"--fit-ctx", "8192"' in src, "fit-auto must floor --fit-ctx at 8192"
