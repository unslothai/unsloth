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


# ── Manual mode (--gpu-layers + --fit off + --n-cpu-moe) ─────────────


def test_load_request_accepts_manual():
    req = LoadRequest(
        model_path = "owner/repo",
        gpu_memory_mode = "manual",
        gpu_layers = 20,
        n_cpu_moe = 8,
        tensor_split = [2, 1],
    )
    assert req.gpu_memory_mode == "manual"
    assert req.gpu_layers == 20
    assert req.n_cpu_moe == 8
    assert req.tensor_split == [2, 1]


def test_load_request_manual_defaults():
    req = LoadRequest(model_path = "owner/repo")
    assert req.gpu_layers == -1
    assert req.n_cpu_moe == 0
    assert req.tensor_split is None


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
            n_cpu_moe = 8,
            tensor_split = [2, 1],
            n_layers = 32,
            n_moe_layers = 32,
        )
    else:
        obj = model_cls(
            gpu_memory_mode = "manual", gpu_layers = 20, n_cpu_moe = 8,
            tensor_split = [2, 1], n_layers = 32, n_moe_layers = 32,
        )
    dumped = obj.model_dump()
    assert dumped["gpu_memory_mode"] == "manual"
    assert dumped["gpu_layers"] == 20
    assert dumped["n_cpu_moe"] == 8
    assert dumped["tensor_split"] == [2, 1]
    assert dumped["n_layers"] == 32
    assert dumped["n_moe_layers"] == 32


def test_manual_properties_default_and_reflect_and_reset():
    backend = LlamaCppBackend()
    assert backend.gpu_layers == -1 and backend.n_cpu_moe == 0
    assert backend.tensor_split is None
    backend._gpu_layers = 20
    backend._n_cpu_moe = 8
    backend._tensor_split = [2, 1]
    assert backend.gpu_layers == 20 and backend.n_cpu_moe == 8
    assert backend.tensor_split == [2, 1]
    backend._process = _FakeProcess()
    backend.unload_model()
    assert backend.gpu_layers == -1 and backend.n_cpu_moe == 0
    assert backend.tensor_split is None


def test_n_moe_layers_property():
    # Behavior: 0 for a dense model (hides the slider); block_count for an
    # all-MoE model; block_count - leading_dense for a leading-dense model
    # (GLM-4.7-Flash: deepseek2, block_count 47, leading_dense 1 -> 46).
    b = LlamaCppBackend()
    b._n_layers = 36
    b._n_experts = None
    assert b.n_moe_layers == 0
    b._n_experts = 128
    b._leading_dense_block_count = None
    assert b.n_moe_layers == 36
    b._n_layers = 47
    b._leading_dense_block_count = 1
    assert b.n_moe_layers == 46


def _target_state_manual(backend, *, gpu_layers, n_cpu_moe, tensor_split = None):
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
        n_cpu_moe = n_cpu_moe,
        tensor_split = tensor_split,
    )


def test_manual_reloads_on_gpu_layers_or_n_cpu_moe_or_split_change():
    backend = _loaded_backend("manual")
    backend._gpu_layers = 20
    backend._n_cpu_moe = 0
    backend._tensor_split = None
    # Same knobs -> no reload.
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0) is True
    # Changed layer count -> reload.
    assert _target_state_manual(backend, gpu_layers = 16, n_cpu_moe = 0) is False
    # Changed MoE offload -> reload.
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 8) is False
    # Added a GPU split -> reload.
    assert (
        _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0, tensor_split = [2, 1])
        is False
    )
    # Same GPU split -> no reload.
    backend._tensor_split = [2, 1]
    assert (
        _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0, tensor_split = [2, 1])
        is True
    )


def test_manual_emits_gpu_layers_fit_off_and_n_cpu_moe():
    src = _load_model_source()
    gate = src.find('elif gpu_memory_mode == "manual":')
    assert gate != -1, "load_model must branch on manual mode"
    block = src[gate : gate + 700]
    # Manual empties the probed set (skips the planner) but no longer forces
    # tensor parallelism off -- the toggle is honored (see test below).
    assert "gpus = []" in block
    assert "tensor_parallel = False" not in block
    # The cmd emits an explicit layer count with fit disabled.
    assert 'cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])' in src
    # MoE offload uses --n-cpu-moe via _resolve_cpu_moe_flag (tested behaviorally
    # below); the cmd only emits it when the helper returns a value.
    assert "_resolve_cpu_moe_flag(" in src
    assert 'cmd.extend(["--n-cpu-moe", str(moe_flag)])' in src
    # Manual forces use_fit False so --fit-ctx is never added under --fit off.
    emit = src.find('cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])')
    assert "use_fit = False" in src[src.rfind("\n", 0, emit) - 200 : emit + 80]


def test_manual_emits_tensor_split():
    # Manual emits --tensor-split from the per-GPU shares, only when provided.
    src = _load_model_source()
    assert "if tensor_split:" in src
    assert '"--tensor-split"' in src
    # Joined as a comma list (e.g. "2,1") within the manual branch.
    gate = src.find('elif gpu_memory_mode == "manual":')
    nxt = src.find("elif use_fit:", gate)
    assert '","' in src[gate:nxt] and "tensor_split" in src[gate:nxt]


def test_resolve_cpu_moe_flag():
    # Behavior: clamp the requested MoE-layer count to the model's MoE layers,
    # then offset past leading dense layers (--n-cpu-moe counts from layer 0).
    R = LlamaCppBackend._resolve_cpu_moe_flag
    assert R(0, 40, 0) is None  # nothing requested
    assert R(8, 0, 0) is None  # dense model (no MoE layers)
    assert R(8, 40, 0) == 8  # all-MoE: direct
    assert R(100, 40, 0) == 40  # clamp to the MoE layer count
    # GLM-4.7-Flash (deepseek2): block_count 47, leading_dense 1, n_moe 46.
    assert R(5, 46, 1) == 6  # offset past the 1 dense layer
    assert R(46, 46, 1) == 47  # all MoE on CPU == block_count


def test_manual_allows_tensor_parallel_via_split_mode():
    # Manual keeps the user's TP choice but skips the memory-based planner
    # (plan_tp excludes manual, so its empty gpu set can't downgrade TP). The
    # --split-mode tensor emission gates on tensor_parallel alone, so manual
    # reaches it -- with tp_tensor_split None it's an even split (no
    # --tensor-split). --fit off means no fit/tensor abort.
    src = _load_model_source()
    assert 'plan_tp = tensor_parallel and gpu_memory_mode != "manual"' in src
    assert "if plan_tp:" in src
    assert "if plan_tp and len(tp_gpus) < 2:" in src
    sm = src.find('cmd.extend(["--split-mode", "tensor"])')
    assert sm != -1, "TP must emit --split-mode tensor"
    guard = src.rfind("if tensor_parallel:", 0, sm)
    assert guard != -1 and sm - guard < 200, "split-mode gates on tensor_parallel"
    # The tensor-split is only emitted for a planned (non-even) split, which
    # manual never produces, so manual stays an even split.
    assert "if tp_tensor_split and len(tp_tensor_split) > 1:" in src


def test_fit_auto_floors_fit_ctx_at_8192():
    # Behavior: --fit on with an auto (0) context floors --fit-ctx at 8192.
    caps = {"supports_fit_ctx": True, "supports_kv_unified": True}
    flags = LlamaCppBackend._ctx_integrity_flags(1, True, 0, 0, caps)
    assert flags[flags.index("--fit-ctx") + 1] == "8192"
    # An explicit requested context floors at that value instead.
    explicit = LlamaCppBackend._ctx_integrity_flags(1, True, 16384, 16384, caps)
    assert explicit[explicit.index("--fit-ctx") + 1] == "16384"
    # No --fit-ctx when the binary lacks support.
    assert "--fit-ctx" not in LlamaCppBackend._ctx_integrity_flags(
        1, True, 0, 0, {"supports_fit_ctx": False}
    )


# ── GPU picker (gpu_ids -> CUDA_VISIBLE_DEVICES) ─────────────────────


def test_load_request_accepts_gpu_ids():
    req = LoadRequest(model_path = "owner/repo", gpu_ids = [1, 0])
    assert req.gpu_ids == [1, 0]
    assert LoadRequest(model_path = "owner/repo").gpu_ids is None


@pytest.mark.parametrize("model_cls", [LoadResponse, InferenceStatusResponse])
def test_response_models_emit_gpu_ids(model_cls):
    if model_cls is LoadResponse:
        obj = model_cls(
            status = "loaded", model = "m", display_name = "m", inference = {}, gpu_ids = [1]
        )
    else:
        obj = model_cls(gpu_ids = [1])
    assert obj.model_dump()["gpu_ids"] == [1]


def test_gpu_ids_property_default_and_reset():
    backend = LlamaCppBackend()
    assert backend.gpu_ids is None
    backend._gpu_ids = [0, 1]
    assert backend.gpu_ids == [0, 1]
    backend._process = _FakeProcess()
    backend.unload_model()
    assert backend.gpu_ids is None


def _target_state_gpu_ids(backend, gpu_ids):
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
        gpu_ids = gpu_ids,
    )


def test_gpu_ids_reload_detection_is_order_insensitive():
    backend = _loaded_backend("auto")
    backend._gpu_ids = [0, 1]
    # Same set, different order -> no reload.
    assert _target_state_gpu_ids(backend, [1, 0]) is True
    # Different set -> reload.
    assert _target_state_gpu_ids(backend, [0]) is False
    # Dropping the pick (auto) -> reload.
    assert _target_state_gpu_ids(backend, None) is False


def test_gpu_picker_filters_probe_and_masks():
    src = _load_model_source()
    # Auto selection only considers the picked devices.
    assert "if gpu_ids:" in src
    assert "g for g in gpus if g[0] in _picked" in src
    # fit/manual (gpu_indices None) get pinned to the picked set.
    assert "if gpu_ids and gpu_indices is None:" in src
    assert "gpu_indices = sorted(gpu_ids)" in src
    # Picked indices follow PCI-bus order so the UI index == llama.cpp's.
    assert 'env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"' in src
