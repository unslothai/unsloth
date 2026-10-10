# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the GPU Memory mode dropdown.

The dropdown threads a single ``gpu_memory_mode`` ("auto" | "manual") from the
chat UI through the load request. "manual" lets the user own the offload: with
``gpu_layers < 0`` (Auto, the default) it hands all memory management to
llama.cpp's ``--fit on`` (no CUDA/HIP device masking, no context auto-reduce, no
gpu-layer or tensor-split planning); with ``gpu_layers >= 0`` it pins the layers
and MoE offload itself (``--fit off``). These tests pin:

  * the pydantic request/response/status contract (snake_case key, default
    "auto", unknown values rejected),
  * the backend ``gpu_memory_mode`` property and its reset on unload,
  * the ``_already_in_target_state`` reload-detection branch, and
  * that the manual + Auto-layers branch in ``load_model`` empties the probed
    GPU set and drops tensor parallelism so the selection below no-ops, while
    the explicit-offload branch emits ``--gpu-layers`` / ``--fit off``.
"""

from __future__ import annotations

import inspect
import struct
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

# Real httpx: a partial stub installed first would poison a combined pytest run.
import httpx  # noqa: F401

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
from models.inference import (
    InferenceStatusResponse,
    LoadRequest,
    LoadResponse,
)


def test_load_request_defaults_gpu_memory_mode_auto():
    assert LoadRequest(model_path = "owner/repo").gpu_memory_mode == "auto"


def test_load_request_round_trips_json_key():
    req = LoadRequest.model_validate({"model_path": "owner/repo", "gpu_memory_mode": "manual"})
    assert req.gpu_memory_mode == "manual"
    assert req.model_dump()["gpu_memory_mode"] == "manual"


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
        manual = model_cls(
            status = "loaded",
            model = "owner/repo",
            display_name = "repo",
            inference = {},
            gpu_memory_mode = "manual",
        )
    else:
        default = model_cls()
        manual = model_cls(gpu_memory_mode = "manual")
    assert default.model_dump()["gpu_memory_mode"] == "auto"
    assert manual.model_dump()["gpu_memory_mode"] == "manual"


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
    backend._gpu_memory_mode = "manual"
    assert backend.gpu_memory_mode == "manual"


def test_unload_resets_gpu_memory_mode():
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._gpu_memory_mode = "manual"
    backend.unload_model()
    assert backend.gpu_memory_mode == "auto"


def _loaded_backend(gpu_memory_mode: str) -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()  # is_loaded only checks is not None
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
    return backend.adopt_load_intent_if_matched(
        GgufLoadIntent(
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
    )


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_already_in_target_state_matches_same_mode(mode):
    assert _target_state(_loaded_backend(mode), mode) is True


@pytest.mark.parametrize("loaded,requested", [("auto", "manual"), ("manual", "auto")])
def test_already_in_target_state_reloads_on_mode_change(loaded, requested):
    assert _target_state(_loaded_backend(loaded), requested) is False


def test_already_in_target_state_ignores_mode_for_diffusion(monkeypatch):
    backend = _loaded_backend("auto")
    backend._is_diffusion = True
    monkeypatch.setenv("LLAMA_ARG_SWA_FULL", "1")
    assert _target_state(backend, "manual") is True


def _load_model_source() -> str:
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def test_auto_layers_branch_empties_gpus_and_drops_tensor_parallel():
    src = _load_model_source()
    gate = src.find('if gpu_memory_mode == "manual" and gpu_layers < 0:')
    assert gate != -1, "load_model must branch on manual + Auto layers (gpu_layers < 0)"
    block = src[gate : gate + 1400]
    assert "gpus = []" in block, "Auto-layers branch must empty the probed GPU set"
    assert "strip_split_mode_only(extra_args)" in block
    assert "requested_ctx if requested_ctx > 0 else 0" in block
    assert gate < src.find("gpu_indices, use_fit = None, True")
    assert 'cmd.extend(["--fit", "on"])' in src
    tp_drop = src.find('if tensor_parallel and gpu_memory_mode == "manual" and gpu_layers < 0:')
    assert tp_drop != -1, "manual + Auto layers must drop tensor_parallel"
    assert "tensor_parallel = False" in src[tp_drop : tp_drop + 400]


def test_auto_layers_never_sends_ctx_size_zero():
    # -c 0 sets fit_params_min_ctx = UINT32_MAX, pinning native ctx and disabling --fit.
    src = _load_model_source()
    base_start = src.find("cmd = [")
    base_end = src.find("\n                ]", base_start)
    base_block = src[base_start:base_end]
    assert '"-c"' not in base_block, "-c must be conditional, not in the base cmd list"
    assert 'cmd.extend(["-c", str(effective_ctx)])' in src, "positive ctx must pass -c"
    assert 'auto_fit = gpu_memory_mode == "manual" and gpu_layers < 0' in src
    zero = src.find('cmd.extend(["-c", "0"])')
    assert zero != -1, '"-c 0" emission must exist outside the Auto-layers case'
    guard = src.rfind("elif not auto_fit:", 0, zero)
    assert guard != -1 and zero - guard < 120, '"-c 0" must sit under the not-auto_fit guard'


def test_manual_mode_clears_inherited_main_model_placement_env():
    env = {name: "inherited" for name in LlamaCppBackend._MANUAL_PLACEMENT_ENV_VARS}
    env["LLAMA_ARG_N_GPU_LAYERS_DRAFT"] = "7"
    env["UNRELATED"] = "kept"

    LlamaCppBackend._clear_manual_placement_env(env)

    assert not (set(env) & set(LlamaCppBackend._MANUAL_PLACEMENT_ENV_VARS))
    assert env["LLAMA_ARG_N_GPU_LAYERS_DRAFT"] == "7"
    assert env["UNRELATED"] == "kept"


def test_load_model_sanitizes_manual_env_after_building_child_env():
    src = _load_model_source()
    env_build = src.find("env = self._llama_server_env_for_binary(binary)")
    env_clear = src.find("self._clear_manual_placement_env(env)", env_build)
    launch = src.find("subprocess.Popen", env_build)
    assert env_build != -1
    assert env_build < env_clear < launch


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


@pytest.mark.parametrize("bad", [[0, 0], [-1, 2], [float("inf"), 1], [float("nan"), 1]])
def test_load_request_rejects_degenerate_tensor_split(bad):
    # Bad splits are dropped at launch but compared raw in dedupe, so they would reload forever.
    with pytest.raises(ValueError):
        LoadRequest(model_path = "owner/repo", tensor_split = bad)


@pytest.mark.parametrize("good", [[2, 1], [1, 1], [], None])
def test_load_request_accepts_valid_tensor_split(good):
    assert LoadRequest(model_path = "owner/repo", tensor_split = good).tensor_split == good


def test_route_normalizes_explicit_extras_before_reload_dedupe():
    route_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    load_impl = route_src[route_src.index("async def _load_model_impl") :]
    preserve = load_impl.index("_gpu_layers_override = parse_gpu_layers_override")
    translate = load_impl.index('_manual_updates["gpu_layers"] = _gpu_layers_override')
    preserve_ts = load_impl.index("_tensor_split_override = parse_tensor_split_override")
    translate_ts = load_impl.index('_manual_updates["tensor_split"] = _tensor_split_override')
    strip = load_impl.index("_stripped_explicit = strip_shadowing_flags")
    normalize = load_impl.index(
        'request = request.model_copy(update = {"llama_extra_args": extra_llama_args})'
    )
    dedupe = load_impl.index("_reuse_loaded_gguf(")
    assert preserve < translate < preserve_ts < translate_ts < strip < normalize < dedupe


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
            gpu_memory_mode = "manual",
            gpu_layers = 20,
            n_cpu_moe = 8,
            tensor_split = [2, 1],
            n_layers = 32,
            n_moe_layers = 32,
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
    # 0 for dense; block_count for all-MoE; else block_count - leading_dense (47 - 1 -> 46).
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


def _target_state_manual(
    backend,
    *,
    gpu_layers,
    n_cpu_moe,
    tensor_split = None,
):
    return backend.adopt_load_intent_if_matched(
        GgufLoadIntent(
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
    )


def test_manual_reloads_on_gpu_layers_or_n_cpu_moe_or_split_change():
    backend = _loaded_backend("manual")
    backend._gpu_layers = 20
    backend._n_cpu_moe = 0
    backend._tensor_split = None
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0) is True
    assert _target_state_manual(backend, gpu_layers = 16, n_cpu_moe = 0) is False
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 8) is False
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0, tensor_split = [2, 1]) is False
    backend._tensor_split = [2, 1]
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0, tensor_split = [2, 1]) is True


def test_auto_layers_reload_tracks_only_gpu_layers():
    # Under Auto the MoE/split knobs don't apply, so leftover values must not reload.
    backend = _loaded_backend("manual")
    backend._gpu_layers = -1
    backend._n_cpu_moe = 0
    backend._tensor_split = None
    assert _target_state_manual(backend, gpu_layers = -1, n_cpu_moe = 8, tensor_split = [2, 1]) is True
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0) is False


def test_manual_offload_emits_gpu_layers_fit_off_and_n_cpu_moe():
    src = _load_model_source()
    gate = src.find('elif gpu_memory_mode == "manual":')
    assert gate != -1, "load_model must have an explicit-offload manual branch"
    block = src[gate : gate + 700]
    assert "gpus = []" in block
    assert "tensor_parallel = False" not in block
    assert 'if gpu_memory_mode == "manual" and gpu_layers >= 0:' in src
    assert 'cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])' in src
    assert "_resolve_cpu_moe_flag(" in src
    assert 'cmd.extend(["--n-cpu-moe", str(moe_flag)])' in src
    moe_emit = src.find('cmd.extend(["--n-cpu-moe", str(moe_flag)])')
    assert "elif n_cpu_moe:" in src[moe_emit : moe_emit + 300]
    assert "self._n_cpu_moe = 0" in src[moe_emit : moe_emit + 300]
    emit = src.find('cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])')
    assert "use_fit = False" in src[src.rfind("\n", 0, emit) - 200 : emit + 80]


def test_status_reports_requested_context_length():
    assert "requested_context_length" in InferenceStatusResponse.model_fields
    s = InferenceStatusResponse(requested_context_length = 8192)
    assert s.model_dump()["requested_context_length"] == 8192
    assert InferenceStatusResponse().model_dump()["requested_context_length"] is None
    from pathlib import Path as _P

    route_src = (_P(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    assert "requested_context_length = llama_backend.requested_n_ctx" in route_src


def test_manual_offload_emits_tensor_split():
    # llama-server aborts on a split/GPU-count mismatch, so stale ratios must not emit.
    src = _load_model_source()
    assert "if tensor_split and _split_gpus > 1:" in src
    assert "_sanitized_split = self._sanitize_tensor_split(tensor_split)" in src
    assert "if len(_sanitized_split) == _split_gpus and _split_total > 0:" in src
    assert '"--tensor-split"' in src
    gate = src.find('if gpu_memory_mode == "manual" and gpu_layers >= 0:')
    nxt = src.find("elif use_fit:", gate)
    assert '","' in src[gate:nxt] and "tensor_split" in src[gate:nxt]
    assert "elif tensor_split:" in src[gate:nxt]
    drop = src.find("elif tensor_split:", gate, nxt)
    assert "self._tensor_split = None" in src[drop : drop + 250]


def test_sanitize_tensor_split_clamps_negative_and_non_finite():
    # inf passes a > 0 total gate and poisons llama.cpp's running-total shares; clamp to 0.
    sanitize = LlamaCppBackend._sanitize_tensor_split
    assert sanitize([2, 1]) == [2.0, 1.0]
    assert sanitize([-1, 2]) == [0.0, 2.0]
    assert sanitize([float("inf"), 1]) == [0.0, 1.0]
    assert sanitize([float("nan"), 1]) == [0.0, 1.0]
    assert sanitize([0, 0]) == [0.0, 0.0]
    assert sanitize(["x", 1]) == []
    assert sanitize([10**400, 1]) == []


def test_zero_offload_mask_honors_device_pin_spellings():
    # llama-server aborts on a device pin it cannot see, in any of its spellings.
    load_src = _load_model_source()
    assert "self._zero_offload_keeps_gpu_visible(cmd, env)" in load_src
    block = inspect.getsource(LlamaCppBackend._cmd_has_gpu_device_pin)
    for flag in (
        '"--device"',
        '"-dev"',
        '"--spec-draft-device"',
        '"-devd"',
        '"--device-draft"',
    ):
        assert flag in block
    assert '"LLAMA_ARG_DEVICE"' in block


def test_resolve_cpu_moe_flag():
    # --n-cpu-moe counts from layer 0, so offset past leading dense layers.
    R = LlamaCppBackend._resolve_cpu_moe_flag
    assert R(0, 40, 0) is None
    assert R(8, 0, 0) is None
    assert R(8, 40, 0) == 8
    assert R(100, 40, 0) == 40
    assert R(5, 46, 1) == 6  # offset past the 1 dense layer
    assert R(46, 46, 1) == 47


def test_manual_allows_tensor_parallel_via_split_mode():
    src = _load_model_source()
    assert 'plan_tp = tensor_parallel and gpu_memory_mode != "manual"' in src
    assert "if plan_tp:" in src
    assert "if plan_tp and len(tp_gpus) < 2:" in src
    sm = src.find('cmd.extend(["--split-mode", "tensor"])')
    assert sm != -1, "TP must emit --split-mode tensor"
    guard = src.rfind("if tensor_parallel:", 0, sm)
    assert guard != -1 and sm - guard < 200, "split-mode gates on tensor_parallel"
    assert "if tp_tensor_split and len(tp_tensor_split) > 1:" in src


def test_fit_sets_target_margin():
    caps = {"supports_fit_target": True}
    flags = LlamaCppBackend._ctx_integrity_flags(1, True, True, 0, 0, caps)
    assert flags[flags.index("--fit-target") + 1] == "512"
    # -c 0 pins native on the legacy auto path, so the tighter margin must not ride along.
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(1, True, False, 0, 0, caps)
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(1, False, False, 0, 0, caps)
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(
        1, True, True, 0, 0, {"supports_fit_target": False}
    )


def test_load_request_accepts_gpu_ids():
    req = LoadRequest(model_path = "owner/repo", gpu_ids = [1, 0])
    assert req.gpu_ids == [1, 0]
    assert LoadRequest(model_path = "owner/repo").gpu_ids is None


def test_gpu_ids_property_default_and_reset():
    backend = LlamaCppBackend()
    assert backend.gpu_ids is None
    assert backend.requested_gpu_ids is None
    backend._gpu_ids = [0, 1]
    backend._requested_gpu_ids = [0, 1, 2]
    assert backend.gpu_ids == [0, 1]
    assert backend.requested_gpu_ids == [0, 1, 2]
    backend._process = _FakeProcess()
    backend.unload_model()
    assert backend.gpu_ids is None
    assert backend.requested_gpu_ids is None


def _target_state_gpu_ids(backend, gpu_ids):
    return backend.adopt_load_intent_if_matched(
        GgufLoadIntent(
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
    )


def test_gpu_ids_reload_detection_is_order_sensitive():
    backend = _loaded_backend("auto")
    backend._gpu_ids = [0, 1]
    backend._requested_gpu_ids = [0, 1]
    assert _target_state_gpu_ids(backend, [1, 0]) is False
    assert _target_state_gpu_ids(backend, [0, 1]) is True
    assert _target_state_gpu_ids(backend, [0]) is False
    assert _target_state_gpu_ids(backend, None) is False


def test_gpu_ids_reload_detection_accepts_raw_and_effective_pin():
    backend = _loaded_backend("auto")
    backend._requested_gpu_ids = [0, 1]
    backend._gpu_ids = [0]
    backend._last_load_intent = GgufLoadIntent(
        gpu_ids = [0, 1],
        model_identifier = "owner/repo",
    )

    assert _target_state_gpu_ids(backend, [0, 1]) is True
    assert backend.requested_gpu_ids == [0, 1]
    assert _target_state_gpu_ids(backend, [0]) is True
    assert backend.requested_gpu_ids == [0]
    assert backend._last_load_intent.gpu_ids == (0,)
    assert backend._last_load_intent.model_identifier == "owner/repo"
    assert _target_state_gpu_ids(backend, [1]) is False
    assert _target_state_gpu_ids(backend, None) is False


@pytest.mark.parametrize(
    ("gpu_ids", "expected_ids", "matches"),
    [([], None, False), ([0], (0,), True)],
)
def test_gpu_ids_control_owned_device_extra_args(gpu_ids, expected_ids, matches):
    backend = _loaded_backend("auto")
    if gpu_ids:
        backend._gpu_ids = backend._requested_gpu_ids = [0]
    intent = GgufLoadIntent(
        model_identifier = "owner/repo",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        speculative_type = "auto",
        gpu_ids = gpu_ids,
        extra_args = ["--main-gpu", "1"],
    )

    assert intent.gpu_ids == expected_ids
    assert backend.adopt_load_intent_if_matched(intent) is matches


def test_gpu_ids_reload_detection_collapses_diffusion_to_single_device():
    # The diffusion runner uses only the lowest device, so dedupe on that.
    backend = _loaded_backend("auto")
    backend._is_diffusion = True
    backend._gpu_ids = [1]
    assert _target_state_gpu_ids(backend, [3, 1]) is True
    assert backend.requested_gpu_ids == [1]
    assert _target_state_gpu_ids(backend, [1]) is True
    assert _target_state_gpu_ids(backend, [3, 2]) is False
    assert _target_state_gpu_ids(backend, None) is False


def test_remote_vulkan_diffusion_preflight_runs_before_teardown(monkeypatch):
    def _mark_diffusion(probe, path):
        assert path == "/cache/model.gguf"
        probe._is_diffusion = True

    monkeypatch.setattr(LlamaCppBackend, "_read_gguf_metadata", _mark_diffusion)
    assert LlamaCppBackend._gguf_path_is_diffusion("/cache/model.gguf", "owner/model") is True

    src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
    preflight = src.index("_preflight_model_path = self._download_gguf(")
    teardown = src.index("# ── Phase 1: kill old process")
    assert preflight < teardown
    phase_two = src.index("model_path = _preflight_model_path", teardown)
    download = src.index("model_path = self._download_gguf(", teardown)
    assert phase_two < download


def test_local_vulkan_diffusion_preflight_runs_before_teardown():
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
    local_preflight = src.index(
        "self._reject_vulkan_diffusion_gpu_ids_before_teardown(\n                    gguf_path,"
    )
    teardown = src.index("# ── Phase 1: kill old process")
    assert local_preflight < teardown


def test_remote_vulkan_diffusion_rejection_keeps_active_server(monkeypatch):
    backend = LlamaCppBackend()
    killed = []
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda **_kwargs: "/bin/llama")
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: True)
    monkeypatch.setattr(backend, "_get_gpu_memory", lambda _binary = None, **_kw: [(0, 1024, 2048)])
    monkeypatch.setattr(
        backend,
        "_download_gguf",
        lambda **_kwargs: "/cache/diffusion.gguf",
    )
    monkeypatch.setattr(backend, "_gguf_path_is_diffusion", lambda *_args: True)
    monkeypatch.setattr(backend, "_kill_process", lambda: killed.append(True))
    monkeypatch.setattr(
        llama_cpp_module,
        "_resolve_repo_id_casing",
        lambda repo: repo,
    )
    monkeypatch.setattr(
        llama_cpp_module,
        "_hf_offline_if_unreachable",
        lambda: __import__("contextlib").nullcontext(),
    )

    with pytest.raises(ValueError, match = "DiffusionGemma"):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "owner/model",
                hf_variant = "Q4_K_M",
                model_identifier = "owner/model",
                gpu_ids = [0],
            )
        )

    assert killed == []


def test_remote_vulkan_preflight_download_failure_keeps_active_server(monkeypatch, tmp_path):
    import hub.utils.gguf as hub_gguf

    cached_shard = tmp_path / "model-00001-of-00003.gguf"
    cached_shard.write_bytes(b"GGUF")
    monkeypatch.setattr(
        hub_gguf,
        "resolve_local_gguf_path",
        lambda _repo, _variant: str(cached_shard),
    )

    for failure in (
        FileNotFoundError("shard 2 of 3 missing"),
        OSError("[Errno 28] No space left on device"),
        ConnectionError("hub unreachable"),
    ):
        backend = LlamaCppBackend()
        order = []

        def _download(_failure = failure, **_kwargs):
            order.append("download")
            raise _failure

        monkeypatch.setattr(backend, "_find_llama_server_binary", lambda **_kwargs: "/bin/llama")
        monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: True)
        monkeypatch.setattr(
            backend, "_get_gpu_memory", lambda _binary = None, **_kw: [(0, 1024, 2048)]
        )
        monkeypatch.setattr(backend, "_download_gguf", _download)
        monkeypatch.setattr(backend, "_gguf_path_is_diffusion", lambda *_args: False)
        monkeypatch.setattr(backend, "_kill_process", lambda: order.append("kill"))
        monkeypatch.setattr(llama_cpp_module, "_resolve_repo_id_casing", lambda repo: repo)
        monkeypatch.setattr(
            llama_cpp_module,
            "_hf_offline_if_unreachable",
            lambda: __import__("contextlib").nullcontext(),
        )

        with pytest.raises(type(failure)):
            backend.load_model(
                GgufLoadIntent(
                    hf_repo = "owner/model",
                    hf_variant = "Q4_K_M",
                    model_identifier = "owner/model",
                    gpu_ids = [0],
                )
            )

        assert order == ["download"], failure


def test_local_vulkan_diffusion_rejection_keeps_active_server(monkeypatch, tmp_path):
    gguf_path = tmp_path / "diffusion.gguf"
    gguf_path.write_bytes(b"GGUF")

    backend = LlamaCppBackend()
    killed = []
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda **_kwargs: "/bin/llama")
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: True)
    monkeypatch.setattr(backend, "_get_gpu_memory", lambda _binary = None, **_kw: [(0, 1024, 2048)])
    monkeypatch.setattr(backend, "_gguf_path_is_diffusion", lambda *_args: True)
    monkeypatch.setattr(backend, "_kill_process", lambda: killed.append(True))

    with pytest.raises(ValueError, match = "DiffusionGemma"):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf_path),
                model_identifier = "local/diffusion",
                gpu_ids = [0],
            )
        )

    assert killed == []


class _ReachedServerStart(Exception):
    """Marks a load getting past the pre-teardown preflight."""


def _write_gguf_header(
    path: Path,
    architecture: str,
    *,
    diffusion: bool = False,
) -> str:
    """Smallest GGUF the header probe can classify: arch, plus the canvas marker."""

    def _kv_str(key: str, value: str) -> bytes:
        kb, vb = key.encode(), value.encode()
        return (
            struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        )

    def _kv_u32(key: str, value: int) -> bytes:
        kb = key.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)

    body = _kv_str("general.architecture", architecture)
    if diffusion:
        body += _kv_u32("diffusion.canvas_length", 256)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 2 if diffusion else 1) + body)
    return str(path)


def _vulkan_pinned_backend(monkeypatch, killed: list) -> LlamaCppBackend:
    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda **_kwargs: "/bin/llama")
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: True)
    monkeypatch.setattr(backend, "_get_gpu_memory", lambda _binary = None, **_kw: [(0, 1024, 2048)])
    monkeypatch.setattr(backend, "_kill_process", lambda: killed.append(True))
    return backend


def test_local_vulkan_pre_teardown_reads_the_real_gguf_header(monkeypatch, tmp_path):
    killed = []
    backend = _vulkan_pinned_backend(monkeypatch, killed)
    monkeypatch.setattr(
        backend,
        "_wait_for_vram_settle",
        lambda **_kwargs: (_ for _ in ()).throw(_ReachedServerStart()),
    )

    with pytest.raises(_ReachedServerStart):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = _write_gguf_header(tmp_path / "chat.gguf", "llama"),
                model_identifier = "local/chat",
                gpu_ids = [0],
            )
        )

    assert killed == [True]


def test_local_vulkan_diffusion_header_rejects_before_teardown(monkeypatch, tmp_path):
    killed = []
    backend = _vulkan_pinned_backend(monkeypatch, killed)

    with pytest.raises(ValueError, match = "DiffusionGemma"):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = _write_gguf_header(tmp_path / "d.gguf", "gemma3", diffusion = True),
                model_identifier = "local/diffusion",
                gpu_ids = [0],
            )
        )

    assert killed == []


def test_local_vulkan_missing_gguf_is_reported_before_teardown(monkeypatch, tmp_path):
    killed = []
    backend = _vulkan_pinned_backend(monkeypatch, killed)

    with pytest.raises(FileNotFoundError):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(tmp_path / "absent.gguf"),
                model_identifier = "local/missing",
                gpu_ids = [0],
            )
        )

    assert killed == []


def test_start_diffusion_server_resets_tensor_parallel():
    # load_model phase 1 skips the unload reset, so diffusion startup must clear TP.
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend._start_diffusion_server)
    assert "self._tensor_parallel = False" in src
    assert "self._requested_gpu_ids = [sorted(gpu_ids)[0]] if gpu_ids else None" in src


@pytest.mark.parametrize(
    ("parent_ids", "expected"),
    [([], None), ([2], (2,)), ([0, 1], None)],
)
def test_unmasked_child_gpu_map_is_known_only_for_one_gpu(monkeypatch, parent_ids, expected):
    import utils.hardware as hw
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: parent_ids)
    assert LlamaCppBackend._unmasked_child_gpu_physical_ids() == expected


def _patch_split_pin_env(monkeypatch, *, inherited, reported):
    """Point the pin helper at a fake inherited mask and picker report.
    ``reported`` None = enumeration unavailable (falls back to ascending)."""
    import utils.hardware as hw

    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: inherited)
    )
    info = (
        {"available": False}
        if reported is None
        else {
            "available": True,
            "index_kind": "physical",
            "devices": [{"index": i} for i in reported],
        }
    )
    monkeypatch.setattr(hw, "get_backend_visible_gpu_info", lambda: info)


def test_split_pin_reorders_inherited_numeric_mask(monkeypatch):
    # The mask must be re-emitted in picker order or the shares land on the wrong cards.
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [1, 3])
    env = {"CUDA_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"


def test_split_pin_keeps_mask_order_when_picker_reported_it(monkeypatch):
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [3, 1])
    env = {"CUDA_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "3,1"


def test_split_pin_falls_back_to_ascending_without_report(monkeypatch):
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = None)
    env = {"CUDA_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"


def test_split_pin_without_mask_only_sets_pci_order(monkeypatch):
    _patch_split_pin_env(monkeypatch, inherited = None, reported = None)
    env = {}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_split_pin_mirrors_hip_mask_on_rocm(monkeypatch):
    # Inherited ROCR is cleared so the mask can't apply twice (ROCR re-indexes first).
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [1, 3])
    _rocm_torch_stub(monkeypatch)
    env = {
        "CUDA_VISIBLE_DEVICES": "3,1",
        "HIP_VISIBLE_DEVICES": "3,1",
        "ROCR_VISIBLE_DEVICES": "3,1",
    }
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert env["HIP_VISIBLE_DEVICES"] == "1,3"
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_split_pin_preserves_inherited_rocr_mask(monkeypatch):
    # Clearing ROCR re-exposes agents to HSA enumeration, which can segfault on unsupported GPUs.
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [1, 3])
    _rocm_torch_stub(monkeypatch)
    env = {"ROCR_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["ROCR_VISIBLE_DEVICES"] == "1,3"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_split_pin_keeps_hip_on_windows_despite_stray_rocr(monkeypatch):
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [1, 3])
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = "6.0")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(sys, "platform", "win32")
    env = {"CUDA_VISIBLE_DEVICES": "3,1", "ROCR_VISIBLE_DEVICES": "9"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert env["HIP_VISIBLE_DEVICES"] == "1,3"
    assert "ROCR_VISIBLE_DEVICES" not in env


def _rocm_torch_stub(monkeypatch):
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = "6.0")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    # prefer_rocr is Linux-only; pin the platform so these pass on Windows too.
    monkeypatch.setattr(sys, "platform", "linux")


def test_subset_pin_masks_via_rocr_on_rocm(monkeypatch):
    # HIP masking still enumerates every agent first, which segfaults on an unsupported GPU.
    _rocm_torch_stub(monkeypatch)
    env = {"HIP_VISIBLE_DEVICES": "9"}
    LlamaCppBackend._emit_child_gpu_visibility(env, "0", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_prefer_rocr_remaps_cuda_to_post_rocr_ordinals(monkeypatch):
    # ROCR re-indexes from 0 and HIP falls back to CUDA_VISIBLE_DEVICES, so CUDA gets ordinals.
    _rocm_torch_stub(monkeypatch)
    env = {}
    LlamaCppBackend._emit_child_gpu_visibility(env, "1", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in env
    env = {}
    LlamaCppBackend._emit_child_gpu_visibility(env, "1,3", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "1,3"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_subset_pin_default_still_uses_hip_and_clears_rocr(monkeypatch):
    _rocm_torch_stub(monkeypatch)
    env = {"ROCR_VISIBLE_DEVICES": "0,1"}
    LlamaCppBackend._emit_child_gpu_visibility(env, "1")
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_cpu_only_pin_keeps_hip_even_with_prefer_rocr(monkeypatch):
    _rocm_torch_stub(monkeypatch)
    env = {}
    LlamaCppBackend._emit_child_gpu_visibility(env, "-1", prefer_rocr = True)
    assert env["HIP_VISIBLE_DEVICES"] == "-1"
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_cpu_only_pin_keeps_an_inherited_rocr_mask(monkeypatch):
    # -1 hides everything; clearing ROCR would re-expose agents to HSA enumeration.
    _rocm_torch_stub(monkeypatch)
    env = {"ROCR_VISIBLE_DEVICES": "1"}
    LlamaCppBackend._emit_child_gpu_visibility(env, "-1")
    assert env["HIP_VISIBLE_DEVICES"] == "-1"
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert env["ROCR_VISIBLE_DEVICES"] == "1"


def _amd_sdk_torch_stub(monkeypatch):
    # AMD SDK wheel: torch.version.hip is None but __version__ encodes rocm.
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = None)
    torch_stub.__version__ = "2.9.1+rocm7.2.1"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(sys, "platform", "linux")


def test_prefer_rocr_falls_back_to_hip_on_windows(monkeypatch):
    # Windows HIP has no ROCr layer, so keep the HIP mask there.
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = "6.0")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(sys, "platform", "win32")
    env = {"ROCR_VISIBLE_DEVICES": "9"}
    LlamaCppBackend._emit_child_gpu_visibility(env, "1", prefer_rocr = True)
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_amd_sdk_wheel_hip_none_still_masks_rocr(monkeypatch):
    _amd_sdk_torch_stub(monkeypatch)
    env = {"HIP_VISIBLE_DEVICES": "9"}
    LlamaCppBackend._emit_child_gpu_visibility(env, "0", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_cuda_wheel_hip_none_gets_no_rocm_mask(monkeypatch):
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = None)
    torch_stub.__version__ = "2.9.1+cu124"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    env = {}
    LlamaCppBackend._emit_child_gpu_visibility(env, "0", prefer_rocr = True)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "ROCR_VISIBLE_DEVICES" not in env
    assert "HIP_VISIBLE_DEVICES" not in env


def test_resolve_physical_ids_reads_rocr_on_amd_sdk_wheel(monkeypatch):
    # On an AMD SDK wheel an inherited ROCR mask IS the ordinal->physical mapping.
    _amd_sdk_torch_stub(monkeypatch)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert LlamaCppBackend._resolve_visible_physical_ids() == [1]


def test_resolve_physical_ids_ignores_rocr_on_cuda_wheel(monkeypatch):
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = None)
    torch_stub.__version__ = "2.9.1+cu124"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert LlamaCppBackend._resolve_visible_physical_ids() is None


def test_resolve_physical_ids_ignores_rocr_on_windows(monkeypatch):
    # Windows HIP has no ROCr layer: a stray ROCR var must not be read as the mapping.
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = None)
    torch_stub.__version__ = "2.9.1+rocm7.2.1"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(sys, "platform", "win32")
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert LlamaCppBackend._resolve_visible_physical_ids() is None
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert LlamaCppBackend._resolve_visible_physical_ids() == [1]


def test_diffusion_gpu_arg_uses_lowest_explicit_physical_id(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setenv("DG_GPU", "7")
    assert LlamaCppBackend._diffusion_gpu_arg([3, 1]) == "1"


def test_diffusion_gpu_arg_preserves_parent_mask_order(monkeypatch):
    monkeypatch.delenv("DG_GPU", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert LlamaCppBackend._diffusion_gpu_arg(None) == "3"


def test_diffusion_gpu_arg_honors_override_and_cpu_mask(monkeypatch):
    monkeypatch.setenv("DG_GPU", "GPU-abc")
    assert LlamaCppBackend._diffusion_gpu_arg(None) == "GPU-abc"
    assert LlamaCppBackend._diffusion_gpu_arg(None, cpu_only = True) == ""


def test_zero_offload_flag_false_without_companions():
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0", "--fit", "off"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is False


@pytest.mark.parametrize(
    "companion",
    ["--mmproj", "-mm", "--model-draft", "-md", "--spec-draft-model", "-hfd"],
)
def test_zero_offload_flag_true_with_companion(companion):
    # mmproj / a drafter offload to GPU regardless of --gpu-layers.
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0", companion, "x.gguf"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is True


def test_zero_offload_flag_true_with_inline_companion_forms():
    cmd = ["llama-server", "-m", "model.gguf", "--spec-draft-model=x.gguf"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is True
    cmd = ["llama-server", "-m", "model.gguf", "--mmproj=proj.gguf"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is True


def test_zero_offload_flag_true_with_env_drafter():
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0"]
    env = {"LLAMA_ARG_SPEC_DRAFT_MODEL": "x.gguf"}
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], env) is True


@pytest.mark.parametrize(
    "device_args",
    [
        ["--device", "CUDA0"],
        ["--device=CUDA0"],
        ["-dev", "CUDA0"],
        ["--spec-draft-device", "CUDA0"],
        ["--device-draft=CUDA0"],
    ],
)
def test_zero_offload_flag_true_with_device_pin(device_args):
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0", *device_args]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is True


def test_zero_offload_flag_true_with_env_device_pin():
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0"]
    env = {"LLAMA_ARG_DEVICE": "CUDA0"}
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], env) is True


@pytest.mark.parametrize(
    ("device_args", "env"),
    [
        (["--device", "cpu"], {}),
        (["--device=none"], {}),
        (["--spec-draft-device", "cpu"], {}),
        ([], {"LLAMA_ARG_DEVICE": "none"}),
        (["--device", "CUDA0", "--device", "cpu"], {}),
    ],
)
def test_zero_offload_flag_false_with_cpu_device_pin(device_args, env):
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0", *device_args]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], env) is False


def test_zero_offload_flag_true_with_surviving_tensor_mode():
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0", "--split-mode", "tensor"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is True


def test_zero_offload_flag_true_for_unmasked_vulkan(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is True


def test_zero_offload_flag_none_without_gpus():
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [], {}) is None


def test_cmd_has_gpu_companion_detection():
    has = LlamaCppBackend._cmd_has_gpu_companion
    assert has(["llama-server", "-m", "m.gguf"], {}) is False
    assert has(["llama-server", "--mmproj", "p.gguf"], {}) is True
    assert has(["llama-server", "--mmproj=p.gguf"], {}) is True
    assert has(["llama-server", "-md", "d.gguf"], {}) is True
    assert has(["llama-server"], {"LLAMA_ARG_SPEC_DRAFT_MODEL": "d.gguf"}) is True


def test_cmd_companion_ignores_cpu_forced_drafter():
    has = LlamaCppBackend._cmd_has_gpu_companion
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-ngl", "0"]
    assert has(cmd, {}) is False
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-device", "cpu"]
    assert has(cmd, {}) is False
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-ngl", "0", "--mmproj", "p.gguf"]
    assert has(cmd, {}) is True


@pytest.fixture
def not_vulkan(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a, **k: False))


def test_zero_vram_chat_load_only_for_a_deliberate_cpu_only_offload(not_vulkan):
    # Manual + gpu_layers=0 is the only shape launched with GPUs hidden, so it may skip the arbiter.
    zero = llama_cpp_module.zero_vram_chat_load
    # An absent mode resolves to auto, which is GPU-bearing.
    assert zero("manual", 0, [], False, "off") is True
    assert zero("auto", 0, [], False, "off") is False
    assert zero("manual", 1, [], False, "off") is False
    assert zero("manual", -1, [], False, "off") is False


def test_zero_vram_chat_load_refuses_every_gpu_companion(not_vulkan):
    zero = llama_cpp_module.zero_vram_chat_load
    assert zero("manual", 0, ["--device", "CUDA0"], False, "off") is False
    assert zero("manual", 0, ["-dev", "CUDA0"], False, "off") is False
    assert zero("manual", 0, ["--split-mode", "tensor"], False, "off") is False
    assert zero("manual", 0, ["--model-draft", "/tmp/draft.gguf"], False, "off") is False
    assert zero("manual", 0, [], True, "off") is False
    assert zero("manual", 0, [], False, "model") is False
    assert zero("manual", 0, ["--device", "none"], False, "off") is True
    assert (
        zero("manual", 0, ["--model-draft", "/tmp/d.gguf", "--spec-draft-ngl", "0"], False, "off")
        is True
    )


def test_zero_vram_chat_load_exempts_disabled_speculation(not_vulkan):
    zero = llama_cpp_module.zero_vram_chat_load
    assert zero("manual", 0, [], False, "off") is True
    assert zero("manual", 0, [], False, " OFF ") is True
    # Empty is not off: it canonicalizes to None, which resolves to auto.
    assert zero("manual", 0, [], False, "") is False
    assert zero("manual", 0, [], False, "auto") is False
    assert zero("manual", 0, [], False, "mtp") is False
    assert zero("manual", 0, [], False, "default") is False


def test_zero_vram_chat_load_is_skipped_on_vulkan(monkeypatch):
    # Vulkan builds are exempt from the CPU-only mask at launch.
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a, **k: True))
    assert llama_cpp_module.zero_vram_chat_load("manual", 0) is False


def test_holds_no_vram_needs_the_launch_to_have_confirmed_it():
    backend = LlamaCppBackend()
    backend._gpu_memory_mode = "manual"
    backend._gpu_layers = 0
    backend._gpu_offload_active = False
    assert backend.holds_no_vram is True
    backend._gpu_offload_active = True
    assert backend.holds_no_vram is False
    backend._gpu_offload_active = None
    assert backend.holds_no_vram is False
    backend._gpu_offload_active = False
    backend._gpu_layers = 20
    assert backend.holds_no_vram is False
    backend._gpu_layers = 0
    backend._gpu_memory_mode = "auto"
    assert backend.holds_no_vram is False


def test_a_cpu_only_chat_load_does_not_take_the_gpu_arbiter():
    route_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    load_impl = route_src[route_src.index("async def _load_model_impl") :]
    assert "chat_load_needs_gpu = not (" in load_impl
    gate = load_impl.index("chat_load_needs_gpu = not (")
    acquire = load_impl.index("if chat_load_needs_gpu:", gate)
    # Release only after the load, or an image/video load could allocate beside the old model.
    release = load_impl.index(
        "await asyncio.to_thread(_release_chat_for_zero_vram_primary)", acquire
    )
    assert load_impl.index("if replacing and not chat_load_needs_gpu:", acquire) < release
    assert load_impl.index("success = await load_with_tensor_fallback(", acquire) < release
    assert "if chat_load_needs_gpu and current_owner() != CHAT:" in load_impl
    assert "if not llama_backend.holds_no_vram:" in load_impl


def test_cmd_companion_ignores_a_projector_pinned_off_the_gpu():
    # --no-mmproj-offload makes clip.cpp skip the GPU backend entirely.
    has = LlamaCppBackend._cmd_has_gpu_companion
    cmd = ["llama-server", "--mmproj", "p.gguf", "--no-mmproj-offload"]
    assert has(cmd, {}) is False
    # llama.cpp assigns rather than accumulates this flag, so the last wins.
    cmd = ["llama-server", "--mmproj", "p.gguf", "--no-mmproj-offload", "--mmproj-offload"]
    assert has(cmd, {}) is True


def test_zero_vram_chat_load_treats_an_absent_mode_as_auto(not_vulkan):
    # An absent mode resolves to auto, which may launch a GPU drafter, so it is not exempt.
    zero = llama_cpp_module.zero_vram_chat_load
    assert zero("manual", 0) is False
    assert zero("manual", 0, [], False, None) is False
    assert zero("manual", 0, [], False, "") is False
    assert zero("manual", 0, [], False, "   ") is False
    assert zero("manual", 0, [], False, "off") is True


def test_manual_auto_layers_never_emits_two_tensor_splits(tmp_path):
    """Manual + Auto layers is the one cell where the route holds the ratio twice: the
    strip is False at ``gpu_layers < 0`` while the promotion still fires (#11330).
    llama.cpp reads the LAST ``--tensor-split``, so a launch path that kept either copy
    would place by the wrong one; both die here, and that is the launcher's doing rather
    than the route's, which is why it is pinned.
    """
    from test_llama_cpp_placement import _backend, _launch

    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 16_000, 16_000), (1, 16_000, 16_000)],
    )
    cmd = _launch(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        gpu_ids = [0, 1],
        n_ctx = 4096,
        tensor_split = [2.2, 1.0],
        extra_args = ["-ts", "2.2,1", "-sm", "layer"],
    )["cmd"]
    tokens = [tok for tok in cmd if tok in ("--tensor-split", "-ts")]
    assert len(tokens) <= 1, f"the ratio reached argv twice: {cmd}"
    assert tokens == []
    assert backend.tensor_split is None


def test_manual_explicit_layers_emits_the_promoted_ratio_once(tmp_path):
    """The control for the cell above: with layers pinned the strip DOES fire, so the
    promoted field is the only copy and it is the one that reaches argv (#11330)."""
    from test_llama_cpp_placement import _backend, _launch

    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 16_000, 16_000), (1, 16_000, 16_000)],
    )
    cmd = _launch(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = 49,
        gpu_ids = [0, 1],
        n_ctx = 4096,
        tensor_split = [2.2, 1.0],
        extra_args = ["-sm", "layer"],
    )["cmd"]
    assert cmd.count("--tensor-split") == 1
    assert cmd[cmd.index("--tensor-split") + 1] == "2.2,1"
    assert backend.tensor_split == [2.2, 1.0]


def test_manual_auto_layers_does_not_judge_a_rewrite_that_never_happens():
    """At Auto layers the launcher drops both copies of the ratio, so the route must not refuse
    a value over the six-digit rendering it will never produce (the strip predicate and the
    reserialized predicate are the same question)."""
    route_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    for marker in (
        "reserialized = _resolved_layers >= 0",
        "reserialized = _validate_resolved_layers >= 0",
    ):
        assert marker in route_src, marker
