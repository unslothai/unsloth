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
    backend._gpu_memory_mode = "manual"
    assert backend.gpu_memory_mode == "manual"


def test_unload_resets_gpu_memory_mode():
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._gpu_memory_mode = "manual"
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


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_already_in_target_state_matches_same_mode(mode):
    assert _target_state(_loaded_backend(mode), mode) is True


@pytest.mark.parametrize("loaded,requested", [("auto", "manual"), ("manual", "auto")])
def test_already_in_target_state_reloads_on_mode_change(loaded, requested):
    # Flipping the dropdown either direction must force a reload so the command
    # is rebuilt with/without the Unsloth GPU masking.
    assert _target_state(_loaded_backend(loaded), requested) is False


def test_already_in_target_state_ignores_mode_for_diffusion():
    # The diffusion runner is mode-agnostic (always reports "auto"), so a standing
    # manual preference in the request must not force a needless reload of an
    # already-loaded diffusion model.
    backend = _loaded_backend("auto")
    backend._is_diffusion = True
    assert _target_state(backend, "manual") is True


# ── load_model: manual + Auto layers bypasses Unsloth GPU management ──


def _load_model_source() -> str:
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def test_auto_layers_branch_empties_gpus_and_drops_tensor_parallel():
    # Emptying the probed set makes the selection / TP planning below no-op, so
    # gpu_indices stays None and use_fit True (--fit on).
    src = _load_model_source()
    gate = src.find('if gpu_memory_mode == "manual" and gpu_layers < 0:')
    assert gate != -1, "load_model must branch on manual + Auto layers (gpu_layers < 0)"
    block = src[gate : gate + 1400]
    assert "gpus = []" in block, "Auto-layers branch must empty the probed GPU set"
    # --fit aborts under --split-mode tensor, so a raw-extras split-mode is stripped.
    assert "strip_split_mode_only(extra_args)" in block
    assert "requested_ctx if requested_ctx > 0 else 0" in block
    # The branch sits before GPU selection assigns gpu_indices; --fit on is its emission.
    assert gate < src.find("gpu_indices, use_fit = None, True")
    assert 'cmd.extend(["--fit", "on"])' in src
    # TP drops for this path, but at a guard BEFORE the quantized-KV cache-drop, so
    # a requested quantized cache survives into the --fit load.
    tp_drop = src.find('if tensor_parallel and gpu_memory_mode == "manual" and gpu_layers < 0:')
    assert tp_drop != -1, "manual + Auto layers must drop tensor_parallel"
    assert "tensor_parallel = False" in src[tp_drop : tp_drop + 400]
    cache_drop = src.find("Tensor parallelism requires a non-quantized KV cache")
    assert cache_drop != -1
    assert (
        tp_drop < cache_drop
    ), "TP must drop before the cache-drop so a quantized KV survives --fit"


def test_auto_layers_never_sends_ctx_size_zero():
    # Sending "-c 0" sets fit_params_min_ctx = UINT32_MAX in llama.cpp, pinning
    # the full native context and disabling --fit's reduction. So the base cmd
    # must never carry -c, "-c 0" is emitted only outside the Auto-layers (--fit)
    # case, and a positive context is passed through (which --fit optimizes
    # layers around).
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


# ── Manual offload (--gpu-layers + --fit off + --n-cpu-moe) ───────────


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


def _target_state_manual(
    backend,
    *,
    gpu_layers,
    n_cpu_moe,
    tensor_split = None,
):
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
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0, tensor_split = [2, 1]) is False
    # Same GPU split -> no reload.
    backend._tensor_split = [2, 1]
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0, tensor_split = [2, 1]) is True


def test_auto_layers_reload_tracks_only_gpu_layers():
    # Under Auto (gpu_layers < 0) the MoE/split knobs don't apply, so a leftover
    # value in the request must not force a reload -- only a change in gpu_layers
    # (Auto -> a pinned count) does.
    backend = _loaded_backend("manual")
    backend._gpu_layers = -1
    backend._n_cpu_moe = 0
    backend._tensor_split = None
    # Same Auto, leftover MoE/split in the request -> still no reload.
    assert _target_state_manual(backend, gpu_layers = -1, n_cpu_moe = 8, tensor_split = [2, 1]) is True
    # Auto -> explicit offload reloads.
    assert _target_state_manual(backend, gpu_layers = 20, n_cpu_moe = 0) is False


def test_manual_offload_emits_gpu_layers_fit_off_and_n_cpu_moe():
    src = _load_model_source()
    gate = src.find('elif gpu_memory_mode == "manual":')
    assert gate != -1, "load_model must have an explicit-offload manual branch"
    block = src[gate : gate + 700]
    # Empties the probed set (skips the planner) but keeps the user's TP choice
    # (only the Auto-layers branch above drops TP).
    assert "gpus = []" in block
    assert "tensor_parallel = False" not in block
    # The cmd emits the layer count with fit disabled, gated on gpu_layers >= 0.
    assert 'if gpu_memory_mode == "manual" and gpu_layers >= 0:' in src
    assert 'cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])' in src
    # MoE offload uses --n-cpu-moe via _resolve_cpu_moe_flag (tested behaviorally below).
    assert "_resolve_cpu_moe_flag(" in src
    assert 'cmd.extend(["--n-cpu-moe", str(moe_flag)])' in src
    # The offload path forces use_fit False so --fit-ctx is never added under --fit off.
    emit = src.find('cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])')
    assert "use_fit = False" in src[src.rfind("\n", 0, emit) - 200 : emit + 80]


def test_manual_offload_emits_tensor_split():
    # The offload path emits --tensor-split from the per-GPU shares, only when
    # provided and only with >1 GPU in use (a stale ratio on a narrowed picker
    # must not emit).
    src = _load_model_source()
    assert "if tensor_split and self._effective_gpu_count(gpu_indices) > 1:" in src
    assert '"--tensor-split"' in src
    # Joined as a comma list (e.g. "2,1") within the explicit-offload cmd branch.
    gate = src.find('if gpu_memory_mode == "manual" and gpu_layers >= 0:')
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
    # Manual offload keeps the user's TP choice but skips the memory-based planner
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


def test_fit_sets_target_margin():
    # Behavior: --fit on tightens the per-device VRAM margin to 512 MiB.
    caps = {"supports_fit_target": True}
    flags = LlamaCppBackend._ctx_integrity_flags(1, True, 0, 0, caps)
    assert flags[flags.index("--fit-target") + 1] == "512"
    # Not emitted when fit is off.
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(1, False, 0, 0, caps)
    # Not emitted when the binary lacks support.
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(
        1, True, 0, 0, {"supports_fit_target": False}
    )


# ── GPU picker (gpu_ids -> CUDA_VISIBLE_DEVICES) ─────────────────────


def test_load_request_accepts_gpu_ids():
    req = LoadRequest(model_path = "owner/repo", gpu_ids = [1, 0])
    assert req.gpu_ids == [1, 0]
    assert LoadRequest(model_path = "owner/repo").gpu_ids is None


@pytest.mark.parametrize("model_cls", [LoadResponse, InferenceStatusResponse])
def test_response_models_emit_gpu_ids(model_cls):
    if model_cls is LoadResponse:
        obj = model_cls(status = "loaded", model = "m", display_name = "m", inference = {}, gpu_ids = [1])
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
    # Auto-layers / manual (gpu_indices None) get pinned to the picked set.
    assert "if gpu_ids and gpu_indices is None:" in src
    assert "gpu_indices = sorted(gpu_ids)" in src
    # Picked indices follow PCI-bus order so the UI index == llama.cpp's.
    assert 'env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"' in src


# ── Manual tensor split: child enumeration pinned to the picker's order ──────


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
    # Parent CUDA_VISIBLE_DEVICES=3,1 makes the child enumerate dev0=phys3, but
    # nvidia-smi reported the picker's list ascending -- the mask must be
    # re-emitted in that order or the per-GPU shares land on the wrong cards.
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [1, 3])
    env = {"CUDA_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"


def test_split_pin_keeps_mask_order_when_picker_reported_it(monkeypatch):
    # Torch-fallback enumeration (no nvidia-smi) reports devices in inherited
    # mask order, so the picker's split list follows the mask -- the pin must
    # keep that order, not re-sort it into a mismatch.
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [3, 1])
    env = {"CUDA_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "3,1"


def test_split_pin_falls_back_to_ascending_without_report(monkeypatch):
    # Enumeration unavailable: ascending physical is the best guess (it matches
    # the dominant nvidia-smi report order).
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = None)
    env = {"CUDA_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"


def test_split_pin_without_mask_only_sets_pci_order(monkeypatch):
    # No inherited mask (or a UUID/MIG one resolving to None): enumeration order
    # is fully fixed by CUDA_DEVICE_ORDER, so no mask is written.
    _patch_split_pin_env(monkeypatch, inherited = None, reported = None)
    env = {}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_split_pin_mirrors_hip_mask_on_rocm(monkeypatch):
    # ROCm: the pin must land in HIP_VISIBLE_DEVICES too, and an inherited ROCR
    # mask is cleared so the mask can't apply twice (ROCR re-indexes, then HIP
    # would index into the already-reduced set).
    _patch_split_pin_env(monkeypatch, inherited = [3, 1], reported = [1, 3])
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = "6.0")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    env = {"CUDA_VISIBLE_DEVICES": "3,1", "ROCR_VISIBLE_DEVICES": "3,1"}
    LlamaCppBackend._pin_visible_gpu_order_for_split(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert env["HIP_VISIBLE_DEVICES"] == "1,3"
    assert "ROCR_VISIBLE_DEVICES" not in env


# ── Deliberate zero-offload (manual gpu_layers=0): training-skip flag ─────────


def test_zero_offload_flag_false_without_companions():
    # CPU-only by construction: False lets the training coordinator skip
    # unloading a server that holds no VRAM.
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0", "--fit", "off"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [(0, 8000, 24000)], {}) is False


@pytest.mark.parametrize(
    "companion",
    ["--mmproj", "--model-draft", "-md", "--spec-draft-model", "-hfd"],
)
def test_zero_offload_flag_true_with_companion(companion):
    # mmproj / a drafter offload to GPU regardless of --gpu-layers, so the
    # server still holds VRAM and training must unload it. Drafter detection
    # reuses the extras parser, so pass-through aliases count too.
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


def test_zero_offload_flag_none_without_gpus():
    cmd = ["llama-server", "-m", "model.gguf", "--gpu-layers", "0"]
    assert LlamaCppBackend._zero_offload_gpu_flag(cmd, [], {}) is None


def test_cmd_has_gpu_companion_detection():
    # The env mask for CPU-only zero-offload loads keys off this scan: any
    # --mmproj form or a drafter (flag aliases / env) keeps the GPUs visible.
    has = LlamaCppBackend._cmd_has_gpu_companion
    assert has(["llama-server", "-m", "m.gguf"], {}) is False
    assert has(["llama-server", "--mmproj", "p.gguf"], {}) is True
    assert has(["llama-server", "--mmproj=p.gguf"], {}) is True
    assert has(["llama-server", "-md", "d.gguf"], {}) is True
    assert has(["llama-server"], {"LLAMA_ARG_SPEC_DRAFT_MODEL": "d.gguf"}) is True


def test_cmd_companion_ignores_cpu_forced_drafter():
    # A drafter pinned to CPU holds no VRAM: the zero-offload mask may hide the
    # GPUs and the training coordinator may leave the server alone.
    has = LlamaCppBackend._cmd_has_gpu_companion
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-ngl", "0"]
    assert has(cmd, {}) is False
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-device", "cpu"]
    assert has(cmd, {}) is False
    # mmproj still counts even alongside a CPU drafter.
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-ngl", "0", "--mmproj", "p.gguf"]
    assert has(cmd, {}) is True
