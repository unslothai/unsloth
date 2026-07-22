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

# httpx is a real, installed backend dependency: import it so the genuine module
# is in sys.modules. A hand-rolled stub here is inevitably incomplete and, since
# setdefault installs it before real httpx loads, would poison a combined pytest
# run -- routes/inference references httpx.Response (and other attrs) at def time.
import httpx  # noqa: F401

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
    # The diffusion runner is mode-agnostic (always "auto"), so a standing manual
    # preference must not force a needless reload.
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


@pytest.mark.parametrize("bad", [[0, 0], [-1, 2], [float("inf"), 1], [float("nan"), 1]])
def test_load_request_rejects_degenerate_tensor_split(bad):
    # A negative/non-finite/all-zero split is dropped at launch but compared raw
    # in the reload dedupe, so it would reload forever -- reject it up front.
    with pytest.raises(ValueError):
        LoadRequest(model_path = "owner/repo", tensor_split = bad)


@pytest.mark.parametrize("good", [[2, 1], [1, 1], [], None])
def test_load_request_accepts_valid_tensor_split(good):
    assert LoadRequest(model_path = "owner/repo", tensor_split = good).tensor_split == good


def test_route_normalizes_explicit_extras_before_reload_dedupe():
    route_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    load_impl = route_src[route_src.index("async def _load_model_impl") :]
    strip = load_impl.index("_stripped_explicit = strip_shadowing_flags")
    normalize = load_impl.index(
        'request = request.model_copy(update = {"llama_extra_args": extra_llama_args})'
    )
    dedupe = load_impl.index("and _request_matches_loaded_settings(")
    assert strip < normalize < dedupe


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
    # 0 for a dense model (hides the slider); block_count for all-MoE;
    # block_count - leading_dense otherwise (GLM-4.7-Flash: 47 - 1 -> 46).
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
    # request value must not reload -- only a gpu_layers change (Auto -> pinned) does.
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
    # A count requested on a dense model is never emitted, so it must also be
    # dropped from the recorded state -- else /status and /load report a count
    # llama-server never received (same rule as the tensor-split drop below).
    moe_emit = src.find('cmd.extend(["--n-cpu-moe", str(moe_flag)])')
    assert "elif n_cpu_moe:" in src[moe_emit : moe_emit + 300]
    assert "self._n_cpu_moe = 0" in src[moe_emit : moe_emit + 300]
    # The offload path forces use_fit False so --fit-ctx is never added under --fit off.
    emit = src.find('cmd.extend(["--gpu-layers", str(gpu_layers), "--fit", "off"])')
    assert "use_fit = False" in src[src.rfind("\n", 0, emit) - 200 : emit + 80]


def test_status_reports_requested_context_length():
    # The hydration path re-seeds a Manual+Auto context pin from the REQUESTED
    # n_ctx (0 = Auto); context_length only exposes the resolved value.
    assert "requested_context_length" in InferenceStatusResponse.model_fields
    s = InferenceStatusResponse(requested_context_length = 8192)
    assert s.model_dump()["requested_context_length"] == 8192
    assert InferenceStatusResponse().model_dump()["requested_context_length"] is None
    # The /status route must actually wire it from the backend (a declared-but-
    # never-populated field would leave hydration silently reverting the pin).
    from pathlib import Path as _P

    route_src = (_P(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    assert "requested_context_length = llama_backend.requested_n_ctx" in route_src


def test_manual_offload_emits_tensor_split():
    # The offload path emits --tensor-split from the per-GPU shares, only when
    # provided, with >1 GPU in use, AND matching that count (a stale ratio on a
    # narrowed picker or a mismatched direct-API list must not emit -- llama-
    # server aborts on a split/GPU-count mismatch).
    src = _load_model_source()
    assert "if tensor_split and _split_gpus > 1:" in src
    # Emit only on a length match AND a positive sanitized total: a mismatched
    # or all-zero split aborts llama-server / assigns nothing, so it's dropped.
    # The emitted list is the sanitized one (clamping tested behaviorally below).
    assert "_sanitized_split = self._sanitize_tensor_split(tensor_split)" in src
    assert "if len(_sanitized_split) == _split_gpus and _split_total > 0:" in src
    assert '"--tensor-split"' in src
    # Joined as a comma list (e.g. "2,1") within the explicit-offload cmd branch.
    gate = src.find('if gpu_memory_mode == "manual" and gpu_layers >= 0:')
    nxt = src.find("elif use_fit:", gate)
    assert '","' in src[gate:nxt] and "tensor_split" in src[gate:nxt]
    # A split with a single effective GPU is never emitted, so it must also be
    # dropped from the recorded state -- else /status and /load report a ratio
    # llama-server never received and the dedupe baseline preserves it.
    assert "elif tensor_split:" in src[gate:nxt]
    drop = src.find("elif tensor_split:", gate, nxt)
    assert "self._tensor_split = None" in src[drop : drop + 250]


def test_sanitize_tensor_split_clamps_negative_and_non_finite():
    # Negative entries would launch a placement different from the ratio the
    # UI showed; inf passes a plain > 0 total gate and would emit
    # "--tensor-split inf,..." (llama.cpp normalizes shares by the running
    # total, so an inf poisons the shares from that entry on). Both clamp to 0.
    sanitize = LlamaCppBackend._sanitize_tensor_split
    assert sanitize([2, 1]) == [2.0, 1.0]
    assert sanitize([-1, 2]) == [0.0, 2.0]
    assert sanitize([float("inf"), 1]) == [0.0, 1.0]
    assert sanitize([float("nan"), 1]) == [0.0, 1.0]
    # All-zero survives sanitization; the call site's total gate drops it.
    assert sanitize([0, 0]) == [0.0, 0.0]
    # Unreadable input -> []; the call site's length gate drops it.
    assert sanitize(["x", 1]) == []
    assert sanitize([10**400, 1]) == []


def test_zero_offload_mask_honors_device_pin_spellings():
    # A user device pin must keep the GPUs visible: llama-server aborts on a
    # pin it can't see ('error: invalid device'). The pin can arrive as
    # --device or its -dev alias, as the draft forms (parsed even with no
    # drafter loaded), or as an inherited LLAMA_ARG_DEVICE env var.
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
    # Clamp the requested MoE-layer count to the model's MoE layers, then offset
    # past leading dense layers (--n-cpu-moe counts from layer 0).
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


def test_fit_sets_target_margin():
    # Manual + Auto (auto_fit) tightens the per-device VRAM margin to 512 MiB.
    caps = {"supports_fit_target": True}
    flags = LlamaCppBackend._ctx_integrity_flags(1, True, True, 0, 0, caps)
    assert flags[flags.index("--fit-target") + 1] == "512"
    # Not emitted on the legacy auto path (fit on but not auto_fit): -c 0 pins
    # native there, so the tighter margin must not ride along.
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(1, True, False, 0, 0, caps)
    # Not emitted when fit is off.
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(1, False, False, 0, 0, caps)
    # Not emitted when the binary lacks support.
    assert "--fit-target" not in LlamaCppBackend._ctx_integrity_flags(
        1, True, True, 0, 0, {"supports_fit_target": False}
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


def test_gpu_ids_reload_detection_collapses_diffusion_to_single_device():
    # The diffusion runner drives only its single lowest device, so the backend
    # records [lowest]. A later multi-GPU request that still resolves to that
    # same lowest device must dedupe (no needless reload); a request whose lowest
    # device moves, or that drops the pick, must reload.
    backend = _loaded_backend("auto")
    backend._is_diffusion = True
    backend._gpu_ids = [1]  # loaded on the lowest of an earlier [3, 1] pick
    assert _target_state_gpu_ids(backend, [3, 1]) is True
    assert _target_state_gpu_ids(backend, [1]) is True
    # Lowest device changes (2, not 1) -> reload.
    assert _target_state_gpu_ids(backend, [3, 2]) is False
    # Dropping the pick (auto) -> reload.
    assert _target_state_gpu_ids(backend, None) is False


def test_start_diffusion_server_resets_tensor_parallel():
    # A prior tensor-parallel chat load leaves self._tensor_parallel True (load_model
    # phase 1 only kills the process, it skips the unload reset). Diffusion is never
    # TP, so startup must clear it -- else /status misreports TP and an identical
    # diffusion re-Apply reloads against stale tensor-parallel state.
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend._start_diffusion_server)
    assert "self._tensor_parallel = False" in src


def test_route_matches_loaded_settings_collapses_diffusion_gpu_ids():
    # The route-level reload dedupe mirrors the backend: for a loaded diffusion
    # model it compares the request against the single recorded device, not the
    # full requested list, or a same-device multi-GPU pick reloads needlessly.
    route_src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text(encoding = "utf-8")
    match_impl = route_src[route_src.index("def _request_matches_loaded_settings") :]
    guard = match_impl.index("if llama_backend.is_diffusion:")
    collapse = match_impl.index("[sorted(request.gpu_ids)[0]] if request.gpu_ids else None")
    compare = match_impl.index("if _req_gpu_ids != llama_backend.gpu_ids:")
    assert guard < collapse < compare


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


def _rocm_torch_stub(monkeypatch):
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = "6.0")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def test_subset_pin_masks_via_rocr_on_rocm(monkeypatch):
    # A GPU-subset pin must exclude the rest at the ROCr/HSA layer: HIP masking
    # still enumerates every agent first, which segfaults the build on an
    # unsupported deselected GPU (e.g. a gfx1103 iGPU under a gfx110X prebuilt).
    # ROCR drops it at the driver layer; only one mask is set (HIP cleared).
    _rocm_torch_stub(monkeypatch)
    env = {"HIP_VISIBLE_DEVICES": "9"}  # stale/inherited HIP mask must not survive
    LlamaCppBackend._emit_child_gpu_visibility(env, "0", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_prefer_rocr_remaps_cuda_to_post_rocr_ordinals(monkeypatch):
    # ROCR re-indexes the visible agents from 0, and HIP (cleared here) falls back
    # to CUDA_VISIBLE_DEVICES -- so on the prefer_rocr path CUDA must carry the
    # post-ROCR ordinals, not the physical ids, else a non-zero pick indexes out
    # of range and the child sees no GPU and drops to CPU (#7272 review).
    _rocm_torch_stub(monkeypatch)
    # Single non-zero GPU: ROCR keeps the physical id, CUDA becomes ordinal 0.
    env = {}
    LlamaCppBackend._emit_child_gpu_visibility(env, "1", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in env
    # Multi-GPU subset: ROCR keeps the physical ids, CUDA is the 0-based ordinals.
    env = {}
    LlamaCppBackend._emit_child_gpu_visibility(env, "1,3", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "1,3"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_subset_pin_default_still_uses_hip_and_clears_rocr(monkeypatch):
    # Without prefer_rocr the masking is unchanged: HIP narrows, inherited ROCR
    # is cleared so the two can't double-mask.
    _rocm_torch_stub(monkeypatch)
    env = {"ROCR_VISIBLE_DEVICES": "0,1"}
    LlamaCppBackend._emit_child_gpu_visibility(env, "1")
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_cpu_only_pin_keeps_hip_even_with_prefer_rocr(monkeypatch):
    # The CPU-only sentinel never routes through ROCR (no portable "hide all"
    # spelling); it hides every GPU via HIP.
    _rocm_torch_stub(monkeypatch)
    env = {}
    LlamaCppBackend._emit_child_gpu_visibility(env, "-1", prefer_rocr = True)
    assert env["HIP_VISIBLE_DEVICES"] == "-1"
    assert "ROCR_VISIBLE_DEVICES" not in env


def _amd_sdk_torch_stub(monkeypatch):
    # AMD SDK wheel: torch.version.hip is None but __version__ encodes rocm.
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = None)
    torch_stub.__version__ = "2.9.1+rocm7.2.1"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def test_amd_sdk_wheel_hip_none_still_masks_rocr(monkeypatch):
    # An AMD SDK wheel leaves torch.version.hip unset but has "rocm" in __version__.
    # It must still get the ROCR mask, else only CUDA_VISIBLE_DEVICES is set and an
    # unsupported iGPU keeps enumerating and can crash llama-server.
    _amd_sdk_torch_stub(monkeypatch)
    env = {"HIP_VISIBLE_DEVICES": "9"}
    LlamaCppBackend._emit_child_gpu_visibility(env, "0", prefer_rocr = True)
    assert env["ROCR_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_cuda_wheel_hip_none_gets_no_rocm_mask(monkeypatch):
    # A CUDA wheel (hip=None, no "rocm" in __version__) must NOT get a HIP/ROCR mask
    # -- only CUDA_VISIBLE_DEVICES -- so the version-string check can't false-positive.
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
    # _resolve_visible_physical_ids must use the same ROCm detection as
    # _emit_child_gpu_visibility: on an AMD SDK wheel (hip=None, rocm in
    # __version__) an inherited ROCR mask IS the ordinal->physical mapping.
    # Reading it as "no mask" labels ordinal 0 as physical 0 and the child's
    # ROCR pin then re-exposes the GPU the mask was hiding (#7272 review).
    _amd_sdk_torch_stub(monkeypatch)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert LlamaCppBackend._resolve_visible_physical_ids() == [1]


def test_resolve_physical_ids_ignores_rocr_on_cuda_wheel(monkeypatch):
    # A CUDA wheel (hip=None, no "rocm") keeps CUDA-only semantics: a stray
    # ROCR var must not be read as the mask.
    torch_stub = _types.ModuleType("torch")
    torch_stub.version = _types.SimpleNamespace(hip = None)
    torch_stub.__version__ = "2.9.1+cu124"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert LlamaCppBackend._resolve_visible_physical_ids() is None


# ── Diffusion single-device selection ───────────────────────────────────────


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


# ── Deliberate zero-offload (manual gpu_layers=0): training-skip flag ─────────


def test_zero_offload_flag_false_without_companions():
    # CPU-only by construction: False lets training skip unloading a server that
    # holds no VRAM.
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
    # The env mask for CPU-only zero-offload loads keys off this scan: any
    # --mmproj form or a drafter (flag aliases / env) keeps the GPUs visible.
    has = LlamaCppBackend._cmd_has_gpu_companion
    assert has(["llama-server", "-m", "m.gguf"], {}) is False
    assert has(["llama-server", "--mmproj", "p.gguf"], {}) is True
    assert has(["llama-server", "--mmproj=p.gguf"], {}) is True
    assert has(["llama-server", "-md", "d.gguf"], {}) is True
    assert has(["llama-server"], {"LLAMA_ARG_SPEC_DRAFT_MODEL": "d.gguf"}) is True


def test_cmd_companion_ignores_cpu_forced_drafter():
    # A CPU-pinned drafter holds no VRAM: the zero-offload mask may hide the GPUs
    # and training may leave the server alone.
    has = LlamaCppBackend._cmd_has_gpu_companion
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-ngl", "0"]
    assert has(cmd, {}) is False
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-device", "cpu"]
    assert has(cmd, {}) is False
    # mmproj still counts even alongside a CPU drafter.
    cmd = ["llama-server", "-md", "d.gguf", "--spec-draft-ngl", "0", "--mmproj", "p.gguf"]
    assert has(cmd, {}) is True
