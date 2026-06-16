# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend contract for the Tensor Parallelism toggle.

The toggle threads a single ``tensor_parallel`` bool from the chat UI
through the load request to a ``--split-mode tensor`` llama-server flag,
and round-trips it back via the load/status responses so the switch
reflects what is actually running. These tests pin:

  * the pydantic request/response/status contract (snake_case key,
    default False),
  * the backend ``tensor_parallel`` property and its reset on unload,
  * the ``_already_in_target_state`` reload-detection branch, and
  * that ``--split-mode tensor`` is emitted only behind the toggle.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import threading
import time
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
from core.inference.llama_server_args import resolve_tensor_parallel
from core.inference.tensor_fallback import load_with_tensor_fallback
from models.inference import (
    InferenceStatusResponse,
    LoadRequest,
    LoadResponse,
)


# ── Pydantic contract (snake_case key, default False) ────────────────


def test_load_request_defaults_tensor_parallel_false():
    req = LoadRequest(model_path = "owner/repo")
    assert req.tensor_parallel is False


def test_load_request_accepts_tensor_parallel():
    req = LoadRequest(model_path = "owner/repo", tensor_parallel = True)
    assert req.tensor_parallel is True


def test_load_request_round_trips_json_key():
    # The frontend sends the snake_case key verbatim.
    req = LoadRequest.model_validate({"model_path": "owner/repo", "tensor_parallel": True})
    assert req.tensor_parallel is True
    assert req.model_dump()["tensor_parallel"] is True


@pytest.mark.parametrize("model_cls", [LoadResponse, InferenceStatusResponse])
def test_response_models_emit_tensor_parallel(model_cls):
    # Default False, and the key is always present in the JSON body.
    if model_cls is LoadResponse:
        default = model_cls(
            status = "loaded",
            model = "owner/repo",
            display_name = "repo",
            inference = {},
        )
        on = model_cls(
            status = "loaded",
            model = "owner/repo",
            display_name = "repo",
            inference = {},
            tensor_parallel = True,
        )
    else:
        default = model_cls()
        on = model_cls(tensor_parallel = True)
    assert default.model_dump()["tensor_parallel"] is False
    assert on.model_dump()["tensor_parallel"] is True


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


def test_tensor_parallel_property_defaults_false():
    assert LlamaCppBackend().tensor_parallel is False


def test_tensor_parallel_property_reflects_field():
    backend = LlamaCppBackend()
    backend._tensor_parallel = True
    assert backend.tensor_parallel is True


def test_unload_resets_tensor_parallel():
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._tensor_parallel = True
    backend.unload_model()
    assert backend.tensor_parallel is False


# ── _already_in_target_state reload-detection branch ─────────────────


def _loaded_backend(tensor_parallel: bool) -> LlamaCppBackend:
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
    backend._tensor_parallel = tensor_parallel
    return backend


def _target_state(backend: LlamaCppBackend, tensor_parallel: bool) -> bool:
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
        tensor_parallel = tensor_parallel,
    )


@pytest.mark.parametrize("flag", [True, False])
def test_already_in_target_state_matches_same_tensor_parallel(flag):
    assert _target_state(_loaded_backend(flag), flag) is True


@pytest.mark.parametrize(
    "loaded,requested",
    [(False, True), (True, False)],
)
def test_already_in_target_state_reloads_on_tensor_parallel_change(loaded, requested):
    # Flipping the toggle either direction must force a reload so the
    # command is rebuilt with/without --split-mode tensor.
    assert _target_state(_loaded_backend(loaded), requested) is False


def test_already_in_target_state_reconciles_split_mode_extras():
    # Tensor engaged via --split-mode in extras (boolean omitted/default False)
    # must match a server already running tensor mode -- no spurious reload.
    backend = _loaded_backend(tensor_parallel = True)
    backend._extra_args = ["--split-mode", "tensor"]
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "owner/repo",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "auto",
            chat_template_override = None,
            extra_args = ["--split-mode", "tensor"],
            is_vision = False,
            tensor_parallel = False,
        )
        is True
    )


# ── --split-mode tensor is emitted only behind the toggle ────────────


def _load_model_source() -> str:
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def test_split_mode_tensor_is_gated_on_the_toggle():
    src = _load_model_source()
    assert (
        'cmd.extend(["--split-mode", "tensor"])' in src
    ), "the tensor-parallel flag emission must be present in load_model"
    # The emission lives behind `if tensor_parallel:` -- it must never be
    # part of the unconditional base cmd list.
    base_start = src.find("cmd = [")
    base_end = src.find("\n                ]", base_start)
    base_block = src[base_start:base_end] if base_end > base_start else ""
    assert (
        "--split-mode" not in base_block
    ), "--split-mode must be conditional, not in the base cmd list"
    gate = src.find("if tensor_parallel:")
    emit = src.find('cmd.extend(["--split-mode", "tensor"])')
    assert 0 <= gate < emit, "emission must sit under `if tensor_parallel:`"


def test_proportional_tensor_split_is_emitted_in_tensor_mode():
    # Asymmetric GPUs (e.g. 48 GB + 24 GB) OOM the smaller card under the
    # even default; the allocator weights --tensor-split by free VRAM. Pin
    # that the flag is emitted from inside the tensor-parallel block.
    src = _load_model_source()
    assert '"--tensor-split"' in src
    gate = src.find("if tensor_parallel:")
    ts = src.find('"--tensor-split"')
    nxt_else = src.find("self._tensor_parallel = False")
    assert 0 <= gate < ts < nxt_else, "--tensor-split must be emitted under `if tensor_parallel:`"


def test_mtp_decode_probe_wired_under_tensor_parallel():
    # MTP-draft can pass /health and crash the CUDA FA kernel only on the first
    # decode under --split-mode tensor. Rather than statically banning MTP+TP
    # (which a future llama.cpp may support), load_model probes a decode and
    # routes a failure into the existing MTP-drop fallback.
    src = _load_model_source()
    probe = src.find("_probe_mtp_decode()")
    assert probe != -1, "load_model must decode-probe MTP under tensor parallelism"
    # Gated on tensor mode AND an MTP request (ordinary MTP loads stay unprobed).
    guard = src[max(0, probe - 400) : probe]
    assert "self._tensor_parallel" in guard and "_spec_requested_mtp" in guard
    # A failed probe flips healthy so the shared MTP-drop fallback fires.
    assert "healthy = False" in src[probe : probe + 400]
    fallback = src.find("if not healthy and _spec_requested_mtp")
    assert 0 <= probe < fallback, "the probe must precede the MTP-drop fallback"


def test_probe_mtp_decode_returns_false_on_crash(monkeypatch):
    # The probe is the decode-time health gate: True only on a clean 200 from a
    # live server; any error (dropped connection, non-200, dead process) is a
    # failed probe so the caller drops MTP and retries.
    backend = LlamaCppBackend()
    backend._port = 0

    class _Resp:
        def __init__(self, code):
            self.status_code = code

    backend._process = None  # liveness check skipped; exercise the HTTP result
    monkeypatch.setattr(llama_cpp_module.httpx, "post", lambda *a, **k: _Resp(200), raising = False)
    assert backend._probe_mtp_decode(timeout = 1.0) is True

    def _drop(*a, **k):
        raise llama_cpp_module.httpx.RemoteProtocolError("peer closed connection")

    monkeypatch.setattr(llama_cpp_module.httpx, "post", _drop, raising = False)
    assert backend._probe_mtp_decode(timeout = 1.0) is False

    monkeypatch.setattr(llama_cpp_module.httpx, "post", lambda *a, **k: _Resp(500), raising = False)
    assert backend._probe_mtp_decode(timeout = 1.0) is False

    # 200 but the server aborted right after (poll() returns an exit code).
    backend._process = _FakeProcess()
    monkeypatch.setattr(llama_cpp_module.httpx, "post", lambda *a, **k: _Resp(200), raising = False)
    assert backend._probe_mtp_decode(timeout = 1.0) is False


# ── generation-time MTP recovery (mid-stream crash) ──────────────────


def _recovery_backend() -> LlamaCppBackend:
    # A backend that loaded MTP under tensor parallelism and whose server has
    # since exited (the _FakeProcess poll() returns 0 -> a dead subprocess).
    b = LlamaCppBackend()
    b._tensor_parallel = True
    b._speculative_type = "draft-mtp"
    b._process = _FakeProcess()
    b._last_load_kwargs = {
        "model_identifier": "owner/repo",
        "tensor_parallel": True,
        "speculative_type": "auto",
        "n_parallel": 4,
    }
    return b


def test_generate_chat_completion_wires_runtime_recovery():
    # The non-tool generation path must route a mid-stream server death into the
    # recovery helper (the tool + passthrough paths do so from the routes).
    src = inspect.getsource(LlamaCppBackend.generate_chat_completion)
    assert "_maybe_recover_from_mtp_crash" in src


def test_runtime_recovery_reloads_without_mtp(monkeypatch):
    # A dead server + tensor + resolved-MTP + snapshot -> one background reload
    # with speculative_type="off" (rest of the snapshot preserved), then
    # spec_fallback_reason="runtime_error" and the single-flight flag released.
    b = _recovery_backend()
    done = threading.Event()
    captured = {}

    def _fake_load_model(**kwargs):
        captured.update(kwargs)
        done.set()
        return True

    monkeypatch.setattr(b, "load_model", _fake_load_model)
    assert b._maybe_recover_from_mtp_crash(RuntimeError("peer closed")) is True
    assert done.wait(timeout = 5)
    assert captured["speculative_type"] == "off"
    assert captured["model_identifier"] == "owner/repo"
    assert captured["n_parallel"] == 4  # snapshot replayed faithfully
    deadline = time.monotonic() + 2
    while b._spec_fallback_reason != "runtime_error" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert b._spec_fallback_reason == "runtime_error"
    assert b._mtp_runtime_fallback_in_progress is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: setattr(b, "_tensor_parallel", False),
        lambda b: setattr(b, "_speculative_type", "ngram-mod"),
        lambda b: setattr(b, "_last_load_kwargs", None),
        lambda b: setattr(b, "_process", None),
        lambda b: b._cancel_event.set(),
    ],
)
def test_runtime_recovery_skips_when_not_applicable(monkeypatch, mutate):
    # No reload when tensor is off, the resolved spec was not MTP, there is no
    # snapshot, the process handle is gone, or the request was cancelled.
    b = _recovery_backend()
    mutate(b)
    calls = []
    monkeypatch.setattr(b, "load_model", lambda **k: calls.append(k))
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is False
    assert calls == []


def test_runtime_recovery_is_single_flight(monkeypatch):
    # Concurrent failures schedule only one reload.
    b = _recovery_backend()
    started = threading.Event()
    release = threading.Event()

    def _slow_load(**kwargs):
        started.set()
        release.wait(timeout = 5)
        return True

    monkeypatch.setattr(b, "load_model", _slow_load)
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is True
    assert started.wait(timeout = 5)
    # Second failure while the first reload is in flight is a no-op.
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is False
    release.set()


def test_runtime_recovery_rechecks_cancel_before_reload():
    # A reload scheduled just before an /unload must not resurrect the model:
    # the background recover() re-checks the cancel flag after the death poll
    # (load_model would otherwise clear the cancel and reload).
    src = inspect.getsource(LlamaCppBackend._maybe_recover_from_mtp_crash)
    cancel = src.rfind("self._cancel_event.is_set()")
    load = src.find("self.load_model(")
    assert 0 <= cancel < load, "recovery must re-check cancel before reloading"


def test_probe_mtp_decode_uses_api_key_auth(monkeypatch):
    # Direct-stream mode runs llama-server with --api-key; the probe must send
    # the same bearer auth or it gets a spurious 401 and falsely drops MTP.
    backend = LlamaCppBackend()
    backend._port = 0
    backend._process = None
    captured = {}

    class _Resp:
        status_code = 200

    def _capture(*a, **k):
        captured.clear()
        captured.update(k)
        return _Resp()

    monkeypatch.setattr(llama_cpp_module.httpx, "post", _capture, raising = False)
    backend._api_key = "secret"
    backend._probe_mtp_decode(timeout = 1.0)
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    backend._api_key = None
    backend._probe_mtp_decode(timeout = 1.0)
    assert captured["headers"] is None


class _ToggleProcess:
    """A subprocess stand-in whose liveness can be flipped at runtime."""

    def __init__(self):
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout = None):
        self._alive = False
        return 0

    def die(self):
        self._alive = False


def test_crash_watchdog_triggers_recovery_on_death(monkeypatch):
    # With MTP + tensor active, the watchdog must notice the server process exit
    # and route it into recovery even when no request handler observed it (the
    # direct llama-server proxy endpoints never call the helper themselves).
    b = _recovery_backend()
    proc = _ToggleProcess()
    b._process = proc
    fired = threading.Event()
    monkeypatch.setattr(b, "_maybe_recover_from_mtp_crash", lambda *a, **k: fired.set())
    b._start_mtp_crash_watchdog()
    assert b._mtp_watchdog_thread is not None
    proc.die()
    assert fired.wait(timeout = 3)


def test_crash_watchdog_ignores_intentional_termination(monkeypatch):
    # A planned reload/unload stops the watchdog before killing the process, so
    # the resulting death must not be mistaken for a crash.
    b = _recovery_backend()
    proc = _ToggleProcess()
    b._process = proc
    fired = threading.Event()
    monkeypatch.setattr(b, "_maybe_recover_from_mtp_crash", lambda *a, **k: fired.set())
    b._start_mtp_crash_watchdog()
    b._stop_mtp_crash_watchdog()  # what _kill_process does first
    proc.die()
    assert not fired.wait(timeout = 2)
    assert b._mtp_watchdog_thread is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: setattr(b, "_tensor_parallel", False),
        lambda b: setattr(b, "_speculative_type", "ngram-mod"),
        lambda b: setattr(b, "_process", None),
    ],
)
def test_crash_watchdog_not_armed_when_inapplicable(mutate):
    # Only a Studio-managed MTP + tensor load with a live process arms it.
    b = _recovery_backend()
    b._process = _ToggleProcess()
    mutate(b)
    b._start_mtp_crash_watchdog()
    assert b._mtp_watchdog_thread is None


def test_kill_process_stops_crash_watchdog(monkeypatch):
    # _kill_process is the single deliberate-termination chokepoint; it must
    # stop the watchdog so the planned kill isn't seen as a crash.
    b = _recovery_backend()
    proc = _ToggleProcess()
    b._process = proc
    fired = threading.Event()
    monkeypatch.setattr(b, "_maybe_recover_from_mtp_crash", lambda *a, **k: fired.set())
    b._start_mtp_crash_watchdog()
    b._kill_process()
    assert b._mtp_watchdog_thread is None
    assert b._process is None
    assert not fired.wait(timeout = 2)


def test_kill_process_stops_watchdog_before_terminate():
    # Ordering matters: stop the watchdog before terminating so the watchdog's
    # post-death stop re-check reliably sees a planned kill.
    src = inspect.getsource(LlamaCppBackend._kill_process)
    stop = src.find("_stop_mtp_crash_watchdog()")
    term = src.find(".terminate(")
    assert 0 <= stop < term, "must stop the watchdog before terminating"


def test_crash_watchdog_rechecks_stop_before_recovery():
    # After a detected exit the watchdog re-checks the stop flag so a kill that
    # raced in between the poll-wait and the poll-read can't fire recovery.
    src = inspect.getsource(LlamaCppBackend._start_mtp_crash_watchdog)
    check = src.find("stop.is_set()")
    recover = src.find("_maybe_recover_from_mtp_crash")
    assert 0 <= check < recover, "must re-check stop before recovering"


def test_load_model_arms_crash_watchdog():
    # The healthy-load commit arms the watchdog for this load.
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "_start_mtp_crash_watchdog" in src


# ── tensor-mode allocation: conservative VRAM budget ─────────────────


def _kv_seeded_backend() -> LlamaCppBackend:
    # Minimal GGUF metadata so _can_estimate_kv() is True (legacy KV path).
    backend = LlamaCppBackend()
    backend._n_layers = 32
    backend._embedding_length = 4096
    backend._n_heads = 32
    backend._n_kv_heads = 8
    backend._context_length = 131072
    return backend


def test_fit_context_budget_frac_override_is_tighter():
    backend = _kv_seeded_backend()
    model_size = 8 * 1024**3
    pool_mib = 24 * 1024  # tight enough that KV capping bites

    fit_default = backend._fit_context_to_vram(131072, pool_mib, model_size, "f16")
    fit_tp = backend._fit_context_to_vram(131072, pool_mib, model_size, "f16", budget_frac = 0.80)
    assert fit_tp < 131072, "expected the context to be capped at this VRAM tier"
    assert fit_tp <= fit_default, "a tighter budget must not allow MORE context"
    # Omitting the override must reproduce the default budget exactly.
    assert backend._fit_context_to_vram(131072, pool_mib, model_size, "f16") == fit_default


# ── unsupported-arch load failure -> clean message ───────────────────


def test_split_mode_tensor_arch_failure_message():
    msg = LlamaCppBackend._classify_llama_start_failure(
        "llama_model_create: LLAMA_SPLIT_MODE_TENSOR not implemented for "
        "architecture 'deepseek2'",
        None,
        "unsloth/DeepSeek-V3-GGUF",
    )
    assert "Tensor parallelism is not supported" in msg


def test_unrelated_arch_failure_not_hijacked_by_tensor_message():
    msg = LlamaCppBackend._classify_llama_start_failure(
        "unknown model architecture: 'flux'", "/models/flux.gguf", None
    )
    assert "Tensor parallelism" not in msg


# ── _plan_tensor_parallel: the allocation math (pure, no model/GPU) ───
# Seeded full-attention KV (~128 KiB/token) via _kv_seeded_backend, so the
# context cap + split are deterministic. Asserts relationships rather than
# magic numbers so the KV estimate can evolve without breaking these.

_GB = 1024**3
_ASYM = [(0, 48000), (1, 24000)]  # asymmetric pool, 72000 MiB
_SYM = [(0, 24000), (1, 24000)]  # symmetric pool


def _plan(
    model_gb,
    target = 131072,
    gpus = _ASYM,
    mtp = False,
):
    b = _kv_seeded_backend()
    return b, b._plan_tensor_parallel(gpus, int(model_gb * _GB), target, mtp_engaged = mtp)


def _kv_budget_b(model_gb, gpus = _ASYM):
    reserve = LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
    return (sum(f for _, f in gpus) - len(gpus) * reserve) * 1024 * 1024 - int(model_gb * _GB)


def test_tp_plan_weighted_split_on_asymmetric_big_model():
    b, (ec, mac, gi, ts) = _plan(50)
    reserve = b._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
    assert gi == [0, 1]
    # split weighted by (free - buffer), not raw free
    assert ts == [48000 - reserve, 24000 - reserve]
    assert ec < 131072  # capped below native


def test_tp_plan_even_split_when_model_fits():
    # A small model whose even share fits the smallest GPU -> llama.cpp's even
    # default (None), which is safe for archs that crash on a weighted split.
    _, (ec, mac, gi, ts) = _plan(4)
    assert ts is None


def test_tp_plan_symmetric_gpus_use_even_split():
    _, (ec, mac, gi, ts) = _plan(8, gpus = _SYM)
    assert ts is None


def test_tp_plan_context_fits_pool_budget_no_oom():
    b, (ec, mac, gi, ts) = _plan(50)
    # the chosen context's KV must fit the pooled budget (weights + buffers)
    assert b._estimate_kv_cache_bytes(ec) <= _kv_budget_b(50)


def test_tp_plan_uses_available_vram_not_wasteful():
    # when the cap engages, the chosen context nearly fills the budget
    b, (ec, mac, gi, ts) = _plan(50)
    assert b._estimate_kv_cache_bytes(ec) >= 0.9 * _kv_budget_b(50)


def test_tp_plan_weights_exceed_pool_floors_context():
    # 70 GB > pool minus per-GPU reserves -> floor (triggers layer fallback)
    _, (ec, mac, gi, ts) = _plan(70)
    assert ec == 2048


def test_tp_plan_floor_never_exceeds_explicit_small_context():
    # An explicit context below the 2048 floor must not be raised: a caller
    # asking for 1024 should not have KV sized for 2048 (avoidable OOM).
    _, (ec, mac, gi, ts) = _plan(70, target = 1024)  # weights exceed pool -> floor path
    assert ec == 1024
    _, (ec2, *_rest) = _plan(50, target = 1024)  # cap path with a tiny budget
    assert ec2 <= 1024


def test_tp_plan_explicit_context_honored_when_it_fits():
    _, (ec, mac, gi, ts) = _plan(50, target = 8192)
    assert ec == 8192


def test_tp_plan_explicit_context_capped_when_too_large():
    _, (ec, mac, gi, ts) = _plan(50, target = 131072)
    assert 2048 <= ec < 131072


def test_tp_plan_max_available_ctx_reports_native_not_explicit_ctx():
    # An explicit small ctx caps effective_ctx but the UI ceiling
    # (max_available_ctx) must reflect the native/hardware cap, not the request.
    b = _kv_seeded_backend()
    ec, mac, _gi, _ts = b._plan_tensor_parallel(_ASYM, int(50 * _GB), 8192, max_target_ctx = 131072)
    _, native_mac, *_ = b._plan_tensor_parallel(_ASYM, int(50 * _GB), 131072)
    assert ec == 8192  # explicit request honored for the load
    assert mac == native_mac > ec  # ceiling reflects the hardware cap


def test_tp_plan_mtp_reserves_extra_and_shrinks_context():
    _, (ec_no, *_rest) = _plan(50)
    _, (ec_mtp, *_rest) = _plan(50, mtp = True)
    assert ec_mtp < ec_no


def test_tp_plan_no_kv_metadata_floors_context():
    b = LlamaCppBackend()  # no KV metadata -> can't size safely
    ec, mac, gi, ts = b._plan_tensor_parallel(_ASYM, int(50 * _GB), 131072)
    assert ec <= 4096


def test_tp_plan_single_gpu_never_splits():
    # The toggle is a no-op without >= 2 GPUs (most dev/CI machines). Even if
    # the planner is reached, it must not emit a tensor split.
    b = _kv_seeded_backend()
    ec, mac, gi, ts = b._plan_tensor_parallel([(0, 24000)], int(8 * _GB), 8192)
    assert ts is None
    assert gi == [0]


def test_tp_plan_zero_gpus_never_splits():
    b = _kv_seeded_backend()
    ec, mac, gi, ts = b._plan_tensor_parallel([], int(8 * _GB), 8192)
    assert ts is None
    assert gi == []


def test_tp_plan_drops_gpu_below_buffer_reserve():
    # A GPU with less free VRAM than the per-device compute-buffer reserve
    # can't host tensor mode; it's excluded, which here leaves <2 usable -> no
    # split (and gpu_indices reflects only the usable device).
    b = _kv_seeded_backend()
    reserve = LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
    ec, mac, gi, ts = b._plan_tensor_parallel([(0, 48000), (1, reserve - 1)], int(8 * _GB), 8192)
    assert gi == [0]
    assert ts is None


# ── route auto-fallback survives a *raised* tensor-load crash ─────────
# A tensor-incompatible model makes load_model RAISE (Gemma 3n aborts) rather
# than return False. The /load fallback helper must catch that and retry with
# layer split -- stripping any --split-mode from the extras so the retry can't
# relaunch tensor -- while a non-tensor load propagates its exception. These
# exercise the real helper with a fake loader (no GPU, no llama-server).


class _RecordingLoader:
    """Fake ``attempt_load``: crashes whenever tensor mode is effectively
    engaged (via the bool or a ``--split-mode`` in extras), like a real
    tensor-incompatible model; succeeds on layer split."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def __call__(self, tensor_parallel, extra_args):
        self.calls.append((tensor_parallel, list(extra_args) if extra_args else extra_args))
        if resolve_tensor_parallel(extra_args, tensor_parallel):
            raise RuntimeError("llama-server failed to start")
        return True


def test_tensor_fallback_retries_layer_on_crash():
    loader = _RecordingLoader()
    ok = asyncio.run(
        load_with_tensor_fallback(loader, requested_tensor = True, extra_args = None, label = "m")
    )
    assert ok is True
    # tensor first (crashes), then layer split.
    assert [c[0] for c in loader.calls] == [True, False]


def test_tensor_fallback_no_retry_on_success():
    calls: list[bool] = []

    async def _ok(tensor_parallel, extra_args):
        calls.append(tensor_parallel)
        return True

    ok = asyncio.run(
        load_with_tensor_fallback(_ok, requested_tensor = True, extra_args = None, label = "m")
    )
    assert ok is True
    assert calls == [True]  # no fallback when the tensor load succeeds


def test_tensor_fallback_retries_when_tensor_returns_false():
    # load_model can signal failure by *returning False* (not only by raising);
    # that must trigger the layer-split retry just like a crash does.
    calls: list[bool] = []

    async def _false_on_tensor(tensor_parallel, extra_args):
        calls.append(tensor_parallel)
        return not resolve_tensor_parallel(extra_args, tensor_parallel)

    ok = asyncio.run(
        load_with_tensor_fallback(
            _false_on_tensor, requested_tensor = True, extra_args = None, label = "m"
        )
    )
    assert ok is True
    assert calls == [True, False]


def test_tensor_fallback_returns_false_when_both_attempts_fail():
    # Tensor fails and the layer retry also fails -> the helper returns False so
    # the route raises its own HTTP 500 (it does not crash mid-flight).
    calls: list[bool] = []

    async def _always_false(tensor_parallel, extra_args):
        calls.append(tensor_parallel)
        return False

    ok = asyncio.run(
        load_with_tensor_fallback(_always_false, requested_tensor = True, extra_args = None, label = "m")
    )
    assert ok is False
    assert calls == [True, False]  # tried tensor, then layer split


def test_tensor_fallback_skips_layer_retry_when_cancelled():
    # load_model returns False on a user cancellation too. When cancelled() is
    # True, the helper must NOT relaunch the load the user just cancelled.
    calls: list[bool] = []

    async def _false_on_tensor(tensor_parallel, extra_args):
        calls.append(tensor_parallel)
        return False

    ok = asyncio.run(
        load_with_tensor_fallback(
            _false_on_tensor,
            requested_tensor = True,
            extra_args = None,
            label = "m",
            cancelled = lambda: True,
        )
    )
    assert ok is False
    assert calls == [True]  # no layer-split retry after cancellation


@pytest.mark.parametrize(
    "extras",
    [
        ["--split-mode", "tensor", "-c", "4096"],
        ["-sm", "tensor", "-c", "4096"],
        ["--split-mode=tensor", "-c", "4096"],
        ["-sm=tensor", "-c", "4096"],
    ],
)
def test_tensor_fallback_strips_split_mode_from_extras_on_retry(extras):
    # Tensor engaged via extras (boolean False); the retry must drop every
    # --split-mode form (long/short, space/=) but keep the user's other flags,
    # else resolve_tensor_parallel re-enables tensor and relaunches the crash.
    loader = _RecordingLoader()
    ok = asyncio.run(
        load_with_tensor_fallback(loader, requested_tensor = False, extra_args = extras, label = "m")
    )
    assert ok is True
    assert len(loader.calls) == 2
    assert loader.calls[1][1] == ["-c", "4096"]  # split-mode stripped, -c kept


def test_tensor_fallback_propagates_non_tensor_crash():
    async def _always_raise(tensor_parallel, extra_args):
        raise RuntimeError("bad model")

    with pytest.raises(RuntimeError, match = "bad model"):
        asyncio.run(
            load_with_tensor_fallback(
                _always_raise, requested_tensor = False, extra_args = None, label = "m"
            )
        )
