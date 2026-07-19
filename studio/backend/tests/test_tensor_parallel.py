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
from core.inference.llama_cpp import _CTX_FIT_VRAM_FRACTION, LlamaCppBackend
from core.inference.llama_server_args import (
    _effective_tensor_parallel,
    resolve_tensor_parallel,
)
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
    # (which a future llama.cpp may support), load_model probes a decode; a hard
    # fault retries --flash-attn off first, else routes into the MTP-drop fallback.
    src = _load_model_source()
    probe = src.find("_probe_mtp_decode()")
    assert probe != -1, "load_model must decode-probe MTP under tensor parallelism"
    # Gated on tensor mode AND an MTP request (ordinary MTP loads stay unprobed).
    guard = src[max(0, probe - 400) : probe]
    assert "self._tensor_parallel" in guard and "_spec_requested_mtp" in guard
    # A hard fault retries FA-off (keeps MTP) before flipping healthy so the
    # shared MTP-drop fallback fires.
    after = src[probe : probe + 900]
    assert "_with_flash_attn_off" in after and "healthy = False" in after
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
    b._mtp_runtime_fallback_active = True
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
    # One background reload with speculative_type="off" (rest of snapshot kept),
    # then spec_fallback_reason="runtime_error" and single-flight released.
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
    # The reload thread clears the single-flight flag in its finally, a beat after
    # it sets the fallback reason -- wait for that instead of racing the thread.
    deadline = time.monotonic() + 2
    while b._mtp_runtime_fallback_in_progress and time.monotonic() < deadline:
        time.sleep(0.02)
    assert b._mtp_runtime_fallback_in_progress is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: setattr(b, "_mtp_runtime_fallback_active", False),
        lambda b: setattr(b, "_last_load_kwargs", None),
        lambda b: setattr(b, "_process", None),
        lambda b: b._cancel_event.set(),
    ],
)
def test_runtime_recovery_skips_when_not_applicable(monkeypatch, mutate):
    # No reload when this launch is not running MTP+tensor, there is no snapshot,
    # the process handle is gone, or the request was cancelled.
    b = _recovery_backend()
    mutate(b)
    calls = []
    monkeypatch.setattr(b, "load_model", lambda **k: calls.append(k))
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is False
    assert calls == []


class _BlockingDeadProc:
    # Reports alive until released, then dead -- lets a test mutate backend state
    # while the recovery thread is still in its death-confirm poll.
    def __init__(self):
        self._dead = threading.Event()

    def poll(self):
        return 0 if self._dead.is_set() else None

    def terminate(self):
        self._dead.set()

    def kill(self):
        self._dead.set()

    def wait(self, timeout = None):
        self._dead.set()
        return 0

    def release(self):
        self._dead.set()


def test_runtime_recovery_fires_for_user_env_mtp(monkeypatch):
    # MTP driven by user extra_args / LLAMA_ARG_SPEC_TYPE leaves _speculative_type
    # unset, but the launch flag still gates recovery on (pass-through MTP).
    b = _recovery_backend()
    b._speculative_type = None  # Unsloth stepped back; user/env owns the spec
    done = threading.Event()
    captured = {}

    def _fake_load_model(**kwargs):
        captured.update(kwargs)
        done.set()
        return True

    monkeypatch.setattr(b, "load_model", _fake_load_model)
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is True
    assert done.wait(timeout = 5)
    assert captured["speculative_type"] == "off"


def test_runtime_recovery_strips_user_mtp_extra_args(monkeypatch):
    # A user --spec-type draft-mtp in extra_args must be neutralised on the reload
    # (append a last-wins --spec-default) so MTP can't re-engage and loop.
    b = _recovery_backend()
    b._last_load_kwargs = dict(b._last_load_kwargs, extra_args = ["--spec-type", "draft-mtp"])
    done = threading.Event()
    captured = {}

    def _fake_load_model(**kwargs):
        captured.update(kwargs)
        done.set()
        return True

    monkeypatch.setattr(b, "load_model", _fake_load_model)
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is True
    assert done.wait(timeout = 5)
    assert captured["speculative_type"] == "off"
    assert captured["extra_args"][-1] == "--spec-default"


def test_runtime_recovery_restores_requested_mode(monkeypatch):
    # After the off-reload, /status must show the user's requested mode + the
    # runtime-error note, not a bare "off" (matches the startup MTP fallback).
    b = _recovery_backend()
    b._last_load_kwargs = dict(b._last_load_kwargs, speculative_type = "mtp")
    done = threading.Event()

    def _fake_load_model(**kwargs):
        b._requested_spec_mode = "off"  # what a real off-reload would leave behind
        done.set()
        return True

    monkeypatch.setattr(b, "load_model", _fake_load_model)
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is True
    assert done.wait(timeout = 5)
    deadline = time.monotonic() + 2
    while b._requested_spec_mode != "mtp" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert b._requested_spec_mode == "mtp"
    assert b._spec_fallback_reason == "runtime_error"


def test_runtime_recovery_skips_when_process_replaced(monkeypatch):
    # A newer user load that replaces the process during the death-confirm poll
    # must not be clobbered by the stale recovery replay.
    b = _recovery_backend()
    p1 = _BlockingDeadProc()
    b._process = p1
    calls = []
    monkeypatch.setattr(b, "load_model", lambda **k: calls.append(k))
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is True  # captures p1
    b._process = _FakeProcess()  # a newer load swapped the live process
    p1.release()  # p1 now reports dead -> recovery runs its staleness check
    time.sleep(0.6)
    assert calls == [], "stale recovery replayed over a newer load"


def test_runtime_recovery_skips_when_snapshot_changed(monkeypatch):
    # If the recorded load changed during the poll, the stale snapshot is dropped.
    b = _recovery_backend()
    p1 = _BlockingDeadProc()
    b._process = p1
    calls = []
    monkeypatch.setattr(b, "load_model", lambda **k: calls.append(k))
    assert b._maybe_recover_from_mtp_crash(RuntimeError()) is True
    b._last_load_kwargs = dict(b._last_load_kwargs, model_identifier = "other/model")
    p1.release()
    time.sleep(0.6)
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
    # recover() must re-check the cancel flag after the death poll (load_model
    # clears it), so a reload scheduled just before /unload can't resurrect it.
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

    assert captured["trust_env"] is False
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
    # The watchdog must notice the process exit and recover even when no request
    # handler observed it (e.g. the direct proxy endpoints).
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
        lambda b: setattr(b, "_mtp_runtime_fallback_active", False),
        lambda b: setattr(b, "_process", None),
    ],
)
def test_crash_watchdog_not_armed_when_inapplicable(mutate):
    # Only a launch actually running MTP+tensor with a live process arms it.
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
    # No totals here, so usable is the legacy free*frac (keeps the 5% cushion).
    reserve = LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
    usable = sum(f * _CTX_FIT_VRAM_FRACTION for _, f in gpus)
    return (usable - len(gpus) * reserve) * 1024 * 1024 - int(model_gb * _GB)


def test_tp_plan_weighted_split_on_asymmetric_big_model():
    b, (ec, mac, gi, ts) = _plan(50)
    reserve = b._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
    assert gi == [0, 1]
    # split weighted by (usable - flat buffer - per-device context compute); with
    # no totals usable is free*frac. The per-device cc is subtracted so the smaller
    # card isn't weighted above its real usable budget (see below).
    cc_per_dev = b._compute_buffer_ctx_bytes(ec, None, None) // (1024 * 1024)
    assert cc_per_dev > 0
    assert ts == [
        int(48000 * _CTX_FIT_VRAM_FRACTION - reserve - cc_per_dev),
        int(24000 * _CTX_FIT_VRAM_FRACTION - reserve - cc_per_dev),
    ]
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


def test_tp_plan_reserves_context_linear_compute_buffer():
    # Tensor mode replicates the compute graph on every device; measured on
    # Qwen3.5-9B at f16 the per-device buffer grows ~n_ubatch*2 B/token (~1024
    # B/tok), so the fit must reserve n_dev x that on top of the flat reserve or
    # it over-pins and OOMs at high context. The chosen KV must leave room for it.
    b, (ec, mac, gi, ts) = _plan(50)
    cc = len(gi) * b._compute_buffer_ctx_bytes(ec, None, "f16")
    assert cc > 0
    assert b._estimate_kv_cache_bytes(ec) + cc <= _kv_budget_b(50)


def test_tp_plan_context_shrinks_vs_compute_unaware():
    # With the context-linear term the pinned context is strictly below what a
    # KV-only (compute-unaware) fit at the same budget would allow.
    b, (ec, *_r) = _plan(50)
    b2 = _kv_seeded_backend()
    b2._embedding_length = 0  # kills the context-linear compute term (returns 0)
    ec_naive, *_r2 = b2._plan_tensor_parallel(_ASYM, int(50 * _GB), 131072)
    assert ec < ec_naive


def test_tp_plan_soft_overhead_shrinks_context():
    # The CUDA-ctx / mmproj / MTP-draft reserve the layer path folds into the fit
    # budget (model_size_fit) must also shrink the tensor context. Tensor mode has
    # no --fit valve, so an unreserved overshoot OOMs at startup instead of
    # offloading. A non-zero soft_overhead must pin a strictly smaller context.
    b = _kv_seeded_backend()
    ec_no, *_r = b._plan_tensor_parallel(_ASYM, int(50 * _GB), 131072)
    ec_soft, *_r2 = b._plan_tensor_parallel(
        _ASYM, int(50 * _GB), 131072, soft_overhead_bytes = 2 * _GB
    )
    assert 2048 < ec_soft < ec_no


def test_tp_plan_soft_overhead_reserved_against_budget():
    # The pinned context must leave the whole soft reserve free on top of KV and
    # the replicated context compute, so the real footprint stays within the pool.
    b = _kv_seeded_backend()
    soft = 2 * _GB
    ec, *_r = b._plan_tensor_parallel(_ASYM, int(50 * _GB), 131072, soft_overhead_bytes = soft)
    cc = len(_ASYM) * b._compute_buffer_ctx_bytes(ec, None, None)
    assert b._estimate_kv_cache_bytes(ec) + cc + soft <= _kv_budget_b(50)


def test_tp_plan_weighted_split_keeps_small_gpu_within_budget():
    # Regression: the weighted split must subtract each device's replicated context
    # compute (cc_bytes/n_dev), not just the flat reserve. Otherwise the smaller
    # card is weighted above its usable budget and OOMs at launch. Model the split:
    # llama.cpp distributes weights+KV by the tensor-split weights; every device
    # also holds the flat reserve plus its per-device context compute.
    b, (ec, mac, gi, ts) = _plan(50)
    assert ts is not None and len(ts) == len(gi) == 2
    reserve = b._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
    cc_per_dev = b._compute_buffer_ctx_bytes(ec, None, None) // (1024 * 1024)
    free_by_idx = {0: 48000, 1: 24000}
    split_content_mib = (int(50 * _GB) + b._estimate_kv_cache_bytes(ec)) / (1024 * 1024)
    total_weight = sum(ts)
    for w, idx in zip(ts, gi):
        placed = split_content_mib * w / total_weight
        usable = free_by_idx[idx] * _CTX_FIT_VRAM_FRACTION
        assert placed + reserve + cc_per_dev <= usable + 1  # +1 MiB for int rounding

    # Lock the regression: under the old formula (flat reserve only) the smaller
    # card was placed over its budget; the cc term is what pulls it back.
    old_adj = [int(free_by_idx[i] * _CTX_FIT_VRAM_FRACTION - reserve) for i in gi]
    old_small_placed = split_content_mib * old_adj[1] / sum(old_adj)
    assert old_small_placed + reserve + cc_per_dev > free_by_idx[1] * _CTX_FIT_VRAM_FRACTION


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
# A tensor-incompatible model makes load_model RAISE (not return False); the
# /load fallback must catch it and retry with layer split (stripping --split-mode
# so the retry can't relaunch tensor), while a non-tensor load propagates.


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
    # --split-mode form (long/short, space/=) and force layer, keeping the user's
    # other flags, else tensor is re-enabled and relaunches the crash.
    loader = _RecordingLoader()
    ok = asyncio.run(
        load_with_tensor_fallback(loader, requested_tensor = False, extra_args = extras, label = "m")
    )
    assert ok is True
    assert len(loader.calls) == 2
    # User --split-mode replaced by an explicit layer override; -c kept.
    assert loader.calls[1][1] == ["-c", "4096", "--split-mode", "layer"]


def test_tensor_fallback_env_tensor_retry_forces_layer(monkeypatch):
    # Env-only tensor (toggle off, no --split-mode extra): load_model engages
    # tensor via LLAMA_ARG_SPLIT_MODE and a tensor-incompatible model crashes. The
    # wrapper must (1) recognise the env tensor request and retry, and (2) force
    # --split-mode layer so the retry doesn't re-engage tensor via the still-set
    # env and crash again (#6312).
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "tensor")
    calls: list = []

    async def _crash_when_effectively_tensor(tensor_parallel, extra_args):
        calls.append(list(extra_args) if extra_args else extra_args)
        # Mirror real load_model: env-aware tensor engagement crashes.
        if _effective_tensor_parallel(extra_args, tensor_parallel):
            raise RuntimeError("llama-server failed to start (tensor)")
        return True

    ok = asyncio.run(
        load_with_tensor_fallback(
            _crash_when_effectively_tensor,
            requested_tensor = False,
            extra_args = None,
            label = "m",
        )
    )
    assert ok is True
    assert len(calls) == 2
    # The forced layer override neutralises the inherited tensor env on retry.
    assert calls[1] == ["--split-mode", "layer"]


def test_tensor_fallback_propagates_non_tensor_crash():
    async def _always_raise(tensor_parallel, extra_args):
        raise RuntimeError("bad model")

    with pytest.raises(RuntimeError, match = "bad model"):
        asyncio.run(
            load_with_tensor_fallback(
                _always_raise, requested_tensor = False, extra_args = None, label = "m"
            )
        )


# ── _plan_tensor_parallel: total-based headroom + ubatch (review fixes) ──


def test_tensor_caps_context_to_total_vram_budget():
    # Partly-used 80 GB cards: 20 GB free each. With total_by_idx the planner must
    # cap occupancy at 0.95*total (not spend the cushion the layer-split paths keep).
    b = _kv_seeded_backend()
    gpus = [(0, 20000), (1, 20000)]
    totals = {0: 81920, 1: 81920}
    model = int(18 * _GB)
    with_total, *_ = b._plan_tensor_parallel(gpus, model, 131072, total_by_idx = totals)
    without, *_ = b._plan_tensor_parallel(gpus, model, 131072)
    assert with_total < without  # total cap tightens the chosen context

    MIB = 1024 * 1024
    reserve = LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB  # flat (no vocab dims)
    pool_usable = sum(f - (1.0 - _CTX_FIT_VRAM_FRACTION) * totals[i] for i, f in gpus)
    foot_total = (model + b._estimate_kv_cache_bytes(with_total, None)) / MIB + len(gpus) * reserve
    foot_free = (model + b._estimate_kv_cache_bytes(without, None)) / MIB + len(gpus) * reserve
    assert foot_total <= pool_usable + 2  # fix: fits the total-based budget
    assert foot_free > pool_usable  # old behavior over-spent the cushion


def test_tensor_unknown_total_keeps_fraction_cushion():
    # A two-column nvidia-smi probe yields total 0. The planner must fall back to
    # free*frac (keep the 5% cushion), like _select_gpus/_gpu_usable, not raw free,
    # or it over-advertises context exactly where the PR is hardening the budget.
    b = _kv_seeded_backend()
    gpus = [(0, 20000), (1, 20000)]
    MIB = 1024 * 1024
    reserve = LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB
    model = int(18 * _GB)
    ec_zero, *_ = b._plan_tensor_parallel(gpus, model, 131072, total_by_idx = {0: 0, 1: 0})
    ec_none, *_ = b._plan_tensor_parallel(gpus, model, 131072)
    assert ec_zero == ec_none  # total 0 == total absent: both use free*frac
    pool_free = sum(f for _, f in gpus)
    foot = (model + b._estimate_kv_cache_bytes(ec_zero, None)) / MIB + len(gpus) * reserve
    assert foot <= pool_free * _CTX_FIT_VRAM_FRACTION + 2  # within free*frac, not raw free


def test_tensor_reserve_scales_with_ubatch():
    # A user --ubatch override must enlarge the per-device reserve -> less ctx room.
    b = _kv_seeded_backend()
    b._vocab_size = 152064  # enable the deterministic compute-buffer estimate
    gpus = [(0, 16000), (1, 16000)]
    model = int(18 * _GB)
    small_ub, *_ = b._plan_tensor_parallel(gpus, model, 131072, n_ubatch = 512)
    big_ub, *_ = b._plan_tensor_parallel(gpus, model, 131072, n_ubatch = 4096)
    assert big_ub < small_ub


def test_plan_tensor_carries_unsized_mtp_flat_reserve():
    # review run3 #1/#5: with a weights-only (KV-unsized) MTP reserve, the planner
    # gets a non-None mtp_overhead_fn but must still subtract the flat unsized-KV
    # cushion, or its binary search spends it on context. Passing the reserve must
    # pick a strictly smaller context than passing 0.
    b = _kv_seeded_backend()
    gpus = [(0, 14000), (1, 14000)]  # tight pool so the context is actually capped
    model = int(8 * _GB)
    weights_only = lambda c: 3 * _GB  # noqa: E731 -- constant drafter weights, no KV term
    ctx_no_flat, *_ = b._plan_tensor_parallel(
        gpus,
        model,
        131072,
        mtp_engaged = True,
        mtp_overhead_fn = weights_only,
        mtp_flat_reserve_bytes = 0,
    )
    ctx_flat, *_ = b._plan_tensor_parallel(
        gpus,
        model,
        131072,
        mtp_engaged = True,
        mtp_overhead_fn = weights_only,
        mtp_flat_reserve_bytes = 2 * _GB,
    )
    assert 0 < ctx_flat < ctx_no_flat


def test_tensor_admission_drops_gpu_below_usable_budget():
    # A partly-used big card can clear the buffer reserve on raw free yet have no
    # usable budget left (free - 0.05*total). Admit by usable budget: GPU 0 here is
    # 6000 free on an 80 GB card -> usable 1904 < flat reserve 5120, so it's dropped
    # (leaving <2 -> no split). Without total_by_idx, raw free 6000 >= 5120 admits it.
    b = _kv_seeded_backend()
    gpus = [(0, 6000), (1, 40000)]
    totals = {0: 81920, 1: 81920}
    _ec, _mac, gi, ts = b._plan_tensor_parallel(gpus, int(8 * _GB), 8192, total_by_idx = totals)
    assert gi == [1] and ts is None  # GPU 0 excluded on usable budget
    _ec2, _mac2, gi_raw, _ts2 = b._plan_tensor_parallel(gpus, int(8 * _GB), 8192)
    assert gi_raw == [0, 1]  # raw free would have admitted both


def test_load_model_tensor_admission_and_capacity_gate_use_usable_budget():
    # load_model is too entangled (subprocess + GPU probe) to drive end-to-end, so
    # assert at the source level that the tensor prefilter admits on the usable
    # budget (_gpu_usable), not raw free, and downgrades to layer split when the
    # pooled budget can't hold weights + per-device compute buffers.
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "_gpu_usable(g) >= reserve_mib" in src  # admit by usable budget
    assert "g[1] >= reserve_mib" not in src  # not raw free
    assert "_tp_weight_budget_mib" in src  # pooled-weight capacity gate
    assert "falling back to layer split" in src  # downgrade on overcommit
    # The gate's required footprint must include the non-shrinkable MTP reserve,
    # not weights alone, or a separate-drafter MTP load can still overcommit.
    assert "_tp_mtp_floor" in src
    assert "model_size + _tp_mtp_floor" in src


def test_load_model_tensor_floor_keeps_flat_reserve_for_weights_only():
    # Tensor mode has no --fit valve, so a weights-only drafter (KV unsized) must
    # keep the flat reserve as the draft-KV cushion, not just the byte weights
    # (Finding H1, the tensor analog of the layer-split _mtp_kv_unsized handling).
    compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    # byte-only floor used only when KV is sizable (not the weights-only case)
    assert "mtp_overhead_fnisnotNoneandnot_mtp_kv_unsized" in compact
    # weights-only / dims-unavailable: flat reserve, never below the byte floor
    assert "_tp_mtp_floor=max(" in compact


def test_load_model_reserves_pipeline_per_device_overhead():
    # Layer split must reserve the fixed per-device overhead per EXTRA device so a
    # tight multi-GPU split can't pin a context that OOMs a device (Finding A); k=1
    # adds nothing.
    assert LlamaCppBackend._PIPELINE_PER_DEVICE_OVERHEAD_MIB > 0
    compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    assert "def_subset_model_size(n_gpus:int)->int:" in compact
    assert "max(0,n_gpus-1)*_pipeline_overhead_bytes" in compact
    assert "_subset_model_size(n_gpus)" in compact  # used in the layer-split fit


def test_load_model_restores_quantized_kv_on_tensor_downgrade():
    # A quantized KV dropped for the tensor attempt must be restored if tensor
    # downgrades to layer split (Finding D); captured once, restored at both the
    # GPU-count and capacity-gate downgrades.
    compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    assert "_tensor_dropped_cache_type_kv=cache_type_kv" in compact  # captured pre-null
    # Restore is shared in one closure, called at every tensor->layer downgrade.
    assert "cache_type_kv=_tensor_dropped_cache_type_kv" in compact  # restored in the closure
    assert "def_restore_after_tensor_downgrade():" in compact  # one shared restore helper
    assert compact.count("_restore_after_tensor_downgrade()") >= 3  # called at each downgrade
