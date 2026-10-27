# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A llama-server whose Metal backend is poisoned is retired, respawned once, then unloaded."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Optional
from unittest.mock import patch

import httpx
import pytest

import core.inference.llama_cpp as llama_cpp
from core.inference import model_slots
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

from . import _sidecar_fit_helpers as h

# After a failed command buffer ggml-metal rejects every later graph until the backend is
# recreated, so the child stays up answering "Compute error.". The reporter's lines:
METAL_OOM = "0.30.864.334 E error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
METAL_LATCHED = (
    "0.30.882.907 E ggml_metal_graph_compute: backend is in error state from a previous "
    "command buffer failure - recreate the backend to recover"
)
DRAFT_FAILED = "0.22.891.174 E process: llama_process(ctx_dft) failed rc=-3 (n_tokens=14, offset=0)"


@dataclass(frozen = True)
class _Intent:
    speculative_type: Optional[str] = None
    extra_args: tuple = ()
    gpu_ids: Optional[tuple] = None
    verified_gguf: Optional[str] = None


_OFF = _Intent(speculative_type = "off")
_MTP = _Intent(speculative_type = "mtp")


def _child(lines = ()):
    code = "import sys, time\n"
    for line in lines:
        code += f"print({line!r}, flush = True)\n"
    code += "time.sleep(120)\n"
    return subprocess.Popen([sys.executable, "-c", code], stdout = subprocess.PIPE, text = True)


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_DEVICE_LOST_LOG_GRACE_S", 0.0)
    children = []

    def make(
        intent: _Intent,
        speculative_type: str = "draft-dspark",
        lines = (),
    ):
        backend = LlamaCppBackend()
        first = _child(lines)
        children.append(first)
        backend._process = first
        backend._healthy = True
        backend._model_identifier = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
        backend._last_load_intent = intent
        backend._speculative_type = speculative_type
        # The child here is a stub with no HTTP server: stand in for a compute that fails.
        backend._metal_compute_failure_confirmed = lambda: True
        loads = []

        def fake_load_model(new_intent):
            loads.append(new_intent)
            replacement = _child()
            children.append(replacement)
            backend._process = replacement
            backend._healthy = True
            backend._last_load_intent = new_intent
            backend._speculative_type = (
                "none" if new_intent.speculative_type == "off" else speculative_type
            )
            return True

        backend.load_model = fake_load_model
        backend._server_socket_is_open = lambda: False
        return backend, first, loads

    yield make
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()


def _wait(predicate, timeout = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _poison(
    backend,
    process,
    draft = False,
):
    backend._retire_device_lost_server(process, metal_poison = {"draft": draft})


@pytest.mark.parametrize(
    "lines, retired",
    [
        ([METAL_OOM], True),
        ([METAL_LATCHED], True),
        (
            [
                "0.01.000.000 I slot launch_slot_: id  0 | prompt: what is kIOGPUCommandBufferCallbackErrorOutOfMemory?"
            ],
            False,
        ),
        (
            ["what does backend is in error state from a previous command buffer failure mean?"],
            False,
        ),
    ],
    ids = ["oom_line", "latched_line", "info_line", "unprefixed_request_dump"],
)
def test_a_poisoned_metal_backend_in_the_log_retires_the_server(harness, lines, retired):
    backend, first, _loads = harness(_OFF, "none", lines)
    threading.Thread(target = backend._drain_stdout, daemon = True).start()
    try:
        first.wait(timeout = 5)
    except subprocess.TimeoutExpired:
        pass
    assert (first.poll() is not None) is retired


def test_echoed_marker_without_a_compute_failure_does_not_retire(harness):
    """A grammar parse error prints the request's grammar, so a forged E-line can carry the
    marker: retirement needs the child to really fail compute."""
    backend, first, _loads = harness(_OFF, "none", [METAL_OOM])
    backend._metal_compute_failure_confirmed = lambda: False
    threading.Thread(target = backend._drain_stdout, daemon = True).start()
    try:
        first.wait(timeout = 1.5)
    except subprocess.TimeoutExpired:
        pass
    assert first.poll() is None
    assert getattr(backend, "_metal_oom_strikes", None) is None


def test_auto_dspark_draft_oom_reloads_once_without_the_drafter_then_stops(harness):
    backend, first, loads = harness(
        _Intent(), "draft-dspark", [METAL_OOM, METAL_LATCHED, DRAFT_FAILED]
    )
    threading.Thread(target = backend._drain_stdout, daemon = True).start()

    assert _wait(lambda: len(loads) == 1)
    assert first.poll() is not None
    assert loads[0].speculative_type == "off"
    assert _wait(lambda: not backend._mtp_runtime_fallback_in_progress)

    # The drafter-free server runs out of Metal memory too: the second strike is terminal.
    second = backend._process
    _poison(backend, second)
    assert second.poll() is not None
    assert backend._respawn_if_dead() is False
    assert len(loads) == 1


def test_explicit_dspark_is_not_silently_dropped(harness):
    intent = _Intent(speculative_type = "dspark")
    backend, first, loads = harness(intent, "draft-dspark")
    _poison(backend, first, draft = True)
    time.sleep(0.3)
    # No drafter-free reload; the next request replays the user's own settings once.
    assert loads == [] and first.poll() is not None
    assert backend._respawn_if_dead() is True
    assert loads == [intent]


def test_retirement_keeps_a_user_owned_drafter(harness):
    """Auto plus a bare --model-draft: an Off reload would still load that file."""
    intent = _Intent(extra_args = ("--model-draft", "/models/custom-dspark.gguf"))
    backend, first, loads = harness(intent, "draft-dspark")
    _poison(backend, first, draft = True)
    assert first.poll() is not None
    _wait(lambda: bool(loads), timeout = 2.0)
    assert all(load.speculative_type != "off" for load in loads)


def test_a_launch_without_a_sidecar_respawns_once_then_stops(harness):
    backend, first, loads = harness(_OFF, "none")
    _poison(backend, first)
    assert backend._respawn_if_dead() is True and loads == [_OFF]
    _poison(backend, backend._process)
    assert backend._respawn_if_dead() is False and loads == [_OFF]


def test_a_terminal_second_oom_is_not_left_healthy(harness):
    backend, first, _loads = harness(_OFF, "none")
    _poison(backend, first)
    assert backend._respawn_if_dead() is True
    _poison(backend, backend._process)
    assert backend._healthy is False


def test_a_terminally_retired_server_is_not_active(harness):
    """After the second Metal OOM nothing respawns the child, so delete, Always mode and the
    training guard must not see a model still loading."""
    backend, first, _loads = harness(_OFF, "none")
    _poison(backend, first)
    assert backend._respawn_if_dead() is True
    assert backend.is_active
    second = backend._process
    _poison(backend, second)
    assert second.poll() is not None
    assert backend._process is None and backend._model_identifier is None
    assert backend._respawn_if_dead() is False
    assert backend.is_loaded is False
    # routes/models.py refuses a delete with 409 while is_active or is_loaded.
    assert backend.is_active is False


def test_unload_clears_the_strikes(harness):
    backend, first, _loads = harness(_OFF, "none")
    backend._metal_oom_strikes = (_OFF, 2)
    backend._metal_oom_terminal_key = _OFF
    backend.unload_model()
    assert backend._metal_oom_strikes is None and backend._metal_oom_terminal_key is None


def test_a_model_swap_during_retirement_is_not_charged_the_strike(harness):
    """An explicit load must not publish another model while the dead child is reaped."""
    backend, first, _loads = harness(_OFF, "none")
    real_kill = first.kill
    blocked = {}

    def kill_and_try_a_load():
        real_kill()
        got = []
        t = threading.Thread(
            target = lambda: got.append(backend._serial_load_lock.acquire(blocking = False))
        )
        t.start()
        t.join()
        blocked["load"] = got == [False]
        if got == [True]:
            backend._serial_load_lock.release()

    first.kill = kill_and_try_a_load
    _poison(backend, first)
    assert blocked.get("load") is True
    assert backend._metal_oom_strikes == (backend._metal_oom_key(_OFF), 1)


def test_the_strike_is_recorded_before_a_respawn_can_run(harness):
    """Retirement decides the recovery before releasing _respawn_lock, so a request racing
    the kill cannot replay the speculative config first."""
    backend, first, loads = harness(_Intent(), "draft-dspark")
    seen = {}
    real_after = backend._after_metal_poisoned_server

    def after(draft_failed):
        seen["respawn_lock_held"] = backend._respawn_lock.locked()
        return real_after(draft_failed)

    backend._after_metal_poisoned_server = after
    _poison(backend, first, draft = True)
    assert seen.get("respawn_lock_held") is True
    assert _wait(lambda: len(loads) == 1)
    assert loads[0].speculative_type == "off"


def test_mtp_recovery_respects_the_terminal_strike(harness):
    backend, first, loads = harness(_Intent(), "draft-dspark")
    backend._metal_oom_terminal_key = backend._metal_oom_key(_Intent())
    backend._mtp_runtime_fallback_active = True
    first.kill()
    first.wait()
    assert backend._maybe_recover_from_mtp_crash(RuntimeError("boom")) is False
    assert backend._maybe_recover_from_mtp_crash(sidecar_oom = True) is False
    time.sleep(0.3)
    assert loads == []


def test_a_retirement_thread_that_cannot_start_releases_the_poison_flag(monkeypatch):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    poison = {"pending": True}
    LlamaCppBackend()._retire_device_lost_server_soon(None, metal_poison = poison)
    assert poison["pending"] is False


# Confirming the poison: probe the route the server really computes on.


def _compute_probe(
    monkeypatch,
    extras = None,
    embedding = None,
    message = "Compute error.",
):
    backend = LlamaCppBackend()
    backend._port = 1
    if extras is not None:
        backend._extra_args = list(extras)
    if embedding is not None:
        monkeypatch.setattr(LlamaCppBackend, "is_embedding_gguf", property(lambda self: embedding))
    resp = type(
        "Resp", (), {"status_code": 500, "json": lambda self: {"error": {"message": message}}}
    )()
    with patch.object(llama_cpp.httpx, "post", return_value = resp, create = True) as post:
        return backend._metal_compute_failure_confirmed(), post


@pytest.mark.parametrize(
    "message, confirmed",
    [
        ("Compute error.", True),
        ("decode() failed: failed to process speculative batch", True),
        ("the request exceeds the available context size", False),
    ],
    ids = ["target", "draft", "unrelated"],
)
def test_the_draft_context_failure_confirms_the_poison(monkeypatch, message, confirmed):
    assert _compute_probe(monkeypatch, message = message)[0] is confirmed


@pytest.mark.parametrize(
    "env, embedding, extras, route, field",
    [
        # --rerank sets embedding mode with RANK pooling: /completion 501s before compute,
        # and send_embedding would read n_embd_out floats out of RANK's n_cls_out.
        ({}, False, ("--rerank",), "/rerank", "documents"),
        ({}, False, ("--reranking",), "/rerank", "documents"),
        ({}, False, ("--embeddings", "--pooling", "rank"), "/rerank", "documents"),
        # A later --pooling replaces --rerank's RANK, as in common/arg.cpp.
        ({}, False, ("--rerank", "--pooling", "mean"), "/embedding", "content"),
        # /v1/embeddings 400s on NONE pooling, explicit or the model's default.
        ({}, False, ("--embeddings", "--pooling", "none"), "/embedding", "content"),
        ({}, True, (), "/embedding", "content"),
        ({}, False, (), "/completion", None),
        # --embedding restricts llama-server to embeddings: /completion never computes.
        ({}, False, ("--embeddings",), "/embedding", None),
        # load_model pops LLAMA_ARG_POOLING / _RERANKING / _EMBEDDINGS from the child's env.
        ({"LLAMA_ARG_RERANKING": "1"}, False, (), "/completion", None),
        ({"LLAMA_ARG_POOLING": "rank"}, True, (), "/embedding", None),
    ],
    ids = [
        "rerank",
        "reranking",
        "pooling-rank",
        "rerank-then-mean",
        "pooling-none",
        "embedding-gguf",
        "completion",
        "embeddings-flag",
        "inherited-reranking-env",
        "inherited-pooling-env",
    ],
)
def test_embedding_and_rerank_servers_are_probed_where_they_compute(
    monkeypatch, env, embedding, extras, route, field
):
    monkeypatch.delenv("LLAMA_ARG_POOLING", raising = False)
    monkeypatch.delenv("LLAMA_ARG_RERANKING", raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    confirmed, post = _compute_probe(monkeypatch, extras, embedding)
    assert confirmed is True
    assert post.call_args.args[0].endswith(route)
    if field is not None:
        assert field in post.call_args.kwargs["json"]


def test_a_reranker_by_gguf_default_alone_is_not_probed(monkeypatch):
    """--embeddings on a RANK-pooling GGUF: /rerank refuses (no --rerank) and /embedding would
    over-read RANK's buffer, so nothing is sent and the server is not retired."""
    monkeypatch.delenv("LLAMA_ARG_POOLING", raising = False)
    monkeypatch.delenv("LLAMA_ARG_RERANKING", raising = False)
    backend = LlamaCppBackend()
    backend._port = 1
    backend._pooling_type = 4
    backend._extra_args = ["--embeddings"]
    with patch.object(llama_cpp.httpx, "post", create = True) as post:
        assert backend._metal_compute_failure_confirmed() is False
    post.assert_not_called()


# Metal OOM episodes: a strike belongs to one load; a caller's load starts a fresh one.


def _stub_child():
    return type(
        "Process",
        (),
        {
            "pid": 123,
            "poll": lambda self: 0,
            "kill": lambda self: None,
            "wait": lambda self, timeout = None: 0,
        },
    )()


def _strike(backend):
    dead = _stub_child()
    backend._process = dead
    backend._retired_process = dead
    backend._after_metal_poisoned_server(False)


def _loaded(tmp_path, **kw):
    backend, gguf = h.cuda_backend(tmp_path)
    backend._speculative_type = "none"
    h.placement._launch(backend, gguf, n_ctx = 4096, **kw)
    return backend, gguf


@pytest.mark.parametrize("retired", [False, True])
def test_a_retired_child_is_not_reused_by_an_identical_load(tmp_path, retired):
    """After a first retirement the killed child is still recorded and _healthy is still
    true: an identical explicit load must spawn a replacement."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"gguf")
    backend = LlamaCppBackend()
    proc = _stub_child()
    proc.poll = lambda: None
    backend._process = proc
    backend._healthy = True
    backend._model_identifier = "m"
    backend._gguf_path = str(gguf)
    if retired:
        backend._retire_device_lost_server(proc)
    intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "m")
    assert backend.matches_load_source(intent) is (not retired)


_HINT = ("someone/repo", "Q4_K_M", "/nonexistent/model.gguf", ())


def test_the_strike_key_ignores_the_verified_cache_hint(tmp_path):
    backend, gguf = h.cuda_backend(tmp_path)
    plain = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096)
    hinted = GgufLoadIntent(
        gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096, verified_gguf = _HINT
    )
    assert backend._metal_oom_key(hinted) == backend._metal_oom_key(plain)
    backend._metal_oom_terminal_key = backend._metal_oom_key(plain)
    assert backend._metal_oom_is_terminal(hinted)


@pytest.mark.parametrize(
    "hint",
    [None, _HINT],
    ids = [
        "test_explicit_reload_clears_a_terminal_metal_oom",
        "test_a_hinted_explicit_reload_clears_a_terminal_metal_oom",
    ],
)
def test_an_explicit_reload_clears_a_terminal_metal_oom(tmp_path, hint):
    backend, gguf = h.cuda_backend(tmp_path)
    # The terminal key comes from a committed snapshot, which never carries the hint.
    snapshot = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096)
    key = backend._metal_oom_key(snapshot)
    backend._metal_oom_strikes = (key, 2)
    backend._metal_oom_terminal_key = key
    h.spawns(backend, gguf, healthy = lambda i, cmd: True, n_ctx = 4096, verified_gguf = hint)
    assert backend._last_load_intent is not None
    assert backend._last_load_intent.verified_gguf is None
    assert backend._metal_oom_terminal_key is None
    assert backend._metal_oom_strikes is None
    assert not backend._metal_oom_is_terminal(backend._last_load_intent)


def test_an_explicit_reload_with_changed_settings_clears_the_terminal_latch(tmp_path):
    """The latch must not survive a reload that changes the GPU pick and re-arm once a
    matching request writes gpu_ids back."""
    backend, gguf = _loaded(tmp_path)
    committed = backend._last_load_intent
    assert committed is not None and committed.gpu_ids is None
    _strike(backend)
    _strike(backend)
    assert backend._metal_oom_is_terminal(committed)
    h.placement._launch(backend, gguf, n_ctx = 4096, gpu_ids = [0])
    assert backend.is_active and backend.is_loaded
    reloaded = backend._last_load_intent
    assert list(reloaded.gpu_ids) == [0]
    # Not vacuous: writing gpu_ids back reproduces the latched load exactly.
    assert backend._metal_oom_key(replace(reloaded, gpu_ids = None)) == backend._metal_oom_key(
        committed
    )
    assert backend._metal_oom_terminal_key is None
    assert backend._metal_oom_strikes is None
    backend._record_matching_gpu_request(None)
    assert backend._metal_oom_is_terminal(backend._last_load_intent) is False


def test_a_callers_load_starts_a_fresh_metal_oom_episode(tmp_path):
    """A (one strike), then B, then A again: A's new session gets its own two strikes."""
    backend, gguf = _loaded(tmp_path)
    _strike(backend)
    assert backend._metal_oom_strikes is not None
    h.placement._launch(backend, gguf, n_ctx = 8192)
    assert backend._metal_oom_strikes is None
    h.placement._launch(backend, gguf, n_ctx = 4096)
    _strike(backend)
    assert backend._metal_oom_strikes[1] == 1
    assert backend._metal_oom_terminal_key is None


def test_automatic_recovery_keeps_the_strike(tmp_path):
    """The respawn and the drafter-free reload replay under the recovery marker, which carries
    the count, so a second OOM still stops respawning."""
    backend, gguf = _loaded(tmp_path)
    _strike(backend)
    with llama_cpp._automatic_recovery_load():
        h.placement._launch(backend, gguf, n_ctx = 4096)
    assert backend._metal_oom_strikes[1] == 1
    _strike(backend)
    assert backend._metal_oom_terminal_key is not None


def test_the_respawn_and_drafter_free_reload_run_as_automatic_recovery():
    for fn in (LlamaCppBackend._respawn_if_dead, LlamaCppBackend._maybe_recover_from_mtp_crash):
        assert "with _automatic_recovery_load():" in inspect.getsource(fn)


def test_a_diffusion_load_drops_the_chat_servers_recovery_state(tmp_path, monkeypatch):
    """A crashed diffusion runner must neither replay the chat load it replaced nor meet that
    load's Metal OOM latch."""
    import test_gpu_memory_mode as _gm

    real_init = LlamaCppBackend.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self._last_load_intent = object()
        self._mtp_runtime_fallback_active = True
        self._metal_oom_strikes = ("chat", 2)
        self._metal_oom_terminal_key = "chat"

    monkeypatch.setattr(LlamaCppBackend, "__init__", init)
    _cmd, _env, backend = _gm._launch_diffusion(
        monkeypatch, tmp_path, gguf_gib = 4.0, gpus = _gm._TWO_3090, gpu_ids = None
    )
    assert backend._last_load_intent is None
    assert backend._mtp_runtime_fallback_active is False
    assert backend._metal_oom_strikes is None
    assert backend._metal_oom_terminal_key is None


# A server Studio retired on purpose is not an MTP crash: _respawn_if_dead replays it.


def _mtp_server(harness):
    backend, first, loads = harness(_MTP, "draft-mtp")
    backend._mtp_runtime_fallback_active = True
    return backend, first, loads


def _retire(backend, first, metal):
    if metal:
        _poison(backend, first)
    else:
        backend._retire_device_lost_server(first)


@pytest.mark.parametrize("metal", [True, False], ids = ["metal-strike-1", "vulkan-device-lost"])
def test_a_retired_mtp_server_is_respawned_not_reloaded_without_mtp(harness, metal):
    backend, first, loads = _mtp_server(harness)
    _retire(backend, first, metal)
    assert first.poll() is not None
    assert backend._maybe_recover_from_mtp_crash(httpx.ConnectError("closed")) is False
    assert backend._respawn_if_dead() is True
    assert [load.speculative_type for load in loads] == ["mtp"]


def test_a_real_mtp_crash_still_reloads_without_mtp(harness):
    backend, first, loads = _mtp_server(harness)
    first.kill()
    first.wait()
    assert backend._maybe_recover_from_mtp_crash(httpx.ConnectError("closed")) is True
    assert _wait(lambda: bool(loads))
    assert loads[-1].speculative_type == "off"


def test_a_reload_queued_before_the_retirement_stands_down(harness):
    """The request saw the port close while the child was still exiting; the retirement
    lands during the queued reload's exit wait."""
    backend, first, loads = _mtp_server(harness)
    assert backend._maybe_recover_from_mtp_crash(httpx.ConnectError("closed")) is True
    backend._retire_device_lost_server(first)
    assert _wait(lambda: not backend._mtp_runtime_fallback_in_progress)
    assert loads == []


def test_the_retirement_is_published_before_the_kill(harness):
    backend, first, _loads = _mtp_server(harness)
    seen = []
    kill = first.kill

    def recording_kill():
        seen.append(getattr(backend, "_retired_process", None) is first)
        kill()

    first.kill = recording_kill
    backend._retire_device_lost_server(first)
    assert seen == [True]


# A retired child takes its engine-stats poller with it.


class _Poller:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


@pytest.mark.parametrize("metal", [True, False], ids = ["metal", "device-lost"])
def test_a_retired_child_stops_its_stats_poller(harness, metal):
    backend, first, _loads = harness(_OFF, "none")
    poller = backend._stats_logger = _Poller()
    _retire(backend, first, metal)
    assert first.poll() is not None
    assert poller.stopped and backend._stats_logger is None


def test_a_terminal_metal_retirement_leaves_no_poller(harness):
    backend, first, _loads = harness(_OFF, "none")
    _poison(backend, first)
    assert backend._respawn_if_dead() is True
    poller = backend._stats_logger = _Poller()
    _poison(backend, backend._process)
    assert backend._process is None and backend._model_identifier is None
    assert poller.stopped and backend._stats_logger is None


def test_the_replacement_load_starts_its_own_poller(tmp_path, monkeypatch):
    import core.inference.llama_stats as _stats

    started = []
    monkeypatch.setattr(
        _stats,
        "maybe_start_stats_logger",
        lambda base_url, logger, headers = None: started.append(base_url) or _Poller(),
    )
    backend, gguf = h.cuda_backend(tmp_path)
    caps = dict(h.CAPS, supports_metrics = True)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    h.placement._launch(backend, gguf, n_ctx = 4096)
    # What a retirement leaves: the poller stopped and cleared.
    backend._stats_logger.stop()
    backend._stats_logger = None
    h.placement._launch(backend, gguf, n_ctx = 4096, force_reload = True)
    assert len(started) == 2 and isinstance(backend._stats_logger, _Poller)
    assert not backend._stats_logger.stopped


# A terminal Metal retirement is a full unload, and the routes release what it held.


class _Fh:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def write(self, *_a):
        return 0

    def flush(self):
        pass


def _two_strikes(backend, first, monkeypatch, calls):
    _poison(backend, first)
    assert backend._respawn_if_dead() is True
    fh = backend._llama_log_fh = _Fh()
    monkeypatch.setattr(backend, "_clear_server_pid", lambda: calls.append("pid"))
    monkeypatch.setattr(backend, "_unload_audio_codec", lambda: calls.append("codec"))
    backend._planned_vram_mib = {0: 9000}
    _poison(backend, backend._process)
    return fh


def test_the_terminal_strike_unloads_the_backend(harness, monkeypatch):
    backend, first, _loads = harness(_OFF, "none")
    calls = []
    fh = _two_strikes(backend, first, monkeypatch, calls)
    assert backend._process is None and backend._model_identifier is None
    assert backend.is_active is False and backend.is_loaded is False
    assert fh.closed and "pid" in calls and "codec" in calls
    assert not backend._planned_vram_mib
    assert backend._respawn_if_dead() is False


def test_a_first_strike_still_respawns_with_the_same_settings(harness, monkeypatch):
    backend, first, loads = harness(_OFF, "none")
    seen = []
    monkeypatch.setattr(llama_cpp, "_TERMINAL_RETIREMENT_HOOKS", [seen.append], raising = False)
    _poison(backend, first)
    assert backend._model_identifier is not None
    assert backend._respawn_if_dead() is True
    assert loads == [_OFF] and seen == []


def test_the_terminal_strike_runs_the_cleanup_hooks_once(harness, monkeypatch):
    backend, first, _loads = harness(_OFF, "none")
    seen = []
    monkeypatch.setattr(
        llama_cpp, "_TERMINAL_RETIREMENT_HOOKS", [lambda *args: seen.append(args)], raising = False
    )
    _two_strikes(backend, first, monkeypatch, [])
    assert seen == [(backend, backend._unload_epoch)]


class _Orchestrator:
    def __init__(self):
        self.active_model_name = None
        self.loading_models = ()
        self.cleaned = False

    def _cleanup(self):
        self.cleaned = True


def _kept_slot(refs = 0, orchestrator = None):
    return model_slots.ExtraSlot(
        llama = LlamaCppBackend(),
        orchestrator = orchestrator or _Orchestrator(),
        account = "a",
        refs = refs,
    )


def _routes(monkeypatch):
    import routes.inference as inf

    assert inf._release_after_terminal_retirement in llama_cpp._TERMINAL_RETIREMENT_HOOKS
    calls = []
    monkeypatch.setattr(inf, "release_chat_after_kept_models", lambda: calls.append("kept") or True)
    monkeypatch.setattr(inf, "release_chat_gpu_claim", lambda: calls.append("claim") or True)
    monkeypatch.setattr(
        inf.account_access, "clear_resident", lambda modality: calls.append(f"clear:{modality}")
    )
    return inf, calls


def _release_primary(inf):
    inf._release_after_terminal_retirement(
        inf._llama_cpp_backend, inf._llama_cpp_backend._unload_epoch
    )


def test_a_retired_kept_slot_is_dropped_and_keeps_the_primarys_residency(monkeypatch):
    """Dead, it holds nothing a request could still use, so a held one goes too; the chat
    residency is the primary's, which may be a live zero-VRAM model."""
    inf, calls = _routes(monkeypatch)
    retired, other = _kept_slot(refs = 1), _kept_slot()
    monkeypatch.setattr(model_slots, "slots", [retired, other])
    monkeypatch.setattr(model_slots, "stuck", [])
    inf._release_after_terminal_retirement(retired.llama, retired.llama._unload_epoch)
    assert model_slots.slots == [other]
    assert calls == ["kept"]
    # Nothing dead is left for a later teardown to count as an unload.
    monkeypatch.setattr(model_slots, "slots", [])
    assert model_slots.unload_llama_slots() == 0


def test_a_retired_primary_clears_its_own_residency(monkeypatch):
    inf, calls = _routes(monkeypatch)
    kept = _kept_slot()
    monkeypatch.setattr(model_slots, "slots", [kept])
    _release_primary(inf)
    assert model_slots.slots == [kept]
    assert calls == ["clear:chat", "claim"]


def _cleanup_after(monkeypatch, inf, land):
    """The routes cleanup, run after ``land`` (a reload queued on _serial_load_lock)."""

    def hook(*args):
        land()
        inf._release_after_terminal_retirement(*args)

    monkeypatch.setattr(llama_cpp, "_TERMINAL_RETIREMENT_HOOKS", [hook])


def test_a_reload_before_the_cleanup_keeps_its_kept_slot(harness, monkeypatch):
    inf, calls = _routes(monkeypatch)
    backend, first, _loads = harness(_OFF, "none")
    slot = model_slots.ExtraSlot(llama = backend, orchestrator = _Orchestrator(), account = "a")
    monkeypatch.setattr(model_slots, "slots", [slot])
    monkeypatch.setattr(model_slots, "stuck", [])
    _cleanup_after(monkeypatch, inf, lambda: backend.load_model(_OFF))
    _two_strikes(backend, first, monkeypatch, [])
    assert model_slots.slots == [slot]
    assert backend._process is not None and backend._process.poll() is None
    assert calls == []


def test_a_reload_before_the_cleanup_keeps_the_primarys_residency(harness, monkeypatch):
    inf, calls = _routes(monkeypatch)
    backend, first, _loads = harness(_OFF, "none")
    monkeypatch.setattr(inf, "_llama_cpp_backend", backend)
    monkeypatch.setattr(model_slots, "slots", [])
    _cleanup_after(monkeypatch, inf, lambda: backend.load_model(_OFF))
    _two_strikes(backend, first, monkeypatch, [])
    assert backend._process is not None and backend._process.poll() is None
    assert calls == []


# A non-GGUF load that hands the GGUF primary off before the terminal cleanup.

_MLX = "mlx-community/Qwen3-4B-4bit"


def _standard_load_lands_first(monkeypatch, inf, backend, orchestrator):
    def land():
        asyncio.run(inf._unload_llama_before_standard_load(backend))
        orchestrator.active_model_name = _MLX

    _cleanup_after(monkeypatch, inf, land)


def _no_settle(monkeypatch, backend, waits):
    monkeypatch.setattr(backend, "_wait_for_vram_settle", lambda **kw: waits.append(kw))


def test_a_standard_load_before_the_cleanup_keeps_the_primarys_residency(harness, monkeypatch):
    inf, calls = _routes(monkeypatch)
    backend, first, _loads = harness(_OFF, "none")
    monkeypatch.setattr(inf, "_llama_cpp_backend", backend)
    monkeypatch.setattr(model_slots, "slots", [])
    std = _Orchestrator()
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: std)
    waits = []
    _no_settle(monkeypatch, backend, waits)
    _standard_load_lands_first(monkeypatch, inf, backend, std)
    _two_strikes(backend, first, monkeypatch, [])
    assert calls == []
    # Nothing was running, so nothing to wait for.
    assert waits == []


def test_a_standard_load_before_the_cleanup_keeps_its_kept_slot(harness, monkeypatch):
    inf, calls = _routes(monkeypatch)
    backend, first, _loads = harness(_OFF, "none")
    orchestrator = _Orchestrator()
    slot = model_slots.ExtraSlot(llama = backend, orchestrator = orchestrator, account = "a")
    monkeypatch.setattr(model_slots, "slots", [slot])
    monkeypatch.setattr(model_slots, "stuck", [])
    _no_settle(monkeypatch, backend, [])
    _standard_load_lands_first(monkeypatch, inf, backend, orchestrator)
    _two_strikes(backend, first, monkeypatch, [])
    assert model_slots.slots == [slot]
    assert orchestrator.cleaned is False
    assert calls == []


def test_the_standard_handoff_advances_the_epoch_of_an_unloaded_backend(monkeypatch):
    import routes.inference as inf

    backend = LlamaCppBackend()
    waits = []
    _no_settle(monkeypatch, backend, waits)
    epoch = backend._unload_epoch
    asyncio.run(inf._unload_llama_before_standard_load(backend))
    assert backend._unload_epoch == epoch + 1
    assert waits == []


def test_the_standard_handoff_skips_the_wait_with_nothing_running(monkeypatch):
    import routes.inference as inf

    backend = LlamaCppBackend()
    waits = []
    _no_settle(monkeypatch, backend, waits)
    asyncio.run(inf._unload_llama_before_standard_load(backend))
    assert waits == []


def test_the_standard_handoff_waits_for_a_cleanup_past_its_check(monkeypatch):
    """A cleanup holding the lock already decided the backend is still retired: the standard
    load may not publish its residency until that cleanup is done."""
    import routes.inference as inf

    backend = LlamaCppBackend()
    _no_settle(monkeypatch, backend, [])
    epoch = backend._unload_epoch
    done = threading.Event()
    with backend._while_still_retired(epoch) as retired:
        assert retired
        worker = threading.Thread(
            target = lambda: (
                asyncio.run(inf._unload_llama_before_standard_load(backend)),
                done.set(),
            )
        )
        worker.start()
        assert not done.wait(0.3)
        assert backend._unload_epoch == epoch
    worker.join(10)
    assert done.is_set() and backend._unload_epoch == epoch + 1


@pytest.mark.parametrize(
    "healthy",
    [True, False],
    ids = [
        "test_the_standard_handoff_still_waits_for_a_loaded_server_to_settle",
        # Starting or recovering: a process, not loaded.
        "test_the_standard_handoff_waits_after_killing_an_unhealthy_server",
    ],
)
def test_the_standard_handoff_waits_after_killing_a_server(harness, monkeypatch, healthy):
    import routes.inference as inf

    backend, _first, _loads = harness(_OFF, "none")
    backend._healthy = healthy
    assert backend._process is not None and backend.is_loaded is healthy
    waits = []
    _no_settle(monkeypatch, backend, waits)
    asyncio.run(inf._unload_llama_before_standard_load(backend))
    assert backend._process is None
    assert len(waits) == 1
    if not healthy:
        assert waits[0]["since_kill"] > 0


def test_the_cleanup_leaves_an_occupied_primary_seat_alone(monkeypatch):
    """No handoff ran (the epoch is still the retirement's): the non-GGUF primary still keeps
    its sharers, and the CHAT release re-checks that the seat is idle."""
    inf, calls = _routes(monkeypatch)
    monkeypatch.setattr(model_slots, "slots", [])
    for field, value in (
        ("active_model_name", _MLX),
        ("loading_models", (_MLX,)),
        ("_managed_engine", object()),
    ):
        std = _Orchestrator()
        setattr(std, field, value)
        monkeypatch.setattr(inf, "_peek_inference_backend", lambda std = std: std)
        calls.clear()
        _release_primary(inf)
        assert calls == ["claim"], field


def test_the_cleanup_leaves_a_kept_slot_with_a_non_gguf_model_alone(monkeypatch):
    inf, calls = _routes(monkeypatch)
    monkeypatch.setattr(model_slots, "stuck", [])
    for field, value in (("active_model_name", _MLX), ("loading_models", (_MLX,))):
        orchestrator = _Orchestrator()
        setattr(orchestrator, field, value)
        slot = _kept_slot(orchestrator = orchestrator)
        monkeypatch.setattr(model_slots, "slots", [slot])
        calls.clear()
        inf._release_after_terminal_retirement(slot.llama, slot.llama._unload_epoch)
        assert model_slots.slots == [slot] and orchestrator.cleaned is False, field
        assert calls == ["kept"]


def test_an_idle_primary_seat_still_clears_its_residency(monkeypatch):
    inf, calls = _routes(monkeypatch)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: _Orchestrator())
    _release_primary(inf)
    assert calls == ["clear:chat", "claim"]
