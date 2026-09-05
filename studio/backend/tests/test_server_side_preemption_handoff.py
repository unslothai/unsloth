# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio hands preemption to a llama-server that can park slots itself.

A build with ``--preempt-ram`` (unslothai/llama.cpp#184) parks a slot's sequence in host
RAM when the unified pool fills and restores it in place, byte-identically, and with the
stream notices it writes ``: preempted`` and ``: resumed`` SSE comments on the way. On
such a build the Studio-side preemption must stand down: it would abort a stream the
server was about to park in place, and re-prefill what the server would have kept. The
chat must still show the pause, and a park must never read as a stall.

On an upstream build without the flag every path here is the one that exists today.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

import httpx
import pytest

from core.inference import llama_cpp as llama_cpp_mod
from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend, _preempt_ram_disabled_in
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptSignal,
    get_preemption_controller,
    reset_preemption_controllers,
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    reset_preemption_controllers()
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    yield
    reset_preemption_controllers()


# --------------------------------------------------------------------------- the mode


class TestMode:
    def test_auto_is_server_only_when_the_build_can_park(self):
        assert preemption.resolve_preempt_mode(True) == preemption.PREEMPT_MODE_SERVER
        assert preemption.resolve_preempt_mode(False) == preemption.PREEMPT_MODE_STUDIO

    def test_studio_can_be_forced(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "studio")
        assert preemption.resolve_preempt_mode(True) == preemption.PREEMPT_MODE_STUDIO

    def test_server_cannot_be_forced_onto_a_build_that_cannot(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "server")
        assert preemption.resolve_preempt_mode(False) == preemption.PREEMPT_MODE_STUDIO
        assert preemption.resolve_preempt_mode(True) == preemption.PREEMPT_MODE_SERVER

    def test_unknown_spellings_read_as_auto(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "whatever")
        assert preemption.preempt_mode_setting() == preemption.PREEMPT_MODE_AUTO

    @pytest.mark.parametrize(
        "args, disabled",
        [
            ([], False),
            (["--preempt-ram", "8192"], False),
            (["--preempt-ram", "0"], True),
            (["--preempt-ram=0"], True),
            (["--preempt-ram", "0", "--preempt-ram", "512"], False),
            (["--preempt-ram"], False),
        ],
    )
    def test_a_hand_typed_zero_switches_parking_off(self, args, disabled):
        assert _preempt_ram_disabled_in(["llama-server", "-m", "x.gguf", *args]) is disabled


class TestBackendProperty:
    def _backend(self, *, flag, unified):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._server_preempts_kv = flag
        backend._kv_cache_unified = unified
        return backend

    def test_true_only_with_the_flag_and_a_unified_cache(self):
        assert self._backend(flag = True, unified = True).server_preempts_kv is True
        assert self._backend(flag = True, unified = False).server_preempts_kv is False
        assert self._backend(flag = False, unified = True).server_preempts_kv is False

    def test_a_backend_that_never_launched_reports_false(self):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        assert backend.server_preempts_kv is False

    def test_forcing_studio_mode_wins(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "studio")
        assert self._backend(flag = True, unified = True).server_preempts_kv is False


# ------------------------------------------------------------------- the capability probe


class TestProbe:
    def _probe(self, monkeypatch, tmp_path, help_text):
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        class _Result:
            returncode = 0
            stdout = help_text
            stderr = ""

        monkeypatch.setattr(llama_cpp_mod.subprocess, "run", lambda *a, **k: _Result())
        LlamaCppBackend._capability_cache.clear()
        return LlamaCppBackend.probe_server_capabilities(str(binary))

    _HELP = (
        "--metrics                               enable prometheus compatible metrics endpoint\n"
        "--kv-unified, -kvu                      use single unified KV buffer\n"
        "--spec-type {none,draft,draft-mtp}      speculative decoding type\n"
    )

    def test_the_flag_is_seen(self, monkeypatch, tmp_path):
        caps = self._probe(
            monkeypatch,
            tmp_path,
            self._HELP
            + "--preempt-ram N                         with a unified KV cache, park a slot "
            "in host RAM instead of failing\n",
        )
        assert caps["supports_preempt_ram"] is True
        assert caps["supports_metrics"] is True

    def test_an_upstream_build_reports_it_absent(self, monkeypatch, tmp_path):
        caps = self._probe(monkeypatch, tmp_path, self._HELP)
        assert caps["supports_preempt_ram"] is False


# ------------------------------------------------------------------------ the controller


def _fill(controller, n = 4, tokens = 2000):
    """Register `n` decoding chats. No sweep runs here: `register` never plans, so the
    caller sees the first decision itself."""
    signals = []
    for i in range(n):
        signal = PreemptSignal()
        controller.register(f"g{i}", tokens = tokens, signal = signal)
        signals.append(signal)
    return signals


class TestController:
    def test_server_mode_chooses_nobody_and_holds_nothing_back(self):
        controller = get_preemption_controller("server")
        controller.configure(
            budget = 8192, kv_unified = True, draft_tokens = 2, slots = 4, batch_tokens = 2048,
            server_mode = True,
        )
        assert controller.server_mode is True
        signals = _fill(controller, n = 4, tokens = 2400)  # 9600 against 8192
        assert controller.plan_preemptions() == []
        # The per-token sweep is the same call, and it must stay quiet too.
        for i in range(4):
            assert controller.observe(f"g{i}", 64) == []
        assert not any(s.is_set() for s in signals)
        snap = controller.snapshot()
        assert snap.mode == "server"
        assert snap.buffer == 0, "the server reserves its own drafts and margin"
        assert snap.committed == 9600 + 4 * 64

    def test_studio_mode_is_unchanged(self):
        controller = get_preemption_controller("studio")
        controller.configure(
            budget = 8192, kv_unified = True, draft_tokens = 2, slots = 4, batch_tokens = 2048,
            server_mode = False,
        )
        signals = _fill(controller, n = 4, tokens = 2400)
        victims = controller.plan_preemptions()
        assert victims, "today's behaviour: somebody must stop"
        assert any(s.is_set() for s in signals)
        assert controller.snapshot().mode == "studio"
        assert controller.snapshot().buffer > 0

    def test_configure_without_the_argument_keeps_the_mode(self):
        controller = get_preemption_controller("keep")
        controller.configure(budget = 8192, kv_unified = True, server_mode = True)
        controller.configure(budget = 8192)
        assert controller.server_mode is True

    def test_the_deferred_wrapper_forwards_the_server_hooks(self):
        """The routes hand the stream a DeferredPreemptionPolicy and bind the real one
        later, so the wrapper has to carry the two new hooks or the ledger never hears
        about a server park. Measured: four chats, one park, the client saw the pause,
        the log showed no `server-parked` line."""
        controller = get_preemption_controller("deferred")
        controller.configure(budget = 8192, kv_unified = True, slots = 4, server_mode = True)
        signal = PreemptSignal()
        controller.register("g", tokens = 100, signal = signal)
        wrapper = preemption.DeferredPreemptionPolicy()
        wrapper.on_server_parked()  # unbound: a no-op, not an error
        wrapper.bind(preemption.ControllerPreemptionPolicy(controller, "g", signal))
        wrapper.on_server_parked()
        assert controller.participant("g").state == ParticipantState.PAUSED
        wrapper.on_server_resumed()
        assert controller.participant("g").state == ParticipantState.DECODING

    def test_the_policy_marks_a_server_park_and_resume(self):
        controller = get_preemption_controller("policy")
        controller.configure(budget = 8192, kv_unified = True, slots = 4, server_mode = True)
        signal = PreemptSignal()
        controller.register("g", tokens = 100, signal = signal)
        controller.observe("g", 5)
        policy = preemption.ControllerPreemptionPolicy(controller, "g", signal)
        policy.on_server_parked()
        assert controller.participant("g").state == ParticipantState.PAUSED
        assert controller.snapshot().paused == 1
        policy.on_server_resumed()
        assert controller.participant("g").state == ParticipantState.DECODING
        assert not signal.is_set(), "a server park never sets the Studio-side signal"


# ----------------------------------------------------------------------- the stream


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n\n"


def _finish(reason: str = "stop") -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n\n"
    )


class _FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def iter_text(self):
        yield from self._chunks

    def close(self):
        pass


class _Recorder:
    """A backend whose upstream stream is a scripted list of raw SSE chunks, read through
    the REAL cancel-aware iterator so the comment lines take the real path."""

    def __init__(self, monkeypatch, chunks, *, server_preempts):
        self.payloads = []
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48851
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False
        backend._server_preempts_kv = server_preempts
        backend._kv_cache_unified = True
        self.backend = backend
        recorder = self

        @contextlib.contextmanager
        def fake_stream_with_retry(_client, _url, payload, _cancel_event, headers = None, **_kw):
            recorder.payloads.append(copy.deepcopy(payload))
            yield _FakeResponse(chunks)

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)


class _HookPolicy:
    def __init__(self):
        self.events = []

    def should_preempt(self):
        return False

    def on_preempted(self, checkpoint):
        self.events.append("preempted")

    def await_resume(self, timeout = None):
        self.events.append("awaited")
        return True

    def on_resumed(self):
        self.events.append("resumed")

    def on_server_parked(self):
        self.events.append("server-parked")

    def on_server_resumed(self):
        self.events.append("server-resumed")


def _client_view(events):
    """What a client assembles: the concatenation of snapshot diffs, plus the dict events."""
    text = ""
    marks = []
    for ev in events:
        if isinstance(ev, dict):
            if ev.get("type") == "preempt":
                marks.append((ev["state"], len(text)))
            continue
        if isinstance(ev, str) and ev.startswith(text):
            text = ev
        elif isinstance(ev, str):
            text = ev
    return text, marks


class TestTheStreamRelaysAServerPark:
    _SCRIPT = [
        _delta("Once upon"),
        ": preempted\n\n",
        ": preempt-keepalive\n\n",
        ": resumed\n\n",
        _delta(" a time"),
        _finish(),
        "data: [DONE]\n\n",
    ]

    def test_the_pause_is_shown_and_the_text_is_untouched(self, monkeypatch):
        recorder = _Recorder(monkeypatch, self._SCRIPT, server_preempts = True)
        policy = _HookPolicy()
        events = list(
            recorder.backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                cancel_event = threading.Event(),
                preempt_event = PreemptSignal(),
                preempt_policy = policy,
            )
        )
        text, marks = _client_view(events)
        assert text == "Once upon a time"
        assert marks == [("paused", len("Once upon")), ("resumed", len("Once upon"))]
        assert policy.events == ["server-parked", "server-resumed"], (
            "the ledger is told, and the Studio-side pause handshake never runs"
        )
        assert len(recorder.payloads) == 1, "nothing was re-opened: the server resumed in place"

    def test_a_policy_without_the_hooks_is_fine(self, monkeypatch):
        recorder = _Recorder(monkeypatch, self._SCRIPT, server_preempts = True)
        events = list(
            recorder.backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                cancel_event = threading.Event(),
                preempt_event = PreemptSignal(),
                preempt_policy = preemption.NullPreemptionPolicy(),
            )
        )
        text, marks = _client_view(events)
        assert text == "Once upon a time"
        assert [m[0] for m in marks] == ["paused", "resumed"]

    def test_an_upstream_stream_without_comments_is_bytewise_todays(self, monkeypatch):
        script = [c for c in self._SCRIPT if not c.startswith(":")]
        recorder = _Recorder(monkeypatch, script, server_preempts = False)
        events = list(
            recorder.backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                cancel_event = threading.Event(),
            )
        )
        text, marks = _client_view(events)
        assert text == "Once upon a time"
        assert marks == []

    def test_a_park_before_the_first_token_is_shown_too(self, monkeypatch):
        script = [": preempted\n\n", ": resumed\n\n", _delta("Hello"), _finish(), "data: [DONE]\n\n"]
        recorder = _Recorder(monkeypatch, script, server_preempts = True)
        events = list(
            recorder.backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                cancel_event = threading.Event(),
                preempt_event = PreemptSignal(),
                preempt_policy = _HookPolicy(),
            )
        )
        text, marks = _client_view(events)
        assert text == "Hello"
        assert [m[0] for m in marks] == ["paused", "resumed"]


class TestTheStreamRelaysAServerParkInTheToolLoop:
    def test_the_tool_loop_surface_relays_the_comments(self, monkeypatch):
        recorder = _Recorder(
            monkeypatch, TestTheStreamRelaysAServerPark._SCRIPT, server_preempts = True
        )
        policy = _HookPolicy()
        events = list(
            recorder.backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "hi"}],
                tools = [],
                cancel_event = threading.Event(),
                preempt_event = PreemptSignal(),
                preempt_policy = policy,
            )
        )
        preempts = [e for e in events if isinstance(e, dict) and e.get("type") == "preempt"]
        assert [e["state"] for e in preempts] == ["paused", "resumed"]
        assert all(e.get("source") == "server" for e in preempts)
        texts = [e["text"] for e in events if isinstance(e, dict) and e.get("type") == "content"]
        assert texts and texts[-1].endswith("Once upon a time")
        assert "server-parked" in policy.events and "server-resumed" in policy.events


# ---------------------------------------------------------------- the stall timeout


class _Obj:
    pass


def _install_wrapper(response, clock, silent_stream, grace):
    """Wire a fake client and pool the way `test_llama_cpp_stall_timeout` does, and
    return the wrapped read."""
    import httpcore  # noqa: F401

    inner = _Obj()
    inner._network_stream = silent_stream
    connection = _Obj()
    connection._connection = inner
    pool = _Obj()
    pool._connections = [connection]
    transport = _Obj()
    transport._pool = pool
    client = _Obj()
    client._transport = transport
    kwargs = {} if grace is None else {"stall_grace": grace}
    LlamaCppBackend._install_cancel_aware_read(client, threading.Event(), response, **kwargs)
    return silent_stream.read


class TestAParkIsNotAStall:
    """The stall lives in the read wrapper (`_install_cancel_aware_read`), below the httpx
    body iterator, because an iterator that has raised is finished and cannot be waited
    through. The wrapper is driven directly here, with a fake clock and a stream that
    never delivers."""

    _STALL = 120.0

    def _drive(self, monkeypatch, grace):
        import httpcore

        clock = {"t": 0.0}
        monkeypatch.setattr(llama_cpp_mod.time, "monotonic", lambda: clock["t"])

        def silent_read(max_bytes, timeout = None):
            clock["t"] += timeout if timeout is not None else 0.0
            raise httpcore.ReadTimeout("silence")

        stream = _Obj()
        stream.read = silent_read
        request = _Obj()
        request.extensions = {"timeout": {"read": self._STALL}}
        response = _Obj()
        response.request = request
        read = _install_wrapper(response, clock, stream, grace)
        return read, clock

    def test_without_grace_the_stall_fires_as_today(self, monkeypatch):
        import httpcore

        read, clock = self._drive(monkeypatch, None)
        with pytest.raises(httpcore.ReadTimeout):
            read(65536, timeout = 1200.0)
        assert clock["t"] == pytest.approx(self._STALL, abs = 1.0)

    def test_a_parked_slot_keeps_the_stream_alive(self, monkeypatch):
        import httpcore

        asked = []

        def grace():
            asked.append(True)
            return len(asked) < 3  # parked at the first two checks, gone at the third

        read, clock = self._drive(monkeypatch, grace)
        with pytest.raises(httpcore.ReadTimeout):
            read(65536, timeout = 1200.0)
        assert len(asked) == 3
        assert clock["t"] == pytest.approx(3 * self._STALL, abs = 1.0)

    def test_the_grace_is_bounded(self, monkeypatch):
        import httpcore

        read, clock = self._drive(monkeypatch, lambda: True)
        with pytest.raises(httpcore.ReadTimeout):
            read(65536, timeout = 1200.0)
        assert clock["t"] >= llama_cpp_mod._SERVER_PARK_STALL_CAP_S
        assert clock["t"] < llama_cpp_mod._SERVER_PARK_STALL_CAP_S + 2 * self._STALL

    def test_a_grace_that_raises_is_a_stall(self, monkeypatch):
        import httpcore

        def grace():
            raise RuntimeError("metrics down")

        read, clock = self._drive(monkeypatch, grace)
        with pytest.raises(httpcore.ReadTimeout):
            read(65536, timeout = 1200.0)
        assert clock["t"] == pytest.approx(self._STALL, abs = 1.0)

    def test_the_open_stream_hands_the_grace_down_only_when_the_build_can_park(self, monkeypatch):
        seen = []

        @contextlib.contextmanager
        def fake_stream_with_retry(_client, _url, _payload, _cancel, **kw):
            seen.append(kw)
            yield _FakeResponse([])

        for flag in (False, True):
            backend = LlamaCppBackend.__new__(LlamaCppBackend)
            backend._port = 48853
            backend._api_key = None
            backend._server_preempts_kv = flag
            backend._kv_cache_unified = True
            monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
            with backend._open_stream("http://x", {}, threading.Event()):
                pass
        assert "stall_grace" not in seen[0], "an upstream build passes nothing new"
        assert seen[1]["stall_grace"] is not None

    def test_the_backend_reads_requests_preempted(self, monkeypatch):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._port = 48852
        from core.inference import llama_stats

        monkeypatch.setattr(
            llama_stats, "scrape_llama_metrics", lambda *_a, **_k: {"requests_preempted": 1.0}
        )
        assert backend._server_park_grace() is True
        monkeypatch.setattr(
            llama_stats, "scrape_llama_metrics", lambda *_a, **_k: {"requests_preempted": 0.0}
        )
        assert backend._server_park_grace() is False
        monkeypatch.setattr(llama_stats, "scrape_llama_metrics", lambda *_a, **_k: None)
        assert backend._server_park_grace() is False


class TestAParkedSlotHoldsNoCells:
    def test_read_slot_occupancy_skips_parked_slots(self):
        from core.inference.llama_preemption import read_slot_occupancy

        slots = [
            {"id": 0, "is_processing": True, "is_preempted": False, "n_prompt_tokens": 3000},
            {"id": 1, "is_processing": True, "is_preempted": True, "n_prompt_tokens": 4000},
            {"id": 2, "is_processing": False, "is_preempted": False, "n_prompt_tokens_cache": 500},
        ]
        occupancy = read_slot_occupancy(lambda: slots)
        assert occupancy["resident"] == 3500, "the parked sequence lives in host RAM"
        assert occupancy["idle_tokens"] == 500
