# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio hands preemption to a llama-server that can park slots itself, and relays the
pause it announces all the way to the browser and to a durable run's follower."""

from __future__ import annotations

import contextlib
import pathlib
import re
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import llama_cpp as llama_cpp_mod
from core.inference import llama_preemption as preemption
from core.inference.chat_generation_runs import _admission_status_chunks
from core.inference.llama_cpp import LlamaCppBackend, _preempt_ram_disabled_in
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptSignal,
    get_preemption_controller,
    reset_preemption_controllers,
)
import routes.inference as inference_route

from .llama_backend_double import FakeLlamaCppBackend
from .preempt_fakes import (
    bare_backend,
    FakeResponse as _FakeResponse,
    PreemptRecorder,
    ServerHookPolicy as _HookPolicy,
    delta as preempt_fakes_delta,
    finish as preempt_fakes_finish,
)


def _delta(content: str) -> str:
    return preempt_fakes_delta(content, terminator = "\n\n")


def _finish(reason: str = "stop") -> str:
    return preempt_fakes_finish(reason, terminator = "\n\n")


def _Recorder(monkeypatch, chunks, *, server_preempts):
    return PreemptRecorder(
        monkeypatch,
        [chunks],
        patch_iter = False,
        response_factory = _FakeResponse,
        _server_preempts_kv = server_preempts,
        _kv_cache_unified = True,
    )


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    reset_preemption_controllers()
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    yield
    reset_preemption_controllers()


class TestMode:
    def test_auto_is_server_only_when_the_build_can_park(self):
        assert preemption.resolve_preempt_mode(True) == preemption.PREEMPT_MODE_SERVER
        assert preemption.resolve_preempt_mode(False) == preemption.PREEMPT_MODE_STUDIO

    def test_studio_can_be_forced_and_server_cannot_be_forced_onto_a_build_that_cannot(
        self, monkeypatch
    ):
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "studio")
        assert preemption.resolve_preempt_mode(True) == preemption.PREEMPT_MODE_STUDIO
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "server")
        assert preemption.resolve_preempt_mode(False) == preemption.PREEMPT_MODE_STUDIO
        assert preemption.resolve_preempt_mode(True) == preemption.PREEMPT_MODE_SERVER
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
        assert LlamaCppBackend.__new__(LlamaCppBackend).server_preempts_kv is False

    def test_forcing_studio_mode_wins(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "studio")
        assert self._backend(flag = True, unified = True).server_preempts_kv is False


def _fill(
    controller,
    n = 4,
    tokens = 2000,
):
    signals = []
    for i in range(n):
        signal = PreemptSignal()
        controller.register(f"g{i}", tokens = tokens, signal = signal)
        signals.append(signal)
    return signals


class TestController:
    def _configured(self, key, *, server_mode):
        controller = get_preemption_controller(key)
        controller.configure(
            budget = 8192,
            kv_unified = True,
            draft_tokens = 2,
            slots = 4,
            batch_tokens = 2048,
            server_mode = server_mode,
        )
        return controller

    def test_server_mode_chooses_nobody_and_holds_nothing_back(self):
        controller = self._configured("server", server_mode = True)
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

    def test_a_server_park_and_resume_move_the_ledger_without_the_studio_signal(self):
        controller = self._configured("policy", server_mode = True)
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


def _client_view(events):
    text = ""
    marks = []
    for ev in events:
        if isinstance(ev, dict):
            if ev.get("type") == "preempt":
                marks.append((ev["state"], len(text)))
        elif isinstance(ev, str):
            text = ev
    return text, marks


_SCRIPT = [
    _delta("Once upon"),
    ": preempted\n\n",
    ": preempt-keepalive\n\n",
    ": resumed\n\n",
    _delta(" a time"),
    _finish(),
    "data: [DONE]\n\n",
]


class TestTheStreamRelaysAServerPark:
    def _plain(
        self,
        monkeypatch,
        script,
        *,
        policy,
        server_preempts = True,
    ):
        recorder = _Recorder(monkeypatch, script, server_preempts = server_preempts)
        events = list(
            recorder.backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                cancel_event = threading.Event(),
                **(
                    {}
                    if policy is None
                    else {"preempt_event": PreemptSignal(), "preempt_policy": policy}
                ),
            )
        )
        return recorder, events

    def test_the_pause_is_shown_and_the_text_is_untouched(self, monkeypatch):
        policy = _HookPolicy()
        recorder, events = self._plain(monkeypatch, _SCRIPT, policy = policy)
        text, marks = _client_view(events)
        assert text == "Once upon a time"
        assert marks == [
            ("paused", len("Once upon")),
            ("keepalive", len("Once upon")),
            ("resumed", len("Once upon")),
        ]
        assert policy.events == [
            "server-parked",
            "server-resumed",
        ], "the ledger is told, and the Studio-side pause handshake never runs"
        assert len(recorder.payloads) == 1, "nothing was re-opened: the server resumed in place"

    def test_the_tool_loop_surface_relays_the_comments_too(self, monkeypatch):
        recorder = _Recorder(monkeypatch, _SCRIPT, server_preempts = True)
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
        assert [e["state"] for e in preempts] == ["paused", "keepalive", "resumed"]
        assert all(e.get("source") == "server" for e in preempts)
        texts = [e["text"] for e in events if isinstance(e, dict) and e.get("type") == "content"]
        assert texts and texts[-1].endswith("Once upon a time")
        assert "server-parked" in policy.events and "server-resumed" in policy.events


class _PausingBackend(FakeLlamaCppBackend):
    context_length = 8192
    _META = {
        "type": "metadata",
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        "timings": {"prompt_n": 11, "predicted_n": 5},
        "finish_reason": "stop",
    }

    def generate_chat_completion_with_tools(self, **kwargs):
        yield {"type": "content", "text": "Introduction: The"}
        yield {"type": "preempt", "state": "paused"}
        yield {"type": "preempt", "state": "resumed"}
        yield {"type": "content", "text": "Introduction: The Paradigm"}
        yield self._META

    def generate_chat_completion(self, **kwargs):
        yield "Introduction: The"
        yield {"type": "preempt", "state": "paused"}
        yield {"type": "preempt", "state": "resumed"}
        yield "Introduction: The Paradigm"
        yield self._META


class TestThePauseReachesTheClient:
    """The GUI path swallowed the pause: a half-written answer stopping dead with no
    explanation is indistinguishable from a wedged backend."""

    def _body(self, monkeypatch, *, tools):
        backend = _PausingBackend()
        backend.supports_tools = tools
        monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: tools)

        async def _fake_select(payload, **_kwargs):
            return [{"type": "function", "function": {"name": "python"}}]

        monkeypatch.setattr(inference_route, "_select_request_tools", _fake_select)
        app = FastAPI()
        app.include_router(inference_route.router)
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
        body = {"messages": [{"role": "user", "content": "write me an essay"}], "stream": True}
        if tools:
            body["enable_tools"] = True
        # The Studio UI's opt-in: tools with confirmation are refused without it.
        response = TestClient(app, headers = {"X-Unsloth-Events": "1"}).post(
            "/chat/completions", json = body
        )
        assert response.status_code == 200
        return response.text

    @pytest.mark.parametrize("tools", [True, False])
    def test_the_pause_and_the_resume_are_announced_around_the_text(self, monkeypatch, tools):
        body = self._body(monkeypatch, tools = tools)
        assert ": preempt-paused" in body
        assert body.index(": preempt-paused") < body.index(": preempt-resumed")
        assert "Introduction: The" in body and " Paradigm" in body
        assert "data: [DONE]" in body


class TestADurableRunRelaysThePause:
    """The GUI streams plain chats through a durable run, whose worker reads the events."""

    def test_the_two_comments_become_status_chunks_in_order(self):
        assert _admission_status_chunks(": preempt-paused\n\n") == [{"_admissionStatus": "paused"}]
        assert _admission_status_chunks(": preempt-resumed\n\n") == [
            {"_admissionStatus": "resumed"}
        ]
        both = ": preempt-paused\n\n: preempt-resumed\r\n\r\n"
        assert [c["_admissionStatus"] for c in _admission_status_chunks(both)] == [
            "paused",
            "resumed",
        ]
        glued = 'data: {"choices":[{"delta":{"content":"x"}}]}\n\n: preempt-paused\n\n'
        assert _admission_status_chunks(glued) == [{"_admissionStatus": "paused"}]
        # Renewed on, never relayed: a keepalive is progress for the lease, not a status.
        assert _admission_status_chunks(": preempt-keepalive\n\n") == []


class TestEverySignalTheClientReadsHasAProducer:
    """The one cross-language contract here: the client understands four comments, and a
    signal it can read that the server never sends is a user staring at a stopped answer."""

    # No behavioural reach: the producers are asserted elsewhere in this file, but only
    # the frontend source says which comments the client is prepared to read.
    def test_the_frontend_declares_exactly_the_four_comments_the_route_emits(self):
        backend_dir = pathlib.Path(__file__).resolve().parent.parent
        ts = backend_dir.parent / "frontend/src/features/chat/utils/admission-status.ts"
        if not ts.exists():
            pytest.skip("frontend not present in this tree")
        pattern = r'^export const ADMISSION_COMMENT_\w+ = "([^"]+)";'
        declared = set(re.findall(pattern, ts.read_text(encoding = "utf-8"), re.M))
        assert declared == {
            "admission-wait",
            "admission-done",
            "preempt-paused",
            "preempt-resumed",
        }


class _Obj:
    pass


class TestTheParkedMetricsProbeIsAuthenticated:
    def test_the_probe_carries_the_key(self, monkeypatch):
        seen = {}

        def scrape(
            base_url,
            timeout_s = 3.0,
            headers = None,
        ):
            seen["headers"] = headers
            return {"requests_preempted": 1.0}

        monkeypatch.setattr("core.inference.llama_stats.scrape_llama_metrics", scrape)
        backend = bare_backend(_api_key = "sk-test")
        assert backend._server_park_grace() is True
        assert seen["headers"] == {"Authorization": "Bearer sk-test"}


class TestAParkIsNotAStall:
    """The stall lives in the read wrapper, below the httpx iterator that cannot resume."""

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

        inner = _Obj()
        inner._network_stream = stream
        connection = _Obj()
        connection._connection = inner
        pool = _Obj()
        pool._connections = [connection]
        transport = _Obj()
        transport._pool = pool
        client = _Obj()
        client._transport = transport
        LlamaCppBackend._install_cancel_aware_read(
            client, threading.Event(), response, **({} if grace is None else {"stall_grace": grace})
        )
        return stream.read, clock

    def _read_until_stall(self, read):
        import httpcore
        with pytest.raises(httpcore.ReadTimeout):
            read(65536, timeout = 1200.0)

    def test_a_parked_slot_keeps_the_stream_alive(self, monkeypatch):
        asked = []

        def grace():
            asked.append(True)
            return len(asked) < 3  # parked at the first two checks, gone at the third

        read, clock = self._drive(monkeypatch, grace)
        self._read_until_stall(read)
        assert len(asked) == 3
        assert clock["t"] == pytest.approx(3 * self._STALL, abs = 1.0)

    def test_the_grace_is_measured_from_the_deadline_it_first_crossed(self, monkeypatch):
        # Parked for good: the wait is the stall window and then the whole grace, not the
        # grace less the window it took to reach the deadline.
        read, clock = self._drive(monkeypatch, lambda: True)
        self._read_until_stall(read)
        cap = llama_cpp_mod._SERVER_PARK_STALL_CAP_S
        assert clock["t"] == pytest.approx(self._STALL + cap, abs = 1.0)

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
