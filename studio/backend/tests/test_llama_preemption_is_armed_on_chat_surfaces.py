# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every local chat surface that takes an admission lease must also arm preemption, hand
the generator the signal and drop the charge again on the way out."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import llama_admission
from core.inference.llama_admission import LlamaAdmissionConfig, get_llama_admission_queue
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptSignal,
    PreemptionController,
    get_preemption_controller,
    preemption_buffer_tokens,
)
from models.inference import AnthropicMessagesRequest
import routes.inference as inference

from .asgi_stream_helpers import wait_for_frame
from .llama_backend_double import FakeLlamaCppBackend
from .preempt_fakes import clean_admission_queues, clean_preemption_registry  # noqa: F401


BASE = "http://llama.test"


def _backend(*, window = 16384, slots = 4, unified = True, url = "http://127.0.0.1:1/"):
    return SimpleNamespace(
        base_url = url,
        context_length = window,
        _kv_cache_context_total = window,
        _kv_cache_unified = unified,
        effective_parallel_slots = slots,
    )


class _ArmedBackend(FakeLlamaCppBackend):
    """Records what the route hands each generator, and finishes normally."""

    base_url = BASE
    effective_parallel_slots = 4
    context_length = 8192
    _kv_cache_unified = True
    _kv_cache_context_total = 8192

    def __init__(self, *, tools = False, raises = False):
        self.supports_tools = tools
        self.seen: list[dict] = []
        self._raises = raises

    _META = {
        "type": "metadata",
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "timings": {"prompt_n": 3, "predicted_n": 1},
        "finish_reason": "stop",
    }

    def generate_chat_completion(self, **kwargs):
        self.seen.append(kwargs)
        yield "hi"
        if self._raises:
            raise RuntimeError("the stream died mid answer")
        yield self._META

    def generate_chat_completion_with_tools(self, **kwargs):
        self.seen.append(kwargs)
        yield {"type": "content", "text": "hi"}
        if self._raises:
            raise RuntimeError("the stream died mid answer")
        yield self._META


@pytest.fixture
def ledger(monkeypatch):
    """A controller double: every registration and every drop, in order."""
    events: list[tuple[str, str]] = []
    real_register = PreemptionController.register
    real_unregister = PreemptionController.unregister
    real_probe = PreemptionController.set_residency_probe

    def _register(self, gen_id, **kwargs):
        events.append(("register", gen_id))
        return real_register(self, gen_id, **kwargs)

    def _unregister(self, gen_id):
        events.append(("unregister", gen_id))
        return real_unregister(self, gen_id)

    def _set_probe(self, probe):
        events.append(("probe", "" if probe is None else "set"))
        return real_probe(self, probe)

    monkeypatch.setattr(PreemptionController, "register", _register)
    monkeypatch.setattr(PreemptionController, "unregister", _unregister)
    monkeypatch.setattr(PreemptionController, "set_residency_probe", _set_probe)
    return events


def _client(monkeypatch, backend, *, reraise = True):
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        inference, "_effective_enable_tools", lambda payload: bool(backend.supports_tools)
    )

    async def _fake_select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "python"}}]

    monkeypatch.setattr(inference, "_select_request_tools", _fake_select)
    app = FastAPI()
    app.include_router(inference.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app, raise_server_exceptions = reraise)


def _post(monkeypatch, backend, *, stream, reraise = True):
    body = {"messages": [{"role": "user", "content": "write me an essay"}], "stream": stream}
    if backend.supports_tools:
        body["enable_tools"] = True
    return _client(monkeypatch, backend, reraise = reraise).post("/chat/completions", json = body)


class TestEveryLocalChatSurfaceArms:
    """A surface that takes a lease and does not arm decodes with no preemption at all,
    and the absence is invisible to every test of what it produces."""

    @pytest.mark.parametrize("tools", [False, True])
    @pytest.mark.parametrize("stream", [False, True])
    def test_it_registers_hands_over_the_signal_and_drops_the_charge(
        self, monkeypatch, ledger, tools, stream
    ):
        backend = _ArmedBackend(tools = tools)
        response = _post(monkeypatch, backend, stream = stream)
        assert response.status_code == 200

        registered = [gen for kind, gen in ledger if kind == "register"]
        dropped = [gen for kind, gen in ledger if kind == "unregister"]
        assert registered, f"tools={tools} stream={stream} took a lease and never armed"
        assert set(registered) <= set(dropped), (
            f"registered {registered} but only dropped {dropped}: a registration that is "
            "never dropped grows the ledger forever, and once it reads full the next chat "
            "waits for room that cannot arrive"
        )
        assert ("probe", "set") in ledger, (
            "a surface that arms without a residency probe can livelock its own resume"
        )

        kwargs = backend.seen[0]
        assert kwargs.get("preempt_event") is not None
        assert kwargs.get("preempt_policy") is not None
        # Armed without this, preemption is inert: four chats reached 16354 of a 16384
        # cache with zero evictions because nothing told the controller they had grown.
        assert callable(kwargs.get("on_tokens"))
        assert get_preemption_controller(BASE).snapshot().committed == 0

    @pytest.mark.parametrize("tools", [False, True])
    def test_a_stream_that_dies_still_drops_the_charge(self, monkeypatch, ledger, tools):
        backend = _ArmedBackend(tools = tools, raises = True)
        # The surface may end the stream or let the error out; either way the charge
        # goes back, so the error is not what is under test here.
        _post(monkeypatch, backend, stream = True, reraise = False)
        registered = [gen for kind, gen in ledger if kind == "register"]
        dropped = [gen for kind, gen in ledger if kind == "unregister"]
        assert registered and set(registered) <= set(dropped), (
            "the disarm does not sit in a finally, so the gave-up, stream-error and "
            "disconnect paths keep their charge forever"
        )
        assert get_preemption_controller(BASE).snapshot().committed == 0

    def test_the_anthropic_surface_arms_too(self, monkeypatch, ledger):
        backend = _ArmedBackend(tools = True)
        backend.supports_tool_passthrough = False
        backend.is_vision = False
        backend.count_chat_tokens = lambda *a, **k: 2
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)

        class _Request:
            state = SimpleNamespace()
            url = SimpleNamespace(path = "/v1/messages")
            method = "POST"

            async def is_disconnected(self):
                return False

        async def _run():
            payload = AnthropicMessagesRequest(
                max_tokens = 16, messages = [{"role": "user", "content": "hi"}]
            )
            response = await inference.anthropic_messages(
                payload, request = _Request(), current_subject = "t"
            )
            assert response.status_code == 200

        asyncio.run(_run())
        registered = [gen for kind, gen in ledger if kind == "register"]
        dropped = [gen for kind, gen in ledger if kind == "unregister"]
        assert registered, "the Anthropic surface took a lease and never armed"
        assert set(registered) <= set(dropped)
        kwargs = backend.seen[0]
        assert kwargs.get("preempt_event") is not None
        assert kwargs.get("preempt_policy") is not None
        assert callable(kwargs.get("on_tokens"))


class TestAQueuedChatIsArmedOnceItIsGranted:
    """Four tool chats at -c 8192 armed one and lost three to `Context size has been
    exceeded`: the arm beside the reservation finds no lease, and nothing armed after the wait."""

    class _OneSlot(_ArmedBackend):
        effective_parallel_slots = 1
        context_length = 8192

    def test_the_arm_after_the_admission_wait_has_the_lease(self, monkeypatch):
        backend = self._OneSlot(tools = True)
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(inference, "_effective_enable_tools", lambda payload: True)

        async def _fake_select(payload, **_kwargs):
            return [{"type": "function", "function": {"name": "python"}}]

        monkeypatch.setattr(inference, "_select_request_tools", _fake_select)

        arms: list[bool] = []
        real_arm = inference._openai_llama_preemption_arm

        def _recording_arm(*, reservation, **kwargs):
            arms.append(reservation is not None and reservation.lease_nowait() is not None)
            return real_arm(reservation = reservation, **kwargs)

        monkeypatch.setattr(inference, "_openai_llama_preemption_arm", _recording_arm)

        app = FastAPI()
        app.include_router(inference.router)
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "enable_tools": True,
            }
        ).encode()

        async def _drive():
            # Somebody else holds the only slot, so the request below must queue.
            holder = (
                get_llama_admission_queue(BASE)
                .reserve(capacity = 1, config = LlamaAdmissionConfig(max_queue = 4))
                .lease_nowait()
            )
            assert holder is not None
            queued, done = asyncio.Event(), asyncio.Event()
            frames: list[str] = []

            async def receive():
                if not frames:
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                if message.get("type") == "http.response.body":
                    chunk = message.get("body", b"").decode()
                    frames.append(chunk)
                    if "admission-wait" in chunk:
                        queued.set()
                    if chunk == inference._SSE_DONE_CHUNK:
                        done.set()

            task = asyncio.create_task(app(_scope(app, body), receive, send))
            try:
                await wait_for_frame(queued, task, what = "the admission-wait comment")
                assert arms and not arms[0], "the first arm should have found no lease yet"
                holder.release()
                await wait_for_frame(done, task, what = "the [DONE] frame")
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions = True)

        asyncio.run(_drive())
        assert any(arms), (
            "the chat was granted after waiting and never armed with its lease: it decodes "
            "outside the preemptor's ledger, which is the three-of-four-dead case"
        )


def _scope(app, body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/completions",
        "raw_path": b"/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app,
    }


class TestARawPassthroughIsCountedAndNeverChosen:
    """It holds real cells and cannot be paused, so a controller that cannot see it
    evicts a chat to make room a passthrough is holding."""

    def test_it_is_registered_at_its_size_and_the_planner_leaves_it_alone(self):
        backend = _backend(url = "http://127.0.0.1:30/")
        controller = get_preemption_controller("http://127.0.0.1:30/")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        inference._openai_llama_count_raw_holder(
            llama_backend = backend,
            lease = SimpleNamespace(tokens = 9000),
            gen_id = "raw",
        )
        assert controller.participant("raw").state == ParticipantState.STREAMING_RAW
        assert controller.committed_tokens() == 9000, "a holder the watermark cannot see"
        controller.register("chat", tokens = 6000, signal = PreemptSignal())
        assert [v.gen_id for v in controller.plan_preemptions(needed = 4000)] == ["chat"]

    def test_a_surface_without_a_lease_is_not_counted(self):
        inference._openai_llama_count_raw_holder(
            llama_backend = _backend(url = "http://127.0.0.1:31/"), lease = None, gen_id = "raw"
        )
        assert get_preemption_controller("http://127.0.0.1:31/").participant("raw") is None


class TestArmingItself:
    def _reserve(self, url, tokens = 4096):
        return get_llama_admission_queue(url).reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = tokens
        )

    def _arm(self, url, backend = None, gen_id = "armed", signal = None):
        return inference._openai_llama_preemption_arm(
            request = None,
            llama_backend = backend or _backend(url = url),
            reservation = self._reserve(url),
            gen_id = gen_id,
            signal = signal or PreemptSignal(),
            loop = None,
        )

    @pytest.mark.asyncio
    async def test_the_charge_and_the_signal_reach_the_controller(self):
        url = "http://127.0.0.1:1/"
        signal = PreemptSignal()
        assert self._arm(url, signal = signal) is not None, "the route did not arm a policy"
        controller = get_preemption_controller(url)
        snapshot = controller.snapshot()
        assert snapshot.committed == 4096, "the real charge did not reach the controller"
        # The budget is the cache llama-server was launched with. Reducing it here to
        # reserve the drafts double-counts them: the watermark buffer holds them back.
        assert snapshot.budget == inference._openai_llama_admission_budget(_backend()) == 16384
        assert 0 < snapshot.buffer < snapshot.budget, (
            "nothing is held back, so the cache can be worked to its last cell"
        )
        assert controller.participant("armed").preempt_event is signal, (
            "the participant holds a different signal than the stream polls, so a preempt "
            "can never reach the stream"
        )

    @pytest.mark.asyncio
    async def test_a_private_cache_per_slot_is_not_armed(self):
        url = "http://127.0.0.1:2/"
        assert self._arm(url, backend = _backend(unified = False, url = url)) is None

    @pytest.mark.asyncio
    async def test_the_rollout_switch_arms_nothing(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        assert self._arm("http://127.0.0.1:3/") is None

    @pytest.mark.asyncio
    async def test_the_drafters_cells_are_reserved_on_top_of_the_batch(self):
        url = "http://127.0.0.1:9/"
        backend = _backend(url = url)
        backend._speculative_type = "draft-mtp"
        backend._spec_draft_n_max = 2
        self._arm(url, backend = backend, gen_id = "spec-armed")
        snapshot = get_preemption_controller(url).snapshot()
        # batch_tokens defaults to llama.cpp's 2048 here, and `pending_prefill` is the 4096
        # this generation was just charged: arming registers a prompt not yet prefilled.
        assert snapshot.buffer == preemption_buffer_tokens(
            snapshot.budget, draft_tokens = 2, slots = 4, batch_tokens = 2048, pending_prefill = 4096
        ), f"the drafter's tokens were not reserved (buffer {snapshot.buffer})"
        assert snapshot.buffer > preemption_buffer_tokens(
            snapshot.budget, slots = 4, batch_tokens = 2048, pending_prefill = 4096
        ), "the drafter must cost something over the same launch without one"

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"_speculative_type": "draft-mtp", "_spec_draft_n_max": 2}, 2),
            # None means the platform default, not zero.
            ({"speculative_type": "draft-mtp", "spec_draft_n_max": None}, 6),
            ({}, 0),
        ],
    )
    def test_the_draft_depth_is_read_off_whichever_field_the_load_set(self, fields, expected):
        backend = _backend()
        for name in (
            "speculative_type",
            "spec_drafter_kind",
            "requested_spec_mode",
            "_speculative_type",
            "spec_draft_n_max",
            "_spec_draft_n_max",
        ):
            setattr(backend, name, None)
        for name, value in fields.items():
            setattr(backend, name, value)
        assert inference._openai_llama_speculative_draft_tokens(backend) == expected

    @pytest.mark.parametrize(
        ("backend", "expected"),
        [
            (SimpleNamespace(requested_n_batch = 512, _requested_n_batch = 512), 512),
            (SimpleNamespace(_requested_n_batch = 512), 512),
            (SimpleNamespace(requested_n_batch = None, _requested_n_batch = None), 2048),
        ],
    )
    def test_the_batch_size_is_read_off_the_launch(self, backend, expected):
        """A `--batch-size 512` load reserved for 2048, the whole cache's worth."""
        assert inference._openai_llama_effective_batch_tokens(backend) == expected


# ------------------------------------------------------------------ dropping the charge


class _Lease:
    def __init__(self, tokens = 2000):
        self.tokens = tokens
        self.slot = 0
        self.finished = False


class _DisarmBackend:
    base_url = "http://127.0.0.1:65001"
    _kv_cache_unified = True
    context_length = 16384


@pytest.fixture
def erasures(monkeypatch):
    seen = []

    monkeypatch.setattr(
        inference,
        "fetch_llama_slots",
        lambda base, headers = None: [{"id": 0, "is_processing": False, "n_ctx_used": 2000}],
    )
    monkeypatch.setattr(
        inference,
        "read_slot_occupancy",
        lambda scrape: {"idle": [(0, 2000)], "resident": 2000, "idle_tokens": 2000},
    )

    def _reclaim(occupancy, erase, *, needed = 0):
        for slot_id, _tokens in occupancy.get("idle", []):
            seen.append(slot_id)
            erase(slot_id)
        return 2000

    monkeypatch.setattr(inference, "reclaim_idle_slots", _reclaim)
    monkeypatch.setattr(inference, "erase_llama_slot", lambda base, slot_id, headers = None: True)
    return seen


@pytest.fixture
def disarm_controller(monkeypatch):
    made = PreemptionController(_DisarmBackend.base_url)
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: made)
    monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
    return made


def _disarm(gen_id = "leaving"):
    inference._openai_llama_preemption_disarm(llama_backend = _DisarmBackend(), gen_id = gen_id)


class TestTheDisarmKeepsThePrefixCache:
    """Dropping the charge while llama-server still holds the cells is worse than
    dropping neither, and erasing them when nobody wants them costs the next turn."""

    def test_a_lone_finished_chat_keeps_its_cells_and_the_ledger_is_still_exact(
        self, disarm_controller, erasures
    ):
        disarm_controller.register("only-chat", lease = _Lease(), tokens = 2000)
        _disarm("only-chat")
        snapshot = disarm_controller.snapshot()
        assert snapshot.committed == 0, "the charge must still be dropped"
        assert snapshot.decoding == 0
        assert erasures == [], (
            "the prompt cache turn two would have reused was erased. This is the "
            "cached_tokens=0 failure, reproduced without a server."
        )

    def test_a_lone_chat_that_reported_residency_keeps_its_cells_too(
        self, disarm_controller, erasures
    ):
        disarm_controller.register("only-chat", lease = _Lease(), tokens = 2000)
        disarm_controller.note_resident(2000, 2000)
        _disarm("only-chat")
        assert erasures == [], (
            "the cached residency is this chat's own cells; judged by it, a lone chat "
            "erased its prompt cache on every turn"
        )

    @pytest.mark.parametrize(
        "state",
        [ParticipantState.PAUSED, ParticipantState.DECODING, ParticipantState.STREAMING_RAW],
    )
    def test_anybody_else_holding_or_waiting_gets_the_room(
        self, disarm_controller, erasures, state
    ):
        disarm_controller.register("leaving", lease = _Lease(), tokens = 2000)
        disarm_controller.register("other", lease = _Lease(), tokens = 3000, state = state)
        _disarm()
        assert erasures == [0], (
            "somebody wants the cells and the finished chat's were left in place, which "
            "is the livelock this reclaim exists to end"
        )

    def test_a_request_waiting_at_admission_counts_too(
        self, disarm_controller, erasures, monkeypatch
    ):
        class _Queue:
            def snapshot(self):
                return SimpleNamespace(queued = 1)

        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda key: _Queue())
        disarm_controller.register("leaving", lease = _Lease(), tokens = 2000)
        _disarm()
        assert erasures == [0], "somebody is queued for the room and the cells were kept"

    def test_neither_half_can_fail_a_response_that_already_succeeded(
        self, disarm_controller, erasures, monkeypatch
    ):
        def _boom(key):
            raise RuntimeError("no queue")

        monkeypatch.setattr(inference, "get_llama_admission_queue", _boom)
        disarm_controller.register("leaving", lease = _Lease(), tokens = 2000)
        _disarm()
        assert erasures == [], "nobody wants the room, the prefix cache stays"

        monkeypatch.setattr(
            inference,
            "get_preemption_controller",
            lambda key: (_ for _ in ()).throw(RuntimeError("no controller")),
        )
        _disarm("x")

    def test_a_backend_without_a_base_url_is_swallowed(self, disarm_controller, erasures):
        class _Headless:
            base_url = ""
            _kv_cache_unified = True

        inference._openai_llama_preemption_disarm(llama_backend = _Headless(), gen_id = "x")
        assert erasures == []


# --------------------------------------------------------------- the residency sweep


class TestTheResidencySweep:
    """Reclaiming only once a victim had been chosen made the erase almost useless."""

    def _observer(self, monkeypatch, slots, *, controller, gen_id = "chat"):
        backend = SimpleNamespace(base_url = BASE, _auth_headers = None)
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
        monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: slots)
        erased: list[int] = []

        def _erase(base, slot_id, headers = None):
            erased.append(slot_id)
            return next(s["n_prompt_tokens_cache"] for s in slots if s["id"] == slot_id)

        monkeypatch.setattr(inference, "erase_llama_slot", _erase)
        refresh, observe, note_state = inference._openai_llama_residency_observer(
            llama_backend = backend, completion_id = gen_id
        )
        return refresh, observe, note_state, erased

    def test_idle_residue_is_reclaimed_on_sight_for_a_waiter_and_the_figure_corrected(
        self, monkeypatch
    ):
        controller = PreemptionController(BASE)
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("waiter", tokens = 5000, signal = PreemptSignal())
        controller.set_state("waiter", ParticipantState.PAUSED)
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens_cache": 9000},
            {"id": 1, "is_processing": True, "n_prompt_tokens_cache": 2000},
        ]
        refresh, _observe, _note, erased = self._observer(
            monkeypatch, slots, controller = controller, gen_id = "waiter"
        )
        refresh(controller, force = True)
        assert erased == [0], (
            "somebody is paused for want of cells and the idle residue was kept until the "
            "cache went over its ceiling, by which time llama-server is already retrying"
        )
        assert controller.room_for("waiter", 5000) is True, (
            "the controller kept planning against the pre-erase figure"
        )

    def test_nothing_is_erased_while_nobody_waits_and_the_cache_is_under_its_ceiling(
        self, monkeypatch
    ):
        controller = PreemptionController(BASE)
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("chat", tokens = 2000, signal = PreemptSignal())
        slots = [{"id": 0, "is_processing": False, "n_prompt_tokens_cache": 900}]
        refresh, _observe, _note, erased = self._observer(
            monkeypatch, slots, controller = controller
        )
        refresh(controller, force = True)
        assert erased == []

    def test_the_sweep_reports_growth_and_reclaims_before_it_pauses_a_live_chat(
        self, monkeypatch
    ):
        controller = PreemptionController(BASE)
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("chat", tokens = 15000, signal = PreemptSignal())
        controller.register("other", tokens = 900, signal = PreemptSignal())
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens_cache": 8000},
            {"id": 1, "is_processing": True, "n_prompt_tokens_cache": 8000},
        ]
        _refresh, observe, _note, erased = self._observer(
            monkeypatch, slots, controller = controller
        )
        observe(2000)
        assert erased == [0], "a live chat was paused without first freeing dead residue"

    def test_note_state_reads_the_ledger_not_a_mirror(self, monkeypatch):
        controller = PreemptionController(BASE)
        controller.configure(budget = 8192, kv_unified = True, slots = 4, batch_tokens = 2048)
        _refresh, _observe, note_state, _erased = self._observer(
            monkeypatch, [], controller = controller
        )
        chat = controller.register("chat", tokens = 1000)
        assert note_state(ParticipantState.TOOLS_RUNNING) == ParticipantState.TOOLS_RUNNING
        assert chat.state == ParticipantState.TOOLS_RUNNING
        # Tokens arrive: the ledger says DECODING on its own.
        controller.observe("chat", 32)
        assert note_state(None) == ParticipantState.DECODING
        # The next tool start is not a repeat of the earlier report.
        assert note_state(ParticipantState.TOOLS_RUNNING) == ParticipantState.TOOLS_RUNNING
        assert chat.state == ParticipantState.TOOLS_RUNNING


class TestTheRouteReportsWhereTheChatIs:
    """A chat stopped on a tool approval stayed DECODING and kept counting after its
    idle slot had been erased."""

    class _ApprovalBackend(FakeLlamaCppBackend):
        base_url = BASE
        effective_parallel_slots = 2
        supports_tools = True
        context_length = 8192

        def _maybe_recover_from_mtp_crash(self, exc):
            return None

        def generate_chat_completion_with_tools(self, **kwargs):
            yield {"type": "content", "text": "Let me check."}
            yield {
                "type": "tool_start",
                "tool_call_id": "call_1",
                "name": "write_file",
                "arguments": {},
                "awaiting_confirmation": True,
            }
            yield {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "name": "write_file",
                "result": "ok",
            }
            yield {"type": "content", "text": "Let me check. Done."}
            yield _ArmedBackend._META

    def test_the_approval_prompt_and_the_next_round_are_reported(self, monkeypatch):
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: self._ApprovalBackend())
        monkeypatch.setattr(inference, "_effective_enable_tools", lambda payload: True)

        async def _fake_select(payload, **_kwargs):
            return [{"type": "function", "function": {"name": "write_file"}}]

        monkeypatch.setattr(inference, "_select_request_tools", _fake_select)
        reported: list[str] = []
        real_note_state = PreemptionController.note_state

        def _recording(self, gen_id, state):
            reported.append(state)
            return real_note_state(self, gen_id, state)

        monkeypatch.setattr(PreemptionController, "note_state", _recording)

        app = FastAPI()
        app.include_router(inference.router)
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "save it"}],
                "stream": True,
                "enable_tools": True,
            }
        ).encode()

        async def _drive():
            done = asyncio.Event()
            frames: list[str] = []

            async def receive():
                if not frames:
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                if message.get("type") == "http.response.body":
                    chunk = message.get("body", b"").decode()
                    frames.append(chunk)
                    if chunk == inference._SSE_DONE_CHUNK:
                        done.set()

            task = asyncio.create_task(app(_scope(app, body), receive, send))
            try:
                await wait_for_frame(done, task, what = "the [DONE] frame")
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions = True)
            return "".join(frames)

        assert "Done." in asyncio.run(_drive())
        assert reported[:3] == [
            ParticipantState.PARKED_ON_TOOL,
            ParticipantState.TOOLS_RUNNING,
            ParticipantState.DECODING,
        ], reported
        assert get_preemption_controller(BASE).snapshot().parked == 0, (
            "the chat is gone from the ledger when it ends"
        )


class TestTheRespawnRetryKeepsItsControls:
    """`_respawn_if_dead()` re-opens the same generation against a replacement server.
    Reproducing it needs a llama-server that dies and comes back, so this is the one
    check here that reads the source rather than driving the path."""

    def test_the_connect_error_retry_forwards_the_preemption_arguments(self):
        from pathlib import Path

        from core.inference import llama_cpp

        source = Path(llama_cpp.__file__).read_text(encoding = "utf-8")
        retry = source[
            source.index(
                "yield from self.generate_chat_completion(\n                    retry_messages,"
            ) :
        ]
        retry = retry[: retry.index("_allow_respawn_retry = False")]
        for kwarg in (
            "admission_output_allowance = admission_output_allowance,",
            '{"preempt_event": preempt_event}',
            "preempt_policy = preempt_policy,",
            "on_tokens = on_tokens,",
        ):
            assert kwarg in retry, f"the respawn retry drops {kwarg}"
