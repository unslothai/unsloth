# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every KV-preemption switch has to reach every surface that acts on it.

The optimistic allowance, the wire clamp, the arm and the disarm's erase are four decisions taken
from one question: can the difference between what a request is charged and what it may generate
be reclaimed? Each check covers one place where the answer was assumed instead of asked.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

import routes.inference as inf
from routes.inference import _OPENAI_LLAMA_ADMISSION_WIRE_RESERVE_TOKENS as _RESERVE
from core.inference import llama_exact as exact
from core.inference.llama_admission import (
    ADMISSION_CONTROL_ENV,
    ADMISSION_KV_BUDGET_ENV,
    LlamaAdmissionQueueFull,
)
from core.inference.llama_cpp import LlamaCppBackend, _preempt_ram_disabled_in
from core.inference.llama_preemption import (
    PREEMPT_ENV,
    PREEMPT_MODE_ENV,
    ParticipantState,
    PreemptSignal,
    get_preemption_controller,
)
from fastapi import HTTPException
from models.inference import AnthropicMessagesRequest, ChatCompletionRequest

from .preempt_fakes import clean_admission_queues, clean_preemption_registry

# pytest finds these by name; named here so the import reads as a use.
_FIXTURES = (clean_admission_queues, clean_preemption_registry)

_BUDGET = 16384
_SLOTS = 4
_KEY = "http://127.0.0.1:65011"


@pytest.fixture(autouse = True)
def _switches_on(monkeypatch):
    """Every switch explicitly on, so a test that turns one off is testing that switch."""
    monkeypatch.setenv(ADMISSION_CONTROL_ENV, "1")
    monkeypatch.setenv(ADMISSION_KV_BUDGET_ENV, "1")
    monkeypatch.setenv(PREEMPT_ENV, "1")
    monkeypatch.delenv("LLAMA_ARG_PREEMPT_RAM", raising = False)
    yield


def _backend(**overrides):
    fields = dict(
        base_url = _KEY,
        context_length = _BUDGET,
        _kv_cache_context_total = _BUDGET,
        effective_parallel_slots = _SLOTS,
        _kv_cache_unified = True,
        server_preempts_kv = False,
        supports_tools = True,
        supports_tool_passthrough = True,
        is_loaded = True,
        is_vision = False,
        model_identifier = "test-model",
        count_chat_tokens = lambda *a, **k: 2,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _chat(cap = None, **fields):
    return ChatCompletionRequest(
        messages = [{"role": "user", "content": "write me a poem about a slot"}],
        max_tokens = cap,
        **fields,
    )


def _prompt_tokens(payload, backend):
    tokens = inf._openai_llama_admission_prompt_tokens(
        payload, image_tokens = inf._openai_llama_admission_image_tokens(backend)
    )
    assert tokens is not None
    return tokens


def _charge(reservation):
    lease = reservation.lease_nowait()
    assert lease is not None, "the fixture backend has free slots, so this must be granted"
    return int(lease.tokens)


# ── 1. Optimistic pricing needs a request that can actually be paused ─────────


class TestPricingARequestThatCannotBePaused:
    def test_an_unpausable_request_is_charged_what_it_may_generate(self):
        backend = _backend()
        payload = _chat(6000)
        prompt = _prompt_tokens(payload, backend)

        async def _run():
            unpausable, _ = inf._openai_llama_admission_reserve(
                request = None, llama_backend = backend, payload = payload, pausable = False
            )
            try:
                assert _charge(unpausable) == prompt + 6000
            finally:
                unpausable.cancel()

        asyncio.run(_run())

    def test_a_pausable_request_keeps_the_optimistic_allowance(self):
        backend = _backend()
        payload = _chat(6000)
        prompt = _prompt_tokens(payload, backend)

        async def _run():
            pausable, _ = inf._openai_llama_admission_reserve(
                request = None, llama_backend = backend, payload = payload
            )
            try:
                charged = _charge(pausable)
                assert charged < prompt + 6000
                assert charged <= max(1, _BUDGET // _SLOTS)
            finally:
                pausable.cancel()

        asyncio.run(_run())

    def test_two_unpausable_requests_cannot_both_be_admitted_beyond_the_cache(self):
        # Two raw streams were once admitted for 8064 tokens while permitted 18016 cells of a 16384
        # cache.
        backend = _backend()
        payload = _chat(6000)
        prompt = 3008
        inf_prompt_tokens = inf._openai_llama_admission_prompt_tokens

        async def _run():
            held = []
            try:
                for _ in range(2):
                    reservation, _config = inf._openai_llama_admission_reserve(
                        request = None,
                        llama_backend = backend,
                        payload = payload,
                        pausable = False,
                    )
                    held.append(reservation)
                granted = [r for r in held if r.lease_nowait() is not None]
                assert len(granted) == 1
                assert _charge(granted[0]) == prompt + 6000
            finally:
                for reservation in held:
                    reservation.cancel()

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(inf, "_openai_llama_admission_prompt_tokens", lambda *a, **k: prompt)
            asyncio.run(_run())
        assert inf._openai_llama_admission_prompt_tokens is inf_prompt_tokens


class _ReserveRecorder:
    """Stops each surface at its reservation and keeps the keywords it asked with."""

    def __init__(self, monkeypatch):
        self.calls: list[dict] = []

        def _reserve(**kwargs):
            self.calls.append(kwargs)
            raise LlamaAdmissionQueueFull("stopped by the test", snapshot = None)

        monkeypatch.setattr(inf, "_openai_llama_admission_reserve", _reserve)

    @property
    def pausable(self):
        assert len(self.calls) == 1, self.calls
        return self.calls[0].get("pausable")


class TestEverySurfaceStatesWhetherItCanPause:
    def test_the_streaming_chat_passthrough_reserves_as_unpausable(self, monkeypatch):
        recorder = _ReserveRecorder(monkeypatch)
        with pytest.raises(HTTPException) as raised:
            asyncio.run(
                inf._openai_passthrough_stream(
                    None,
                    threading.Event(),
                    _backend(),
                    _chat(),
                    "test-model",
                    "cmpl-raw-stream",
                )
            )
        assert raised.value.status_code == 429
        assert recorder.pausable is False

    def test_the_non_streaming_chat_passthrough_reserves_as_unpausable(self, monkeypatch):
        recorder = _ReserveRecorder(monkeypatch)
        with pytest.raises(HTTPException) as raised:
            asyncio.run(
                inf._openai_passthrough_non_streaming(
                    _backend(),
                    _chat(),
                    "test-model",
                    request = None,
                    cancel_event = threading.Event(),
                )
            )
        assert raised.value.status_code == 429
        assert recorder.pausable is False

    def test_the_anthropic_client_tool_passthrough_reserves_as_unpausable(self, monkeypatch):
        recorder = _ReserveRecorder(monkeypatch)
        monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _backend())
        payload = AnthropicMessagesRequest(
            max_tokens = 64,
            messages = [{"role": "user", "content": "hi"}],
            tools = [
                {
                    "name": "web_search",
                    "description": "search",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
        with pytest.raises(HTTPException) as raised:
            asyncio.run(
                inf.anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t")
            )
        assert raised.value.status_code == 429
        assert recorder.pausable is False

    def test_an_anthropic_chat_studio_composes_stays_pausable(self, monkeypatch):
        recorder = _ReserveRecorder(monkeypatch)
        monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _backend())
        payload = AnthropicMessagesRequest(
            max_tokens = 64, messages = [{"role": "user", "content": "hi"}]
        )
        with pytest.raises(HTTPException):
            asyncio.run(
                inf.anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t")
            )
        assert recorder.pausable is True


class _AnthropicRequest:
    def __init__(self):
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(path = "/v1/messages")
        self.method = "POST"

    async def is_disconnected(self):
        return False


# ── 2. A zero RAM budget from the child's environment ────────────────────────


class TestServerParkingSwitchedOffThroughTheEnvironment:
    def test_a_zero_budget_in_the_child_environment_is_seen(self):
        assert _preempt_ram_disabled_in(["--kv-unified"], env = {"LLAMA_ARG_PREEMPT_RAM": "0"})

    def test_studios_own_environment_is_read_when_no_child_env_is_given(self, monkeypatch):
        monkeypatch.setenv("LLAMA_ARG_PREEMPT_RAM", "0")
        assert _preempt_ram_disabled_in(["--kv-unified"])

    def test_a_budget_on_the_launch_line_still_wins_over_the_environment(self):
        # llama.cpp applies the variable before parsing argv, so the flag is the later word.
        assert not _preempt_ram_disabled_in(
            ["--kv-unified", "--preempt-ram", "8192"], env = {"LLAMA_ARG_PREEMPT_RAM": "0"}
        )
        assert _preempt_ram_disabled_in(
            ["--kv-unified", "--preempt-ram", "0"], env = {"LLAMA_ARG_PREEMPT_RAM": "8192"}
        )

    def test_a_nonzero_environment_budget_leaves_parking_on(self):
        assert not _preempt_ram_disabled_in(["--kv-unified"], env = {"LLAMA_ARG_PREEMPT_RAM": "8192"})

    def test_naming_no_budget_at_all_is_parking_off(self):
        assert _preempt_ram_disabled_in(["--kv-unified"], env = {})

    @pytest.mark.parametrize("zero", ["00", "+0", " 0 ", "-0"])
    def test_every_numeric_zero_switches_parking_off(self, zero):
        assert _preempt_ram_disabled_in(["--kv-unified", "--preempt-ram", zero])
        assert _preempt_ram_disabled_in([f"--preempt-ram={zero}"])
        assert _preempt_ram_disabled_in(["--kv-unified"], env = {"LLAMA_ARG_PREEMPT_RAM": zero})

    def test_an_unparseable_budget_is_not_a_zero(self):
        assert not _preempt_ram_disabled_in(["--preempt-ram", "none"])


# ── 3. Exact concurrency needs evidence the binary implements it ─────────────


class TestExactModeIsOnlyReportedOnEvidence:
    @pytest.mark.parametrize("setting", ["auto", "on"])
    def test_a_binary_without_the_fork_flag_is_unavailable(self, setting):
        # A build predating unslothai/llama.cpp#194 ignores LLAMA_EXACT_CONCURRENCY and still comes
        # up healthy.
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = setting,
                env = {exact.CHILD_ENV: "1"},
                args = ["llama-server", "--kv-unified", "--flash-attn", "on"],
                supports_exact = False,
            )
            == exact.EXACT_STATE_UNAVAILABLE
        )

    def test_the_capability_is_what_turns_it_on(self):
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = "on",
                env = {exact.CHILD_ENV: "1"},
                args = ["llama-server", "--kv-unified", "--flash-attn", "on"],
                supports_exact = True,
            )
            == exact.EXACT_STATE_ON
        )

    def test_the_capability_alone_does_not_turn_it_on(self):
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = "auto",
                env = {},
                args = ["llama-server", "--kv-unified", "--flash-attn", "on"],
                supports_exact = True,
            )
            == exact.EXACT_STATE_UNAVAILABLE
        )

    def test_the_load_reads_the_mode_off_the_running_server(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "_server_props = self._server_props_or_last()" in source
        assert '_exact_running = _server_props.get("exact_concurrency") is True' in source
        assert "supports_exact = _exact_running," in source
        assert "supports_exact = bool(" not in source

    @pytest.mark.parametrize(
        ("props", "expected"),
        [
            ({"exact_concurrency": True}, True),
            ({"exact_concurrency": False}, False),
            ({"total_slots": 4}, False),  # a build that does not advertise the mode
            (None, False),  # /props unreadable
            ({"exact_concurrency": "1"}, False),  # only the boolean the server sends
        ],
    )
    def test_what_the_server_says_on_props_is_the_evidence(self, monkeypatch, props, expected):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        monkeypatch.setattr(backend, "_query_server_props", lambda: props)
        assert backend._server_reports_exact_concurrency() is expected


# ── 4. Arming obeys the same eligibility gate as pricing ─────────────────────


def _arm(
    backend,
    gen_id,
    *,
    tokens = None,
):
    reservation, _config = inf._openai_llama_admission_reserve(
        request = None, llama_backend = backend, payload = _chat(64)
    )
    signal = PreemptSignal()
    policy = inf._openai_llama_preemption_arm(
        request = None,
        llama_backend = backend,
        reservation = reservation,
        gen_id = gen_id,
        signal = signal,
    )
    return reservation, signal, policy


class TestArmingHonoursTheAccountingOptOuts:
    @pytest.mark.parametrize("switch", [ADMISSION_CONTROL_ENV, ADMISSION_KV_BUDGET_ENV])
    def test_a_disabled_switch_arms_nothing(self, monkeypatch, switch):
        monkeypatch.setenv(switch, "0")
        backend = _backend()
        assert inf._openai_llama_preemption_will_apply(backend, _BUDGET) is False

        async def _run():
            held = []
            signals = []
            try:
                for index in range(_SLOTS):
                    reservation, signal, policy = _arm(backend, f"disabled-{index}")
                    held.append(reservation)
                    signals.append(signal)
                    assert policy is None
                controller = get_preemption_controller(_KEY)
                for index in range(_SLOTS):
                    controller.observe(f"disabled-{index}", 4000)
                assert not any(signal.is_set() for signal in signals)
                assert all(controller.participant(f"disabled-{i}") is None for i in range(_SLOTS))
            finally:
                for reservation in held:
                    reservation.cancel()

        asyncio.run(_run())

    @pytest.mark.parametrize("switch", [ADMISSION_CONTROL_ENV, ADMISSION_KV_BUDGET_ENV])
    def test_the_controller_agrees_it_is_not_active(self, monkeypatch, switch):
        controller = get_preemption_controller(_KEY)
        controller.configure(budget = _BUDGET, kv_unified = True, slots = _SLOTS)
        assert controller.active is True
        monkeypatch.setenv(switch, "0")
        assert controller.active is False
        assert controller.plan_preemptions(needed = _BUDGET) == []

    def test_a_fully_switched_on_backend_still_arms(self):
        backend = _backend()

        async def _run():
            reservation, _signal, policy = _arm(backend, "armed")
            try:
                assert policy is not None
                assert get_preemption_controller(_KEY).participant("armed") is not None
            finally:
                reservation.cancel()

        asyncio.run(_run())


# ── 5. The wire clamp only enforces a reservation that exists ────────────────


class TestTheWireClampFollowsTheSwitches:
    @pytest.mark.parametrize("switch", [ADMISSION_CONTROL_ENV, ADMISSION_KV_BUDGET_ENV])
    @pytest.mark.parametrize("cap", [None, 512, 20000])
    def test_a_disabled_switch_sends_the_callers_own_cap(self, monkeypatch, switch, cap):
        monkeypatch.setenv(switch, "0")
        assert (
            inf._openai_llama_admission_enforced_max_tokens(
                _chat(cap), request = None, llama_backend = _backend()
            )
            is None
        )

    def test_admission_on_and_preemption_off_sends_the_callers_own_cap(self, monkeypatch):
        """Off is what Studio did before preemption existed: no admission cap on the wire."""
        monkeypatch.setenv(PREEMPT_ENV, "0")
        assert (
            inf._openai_llama_admission_enforced_max_tokens(
                _chat(), request = None, llama_backend = _backend()
            )
            is None
        )

    def test_with_nothing_to_reclaim_the_share_pricing_clamps_to_a_share(
        self, monkeypatch, share_pricing
    ):
        monkeypatch.setenv(PREEMPT_ENV, "0")
        backend = _backend()
        payload = _chat()
        prompt = _prompt_tokens(payload, backend)
        clamp = inf._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        assert clamp == max(1, _BUDGET // _SLOTS) - prompt - _RESERVE

    def test_a_stated_cap_at_or_above_the_window_is_still_unstated(self):
        backend = _backend()
        payload = _chat(20000)
        prompt = _prompt_tokens(payload, backend)
        assert (
            inf._openai_llama_admission_enforced_max_tokens(
                payload, request = None, llama_backend = backend
            )
            == _BUDGET - prompt - _RESERVE
        )

    def test_a_stated_cap_below_the_window_is_left_alone(self):
        assert (
            inf._openai_llama_admission_enforced_max_tokens(
                _chat(512), request = None, llama_backend = _backend()
            )
            is None
        )


# ── 6. The disarm erases only for a generation that armed ────────────────────


class _Erases:
    def __init__(
        self,
        monkeypatch,
        *,
        queued = 1,
        idle_tokens = 2000,
    ):
        self.erased: list[int] = []
        monkeypatch.setattr(
            inf,
            "get_llama_admission_queue",
            lambda _key: SimpleNamespace(snapshot = lambda: SimpleNamespace(queued = queued)),
        )
        monkeypatch.setattr(
            inf,
            "fetch_llama_slots",
            lambda *_a, **_k: [{"id": 0, "is_processing": False, "n_prompt_tokens": idle_tokens}],
        )

        def _erase(_base, slot_id, **_kwargs):
            self.erased.append(slot_id)
            return idle_tokens

        monkeypatch.setattr(inf, "erase_llama_slot", _erase)


class TestDisarmDoesNotEraseWhatItNeverArmed:
    def test_a_switched_off_generation_keeps_the_prefix_cache(self, monkeypatch):
        monkeypatch.setenv(PREEMPT_ENV, "0")
        erases = _Erases(monkeypatch)
        inf._openai_llama_preemption_disarm(llama_backend = _backend(), gen_id = "never-armed")
        assert erases.erased == []

    def test_a_non_unified_cache_keeps_the_prefix_cache(self, monkeypatch):
        erases = _Erases(monkeypatch)
        inf._openai_llama_preemption_disarm(
            llama_backend = _backend(_kv_cache_unified = False), gen_id = "never-armed"
        )
        assert erases.erased == []

    def test_a_generation_that_did_arm_still_hands_its_cells_back(self, monkeypatch):
        erases = _Erases(monkeypatch)
        backend = _backend()
        controller = get_preemption_controller(_KEY)
        controller.configure(budget = _BUDGET, kv_unified = True, slots = _SLOTS)
        controller.register("armed", tokens = 1000, state = ParticipantState.DECODING)
        inf._openai_llama_preemption_disarm(llama_backend = backend, gen_id = "armed")
        assert erases.erased == [0]


# ── 7. The residency sweep answers to the same switches ──────────────────────


class TestTheResidencySweepAnswersToTheSameSwitches:
    @pytest.mark.parametrize(
        ("env", "unified"),
        [
            ({PREEMPT_ENV: "0"}, True),
            ({"UNSLOTH_LLAMA_ADMISSION_CONTROL": "0"}, True),
            ({"UNSLOTH_LLAMA_ADMISSION_KV_BUDGET": "0"}, True),
            ({}, False),
        ],
    )
    def test_an_ineligible_generation_erases_no_idle_slot(self, monkeypatch, env, unified):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        erases = _Erases(monkeypatch)
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 1000},
            {"id": 1, "is_processing": True, "n_prompt_tokens": 64},
        ]
        monkeypatch.setattr(inf, "fetch_llama_slots", lambda *a, **k: slots)
        backend = _backend(_kv_cache_unified = unified)
        assert not inf._openai_llama_preemption_will_apply(backend, _BUDGET)
        _refresh, observe, _note = inf._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "active"
        )
        # llama.cpp skips the token callback outright when `on_tokens` is None.
        assert observe is None
        assert erases.erased == []

    def test_an_eligible_generation_still_sweeps(self, monkeypatch):
        erases = _Erases(monkeypatch)
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 1000},
            {"id": 1, "is_processing": True, "n_prompt_tokens": 64},
        ]
        monkeypatch.setattr(inf, "fetch_llama_slots", lambda *a, **k: slots)
        backend = _backend()
        assert inf._openai_llama_preemption_will_apply(backend, _BUDGET)
        controller = get_preemption_controller(_KEY)
        controller.configure(budget = 512, kv_unified = True, slots = _SLOTS)
        controller.register("active", tokens = 64)
        controller.register("theirs", tokens = 64)
        _refresh, observe, _note = inf._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "active"
        )
        observe(32)
        assert erases.erased == [0]


# ── 8. An unpausable request is charged what it is permitted ─────────────────


class TestAnUnpausableRequestIsChargedWhatItIsPermitted:
    def test_an_unstated_unpausable_request_reserves_the_rest_of_its_share(self):
        share = _BUDGET // _SLOTS
        prompt = 8
        charged = inf._openai_llama_admission_output_allowance(
            None,
            budget = _BUDGET,
            prompt_tokens = prompt,
            context_window = _BUDGET,
            share = share,
            preemption_active = False,
        )
        assert charged == share - prompt

    def test_the_charge_is_the_wire_cap_plus_the_reserve_it_never_sends(self):
        backend = _backend()
        payload = _chat(None)
        prompt = _prompt_tokens(payload, backend)
        wire = inf._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, pausable = False
        )

        async def _run():
            reservation, _ = inf._openai_llama_admission_reserve(
                request = None, llama_backend = backend, payload = payload, pausable = False
            )
            try:
                assert _charge(reservation) == prompt + wire + _RESERVE
            finally:
                reservation.cancel()

        asyncio.run(_run())

    def test_a_pausable_unstated_request_keeps_the_flat_allowance(self):
        share = _BUDGET // _SLOTS
        charged = inf._openai_llama_admission_output_allowance(
            None,
            budget = _BUDGET,
            prompt_tokens = 8,
            context_window = _BUDGET,
            share = share,
            preemption_active = True,
        )
        assert charged == inf._OPENAI_LLAMA_ADMISSION_UNSTATED_OUTPUT_TOKENS


# ── 9. The Anthropic passthrough is sent the cap it was charged for ──────────


class TestTheAnthropicPassthroughIsSentTheCapItWasChargedFor:
    @pytest.mark.parametrize("stream", [True, False])
    def test_an_omitted_cap_is_held_to_the_share(self, monkeypatch, stream):
        backend = _backend()
        monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
        sent: dict = {}

        async def _stream(
            request,
            cancel_event,
            llama_backend,
            messages,
            tools,
            temperature,
            top_p,
            top_k,
            max_tokens,
            *a,
            **k,
        ):
            sent.update(max_tokens = max_tokens, messages = messages, tools = tools)
            raise HTTPException(status_code = 418)

        async def _non_streaming(
            llama_backend, messages, tools, temperature, top_p, top_k, max_tokens, *a, **k
        ):
            sent.update(max_tokens = max_tokens, messages = messages, tools = tools)
            raise HTTPException(status_code = 418)

        monkeypatch.setattr(inf, "_anthropic_passthrough_stream", _stream)
        monkeypatch.setattr(inf, "_anthropic_passthrough_non_streaming", _non_streaming)
        payload = AnthropicMessagesRequest(
            max_tokens = _BUDGET,  # at the window, which is unstated
            stream = stream,
            messages = [{"role": "user", "content": "hi"}],
            tools = [
                {
                    "name": "web_search",
                    "description": "search",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
        with pytest.raises(HTTPException) as raised:
            asyncio.run(
                inf.anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t")
            )
        assert raised.value.status_code == 418
        expected = inf._openai_llama_admission_enforced_max_tokens(
            payload,
            request = None,
            llama_backend = backend,
            conversation = sent["messages"],
            injected_tools = sent["tools"],
            pausable = False,
        )
        assert expected is not None and expected < _BUDGET
        assert sent["max_tokens"] == expected


# ── 10. One switch stands the server's own parking down as well ──────────────


class TestOneSwitchStandsTheChildsParkingDown:
    def test_preemption_off_leaves_a_clean_launch_exactly_as_upstream_ships_it(self, monkeypatch):
        from core.inference.llama_cpp import _stand_down_child_parking

        monkeypatch.setenv(PREEMPT_ENV, "0")
        env: dict = {}
        assert _stand_down_child_parking(env, ["llama-server", "--kv-unified"]) == []
        assert env == {}
        assert _preempt_ram_disabled_in(["llama-server", "--kv-unified"], env = env)

    def test_preemption_on_leaves_the_child_alone(self):
        from core.inference.llama_cpp import _stand_down_child_parking

        env: dict = {}
        assert _stand_down_child_parking(env, ["llama-server"]) is None
        assert env == {}

    def test_studio_pausing_stands_the_child_down_too(self, monkeypatch):
        from core.inference.llama_cpp import _stand_down_child_parking

        monkeypatch.setenv(PREEMPT_MODE_ENV, "studio")
        env: dict = {}
        assert _stand_down_child_parking(env, ["llama-server", "--kv-unified"]) == []
        assert env["LLAMA_ARG_PREEMPT_RAM"] == "0"
        named = {"LLAMA_ARG_PREEMPT_RAM": "4096"}
        assert _stand_down_child_parking(named, ["llama-server"]) == ["LLAMA_ARG_PREEMPT_RAM=4096"]
        assert named["LLAMA_ARG_PREEMPT_RAM"] == "0"

    @pytest.mark.parametrize(
        ("env", "args", "overridden", "line"),
        [
            (
                {"LLAMA_ARG_PREEMPT_RAM": "4096"},
                ["llama-server"],
                ["LLAMA_ARG_PREEMPT_RAM=4096"],
                ["llama-server"],
            ),
            (
                {},
                ["llama-server", "--preempt-ram", "4096"],
                ["--preempt-ram 4096"],
                ["llama-server", "--preempt-ram", "0"],
            ),
            (
                {},
                ["llama-server", "--preempt-ram=4096"],
                ["--preempt-ram=4096"],
                ["llama-server", "--preempt-ram=0"],
            ),
        ],
    )
    def test_a_budget_someone_named_loses_to_the_switch(
        self, monkeypatch, env, args, overridden, line
    ):
        from core.inference.llama_cpp import _stand_down_child_parking

        monkeypatch.setenv(PREEMPT_ENV, "0")
        assert _stand_down_child_parking(env, args) == overridden
        assert env["LLAMA_ARG_PREEMPT_RAM"] == "0"
        assert args == line
        assert _preempt_ram_disabled_in(args, env = env)

    def test_the_load_warning_names_the_flag_and_the_setting_that_overrode_it(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        source = inspect.getsource(LlamaCppBackend.load_model)
        site = source.index("_parking_overridden = _stand_down_child_parking(")
        window = source[site : site + 1400]
        assert "_stand_down_why = _child_parking_stand_down_reason(" in window
        assert "self._record_load_warning(" in window
        assert '", ".join(_parking_overridden)' in window


class TestAnInconclusiveProbeStandsTheChildDown:
    """The capability probe not confirming --preempt-ram makes Studio the owner, so a child
    that parks after all is told not to, and a named budget is overridden like any other."""

    def test_the_child_is_told_not_to_park(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_LLAMA_PREEMPT_MODE", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        from core.inference import llama_cpp as llama_mod

        env: dict = {}
        assert llama_mod._stand_down_child_parking(env, [], server_supports = True) is None
        assert "LLAMA_ARG_PREEMPT_RAM" not in env
        assert llama_mod._stand_down_child_parking(env, [], server_supports = False) == []
        assert env["LLAMA_ARG_PREEMPT_RAM"] == "0"
        args = ["--preempt-ram", "8192"]
        assert llama_mod._stand_down_child_parking({}, args, server_supports = False) == [
            "--preempt-ram 8192"
        ]
        assert args == ["--preempt-ram", "0"]
        assert "did not confirm --preempt-ram" in llama_mod._child_parking_stand_down_reason(False)
        assert llama_mod._child_parking_stand_down_reason(True) is None
        assert llama_mod._child_parking_stand_down_reason() is None

    def test_the_launch_passes_the_probes_answer(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert (
            '_stand_down_child_parking( env, cmd, server_supports = bool(server_caps.get("supports_preempt_ram")),'
            ' probe_inconclusive = bool(server_caps.get("reasoning_budget_probe_inconclusive")), )'
            in source
        )
