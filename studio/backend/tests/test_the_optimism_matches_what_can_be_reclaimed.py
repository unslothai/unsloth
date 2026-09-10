# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Four places where the mechanism promised more than it delivered.

Admission is optimistic because something reclaims the difference. Where nothing can --
the raw surfaces, a victim whose parked identity was erased, an admission that never read
the cache -- the optimism is a bare overcommit, which is the failure this branch exists to
remove. And a Stop during a pause is a cancel, not a chat that ran out of cache.
"""

from __future__ import annotations

import ast
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference.llama_admission import LlamaAdmissionConfig, get_llama_admission_queue
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptionController,
    PreemptSignal,
    get_preemption_controller,
)
import routes.inference as inference_route


ROUTES = pathlib.Path(inference_route.__file__)
LLAMA_CPP = ROUTES.parent.parent / "core" / "inference" / "llama_cpp.py"


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _chat(text = "hi", **fields):
    return _Payload(messages = [{"role": "user", "content": text}], **fields)


def _backend(
    *,
    window = 16384,
    slots = 4,
    unified = True,
    base_url = "http://raw.test",
):
    return SimpleNamespace(
        context_length = window,
        _kv_cache_context_total = window,
        effective_parallel_slots = slots,
        _kv_cache_unified = unified,
        base_url = base_url,
    )


class TestARawSurfaceKeepsTheShare:
    """`STREAMING_RAW` is counted and never chosen, so nothing reclaims what it overruns.

    The flag is spelled `pausable`, and a backend whose server parks slots itself turns it
    back on for these surfaces; the fixture is a server that does not.
    """

    def test_preemption_must_actually_apply_for_the_whole_window_to_be_offered(self):
        backend = _backend()
        assert inference_route._openai_llama_preemption_will_apply(
            backend, inference_route._openai_llama_admission_budget(backend)
        ), "the fixture must be a backend where the optimistic pricing is on"

    def test_a_raw_surface_is_bounded_by_the_share(self):
        backend = _backend()
        payload = _chat(max_tokens = 16384)
        armed = inference_route._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        raw = inference_route._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, pausable = False
        )
        assert armed is not None and raw is not None
        # Armed: the window, because the controller polices it. Raw: a share, because
        # nobody can pause it and it never says how big it has become.
        assert armed > 16384 // 2
        assert raw <= 16384 // 4, f"a raw surface was offered {raw} of a 16384 cache"

    def test_a_raw_retry_is_bounded_by_the_share_too(self):
        backend = _backend()
        body = {"messages": [{"role": "user", "content": "hi"}]}
        armed = inference_route._openai_llama_admission_retry_max_tokens(
            body,
            admission_output_allowance = 8000,
            request = None,
            llama_backend = backend,
        )
        raw = inference_route._openai_llama_admission_retry_max_tokens(
            body,
            admission_output_allowance = 8000,
            request = None,
            llama_backend = backend,
            pausable = False,
        )
        assert armed is not None and raw is not None
        assert raw < armed

    def test_a_raw_reservation_charges_a_stated_cap_conservatively(self):
        payload = _chat(max_tokens = 12000)
        common = dict(
            budget = 16384,
            capacity = 4,
            context_window = 16384,
        )
        armed = inference_route._openai_llama_admission_tokens(
            payload, preemption_active = True, **common
        )
        raw = inference_route._openai_llama_admission_tokens(
            payload, preemption_active = False, **common
        )
        # Charging less than a request may write is safe only where something reclaims the
        # difference. A raw stream pays for what it asked for.
        assert raw > armed, (
            f"raw charged {raw}, armed charged {armed}: the raw surface bought optimism "
            "nothing can pay for"
        )

    def test_the_reserve_threads_the_flag_into_the_charge(self):
        source = ROUTES.read_text(encoding = "utf-8")
        body = source[source.index("def _openai_llama_admission_reserve(") :]
        body = body[: body.index("\ndef _openai_llama_admission_recost(")]
        flat = " ".join(body.split())
        assert (
            "preemption_active = pausable and _openai_llama_preemption_will_apply("
            "llama_backend, budget)" in flat
        ), "the flag has to reach the charge, not just the signature"

    @pytest.mark.parametrize(
        "anchor",
        [
            # The Responses stream.
            "payload = chat_req,",
            # Both OpenAI passthrough branches, and the shared body builder they clamp with.
            '_raw_gen_id = monitor_id or f"passthrough-nonstream-{id(payload):x}"',
            "or _effective_openai_max_tokens(payload)",
        ],
    )
    def test_every_raw_call_site_says_so(self, anchor):
        source = ROUTES.read_text(encoding = "utf-8")
        window = source[source.index(anchor) - 1200 : source.index(anchor) + 1200]
        assert (
            "pausable = False" in window
        ), "a raw surface priced as pausable takes the whole window and gives nothing back"

    def test_the_anthropic_reserve_follows_its_own_raw_flag(self):
        source = ROUTES.read_text(encoding = "utf-8")
        body = source[source.index("async def _admitted_anthropic(") :]
        body = body[: body.index("if payload.stream:")]
        assert "pausable = not raw," in body, (
            "the client-tool branch registers STREAMING_RAW, so it may not be priced as "
            "though it would be paused"
        )


def _controller(key = "test://parked") -> PreemptionController:
    made = PreemptionController(key)
    made.configure(budget = 8192, kv_unified = True, slots = 4, draft_tokens = 2, batch_tokens = 2048)
    return made


class TestAChosenVictimKeepsItsParkedIdentity:
    """The sweep runs before the reclaim reads the ledger, and PREEMPTING hid the parking."""

    def _two_holders_over_the_ceiling(self):
        controller = _controller()
        parked = controller.register("parked", tokens = 3000)
        controller.note_tokens("parked", 3000)
        controller.note_state("parked", ParticipantState.PARKED_ON_TOOL)
        decoder = controller.register("decoder", tokens = 3000)
        controller.note_tokens("decoder", 3000)
        return controller, parked, decoder

    def test_the_parked_victim_is_the_one_chosen(self):
        controller, parked, _decoder = self._two_holders_over_the_ceiling()
        victims = controller.observe("decoder", 3000)
        assert [v.gen_id for v in victims] == ["parked"], "parked holders go first"
        assert parked.state == ParticipantState.PREEMPTING

    def test_it_is_still_visible_to_the_reclaim(self):
        """`observe()` sweeps, then the route reads parked holders and erases idle slots."""
        controller, parked, _decoder = self._two_holders_over_the_ceiling()
        controller.observe("decoder", 3000)
        # This is the reading the route takes AFTER the sweep, and hands back below.
        parked_before = controller.parked_holders()
        assert "parked" in parked_before, (
            "an approval has no stream to abort, so its pause may not land until the user "
            "answers; invisible here it keeps a charge for cells the erase already took"
        )
        assert controller.note_cells_reclaimed(parked_before) == 1
        assert parked.cells_reclaimed is True
        assert parked.holds_kv is False, "its cells are gone, so it stops counting"

    def test_a_holder_that_parked_after_the_reading_still_keeps_its_charge(self):
        """The park_seq guard has to survive the same transition."""
        controller, parked, _decoder = self._two_holders_over_the_ceiling()
        controller.observe("decoder", 3000)
        stale = {"parked": parked.park_seq - 1}
        assert controller.note_cells_reclaimed(stale) == 0
        assert parked.cells_reclaimed is False

    def test_a_decoding_victim_is_not_mistaken_for_a_parked_one(self):
        controller = _controller("test://decoding-victim")
        a = controller.register("a", tokens = 3000)
        controller.note_tokens("a", 3000)
        b = controller.register("b", tokens = 3000)
        controller.note_tokens("b", 3000)
        controller.observe("b", 3000)
        chosen = a if a.state == ParticipantState.PREEMPTING else b
        assert chosen.state == ParticipantState.PREEMPTING
        assert chosen.parked_on_a_tool is False
        assert controller.parked_holders() == {}
        assert controller.note_cells_reclaimed() == 0


class TestAdmissionReadsTheCacheAfresh:
    """`contended()` says admission does; only the resume wait did."""

    @pytest.mark.asyncio
    async def test_arming_probes_before_it_plans(self, monkeypatch):
        controller = _controller("http://arm-probe.test")
        monkeypatch.setattr(inference_route, "get_preemption_controller", lambda key: controller)
        order: list = []
        controller.set_residency_probe(lambda: order.append("probe"))
        real_plan = controller.plan_preemptions

        def _plan(_self, **kwargs):
            order.append("plan")
            return real_plan(**kwargs)

        monkeypatch.setattr(PreemptionController, "plan_preemptions", _plan)

        queue = get_llama_admission_queue("http://arm-probe.test")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 8192, tokens = 2048
        )
        policy = inference_route._openai_llama_preemption_arm(
            request = None,
            llama_backend = _backend(base_url = "http://arm-probe.test"),
            reservation = reservation,
            gen_id = "armed",
            signal = PreemptSignal(),
            loop = None,
        )
        assert policy is not None
        assert order == ["probe", "plan"], (
            "a stale reading lets the next prompt prefill beside a window of retained "
            "idle cells, and the first live one arrives 32 generated tokens later"
        )

    def test_a_slow_probe_does_not_hold_the_loop(self):
        """Arming runs inside async route bodies, so the read is bounded, not awaited."""
        import asyncio

        controller = _controller("http://arm-slow.test")
        released = threading.Event()
        controller.set_residency_probe(released.wait)

        async def _drive():
            started = asyncio.get_event_loop().time()
            inference_route._refresh_residency_before_planning(controller)
            return asyncio.get_event_loop().time() - started

        try:
            waited = asyncio.run(_drive())
        finally:
            released.set()
        assert waited < inference_route._ARM_RESIDENCY_READ_S + 1.0
        assert waited >= inference_route._ARM_RESIDENCY_READ_S

    def test_off_the_loop_it_simply_reads(self):
        controller = _controller("http://arm-sync.test")
        read = []
        controller.set_residency_probe(lambda: read.append(1))
        inference_route._refresh_residency_before_planning(controller)
        assert read == [1]


class TestAStopDuringAPauseIsACancel:
    """`await_resume` answers False for both, and they are not the same ending."""

    @pytest.mark.parametrize(
        "anchor",
        [
            "resumed_p = yield from _await_resume(preempt_policy, cancel_event)",
            "_resumed = yield from _await_resume(preempt_policy, cancel_event)",
            "_resumed_f = yield from _await_resume(preempt_policy, cancel_event)",
        ],
    )
    def test_the_wait_is_followed_by_a_cancel_check(self, anchor):
        source = LLAMA_CPP.read_text(encoding = "utf-8")
        after = source[source.index(anchor) + len(anchor) :]
        cancel = after.index("cancel_event is not None and cancel_event.is_set()")
        gave_up = min(
            index
            for index in (
                after.find("_preempt_gave_up_event("),
                after.find("_final_pause_gave_up("),
            )
            if index >= 0
        )
        assert cancel < gave_up, (
            "a Stop taken for contention tells the client its turn ran out of cache and "
            "offers to continue a turn the user just stopped"
        )

    def test_the_check_returns_rather_than_reporting_a_length_finish(self):
        """Every other cancel in this generator returns silently; so does this one."""
        source = LLAMA_CPP.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        found = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = ast.unparse(node.test)
            if "cancel_event.is_set()" not in test or "cancel_event is not None" not in test:
                continue
            if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                found += 1
        assert found >= 6, f"only {found} bare cancel returns; the pause waits added three"


class TestAFailedResumeGivesTheRoomBack:
    """The grant is taken before the wait; every exit from it has to hand it back."""

    def _granted(self, monkeypatch):
        from core.inference.llama_preemption import ControllerPreemptionPolicy

        controller = _controller("test://rollback")
        held = controller.register("gen", tokens = 1000, prompt_tokens = 900)
        controller.set_state("gen", ParticipantState.PAUSED)

        class _Lease:
            tokens = 1000

            def resume_async(self, *_a, **_k):
                raise RuntimeError("the loop is gone")

        held.lease = _Lease()
        policy = ControllerPreemptionPolicy(controller, "gen", held.preempt_event, loop = object())
        return controller, held, policy

    def test_a_raising_resume_rolls_the_grant_back(self, monkeypatch):
        controller, held, policy = self._granted(monkeypatch)
        assert policy.await_resume(timeout = 0.5) is False
        # Not RESUMING: that is in `_HOLDS_KV` and out of `_PREEMPTABLE`, so a holder left
        # there is room nothing will fill and no sweep can choose.
        assert held.state == ParticipantState.PAUSED
        assert held.prefill_pending(time.monotonic()) == 0

    def test_the_two_failure_paths_agree(self):
        """`got == False` and the exception path are the same outcome for the ledger."""
        source = (
            pathlib.Path(inference_route.__file__).parent.parent
            / "core"
            / "inference"
            / "llama_preemption.py"
        )
        body = source.read_text(encoding = "utf-8")
        # The concrete policy, not the Protocol stub of the same name above it.
        body = body[body.index("class ControllerPreemptionPolicy:") :]
        body = body[body.index("    def await_resume(") :]
        body = body[: body.index("    def on_resumed(")]
        assert (
            body.count("note_resume_failed(") == 2
        ), "one of the two ways the resume can fail leaves the grant booked"


class TestNMoreChoicesNeedTheLeaseBack:
    def test_a_preempted_lease_stops_the_remaining_choices(self):
        source = pathlib.Path(inference_route.__file__).read_text(encoding = "utf-8")
        body = source[source.index("def _drain_gguf_choices():") :]
        body = body[: body.index("_plain_preempt_policy.restart()")]
        assert 'getattr(admission_lease, "is_preempted", False)' in body, (
            "restart() resets the ledger but not the lease, so the next choice would "
            "decode outside slot admission and outside the KV budget"
        )
        assert body.index("if _idx:") < body.index("is_preempted")
