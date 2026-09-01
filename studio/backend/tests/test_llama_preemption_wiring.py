# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The policy and the stream were built apart; this is the seam between them.

Both halves passed their own tests while nothing called `plan_preemptions()`, so a build
containing every line of this feature still could not pause a chat. These tests assert
the wiring itself: that the route arms a policy, that arming it charges the real token
figure rather than a silent zero, and that a finished generation stops counting.
"""

import sys
from types import SimpleNamespace

import pytest

from core.inference.llama_admission import (
    LlamaAdmissionConfig,
    get_llama_admission_queue,
)
from core.inference.llama_preemption import (
    ControllerPreemptionPolicy,
    DeferredPreemptionPolicy,
    ParticipantState,
    PreemptSignal,
    PreemptionController,
    get_preemption_controller,
    reset_preemption_controllers,
)


@pytest.fixture(autouse = True)
def _clean_registry():
    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


def _backend(*, window = 16384, slots = 4, unified = True, url = "http://127.0.0.1:1/"):
    return SimpleNamespace(
        base_url = url,
        context_length = window,
        _kv_cache_context_total = window,
        _kv_cache_unified = unified,
        effective_parallel_slots = slots,
    )


class TestTheLeaseReportsWhatItHolds:
    """`getattr(lease, "tokens", 0)` silently returned 0 before this property existed.

    That is the failure mode this whole file exists to catch: everything imports, every
    unit test passes, and preemption never triggers because it believes nobody is
    holding anything.
    """

    @pytest.mark.asyncio
    async def test_a_lease_exposes_its_charge(self):
        queue = get_llama_admission_queue("wiring-tokens")
        reservation = queue.reserve(
            capacity = 4,
            config = LlamaAdmissionConfig(),
            budget = 16384,
            tokens = 4096,
        )
        lease = reservation.lease_nowait()
        assert lease is not None
        assert lease.tokens == 4096, "the charge must be readable, not silently zero"

    @pytest.mark.asyncio
    async def test_a_released_lease_says_so(self):
        queue = get_llama_admission_queue("wiring-released")
        reservation = queue.reserve(
            capacity = 4,
            config = LlamaAdmissionConfig(),
            budget = 16384,
            tokens = 1024,
        )
        lease = reservation.lease_nowait()
        assert lease.is_released is False
        lease.release()
        assert lease.is_released is True


class TestAFinishedChatStopsCounting:
    """A missed unregister would preempt everyone forever, so it cannot depend on the
    route remembering. Release happens on at least eight branches."""

    @pytest.mark.asyncio
    async def test_a_released_participant_is_pruned_without_unregister(self):
        queue = get_llama_admission_queue("wiring-prune")
        controller = PreemptionController("wiring-prune")
        controller.configure(budget = 16384, kv_unified = True)
        leases = []
        for index in range(2):
            reservation = queue.reserve(
                capacity = 4,
                config = LlamaAdmissionConfig(),
                budget = 16384,
                tokens = 4096,
            )
            lease = reservation.lease_nowait()
            leases.append(lease)
            controller.register(f"gen{index}", lease = lease, tokens = 4096)
        assert controller.committed_tokens() == 8192
        # The route never says a word; the lease ending is enough.
        leases[0].release()
        assert controller.committed_tokens() == 4096, (
            "a finished generation still counted against the budget"
        )

    @pytest.mark.asyncio
    async def test_pruning_frees_the_epoch(self):
        queue = get_llama_admission_queue("wiring-epoch")
        controller = PreemptionController("wiring-epoch")
        controller.configure(budget = 16384, kv_unified = True)
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        lease = reservation.lease_nowait()
        controller.register("winner", lease = lease, tokens = 4096)
        controller.plan_preemptions(needed = 999999)
        assert controller.snapshot().winner == "winner"
        lease.release()
        controller.committed_tokens()
        assert controller.snapshot().winner is None


class TestTheDeferredHandoff:
    """The generator is BUILT before admission returns and ITERATED after, so the policy
    passed into it cannot yet know its lease."""

    def test_unbound_is_inert_rather_than_crashing(self):
        policy = DeferredPreemptionPolicy()
        assert policy.bound is False
        assert policy.should_preempt() is False
        # False means "finish the turn", the behaviour that predates preemption.
        assert policy.await_resume(timeout = 0.01) is False
        policy.on_resumed()

    def test_binding_forwards(self):
        controller = PreemptionController("wiring-bind")
        controller.configure(budget = 16384, kv_unified = True)
        signal = PreemptSignal()
        controller.register("g", tokens = 10)
        policy = DeferredPreemptionPolicy()
        policy.bind(ControllerPreemptionPolicy(controller, "g", signal))
        assert policy.bound is True
        assert policy.should_preempt() is False
        signal.set()
        assert policy.should_preempt() is True


class TestTheRouteActuallyArmsIt:
    """Everything above can pass while the route never calls any of it."""

    def _route(self):
        import routes.inference as inference

        return inference

    @pytest.mark.asyncio
    async def test_the_route_arms_and_plans(self):
        inference = self._route()
        queue = get_llama_admission_queue("http://127.0.0.1:1/")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        signal = PreemptSignal()
        policy = inference._openai_llama_preemption_arm(
            request = None,
            llama_backend = _backend(),
            reservation = reservation,
            gen_id = "armed",
            signal = signal,
            loop = None,
        )
        assert policy is not None, "the route did not arm a policy"
        controller = get_preemption_controller("http://127.0.0.1:1/")
        snapshot = controller.snapshot()
        assert snapshot.committed == 4096, "the real charge did not reach the controller"
        assert snapshot.budget == 16384

    @pytest.mark.asyncio
    async def test_a_private_cache_per_slot_is_not_armed(self):
        """Without --kv-unified a preempted slot's cells are never purged for anyone
        else, so pausing would stall the victim for nothing."""
        inference = self._route()
        queue = get_llama_admission_queue("http://127.0.0.1:2/")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        policy = inference._openai_llama_preemption_arm(
            request = None,
            llama_backend = _backend(unified = False, url = "http://127.0.0.1:2/"),
            reservation = reservation,
            gen_id = "not-armed",
            signal = PreemptSignal(),
            loop = None,
        )
        assert policy is None

    @pytest.mark.asyncio
    async def test_the_switch_off_arms_nothing(self, monkeypatch):
        inference = self._route()
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        queue = get_llama_admission_queue("http://127.0.0.1:3/")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        policy = inference._openai_llama_preemption_arm(
            request = None,
            llama_backend = _backend(url = "http://127.0.0.1:3/"),
            reservation = reservation,
            gen_id = "off",
            signal = PreemptSignal(),
            loop = None,
        )
        assert policy is None, "the rollout switch did not turn it off"

    def test_the_tool_loop_is_handed_both_the_signal_and_the_policy(self):
        """The wiring that makes any of this reachable at runtime.

        Read from source: the call site is inside a large async generator that cannot be
        invoked here without a live backend, and asserting on the source is honest about
        that rather than pretending to exercise it.
        """
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        assert "preempt_event = _gguf_preempt_signal," in source, (
            "the tool loop is not handed the preempt signal"
        )
        assert "preempt_policy = _gguf_preempt_policy_hold," in source, (
            "the tool loop is not handed the policy"
        )
        assert "_openai_llama_preemption_arm(" in source
        assert source.count("_gguf_preempt_policy_hold.bind(") == 1, (
            "the policy is never bound, so it stays inert forever"
        )
