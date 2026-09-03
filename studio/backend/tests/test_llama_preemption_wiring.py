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
def _isolate_process_wide_admission_state():
    """The park budget and the queue registry are process-wide, not per test.

    `_parked_total` in llama_admission is deliberately global: there is one executor and
    base_url takes a fresh port on every load, so a per-queue budget would hand the same
    allowance to each backend. That makes it shared state between tests, and a test that
    parks without releasing starves every later one.

    Caught by `test_parking_does_not_reopen_the_whole_cache` failing only when this file
    ran after the preemption wiring suite, and passing alone. An order-dependent failure
    is worse than a plain one: it moves whenever a test is added.
    """
    from core.inference.llama_admission import reset_llama_admission_queues

    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


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
    async def test_pruning_frees_the_charge(self):
        """This asserted that pruning freed the EPOCH, which no longer exists.

        No generation is exempt from eviction now, so there is no crown to release; what
        pruning has to free is the charge, or the ledger reports a cache fuller than it is
        and admits nobody.
        """
        queue = get_llama_admission_queue("wiring-epoch")
        controller = PreemptionController("wiring-epoch")
        controller.configure(budget = 16384, kv_unified = True)
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        lease = reservation.lease_nowait()
        controller.register("holder", lease = lease, tokens = 4096)
        assert controller.snapshot().winner is None, "nobody is crowned any more"
        assert controller.committed_tokens() == 4096
        lease.release()
        controller.unregister("holder")
        assert controller.committed_tokens() == 0


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
        # The budget is the cache llama-server was launched with. It was briefly reduced
        # here to reserve the speculative drafts, which double-counted them: the drafts
        # are held back by the watermark buffer, which the snapshot reports separately.
        assert snapshot.budget == inference._openai_llama_admission_budget(_backend())
        assert snapshot.budget == 16384
        assert 0 < snapshot.buffer < snapshot.budget, (
            "nothing is held back, so the cache can be worked to its last cell"
        )

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


class TestSpeculativeDraftsAreReserved:
    """Drafts occupy cells nobody is charged for.

    Measured 2026-09-01 on `-c 16384 --parallel 4 --kv-unified --spec-type draft-mtp
    --spec-draft-n-max 2`: the cache filled, llama-server halved n_batch 128 -> 64 -> 32
    -> 16 -> 8 -> 4 hunting for room, and at that width the speculative indices fell
    outside the sub-batch and it threw. That is upstream ggml-org/llama.cpp#24840, whose
    retry path shifts `slot.i_batch` by the offset but never `slot.spec_i_batch`. We
    cannot patch it, so the fix here is to keep the cache off that retry path.
    """

    def test_drafts_are_added_on_top_of_the_ratio(self):
        from core.inference.llama_preemption import preemption_buffer_tokens

        # Same slot count on both sides. The reaction term is per-slot now, so
        # comparing a four-slot reserve against a one-slot one measured that change
        # rather than the drafts.
        plain = preemption_buffer_tokens(16384, slots = 4)
        drafted = preemption_buffer_tokens(16384, draft_tokens = 2, slots = 4)
        assert drafted == plain + 8, "every slot may hold n_draft unaccounted tokens"

    def test_no_speculation_reserves_nothing_extra(self):
        from core.inference.llama_preemption import preemption_buffer_tokens

        assert preemption_buffer_tokens(16384, draft_tokens = 0, slots = 4) == (
            preemption_buffer_tokens(16384, slots = 4)
        )

    def test_a_huge_draft_window_cannot_swallow_a_small_cache(self):
        """Still never a ceiling of zero, which would preempt everyone forever."""
        from core.inference.llama_preemption import preemption_buffer_tokens

        buffer = preemption_buffer_tokens(512, draft_tokens = 64, slots = 8)
        assert buffer <= 256
        assert 512 - buffer > 0

    def test_the_controller_reserves_them(self):
        controller = PreemptionController("spec")
        controller.configure(
            budget = 16384, kv_unified = True, draft_tokens = 2, slots = 4
        )
        from core.inference.llama_preemption import preemption_buffer_tokens

        # Derived, not a constant: the ratio is tunable and was raised after measurement,
        # and an earlier revision pinned 828 here twice over.
        assert controller.snapshot().buffer == preemption_buffer_tokens(
            16384, draft_tokens = 2, slots = 4
        )
        assert controller.snapshot().buffer > preemption_buffer_tokens(16384)

    @pytest.mark.asyncio
    async def test_the_route_passes_the_backends_draft_settings(self):
        import routes.inference as inference

        queue = get_llama_admission_queue("http://127.0.0.1:9/")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        backend = _backend(url = "http://127.0.0.1:9/")
        backend._speculative_type = "draft-mtp"
        backend._spec_draft_n_max = 2
        inference._openai_llama_preemption_arm(
            request = None,
            llama_backend = backend,
            reservation = reservation,
            gen_id = "spec-armed",
            signal = PreemptSignal(),
            loop = None,
        )
        snapshot = get_preemption_controller("http://127.0.0.1:9/").snapshot()
        from core.inference.llama_preemption import preemption_buffer_tokens

        # batch_tokens is the launch's --batch-size, unstated here so it defaults to
        # llama.cpp's 2048; the drafter's cells are added ON TOP of it.
        assert snapshot.buffer == preemption_buffer_tokens(
            snapshot.budget, draft_tokens = 2, slots = 4, batch_tokens = 2048
        ), f"the drafter's tokens were not reserved (buffer {snapshot.buffer})"
        assert snapshot.buffer > preemption_buffer_tokens(
            snapshot.budget, slots = 4, batch_tokens = 2048
        ), "the drafter must cost something over the same launch without one"

    @pytest.mark.asyncio
    async def test_a_backend_without_speculation_reserves_nothing_extra(self):
        import routes.inference as inference

        queue = get_llama_admission_queue("http://127.0.0.1:10/")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        backend = _backend(url = "http://127.0.0.1:10/")
        # Both cleared together, which is the only way the backend leaves them: a stated
        # depth is set at the same point a drafter is configured, so "depth 2 but nothing
        # drafting" is a state that cannot occur. An earlier revision of this test
        # asserted against exactly that impossible shape.
        backend.speculative_type = None
        backend.spec_drafter_kind = None
        backend.requested_spec_mode = None
        backend._speculative_type = None
        backend.spec_draft_n_max = None
        backend._spec_draft_n_max = None
        inference._openai_llama_preemption_arm(
            request = None,
            llama_backend = backend,
            reservation = reservation,
            gen_id = "no-spec",
            signal = PreemptSignal(),
            loop = None,
        )
        from core.inference.llama_preemption import preemption_buffer_tokens

        snapshot = get_preemption_controller("http://127.0.0.1:10/").snapshot()
        assert snapshot.buffer == preemption_buffer_tokens(
            snapshot.budget, slots = snapshot.slots, batch_tokens = 2048
        )


class TestTheBufferCanHoldOnePrefillChunk:
    """The term every earlier buffer was missing.

    The cache does not fail when it is full of tokens, it fails when the next BATCH does
    not fit. llama-server prefills in chunks of --batch-size, so a resumed chat replaying
    5000 tokens asks for a whole chunk of free cells at once.

    Measured 2026-09-03 and it is why the watermark kept looking innocent: across 1329
    samples peak residency was 13540 against a 15592 ceiling, never once over, while
    llama-server halved its batch 19 times and threw 4 speculative sub-batch errors. The
    buffer was 792 against a 2048 chunk.
    """

    def test_the_buffer_covers_the_batch(self):
        from core.inference.llama_preemption import preemption_buffer_tokens

        buffer = preemption_buffer_tokens(16384, slots = 4, batch_tokens = 2048)
        assert buffer >= 2048, (
            "a buffer smaller than one prefill chunk cannot prevent the decode failure "
            "that starts the shrinking-batch retry"
        )

    def test_drafts_are_added_on_top_of_the_batch(self):
        """They are cells the drafter puts in BEFORE acceptance, not part of the chunk."""
        from core.inference.llama_preemption import preemption_buffer_tokens

        plain = preemption_buffer_tokens(16384, slots = 4, batch_tokens = 2048)
        drafted = preemption_buffer_tokens(
            16384, slots = 4, batch_tokens = 2048, draft_tokens = 6
        )
        assert drafted == plain + 6 * 4

    def test_reaction_headroom_still_wins_when_it_is_larger(self):
        """max(), not a sum: both buy space for the next step, so the larger covers both.

        A tiny --batch-size must not shrink the buffer below the reaction headroom that
        several slots decoding at once still needs.
        """
        from core.inference.llama_preemption import preemption_buffer_tokens

        assert preemption_buffer_tokens(16384, slots = 8, batch_tokens = 64) == (
            preemption_buffer_tokens(16384, slots = 8)
        )

    def test_a_small_cache_is_not_given_a_ceiling_of_zero(self):
        """The batch can exceed a small -c outright; the cap has to survive it."""
        from core.inference.llama_preemption import preemption_buffer_tokens

        buffer = preemption_buffer_tokens(2048, slots = 4, batch_tokens = 2048)
        assert 0 < buffer <= 1024, buffer


class TestTheLiveCrashOf20260901:
    """Four chats armed, two were chosen as victims, neither ever paused.

    The log said `preempted=chatcmpl-...` and then nothing: no `paused`, no
    `awaiting-room`, no `resumed`. Meanwhile `committed` sat at 8192 of a 16384 cache
    while four chats really held 16384, and the model ran out of context space exactly
    as it did before any of this was written.

    Two independent defects, both of which these tests fail against.
    """

    @pytest.mark.asyncio
    async def test_the_participant_polls_the_signal_the_stream_polls(self):
        """Defect one. Without a shared signal, `register` builds its own, the caller
        hands a different one to the stream, and selecting a victim reaches nobody."""
        import routes.inference as inference

        queue = get_llama_admission_queue("http://127.0.0.1:20/")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )
        signal = PreemptSignal()
        inference._openai_llama_preemption_arm(
            request = None,
            llama_backend = _backend(url = "http://127.0.0.1:20/"),
            reservation = reservation,
            gen_id = "shared",
            signal = signal,
            loop = None,
        )
        controller = get_preemption_controller("http://127.0.0.1:20/")
        participant = controller.participant("shared")
        assert participant is not None
        assert participant.preempt_event is signal, (
            "the participant holds a different signal than the stream polls, so a "
            "preempt can never reach the stream"
        )

    def test_selecting_a_victim_actually_sets_the_streams_signal(self):
        controller = PreemptionController("victim-signal")
        controller.configure(budget = 8192, kv_unified = True)
        big = PreemptSignal()
        small = PreemptSignal()
        controller.register("big", tokens = 6000, signal = big)
        controller.register("small", tokens = 2000, signal = small)
        victims = controller.plan_preemptions(needed = 4000)
        assert victims, "nobody was selected although the cache is over its ceiling"
        assert small.is_set(), "the victim's own signal was never set"
        assert not big.is_set(), "the winner must keep decoding"

    def test_a_victim_keeps_holding_kv_until_the_pause_is_confirmed(self):
        """Defect two. Marking a victim PAUSED at planning time frees room that is
        still occupied, so the next arrival is admitted against a cache that is full."""
        controller = PreemptionController("honest-accounting")
        controller.configure(budget = 16384, kv_unified = True)
        for index in range(4):
            controller.register(f"c{index}", tokens = 4096, signal = PreemptSignal())
        before = controller.committed_tokens()
        assert before == 16384
        controller.plan_preemptions(needed = 4096)
        after = controller.committed_tokens()
        assert after == before, (
            f"asking for a pause freed {before - after} tokens that are still occupied; "
            "the room is only free once the stream confirms it stopped"
        )

    def test_the_room_is_released_only_on_confirmation(self):
        controller = PreemptionController("confirm")
        controller.configure(budget = 16384, kv_unified = True)
        for index in range(4):
            controller.register(f"c{index}", tokens = 4096, signal = PreemptSignal())
        victims = controller.plan_preemptions(needed = 4096)
        assert victims
        assert controller.committed_tokens() == 16384
        # What ControllerPreemptionPolicy.on_preempted does once the stream really stopped.
        controller.set_state(victims[0].gen_id, ParticipantState.PAUSED)
        assert controller.committed_tokens() == 16384 - 4096

    def test_an_already_asked_victim_is_not_asked_twice(self):
        """Otherwise a second arrival double-counts the room the first pause will free."""
        controller = PreemptionController("no-double")
        controller.configure(budget = 16384, kv_unified = True)
        for index in range(4):
            controller.register(f"c{index}", tokens = 4096, signal = PreemptSignal())
        first = {v.gen_id for v in controller.plan_preemptions(needed = 4096)}
        second = {v.gen_id for v in controller.plan_preemptions(needed = 4096)}
        assert not (first & second), f"re-selected {first & second}"


class TestTheDraftReserveActuallyApplied:
    """Defect three. The live buffer read 820 where 828 was expected, so the reserve
    silently did nothing on the one configuration that needed it."""

    def test_a_load_whose_spec_block_came_from_extra_args(self):
        import routes.inference as inference

        backend = _backend()
        backend.speculative_type = None
        backend.spec_drafter_kind = None
        backend.requested_spec_mode = None
        backend._speculative_type = "draft-mtp"
        backend._spec_draft_n_max = 2
        assert inference._openai_llama_speculative_draft_tokens(backend) == 2

    def test_an_active_drafter_with_no_stated_depth_reserves_the_default(self):
        import routes.inference as inference

        backend = _backend()
        backend.speculative_type = "draft-mtp"
        backend.spec_draft_n_max = None
        backend._spec_draft_n_max = None
        assert inference._openai_llama_speculative_draft_tokens(backend) == 6, (
            "None means the platform default, not zero"
        )

    def test_nothing_drafting_reserves_nothing(self):
        import routes.inference as inference

        backend = _backend()
        backend.speculative_type = None
        backend.spec_drafter_kind = None
        backend.requested_spec_mode = None
        backend._speculative_type = None
        backend.spec_draft_n_max = None
        backend._spec_draft_n_max = None
        assert inference._openai_llama_speculative_draft_tokens(backend) == 0


class TestAPauseCannotOutliveTheRoomItWaitsFor:
    """A 33-minute hang with nothing decoding, observed 2026-09-01.

    Three chats were interrupted, entered the pause handshake, and waited forever: the
    stream calls `await_resume()` with NO argument, the adapter mapped None to
    `future.result(timeout=None)`, and `resume_async` mapped it to no deadline. Both
    layers unbounded. That is strictly worse for a user than the crash it replaced,
    because a crash at least ends.
    """

    def test_no_argument_does_not_mean_forever(self):
        import inspect

        from core.inference.llama_preemption import DEFAULT_RESUME_WAIT_TIMEOUT_S

        source = inspect.getsource(ControllerPreemptionPolicy.await_resume)
        assert "if timeout is None:" in source, "an unstated timeout must acquire one"
        assert "DEFAULT_RESUME_WAIT_TIMEOUT_S" in source
        assert DEFAULT_RESUME_WAIT_TIMEOUT_S > 0

    def test_the_future_is_never_awaited_unbounded(self):
        import inspect

        source = inspect.getsource(ControllerPreemptionPolicy.await_resume)
        assert "timeout = None if timeout is None" not in source, (
            "the future would block forever when the caller stated no timeout"
        )

    def test_it_always_returns_promptly_rather_than_blocking(self):
        """The value depends on the path taken; the point is that it RETURNS.

        A participant holding no lease answers True at once (there is nothing to take
        back), one with no loop answers False. Neither may block, which is the property
        the hang violated.
        """
        import time

        for gen_id in ("known", "missing"):
            controller = PreemptionController(f"prompt-{gen_id}")
            controller.configure(budget = 16384, kv_unified = True)
            if gen_id == "known":
                controller.register(gen_id, tokens = 100, signal = PreemptSignal())
            policy = ControllerPreemptionPolicy(
                controller, gen_id, PreemptSignal(), loop = None
            )
            started = time.monotonic()
            result = policy.await_resume()
            assert result in (True, False)
            assert time.monotonic() - started < 5, f"{gen_id}: await_resume blocked"

    def test_giving_up_is_reported_as_false_not_raised(self):
        """False means "finish the turn with what you have", which the stream already
        knows how to do. An exception here would surface as a failed generation."""
        controller = PreemptionController("gave-up")
        controller.configure(budget = 16384, kv_unified = True)
        policy = ControllerPreemptionPolicy(
            controller, "missing", PreemptSignal(), loop = None
        )
        assert policy.await_resume(timeout = 0.01) is False

    @staticmethod
    def _waiting_policy(controller, gen_id, tokens):
        """A policy that actually reaches the wait loop.

        await_resume short-circuits before the loop when the participant has no lease
        (nothing to take back) or no event loop (nothing to take it back ON), so a stub
        for each is the only way to exercise the thing under test.
        """
        import asyncio
        import threading

        class _Lease:
            is_released = False
            tokens = 0

            async def resume_async(self, want, **kwargs):
                return True

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target = loop.run_forever, daemon = True)
        thread.start()
        controller.register(
            gen_id, lease = _Lease(), tokens = tokens, signal = PreemptSignal()
        )
        policy = ControllerPreemptionPolicy(
            controller, gen_id, PreemptSignal(), loop = loop
        )
        return policy, loop

    @staticmethod
    def _shutdown(loop):
        loop.call_soon_threadsafe(loop.stop)

    def test_two_waiters_cannot_both_be_granted_the_same_room(self):
        """The 2026-09-03 run 2 failure: 3 context-exhaustion errors and 4 speculative
        sub-batch errors while sampled residency never once passed the ceiling.

        `room_for` only answers a question. PAUSED is not in `_HOLDS_KV`, so two paused
        chats asking at the same moment do not appear in each other's arithmetic and both
        get yes, then both prefill. The overflow happens inside one prefill, between two
        residency samples, which is why the watermark never saw it.
        """
        controller = PreemptionController("double-grant")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        for gen_id in ("a", "b"):
            controller.register(gen_id, tokens = 9000, signal = PreemptSignal())
            controller.set_state(gen_id, ParticipantState.PAUSED)

        snapshot = controller.snapshot()
        ceiling = snapshot.budget - snapshot.buffer
        # Each fits alone; together they do not. That is exactly the shape that crashed.
        assert 9000 <= ceiling < 18000, ceiling

        assert controller.room_for("a", 9000) is True
        assert controller.room_for("b", 9000) is True, (
            "the question is answered the same way twice, which is the bug"
        )

        assert controller.try_grant_resume("a", 9000) is True
        assert controller.try_grant_resume("b", 9000) is False, (
            "the second waiter must be refused: the first has already booked the room"
        )

    def test_a_grant_that_does_not_resume_is_handed_back(self):
        controller = PreemptionController("grant-rollback")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        for gen_id in ("a", "b"):
            controller.register(gen_id, tokens = 9000, signal = PreemptSignal())
            controller.set_state(gen_id, ParticipantState.PAUSED)
        assert controller.try_grant_resume("a", 9000) is True
        assert controller.try_grant_resume("b", 9000) is False
        controller.note_resume_failed("a")
        assert controller.try_grant_resume("b", 9000) is True, (
            "room booked by a resume that never happened must not stay booked"
        )

    def test_a_stalled_wait_still_ends(self):
        """The hang this bound exists for: nothing decoding, nothing moving.

        A full cache that nobody is draining produces no progress at all, so the stall
        deadline expires on schedule and the turn finishes with what it has.
        """
        import time

        controller = PreemptionController("stalled")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("holder", tokens = 14000, signal = PreemptSignal())
        controller.note_tokens("holder", 14000)  # measured, and going nowhere
        policy, loop = self._waiting_policy(controller, "waiter", 8000)
        try:
            started = time.monotonic()
            assert policy.await_resume(timeout = 0.5) is False
            elapsed = time.monotonic() - started
        finally:
            self._shutdown(loop)
        assert 0.4 < elapsed < 5.0, f"stall must end near the timeout, took {elapsed}s"

    def test_progress_buys_more_patience_than_the_wall_clock_allows(self):
        """The defect that killed two live chats: 90s of wall clock against an answer
        that legitimately takes minutes, while the cache was steadily turning over.

        Room is released from another thread AFTER the flat deadline would have fired.
        A wall-clock bound gives up; a stall bound waits, because the cache kept giving
        room back the whole time.
        """
        import threading
        import time

        controller = PreemptionController("progressing")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("holder", tokens = 14000, signal = PreemptSignal())
        controller.note_tokens("holder", 14000)

        stop = threading.Event()

        def drain():
            # Room appearing, a little at a time, for well longer than `timeout`.
            held = 14000
            while not stop.is_set() and held > 1000:
                time.sleep(0.05)
                held -= 200
                controller.note_tokens("holder", held)

        worker = threading.Thread(target = drain, daemon = True)
        worker.start()
        policy, loop = self._waiting_policy(controller, "waiter", 4000)
        try:
            started = time.monotonic()
            policy.await_resume(timeout = 0.3)
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            worker.join(timeout = 5)
            self._shutdown(loop)
        assert elapsed > 0.4, (
            f"gave up after {elapsed}s despite the cache draining throughout; a stall "
            "deadline must reset on progress"
        )

    def test_growth_is_not_progress(self):
        """Otherwise the deadline never expires: other chats decoding into the cache is
        the opposite of room appearing, and resetting on it would restore the hang."""
        controller = PreemptionController("growing")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("a", tokens = 1000, signal = PreemptSignal())
        controller.observe("a", 0)
        controller.note_resident(1000)
        before = controller.progress_signature()
        controller.observe("a", 5000)
        after = controller.progress_signature()
        assert after != before, "the signature must move when occupancy moves"
        assert after[0] > before[0], "and growth must be visible as growth"
        assert after[1] == before[1], "with the same holders"

    def test_the_events_reach_the_logger_studio_actually_configures(self):
        """`paused` never appeared in the live log, which read as "the handshake never
        ran". The handshake may well have run: the module was writing to a stdlib
        logger Studio does not configure, so the evidence was simply discarded."""
        from pathlib import Path

        import core.inference.llama_preemption as module

        source = Path(module.__file__).read_text()
        assert "from loggers import get_logger" in source
        assert "logging.getLogger" not in source, (
            "a stdlib logger here is dropped, and silence gets read as absence"
        )


class TestTheCacheIsNeverHandedOutToTheLastToken:
    """Four chats died with preemption working perfectly, 2026-09-01.

    armed 4, paused 3, resumed 3, peak requests_processing 4: the pause/resume cycle did
    exactly what it was built to do. They still all died, because admission had handed
    out `4 * 4096 = 16384` of a 16384 cache. At 100% with zero headroom the speculative
    drafts had nowhere to go, and the overrun was 24 cells.

    This class first asserted the reserve by subtracting it from the admission budget and
    checking `capacity * share` left headroom. Both halves of that belong to the design
    where a chat was confined to a share of the cache. A chat is now clamped to the whole
    window and the cache is deliberately overcommitted, so `capacity * share` describes
    nothing, and subtracting the reserve from the budget reserved the drafts twice over
    while reporting a cache smaller than llama-server was launched with.

    The property is unchanged and is asserted where it now lives: the watermark holds
    back at least the drafts, so the cache is never worked right up to its last cell.
    """

    def _buffer(self, budget, drafts, slots):
        from core.inference.llama_preemption import preemption_buffer_tokens

        return preemption_buffer_tokens(budget, draft_tokens = drafts, slots = slots)

    def test_the_headroom_covers_the_drafts(self):
        # The measured case: 16384 over four slots, two draft tokens each.
        assert self._buffer(16384, 2, 4) >= 2 * 4

    def test_the_ceiling_is_below_the_cache(self):
        budget = 16384
        assert budget - self._buffer(budget, 2, 4) < budget, (
            "the cache would be worked to its last cell, which is how the drafts overran"
        )

    def test_it_holds_across_cache_sizes_and_slot_counts(self):
        for total in (2048, 4096, 16384, 65536, 262144):
            for slots in (2, 3, 4, 8):
                buffer = self._buffer(total, 2, slots)
                assert buffer >= 2 * slots, f"{total}/{slots}: drafts not covered"
                assert total - buffer < total, f"{total}/{slots}: no headroom"

    def test_no_speculation_still_leaves_a_margin(self):
        """The token figures are estimates, not tokenisations, in every case.

        `estimate_messages_tokens_dense` approximates, so a cache handed out to the last
        token only works if the arithmetic is exact, and it is not.
        """
        assert self._buffer(16384, 0, 4) > 0

    def test_a_tiny_cache_is_reduced_not_erased(self):
        """A buffer that swallows the budget leaves a ceiling of zero.

        That reads as "no room for anyone" and would preempt every participant on every
        call, forever, so it is capped at half.
        """
        for total in (256, 512, 1024):
            buffer = self._buffer(total, 8, 8)
            assert 0 < buffer <= total // 2, f"{total}: buffer {buffer} erases the cache"

    def test_the_budget_is_the_cache_llama_server_was_launched_with(self):
        """No hidden subtraction. The reserve is the watermark's job, in one place."""
        import routes.inference as inference

        backend = _backend()
        backend.speculative_type = "draft-mtp"
        backend.spec_drafter_kind = None
        backend.requested_spec_mode = None
        backend.spec_draft_n_max = 2
        assert inference._openai_llama_admission_budget(backend) == 16384

class TestEveryChatGetsTheWholeWindow:
    """N for everyone, then evict. The design asked for from the start.

    Dividing the cache into `N / slots` was the stopgap while nothing could pause, and
    its cost was measured on 2026-09-01: a chat held to ~4049 tokens an attempt, a long
    answer grinding through length continuations, two of four not finishing inside 900s
    while the cache sat mostly idle. vLLM admits against the full max_model_len and
    preempts at a watermark; so do we now.
    """

    def _backend(self, slots = 4, total = 16384):
        backend = _backend()
        backend.effective_parallel_slots = slots
        backend.context_length = total
        backend._kv_cache_context_total = total
        backend.speculative_type = "draft-mtp"
        backend.spec_draft_n_max = 2
        backend.spec_drafter_kind = None
        backend.requested_spec_mode = None
        return backend

    def test_a_chat_is_permitted_the_window_not_a_share_of_it(self):
        import routes.inference as inference

        backend = self._backend()
        payload = _chat_payload()
        permitted = inference._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        share = inference._openai_llama_admission_budget(backend) // 4
        assert permitted > share * 3, (
            f"permitted {permitted} is still a share ({share}), not the window"
        )

    def test_four_chats_are_each_permitted_the_window(self):
        """They collectively exceed the cache ON PURPOSE. Preemption reclaims."""
        import routes.inference as inference

        backend = self._backend()
        total = 0
        for _ in range(4):
            total += inference._openai_llama_admission_enforced_max_tokens(
                _chat_payload(), request = None, llama_backend = backend
            )
        assert total > 16384, "the cache is meant to be overcommitted now"

    def test_a_stated_cap_is_still_honoured(self):
        import routes.inference as inference

        assert inference._openai_llama_admission_enforced_max_tokens(
            _chat_payload(max_tokens = 512), request = None, llama_backend = self._backend()
        ) is None


class TestTheWatermarkSweep:
    """With the arithmetic no longer preventing an overrun, this is what does."""

    def _filled(self, budget = 16384, chats = 4, each = 1000):
        controller = PreemptionController(f"sweep-{budget}-{chats}-{each}")
        controller.configure(budget = budget, kv_unified = True)
        signals = {}
        for index in range(chats):
            signals[index] = PreemptSignal()
            controller.register(f"c{index}", tokens = each, signal = signals[index])
        return controller, signals

    def test_growth_below_the_watermark_evicts_nobody(self):
        controller, _ = self._filled()
        assert controller.observe("c0", 500) == []

    def test_growth_past_the_watermark_evicts(self):
        controller, signals = self._filled()
        victims = []
        for grown in (2000, 3000, 4000):
            for index in range(4):
                victims += controller.observe(f"c{index}", grown)
        assert victims, "the cache passed its ceiling and nobody was asked to stop"
        assert any(s.is_set() for s in signals.values())

    def test_the_newest_arrival_is_the_first_victim(self):
        """This asserted the reverse, that the biggest chat is never a victim.

        Largest-first ranked 5th of 7 policies over nine simulated regimes and last on
        fairness. vLLM V1 evicts the most recently arrived instead, so the work already
        done survives, and that ranked best overall.
        """
        controller, signals = self._filled(each = 2000)
        controller.register("big", tokens = 9000, signal = PreemptSignal())
        victims = {v.gen_id for v in controller.observe("big", 3000)}
        assert "big" in victims, "the newest registration should stop first"

    def test_live_growth_is_added_to_the_admitted_charge(self):
        """Reporting "n generated" must not drop the prompt already resident."""
        controller, _ = self._filled(chats = 1, each = 4000)
        controller.observe("c0", 1000)
        assert controller.committed_tokens() == 5000

    def test_a_round_boundary_rebaselines(self):
        """note_tokens restates the whole conversation, so later growth counts from there."""
        controller, _ = self._filled(chats = 1, each = 1000)
        controller.observe("c0", 500)
        assert controller.committed_tokens() == 1500
        controller.note_tokens("c0", 6000)
        controller.observe("c0", 100)
        assert controller.committed_tokens() == 6100, (
            "growth after a round must be measured from the round, not from admission"
        )


def _chat_payload(**fields):
    class _P:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, _name):
            return None

    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16384}
    base.update(fields)
    return _P(**base)


class TestTheStreamActuallyReportsGrowth:
    """The sweep is only as good as the thing feeding it.

    Deleting the counter increment from the chunk loop left every other test green: the
    controller's own eviction logic is fine in isolation, and nothing asserted that the
    stream still tells it anything. That is the same shape as the two live failures
    today, where the policy was correct and simply never reached.
    """

    def _source(self):
        from pathlib import Path

        import core.inference.llama_cpp as llama_cpp

        return Path(llama_cpp.__file__).read_text()

    def test_the_chunk_loop_counts_tokens(self):
        assert "_tokens_this_stream += 1" in self._source(), (
            "nothing increments the live token count, so the sweep sees a frozen n_i"
        )

    def test_the_count_is_reported_to_the_preemptor(self):
        source = self._source()
        assert "on_tokens(_tokens_this_stream)" in source, (
            "the count is kept but never handed to the watermark sweep"
        )

    def test_the_report_is_batched_not_per_token(self):
        """A lock per token would put the preemptor on the hot path."""
        source = self._source()
        assert "_tokens_this_stream % _TOKEN_REPORT_EVERY == 0" in source

    def test_the_batch_is_small_enough_to_be_caught_by_the_buffer(self):
        """Overshoot between reports must fit inside the headroom, or the sweep learns
        about the overrun after llama-server does."""
        from core.inference.llama_cpp import _TOKEN_REPORT_EVERY
        from core.inference.llama_preemption import preemption_buffer_tokens

        assert 0 < _TOKEN_REPORT_EVERY <= 64
        # This is the whole reason the buffer is per-slot. Worst case every slot
        # overshoots by a full batch between reports, so the lag scales with the slot
        # count and the headroom has to scale with it too. Computing the buffer for one
        # slot while computing the lag for eight compared two different systems.
        for slots in (1, 2, 4, 8, 16):
            worst = _TOKEN_REPORT_EVERY * slots
            buffer = preemption_buffer_tokens(16384, slots = slots)
            assert worst < buffer, (
                f"{slots} slots lag by {worst} tokens against a {buffer} buffer"
            )

    def test_the_route_supplies_the_callback(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        assert "on_tokens = _gguf_observe_tokens," in source
        assert ".observe(completion_id, generated)" in source


class TestResumingDoesNotThrash:
    """44 preemptions across four chats, one producing 611 characters in 374 seconds.

    Observed 2026-09-01 the first time the cache was deliberately overcommitted. Eviction
    was gated on the LIVE total while resume was gated on the admission queue's optimistic
    charge, so a chat was let back in while the cache was still over its watermark and the
    next sweep threw it straight back out. A resume undone by the next sweep is worse than
    waiting: it pays a prefill for nothing.
    """

    def test_no_room_while_the_others_still_hold_it(self):
        controller = PreemptionController("thrash-1")
        controller.configure(budget = 16384, kv_unified = True)
        for index in range(2):
            controller.register(f"c{index}", tokens = 7000, signal = PreemptSignal())
        assert controller.room_for("c0", 12000) is False, (
            "a resume was permitted while the cache was already over its watermark"
        )

    def test_room_appears_once_a_holder_stops(self):
        """The approved policy: the longest chat continues, and the rest resume once it
        frees its room."""
        controller = PreemptionController("thrash-2")
        controller.configure(budget = 16384, kv_unified = True)
        for index in range(2):
            controller.register(f"c{index}", tokens = 7000, signal = PreemptSignal())
        assert controller.room_for("c0", 12000) is False
        controller.set_state("c1", ParticipantState.PAUSED)
        assert controller.room_for("c0", 12000) is True

    def test_a_generation_does_not_count_against_its_own_resume(self):
        controller = PreemptionController("thrash-3")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("solo", tokens = 9000, signal = PreemptSignal())
        assert controller.room_for("solo", 9000) is True

    def test_the_gate_is_off_when_preemption_is(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        controller = PreemptionController("thrash-4")
        controller.configure(budget = 16384, kv_unified = True)
        for index in range(4):
            controller.register(f"c{index}", tokens = 9000, signal = PreemptSignal())
        assert controller.room_for("c0", 9000) is True, (
            "with preemption off nothing should wait on its watermark"
        )

    def test_the_resume_path_consults_it(self):
        import inspect

        source = inspect.getsource(ControllerPreemptionPolicy.await_resume)
        assert "room_for" in source, (
            "await_resume takes the lease back without checking the live cache"
        )
        assert "gave-up" in source, "giving up after the timeout must be visible"


class TestTheCacheHoldsMoreThanTheLedgerKnows:
    """`purging slot 1 with 16383 tokens`, observed 2026-09-01.

    An idle slot held the ENTIRE 16384-cell cache, left behind by a request that had
    already finished, while four chats were scheduled against a ledger that believed the
    cache was nearly empty. llama.cpp keeps a slot's prompt cache for prefix reuse; the
    admission ledger cannot see it and /metrics does not report it. That residue is why
    the watermark kept firing too late and llama-server kept dropping into the
    shrinking-batch retry where upstream #24840 throws.

    The original design said explicitly not to add a GET /slots reader. That was written
    before this was understood, and it is the only endpoint that can report it.
    """

    def test_residency_counts_what_the_ledger_cannot_see(self):
        from core.inference.llama_preemption import read_slot_occupancy

        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens_cache": 16383},
            {"id": 1, "is_processing": True, "n_prompt_tokens_cache": 2000},
            {"id": 2, "is_processing": False, "n_prompt_tokens_cache": 9209},
        ]
        occupancy = read_slot_occupancy(lambda: slots)
        assert occupancy["resident"] == 27592
        assert [slot for slot, _ in occupancy["idle"]] == [0, 2], "largest idle first"

    def test_generated_tokens_count_as_residency(self):
        """The undercount that made every watermark diagnostic come back innocent.

        `/slots` reports the prompt. The tokens decoded since occupy cells too, and on a
        chat writing a long answer they are most of it. Sampled live 2026-09-03 while one
        slot decoded: reported 12632, decoded 6323, true 18955, in a 16384 cache, against
        a ceiling of 14312. The watermark never fired because the number it watches never
        passed the ceiling.
        """
        from core.inference.llama_preemption import read_slot_occupancy

        slots = [
            {
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens": 12632,
                "next_token": [{"n_decoded": 6323}],
            },
        ]
        occupancy = read_slot_occupancy(lambda: slots)
        assert occupancy["resident"] == 18955, (
            "a decoding slot holds its prompt AND everything it has generated"
        )

    def test_a_finished_slot_is_not_counted_twice(self):
        """An idle slot's prompt cache already holds the whole sequence it produced, so
        adding a stale n_decoded on top would count the generated half twice."""
        from core.inference.llama_preemption import read_slot_occupancy

        slots = [
            {
                "id": 0,
                "is_processing": False,
                "n_prompt_tokens_cache": 9000,
                "next_token": [{"n_decoded": 4000}],
            },
        ]
        assert read_slot_occupancy(lambda: slots)["resident"] == 9000

    def test_the_decoded_count_is_read_from_either_shape(self):
        """llama-server nests it in a one-element list here and a bare object elsewhere."""
        from core.inference.llama_preemption import read_slot_occupancy

        for shape in ([{"n_decoded": 500}], {"n_decoded": 500}):
            slots = [
                {"id": 0, "is_processing": True, "n_prompt_tokens": 1000,
                 "next_token": shape},
            ]
            assert read_slot_occupancy(lambda: slots)["resident"] == 1500, shape

    def test_a_missing_or_malformed_next_token_reads_as_zero(self):
        """An occupancy read that raises takes the whole watermark sweep with it."""
        from core.inference.llama_preemption import read_slot_occupancy

        for shape in (None, [], "nonsense", {"n_decoded": "abc"}, {}):
            slots = [
                {"id": 0, "is_processing": True, "n_prompt_tokens": 1000,
                 "next_token": shape},
            ]
            assert read_slot_occupancy(lambda: slots)["resident"] == 1000, shape

    def test_an_unreadable_endpoint_is_not_an_empty_cache(self):
        from core.inference.llama_preemption import read_slot_occupancy

        assert read_slot_occupancy(lambda: None) is None
        assert read_slot_occupancy(lambda: []) is None

    def test_a_chat_that_has_not_prefilled_is_added_to_what_the_cache_holds(self):
        """An idle slot's residue plus a prompt still to be sent are different cells."""
        controller = PreemptionController("resident")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("a", tokens = 2000, signal = PreemptSignal())
        assert controller.committed_tokens() == 2000
        controller.note_resident(16383)
        # "a" has not decoded a token, so its 2000 are NOT among the 16383 the cache is
        # already holding: they are a prefill still to come, and both have to fit.
        assert controller.committed_tokens() == 18383, (
            "the cache is full and the ledger does not know it"
        )
        controller.note_resident(None)
        assert controller.committed_tokens() == 2000, "a failed read falls back, not to zero"

    def test_a_chat_already_in_the_cache_is_not_counted_a_second_time(self):
        """The regression that paused chats against a cache that was half empty.

        Taking max(ledger, resident) was safe; adding the two was not, and the old split
        did neither cleanly. Measured over 1218 samples of a live four-chat run, the
        ledger overstated the cache in 1212 of them because every chat carried an equal
        share reservation on top of cells llama-server had already counted.
        """
        controller = PreemptionController("resident-measured")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("a", tokens = 4096, signal = PreemptSignal())
        # One token comes back: whatever "a" really holds, the cache has now reported it.
        controller.observe("a", 1)
        controller.note_resident(1400)
        assert controller.committed_tokens() == 4097, (
            "the ledger is the larger of the two opinions about the same cells"
        )
        # And once the round boundary restates the conversation honestly, the reservation
        # stops inflating the total at all.
        controller.note_tokens("a", 1400)
        assert controller.committed_tokens() == 1400, (
            "a measured chat must be counted once, not once per source"
        )

    def test_a_lagging_reading_cannot_shrink_a_measured_chat(self):
        """resident lags a prefill in progress, so the larger figure still wins."""
        controller = PreemptionController("resident-lag")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("a", tokens = 9000, signal = PreemptSignal())
        controller.observe("a", 0)
        controller.note_resident(12)  # mid prefill, almost nothing visible yet
        assert controller.committed_tokens() == 9000

    def test_a_resume_is_refused_against_a_cache_only_slots_can_see(self):
        controller = PreemptionController("resident-room")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("a", tokens = 2000, signal = PreemptSignal())
        controller.note_resident(16383)
        assert controller.room_for("a", 4000) is False
        controller.note_resident(3000)
        assert controller.room_for("a", 4000) is True

    def test_idle_residue_is_reclaimed_before_a_live_chat_is_paused(self):
        from core.inference.llama_preemption import read_slot_occupancy, reclaim_idle_slots

        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens_cache": 16383},
            {"id": 1, "is_processing": True, "n_prompt_tokens_cache": 2000},
        ]
        occupancy = read_slot_occupancy(lambda: slots)
        erased = []

        def _erase(slot_id):
            erased.append(slot_id)
            return dict(occupancy["idle"])[slot_id]

        freed = reclaim_idle_slots(occupancy, _erase, needed = 10000)
        assert freed == 16383
        assert erased == [0], "the busy slot must never be erased"

    def test_a_failing_erase_does_not_take_the_generation_with_it(self):
        from core.inference.llama_preemption import read_slot_occupancy, reclaim_idle_slots

        occupancy = read_slot_occupancy(
            lambda: [{"id": 0, "is_processing": False, "n_prompt_tokens_cache": 900}]
        )

        def _boom(_slot_id):
            raise RuntimeError("endpoint disabled")

        assert reclaim_idle_slots(occupancy, _boom, needed = 500) == 0

    def test_nothing_is_erased_when_nothing_is_needed(self):
        from core.inference.llama_preemption import read_slot_occupancy, reclaim_idle_slots

        occupancy = read_slot_occupancy(
            lambda: [{"id": 0, "is_processing": False, "n_prompt_tokens_cache": 900}]
        )
        assert reclaim_idle_slots(occupancy, lambda _i: 900, needed = 0) == 0

    def test_the_route_polls_residency_and_reclaims_first(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        assert "_gguf_refresh_residency(controller)" in source
        assert "controller.note_resident(" in source
        assert "reclaim_idle_slots(" in source, (
            "a live chat is paused without first freeing dead residue"
        )


class TestIdleResidueIsFreedWhenSeenNotWhenDesperate:
    """Reclaiming only once a victim had been chosen made the erase almost useless.

    Measured 2026-09-02: three reclaims freeing 1949, 3103 and 844 tokens while
    llama-server still entered its shrinking-batch retry 21 times and threw the
    speculative sub-batch error 4 times. By the time a victim is being chosen the cache
    is already in trouble. An idle slot's cache belongs to a request that has finished,
    so there is no reason to wait for trouble before freeing it.
    """

    def test_the_route_reclaims_on_sight(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        assert "reclaimed-idle-early" in source, (
            "residue is only reclaimed once a live chat is already being paused"
        )
        # And the reclaim must sit in the residency refresh, not behind the victim check.
        refresh = source.split("def _gguf_refresh_residency", 1)[1].split("def ", 1)[0]
        assert "reclaim_idle_slots(" in refresh

    def test_the_residency_figure_is_corrected_after_freeing(self):
        from pathlib import Path

        import routes.inference as inference

        refresh = (
            Path(inference.__file__).read_text()
            .split("def _gguf_refresh_residency", 1)[1].split("def ", 1)[0]
        )
        assert "controller.note_resident(" in refresh
        assert "- freed" in refresh, (
            "the controller would keep planning against the pre-erase figure"
        )


class TestEveryNameThePreemptPathCallsIsActuallyBound:
    """A NameError on the pause path kills the chat it was trying to save.

    Shipped 2026-09-02 and caught only by a live run: an import was inserted into the
    first textual match of the import statement, which was an unrelated local import
    deep in the GGUF metadata reader, so the names were bound in a function that never
    used them and unbound in the one that did. The file still compiled, `ast.parse`
    still passed, the helpers still imported cleanly from their own module, and the
    first preemption raised `NameError: name 'trailing_assistant_reasoning' is not
    defined`. Two chats died in under twenty seconds.

    Checking that a helper exists is not the same as checking its caller can see it.
    """

    def test_the_resume_helpers_are_bound_wherever_they_are_called(self):
        """Every function that calls them must import them, whichever one that is.

        Pinned to a function name at first, which broke the moment the block was
        extracted into a helper. The invariant is not "this function imports it", it is
        "no function calls a name it cannot see".
        """
        import ast
        from pathlib import Path

        from core.inference import llama_cpp

        watched = {"trailing_assistant_reasoning", "trailing_assistant_resumable"}
        tree = ast.parse(Path(llama_cpp.__file__).read_text())
        module_level = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_level.update((a.asname or a.name).split(".")[0] for a in node.names)

        seen_any = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = {
                n.func.id
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            wanted = watched & called
            if not wanted:
                continue
            seen_any = True
            bound = set(module_level)
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    bound.update((a.asname or a.name).split(".")[0] for a in inner.names)
            missing = wanted - bound
            assert not missing, (
                f"{node.name} calls {sorted(missing)} without importing them; the first "
                f"preemption raises NameError"
            )
        assert seen_any, "nothing calls the resume helpers any more; drop this test"

    def test_the_helpers_import_from_their_own_module(self):
        import core.inference.chat_template_helpers as helpers

        assert callable(helpers.trailing_assistant_reasoning)
        assert callable(helpers.trailing_assistant_resumable)

    def test_the_gguf_metadata_reader_did_not_keep_the_stray_import(self):
        """The block the bad insertion landed in. It compiled, which is the whole point."""
        from pathlib import Path

        from core.inference import llama_cpp

        source = Path(llama_cpp.__file__).read_text()
        assert "                                        trailing_assistant_reasoning,\n" not in source


class TestAFailureInTheResumeCannotKillTheChat:
    def test_the_assembly_is_guarded(self):
        from pathlib import Path

        from core.inference import llama_cpp

        source = Path(llama_cpp.__file__).read_text()
        block = source.split("_carried_truncations = list(_respawn_truncations)", 1)[1]
        block = block.split("preempt_policy.on_preempted", 1)[0]
        assert "_assemble_preempt_resume(" in block
        assert "except Exception:" in block, (
            "a bug in the save-the-chat path would end the chat it was saving"
        )

    def test_a_raising_assembly_degrades_to_re_issuing_whole(self):
        from core.inference.llama_preemption import StreamCheckpoint

        # The behaviour the guard buys: a broken assembly leaves the conversation
        # untouched and the turn continues, rather than propagating out of the
        # generator as an api_error.
        convo = [{"role": "user", "content": "hi"}]

        class Boom:
            def _assemble_preempt_resume(self, *a, **k):
                raise NameError("trailing_assistant_reasoning is not defined")

        resumed = None
        try:
            resumed = Boom()._assemble_preempt_resume(
                convo, StreamCheckpoint(reasoning_text = "x"), "", "x",
            )
        except Exception:
            resumed = False
        assert resumed is False
        assert convo == [{"role": "user", "content": "hi"}]

    def test_assembly_returns_false_when_nothing_was_produced(self):
        from core.inference.llama_cpp import LlamaCppBackend
        from core.inference.llama_preemption import StreamCheckpoint

        convo = [{"role": "user", "content": "hi"}]
        out = LlamaCppBackend._assemble_preempt_resume(
            object(), convo, StreamCheckpoint(), "", "",
        )
        assert out is False
        assert len(convo) == 1, "an empty assistant turn would be refused downstream"

    def test_assembly_carries_a_thought_as_reasoning_not_content(self):
        from core.inference.llama_cpp import LlamaCppBackend
        from core.inference.llama_preemption import StreamCheckpoint

        convo = [{"role": "user", "content": "hi"}]
        out = LlamaCppBackend._assemble_preempt_resume(
            object(), convo, StreamCheckpoint(reasoning_text = "half a thought"),
            "", "half a thought",
        )
        assert out is True
        assert convo[-1]["reasoning_content"] == "half a thought"
        assert convo[-1]["content"] == "", (
            "a thought placed in content would be rendered as the answer"
        )

    def test_a_second_pause_merges_rather_than_replaces_the_thought(self):
        from core.inference.llama_cpp import LlamaCppBackend
        from core.inference.llama_preemption import StreamCheckpoint

        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "first half "},
        ]
        LlamaCppBackend._assemble_preempt_resume(
            object(), convo, StreamCheckpoint(reasoning_text = "second half"),
            "", "second half",
        )
        # The accumulators reset each round, so replacing would lose the first pause.
        assert convo[-1]["reasoning_content"] == "first half second half"
        assert len(convo) == 2


class TestAPauseActuallyFreesCells:
    """Aborting the upstream request stops the decode; it does not free the cache.

    llama-server keeps a finished slot's prompt cache for prefix reuse, so a paused chat
    still occupies its cells. Three chats paused together therefore hold the whole cache
    between them and every one of them waits for room only the others could return.

    Measured 2026-09-02 on the build that carried thoughts across a pause and charged
    them correctly: `want` climbed 4049 -> 4625 -> 9532 as each replayed thought became
    prompt, only two reclaims fired totalling 5620 tokens against a 16384 cache, and all
    three chats hit "no room within 90.0s". Correct accounting was not enough; the cells
    have to actually go, which is what vLLM's RECOMPUTE does.
    """

    def _refresh_source(self):
        from pathlib import Path

        import routes.inference as inference

        return (
            Path(inference.__file__).read_text()
            .split("def _gguf_refresh_residency", 1)[1].split("\n            def ", 1)[0]
        )

    def test_waiting_alone_triggers_a_reclaim(self):
        source = self._refresh_source()
        assert "snapshot, \"paused\"" in source or "getattr(snapshot, \"paused\"" in source, (
            "reclaim still fires only when over the watermark, so a pause frees nothing"
        )

    def test_the_reclaim_is_not_gated_on_being_over_the_ceiling(self):
        source = self._refresh_source()
        assert "if needed > 0 and occupancy.get(\"idle\")" in source
        assert "if over > 0 and occupancy.get(\"idle\")" not in source, (
            "the old gate is back; a paused chat's cells would be held until overflow"
        )

    def test_the_waiter_count_is_logged(self):
        """So a run can be read afterwards without guessing why a reclaim fired."""
        source = self._refresh_source()
        assert "waiting = waiting" in source


class TestAFinishedGenerationGivesItsChargeBack:
    """The ledger only ever grew, so eventually nobody could be admitted.

    `_openai_llama_preemption_disarm` was written and never called. Every chat that
    ended stayed registered with its tokens committed: the ones that finished normally,
    the ones that gave up waiting for room, and the ones killed by a stream error. Once
    the accumulated total passed the ceiling the next chat waited for room that could
    never arrive, and unlike the resume wait that path has no timeout, so it waited until
    the client disconnected. Observed 2026-09-02: one chat of four open for the full
    2400s deadline while llama-server sat idle with every slot released and
    `requests_processing` at 0.
    """

    def test_the_disarm_is_called(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        calls = source.count("_openai_llama_preemption_disarm(")
        assert calls >= 2, (
            "disarm is defined but never called, so charges accumulate forever"
        )

    def test_it_runs_in_a_finally_so_an_error_path_still_releases(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        call_at = source.index("                    _openai_llama_preemption_disarm(")
        # The nearest preceding block opener must be a `finally:`, or the paths that
        # matter most here (gave-up, stream error, disconnect) would skip it.
        preceding = source[:call_at]
        # The nearest block opener before the call, ignoring comments and blank lines.
        openers = [
            line.strip() for line in preceding.splitlines()
            if line.strip().endswith(":") and not line.strip().startswith("#")
        ]
        assert openers and openers[-1] == "finally:", (
            f"disarm sits under {openers[-1] if openers else 'nothing'}, not a finally, "
            f"so gave-up, stream-error and disconnect paths would skip it"
        )

    def _controller(self, budget = 16384):
        from core.inference.llama_preemption import PreemptionController

        controller = PreemptionController("disarm-test")
        controller.configure(budget = budget, kv_unified = True)
        return controller

    def test_unregister_actually_drops_the_charge(self):
        from core.inference.llama_preemption import ParticipantState

        controller = self._controller()
        controller.register("a", tokens = 8000)
        controller.register("b", tokens = 4000)
        assert controller.snapshot().committed == 12000

        controller.unregister("a")
        assert controller.snapshot().committed == 4000, (
            "a finished chat still counts against the cache"
        )
        assert controller.participant("a") is None
        # And the survivor is untouched.
        assert controller.participant("b").state == ParticipantState.DECODING

    def test_unregistering_twice_is_harmless(self):
        controller = self._controller()
        controller.register("a", tokens = 100)
        controller.unregister("a")
        controller.unregister("a")
        assert controller.snapshot().committed == 0

    def test_a_replayed_charge_is_released_too(self):
        """note_replayed raises base_tokens, so a leak here is larger than the prompt."""
        controller = self._controller()
        controller.register("a", tokens = 1000)
        controller.note_replayed("a", 4000)
        controller.observe("a", 0)
        assert controller.snapshot().committed == 5000
        controller.unregister("a")
        assert controller.snapshot().committed == 0


class TestDroppingTheChargeWithoutTheCellsIsWorseThanNeither:
    """Freeing a finished chat's charge while llama-server still holds its cells.

    The ledger then reports room the cache does not have, and admission overcommits
    against space that does not exist. That is the crash, not a stall, which makes it
    strictly worse than the leak it replaced: an over-counted ledger is only pessimistic.

    Measured 2026-09-02 across two consecutive builds on the same scenario. Before the
    disarm: 3 of 4 chats completed, 0 context-exhaustion errors, 1 KV retry. After the
    disarm alone: 0 of 4, 3 context-exhaustion errors, 42 KV retries, and idle reclaims
    fell from 5 to 2 because the emptier-looking ledger left nobody waiting to trigger
    them.
    """

    def _disarm_source(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        return source.split("def _openai_llama_preemption_disarm", 1)[1].split("\ndef ", 1)[0]

    def test_the_disarm_releases_cells_as_well_as_the_charge(self):
        body = self._disarm_source()
        assert "unregister(" in body, "the charge is not released"
        assert "reclaim_idle_slots(" in body, (
            "only the charge is released, so the ledger reports room the cache lacks"
        )

    def test_the_cells_go_after_the_charge_not_before(self):
        """Order matters: the slot is only idle once the response is finished."""
        body = self._disarm_source()
        assert body.index("unregister(") < body.index("reclaim_idle_slots(")

    def test_neither_half_can_fail_the_response(self):
        body = self._disarm_source()
        assert body.count("except Exception:") >= 2, (
            "bookkeeping must never fail a response that already succeeded"
        )

    def test_the_release_is_logged(self):
        body = self._disarm_source()
        assert "released-cells" in body

    def test_the_residency_figure_is_corrected(self):
        body = self._disarm_source()
        assert "note_resident(" in body, (
            "the controller would keep planning against the pre-erase figure"
        )


def _await_resume_body(source: str) -> str:
    """The adapter's await_resume, not one of the four other definitions of that name.

    Splitting on the bare name picked up a Protocol stub, whose body is 53 characters and
    contains none of what these assertions look for, so the tests failed against correct
    code.
    """
    cls = source.split("class ControllerPreemptionPolicy", 1)[1]
    return cls.split("def await_resume", 1)[1].split("\n    def ", 1)[0]


class TestTheResumeWaitNeverWaitsForImpossibleRoom:
    """The live hang, at its source.

    A resumed run replays what it generated as prompt, so `want` grows with every pause.
    Once it passes the shared ceiling no eviction can admit it, and once it passes the
    cache itself nothing can. The wait loop asked neither question and simply spun until
    its client disconnected: one chat of four open for a full 2400s deadline with
    llama-server idle and every slot released.
    """

    def _adapter(self, *, budget = 16384, drafts = 2, slots = 4):
        from core.inference.llama_preemption import (
            ControllerPreemptionPolicy,
            PreemptSignal,
            PreemptionController,
        )

        controller = PreemptionController("resume-wait")
        controller.configure(
            budget = budget, kv_unified = True, draft_tokens = drafts, slots = slots
        )
        return controller, ControllerPreemptionPolicy, PreemptSignal

    def test_the_wait_is_told_when_nothing_can_ever_fit(self):
        controller, _, _ = self._adapter()
        assert controller.cannot_ever_fit(controller.snapshot().budget)

    def test_a_want_between_the_ceilings_is_not_impossible(self):
        controller, _, _ = self._adapter()
        snapshot = controller.snapshot()
        want = snapshot.budget - snapshot.buffer + 1
        assert controller.outgrew_the_shared_ceiling(want)
        assert not controller.cannot_ever_fit(want), (
            "it fits with the cache to itself, so ending the turn would be premature"
        )

    def test_both_questions_are_asked_before_the_spin(self):
        from pathlib import Path

        from core.inference import llama_preemption

        body = _await_resume_body(Path(llama_preemption.__file__).read_text())
        spin = body.index("while not self._controller.try_grant_resume")
        assert body.index("cannot_ever_fit(") < spin, (
            "a chat larger than the cache would spin until its client gave up"
        )
        assert body.index("outgrew_the_shared_ceiling(") < spin

    def test_giving_up_is_reported_as_finishing_not_as_failure(self):
        """`length` is resumable by the continuation path; a hang is not."""
        from pathlib import Path

        from core.inference import llama_preemption

        body = _await_resume_body(Path(llama_preemption.__file__).read_text())
        assert "too-large" in body
        assert "finishing the turn" in body


class TestTheResumeGrantReadsTheCacheAfresh:
    """A resume is granted against an exact figure, not one up to a second old.

    The ledger adds up prompt ESTIMATES; llama-server's per-slot totals are exact, and on
    code-heavy answers the two drift. A run overran the cache with three slots holding
    4237 + 5400 + 7390 = 17027 tokens against 16384, and nothing recorded whether the
    ledger had drifted low or the sweep had simply not run during prefill. Both are now
    addressed: the drift is logged when it exceeds 256 tokens, and the grant re-reads.
    """

    def test_the_controller_can_be_given_a_probe(self):
        from core.inference.llama_preemption import PreemptionController

        controller = PreemptionController("probe")
        controller.configure(budget = 16384, kv_unified = True)
        calls = []
        controller.set_residency_probe(lambda: calls.append(1))
        controller.refresh_residency()
        assert calls == [1]

    def test_a_failing_probe_never_raises(self):
        from core.inference.llama_preemption import PreemptionController

        controller = PreemptionController("probe-boom")
        controller.configure(budget = 16384, kv_unified = True)

        def boom():
            raise RuntimeError("slots endpoint down")

        controller.set_residency_probe(boom)
        controller.refresh_residency()  # must not raise

    def test_no_probe_is_harmless(self):
        from core.inference.llama_preemption import PreemptionController

        PreemptionController("no-probe").refresh_residency()

    def test_the_resume_wait_refreshes_before_it_asks(self):
        from pathlib import Path

        from core.inference import llama_preemption

        body = _await_resume_body(Path(llama_preemption.__file__).read_text())
        first_ask = body.index("while not self._controller.try_grant_resume")
        assert body.index("refresh_residency()") < first_ask, (
            "the first grant would be decided on a cached figure"
        )

    def test_the_route_registers_the_probe(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        assert "set_residency_probe(" in source
        assert "force = True" in source, "the probe must bypass the one second cache"

    def test_the_drift_between_the_two_figures_is_logged(self):
        from pathlib import Path

        from core.inference import llama_preemption

        source = Path(llama_preemption.__file__).read_text()
        assert "ledger-drift" in source, (
            "a cache overrun could not be attributed to the ledger or to the sweep"
        )


class TestTheSweepIsNotBlindDuringPrefill:
    """A round boundary grows the prompt, so it has to be a sweep point.

    The watermark ran from the token callback alone. `_gguf_recost` told the controller
    the new size at each round through `note_tokens`, but note_tokens only records and
    only observe() plans evictions, so the larger figure sat in the ledger unacted on
    until 32 more tokens had been generated.

    That window is exactly where the prompts are biggest: after a tool result, and after
    a resume replays its whole partial. Three chats prefilling together could pass the
    cache before the sweep next ran, which is how three slots came to hold
    4237 + 5400 + 7390 = 17027 tokens against a 16384 cache with the watermark at 15608.
    """

    def _recost_body(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text()
        return source.split("def _gguf_recost", 1)[1].split("\n            def ", 1)[0]

    def test_the_round_boundary_sweeps_and_does_not_merely_record(self):
        body = self._recost_body()
        assert "note_tokens(" in body, "the new size must still be recorded"
        assert "_gguf_observe_tokens(" in body, (
            "recording without sweeping leaves the eviction until the next 32 tokens"
        )

    def test_it_sweeps_after_recording_not_before(self):
        body = self._recost_body()
        assert body.index("note_tokens(") < body.index("_gguf_observe_tokens("), (
            "sweeping first would plan against the previous round's figure"
        )

    def test_note_tokens_rebaselines_so_zero_growth_is_correct(self):
        """The sweep is passed zero generated, which is only right after a re-baseline."""
        from core.inference.llama_preemption import PreemptionController

        controller = PreemptionController("prefill")
        controller.configure(budget = 16384, kv_unified = True)
        controller.register("a", tokens = 1000)
        controller.observe("a", 500)
        assert controller.participant("a").tokens == 1500

        controller.note_tokens("a", 9000)          # a round restated the conversation
        controller.observe("a", 0)                 # the sweep that follows it
        assert controller.participant("a").tokens == 9000, (
            "growth must be measured from the new round, not added to the old total"
        )

    def test_a_round_that_grows_past_the_watermark_evicts_someone(self):
        from core.inference.llama_preemption import PreemptionController

        controller = PreemptionController("prefill-evict")
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("older", tokens = 4000)
        controller.register("newer", tokens = 4000)
        assert controller.plan_preemptions() == [], "nothing to do yet"

        snapshot = controller.snapshot()
        controller.note_tokens("newer", snapshot.budget - snapshot.buffer)
        victims = [p.gen_id for p in controller.observe("newer", 0)]
        assert victims, "the round grew past the watermark and nobody was asked to stop"
