# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the ledger is told after an erase, and how long a plain chat may keep resuming.

An early idle-slot erase subtracts what it freed from the reading it planned from, and an
approval erase takes a parked chat's cells: both must reach the ledger, or the next plan
still sees cells that are gone, or the next round reserves only its growth for a whole
prompt it prefills again. A plain chat resumes by nesting its generator, so productive
pauses need a bound of their own.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


BASE = "http://127.0.0.1:65071"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 8192


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _controller() -> PreemptionController:
    made = PreemptionController(BASE)
    made.configure(budget = 8192, kv_unified = True, slots = 4)
    return made


class TestThePostEraseReadingLands:
    def test_the_plan_after_an_early_erase_sees_the_freed_cells(self, monkeypatch):
        controller = _controller()
        controller.register("live", lease = _Lease(), tokens = 2000)
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
        monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: [])
        monkeypatch.setattr(
            inference,
            "read_slot_occupancy",
            lambda scrape: {"idle": [(1, 3000)], "resident": 9000, "idle_tokens": 3000},
        )
        monkeypatch.setattr(
            inference, "reclaim_idle_slots", lambda occupancy, erase, *, needed = 0: 3000
        )
        refresh, _observe, _note = inference._openai_llama_residency_observer(
            llama_backend = _Backend(), completion_id = "live"
        )
        refresh(controller, force = True)
        assert controller._resident == 6000, (
            "the post-erase figure was dropped as stale, so the next plan still counts "
            f"the erased cells: resident {controller._resident}"
        )
        assert controller._reclaimable == 0

    def test_a_newer_sample_outranks_the_derived_one(self):
        controller = _controller()
        recorded = controller.note_resident(8000, 3000, started_at_seq = controller.residency_epoch())
        assert recorded is not None
        controller.note_resident(7000, 0, started_at_seq = controller.residency_epoch())
        assert controller.note_erased_since(recorded, 5000, 0) is False
        assert controller._resident == 7000

    def test_a_measure_in_between_does_not_void_it(self):
        controller = _controller()
        controller.register("raw", lease = _Lease(), tokens = 2000)
        recorded = controller.note_resident(8000, 3000, started_at_seq = controller.residency_epoch())
        controller.note_measured("raw")
        assert controller.note_erased_since(recorded, 5000, 0) is True
        assert controller._resident == 5000
        assert controller.participant("raw").measured is False, (
            "measured after the probe left, so this reading cannot hold its cells"
        )

    def test_a_dropped_or_failed_reading_records_nothing(self):
        controller = _controller()
        epoch = controller.residency_epoch()
        controller.note_resident(8000, 0, started_at_seq = epoch)
        assert controller.note_resident(4000, 0, started_at_seq = epoch) is None
        assert controller.note_resident(None) is None


class TestAnApprovalEraseIsRecorded:
    def test_the_next_round_reserves_the_whole_prompt(self, monkeypatch):
        controller = _controller()
        controller.register("chat", lease = _Lease(), tokens = 4000)
        controller.note_tokens("chat", 4000, 4000)
        controller.note_measured("chat")
        controller.note_state("chat", ParticipantState.PARKED_ON_TOOL)
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)

        parked = inference._openai_llama_parked_self(_Backend(), "chat")
        assert parked == {"chat": controller.participant("chat").park_seq}
        inference._openai_llama_note_self_erased(_Backend(), parked)
        assert controller.participant("chat").holds_kv is False

        controller.note_tokens("chat", 4200, 4200)
        assert controller.snapshot().prefilling == 4200, (
            "the erase took every cell, so the round prefills the whole prompt, not its growth"
        )

    def test_nothing_is_told_when_it_was_not_parked(self, monkeypatch):
        controller = _controller()
        controller.register("chat", lease = _Lease(), tokens = 4000)
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
        assert inference._openai_llama_parked_self(_Backend(), "chat") is None
        inference._openai_llama_note_self_erased(_Backend(), None)
        assert controller.participant("chat").cells_reclaimed is False

    def test_the_approval_reclaim_reads_before_and_tells_after(self):
        source = inspect.getsource(inference)
        body = source[source.index("async def _reclaim_approval_cache(lease):") :]
        body = body[: body.index("def _watch_for_reclaim(lease)")]
        before = body.index("_openai_llama_parked_self(llama_backend, completion_id)")
        erase = body.index("_erase_and_record, target")
        told = body.index("_openai_llama_note_self_erased(llama_backend, parked_now)")
        assert before < erase < told
        assert body.index("lease.release_parked_cache()") < told


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"


class _Policy:
    def __init__(self):
        self.resumes = 0

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint):
        pass

    def await_resume(self, timeout = None) -> bool:
        return True

    def on_resumed(self) -> None:
        self.resumes += 1


class TestAPlainChatCannotNestPastTheLimit:
    def test_productive_pauses_end_the_turn_continuable(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _Policy()
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48854
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False
        opened: list[dict] = []

        @contextlib.contextmanager
        def fake_stream_with_retry(
            _client, _url, payload, _cancel_event, headers = None, first_token_deadline = None,
            preempt_event = None,
        ):
            opened.append(copy.deepcopy(payload))
            yield type("FakeResponse", (), {"status_code": 200})()

        def fake_iter_text_cancellable(
            response, _cancel_event, first_token_deadline = None, preempt_event = None
        ):
            # Every attempt decodes a token and is paused: progress each time, so the
            # zero-output guard never ends it.
            yield _delta("ab")
            signal.request("kv_pressure")
            raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)

        events = list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "write me a poem"}],
                cancel_event = threading.Event(),
                preempt_event = signal,
                preempt_policy = policy,
            )
        )
        assert len(opened) == preemption.MAX_NESTED_RESUMES, len(opened)
        metas = [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"]
        assert metas and metas[-1]["finish_reason"] == "length"
        assert policy.resumes == preemption.MAX_NESTED_RESUMES - 1

    def test_the_limit_is_far_below_the_interpreter_stack(self):
        import sys

        assert preemption.MAX_NESTED_RESUMES * 2 < sys.getrecursionlimit()
        assert not preemption.resumes_exhausted(0)
        assert preemption.resumes_exhausted(preemption.MAX_NESTED_RESUMES - 1)
