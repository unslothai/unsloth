# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unpausable request is not sent into cells its victims still hold.

The first wait is short, since a pause lands within a token or two of the signal. When it ran
out the request went upstream anyway, into the cells the plan had counted as freed, and its
prefill could overflow the shared cache for every chat in it. It now waits on while there is no
room, and a victim wedged past the hard bound refuses the request as an admission timeout.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_admission import LlamaAdmissionTimeout
from core.inference.llama_preemption import ParticipantState, get_preemption_controller
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_openai_llama_preemption_will_apply", lambda *a, **k: True)
    monkeypatch.setattr(inference, "_openai_llama_admission_budget", lambda *a, **k: 16384)
    monkeypatch.setattr(inference, "_refresh_residency_before_planning", lambda controller: None)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 0.2)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 1.0, raising = False)
    fake = SimpleNamespace(base_url = "http://127.0.0.1:65083", _admission_key = "direct-never-test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    controller.register("chat", tokens = 15000)
    controller.note_measured("chat")
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def _later(after_s, action):
    def run():
        time.sleep(after_s)
        action()

    threading.Thread(target = run, daemon = True).start()


def _route(fake, controller, then = None):
    async def route():
        holder = inference._openai_llama_count_completions_holder(fake, "x" * 16000)
        assert controller.participant("chat").state == ParticipantState.PREEMPTING
        if then is not None:
            then()
        started = time.monotonic()
        try:
            await inference._openai_llama_let_victims_go(fake, holder)
            return time.monotonic() - started, None
        except LlamaAdmissionTimeout as exc:
            return time.monotonic() - started, exc
        finally:
            inference._openai_llama_release_completions_holder(fake, holder)

    return asyncio.run(route())


def test_a_victim_slower_than_the_first_wait_is_still_waited_for(backend):
    fake, controller = backend
    waited, refused = _route(
        fake,
        controller,
        then = lambda: _later(0.5, lambda: controller.set_state("chat", ParticipantState.PAUSED)),
    )
    assert refused is None
    assert waited >= 0.45, f"sent after {waited:.2f}s, while the victim still held its cells"
    assert controller.participant("chat").state == ParticipantState.PAUSED


def test_a_victim_that_never_lets_go_refuses_the_request(backend):
    fake, controller = backend
    waited, refused = _route(fake, controller)
    assert isinstance(refused, LlamaAdmissionTimeout), "sent upstream into cells still held"
    assert 0.9 <= waited < 3.0


def test_room_freed_another_way_lets_it_go(backend):
    # The victim finished instead of pausing: its cells left with it.
    fake, controller = backend
    waited, refused = _route(
        fake, controller, then = lambda: _later(0.4, lambda: controller.unregister("chat"))
    )
    assert refused is None
    assert 0.35 <= waited < 0.9
