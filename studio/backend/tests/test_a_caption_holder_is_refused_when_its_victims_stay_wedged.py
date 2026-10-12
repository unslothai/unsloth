# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A caption or OCR call is refused, not sent, when the chats paused for it never let go.

Those calls count their holder on a worker thread and wait there for the victims. The wait's
answer was thrown away, so after the hard bound the vision prompt still went upstream into the
cells the plan had counted as freed. It now raises the admission timeout, as the loop's callers
do, and leaves nothing registered behind it.
"""

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
    fake = SimpleNamespace(base_url = "http://127.0.0.1:65085", _admission_key = "caption-wedged")
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: fake)
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    controller.register("chat", tokens = 15000)
    controller.note_measured("chat")
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def test_a_wedged_victim_refuses_the_call_before_it_goes_upstream(backend):
    _fake, controller = backend
    sent: list = []
    with pytest.raises(LlamaAdmissionTimeout):
        with inference._direct_llama_request(kv_tokens = 4000):
            sent.append(True)
    assert not sent, "the vision prompt went upstream into cells the victim still holds"
    assert set(controller._participants) == {"chat"}, "the refused holder was left registered"
    assert not inference._direct_llama_is_busy()


def test_a_victim_that_pauses_lets_the_call_go(backend):
    _fake, controller = backend

    def pause():
        time.sleep(0.3)
        controller.set_state("chat", ParticipantState.PAUSED)

    threading.Thread(target = pause, daemon = True).start()
    sent: list = []
    with inference._direct_llama_request(kv_tokens = 4000) as holder:
        sent.append(holder)
    assert sent and sent[0] is not None
    assert set(controller._participants) == {"chat"}
