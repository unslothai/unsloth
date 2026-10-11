# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/v1/completions takes no admission lease, but it fills the same unified cache as the chats."""

import inspect
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import ParticipantState, get_preemption_controller
import routes.inference as inference


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_openai_llama_preemption_will_apply", lambda *a, **k: True)
    monkeypatch.setattr(inference, "_openai_llama_admission_budget", lambda *a, **k: 16384)
    fake = SimpleNamespace(base_url = "http://127.0.0.1:9", _admission_key = "cmpl-holder-test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def test_a_chat_beside_a_completion_is_contended_and_the_completion_is_never_chosen(backend):
    fake, controller = backend
    controller.register("chat", tokens = 4000)
    assert not controller.contended(), "alone, the chat reads nothing"
    holder = inference._openai_llama_count_completions_holder(fake, "x" * 16000)
    assert holder is not None
    assert controller.participant(holder).state == ParticipantState.STREAMING_RAW
    assert controller.contended(), "the completion's cells are somebody else's"
    controller.note_resident(16384)
    victims = controller.plan_preemptions(needed = 0)
    assert [v.gen_id for v in victims] == ["chat"], "only the chat can pause"
    inference._openai_llama_release_completions_holder(fake, holder)
    assert controller.participant(holder) is None


def test_nothing_is_counted_with_preemption_off(backend, monkeypatch):
    fake, controller = backend
    monkeypatch.setenv(preemption.PREEMPT_ENV, "0")
    assert inference._openai_llama_count_completions_holder(fake, "hello") is None


def test_both_completions_branches_count_and_release():
    source = inspect.getsource(inference.openai_completions)
    assert source.count("_openai_llama_count_completions_holder(llama_backend, prompt_text)") == 2
    assert (
        source.count("_openai_llama_release_completions_holder(llama_backend, _cmpl_holder)") == 2
    )
