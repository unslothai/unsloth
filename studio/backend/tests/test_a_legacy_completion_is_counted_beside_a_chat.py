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
    assert source.count('_openai_llama_count_completions_holder(llama_backend, body.get("prompt", ""))') == 2
    assert (
        source.count("_openai_llama_release_completions_holder(llama_backend, _cmpl_holder)") == 2
    )


def test_an_unpausable_holder_plans_before_its_request_goes_out(backend, monkeypatch):
    # The chat has outgrown its lease: the cache already holds near all of it. Registering the
    # completion has to pick the chat at once, not at the chat's next token report.
    fake, controller = backend
    monkeypatch.setattr(
        inference, "_refresh_residency_before_planning", lambda c: c.note_resident(16000)
    )
    controller.register("chat", tokens = 4000)
    holder = inference._openai_llama_count_completions_holder(fake, "x" * 16000)
    assert controller.participant("chat").state == ParticipantState.PREEMPTING
    assert controller.participant(holder).state == ParticipantState.STREAMING_RAW
    inference._openai_llama_release_completions_holder(fake, holder)


def test_both_unpausable_registrations_plan():
    for helper in (
        inference._openai_llama_count_raw_holder,
        inference._openai_llama_count_direct_holder,
    ):
        assert "_plan_for_unpausable_holder(controller, gen_id)" in inspect.getsource(helper)
    assert "_openai_llama_count_direct_holder(" in inspect.getsource(
        inference._openai_llama_count_completions_holder
    )


def test_the_prompt_estimate_does_not_undercount_cjk_or_token_arrays():
    assert inference._prompt_kv_tokens([101, 102, 103]) == 3
    assert inference._prompt_kv_tokens([[1, 2], [3]]) == 3
    assert inference._prompt_kv_tokens("你好世界") == 4, "one token a character, not a quarter"
    assert inference._prompt_kv_tokens("hello world!") == 3
    assert inference._prompt_kv_tokens(["你好", "abcd"]) == 3


def test_a_caption_or_ocr_call_is_counted_while_it_runs(backend, monkeypatch):
    fake, controller = backend
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: fake)
    with inference._direct_llama_request(kv_tokens = 900):
        held = [p for p in controller._participants if p.startswith("direct-holder-")]
        assert len(held) == 1
        assert controller.participant(held[0]).state == ParticipantState.STREAMING_RAW
    assert not [p for p in controller._participants if p.startswith("direct-holder-")]


def test_the_captioner_states_its_footprint():
    from core.rag import captioner

    source = inspect.getsource(captioner._vision_complete)
    assert "_direct_llama_slot(prompt = prompt, max_tokens = max_tokens)" in source
    assert "_direct_llama_request(kv_tokens = kv_tokens)" in inspect.getsource(captioner._direct_llama_slot)
