# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF tool round hands the preemptor the tokenizer count its recost just took.

The recost counts the prepared prompt exactly, and the publication then priced it again with the
character estimate. Dense numeric or code text tokenizes far above that, so the round's resident
figure fell short by the gap and overwrote the count arming had used.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    tokens = 9000

    def recost_waiting(self, *args, **kwargs):
        return True


class _Reservation:
    def lease_nowait(self):
        return _Lease()


def _backend():
    return SimpleNamespace(
        base_url = "http://round.test",
        admission_key = "test://round",
        _kv_cache_unified = True,
        _kv_cache_context_total = 16384,
        is_vision = False,
    )


def test_the_recost_hands_back_the_count_of_what_the_round_sends(monkeypatch):
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    monkeypatch.setattr(
        inference,
        "_count_gguf_admission_prompt_detail",
        lambda *a, **k: (5000, True),
    )
    counted: dict = {}
    inference._openai_llama_admission_recost(
        _Reservation(),
        [{"role": "user", "content": "1 2 3 4 5 6 7 8 9"}],
        request = None,
        llama_backend = _backend(),
        payload = SimpleNamespace(),
        injected_tools = tools,
        wire_tools = tools,
        count_prepared_prompt = True,
        counted_out = counted,
    )
    assert counted == {"prompt": 5000}


def test_another_catalogue_is_counted_on_its_own(monkeypatch):
    counts = {True: 5000, False: 4900}
    monkeypatch.setattr(
        inference,
        "_count_gguf_admission_prompt_detail",
        lambda backend, payload, messages, tools = None, **k: (counts[bool(tools)], True),
    )
    counted: dict = {}
    inference._openai_llama_admission_recost(
        _Reservation(),
        [{"role": "user", "content": "hi"}],
        request = None,
        llama_backend = _backend(),
        payload = SimpleNamespace(),
        injected_tools = [{"type": "function", "function": {"name": "a"}}],
        # The final pass renders no catalogue, so the count of the charged one does not stand.
        wire_tools = None,
        count_prepared_prompt = True,
        counted_out = counted,
    )
    assert counted == {"prompt": 4900}


def test_an_estimated_count_is_not_handed_back(monkeypatch):
    monkeypatch.setattr(
        inference,
        "_count_gguf_admission_prompt_detail",
        lambda *a, **k: (5000, False),
    )
    counted: dict = {}
    inference._openai_llama_admission_recost(
        _Reservation(),
        [{"role": "user", "content": "hi"}],
        request = None,
        llama_backend = _backend(),
        payload = SimpleNamespace(),
        injected_tools = [{"type": "function", "function": {"name": "a"}}],
        wire_tools = None,
        count_prepared_prompt = True,
        counted_out = counted,
    )
    assert counted == {}


def test_the_publication_records_the_count_not_an_estimate(monkeypatch):
    noted: list = []

    class _Controller:
        def note_tokens(
            self,
            gen_id,
            tokens,
            prompt_tokens = None,
        ):
            noted.append((gen_id, tokens, prompt_tokens))

    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: _Controller())
    inference._openai_llama_publish_round_charge(
        llama_backend = _backend(),
        gen_id = "loop",
        reservation = _Reservation(),
        payload = SimpleNamespace(),
        conversation = [{"role": "user", "content": "1 2 3 4 5 6 7 8 9"}],
        rendered_tools = None,
        observe_tokens = lambda generated: None,
        counted_prompt_tokens = 5000,
    )
    assert noted == [("loop", 9000, 5000)], noted


def test_the_gguf_loop_threads_the_count_through():
    source = inspect.getsource(inference)
    body = source[source.index("def _gguf_recost(conversation, round_tools = None)") :]
    body = body[: body.index("return _recosted_allowance")]
    assert "counted_out = _round_counted," in body
    assert 'counted_prompt_tokens = _round_counted.get("prompt"),' in body
