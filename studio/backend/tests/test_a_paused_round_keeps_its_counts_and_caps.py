# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A resumed attempt starts unmeasured, a copied catalogue keeps its count, and the
`/v1/messages` tool loop publishes the count its recost took.

A resume grant kept the paused attempt's measurement stamp, so a probe landing before the
replay was sent promoted the holder and dropped the replay from the ledger. The tool loop
hands each round a deep copy of its catalogue, so an identity check discarded every exact
count. The `/v1/messages` loop priced its rounds with the character estimate alone, which
tool results run far above.
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace

from core.inference.llama_preemption import PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    tokens = 9000

    def yield_parked_commitment(self, *, charged_at = None):
        return 0

    def recost_waiting(self, *args, **kwargs):
        return True


class _Reservation:
    def lease_nowait(self):
        return _Lease()


def _backend():
    return SimpleNamespace(
        base_url = "http://paused.test",
        admission_key = "test://paused",
        _kv_cache_unified = True,
        _kv_cache_context_total = 16384,
        is_vision = False,
    )


class TestAResumeStartsUnmeasured:
    def test_a_probe_before_the_replay_does_not_promote_it(self):
        controller = PreemptionController("http://127.0.0.1:65077")
        controller.configure(budget = 32768, kv_unified = True, slots = 4)
        controller.register("chat", lease = _Lease(), tokens = 7000, prompt_tokens = 7000)
        # A token is stamped, and the probe that would have confirmed it never lands.
        controller.observe("chat", 1, plan = False)
        assert controller.participant("chat").measured_at_seq is not None
        assert controller.try_grant_resume("chat", 12001) is True
        controller.note_resident(7000, started_at_seq = controller.residency_epoch())
        assert controller.participant("chat").measured is False
        assert controller.snapshot().committed >= 12001, controller.snapshot().committed


class TestACopiedCatalogueKeepsItsCount:
    def test_an_equal_copy_hands_the_count_back(self, monkeypatch):
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        monkeypatch.setattr(
            inference, "_count_gguf_admission_prompt_detail", lambda *a, **k: (5000, True)
        )
        counted: dict = {}
        inference._openai_llama_admission_recost(
            _Reservation(),
            [{"role": "user", "content": "1 2 3 4 5 6 7 8 9"}],
            request = None,
            llama_backend = _backend(),
            payload = SimpleNamespace(),
            injected_tools = tools,
            wire_tools = copy.deepcopy(tools),
            count_prepared_prompt = True,
            counted_out = counted,
        )
        assert counted == {"prompt": 5000}


def _anthropic_recost(namespace: dict):
    source = textwrap.dedent(inspect.getsource(inference.anthropic_messages))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_anthropic_recost":
            module = ast.Module(body = [node], type_ignores = [])
            exec(compile(module, "<anthropic_messages>", "exec"), namespace)
            return namespace["_anthropic_recost"]
    raise AssertionError("_anthropic_recost is gone")


class TestTheMessagesLoopPublishesItsCount:
    def test_the_count_reaches_the_ledger(self, monkeypatch):
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        noted: list = []

        class _Controller:
            def note_tokens(
                self,
                gen_id,
                tokens,
                prompt_tokens = None,
            ):
                noted.append((gen_id, tokens, prompt_tokens))

        monkeypatch.setattr(
            inference, "_count_gguf_admission_prompt_detail", lambda *a, **k: (7249, True)
        )
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: _Controller())
        namespace = dict(vars(inference))
        namespace.update(
            _anthropic_admission_hold = {"reservation": _Reservation()},
            request = None,
            llama_backend = _backend(),
            payload = SimpleNamespace(max_tokens = None),
            openai_tools = tools,
            cancel_event = None,
            message_id = "msg",
            _anthropic_observe_tokens = lambda generated: None,
        )
        recost = _anthropic_recost(namespace)
        recost([{"role": "tool", "content": "0.123 4.56 " * 300}], copy.deepcopy(tools))
        assert noted and noted[-1][2] == 7249, noted
