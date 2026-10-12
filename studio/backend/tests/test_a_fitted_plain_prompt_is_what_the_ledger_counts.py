# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A plain chat whose oldest messages were fitted away is counted at its fitted size.

The refit repriced the output allowance and left the preemptor holding the pre-fit prompt, so a
chat sending a few tokens was counted at thousands: paused for room it never takes, then refused
its resume as too large.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

from core.inference.llama_preemption import PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    tokens = 12000

    def yield_parked_commitment(self, *, charged_at = None):
        return 0


class _Reservation:
    def lease_nowait(self):
        return _Lease()


def _backend():
    return SimpleNamespace(
        base_url = "http://fitted.test",
        admission_key = "test://fitted",
        _kv_cache_unified = True,
        _kv_cache_context_total = 16384,
        is_vision = False,
    )


def _lift(namespace: dict):
    source = textwrap.dedent(inspect.getsource(inference.produce_openai_chat_completions))
    wanted = {"_gguf_prompt_estimate", "_gguf_refit_allowance"}
    nodes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} >= {"_gguf_refit_allowance"}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "<route>", "exec"), namespace)
    return namespace["_gguf_refit_allowance"]


def _refit(monkeypatch, controller, original, fitted, counted = 5):
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(
        inference, "_count_gguf_admission_prompt_detail", lambda *a, **k: (counted, True)
    )
    monkeypatch.setattr(
        inference, "_openai_llama_admission_enforced_max_tokens", lambda *a, **k: 2048
    )
    namespace = dict(vars(inference))
    namespace.update(
        payload = SimpleNamespace(messages = original),
        request = None,
        llama_backend = _backend(),
        gguf_messages = original,
        cancel_event = None,
        completion_id = "plain",
        reservation = _Reservation(),
        _plain_observe_tokens = lambda generated: None,
    )
    return _lift(namespace)(fitted)


def _controller() -> PreemptionController:
    controller = PreemptionController("http://127.0.0.1:65081")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    controller.register("plain", lease = _Lease(), tokens = 10015, prompt_tokens = 10015)
    controller.participant("plain").announce_prefill(10015)
    return controller


_LONG = [{"role": "user", "content": "1 2 3 4 5 " * 2000}, {"role": "user", "content": "hi"}]


def test_a_fit_that_dropped_messages_is_counted_at_its_fitted_size(monkeypatch):
    controller = _controller()
    assert _refit(monkeypatch, controller, _LONG, _LONG[-1:]) == 2048
    participant = controller.participant("plain")
    assert participant.tokens == 5, participant.tokens
    assert participant.pending_prefill <= 5, participant.pending_prefill


def test_a_fit_that_kept_everything_counts_nothing_again(monkeypatch):
    controller = _controller()

    def no_count(*a, **k):
        raise AssertionError("a fit that kept everything was counted again")

    monkeypatch.setattr(inference, "_count_gguf_admission_prompt_detail", no_count)
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(
        inference, "_openai_llama_admission_enforced_max_tokens", lambda *a, **k: 2048
    )
    namespace = dict(vars(inference))
    namespace.update(
        payload = SimpleNamespace(messages = _LONG),
        request = None,
        llama_backend = _backend(),
        gguf_messages = _LONG,
        cancel_event = None,
        completion_id = "plain",
        reservation = _Reservation(),
        _plain_observe_tokens = lambda generated: None,
    )
    assert _lift(namespace)(list(_LONG)) == 2048
    assert controller.participant("plain").tokens == 10015
