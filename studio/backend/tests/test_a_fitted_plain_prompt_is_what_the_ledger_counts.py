# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A plain chat whose oldest messages were fitted away is counted at its fitted size."""

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


def _refit(
    monkeypatch,
    controller,
    original,
    fitted,
    counted = 5,
):
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
        _admission_output_allowance = 3000,
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
        _admission_output_allowance = 3000,
    )
    assert _lift(namespace)(list(_LONG)) == 3000, "the admission count still stands"
    assert controller.participant("plain").tokens == 10015


def _lift_counted(namespace: dict):
    source = textwrap.dedent(inspect.getsource(inference.produce_openai_chat_completions))
    nodes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_gguf_counted_allowance"
    ]
    assert nodes, "_gguf_counted_allowance is gone from the plain path"
    exec(compile(ast.Module(body = nodes, type_ignores = []), "<route>", "exec"), namespace)
    return namespace["_gguf_counted_allowance"]


class TestAPlainPromptIsBoundedByItsCount:
    """Reported on Qwen3 at 4096 x 4 slots: " hello" x 2800 is 2812 tokens, estimated at 4207."""

    def _counted(self, counted):
        backend = SimpleNamespace(
            context_length = 4096,
            _kv_cache_context_total = 4096,
            effective_parallel_slots = 4,
            _kv_cache_unified = True,
        )
        payload = SimpleNamespace(
            messages = [{"role": "user", "content": " hello" * 2800}], max_tokens = None
        )
        estimated = inference._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = payload.messages
        )
        namespace = dict(vars(inference))
        namespace.update(
            payload = payload,
            request = None,
            llama_backend = backend,
            gguf_messages = payload.messages,
            _admission_output_allowance = estimated,
        )
        return estimated, _lift_counted(namespace)(counted)

    def test_the_count_replaces_an_estimate_that_left_one_token(self):
        estimated, counted = self._counted(2812)
        assert estimated == 1, f"the estimate this is about overcounts: {estimated}"
        assert counted > 1000, f"a 2812-token prompt in a 4096 window was sent {counted}"
        assert counted <= 4096 - 2812

    def test_a_failed_count_keeps_the_estimate(self):
        estimated, counted = self._counted(4096)
        assert counted == estimated

    def test_both_plain_branches_rebound_after_admission(self):
        source = inspect.getsource(inference.produce_openai_chat_completions)
        call = "_admission_output_allowance = _gguf_counted_allowance("
        assert source.count(call) == 2
        stream = source.index("async def gguf_stream_chunks():")
        assert source.rindex("_reserve_counted_gguf_chat(", 0, stream) < source.rindex(
            call, 0, stream
        )
        nonstream = source.index(call, stream)
        assert source.rindex("_reserve_counted_gguf_chat(", 0, nonstream) > stream
