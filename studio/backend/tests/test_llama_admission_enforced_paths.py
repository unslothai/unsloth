# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bound has to hold on every path that builds a wire ``max_tokens``.

``test_llama_admission_enforced_max_tokens.py`` proves the arithmetic: a request charged
its share is permitted its share. That is only worth anything if the figure actually
reaches every request the run sends, and the first revision reached two of them. A tool
loop sends one payload per round plus a synthesized final answer, either of them can be
rebuilt by a respawn refit, a plain chat can be retried after a respawn, and /v1/messages
draws on the same slots through a different pair of call sites. Each of those rebuilt the
cap from the whole context window while the ledger still held a share, which is the exact
overcommit the change exists to close.

Driven through the real generators with fake llama-server streams, so what is asserted is
the payload llama-server would have received.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.inference as inf_mod
from core.inference.api_monitor import ApiMonitor
from core.inference.llama_admission import (
    ADMISSION_CONTROL_ENV,
    ADMISSION_KV_BUDGET_ENV,
    reset_llama_admission_queues,
)
from core.inference.llama_cpp import LlamaCppBackend
from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
from routes.inference import (
    _build_openai_passthrough_body,
    _openai_llama_admission_enforced_max_tokens,
    _openai_llama_admission_prompt_tokens,
    _openai_llama_admission_recost,
    _openai_llama_admission_tokens,
    anthropic_messages,
)

_CTX = 4096
# Four slots on one unified cache, so a share is a quarter of the window.
_SHARE = _CTX // 4


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _tool_call(name: str, arguments: dict, call_id: str) -> list[str]:
    return [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
            }
        ),
        _done(),
    ]


def _make_backend(monkeypatch, streams: list[object], payloads: list[dict]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48851
    backend._api_key = None
    backend._effective_context_length = _CTX
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        payloads.append(copy.deepcopy(payload))
        stream = streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream
        yield type("FakeResponse", (), {"status_code": 200, "chunks": stream})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return backend


def _caps(payloads: list[dict]) -> list[int]:
    return [payload["max_tokens"] for payload in payloads]


class TestTheGeneratorsSendIt:
    """Every request a run makes, not only the first one of each kind."""

    def test_the_plain_stream_sends_the_bound(self, monkeypatch):
        payloads: list[dict] = []
        backend = _make_backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)

        list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                admission_output_allowance = _SHARE,
            )
        )

        assert _caps(payloads) == [_SHARE]

    def test_a_respawn_retry_keeps_the_bound(self, monkeypatch):
        """The lease outlives the respawn, so the retry is inside the same reservation.

        The retry re-enters the generator, and the replacement window is what it rebuilds
        an uncapped `max_tokens` from. Two post-respawn retries generating into one cache
        is the collision this change exists to prevent, arrived at from the other side.
        """
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [httpx.ConnectError("dead"), [_sse({"content": "hi"}), _done()]],
            payloads,
        )
        monkeypatch.setattr(backend, "_respawn_if_dead", lambda *_a, **_k: True)

        list(
            backend.generate_chat_completion(
                messages = [{"role": "user", "content": "hello"}],
                admission_output_allowance = _SHARE,
            )
        )

        assert len(payloads) == 2, "expected the first attempt and the post-respawn retry"
        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_the_tool_round_and_the_final_pass_both_send_it(self, monkeypatch):
        """The final pass carries the whole run's history and skips the top of the loop."""
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )

        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "Which kernel?"}],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                max_tool_iterations = 1,
                permission_mode = "off",
                admission_output_allowance = _SHARE,
            )
        )

        assert len(payloads) == 2, "expected one tool round and one synthesized final pass"
        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_a_re_cost_moves_the_bound_with_the_conversation(self, monkeypatch):
        """A cap frozen at the opening prompt drifts exactly as far as the loop grows.

        The ledger re-prices each round from the conversation as it now stands; a wire cap
        still measured against the opening prompt permits `grown_prompt + (share -
        opening_prompt)` while the charge is only `grown_prompt + the flat allowance`.
        The callback hands the fresh figure back, so the two move together.
        """
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )
        recosted = iter([_SHARE - 100, _SHARE - 400])

        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "Which kernel?"}],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                max_tool_iterations = 1,
                permission_mode = "off",
                admission_output_allowance = _SHARE,
                on_conversation_grew = lambda _conversation: next(recosted, None),
            )
        )

        assert _caps(payloads) == [_SHARE - 100, _SHARE - 400]

    def test_a_re_cost_that_says_nothing_leaves_the_bound_alone(self, monkeypatch):
        """Accounting that declined to re-price must not read as "no bound"."""
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )

        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "Which kernel?"}],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                max_tool_iterations = 1,
                permission_mode = "off",
                admission_output_allowance = _SHARE,
                on_conversation_grew = lambda _conversation: None,
            )
        )

        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_a_respawn_refit_does_not_restore_the_window(self, monkeypatch):
        """A replacement server reporting a bigger window is not a bigger reservation.

        The refit rebuilds an uncapped `max_tokens` from the new context length, which is
        the whole cache; the lease it is retrying under is still one share.
        """
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                httpx.ConnectError("dead"),
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10"}), _done()],
            ],
            payloads,
        )

        def _respawn_bigger(*_args, **_kwargs):
            backend._effective_context_length = _CTX * 2
            return True

        monkeypatch.setattr(backend, "_respawn_if_dead", _respawn_bigger)
        # The refit prices its fit against llama-server; the window is what this test is
        # about, not the count.
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )

        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "Which kernel?"}],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                max_tool_iterations = 1,
                permission_mode = "off",
                context_overflow = "truncate_oldest",
                admission_output_allowance = _SHARE,
            )
        )

        assert payloads, "no request was sent"
        assert all(cap <= _SHARE for cap in _caps(payloads)), _caps(payloads)


def _backend_stub(*, window, total, slots):
    return SimpleNamespace(
        context_length = window,
        _kv_cache_context_total = total,
        effective_parallel_slots = slots,
    )


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


def _chat(text = "hi", **fields):
    return _Payload(messages = [{"role": "user", "content": text}], **fields)


_CATALOGUE = [
    {
        "type": "function",
        "function": {
            "name": "search_conversation",
            "description": "Search the conversation. " * 60,
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class TestEveryCallSiteCarriesIt:
    """A generation call site added without the bound is the whole defect, again.

    Read off the route source rather than exercised: `produce_openai_chat_completions` and
    `anthropic_messages` each reach llama-server through several branches, and what has to
    hold is that NONE of them is left out -- which is a property of the set of call sites,
    not of any one run. The behaviour of each branch is asserted above and in
    TestTheAnthropicSurface.
    """

    def _routes_tree(self):
        import ast
        return ast.parse(Path(inf_mod.__file__).read_text())

    def _generator_calls(self, tree):
        """Calls on `llama_backend`, which is the only receiver that holds a KV lease.

        The safetensors twin (`backend.generate_chat_completion_with_tools`) decodes in
        this process against no llama-server cache, so it takes no reservation and there
        is nothing to enforce on it.
        """
        import ast
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None)
            receiver = getattr(getattr(node.func, "value", None), "id", None)
            if receiver != "llama_backend":
                continue
            if name in ("generate_chat_completion", "generate_chat_completion_with_tools"):
                yield name, node

    def test_no_generation_call_site_is_left_unbounded(self):
        import ast

        tree = self._routes_tree()
        unbounded = [
            name
            for name, call in self._generator_calls(tree)
            if not any(keyword.arg == "admission_output_allowance" for keyword in call.keywords)
        ]
        assert not unbounded, f"these call sites send the whole window: {unbounded}"

    def test_a_tool_loop_bound_is_priced_with_the_catalogue_it_sends(self):
        """Studio resolves its own catalogue server-side, and the reservation charges it.

        `payload.tools` does not carry it, so an allowance priced from the client's
        messages alone permits `share + catalogue` per slot on a cache sized for `share`.
        The pairing, not the arithmetic: the helper has always accepted `injected_tools`,
        and the defect was a tool-loop call site that did not pass any.
        """
        import ast

        tree = self._routes_tree()
        priced_with_catalogue = {
            target.id: any(keyword.arg == "injected_tools" for keyword in node.value.keywords)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None)
            == "_openai_llama_admission_enforced_max_tokens"
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert priced_with_catalogue, "no allowance is computed in the route at all"

        blind = []
        for name, call in self._generator_calls(tree):
            if name != "generate_chat_completion_with_tools":
                continue
            for keyword in call.keywords:
                if keyword.arg != "admission_output_allowance":
                    continue
                bound_to = getattr(keyword.value, "id", None)
                if not priced_with_catalogue.get(bound_to, False):
                    blind.append(bound_to)
        assert not blind, f"tool-loop bounds priced without the catalogue they send: {blind}"


class TestWhatThePromptIsMeasuredAgainst:
    def test_the_injected_catalogue_is_inside_the_bound(self):
        """Studio resolves its own tool catalogue server-side, and the reservation charges
        it. `payload.tools` does not carry it, so a cap measured against the client's
        messages alone permits `share + catalogue` on a cache sized for `share`."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        prompt = _openai_llama_admission_prompt_tokens(payload, injected_tools = _CATALOGUE)
        blind = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        aware = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, injected_tools = _CATALOGUE
        )
        charged = _openai_llama_admission_tokens(
            payload,
            budget = 16384,
            capacity = 4,
            tool_loop = True,
            injected_tools = _CATALOGUE,
            context_window = 16384,
        )
        assert prompt + blind > charged, "the catalogue has to be big enough to matter"
        assert prompt + aware <= charged
        assert (prompt + aware) * 4 <= 16384

    def test_a_grown_conversation_earns_a_smaller_cap(self):
        """The re-cost hands back a cap for the prompt it just charged, so the two cannot
        drift once tool results carry the conversation past its share."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)

        class _Lease:
            def recost_waiting(self, *_args, **_kwargs):
                return None

        reservation = SimpleNamespace(lease_nowait = lambda: _Lease())

        opening = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        grown = [{"role": "user", "content": "word " * 900}]
        recosted = _openai_llama_admission_recost(
            reservation,
            grown,
            request = None,
            llama_backend = backend,
            payload = payload,
            output_tokens = 16384,
        )
        assert recosted is not None and recosted < opening, (opening, recosted)
        prompt = _openai_llama_admission_prompt_tokens(_Payload(messages = grown))
        assert prompt + recosted <= 16384 // 4


class TestTheOperatorSwitches:
    """Both switches turn the reservation itself off, so there is nothing to enforce."""

    @pytest.mark.parametrize("variable", [ADMISSION_CONTROL_ENV, ADMISSION_KV_BUDGET_ENV])
    def test_admission_off_leaves_generation_alone(self, monkeypatch, variable):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        assert (
            _openai_llama_admission_enforced_max_tokens(
                payload, request = None, llama_backend = backend
            )
            is not None
        )
        monkeypatch.setenv(variable, "off")
        assert (
            _openai_llama_admission_enforced_max_tokens(
                payload, request = None, llama_backend = backend
            )
            is None
        )


class TestThePassthroughSurface:
    def test_the_body_reads_capacity_from_the_request(self):
        """`effective_parallel_slots` is unset until the backend commits its runtime slots.
        Admission falls back to the launch intent on the request for exactly that window;
        without the request the body reads capacity 1, declines to clamp, and is permitted
        the whole window while the reservation charges a multi-slot share."""
        backend = SimpleNamespace(
            context_length = 16384,
            _kv_cache_context_total = 16384,
            effective_parallel_slots = None,
            markup_profile = None,
            _request_reasoning_kwargs = lambda *_a, **_k: None,
        )
        payload = ChatCompletionRequest.model_validate(
            {
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16384,
                "stream": False,
            }
        )
        request = SimpleNamespace(
            app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 4))
        )

        blind = _build_openai_passthrough_body(payload, backend_ctx = 16384, llama_backend = backend)
        aware = _build_openai_passthrough_body(
            payload, backend_ctx = 16384, llama_backend = backend, request = request
        )
        assert blind["max_tokens"] == 16384
        assert aware["max_tokens"] <= 16384 // 4


_ANTHROPIC_KEY = "http://llama.enforced.test:9999"


class _AnthropicRequest:
    def __init__(self):
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(path = "/v1/messages")
        self.method = "POST"
        self.app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 4))

    async def is_disconnected(self):
        return False


class TestTheAnthropicSurface:
    """/v1/messages draws on the same slots through its own call sites.

    `max_tokens` is optional here and a value at the window reads as unstated, so an
    Anthropic client was charged a share and permitted the whole cache.
    """

    @pytest.fixture(autouse = True)
    def _isolate(self, monkeypatch):
        reset_llama_admission_queues()
        monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
        monkeypatch.setattr(inf_mod, "_CANCEL_REGISTRY", {})
        yield
        reset_llama_admission_queues()

    def _install(self, monkeypatch, seen: dict):
        def _gen_plain(**kwargs):
            seen["plain"] = kwargs
            yield "ok"

        def _gen_tools(**kwargs):
            seen["tools"] = kwargs
            yield {"type": "content", "text": "ok"}

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            supports_tool_passthrough = False,
            model_identifier = "test-model",
            context_length = 16384,
            _kv_cache_context_total = 16384,
            effective_parallel_slots = 4,
            count_chat_tokens = lambda *a, **k: 2,
            generate_chat_completion = _gen_plain,
            generate_chat_completion_with_tools = _gen_tools,
            base_url = _ANTHROPIC_KEY,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        return backend

    def test_the_plain_generator_is_bounded(self, monkeypatch):
        seen: dict = {}
        self._install(monkeypatch, seen)
        payload = AnthropicMessagesRequest(
            max_tokens = 16384,
            messages = [{"role": "user", "content": "hi"}],
        )

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        allowance = seen["plain"]["admission_output_allowance"]
        assert allowance is not None and allowance <= 16384 // 4

    def test_the_tool_generator_is_bounded(self, monkeypatch):
        seen: dict = {}
        self._install(monkeypatch, seen)
        # Unsloth's own server-side loop, which is what takes the multi-round lease;
        # a client catalogue would take the passthrough instead.
        payload = AnthropicMessagesRequest.model_validate(
            {
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": "hi"}],
                "enable_tools": True,
                # No confirmation channel on this surface, so the route requires an
                # explicit permission mode before it will run server tools.
                "permission_mode": "off",
            }
        )

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        allowance = seen["tools"]["admission_output_allowance"]
        assert allowance is not None and allowance <= 16384 // 4
