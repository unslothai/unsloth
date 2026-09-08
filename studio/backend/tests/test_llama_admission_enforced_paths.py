# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bound has to hold on every path that builds a wire ``max_tokens``.

The arithmetic is proved next door; this proves the figure reaches every request, since a
tool round, the final answer, both respawn refits, the post-respawn retry and /v1/messages
each rebuilt the cap from the whole window. Driven through the real generators, so what is
asserted is the payload llama-server would have received.
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
from fastapi.responses import JSONResponse

from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
from routes.inference import (
    _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS,
    _build_openai_passthrough_body,
    _openai_llama_admission_retry_max_tokens,
    _openai_llama_admission_wire_prompt_tokens,
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


def _finish(reason: str) -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


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


_TOOL = {
    "type": "function",
    "function": {"name": "web_search", "parameters": {"type": "object", "properties": {}}},
}


def _run_tool_loop(
    monkeypatch,
    payloads,
    *,
    streams = None,
    backend = None,
    **kwargs,
):
    if backend is None:
        backend = _make_backend(
            monkeypatch,
            streams
            if streams is not None
            else [
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
            tools = [_TOOL],
            max_tool_iterations = 1,
            permission_mode = "off",
            **kwargs,
        )
    )
    return backend


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
        """The retry re-enters the generator, which rebuilds an uncapped cap."""
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
        _run_tool_loop(monkeypatch, payloads, admission_output_allowance = _SHARE)

        assert len(payloads) == 2, "expected one tool round and one synthesized final pass"
        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_a_re_cost_moves_the_bound_with_the_conversation(self, monkeypatch):
        """A cap frozen at the opening prompt drifts as far as the loop grows."""
        payloads: list[dict] = []
        recosted = iter([_SHARE - 100, _SHARE - 400])
        _run_tool_loop(
            monkeypatch,
            payloads,
            admission_output_allowance = _SHARE,
            on_conversation_grew = lambda _conversation, _tools: next(recosted, None),
        )

        assert _caps(payloads) == [_SHARE - 100, _SHARE - 400]

    def test_a_re_cost_that_says_nothing_leaves_the_bound_alone(self, monkeypatch):
        """Accounting that declined to re-price must not read as "no bound"."""
        payloads: list[dict] = []
        _run_tool_loop(
            monkeypatch,
            payloads,
            admission_output_allowance = _SHARE,
            on_conversation_grew = lambda _conversation, _tools: None,
        )

        assert _caps(payloads) == [_SHARE, _SHARE]

    def test_the_hook_is_told_which_catalogue_each_request_sends(self, monkeypatch):
        """Rounds carry the catalogue, the final answer sends none, and the loop narrows."""
        payloads: list[dict] = []
        seen: list = []

        def _recost(_conversation, tools):
            seen.append(tools)
            return _SHARE - (500 if tools else 100)

        _run_tool_loop(
            monkeypatch,
            payloads,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _recost,
        )

        assert len(seen) == 2
        assert [tool["function"]["name"] for tool in seen[0]] == ["web_search"]
        assert seen[1] is None, "the final pass sends no tools array"
        assert _caps(payloads) == [_SHARE - 500, _SHARE - 100]

    def test_a_respawn_refit_does_not_restore_the_window(self, monkeypatch):
        """A replacement server reporting a bigger window is not a bigger reservation."""
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
        # The window is what this test is about, not the fit's token count.
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        _run_tool_loop(
            monkeypatch,
            payloads,
            backend = backend,
            context_overflow = "truncate_oldest",
            admission_output_allowance = _SHARE,
        )

        assert payloads, "no request was sent"
        assert all(cap <= _SHARE for cap in _caps(payloads)), _caps(payloads)

    def test_every_final_continuation_is_re_costed(self, monkeypatch):
        """A continuation is a bigger prompt on the same lease."""
        payloads: list[dict] = []
        backend = _make_backend(
            monkeypatch,
            [
                _tool_call("web_search", {"query": "kernel"}, "c1"),
                [_sse({"content": "6.10 and then some"}), _finish("length"), _done()],
                [_sse({"content": " more"}), _done()],
            ],
            payloads,
        )
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 10)
        seen: list[int] = []
        recosted = iter([None, _SHARE - 100, _SHARE - 700])

        def _recost(messages, tools):
            if tools is None:
                seen.append(len(messages))
            return next(recosted, None)

        _run_tool_loop(
            monkeypatch,
            payloads,
            backend = backend,
            admission_output_allowance = _SHARE,
            on_conversation_grew = _recost,
        )

        assert len(seen) == 2, f"the final re-cost ran {len(seen)} time(s), not once per attempt"
        assert seen[1] >= seen[0], "the retry should carry at least the first attempt's prompt"
        assert _caps(payloads)[1:] == [_SHARE - 100, _SHARE - 700]


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


def _reservation():
    """A lease that accepts any re-cost, so a test reads the figure it hands back."""

    class _Lease:
        def recost_waiting(self, *_args, **_kwargs):
            return None

    return SimpleNamespace(lease_nowait = lambda: _Lease())


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
    """A generation call site added without the bound is the whole defect, again. Read off
    the route source: what has to hold is that NO branch is left out."""

    def _routes_tree(self):
        import ast
        return ast.parse(Path(inf_mod.__file__).read_text(encoding = "utf-8"))

    def _generator_calls(self, tree):
        """Calls on `llama_backend`, the only receiver holding a KV lease. The safetensors
        twin decodes in-process against no cache, so it takes no reservation."""
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
        """`payload.tools` omits Studio's server-side catalogue, which the lease charges."""
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
        """The catalogue is inside the bound, so no request is permitted `share + it`."""
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
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)

        reservation = _reservation()

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

    def test_a_client_that_named_a_cap_is_never_bounded_at_either_end(self):
        """A caller under the window is charged what it asked for, so a bound handed back
        mid-loop would cut its answer at a share with nothing to say why."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)

        reservation = _reservation()
        grown = [{"role": "user", "content": "word " * 900}]
        for payload, cap in (
            (_chat(max_tokens = 512), 512),
            (_chat(max_completion_tokens = 512), 512),
        ):
            assert (
                _openai_llama_admission_enforced_max_tokens(
                    payload, request = None, llama_backend = backend
                )
                is None
            )
            assert (
                _openai_llama_admission_recost(
                    reservation,
                    grown,
                    request = None,
                    llama_backend = backend,
                    payload = payload,
                    output_tokens = cap,
                )
                is None
            )


class TestWhatTheWireActuallyCarries:
    """The charge is deliberately conservative; the bound cannot be: the ledger re-adds
    terms the next request does not carry, and each would come off the answer."""

    def test_a_translated_system_prompt_is_not_charged_to_the_answer_twice(self):
        """`anthropic_messages_to_openai` folds `system` into the re-costed conversation,
        so the ledger's extra-prompt term counts it twice; on the wire that costs tokens."""
        backend = _backend_stub(window = 65536, total = 65536, slots = 4)
        system = "You are a careful assistant. " * 200
        payload = _Payload(
            messages = [{"role": "user", "content": "hi"}],
            system = system,
            max_tokens = 65536,
        )
        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": "hi"},
        ]

        reservation = _reservation()
        wire = _openai_llama_admission_recost(
            reservation,
            conversation,
            request = None,
            llama_backend = backend,
            payload = payload,
            output_tokens = 65536,
        )
        share = 65536 // 4
        conversation_tokens = _openai_llama_admission_wire_prompt_tokens(
            conversation, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        assert wire == share - conversation_tokens, (wire, share, conversation_tokens)

    def test_image_transport_bytes_do_not_come_off_the_answer(self):
        """Only OpenAI `image_url` parts are compacted, so an Anthropic image keeps base64."""
        # A share (8192) above one image's allowance but below the base64 as prompt text.
        backend = _backend_stub(window = 32768, total = 32768, slots = 4)
        data = "A" * 40000
        payload = _Payload(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": data},
                        },
                    ],
                }
            ],
            max_tokens = 32768,
        )
        translated = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                ],
            }
        ]
        raw = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        wire = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = translated
        )
        assert raw == 1, "the base64 transport should have swamped the share"
        assert wire > 1000, "the normalised part is priced as an image, not as prompt text"

    def test_a_request_is_not_charged_a_catalogue_it_does_not_send(self):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)

        reservation = _reservation()
        conversation = [{"role": "user", "content": "word " * 700}]

        def _recost(wire_tools):
            return _openai_llama_admission_recost(
                reservation,
                conversation,
                request = None,
                llama_backend = backend,
                payload = payload,
                output_tokens = 16384,
                injected_tools = _CATALOGUE,
                wire_tools = wire_tools,
            )

        with_tools = _recost(_CATALOGUE)
        without = _recost(None)
        assert without > with_tools, (with_tools, without)
        catalogue = _openai_llama_admission_prompt_tokens(
            _Payload(messages = [{"role": "user", "content": ""}]), injected_tools = _CATALOGUE
        ) - _openai_llama_admission_prompt_tokens(
            _Payload(messages = [{"role": "user", "content": ""}])
        )
        assert without - with_tools == catalogue

    def test_a_prompt_injected_after_the_payload_is_still_inside_the_bound(self):
        """The plain GGUF path splices in a date prompt and has no re-cost to notice it."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        payload = _chat(max_tokens = 16384)
        injected = [
            {"role": "system", "content": "Today's date is 2026-09-08. " * 40},
            {"role": "user", "content": "hi"},
        ]
        raw = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend
        )
        wire = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = injected
        )
        assert wire < raw, (raw, wire)
        assert (
            _openai_llama_admission_wire_prompt_tokens(
                injected, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            )
            + wire
            <= 16384 // 4
        )

    def test_audio_and_video_are_left_unenforced(self):
        """Nothing here can size their prompt KV: transport length as a prompt count would
        floor any real recording's answer at one token. Images keep a real bound."""
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        clip = "A" * 200000
        text_only = _chat("listen", max_tokens = 16384)
        assert (
            _openai_llama_admission_enforced_max_tokens(
                text_only, request = None, llama_backend = backend
            )
            is not None
        )
        with_audio = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "listen"},
                    {"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}},
                ],
            }
        ]
        assert (
            _openai_llama_admission_enforced_max_tokens(
                text_only,
                request = None,
                llama_backend = backend,
                conversation = with_audio,
            )
            is None
        )
        assert (
            _openai_llama_admission_enforced_max_tokens(
                _Payload(
                    messages = [{"role": "user", "content": "listen"}],
                    audio_base64 = clip,
                    max_tokens = 16384,
                ),
                request = None,
                llama_backend = backend,
            )
            is None
        )

    def test_a_legacy_image_already_spliced_in_is_charged_once(self):
        """The splice already counts `image_base64`; the separate charge doubles it."""
        backend = _backend_stub(window = 32768, total = 32768, slots = 4)
        image = "iVBORw0KGgo="
        payload = _Payload(
            messages = [{"role": "user", "content": "what is this?"}],
            image_base64 = image,
            max_tokens = 32768,
        )
        spliced = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ],
            }
        ]
        one_image = _openai_llama_admission_wire_prompt_tokens(
            spliced, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        text_only = _openai_llama_admission_wire_prompt_tokens(
            [{"role": "user", "content": "what is this?"}],
            image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS,
        )
        assert one_image - text_only < 2 * _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS, (
            one_image,
            text_only,
        )
        bound = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = spliced
        )
        assert bound is not None and bound > 1

    def test_the_lease_is_taken_on_the_same_finalized_messages(self):
        """Charged and permitted must be the same prompt, so both use `gguf_messages`."""
        budget, slots = 16384, 4
        share = budget // slots
        # Sized so the raw payload sits under its share and the finalized prompt over it.
        turn = "word " * 3200
        payload = _chat(turn, max_tokens = budget)
        injected = [
            {"role": "system", "content": "Today's date is 2026-09-08. " * 40},
            {"role": "user", "content": turn},
        ]
        charged = _openai_llama_admission_tokens(
            payload,
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = injected,
        )
        backend = _backend_stub(window = budget, total = budget, slots = slots)
        bound = _openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = backend, conversation = injected
        )
        prompt = _openai_llama_admission_wire_prompt_tokens(
            injected, image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
        )
        raw = _openai_llama_admission_prompt_tokens(payload)
        assert raw < share < prompt, (raw, share, prompt)
        assert charged >= prompt + bound, (charged, prompt, bound)

    def test_transport_bytes_stay_on_the_ledger(self):
        """The charge must keep them: they stop an audio request sharing the cache."""
        budget, slots = 262144, 4
        clip = "A" * 400000
        conversation = [{"role": "user", "content": "listen"}]
        without = _openai_llama_admission_tokens(
            _chat("listen", max_tokens = budget),
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = conversation,
        )
        with_clip = _openai_llama_admission_tokens(
            _Payload(
                messages = [{"role": "user", "content": "listen"}],
                audio_base64 = clip,
                max_tokens = budget,
            ),
            budget = budget,
            capacity = slots,
            context_window = budget,
            conversation = conversation,
        )
        assert with_clip > without
        assert with_clip >= len(clip) // 4


class TestARetryThatGrewItsPrompt:
    def test_the_nudge_retry_is_bounded_by_its_own_prompt(self):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        first = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4088}
        grown = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "word " * 500},
                {"role": "user", "content": "please call the tool"},
            ],
            "max_tokens": 4088,
        }
        bound = _openai_llama_admission_retry_max_tokens(
            grown, admission_output_allowance = 4088, request = None, llama_backend = backend
        )
        assert bound is not None and bound < first["max_tokens"]
        assert (
            _openai_llama_admission_wire_prompt_tokens(
                grown["messages"], image_tokens = _OPENAI_LLAMA_ADMISSION_IMAGE_TOKENS
            )
            + bound
            <= 16384 // 4
        )

    def test_a_client_that_named_a_cap_is_left_alone(self):
        backend = _backend_stub(window = 16384, total = 16384, slots = 4)
        grown = {"messages": [{"role": "user", "content": "word " * 500}], "max_tokens": 512}
        assert (
            _openai_llama_admission_retry_max_tokens(
                grown, admission_output_allowance = None, request = None, llama_backend = backend
            )
            is None
        )


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
        """`effective_parallel_slots` is unset early, so without the request capacity is 1."""
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
    """/v1/messages shares the slots, and its optional `max_tokens` reads as unstated."""

    @pytest.fixture(autouse = True)
    def _isolate(self, monkeypatch):
        reset_llama_admission_queues()
        monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
        monkeypatch.setattr(inf_mod, "_CANCEL_REGISTRY", {})
        yield
        reset_llama_admission_queues()

    def _install(
        self,
        monkeypatch,
        seen: dict,
        *,
        supports_tool_passthrough = False,
    ):
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
            supports_tool_passthrough = supports_tool_passthrough,
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

    def test_the_client_tool_passthrough_is_bounded(self, monkeypatch):
        """Returns through the passthrough builders, yet still takes a lease."""
        seen: dict = {}
        self._install(monkeypatch, seen, supports_tool_passthrough = True)
        captured: dict = {}

        async def _fake_passthrough(
            llama_backend,
            openai_messages,
            openai_tools,
            temperature,
            top_p,
            top_k,
            max_tokens,
            *args,
            **kwargs,
        ):
            captured["max_tokens"] = max_tokens
            captured["allowance"] = kwargs.get("admission_output_allowance")
            return JSONResponse(content = {"id": "msg_x"})

        monkeypatch.setattr(inf_mod, "_anthropic_passthrough_non_streaming", _fake_passthrough)
        payload = AnthropicMessagesRequest.model_validate(
            {
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Search the web.",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            }
        )

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        assert captured["max_tokens"] is not None
        assert captured["max_tokens"] <= 16384 // 4, captured
        assert captured["allowance"] == captured["max_tokens"]

    def test_the_tool_generator_is_bounded(self, monkeypatch):
        seen: dict = {}
        self._install(monkeypatch, seen)
        # Unsloth's own server-side loop; a client catalogue takes the passthrough instead.
        payload = AnthropicMessagesRequest.model_validate(
            {
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": "hi"}],
                "enable_tools": True,
                # The route requires an explicit permission mode to run server tools.
                "permission_mode": "off",
            }
        )

        asyncio.run(anthropic_messages(payload, request = _AnthropicRequest(), current_subject = "t"))

        allowance = seen["tools"]["admission_output_allowance"]
        assert allowance is not None and allowance <= 16384 // 4
