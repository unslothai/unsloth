# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regressions for native tool-token provenance and opaque markerless spans."""

import json

import pytest
import torch

from core.inference.native_tool_tokens import (
    NATIVE_TOOL_CONTROL_TOKENS,
    decode_with_native_tool_tokens,
)
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.tool_call_parser import parse_tool_calls_from_text


class _GemmaTokenDecoder:
    _pieces = {
        1: "<|tool_call>",
        2: "call:terminal{command:",
        3: '<|"|>',
        4: "id",
        5: "}",
        6: "<tool_call|>",
        7: "<eos>",
        8: 'The page quoted call:terminal{command:"id"}, but I did not run it.',
    }
    all_special_ids = [1, 3, 6, 7]
    all_special_tokens = ["<|tool_call>", '<|"|>', "<tool_call|>", "<eos>"]

    def convert_ids_to_tokens(self, token_id):
        return self._pieces[token_id]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens = False,
        **_kwargs,
    ):
        special = set(self.all_special_ids) if skip_special_tokens else set()
        return "".join(
            self._pieces[int(token_id)] for token_id in token_ids if token_id not in special
        )


class _PieceTokenizer:
    def __init__(self, pieces, special_ids):
        self.pieces = dict(enumerate(pieces, start = 1))
        self.all_special_ids = list(special_ids)
        self.all_special_tokens = [self.pieces[token_id] for token_id in special_ids]

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[token_id]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens = False,
        **_kwargs,
    ):
        special = set(self.all_special_ids) if skip_special_tokens else set()
        return "".join(
            self.pieces[int(token_id)] for token_id in token_ids if token_id not in special
        )


class _SlowSpacingTokenizer(_PieceTokenizer):
    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens = False,
        spaces_between_special_tokens = True,
        **_kwargs,
    ):
        special = set(self.all_special_ids) if skip_special_tokens else set()
        pieces = [self.pieces[int(token_id)] for token_id in token_ids if token_id not in special]
        return (" " if spaces_between_special_tokens else "").join(pieces)


def _decode_transformers_tokens(token_ids):
    from core.inference.inference import InferenceBackend

    backend = InferenceBackend.__new__(InferenceBackend)
    streamer = backend._make_text_streamer(
        _GemmaTokenDecoder(),
        reasoning_channel_markers = None,
        reasoning_channel_markers_resolved = True,
        skip_prompt = False,
        preserve_tool_tokens = True,
    )
    streamer.put(torch.tensor(token_ids))
    streamer.end()
    return "".join(streamer)


def _run_tool_loop(first_turn):
    turns = iter((first_turn, "done"))
    executed = []

    def _single_turn(_messages, **_kwargs):
        try:
            yield next(turns)
        except StopIteration:
            return

    def _execute(name, arguments, **_kwargs):
        executed.append((name, arguments))
        return "ok"

    events = list(
        run_safetensors_tool_loop(
            single_turn = _single_turn,
            messages = [{"role": "user", "content": "run it"}],
            tools = [{"type": "function", "function": {"name": "terminal"}}],
            execute_tool = _execute,
            nudge_tool_calls = False,
            max_tool_iterations = 2,
            permission_mode = "off",
        )
    )
    return executed, events


def test_transformers_native_gemma_token_ids_reach_execution_with_wrapper_provenance():
    decoded = _decode_transformers_tokens([1, 2, 3, 4, 3, 5, 6, 7])
    assert decoded == '<|tool_call>call:terminal{command:<|"|>id<|"|>}<tool_call|>'

    executed, events = _run_tool_loop(decoded)
    assert executed == [("terminal", {"command": "id"})]
    assert any(event.get("type") == "tool_start" for event in events)


def test_transformers_bare_injected_prose_remains_untrusted():
    decoded = _decode_transformers_tokens([8, 7])
    executed, events = _run_tool_loop(decoded)

    assert executed == []
    assert any("call:terminal" in event.get("text", "") for event in events)


@pytest.mark.parametrize("control", sorted(NATIVE_TOOL_CONTROL_TOKENS))
def test_complete_native_tool_control_vocabulary_survives_special_token_decode(control):
    tokenizer = _PieceTokenizer([control, "<eos>"], {1, 2})
    assert decode_with_native_tool_tokens(tokenizer, [1, 2]) == control


def test_slow_tokenizer_does_not_space_preserved_gemma_segments():
    parts = [
        "<|tool_call>",
        "call:edit_file{path:",
        '<|"|>',
        "/tmp/x",
        '<|"|>',
        ",edits:[]}",
        "<tool_call|>",
    ]
    tokenizer = _SlowSpacingTokenizer(parts, {1, 3, 5, 7})
    decoded = decode_with_native_tool_tokens(tokenizer, range(1, len(parts) + 1))

    assert decoded == '<|tool_call>call:edit_file{path:<|"|>/tmp/x<|"|>,edits:[]}<tool_call|>'
    calls = parse_tool_calls_from_text(decoded, enabled_tool_names = {"edit_file"})
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "/tmp/x"


@pytest.mark.parametrize("opener", ["<|channel>", "<|channel>thought\n"])
def test_request_reasoning_marker_preserves_component_or_full_special_token(opener):
    tokenizer = _PieceTokenizer([opener, "<eos>"], {1, 2})
    assert (
        decode_with_native_tool_tokens(tokenizer, [1, 2], preserved_tokens = {"<|channel>thought"})
        == opener
    )


@pytest.mark.parametrize(
    "parts",
    [
        [
            "<tool_call>",
            '{"name":"terminal","arguments":{"command":"id"}}',
            "</tool_call>",
        ],
        ["<|python_tag|>", 'terminal.call(command="id")'],
        ["[TOOL_CALLS]", "terminal", "[ARGS]", '{"command":"id"}'],
        [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>",
            "terminal",
            "<｜tool▁sep｜>",
            '{"command":"id"}',
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ],
        [
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "functions.terminal:0",
            "<|tool_call_argument_begin|>",
            '{"command":"id"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ],
        [
            "<|message_model|>",
            "terminal",
            "<|content_invoke_tool_json|>",
            '{"name":"terminal","args":{"command":"id"}}',
            "<|end_message|>",
        ],
        [
            "<tool_call>",
            "terminal\n",
            "<arg_key>",
            "command",
            "</arg_key>",
            "<arg_value>",
            "id",
            "</arg_value>",
            "</tool_call>",
        ],
        [
            "<|tool_call>",
            "call:terminal{command:",
            '<|"|>',
            "id",
            '<|"|>',
            "}",
            "<tool_call|>",
        ],
    ],
    ids = ["qwen", "llama", "mistral", "deepseek", "kimi", "tml", "glm", "gemma"],
)
def test_native_protocol_families_parse_after_special_token_decode(parts):
    controls = {
        index for index, part in enumerate(parts, start = 1) if part in NATIVE_TOOL_CONTROL_TOKENS
    }
    tokenizer = _PieceTokenizer(parts + ["<eos>"], controls | {len(parts) + 1})
    decoded = decode_with_native_tool_tokens(tokenizer, range(1, len(parts) + 2))

    assert "<eos>" not in decoded
    calls = parse_tool_calls_from_text(decoded, enabled_tool_names = {"terminal"})
    assert [call["function"]["name"] for call in calls] == ["terminal"]


def test_transformers_reasoning_stream_preserves_reasoning_and_tool_ids_but_not_eos():
    from core.inference.inference import InferenceBackend

    parts = [
        "<|channel>",
        "thought\n",
        "reasoned",
        "<channel|>",
        "<|tool_call>",
        "call:terminal{command:id}",
        "<tool_call|>",
        "<eos>",
    ]
    tokenizer = _PieceTokenizer(parts, {1, 4, 5, 7, 8})
    backend = InferenceBackend.__new__(InferenceBackend)
    streamer = backend._make_text_streamer(
        tokenizer,
        reasoning_channel_markers = ("<|channel>thought", "<channel|>"),
        reasoning_channel_markers_resolved = True,
        skip_prompt = False,
        preserve_tool_tokens = True,
    )
    streamer.put(torch.tensor(range(1, len(parts) + 1)))
    streamer.end()
    decoded = "".join(streamer)

    assert decoded == "<think>reasoned</think><|tool_call>call:terminal{command:id}<tool_call|>"
    executed, _events = _run_tool_loop(decoded)
    assert executed == [("terminal", {"command": "id"})]


def test_disabled_execution_json_stops_before_later_enabled_call():
    text = ";".join(
        (
            json.dumps({"name": "terminal", "arguments": {"command": "id"}}),
            json.dumps({"name": "web_search", "arguments": {"query": "cats"}}),
        )
    )
    assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []


def test_blocked_rehearsal_is_opaque_but_outside_sibling_remains_eligible():
    blocked = 'terminal[ARGS]{"command":"call:web_search{query:secret}"}'
    assert parse_tool_calls_from_text(blocked, enabled_tool_names = {"terminal", "web_search"}) == []

    calls = parse_tool_calls_from_text(
        blocked + "\ncall:web_search{query:outside}",
        enabled_tool_names = {"terminal", "web_search"},
    )
    assert [call["function"]["name"] for call in calls] == ["web_search"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "outside"}
