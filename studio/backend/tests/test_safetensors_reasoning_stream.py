# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safetensors/MLX reasoning-block parity with GGUF.

enable_thinking templates (Qwen3/GLM) prefill an unclosed ``<think>`` so the model
emits only the closing ``</think>`` then the answer; the safetensors stream must
split the leading text into ``reasoning_content`` deltas (plain stream and tool
loop), resetting per turn and appending only visible text to the monitor. Replays a
copy of ``sf_tool_stream``'s reasoning loop against synthetic events.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.inference import (
    _ResponsesReasoningExtractor,
    _sf_reasoning_prefill_mode,
    _strip_tool_xml_for_display,
)


_THINK_TPL = "...<think>...</think>..."
_ETHINK = {"reasoning_style": "enable_thinking", "supports_reasoning": True}
_ETHINK_EFFORT = {"reasoning_style": "enable_thinking_effort", "supports_reasoning": True}


def test_prefill_mode_on_for_enable_thinking_default():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _THINK_TPL) is True


def test_prefill_mode_off_when_thinking_disabled():
    assert _sf_reasoning_prefill_mode(_ETHINK, False, _THINK_TPL) is False


def test_prefill_mode_off_for_reasoning_effort_none():
    # enable_thinking_effort turns thinking off via reasoning_effort="none"; prefilled mode
    # would capture the whole answer as reasoning_content.
    assert (
        _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _THINK_TPL, reasoning_effort = "none")
        is False
    )
    assert (
        _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _THINK_TPL, reasoning_effort = "high")
        is True
    )


def test_prefill_mode_off_without_think_markers():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, "no markers here") is False


def _replay_sf_reasoning_stream(events: list[dict], *, prefilled: bool) -> dict:
    """Mirror sf_tool_stream's reasoning loop: diff each cumulative ``content``
    snapshot, feed the delta through the extractor, and reset (flushing first) on
    ``tool_start`` / empty ``status`` so each turn splits independently."""
    prev_text = ""
    extractor = _ResponsesReasoningExtractor(
        parse_think_markers = True, reasoning_prefilled = prefilled
    )
    reasoning_deltas: list[str] = []
    visible_deltas: list[str] = []
    monitor: list[str] = []
    tool_starts: list[dict] = []
    order: list[str] = []  # sequence of ("reasoning"|"visible"|"tool_start") events

    def _flush():
        fr, fv = extractor.finish()
        if fr:
            reasoning_deltas.append(fr)
            order.append("reasoning")
        if fv:
            visible_deltas.append(fv)
            monitor.append(fv)
            order.append("visible")

    for event in events:
        etype = event["type"]
        if etype == "status":
            if not event["text"]:
                _flush()
                prev_text = ""
                extractor = _ResponsesReasoningExtractor(
                    parse_think_markers = True, reasoning_prefilled = prefilled
                )
            continue
        if etype in ("tool_start", "tool_end"):
            if etype == "tool_start":
                _flush()
                prev_text = ""
                extractor = _ResponsesReasoningExtractor(
                    parse_think_markers = True, reasoning_prefilled = prefilled
                )
                tool_starts.append(event)
                order.append("tool_start")
            continue
        clean = _strip_tool_xml_for_display(event.get("text", ""), auto_heal_tool_calls = True)
        new_text = clean[len(prev_text) :]
        prev_text = clean
        if not new_text:
            continue
        r, v = extractor.feed(new_text)
        if r:
            reasoning_deltas.append(r)
            order.append("reasoning")
        if v:
            visible_deltas.append(v)
            monitor.append(v)
            order.append("visible")
    _flush()
    return {
        "reasoning": "".join(reasoning_deltas),
        "visible": "".join(visible_deltas),
        "monitor": "".join(monitor),
        "tool_starts": tool_starts,
        "order": order,
    }


def test_s1_plain_stream_splits_prefilled_reasoning():
    # S1: plain/MLX single turn -> reasoning delta + visible delta; monitor visible-only.
    events = [
        {"type": "content", "text": "Let me compute 17*23"},
        {"type": "content", "text": "Let me compute 17*23 = 391</think>The answer is 391."},
    ]
    out = _replay_sf_reasoning_stream(events, prefilled = True)
    assert out["reasoning"] == "Let me compute 17*23 = 391"
    assert out["visible"] == "The answer is 391."
    assert out["monitor"] == "The answer is 391."
    assert "<think>" not in out["reasoning"] and "</think>" not in out["visible"]


def test_s2_reasoning_flushed_before_tool_start():
    # S2: reasoning streamed as reasoning_content, then flushed BEFORE tool_start.
    events = [
        {"type": "content", "text": "I should search"},
        {"type": "content", "text": "I should search Sydney weather</think>"},
        {"type": "tool_start", "tool_name": "web_search", "tool_call_id": "c0"},
        {"type": "tool_end", "tool_name": "web_search", "tool_call_id": "c0"},
        {"type": "status", "text": ""},
        {"type": "content", "text": "Found it</think>Sydney is 21C today."},
    ]
    out = _replay_sf_reasoning_stream(events, prefilled = True)
    # Both turns' reasoning surfaced, answer only from turn 2.
    assert "I should search Sydney weather" in out["reasoning"]
    assert "Found it" in out["reasoning"]
    assert out["visible"] == "Sydney is 21C today."
    assert out["monitor"] == "Sydney is 21C today."
    # Ordering: the pre-tool reasoning is emitted before the tool_start.
    assert out["order"].index("reasoning") < out["order"].index("tool_start")


def test_s3_extractor_resets_each_turn():
    # S3: multi-turn -> the two turns' reasoning are distinct (fresh extractor each).
    events = [
        {"type": "content", "text": "turn1 thoughts</think>partial"},
        {"type": "status", "text": ""},
        {"type": "content", "text": "turn2 thoughts</think>final answer"},
    ]
    out = _replay_sf_reasoning_stream(events, prefilled = True)
    assert out["reasoning"] == "turn1 thoughtsturn2 thoughts"
    assert out["visible"] == "partialfinal answer"


def test_s4_harmony_full_tags_normal_mode():
    # S4: gpt-oss / explicit-tag models use normal mode (prefilled=False).
    events = [{"type": "content", "text": "<think>reasoning here</think>visible answer"}]
    out = _replay_sf_reasoning_stream(events, prefilled = False)
    assert out["reasoning"] == "reasoning here"
    assert out["visible"] == "visible answer"


def test_s5_thinking_off_no_reasoning_deltas():
    # S5: thinking disabled -> not prefilled, no </think>, all content is visible.
    events = [{"type": "content", "text": "Just the plain answer, no thinking."}]
    out = _replay_sf_reasoning_stream(events, prefilled = False)
    assert out["reasoning"] == ""
    assert out["visible"] == "Just the plain answer, no thinking."
    assert out["monitor"] == "Just the plain answer, no thinking."


def test_s6_reasoning_effort_none_disables_prefill_for_enable_thinking_effort():
    # GLM-5.2-style enable_thinking_effort: a request with reasoning_effort="none" (and
    # enable_thinking omitted) disables thinking exactly like enable_thinking=False, so
    # prefilled mode must be OFF. Otherwise the model emits no </think> and a plain
    # answer is swallowed whole into reasoning_content, leaving the visible response
    # empty (the exact bug: prefilled=True below eats the whole answer).
    feats = {"reasoning_style": "enable_thinking_effort", "supports_reasoning": True}
    assert _sf_reasoning_prefill_mode(feats, None, _THINK_TPL, "none") is False
    # Thinking on (effort level or default) still prefills.
    assert _sf_reasoning_prefill_mode(feats, None, _THINK_TPL, "high") is True
    assert _sf_reasoning_prefill_mode(feats, None, _THINK_TPL, None) is True
    # An explicit enable_thinking=False also disables (unchanged).
    assert _sf_reasoning_prefill_mode(feats, False, _THINK_TPL, "high") is False
    # reasoning_always_on wins regardless of reasoning_effort.
    always = {**feats, "reasoning_always_on": True}
    assert _sf_reasoning_prefill_mode(always, None, _THINK_TPL, "none") is True
    # Plain enable_thinking models (Qwen) have no "none" sentinel; unaffected.
    plain = {"reasoning_style": "enable_thinking", "supports_reasoning": True}
    assert _sf_reasoning_prefill_mode(plain, None, _THINK_TPL, "none") is True

    # End-to-end: with the corrected prefilled=False, a plain no-</think> answer is
    # emitted as visible content rather than swallowed into the thinking drawer.
    events = [{"type": "content", "text": "The capital of France is Paris."}]
    out = _replay_sf_reasoning_stream(events, prefilled = False)
    assert out["visible"] == "The capital of France is Paris."
    assert out["reasoning"] == ""
    # The buggy prefilled=True path is what swallowed the whole answer (guard the delta).
    swallowed = _replay_sf_reasoning_stream(events, prefilled = True)
    assert swallowed["visible"] == ""
    assert swallowed["reasoning"] == "The capital of France is Paris."


def test_native_reasoning_streamer_selected_and_errors_raise():
    import threading
    import pytest

    torch = pytest.importorskip("torch")
    inf = pytest.importorskip("core.inference.inference")

    class Batch(dict):
        def to(self, _device):
            return self

    class Tok:
        chat_template = "<|channel>thought\n...<channel|>"
        all_special_tokens = []
        eos_token_id = 1
        pad_token_id = None
        pieces = {10: "<|channel>thought\n", 11: "r", 12: "<channel|>", 13: "a"}

        def __call__(self, *_args, **_kwargs):
            return Batch({"input_ids": torch.zeros((1, 1), dtype = torch.long)})

        def decode(self, ids, **_kwargs):
            return "".join(self.pieces.get(int(token_id), "") for token_id in ids)

    class Model:
        device = "cpu"
        generation_config = type("Cfg", (), {"eos_token_id": 1})()
        config = generation_config

        def __init__(self, fail = False):
            self.fail = fail
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            streamer = kwargs["streamer"]
            streamer.put(torch.zeros((1, 1), dtype = torch.long))
            for token_id in [10, 11, 12, 13]:
                streamer.put(torch.tensor([token_id]))
                if self.fail:
                    raise RuntimeError("boom")

    backend = inf.InferenceBackend.__new__(inf.InferenceBackend)
    backend.active_model_name = "gemma-test"
    backend._generation_lock = threading.Lock()
    backend.models = {"gemma-test": {"model": Model(), "tokenizer": Tok()}}

    assert list(backend.generate_stream("prompt", max_new_tokens = 4))[-1] == "<think>r</think>a"

    backend.models["gemma-test"]["model"] = Model(fail = True)

    with pytest.raises(inf._GenerationThreadError, match = "boom"):
        list(backend.generate_stream("prompt", max_new_tokens = 4))
