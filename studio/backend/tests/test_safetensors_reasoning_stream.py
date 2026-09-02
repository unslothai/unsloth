# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safetensors/MLX reasoning-block parity with GGUF.

Some enable_thinking templates prefill an unclosed ``<think>`` so the model emits only
the closing ``</think>`` then the answer; the safetensors stream must split the leading
text into ``reasoning_content`` deltas (plain stream and tool loop), resetting per turn
and appending only visible text to the monitor. Others render a closed block or none at
all and the answer is visible from the first token, so the prefill mode is read from the
generation prompt the request renders. Replays a copy of ``sf_tool_stream``'s reasoning
loop against synthetic events, and covers the split through the route itself.
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

import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Register a stub module for a dep the backend pytest job does not install.

    Same helper and reason as test_audio_type_inconclusive.py and
    test_trainer_stdout_quiet.py: ``core.inference.inference`` imports ``unsloth``
    (and through it ``unsloth_zoo``) at module scope, while the pytest matrix in
    studio-backend-ci.yml installs studio.txt plus torch and transformers and
    deliberately stops there. A real install is left alone.

    This file used to have no stub at all. The three tests below reach
    ``core.inference.inference`` through ``pytest.importorskip`` inside the test
    body, which is lazy enough that the module-scope guard in
    test_backend_tests_stub_heavy_imports.py does not look at it, so the omission
    was invisible. They passed anyway, because some earlier file in the same
    session had installed this stub and left the imported module in
    ``sys.modules`` for them. Run this file first, or on its own, and the import
    raises ``ImportError: Please install unsloth_zoo``, which pytest 8.2+ no
    longer converts to a skip (only ``ModuleNotFoundError`` does that), so it is
    a hard failure rather than the intended skip.

    Stubbing here rather than switching to skipif keeps the coverage: the module
    under test is the real ``core.inference.inference``, and only ``unsloth``
    itself is faked.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - unusable here either way, so stub it
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

# Build the module while the stubs are live, then drop them, exactly as
# test_audio_type_inconclusive.py does. The three tests below reach this through
# pytest.importorskip and get it from sys.modules, so the stubs only have to exist
# for this one import.
#
# Dropping them is not tidiness. A stub left in sys.modules is a cross-file leak:
# test_audio_type_inconclusive.py::test_the_stubs_do_not_outlive_this_module asserts
# nobody does it, and every other file's _stub_if_missing returns early when the name
# is already present, so its own bookkeeping never runs and its cleanup has nothing to
# undo. Leaving them installed traded this file's order dependency for a worse one.
_EAGER_IMPORT_ERROR: str | None = None
try:
    import core.inference.inference  # noqa: E402,F401
except ImportError as _error:  # recorded, not swallowed; see the test at the bottom
    _EAGER_IMPORT_ERROR = f"{type(_error).__name__}: {_error}"

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


# DeepSeek-R1 / QwQ / GLM shape: the generation prompt opens an unclosed ``<think>``.
_THINK_TPL = (
    "{% for m in messages %}<|user|>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n<think>\n{% endif %}"
)
# Qwen3.5 shape: a CLOSED ``<think></think>`` unless thinking is explicitly requested.
_TEMPLATE_DEFAULT_OFF_TPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is true %}<think>\n"
    "{% else %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
# Opens ``<think>`` only at high effort.
_EFFORT_SHAPE_TPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if reasoning_effort is defined and reasoning_effort == 'high' %}<think>\n"
    "{% else %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
# Stamps a date through the ``strftime_now`` global transformers exposes to every template.
_STRFTIME_TPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{{ strftime_now('%Y') }}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
# Nemotron shape: thinking is off until a message opts in with ``/think``.
_MESSAGE_SHAPE_TPL = (
    "{% set ns = namespace(think = false) %}"
    "{% for m in messages %}{% if '/think' in m['content'] %}{% set ns.think = true %}{% endif %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if ns.think %}<think>\n{% else %}<think></think>{% endif %}{% endif %}"
)
# Kimi shape: renders history into the prompt but opens no block of its own.
_HISTORY_ONLY_TPL = (
    "{% for m in messages %}<|im_user|>{{ m['role'] }}<|im_middle|>{{ m['content'] }}<|im_end|>"
    "{% if m['role'] == 'assistant' %}<think></think>{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_assistant|>assistant<|im_middle|>{% endif %}"
)
# Closes its block, and raises on a tool turn the way a strict template does.
_STRICT_HISTORY_TPL = (
    "{% for m in messages %}{% if m['role'] == 'tool' %}{{ raise_exception('no tool turns') }}"
    "{% endif %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
_ETHINK = {"reasoning_style": "enable_thinking", "supports_reasoning": True}
_ETHINK_EFFORT = {"reasoning_style": "enable_thinking_effort", "supports_reasoning": True}


def test_prefill_mode_on_for_enable_thinking_default():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _THINK_TPL) is True


def test_prefill_mode_follows_template_default_not_the_request_flag():
    # The kwarg is omitted when the request says nothing, so the template's own default
    # decides. Assuming a prefill blanked ``content`` for every OpenAI client.
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _TEMPLATE_DEFAULT_OFF_TPL) is False
    assert _sf_reasoning_prefill_mode(_ETHINK, True, _TEMPLATE_DEFAULT_OFF_TPL) is True


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


def test_prefill_mode_renders_with_the_requested_reasoning_effort():
    # Rendering without the request's effort would report the closed shape for both.
    assert _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _EFFORT_SHAPE_TPL, "high") is True
    assert _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _EFFORT_SHAPE_TPL, "low") is False


def test_a_template_that_stamps_a_date_still_renders():
    # Without ``strftime_now`` the render raises and the failure reads as a prefill.
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _STRFTIME_TPL) is False


def test_prefill_mode_off_without_think_markers():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, "no markers here") is False


def test_prefill_mode_renders_the_request_messages():
    # The template reads the shape off the conversation, so a fixed stand-in classifies a
    # request nobody made and the answer to the opted-in one lands in visible content.
    plain = [{"role": "user", "content": "what is 2+2"}]
    opted_in = [{"role": "user", "content": "/think what is 2+2"}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, plain) is False
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, opted_in) is True


def test_messages_the_template_refuses_fall_back_to_the_single_user_probe():
    # A history the template raises on must not itself read as a prefill: fall back to the
    # stand-in every caller got before the messages were threaded through.
    refused = [{"role": "user", "content": "hi"}, {"role": "tool", "content": "42"}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _STRICT_HISTORY_TPL, None, refused) is False
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _STRICT_HISTORY_TPL) is False


def test_a_think_tag_the_user_typed_does_not_prefill():
    # Only the assistant prefix counts. Weighing the whole prompt would let anyone asking
    # about a <think> tag become the last opener and blank their own answer.
    for text in ("how do I emit a <think> tag?", "use <think>x</think> tags", "plain"):
        msgs = [{"role": "user", "content": text}]
        assert _sf_reasoning_prefill_mode(_ETHINK, None, _HISTORY_ONLY_TPL, None, msgs) is False
    # A template that really does open one is still read as prefilled.
    typed = [{"role": "user", "content": "a <think> b"}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _THINK_TPL, None, typed) is True


def test_content_parts_must_reach_the_probe_flattened():
    # Why the route probes the normalized conversation: the shapes disagree, so a raw array
    # would test ``'/think' in <list>``, miss the opt-in, and classify a prompt never rendered.
    flattened = [{"role": "user", "content": "/think what is 2+2"}]
    raw_parts = [{"role": "user", "content": [{"type": "text", "text": "/think what is 2+2"}]}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, flattened) is True
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, raw_parts) is False


def test_control_markup_must_reach_the_probe_swept():
    # The renderer neutralizes control markup first, so a user's ``<think>`` arrives as
    # ``< think>``. A template branching on it would otherwise split from generation.
    from core.inference.chat_template_helpers import neutralize_control_markup_in_messages

    raw = [{"role": "user", "content": "<think> tags"}]
    swept = neutralize_control_markup_in_messages([dict(m) for m in raw])
    assert swept[0]["content"] == "< think> tags"
    marker_tpl = _MESSAGE_SHAPE_TPL.replace("'/think' in", "'<think>' in")
    assert _sf_reasoning_prefill_mode(_ETHINK, None, marker_tpl, None, raw) is True
    assert _sf_reasoning_prefill_mode(_ETHINK, None, marker_tpl, None, swept) is False


def test_prefill_mode_without_messages_matches_the_single_user_probe():
    # Omitting the argument keeps the pre-existing signature and verdict.
    for tpl in (_THINK_TPL, _TEMPLATE_DEFAULT_OFF_TPL, _MESSAGE_SHAPE_TPL):
        assert _sf_reasoning_prefill_mode(_ETHINK, None, tpl) == _sf_reasoning_prefill_mode(
            _ETHINK, None, tpl, None, [{"role": "user", "content": "hi"}]
        )


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


def test_native_reasoning_streamer_starts_inside_prompt_opened_channel():
    """A post-tool prompt opens the channel, so generation emits only its close."""
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
        pieces = {11: "reasoned", 12: "<channel|>", 13: "answer"}

        def __call__(self, *_args, **_kwargs):
            return Batch({"input_ids": torch.zeros((1, 1), dtype = torch.long)})

        def decode(self, ids, **_kwargs):
            return "".join(self.pieces.get(int(token_id), "") for token_id in ids)

    class Model:
        device = "cpu"
        generation_config = type("Cfg", (), {"eos_token_id": 1})()
        config = generation_config

        def generate(self, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put(torch.zeros((1, 1), dtype = torch.long))
            for token_id in [11, 12, 13]:
                streamer.put(torch.tensor([token_id]))

    backend = inf.InferenceBackend.__new__(inf.InferenceBackend)
    backend.active_model_name = "gemma-test"
    backend._generation_lock = threading.Lock()
    backend.models = {"gemma-test": {"model": Model(), "tokenizer": Tok()}}

    post_tool_prompt = "<|tool_response>response:web_search{}<tool_response|><|channel>thought\n"
    assert list(backend.generate_stream(post_tool_prompt, max_new_tokens = 4))[-1] == (
        "<think>reasoned</think>answer"
    )


def test_text_only_vlm_fallback_resolves_native_markers_off():
    import threading
    import pytest

    torch = pytest.importorskip("torch")
    inf = pytest.importorskip("core.inference.inference")

    class Batch(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        all_special_tokens = []
        eos_token_id = 1
        pad_token_id = None

        def __call__(self, *_args, **_kwargs):
            return Batch({"input_ids": torch.zeros((1, 1), dtype = torch.long)})

    class Processor:
        chat_template = "<|channel>thought\n...<channel|>"
        tokenizer = Tokenizer()

    class Model:
        device = "cpu"
        generation_config = type("Cfg", (), {"eos_token_id": 1})()
        config = generation_config

        def generate(self, **_kwargs):
            return None

    class EmptyStreamer:
        def __next__(self):
            raise StopIteration

        def end(self):
            return None

    captured = {}
    backend = inf.InferenceBackend.__new__(inf.InferenceBackend)
    backend.active_model_name = "vision-test"
    backend._generation_lock = threading.Lock()
    backend.models = {
        "vision-test": {
            "model": Model(),
            "processor": Processor(),
            "tokenizer": Processor(),
        }
    }
    backend.format_chat_prompt = lambda *_args, **_kwargs: "manual text-only prompt"

    def make_streamer(*_args, **kwargs):
        captured.update(kwargs)
        return EmptyStreamer()

    backend._make_text_streamer = make_streamer

    assert (
        list(
            backend._generate_vision_response(
                messages = [{"role": "user", "content": "hello"}],
                system_prompt = "",
                image = None,
                temperature = 0.7,
                top_p = 0.9,
                top_k = 40,
                min_p = 0.0,
                max_new_tokens = 1,
                repetition_penalty = 1.0,
            )
        )
        == []
    )
    assert captured["reasoning_channel_markers"] is None
    assert captured["reasoning_channel_markers_resolved"] is True
    # No markers on this branch, so this pins forwarding only.
    assert captured["prompt"] == "manual text-only prompt"


def test_the_eager_import_under_the_stubs_actually_succeeded():
    """A failed eager import turns the three importorskip tests into silent skips.

    They resolve out of ``sys.modules``, so if the import above did not put
    ``core.inference.inference`` there, ``importorskip`` finds the dependency
    genuinely missing and skips. The job stays green while a third of this file
    stops running, which is how the missing ``peft`` stub went unnoticed: 10 passed
    and 3 skipped on the matrix, reported as success.

    So the swallow records the error instead of dropping it, and this reads it back.
    A module-scope dependency added to core/inference/inference.py that the backend
    job does not install fails here by name rather than quietly reducing coverage.
    """
    assert _EAGER_IMPORT_ERROR is None, (
        f"the eager import of core.inference.inference failed ({_EAGER_IMPORT_ERROR}), so the "
        f"importorskip tests in this file skip instead of running. Install what it names in "
        f"the backend job's extras, or stub it above the import where a stub is safe (it is "
        f"not for anything transformers probes with importlib.util.find_spec)."
    )
    assert "core.inference.inference" in sys.modules


def _sf_route_message(monkeypatch, template, snapshots, **body):
    """POST a non-streaming safetensors chat completion and return the assistant message."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import routes.inference as inference_route
    from auth.authentication import get_current_subject
    from utils.api_errors import install_api_error_handlers

    class _NoGGUF:
        is_loaded = False
        supports_tools = False

    class _Safetensors:
        active_model_name = "qwen"
        models = {"qwen": {"chat_template_info": {"template": template}}}

        def generate_chat_response(self, **kwargs):
            yield from snapshots

        def reset_generation_state(self, *_args):
            return None

    monkeypatch.setattr(
        inference_route,
        "_detect_safetensors_features",
        lambda backend, chat_template, tools = None: dict(_ETHINK, supports_tools = False),
    )
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _NoGGUF())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Safetensors())

    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            **body,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["choices"][0]["message"]


def test_route_returns_the_answer_as_content_when_the_template_closes_its_block(monkeypatch):
    """The user-visible symptom: a plain answer reaching ``content``, not the thinking drawer."""
    # generate_chat_response yields cumulative text snapshots, not events.
    snapshots = ["The capital", "The capital of Japan is Tokyo."]
    message = _sf_route_message(monkeypatch, _TEMPLATE_DEFAULT_OFF_TPL, snapshots)
    assert message["content"] == "The capital of Japan is Tokyo."
    assert not message["reasoning_content"]
