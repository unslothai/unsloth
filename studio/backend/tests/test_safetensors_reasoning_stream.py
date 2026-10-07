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

import json
import sys
from pathlib import Path


def _shared_setup_1():
    import threading
    import pytest

    torch = pytest.importorskip("torch")
    inf = pytest.importorskip("core.inference.inference")
    return inf, pytest, threading, torch


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.inference import (
    _ResponsesReasoningExtractor,
    _sf_parse_think_markers,
    _sf_reasoning_prefill_mode,
    _strip_tool_xml_for_display,
)

import importlib  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
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

# Stubs live only for this import: a leftover stub in sys.modules leaks into other test files.
_EAGER_IMPORT_ERROR: str | None = None
try:
    import core.inference.inference  # noqa: E402,F401
except ImportError as _error:  # recorded, not swallowed; see the test at the bottom
    _EAGER_IMPORT_ERROR = f"{type(_error).__name__}: {_error}"

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


_THINK_TPL = (
    "{% for m in messages %}<|user|>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n<think>\n{% endif %}"
)
_TEMPLATE_DEFAULT_OFF_TPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is true %}<think>\n"
    "{% else %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
_EFFORT_SHAPE_TPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if reasoning_effort is defined and reasoning_effort == 'high' %}<think>\n"
    "{% else %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
_STRFTIME_TPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{{ strftime_now('%Y') }}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
_MESSAGE_SHAPE_TPL = (
    "{% set ns = namespace(think = false) %}"
    "{% for m in messages %}{% if '/think' in m['content'] %}{% set ns.think = true %}{% endif %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if ns.think %}<think>\n{% else %}<think></think>{% endif %}{% endif %}"
)
_THINK_OFF_TAG_TPL = (
    "{% set ns = namespace(off = false) %}"
    "{% for m in messages %}{% if '<|think_off|>' in m['content'] %}{% set ns.off = true %}"
    "{% endif %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if ns.off %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}"
)
_HISTORY_ONLY_TPL = (
    "{% for m in messages %}<|im_user|>{{ m['role'] }}<|im_middle|>{{ m['content'] }}<|im_end|>"
    "{% if m['role'] == 'assistant' %}<think></think>{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_assistant|>assistant<|im_middle|>{% endif %}"
)
_STRICT_HISTORY_TPL = (
    "{% for m in messages %}{% if m['role'] == 'tool' %}{{ raise_exception('no tool turns') }}"
    "{% endif %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAE0lEQVR4nGM8ISfHAANMcBZeDgA0dgEMydTl/QAAAABJRU5ErkJggg=="
_ETHINK = {"reasoning_style": "enable_thinking", "supports_reasoning": True}
_ETHINK_EFFORT = {"reasoning_style": "enable_thinking_effort", "supports_reasoning": True}


def test_prefill_mode_on_for_enable_thinking_default():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _THINK_TPL) is True


def test_prefill_mode_follows_template_default_not_the_request_flag():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _TEMPLATE_DEFAULT_OFF_TPL) is False
    assert _sf_reasoning_prefill_mode(_ETHINK, True, _TEMPLATE_DEFAULT_OFF_TPL) is True


def test_prefill_mode_off_when_thinking_disabled():
    assert _sf_reasoning_prefill_mode(_ETHINK, False, _THINK_TPL) is False


def test_prefill_mode_off_for_reasoning_effort_none():
    assert (
        _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _THINK_TPL, reasoning_effort = "none")
        is False
    )
    assert (
        _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _THINK_TPL, reasoning_effort = "high")
        is True
    )


def test_prefill_mode_renders_with_the_requested_reasoning_effort():
    assert _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _EFFORT_SHAPE_TPL, "high") is True
    assert _sf_reasoning_prefill_mode(_ETHINK_EFFORT, None, _EFFORT_SHAPE_TPL, "low") is False


def test_a_template_that_stamps_a_date_still_renders():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _STRFTIME_TPL) is False


def test_prefill_mode_off_without_think_markers():
    assert _sf_reasoning_prefill_mode(_ETHINK, None, "no markers here") is False


def test_prefill_mode_renders_the_request_messages():
    plain = [{"role": "user", "content": "what is 2+2"}]
    opted_in = [{"role": "user", "content": "/think what is 2+2"}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, plain) is False
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, opted_in) is True


def test_messages_the_template_refuses_fall_back_to_the_single_user_probe():
    refused = [{"role": "user", "content": "hi"}, {"role": "tool", "content": "42"}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _STRICT_HISTORY_TPL, None, refused) is False
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _STRICT_HISTORY_TPL) is False


def test_a_think_tag_the_user_typed_does_not_prefill():
    for text in ("how do I emit a <think> tag?", "use <think>x</think> tags", "plain"):
        msgs = [{"role": "user", "content": text}]
        assert _sf_reasoning_prefill_mode(_ETHINK, None, _HISTORY_ONLY_TPL, None, msgs) is False
    typed = [{"role": "user", "content": "a <think> b"}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _THINK_TPL, None, typed) is True


def test_content_parts_must_reach_the_probe_flattened():
    flattened = [{"role": "user", "content": "/think what is 2+2"}]
    raw_parts = [{"role": "user", "content": [{"type": "text", "text": "/think what is 2+2"}]}]
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, flattened) is True
    assert _sf_reasoning_prefill_mode(_ETHINK, None, _MESSAGE_SHAPE_TPL, None, raw_parts) is False


def test_control_markup_must_reach_the_probe_swept():
    from core.inference.chat_template_helpers import neutralize_control_markup_in_messages

    raw = [{"role": "user", "content": "<think> tags"}]
    swept = neutralize_control_markup_in_messages([dict(m) for m in raw])
    assert swept[0]["content"] == "< think> tags"
    marker_tpl = _MESSAGE_SHAPE_TPL.replace("'/think' in", "'<think>' in")
    assert _sf_reasoning_prefill_mode(_ETHINK, None, marker_tpl, None, raw) is True
    assert _sf_reasoning_prefill_mode(_ETHINK, None, marker_tpl, None, swept) is False


def test_prefill_mode_without_messages_matches_the_single_user_probe():
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
    order: list[str] = []

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
    events = [
        {"type": "content", "text": "I should search"},
        {"type": "content", "text": "I should search Sydney weather</think>"},
        {"type": "tool_start", "tool_name": "web_search", "tool_call_id": "c0"},
        {"type": "tool_end", "tool_name": "web_search", "tool_call_id": "c0"},
        {"type": "status", "text": ""},
        {"type": "content", "text": "Found it</think>Sydney is 21C today."},
    ]
    out = _replay_sf_reasoning_stream(events, prefilled = True)
    assert "I should search Sydney weather" in out["reasoning"]
    assert "Found it" in out["reasoning"]
    assert out["visible"] == "Sydney is 21C today."
    assert out["monitor"] == "Sydney is 21C today."
    assert out["order"].index("reasoning") < out["order"].index("tool_start")


def test_s3_extractor_resets_each_turn():
    events = [
        {"type": "content", "text": "turn1 thoughts</think>partial"},
        {"type": "status", "text": ""},
        {"type": "content", "text": "turn2 thoughts</think>final answer"},
    ]
    out = _replay_sf_reasoning_stream(events, prefilled = True)
    assert out["reasoning"] == "turn1 thoughtsturn2 thoughts"
    assert out["visible"] == "partialfinal answer"


def test_s4_harmony_full_tags_normal_mode():
    events = [{"type": "content", "text": "<think>reasoning here</think>visible answer"}]
    out = _replay_sf_reasoning_stream(events, prefilled = False)
    assert out["reasoning"] == "reasoning here"
    assert out["visible"] == "visible answer"


def test_s5_thinking_off_no_reasoning_deltas():
    events = [{"type": "content", "text": "Just the plain answer, no thinking."}]
    out = _replay_sf_reasoning_stream(events, prefilled = False)
    assert out["reasoning"] == ""
    assert out["visible"] == "Just the plain answer, no thinking."
    assert out["monitor"] == "Just the plain answer, no thinking."


def test_s6_reasoning_effort_none_disables_prefill_for_enable_thinking_effort():
    feats = {"reasoning_style": "enable_thinking_effort", "supports_reasoning": True}
    assert _sf_reasoning_prefill_mode(feats, None, _THINK_TPL, "none") is False
    assert _sf_reasoning_prefill_mode(feats, None, _THINK_TPL, "high") is True
    assert _sf_reasoning_prefill_mode(feats, None, _THINK_TPL, None) is True
    assert _sf_reasoning_prefill_mode(feats, False, _THINK_TPL, "high") is False
    always = {**feats, "reasoning_always_on": True}
    assert _sf_reasoning_prefill_mode(always, None, _THINK_TPL, "none") is True
    plain = {"reasoning_style": "enable_thinking", "supports_reasoning": True}
    assert _sf_reasoning_prefill_mode(plain, None, _THINK_TPL, "none") is True

    events = [{"type": "content", "text": "The capital of France is Paris."}]
    out = _replay_sf_reasoning_stream(events, prefilled = False)
    assert out["visible"] == "The capital of France is Paris."
    assert out["reasoning"] == ""
    swallowed = _replay_sf_reasoning_stream(events, prefilled = True)
    assert swallowed["visible"] == ""
    assert swallowed["reasoning"] == "The capital of France is Paris."


def test_native_reasoning_streamer_selected_and_errors_raise():
    inf, pytest, threading, torch = _shared_setup_1()

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
    inf, pytest, threading, torch = _shared_setup_1()

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
    inf, pytest, threading, torch = _shared_setup_1()

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


def _sf_route_message(
    monkeypatch,
    template,
    snapshots,
    is_vision = False,
    is_mlx = False,
    features = None,
    model_info = None,
    seen = None,
    status = 200,
    **body,
):
    """POST a safetensors chat completion and return the assistant message, joined if streamed."""
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
        models = {
            "qwen": {
                "chat_template_info": {"template": template},
                "is_vision": is_vision,
                "is_mlx": is_mlx,
                **(model_info or {}),
            }
        }

        def generate_chat_response(self, **kwargs):
            if seen is not None:
                seen.update(kwargs)
            yield from snapshots

        def reset_generation_state(self, *_args):
            return None

        def resize_image(self, image):
            return image

    monkeypatch.setattr(
        inference_route,
        "_detect_safetensors_features",
        lambda backend, chat_template, tools = None: dict(features or _ETHINK, supports_tools = False),
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
    assert resp.status_code == status, resp.text
    if status != 200:
        return resp.json()["error"]
    if not body.get("stream"):
        return resp.json()["choices"][0]["message"]
    message = {"content": "", "reasoning_content": ""}
    for line in resp.text.splitlines():
        if line.startswith("data: {"):
            for choice in json.loads(line[6:]).get("choices") or []:
                for key in message:
                    message[key] += (choice.get("delta") or {}).get(key) or ""
    return message


@pytest.mark.parametrize("stream", [False, True], ids = ["json", "sse"])
def test_route_resumes_an_mlx_thought_as_reasoning(monkeypatch, stream):
    seen = {}
    message = _sf_route_message(
        monkeypatch,
        _TEMPLATE_DEFAULT_OFF_TPL,
        ["<think>tail", "<think>tail.</think>\n\nanswer"],
        is_mlx = True,
        seen = seen,
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "The head, then the "},
        ],
        continue_final_message = True,
        enable_thinking = False,
        stream = stream,
    )
    assert seen["continue_final_message"] is True
    assert message["reasoning_content"] == "tail."
    assert message["content"] == "\n\nanswer"


def test_route_refuses_response_format_on_an_mlx_thought_resume(monkeypatch):
    error = _sf_route_message(
        monkeypatch,
        _TEMPLATE_DEFAULT_OFF_TPL,
        [],
        is_mlx = True,
        status = 400,
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "The head"},
        ],
        continue_final_message = True,
        response_format = {"type": "json_object"},
    )
    assert error["param"] == "response_format"


def test_route_returns_the_answer_as_content_when_the_template_closes_its_block(monkeypatch):
    """The user-visible symptom: a plain answer reaching ``content``, not the thinking drawer."""
    snapshots = ["The capital", "The capital of Japan is Tokyo."]
    message = _sf_route_message(monkeypatch, _TEMPLATE_DEFAULT_OFF_TPL, snapshots)
    assert message["content"] == "The capital of Japan is Tokyo."
    assert not message["reasoning_content"]


def test_route_keeps_literal_think_text_when_the_request_turns_thinking_off(monkeypatch):
    answer = "Use <think>hi</think> in your prompt."
    message = _sf_route_message(
        monkeypatch,
        _TEMPLATE_DEFAULT_OFF_TPL,
        [answer],
        enable_thinking = False,
    )
    assert message["content"] == answer
    assert not message["reasoning_content"]


def test_route_still_splits_real_thinking_when_the_request_leaves_it_on(monkeypatch):
    message = _sf_route_message(
        monkeypatch,
        _TEMPLATE_DEFAULT_OFF_TPL,
        ["<think>plan</think>answer"],
        enable_thinking = True,
    )
    assert message["content"] == "answer"
    assert message["reasoning_content"] == "plan"


def test_route_keeps_literal_think_text_on_a_transformers_image_turn(monkeypatch):
    answer = "Use <think>hi</think> in your prompt."
    message = _sf_route_message(
        monkeypatch,
        _TEMPLATE_DEFAULT_OFF_TPL,
        [answer],
        is_vision = True,
        enable_thinking = False,
        image_base64 = _TINY_PNG_B64,
    )
    assert message["content"] == answer
    assert not message["reasoning_content"]


def test_route_keeps_literal_think_text_on_an_mlx_image_turn(monkeypatch):
    """Effort alone, so a guard that drops it cannot pass on the boolean."""
    answer = "Use <think>hi</think> in your prompt."
    message = _sf_route_message(
        monkeypatch,
        _EFFORT_SHAPE_TPL,
        [answer],
        features = _ETHINK_EFFORT,
        is_vision = True,
        is_mlx = True,
        reasoning_effort = "none",
        image_base64 = _TINY_PNG_B64,
    )
    assert message["content"] == answer
    assert not message["reasoning_content"]


_THINK_OFF_MESSAGES = [
    {"role": "system", "content": "You are helpful. <|think_off|>"},
    {"role": "user", "content": "What is the capital of France?"},
]


@pytest.mark.parametrize(
    "reason, content, reasoning",
    [
        (None, "Paris.", ""),
        ("it could not render a conversation", "", "Paris."),
    ],
)
def test_route_probes_the_applied_mlx_override_not_the_shipped_template(
    monkeypatch, reason, content, reasoning
):
    """An override closing the block on an inline tag must not be read through the shipped one."""
    message = _sf_route_message(
        monkeypatch,
        _THINK_TPL,
        ["Paris."],
        is_mlx = True,
        model_info = {
            "chat_template_override_requested": _THINK_OFF_TAG_TPL,
            "chat_template_override_reason": reason,
        },
        messages = _THINK_OFF_MESSAGES,
    )
    assert (message["content"] or "") == content
    assert (message["reasoning_content"] or "") == reasoning


def test_route_carries_the_effort_field_into_the_gate(monkeypatch):
    answer = "Use <think>hi</think> in your prompt."
    message = _sf_route_message(
        monkeypatch,
        _EFFORT_SHAPE_TPL,
        [answer],
        features = _ETHINK_EFFORT,
        reasoning_effort = "none",
    )
    assert message["content"] == answer
    assert not message["reasoning_content"]


def test_parse_think_markers_gates_on_capability_and_request():
    assert _sf_parse_think_markers(_ETHINK) is True
    assert _sf_parse_think_markers(_ETHINK, True) is True
    assert _sf_parse_think_markers(_ETHINK, False) is False
    assert _sf_parse_think_markers(dict(_ETHINK, reasoning_always_on = True), False) is True
    assert _sf_parse_think_markers({"supports_reasoning": False}, True) is False


def test_parse_think_markers_reads_only_the_dial_the_template_branches_on():
    assert _sf_parse_think_markers(_ETHINK, None, "none") is True
    _EFFORT_ONLY = {"reasoning_style": "reasoning_effort", "supports_reasoning": True}
    assert _sf_parse_think_markers(_EFFORT_ONLY, False, None) is True
    assert _sf_parse_think_markers(_EFFORT_ONLY, None, "none") is False
    assert _sf_parse_think_markers(_EFFORT_ONLY, None, "low") is True
    assert _sf_parse_think_markers(_ETHINK_EFFORT, True, "none") is True
    assert _sf_parse_think_markers(_ETHINK_EFFORT, False, "high") is False
    assert _sf_parse_think_markers(_ETHINK_EFFORT, None, "none") is False
    assert _sf_parse_think_markers(_ETHINK_EFFORT, None, "high") is True
