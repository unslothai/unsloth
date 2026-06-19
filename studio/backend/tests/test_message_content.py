# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`content_to_text` normalizes OpenAI-style message content to plain text.

Regression for issue #4383: vision / audio / manual chat-template formatting
paths called string ops (``.strip()``, ``re.sub``, f-string interpolation) on
``content`` directly. When a client sends the OpenAI multimodal *list* form
``[{"type": "text", "text": "hi"}, {"type": "image_url", ...}]`` those paths
raised ``'list' object has no attribute 'replace'`` (or rendered the list repr
into the prompt). ``content_to_text`` collapses either shape to text.

Loaded by file path so the test does not import ``core.inference`` (whose
``__init__`` eagerly pulls in the orchestrator + llama_cpp / torch).
"""

import importlib.util
from pathlib import Path


_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_message_content():
    path = _BACKEND_DIR / "core/inference/message_content.py"
    spec = importlib.util.spec_from_file_location("message_content_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_string_is_returned_unchanged():
    mc = _load_message_content()
    assert mc.content_to_text("hello world") == "hello world"
    assert mc.content_to_text("") == ""


def test_none_becomes_empty_string():
    mc = _load_message_content()
    assert mc.content_to_text(None) == ""


def test_single_text_part_list():
    mc = _load_message_content()
    content = [{"type": "text", "text": "hello"}]
    assert mc.content_to_text(content) == "hello"


def test_multimodal_list_drops_non_text_parts():
    mc = _load_message_content()
    content = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    # The image part is dropped; only the text survives.
    assert mc.content_to_text(content) == "describe this"


def test_multiple_text_parts_joined_with_space():
    mc = _load_message_content()
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert mc.content_to_text(content) == "first second"


def test_bare_string_items_in_list():
    mc = _load_message_content()
    assert mc.content_to_text(["a", "b"]) == "a b"


def test_audio_and_image_only_list_is_empty():
    mc = _load_message_content()
    content = [
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "input_audio", "input_audio": {"data": "y", "format": "wav"}},
    ]
    assert mc.content_to_text(content) == ""


def test_part_without_type_treated_as_text():
    mc = _load_message_content()
    # A dict carrying a ``text`` field but no explicit ``type`` is treated as text.
    assert mc.content_to_text([{"text": "untyped"}]) == "untyped"


def test_empty_text_parts_skipped():
    mc = _load_message_content()
    content = [
        {"type": "text", "text": ""},
        {"type": "text", "text": "kept"},
    ]
    assert mc.content_to_text(content) == "kept"


def test_tuple_behaves_like_list():
    mc = _load_message_content()
    content = ({"type": "text", "text": "x"}, {"type": "text", "text": "y"})
    assert mc.content_to_text(content) == "x y"


def test_result_supports_string_ops():
    mc = _load_message_content()
    # The crux of #4383: the return value must be a plain str so callers can
    # safely call .strip()/.replace() on it.
    out = mc.content_to_text([{"type": "text", "text": "  padded  "}])
    assert out.strip() == "padded"
    assert isinstance(out, str)
