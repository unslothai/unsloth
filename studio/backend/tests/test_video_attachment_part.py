# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video attachments ride the message list as llama-server's `input_video` part.

llama.cpp takes video through its OpenAI-compatible chat endpoint as
``{"type": "input_video", "input_video": {"data": ...}}`` (tools/server/
server-common.cpp), refusing it unless the projector, the build and ffmpeg all
line up -- which it reports at ``/props`` under ``modalities.video``. These tests
pin the wire shape and that capability read, since neither is visible from the
GGUF alone.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from routes.inference import _inject_video_part  # noqa: E402


def test_a_video_part_is_appended_to_the_last_user_message():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "what happens here?"}]},
    ]
    _inject_video_part(messages, "AAAA")
    assert messages[1]["content"][-1] == {"type": "input_video", "input_video": {"data": "AAAA"}}
    # The system message is untouched.
    assert messages[0]["content"] == "be brief"


def test_a_string_content_turn_is_promoted_to_parts():
    messages = [{"role": "user", "content": "describe the clip"}]
    _inject_video_part(messages, "BBBB")
    assert messages[0]["content"] == [
        {"type": "text", "text": "describe the clip"},
        {"type": "input_video", "input_video": {"data": "BBBB"}},
    ]


def test_only_the_newest_user_turn_carries_the_clip():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    _inject_video_part(messages, "CCCC")
    assert messages[0]["content"] == "first"
    assert messages[2]["content"][-1]["type"] == "input_video"


def test_a_turn_with_no_user_message_is_left_alone():
    messages = [{"role": "assistant", "content": "hello"}]
    _inject_video_part(messages, "DDDD")
    assert messages == [{"role": "assistant", "content": "hello"}]


def test_video_capability_is_read_from_the_server_props():
    """Only llama-server knows: the mmproj, MTMD_VIDEO and ffmpeg all have a vote."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._has_video_input = False
    backend._query_server_props = lambda: {
        "default_generation_settings": {"n_ctx": 4096},
        "modalities": {"vision": True, "video": True, "audio": False},
    }
    assert backend._query_server_n_ctx() == 4096
    assert backend._has_video_input is True


def test_a_server_without_video_leaves_the_capability_off():
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._has_video_input = True
    backend._query_server_props = lambda: {
        "default_generation_settings": {"n_ctx": 2048},
        "modalities": {"vision": True, "video": False, "audio": False},
    }
    backend._query_server_n_ctx()
    assert backend._has_video_input is False


def test_an_unreadable_props_does_not_claim_video():
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._has_video_input = False
    backend._query_server_props = lambda: None
    assert backend._query_server_n_ctx() is None
    assert backend._has_video_input is False
