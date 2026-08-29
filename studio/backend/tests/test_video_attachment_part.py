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

import base64
from pathlib import Path

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


def test_the_cap_admits_a_clip_of_exactly_the_composer_limit():
    """Flooring the 4/3 inflation refused a file of exactly the allowed size."""
    import math

    from routes.inference import _MAX_VIDEO_B64_CHARS

    limit_bytes = 64 * 1024 * 1024
    # Padded base64 is 4 characters per 3 bytes, rounded up.
    assert len(base64.b64encode(b"x" * 3001)) == 4 * math.ceil(3001 / 3)
    assert 4 * math.ceil(limit_bytes / 3) <= _MAX_VIDEO_B64_CHARS
    assert 4 * math.ceil((limit_bytes + 1024) / 3) > _MAX_VIDEO_B64_CHARS


def _inference_source() -> str:
    return (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )


def test_video_is_refused_on_the_tool_passthrough_path():
    """That branch forwards an explicit field list and returns before the
    injection below, so the clip would be dropped and the model would answer
    without it. The audio path already refuses; video has to match."""
    source = _inference_source()
    start = source.index("if using_gguf and _takes_tool_passthrough(payload, llama_backend):")
    branch = source[start : start + 2500]
    assert "payload.audio_base64" in branch
    assert "payload.video_base64" in branch
    assert "Video input is not supported together with guided decoding" in branch


def test_the_size_check_runs_before_the_automatic_switch():
    """A cheap length check must not cost a model load first: an oversized clip
    would otherwise evict a working model and 413 only afterwards."""
    source = _inference_source()
    # Anchor inside the chat-completions handler; other routes switch too.
    handler = source.index("_needs_image = bool(_pre_parsed[2])")
    guard = source.index("_video_b64_rejection(payload.video_base64)", handler)
    switch = source.index("await _maybe_auto_switch_model(", handler)
    assert guard < switch


def test_video_joins_the_projector_requirement_before_switching():
    """Video rides the same companion mmproj as vision, so a text-only target
    cannot serve it either. Audio already votes here."""
    source = _inference_source()
    start = source.index("_needs_image = bool(_pre_parsed[2])")
    # 600, not 400: the block gained explanatory comments and a widened
    # _needs_image derivation, which pushed the modality votes past the old
    # window while leaving the wiring itself intact.
    block = source[start : start + 600]
    assert "payload.audio_base64" in block
    assert "payload.video_base64" in block


def test_an_external_provider_refuses_video_rather_than_ignoring_it():
    """input_video is llama.cpp's own part type, so the proxy has nowhere to put
    the clip and returns before any video handling below."""
    source = _inference_source()
    start = source.index("if payload.provider_id or payload.provider_type:")
    branch = source[start : source.index("_proxy_to_external_provider(payload", start)]
    assert "payload.video_base64" in branch
    assert "Video input is only supported on a local GGUF model" in branch


def test_a_non_gguf_model_refuses_video_rather_than_ignoring_it():
    """Injection lives in the GGUF branch, so a transformers model would answer
    as if nothing were attached."""
    source = _inference_source()
    assert "if payload.video_base64 and not using_gguf:" in source


def test_token_counting_refuses_video_like_image_and_audio():
    """The completion injects the clip; this route cannot, so counting here
    would silently undercount the turn."""
    source = _inference_source()
    start = source.index("Cannot count tokens for messages containing images.")
    block = source[start : start + 700]
    assert "Cannot count tokens for messages containing audio." in block
    assert "Cannot count tokens for messages containing video." in block


def test_both_video_checks_share_one_rule():
    """Two size checks that drift let the pre-switch one pass what the post-load
    one refuses, which is the model load this was meant to avoid."""
    source = _inference_source()
    assert source.count("_video_b64_rejection(payload.video_base64)") == 2
