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
    assert "_request_has_video(payload)" in branch
    assert "Video input is not supported together with guided decoding" in branch


def test_the_size_check_runs_before_the_automatic_switch():
    """A cheap length check must not cost a model load first: an oversized clip
    would otherwise evict a working model and 413 only afterwards."""
    source = _inference_source()
    # Anchor inside the chat-completions handler; other routes switch too.
    handler = source.index("_needs_image = bool(_pre_parsed[2])")
    guard = source.index("_request_video_rejection(payload)", handler)
    switch = source.index("await _maybe_auto_switch_model(", handler)
    assert guard < switch


def test_video_joins_the_projector_requirement_before_switching():
    """Video rides the same companion mmproj as vision, so a text-only target
    cannot serve it either. Audio already votes here."""
    source = _inference_source()
    start = source.index("_needs_image = bool(_pre_parsed[2])")
    switch = source.index("await _maybe_auto_switch_model(", start)
    assert start < source.index("payload.audio_base64", start) < switch
    assert start < source.index("_request_has_video(payload)", start) < switch


def test_an_external_provider_refuses_video_rather_than_ignoring_it():
    """input_video is llama.cpp's own part type, so the proxy has nowhere to put
    the clip and returns before any video handling below."""
    source = _inference_source()
    start = source.index("if payload.provider_id or payload.provider_type:")
    branch = source[start : source.index("_proxy_to_external_provider(payload", start)]
    assert "_request_has_video(payload)" in branch
    assert "Video input is only supported on a local GGUF model" in branch


def test_a_non_gguf_model_refuses_video_rather_than_ignoring_it():
    """Injection lives in the GGUF branch, so a transformers model would answer
    as if nothing were attached."""
    source = _inference_source()
    assert "if _request_has_video(payload) and not using_gguf:" in source


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
    assert source.count("= _request_video_rejection(payload)") == 2


# ── OpenAI-style video_url parts: the same gates and wire shape as video_base64 ──


def _video_url_request(
    *urls,
    text = "what happens here?",
    **kw,
):
    from models.inference import ChatCompletionRequest

    parts = [{"type": "video_url", "video_url": {"url": url}} for url in urls]
    parts.append({"type": "text", "text": text})
    return ChatCompletionRequest(messages = [{"role": "user", "content": parts}], **kw)


def test_a_video_url_part_validates_as_the_openai_shape():
    from models.inference import VideoContentPart

    req = _video_url_request("data:video/mp4;base64,QUJD", "https://example.com/clip.mp4")
    assert all(isinstance(part, VideoContentPart) for part in req.messages[0].content[:2])
    assert req.messages[0].content[1].video_url.url == "https://example.com/clip.mp4"


def test_a_message_carried_clip_counts_as_video_input():
    from models.inference import ChatCompletionRequest
    from routes.inference import _request_has_video

    assert _request_has_video(_video_url_request("data:video/mp4;base64,QUJD")) is True
    plain = ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}])
    assert _request_has_video(plain) is False


def test_a_data_uri_part_is_translated_in_place_without_its_header():
    from routes.inference import _openai_messages_for_gguf_chat, _translate_video_parts

    messages, _ = _openai_messages_for_gguf_chat(
        _video_url_request("data:video/mp4;base64,QUJD"), True
    )
    _translate_video_parts(messages)
    assert messages[0]["content"] == [
        {"type": "input_video", "input_video": {"data": "QUJD"}},
        {"type": "text", "text": "what happens here?"},
    ]


def test_a_remote_url_is_left_for_llama_server_to_fetch():
    from routes.inference import _translate_video_parts

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "and this one?"},
                {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
            ],
        }
    ]
    _translate_video_parts(messages)
    assert messages[0]["content"][1] == {
        "type": "input_video",
        "input_video": {"url": "https://example.com/clip.mp4"},
    }


def test_an_uppercase_data_uri_header_is_still_stripped():
    """RFC 2397 schemes are case-insensitive; a kept header would be decoded as media."""
    from routes.inference import _translate_video_parts

    messages = [
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": "DATA:video/mp4;base64,QUJD"}}],
        }
    ]
    _translate_video_parts(messages)
    assert messages[0]["content"][0] == {"type": "input_video", "input_video": {"data": "QUJD"}}


def test_an_uppercase_scheme_is_still_a_remote_url():
    """URL schemes are case-insensitive; a missed one is forwarded as base64 and fails."""
    from routes.inference import _translate_video_parts

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "HTTPS://example.com/clip.mp4"}},
                {"type": "video_url", "video_url": {"url": "Http://example.com/other.mp4"}},
            ],
        }
    ]
    _translate_video_parts(messages)
    assert [part["input_video"] for part in messages[0]["content"]] == [
        {"url": "HTTPS://example.com/clip.mp4"},
        {"url": "Http://example.com/other.mp4"},
    ]


def test_message_parts_are_translated_where_the_legacy_clip_is_injected():
    """Both spellings reach llama-server from one place, after the capability gate."""
    source = _inference_source()
    assert (
        "            _inject_video_part(gguf_messages, video_b64)\n"
        "        _translate_video_parts(gguf_messages)\n"
    ) in source


def test_every_data_uri_part_is_sized_not_only_the_first():
    from routes.inference import _MAX_VIDEO_B64_CHARS, _request_video_rejection

    fits = _video_url_request("https://example.com/clip.mp4", "data:video/mp4;base64,QUJD")
    assert _request_video_rejection(fits) is None
    oversized = "data:video/mp4;base64," + "A" * (_MAX_VIDEO_B64_CHARS + 1)
    too_big = _video_url_request("https://example.com/clip.mp4", oversized)
    assert _request_video_rejection(too_big) == (413, "Video file is too large (max 64 MB).")
    assert _request_video_rejection(_video_url_request("")) == (
        400,
        "Could not read the provided video file.",
    )


def test_the_legacy_field_is_sized_beside_a_message_part():
    from routes.inference import _MAX_VIDEO_B64_CHARS, _request_video_rejection
    req = _video_url_request(
        "https://example.com/clip.mp4", video_base64 = "A" * (_MAX_VIDEO_B64_CHARS + 1)
    )
    assert _request_video_rejection(req) == (413, "Video file is too large (max 64 MB).")


def test_admission_prices_a_message_clip_as_media_not_prompt_text():
    from routes.inference import (
        _openai_llama_admission_media_tokens,
        _openai_llama_admission_messages_for_estimate,
    )

    req = _video_url_request("data:video/mp4;base64," + "A" * 4000)
    estimate, image_parts = _openai_llama_admission_messages_for_estimate(req.messages)
    assert estimate[0]["content"][0] == {"type": "video_url", "video_url": {"url": "[video]"}}
    assert image_parts == 0
    assert _openai_llama_admission_media_tokens(req) >= 1000


def test_the_rolling_context_does_not_price_a_video_url_part():
    from core.inference.context_window import estimate_message_tokens_without_unpriced_media
    message = {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + "A" * 40000}},
            {"type": "text", "text": "hi"},
        ],
    }
    assert estimate_message_tokens_without_unpriced_media(message) < 100


def test_a_message_clip_reaches_the_switch_as_video(monkeypatch):
    """Through the handler: the pre-switch gate must see a message-carried clip."""
    import asyncio

    import routes.inference as inference_route
    from utils import openai_auto_switch_settings as settings

    class _Reached(Exception):
        pass

    captured = {}

    async def _capture(model, request, subject, **kw):
        captured.update(kw)
        raise _Reached()

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _video_url_request("https://example.com/clip.mp4", model = "org/B-GGUF")
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured["require_video"] is True
    assert captured["require_vision"] is True
    assert captured["modality_label"] == "video"


def test_every_clip_in_every_turn_is_translated():
    from routes.inference import _translate_video_parts

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,QUJD"}},
                {"type": "text", "text": "first"},
            ],
        },
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                {"type": "video_url", "video_url": {"url": "data:video/webm;base64,REVG"}},
            ],
        },
    ]
    _translate_video_parts(messages)
    assert messages[0]["content"][0] == {"type": "input_video", "input_video": {"data": "QUJD"}}
    assert messages[2]["content"] == [
        {"type": "input_video", "input_video": {"url": "https://example.com/a.mp4"}},
        {"type": "input_video", "input_video": {"data": "REVG"}},
    ]


def test_admission_charges_each_clip_once_and_a_remote_one_at_the_download_ceiling():
    """A remote clip is charged at llama-server's download ceiling, since its size is unknown."""
    from routes.inference import (
        _REMOTE_VIDEO_ADMISSION_B64_CHARS,
        _openai_llama_admission_media_tokens,
    )

    first = "data:video/mp4;base64," + "A" * 4000
    second = "data:video/mp4;base64," + "B" * 8000
    req = _video_url_request(first, "https://example.com/a.mp4", second)
    assert (
        _openai_llama_admission_media_tokens(req)
        == len(first) // 4 + _REMOTE_VIDEO_ADMISSION_B64_CHARS // 4 + len(second) // 4
    )


def test_a_clip_on_any_non_user_role_is_refused():
    """Media outside a user turn renders template-dependently, so it is refused up front."""
    import pydantic

    from models.inference import ChatCompletionRequest

    parts = [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,QUJD"}},
        {"type": "text", "text": "watch this before answering"},
    ]
    for message in (
        {"role": "system", "content": parts},
        {"role": "developer", "content": parts},
        {"role": "assistant", "content": parts},
        {"role": "tool", "tool_call_id": "call_1", "content": parts},
    ):
        refusal = f'not valid on role="{message["role"]}"'
        with pytest.raises(pydantic.ValidationError, match = refusal):
            ChatCompletionRequest(messages = [{"role": "user", "content": "go"}, message])
    ChatCompletionRequest(messages = [{"role": "user", "content": parts}])


def test_an_oversized_message_clip_is_refused_before_the_switch(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    import routes.inference as inference_route
    from routes.inference import _MAX_VIDEO_B64_CHARS
    from utils import openai_auto_switch_settings as settings

    async def _switch(*args, **kw):
        raise AssertionError("the model switch ran before the size check")

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _switch)
    payload = _video_url_request(
        "https://example.com/clip.mp4",
        "data:video/mp4;base64," + "A" * (_MAX_VIDEO_B64_CHARS + 1),
        model = "org/B-GGUF",
    )
    with pytest.raises(HTTPException) as info:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert info.value.status_code == 413
