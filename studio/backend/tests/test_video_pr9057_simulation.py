# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PR 9057 review simulation: every axis a video attachment can travel.

Not part of the PR. Written during review to answer "does this break anything,
and does the fix actually work", covering: the common no-video-capability model,
a swap from a capable model to a non-capable one, the external-provider and
non-GGUF passthroughs, an oversized clip, a data-URI wrapper, an llama.cpp build
too old to declare modalities at all, and the shape of the runtime fields old
clients read.
"""

from __future__ import annotations

import base64
import math

import pytest

pytest.importorskip("torch")

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from models.inference import (  # noqa: E402
    ChatCompletionRequest,
    InferenceStatusResponse,
    _InferenceRuntimeFields,
)
from routes.inference import (  # noqa: E402
    _MAX_VIDEO_B64_CHARS,
    _inject_video_part,
    _video_b64_rejection,
)

LIMIT = 64 * 1024 * 1024


def _props_backend(props):
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._has_video_input = False
    b._query_server_props = lambda: props
    return b


# --------------------------------------------------------------------------
# A. the base64 ceiling, measured against the real encoder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 1023, 1024, 3000, 3001, 3002])
def test_the_ceiling_formula_matches_what_base64_actually_produces(n):
    assert len(base64.b64encode(b"x" * n)) == 4 * math.ceil(n / 3)


def test_a_clip_of_exactly_the_composer_limit_is_admitted():
    # 67108864 = 3*22369621 + 1, so the last byte costs a full padded quad.
    assert _MAX_VIDEO_B64_CHARS == 89478488
    assert 4 * math.ceil(LIMIT / 3) == 89478488
    exact = "A" * 89478488
    assert _video_b64_rejection(exact)[1] is None


def test_the_old_floor_expression_would_have_refused_it():
    # What the review flagged: floor(64MiB * 4 / 3) == 89478485, three characters
    # short, so the largest file the composer offers 413s.
    floored = (LIMIT * 4) // 3
    assert floored == 89478485
    assert floored < 4 * math.ceil(LIMIT / 3)


def test_one_character_over_the_ceiling_is_refused_413():
    assert _video_b64_rejection("A" * (_MAX_VIDEO_B64_CHARS + 1))[1] == (
        413,
        "Video file is too large (max 64 MB).",
    )


def test_the_data_uri_header_is_not_counted_against_the_cap():
    # A composer that sends a data URI must not lose bytes to its own header.
    payload = "A" * _MAX_VIDEO_B64_CHARS
    stripped, rejection = _video_b64_rejection(f"data:video/mp4;base64,{payload}")
    assert rejection is None
    assert stripped == payload


@pytest.mark.parametrize(
    "mime",
    ["video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/x-msvideo"],
)
def test_every_container_the_composer_accepts_survives_the_data_uri_strip(mime):
    stripped, rejection = _video_b64_rejection(f"data:{mime};base64,QUJD")
    assert rejection is None and stripped == "QUJD"


@pytest.mark.parametrize("bad", ["", "data:", "data:video/mp4;base64", "data:,"])
def test_an_unreadable_payload_is_a_400_not_a_crash(bad):
    stripped, rejection = _video_b64_rejection(bad)
    assert rejection == (400, "Could not read the provided video file.")


def test_a_bare_payload_with_no_header_is_passed_through_untouched():
    assert _video_b64_rejection("QUJD") == ("QUJD", None)


# --------------------------------------------------------------------------
# B. capability read: old builds, odd payloads, and swaps
# --------------------------------------------------------------------------


def test_a_build_too_old_to_declare_modalities_reports_no_video():
    """The key backwards-compat case: llama.cpp only grew `modalities` in /props
    recently, and every older build simply omits the key."""
    b = _props_backend({"default_generation_settings": {"n_ctx": 4096}})
    assert b._query_server_n_ctx() == 4096
    assert b._has_video_input is False


@pytest.mark.parametrize(
    "props",
    [
        {"modalities": None},
        {"modalities": []},
        {"modalities": "vision"},
        {"modalities": {"vision": True}},
        {"modalities": {"video": None}},
        {"modalities": {"video": 0}},
        {"modalities": {"video": "false"}},  # a non-empty string is truthy: see below
        {},
    ],
)
def test_a_malformed_modalities_block_never_crashes_the_context_readback(props):
    b = _props_backend({**props, "default_generation_settings": {"n_ctx": 2048}})
    assert b._query_server_n_ctx() == 2048
    assert isinstance(b._has_video_input, bool)


def test_only_a_real_json_true_turns_the_capability_on():
    for value, expected in ((True, True), (False, False), (None, False), (0, False)):
        b = _props_backend({"modalities": {"video": value}})
        b._query_server_n_ctx()
        assert b._has_video_input is expected, value


def test_a_swap_to_a_model_without_video_does_not_inherit_the_old_answer():
    """A stale True here would offer video on a model that cannot take it, and
    llama-server would refuse the completion after the upload."""
    b = _props_backend(
        {"modalities": {"video": True}, "default_generation_settings": {"n_ctx": 8192}}
    )
    b._query_server_n_ctx()
    assert b._has_video_input is True
    b._query_server_props = lambda: {
        "modalities": {"video": False},
        "default_generation_settings": {"n_ctx": 8192},
    }
    b._query_server_n_ctx()
    assert b._has_video_input is False


def test_an_unreachable_props_leaves_the_capability_off_rather_than_guessing():
    b = _props_backend(None)
    b._has_video_input = True
    assert b._query_server_n_ctx() is None
    # Nothing clears it here, which is why the load path clears it explicitly:
    assert "self._has_video_input = False" in _llama_cpp_source()


def _llama_cpp_source() -> str:
    from pathlib import Path
    return (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
    ).read_text(encoding = "utf-8")


def test_the_load_path_and_the_unload_path_both_clear_the_capability():
    src = _llama_cpp_source()
    assert src.count("self._has_video_input = False") == 2


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _stub_props_http(
    monkeypatch,
    resp,
    record = None,
):
    """Replace httpx.get inside the backend module and capture the call."""
    import core.inference.llama_cpp as llama_mod

    def _get(url, **kwargs):
        if record is not None:
            record.update(url = url, **kwargs)
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(llama_mod.httpx, "get", _get)


def _live_backend(api_key = None):
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._has_video_input = False
    b._api_key = api_key
    b._port = 9999
    b._host = "127.0.0.1"
    return b


@pytest.mark.parametrize("junk", [[1, 2], "props", 7, None, 3.5])
def test_a_props_body_that_is_not_an_object_is_rejected_not_crashed(monkeypatch, junk):
    """A proxy or a future build could answer with a list; the readback must
    degrade to "cannot tell", not raise into the load path."""
    b = _live_backend()
    _stub_props_http(monkeypatch, _Resp(200, junk))
    assert b._query_server_props() is None
    assert b._query_server_n_ctx() is None
    assert b._has_video_input is False


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
def test_a_non_200_props_never_claims_video(monkeypatch, status):
    b = _live_backend()
    _stub_props_http(monkeypatch, _Resp(status, {"modalities": {"video": True}}))
    assert b._query_server_props() is None
    assert b._has_video_input is False


def test_an_undecodable_props_body_is_swallowed(monkeypatch):
    b = _live_backend()
    _stub_props_http(monkeypatch, _Resp(200, ValueError("not json")))
    assert b._query_server_props() is None


def test_a_dead_server_is_swallowed(monkeypatch):
    b = _live_backend()
    _stub_props_http(monkeypatch, OSError("connection refused"))
    assert b._query_server_props() is None


def test_the_props_request_carries_the_child_api_key_when_direct_stream_set_one(monkeypatch):
    """llama-server's api-key middleware protects /props (it is not in the
    public_endpoints set), so an unauthenticated read 401s and the capability
    silently reads False under UNSLOTH_DIRECT_STREAM=1."""
    record: dict = {}
    b = _live_backend(api_key = "secret-token")
    _stub_props_http(monkeypatch, _Resp(200, {"modalities": {"video": True}}), record = record)
    b._query_server_props()
    assert record.get("headers") == {"Authorization": "Bearer secret-token"}


def test_the_props_request_sends_no_auth_header_when_there_is_no_child_key(monkeypatch):
    record: dict = {}
    b = _live_backend(api_key = None)
    _stub_props_http(monkeypatch, _Resp(200, {}), record = record)
    b._query_server_props()
    assert record.get("headers") is None


# --------------------------------------------------------------------------
# C. the wire shape old and new clients see
# --------------------------------------------------------------------------


def test_the_runtime_field_is_declared_and_defaults_off_for_every_non_gguf_model():
    """A transformers or MLX model never sets it, so the composer must read
    False and refuse video rather than offering it."""
    assert "has_video_input" in _InferenceRuntimeFields.model_fields
    assert InferenceStatusResponse().has_video_input is False


def test_the_generic_runtime_mapper_actually_picks_the_capability_up():
    """`_llama_runtime_fields` maps a response field to `_<name>` on the backend.
    If that mapping missed, `has_video_input` would be hardcoded False on the
    wire and the whole feature would be unreachable from the UI."""
    from routes.inference import _llama_runtime_fields

    class _Stub:
        pass

    backend = _Stub()
    for name in _InferenceRuntimeFields.model_fields:
        setattr(backend, f"_{name}", None)
    backend._has_video_input = True
    backend._has_audio_input = False
    for extra in (
        "requested_spec_mode",
        "requested_parallel_slots",
        "effective_parallel_slots",
        "requested_extra_args",
        "is_diffusion",
    ):
        setattr(backend, extra, None)
    fields = _llama_runtime_fields(backend)
    assert fields["has_video_input"] is True
    assert fields["has_audio_input"] is False


def test_an_old_client_that_sends_no_video_field_is_unaffected():
    req = ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}])
    assert req.video_base64 is None


def test_an_old_backend_would_ignore_the_new_field_rather_than_422():
    """`extra: allow`, so a newer desktop app talking to an older backend loses
    the clip silently instead of breaking every message. Worth knowing; not
    something this PR can fix from the new side."""
    assert ChatCompletionRequest.model_config["extra"] == "allow"


def test_the_field_round_trips_through_json_unchanged():
    payload = "data:video/mp4;base64,QUJD"
    req = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "video_base64": payload}
    )
    assert req.video_base64 == payload
    assert req.model_dump()["video_base64"] == payload


# --------------------------------------------------------------------------
# D. injection, on every message shape a real session produces
# --------------------------------------------------------------------------


def test_an_empty_message_list_is_a_no_op():
    messages: list[dict] = []
    _inject_video_part(messages, "AAAA")
    assert messages == []


def test_a_user_turn_with_a_none_content_is_promoted_without_losing_the_clip():
    messages = [{"role": "user", "content": None}]
    _inject_video_part(messages, "AAAA")
    assert messages[0]["content"] == [
        {"type": "text", "text": ""},
        {"type": "input_video", "input_video": {"data": "AAAA"}},
    ]


def test_the_clip_lands_beside_an_image_rather_than_replacing_it():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
    ]
    _inject_video_part(messages, "VVVV")
    types = [p["type"] for p in messages[0]["content"]]
    assert types == ["text", "image_url", "input_video"]


def test_a_tool_turn_after_the_last_user_turn_does_not_steal_the_clip():
    messages = [
        {"role": "user", "content": "watch this"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "42"},
    ]
    _inject_video_part(messages, "VVVV")
    assert messages[0]["content"][-1]["type"] == "input_video"
    assert messages[2]["content"] == "42"


def test_a_long_multi_turn_thread_only_carries_one_clip():
    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    messages.append({"role": "user", "content": "last"})
    _inject_video_part(messages, "VVVV")
    injected = [
        m
        for m in messages
        if isinstance(m["content"], list) and any(p["type"] == "input_video" for p in m["content"])
    ]
    assert len(injected) == 1
    assert injected[0]["content"][0]["text"] == "last"


def test_the_part_shape_is_exactly_what_llama_server_parses():
    messages = [{"role": "user", "content": "x"}]
    _inject_video_part(messages, "PAYLOAD")
    part = messages[0]["content"][-1]
    assert set(part) == {"type", "input_video"}
    assert part["type"] == "input_video"
    assert set(part["input_video"]) == {"data"}
    assert part["input_video"]["data"] == "PAYLOAD"


# --------------------------------------------------------------------------
# E. the refusal paths, read off the handler
# --------------------------------------------------------------------------


def _routes_source() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )


@pytest.mark.parametrize(
    "needle",
    [
        # external provider (OpenAI / Anthropic / any proxied backend)
        'raise HTTPException(\n                status_code = 400,\n                detail = "Video input is only supported on a local GGUF model with video support.",',
        # local non-GGUF (transformers, MLX)
        "if payload.video_base64 and not using_gguf:",
        # GGUF that cannot take video
        'if not getattr(llama_backend, "_has_video_input", False):',
        # tool / guided-decoding passthrough
        '"Video input is not supported together with guided decoding or client-supplied tools yet."',
        # token counting
        '"Cannot count tokens for messages containing video."',
    ],
)
def test_every_path_that_cannot_serve_a_clip_refuses_out_loud(needle):
    assert needle in _routes_source()


def test_the_size_check_is_paid_before_the_model_switch_not_after():
    src = _routes_source()
    handler = src.index("_needs_image = bool(_pre_parsed[2])")
    assert src.index("_video_b64_rejection(payload.video_base64)", handler) < src.index(
        "await _maybe_auto_switch_model(", handler
    )


def test_the_external_provider_refusal_precedes_the_proxy_call():
    src = _routes_source()
    start = src.index("if payload.provider_id or payload.provider_type:")
    branch = src[start : src.index("_proxy_to_external_provider(payload", start)]
    assert "payload.video_base64" in branch
