# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a client gets back when it attaches a clip, by status code and wire shape.

The other video tests assert a line is present in ``routes/inference.py``, which is how the
non-GGUF regression went unnoticed: the source assertion passed while the request 400'd. These
take the ``messages`` kwarg of ``generate_chat_completion`` as the JSON llama-server receives.

Invariant, asserted per route: a ``video_url`` part and the legacy ``video_base64`` field behave
identically, so picking a spelling cannot pick a different set of refusals.

Wire shapes follow llama.cpp ``handle_media``: ``input_video`` is ``data`` else ``url``, remote
downloads cap at 10 MB.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from auth.authentication import get_current_subject  # noqa: E402
import routes.inference as inference_route  # noqa: E402

from .llama_backend_double import FakeLlamaCppBackend  # noqa: E402

_CLIP_B64 = "AAAAGGZ0eXBtcDQy"  # a bare mp4 box header
_DATA_URI = f"data:video/mp4;base64,{_CLIP_B64}"
_REMOTE = "https://example.com/clip.mp4"


class _VideoGguf(FakeLlamaCppBackend):
    """A loaded GGUF whose llama-server /props reported ``modalities.video``."""

    is_vision = True
    _has_video_input = True

    def __init__(self, *, has_video = True):
        self._has_video_input = has_video
        self.dispatched: list[dict] = []

    def generate_chat_completion(self, **kwargs):
        self.dispatched.append(kwargs)
        yield "ok"
        yield {"type": "metadata", "usage": {}, "timings": {}}


def _no_gguf():
    """The resident backend is transformers or MLX, so ``using_gguf`` is False."""
    return SimpleNamespace(
        is_loaded = False, supports_tools = False, is_vision = False, context_length = None
    )


def _client(
    monkeypatch,
    backend = None,
    *,
    prefix = "/v1",
):
    async def _no_switch(*_a, **_k):
        return None

    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: backend if backend else _no_gguf()
    )
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _no_switch)

    app = FastAPI()
    app.include_router(inference_route.router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app, raise_server_exceptions = False)


@pytest.fixture(autouse = True)
def _hosts_resolve_publicly(monkeypatch):
    """A remote clip's host is resolved now, and conftest blocks the real lookup.

    Default every name to one public address so each test below still asserts its own subject;
    the tests about the destination guard install their own resolver over this one.
    """
    import socket

    def _resolve(host, port, *_a, **_k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _resolve)


def _part_body(
    *urls,
    text = "what happens here?",
    **extra,
):
    parts = [{"type": "video_url", "video_url": {"url": u}} for u in urls]
    parts.append({"type": "text", "text": text})
    return {
        "model": "test/model.gguf",
        "stream": False,
        "messages": [{"role": "user", "content": parts}],
        **extra,
    }


def _field_body(
    clip = _DATA_URI,
    text = "what happens here?",
    **extra,
):
    return {
        "model": "test/model.gguf",
        "stream": False,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        "video_base64": clip,
        **extra,
    }


def _detail(response) -> str:
    """The human message, from a bare detail or an OpenAI error body.

    Falls back to raw text, since a request that gets past the refusals answers with SSE or an
    empty body and a test asserting a refusal is *absent* must read those too.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text
    if not isinstance(body, dict):
        return str(body)
    detail = body.get("detail", body)
    if isinstance(detail, dict):
        return str(detail.get("error", detail).get("message", detail))
    return str(detail)


def _sent_parts(backend, index = -1):
    """Parts of a dispatched user turn. A system turn is prepended, so index from the user ones."""
    user_turns = [
        m
        for m in backend.dispatched[0]["messages"]
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    return user_turns[index]["content"]


def _sent_media(backend):
    """Every input_video part across every dispatched turn, in order."""
    return [
        part
        for message in backend.dispatched[0]["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "input_video"
    ]


@pytest.mark.parametrize(
    "url, expected",
    [
        (_DATA_URI, {"data": _CLIP_B64}),
        # The header is not payload; llama.cpp's own base64 decoder takes the body alone.
        (f"DATA:video/mp4;BASE64,{_CLIP_B64}", {"data": _CLIP_B64}),
        ("data:video/webm;base64,REVG", {"data": "REVG"}),
        # Bare base64 with no header: handle_media's final fallback treats it as payload.
        (_CLIP_B64, {"data": _CLIP_B64}),
        # Remote is forwarded whole; llama-server fetches it under its own 10 MB ceiling.
        (_REMOTE, {"url": _REMOTE}),
        ("http://example.com/clip.mp4", {"url": "http://example.com/clip.mp4"}),
        # handle_media matches the scheme case-sensitively, so it is lowercased before
        # forwarding. Only the scheme: the path is case-sensitive and must survive verbatim.
        ("HTTPS://EXAMPLE.COM/Clip.MP4", {"url": "https://EXAMPLE.COM/Clip.MP4"}),
    ],
)
def test_a_clip_reaches_llama_server_as_input_video(monkeypatch, url, expected):
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _part_body(url))
    assert response.status_code == 200
    assert _sent_media(backend) == [{"type": "input_video", "input_video": expected}]


def test_only_the_scheme_is_lowercased_not_the_path(monkeypatch):
    """A case-folded path would 404 on any host that serves case-sensitive paths."""
    backend = _VideoGguf()
    url = "HtTpS://Example.COM/A/Mixed/Case-Path.MP4?Token=AbC"
    with _client(monkeypatch, backend) as client:
        client.post("/v1/chat/completions", json = _part_body(url))
    sent = _sent_media(backend)[0]["input_video"]["url"]
    assert sent == "https://Example.COM/A/Mixed/Case-Path.MP4?Token=AbC"
    assert sent.startswith(
        "http"
    ), "handle_media's string_starts_with(url, 'http') is case-sensitive"


def test_the_translated_part_is_the_only_video_key_left(monkeypatch):
    """A part left spelled video_url would be refused by llama-server as an unsupported type:
    it accepts image_url, input_audio and input_video, and nothing else."""
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        client.post("/v1/chat/completions", json = _part_body(_DATA_URI))
    types = [p.get("type") for p in _sent_parts(backend)]
    assert "video_url" not in types
    assert len(_sent_media(backend)) == 1


def test_the_text_of_the_turn_survives_translation(monkeypatch):
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        client.post("/v1/chat/completions", json = _part_body(_DATA_URI, text = "what colour?"))
    assert {"type": "text", "text": "what colour?"} in _sent_parts(backend)


def test_every_clip_in_a_turn_is_translated(monkeypatch):
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        client.post("/v1/chat/completions", json = _part_body(_DATA_URI, _REMOTE))
    assert [p["input_video"] for p in _sent_media(backend)] == [
        {"data": _CLIP_B64},
        {"url": _REMOTE},
    ]


def test_a_clip_in_an_older_turn_is_translated_too(monkeypatch):
    """_inject_video_part only ever touches the newest user turn, so a parts-carried clip on an
    earlier turn would otherwise reach llama-server still spelled video_url."""
    backend = _VideoGguf()
    body = {
        "model": "test/model.gguf",
        "stream": False,
        "messages": [
            {"role": "user", "content": [{"type": "video_url", "video_url": {"url": _DATA_URI}}]},
            {"role": "assistant", "content": "a red square"},
            {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
        ],
    }
    with _client(monkeypatch, backend) as client:
        client.post("/v1/chat/completions", json = body)
    # Translated in place, on the turn that carried it.
    assert _sent_parts(backend, 0)[0]["type"] == "input_video"
    assert len(_sent_media(backend)) == 1


def test_the_legacy_field_still_rides_the_newest_user_turn(monkeypatch):
    """Backwards compatibility: the spelling the Studio frontend sends is unchanged."""
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _field_body())
    assert response.status_code == 200
    assert _sent_parts(backend)[-1] == {"type": "input_video", "input_video": {"data": _CLIP_B64}}


def test_both_spellings_of_the_same_clip_produce_the_same_wire_shape(monkeypatch):
    """The invariant the whole change rests on, asserted on the dispatched body itself."""
    part_backend, field_backend = _VideoGguf(), _VideoGguf()
    with _client(monkeypatch, part_backend) as client:
        client.post("/v1/chat/completions", json = _part_body(_DATA_URI))
    with _client(monkeypatch, field_backend) as client:
        client.post("/v1/chat/completions", json = _field_body())

    assert _sent_media(part_backend) == _sent_media(field_backend)


@pytest.mark.parametrize("body", [_part_body(_DATA_URI), _field_body()])
def test_a_gguf_without_a_video_projector_refuses_either_spelling(monkeypatch, body):
    with _client(monkeypatch, _VideoGguf(has_video = False)) as client:
        response = client.post("/v1/chat/completions", json = body)
    assert response.status_code == 400
    assert "cannot take video input" in _detail(response)


@pytest.mark.parametrize("body", [_part_body(_DATA_URI), _field_body()])
def test_a_non_gguf_backend_is_offered_the_clip_rather_than_refused_outright(monkeypatch, body):
    """The regression guard. A blanket 'not using_gguf' refusal ahead of _local_video_clip
    refuses video on transformers and MLX for BOTH spellings, including the legacy field the
    frontend sends, breaking working MLX video on macOS.
    """
    reached: list[dict] = []

    def _gate(payload, model_info):
        reached.append(model_info)
        return _CLIP_B64

    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: SimpleNamespace(
            active_model_name = "mlx-model",
            models = {"mlx-model": {"is_vision": True, "has_video_input": True}},
        ),
    )
    monkeypatch.setattr(inference_route, "_local_video_clip", _gate)
    with _client(monkeypatch, None) as client:
        response = client.post("/v1/chat/completions", json = body)

    # Reached-the-gate alone cannot tell the regression from the fix: a blanket refusal placed
    # after the gate still lets the gate run.
    assert reached == [{"is_vision": True, "has_video_input": True}]
    assert "only supported on a local GGUF model" not in _detail(response)
    assert inference_route._VIDEO_INPUT_REFUSAL not in _detail(response)


def test_the_non_gguf_gate_reads_both_spellings(monkeypatch):
    """_local_video_clip predates the part, so left reading payload.video_base64 alone it would
    hand an MLX model nothing while the legacy field worked."""
    from models.inference import ChatCompletionRequest

    info = {"is_vision": True, "has_video_input": True}
    field = ChatCompletionRequest(model = "m", messages = [], video_base64 = _DATA_URI)
    part = ChatCompletionRequest.model_validate(_part_body(_DATA_URI))
    assert inference_route._local_video_clip(field, info) == _CLIP_B64
    assert inference_route._local_video_clip(part, info) == _CLIP_B64


def test_a_non_gguf_backend_without_video_refuses_by_name():
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest.model_validate(_part_body(_DATA_URI))
    with pytest.raises(HTTPException) as exc:
        inference_route._local_video_clip(payload, {"is_vision": True})
    assert exc.value.status_code == 400
    assert exc.value.detail == inference_route._VIDEO_INPUT_REFUSAL
    # The refusal names MLX: that backend serves video, so a GGUF-only message misinforms.
    assert "MLX" in inference_route._VIDEO_INPUT_REFUSAL


def test_a_second_clip_is_refused_rather_than_dropped_on_a_non_gguf_backend():
    """Generation takes one video kwarg. Taking clips[0] silently changed the prompt by
    backend: GGUF forwards every clip, so the same request meant two different things."""
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    info = {"is_vision": True, "has_video_input": True}
    two = ChatCompletionRequest.model_validate(_part_body(_DATA_URI, _DATA_URI))
    with pytest.raises(HTTPException) as exc:
        inference_route._local_video_clip(two, info)
    assert exc.value.status_code == 400
    assert "Only one video" in exc.value.detail

    # The legacy field beside a part is the same two-clip request in a different spelling.
    both = ChatCompletionRequest.model_validate(
        _part_body(_DATA_URI, **{"video_base64": _DATA_URI})
    )
    with pytest.raises(HTTPException) as exc:
        inference_route._local_video_clip(both, info)
    assert exc.value.status_code == 400

    # One clip still serves, so the guard did not swallow the ordinary case.
    one = ChatCompletionRequest.model_validate(_part_body(_DATA_URI))
    assert inference_route._local_video_clip(one, info) == _CLIP_B64


def test_an_unsupported_scheme_is_refused_on_the_non_gguf_path_too():
    """_request_video_rejection runs only when a pre-switch validation happens, so with
    auto-switch off a file:// clip reached the backend and was decoded as base64."""
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest.model_validate(_part_body("file:///etc/passwd"))
    with pytest.raises(HTTPException) as exc:
        inference_route._local_video_clip(payload, {"is_vision": True, "has_video_input": True})
    assert exc.value.status_code == 400
    assert "Unsupported video URL scheme" in exc.value.detail


def test_admission_compacts_a_clip_after_translation_too():
    """The server-side tool loop recosts the conversation after _translate_video_parts has
    renamed the part, so matching video_url alone priced the payload as dense prompt text."""
    import copy

    big = "data:video/mp4;base64," + "A" * 40_000
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": big}},
                {"type": "text", "text": "hi"},
            ],
        }
    ]

    translated = copy.deepcopy(msgs)
    inference_route._translate_video_parts(translated)
    before, _ = inference_route._openai_llama_admission_messages_for_estimate(copy.deepcopy(msgs))
    after, _ = inference_route._openai_llama_admission_messages_for_estimate(translated)

    assert "A" * 40_000 not in str(after)
    # Both spellings compact to about the same size; the clip is a marker, not prompt text.
    assert abs(len(str(after)) - len(str(before))) < 200


def test_a_remote_clip_is_refused_on_a_non_gguf_backend():
    """Only llama-server fetches a clip for itself. A transformers or MLX model is handed bytes,
    so forwarding the URL would feed it the text of the URL instead of the video."""
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest.model_validate(_part_body(_REMOTE))
    with pytest.raises(HTTPException) as exc:
        inference_route._local_video_clip(payload, {"is_vision": True, "has_video_input": True})
    assert exc.value.status_code == 400
    assert "remote video URL" in exc.value.detail


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "FILE:///etc/passwd",
        "ftp://example.com/clip.mp4",
        "gopher://example.com/clip.mp4",
    ],
)
def test_an_unsupported_scheme_is_refused_by_name(monkeypatch, url):
    """handle_media reads input_video as data-or-url and treats the one string it finds the
    same way, honouring file:// under --media-path. Forwarding the clip as opaque payload does
    not neutralise the scheme, so refuse it here.
    """
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body(url))
    assert response.status_code == 400
    assert "Unsupported video URL scheme" in _detail(response)


def test_a_bare_path_is_refused_without_naming_a_scheme(monkeypatch):
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body("/tmp/clip.mp4"))
    # No colon: indistinguishable from base64 payload, so llama-server owns the verdict.
    assert response.status_code == 200


def test_bare_base64_is_not_mistaken_for_a_scheme(monkeypatch):
    """':' is not in the base64 alphabet, which is what separates a URL from payload. A scheme
    check that scanned the whole string would refuse legitimate clips."""
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _part_body("QUJDREVGR0hJSktM"))
    assert response.status_code == 200
    assert _sent_media(backend) == [
        {"type": "input_video", "input_video": {"data": "QUJDREVGR0hJSktM"}}
    ]


def test_an_oversized_clip_is_refused_in_either_spelling():
    """On the shared rule rather than over HTTP, since the body would be 85 MB of JSON. The
    ordering guarantee is pinned in test_video_attachment_part.py."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _MAX_VIDEO_B64_CHARS

    over = "data:video/mp4;base64," + "A" * (_MAX_VIDEO_B64_CHARS + 1)
    for payload in (
        ChatCompletionRequest.model_validate(_part_body(over)),
        ChatCompletionRequest.model_validate(_field_body(over)),
    ):
        assert inference_route._request_video_rejection(payload)[0] == 413


def test_a_clip_of_exactly_the_cap_is_admitted():
    """The boundary the cap exists to allow: refusing it would reject a file the composer offers."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _MAX_VIDEO_B64_CHARS

    at_cap = "data:video/mp4;base64," + "A" * _MAX_VIDEO_B64_CHARS
    payload = ChatCompletionRequest.model_validate(_part_body(at_cap))
    assert inference_route._request_video_rejection(payload) is None


def test_a_data_uri_with_no_payload_is_refused(monkeypatch):
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body("data:video/mp4;base64,"))
    assert response.status_code == 400
    assert "Could not read the provided video file" in _detail(response)


def test_a_remote_url_is_not_measured_against_the_64_mb_cap(monkeypatch):
    """The bytes never reach us, so llama.cpp's 10 MB ceiling is the limit that holds.
    Measuring the URL string would make the cap look enforced while admitting any size."""
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _part_body(_REMOTE))
    assert response.status_code == 200
    assert inference_route._REMOTE_VIDEO_ADMISSION_B64_CHARS < inference_route._MAX_VIDEO_B64_CHARS


def test_admission_prices_a_remote_clip_at_llama_cpp_s_download_ceiling():
    """10 MB, matching common_remote_params.max_size in handle_media."""
    import math
    assert inference_route._REMOTE_VIDEO_ADMISSION_B64_CHARS == 4 * math.ceil(
        (10 * 1024 * 1024) / 3
    )


@pytest.mark.parametrize("body", [_part_body(_DATA_URI), _field_body()])
def test_an_external_provider_refuses_either_spelling(monkeypatch, body):
    """_build_external_messages rebuilds from an allowlist, so a clip left standing is dropped
    and the provider answers a prompt the caller did not send."""
    body = {**body, "provider_type": "openai", "provider_api_key": "sk-test"}
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = body)
    assert response.status_code == 400
    assert response.json()["detail"] == inference_route._VIDEO_INPUT_REFUSAL


@pytest.mark.parametrize("body", [_part_body(_DATA_URI), _field_body()])
def test_token_counting_refuses_either_spelling(monkeypatch, body):
    """llama-server samples the frames at completion time, so counting here undercounts."""
    body = {k: v for k, v in body.items() if k != "stream"}
    with _client(monkeypatch, _VideoGguf(), prefix = "") as client:
        response = client.post("/chat/count_tokens", json = body)
    assert response.status_code == 503
    assert "video" in _detail(response)


@pytest.mark.parametrize("body", [_part_body(_DATA_URI), _field_body()])
def test_the_speech_route_refuses_a_clip_rather_than_speaking_past_it(monkeypatch, body):
    """/audio/generate keeps only text, so without this the clip was dropped in silence.
    Registering video_url as a known tag removed the unknown-part guard that covered it."""
    body = {k: v for k, v in body.items() if k != "stream"}
    with _client(monkeypatch, _VideoGguf(), prefix = "") as client:
        response = client.post("/audio/generate", json = body)
    assert response.status_code == 400
    assert "Video input is not supported here" in _detail(response)


def test_the_speech_route_still_speaks_a_plain_text_turn(monkeypatch):
    """The control for the refusal above: it must not swallow ordinary requests."""
    body = {"model": "default", "messages": [{"role": "user", "content": "read this out"}]}
    with _client(monkeypatch, _VideoGguf(), prefix = "") as client:
        response = client.post("/audio/generate", json = body)
    assert "Video input is not supported here" not in _detail(response)


@pytest.mark.parametrize("body", [_part_body(_DATA_URI), _field_body()])
def test_the_tool_passthrough_path_refuses_either_spelling(monkeypatch, body):
    """That branch forwards an explicit field list, so the clip would be dropped silently."""
    body = {
        **body,
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    }

    class _ToolGguf(_VideoGguf):
        supports_tools = True

    with _client(monkeypatch, _ToolGguf()) as client:
        response = client.post("/v1/chat/completions", json = body)
    assert response.status_code == 400
    assert "guided decoding" in _detail(response)


def test_a_durable_chat_run_refuses_a_clip_in_either_spelling():
    """The request payload persists verbatim, so a data URI would sit in request_json for the
    life of the thread. The gate is field-shaped, so the part needs its own predicate."""
    from fastapi import HTTPException
    from routes.chat_generation_runs import CreateChatGenerationRun, _sanitize_request

    for body in (_part_body(_DATA_URI), _field_body()):
        run = CreateChatGenerationRun(
            runId = "run-1",
            threadId = "thread-1",
            userMessageId = "user-1",
            assistantMessageId = "assistant-1",
            requestPayload = {k: v for k, v in body.items() if k != "stream"},
        )
        with pytest.raises(HTTPException) as exc:
            _sanitize_request(run)
        assert exc.value.status_code == 400
        assert "Media chat runs" in str(exc.value.detail)


def test_a_durable_chat_run_still_takes_a_plain_text_turn():
    """The control: the media gate must not start refusing ordinary durable runs."""
    from routes.chat_generation_runs import CreateChatGenerationRun, _sanitize_request

    run = CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert _sanitize_request(run) is not None


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_a_clip_on_a_non_user_turn_is_refused(monkeypatch, role):
    """OpenAI places media on user turns only, and llama-server renders the marker into whichever
    turn carried it, so anywhere else the result is template-dependent."""
    body = {
        "model": "test/model.gguf",
        "stream": False,
        "messages": [
            {"role": role, "content": [{"type": "video_url", "video_url": {"url": _DATA_URI}}]},
            {"role": "user", "content": "go"},
        ],
    }
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = body)
    assert response.status_code == 422


def test_a_clip_on_a_user_turn_is_accepted(monkeypatch):
    """The control: the role rule must not refuse the placement it exists to protect."""
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body(_DATA_URI))
    assert response.status_code == 200


def test_admission_charges_each_clip_once_and_not_as_prompt_text():
    """Pricing a megabytes-long data URI as prompt text would swamp the estimate."""
    from models.inference import ChatCompletionRequest

    big = "data:video/mp4;base64," + "A" * 40_000
    one = ChatCompletionRequest.model_validate(_part_body(big))
    two = ChatCompletionRequest.model_validate(_part_body(big, big))
    charge_one = inference_route._openai_llama_admission_media_tokens(one)
    charge_two = inference_route._openai_llama_admission_media_tokens(two)
    assert charge_two - charge_one == pytest.approx(charge_one, rel = 0.05)

    estimate, _ = inference_route._openai_llama_admission_messages_for_estimate(
        [m.model_dump(exclude_none = True) for m in one.messages]
    )
    assert "A" * 40_000 not in str(estimate)


def test_admission_prices_the_two_spellings_of_one_clip_alike():
    from models.inference import ChatCompletionRequest

    clip = "data:video/mp4;base64," + "A" * 40_000
    part = ChatCompletionRequest.model_validate(_part_body(clip))
    field = ChatCompletionRequest.model_validate(_field_body(clip))
    assert inference_route._openai_llama_admission_media_tokens(
        part
    ) == inference_route._openai_llama_admission_media_tokens(field)


def test_the_rolling_context_does_not_price_a_clip_as_text():
    """Trimming counts a turn to decide what to drop; a clip counted as text would evict the
    conversation around it."""
    from core.inference.context_window import _UNPRICED_MEDIA_TYPES
    assert {"video_url", "input_video"} <= set(_UNPRICED_MEDIA_TYPES)


def test_an_unknown_part_type_is_still_refused_by_name(monkeypatch):
    """video_url used to land here. Registering it must not have widened the catch-all: an
    unregistered type still has to be refused rather than silently dropped."""
    body = {
        "model": "test/model.gguf",
        "stream": False,
        "messages": [
            {"role": "user", "content": [{"type": "hologram_url", "hologram_url": {"url": "x"}}]}
        ],
    }
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = body)
    assert response.status_code == 400
    assert "hologram_url" in _detail(response)


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/clip.mp4",
        "http://localhost/clip.mp4",
        "http://[::1]/clip.mp4",
        "http://[::ffff:127.0.0.1]/clip.mp4",
        "http://192.168.1.10/clip.mp4",
        "http://10.0.0.5/clip.mp4",
    ],
)
def test_a_remote_clip_aimed_at_a_private_host_is_refused(monkeypatch, url):
    """llama-server downloads the URL from this machine, so an unchecked host turns a clip into
    a server-side fetch of loopback, the LAN or a metadata endpoint."""
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body(url))
    assert response.status_code == 400
    assert "must point at a public host" in _detail(response)


def test_a_public_remote_clip_is_still_forwarded(monkeypatch):
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _part_body(_REMOTE))
    assert response.status_code == 200
    assert _sent_media(backend) == [{"type": "input_video", "input_video": {"url": _REMOTE}}]


def test_the_destination_guard_also_runs_without_a_pre_switch_validation():
    """_request_video_rejection only runs on a pre-switch validation, so the dispatch boundary
    has to refuse the same host on its own."""
    from fastapi import HTTPException

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "http://169.254.169.254/clip.mp4"}}
            ],
        }
    ]
    with pytest.raises(HTTPException) as exc:
        inference_route._translate_video_parts(messages)
    assert exc.value.status_code == 400
    assert "must point at a public host" in exc.value.detail


def test_a_hostname_is_not_resolved_in_the_request_path(monkeypatch):
    """Resolving would block the event loop, and llama-server re-resolves anyway, so a name is
    forwarded rather than classified. Pinned so the tradeoff is not lost by accident."""
    import socket

    def _boom(*_a, **_k):
        raise AssertionError("the request path must not resolve a video hostname")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    assert inference_route._remote_video_destination_rejection(_REMOTE) is None


def test_the_pre_switch_validation_refuses_a_private_host_before_any_model_loads():
    """The dispatch boundary would catch it, but only after a switch had already been paid for."""
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest.model_validate(
        _part_body("http://169.254.169.254/latest/meta-data/")
    )
    rejection = inference_route._request_video_rejection(payload)
    assert rejection is not None
    assert rejection[0] == 400
    assert "must point at a public host" in rejection[1]
    assert (
        inference_route._request_video_rejection(
            ChatCompletionRequest.model_validate(_part_body(_REMOTE))
        )
        is None
    )


@pytest.mark.parametrize(
    "host",
    ["2130706433", "127.1", "0177.0.0.1", "0x7f000001", "0x7f.1", "017700000001"],
)
def test_a_legacy_numeric_form_of_loopback_is_refused_too(monkeypatch, host):
    """ip_address reads dotted-quad only, but a resolver reads all of these as 127.0.0.1, so
    classifying with ip_address alone left the guard bypassable."""
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body(f"http://{host}/clip.mp4"))
    assert response.status_code == 400
    assert "must point at a public host" in _detail(response)


@pytest.mark.parametrize("url", ["http://[::1", "http://[bad]/clip.mp4", "https://[::1]:x/c.mp4"])
def test_a_malformed_remote_url_is_a_client_error_not_a_crash(monkeypatch, url):
    """urlsplit raises on a broken authority; uncaught it would surface as a 500."""
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body(url))
    assert response.status_code == 400


def test_a_public_numeric_host_is_still_allowed(monkeypatch):
    """The numeric check must refuse loopback, not every address written as digits."""
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _part_body("http://8.8.8.8/c.mp4"))
    assert response.status_code == 200
    assert _sent_media(backend) == [
        {"type": "input_video", "input_video": {"url": "http://8.8.8.8/c.mp4"}}
    ]


def _fake_resolver(mapping):
    """getaddrinfo shaped like socket's, answering from a dict of host -> addresses."""
    import socket

    def _resolve(host, port, *a, **k):
        if host not in mapping:
            raise socket.gaierror(-2, "Name or service not known")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
            for address in mapping[host]
        ]

    return _resolve


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "169.254.169.254", "10.1.2.3", "192.168.0.9", "::1"]
)
def test_a_hostname_resolving_somewhere_private_is_refused(monkeypatch, address):
    """The literal checks never see a name, and a name the caller controls can answer loopback."""
    import socket

    monkeypatch.setattr(
        socket, "getaddrinfo", _fake_resolver({"clips.example": [address]})
    )
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post(
            "/v1/chat/completions", json = _part_body("http://clips.example/clip.mp4")
        )
    assert response.status_code == 400
    assert "must point at a public host" in _detail(response)


def test_a_hostname_with_one_private_answer_among_public_ones_is_refused(monkeypatch):
    """Checking only the first answer would let a name volunteer a public address and still
    hand llama-server a private one."""
    import socket

    monkeypatch.setattr(
        socket, "getaddrinfo", _fake_resolver({"clips.example": ["93.184.216.34", "127.0.0.1"]})
    )
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post(
            "/v1/chat/completions", json = _part_body("http://clips.example/clip.mp4")
        )
    assert response.status_code == 400


def test_a_hostname_resolving_publicly_is_still_forwarded(monkeypatch):
    import socket

    monkeypatch.setattr(
        socket, "getaddrinfo", _fake_resolver({"clips.example": ["93.184.216.34"]})
    )
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post(
            "/v1/chat/completions", json = _part_body("http://clips.example/clip.mp4")
        )
    assert response.status_code == 200
    assert _sent_media(backend) == [
        {"type": "input_video", "input_video": {"url": "http://clips.example/clip.mp4"}}
    ]


def test_a_host_that_cannot_be_resolved_is_refused_by_name(monkeypatch):
    """llama-server would fail the fetch too; saying so beats letting it try."""
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver({}))
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post(
            "/v1/chat/completions", json = _part_body("http://nowhere.example/clip.mp4")
        )
    assert response.status_code == 400
    assert "could not be resolved" in _detail(response)


def test_the_lookup_does_not_run_on_the_event_loop(monkeypatch):
    """A blocking getaddrinfo in the request path would stall every other request."""
    import asyncio
    import socket

    seen = {}

    def _resolve(host, port, *a, **k):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _resolve)
    with _client(monkeypatch, _VideoGguf()) as client:
        client.post("/v1/chat/completions", json = _part_body("http://clips.example/clip.mp4"))
    assert seen == {"on_loop": False}


@pytest.mark.parametrize(
    "url",
    [
        "https://clips.example:bad/clip.mp4",
        "https://clips.example:99999/clip.mp4",
    ],
)
def test_an_invalid_port_on_a_resolvable_host_is_a_client_error(monkeypatch, url):
    """urlsplit parses the hostname but raises on .port, so reading it outside the guarded
    block surfaced a malformed URL as a 500."""
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body(url))
    assert response.status_code == 400
    assert "could not be parsed" in _detail(response)
