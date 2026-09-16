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
    ],
)
def test_a_clip_reaches_llama_server_as_input_video(monkeypatch, url, expected):
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _part_body(url))
    assert response.status_code == 200
    assert _sent_media(backend) == [{"type": "input_video", "input_video": expected}]


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
        client.post(
            "/v1/chat/completions", json = _part_body(_DATA_URI, "data:video/webm;base64,REVG")
        )
    assert [p["input_video"] for p in _sent_media(backend)] == [
        {"data": _CLIP_B64},
        {"data": "REVG"},
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
    assert "Remote video URLs are not supported" in exc.value.detail


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
        "https://example.com/clip.mp4",
        "http://example.com/clip.mp4",
        "HTTPS://EXAMPLE.COM/Clip.MP4",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/clip.mp4",
        "http://[::1",
        "https://clips.example:bad/clip.mp4",
    ],
)
def test_a_remote_video_url_is_refused(monkeypatch, url):
    """llama-server, not Studio, would fetch the clip, and its downloader follows redirects
    (set_follow_location in common/http.h), so a public URL redirecting to a private address
    defeats any host check made here. The shape is refused rather than guarded."""
    with _client(monkeypatch, _VideoGguf()) as client:
        response = client.post("/v1/chat/completions", json = _part_body(url))
    assert response.status_code == 400
    assert "Remote video URLs are not supported" in _detail(response)


def test_no_url_ever_reaches_llama_server_as_a_video(monkeypatch):
    """The refusal is the point: input_video must always carry bytes, never something to dial."""
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        client.post("/v1/chat/completions", json = _part_body(_DATA_URI))
    assert all("data" in part["input_video"] for part in _sent_media(backend))
    assert not any("url" in part["input_video"] for part in _sent_media(backend))


def test_the_dispatch_boundary_refuses_a_remote_url_on_its_own():
    """_request_video_rejection only runs on a pre-switch validation, so translation has to
    refuse it too rather than trust that it was already checked."""
    from fastapi import HTTPException

    messages = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": _REMOTE}}]}]
    with pytest.raises(HTTPException) as exc:
        inference_route._translate_video_parts(messages)
    assert exc.value.status_code == 400
    assert "Remote video URLs are not supported" in exc.value.detail


def test_both_spellings_refuse_a_remote_clip_alike(monkeypatch):
    with _client(monkeypatch, _VideoGguf()) as client:
        part = client.post("/v1/chat/completions", json = _part_body(_REMOTE))
        field = client.post("/v1/chat/completions", json = _field_body(_REMOTE))
    assert part.status_code == field.status_code == 400
    assert _detail(part) == _detail(field)


def _older_turn_body(clip = _DATA_URI):
    return {
        "model": "test/model.gguf",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": clip}},
                    {"type": "text", "text": "what is this?"},
                ],
            },
            {"role": "assistant", "content": "a clip"},
            {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
        ],
    }


def test_a_clip_on_an_older_turn_is_refused_on_a_non_gguf_backend():
    """GGUF keeps the part on the turn that carried it; the non-GGUF path flattens the parts and
    hands generation one video kwarg, which MLX attaches to the newest turn. Refusing keeps the
    two backends from reading the same request as different conversations."""
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest.model_validate(_older_turn_body())
    with pytest.raises(HTTPException) as exc:
        inference_route._local_video_clip(payload, {"is_vision": True, "has_video_input": True})
    assert exc.value.status_code == 400
    assert "attached to the latest message" in exc.value.detail


def test_a_clip_on_the_latest_turn_is_still_served_on_a_non_gguf_backend():
    from models.inference import ChatCompletionRequest
    payload = ChatCompletionRequest.model_validate(_part_body(_DATA_URI))
    assert (
        inference_route._local_video_clip(payload, {"is_vision": True, "has_video_input": True})
        == _CLIP_B64
    )


def test_the_legacy_field_is_not_treated_as_an_older_turn():
    """video_base64 has no turn to belong to, so it is always the newest attachment."""
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest.model_validate(_field_body())
    assert inference_route._video_is_on_an_older_turn(payload) is False
    assert (
        inference_route._local_video_clip(payload, {"is_vision": True, "has_video_input": True})
        == _CLIP_B64
    )


def test_gguf_still_keeps_a_clip_on_the_turn_that_carried_it(monkeypatch):
    """The refusal above is a non-GGUF limit, not a new rule for everyone."""
    backend = _VideoGguf()
    with _client(monkeypatch, backend) as client:
        response = client.post("/v1/chat/completions", json = _older_turn_body())
    assert response.status_code == 200
    turns = [
        m
        for m in backend.dispatched[0]["messages"]
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    assert [p.get("type") for p in turns[0]["content"]] == ["input_video", "text"]
    assert [p.get("type") for p in turns[-1]["content"]] == ["text"]


def test_validation_does_not_copy_the_clip():
    """Validation runs twice per request and only wants the verdict, so slicing the header off
    to get it copied the whole payload each time: 89 MB for a clip at the limit."""
    import tracemalloc

    from models.inference import ChatCompletionRequest

    clip = "data:video/mp4;base64," + "A" * (8 * 1024 * 1024)
    payload = ChatCompletionRequest.model_validate(_part_body(clip))
    tracemalloc.start()
    try:
        assert inference_route._request_video_rejection(payload) is None
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024, f"validation allocated {peak} bytes for an 8 MB clip"


@pytest.mark.parametrize(
    "clip, expected",
    [
        ("data:video/mp4;base64," + "A" * 40, None),
        ("A" * 40, None),
        ("", (400, "Could not read the provided video file.")),
        ("data:video/mp4;base64,", (400, "Could not read the provided video file.")),
        ("data:video/mp4;base64", (400, "Could not read the provided video file.")),
    ],
)
def test_the_measured_verdict_matches_the_sliced_one(clip, expected):
    """The fast path has to agree with _video_b64_rejection, header handling included."""
    assert inference_route._video_size_rejection(clip) == expected
    assert inference_route._video_b64_rejection(clip)[1] == expected


def test_the_measured_verdict_agrees_on_the_cap_boundary():
    limit = inference_route._MAX_VIDEO_B64_CHARS
    for length in (limit, limit + 1):
        clip = "data:video/mp4;base64," + "A" * length
        assert (
            inference_route._video_size_rejection(clip)
            == (inference_route._video_b64_rejection(clip)[1])
        )


def test_a_part_carried_clip_is_transcoded_like_the_legacy_field(monkeypatch):
    """main shrinks the clip before llama-server samples it, but only on the legacy field's
    path. Without the same pass over the parts, the same clip would reach llama-server at a
    different resolution depending on which spelling carried it."""
    seen = []

    def _shrink(
        clip,
        cap,
        *,
        sampled_fps = None,
    ):
        seen.append(clip)
        return "SHRUNK"

    monkeypatch.setattr(inference_route, "shrink_video_for_llama", _shrink)

    part_backend = _VideoGguf()
    with _client(monkeypatch, part_backend) as client:
        assert client.post("/v1/chat/completions", json = _part_body(_DATA_URI)).status_code == 200
    field_backend = _VideoGguf()
    with _client(monkeypatch, field_backend) as client:
        assert client.post("/v1/chat/completions", json = _field_body()).status_code == 200

    assert seen == [_CLIP_B64, _CLIP_B64], "both spellings must reach the transcoder"
    assert _sent_media(part_backend) == _sent_media(field_backend)
    assert _sent_media(part_backend) == [{"type": "input_video", "input_video": {"data": "SHRUNK"}}]


def test_several_clips_are_capped_in_aggregate_not_only_per_clip():
    """Every clip is transcoded and shrink_video_for_llama allows 300s each, so N clips just
    under the per-clip limit is N*300s of ffmpeg for one request."""
    from models.inference import ChatCompletionRequest

    half = "data:video/mp4;base64," + "A" * (inference_route._MAX_VIDEO_B64_CHARS // 2 + 10)
    one = ChatCompletionRequest.model_validate(_part_body(half))
    assert inference_route._request_video_rejection(one) is None
    two = ChatCompletionRequest.model_validate(_part_body(half, half))
    rejection = inference_route._request_video_rejection(two)
    assert rejection is not None and rejection[0] == 413
    assert "per request" in rejection[1]


def test_the_aggregate_cap_counts_the_legacy_field_too():
    """Otherwise the field plus a part could exceed it together."""
    from models.inference import ChatCompletionRequest

    half = "data:video/mp4;base64," + "A" * (inference_route._MAX_VIDEO_B64_CHARS // 2 + 10)
    body = _part_body(half)
    body["video_base64"] = half
    rejection = inference_route._request_video_rejection(ChatCompletionRequest.model_validate(body))
    assert rejection is not None and rejection[0] == 413


def test_recosting_drops_a_clip_the_conversation_has_evicted():
    """truncate_oldest can evict the turn that carried the clip. Charging it from the opening
    payload kept every later round reserved at the full budget for media no longer sent."""
    from models.inference import ChatCompletionRequest

    clip = "data:video/mp4;base64," + "A" * 40_000
    payload = ChatCompletionRequest.model_validate(_part_body(clip))
    opening = inference_route._openai_llama_admission_media_tokens(payload)
    assert opening > 1000

    evicted = inference_route._openai_llama_admission_media_tokens(
        payload, message_video_clips = inference_route._conversation_video_clips([])
    )
    assert evicted == 0


def test_recosting_still_charges_a_clip_the_conversation_kept():
    """Reading video_url alone would find nothing post-translation and charge no video at all."""
    from models.inference import ChatCompletionRequest

    clip = "A" * 40_000
    payload = ChatCompletionRequest.model_validate(_part_body("data:video/mp4;base64," + clip))
    conversation = [
        {"role": "user", "content": [{"type": "input_video", "input_video": {"data": clip}}]}
    ]
    kept = inference_route._openai_llama_admission_media_tokens(
        payload, message_video_clips = inference_route._conversation_video_clips(conversation)
    )
    assert kept == len(clip) // 4


def test_the_legacy_field_is_not_charged_twice_during_recosting():
    """_inject_video_part splices the legacy clip into the conversation as input_video before
    the loop starts, so during a recost the conversation clips already include it. Charging the
    field as well priced it at exactly 2x and clamped the lease toward the whole KV budget."""
    from models.inference import ChatCompletionRequest

    clip = "A" * 40_000
    payload = ChatCompletionRequest.model_validate(
        _field_body("data:video/mp4;base64," + clip, text = "hi")
    )
    conversation = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    inference_route._inject_video_part(conversation, clip)

    opening = inference_route._openai_llama_admission_media_tokens(payload)
    recost = inference_route._openai_llama_admission_media_tokens(
        payload,
        message_video_clips = inference_route._conversation_video_clips(conversation),
    )
    assert recost == pytest.approx(opening, rel = 0.01)


def test_audio_is_still_charged_from_the_field_during_recosting():
    """Only video is spliced into the conversation; dropping audio with it would undercharge."""
    from models.inference import ChatCompletionRequest

    body = _field_body("data:video/mp4;base64," + "A" * 400)
    body["audio_base64"] = "B" * 4000
    payload = ChatCompletionRequest.model_validate(body)
    assert (
        inference_route._openai_llama_admission_media_tokens(payload, message_video_clips = [])
        == 1000
    )


def test_many_tiny_clips_are_refused_by_count_not_only_by_bytes():
    """Each clip is its own ffprobe and ffmpeg (30s and 300s ceilings), so thousands of tiny
    clips stay under the byte cap while holding the request and its lease for hours."""
    from models.inference import ChatCompletionRequest

    tiny = "data:video/mp4;base64,QUJD"
    many = ChatCompletionRequest.model_validate(
        _part_body(*[tiny] * (inference_route._MAX_VIDEO_CLIPS_PER_REQUEST + 1))
    )
    rejection = inference_route._request_video_rejection(many)
    assert rejection is not None and rejection[0] == 400
    assert "Too many videos" in rejection[1]


def test_a_request_at_the_clip_limit_is_still_served():
    """The count cap must bound abuse, not ordinary multi-clip use."""
    from models.inference import ChatCompletionRequest

    tiny = "data:video/mp4;base64,QUJD"
    ok = ChatCompletionRequest.model_validate(
        _part_body(*[tiny] * inference_route._MAX_VIDEO_CLIPS_PER_REQUEST)
    )
    assert inference_route._request_video_rejection(ok) is None


def test_the_clip_count_includes_the_legacy_field():
    """Otherwise the field plus the maximum parts exceeds the bound together."""
    from models.inference import ChatCompletionRequest

    tiny = "data:video/mp4;base64,QUJD"
    body = _part_body(*[tiny] * inference_route._MAX_VIDEO_CLIPS_PER_REQUEST)
    body["video_base64"] = tiny
    rejection = inference_route._request_video_rejection(ChatCompletionRequest.model_validate(body))
    assert rejection is not None and rejection[0] == 400
