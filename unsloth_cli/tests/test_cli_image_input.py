# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth inference --image` and `unsloth chat --image` / `/image` (#16340)."""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

import unsloth_cli.commands.chat as chatmod
import unsloth_cli.commands.inference as infermod
from unsloth_cli._inference import (
    ChatBackend,
    HttpChatBackend,
    _local_image_request,
    ensure_studio_backend_path,
    image_part,
    user_turn,
)


class _FakeConfig:
    is_gguf = False
    is_lora = False
    display_name = "fake-vlm"
    base_model = None
    path = None


def _png(path, color = (255, 0, 0)):
    Image.new("RGB", (8, 8), color).save(path, format = "PNG")
    return path


def _decoded(part) -> Image.Image:
    url = part["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2])))


def _app(command):
    cli = typer.Typer()
    cli.command()(command)
    return cli


class _RecordingBackend:
    def __init__(self, supports_images = True):
        self.supports_images = supports_images
        self.streams = []
        self.closed = False

    def stream(self, messages, **kwargs):
        self.streams.append([dict(m) for m in messages])
        return iter(["a red square"])

    def close(self):
        self.closed = True


_STREAM_KWARGS = dict(
    system_prompt = "",
    temperature = 0.7,
    top_p = 0.9,
    top_k = 40,
    max_new_tokens = 8,
    repetition_penalty = 1.0,
    enable_thinking = False,
)


# ── image_part ────────────────────────────────────────────────────


def test_image_part_carries_the_file_as_a_data_url(tmp_path):
    part = image_part(_png(tmp_path / "red.png"))

    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert _decoded(part).getpixel((0, 0)) == (255, 0, 0)


def test_image_part_names_jpeg_by_its_content_not_its_extension(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 4), (0, 0, 255)).save(path, format = "JPEG")

    assert image_part(path)["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_image_part_rejects_a_missing_or_non_image_file(tmp_path):
    with pytest.raises(ValueError, match = "can't read image"):
        image_part(tmp_path / "missing.png")
    notes = tmp_path / "notes.png"
    notes.write_text("not a picture", encoding = "utf-8")
    with pytest.raises(ValueError, match = "not an image file"):
        image_part(notes)


def test_user_turn_puts_images_ahead_of_the_text(tmp_path):
    part = image_part(_png(tmp_path / "a.png"))

    assert user_turn("hi") == {"role": "user", "content": "hi"}
    assert user_turn("what is this?", [part]) == {
        "role": "user",
        "content": [part, {"type": "text", "text": "what is this?"}],
    }


# ── unsloth inference --image ─────────────────────────────────────


def _local_inference(monkeypatch, backend):
    monkeypatch.setattr(infermod, "connect_studio_server", lambda *a, **k: None)
    monkeypatch.setattr(infermod, "load_chat_backend", lambda *a, **k: backend)


def test_inference_sends_each_image_with_the_prompt(monkeypatch, tmp_path):
    backend = _RecordingBackend()
    _local_inference(monkeypatch, backend)
    red, blue = _png(tmp_path / "red.png"), _png(tmp_path / "blue.png", (0, 0, 255))

    result = CliRunner().invoke(
        _app(infermod.inference),
        ["fake-vlm", "compare these", "--image", str(red), "--image", str(blue)],
    )

    assert result.exit_code == 0, result.output
    assert "a red square" in result.stdout
    [[turn]] = backend.streams
    *images, text = turn["content"]
    assert text == {"type": "text", "text": "compare these"}
    assert [_decoded(part).getpixel((0, 0)) for part in images] == [(255, 0, 0), (0, 0, 255)]
    assert backend.closed


def test_inference_without_image_keeps_a_plain_text_turn(monkeypatch):
    backend = _RecordingBackend(supports_images = False)
    _local_inference(monkeypatch, backend)

    result = CliRunner().invoke(_app(infermod.inference), ["fake-model", "hello"])

    assert result.exit_code == 0, result.output
    assert backend.streams == [[{"role": "user", "content": "hello"}]]


def test_inference_reports_a_bad_image_path_before_loading(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(infermod, "connect_studio_server", lambda *a, **k: loads.append(1))
    monkeypatch.setattr(infermod, "load_chat_backend", lambda *a, **k: loads.append(1))

    result = CliRunner().invoke(
        _app(infermod.inference),
        ["fake-vlm", "hi", "--image", str(tmp_path / "missing.png")],
    )

    assert result.exit_code == 1
    assert "can't read image" in result.output
    assert loads == []


def test_inference_refuses_an_image_for_a_text_only_model(monkeypatch, tmp_path):
    backend = _RecordingBackend(supports_images = False)
    _local_inference(monkeypatch, backend)

    result = CliRunner().invoke(
        _app(infermod.inference),
        ["text-model", "hi", "--image", str(_png(tmp_path / "a.png"))],
    )

    assert result.exit_code == 1
    assert "not a vision model" in result.output
    assert backend.streams == []
    assert backend.closed


def test_inference_and_chat_list_the_image_option():
    for command in (infermod.inference, chatmod.chat):
        result = CliRunner().invoke(_app(command), ["--help"])
        assert "--image" in result.output


# ── unsloth chat --image and /image ───────────────────────────────


def _local_chat(monkeypatch, backend):
    monkeypatch.setattr(chatmod, "resolve_model_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(chatmod, "connect_studio_server", lambda *a, **k: None)
    monkeypatch.setattr(chatmod, "load_chat_backend", lambda *a, **k: backend)


def _user_turns(stream):
    return [m for m in stream if m["role"] == "user"]


def test_chat_image_flag_rides_on_the_first_message_only(monkeypatch, tmp_path):
    backend = _RecordingBackend()
    _local_chat(monkeypatch, backend)

    result = CliRunner().invoke(
        _app(chatmod.chat),
        ["fake-vlm", "--image", str(_png(tmp_path / "red.png"))],
        input = "what colour is it?\nand now?\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    first, second = backend.streams
    [turn] = _user_turns(first)
    assert turn["content"][0]["type"] == "image_url"
    assert turn["content"][-1] == {"type": "text", "text": "what colour is it?"}
    # The history keeps the picture; the follow-up question carries none of its own.
    assert _user_turns(second)[0] == turn
    assert _user_turns(second)[1] == {"role": "user", "content": "and now?"}


def test_chat_slash_image_attaches_to_the_next_message(monkeypatch, tmp_path):
    backend = _RecordingBackend()
    _local_chat(monkeypatch, backend)
    folder = tmp_path / "my pictures"
    folder.mkdir()
    blue = _png(folder / "blue.png", (0, 0, 255))

    result = CliRunner().invoke(
        _app(chatmod.chat),
        ["fake-vlm"],
        input = f'hello\n/image "{blue}"\nwhat is this?\n/exit\n',
    )

    assert result.exit_code == 0, result.output
    assert "(1 image attached to your next message)" in result.output
    first, second = backend.streams
    assert _user_turns(first) == [{"role": "user", "content": "hello"}]
    attached = _user_turns(second)[1]["content"]
    assert _decoded(attached[0]).getpixel((0, 0)) == (0, 0, 255)
    assert attached[1] == {"type": "text", "text": "what is this?"}


def test_chat_reset_drops_a_pending_image(monkeypatch, tmp_path):
    backend = _RecordingBackend()
    _local_chat(monkeypatch, backend)

    result = CliRunner().invoke(
        _app(chatmod.chat),
        ["fake-vlm"],
        input = f"/image {_png(tmp_path / 'a.png')}\n/reset\nhi\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert backend.streams == [[{"role": "user", "content": "hi"}]]


def test_chat_slash_image_explains_a_bad_path_and_keeps_chatting(monkeypatch, tmp_path):
    backend = _RecordingBackend()
    _local_chat(monkeypatch, backend)

    result = CliRunner().invoke(
        _app(chatmod.chat),
        ["fake-vlm"],
        input = f"/image {tmp_path / 'missing.png'}\n/image\nhi\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "can't read image" in result.output
    assert "(usage: /image <path>)" in result.output
    assert backend.streams == [[{"role": "user", "content": "hi"}]]


def test_chat_text_only_model_refuses_images(monkeypatch, tmp_path):
    picture = _png(tmp_path / "a.png")

    backend = _RecordingBackend(supports_images = False)
    _local_chat(monkeypatch, backend)
    result = CliRunner().invoke(_app(chatmod.chat), ["fake-model", "--image", str(picture)])
    assert result.exit_code == 1
    assert "not a vision model" in result.output
    assert backend.closed and backend.streams == []

    backend = _RecordingBackend(supports_images = False)
    _local_chat(monkeypatch, backend)
    result = CliRunner().invoke(
        _app(chatmod.chat), ["fake-model"], input = f"/image {picture}\nhi\n/exit\n"
    )
    assert result.exit_code == 0, result.output
    assert "not a vision model, so it can't read images" in result.output
    assert backend.streams == [[{"role": "user", "content": "hi"}]]


def test_chat_help_lists_the_image_command():
    assert "/image <path>" in chatmod._HELP


# ── backends ──────────────────────────────────────────────────────


class _FakeOrchestrator:
    def __init__(self, *, is_vision = True, several = False):
        self.active_model_name = "fake/vlm"
        self.models = {
            "fake/vlm": {
                "is_vision": is_vision,
                "chat_template_info": {"accepts_multiple_images": several},
            }
        }
        self.calls = []

    def generate_chat_response(self, **kwargs):
        self.calls.append(kwargs)
        return iter(["ok"])


def _parts(tmp_path, *colors):
    return [
        image_part(_png(tmp_path / f"{index}.png", color)) for index, color in enumerate(colors)
    ]


def _payload(part):
    return part["image_url"]["url"].partition(",")[2]


def test_chatbackend_reports_vision_support_per_backend():
    assert ChatBackend("unsloth", _FakeOrchestrator()).supports_images
    assert not ChatBackend("unsloth", _FakeOrchestrator(is_vision = False)).supports_images

    class _Gguf:
        is_vision = True

    assert ChatBackend("gguf", _Gguf()).supports_images
    assert not ChatBackend("gguf", object()).supports_images


def test_unsloth_backend_gets_markers_and_payloads_like_the_studio_route(tmp_path):
    red, blue = _parts(tmp_path, (255, 0, 0), (0, 0, 255))
    fake = _FakeOrchestrator(several = True)
    messages = [
        user_turn("first", [red]),
        {"role": "assistant", "content": "a red square"},
        user_turn("second", [blue]),
    ]

    list(ChatBackend("unsloth", fake).stream(messages, **_STREAM_KWARGS))

    [call] = fake.calls
    assert call["messages"] == [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "first"}]},
        {"role": "assistant", "content": "a red square"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "second"}]},
    ]
    assert call["images"] == [_payload(red), _payload(blue)]
    assert call["image_ordinal"] == 1


def test_single_image_model_gets_only_the_newest_image(tmp_path):
    red, blue = _parts(tmp_path, (255, 0, 0), (0, 0, 255))
    fake = _FakeOrchestrator(several = False)
    messages = [
        user_turn("first", [red]),
        {"role": "assistant", "content": "a red square"},
        user_turn("second", [blue]),
        {"role": "assistant", "content": "a blue square"},
        user_turn("third"),
    ]

    list(ChatBackend("unsloth", fake).stream(messages, **_STREAM_KWARGS))

    [call] = fake.calls
    assert call["messages"][0] == {"role": "user", "content": "first"}
    assert call["messages"][2] == {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "second"}],
    }
    assert call["messages"][4] == {"role": "user", "content": "third"}
    assert call["images"] == [_payload(blue)]
    assert call["image_ordinal"] == 1


def test_single_image_model_refuses_two_images_in_one_message(tmp_path):
    fake = _FakeOrchestrator(several = False)
    two = user_turn("compare", _parts(tmp_path, (255, 0, 0), (0, 0, 255)))

    with pytest.raises(ValueError, match = "one image per message"):
        ChatBackend("unsloth", fake).stream([two], **_STREAM_KWARGS)
    assert fake.calls == []


def test_text_only_conversation_reaches_the_unsloth_backend_unchanged():
    fake = _FakeOrchestrator()

    list(ChatBackend("unsloth", fake).stream([user_turn("hi")], **_STREAM_KWARGS))

    assert fake.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert "images" not in fake.calls[0]


def test_gguf_backend_gets_openai_image_parts_for_llama_server(tmp_path):
    calls = []

    class _Gguf:
        is_vision = True

        def generate_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return iter(["ok"])

    [red] = _parts(tmp_path, (255, 0, 0))
    turn = user_turn("what is this?", [red])

    list(ChatBackend("gguf", _Gguf()).stream([turn], **_STREAM_KWARGS))

    assert calls[0]["messages"] == [turn]


def test_markers_survive_studios_real_vision_message_assembly(tmp_path):
    """The shapes the CLI builds go through the same helpers the transformers backend runs."""
    ensure_studio_backend_path()
    from core.inference.chat_template_helpers import messages_with_attached_image
    from core.inference.mcp_images import image_marker_parts, top_up_image_markers

    red, blue = _parts(tmp_path, (255, 0, 0), (0, 0, 255))
    history = [
        user_turn("first", [red]),
        {"role": "assistant", "content": "a red square"},
        user_turn("second", [blue]),
        {"role": "assistant", "content": "a blue square"},
        user_turn("which was first?"),
    ]
    for several, expected in ((True, 2), (False, 1)):
        messages, images, ordinal = _local_image_request(history, several)
        assembled = top_up_image_markers(
            messages_with_attached_image(
                messages, fallback_user_text = "which was first?", structured_content = True
            ),
            len(images),
            ordinal = ordinal,
        )
        # One marker per picture, each still on the turn that sent it; the question carries none.
        assert len(image_marker_parts(assembled)) == len(images) == expected
        assert assembled[-1]["content"] == [{"type": "text", "text": "which was first?"}]


# ── a running Unsloth server ──────────────────────────────────────


class _StudioStub(BaseHTTPRequestHandler):
    received: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append((self.path, body))
        if self.path == "/api/inference/load":
            reply = json.dumps({"status": "loaded", "model": "m", "is_vision": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
            return
        # Answer with what the server decoded, so the test sees the picture arrive intact.
        sizes = []
        for message in body["messages"]:
            for part in message["content"] if isinstance(message["content"], list) else ():
                if part["type"] == "image_url":
                    sizes.append("x".join(map(str, _decoded(part).size)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunk = {"choices": [{"delta": {"content": f"saw {','.join(sizes)}"}}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture
def studio_stub():
    _StudioStub.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StudioStub)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_server_backend_posts_images_as_openai_parts(studio_stub, tmp_path):
    backend = HttpChatBackend(studio_stub, "token")
    backend.ensure_loaded("m", hf_token = None, max_seq_length = 0, load_in_4bit = None)
    assert backend.supports_images

    part = image_part(_png(tmp_path / "a.png"))
    answer = list(backend.stream([user_turn("what is this?", [part])], **_STREAM_KWARGS))

    assert answer[-1] == "saw 8x8"
    path, body = _StudioStub.received[-1]
    assert path == "/v1/chat/completions"
    assert body["messages"] == [user_turn("what is this?", [part])]
