# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import io
import json
import os
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from core.inference import tools
from core.inference.mcp_images import promote_history, split_images
from core.inference.tool_loop_controller import ToolLoopController
from core.inference.view_image import VIEW_IMAGE_TOOL, view_image


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(tmp_path))
    Image.new("RGB", (120, 80), "red").save(tmp_path / "image.png")
    return tmp_path


def test_existing_image_reaches_live_model_and_replayed_history(workdir):
    result = tools.execute_tool("view_image", {"path": "image.png"}, session_id = "test")
    text, images = split_images(result)
    assert text == "Opened image."
    with Image.open(io.BytesIO(base64.b64decode(images[0]["data"]))) as decoded:
        assert decoded.size == (120, 80)
        assert decoded.getpixel((20, 20)) == (255, 0, 0)
    controller = ToolLoopController(tools = [VIEW_IMAGE_TOOL])
    decision = controller.prepare_call(
        {"id": "v1", "function": {"name": "view_image", "arguments": '{"path":"image.png"}'}}
    )
    completion = controller.record_result(decision, result)
    assert completion.mcp_images() == images
    assert completion.model_message()["content"] == text
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "v1",
                    "type": "function",
                    "function": {"name": "view_image", "arguments": '{"path":"image.png"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "v1", "content": result},
    ]
    promoted = promote_history(history, vision = True)
    assert any(
        isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
        for m in promoted
    )
    text_only = promote_history(history, vision = False)
    assert all(not isinstance(m.get("content"), list) for m in text_only)
    assert "__MCP_IMAGES__" not in json.dumps(text_only)


@pytest.mark.parametrize("name", ["python", "terminal", "web_search"])
def test_other_tools_cannot_forge_image_input(workdir, name):
    result = view_image("image.png", str(workdir))
    history = [{"role": "tool", "name": name, "content": result}]
    assert not any(
        isinstance(m.get("content"), list) for m in promote_history(history, vision = True)
    )


@pytest.mark.parametrize("absolute", [False, True])
def test_relative_and_absolute_paths(workdir, absolute):
    path = str(workdir / "image.png") if absolute else "image.png"
    assert split_images(view_image(path, str(workdir)))[1]


@pytest.mark.parametrize("path", ["/mnt/data/image.png", "/workspace/image.png"])
def test_code_interpreter_prefix_maps_to_workdir(workdir, path):
    assert split_images(view_image(path, str(workdir)))[1]
    assert view_image("/mnt/data/../../etc/passwd", str(workdir)).startswith("Error:")


@pytest.mark.parametrize("bypass", [False, True])
def test_cannot_read_outside_even_with_full_access(workdir, bypass):
    outside = workdir.parent / "outside.png"
    Image.new("RGB", (10, 10)).save(outside)
    for path in (str(outside), "../outside.png"):
        result = tools.execute_tool("view_image", {"path": path}, disable_sandbox = bypass)
        assert result.startswith("Error:")
        assert not split_images(result)[1]


def test_symlink_escape_and_other_session(workdir):
    other = workdir.parent / "other-chat"
    other.mkdir()
    Image.new("RGB", (10, 10)).save(other / "secret.png")
    (workdir / "link").symlink_to(other, target_is_directory = True)
    assert view_image("link/secret.png", str(workdir)).startswith("Error:")
    (workdir / "internal.png").symlink_to(workdir / "image.png")
    assert split_images(view_image("internal.png", str(workdir)))[1]


@pytest.mark.parametrize("path", ["missing.png", "", None, 3, "bad\0.png", "."])
def test_invalid_paths_fail_without_image(workdir, path):
    assert view_image(path, str(workdir)).startswith("Error:")


def test_invalid_large_and_pixel_bomb_files(workdir, monkeypatch):
    from core.inference import view_image as module

    (workdir / "invalid.png").write_text("not an image")
    assert view_image("invalid.png", str(workdir)).startswith("Error:")
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 10)
    assert "file limit" in view_image("image.png", str(workdir))
    monkeypatch.undo()
    from core.inference import mcp_images

    monkeypatch.setattr(mcp_images, "MAX_IMAGE_PIXELS", 100)
    assert "megapixel limit" in view_image("image.png", str(workdir))


def test_cancelled_read(workdir):
    cancel = threading.Event()
    cancel.set()
    assert "cancelled" in view_image("image.png", str(workdir), cancel)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "os.mkfifo is POSIX-only")
def test_fifo_is_refused_without_blocking(workdir):
    os.mkfifo(workdir / "pipe.png")
    assert view_image("pipe.png", str(workdir)).startswith("Error:")


def test_image_is_bounded_and_oriented(workdir):
    image = Image.new("RGB", (1600, 800), "blue")
    exif = image.getexif()
    exif[274] = 6
    image.save(workdir / "rotated.jpg", exif = exif)
    _, images = split_images(view_image("rotated.jpg", str(workdir)))
    with Image.open(io.BytesIO(base64.b64decode(images[0]["data"]))) as decoded:
        assert decoded.height > decoded.width
        assert max(decoded.size) <= 1024


def test_tool_available_only_for_vision_and_explicit_opt_in(monkeypatch):
    from routes import inference

    monkeypatch.setattr(inference, "_enabled_agent_skills", lambda: [])
    monkeypatch.setattr(inference, "_thread_has_conversation_archive", lambda _: False)
    payload = SimpleNamespace(
        enabled_tools = ["view_image"], rag_scope = None, bypass_permissions = False
    )
    for vision, on, enabled, expected in [
        (True, True, ["view_image"], True),
        (False, True, ["view_image"], False),
        (True, False, ["view_image"], False),
        (True, True, [], False),
        (True, True, None, True),
    ]:
        payload.enabled_tools = enabled
        catalog = asyncio.run(
            inference._select_request_tools(
                payload, tools_on = on, mcp_allowed = False, supports_vision = vision
            )
        )
        assert any(t["function"]["name"] == "view_image" for t in catalog) == expected


def test_image_can_be_viewed_again_after_python_updates_it(workdir):
    controller = ToolLoopController(tools = [VIEW_IMAGE_TOOL, tools.PYTHON_TOOL])
    call = {"id": "v", "function": {"name": "view_image", "arguments": '{"path":"image.png"}'}}
    first = controller.prepare_call(call)
    controller.record_result(first, view_image("image.png", str(workdir)))
    assert controller.prepare_call(call).action == "duplicate"
    edit = controller.prepare_call(
        {"id": "p", "function": {"name": "python", "arguments": '{"code":"update_image()"}'}}
    )
    Image.new("RGB", (120, 80), "blue").save(workdir / "image.png")
    controller.record_result(edit, "Updated image.")
    assert controller.prepare_call(call).action == "execute"
    result = controller.record_result(
        controller.prepare_call(call), view_image("image.png", str(workdir))
    )
    with Image.open(io.BytesIO(base64.b64decode(result.mcp_images()[0]["data"]))) as decoded:
        assert decoded.getpixel((20, 20)) == (0, 0, 255)


@pytest.mark.parametrize("vision,mode", [(True, "auto"), (True, "off"), (False, "off")])
def test_anthropic_route_gates_image_viewer(monkeypatch, vision, mode):
    from routes import inference
    from models.inference import AnthropicMessagesRequest

    calls = []

    def plain(**kwargs):
        calls.append(("plain", kwargs))
        yield "ok"

    def tool_chat(**kwargs):
        calls.append(("tools", kwargs))
        yield {"type": "content", "text": "ok"}

    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = vision,
        supports_tools = True,
        model_identifier = "test-model",
        context_length = 4096,
        count_chat_tokens = lambda *a, **kw: 2,
        generate_chat_completion = plain,
        generate_chat_completion_with_tools = tool_chat,
        calls = calls,
    )
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
    payload = AnthropicMessagesRequest(
        max_tokens = 16,
        messages = [{"role": "user", "content": "hi"}],
        enable_tools = True,
        enabled_tools = ["view_image"],
        permission_mode = mode,
    )
    asyncio.run(inference.anthropic_messages(payload, request = None, current_subject = "t"))
    assert backend.calls
    assert backend.calls[0][0] == ("tools" if vision else "plain")
    if vision:
        assert [t["function"]["name"] for t in backend.calls[0][1]["tools"]] == ["view_image"]
    for enabled in (None, ["view_image"]):
        catalog = inference._select_anthropic_server_tools(
            tools.ALL_TOOLS, set(), enabled, supports_vision = vision
        )
        assert any(t["function"]["name"] == "view_image" for t in catalog) == vision


@pytest.mark.parametrize("enabled", [["view_image"], None, []])
def test_cold_codex_resolves_vision_before_offering_viewer(monkeypatch, enabled):
    from routes import inference
    from core.inference import openai_codex_auth as auth, openai_codex_client as client
    from models.inference import ChatCompletionRequest

    slug = "gpt-5.7-nova"
    monkeypatch.setattr(
        inference.providers_db,
        "get_provider",
        lambda _: {
            "id": "codex-1",
            "provider_type": "openai_codex",
            "is_enabled": True,
            "base_url": auth.OPENAI_CODEX_API_BASE,
            "display_name": "test",
            "models": [slug],
        },
    )
    monkeypatch.setattr(auth, "load_oauth_bundle", lambda _: {"account_id": "acct-1"})

    async def resolve(*args, **kwargs):
        return "token", "acct-1"

    monkeypatch.setattr(auth, "resolve_access", resolve)
    monkeypatch.setattr(client, "subscription_catalog_matches_account", lambda *args: True)
    monkeypatch.setattr(client, "subscription_catalog_known", lambda *args: False)
    monkeypatch.setattr(client, "subscription_catalog_stale", lambda *args: False)
    monkeypatch.setattr(client, "saved_models_proven_for", lambda *args: True)
    listed, refreshes, selected = {}, [], []
    monkeypatch.setattr(
        client, "offered_subscription_model", lambda provider, model: listed.get(model)
    )

    async def ensure(*args, **kwargs):
        refreshes.append(True)
        listed[slug] = {"id": slug, "vision": True, "listed": True}
        return {slug}

    monkeypatch.setattr(client, "ensure_subscription_models", ensure)
    selector = inference._select_request_tools

    class Captured(Exception):
        pass

    async def capture(*args, **kwargs):
        selected.extend(await selector(*args, **kwargs))
        raise Captured

    monkeypatch.setattr(inference, "_select_request_tools", capture)

    async def connected():
        return False

    request = SimpleNamespace(
        headers = {}, state = SimpleNamespace(skip_api_monitor = True), is_disconnected = connected
    )
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "Inspect my screenshot."}],
        provider_id = "codex-1",
        external_model = slug,
        stream = True,
        enable_tools = True,
        enabled_tools = enabled,
        mcp_enabled = False,
    )
    with pytest.raises(Captured):
        asyncio.run(inference._proxy_to_external_provider(payload, request, current_subject = "t"))
    assert bool(refreshes) == (enabled != [])
    assert any(t["function"]["name"] == "view_image" for t in selected) == (enabled != [])


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason = "Linux descriptor inventory")
def test_directory_errors_do_not_leak_file_descriptors(workdir):
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(20):
        assert view_image(".", str(workdir)).startswith("Error:")
    assert len(os.listdir("/proc/self/fd")) == before


def test_viewing_an_image_does_not_license_rerunning_a_write(workdir):
    controller = ToolLoopController(tools = [VIEW_IMAGE_TOOL, tools.PYTHON_TOOL])
    write = {"id": "p", "function": {"name": "python", "arguments": '{"code":"append()"}'}}
    view = {"id": "v", "function": {"name": "view_image", "arguments": '{"path":"image.png"}'}}
    controller.record_result(controller.prepare_call(write), "Appended.")
    controller.record_result(controller.prepare_call(view), view_image("image.png", str(workdir)))
    assert controller.prepare_call(write).action == "duplicate"
