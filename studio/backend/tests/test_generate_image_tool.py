# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generate_image chat tool renders on the loaded image model, as the calling account."""

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from auth import policy
from core.inference import (
    diffusion_engine_router,
    generate_image_tool as tool,
    gpu_arbiter,
    media_chat_tools,
    media_keepwarm,
    tools,
)
from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
from core.inference.tool_loop_controller import (
    ToolLoopController,
    status_for_tool,
    strip_result_for_model,
)
from hub.services.models import account_access as access
from models.inference import DiffusionGenerateRequest, DiffusionGenerateResponse, GalleryImage
from routes import inference
from state import active_generations
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
STATUS = {"loaded": True, "repo_id": "org/public-model", "family": "z-image", "base_repo": None}


class FakeEngine:
    def __init__(self, status = None):
        self.status_dict = dict(STATUS if status is None else status)
        self.calls: list[dict] = []
        self.release = threading.Event()
        self.started = threading.Event()
        self.block = False
        self.active = False
        self.cancels: list[str] = []

    def status(self):
        return dict(self.status_dict)

    def generate_progress(self):
        return {"active": self.active}

    def cancel_generate(self, expected_account = None):
        self.release.set()
        return True

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if self.block:
            self.release.wait(20)
            raise RuntimeError(DIFFUSION_CANCELLED_MSG)
        size = (kwargs.get("width") or 1024, kwargs.get("height") or 1024)
        return {
            "images": [Image.new("RGB", size, "blue")],
            "seed": 7,
            "repo_id": "org/public-model",
        }


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_generation_accounts", {})
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: True)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    monkeypatch.setattr(tool, "_STOP_WAIT_S", 5.0)
    monkeypatch.setattr(inference, "_enabled_agent_skills", lambda: [])
    monkeypatch.setattr(inference, "_thread_has_conversation_archive", lambda _: False)
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setitem(media_keepwarm._ENGINES, gpu_arbiter.DIFFUSION, lambda: fake)
    monkeypatch.setitem(media_keepwarm._ENGINES, gpu_arbiter.VIDEO, lambda: None)
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: fake)
    return fake


@pytest.fixture
def server_loop(monkeypatch):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    monkeypatch.setattr(media_chat_tools, "_server_loop", loop)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


@pytest.fixture
def keepwarm(monkeypatch):
    counts = {"begin": 0, "end": 0}

    async def begin(owner):
        assert owner == gpu_arbiter.DIFFUSION
        counts["begin"] += 1

    def end(owner, *, counted = True):
        counts["end"] += 1

    monkeypatch.setattr(media_keepwarm, "begin_request", begin)
    monkeypatch.setattr(media_keepwarm, "end_request", end)
    return counts


def _alice_loaded(monkeypatch):
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", ALICE.account_id)


def _in_thread(account, fn, *args, **kwargs):
    """Run like the chat tool loop does: on a worker thread bound to the caller's account."""
    out = {}
    thread = threading.Thread(target = lambda: out.update(r = run_as(account, fn, *args, **kwargs)))
    thread.start()
    thread.join(60)
    return out["r"]


def _select(enabled, *, tools_on = True):
    payload = SimpleNamespace(enabled_tools = enabled, rag_scope = None, bypass_permissions = False)

    async def go():
        selected = await inference._select_request_tools(
            payload, tools_on = tools_on, mcp_allowed = False, supports_vision = True
        )
        return selected, asyncio.get_running_loop()

    selected, loop = asyncio.run(go())
    return {t["function"]["name"]: t for t in selected}, loop


def test_offered_only_while_a_usable_text_to_image_model_is_loaded(engine, monkeypatch):
    media_chat_tools._server_loop = None
    names, loop = _select(["generate_image"])
    assert "generate_image" in names and media_chat_tools._server_loop is loop
    assert "generate_image" not in _select(["web_search"])[0]
    assert "generate_image" not in _select(None)[0]
    assert "generate_image" not in _select(["generate_image"], tools_on = False)[0]
    # A video model isn't loaded, so its tool isn't offered alongside.
    assert "generate_video" not in _select(["generate_image", "generate_video"])[0]

    engine.status_dict["workflows"] = ["edit"]
    assert "generate_image" not in _select(["generate_image"])[0]
    engine.status_dict["workflows"] = ["txt2img", "edit"]
    assert "generate_image" in _select(["generate_image"])[0]

    engine.status_dict["loaded"] = False
    assert "generate_image" not in _select(["generate_image"])[0]
    engine.status_dict["loaded"] = True

    monkeypatch.setattr(access, "resident_hidden", lambda *a, **k: True)
    assert "generate_image" not in _select(["generate_image"])[0]


def test_a_managed_account_refused_the_model_is_not_offered_it(engine, monkeypatch):
    def refuse(status, modality):
        raise HTTPException(status_code = 403, detail = "not allowed")

    _alice_loaded(monkeypatch)
    assert "generate_image" in run_as(ALICE, _select, ["generate_image"])[0]
    monkeypatch.setattr(access, "require_media_generation_access", refuse)
    assert "generate_image" in _select(["generate_image"])[0]  # the owner is never refused
    assert "generate_image" not in run_as(ALICE, _select, ["generate_image"])[0]


def test_unselected_loop_is_not_remembered(engine):
    media_chat_tools._server_loop = None
    _select(["web_search"])
    assert media_chat_tools._server_loop is None


@pytest.mark.parametrize("supports", [True, False])
def test_negative_prompt_follows_the_model_family(supports):
    status = {**STATUS, "supports_negative_prompt": supports, "prompt_hint": "Use tags."}
    schema = tool.generate_image_schema(status)["function"]
    assert ("negative_prompt" in schema["parameters"]["properties"]) is supports
    assert schema["parameters"]["required"] == ["prompt"]
    assert schema["description"].endswith(" Use tags.")
    assert "Unsloth Studio" in schema["description"]


def test_managed_account_run_lands_in_its_own_gallery(
    engine, server_loop, keepwarm, monkeypatch, tmp_path
):
    _alice_loaded(monkeypatch)
    checked = []
    real_check = access.require_media_generation_access
    monkeypatch.setattr(
        access,
        "require_media_generation_access",
        lambda status, modality: (checked.append(modality), real_check(status, modality))[1],
    )
    workdir = tmp_path / "work"
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    result = _in_thread(
        ALICE,
        tools.execute_tool,
        "generate_image",
        {"prompt": "a  sloth\n on a branch", "width": 512, "height": 512},
        session_id = "s",
    )
    assert result.startswith("Generated a 512x512 image (seed 7)"), result
    assert "Prompt used: a sloth on a branch" in result
    assert checked and set(checked) == {"diffusion"}
    from core.inference import image_gallery

    alice_ids = run_as(ALICE, lambda: [p.stem for p in image_gallery.gallery_dir().glob("*.png")])
    owner_dir = run_as(OWNER, image_gallery.gallery_dir)
    assert len(alice_ids) == 1 and not list(owner_dir.glob("*.png"))
    saved = workdir / f"generated-image-{alice_ids[0][:8]}.png"
    assert saved.is_file() and Image.open(saved).size == (512, 512)
    assert keepwarm == {"begin": 1, "end": 1}


@pytest.mark.parametrize(
    "status, expected",
    [
        ({"generation_defaults": {"steps": 28, "guidance": 3.5}}, (28, 3.5)),
        # sd.cpp and MLX report no recipe; the family table decides from the base repo.
        ({"repo_id": "/models/my-flux", "base_repo": "black-forest-labs/FLUX.1-dev"}, (20, 3.5)),
    ],
)
def test_omitted_steps_and_guidance_follow_the_loaded_model(
    engine, server_loop, keepwarm, monkeypatch, tmp_path, status, expected
):
    engine.status_dict.update(status)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(tmp_path / "w"))
    result = _in_thread(
        OWNER, tools.execute_tool, "generate_image", {"prompt": "p"}, session_id = "s"
    )
    assert result.startswith("Generated a 1024x1024 image"), result
    assert (engine.calls[0]["steps"], engine.calls[0]["guidance"]) == expected


def test_the_core_runs_under_the_account_bound_in_the_coroutine(
    engine, server_loop, keepwarm, monkeypatch, tmp_path
):
    import utils.account_context as account_context

    bound = []
    real = account_context.arun_as

    def spy(account, awaitable):
        bound.append(account)
        return real(account, awaitable)

    monkeypatch.setattr(account_context, "arun_as", spy)
    _alice_loaded(monkeypatch)
    result = _in_thread(
        ALICE, tool.run_generate_image, {"prompt": "x"}, workdir = str(tmp_path), timeout = 30
    )
    assert result.startswith("Generated"), result
    assert bound == [ALICE]


def _fake_core(monkeypatch, behaviour):
    seen = []

    async def core(request):
        seen.append(request)
        return await behaviour(request)

    monkeypatch.setattr(inference, "_generate_diffusion_image_core", core)
    return seen


def _response(tmp_path, monkeypatch):
    from core.inference import image_gallery

    png = tmp_path / "abcdef0123456789.png"
    Image.new("RGB", (1024, 768)).save(png)
    monkeypatch.setattr(image_gallery, "image_path", lambda image_id: png)
    record = {
        "id": "abcdef0123456789",
        "url": "/x",
        "prompt": "p",
        "width": 1024,
        "height": 768,
        "steps": 9,
        "guidance": 0.0,
        "seed": 3,
        "created_at": 0.0,
    }
    return DiffusionGenerateResponse(images = [GalleryImage(**record)])


def test_omitted_settings_are_not_sent_and_sizes_snap(engine, server_loop, monkeypatch, tmp_path):
    response = _response(tmp_path, monkeypatch)

    async def ok(request):
        return response

    seen = _fake_core(monkeypatch, ok)
    work = tmp_path / "w"
    tool.run_generate_image({"prompt": "x", "width": 1030, "height": 100}, workdir = str(work))
    tool.run_generate_image({"prompt": "x", "negative_prompt": "blur"}, workdir = str(work))
    first, second = seen
    assert "steps" not in first.model_fields_set and "guidance" not in first.model_fields_set
    assert (first.width, first.height) == (1024, 256)
    assert first.model_fields_set == {"prompt", "width", "height"}
    assert second.negative_prompt is None  # the family has no negative prompt
    engine.status_dict["supports_negative_prompt"] = True
    tool.run_generate_image({"prompt": "x", "negative_prompt": "blur"}, workdir = str(work))
    assert seen[-1].negative_prompt == "blur"


def test_sentinels_stay_out_of_the_model_text(engine, server_loop, monkeypatch, tmp_path):
    response = _response(tmp_path, monkeypatch)

    async def ok(request):
        return response

    _fake_core(monkeypatch, ok)
    result = tool.run_generate_image({"prompt": "x"}, workdir = str(tmp_path / "w"))
    assert '\n__FILES__:[{"name": "generated-image-abcdef01.png"' in result
    assert result.endswith('\n__IMAGES__:["generated-image-abcdef01.png"]')
    assert (tmp_path / "w" / "generated-image-abcdef01.png").is_file()
    stripped = strip_result_for_model(result, "generate_image")
    assert "__FILES__" not in stripped and "__IMAGES__" not in stripped
    assert stripped.startswith("Generated a 1024x768 image (seed 3)")


@pytest.mark.parametrize(
    "error, expected",
    [
        (gpu_arbiter.GpuBusyForAnotherAccountError("chat", 1), tool.ERR_OTHER_ACCOUNT),
        (
            HTTPException(409, detail = {"error": "gpu_busy", "message": "busy"}),
            tool.ERR_OTHER_ACCOUNT,
        ),
        (HTTPException(400, detail = "Width too large for this model."), "Error: Width too large"),
        (HTTPException(409, detail = "Model was replaced."), "Error: Model was replaced."),
        (HTTPException(500, detail = "Image generation failed: OOM"), "Error: Image generation"),
        (HTTPException(404, detail = {"error": {"message": "Model not found"}}), "Error: Model not"),
        (HTTPException(409, detail = DIFFUSION_CANCELLED_MSG), tool.ERR_STOPPED),
        (RuntimeError("boom"), "Error: image generation failed: boom"),
    ],
)
def test_failures_become_error_strings(engine, server_loop, monkeypatch, tmp_path, error, expected):
    async def fail(request):
        raise error

    _fake_core(monkeypatch, fail)
    assert tool.run_generate_image({"prompt": "x"}, workdir = str(tmp_path)).startswith(expected)


def test_refusals_before_any_render(engine, server_loop, monkeypatch, tmp_path):
    run = lambda args: tool.run_generate_image(args, workdir = str(tmp_path))  # noqa: E731
    assert run({"prompt": "  "}) == tool.ERR_NO_PROMPT
    assert run("not a dict") == tool.ERR_NO_PROMPT
    engine.active = True
    assert run({"prompt": "x"}) == tool.ERR_BUSY
    engine.active = False
    engine.status_dict["workflows"] = ["edit"]
    assert run({"prompt": "x"}) == tool.ERR_EDIT_ONLY
    engine.status_dict["loaded"] = False
    assert run({"prompt": "x"}) == tool.ERR_NOT_LOADED
    monkeypatch.setattr(media_chat_tools, "_server_loop", None)
    assert run({"prompt": "x"}) == tool.ERR_NO_LOOP
    assert not engine.calls


@pytest.mark.parametrize("how", ["cancel", "timeout"])
def test_stop_and_timeout_stop_the_engine_too(
    engine, server_loop, keepwarm, monkeypatch, tmp_path, how
):
    _alice_loaded(monkeypatch)
    engine.block = True
    stopped = []

    def cancel_for(account_id):
        stopped.append(account_id)
        return engine.cancel_generate(expected_account = account_id)

    monkeypatch.setattr(diffusion_engine_router, "cancel_generation_for_account", cancel_for)
    cancel = threading.Event()
    if how == "cancel":
        threading.Thread(target = lambda: engine.started.wait(20) and cancel.set()).start()
    result = _in_thread(
        ALICE,
        tool.run_generate_image,
        {"prompt": "x"},
        workdir = str(tmp_path),
        cancel_event = cancel,
        timeout = 2 if how == "timeout" else 60,
    )
    assert result == (tool.ERR_STOPPED if how == "cancel" else tool.ERR_TIMEOUT)
    assert stopped and set(stopped) == {ALICE.account_id}
    # The run unwound before the tool returned: the keep-warm count is already closed.
    assert keepwarm == {"begin": 1, "end": 1}
    assert not os.listdir(tmp_path)


def test_a_render_that_finishes_inside_the_stop_window_is_still_returned(
    engine, server_loop, keepwarm, monkeypatch, tmp_path
):
    # The timeout lands at the last step: the engine finishes the image instead of cancelling.
    _alice_loaded(monkeypatch)
    finish = engine.generate

    def generate(**kwargs):
        engine.started.set()
        engine.release.wait(20)
        return finish(**kwargs)

    monkeypatch.setattr(engine, "generate", generate)
    monkeypatch.setattr(
        diffusion_engine_router,
        "cancel_generation_for_account",
        lambda account_id: engine.cancel_generate(expected_account = account_id),
    )
    result = _in_thread(
        ALICE, tool.run_generate_image, {"prompt": "x"}, workdir = str(tmp_path), timeout = 1
    )
    assert result.startswith("Generated a 1024x1024 image"), result
    assert [n for n in os.listdir(tmp_path) if n.startswith("generated-image-")]


def test_keepwarm_counts_once_per_tool_call_and_never_in_the_core(
    engine, server_loop, keepwarm, tmp_path
):
    assert media_keepwarm.owner_for_path("/api/inference/images/generate") == gpu_arbiter.DIFFUSION
    asyncio.run(inference._generate_diffusion_image_core(DiffusionGenerateRequest(prompt = "x")))
    assert keepwarm == {"begin": 0, "end": 0}
    for _ in range(2):
        tool.run_generate_image({"prompt": "x"}, workdir = str(tmp_path))
    assert keepwarm == {"begin": 2, "end": 2}


def test_registries_treat_both_media_tools_alike():
    from core.inference.tool_loop_controller import (
        _FILE_SENTINEL_TOOLS,
        _IMAGE_SENTINEL_TOOLS,
        REPEATABLE_TOOL_NAMES,
    )

    for name in ("generate_image", "generate_video"):
        assert not tools.is_potentially_unsafe_tool_call(name, {"prompt": "x"})
        assert name in _FILE_SENTINEL_TOOLS and name in REPEATABLE_TOOL_NAMES
        assert name in inference._ANTHROPIC_UNPROMPTED_SAFE_TOOLS
    assert "generate_image" in _IMAGE_SENTINEL_TOOLS
    assert "generate_video" not in _IMAGE_SENTINEL_TOOLS
    assert (
        status_for_tool("generate_image", {"prompt": "p" * 99}) == "Generating image: " + "p" * 60
    )
    assert status_for_tool("generate_video", {"prompt": "dog"}) == "Generating video: dog"


def test_anthropic_never_selects_media_tools(engine):
    picked = inference._select_anthropic_server_tools(
        tools.ALL_TOOLS, {"generate_image", "generate_video"}, ["generate_image", "web_search"]
    )
    assert [t["function"]["name"] for t in picked] == ["web_search"]
    assert not {t["function"]["name"] for t in tools.ALL_TOOLS} & media_chat_tools.MEDIA_CHAT_TOOLS


def test_the_same_prompt_twice_is_two_renders():
    controller = ToolLoopController(tools = [tool.generate_image_schema(STATUS)])
    call = {"id": "g", "function": {"name": "generate_image", "arguments": '{"prompt":"x"}'}}
    first = controller.prepare_call(call)
    controller.record_result(first, "Generated a 1024x1024 image.")
    assert controller.prepare_call(call).action == "execute"
