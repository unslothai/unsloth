# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No MCP tool may hand a host path to the agent. Every tool gets a case here: the fake Studio answers each route with its payload poisoned by sentinel paths, and nothing the tool returns may contain one."""

import asyncio
import json
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient
from fastmcp import FastMCP
from pydantic import ValidationError

from hub.utils.host_paths import response_leaks_host_path
from mcp_server import create_studio_mcp
from studio_mcp.caller import current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.outputs import RouteText, ToolOutput

from .mcp_harness import SENTINEL_ROOTS, call_tool, fake_studio, poison, served
from .test_mcp_tools_audio import PAYLOADS as AUDIO_PAYLOADS
from .test_mcp_tools_audio import TRANSCRIBE_PAYLOADS
from .test_mcp_tools_datasets import PAYLOADS as DATASETS_PAYLOADS
from .test_mcp_tools_images import PAYLOADS as IMAGES_PAYLOADS
from .test_mcp_tools_loading import PAYLOADS as LOADING_PAYLOADS
from .test_mcp_tools_models import PAYLOADS as MODELS_PAYLOADS
from .test_mcp_tools_recipe import PAYLOADS as RECIPE_PAYLOADS
from .test_mcp_tools_status import PAYLOADS as STATUS_PAYLOADS
from .test_mcp_tools_text import PAYLOADS as TEXT_PAYLOADS
from .test_mcp_tools_training import PAYLOADS as TRAINING_PAYLOADS
from .test_mcp_tools_video import PAYLOADS as VIDEO_PAYLOADS

# The direct-call tools from before forwarding. Each commit that replaces one removes it here and adds its case.
LEGACY_UNCHECKED = {
    "stop_training",
    "list_training_runs",
    "load_checkpoint",
    "export_gguf",
}

# tool name -> (fake Studio routes as {(method, path): payload}, tool arguments), or a list of them
CASES: dict[str, Any] = {
    "studio_status": (STATUS_PAYLOADS, {}),
    "list_models": (MODELS_PAYLOADS, {"model": "unsloth/Qwen3-0.6B"}),
    "load_model": (LOADING_PAYLOADS, {"model": "unsloth/Llama-3.2-1B-Instruct-GGUF"}),
    "unload_model": (LOADING_PAYLOADS, {}),
    "chat": (TEXT_PAYLOADS, {"prompt": "hi"}),
    "embed": (TEXT_PAYLOADS, {"texts": ["a", "b"]}),
    "system_one": (TEXT_PAYLOADS, {"state": "x", "questions": {"urgent": {"type": "noul"}}}),
    "generate_image": (IMAGES_PAYLOADS, {"prompt": "a red fox"}),
    "generate_video": (VIDEO_PAYLOADS, {"prompt": "waves"}),
    "get_job": [
        (VIDEO_PAYLOADS, {"kind": "video"}),
        (VIDEO_PAYLOADS, {"kind": "video", "id": "video-1"}),
        (RECIPE_PAYLOADS, {"kind": "recipe", "id": "job-1", "rows": 2}),
    ],
    "start_training": [
        (TRAINING_PAYLOADS, {"config": {"model_name": "m"}}),
        (TRAINING_PAYLOADS, {"config": {"base_model": "m", "data_dir": "d"}, "kind": "diffusion"}),
    ],
    "run_recipe": (RECIPE_PAYLOADS, {"recipe": {"columns": [{"name": "q"}]}}),
    "datasets": [
        (DATASETS_PAYLOADS, {"action": "list"}),
        (DATASETS_PAYLOADS, {"action": "check_format", "name": "a/b"}),
        (DATASETS_PAYLOADS, {"action": "status", "repo_id": "a/b"}),
    ],
    "transcribe": (
        TRANSCRIBE_PAYLOADS,
        {
            "audio": {"data_base64": "UklGRg==", "filename": "a.wav"},
            "timestamps": True,
            "language": "en",
        },
    ),
    "generate_audio": (
        AUDIO_PAYLOADS,
        {
            "workflow": "clone",
            "text": "hi",
            "reference": {"data_base64": "UklGRg==", "filename": "a.wav"},
        },
    ),
}

# Output keys that carry model-written text, which is the model's to say and is never rewritten.
MODEL_TEXT = {"chat": {"text"}, "transcribe": {"text"}, "get_job": {"data_rows"}}


def leaks(result: dict, ignore: frozenset = frozenset()) -> Optional[str]:
    found = response_leaks_host_path(
        result.get("structuredContent"), roots = SENTINEL_ROOTS, ignore = ignore
    )
    if found is not None:
        return found
    for item in result.get("content") or []:
        if item.get("type") == "text":
            try:
                payload = json.loads(item.get("text"))
            except (TypeError, ValueError):
                payload = item.get("text")
            found = response_leaks_host_path(payload, roots = SENTINEL_ROOTS, ignore = ignore)
            if found is not None:
                return found
    return None


def _poisoned_studio(routes: dict[tuple[str, str], Any]):
    def answer(payload):
        return lambda request, body: poison(payload)

    return fake_studio({key: answer(payload) for key, payload in routes.items()})


def run_case(monkeypatch, mcp, name: str, routes: dict, args: dict) -> dict:
    app = served(mcp, _poisoned_studio(routes), monkeypatch = monkeypatch)
    with TestClient(app) as http:
        return call_tool(http, name, args)


def _registered_names() -> set[str]:
    return {tool.name for tool in asyncio.run(create_studio_mcp().list_tools())}


def test_every_registered_tool_has_a_host_path_case():
    names = _registered_names()
    assert set(MODEL_TEXT) <= set(CASES)
    assert not (set(CASES) & LEGACY_UNCHECKED)
    assert LEGACY_UNCHECKED <= names, "a legacy tool was removed; drop it from LEGACY_UNCHECKED"
    assert names - LEGACY_UNCHECKED == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_host_path_in_tool_output(monkeypatch, name):
    cases = CASES[name] if isinstance(CASES[name], list) else [CASES[name]]
    for routes, args in cases:
        result = run_case(monkeypatch, create_studio_mcp(), name, routes, args)
        assert leaks(result, frozenset(MODEL_TEXT.get(name, ()))) is None, args


class _Status(ToolOutput):
    message: RouteText
    error: Optional[RouteText] = None
    notes: list[RouteText] = []
    url: str
    text: str


def test_tool_outputs_refuse_undeclared_fields():
    with pytest.raises(ValidationError):
        _Status(message = "ok", url = "http://127.0.0.1:8888/x", text = "hi", output_dir = "/srv/x")


def test_route_text_is_scrubbed_but_urls_and_model_text_are_not():
    status = _Status(
        message = f"Saved to {SENTINEL_ROOTS[0]}/run",
        error = f"Failed at {SENTINEL_ROOTS[1]}\\run",
        notes = [f"cache {SENTINEL_ROOTS[0]}/cache"],
        url = "http://127.0.0.1:8888/v1/audio/gallery/abc/file",
        text = "The file lives at /etc/hosts on most systems.",
    )
    dumped = status.model_dump()
    assert (
        response_leaks_host_path(
            {k: dumped[k] for k in ("message", "error", "notes")}, roots = SENTINEL_ROOTS
        )
        is None
    )
    assert dumped["message"].startswith("Saved to")
    assert dumped["url"] == "http://127.0.0.1:8888/v1/audio/gallery/abc/file"
    assert dumped["text"] == "The file lives at /etc/hosts on most systems."


def test_the_output_schema_forbids_extra_properties():
    assert _Status.model_json_schema()["additionalProperties"] is False


def _probe_mcp():
    mcp = FastMCP("probe")

    class Echo(ToolOutput):
        message: RouteText

    @mcp.tool
    async def passthrough() -> dict:
        return raise_for_route(await forward(current_caller(), "GET", "/api/train/status"))

    @mcp.tool
    async def typed() -> Echo:
        payload = raise_for_route(await forward(current_caller(), "GET", "/api/train/status"))
        return Echo(message = payload["message"])

    @mcp.tool
    async def failing() -> Echo:
        raise_for_route(await forward(current_caller(), "GET", "/api/train/missing"))

    return mcp


ROUTES = {
    ("GET", "/api/train/status"): {"message": "Training", "output_dir": "/srv/out"},
}


def test_the_checker_sees_a_passed_through_route_payload(monkeypatch):
    result = run_case(monkeypatch, _probe_mcp(), "passthrough", ROUTES, {})
    assert leaks(result) is not None


def test_a_typed_output_and_a_route_error_do_not_leak(monkeypatch):
    typed = run_case(monkeypatch, _probe_mcp(), "typed", ROUTES, {})
    assert typed["structuredContent"]["message"].startswith("Training")
    assert leaks(typed) is None

    def missing(request, body):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": poison("Run not found")}, status_code = 404)

    studio = fake_studio({("GET", "/api/train/missing"): missing})
    with TestClient(served(_probe_mcp(), studio, monkeypatch = monkeypatch)) as http:
        failed = call_tool(http, "failing")
    assert failed["isError"] is True
    assert "Run not found" in failed["content"][0]["text"]
    assert leaks(failed) is None
