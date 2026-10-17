# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No MCP tool may hand a host path to the agent. Every tool gets a case here: the fake Studio answers each route with its payload poisoned by sentinel paths, and nothing the tool returns may contain one."""

import asyncio
import json
from typing import Any, Optional

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from fastmcp import FastMCP
from pydantic import ValidationError

from hub.utils.host_paths import response_leaks_host_path
from mcp_server import create_studio_mcp
from studio_mcp.caller import current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.outputs import RouteText, ToolOutput

from .mcp_harness import (
    OUT,
    PNG_URL,
    RUN_CHECKPOINTS,
    SENTINEL_ROOTS,
    WAV_INPUT,
    call_tool,
    fake_studio,
    poison,
    served,
)
from .test_mcp_tools_audio import PAYLOADS as AUDIO_PAYLOADS
from .test_mcp_tools_audio import TRANSCRIPT, VERBOSE
from .test_mcp_tools_cancel import PAYLOADS as CANCEL_PAYLOADS
from .test_mcp_tools_datasets import PAYLOADS as DATASETS_PAYLOADS
from .test_mcp_tools_datasets_images import PAYLOADS as IMAGE_DATASET_PAYLOADS
from .test_mcp_tools_download import TOOL_PAYLOADS as DOWNLOAD_PAYLOADS
from .test_mcp_train_to_use import _routes as TRAIN_TO_USE_ROUTES
from .test_mcp_tools_datasets_upload import PAYLOADS as UPLOAD_PAYLOADS
from .test_mcp_tools_images import PAYLOADS as IMAGES_PAYLOADS
from .test_mcp_tools_loading import PAYLOADS as LOADING_PAYLOADS
from .test_mcp_tools_plan_load import _world as plan_world
from .test_mcp_tools_models import PAYLOADS as MODELS_PAYLOADS
from .test_mcp_tools_recipe import PAYLOADS as RECIPE_PAYLOADS
from .test_mcp_tools_status import PAYLOADS as STATUS_PAYLOADS
from .test_mcp_tools_text import PAYLOADS as TEXT_PAYLOADS
from .test_mcp_tools_training import PAYLOADS as TRAINING_PAYLOADS
from .test_mcp_tools_training import RUN_PAYLOADS
from .test_mcp_tools_video import PAYLOADS as VIDEO_PAYLOADS

TRANSCRIBE_PAYLOADS = {
    ("POST", "/v1/audio/transcriptions"): VERBOSE,
    ("POST", "/v1/audio/translations"): TRANSCRIPT,
}
EXPORT_PAYLOADS = {
    ("GET", "/api/models/checkpoints"): {
        **RUN_CHECKPOINTS,
        "models": RUN_CHECKPOINTS["models"][:1],
    },
    ("GET", "/api/export/status"): {
        "last_op_seq": 1,
        "current_checkpoint": f"{OUT}/qwen-lora",
        "last_op_output_path": f"{OUT}/x",
    },
    ("POST", "/api/export/load-checkpoint"): {
        "success": True,
        "message": f"Loaded {OUT}/qwen-lora",
    },
    ("POST", "/api/export/export/gguf"): {"success": True, "message": f"Saved to {OUT}/x"},
}

# tool name -> (fake Studio routes, tool arguments), or a list of them
CASES: dict[str, Any] = {
    "studio_status": (STATUS_PAYLOADS, {}),
    "list_models": (MODELS_PAYLOADS, {"model": "unsloth/Qwen3-0.6B"}),
    "load_model": [
        (LOADING_PAYLOADS, {"model": "unsloth/Llama-3.2-1B-Instruct-GGUF"}),
        (TRAIN_TO_USE_ROUTES([f"{OUT}/qwen-lora"]), {"checkpoint": "qwen-lora"}),
        (TRAIN_TO_USE_ROUTES([f"{OUT}/qwen-lora"]), {"export": "qwen-GGUF", "variant": "Q4_K_M"}),
    ],
    "unload_model": (LOADING_PAYLOADS, {}),
    "plan_load": (
        plan_world("off", ["unsloth/Qwen3-0.6B-GGUF"]),
        {"model": "unsloth/Qwen3-0.6B-GGUF"},
    ),
    "download_model": (DOWNLOAD_PAYLOADS, {"model": "unsloth/Qwen3-0.6B-GGUF"}),
    "chat": (TEXT_PAYLOADS, {"prompt": "hi"}),
    "embed": (TEXT_PAYLOADS, {"texts": ["a", "b"]}),
    "system_one": (TEXT_PAYLOADS, {"state": "x", "questions": {"urgent": {"type": "noul"}}}),
    "generate_image": (IMAGES_PAYLOADS, {"prompt": "a red fox"}),
    "generate_video": (VIDEO_PAYLOADS, {"prompt": "waves"}),
    "get_job": [
        (VIDEO_PAYLOADS, {"kind": "video"}),
        (VIDEO_PAYLOADS, {"kind": "video", "id": "video-1"}),
        (RECIPE_PAYLOADS, {"kind": "recipe", "id": "job-1", "rows": 2}),
        (DOWNLOAD_PAYLOADS, {"kind": "download", "id": "a/b::Q4_K_M"}),
    ],
    "start_training": [
        (TRAINING_PAYLOADS, {"config": {"model_name": "m"}}),
        (TRAINING_PAYLOADS, {"config": {"base_model": "m", "data_dir": "d"}, "kind": "diffusion"}),
    ],
    "list_training_runs": (RUN_PAYLOADS, {"run_id": "run-1", "include_checkpoints": True}),
    "export_model": [
        (EXPORT_PAYLOADS, {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "out"}),
        (EXPORT_PAYLOADS, {"checkpoint": "missing", "format": "gguf", "save_directory": "out"}),
    ],
    "cancel": [
        (CANCEL_PAYLOADS, {"kind": "training"}),
        (CANCEL_PAYLOADS, {"kind": "recipe", "id": "rec-1"}),
        (DOWNLOAD_PAYLOADS, {"kind": "model_download", "id": "a/b::Q4_K_M"}),
    ],
    "run_recipe": (RECIPE_PAYLOADS, {"recipe": {"columns": [{"name": "q"}]}}),
    "datasets": [
        (DATASETS_PAYLOADS, {"action": "list"}),
        (DATASETS_PAYLOADS, {"action": "check_format", "name": "a/b"}),
        (DATASETS_PAYLOADS, {"action": "status", "repo_id": "a/b"}),
        (UPLOAD_PAYLOADS, {"action": "upload", "rows": [{"text": "hi"}]}),
        (UPLOAD_PAYLOADS, {"action": "upload", "data_base64": "YSxi", "filename": "a.csv"}),
        (
            IMAGE_DATASET_PAYLOADS,
            {"action": "upload_images", "name": "p", "images": [PNG_URL], "captions": ["c"]},
        ),
    ],
    "transcribe": (TRANSCRIBE_PAYLOADS, {"audio": WAV_INPUT, "timestamps": True, "language": "en"}),
    "generate_audio": (AUDIO_PAYLOADS, {"workflow": "clone", "text": "hi", "reference": WAV_INPUT}),
}

# Output keys that carry model-written text, which is the model's to say and is never rewritten.
MODEL_TEXT = {"chat": {"text"}, "transcribe": {"text"}, "get_job": {"data_rows"}}


def leaks(result: dict, ignore: frozenset = frozenset()) -> Optional[str]:
    payloads = [result.get("structuredContent")]
    for item in result.get("content") or []:
        if item.get("type") == "text":
            try:
                payloads.append(json.loads(item["text"]))
            except ValueError:
                payloads.append(item["text"])
    for payload in payloads:
        found = response_leaks_host_path(payload, roots = SENTINEL_ROOTS, ignore = ignore)
        if found is not None:
            return found
    return None


def run_case(monkeypatch, mcp, name: str, routes: dict, args: dict) -> dict:
    with TestClient(served(mcp, fake_studio(poison(routes)), monkeypatch = monkeypatch)) as http:
        return call_tool(http, name, args)


def test_every_registered_tool_has_a_host_path_case():
    names = {tool.name for tool in asyncio.run(create_studio_mcp().list_tools())}
    assert set(MODEL_TEXT) <= set(CASES) == names
    assert len(names) == len(CASES)


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
    assert _Status.model_json_schema()["additionalProperties"] is False
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
    assert status.model_dump() == {
        "message": "Saved to <path>",
        "error": "Failed at <path>",
        "notes": ["cache <path>"],
        "url": "http://127.0.0.1:8888/v1/audio/gallery/abc/file",
        "text": "The file lives at /etc/hosts on most systems.",
    }


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


ROUTES = {("GET", "/api/train/status"): {"message": "Training", "output_dir": "/srv/out"}}


def test_the_checker_sees_a_leak_but_typed_outputs_and_route_errors_do_not_leak(monkeypatch):
    # Positive control: a passed-through route payload is caught.
    assert leaks(run_case(monkeypatch, _probe_mcp(), "passthrough", ROUTES, {})) is not None
    typed = run_case(monkeypatch, _probe_mcp(), "typed", ROUTES, {})
    assert typed["structuredContent"]["message"].startswith("Training")
    assert leaks(typed) is None

    missing = JSONResponse({"detail": poison("Run not found")}, status_code = 404)
    failed = run_case(
        monkeypatch, _probe_mcp(), "failing", {("GET", "/api/train/missing"): missing}, {}
    )
    assert failed["isError"] is True
    assert "Run not found" in failed["content"][0]["text"]
    assert leaks(failed) is None
