# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from fastmcp.exceptions import ToolError

from hub.utils.host_paths import cache_reference

from mcp_server import create_studio_mcp
from studio_mcp import checkpoints
from studio_mcp.caller import Caller

from .mcp_harness import call_tool, fake_studio, served

CONFIG = {
    "model_name": "unsloth/Qwen3-0.6B",
    "dataset": "mlabonne/FineTome-100k",
    "max_steps": 30,
    "hf_token": "hf_cfg",
}
STARTED = {"job_id": "job-7", "status": "queued", "message": "Training job queued", "error": None}

PAYLOADS = {
    ("POST", "/api/train/start"): STARTED,
    ("POST", "/api/train/diffusion/start"): {
        "job_id": "diff-1",
        "status": "queued",
        "output_dir": "/srv/out",
    },
}


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, name, args):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, name, args)


def _bodies(studio, path):
    return [json.loads(b) for _m, p, _h, b in studio.state.calls if p == path]


def test_start_forwards_the_config_as_sent(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "start_training", {"config": CONFIG})
    assert result["structuredContent"] == {
        "job_id": "job-7",
        "status": "queued",
        "message": "Training job queued",
    }
    assert _bodies(studio, "/api/train/start") == [CONFIG]
    headers = studio.state.calls[0][2]
    assert "x-unsloth-hf-token" not in headers


def test_an_already_active_200_is_a_tool_error(monkeypatch):
    refused = {
        "job_id": "",
        "status": "error",
        "message": "Training is already in progress. Stop current training before starting a new one.",
        "error": "Training already active",
    }
    studio = _studio({("POST", "/api/train/start"): lambda request, body: refused})
    result = _call(monkeypatch, studio, "start_training", {"config": CONFIG})
    assert result["isError"] is True
    assert result["content"][0]["text"] == refused["message"]


def test_start_training_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["start_training"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None


DIFFUSION = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "data_dir": "my-photos",
    "output_dir": "my-lora",
    "train_steps": 500,
}


def test_kind_routes_to_the_matching_start_route(monkeypatch):
    studio = _studio(
        {
            ("POST", "/api/train/diffusion/start"): lambda request, body: {
                "job_id": "diff-1",
                "status": "queued",
                "output_dir": "/srv/unsloth/outputs/my-lora",
            }
        }
    )
    result = _call(
        monkeypatch, studio, "start_training", {"config": DIFFUSION, "kind": "diffusion"}
    )
    assert result["structuredContent"] == {"job_id": "diff-1", "status": "queued", "message": None}
    assert "/srv" not in json.dumps(result)
    assert _bodies(studio, "/api/train/diffusion/start") == [DIFFUSION]
    assert _bodies(studio, "/api/train/start") == []


def test_a_diffusion_start_during_an_llm_run_is_refused(monkeypatch):
    from fastapi.responses import JSONResponse

    def busy(request, body):
        return JSONResponse(
            {"detail": "An LLM training run is active. Stop it before starting image training."},
            status_code = 409,
        )

    studio = _studio({("POST", "/api/train/diffusion/start"): busy})
    result = _call(
        monkeypatch, studio, "start_training", {"config": DIFFUSION, "kind": "diffusion"}
    )
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "An LLM training run is active. Stop it before starting image training. (HTTP 409)"
    )


# ---------------------------------------------------------------- runs and checkpoints

OUT = "/srv/unsloth/outputs"
CHECKPOINTS = {
    "outputs_dir": OUT,
    "models": [
        {
            "name": "qwen-lora",
            "checkpoints": [
                {"display_name": "qwen-lora", "path": f"{OUT}/qwen-lora", "loss": 0.9},
                {
                    "display_name": "checkpoint-30",
                    "path": f"{OUT}/qwen-lora/checkpoint-30",
                    "loss": 1.1,
                },
            ],
            "base_model": "unsloth/Qwen3-0.6B",
            "peft_type": "LORA",
            "lora_rank": 16,
            "is_quantized": True,
        },
        {
            "name": "qwen-lora-2",
            "checkpoints": [
                {
                    "display_name": "checkpoint-30",
                    "path": f"{OUT}/qwen-lora-2/checkpoint-30",
                    "loss": 1.0,
                }
            ],
            "base_model": "unsloth/Qwen3-0.6B",
        },
    ],
}


def _run(run_id, folder):
    return {
        "id": run_id,
        "status": "completed",
        "model_name": "unsloth/Qwen3-0.6B",
        "dataset_name": "mlabonne/FineTome-100k",
        "started_at": "2026-10-08T10:00:00Z",
        "final_step": 30,
        "final_loss": 0.9,
        "output_dir": cache_reference(f"{OUT}/{folder}"),
        "can_resume": False,
    }


RUNS = {"runs": [_run("run-1", "qwen-lora"), _run("run-2", "qwen-lora-2")], "total": 2}
DETAIL = {
    "run": _run("run-1", "qwen-lora"),
    "config": {
        "model_name": "unsloth/Qwen3-0.6B",
        "learning_rate": 0.0002,
        "max_steps": 30,
        "output_dir": "/srv/out",
        "hf_token": "hf_secret",
    },
    "metrics": {"step_history": [1, 2], "loss_history": [1.5, 1.2]},
}
RUN_PAYLOADS = {
    ("GET", "/api/models/checkpoints"): CHECKPOINTS,
    ("GET", "/api/train/runs"): RUNS,
    ("GET", "/api/train/runs/run-1"): DETAIL,
}


def _runs_studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in RUN_PAYLOADS.items()
    }
    routes.update(overrides or {})
    return fake_studio(routes)


def test_runs_name_their_folder_and_checkpoints_by_name(monkeypatch):
    studio = _runs_studio()
    result = _call(monkeypatch, studio, "list_training_runs", {"include_checkpoints": True})
    out = result["structuredContent"]
    assert [(r["id"], r["run_folder"]) for r in out["runs"]] == [
        ("run-1", "qwen-lora"),
        ("run-2", "qwen-lora-2"),
    ]
    assert [c["name"] for c in out["checkpoints"]] == [
        "qwen-lora",
        "qwen-lora/checkpoint-30",
        "qwen-lora-2/checkpoint-30",
    ]
    assert out["checkpoints"][0] == {
        "run": "qwen-lora",
        "name": "qwen-lora",
        "loss": 0.9,
        "base_model": "unsloth/Qwen3-0.6B",
        "peft_type": "LORA",
        "lora_rank": 16,
        "is_quantized": True,
    }
    assert OUT not in json.dumps(result)


def test_names_resolve_back_to_paths_in_process():
    studio = _runs_studio()
    caller = Caller(
        token = "sk-unsloth-test",
        account_id = "owner",
        direct_local = False,
        public_base = "http://h",
        studio_app = studio,
    )
    found = {c.name: c.path for c in asyncio.run(checkpoints.list_checkpoints(caller))[0]}
    assert found == {
        "qwen-lora": f"{OUT}/qwen-lora",
        "qwen-lora/checkpoint-30": f"{OUT}/qwen-lora/checkpoint-30",
        "qwen-lora-2/checkpoint-30": f"{OUT}/qwen-lora-2/checkpoint-30",
    }
    for name, path in found.items():
        assert asyncio.run(checkpoints.resolve(caller, name)).path == path
    with pytest.raises(
        ToolError, match = "Available: qwen-lora, qwen-lora-2/checkpoint-30, qwen-lora/checkpoint-30"
    ):
        asyncio.run(checkpoints.resolve(caller, "missing"))


def test_a_run_detail_keeps_only_scalar_settings(monkeypatch):
    studio = _runs_studio()
    result = _call(monkeypatch, studio, "list_training_runs", {"run_id": "run-1"})
    run = result["structuredContent"]["run"]
    assert run["config"] == {
        "model_name": "unsloth/Qwen3-0.6B",
        "learning_rate": 0.0002,
        "max_steps": 30,
    }
    assert run["loss_history"] == [1.5, 1.2]
    assert run["run_folder"] == "qwen-lora"
    assert result["structuredContent"]["checkpoints"] is None


def test_limit_is_clamped(monkeypatch):
    queries = []
    studio = _runs_studio(
        {
            ("GET", "/api/train/runs"): lambda request, body: queries.append(
                dict(request.query_params)
            )
            or RUNS
        }
    )
    _call(monkeypatch, studio, "list_training_runs", {"limit": 10_000, "offset": -5})
    _call(monkeypatch, studio, "list_training_runs", {"limit": 0})
    assert queries == [{"limit": "200", "offset": "0"}, {"limit": "1", "offset": "0"}]


def test_list_training_runs_is_read_only():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["list_training_runs"]
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
