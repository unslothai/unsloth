# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest
from fastapi.responses import JSONResponse
from fastmcp.exceptions import ToolError

from hub.utils.host_paths import cache_reference

from studio_mcp import checkpoints

from .mcp_harness import (
    CONFIG,
    DIFFUSION,
    OUT,
    RUN_CHECKPOINTS,
    bodies,
    fake_studio,
    make_caller,
    queries,
    run_tool,
)

STARTED = {"job_id": "job-7", "status": "queued", "message": "Training job queued", "error": None}

PAYLOADS = {
    ("POST", "/api/train/start"): STARTED,
    ("POST", "/api/train/diffusion/start"): {
        "job_id": "diff-1",
        "status": "queued",
        "output_dir": "/srv/unsloth/outputs/my-lora",
    },
}


def _start(
    monkeypatch,
    args,
    overrides = None,
):
    return run_tool(monkeypatch, {**PAYLOADS, **(overrides or {})}, "start_training", args)


def test_start_forwards_the_config_as_sent(monkeypatch):
    result, studio = _start(monkeypatch, {"config": CONFIG})
    (sent,) = bodies(studio, "/api/train/start")
    request_id = sent.pop("start_request_id")
    assert request_id.startswith("mcp-")
    assert sent == CONFIG
    assert result["structuredContent"] == {
        "job_id": "job-7",
        "status": "queued",
        "message": "Training job queued",
        "start_request_id": request_id,
    }
    headers = studio.state.calls[0][2]
    assert "x-unsloth-hf-token" not in headers


ACKNOWLEDGE = ("POST", "/api/train/start-requests/{start_request_id}/acknowledge")


@pytest.mark.parametrize("refusal", ["status_error", "http_409"])
def test_a_refused_start_is_acknowledged(monkeypatch, refusal):
    refused = {
        "job_id": "",
        "status": "error",
        "message": "Training already active",
        "error": "Training already active",
    }
    answer = (
        refused
        if refusal == "status_error"
        else JSONResponse(
            {"detail": "A transformers installation is in progress."}, status_code = 409
        )
    )
    result, studio = _start(
        monkeypatch,
        {"config": CONFIG},
        {("POST", "/api/train/start"): answer, ACKNOWLEDGE: {"status": "ok"}},
    )
    assert result["isError"] is True
    (sent,) = bodies(studio, "/api/train/start")
    acknowledged = [p for _m, p, _h, _b in studio.state.calls if p.endswith("/acknowledge")]
    # Without this the refusal would stay the training status and hide the run that is going.
    assert acknowledged == [f"/api/train/start-requests/{sent['start_request_id']}/acknowledge"]


def test_a_started_job_is_not_acknowledged(monkeypatch):
    _result, studio = _start(monkeypatch, {"config": CONFIG}, {ACKNOWLEDGE: {"status": "ok"}})
    assert not [p for _m, p, _h, _b in studio.state.calls if p.endswith("/acknowledge")]


def test_validate_only_starts_nothing(monkeypatch):
    result, studio = _start(monkeypatch, {"config": CONFIG, "validate_only": True})
    assert result["structuredContent"]["status"] == "valid"
    assert result["structuredContent"]["job_id"] is None
    assert studio.state.calls == []


def test_a_callers_start_request_id_is_kept(monkeypatch):
    result, studio = _start(monkeypatch, {"config": {**CONFIG, "start_request_id": "mine-1"}})
    assert bodies(studio, "/api/train/start")[0]["start_request_id"] == "mine-1"
    assert result["structuredContent"]["start_request_id"] == "mine-1"


def test_kind_routes_to_the_matching_start_route(monkeypatch):
    result, studio = _start(monkeypatch, {"config": DIFFUSION, "kind": "diffusion"})
    assert result["structuredContent"] == {
        "job_id": "diff-1",
        "status": "queued",
        "message": None,
        "start_request_id": None,
    }
    assert "/srv" not in json.dumps(result)
    assert bodies(studio, "/api/train/diffusion/start") == [DIFFUSION]
    assert bodies(studio, "/api/train/start") == []


# ---------------------------------------------------------------- runs and checkpoints


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
    ("GET", "/api/models/checkpoints"): RUN_CHECKPOINTS,
    ("GET", "/api/train/runs"): RUNS,
    ("GET", "/api/train/runs/run-1"): DETAIL,
}


def _runs(monkeypatch, args):
    return run_tool(monkeypatch, RUN_PAYLOADS, "list_training_runs", args)[0]


def test_runs_name_their_folder_and_checkpoints_by_name(monkeypatch):
    result = _runs(monkeypatch, {"include_checkpoints": True})
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
    caller = make_caller(
        direct_local = False, public_base = "http://h", studio_app = fake_studio(RUN_PAYLOADS)
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
    result = _runs(monkeypatch, {"run_id": "run-1"})
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
    studio = fake_studio(RUN_PAYLOADS)
    run_tool(monkeypatch, studio, "list_training_runs", {"limit": 10_000, "offset": -5})
    run_tool(monkeypatch, studio, "list_training_runs", {"limit": 0})
    assert queries(studio, "/api/train/runs") == [
        {"limit": "200", "offset": "0"},
        {"limit": "1", "offset": "0"},
    ]
