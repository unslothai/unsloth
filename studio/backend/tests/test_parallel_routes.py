# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /api/train/parallel/capabilities, POST /api/train/parallel/plan, and /start's plan step (400 before any
VRAM moves), on a spoofed 2x T4 host."""

from __future__ import annotations

import asyncio
import types

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.training import parallel_plan
from hub.dependencies import get_hf_token
from routes import training as training_routes


def _host(reason = None):
    return {
        "platform": "windows" if reason else "linux",
        "device": "cuda",
        "distributed_backend": None if reason else "nccl",
        "multi_process_reason": reason,
        "gpus": [
            {"index": 0, "name": "Tesla T4", "total_gb": 14.7, "free_gb": 14.5},
            {"index": 1, "name": "Tesla T4", "total_gb": 14.7, "free_gb": 14.5},
        ],
        "core": {"ddp": True, "fsdp2": True, "tp": True, "pp": True, "cp": True, "ep": False},
        "torch": (2, 9),
        "sm": 7,
    }


@pytest.fixture
def spoof(monkeypatch):
    state = {"host": _host(), "need": 6.0}
    monkeypatch.setattr(parallel_plan, "detect_host", lambda max_age_s = 5.0: state["host"])
    monkeypatch.setattr(
        parallel_plan,
        "model_facts",
        lambda name, **kw: {
            "required_gb": state["need"],
            "per_gpu_gb": {1: state["need"], 2: state["need"] / 2 + 1},
            "is_moe": False,
            "family": "qwen3",
            "multimodal": False,
            "mode": "lora",
            "trainer": "sft",
        },
    )
    return state


@pytest.fixture
def client(spoof):
    app = FastAPI()
    app.include_router(training_routes.router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.dependency_overrides[get_hf_token] = lambda: None
    with TestClient(app) as http:
        yield http


def test_capabilities(client):
    body = client.get("/api/train/parallel/capabilities").json()
    assert body["multi_process"] and body["distributed_backend"] == "nccl"
    assert [g["index"] for g in body["gpus"]] == [0, 1]
    modes = {row["mode"]: row for row in body["modes"]}
    assert modes["ddp"]["available"] and not modes["ep"]["available"]


def test_plan_auto_is_ddp(client):
    body = client.post("/api/train/parallel/plan", json = {"model_name": "unsloth/Qwen3-0.6B"}).json()
    assert body["mode"] == "ddp" and body["spec"] == "dp=2" and body["gpu_ids"] == [0, 1]
    assert body["error"] is None


def test_plan_reports_a_refusal_inline(client, spoof):
    spoof["host"] = _host(reason = "Windows has no NCCL ...")
    response = client.post(
        "/api/train/parallel/plan",
        json = {"model_name": "unsloth/Qwen3-0.6B", "parallel_mode": "ddp"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"]["code"] == "parallel_platform"
    assert body["requested_mode"] == "ddp" and body["mode"] == "single"


def _request(mode, spec = None):
    return types.SimpleNamespace(parallel_mode = mode, parallel_spec = spec)


def _kwargs():
    return {
        "model_name": "unsloth/Qwen3-0.6B",
        "training_type": "LoRA/QLoRA",
        "load_in_4bit": False,
        "gpu_ids": None,
    }


def test_start_plan_step_pins_gpus_and_spec(spoof):
    kwargs = _kwargs()
    asyncio.run(training_routes._plan_parallel_training(_request("auto"), kwargs, None))
    assert kwargs["parallel"]["mode"] == "ddp" and kwargs["parallel"]["launcher"] == "multi_process"
    assert kwargs["parallel"]["spec"] == "dp=2" and kwargs["gpu_ids"] == [0, 1]


def test_start_plan_step_refuses_with_400(spoof):
    spoof["host"] = _host(reason = "Windows has no NCCL ...")
    with pytest.raises(HTTPException) as err:
        asyncio.run(training_routes._plan_parallel_training(_request("fsdp2"), _kwargs(), None))
    assert err.value.status_code == 400
    assert err.value.detail["code"] == "parallel_platform"


def test_start_plan_step_keeps_auto_selection_for_one_process(spoof):
    spoof["host"]["gpus"] = spoof["host"]["gpus"][:1]
    kwargs = _kwargs()
    asyncio.run(training_routes._plan_parallel_training(_request("auto"), kwargs, None))
    assert kwargs["parallel"]["mode"] == "single" and kwargs["gpu_ids"] is None


def test_request_model_defaults_to_legacy():
    from models.training import TrainingStartRequest
    fields = TrainingStartRequest.model_fields
    assert fields["parallel_mode"].default is None and fields["parallel_spec"].default is None
