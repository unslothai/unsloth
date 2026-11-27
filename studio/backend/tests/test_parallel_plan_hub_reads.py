# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /api/train/parallel/plan reads a repo's model_info once per request, not once per probe."""

from __future__ import annotations

import json
import types
from collections import Counter

import huggingface_hub
import pytest
import requests.adapters
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.training import parallel_plan
from hub.dependencies import get_hf_token
from routes import training as training_routes
from utils.hardware import hardware as hw

REPO = "unsloth/r42-plan-probe-8B"


def _host():
    return {
        "platform": "linux",
        "device": "cuda",
        "distributed_backend": "nccl",
        "multi_process_reason": None,
        "gpus": [
            {"index": 0, "name": "Tesla T4", "total_gb": 14.7, "free_gb": 14.5},
            {"index": 1, "name": "Tesla T4", "total_gb": 14.7, "free_gb": 14.5},
        ],
        "core": {"ddp": True, "fsdp2": True, "tp": True, "pp": True, "cp": True, "ep": False},
        "torch": (2, 9),
        "sm": 7,
    }


def _config():
    return types.SimpleNamespace(
        model_type = "llama",
        architectures = ["LlamaForCausalLM"],
        hidden_size = 4096,
        intermediate_size = 14336,
        num_hidden_layers = 32,
        num_attention_heads = 32,
        num_key_value_heads = 8,
        vocab_size = 128256,
        max_position_embeddings = 131072,
        tie_word_embeddings = False,
    )


class RepositoryNotFoundError(Exception):
    def __init__(self):
        super().__init__("404 Client Error: Repository Not Found")
        self.response = types.SimpleNamespace(status_code = 404, headers = {})


@pytest.fixture
def hub(monkeypatch):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising = False)
    state = {"calls": Counter(), "missing": False}

    def model_info(repo_id, **kwargs):
        state["calls"][repo_id] += 1
        if state["missing"]:
            raise RepositoryNotFoundError()
        return types.SimpleNamespace(
            id = repo_id,
            siblings = [
                types.SimpleNamespace(rfilename = "config.json", size = 900),
                types.SimpleNamespace(rfilename = "model.safetensors", size = 16 * 1024**3),
            ],
            safetensors = {"total": 8_030_261_248},
            tags = ["safetensors", "llama", "text-generation"],
            pipeline_tag = "text-generation",
            card_data = None,
            cardData = None,
            gated = False,
            private = False,
            config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        )

    def no_network(self, request, *args, **kwargs):
        raise requests.exceptions.ConnectionError(f"network blocked in test: {request.url}")

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", no_network)
    monkeypatch.setattr(
        hw, "_load_config_for_gpu_estimate", lambda model_name, hf_token = None: _config()
    )
    monkeypatch.setattr(parallel_plan, "detect_host", lambda max_age_s = 5.0: _host())
    return state


@pytest.fixture
def client(hub):
    app = FastAPI()
    app.include_router(training_routes.router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.dependency_overrides[get_hf_token] = lambda: None
    with TestClient(app) as http:
        yield http


def _plan(client):
    response = client.post("/api/train/parallel/plan", json = {"model_name": REPO})
    assert response.status_code == 200
    return response.json()


def test_one_request_shares_the_model_info_between_probes(client, hub):
    body = _plan(client)
    assert body["error"] is None and body["estimate"]["required_gb"]
    # One read for the resolver probes, one for the safetensors parameter count.
    assert hub["calls"][REPO] <= 2
    assert body == _unscoped_plan()


def test_nothing_is_kept_between_requests(client, hub):
    _plan(client)
    first = hub["calls"][REPO]
    _plan(client)
    assert hub["calls"][REPO] == 2 * first


def test_a_missing_repo_still_plans_without_model_facts(client, hub):
    hub["missing"] = True
    body = _plan(client)
    assert body["error"] is None
    assert body == _unscoped_plan()
    assert hub["calls"][REPO] >= 2


def _unscoped_plan():
    from models.training import ParallelPlanRequest

    values = {**ParallelPlanRequest(model_name = REPO).model_dump(), "drafter": None}
    plan = parallel_plan.plan_for_request(values, hf_token = None, inline_error = True)
    return json.loads(json.dumps(plan))
