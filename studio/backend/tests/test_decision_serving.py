# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import shutil
import struct
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import external_provider
from core.systemone import catalog, laya_runtime
from routes import systemone
from storage import credential_secrets, providers_db
from utils import systemone_settings
from utils.account_context import OWNER, run_as
from utils.paths import outputs_root

_REAL_LOAD = laya_runtime._load_checkpoint
QUESTIONS = {"urgent": {"type": "noul", "instructions": "Does this need a reply now?"}}
UPSTREAM = {
    "model": "jev-1.13",
    "answers": {"urgent": {"type": "noul", "noul": 0.93}},
    "usage": {"input_tokens": 5, "output_tokens": 0},
}


def _answer(agent, state, questions):
    answers = {name: {"type": "noul", "noul": 0.5, "confidence": 0.5} for name in questions}
    return {"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}, False


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    for name in (
        "UNSLOTH_SYSTEMONE_MODEL",
        "UNSLOTH_SYSTEMONE_DISABLE",
        "UNSLOTH_SYSTEMONE_DEVICE",
    ):
        monkeypatch.delenv(name, raising = False)
    for name in ("_agent", "_loaded", "_device_name", "_loader", "_loading", "_failure"):
        monkeypatch.setattr(laya_runtime, name, None)
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        laya_runtime, "_load_checkpoint", lambda checkpoint: (SimpleNamespace(), "cpu")
    )
    monkeypatch.setattr(laya_runtime, "_predict", _answer)
    outputs = run_as(OWNER, outputs_root)
    outputs.mkdir(parents = True)
    yield outputs
    if laya_runtime._loader is not None:
        laya_runtime._loader.join(5)


@pytest.fixture
def client(home):
    from routes.settings import router as settings_router

    app = FastAPI()
    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    return TestClient(app)


def _fine_tune(outputs, folder):
    path = outputs / folder
    for name in ("encoder", "tokenizer"):
        (path / name).mkdir(parents = True)
    # A float16 safetensors header, as the trainer saves.
    header = json.dumps({"w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))
    (path / "rl_agent_config.json").write_text("{}", encoding = "utf-8")
    return catalog.FINE_TUNE_PREFIX + folder


def _post(client, model = "default"):
    return client.post(
        "/v1/systemone", json = {"model": model, "state": "The site is down.", "questions": QUESTIONS}
    )


def _put(client, **settings):
    return client.put("/api/settings/systemone", json = settings)


def _listed(client):
    models = client.get("/api/settings/systemone").json()["models"]
    return [m["name"] for m in models if m["kind"] == "fine_tune"]


def test_hidden_and_staged_folders_are_not_models(home, client):
    kept = _fine_tune(home, "laya_kept_1")
    staged = _fine_tune(home, ".laya_gone_2.deleting-" + "0" * 32)
    hidden = _fine_tune(home, ".laya_hidden_3")
    assert _listed(client) == [kept]
    for name in (staged, hidden):
        assert _put(client, model = name).status_code == 400, name


@pytest.mark.parametrize("folder", ["ft\x001", "loop"])
def test_unusable_fine_tune_names_are_unknown_models(home, client, folder):
    # outputs/loop -> loop
    os.symlink("loop", home / "loop")
    name = catalog.FINE_TUNE_PREFIX + folder
    assert _put(client, enabled = True).status_code == 200
    assert _post(client, name).status_code == 400
    assert client.get("/api/settings/systemone/resolve", params = {"model": name}).status_code == 400
    assert _put(client, model = name).status_code == 400


def test_keyless_callers_cannot_probe_fine_tune_names(home, client, monkeypatch):
    import auth.authentication

    served = _fine_tune(home, "laya_served_1")
    _fine_tune(home, "laya_private_2")
    assert _put(client, enabled = True, model = served).status_code == 200
    monkeypatch.setattr(auth.authentication, "request_admitted_without_credential", lambda r: True)
    existing = _post(client, catalog.FINE_TUNE_PREFIX + "laya_private_2")
    missing = _post(client, catalog.FINE_TUNE_PREFIX + "laya_missing_3")
    assert existing.status_code == missing.status_code == 403
    assert existing.json() == missing.json()
    assert _post(client, served).status_code == 200


@pytest.fixture
def gpu(home, monkeypatch):
    pytest.importorskip("torch")
    from utils import torch_device_probe
    from utils.hardware import hardware

    monkeypatch.setenv("UNSLOTH_SYSTEMONE_DEVICE", "gpu")
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(torch_device_probe, "device_can_allocate", lambda device: True)
    # The real loader around a stand-in model; placing it only records where it would go.
    state = SimpleNamespace(training = False, placed = [], on_build = None)

    def build(path, **kwargs):
        if state.on_build:
            state.on_build()
        return SimpleNamespace()

    def place(agent, device, fp16_checkpoint):
        state.placed.append(device.type)
        agent.device = device

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(laya_runtime, "_load_laya", build)
    monkeypatch.setattr(laya_runtime, "_place", place)
    monkeypatch.setattr(laya_runtime, "_training_active", lambda: state.training)
    # Imported here, so a cold import does not count against a request's load wait.
    laya_runtime._laya()
    return state


def test_the_model_returns_to_the_gpu_after_training(home, client, gpu):
    assert _put(client, enabled = True, model = _fine_tune(home, "laya_served_1")).status_code == 200
    assert _post(client).status_code == 200
    gpu.training = True
    # What the training start hook does with a model on the GPU.
    laya_runtime.unload()
    assert _post(client).status_code == 200
    assert _post(client).status_code == 200
    assert gpu.placed == ["cuda", "cpu"]
    gpu.training = False
    assert _post(client).status_code == 200
    assert _post(client).status_code == 200
    assert gpu.placed == ["cuda", "cpu", "cuda"]
    assert client.get("/api/settings/systemone").json()["loaded_device"] == "cuda"


def test_training_takes_the_gpu_from_a_load_in_flight_or_a_model_it_missed(home, client, gpu):
    def training_starts():
        gpu.training, gpu.on_build = True, None

    gpu.on_build = training_starts
    assert _put(client, enabled = True, model = _fine_tune(home, "laya_served_1")).status_code == 200
    assert _post(client).status_code == 200
    assert gpu.placed == ["cpu"]
    gpu.training = False
    assert _post(client).status_code == 200
    gpu.training = True
    assert _post(client).status_code == 200
    assert gpu.placed == ["cpu", "cuda", "cpu"]


@pytest.fixture
def upstream(home, monkeypatch):
    for module in (credential_secrets, providers_db):
        monkeypatch.setattr(module, "_schema_ready", set())
    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32
    )
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json = UPSTREAM)

    monkeypatch.setattr(
        external_provider,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handle)),
    )
    providers_db.create_provider(
        id = "deciders",
        provider_type = "typesafe",
        display_name = "TypeSafe",
        base_url = "https://api.typesafe.ai/v1",
        models = ["jev-latest"],
        api_type = "chat_completions",
    )
    credential_secrets.save_provider_api_key("deciders", "ts-key")
    return calls


def _mcp_model():
    from fastmcp import Client
    async def call():
        async with Client(systemone.decisions_mcp) as mcp:
            return await mcp.call_tool("decide", {"state": "x", "questions": QUESTIONS})

    return asyncio.run(call()).structured_content["model"]


def test_connections_and_fine_tunes_work_side_by_side(home, client, upstream):
    connection = "connection:deciders:jev-latest"
    served = _fine_tune(home, "laya_served_1")
    other = _fine_tune(home, "laya_other_2")
    assert _listed(client) == [other, served]
    listed = client.get("/api/settings/systemone/connections").json()
    assert [option["name"] for option in listed] == [connection]

    assert _put(client, enabled = True, model = connection).status_code == 200
    assert _post(client, "jev-latest").json() == UPSTREAM
    assert _post(client, connection).json() == UPSTREAM
    assert _post(client, served).json()["model"] == served
    assert _mcp_model() == "jev-1.13"
    assert _listed(client) == [other, served]

    assert _put(client, model = served).status_code == 200
    assert client.get("/api/settings/systemone").json()["model"] == served
    assert _post(client, "jev-latest").json()["model"] == served
    assert _post(client, other).json()["model"] == other
    assert _post(client, connection).status_code == 400
    assert _mcp_model() == served
    for bad in ("connection:deciders:jev-1.13", "connection:gone:jev-latest", "laya-ft:laya_gone"):
        response = client.post("/api/settings/systemone/validate", json = {"model": bad})
        assert response.status_code == 400, bad
    assert len(upstream) == 3

    # A removed connection stays configured and says so; a deleted fine-tune falls back.
    assert _put(client, model = connection).status_code == 200
    providers_db.delete_provider("deciders")
    assert client.get("/api/settings/systemone").json()["model"] == connection
    assert "removed" in _post(client).json()["detail"]["message"]
    assert _put(client, model = served).status_code == 200
    shutil.rmtree(home / "laya_served_1")
    assert client.get("/api/settings/systemone").json()["model"] == "laya-multilingual"


def test_the_environment_can_pin_a_fine_tune_or_a_folder(home, client, monkeypatch):
    served = _fine_tune(home, "laya_served_1")
    assert _put(client, enabled = True).status_code == 200
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", served)
    settings = client.get("/api/settings/systemone").json()
    assert (settings["model"], settings["model_locked"]) == (served, True)
    assert _post(client).json()["model"] == served
    assert _put(client, model = "laya-english").status_code == 400

    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", str(home / "laya_served_1"))
    assert client.get("/api/settings/systemone").json()["model"] == catalog.LOCAL_NAME
    assert _post(client, catalog.LOCAL_NAME).json()["model"] == catalog.LOCAL_NAME
    assert _post(client, served).json()["model"] == served
