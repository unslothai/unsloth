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


def _llm_fine_tune(outputs, folder):
    path = outputs / folder
    path.mkdir(parents = True)
    (path / "decision_config.json").write_text('{"format": "causal"}', encoding = "utf-8")
    (path / "decision_head.safetensors").write_bytes(b"")
    return catalog.LLM_FINE_TUNE_PREFIX + folder


@pytest.fixture
def llm_worker(home, monkeypatch):
    from core.systemone import llm_runtime

    state = SimpleNamespace(agents = [], training = False, fail = False)

    class Agent:
        device = "cuda"

        def __init__(self, folder):
            self.folder, self.closed = folder, False
            state.agents.append(self)

        def decide(self, state_text, questions):
            if state.fail:
                raise llm_runtime.LLMWorkerError("The decision model worker exited (code -9)")
            answers = {
                name: {"type": "noul", "noul": 0.81, "confidence": 0.81} for name in questions
            }
            return {"answers": answers, "input_tokens": 42, "truncated": False}

        def close(self):
            self.closed = True

    def load(checkpoint):
        if checkpoint.layout == "llm":
            return _REAL_LOAD(checkpoint)
        # As the real Laya load: the resident model goes before the new one comes.
        laya_runtime._evict()
        return SimpleNamespace(), "cpu"

    monkeypatch.setattr(laya_runtime, "_load_checkpoint", load)
    monkeypatch.setattr(llm_runtime, "LLMDecisionAgent", Agent)
    monkeypatch.setattr(catalog, "llm_unsupported_reason", lambda: None)
    monkeypatch.setattr(laya_runtime, "_training_active", lambda: state.training)
    return state


def test_llm_decision_models_are_listed_and_served_by_their_worker(home, client, llm_worker):
    laya = _fine_tune(home, "laya_run_1")
    llm = _llm_fine_tune(home, "qwen_run_2")
    assert _listed(client) == [laya, llm]
    assert _put(client, enabled = True, model = llm).status_code == 200

    response = _post(client)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "model": llm,
        "answers": {"urgent": {"type": "noul", "noul": 0.81}},
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }
    assert [agent.folder for agent in llm_worker.agents] == [home / "qwen_run_2"]
    assert laya_runtime.status()["device"] == "cuda"

    # Switching to Laya ends the worker, so its GPU memory goes with it.
    assert _post(client, laya).json()["model"] == laya
    assert llm_worker.agents[0].closed


def test_an_llm_decision_model_waits_for_training_and_restarts_a_dead_worker(
    home, client, llm_worker
):
    llm = _llm_fine_tune(home, "qwen_run_1")
    assert _put(client, enabled = True, model = llm).status_code == 200
    assert _post(client).status_code == 200

    llm_worker.training = True
    busy = _post(client)
    assert busy.status_code == 503 and "training run" in busy.json()["detail"]["message"]
    assert llm_worker.agents[0].closed
    llm_worker.training = False

    assert _post(client).status_code == 200
    llm_worker.fail = True
    assert _post(client).status_code == 503
    if laya_runtime._loader is not None:
        laya_runtime._loader.join(5)
    llm_worker.fail = False
    assert _post(client).status_code == 200
    assert len(llm_worker.agents) == 3 and llm_worker.agents[1].closed


def test_the_train_page_button_stores_an_llm_run_under_its_own_name(home, client, llm_worker):
    llm = _llm_fine_tune(home, "qwen_run_1")
    # The Train page names every finished decision run laya-ft:<folder>.
    assert _put(client, enabled = True, model = "laya-ft:qwen_run_1").status_code == 200
    assert client.get("/api/settings/systemone").json()["model"] == llm
    assert _post(client).json()["model"] == llm


def test_the_environment_can_point_at_an_llm_decision_folder(home, client, llm_worker, monkeypatch):
    _llm_fine_tune(home, "qwen_run_1")
    assert _put(client, enabled = True).status_code == 200
    monkeypatch.setenv("UNSLOTH_SYSTEMONE_MODEL", str(home / "qwen_run_1"))
    assert client.get("/api/settings/systemone").json()["model"] == catalog.LOCAL_NAME
    assert _post(client).json()["model"] == catalog.LOCAL_NAME
    assert llm_worker.agents[0].folder == home / "qwen_run_1"


def test_llm_decision_models_need_a_gpu(home, client, llm_worker, monkeypatch):
    llm = _llm_fine_tune(home, "qwen_run_1")
    monkeypatch.setattr(catalog, "llm_unsupported_reason", lambda: catalog.LLM_NEEDS_GPU)
    assert _put(client, enabled = True, model = llm).status_code == 200
    response = _post(client)
    assert response.status_code == 400
    assert response.json()["detail"]["message"] == catalog.LLM_NEEDS_GPU
    assert llm_worker.agents == []


def test_llm_answers_have_the_laya_shapes():
    from core.systemone.llm_runtime import _answer

    confidence = laya_runtime._laya().common.confidence_from_probs
    choice = _answer(
        {"t": "choice", "crit": {"a": "", "b": ""}}, ["a", "b"], [0.25, 0.75], confidence
    )
    assert (choice["choice"], choice["probabilities"]) == ("b", {"a": 0.25, "b": 0.75})
    score = _answer(
        {"t": "score", "crit": ["low", "mid", "high"]}, ["0", "1", "2"], [0.2, 0.3, 0.5], confidence
    )
    assert score["score"] == 1.3 and score["legend"] == {"0": "low", "1": "mid", "2": "high"}
    noul = _answer({"t": "noul", "crit": None}, ["false", "true"], [0.1, 0.9], confidence)
    assert noul == {"type": "noul", "noul": 0.9, "confidence": 0.9}
    for answer in (choice, score, noul):
        assert laya_runtime._wire_answer(answer)["type"] == answer["type"]
