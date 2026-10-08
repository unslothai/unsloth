# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import shutil
import struct
import sys
import threading
import time
from pathlib import Path
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
_REAL_ENGINE = laya_runtime._engine_available
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
    # The MLX engine is opted into per test, so a run on Apple Silicon selects what CI does.
    monkeypatch.setattr(laya_runtime, "_engine_available", lambda: False)
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


def _clef_fine_tune(outputs, folder):
    path = outputs / folder
    path.mkdir(parents = True)
    for name in ("config.json", "joint_head.safetensors", "joint_head_config.json"):
        (path / name).write_text("{}", encoding = "utf-8")
    (path / "model.safetensors").write_bytes(b"")
    return catalog.CLEF_FINE_TUNE_PREFIX + folder


def _spoof_device(monkeypatch, kind):
    # ROCm hosts report DeviceType.CUDA, like NVIDIA ones.
    from utils.hardware import hardware

    device = {
        "cuda": hardware.DeviceType.CUDA,
        "rocm": hardware.DeviceType.CUDA,
        "mlx": hardware.DeviceType.MLX,
        "xpu": hardware.DeviceType.XPU,
        "cpu": hardware.DeviceType.CPU,
    }[kind]
    monkeypatch.setattr(hardware, "get_device", lambda: device)
    monkeypatch.setattr(hardware, "DEVICE", device)


@pytest.fixture
def clef(home, monkeypatch):
    from core.systemone import clef_runtime

    state = SimpleNamespace(training = False, agents = [])

    class Agent:
        device = "cuda"

        def __init__(
            self,
            folder,
            cancelled = None,
        ):
            self.folder, self.closed, self.cancelled = folder, False, cancelled
            state.agents.append(self)

        def decide(self, state_, questions):
            answers = {name: {"type": "noul", "noul": 0.25} for name in questions}
            return {"answers": answers, "input_tokens": 7, "truncated": False}

        def close(self):
            self.closed = True

    monkeypatch.setattr(clef_runtime, "ClefAgent", Agent)
    _spoof_device(monkeypatch, "cuda")
    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(laya_runtime, "_training_active", lambda: state.training)
    return state


def test_a_clef_fine_tune_serves_through_its_worker(home, client, clef):
    served = _clef_fine_tune(home, "clef_served_1")
    assert _listed(client) == [served]
    # The trainer's "Use in Decision API" names every run laya-ft:; the layout picks the prefix.
    assert _put(client, enabled = True, model = "laya-ft:clef_served_1").status_code == 200
    assert client.get("/api/settings/systemone").json()["model"] == served

    answer = _post(client).json()
    assert answer["model"] == served
    assert answer["answers"]["urgent"] == {"type": "noul", "noul": 0.25}
    assert answer["usage"] == {"input_tokens": 7, "output_tokens": 0}
    assert client.get("/api/settings/systemone").json()["loaded_device"] == "cuda"

    # Clef has no CPU fallback: during training it waits, and its worker is ended.
    clef.training = True
    busy = _post(client)
    assert busy.status_code == 503
    assert "training run" in busy.json()["detail"]["message"]
    assert clef.agents[0].closed
    clef.training = False
    assert _post(client).status_code == 200
    assert len(clef.agents) == 2
    laya_runtime.unload()
    assert clef.agents[1].closed


def test_an_adapter_clef_fine_tune_serves_through_its_worker(home, client, clef):
    # FastDecisionModel.save_pretrained: LoRA adapters plus the Clef head over the base LLM.
    path = home / "qwen_decisions_1"
    path.mkdir(parents = True)
    for name in ("adapter_config.json", "joint_head.safetensors", "joint_head_config.json"):
        (path / name).write_text("{}", encoding = "utf-8")
    served = catalog.CLEF_FINE_TUNE_PREFIX + "qwen_decisions_1"
    # Not complete until the adapter weights are there.
    assert _listed(client) == []
    (path / "adapter_model.safetensors").write_bytes(b"")
    assert _listed(client) == [served]
    assert _put(client, enabled = True, model = served).status_code == 200

    answer = _post(client).json()
    assert answer["model"] == served
    assert answer["answers"]["urgent"] == {"type": "noul", "noul": 0.25}
    assert clef.agents[0].folder == path.resolve() or str(clef.agents[0].folder) == str(path)


def test_a_clef_load_that_training_overtakes_frees_the_gpu(home, clef, monkeypatch):
    from core.systemone import clef_runtime

    # The loader thread outlives the request that started it, so no later request evicts it.
    checkpoint = catalog.resolve(_clef_fine_tune(home, "clef_overtaken_1"))
    real = clef_runtime.ClefAgent

    def overtaken(folder, cancelled = None):
        agent = real(folder, cancelled)
        clef.training = True
        return agent

    monkeypatch.setattr(clef_runtime, "ClefAgent", overtaken)
    laya_runtime._load(checkpoint)
    assert clef.agents[0].closed
    assert laya_runtime._agent is None and laya_runtime._device_name is None
    # The loader also stops the worker mid-load, not only once it reports ready.
    assert clef.agents[0].cancelled is laya_runtime._training_active


def test_a_clef_worker_that_never_reports_ready_is_stopped(monkeypatch):
    from core.systemone import clef_runtime

    calls = []

    class Process:
        exitcode = None

        def __init__(self, **kwargs):
            self.child = kwargs["args"][0]

        def start(self):
            # A live child holds its own end of the pipe, so the parent sees silence, not EOF.
            # Keeping the parent's close from releasing it works on Windows too, where a pipe
            # end is a handle os.dup cannot copy.
            held.append(self.child)
            self.child.close = lambda: None

        def join(self, timeout = None):
            calls.append("join")

        def is_alive(self):
            return "kill" not in calls

        def kill(self):
            calls.append("kill")

    held = []
    monkeypatch.setattr(clef_runtime._CTX, "Process", Process)
    monkeypatch.setattr(clef_runtime, "LOAD_TIMEOUT_S", 0.01)
    try:
        with pytest.raises(clef_runtime.ClefWorkerError, match = "did not answer"):
            clef_runtime.ClefAgent("unused")
    finally:
        for child in held:
            del child.close
            child.close()
    assert "kill" in calls


def test_a_training_run_stops_a_clef_worker_still_loading(monkeypatch):
    from core.systemone import clef_runtime

    calls = []

    class Process:
        exitcode = None

        def __init__(self, **kwargs):
            self.child = kwargs["args"][0]

        def start(self):
            # A live child holds its own end of the pipe, so the parent sees silence, not EOF.
            # Keeping the parent's close from releasing it works on Windows too, where a pipe
            # end is a handle os.dup cannot copy.
            held.append(self.child)
            self.child.close = lambda: None

        def join(self, timeout = None):
            calls.append("join")

        def is_alive(self):
            return "kill" not in calls

        def kill(self):
            calls.append("kill")

    held = []
    monkeypatch.setattr(clef_runtime._CTX, "Process", Process)
    monkeypatch.setattr(clef_runtime, "LOAD_TIMEOUT_S", 600)
    try:
        started = time.monotonic()
        with pytest.raises(clef_runtime.ClefWorkerError, match = "training run"):
            clef_runtime.ClefAgent("unused", cancelled = lambda: True)
        assert time.monotonic() - started < 30
    finally:
        for child in held:
            del child.close
            child.close()
    assert "kill" in calls


def test_the_catalog_offers_the_stock_clef_models():
    for name, repo in (("clef", "Cloudflare/clef"), ("clef-flash", "Cloudflare/clef-flash")):
        checkpoint = catalog.CHECKPOINTS[name]
        assert (checkpoint.source, checkpoint.layout, checkpoint.subfolder) == (repo, "clef", None)
    assert all(
        c.layout == "laya"
        for n, c in catalog.CHECKPOINTS.items()
        if n.startswith("laya") and c.backend == "pytorch"
    )


@pytest.mark.parametrize("kind", ["mlx", "xpu", "cpu"])
def test_clef_refuses_a_machine_without_an_nvidia_or_amd_gpu(home, client, clef, monkeypatch, kind):
    served = _clef_fine_tune(home, "clef_nogpu_1")
    assert _put(client, enabled = True, model = served).status_code == 200
    _spoof_device(monkeypatch, kind)

    refused = _post(client)
    assert refused.status_code == 400
    assert refused.json()["detail"]["message"] == catalog.CLEF_NEEDS_GPU
    assert clef.agents == []
    models = {m["name"]: m for m in client.get("/api/settings/systemone").json()["models"]}
    for name in (served, "clef", "clef-flash"):
        assert models[name]["available"] is False
        assert models[name]["unavailable_reason"] == catalog.CLEF_NEEDS_GPU
    assert all(
        m["available"]
        for n, m in models.items()
        if n.startswith("laya") and not m["llama_cpp_only"]
    )


@pytest.mark.parametrize("kind", ["cuda", "rocm"])
def test_clef_serves_on_nvidia_and_amd_gpus(home, client, clef, monkeypatch, kind):
    served = _clef_fine_tune(home, "clef_gpu_1")
    assert _put(client, enabled = True, model = served).status_code == 200
    _spoof_device(monkeypatch, kind)
    assert _post(client).status_code == 200
    models = client.get("/api/settings/systemone").json()["models"]
    # The GGUF-only entries depend on a llama-server, which this test does not install.
    assert all(
        m["available"] and m["unavailable_reason"] is None
        for m in models
        if not m["llama_cpp_only"]
    )


def test_a_failed_device_probe_does_not_refuse_clef(monkeypatch):
    from utils.hardware import hardware

    def broken():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(hardware, "get_device", broken)
    assert catalog.clef_unsupported_reason() is None


def test_settings_never_wait_on_device_detection(monkeypatch):
    from utils.hardware import hardware

    def slow():
        raise AssertionError("Settings must not run device detection")

    monkeypatch.setattr(hardware, "get_device", slow)
    monkeypatch.setattr(hardware, "DEVICE", None)
    assert catalog.clef_unsupported_reason(wait = False) is None
    monkeypatch.setattr(hardware, "DEVICE", hardware.DeviceType.MLX)
    assert catalog.clef_unsupported_reason(wait = False) == catalog.CLEF_NEEDS_GPU


def _clef_agent(conn):
    import threading

    from core.systemone import clef_runtime

    agent = object.__new__(clef_runtime.ClefAgent)
    agent._lock, agent._broken, agent._conn = threading.Lock(), None, conn
    agent._process = SimpleNamespace(exitcode = 1)
    return agent


def test_a_timed_out_clef_worker_is_never_asked_again(monkeypatch):
    from core.systemone import clef_runtime

    class Conn:
        sent = 0

        def send(self, message):
            self.sent += 1

        def poll(self, timeout):
            return self.sent > 1  # the first answer comes late, after its request timed out

        def recv(self):
            return ("ok", {"answers": "for the request that timed out"})

    monkeypatch.setattr(clef_runtime, "DECIDE_TIMEOUT_S", 0.01)
    conn = Conn()
    agent = _clef_agent(conn)
    for _ in range(2):
        with pytest.raises(clef_runtime.ClefWorkerError, match = "did not answer"):
            agent.decide("state", {})
    assert conn.sent == 1


def test_a_clef_worker_that_died_after_loading_is_a_worker_error():
    from core.systemone import clef_runtime
    class Conn:
        def send(self, message):
            raise BrokenPipeError(32, "Broken pipe")

    with pytest.raises(clef_runtime.ClefWorkerError, match = "exited"):
        _clef_agent(Conn()).decide("state", {})


@pytest.fixture
def engine(home, monkeypatch, tmp_path):
    _spoof_device(monkeypatch, "mlx")
    monkeypatch.setattr("utils.hardware.hardware.DETECTION_COMPLETE", done := threading.Event())
    done.set()
    state = SimpleNamespace(loaded = [], asked = [], error = None, cached = set())

    def answer(state_, questions):
        state.asked.append(questions)
        if state.error:
            raise state.error("no")
        answers = {name: {"type": "noul", "noul": 0.5} for name in questions}
        return {"answers": answers, "usage": {"input_tokens": 3}}

    def load(folder, family, base_model):
        state.loaded.append((folder.name, family, base_model and base_model.name))
        tokenizer = SimpleNamespace(encode = lambda text, **_: [0] * len(text.encode()))
        return SimpleNamespace(answer = answer, tokenizer = tokenizer)

    def dirs(checkpoint, local_only):
        if local_only and checkpoint.name not in state.cached:
            raise FileNotFoundError(checkpoint.name)
        if checkpoint.layout == "clef":
            return Path(checkpoint.source), None
        base = catalog.MLX_COMPANIONS[checkpoint.name].base
        return tmp_path / checkpoint.name, base and tmp_path / base.repo.split("/")[1]

    zoo = SimpleNamespace(load_decision_model = load)
    zoo.DecisionRequestError = type("RequestError", (ValueError,), {})
    zoo.DecisionUnsupportedError = type("UnsupportedError", (zoo.DecisionRequestError,), {})
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.decision", zoo)
    monkeypatch.setattr(laya_runtime, "_engine_available", _REAL_ENGINE)
    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(laya_runtime, "_mlx_dirs", dirs)
    monkeypatch.setattr(laya_runtime, "_training_active", lambda: False)
    # Whatever an earlier test in this process left holding the GPU.
    monkeypatch.setattr("core.inference.gpu_arbiter._owner", None)
    yield state
    laya_runtime.unload()


def test_apple_silicon_answers_text_through_the_mlx_engine(home, client, engine, monkeypatch):
    served, folder = _clef_fine_tune(home, "clef_mlx_1"), home / "clef_mlx_1"
    assert _put(client, enabled = True, model = "kev-4b").status_code == 200
    answered = _post(client)
    assert answered.status_code == 200 and answered.json()["answers"]["urgent"]["noul"] == 0.5
    assert engine.loaded == [("kev-4b", "kev", "Qwen3.5-4B-Base")]
    settings = client.get("/api/settings/systemone").json()
    assert (settings["effective_backend"], settings["loaded_backend"]) == ("mlx", "mlx")
    models = {m["name"]: m for m in settings["models"]}
    assert all(models[name]["available"] for name in (served, "clef-flash", "kev-4b"))
    assert _post(client, served).headers["x-unsloth-decision-backend"] == "mlx"
    assert engine.loaded[-1] == ("clef_mlx_1", "clef", None)
    # Images, a forced runtime and a Clef saved as adapters are not the engine's.
    kev = catalog.CHECKPOINTS["kev-4b"]
    assert laya_runtime._mlx_choice(kev, ["png"], None, "auto") is None
    assert laya_runtime._mlx_choice(kev, None, None, "llama.cpp") is None
    (folder / "config.json").rename(folder / "adapter_config.json")
    (folder / "model.safetensors").rename(folder / "adapter_model.safetensors")
    assert laya_runtime._mlx_target(catalog.fine_tune(served)) is None
    # A downloaded GGUF serves before the MLX form is fetched.
    laya_runtime.unload()
    monkeypatch.setattr(laya_runtime, "_native_unavailable", lambda *args: None)
    monkeypatch.setattr(laya_runtime, "is_cached", laya_runtime._is_native)
    assert laya_runtime.select(kev)[0].backend == "llama.cpp"
    monkeypatch.setattr(laya_runtime, "is_cached", lambda checkpoint: True)
    assert laya_runtime.select(kev)[0].backend == "mlx"


def test_the_mlx_runtime_serves_only_through_the_engine(home, client, engine, monkeypatch):
    served, folder = _clef_fine_tune(home, "clef_mlx_2"), home / "clef_mlx_2"
    kev = catalog.CHECKPOINTS["kev-4b"]
    assert _put(client, enabled = True, model = "kev-4b", backend = "mlx").status_code == 200
    settings = client.get("/api/settings/systemone").json()
    assert settings["mlx_available"] and settings["effective_backend"] == "mlx"
    described = {m["name"]: "llama.cpp or MLX" in m["description"] for m in settings["models"]}
    assert described["kev-9b"] and not described["laya-gguf"]
    assert _post(client, served).headers["x-unsloth-decision-backend"] == "mlx"
    tuned = catalog.fine_tune(served)
    # Unlike Auto, a downloaded GGUF does not take the request.
    monkeypatch.setattr(laya_runtime, "_native_unavailable", lambda *args: None)
    monkeypatch.setattr(laya_runtime, "is_cached", laya_runtime._is_native)
    assert laya_runtime.select(kev)[0].backend == "mlx"
    assert laya_runtime.mlx_ready(kev) and not laya_runtime.native_ready(kev)
    assert _post(client).headers["x-unsloth-decision-backend"] == "mlx"
    with pytest.raises(laya_runtime.Unavailable, match = "Images are served only by llama.cpp"):
        laya_runtime.select(kev, ["png"])
    (folder / "config.json").rename(folder / "adapter_config.json")
    (folder / "model.safetensors").rename(folder / "adapter_model.safetensors")
    with pytest.raises(laya_runtime.Unavailable, match = "has no MLX form"):
        laya_runtime.select(tuned)
    assert _put(client, device = "cpu").status_code == 200
    settings = client.get("/api/settings/systemone").json()
    assert not settings["mlx_available"] and settings["effective_backend"] is None
    assert not any("MLX" in m["description"] for m in settings["models"])
    assert "needs Apple Silicon" in settings["fallback_reason"] and _post(client).status_code == 400
    plan = client.get("/api/settings/systemone/resolve", params = {"backend": "mlx"}).json()
    assert (plan["repo"], plan["cached"]) == (None, False) and "needs Apple Silicon" in plan[
        "error"
    ]
    assert _put(client, backend = "mlx-lm").status_code == 400


def test_the_mlx_engine_asks_by_id_and_reports_what_it_refuses(home, client, engine):
    assert _put(client, enabled = True, model = "julia-1").status_code == 200
    questions = {"urgent": {"type": "noul"}, "late": {"type": "noul", "instructions": "Late?"}}
    body = {"model": "default", "state": "s", "questions": questions}
    assert client.post("/v1/systemone", json = body).status_code == 200
    assert [q["instructions"] for q in engine.asked[0].values()] == ["urgent", "Late?"]
    zoo = sys.modules["unsloth_zoo.mlx.decision"]
    for error, status in ((zoo.DecisionRequestError, 422), (zoo.DecisionUnsupportedError, 501)):
        engine.error = error
        assert _post(client).status_code == status
    engine.error, asked = None, len(engine.asked)
    states = ("w " * 8000, "w " * 8193, "é" * 8193, ["é" * 8000], "s")
    for state, status in zip(states, (200, 422, 422, 200, 422)):
        questions["late"]["instructions"] = "w " * 8193 if state == "s" else "Late?"
        body["state"] = state
        assert client.post("/v1/systemone", json = body).status_code == status
    assert len(engine.asked) == asked + 2


def test_an_mlx_decoder_waits_for_the_gpu(home, client, engine, monkeypatch):
    from core.inference import gpu_arbiter

    assert _put(client, enabled = True).status_code == 200
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    assert _post(client, "kev-4b").status_code == 409
    assert _post(client, "julia-1").status_code == 200
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    assert _post(client, "kev-4b").status_code == 200
    agent = laya_runtime._agent
    assert gpu_arbiter.current_owner() == gpu_arbiter.DECISIONS
    # A chat load ends the idle decoder through the arbiter, and its weights go with it.
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    gpu_arbiter.acquire_for(gpu_arbiter.CHAT)
    assert laya_runtime._agent is None and agent.model is None
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    assert _post(client, "kev-4b").status_code == 200
    laya_runtime.unload()
    assert gpu_arbiter.current_owner() is None
    # A load that loses the GPU to a training run gives its memory back before the claim.
    freed = []
    monkeypatch.setattr(laya_runtime, "_release_memory", lambda: freed.append(gpu_arbiter._owner))
    engine.loaded.clear()
    monkeypatch.setattr(laya_runtime, "_training_active", lambda: bool(engine.loaded))
    assert _post(client, "kev-4b").status_code != 200
    assert freed[-1] == gpu_arbiter.DECISIONS and gpu_arbiter.current_owner() is None


def test_the_mlx_engine_needs_apple_silicon_and_the_gpu(home, client, engine, monkeypatch):
    hardware = sys.modules["utils.hardware.hardware"]
    kev = catalog.CHECKPOINTS["kev-4b"]
    assert _put(client, device = "cpu").status_code == 200
    assert laya_runtime._mlx_target(kev) is None
    assert _put(client, enabled = True, device = "gpu").status_code == 200
    assert laya_runtime._mlx_target(kev) is not None
    # While detection is still running the answer is no: a settings read never waits for it.
    waited = []
    hardware.DETECTION_COMPLETE.clear()
    monkeypatch.setattr(hardware, "get_device", lambda: waited.append(1) or hardware.DeviceType.MLX)
    assert laya_runtime._mlx_target(kev) is None and not waited
    hardware.DETECTION_COMPLETE.set()
    for apple in (False, True):
        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: apple)
        assert _post(client, "kev-4b").status_code == 200 and bool(waited) == apple
    assert laya_runtime._mlx_target(kev) is not None
    _spoof_device(monkeypatch, "cuda")
    assert laya_runtime._mlx_target(kev) is None
    _spoof_device(monkeypatch, "mlx")
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.decision", None)
    assert laya_runtime._mlx_target(kev) is None


def test_settings_report_mlx_as_detection_settles_and_where_it_runs(
    home, client, engine, monkeypatch
):
    hardware = sys.modules["utils.hardware.hardware"]
    assert _put(client, enabled = True, model = "kev-4b").status_code == 200
    # No device stored: MLX takes the GPU, as llama.cpp does, and the response says so.
    settings = client.get("/api/settings/systemone").json()
    assert (settings["effective_backend"], settings["device"]) == ("mlx", "gpu")
    # Detection finishing inside this read's own wait for it is seen by the MLX answers too.
    hardware.DETECTION_COMPLETE.clear()
    monkeypatch.setattr(
        hardware,
        "get_device",
        lambda: hardware.DETECTION_COMPLETE.set() or hardware.DeviceType.MLX,
    )
    settings = client.get("/api/settings/systemone").json()
    assert settings["mlx_available"] and settings["effective_backend"] == "mlx"


def test_mlx_sources_are_fetched_at_their_pinned_revisions(home, monkeypatch, tmp_path):
    fetched = []

    def download(repo, revision, allow_patterns, local_files_only, **kwargs):
        folder = tmp_path / f"{repo.replace('/', '--')}@{revision}"
        if local_files_only and not folder.is_dir():
            raise FileNotFoundError(revision)
        if not local_files_only:
            fetched.append((repo, revision))
            for name in allow_patterns:
                (folder / name).parent.mkdir(parents = True, exist_ok = True)
                (folder / name).touch()
        return str(folder)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr(laya_runtime, "_engine_available", lambda: True)
    kev = catalog.MLX_COMPANIONS["kev-4b"]
    target = laya_runtime._mlx_target(catalog.CHECKPOINTS["kev-4b"])
    plan = laya_runtime.download_plan(target)
    assert (plan["repo"], plan["revision"]) == (kev.base.repo, kev.base.revision)
    assert not plan["cached"] and not fetched
    laya_runtime._repo_dir(kev.base, False)
    plan = laya_runtime.download_plan(target)
    assert plan == {**plan, "repo": kev.repo, "revision": kev.revision, "files": sorted(kev.files)}
    folder, base = laya_runtime._mlx_dirs(target, local_only = False)
    assert fetched == [(kev.base.repo, kev.base.revision), (kev.repo, kev.revision)]
    assert (base / "config.json").is_file() and laya_runtime.is_cached(target)
    # A settings download holds the repo's main: it serves, without another fetch, where the pinned revision is not complete.
    (folder / "head.pt").unlink()
    assert not laya_runtime.is_cached(target)
    main = Path(download(kev.repo, None, kev.files, False))
    del fetched[:]
    assert laya_runtime.is_cached(target)
    assert laya_runtime._mlx_dirs(target, local_only = False)[0] == main and not fetched
    (main / "head.pt").unlink()
    assert not laya_runtime.is_cached(target)
