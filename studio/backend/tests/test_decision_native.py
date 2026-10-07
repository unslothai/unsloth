# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clef through llama.cpp: routing, flags, capability, errors and answers, against a stub llama-server."""

import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import gpu_arbiter
from core.inference.llama_cpp import LlamaCppBackend
from core.systemone import catalog, gguf_export_contract as contract, laya_runtime, native_worker
from routes import systemone
from utils import systemone_settings
from utils.account_context import OWNER, run_as
from utils.paths import outputs_root

_REAL_LOAD = laya_runtime._load_checkpoint
REPO_ROOT = Path(__file__).resolve().parents[3]
PHYSICAL_BATCH = (
    "input ({tokens} tokens) is too large to process. increase the physical batch size "
    "(current batch size: {limit})"
)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"shipping": "parcels", "billing": "charges"},
    },
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "angry": {"type": "noul", "instructions": None},
}

# llama-server's /health, /v1/models and /v1/systemone, with llama.cpp's own answer and error shapes.
STUB = r"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
opt = lambda name: args[args.index(name) + 1] if name in args else None
mode = os.environ.get("STUB_MODE", "")
record = os.environ["STUB_RECORD"]


def log(kind, **data):
    with open(record, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, **data}) + "\n")


log("start", argv=args, env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")})
if mode == "exit":
    print("stub: failed to load model", flush=True)
    sys.exit(3)
key = opt("--api-key") or open(opt("--api-key-file"), encoding="utf-8").read().strip()
port, alias, ub, mmproj = int(opt("--port")), opt("--alias"), int(opt("-ub")), "--mmproj" in args


def answer(q):
    if q["type"] == "noul":
        return {"type": "noul", "noul": 0.7}
    keys = sorted(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    weights = [i + 1 for i in range(len(keys))]
    probs = {k: w / sum(weights) for k, w in zip(keys, weights)}
    best = max(probs, key=probs.get)
    uniform = 1 / len(keys)
    out = {"type": q["type"], "probabilities": probs, "confidence": (probs[best] - uniform) / (1 - uniform)}
    if q["type"] == "choice":
        out["choice"] = best
    else:
        out["score"] = sum(i * p for i, p in enumerate(probs.values()))
        out["legend"] = dict(zip(keys, q["criteria"]))
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def error(self, status, message):
        self.send(status, {"error": {"code": status, "message": message, "type": "x"}})

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"status": "ok"})
        if self.headers.get("Authorization") != f"Bearer {key}":
            return self.error(401, "Invalid API Key")
        outputs = ["text"] if mode == "nodecisions" else ["decisions"]
        inputs = ["text", "image", "video"] if mmproj else ["text"]
        served = "someone-else" if mode == "wrongalias" else alias
        self.send(200, {"object": "list", "data": [{"id": served, "aliases": [served],
                  "architecture": {"input_modalities": inputs, "output_modalities": outputs}}]})

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {key}":
            return self.error(401, "Invalid API Key")
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        log("request", body=body)
        state = body["state"] if isinstance(body["state"], str) else json.dumps(body["state"])
        if state == "crash":
            os._exit(1)
        if state == "badimage":
            return self.error(500, "Failed to load image or audio file")
        if state == "reject":
            return self.error(400, 'questions.q: "criteria" must be a non-empty object')
        if state == "context":
            return self.error(400, "request (9000 tokens) exceeds the available context size (8192 tokens), try increasing it")
        tokens = len(state.split()) + 10
        if tokens > ub:
            return self.error(500, f"input ({tokens} tokens) is too large to process. increase the physical batch size (current batch size: {ub})")
        answers = {name: answer(q) for name, q in body["questions"].items()}
        self.send(200, {"model": body["model"], "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}})


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def _formatter():
    # The vendored reference, loaded independently of the module under test.
    path = REPO_ROOT / "unsloth" / "_vendor" / "clef" / "joint_schema_model.py"
    spec = importlib.util.spec_from_file_location("_test_clef_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_clef_reference"] = module
    spec.loader.exec_module(module)
    return module.systemone_answer


def _stub_probabilities(question):
    if question["type"] == "noul":
        return {"true": 0.7, "false": 0.3}
    keys = (
        sorted(question["criteria"])
        if question["type"] == "choice"
        else [str(i) for i in range(len(question["criteria"]))]
    )
    weights = [i + 1 for i in range(len(keys))]
    return {k: w / sum(weights) for k, w in zip(keys, weights)}


@pytest.fixture
def stub(tmp_path, monkeypatch):
    if os.name == "nt":
        # The stub server is a shebang script; CreateProcess cannot run it (WinError 193).
        pytest.skip("stub llama-server needs a POSIX shebang")
    record = tmp_path / "stub_record.jsonl"
    script = tmp_path / "stub_server.py"
    script.write_text(STUB, encoding = "utf-8")
    binary = tmp_path / "llama-server"
    binary.write_text(f"#!{sys.executable}\nexec(open({str(script)!r}).read())\n", encoding = "utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_RECORD", str(record))
    monkeypatch.setattr(native_worker, "resolve_binary", lambda: str(binary))
    monkeypatch.setattr(
        LlamaCppBackend, "_llama_server_env_for_binary", staticmethod(lambda _: os.environ.copy())
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_enumerated_gpu_devices", staticmethod(lambda *_: ["CUDA0", "CUDA1"])
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_get_gpu_free_memory",
        staticmethod(lambda *_, **__: [(0, 1000), (1, 90000)]),
    )

    def records(kind):
        if not record.exists():
            return []
        rows = [json.loads(line) for line in record.read_text(encoding = "utf-8").splitlines()]
        return [row for row in rows if row["kind"] == kind]

    yield SimpleNamespace(
        binary = binary, records = records, set_mode = lambda m: monkeypatch.setenv("STUB_MODE", m)
    )
    laya_runtime.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    for name in (
        "UNSLOTH_SYSTEMONE_MODEL",
        "UNSLOTH_SYSTEMONE_DISABLE",
        "UNSLOTH_SYSTEMONE_DEVICE",
        "STUB_MODE",
    ):
        monkeypatch.delenv(name, raising = False)
    for name in (
        "_agent",
        "_loaded",
        "_device_name",
        "_loader",
        "_loading",
        "_failure",
        "_fallback_reason",
    ):
        monkeypatch.setattr(laya_runtime, name, None)
    monkeypatch.setattr(laya_runtime, "_incapable", set())
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: None)
    monkeypatch.setattr(systemone_settings, "gpu_available", lambda: False)
    monkeypatch.setattr(laya_runtime, "_load_checkpoint", _REAL_LOAD)
    monkeypatch.setattr(laya_runtime, "_training_active", lambda: state.training)
    # The stock Clef safetensors are never fetched here: the PyTorch worker below is a fake.
    clef_dir = tmp_path / "clef_safetensors"
    clef_dir.mkdir()
    monkeypatch.setattr(laya_runtime, "_clef_dir", lambda checkpoint, local_only: clef_dir)
    cache = tmp_path / "hub"
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(cache))
    state = SimpleNamespace(training = False, torch_agents = [], cache = cache)

    from core.systemone import clef_runtime

    class TorchAgent:
        device = "cuda"

        def __init__(
            self,
            folder,
            cancelled = None,
        ):
            self.closed = False
            state.torch_agents.append(self)

        def decide(self, state_, questions):
            systemone_answer = _formatter()
            answers = {n: systemone_answer(q, _stub_probabilities(q)) for n, q in questions.items()}
            return {"answers": answers, "input_tokens": 13, "truncated": False}

        def close(self):
            self.closed = True

    monkeypatch.setattr(clef_runtime, "ClefAgent", TorchAgent)
    from utils.hardware import hardware

    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "DEVICE", hardware.DeviceType.CUDA)
    outputs = run_as(OWNER, outputs_root)
    outputs.mkdir(parents = True)
    state.outputs = outputs
    yield state
    if laya_runtime._loader is not None:
        laya_runtime._loader.join(5)
    laya_runtime.shutdown()


def _cache_gguf(cache: Path, name: str = "clef-flash") -> None:
    companion = catalog.GGUF_COMPANIONS[name]
    folder = (
        cache / f"models--{companion.repo.replace('/', '--')}" / "snapshots" / companion.revision
    )
    folder.mkdir(parents = True)
    for file in (companion.model, companion.mmproj):
        if file:
            (folder / file).write_bytes(b"GGUF")


@pytest.fixture
def client(home):
    from routes.settings import router as settings_router

    app = FastAPI()
    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    return TestClient(app)


def _put(client, **settings):
    response = client.put("/api/settings/systemone", json = settings)
    assert response.status_code == 200, response.text
    return response.json()


def _post(
    client,
    state = "The parcel never arrived.",
    questions = QUESTIONS,
    images = None,
    model = "default",
):
    body = {"model": model, "state": state, "questions": questions}
    if images is not None:
        body["images"] = images
    for _ in range(100):
        response = client.post("/v1/systemone", json = body)
        if (
            response.status_code != 503
            or response.json()["detail"]["error_type"] != "model_loading"
        ):
            return response
        time.sleep(0.1)
    return response


PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _clef_folder(root: Path, name = "clef_run") -> Path:
    folder = root / name
    folder.mkdir(parents = True)
    for file in ("config.json", "joint_head_config.json", "unsloth_decision_config.json"):
        (folder / file).write_text("{}", encoding = "utf-8")
    (folder / "joint_head.safetensors").write_bytes(b"head")
    (folder / "model.safetensors").write_bytes(b"")
    return folder


def _export(
    folder: Path,
    quants = ("Q8_0",),
    mmproj = True,
    fingerprint = None,
) -> None:
    directory = folder / contract.EXPORT_DIR
    directory.mkdir(exist_ok = True)
    files = {}
    for quant in quants:
        (directory / f"model-{quant}.gguf").write_bytes(b"GGUF")
        if mmproj:
            (directory / f"mmproj-{quant}.gguf").write_bytes(b"GGUF")
        files[quant] = {
            "model": f"model-{quant}.gguf",
            "mmproj": f"mmproj-{quant}.gguf" if mmproj else None,
        }
    (directory / contract.EXPORT_FILE).write_text(
        json.dumps(
            {
                "format": contract.FORMAT,
                "version": 1,
                "layout": "clef",
                "quantizations": list(quants),
                "files": files,
                "source_fingerprint": fingerprint or contract.fingerprint(folder, "clef"),
            }
        ),
        encoding = "utf-8",
    )


def test_fingerprint_follows_the_calibration_and_head_bytes(tmp_path):
    folder = _clef_folder(tmp_path)
    first = contract.fingerprint(folder, "clef")
    assert first == contract.fingerprint(folder, "clef") and len(first) == 64
    # The backbone weights are not part of it; the head and the calibration are.
    (folder / "model.safetensors").write_bytes(b"other")
    assert contract.fingerprint(folder, "clef") == first
    time.sleep(0.01)
    (folder / "unsloth_decision_config.json").write_text(
        '{"temperature": [1, 2, 3]}', encoding = "utf-8"
    )
    assert contract.fingerprint(folder, "clef") != first
    laya = tmp_path / "laya"
    laya.mkdir()
    (laya / "rl_agent_config.json").write_text("{}", encoding = "utf-8")
    assert contract.fingerprint(laya, "laya") != contract.fingerprint(laya, "clef")
    with pytest.raises(ValueError):
        contract.fingerprint(laya, "other")


def test_the_served_export_prefers_q8_then_f16_and_ignores_stale_ones(tmp_path, caplog):
    folder = _clef_folder(tmp_path)
    assert contract.read_export(folder) is None and contract.served_files(folder, "clef") is None
    _export(folder, quants = ("Q4_K_M", "F16", "Q8_0"))
    quant, model, mmproj = contract.served_files(folder, "clef")
    assert (quant, model.name, mmproj.name) == ("Q8_0", "model-Q8_0.gguf", "mmproj-Q8_0.gguf")
    (folder / "gguf" / "model-Q8_0.gguf").unlink()
    assert contract.served_files(folder, "clef")[0] == "F16"
    assert contract.served_files(folder, "laya") is None

    stale = _clef_folder(tmp_path, "stale_run")
    _export(stale, fingerprint = "0" * 64)
    with caplog.at_level("INFO"):
        assert contract.served_files(stale, "clef") is None
        assert contract.served_files(stale, "clef") is None
    assert sum("stale GGUF export" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(format = "other"),
        lambda d: d.update(version = 2),
        lambda d: d.update(layout = "llm"),
        lambda d: d["files"]["Q8_0"].update(model = "../escape.gguf"),
        lambda d: d["files"]["Q8_0"].update(mmproj = "sub/mmproj.gguf"),
        lambda d: d.update(quantizations = ["Q4_0"]),
    ],
)
def test_a_malformed_export_is_not_served(tmp_path, edit):
    folder = _clef_folder(tmp_path)
    _export(folder)
    path = folder / "gguf" / "export.json"
    data = json.loads(path.read_text(encoding = "utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding = "utf-8")
    assert contract.read_export(folder) is None


def test_cpu_flags_hide_the_gpu_and_keep_weights_unrepacked(home, stub, tmp_path):
    model, mmproj = tmp_path / "m.gguf", tmp_path / "mm.gguf"
    model.write_bytes(b"GGUF")
    mmproj.write_bytes(b"GGUF")
    agent = native_worker.NativeClefAgent(model, mmproj, "clef-flash", gpu = False, ctx = 4096)
    try:
        start = stub.records("start")[0]
        argv = start["argv"]
        value = lambda flag: argv[argv.index(flag) + 1]
        assert (value("-m"), value("--mmproj"), value("--alias")) == (
            str(model),
            str(mmproj),
            "clef-flash",
        )
        assert (value("--host"), value("--parallel")) == ("127.0.0.1", "1")
        assert value("-c") == value("-b") == value("-ub") == "4096"
        assert (value("-ngl"), value("--device")) == ("0", "none")
        assert "--no-repack" in argv and "--no-mmproj-offload" in argv
        assert start["env"] == {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "-1"}
        assert agent.device == "cpu" and agent.accepts_images
    finally:
        agent.close()
    assert not agent.is_alive()


def test_the_key_is_passed_in_a_private_file_and_files_go_with_the_server(home, stub, tmp_path):
    from utils.paths.storage_roots import auth_root

    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    agent = native_worker.NativeClefAgent(model, None, "clef-flash", gpu = False)
    argv = stub.records("start")[0]["argv"]
    key_file, log_file = Path(argv[argv.index("--api-key-file") + 1]), agent._log_path
    assert "--api-key" not in argv and agent._key not in " ".join(argv)
    assert key_file.parent == auth_root() and key_file.read_text(encoding = "utf-8") == agent._key
    assert len(agent._key) >= 32
    if os.name != "nt":
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    second = native_worker.NativeClefAgent(model, None, "clef-flash", gpu = False)
    assert second._log_path != log_file and second._key_file != key_file
    for each in (agent, second):
        each.close()
    assert not key_file.exists() and not log_file.exists() and not second._key_file.exists()


def test_gpu_flags_offload_to_the_freest_device(home, stub, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    agent = native_worker.NativeClefAgent(model, None, "clef-ft:run", gpu = True)
    try:
        argv = stub.records("start")[0]["argv"]
        assert argv[argv.index("-ngl") + 1] == "-1" and argv[argv.index("--device") + 1] == "CUDA1"
        assert argv[argv.index("-ub") + 1] == str(native_worker.DEFAULT_CTX)
        assert "--mmproj" not in argv and "--no-repack" not in argv
        # Without a projector the server reads text only.
        assert agent.device == "CUDA1" and not agent.accepts_images
        api_key = agent._key
    finally:
        agent.close()
    second = native_worker.NativeClefAgent(model, None, "x", gpu = True)
    second.close()
    assert second._key != api_key


@pytest.mark.parametrize("mask, expected", [("1,0", "CUDA0"), ("0,1", "CUDA1"), ("5", "CUDA1")])
def test_freest_gpu_follows_the_visibility_order(stub, monkeypatch, mask, expected):
    # Physical GPU 1 is the freest; llama.cpp numbers devices in CUDA_VISIBLE_DEVICES order.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = mask)
    assert native_worker._pick_device("llama-server", env) == expected
    # Vulkan ordinals ignore the CUDA mask.
    monkeypatch.setattr(
        LlamaCppBackend, "_enumerated_gpu_devices", staticmethod(lambda *_: ["Vulkan0", "Vulkan1"])
    )
    assert native_worker._pick_device("llama-server", env) == "Vulkan1"
    # ROCm follows HIP_VISIBLE_DEVICES.
    monkeypatch.setattr(
        LlamaCppBackend, "_enumerated_gpu_devices", staticmethod(lambda *_: ["ROCm0", "ROCm1"])
    )
    hip = dict(os.environ, HIP_VISIBLE_DEVICES = mask, CUDA_VISIBLE_DEVICES = mask)
    hip.pop("ROCR_VISIBLE_DEVICES", None)
    assert native_worker._pick_device("llama-server", hip) == expected.replace("CUDA", "ROCm")
    hip.pop("HIP_VISIBLE_DEVICES")
    assert native_worker._pick_device("llama-server", hip) == expected.replace("CUDA", "ROCm")


@pytest.mark.parametrize("mode", ["nodecisions", "wrongalias"])
def test_a_server_without_a_decisions_output_is_not_capable(home, stub, tmp_path, mode):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    stub.set_mode(mode)
    with pytest.raises(native_worker.NativeIncapable, match = "decisions"):
        native_worker.NativeClefAgent(model, None, "clef-flash", gpu = False)


def test_a_server_that_dies_at_startup_is_a_startup_error(home, stub, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    stub.set_mode("exit")
    with pytest.raises(native_worker.NativeError, match = "exited while loading") as failed:
        native_worker.NativeClefAgent(model, None, "clef-flash", gpu = False)
    # The server's own output is quoted, and a server that died by itself keeps its log.
    assert "code 3" in str(failed.value) and "stub: failed to load model" in str(failed.value)
    from utils.paths.storage_roots import studio_root

    kept = list((studio_root() / "logs").glob("decision-llama-server-*.log"))
    assert len(kept) == 1 and "stub: failed to load model" in kept[0].read_text(encoding = "utf-8")


def test_the_binary_comes_from_studios_resolver(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding = "utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(binary))
    assert Path(native_worker.resolve_binary()).resolve() == binary.resolve()


def test_error_mapping():
    batch = native_worker.map_error(500, PHYSICAL_BATCH.format(tokens = 20158, limit = 16384))
    assert isinstance(batch, native_worker.NativeContextOverflow)
    assert (batch.tokens, batch.limit, batch.status) == (20158, 16384, 422)
    assert "at most 16384" in str(batch)
    for status, text in (
        (
            400,
            "request (9000 tokens) exceeds the available context size (8192 tokens), try increasing it",
        ),
        (400, "input (9000 tokens) is larger than the max context size (8192 tokens). skipping"),
        (
            400,
            "the question and its options (3000 tokens) are too large to process. increase the batch size (current batch size: 2048)",
        ),
    ):
        assert isinstance(
            native_worker.map_error(status, text), native_worker.NativeContextOverflow
        )
    # llama.cpp answers an image mtmd cannot decode with HTTP 500, from a healthy server.
    undecodable = native_worker.map_error(500, "Failed to load image or audio file")
    assert type(undecodable) is native_worker.NativeInputError and not undecodable.retire
    other = native_worker.map_error(400, 'questions.q: "type" must be one of: choice, score, noul')
    assert (
        type(other) is native_worker.NativeInputError and other.status == 422 and not other.retire
    )
    for status in (401, 403, 500, 502):
        failure = native_worker.map_error(status, "boom")
        assert (
            type(failure) is native_worker.NativeError and failure.status == 503 and failure.retire
        )


def test_answers_are_rebuilt_with_the_pytorch_formatter():
    # llama.cpp's own confidence for 0.8 / 0.2 is the normalised 0.6; PyTorch answers the winning probability.
    questions = {
        "c": {"type": "choice", "instructions": "x", "criteria": {"b": "B", "a": "A"}},
        "s": {"type": "score", "instructions": "x", "criteria": ["low", "high"]},
        "n": {"type": "noul", "instructions": "x", "criteria": None},
    }
    native = {
        "answers": {
            "c": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.8, "b": 0.2},
                "confidence": 0.6,
            },
            "s": {
                "type": "score",
                "score": 0.75,
                "probabilities": {"0": 0.25, "1": 0.75},
                "confidence": 0.5,
                "legend": {"0": "low", "1": "high"},
            },
            "n": {"type": "noul", "noul": 0.123456},
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }
    result = native_worker.normalise(native, questions)
    systemone_answer = _formatter()
    assert result["answers"] == {
        "c": systemone_answer(questions["c"], {"a": 0.8, "b": 0.2}),
        "s": systemone_answer(questions["s"], {"0": 0.25, "1": 0.75}),
        "n": systemone_answer(questions["n"], {"true": 0.123456, "false": 1 - 0.123456}),
    }
    assert result["answers"]["c"]["confidence"] == 0.8
    assert result["answers"]["s"]["confidence"] == 0.75
    assert result["answers"]["n"] == {"type": "noul", "noul": 0.1235}
    assert (result["input_tokens"], result["truncated"]) == (42, False)
    bad = {"answers": {**native["answers"], "c": {"probabilities": {"a": 0.8}}}}
    with pytest.raises(native_worker.NativeError, match = '"c"'):
        native_worker.normalise(bad, questions)


def test_auto_serves_a_stock_clef_through_llama_cpp(home, client, stub):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    settings = client.get("/api/settings/systemone").json()
    assert (settings["backend"], settings["effective_backend"]) == ("auto", "llama.cpp")
    assert settings["input_modalities"] == ["text", "image"]
    response = _post(client)
    assert response.status_code == 200, response.text
    assert response.headers["x-unsloth-decision-backend"] == "llama.cpp"
    body = response.json()
    assert body["model"] == "clef-flash" and body["usage"]["input_tokens"] > 0
    # GPU by default when there is one; the device is the CPU here (gpu_available is False).
    argv = stub.records("start")[0]["argv"]
    companion = catalog.GGUF_COMPANIONS["clef-flash"]
    assert argv[argv.index("-m") + 1].endswith(f"{companion.revision}/{companion.model}")
    assert argv[argv.index("--mmproj") + 1].endswith(companion.mmproj)
    wire = stub.records("request")[0]["body"]
    assert wire["model"] == "clef-flash"
    # A missing instruction is the question id, as in Cloudflare's reference encoder.
    assert wire["questions"]["angry"] == {"type": "noul", "instructions": "angry"}
    status = client.get("/api/settings/systemone").json()
    assert (status["loaded_model"], status["loaded_backend"], status["loaded_device"]) == (
        "clef-flash",
        "llama.cpp",
        "cpu",
    )
    assert home.torch_agents == []


def test_both_backends_answer_the_same_body(home, client, stub):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    native = _post(client)
    _put(client, backend = "pytorch")
    torch = _post(client)
    assert torch.headers["x-unsloth-decision-backend"] == "pytorch"
    assert home.torch_agents and native.headers["x-unsloth-decision-backend"] == "llama.cpp"
    native_body, torch_body = native.json(), torch.json()
    native_body["usage"] = torch_body["usage"] = None
    assert native_body == torch_body


def test_auto_without_llama_server_uses_pytorch_and_says_why(home, client, monkeypatch):
    _cache_gguf(home.cache)
    monkeypatch.setattr(native_worker, "resolve_binary", lambda: None)
    _put(client, enabled = True, model = "clef-flash")
    settings = client.get("/api/settings/systemone").json()
    assert settings["effective_backend"] == "pytorch"
    assert settings["fallback_reason"] == "llama-server is not installed."
    assert settings["input_modalities"] == ["text"]
    response = _post(client)
    assert (
        response.status_code == 200 and response.headers["x-unsloth-decision-backend"] == "pytorch"
    )
    refused = _post(client, images = [PNG])
    assert refused.status_code == 400
    assert "Images are served only by llama.cpp" in refused.json()["detail"]["message"]


def test_auto_retries_on_pytorch_when_the_build_has_no_decisions(home, client, stub):
    _cache_gguf(home.cache)
    stub.set_mode("nodecisions")
    _put(client, enabled = True, model = "clef-flash")
    response = _post(client)
    assert (
        response.status_code == 200 and response.headers["x-unsloth-decision-backend"] == "pytorch"
    )
    assert len(stub.records("start")) == 1
    assert _post(client).headers["x-unsloth-decision-backend"] == "pytorch"
    assert len(stub.records("start")) == 1
    assert (
        "cannot serve decision models"
        in client.get("/api/settings/systemone").json()["fallback_reason"]
    )


def test_forced_backends_give_clear_errors(home, client, stub, monkeypatch):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash", backend = "pytorch")
    refused = _post(client, images = [PNG])
    assert refused.status_code == 400 and "Auto or llama.cpp" in refused.json()["detail"]["message"]

    _put(client, backend = "llama.cpp")
    gap = {"one": {"type": "score", "instructions": "x", "criteria": ["only"]}}
    refused = _post(client, questions = gap)
    assert (
        refused.status_code == 422 and "at least two levels" in refused.json()["detail"]["message"]
    )
    monkeypatch.setattr(native_worker, "resolve_binary", lambda: None)
    refused = _post(client)
    assert refused.status_code == 503
    assert refused.json()["detail"]["message"].startswith("llama-server is not installed.")
    assert home.torch_agents == []

    # A fine-tune without an export has no GGUF at all.
    folder = _clef_folder(home.outputs, "clef_noexport")
    _put(client, model = catalog.CLEF_FINE_TUNE_PREFIX + folder.name)
    refused = _post(client)
    assert (
        refused.status_code == 400
        and "no current GGUF export" in refused.json()["detail"]["message"]
    )


def test_auto_sends_what_llama_cpp_cannot_parse_to_pytorch(home, client, stub):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    gap = {"one": {"type": "score", "instructions": "x", "criteria": ["only"]}}
    response = _post(client, questions = gap)
    assert (
        response.status_code == 200 and response.headers["x-unsloth-decision-backend"] == "pytorch"
    )
    assert stub.records("start") == []


def test_images_reach_llama_cpp_and_laya_refuses_them(home, client, stub):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    response = _post(client, images = [PNG])
    assert (
        response.status_code == 200
        and response.headers["x-unsloth-decision-backend"] == "llama.cpp"
    )
    assert stub.records("request")[0]["body"]["images"] == [PNG]
    for bad in (
        ["https://example.com/a.png"],
        ["data:image/gif;base64,R0lGOD"],
        ["data:image/png;base64,%%%"],
        [PNG] * 5,
    ):
        assert _post(client, images = bad).status_code == 422
    _put(client, model = "laya-multilingual")
    refused = _post(client, images = [PNG])
    assert (
        refused.status_code == 400 and "Laya reads text only" in refused.json()["detail"]["message"]
    )


@pytest.mark.parametrize("ctx, retried", [(1024, True), (16384, False)])
def test_auto_retries_an_over_long_state_on_pytorch_only_when_it_is_longer(
    home, client, stub, ctx, retried
):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash", native_ctx = ctx)
    long_state = "word " * (ctx + 50)
    response = _post(client, state = long_state)
    if retried:
        assert (
            response.status_code == 200
            and response.headers["x-unsloth-decision-backend"] == "pytorch"
        )
        status = client.get("/api/settings/systemone").json()
        assert status["loaded_backend"] == "pytorch"
        assert status["fallback_reason"] == "The request is longer than the llama.cpp context."
        # The resident PyTorch model keeps serving text under Auto instead of reloading.
        assert _post(client).headers["x-unsloth-decision-backend"] == "pytorch"
        assert len(stub.records("start")) == 1
    else:
        assert response.status_code == 422
        assert f"at most {ctx}" in response.json()["detail"]["message"]
        assert home.torch_agents == []


def test_other_native_failures_are_surfaced_not_retried(home, client, stub):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    rejected = _post(client, state = "reject")
    assert rejected.status_code == 422 and "criteria" in rejected.json()["detail"]["message"]
    assert _post(client).status_code == 200
    crashed = _post(client, state = "crash")
    assert (
        crashed.status_code == 503 and crashed.json()["detail"]["error_type"] == "model_unavailable"
    )
    assert home.torch_agents == []
    # The crashed server was retired; the next request starts a fresh one.
    deadline = time.monotonic() + 5
    while laya_runtime._agent is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _post(client).status_code == 200
    assert len(stub.records("start")) == 2
    # A context overflow below PyTorch's window is the one failure Auto answers on PyTorch.
    overflow = _post(client, state = "context")
    assert (
        overflow.status_code == 200 and overflow.headers["x-unsloth-decision-backend"] == "pytorch"
    )
    _put(client, backend = "llama.cpp")
    overflow = _post(client, state = "context")
    assert overflow.status_code == 422 and "at most 8192" in overflow.json()["detail"]["message"]


def test_training_and_the_gpu_arbiter_gate_a_gpu_server(home, client, stub, monkeypatch):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash", device = "gpu")
    home.training = True
    busy = _post(client)
    assert busy.status_code == 503 and "training run" in busy.json()["detail"]["message"]
    assert stub.records("start") == []
    home.training = False
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    # Auto answers text on PyTorch beside the chat model, as before llama.cpp served decisions.
    beside = _post(client)
    assert beside.status_code == 200 and beside.headers["x-unsloth-decision-backend"] == "pytorch"
    assert client.get("/api/settings/systemone").json()["fallback_reason"] == (
        "Another model is using the GPU."
    )
    # Only llama.cpp reads images: those still wait for the GPU.
    busy = _post(client, images = [PNG])
    assert (
        busy.status_code == 409 and "Unload the resident chat" in busy.json()["detail"]["message"]
    )
    laya_runtime.unload()
    _put(client, backend = "llama.cpp")
    busy = _post(client)
    assert (
        busy.status_code == 409 and "Unload the resident chat" in busy.json()["detail"]["message"]
    )
    assert stub.records("start") == []
    _put(client, backend = "auto")
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    assert _post(client).headers["x-unsloth-decision-backend"] == "llama.cpp"
    assert gpu_arbiter.current_owner() == gpu_arbiter.DECISIONS
    # A chat load evicts the idle server through the arbiter.
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.CHAT, lambda: None)
    gpu_arbiter.acquire_for(gpu_arbiter.CHAT)
    assert laya_runtime._agent is None and gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    # Training that starts while it is resident ends it and answers 503.
    assert _post(client).status_code == 200
    home.training = True
    assert _post(client).status_code == 503
    assert laya_runtime._agent is None and gpu_arbiter.current_owner() is None


def test_a_cpu_server_keeps_serving_during_training(home, client, stub):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash", device = "cpu")
    home.training = True
    assert _post(client).status_code == 200
    # A request only PyTorch can answer waits for the GPU, without ending the CPU server.
    gap = {"one": {"type": "score", "instructions": "x", "criteria": ["only"]}}
    assert _post(client, questions = gap).status_code == 503
    assert laya_runtime._agent is not None
    assert _post(client).status_code == 200
    assert len(stub.records("start")) == 1


def test_a_re_export_replaces_the_resident_server(home, client, stub):
    folder = _clef_folder(home.outputs, "clef_reexported")
    _export(folder)
    _put(client, enabled = True, model = catalog.CLEF_FINE_TUNE_PREFIX + folder.name)
    assert _post(client).headers["x-unsloth-decision-backend"] == "llama.cpp"
    time.sleep(0.01)
    replacement = folder / contract.EXPORT_DIR / "model-Q8_0.gguf.tmp"
    replacement.write_bytes(b"GGUF re-exported")
    os.replace(replacement, folder / contract.EXPORT_DIR / "model-Q8_0.gguf")
    assert _post(client).headers["x-unsloth-decision-backend"] == "llama.cpp"
    assert len(stub.records("start")) == 2


def test_a_fine_tune_serves_its_current_export_and_ignores_a_stale_one(home, client, stub):
    folder = _clef_folder(home.outputs, "clef_exported")
    name = catalog.CLEF_FINE_TUNE_PREFIX + folder.name
    _put(client, enabled = True, model = name)
    assert _post(client).headers["x-unsloth-decision-backend"] == "pytorch"
    _export(folder)
    laya_runtime.unload()
    response = _post(client)
    assert response.headers["x-unsloth-decision-backend"] == "llama.cpp"
    argv = stub.records("start")[0]["argv"]
    assert argv[argv.index("-m") + 1] == str(folder / "gguf" / "model-Q8_0.gguf")
    assert argv[argv.index("--alias") + 1] == name
    # Recalibrating the folder makes the export stale: PyTorch again.
    laya_runtime.unload()
    time.sleep(0.01)
    (folder / "unsloth_decision_config.json").write_text(
        '{"temperature": [2, 2, 2]}', encoding = "utf-8"
    )
    assert _post(client).headers["x-unsloth-decision-backend"] == "pytorch"


def test_laya_serving_is_unchanged(home, client, stub, monkeypatch):
    monkeypatch.setattr(
        laya_runtime, "_load_checkpoint", lambda checkpoint: (SimpleNamespace(), "cpu")
    )
    monkeypatch.setattr(
        laya_runtime,
        "_predict",
        lambda agent, state, questions: (
            {
                "answers": {n: {"type": "noul", "noul": 0.5, "confidence": 0.5} for n in questions},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
            False,
        ),
    )
    _put(client, enabled = True, model = "laya-multilingual", backend = "llama.cpp")
    response = _post(client, questions = {"q": {"type": "noul", "instructions": "x"}})
    assert (
        response.status_code == 200 and response.headers["x-unsloth-decision-backend"] == "pytorch"
    )
    assert stub.records("start") == []


def test_an_idle_server_unloads_and_a_stale_timer_does_nothing(home, client, stub, monkeypatch):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    monkeypatch.setattr(laya_runtime, "IDLE_UNLOAD_S", 0.3)
    assert _post(client).status_code == 200
    agent = laya_runtime._agent
    generation = laya_runtime._generation
    # Generous: a loaded host can take seconds to reap the stub server; the loop exits early.
    deadline = time.monotonic() + 30
    while laya_runtime._agent is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert laya_runtime._agent is None and not agent.is_alive()
    # A timer from that generation never touches the server loaded after it.
    assert _post(client).status_code == 200
    newer = laya_runtime._agent
    monkeypatch.setattr(laya_runtime, "_last_used", 0.0)
    laya_runtime._idle_unload(generation)
    assert laya_runtime._agent is newer and newer.is_alive()


def test_a_load_that_loses_its_generation_is_not_published(home, stub, monkeypatch, tmp_path):
    _cache_gguf(home.cache)
    target, _ = laya_runtime.select(catalog.CHECKPOINTS["clef-flash"])
    real = native_worker.NativeClefAgent
    built = []

    def racing(*args, **kwargs):
        agent = real(*args, **kwargs)
        built.append(agent)
        # Something unloaded the Decision API while this server started.
        laya_runtime._unloads += 1
        return agent

    monkeypatch.setattr(native_worker, "NativeClefAgent", racing)
    laya_runtime._load(target)
    assert laya_runtime._agent is None and not built[0].is_alive()


def test_the_model_list_shows_image_input_only_where_llama_cpp_serves_it(
    home, client, stub, monkeypatch
):
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    listed = {
        m["id"]: m["architecture"]["input_modalities"] for m in systemone.decision_model_objects()
    }
    assert listed["clef-flash"] == listed["clef"] == listed["default"] == ["text", "image"]
    assert listed["laya-multilingual"] == ["text"]
    _put(client, backend = "pytorch")
    listed = {
        m["id"]: m["architecture"]["input_modalities"] for m in systemone.decision_model_objects()
    }
    assert listed["clef-flash"] == ["text"]


@pytest.mark.parametrize(
    "owner, status", [(gpu_arbiter.DECISIONS, 409), (gpu_arbiter.DIFFUSION, None)]
)
def test_a_chat_load_refused_by_a_busy_decision_server_is_retryable(owner, status):
    import asyncio
    from unittest.mock import MagicMock, patch

    from fastapi import HTTPException

    from models.inference import LoadRequest
    from routes import inference as inference_route

    path = "unsloth/Qwen3-0.6B-GGUF"
    with (
        patch.object(
            inference_route,
            "_resolve_model_identifier_for_request",
            return_value = (path, path, False),
        ),
        patch.object(
            inference_route, "resolve_effective_chat_template_override", return_value = None
        ),
        patch.object(
            inference_route, "get_inference_backend", return_value = MagicMock(active_model_name = None)
        ),
        patch.object(inference_route, "get_llama_cpp_backend", return_value = MagicMock()),
        patch.object(
            inference_route.ModelConfig,
            "from_identifier",
            side_effect = gpu_arbiter.GpuOwnerBusyError(owner),
        ),
    ):
        load = inference_route.load_model(LoadRequest(model_path = path), MagicMock(), "unsloth")
        if status is None:
            # The preview route maps the image/video refusal itself.
            with pytest.raises(gpu_arbiter.GpuOwnerBusyError):
                asyncio.run(load)
            return
        with pytest.raises(HTTPException) as refused:
            asyncio.run(load)
    assert refused.value.status_code == 409 and refused.value.headers["Retry-After"] == "5"
    assert refused.value.detail == "The Decision API is using the GPU; retry shortly."


def _image_url(kind, data):
    import base64
    return f"data:image/{kind};base64," + base64.b64encode(data).decode()


def test_an_unreadable_image_is_refused_and_the_server_keeps_serving(home, client, stub):
    import io

    from PIL import Image

    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    webp = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(webp, "WEBP")
    jpeg = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(jpeg, "JPEG")
    for bad in (
        _image_url("webp", webp.getvalue()),
        _image_url("png", b"\x89PNG\r\n\x1a\n" + b"garbage" * 20),
        _image_url("png", webp.getvalue()),
    ):
        refused = _post(client, images = [bad])
        assert refused.status_code == 422, refused.text
    assert stub.records("start") == []
    assert _post(client, images = [_image_url("jpeg", jpeg.getvalue())]).status_code == 200
    # One the server still cannot decode is the caller's error, not a crash.
    refused = _post(client, state = "badimage", images = [PNG])
    assert refused.status_code == 422 and "Failed to load image" in refused.text
    assert _post(client).status_code == 200
    assert len(stub.records("start")) == 1


def test_a_slow_close_keeps_the_claim_of_a_server_loading_after_it(home, monkeypatch, tmp_path):
    import threading

    native = laya_runtime._native_target(catalog.CHECKPOINTS["clef-flash"])
    closing, closed = threading.Event(), threading.Event()

    class Server:
        backend, gpu, device = "llama.cpp", True, "CUDA0"

        def __init__(self, slow = False):
            self.slow = slow

        def close(self):
            if self.slow:
                closing.set()
                # llama-server takes its time to exit; the next load starts meanwhile.
                time.sleep(0.3)
                closed.set()

    def starting(*args, **kwargs):
        assert closed.wait(5)
        return Server()

    monkeypatch.setattr(laya_runtime, "_native_gpu", lambda: True)
    monkeypatch.setattr(laya_runtime, "_native_files", lambda *a, **k: (tmp_path / "m.gguf", None))
    monkeypatch.setattr(native_worker, "NativeClefAgent", starting)
    gpu_arbiter.acquire_for(gpu_arbiter.DECISIONS, allow_evict = False)
    old = Server(slow = True)
    monkeypatch.setattr(laya_runtime, "_agent", old)
    monkeypatch.setattr(laya_runtime, "_loaded", native)
    retire = threading.Thread(target = laya_runtime._evict_agent, args = (old,))
    retire.start()
    assert closing.wait(5)
    monkeypatch.setattr(laya_runtime, "_loading", native)
    laya_runtime._load(native)
    retire.join(5)
    assert isinstance(laya_runtime._agent, Server) and laya_runtime._agent is not old
    assert gpu_arbiter.current_owner() == gpu_arbiter.DECISIONS
    # With nothing loading or resident, a close hands the GPU back.
    laya_runtime.unload()
    assert gpu_arbiter.current_owner() is None


def test_auto_keeps_pytorch_while_the_gguf_is_not_downloaded(home, client, stub):
    _put(client, enabled = True, model = "clef-flash")
    clef = catalog.CHECKPOINTS["clef-flash"]
    target, reason = laya_runtime.select(clef)
    assert target == clef and "not downloaded" in reason
    assert client.get("/api/settings/systemone").json()["effective_backend"] == "pytorch"
    response = _post(client)
    assert response.headers["x-unsloth-decision-backend"] == "pytorch"
    assert stub.records("start") == []
    # Images and a forced llama.cpp still take the GGUF.
    assert laya_runtime.select(clef, images = [PNG])[0].backend == "llama.cpp"
    assert laya_runtime.select(clef, preference = "llama.cpp")[0].backend == "llama.cpp"
    laya_runtime.unload()
    _cache_gguf(home.cache)
    assert laya_runtime.select(clef) == (laya_runtime._native_target(clef), None)


def test_an_over_long_state_is_not_retried_where_pytorch_cannot_serve_clef(
    home, client, stub, monkeypatch
):
    from utils.hardware import hardware

    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.MLX)
    monkeypatch.setattr(hardware, "DEVICE", hardware.DeviceType.MLX)
    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash", native_ctx = 1024)
    response = _post(client, state = "word " * 1100)
    assert response.status_code == 422 and "at most 1024" in response.json()["detail"]["message"]
    assert home.torch_agents == []


def _cache_blob(
    cache: Path,
    name: str,
    revision: str,
    sha256s = None,
) -> None:
    # The Hub cache layout: snapshot files link to blobs named by their LFS sha256.
    companion = catalog.GGUF_COMPANIONS[name]
    repo = cache / f"models--{companion.repo.replace('/', '--')}"
    (repo / "blobs").mkdir(parents = True, exist_ok = True)
    (repo / "snapshots" / revision).mkdir(parents = True)
    files = [f for f in (companion.model, companion.mmproj) if f]
    for file, sha256 in zip(files, sha256s or companion.sha256):
        (repo / "blobs" / sha256).write_bytes(b"GGUF")
        (repo / "snapshots" / revision / file).symlink_to(Path("..", "..", "blobs", sha256))


def test_gguf_entries_are_listed_as_llama_cpp_only(home, client, stub):
    gguf = {n for n, c in catalog.CHECKPOINTS.items() if c.layout == "gguf"}
    assert gguf == {
        "kev-0.8b",
        "kev-4b",
        "kev-9b",
        "lev",
        "bespoke-nimble-9b-v3",
        "openjev",
        "laya-gguf",
        "julia-1",
    }
    for name in gguf:
        checkpoint, companion = catalog.CHECKPOINTS[name], catalog.GGUF_COMPANIONS[name]
        assert checkpoint.backend == "llama.cpp" and checkpoint.source == companion.repo
        assert len(companion.sha256) == (2 if companion.mmproj else 1)
    assert systemone_settings.DEFAULT_MODEL == "laya-multilingual"
    _put(client, enabled = True)
    options = {m["name"]: m for m in client.get("/api/settings/systemone").json()["models"]}
    assert options["kev-0.8b"]["llama_cpp_only"] and options["kev-0.8b"]["label"] == "Kev 0.8B"
    assert options["kev-0.8b"]["available"] and options["kev-0.8b"]["download_bytes"] == 812_406_304
    assert not options["clef-flash"]["llama_cpp_only"] and options["clef-flash"]["label"] is None
    listed = {
        m["id"]: m["architecture"]["input_modalities"] for m in systemone.decision_model_objects()
    }
    assert listed["openjev"] == ["text", "image"] and listed["kev-0.8b"] == ["text"]
    stub.set_mode("nodecisions")
    laya_runtime._incapable.add(laya_runtime._binary_key(str(stub.binary)))
    option = {m["name"]: m for m in client.get("/api/settings/systemone").json()["models"]}["lev"]
    assert (
        not option["available"] and "cannot serve decision models" in option["unavailable_reason"]
    )


def test_a_gguf_entry_is_served_only_by_llama_cpp(home, client, stub, monkeypatch):
    _cache_gguf(home.cache, "kev-0.8b")
    _put(client, enabled = True, model = "kev-0.8b")
    settings = client.get("/api/settings/systemone").json()
    assert (settings["effective_backend"], settings["input_modalities"]) == ("llama.cpp", ["text"])
    response = _post(client)
    assert response.status_code == 200, response.text
    assert response.headers["x-unsloth-decision-backend"] == "llama.cpp"
    argv = stub.records("start")[0]["argv"]
    companion = catalog.GGUF_COMPANIONS["kev-0.8b"]
    assert argv[argv.index("-m") + 1].endswith(f"{companion.revision}/{companion.model}")
    assert "--mmproj" not in argv and argv[argv.index("--alias") + 1] == "kev-0.8b"
    refused = _post(client, images = [PNG])
    assert (
        refused.status_code == 422 and "no vision projector" in refused.json()["detail"]["message"]
    )

    _put(client, backend = "pytorch")
    settings = client.get("/api/settings/systemone").json()
    assert settings["effective_backend"] is None
    assert "served only by llama.cpp" in settings["fallback_reason"]
    refused = _post(client)
    assert refused.status_code == 400
    assert "Auto or llama.cpp" in refused.json()["detail"]["message"]

    _put(client, backend = "llama.cpp")
    assert _post(client).status_code == 200
    monkeypatch.setattr(native_worker, "resolve_binary", lambda: None)
    _put(client, backend = "auto")
    refused = _post(client)
    assert refused.status_code == 503
    assert refused.json()["detail"]["message"] == (
        "kev-0.8b needs llama.cpp: llama-server is not installed."
    )
    assert home.torch_agents == []


def test_a_build_without_decisions_is_a_clear_error_for_a_gguf_entry(home, client, stub):
    _cache_gguf(home.cache, "julia-1")
    stub.set_mode("nodecisions")
    _put(client, enabled = True, model = "julia-1")
    for _ in range(2):
        response = _post(client)
        assert response.status_code == 503
        assert response.json()["detail"]["message"] == (
            "julia-1 needs llama.cpp: This llama.cpp build cannot serve decision models; "
            "update llama.cpp in Studio."
        )
    assert len(stub.records("start")) == 1 and home.torch_agents == []


def test_a_gguf_entry_passes_llama_cpp_answers_through(home, client, stub):
    _cache_gguf(home.cache, "laya-gguf")
    _put(client, enabled = True, model = "laya-gguf")
    response = _post(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "laya-gguf"
    assert body["usage"] == {"input_tokens": 14, "output_tokens": 0}
    # llama.cpp's normalised confidence, not the Clef formatter's winning probability.
    assert body["answers"]["route"] == {
        "type": "choice",
        "choice": "shipping",
        "confidence": pytest.approx((2 / 3 - 1 / 2) / (1 - 1 / 2)),
        "probabilities": {"billing": pytest.approx(1 / 3), "shipping": pytest.approx(2 / 3)},
    }
    assert body["answers"]["urgency"]["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert body["answers"]["urgency"]["score"] == pytest.approx(2 / 6 + 2 * 3 / 6)
    assert body["answers"]["angry"] == {"type": "noul", "noul": 0.7}

    # An over-long state is refused, never retried on PyTorch.
    _put(client, native_ctx = 1024)
    response = _post(client, state = "word " * 1100)
    assert response.status_code == 422 and "at most 1024" in response.json()["detail"]["message"]
    assert home.torch_agents == []


def test_gguf_answers_are_checked_before_they_pass_through():
    questions = {"c": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}}
    good = {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.8, "b": 0.2},
        "confidence": 0.6,
    }
    result = native_worker.normalise({"answers": {"c": good}}, questions, clef_answers = False)
    assert result["answers"]["c"] == good
    for bad in (
        {**good, "choice": "z"},
        {**good, "confidence": None},
        {**good, "probabilities": {"a": 0.8}},
    ):
        with pytest.raises(native_worker.NativeError, match = '"c"'):
            native_worker.normalise({"answers": {"c": bad}}, questions, clef_answers = False)


def test_the_download_plan_pins_the_revision_and_main_with_the_same_blob_counts(home, client):
    companion = catalog.GGUF_COMPANIONS["openjev"]
    checkpoint = catalog.CHECKPOINTS["openjev"]
    assert laya_runtime.download_plan(checkpoint, preference = "pytorch")["revision"] == (
        companion.revision
    )
    plan = client.get("/api/settings/systemone/resolve", params = {"model": "openjev"}).json()
    assert plan == {
        "repo": "ggml-org/OpenJev-GGUF",
        "files": ["OpenJev-Q8_0.gguf", "mmproj-OpenJev-Q8_0.gguf"],
        "size_bytes": 28_595_765_408 + 629_247_232,
        "cached": False,
        "error": None,
    }
    # The settings download fetches main: another commit holding different bytes is not the model.
    _cache_blob(home.cache, "openjev", "f" * 40, ("0" * 64, "1" * 64))
    assert not laya_runtime.is_cached(checkpoint)
    _cache_blob(home.cache, "openjev", "e" * 40)
    assert laya_runtime.is_cached(checkpoint)
    model, mmproj = laya_runtime._native_files(checkpoint, local_only = True)
    assert model.parent.name == mmproj.parent.name == "e" * 40
    assert client.get("/api/settings/systemone/resolve", params = {"model": "openjev"}).json()[
        "cached"
    ]
    # The Clef companions resolve the same way.
    _cache_blob(home.cache, "clef-flash", "e" * 40)
    assert laya_runtime.is_cached(laya_runtime._native_target(catalog.CHECKPOINTS["clef-flash"]))


def test_an_image_that_decodes_to_too_many_pixels_is_refused(home, client, stub):
    import io

    from PIL import Image

    _cache_gguf(home.cache)
    _put(client, enabled = True, model = "clef-flash")
    big = io.BytesIO()
    Image.new("1", (8192, 8192)).save(big, "PNG")
    assert len(big.getvalue()) < 4 * 1024 * 1024
    refused = _post(client, images = [_image_url("png", big.getvalue())])
    assert refused.status_code == 422 and "4096 x 4096" in refused.text
    assert stub.records("start") == []


def test_a_laya_model_never_reports_an_earlier_clefs_fallback_reason(monkeypatch):
    from core.systemone import laya_runtime

    monkeypatch.setattr(laya_runtime, "_fallback_reason", "The GGUF is not downloaded.")
    clef = SimpleNamespace(name = "clef-flash", backend = "pytorch", layout = "clef")
    laya = SimpleNamespace(name = "laya-multilingual", backend = "pytorch", layout = "laya")
    monkeypatch.setattr(laya_runtime, "_loaded", clef)
    assert laya_runtime.status()["fallback_reason"] == "The GGUF is not downloaded."
    monkeypatch.setattr(laya_runtime, "_loaded", laya)
    assert laya_runtime.status()["fallback_reason"] is None
    # A Laya request leaves the reason alone, so a concurrent Clef load keeps its own.
    monkeypatch.setattr(laya_runtime, "select", lambda checkpoint, *a, **k: (checkpoint, None))
    monkeypatch.setattr(laya_runtime, "_decide", lambda *_: {"ok": True})
    assert laya_runtime._route(laya, "s", {}, None) == {"ok": True}
    assert laya_runtime._fallback_reason == "The GGUF is not downloaded."


def test_a_uuid_mask_keeps_its_order_when_choosing_the_gpu(stub, monkeypatch):
    from utils.hardware import nvidia

    monkeypatch.setattr(nvidia, "resolve_uuid_mask", lambda mask: {"GPU-b,GPU-a": [1, 0]}.get(mask))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "GPU-b,GPU-a")
    assert native_worker._pick_device("llama-server", env) == "CUDA0"


def test_a_no_torch_install_finds_the_gpu_through_llama_cpp(monkeypatch):
    from utils import systemone_settings
    from utils.hardware import hardware

    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CPU)
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_free_memory", staticmethod(lambda *_, **__: [(0, 9000)])
    )
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: None)
    assert not systemone_settings.gpu_available()
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: "no torch")
    assert systemone_settings.gpu_available()
