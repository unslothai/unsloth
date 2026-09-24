# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Lifecycle and isolation tests using small environments and a real HTTP peer."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.inference import engine_install as install
from core.inference.managed_engine import ManagedEngine, launch_arguments


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    monkeypatch.setattr(install, "support_reason", lambda engine = "vllm": None)
    monkeypatch.setattr(install, "_jobs", {})
    monkeypatch.setattr(install, "_cancels", {})
    # Studio without torch: engines get a complete isolated environment.
    monkeypatch.setattr(install, "_studio_packages", lambda: {})
    return tmp_path


def active(
    root,
    engine = "vllm",
    directory = "env-prior",
):
    folder = root / engine / directory
    (folder / "bin").mkdir(parents = True)
    (folder / "bin" / "python").touch()
    marker = root / engine / "active.json"
    marker.write_text(json.dumps({"directory": directory, "version": "old"}))
    return marker


def test_failed_update_preserves_active_environment(isolated, monkeypatch):
    marker = active(isolated)
    before = marker.read_bytes()
    monkeypatch.setattr(install.shutil, "which", lambda _: "/uv")

    def fail(*args):
        raise RuntimeError("download failed")

    monkeypatch.setattr(install, "_run", fail)
    install._install("vllm", threading.Event())
    assert marker.read_bytes() == before
    assert install.status("vllm")["job"]["state"] == "error"
    assert [p.name for p in marker.parent.iterdir() if p.is_dir()] == ["env-prior"]


def test_activation_only_after_check_and_keeps_previous(isolated, monkeypatch):
    marker = active(isolated)
    calls = []
    monkeypatch.setattr(install.shutil, "which", lambda _: "/uv")

    def run(engine, argv, cancel):
        assert json.loads(marker.read_text())["directory"] == "env-prior"
        calls.append(argv)
        if argv[1] == "venv":
            destination = Path(argv[-1])
            (destination / "bin").mkdir(parents = True)
            (destination / "bin" / "python").touch()

    monkeypatch.setattr(install, "_run", run)
    install._install("vllm", threading.Event())
    result = install.installed("vllm")
    assert result["directory"] != "env-prior"
    assert result["previous_directory"] == "env-prior"
    assert any("--require-hashes" in argv for argv in calls)
    assert any(install._CHECK in argv for argv in calls)
    assert install.status("vllm")["job"]["state"] == "success"
    install.rollback("vllm")
    assert install.installed("vllm")["directory"] == "env-prior"
    install.rollback("vllm")
    assert install.installed("vllm")["directory"] == result["directory"]


def fake_venv(engine, argv, cancel):
    if argv[1] == "venv":
        destination = Path(argv[-1])
        (destination / "bin").mkdir(parents = True)
        (destination / "bin" / "python").touch()
        (destination / "lib" / "python3.13" / "site-packages").mkdir(parents = True)


def studio_with_engine_torch(monkeypatch, engine, **changes):
    pins = {name: version for name, (version, _) in install._pins(engine).items()}
    runtime = {"torch", "triton", "nvidia-cublas", "nvidia-nccl-cu13"}
    packages = {name: pins[name] for name in (*runtime, "numpy", "fastapi")}
    packages.update(changes)
    monkeypatch.setattr(install, "_studio_packages", lambda: dict(packages))
    monkeypatch.setattr(install, "_torch_runtime", lambda: runtime)
    monkeypatch.setattr(install.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(install.sys.implementation, "name", "cpython")
    monkeypatch.setattr(install.platform, "python_version", lambda: "3.13.0")
    return packages


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_shared_environment_installs_only_what_studio_lacks(isolated, monkeypatch, engine):
    studio = studio_with_engine_torch(monkeypatch, engine, fastapi = "0.0.1")
    monkeypatch.setattr(install.shutil, "which", lambda _: "/uv")
    monkeypatch.setattr(install, "_run", fake_venv)
    install._install(engine, threading.Event())
    info = install.installed(engine)
    assert info["shared"] is True
    # Only packages at the locked version are shared; a different fastapi is installed.
    assert info["provided"] == {
        name: version for name, version in studio.items() if name != "fastapi"
    }
    packages = (Path(info["path"]) / "engine-requirements.txt").read_text()
    assert not any(
        line.startswith(("torch==", "triton==", "numpy==")) for line in packages.splitlines()
    )
    assert any(line.startswith((engine + "==", "fastapi==")) for line in packages.splitlines())
    pth = Path(info["path"]) / "lib" / "python3.13" / "site-packages" / install._BASE_PTH
    assert "site.addsitedir(" in pth.read_text()
    assert install.status(engine)["current"] is True

    # A Studio update that changes a provided package invalidates the environment.
    studio["numpy"] = "0.0.1"
    status = install.status(engine)
    assert status["installed"] is True and status["current"] is False
    from core.inference import managed_engine
    from types import SimpleNamespace

    monkeypatch.setattr(managed_engine, "support_reason", lambda *args: None)
    request = SimpleNamespace(gpu_ids = [0], gguf_variant = None, model_path = "m")
    with pytest.raises(ValueError, match = "Repair it"):
        managed_engine.validate_load(engine, request)


def test_changed_studio_torch_uses_an_isolated_environment(isolated, monkeypatch):
    studio_with_engine_torch(monkeypatch, "vllm", torch = "2.99.0")
    monkeypatch.setattr(install.shutil, "which", lambda _: "/uv")
    calls = []

    def run(engine, argv, cancel):
        calls.append(argv)
        fake_venv(engine, argv, cancel)

    monkeypatch.setattr(install, "_run", run)
    install._install("vllm", threading.Event())
    info = install.installed("vllm")
    assert info["shared"] is False and info["provided"] == {}
    packages = (Path(info["path"]) / "engine-requirements.txt").read_text()
    assert packages == "".join(line for _, line in install._pins("vllm").values())
    assert not (
        Path(info["path"]) / "lib" / "python3.13" / "site-packages" / install._BASE_PTH
    ).exists()
    assert install.status("vllm")["current"] is True


def test_rollback_refuses_a_shared_environment_studio_no_longer_matches(isolated, monkeypatch):
    studio = studio_with_engine_torch(monkeypatch, "vllm")
    monkeypatch.setattr(install.shutil, "which", lambda _: "/uv")
    monkeypatch.setattr(install, "_run", fake_venv)
    install._install("vllm", threading.Event())
    install._install("vllm", threading.Event())
    assert install.status("vllm")["can_rollback"] is True
    studio["numpy"] = "0.0.1"
    assert install.status("vllm")["can_rollback"] is False
    with pytest.raises(RuntimeError, match = "no longer has"):
        install.rollback("vllm")


def test_engine_check_sees_every_locked_version_and_requirement(tmp_path):
    import os
    import subprocess
    import sys

    site = tmp_path / "site"
    (site / "demo-1.0.dist-info").mkdir(parents = True)
    (site / "demo-1.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\nRequires-Dist: absent-dependency>=1\n"
    )
    lock = tmp_path / "lock.txt"
    env = {**os.environ, "PYTHONPATH": str(site)}

    def check(pins, omitted = ""):
        lock.write_text(pins)
        return subprocess.run(
            [sys.executable, "-c", install._CHECK, str(lock), omitted],
            capture_output = True,
            text = True,
            env = env,
        )

    assert (
        check("demo==2.0 \\\n    --hash=sha256:00\n").stderr.strip()
        == "demo==2.0 is required, found 1.0"
    )
    assert "demo requires absent-dependency>=1, found None" in check("demo==1.0\n").stderr
    assert check("demo==1.0\n", "absent-dependency").returncode == 0


def test_cancel_before_activation_keeps_previous(isolated, monkeypatch):
    marker = active(isolated)
    monkeypatch.setattr(install.shutil, "which", lambda _: "/uv")
    cancelled = threading.Event()

    def run(*args):
        cancelled.set()

    monkeypatch.setattr(install, "_run", run)
    install._install("vllm", cancelled)
    assert json.loads(marker.read_text())["directory"] == "env-prior"
    assert install.status("vllm")["job"]["state"] == "cancelled"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_explicit_rollback_allows_old_profile_until_replaced(isolated, monkeypatch, engine):
    marker = active(isolated, engine)
    monkeypatch.setattr(install.shutil, "which", lambda _: "/uv")

    def run(engine, argv, cancel):
        if argv[1] == "venv":
            destination = Path(argv[-1])
            (destination / "bin").mkdir(parents = True)
            (destination / "bin" / "python").touch()

    monkeypatch.setattr(install, "_run", run)
    assert install.status(engine)["current"] is False
    assert install.status(engine)["restored"] is False
    install._install(engine, threading.Event())
    assert install.status(engine)["current"] is True
    install.rollback(engine)
    # Restoration survives a new process reading the on-disk marker.
    monkeypatch.setattr(install, "_jobs", {})
    assert install.status(engine)["current"] is False
    assert install.status(engine)["restored"] is True
    before = marker.read_bytes()

    def fail(*args):
        raise RuntimeError("download failed")

    monkeypatch.setattr(install, "_run", fail)
    install._install(engine, threading.Event())
    assert marker.read_bytes() == before
    assert install.status(engine)["restored"] is True
    monkeypatch.setattr(install, "_run", run)
    install._install(engine, threading.Event())
    assert install.status(engine)["current"] is True
    assert install.status(engine)["restored"] is False


def test_runtime_lease_blocks_removal(isolated):
    marker = active(isolated)
    with install.engine_lease("vllm"):
        assert install.status("vllm")["in_use"] is True
        with pytest.raises(RuntimeError, match = "another Studio"):
            install.remove("vllm")
    assert marker.exists()
    install.remove("vllm")
    assert not marker.exists()


def test_removal_keeps_shared_models_and_cache(isolated):
    active(isolated)
    shared = isolated / "shared-model.safetensors"
    shared.write_bytes(b"model")
    install.remove("vllm")
    assert shared.read_bytes() == b"model"


def test_marker_cannot_escape_engine_root(isolated):
    marker = active(isolated)
    marker.write_text(json.dumps({"directory": "../env-other"}))
    assert install.installed("vllm") is None


def test_unknown_engine_cannot_become_a_path(isolated):
    with pytest.raises(ValueError):
        install.remove("../vllm")


def test_installer_does_not_inherit_base_python_or_secrets(monkeypatch):
    for key in ("HF_TOKEN", "PYTHONPATH", "VIRTUAL_ENV", "UV_OVERRIDE", "PIP_INDEX_URL"):
        monkeypatch.setenv(key, "must-not-inherit")
    env = install.install_environment()
    assert not any(
        env.get(key) == "must-not-inherit"
        for key in (
            "HF_TOKEN",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "UV_OVERRIDE",
            "PIP_INDEX_URL",
        )
    )
    assert env["UV_LINK_MODE"] == "copy"


def test_installer_reuses_recorded_cache(monkeypatch, tmp_path):
    from utils.paths import storage_roots

    cache = tmp_path / "existing cache"
    cache.mkdir()
    (tmp_path / "uv-cache-dir").write_text(str(cache) + "\n")
    monkeypatch.setattr(storage_roots, "cache_root", lambda: tmp_path)
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    assert install.install_environment()["UV_CACHE_DIR"] == str(cache)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "explicit"))
    assert install.install_environment()["UV_CACHE_DIR"] == str(tmp_path / "explicit")


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_argv_owns_network_and_memory_settings(engine):
    argv = launch_arguments(engine, "/env/bin/python", "org/model", 45678, "private", 4096)
    assert argv[0:2] == ["/env/bin/python", "-I"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--api-key") + 1] == "private"
    assert "4096" in argv
    assert "--trust-remote-code" not in argv


@pytest.fixture
def peer():
    requests = []
    waiting = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
                self.wfile.flush()
                if body.get("seed") == 42:
                    waiting.wait(10)
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"length"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\ndata: [DONE]\n\n'
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    engine = ManagedEngine("vllm")
    engine.base_url = f"http://127.0.0.1:{server.server_port}"
    engine.model = "org/model"
    yield engine, requests
    waiting.set()
    server.shutdown()
    server.server_close()
    thread.join(2)


@pytest.mark.parametrize("deadline,completes", [(0.3, False), (10, True)])
def test_engine_stream_waits_for_the_first_token_deadline(monkeypatch, deadline, completes):
    import time
    import httpx
    from core.inference.engine_transport import stream_chat_events

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            # A long prefill sends nothing before the first token.
            time.sleep(1)
            try:
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    monkeypatch.setenv("UNSLOTH_OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT", str(deadline))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        events = stream_chat_events(f"http://127.0.0.1:{server.server_port}", {}, {}, lambda: False)
        if completes:
            assert [e["choices"][0]["delta"]["content"] for e in events] == ["hi"]
        else:
            with pytest.raises(httpx.ReadTimeout):
                list(events)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_real_http_stream_and_usage(peer):
    engine, requests = peer
    stats = {}
    assert list(
        engine.generate(messages = [{"role": "user", "content": "Hi"}], stats_holder = stats)
    ) == ["hello", " world"]
    assert stats["stats"]["usage"]["prompt_tokens"] == 3
    assert stats["stats"]["finish_reason"] == "length"
    path, auth, payload = requests[0]
    assert path == "/v1/chat/completions"
    assert auth == "Bearer " + engine.key
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_cancellation_does_not_wait_for_next_token(peer):
    import time

    engine, _ = peer
    cancel = threading.Event()
    stream = engine.generate(messages = [], seed = 42, cancel_event = cancel)
    assert next(stream) == "hello"
    cancel.set()
    start = time.monotonic()
    assert list(stream) == []
    assert time.monotonic() - start < 2


def test_unsupported_tools_are_not_silently_dropped(peer):
    engine, requests = peer
    with pytest.raises(ValueError, match = "Tools"):
        list(engine.generate(messages = [], tools = [{"type": "function"}]))
    assert requests == []


def test_interrupted_install_is_visible_after_restart(isolated):
    (isolated / "vllm.job.json").write_text(json.dumps({"state": "running", "phase": "installing"}))
    assert install.status("vllm")["job"]["state"] == "error"
    assert "interrupted" in install.status("vllm")["job"]["message"]


def test_install_routes_require_owner(isolated, monkeypatch):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from routes.engines import router
    from auth import policy
    from auth.authentication import get_current_subject

    app = FastAPI()
    app.include_router(router, prefix = "/api/engines")
    app.dependency_overrides[get_current_subject] = lambda: "member"

    def deny():
        raise HTTPException(status_code = 403)

    app.dependency_overrides[policy.require_owner] = deny
    called = []
    monkeypatch.setattr(install, "start_install", lambda name: called.append(name))
    with TestClient(app) as client:
        assert client.post("/api/engines/vllm/install").status_code == 403
        assert client.delete("/api/engines/vllm").status_code == 403
        assert client.post("/api/engines/vllm/cancel").status_code == 403
        assert client.post("/api/engines/vllm/rollback").status_code == 403
    assert called == []


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_model_metadata_allows_native_vision_and_quantization(tmp_path, engine):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model

    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "vision_config": {"hidden_size": 32},
                "text_config": {"num_attention_heads": 8},
                "quantization_config": {"quant_method": "awq"},
            }
        )
    )
    options = validate_model(config, gpu_ids = [0, 1], engine = engine)
    assert options["is_vision"] and options["quantization"] == "awq"
    with pytest.raises(ValueError, match = "already quantized"):
        validate_model(config, engine = engine, precision = "int4")
    path.write_text(
        json.dumps({"quantization_config": {"quant_method": "bitsandbytes", "load_in_8bit": True}})
    )
    assert validate_model(config, engine = engine)["load_format"] == "bitsandbytes"
    with pytest.raises(ValueError, match = "do not support tensor parallelism"):
        validate_model(config, gpu_ids = [0, 1], engine = engine)


def test_engine_override_only_applies_to_safetensors():
    from utils.openai_auto_switch_settings import (
        normalize_model_override,
        model_override_load_kwargs,
    )

    saved = normalize_model_override({"engine": "sglang"})
    assert model_override_load_kwargs(saved, is_gguf = False) == {
        "engine": "sglang",
        "engine_precision": "auto",
        "engine_parallelism": "tensor",
        "load_in_4bit": False,
    }
    assert "engine" not in model_override_load_kwargs(saved, is_gguf = True)
    assert normalize_model_override({"engine": "not-an-engine"}) == {}


def test_orchestrator_dispatches_managed_generation_without_worker(peer):
    from core.inference.orchestrator import InferenceOrchestrator

    engine, _ = peer
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator._managed_engine = engine
    assert list(orchestrator._generate_inner(messages = [{"role": "user", "content": "Hello"}])) == [
        "hello",
        "hello world",
    ]


def test_managed_load_launches_the_validated_local_path(monkeypatch):
    import threading as _threading
    from types import SimpleNamespace
    from core.inference import managed_engine
    from core.inference.engine_adapters import ADAPTERS
    from core.inference.orchestrator import InferenceOrchestrator

    started = {}

    class Engine:
        context = 4096

        def __init__(self, name):
            pass

        def start(
            self,
            model,
            *args,
            model_path = None,
        ):
            started.update(model = model, model_path = model_path)

        def alive(self):
            return True

    monkeypatch.setattr(managed_engine, "ManagedEngine", Engine)
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator._managed_engine = None
    orchestrator._subprocess_shutdown_lock = _threading.RLock()
    orchestrator._shutdown_subprocess = lambda *args, **kwargs: True
    orchestrator.models, orchestrator.loading_models = {}, set()
    orchestrator.active_model_name, orchestrator.load_generation = None, 0
    config = SimpleNamespace(
        identifier = "C:\\models\\foo", path = "/mnt/c/models/foo", is_local = True, is_vision = False
    )
    assert orchestrator._load_managed_engine("vllm", config, 4096, [0], None, None, None, False)
    assert started == {"model": "C:\\models\\foo", "model_path": "/mnt/c/models/foo"}
    assert orchestrator.active_model_name == "C:\\models\\foo"

    args = ADAPTERS["vllm"].command(
        "python", "/mnt/c/models/foo", 1, "key", 4096, 0.8, served_model_name = "C:\\models\\foo"
    )
    assert args[args.index("--model") + 1] == "/mnt/c/models/foo"
    assert args[args.index("--served-model-name") + 1] == "C:\\models\\foo"


def test_validate_rejects_engine_settings_before_the_picker_unloads(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from fastapi import HTTPException
    from core.inference import managed_engine
    from models.inference import ValidateModelRequest
    from routes import inference as route

    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "awq"}})
    )
    config = SimpleNamespace(
        identifier = str(tmp_path),
        path = str(tmp_path),
        display_name = "model",
        is_local = True,
        is_gguf = False,
        is_lora = False,
        is_audio = False,
        is_vision = False,
    )
    monkeypatch.setattr(managed_engine, "support_reason", lambda *args: None)
    monkeypatch.setattr(
        managed_engine,
        "installed",
        lambda engine: {"path": "/env", "profile_digest": install.profile_digest(engine)},
    )
    monkeypatch.setattr(
        route,
        "_resolve_model_identifier_for_request",
        lambda *args, **kwargs: (str(tmp_path), str(tmp_path), False),
    )
    monkeypatch.setattr(route.ModelConfig, "from_identifier", lambda **kwargs: config)
    request = ValidateModelRequest(model_path = str(tmp_path), engine = "vllm", engine_precision = "bf16")
    with pytest.raises(HTTPException) as raised:
        asyncio.run(route.validate_model(request, current_subject = "test-user"))
    assert raised.value.status_code == 400
    assert "already quantized" in raised.value.detail


def test_busy_install_does_not_overwrite_another_job(isolated):
    job = {"state": "running", "phase": "installing", "message": "Downloading"}
    (isolated / "vllm.job.json").write_text(json.dumps(job))
    with install.engine_lease("vllm", exclusive = True):
        with pytest.raises(RuntimeError, match = "another Studio"):
            install.start_install("vllm")
        assert install.status("vllm")["job"] == job


def test_another_instance_can_request_install_cancellation(isolated):
    (isolated / "vllm.job.json").write_text(json.dumps({"state": "running"}))
    with install.engine_lease("vllm", exclusive = True):
        install.cancel_install("vllm")
    assert (isolated / "vllm.cancel").exists()


def test_rollback_rejects_traversal(isolated):
    marker = active(isolated)
    info = json.loads(marker.read_text())
    info["previous"] = {"directory": "../env-other"}
    marker.write_text(json.dumps(info))
    with pytest.raises(RuntimeError, match = "No previous"):
        install.rollback("vllm")
    assert install.installed("vllm")["directory"] == "env-prior"


def test_training_always_evicts_managed_engine(monkeypatch):
    from routes import training_vram

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": True, "managed_engine": True}
    )
    monkeypatch.setattr(training_vram, "summarize_resident_stt", lambda: {"any": False})
    monkeypatch.setattr(training_vram, "free_stt_model_for_training", lambda **kw: [])
    monkeypatch.setattr(training_vram, "free_chat_models_for_training", lambda **kw: ["hf:model"])

    def would_keep():
        pytest.fail("Managed servers must be evicted even when idle VRAM appears sufficient")

    assert training_vram.coordinate_models_for_training(would_keep) == ["hf:model"]


def test_failed_managed_stop_blocks_training(monkeypatch):
    from types import SimpleNamespace
    import core.inference
    from routes.training_vram import free_chat_models_for_training

    backend = SimpleNamespace(
        active_model_name = "model",
        loading_models = set(),
        models = {"model": {}},
        _managed_engine = object(),
        _shutdown_subprocess = lambda: False,
    )
    monkeypatch.setattr(core.inference, "get_inference_backend", lambda: backend)
    with pytest.raises(RuntimeError, match = "could not be stopped"):
        free_chat_models_for_training("test")
    assert backend.active_model_name == "model"


def test_failed_managed_stop_keeps_the_chat_gpu_claim_on_unload(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from fastapi import HTTPException
    from core.inference.orchestrator import InferenceOrchestrator
    from models.inference import UnloadRequest
    import routes.inference as inference_route

    backend = InferenceOrchestrator.__new__(InferenceOrchestrator)
    backend._subprocess_shutdown_lock = threading.RLock()
    backend.active_model_name = "model"
    backend.models = {"model": {"engine": "vllm"}}
    backend.loading_models = set()
    backend._managed_engine = SimpleNamespace(stop = lambda: False)
    released = []
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_active = False, is_loaded = False, model_identifier = None),
    )
    monkeypatch.setattr(inference_route, "release_chat_gpu_claim", lambda: released.append(True))

    with pytest.raises(HTTPException) as raised:
        asyncio.run(inference_route._unload_model_impl(UnloadRequest(model_path = "model"), "tester"))
    assert raised.value.status_code == 500
    assert released == []
    assert backend.active_model_name == "model"


def test_dead_managed_server_is_reaped_without_clearing_live_server(monkeypatch):
    from types import SimpleNamespace
    from core.inference.orchestrator import InferenceOrchestrator

    backend = InferenceOrchestrator.__new__(InferenceOrchestrator)
    backend._subprocess_shutdown_lock = threading.RLock()
    backend.active_model_name = "model"
    backend._managed_engine = SimpleNamespace(phase = "ready", alive = lambda: True)
    stopped = []
    backend._shutdown_subprocess = lambda: stopped.append(True)
    backend.reap_dead_managed_engine()
    assert stopped == []
    backend._managed_engine.alive = lambda: False
    backend.reap_dead_managed_engine()
    assert stopped == [True]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_adapter_uses_measured_memory_fraction(engine):
    from core.inference.engine_adapters import ADAPTERS

    args = ADAPTERS[engine].command("/env/python", "model", 40000, "key", 2048, 0.653)
    option = "--gpu-memory-utilization" if engine == "vllm" else "--mem-fraction-static"
    assert args[args.index(option) + 1] == "0.653"


def test_memory_budget_uses_selected_gpu_and_reserves_headroom(monkeypatch):
    from types import SimpleNamespace
    from core.inference import engine_adapters
    from utils import vram_budget_settings

    commands = []

    def query(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(stdout = "24576, 16384")

    monkeypatch.setattr(engine_adapters.subprocess, "run", query)
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: 0.97)
    assert engine_adapters.gpu_memory_fraction([1]) == 0.645
    assert commands[0][commands[0].index("--id") + 1] == "1"
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: 0.5)
    assert engine_adapters.gpu_memory_fraction([1]) == 0.5
    monkeypatch.setattr(
        engine_adapters.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout = "24576, 256")
    )
    with pytest.raises(RuntimeError, match = "reserve memory"):
        engine_adapters.gpu_memory_fraction([1])


@pytest.mark.parametrize("gpu_ids", [[1], [1, 0]])
def test_server_outlives_short_lived_start_thread(isolated, monkeypatch, gpu_ids):
    import os
    import sys
    import time
    from types import SimpleNamespace
    from core.inference import managed_engine

    active(isolated)
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda _: 0.8)
    engine = ManagedEngine("vllm")

    def command(python, model, port, key, context, memory, tensor_parallel_size):
        assert tensor_parallel_size == len(gpu_ids)
        code = (
            "import os\n"
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            " def do_GET(self):\n"
            "  self.send_response(200); self.end_headers()\n"
            "  self.wfile.write((os.environ['CUDA_VISIBLE_DEVICES'] + '|' + os.environ['TRITON_CACHE_DIR']).encode())\n"
            " def log_message(self, *args): pass\n"
            f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
        )
        return [sys.executable, "-u", "-c", code]

    engine.adapter = SimpleNamespace(
        command = command, progress = lambda _: None, environment = lambda _: {}
    )
    errors = []

    def start():
        try:
            engine.start("model", 2048, gpu_ids, dict(os.environ, TRITON_CACHE_DIR = "/shared-cache"))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target = start)
    try:
        thread.start()
        thread.join(15)
        assert not thread.is_alive() and not errors
        time.sleep(0.2)
        assert engine.alive()
        import httpx

        visible, cache_path = httpx.get(engine.base_url, trust_env = False).text.split("|")
        assert visible == ",".join(map(str, gpu_ids))
        assert Path(cache_path).parent.parent == isolated / "vllm" / "cache"
        assert Path(cache_path).name == "triton"
        with pytest.raises(RuntimeError):
            install.remove("vllm")
    finally:
        assert engine.stop()
        thread.join(5)
    with install.engine_lease("vllm", exclusive = True):
        pass


def test_shared_http_reader_closes_on_early_exit():
    import asyncio
    from core.inference.http_stream import closing_response_lines

    closed = []

    class Response:
        async def aiter_lines(self):
            try:
                yield "data: first"
                yield "data: second"
            finally:
                closed.append("iterator")

        async def aclose(self):
            closed.append("response")

    async def consume():
        lines = closing_response_lines(Response())
        assert await anext(lines) == "data: first"
        await lines.aclose()

    asyncio.run(consume())
    assert closed == ["response", "iterator"]


def test_quiet_installer_can_be_cancelled(isolated):
    import sys
    import time

    cancel = threading.Event()
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match = "cancelled"):
            install._run("vllm", [sys.executable, "-c", "import time; time.sleep(30)"], cancel)
        assert time.monotonic() - started < 5
    finally:
        timer.cancel()


@pytest.mark.parametrize("kind", ["installer", "server"])
def test_shutdown_during_adoption_reaps_child(isolated, monkeypatch, kind):
    import os
    import sys
    from types import SimpleNamespace
    from core.inference import managed_engine
    from utils import process_lifetime

    shutting_down = threading.Event()
    children = []
    original_spawn = process_lifetime.spawn_on_lifetime_thread
    original_adopt = process_lifetime.adopt_pid

    def spawn(factory):
        child = original_spawn(factory)
        children.append(child)
        return child

    def adopt(pid):
        original_adopt(pid)
        # Simulate a shutdown sweep completing just before adoption.
        shutting_down.set()

    monkeypatch.setattr(process_lifetime, "spawn_on_lifetime_thread", spawn)
    monkeypatch.setattr(process_lifetime, "adopt_pid", adopt)
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", shutting_down.is_set)
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    engine = None
    try:
        with pytest.raises(RuntimeError, match = "shutting down"):
            if kind == "installer":
                install._run("vllm", command, threading.Event())
            else:
                active(isolated)
                monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda _: 0.8)
                engine = ManagedEngine("vllm")
                engine.adapter = SimpleNamespace(
                    command = lambda *args: command, environment = lambda _: {}
                )
                engine.start("model", 2048, [0], dict(os.environ))
        assert len(children) == 1
        assert children[0].poll() is not None
        assert children[0].stdout.closed
        if engine is not None:
            assert engine.process is None
            with install.engine_lease("vllm", exclusive = True):
                pass
    finally:
        for child in children:
            if child.poll() is None:
                process_lifetime.terminate_pid(child.pid, timeout = 5, owner_verified = True)
                child.wait(timeout = 5)
            process_lifetime.forget_pid(child.pid)
        if engine is not None:
            engine.stop()


@pytest.mark.parametrize("supported", [True, False])
def test_package_clone_probe_falls_back_to_independent_copies(tmp_path, monkeypatch, supported):
    import errno
    import fcntl

    cache = tmp_path / "cache"
    destination = tmp_path / "environment"
    destination.mkdir()
    monkeypatch.setattr(install, "install_environment", lambda: {"UV_CACHE_DIR": str(cache)})
    calls = []

    def clone(target, operation, source):
        calls.append(operation)
        if not supported:
            raise OSError(errno.EOPNOTSUPP, "No reflinks on this filesystem")

    monkeypatch.setattr(fcntl, "ioctl", clone)
    assert install.package_link_mode(destination) == ("clone" if supported else "copy")
    assert calls == [0x40049409]
    assert list(cache.iterdir()) == []
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_managed_capabilities_validate_in_load_and_status_responses(engine):
    from types import SimpleNamespace
    from routes.inference import _detect_safetensors_features
    from models.inference import _InferenceRuntimeFields

    backend = SimpleNamespace(active_model_name = "model", models = {"model": {"engine": engine}})
    features = _detect_safetensors_features(backend, None)
    runtime = _InferenceRuntimeFields(**features)
    assert runtime.supports_reasoning is False
    assert runtime.supports_tools is False
    assert runtime.reasoning_style == "enable_thinking"


@pytest.mark.parametrize("limit", [None, -1, 0, 64, 4096, 8192])
def test_max_tokens_sentinel_is_not_sent_as_a_negative_limit(peer, limit):
    engine, requests = peer
    engine.context = 4096
    assert list(engine.generate(messages = [{"role": "user", "content": "Hi"}], max_new_tokens = limit))
    payload = requests[0][2]
    if limit is not None and 0 < limit < engine.context:
        assert payload["max_tokens"] == limit
    else:
        assert "max_tokens" not in payload


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_managed_gpu_change_does_not_reuse_resident(engine):
    from types import SimpleNamespace
    from models.inference import LoadRequest
    from routes.inference import _non_gguf_runtime_settings_match

    backend = SimpleNamespace(
        active_model_name = "model",
        models = {
            "model": {
                "engine": engine,
                "gpu_ids_requested": [1],
                "max_seq_length_requested": 4096,
            }
        },
    )
    request = LoadRequest(model_path = "model", engine = engine, gpu_ids = [1], max_seq_length = 4096)
    assert _non_gguf_runtime_settings_match(backend, request)
    assert not _non_gguf_runtime_settings_match(
        backend, request.model_copy(update = {"gpu_ids": [0]})
    )
    assert not _non_gguf_runtime_settings_match(
        backend, request.model_copy(update = {"max_seq_length": 8192})
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_managed_token_count_route_uses_engine_capabilities(monkeypatch, engine):
    import asyncio
    from types import SimpleNamespace
    from fastapi import HTTPException
    from models.inference import ChatCountTokensRequest
    from routes import inference

    calls = []

    def count(messages, system_prompt, **kwargs):
        calls.append(messages)
        return 17, "model"

    backend = SimpleNamespace(
        active_model_name = "model",
        models = {"model": {"engine": engine}},
        load_generation = 1,
        count_chat_tokens = count,
    )
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    payload = ChatCountTokensRequest(
        model = "model", messages = [{"role": "user", "content": "Hi"}], enable_tools = False
    )
    if engine == "sglang":
        with pytest.raises(HTTPException, match = "SGLang") as error:
            asyncio.run(inference._mlx_count_chat_tokens(payload))
        assert error.value.status_code == 503
        assert not calls
    else:
        response = asyncio.run(inference._mlx_count_chat_tokens(payload))
        assert json.loads(response.body)["input_tokens"] == 17
        assert calls == [[{"role": "user", "content": "Hi"}]]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_tensor_parallel_launch_stays_on_selected_local_devices(engine, size):
    args = launch_arguments(engine, "/env/python", "model", 40000, "key", 4096, 0.8, size)
    assert args[args.index("--tensor-parallel-size") + 1] == str(size)
    if engine == "vllm" and size > 1:
        assert args[args.index("--distributed-executor-backend") + 1] == "mp"
    if engine == "sglang":
        assert ("--enable-p2p-check" in args) == (size > 1)
    from core.inference.engine_adapters import ADAPTERS

    assert ADAPTERS[engine].environment(size) == (
        {
            "vllm": {"VLLM_HOST_IP": "127.0.0.1"},
            "sglang": {
                "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
                "FLASHINFER_USE_CUDA_NORM": "1",
            },
        }[engine]
        if size > 1
        else {}
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_multi_gpu_preflight_checks_every_device(engine, monkeypatch):
    from core.inference import managed_engine
    from models.inference import LoadRequest

    checked = []
    info = {"path": "/env", "profile_digest": install.profile_digest(engine)}
    monkeypatch.setattr(managed_engine, "installed", lambda _: info)
    monkeypatch.setattr(managed_engine, "support_reason", lambda _, gpu: checked.append(gpu))
    request = LoadRequest(model_path = "model", gpu_ids = [1, 0], engine = engine)
    managed_engine.validate_load(engine, request)
    assert checked == [1, 0]
    # An outdated profile needs an update, unless the user explicitly restored it.
    info["profile_digest"] = "outdated"
    with pytest.raises(ValueError, match = "Update"):
        managed_engine.validate_load(engine, request)
    info["restored"] = True
    managed_engine.validate_load(engine, request)
    managed_engine.validate_load(engine, request.model_copy(update = {"tensor_parallel": True}))
    for ids in ([0, 0], [-1, 0]):
        with pytest.raises(ValueError, match = "distinct"):
            managed_engine.validate_load(engine, request.model_copy(update = {"gpu_ids": ids}))
    monkeypatch.setattr(
        managed_engine, "support_reason", lambda _, gpu: "Unsupported device" if gpu == 0 else None
    )
    with pytest.raises(ValueError, match = "GPU 0: Unsupported"):
        managed_engine.validate_load(engine, request)


def test_multi_gpu_memory_budget_uses_most_constrained_device(monkeypatch):
    from types import SimpleNamespace
    from core.inference import engine_adapters
    from utils import vram_budget_settings

    commands = []

    def query(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout = "49152, 40000\n24576, 8192\n")

    monkeypatch.setattr(engine_adapters.subprocess, "run", query)
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: 0.97)
    assert engine_adapters.gpu_memory_fraction([0, 1]) == 0.312
    assert commands[0][commands[0].index("--id") + 1] == "0,1"
    for output in ("49152, 40000", "49152, 40000\n24576, 256"):
        monkeypatch.setattr(
            engine_adapters.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout = output)
        )
        with pytest.raises(RuntimeError, match = "every selected GPU"):
            engine_adapters.gpu_memory_fraction([0, 1])


@pytest.mark.parametrize(
    "overrides,size,valid",
    [
        ({}, 2, True),
        ({}, 4, True),  # Two KV heads replicated over four ranks.
        ({}, 3, False),  # Three ranks cannot shard or replicate two KV heads.
        ({"num_attention_heads": 7}, 2, False),
        ({"intermediate_size": 1537}, 2, False),
        ({"hidden_size": 769}, 2, False),
        ({"num_key_value_heads": 6}, 4, False),
    ],
)
def test_tensor_parallel_model_validation_before_handoff(tmp_path, overrides, size, valid):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model

    metadata = {
        "model_type": "qwen2",
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "hidden_size": 768,
        "intermediate_size": 1536,
        **overrides,
    }
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    if valid:
        validate_model(config, gpu_ids = list(range(size)))
    else:
        with pytest.raises(ValueError, match = "GPUs"):
            validate_model(config, gpu_ids = list(range(size)))


def test_installer_publishes_bounded_output_before_exit(isolated):
    import sys
    import time

    errors = []
    cancel = threading.Event()
    code = "import time; print('Downloading torch (900 MiB)', flush=True); time.sleep(2); [print('Downloaded package-' + str(i), flush=True) for i in range(30)]"

    def run():
        try:
            install._run("vllm", [sys.executable, "-u", "-c", code], cancel)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target = run)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            path = isolated / "vllm.job.json"
            if path.exists():
                job = json.loads(path.read_text())
                if job.get("activity") == "Downloading torch (900 MiB)":
                    break
            time.sleep(0.02)
        else:
            pytest.fail("Installer output was not published while the subprocess was running")
        assert thread.is_alive()
    finally:
        thread.join(10)
    assert not thread.is_alive() and not errors
    job = json.loads((isolated / "vllm.job.json").read_text())
    assert len(job["log"]) == 20
    assert job["activity"] == "Downloaded package-29"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("precision", ["auto", "bf16", "fp16", "int4", "int8", "fp8"])
def test_native_precision_arguments(engine, precision):
    from core.inference.engine_adapters import ADAPTERS

    args = ADAPTERS[engine].command(
        "python",
        "model",
        12345,
        "key",
        4096,
        0.8,
        options = {
            "precision": precision,
            "load_format": "bitsandbytes" if engine == "vllm" and precision == "int4" else "auto",
        },
    )
    expected = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "int4": "bitsandbytes" if engine == "vllm" else "int4wo-32",
        "fp8": "torchao" if engine == "vllm" else "fp8",
        "int8": "torchao" if engine == "vllm" else "int8wo",
    }
    if precision == "auto":
        assert "--quantization" not in args and "--dtype" not in args
    else:
        assert expected[precision] in args
    assert args[args.index("--load-format") + 1] == (
        "bitsandbytes" if engine == "vllm" and precision == "int4" else "auto"
    )


@pytest.mark.parametrize("encoded", [False, True])
def test_managed_images_keep_their_conversation_positions(peer, encoded):
    from PIL import Image

    engine, requests = peer
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "First?"}]},
        {"role": "assistant", "content": "Red"},
        {"role": "user", "content": [{"type": "text", "text": "Second?"}, {"type": "image"}]},
    ]
    images = [Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")]
    if encoded:
        from core.inference.orchestrator import InferenceOrchestrator
        images = [InferenceOrchestrator._pil_to_base64(image) for image in images]
    list(engine.generate(messages = messages, images = images))
    sent = requests[0][2]["messages"]
    assert sent[0]["content"][0]["type"] == "image_url"
    assert sent[2]["content"][1]["type"] == "image_url"
    assert sent[0]["content"][0]["image_url"] != sent[2]["content"][1]["image_url"]
    assert messages[0]["content"][0]["type"] == "image"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_prequantized_int8_uses_uncaptured_sglang_execution(tmp_path, engine):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model
    from core.inference.engine_adapters import ADAPTERS

    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "bitsandbytes", "load_in_8bit": True}})
    )
    options = validate_model(SimpleNamespace(is_local = True, path = str(tmp_path)), engine = engine)
    args = ADAPTERS[engine].command("python", "model", 12345, "key", 4096, 0.8, options = options)
    assert ("--disable-cuda-graph" in args) == (engine == "sglang")
    assert ("--disable-piecewise-cuda-graph" in args) == (engine == "sglang")


@pytest.mark.parametrize(
    "capabilities, eager", [("8.9\n", False), ("8.6\n", True), ("8.9\n8.6\n", True)]
)
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_fp8_conversion_uses_eager_on_ampere(tmp_path, monkeypatch, capabilities, eager, engine):
    from types import SimpleNamespace
    from core.inference import managed_engine

    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    monkeypatch.setattr(
        managed_engine.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout = capabilities),
    )
    options = managed_engine.validate_model(
        SimpleNamespace(is_local = True, path = str(tmp_path)), engine = engine, precision = "fp8"
    )
    assert options["disable_cuda_graph"] is eager


@pytest.mark.parametrize("engine, load_format", [("vllm", "bitsandbytes"), ("sglang", "auto")])
def test_online_int4_uses_the_native_loader(tmp_path, engine, load_format):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model

    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    options = validate_model(
        SimpleNamespace(is_local = True, path = str(tmp_path)), engine = engine, precision = "int4"
    )
    assert options["load_format"] == load_format


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["tensor", "pipeline", "data"])
@pytest.mark.parametrize("devices", [1, 2, 4])
def test_parallel_mode_uses_exactly_the_selected_devices(engine, mode, devices):
    from core.inference.engine_adapters import ADAPTERS

    args = ADAPTERS[engine].command(
        "python",
        "model",
        12345,
        "key",
        2048,
        0.8,
        devices,
        options = {"parallelism": mode},
    )
    sizes = {}
    for name in ("tensor", "pipeline", "data"):
        flag = f"--{name}-parallel-size"
        sizes[name] = int(args[args.index(flag) + 1]) if flag in args else 1
    assert sizes[mode] == devices
    assert all(size == 1 for name, size in sizes.items() if name != mode)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    if engine == "vllm" and mode == "data":
        assert "vllm.entrypoints.cli.main" in args and "serve" in args
        assert "--distributed-executor-backend" not in args
        if devices > 1:
            assert args[args.index("--data-parallel-backend") + 1] == "mp"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["pipeline", "data"])
def test_non_tensor_modes_do_not_require_divisible_heads(tmp_path, engine, mode):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_attention_heads": 3,
                    "hidden_size": 15,
                    "num_hidden_layers": 12,
                },
            }
        )
    )
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    with pytest.raises(ValueError, match = "cannot be split"):
        validate_model(config, gpu_ids = [0, 1], engine = engine)
    options = validate_model(config, gpu_ids = [0, 1], engine = engine, parallelism = mode)
    assert options["parallelism"] == mode


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["pipeline", "data"])
def test_non_tensor_modes_validate_prequantized_bitsandbytes(tmp_path, engine, mode):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model

    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "bitsandbytes"}})
    )
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    if engine == "sglang" and mode == "pipeline":
        with pytest.raises(ValueError, match = "SGLang cannot load prequantized BitsAndBytes"):
            validate_model(config, gpu_ids = [0, 1], engine = engine, parallelism = mode)
        # A retained pipeline selection still permits a single-device load.
        assert (
            validate_model(config, gpu_ids = [0], engine = engine, parallelism = mode)["load_format"]
            == "bitsandbytes"
        )
        return
    options = validate_model(
        config,
        gpu_ids = [0, 1],
        engine = engine,
        parallelism = mode,
    )
    assert options["parallelism"] == mode and options["load_format"] == "bitsandbytes"


@pytest.mark.parametrize("mode", ["pipeline", "data"])
def test_resident_engine_parallel_mode_requires_reload(mode):
    from types import SimpleNamespace
    from routes.inference import _non_gguf_runtime_settings_match
    from models.inference import LoadRequest

    backend = SimpleNamespace(
        active_model_name = "model",
        models = {
            "model": {
                "engine": "vllm",
                "engine_parallelism": mode,
                "gpu_ids_requested": [0, 1],
            }
        },
    )
    request = LoadRequest(
        model_path = "model", engine = "vllm", engine_parallelism = mode, gpu_ids = [0, 1]
    )
    assert _non_gguf_runtime_settings_match(backend, request)
    assert not _non_gguf_runtime_settings_match(
        backend, request.model_copy(update = {"engine_parallelism": "tensor"})
    )


def test_native_api_key_cannot_be_parsed_as_a_cli_option(monkeypatch):
    import core.inference.managed_engine as managed
    monkeypatch.setattr(managed.secrets, "token_urlsafe", lambda _: "--random-token")
    assert managed.ManagedEngine("vllm").key == "studio---random-token"


def test_vllm_int4_pipeline_uses_native_torchao_loader(tmp_path):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model
    from core.inference.engine_adapters import ADAPTERS

    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    options = validate_model(
        SimpleNamespace(is_local = True, path = str(tmp_path)),
        engine = "vllm",
        gpu_ids = [0, 1],
        precision = "int4",
        parallelism = "pipeline",
    )
    args = ADAPTERS["vllm"].command("python", "model", 12345, "key", 2048, 0.8, 2, options = options)
    assert args[args.index("--load-format") + 1] == "auto"
    assert args[args.index("--quantization") + 1] == "torchao"
    config = json.loads(
        json.loads(args[args.index("--hf-overrides") + 1])["quantization_config_dict_json"]
    )
    assert config["_type"] == "Int4WeightOnlyConfig"
    assert config["_data"]["int4_choose_qparams_algorithm"]["_data"] == "HQQ"


def test_switching_from_managed_engine_keeps_default_precision():
    from types import SimpleNamespace
    from models.inference import LoadRequest
    from routes.inference import _inherit_resident_load_in_4bit

    backend = SimpleNamespace(
        active_model_name = "model",
        models = {"model": {"engine": "vllm", "load_in_4bit_requested": False}},
    )
    request = LoadRequest(model_path = "model")
    _inherit_resident_load_in_4bit(backend, request, "model")
    assert request.load_in_4bit is True
