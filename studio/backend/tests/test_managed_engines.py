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
    assert any("check" in argv for argv in calls)
    assert install.status("vllm")["job"]["state"] == "success"
    install.rollback("vllm")
    assert install.installed("vllm")["directory"] == "env-prior"
    install.rollback("vllm")
    assert install.installed("vllm")["directory"] == result["directory"]


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


def test_model_metadata_validation_rejects_quantized_and_multimodal(tmp_path):
    from types import SimpleNamespace
    from core.inference.managed_engine import validate_model

    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "qwen2"}))
    validate_model(config)
    for extra in (
        {"quantization_config": {"quant_method": "awq"}},
        {"vision_config": {"hidden_size": 32}},
    ):
        path.write_text(json.dumps({"model_type": "qwen2", **extra}))
        with pytest.raises(ValueError, match = "full precision"):
            validate_model(config)


def test_engine_override_only_applies_to_safetensors():
    from utils.openai_auto_switch_settings import (
        normalize_model_override,
        model_override_load_kwargs,
    )

    saved = normalize_model_override({"engine": "sglang"})
    assert model_override_load_kwargs(saved, is_gguf = False) == {
        "engine": "sglang",
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

        assert httpx.get(engine.base_url, trust_env = False).text == (
            ",".join(map(str, gpu_ids)) + "|" + str(isolated / "vllm" / "cache" / "triton")
        )
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
            "sglang": {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0"},
        }[engine]
        if size > 1
        else {}
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_multi_gpu_preflight_checks_every_device(engine, monkeypatch):
    from core.inference import managed_engine
    from models.inference import LoadRequest

    checked = []
    monkeypatch.setattr(managed_engine, "installed", lambda _: {"path": "/env"})
    monkeypatch.setattr(managed_engine, "support_reason", lambda _, gpu: checked.append(gpu))
    request = LoadRequest(model_path = "model", gpu_ids = [1, 0], engine = engine)
    managed_engine.validate_load(engine, request)
    assert checked == [1, 0]
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
