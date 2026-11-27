# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import subprocess
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import engine_install, managed_engine, managed_engine_log, wsl_host
from utils import debug_log_sources


_TOKEN = "fixture-opaque-hf-credential"
_FILE_TOKEN = "fixture-opaque-hf-file-credential"


@pytest.fixture(params = ["sglang", "vllm"])
def engine_name(request):
    return request.param


@pytest.fixture
def engine(tmp_path, monkeypatch, engine_name):
    info = {
        "path": str(tmp_path / "env"),
        "host": "native",
    }
    monkeypatch.setattr(managed_engine, "installed", lambda _: info)
    monkeypatch.setattr(managed_engine, "engine_lease", lambda _: nullcontext())
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda *_: 0.8)
    monkeypatch.setattr(managed_engine, "_child_cuda_mask", lambda _: "0")
    monkeypatch.setattr(engine_install, "cuda_environment", lambda _: {})
    monkeypatch.setattr(engine_install, "engine_root", lambda: tmp_path)
    runner = managed_engine.ManagedEngine(engine_name)
    runner.adapter = SimpleNamespace(
        command = lambda *args, **kwargs: pytest.fail("No command was arranged for this test"),
        progress = lambda _: None,
        environment = lambda _: {},
        key_environment = lambda _: {},
    )
    try:
        yield runner
    finally:
        runner.stop()


def _arrange_process(engine, monkeypatch, host, *, attempt, fail):
    def command(python, model, port, key, *_):
        header = (
            "import sys\n"
            f"print({('stdout ' + attempt + ' ' + key)!r}, flush=True)\n"
            f"print({('stderr ' + attempt + ' ' + _TOKEN + ' ' + _FILE_TOKEN)!r}, "
            "file=sys.stderr, flush=True)\n"
            "print('Authorization: Bearer auth-test-112233', file=sys.stderr, flush=True)\n"
        )
        if fail:
            body = "sys.exit(1)\n"
        elif host == "wsl":
            body = (
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                " def do_GET(self): self.send_response(200); self.end_headers()\n"
                " def log_message(self, *args): pass\n"
                f"server = HTTPServer(('127.0.0.1', {port}), Handler)\n"
                "import threading\n"
                "threading.Thread(target=server.serve_forever, daemon=True).start()\n"
                "sys.stdin.read()\n"
                "print('guest received EOF', flush=True)\n"
            )
        else:
            body = (
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                " def do_GET(self): self.send_response(200); self.end_headers()\n"
                " def log_message(self, *args): pass\n"
                f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
            )
        return [sys.executable, "-u", "-c", header + body]

    engine.adapter.command = command
    if host == "wsl":
        # Keep the same host-side Popen, stdin shutdown and stdout drain as wsl.exe.
        monkeypatch.setattr(
            engine,
            "_wsl_command",
            lambda info, env, ids, options, trust, model, path, port: (
                command(None, model, port, engine.key),
                dict(os.environ),
            ),
        )


@pytest.mark.parametrize("host", ["native", "wsl"])
def test_failed_launch_stays_selectable_after_retry(tmp_path, monkeypatch, engine, host):
    info = managed_engine.installed(engine.engine)
    info["host"] = host
    token_path = tmp_path / "token"
    token_path.write_text(_FILE_TOKEN, encoding = "utf-8")
    env = dict(os.environ, HF_TOKEN = _TOKEN, HF_TOKEN_PATH = str(token_path))

    _arrange_process(engine, monkeypatch, host, attempt = "first", fail = True)
    with pytest.raises(RuntimeError, match = "Engine failed to start") as failure:
        engine.start("model", 1024, [0], env)
    failed_path = Path(failure.value.managed_engine_log_path)
    assert engine.process is None and engine._attempt_log is None

    retry = managed_engine.ManagedEngine(engine.engine)
    retry.adapter = engine.adapter
    _arrange_process(retry, monkeypatch, host, attempt = "second", fail = False)
    try:
        retry.start("model", 1024, [0], env)
        assert retry.phase == "ready"
        live_log = retry._attempt_log
        assert live_log is not None
        ready_path = Path(live_log.path)
        assert ready_path != failed_path
        assert "Managed engine is ready." in ready_path.read_text(encoding = "utf-8")
    finally:
        assert retry.stop()

    assert live_log._stream.closed
    for path, attempt in ((failed_path, "first"), (ready_path, "second")):
        raw = path.read_text(encoding = "utf-8")
        assert f"stdout {attempt}" in raw and f"stderr {attempt}" in raw
        assert "Managed engine stopped." in raw
        for secret in (engine.key, retry.key, _TOKEN, _FILE_TOKEN, "auth-test-112233"):
            assert secret not in raw
    assert "Engine exited during startup." in failed_path.read_text(encoding = "utf-8")
    if host == "wsl":
        assert "guest received EOF" in ready_path.read_text(encoding = "utf-8")

    sources = debug_log_sources.list_sources()
    assert next(source for source in sources if source.family == engine.engine).realpath == str(
        ready_path
    )
    old_id = debug_log_sources.source_id_for_path(str(failed_path), sources)
    assert debug_log_sources.resolve_source_id(old_id) == failed_path


def test_guest_output_masks_a_registered_project_folder(engine, monkeypatch):
    from io import StringIO
    from utils import native_path_leases

    folder = r"C:\Users\Fixture\PrivateProject"
    monkeypatch.setattr(native_path_leases, "_NATIVE_PATH_REDACTIONS", [folder])
    log = managed_engine_log.ManagedEngineLog.open(engine.engine)
    assert log is not None
    guest_path = wsl_host.to_guest_path(folder) + "/checkpoints/model"
    try:
        engine._drain(SimpleNamespace(stdout = StringIO(guest_path)), log, (engine.key,))
    finally:
        log.close()
    raw = Path(log.path).read_text(encoding = "utf-8")
    assert "PrivateProject" not in raw
    assert "<native_path>/checkpoints/model" in raw


def test_diagnostic_token_lookup_skips_devices(engine):
    # The null device returns EOF, so this also terminates on a regressed implementation.
    from unittest.mock import patch
    with patch.object(Path, "open", side_effect = AssertionError("opened credential device")):
        masked = engine._log_secrets({"HF_TOKEN": _TOKEN, "HF_TOKEN_PATH": os.devnull})
    assert _TOKEN in masked


def test_spawn_failure_retains_a_closed_redacted_log(engine, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError(f"spawn failed with {_TOKEN}")

    monkeypatch.setattr(subprocess, "Popen", fail)
    _arrange_process(engine, monkeypatch, "native", attempt = "first", fail = True)
    with pytest.raises(OSError, match = "spawn failed") as failure:
        engine.start("model", 1024, [0], dict(os.environ, HF_TOKEN = _TOKEN))
    assert engine._attempt_log is None
    raw = Path(failure.value.managed_engine_log_path).read_text(encoding = "utf-8")
    assert "Engine startup failed: spawn failed" in raw
    assert _TOKEN not in raw


@pytest.mark.parametrize("fail_at", ["open", "write", "redact"])
def test_logging_failure_does_not_prevent_load_or_unload(engine, monkeypatch, fail_at):
    if fail_at == "open":

        def open_failed(*args, **kwargs):
            raise PermissionError("unwritable logs")

        monkeypatch.setattr(managed_engine_log.tempfile, "NamedTemporaryFile", open_failed)
    elif fail_at == "write":
        original = managed_engine_log.ManagedEngineLog.open

        class BrokenStream:
            def __init__(self, stream):
                self.stream = stream

            def write(self, text):
                raise OSError("disk full")

            def close(self):
                self.stream.close()

            @property
            def closed(self):
                return self.stream.closed

        def open_broken(name):
            log = original(name)
            log._stream = BrokenStream(log._stream)
            return log

        monkeypatch.setattr(managed_engine_log.ManagedEngineLog, "open", open_broken)
    else:

        def broken_redaction(*args):
            raise ValueError("scrubber failed")

        monkeypatch.setattr(engine, "_redact_output", broken_redaction)
    _arrange_process(engine, monkeypatch, "native", attempt = "ready", fail = False)
    engine.start("model", 1024, [0], dict(os.environ))
    assert engine.phase == "ready"
    log = engine._attempt_log
    assert engine.stop()
    if fail_at == "open":
        assert log is None
        assert not any(
            source.family == engine.engine for source in debug_log_sources.list_sources()
        )
    else:
        assert log._stream.closed
        assert engine.key not in Path(log.path).read_text(encoding = "utf-8")
        assert "stdout ready" not in Path(log.path).read_text(encoding = "utf-8")


def test_cancelled_before_spawn_opens_no_file(engine):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match = "cancelled") as failure:
        engine.start("model", 1024, [0], dict(os.environ), cancel)
    assert not hasattr(failure.value, "managed_engine_log_path")


def test_darwin_has_no_managed_engine_logs(monkeypatch, engine_name):
    monkeypatch.setattr(wsl_host, "active", lambda: False)
    monkeypatch.setattr(managed_engine, "resolve_requested_gpu_ids", lambda _: [0])
    monkeypatch.setattr(managed_engine, "_mig_rows", lambda _: [])
    monkeypatch.setattr(engine_install.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match = "require Linux"):
        managed_engine.validate_load(engine_name, SimpleNamespace(gpu_ids = None))
    assert not any(source.family == engine_name for source in debug_log_sources.list_sources())


def test_all_retained_attempts_are_resolvable_alongside_existing_sources(engine_name):
    from utils.log_retention import DEFAULT_KEEP
    from utils.paths import studio_root

    server = studio_root() / "logs" / "server"
    server.mkdir(parents = True)
    (server / "server-fixture.log").touch()
    first = None
    for index in range(DEFAULT_KEEP + 1):
        log = managed_engine_log.ManagedEngineLog.open(engine_name)
        assert log is not None
        log.close()
        if index == 0:
            first = log.path
            os.utime(first, (1, 1))

    sources = debug_log_sources.list_sources()
    assert len([source for source in sources if source.family == engine_name]) == DEFAULT_KEEP
    assert any(source.family == "server" for source in sources)
    assert not Path(first).exists()
    for source in sources:
        assert debug_log_sources.resolve_source_id(source.id) == Path(source.realpath)
