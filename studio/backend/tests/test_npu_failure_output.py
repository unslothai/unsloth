# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What Lemonade printed around a failed NPU load reaches the log and the error, fed through _drain."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from core.inference import npu_backend as nb
from core.inference.lemonade_server import LemonadeServer

_MODEL = "qwen3.6-moe-35b-a3b-FLM"


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message, *args, **fields) -> None:
        text = message % args if args else message
        self.warnings.append("\n".join([text, *map(str, fields.get("lemonade_output") or [])]))
        self.fields = fields

    def info(self, *args) -> None:
        pass

    debug = info


def _server(tmp_path) -> LemonadeServer:
    return LemonadeServer(
        tmp_path / "lemond",
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
    )


def _prints(server: LemonadeServer, *lines: str) -> None:
    server._drain(SimpleNamespace(stdout = [line + "\n" for line in lines]))


def _line(severity: str, tag: str, message: str) -> str:
    return f"2026-11-11 04:20:01.123 [{severity}] ({tag}) {message}"


@pytest.fixture
def log(monkeypatch):
    recorder = _Log()
    monkeypatch.setattr(nb, "_events", recorder)
    return recorder


def _backend(tmp_path, monkeypatch, server, answer):
    def request(method, path, **kwargs):
        lines, response = answer(path)
        _prints(server, *lines)
        return response

    monkeypatch.setattr(server, "request", request)
    backend = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    monkeypatch.setattr(backend, "_ensure_running", lambda: server)
    monkeypatch.setattr(
        backend,
        "_model",
        lambda model_id: nb.NpuModel(
            id = model_id,
            checkpoint = "qwen3.6-moe:35b-a3b",
            size_gb = 20.0,
            downloaded = True,
            labels = ("chat",),
            max_context_length = 32768,
        ),
    )
    return backend


def test_a_failed_load_logs_lemonades_output_and_names_its_reason(tmp_path, monkeypatch, log):
    server = _server(tmp_path)
    _prints(server, _line("Info", "Server", "Lemonade Server started on port 1234"))

    def answer(path):
        return (
            [
                _line("Info", "FastFlowLM", f"Loading model: {_MODEL}"),
                _line("Error", "WrappedServer", "FastFlowLM process has terminated"),
                _line("Error", "WrappedServer", "This usually means:"),
                _line("Error", "Router", "Backend load failed: FastFlowLM failed to start"),
            ],
            httpx.Response(
                500,
                json = {
                    "error": {
                        "message": f"Failed to load model '{_MODEL}': FastFlowLM failed to start"
                    }
                },
            ),
        )

    backend = _backend(tmp_path, monkeypatch, server, answer)
    with pytest.raises(nb.NpuError) as raised:
        backend.load(_MODEL, 32768)

    message = str(raised.value)
    assert message.startswith(f"Loading {_MODEL} failed: Failed to load model")
    assert "FastFlowLM process has terminated" in message
    logged = "\n".join(log.warnings)
    assert "FastFlowLM process has terminated" in logged
    assert "Backend load failed: FastFlowLM failed to start" in logged
    # Only what lemond printed for this load, not everything since it started.
    assert "started on port" not in logged


@pytest.mark.parametrize("has_log", [True, False])
def test_a_failure_lemonade_only_summarizes_points_at_the_log(tmp_path, monkeypatch, log, has_log):
    server = _server(tmp_path)
    if has_log:
        server.log_dir = tmp_path / "logs" / "lemonade"
        server._open_log_file(4242)
    summary = f"Failed to load {_MODEL} model!"

    def answer(path):
        return (
            [_line("Error", "Router", f"Failed to load model: {summary}")],
            httpx.Response(500, json = {"error": {"message": summary}}),
        )

    backend = _backend(tmp_path, monkeypatch, server, answer)
    with pytest.raises(nb.NpuError) as raised:
        backend.load(_MODEL, 32768)

    assert summary in str(raised.value)
    # No file means nothing on the Logs page to point at.
    assert ("Logs" in str(raised.value)) == has_log
    assert any(f"Failed to load model: {summary}" in warning for warning in log.warnings)


def test_a_failure_with_nothing_printed_keeps_the_summary_and_logs_nothing(
    tmp_path, monkeypatch, log
):
    server = _server(tmp_path)
    backend = _backend(
        tmp_path,
        monkeypatch,
        server,
        lambda path: ([], httpx.Response(500, json = {"error": {"message": "boom"}})),
    )
    with pytest.raises(nb.NpuError) as raised:
        backend.load(_MODEL, 32768)

    assert str(raised.value) == f"Loading {_MODEL} failed: boom"
    assert log.warnings == []


def test_output_since_a_mark_is_only_what_came_after(tmp_path):
    server = _server(tmp_path)
    _prints(server, "before")
    mark = server.output_mark()
    assert server.output_since(mark, settle = 0) == []
    _prints(server, "one", "two", "three")
    assert server.output_since(mark, settle = 0) == ["one", "two", "three"]
    assert server.output_since(mark, lines = 2, settle = 0) == ["two", "three"]


def test_output_since_waits_for_a_drain_that_runs_late(tmp_path):
    import threading

    server = _server(tmp_path)
    mark = server.output_mark()
    threading.Timer(0.15, _prints, (server, "late error")).start()
    threading.Timer(0.25, _prints, (server, "later detail")).start()
    assert server.output_since(mark, settle = 1.0) == ["late error", "later detail"]


def test_lemond_output_goes_to_a_log_file_settings_lists(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    server = LemonadeServer(
        tmp_path / "lemond",
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
        log_dir = home / "logs" / "lemonade",
    )
    handle = server._open_log_file(4242)
    failure = _line("Error", "Router", "Failed to load model qwen3.6-moe:35b-a3b")
    server._drain(SimpleNamespace(stdout = [failure + "\n"]), handle)

    from utils import debug_log_sources

    sources = [s for s in debug_log_sources.list_sources() if s.family == "lemonade"]
    assert [s.label for s in sources] == [
        f"lemonade-{sources[0].label.split('-')[1]}-port-4242.log"
    ]
    assert failure in (home / "logs" / "lemonade" / sources[0].label).read_text(encoding = "utf-8")


def test_no_log_dir_writes_no_file(tmp_path):
    server = _server(tmp_path)
    assert server._open_log_file(4242) is None
    _prints(server, _line("Info", "Server", "ready"))
    assert not any(tmp_path.rglob("lemonade-*.log"))


def test_the_lemonade_log_says_what_was_asked_and_how_it_ended(tmp_path, monkeypatch, log):
    server = LemonadeServer(
        tmp_path / "lemond",
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
        log_dir = tmp_path / "logs" / "lemonade",
    )
    handle = server._open_log_file(4242, ["lemond", "--port", "4242"])
    server._drain(SimpleNamespace(stdout = []), handle)
    server._open_log_file(4243)
    summary = f"Failed to load {_MODEL} model!"

    def answer(path):
        return (
            [_line("Error", "WrappedServer", "FastFlowLM process has terminated with exit code 1")],
            httpx.Response(500, json = {"error": {"message": summary}}),
        )

    backend = _backend(tmp_path, monkeypatch, server, answer)
    with pytest.raises(nb.NpuError):
        backend.load(_MODEL, 32768)

    first = tmp_path / "logs" / "lemonade"
    texts = {p.name: p.read_text(encoding = "utf-8") for p in first.glob("lemonade-*.log")}
    assert any(
        "(Unsloth Studio) Starting Lemonade: lemond --port 4242" in t for t in texts.values()
    )
    current = texts[server.log_path.name]
    assert f"(Unsloth Studio) Loading {_MODEL} with a" in current
    assert "(Unsloth Studio) Request failed: " + summary in current
    assert "exit code 1" in current
    assert log.fields["lemonade_log"] == server.log_path.name


def test_debug_configuration_cannot_persist_chat_previews(tmp_path):
    server = _server(tmp_path)
    server.config_dir.mkdir()
    config = server.config_dir / "config.json"
    config.write_text('{"log_level":"debug"}', encoding = "utf-8")
    server._write_config()
    assert json.loads(config.read_text(encoding = "utf-8"))["log_level"] == "info"

    server.log_dir = tmp_path / "logs"
    lines = [
        _line("Debug", "Server", "chat preview"),
        "continuation of chat preview",
        _line("Info", "Process", "backend request body"),
        _line("Error", "Router", "model allocation failed"),
    ]
    server._drain(SimpleNamespace(stdout = lines), server._open_log_file(4242))
    for output in (server.log_tail(), server.log_path.read_text(encoding = "utf-8")):
        assert "chat preview" not in output
        assert "backend request body" not in output
        assert "model allocation failed" in output


def test_daemon_spawn_does_not_inherit_forced_debug(tmp_path, monkeypatch):
    from core.inference import lemonade_server as ls

    server = _server(tmp_path)
    monkeypatch.setenv("LEMONADE_CI_MODE", "1")
    monkeypatch.setattr(ls, "spawn_on_lifetime_thread", lambda spawn: spawn())

    def capture(command, **kwargs):
        assert "LEMONADE_CI_MODE" not in kwargs["env"]
        raise OSError("test stopped before spawn")

    monkeypatch.setattr(ls.subprocess, "Popen", capture)
    with pytest.raises(OSError, match = "test stopped before spawn"):
        server._start_once(5)


def test_silent_load_failure_points_to_the_existing_log(tmp_path, monkeypatch, log):
    server = _server(tmp_path)
    server.log_dir = tmp_path / "logs"
    handle = server._open_log_file(4242)
    try:
        backend = _backend(
            tmp_path,
            monkeypatch,
            server,
            lambda path: ([], httpx.Response(500, json = {"error": "load refused"})),
        )
        with pytest.raises(nb.NpuError, match = "Logs page"):
            backend.load(_MODEL)
        assert "Request failed: load refused" in server.log_path.read_text(encoding = "utf-8")
    finally:
        server._close_log_file(handle)


def test_pending_chat_keeps_its_original_daemon(tmp_path, log):
    original = _server(tmp_path)
    original.api_key = "first-key"
    upstream = SimpleNamespace(api_key = original.api_key)
    backend = nb.LemonadeNpuBackend(root = tmp_path)
    backend._server = original
    window = backend.output_mark(upstream)
    _prints(original, _line("Error", "Router", "original failure"))

    replacement = _server(tmp_path)
    replacement.api_key = "second-key"
    backend._server = replacement
    _prints(replacement, _line("Error", "Router", "unrelated failure"))
    assert backend.output_mark(upstream) is None
    assert backend.explain_failure(window, "chat failed") == "chat failed (original failure)"


@pytest.mark.parametrize("path", ["/v1/pull", "/v1/delete"])
def test_chat_cannot_claim_a_background_model_operation(tmp_path, monkeypatch, log, path):
    server = _server(tmp_path)
    server.port = 4242
    monkeypatch.setattr(server, "is_alive", lambda: True)
    chat = server.open_window()

    def handler(request):
        _prints(server, _line("Error", "DownloadManager", "unrelated model failed"))
        assert chat.shared
        return httpx.Response(500)

    with httpx.Client(transport = httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(server, "_client", client)
        if path == "/v1/pull":
            with server.stream("POST", path, json_body = {}) as response:
                assert response.status_code == 500
        else:
            assert server.request("POST", path).status_code == 500

    assert nb._explain(server, chat, "chat failed") == "chat failed"
    del chat
    assert not server.open_window().shared


def test_load_disconnect_uses_the_request_output(tmp_path, monkeypatch, log):
    server = _server(tmp_path)

    def disconnect(path):
        assert path == "/v1/load"
        _prints(server, _line("Error", "Router", "model allocation failed"))
        raise nb.LemonadeUnavailable("connection closed")

    backend = _backend(tmp_path, monkeypatch, server, disconnect)
    with pytest.raises(nb.NpuError, match = r"connection closed \(model allocation failed\)"):
        backend.load(_MODEL, 8192)
    assert log.fields["lemonade_output"] == [_line("Error", "Router", "model allocation failed")]


def test_a_failed_spawn_closes_the_log_file(tmp_path, monkeypatch):
    from core.inference import lemonade_server as ls

    server = LemonadeServer(
        tmp_path / "lemond",
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
        log_dir = tmp_path / "logs" / "lemonade",
    )
    opened = []
    real_open = server._open_log_file
    monkeypatch.setattr(
        server, "_open_log_file", lambda *a: opened.append(real_open(*a)) or opened[-1]
    )

    def boom(spawn):
        raise OSError("no lemond")

    monkeypatch.setattr(ls, "spawn_on_lifetime_thread", boom)
    with pytest.raises(OSError):
        server._start_once(5)
    assert opened and opened[0].closed
    assert server._log_file is None


def test_overlapping_requests_get_no_reason_pinned_on_them(tmp_path, log):
    server = _server(tmp_path)
    first = server.open_window()
    second = server.open_window()
    _prints(server, _line("Error", "WrappedServer", "FastFlowLM process has terminated"))
    for window in (first, second):
        message = nb._explain(server, window, "boom")
        assert "terminated" not in message
    assert log.fields["overlapped_another_request"] is True


def test_a_window_alone_still_gets_its_reason(tmp_path, log):
    server = _server(tmp_path)
    server.open_window()  # dropped at once, like a finished request
    window = server.open_window()
    _prints(server, _line("Error", "WrappedServer", "FastFlowLM process has terminated"))
    assert "terminated" in nb._explain(server, window, "boom")
    assert log.fields["overlapped_another_request"] is False
