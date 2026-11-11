# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What Lemonade printed around a failed NPU load reaches the log and the error, fed through _drain."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from core.inference import npu_backend as nb
from core.inference.lemonade_server import LemonadeServer

_MODEL = "qwen3.6-moe-35b-a3b-FLM"


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message, *args) -> None:
        self.warnings.append(message % args if args else message)

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
    monkeypatch.setattr(nb, "logger", recorder)
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


def test_a_failure_lemonade_only_summarizes_points_at_the_log(tmp_path, monkeypatch, log):
    server = _server(tmp_path)
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
    assert "Logs" in str(raised.value)
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
