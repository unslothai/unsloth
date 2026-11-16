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


def test_a_failure_with_nothing_printed_still_points_at_the_log(tmp_path, monkeypatch, log):
    server = _server(tmp_path)
    server.log_dir = tmp_path / "logs" / "lemonade"
    server._open_log_file(4242)
    backend = _backend(
        tmp_path,
        monkeypatch,
        server,
        lambda path: ([], httpx.Response(500, json = {"error": {"message": "boom"}})),
    )
    with pytest.raises(nb.NpuError) as raised:
        backend.load(_MODEL, 32768)

    assert "boom (Lemonade's output is on the Logs page.)" in str(raised.value)
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
    threading.Timer(0.15, _prints, (server, "late error", "later detail")).start()
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


@pytest.mark.parametrize("path", ["/v1/pull", "/v1/delete", "/v1/models?show_all=true"])
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
            method = "GET" if path.startswith("/v1/models") else "POST"
            assert server.request(method, path).status_code == 500

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


def _flm(severity: str, text: str) -> str:
    return _line(severity, "Process", text)


_GENERIC = f"Failed to load {_MODEL} model!"
_TERMINATED = _line("Error", "WrappedServer", "FastFlowLM process has terminated")


@pytest.mark.parametrize(
    "printed, summary, raw, advice",
    [
        (
            [
                _flm(
                    "Error",
                    "[ERROR]  Failed to load model: Could not create allocation for hw context "
                    "instruction buffers.",
                ),
                _TERMINATED,
            ],
            _GENERIC,
            "Could not create allocation for hw context instruction buffers.",
            "ran out of memory",
        ),
        (
            [_flm("Error", "Error: Failed to allocate flm_rt::ext::bo: bad allocation")],
            _GENERIC,
            "Failed to allocate flm_rt::ext::bo: bad allocation",
            "ran out of memory",
        ),
        (
            [
                _flm(
                    "Error",
                    "\x1b[31m[ERROR]  Failed to load model: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL "
                    "failed\x1b[0m",
                )
            ],
            _GENERIC,
            "[ERROR]  Failed to load model: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed)",
            "hardware context",
        ),
        (
            [
                _flm("Info", "[WARNING]  Local model qwen3.6-moe:35b-a3b version: 0.9.45 < 1.0.2"),
                _TERMINATED,
            ],
            _GENERIC,
            "FastFlowLM process has terminated",
            "Delete the model and download it again",
        ),
        (
            [],
            "FLM NPU validation failed: flm validate failed with exit code -1058471934",
            "exit code -1058471934",
            "Smart App Control",
        ),
    ],
)
def test_a_known_fastflowlm_failure_gets_advice_after_the_raw_line(
    tmp_path, monkeypatch, log, printed, summary, raw, advice
):
    server = _server(tmp_path)
    backend = _backend(
        tmp_path,
        monkeypatch,
        server,
        lambda path: (printed, httpx.Response(500, json = {"error": {"message": summary}})),
    )
    with pytest.raises(nb.NpuError) as raised:
        backend.load(_MODEL, 32768)

    message = str(raised.value)
    assert raw in message
    assert advice in message
    assert message.index(raw) < message.index(advice)
    assert "\x1b" not in message


def test_an_unknown_failure_gets_no_advice(tmp_path, monkeypatch, log):
    server = _server(tmp_path)
    backend = _backend(
        tmp_path,
        monkeypatch,
        server,
        lambda path: (
            [_flm("Error", "[ERROR]  Failed to load model: something new"), _TERMINATED],
            httpx.Response(500, json = {"error": {"message": _GENERIC}}),
        ),
    )
    with pytest.raises(nb.NpuError) as raised:
        backend.load(_MODEL, 32768)

    assert str(raised.value) == (
        f"Loading {_MODEL} failed: {_GENERIC} ([ERROR]  Failed to load model: something new)"
    )


def test_flm_output_is_kept_only_while_a_load_is_open(tmp_path, monkeypatch):
    server = LemonadeServer(
        tmp_path / "lemond",
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
        log_dir = tmp_path / "logs" / "lemonade",
    )
    handle = server._open_log_file(4242)
    _level_calls(server, monkeypatch)
    # flm prints part of every chat request; lemond relays it for a backend started at debug.
    prompt = _flm("Info", '        "content": "my secret ...lan for today"')
    loading = _flm("Info", "[FLM]  Loading model: Qwen3.6-35B-A3B-NPU2")
    own = _line("Info", "Router", "Model loaded")

    with server.backend_output(server.open_window()):
        server._drain(SimpleNamespace(stdout = [loading + "\n"]), None)
    server._drain(SimpleNamespace(stdout = [prompt + "\n", own + "\n"]), handle)

    assert server.output_since(0, settle = 0) == [loading, own]
    assert "my secret" not in server.log_path.read_text(encoding = "utf-8")


def test_lemond_is_at_debug_for_the_load_and_back_at_info_after(tmp_path, monkeypatch, log):
    server = _server(tmp_path)
    server.port = 4242
    calls: list[str] = []

    def post(url, *, json, **kwargs):
        assert url == "http://127.0.0.1:4242/internal/set"
        calls.append(json["log_level"])
        return httpx.Response(200, json = {"status": "success"})

    def answer(path):
        calls.append(path)
        return ([], httpx.Response(500, json = {"error": {"message": "boom"}}))

    monkeypatch.setattr(server, "is_alive", lambda: True)
    monkeypatch.setattr(server, "_client", SimpleNamespace(post = post))
    backend = _backend(tmp_path, monkeypatch, server, answer)
    with pytest.raises(nb.NpuError):
        backend.load(_MODEL, 32768)

    assert calls == ["debug", "/v1/load", "info"]


def test_a_restart_never_inherits_debug_from_the_saved_config(tmp_path):
    import json

    server = _server(tmp_path)
    server.config_dir.mkdir(parents = True)
    # /internal/set saves what it is given, so a crash mid-load leaves this behind.
    (server.config_dir / "config.json").write_text('{"log_level": "debug"}', encoding = "utf-8")
    server._write_config()
    saved = json.loads((server.config_dir / "config.json").read_text(encoding = "utf-8"))
    assert saved["log_level"] == "info"


def _level_calls(server, monkeypatch) -> list[str]:
    calls: list[str] = []

    def post(url, *, json, **kwargs):
        calls.append(json["log_level"])
        return httpx.Response(200, json = {"status": "success"})

    server.port = 4242
    monkeypatch.setattr(server, "is_alive", lambda: True)
    monkeypatch.setattr(server, "_client", SimpleNamespace(post = post))
    return calls


def test_a_load_that_starts_during_a_chat_leaves_lemond_at_info(tmp_path, monkeypatch):
    server = _server(tmp_path)
    calls = _level_calls(server, monkeypatch)
    chat = server.open_window()
    prompt = _flm("Info", '        "content": "my secret ...lan for today"')

    with server.backend_output(server.open_window()):
        _prints(server, prompt)

    assert chat.shared
    assert calls == []
    assert server.output_since(0, settle = 0) == []


def test_a_chat_that_joins_a_load_keeps_its_content_out_of_the_log(tmp_path, monkeypatch):
    server = _server(tmp_path)
    calls = _level_calls(server, monkeypatch)
    loading = _flm("Info", "[FLM]  Loading model: Qwen3.6-35B-A3B-NPU2")
    failed = _line("Error", "Router", "Backend load failed: FastFlowLM failed to start")

    with server.backend_output(server.open_window()):
        _prints(server, loading)
        chat = server.open_window()
        _prints(
            server,
            _flm("Info", '        "content": "my secret ...lan for today"'),
            _line("Debug", "Server", "Message content: my secret plan for today"),
            "and its second line",
            failed,
        )

    assert chat.shared
    assert calls == ["debug", "info"]
    assert server.output_since(0, settle = 0) == [loading, failed]


def test_content_is_filtered_while_either_level_change_is_in_flight(tmp_path, monkeypatch):
    server = _server(tmp_path)
    content = _line("Debug", "Server", "Message content: my secret plan for today")
    chats = []

    def post(url, *, json, **kwargs):
        # A chat opens its window, and lemond prints at debug, before the call returns.
        if json["log_level"] == "debug":
            chats.append(server.open_window())
        _prints(server, content, "and its second line")
        return httpx.Response(200, json = {"status": "success"})

    server.port = 4242
    monkeypatch.setattr(server, "is_alive", lambda: True)
    monkeypatch.setattr(server, "_client", SimpleNamespace(post = post))
    with server.backend_output(server.open_window()):
        pass
    # Still drained after the load, before lemond says it is back at info.
    _prints(server, content, "and its second line")
    assert server.output_since(0, settle = 0) == []

    back = _line("Info", "Server", "Logging configuration updated (level=info, file=disabled)")
    _prints(server, back, "a line with no tag", content)
    assert server.output_since(0, settle = 0) == [back, "a line with no tag"]


def test_flm_output_drained_after_lemond_died_mid_load_is_kept(tmp_path, monkeypatch):
    server = _server(tmp_path)
    alive = [True]
    _level_calls(server, monkeypatch)
    monkeypatch.setattr(server, "is_alive", lambda: alive[0])
    reason = _flm("Error", "[ERROR]  Failed to load model: bad allocation")

    with server.backend_output(server.open_window()):
        alive[0] = False
    _prints(server, reason)

    assert server.output_since(0, settle = 0) == [reason]


def test_a_slow_or_failed_switch_back_to_info_never_lets_the_filter_lapse(tmp_path, monkeypatch):
    from core.inference import lemonade_server as ls

    server = _server(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(ls.time, "monotonic", lambda: clock[0])
    content = _line("Debug", "Server", "Message content: my secret plan for today")

    def post(url, *, json, **kwargs):
        if json["log_level"] == "info":
            # Longer than the deadline a finished switch gets, and lemond is still at debug.
            clock[0] += 9.0
            _prints(server, content, "and its second line")
            return httpx.Response(500)
        return httpx.Response(200, json = {"status": "success"})

    server.port = 4242
    monkeypatch.setattr(server, "is_alive", lambda: True)
    monkeypatch.setattr(server, "_client", SimpleNamespace(post = post))
    with server.backend_output(server.open_window()):
        pass
    clock[0] += 60.0
    _prints(server, content, "and its second line")

    assert server.output_since(0, settle = 0) == []


def test_chat_content_that_looks_like_a_lemonade_line_is_not_kept(tmp_path, monkeypatch):
    server = _server(tmp_path)
    _level_calls(server, monkeypatch)

    with server.backend_output(server.open_window()):
        chat = server.open_window()
        _prints(
            server,
            _line("Debug", "Server", "Message content: my secret plan for today"),
            "then [Info] (Router) the rest of my secret plan",
        )

    assert chat.shared
    assert server.output_since(0, settle = 0) == []


def test_only_lemonds_own_line_ends_the_filter_after_a_load(tmp_path, monkeypatch):
    server = _server(tmp_path)
    _level_calls(server, monkeypatch)
    with server.backend_output(server.open_window()):
        pass
    _prints(
        server,
        _line("Debug", "Server", "Message content: Logging configuration updated (level=info)"),
        "the rest of my secret plan",
    )
    assert server.output_since(0, settle = 0) == []


def test_unprefixed_lines_follow_the_record_they_continue(tmp_path):
    server = _server(tmp_path)
    _prints(
        server,
        _line("Debug", "Server", "chat preview"),
        "continuation of the chat preview",
        _line("Info", "Process", "backend request body"),
        "continuation of the request body",
        _line("Error", "Router", "model allocation failed"),
        "continuation of the error",
    )

    assert server.output_since(0, settle = 0) == [
        _line("Error", "Router", "model allocation failed"),
        "continuation of the error",
    ]


def test_a_log_that_cannot_reopen_names_no_older_file(tmp_path, monkeypatch):
    server = _server(tmp_path)
    server.log_dir = tmp_path / "logs"
    assert server._open_log_file(4242) is not None
    assert server.log_path is not None
    monkeypatch.setattr(
        "builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    assert server._open_log_file(4243) is None
    assert server.log_path is None


def test_a_reason_lemond_only_wraps_in_a_prefix_is_not_repeated(tmp_path, log):
    # As seen on a real XDNA2 NPU: lemond's line restates the error it returned.
    server = _server(tmp_path)
    window = server.open_window()
    _prints(
        server, _line("Error", "Router", "Backend load failed: FLM pull failed with exit code: 1")
    )
    message = "Failed to load model 'qwen3-0.6b-FLM': FLM pull failed with exit code: 1"
    assert nb._explain(server, window, message) == message


def test_a_model_name_digit_does_not_hide_the_reason(tmp_path, log):
    server = _server(tmp_path)
    window = server.open_window()
    _prints(
        server, _line("Error", "Router", "Backend load failed: FLM pull failed with exit code: 1")
    )
    explained = nb._explain(server, window, "Failed to load qwen3:1.7b model!")
    assert "FLM pull failed with exit code: 1" in explained
