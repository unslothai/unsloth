# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth run`: the latest-transformers consent before a load whose architecture no installed transformers
ships (nvidia/Cosmos3-Edge on the 5.5.0 default), asked through /validate with the /load body, as the UI asks."""

from __future__ import annotations

import ast
import inspect
import io
import json
import sys
import textwrap
import urllib.error
from pathlib import Path

import pytest
import typer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.commands import studio as studio_mod

VALIDATE = "/api/inference/validate"
INSTALL = "/api/inference/install-latest-transformers"
EDGE = "nvidia/Cosmos3-Edge"
_UPGRADE = {
    "model_type": "cosmos3_edge",
    "pypi_version": "5.20.2",
    "supported_in_pypi": True,
    "supported_in_main": True,
    "main_version": "5.21.0.dev0",
}
_EDGE_ANSWER = {
    "valid": True,
    "requires_transformers_upgrade": True,
    "transformers_upgrade": _UPGRADE,
}
_OK = {"success": True, "version": "5.20.2", "message": "ok"}


@pytest.fixture
def server(monkeypatch):
    """Stand in for the in-process server: answers per route (a list is served in order), records every POST with
    its raise_on_error flag, and honours that flag for a simulated HTTP failure (an Exception instance)."""
    calls = []

    def install(validate = None, install = None):
        answers = {VALIDATE: validate, INSTALL: install}

        def fake_post(
            port,
            api_key,
            path,
            payload,
            request_host,
            timeout,
            raise_on_error = False,
        ):
            calls.append({"path": path, "payload": payload, "raise_on_error": raise_on_error})
            answer = answers[path]
            if isinstance(answer, list):
                answer = answer.pop(0)
            if isinstance(answer, Exception):
                if raise_on_error:
                    raise RuntimeError(str(answer))
                return None
            return answer

        monkeypatch.setattr(studio_mod, "_studio_post_json", fake_post)
        return calls

    return install


@pytest.fixture
def terminal(monkeypatch):
    """Who is at the terminal: None (nobody can or does answer) or the answer typed."""
    asked = []

    def set_answer(answer):
        def fake_confirm(question):
            asked.append(question)
            return answer

        monkeypatch.setattr(studio_mod, "_confirm_while_attended", fake_confirm)
        return asked

    return set_answer


def _payload(model = EDGE, **kw):
    return studio_mod._load_payload(model, None, 4096, True, **kw)


def _ensure(yes = False, payload = None):
    studio_mod._ensure_transformers_for_model(
        8888, "sk-test", payload or _payload(), "127.0.0.1", yes, True
    )


def _installs(calls):
    return [c["payload"] for c in calls if c["path"] == INSTALL]


def test_the_preflight_is_validate_with_the_load_body(server):
    calls = server(validate = {"valid": True, "requires_transformers_upgrade": False})
    payload = _payload(gpu_memory_mode = "manual", gpu_ids = [0])
    _ensure(yes = True, payload = payload)
    assert calls == [{"path": VALIDATE, "payload": payload, "raise_on_error": False}]


def test_a_gguf_never_asks(server, terminal):
    """/validate skips the upgrade check for a GGUF (llama.cpp serves it without transformers), with or without
    an explicit variant."""
    calls = server(
        validate = {"valid": True, "is_gguf": True, "requires_transformers_upgrade": False}
    )
    asked = terminal(True)
    _ensure(yes = False, payload = _payload("unsloth/NewArch-GGUF"))
    assert _installs(calls) == [] and asked == []


def test_an_unavailable_check_never_blocks_the_load(server, terminal):
    calls = server(validate = urllib.error.URLError("refused"))
    asked = terminal(True)
    _ensure(yes = False)
    assert _installs(calls) == [] and asked == []


def test_nobody_at_the_terminal_needs_yes(server, terminal):
    calls = server(validate = _EDGE_ANSWER)
    terminal(None)
    with pytest.raises(RuntimeError, match = "--yes"):
        _ensure(yes = False)
    assert _installs(calls) == []


def test_a_declined_prompt_installs_nothing(server, terminal):
    calls = server(validate = _EDGE_ANSWER)
    asked = terminal(False)
    with pytest.raises(RuntimeError, match = "not loaded"):
        _ensure(yes = False)
    assert _installs(calls) == []
    assert asked and "cosmos3_edge" in asked[0] and "5.20.2" in asked[0]


def test_an_accepted_prompt_installs_the_release(server, terminal):
    calls = server(validate = _EDGE_ANSWER, install = _OK)
    terminal(True)
    _ensure(yes = False)
    assert _installs(calls) == [{"version": "5.20.2"}]
    assert [c["raise_on_error"] for c in calls] == [False, True]


def test_yes_installs_a_release_without_asking(server, terminal):
    calls = server(validate = _EDGE_ANSWER, install = _OK)
    asked = terminal(True)
    _ensure(yes = True)
    assert _installs(calls) == [{"version": "5.20.2"}] and asked == []


def test_yes_does_not_cover_a_development_build(server, terminal):
    upgrade = {**_UPGRADE, "supported_in_pypi": False}
    calls = server(validate = {**_EDGE_ANSWER, "transformers_upgrade": upgrade}, install = _OK)
    terminal(None)
    with pytest.raises(RuntimeError, match = "development build"):
        _ensure(yes = True)
    assert _installs(calls) == []


def test_a_development_build_is_named_and_installed_on_a_yes(server, terminal):
    upgrade = {**_UPGRADE, "supported_in_pypi": False}
    calls = server(
        validate = {**_EDGE_ANSWER, "transformers_upgrade": upgrade}, install = {"success": True}
    )
    asked = terminal(True)
    _ensure(yes = True)
    assert _installs(calls) == [{"version": "5.21.0.dev0"}]
    assert "unreleased development build" in asked[0]


def test_an_architecture_no_release_ships_is_left_to_the_load(server, terminal):
    upgrade = {**_UPGRADE, "supported_in_pypi": False, "supported_in_main": False}
    calls = server(validate = {**_EDGE_ANSWER, "transformers_upgrade": upgrade})
    asked = terminal(True)
    _ensure(yes = True)
    assert _installs(calls) == [] and asked == []


def test_a_release_published_since_the_check_is_installed_instead(server, terminal):
    """The install re-checks PyPI and names the newer release; the UI retries with it once, and so does --yes."""
    stale = {"success": False, "message": "not the latest", "latest_version": "5.20.3"}
    calls = server(validate = _EDGE_ANSWER, install = [stale, {"success": True}])
    terminal(None)
    _ensure(yes = True)
    assert _installs(calls) == [{"version": "5.20.2"}, {"version": "5.20.3"}]


def test_a_failed_install_is_an_error(server):
    server(validate = _EDGE_ANSWER, install = {"success": False, "message": "see the Unsloth logs"})
    with pytest.raises(RuntimeError, match = "see the Unsloth logs"):
        _ensure(yes = True)


def test_a_refused_install_carries_the_server_reason(server):
    server(
        validate = _EDGE_ANSWER,
        install = urllib.error.URLError('HTTP 409: {"detail":"A training run is active."}'),
    )
    with pytest.raises(RuntimeError, match = "training run is active"):
        _ensure(yes = True)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_post_helper_builds_the_request_and_reads_the_answer(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen.update(
            url = request.full_url,
            method = request.get_method(),
            auth = request.get_header("Authorization"),
        )
        seen["body"] = json.loads(request.data)
        return _Resp(json.dumps(seen.get("answer", {"ok": True})).encode())

    monkeypatch.setattr(studio_mod, "_direct_urlopen", fake_urlopen)
    answer = studio_mod._studio_post_json(
        8888, "sk-test", VALIDATE, {"model_path": EDGE}, "127.0.0.1", 5
    )
    assert answer == {"ok": True}
    assert seen == {
        "url": "http://127.0.0.1:8888/api/inference/validate",
        "method": "POST",
        "auth": "Bearer sk-test",
        "body": {"model_path": EDGE},
    }
    monkeypatch.setattr(studio_mod, "_direct_urlopen", lambda request, timeout: _Resp(b"[1, 2]"))
    assert studio_mod._studio_post_json(8888, "sk-test", VALIDATE, {}, "127.0.0.1", 5) is None


def test_the_post_helper_reports_http_errors_only_when_asked(monkeypatch):
    def refuse(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 409, "Conflict", {}, io.BytesIO(b'{"detail":"busy"}')
        )

    monkeypatch.setattr(studio_mod, "_direct_urlopen", refuse)
    args = (8888, "sk-test", INSTALL, {"version": "5.20.2"}, "127.0.0.1", 5)
    assert studio_mod._studio_post_json(*args) is None
    with pytest.raises(RuntimeError, match = "HTTP 409.*busy"):
        studio_mod._studio_post_json(*args, raise_on_error = True)


@pytest.mark.parametrize(
    "streams, owns", [(False, True), (True, False)], ids = ["no_terminal", "background_job"]
)
def test_nobody_can_answer_without_a_driven_terminal(monkeypatch, streams, owns):
    monkeypatch.setattr(studio_mod, "_prompt_streams_interactive", lambda: streams)
    monkeypatch.setattr(studio_mod, "_prompt_owns_the_terminal", lambda: owns)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: pytest.fail("must not prompt"))
    assert studio_mod._confirm_while_attended("Install?") is None


@pytest.mark.parametrize(
    "typed, expected", [(None, None), ("y\n", True), ("yes\n", True), ("\n", False), ("n\n", False)]
)
def test_an_attended_prompt_waits_a_bounded_time(monkeypatch, typed, expected):
    import select

    monkeypatch.setattr(studio_mod, "_prompt_streams_interactive", lambda: True)
    monkeypatch.setattr(studio_mod, "_prompt_owns_the_terminal", lambda: True)
    waits = []

    def fake_select(r, w, x, timeout):
        waits.append(timeout)
        return (r if typed is not None else [], [], [])

    monkeypatch.setattr(select, "select", fake_select)
    monkeypatch.setattr(sys, "stdin", io.StringIO(typed or ""))
    assert studio_mod._confirm_while_attended("Install?") is expected
    assert waits == [studio_mod._UNATTENDED_PROMPT_SECONDS]


def test_run_asks_before_the_load_and_passes_its_yes():
    """run() asks for every transformers load (engine auto, not NPU) and hands over --yes and --silent; the GGUF
    decision is the server's (/validate), not the flag's."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(studio_mod.run)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "_ensure_transformers_for_model"
    ]
    assert len(calls) == 1
    args = [ast.unparse(a) for a in calls[0].args]
    assert args[0:2] == ["actual_port", "api_key"] and args[3:] == ["request_host", "yes", "silent"]
    assert args[2].startswith("_load_payload(")
    src = inspect.getsource(studio_mod.run)
    assert 'if engine == "auto" and not npu_path:' in src
    assert src.index("_ensure_transformers_for_model(") < src.index("_load_model_via_http(")
