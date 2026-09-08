# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Settings > Logs endpoints. These are polled once a second while the
tab is open, so "the file is not there yet" has to be a 200 with a status, not
an error the UI flashes on every tick."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.settings as settings_route


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route._require_ui_session] = lambda: None
    return TestClient(app, raise_server_exceptions = False)


def _seed_server_log(body: str = "hello\n") -> Path:
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / f"server-20260813-120000-pid{os.getpid()}.log"
    path.write_text(body, encoding = "utf-8")
    return path


def test_the_sources_list_names_the_running_session(client):
    path = _seed_server_log()
    body = client.get("/api/settings/debug/logs/sources").json()
    assert body["default_source_id"]
    assert any(s["label"] == path.name and s["is_current"] for s in body["sources"])
    assert any(s["realpath"] == str(path.resolve()) for s in body["sources"])


def test_the_first_read_returns_the_tail_and_a_cursor(client):
    _seed_server_log("".join(f"line{i}\n" for i in range(20)))
    body = client.get("/api/settings/debug/logs").json()
    assert body["status"] == "ok"
    assert body["lines"][-1] == "line19"
    assert body["reset"] is True
    assert body["cursor"]
    assert body["realpath"]


def test_a_second_read_returns_only_what_was_appended(client):
    path = _seed_server_log("a\n")
    cursor = client.get("/api/settings/debug/logs").json()["cursor"]
    with open(path, "a", encoding = "utf-8") as handle:
        handle.write("b\nc\n")
    body = client.get("/api/settings/debug/logs", params = {"cursor": cursor}).json()
    assert body["lines"] == ["b", "c"]
    assert body["reset"] is False


def test_an_idle_poll_is_a_200_with_no_lines(client):
    _seed_server_log("a\n")
    cursor = client.get("/api/settings/debug/logs").json()["cursor"]
    response = client.get("/api/settings/debug/logs", params = {"cursor": cursor})
    assert response.status_code == 200
    assert response.json()["lines"] == []


def test_an_unknown_source_is_a_404_so_the_picker_refetches(client):
    _seed_server_log()
    response = client.get("/api/settings/debug/logs", params = {"source": "server:" + "0" * 16})
    assert response.status_code == 404


@pytest.mark.parametrize(
    "hostile",
    ["server:../../../../etc/passwd", "../../etc/passwd", "nosuch:abcdef0123456789"],
)
def test_a_crafted_source_id_never_reads_a_file(client, hostile):
    _seed_server_log()
    response = client.get("/api/settings/debug/logs", params = {"source": hostile})
    assert response.status_code == 404
    assert "root:" not in response.text


def test_no_logs_at_all_reports_a_reason_rather_than_an_empty_view(client):
    body = client.get("/api/settings/debug/logs").json()
    assert body["status"] in {"missing", "disabled"}
    assert body["reason"]


def test_file_logging_turned_off_says_so(client, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_NO_FILE_LOG", "1")
    body = client.get("/api/settings/debug/logs").json()
    assert body["status"] == "disabled"
    assert "UNSLOTH_STUDIO_NO_FILE_LOG" in body["reason"]


def test_a_deleted_file_is_not_a_500(client):
    path = _seed_server_log()
    cursor = client.get("/api/settings/debug/logs").json()["cursor"]
    path.unlink()
    response = client.get("/api/settings/debug/logs", params = {"cursor": cursor})
    assert response.status_code in (200, 404)
    assert response.status_code != 500


def test_credentials_in_the_log_never_reach_the_response(client):
    _seed_server_log(
        "loading with hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345\n"
        'auth: {"api_key":"abcdef123456"}\n'
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg\n"
    )
    text = client.get("/api/settings/debug/logs").text
    for secret in (
        "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "abcdef123456",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefg",
    ):
        assert secret not in text


def test_an_api_key_session_cannot_read_the_logs():
    """Log lines and a local realpath are UI-operator material, not something a
    remote API key should be able to pull."""
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[settings_route.get_current_subject] = lambda: "admin"
    app.dependency_overrides[settings_route.authenticated_via_api_key] = lambda: True
    api_client = TestClient(app, raise_server_exceptions = False)
    _seed_server_log()
    assert api_client.get("/api/settings/debug/logs").status_code == 403
    assert api_client.get("/api/settings/debug/logs/sources").status_code == 403


def test_the_endpoints_stay_out_of_the_access_log():
    """Load bearing, not tidiness. These paths are polled while the tab is open
    and they read the very file the access log writes to, so without the
    suppression each poll appends a line the next poll reads back."""
    from loggers.handlers import _is_quiet_success

    # Behaviour, not membership: these go through _SELF_READ_PATHS rather than
    # _QUIET_SUCCESS_PATHS because --verbose must not lift them.
    # test_debug_log_self_feedback.py proves it over the real middleware.
    assert _is_quiet_success("GET", "/api/settings/debug/logs", 200, False) is True
    assert _is_quiet_success("GET", "/api/settings/debug/logs/sources", 200, False) is True


def test_a_stale_session_is_flagged_when_file_logging_is_off(client, monkeypatch):
    """An old file with logging now off must not read as a live view.

    Reported on the PR: with UNSLOTH_STUDIO_NO_FILE_LOG=1 and a log left over
    from a previous run, the read path answered a plain "ok" and the viewer sat
    there looking live while nothing would ever be appended to it again.
    """
    _seed_server_log("a previous session\n")
    monkeypatch.setenv("UNSLOTH_STUDIO_NO_FILE_LOG", "1")
    body = client.get("/api/settings/debug/logs").json()
    assert body["status"] == "ok"
    assert body["lines"] == ["a previous session"]
    assert body["file_logging_disabled"] is True


def test_file_logging_disabled_is_false_on_an_ordinary_read(client):
    _seed_server_log()
    assert client.get("/api/settings/debug/logs").json()["file_logging_disabled"] is False


def test_a_burst_larger_than_one_response_says_more_is_pending(client):
    """The remainder is delivered, and the caller is told to come back for it."""
    from utils import debug_log_reader

    path = _seed_server_log()
    cursor = client.get("/api/settings/debug/logs").json()["cursor"]
    burst = debug_log_reader.MAX_LINES_PER_RESPONSE + 500
    with path.open("a", encoding = "utf-8") as handle:
        handle.write("".join(f"line {index}\n" for index in range(burst)))

    first = client.get("/api/settings/debug/logs", params = {"cursor": cursor}).json()
    assert len(first["lines"]) == debug_log_reader.MAX_LINES_PER_RESPONSE
    assert first["more_pending"] is True

    second = client.get("/api/settings/debug/logs", params = {"cursor": first["cursor"]}).json()
    assert second["more_pending"] is False
    assert first["lines"] + second["lines"] == [f"line {index}" for index in range(burst)]


def _seed_llama_log(body: str = "llama runner line\n") -> Path:
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "llama-server"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / "llama-1786000000.log"
    path.write_text(body, encoding = "utf-8")
    return path


def _source_id(client, family: str) -> str:
    body = client.get("/api/settings/debug/logs/sources").json()
    return next(s["id"] for s in body["sources"] if s["family"] == family)


def test_a_runner_log_is_not_called_stale_when_only_the_server_tee_is_off(client, monkeypatch):
    """UNSLOTH_STUDIO_NO_FILE_LOG only skips run.py's tee.

    The llama and diffusion runners and the desktop shell keep writing, so
    treating the setting as global told a user watching a live llama-server log
    that it would not update while the failure was still being appended to it.
    """
    _seed_llama_log()
    monkeypatch.setenv("UNSLOTH_STUDIO_NO_FILE_LOG", "1")
    source_id = _source_id(client, "llama-server")
    body = client.get("/api/settings/debug/logs", params = {"source": source_id}).json()
    assert body["status"] == "ok"
    assert body["lines"] == ["llama runner line"]
    assert body["file_logging_disabled"] is False


def test_the_server_log_is_still_called_stale(client, monkeypatch):
    _seed_server_log()
    _seed_llama_log()
    monkeypatch.setenv("UNSLOTH_STUDIO_NO_FILE_LOG", "1")
    source_id = _source_id(client, "server")
    body = client.get("/api/settings/debug/logs", params = {"source": source_id}).json()
    assert body["file_logging_disabled"] is True
