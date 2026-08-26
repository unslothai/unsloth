# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Settings > Logs endpoints. These are polled once a second while the
tab is open, so "the file is not there yet" has to be a 200 with a status, not
an error the UI flashes on every tick."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

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
    assert api_client.get("/api/settings/debug/logs/export").status_code == 403


def test_export_contains_every_visible_source_and_masks_credentials(client):
    secret = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    cookie_secret = "abc123def456COOKIESECRET"
    pair_cookie_secret = "abc123def456PAIRSECRET"
    redis_secret = "correct-horse-battery"
    server = _seed_server_log(
        f"server line\ntoken={secret}\n"
        rf'payload="{{\"Cookie\":\"session={cookie_secret}\"}}"'
        f"\nheaders=[('Cookie', 'session={pair_cookie_secret}')]"
        f"\nredis://:{redis_secret}@localhost:6379/0\n"
    )
    llama = _seed_llama_log("llama runner line\n")

    response = client.get("/api/settings/debug/logs/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"].startswith('attachment; filename="unsloth-logs-')
    assert response.headers["cache-control"] == "private, no-store"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {f"server/{server.name}", f"llama-server/{llama.name}"}
        exported = "\n".join(archive.read(name).decode("utf-8") for name in archive.namelist())
    assert "server line" in exported
    assert "llama runner line" in exported
    assert secret not in exported
    assert cookie_secret not in exported
    assert pair_cookie_secret not in exported
    assert redis_secret not in exported
    assert "hf_<redacted>" in exported


def test_export_with_no_logs_is_still_a_valid_zip(client):
    response = client.get("/api/settings/debug/logs/export")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == []


def test_browser_export_refuses_history_that_cannot_be_downloaded_boundedly(client, monkeypatch):
    from utils import debug_log_sources

    monkeypatch.setattr(
        debug_log_sources,
        "list_sources",
        lambda **_: [
            SimpleNamespace(size_bytes = settings_route.BROWSER_LOG_EXPORT_MAX_SOURCE_BYTES + 1)
        ],
    )

    response = client.get("/api/settings/debug/logs/export?browser=true")

    assert response.status_code == 413
    assert "desktop app" in response.json()["detail"].lower()


def test_export_starts_the_response_before_building_the_archive(monkeypatch):
    from utils import debug_log_export, debug_log_sources

    built = []
    monkeypatch.setattr(debug_log_sources, "list_sources", lambda **_: [])

    def _build(sources):
        built.append(list(sources))
        return io.BytesIO(b"zip bytes")

    monkeypatch.setattr(debug_log_export, "build_debug_log_archive", _build)
    response = settings_route.export_debug_logs("admin", None)
    assert built == []

    async def _first_chunk():
        return await response.body_iterator.__anext__()

    assert asyncio.run(_first_chunk()) == b"zip bytes"
    assert built == [[]]


def test_export_is_not_limited_to_the_viewers_per_family_window(client):
    directory = Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "logs" / "llama-server"
    directory.mkdir(parents = True)
    expected = set()
    for index in range(12):
        name = f"llama-{1786000000 + index}.log"
        (directory / name).write_text(f"attempt {index}\n", encoding = "utf-8")
        expected.add(f"llama-server/{name}")

    # The picker remains deliberately capped; the explicit export does not.
    listed = client.get("/api/settings/debug/logs/sources").json()["sources"]
    assert len([source for source in listed if source["family"] == "llama-server"]) == 10

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == expected


def test_export_omits_an_oversized_record_instead_of_splitting_a_secret(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    secret = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    path = _seed_server_log("x" * EXPORT_READ_BYTES + secret + "\nkept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "oversized log record omitted" in exported
    assert exported.splitlines()[-1] == "kept"


def test_export_masks_a_credential_value_on_the_next_line(client):
    secret = "correct-horse-battery-staple"
    path = _seed_server_log(f"password:\n  {secret}\nkept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "  <redacted>" in exported
    assert exported.splitlines()[-1] == "kept"


def test_export_masks_every_line_of_a_multiline_plain_yaml_secret(client):
    first = "correct-horse-battery"
    second = "staple-secret-value"
    path = _seed_server_log(f"password:\n  {first}\n  {second}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert exported.count("  <redacted>") == 2
    assert "ordinary: kept" in exported


def test_export_masks_plain_yaml_continuations_after_an_inline_value(client):
    first = "correct-horse-battery"
    second = "staple-secret-value"
    path = _seed_server_log(f"password: {first}\n  {second}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


@pytest.mark.parametrize("indicator", ["|-", ">", "|2-"])
def test_export_masks_every_line_of_a_yaml_block_scalar(client, indicator):
    first = "correct-horse-battery-staple"
    second = "another-secret-value"
    path = _seed_server_log(
        f"credentials:\n  password: {indicator}\n    {first}\n    {second}\n  ordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert exported.count("    <redacted>") == 2
    assert "  ordinary: kept" in exported


def test_export_honors_an_explicit_yaml_block_indent(client):
    first = "deep-secret-value"
    second = "shallower-secret-value"
    path = _seed_server_log(
        f"credentials:\n  password: |2-\n      {first}\n    {second}\n  ordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "  ordinary: kept" in exported


def test_export_masks_a_continued_cookie_header(client):
    secret = "session-cookie-secret"
    path = _seed_server_log(f"Cookie:\n  session={secret}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "  <redacted>" in exported
    assert "ordinary: kept" in exported


def test_export_masks_every_record_of_a_multiline_quoted_secret(client):
    first = "correct horse"
    second = "battery staple"
    path = _seed_server_log(f'password: "{first}\n  {second}"\nordinary: kept\n')

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


@pytest.mark.parametrize(
    "name",
    ["MYSQL_PWD", "AWS_ACCESS_KEY_ID", "rediscli_auth"],
)
def test_export_masks_continued_classified_environment_values(client, name):
    secret = "NOTSTANDARDSECRET123"
    path = _seed_server_log(f"{name}=\n  {secret}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "ordinary: kept" in exported


def test_export_masks_classified_environment_values_in_structured_logs(client):
    secret = "plainopaquecredential123456"
    path = _seed_server_log(f'{{"GITHUB_TOKEN":"{secret}","request_id":"req-42"}}\n')

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert '"GITHUB_TOKEN":"<redacted>"' in exported
    assert '"request_id":"req-42"' in exported


@pytest.mark.parametrize(
    "parameter",
    ["token", "x-amz-signature", "x-goog-signature"],
)
def test_export_masks_query_credentials_wrapped_onto_the_next_record(client, parameter):
    secret = "plainopaquecredential123456"
    path = _seed_server_log(
        f"https://storage.example/object?{parameter}=\n  {secret}\nordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "  <redacted>" in exported
    assert "ordinary: kept" in exported


def test_export_preserves_siblings_after_an_indented_quoted_secret_key(client):
    secret = "correct-horse-battery-staple"
    path = _seed_server_log(
        f'  "password":\n    {secret}\n  "request_id": "req-42"\n  "status": 500\n'
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert '  "request_id": "req-42"' in exported
    assert '  "status": 500' in exported


def test_export_masks_a_tuple_style_continued_cookie(client):
    secret = "abc123SESSIONSECRET"
    path = _seed_server_log(
        f"headers=[('Cookie',\n  'unsloth_session={secret}')]\nordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "ordinary: kept" in exported


@pytest.mark.parametrize("label", ["PRIVATE KEY", "PGP PRIVATE KEY BLOCK"])
def test_export_masks_a_pem_private_key_block(client, label):
    key_body = "PRIVATEKEYBODYSHOULDNOTSURVIVE"
    path = _seed_server_log(
        f"-----BEGIN {label}-----\n{key_body}\n-----END {label}-----\nordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert key_body not in exported
    assert "BEGIN PRIVATE KEY" not in exported
    assert "END PRIVATE KEY" not in exported
    assert "ordinary: kept" in exported


def test_export_keeps_private_key_state_after_an_oversized_record(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    key_body = "OVERSIZEDPEMBODYSHOULDNOTSURVIVE"
    path = _seed_server_log(
        "x" * EXPORT_READ_BYTES
        + "-----BEGIN PRIVATE KEY-----\n"
        + key_body
        + "\n-----END PRIVATE KEY-----\nordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert key_body not in exported
    assert "END PRIVATE KEY" not in exported
    assert "ordinary: kept" in exported


def test_export_tracks_a_quoted_cookie_after_an_oversized_record(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    second = "SECONDCOOKIEPARTSECRET"
    path = _seed_server_log(
        "x" * EXPORT_READ_BYTES + ' Cookie: "session=FIRST\n' + second + '"\nordinary: kept\n'
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_treats_doubled_yaml_single_quotes_as_escapes(client):
    first = "correct"
    second = "it''s still secret"
    third = "and-this-is-too"
    path = _seed_server_log(f"password: '{first}\n  {second}\n  {third}'\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert third not in exported
    assert "ordinary: kept" in exported


def test_export_keeps_secret_context_after_an_oversized_record(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    secret = "correct-horse-battery-staple"
    path = _seed_server_log("x" * EXPORT_READ_BYTES + f" password:\n  {secret}\nkept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "oversized log record omitted" in exported
    assert exported.splitlines()[-1] == "kept"


def test_export_does_not_carry_a_closed_secret_past_an_oversized_record(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    path = _seed_server_log(
        "password=done "
        + "x" * EXPORT_READ_BYTES
        + "\n  traceback detail stays visible\nordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert "oversized log record omitted" in exported
    assert "traceback detail stays visible" in exported
    assert exported.splitlines()[-1] == "ordinary: kept"


def test_export_scans_the_whole_oversized_record_for_secret_context(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    secret = "correct-horse-battery-staple"
    path = _seed_server_log("password:" + " " * (EXPORT_READ_BYTES + 32) + f"\n  {secret}\nkept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert exported.splitlines()[-1] == "kept"


def test_export_keeps_repeated_continuation_state_after_an_oversized_record(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    second = "second-secret-part"
    third = "third-secret-part"
    path = _seed_server_log(
        "PASSWORD=" + "x" * EXPORT_READ_BYTES + f"\\\n{second}\\\n{third}\nordinary: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert second not in exported
    assert third not in exported
    assert "ordinary: kept" in exported


def test_export_keeps_quoted_state_after_an_oversized_record(client):
    from utils.debug_log_export import EXPORT_READ_BYTES

    second = "second-secret-part"
    third = "third-secret-part"
    path = _seed_server_log(
        'password="' + "x" * EXPORT_READ_BYTES + f'\n{second}\n{third}"\nordinary: kept\n'
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert second not in exported
    assert third not in exported
    assert "ordinary: kept" in exported


@pytest.mark.parametrize(
    "body",
    [
        "2026-08-25 INFO password:\n  correct-horse-battery-staple\nkept\n",
        '2026-08-25 INFO password: "correct horse\n  battery staple"\nkept\n',
    ],
)
def test_export_tracks_multiline_secrets_after_a_log_prefix(client, body):
    path = _seed_server_log(body)

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert "correct" not in exported
    assert "battery" not in exported
    assert exported.splitlines()[-1] == "kept"


def test_export_tracks_a_quoted_secret_when_its_value_contains_a_key_word(client):
    first = "my secret phrase"
    second = "continuation-value"
    path = _seed_server_log(f'password: "{first}\n  {second}"\nordinary: kept\n')

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_preserves_yaml_sequence_boundaries_around_a_secret(client):
    secret = "correct-horse-battery-staple"
    path = _seed_server_log(
        f"items:\n  - password: {secret}\n    username: alice\n  - endpoint: kept\n"
    )

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "username: alice" in exported
    assert "endpoint: kept" in exported


def test_export_tracks_an_unindented_shell_secret_continuation(client):
    first = "correct-horse"
    second = "battery-staple"
    path = _seed_server_log(f"PASSWORD={first}-\\\n{second}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_tracks_a_quoted_shell_secret_continuation(client):
    first = "correct-horse"
    second = "battery-staple"
    path = _seed_server_log(f'PASSWORD="{first}-\\\n{second}"\nordinary: kept\n')

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_tracks_a_quoted_flag_secret_continuation(client):
    first = "correct-horse"
    second = "battery-staple"
    path = _seed_server_log(f'llama --api-key "{first}-\\\n{second}"\nordinary: kept\n')

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_tracks_an_unquoted_flag_secret_continuation(client):
    first = "correct-horse"
    second = "battery-staple"
    path = _seed_server_log(f"llama --api-key {first}-\\\n{second}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_does_not_extend_flag_context_from_a_later_argument(client):
    secret = "correct-horse"
    visible = "ordinary-visible"
    path = _seed_server_log(f"llama --api-key {secret} --output path-\\\n{visible}\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert visible in exported


def test_export_tracks_folded_cookie_pairs_after_an_inline_fragment(client):
    first = "first-session-secret"
    second = "second-refresh-secret"
    path = _seed_server_log(f"Cookie: session={first};\n  refresh={second}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_tracks_a_folded_cookie_value_continuation(client):
    first = "correct-horse"
    second = "battery-staple"
    path = _seed_server_log(f"Cookie: session={first}-\n  {second}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert "ordinary: kept" in exported


def test_export_tracks_a_token_after_a_standalone_authorization_scheme(client):
    secret = "short-auth-token"
    path = _seed_server_log(f"Authorization: Bearer\n  {secret}\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "Authorization: Bearer" in exported
    assert "ordinary: kept" in exported


def test_export_uses_the_opening_quote_for_multiline_secret_state(client):
    first = 'first "nickname'
    second = 'second "alias'
    third = "third-secret"
    path = _seed_server_log(f"password: '{first}\n  {second}\n  {third}'\nordinary: kept\n")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert first not in exported
    assert second not in exported
    assert third not in exported
    assert "ordinary: kept" in exported


def test_export_preserves_many_carriage_return_delimited_records(client):
    records = [f"progress {index:05d} " + "x" * 48 for index in range(20_000)]
    path = _seed_server_log("\r".join(records) + "\r")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert "oversized log record omitted" not in exported
    assert records[0] in exported
    assert records[-1] in exported


def test_export_masks_carriage_return_delimited_secret_continuations(client):
    secret = "correct-horse-battery-staple"
    path = _seed_server_log(f"password:\r  {secret}\rordinary: kept\r")

    response = client.get("/api/settings/debug/logs/export")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        exported = archive.read(f"server/{path.name}").decode("utf-8")
    assert secret not in exported
    assert "<redacted>\rordinary: kept\r" in exported


def test_export_stops_at_each_sources_enumerated_size(tmp_path):
    from utils.debug_log_export import build_debug_log_archive
    from utils.debug_log_sources import LogSource

    path = tmp_path / "growing.log"
    initial = b"snapshot\n"
    path.write_bytes(initial)
    source = LogSource(
        id = "server:growing",
        family = "server",
        label = path.name,
        realpath = str(path),
        size_bytes = len(initial),
        modified_at = 0,
        is_current = False,
    )
    with open(path, "ab") as handle:
        handle.write(b"appended-after-enumeration\n")

    output = build_debug_log_archive([source])
    try:
        with zipfile.ZipFile(output) as archive:
            exported = archive.read("server/growing.log")
    finally:
        output.close()
    assert exported == initial


def test_export_warns_when_a_source_shrinks_after_enumeration(tmp_path):
    from utils.debug_log_export import build_debug_log_archive
    from utils.debug_log_sources import LogSource

    path = tmp_path / "shrinking.log"
    path.write_bytes(b"abc")
    source = LogSource(
        id = "server:shrinking",
        family = "server",
        label = path.name,
        realpath = str(path),
        size_bytes = 10,
        modified_at = 0,
        is_current = False,
    )

    output = build_debug_log_archive([source])
    try:
        with zipfile.ZipFile(output) as archive:
            assert archive.read("server/shrinking.log") == b"abc"
            warning = archive.read("EXPORT_WARNINGS.txt").decode("utf-8")
    finally:
        output.close()
    assert "server/shrinking.log" in warning
    assert "OSError" in warning


def test_export_refuses_a_source_replaced_by_a_symlink(tmp_path):
    from utils.debug_log_export import build_debug_log_archive
    from utils.debug_log_sources import LogSource

    secret = tmp_path / "private.txt"
    secret.write_text("must-not-export", encoding = "utf-8")
    path = tmp_path / "server.log"
    path.symlink_to(secret)
    source = LogSource(
        id = "server:swapped",
        family = "server",
        label = path.name,
        realpath = str(path),
        size_bytes = secret.stat().st_size,
        modified_at = 0,
        is_current = False,
    )

    output = build_debug_log_archive([source])
    try:
        with zipfile.ZipFile(output) as archive:
            assert "server/server.log" not in archive.namelist()
            warning = archive.read("EXPORT_WARNINGS.txt").decode("utf-8")
    finally:
        output.close()
    assert "must-not-export" not in warning


def test_export_refuses_a_regular_file_replaced_after_enumeration(tmp_path):
    from utils.debug_log_export import build_debug_log_archive
    from utils.debug_log_sources import LogSource

    path = tmp_path / "server.log"
    path.write_text("original failure\n", encoding = "utf-8")
    enumerated = path.stat()
    source = LogSource(
        id = "server:rotated",
        family = "server",
        label = path.name,
        realpath = str(path),
        size_bytes = enumerated.st_size,
        modified_at = enumerated.st_mtime,
        is_current = False,
        device_id = enumerated.st_dev,
        inode = enumerated.st_ino,
    )
    path.replace(tmp_path / "server.log.1")
    path.write_text("replacement content\n", encoding = "utf-8")

    output = build_debug_log_archive([source])
    try:
        with zipfile.ZipFile(output) as archive:
            assert "server/server.log" not in archive.namelist()
            warning = archive.read("EXPORT_WARNINGS.txt").decode("utf-8")
    finally:
        output.close()
    assert "OSError" in warning


def test_export_warning_escapes_a_surrogate_filename(tmp_path):
    from utils.debug_log_export import build_debug_log_archive
    from utils.debug_log_sources import LogSource

    label = "broken-\udcff.log"
    source = LogSource(
        id = "server:surrogate",
        family = "server",
        label = label,
        realpath = str(tmp_path / label),
        size_bytes = 0,
        modified_at = 0,
        is_current = False,
    )

    output = build_debug_log_archive([source])
    try:
        with zipfile.ZipFile(output) as archive:
            warning = archive.read("EXPORT_WARNINGS.txt").decode("utf-8")
    finally:
        output.close()
    assert "broken-\\udcff.log" in warning


def test_export_warning_does_not_expose_a_source_realpath(tmp_path):
    from utils.debug_log_export import build_debug_log_archive
    from utils.debug_log_sources import LogSource

    missing = tmp_path / "private-host-path" / "missing.log"
    source = LogSource(
        id = "server:missing",
        family = "server",
        label = missing.name,
        realpath = str(missing),
        size_bytes = 0,
        modified_at = 0,
        is_current = False,
    )

    output = build_debug_log_archive([source])
    try:
        with zipfile.ZipFile(output) as archive:
            warning = archive.read("EXPORT_WARNINGS.txt").decode("utf-8")
    finally:
        output.close()
    assert str(missing) not in warning
    assert "server/missing.log: FileNotFoundError (errno 2)" in warning


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
