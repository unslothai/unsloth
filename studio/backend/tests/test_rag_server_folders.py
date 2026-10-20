# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core.rag import folder_sync, server_folders
from utils.account_context import AccountContext, bind_account, reset_account


@pytest.fixture
def client(monkeypatch):
    from routes import rag

    app = FastAPI()
    app.include_router(rag.router, prefix = "/api/rag")
    app.dependency_overrides[rag.get_current_subject] = lambda: "owner"
    app.dependency_overrides[rag.authenticated_via_api_key] = lambda: False
    return TestClient(app), app


@pytest.fixture
def source(rag_home, monkeypatch):
    folder = rag_home / "documents"
    folder.mkdir()
    monkeypatch.setenv(server_folders.ENV_NAME, json.dumps([str(folder)]))
    return folder


@pytest.mark.parametrize("setting", [None, "", "invalid", "{}", '["relative/path"]', "[12]"])
def test_server_folders_fail_closed(setting, monkeypatch):
    if setting is None:
        monkeypatch.delenv(server_folders.ENV_NAME, raising = False)
    else:
        monkeypatch.setenv(server_folders.ENV_NAME, setting)
    assert server_folders.available_folders() == []


def test_bad_setting_and_missing_folder_warn_once(rag_home, monkeypatch, caplog):
    monkeypatch.setattr(server_folders, "_warned", set())
    monkeypatch.setenv(server_folders.ENV_NAME, "invalid")
    for _ in range(3):
        server_folders.available_folders()
    monkeypatch.setenv(server_folders.ENV_NAME, json.dumps([str(rag_home / "missing")]))
    for _ in range(3):
        server_folders.available_folders()
    warnings = [r.getMessage() for r in caplog.records if r.name == server_folders.logger.name]
    assert len(warnings) == 2
    assert "missing" in warnings[1]


def test_an_undecodable_path_does_not_break_the_listing(source, monkeypatch):
    monkeypatch.setenv(
        server_folders.ENV_NAME, json.dumps([str(source.parent / "\udcff"), str(source)])
    )
    assert [f["displayName"] for f in server_folders.available_folders()] == ["documents"]


def test_only_configured_folder_ids_resolve(source):
    choices = server_folders.available_folders()
    assert len(choices) == 1
    assert choices[0]["displayName"] == "documents"
    assert str(source) not in json.dumps(choices)
    path, identity = server_folders.resolve_folder(choices[0]["id"])
    assert Path(path) == source.resolve()
    assert identity == folder_sync._root_identity(str(source.resolve()))
    for unconfigured in [str(source), "../documents", "unknown"]:
        with pytest.raises(ValueError, match = "not configured"):
            server_folders.resolve_folder(unconfigured)


def test_same_named_folders_are_told_apart(rag_home, monkeypatch):
    paths = [rag_home / "team-a" / "docs", rag_home / "team-b" / "docs", rag_home / "notes"]
    for path in paths:
        path.mkdir(parents = True)
    spellings = [str(p) for p in paths] + [str(paths[0]) + "/"]
    monkeypatch.setenv(server_folders.ENV_NAME, json.dumps(spellings))
    names = [f["displayName"] for f in server_folders.available_folders()]
    assert names == [str(Path("team-a", "docs")), str(Path("team-b", "docs")), "notes"]


def test_a_suffix_of_another_folder_is_not_labelled_by_its_absolute_path(monkeypatch):
    from pathlib import PurePosixPath
    monkeypatch.setattr(server_folders, "Path", PurePosixPath)
    assert server_folders._labels(["/srv/docs", "/archive/srv/docs"]) == [
        "srv/docs",
        "archive/srv/docs",
    ]


def test_same_named_folders_on_different_windows_drives_are_told_apart(monkeypatch):
    from pathlib import PureWindowsPath

    monkeypatch.setattr(server_folders, "Path", PureWindowsPath)
    paths = [r"C:\docs", r"D:\docs", r"\\nas\team\docs", r"C:\notes"]
    assert server_folders._labels(paths) == [
        "docs (C:)",
        "docs (D:)",
        r"docs (\\nas\team)",
        "notes",
    ]


def test_removed_configuration_invalidates_selection(source, monkeypatch):
    choice = server_folders.available_folders()[0]
    monkeypatch.delenv(server_folders.ENV_NAME)
    with pytest.raises(ValueError, match = "not configured"):
        server_folders.resolve_folder(choice["id"])


def test_folder_disappearing_after_listing_is_rejected(source):
    choice = server_folders.available_folders()[0]
    source.rmdir()
    assert server_folders.available_folders() == []
    with pytest.raises(ValueError):
        server_folders.resolve_folder(choice["id"])


def test_configured_symlink_is_not_a_source(source, tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(source, target_is_directory = True)
    except OSError:
        pytest.skip(reason = "directory symlinks need admin or Developer Mode on Windows")
    monkeypatch.setenv(server_folders.ENV_NAME, json.dumps([str(alias)]))
    assert server_folders.available_folders() == []


@pytest.mark.parametrize("kind", ["file", "sensitive", "root"])
def test_normal_folder_restrictions_apply(source, tmp_path, monkeypatch, kind):
    if kind == "file":
        path = source / "file.txt"
        path.write_text("hello")
    elif kind == "sensitive":
        path = tmp_path / ".ssh"
        path.mkdir()
    else:
        path = Path(source.anchor)
    monkeypatch.setenv(server_folders.ENV_NAME, json.dumps([str(path)]))
    assert server_folders.available_folders() == []


def test_api_keys_and_managed_accounts_cannot_link_server_folders():
    from routes import rag

    with pytest.raises(HTTPException) as error:
        rag._require_server_folder_session(True)
    assert error.value.status_code == 403
    token = bind_account(AccountContext("guest", "guest", "user"))
    try:
        with pytest.raises(HTTPException) as error:
            rag._require_server_folder_session(False)
        assert error.value.status_code == 403
    finally:
        reset_account(token)


def test_native_link_request_still_requires_signed_grant():
    from routes import rag
    with pytest.raises(ValidationError):
        rag.LinkFolderRequest(serverFolderId = "anything", path = "/documents")


@pytest.mark.parametrize("scope_type", ["project", "knowledge_base"])
def test_server_link_reuses_indexing_with_resolved_identity(source, monkeypatch, scope_type):
    from routes import rag

    choice = server_folders.available_folders()[0]
    monkeypatch.setattr(rag, "_require_rag", lambda: None)
    scopes = []
    monkeypatch.setattr(rag, "_require_scope_owner", lambda *args: scopes.append(args))
    calls = []
    monkeypatch.setattr(
        rag, "_create_linked_folder_at_path", lambda *args: calls.append(args) or {"ok": True}
    )
    payload = rag.ServerLinkFolderRequest(serverFolderId = choice["id"])
    assert rag._link_server_source_folder(scope_type, "scope", payload, False) == {"ok": True}
    assert scopes == [(scope_type, "scope")]
    assert calls[0][:2] == (scope_type, "scope")
    assert calls[0][2].server_folder_id == choice["id"]
    assert calls[0][2].name == choice["displayName"]
    assert calls[0][3:] == server_folders.resolve_folder(choice["id"])


def test_http_link_creates_a_real_folder_and_sync_job(source, rag_home, client):
    from core.rag import store
    from storage import rag_db

    if not rag_db.rag_available():
        pytest.skip(
            reason = "sqlite-vec cannot load here (a Python built without SQLite extension loading)"
        )
    http, _ = client
    with closing(rag_db.get_connection()) as conn:
        store.create_kb(conn, name = "Knowledge", kb_id = "knowledge")
    choices = http.get("/api/rag/server-source-folders")
    assert choices.status_code == 200
    folder_id = choices.json()["folders"][0]["id"]
    response = http.post(
        "/api/rag/knowledge-bases/knowledge/server-linked-folders",
        json = {"serverFolderId": folder_id},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    folder = folder_sync.get_folder(result["linkedFolder"]["id"])
    assert Path(folder["path"]) == source.resolve()
    assert folder["scope"] == store.kb_scope("knowledge")
    assert result["job"]["linkedFolderId"] == folder["id"]
    assert result["job"]["status"] == "pending"
    assert "path" not in result["linkedFolder"]


def test_same_named_folders_stay_apart_once_linked(rag_home, client, monkeypatch):
    from core.rag import store
    from storage import rag_db

    if not rag_db.rag_available():
        pytest.skip(
            reason = "sqlite-vec cannot load here (a Python built without SQLite extension loading)"
        )
    paths = [rag_home / "team-a" / "docs", rag_home / "team-b" / "docs"]
    for path in paths:
        path.mkdir(parents = True)
    monkeypatch.setenv(server_folders.ENV_NAME, json.dumps([str(p) for p in paths]))
    http, _ = client
    with closing(rag_db.get_connection()) as conn:
        store.create_kb(conn, name = "Knowledge", kb_id = "knowledge")
    names = []
    for choice in http.get("/api/rag/server-source-folders").json()["folders"]:
        response = http.post(
            "/api/rag/knowledge-bases/knowledge/server-linked-folders",
            json = {"serverFolderId": choice["id"]},
        )
        assert response.status_code == 200, response.text
        names.append(response.json()["linkedFolder"]["displayName"])
    assert names == [str(Path("team-a", "docs")), str(Path("team-b", "docs"))]


@pytest.mark.parametrize("api_key,managed", [(True, False), (False, True)])
def test_http_routes_enforce_owner_session_before_filesystem_access(
    client, monkeypatch, api_key, managed
):
    from routes import rag

    http, app = client
    app.dependency_overrides[rag.authenticated_via_api_key] = lambda: api_key
    monkeypatch.setattr(
        server_folders, "configured_folders", lambda: pytest.fail("filesystem policy reached")
    )
    token = bind_account(AccountContext("guest", "guest", "user")) if managed else None
    try:
        assert http.get("/api/rag/server-source-folders").status_code == 403
        for scope in ["projects", "knowledge-bases"]:
            response = http.post(
                f"/api/rag/{scope}/scope/server-linked-folders",
                json = {"serverFolderId": "unknown"},
            )
            assert response.status_code == 403
    finally:
        if token is not None:
            reset_account(token)
