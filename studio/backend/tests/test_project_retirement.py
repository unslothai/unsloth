# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core import project_retirement as lifecycle
from routes import chat_history
from storage import studio_db


@pytest.fixture(autouse = True)
def isolated_projects(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))


def _client():
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/history")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    return TestClient(app)


def _project():
    return studio_db.upsert_chat_project(
        {"id": "lifecycle", "name": "Lifecycle", "createdAt": 1, "updatedAt": 1}
    )


def test_no_feature_archive_and_delete_keep_existing_behavior():
    _project()
    with _client() as client:
        archived = client.patch("/history/projects/lifecycle", json = {"archived": True})
        assert archived.status_code == 200
        assert archived.json()["archived"] is True
        assert client.delete("/history/projects/lifecycle").status_code == 200
        assert client.delete("/history/projects/lifecycle").status_code == 404


@pytest.mark.parametrize("operation", ["archive", "delete"])
def test_failed_retirement_preserves_project_and_its_workspace(monkeypatch, operation):
    project = _project()

    def refuse(*args, **kwargs):
        raise RuntimeError("Recover owned state first")

    monkeypatch.setattr(
        lifecycle, "_feature", lambda _name: SimpleNamespace(begin_git_retirement = refuse)
    )
    with _client() as client:
        response = (
            client.patch("/history/projects/lifecycle", json = {"archived": True})
            if operation == "archive"
            else client.delete("/history/projects/lifecycle?delete_files=true")
        )
        assert response.status_code == 409
    assert studio_db.get_chat_project("lifecycle")["archived"] is False
    assert studio_db.get_chat_project("lifecycle")["sandboxPath"] == project["sandboxPath"]


def test_combined_features_use_only_git_retirement_owner(monkeypatch):
    events = []
    sentinel = object()

    def begin(project_id, *, deleting):
        events.append(("begin", project_id, deleting))
        return sentinel

    def finish(project_id, token):
        assert token is sentinel
        events.append(("finish", project_id))

    git = SimpleNamespace(begin_git_retirement = begin, finish_git_retirement = finish)

    def feature(name):
        assert name == "git_retirement", "Verification is retired by Git, never a second owner"
        return git

    monkeypatch.setattr(lifecycle, "_feature", feature)
    token = lifecycle.begin_project_retirement("p", deleting = True)
    lifecycle.finish_project_retirement("p", token)
    lifecycle.finish_project_retirement("p", token)
    assert events == [("begin", "p", True), ("finish", "p")]


def test_verification_cancellation_failure_releases_admission(monkeypatch):
    active = set()

    def cannot_stop(project_id):
        assert project_id in active
        raise RuntimeError("Still running")

    verification = SimpleNamespace(
        begin_project_deletion = active.add,
        cancel_project_verifications_and_wait = cannot_stop,
        finish_project_deletion = active.remove,
    )
    monkeypatch.setattr(
        lifecycle, "_feature", lambda name: verification if name == "verification" else None
    )
    with pytest.raises(RuntimeError, match = "Still running"):
        lifecycle.begin_project_retirement("p")
    assert not active


def test_unexpected_feature_import_failure_is_not_treated_as_absent(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError("broken dependency", name = "unavailable_dependency")

    monkeypatch.setattr(lifecycle.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        lifecycle.begin_project_retirement("p")
