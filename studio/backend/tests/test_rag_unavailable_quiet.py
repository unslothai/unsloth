# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A missing sqlite-vec binary must be reported once, not on every KB poll.

sqlite_vec imports fine while its native vec0 library is absent from the venv (a
real macOS condition), so RAG_AVAILABLE stays True and every
/api/rag/knowledge-bases poll used to raise, producing a 500 plus two full
tracebacks a few seconds apart for a condition that cannot change mid-session.
rag_db now warns once and raises RagExtensionUnavailable.

The router answers that with one contract: the polled KB list degrades to an empty
list plus an availability marker, so the frontend can tell an empty store from a
machine where RAG cannot run, and every other endpoint answers a clean 503 carrying
the same reason, so clicking Create in that state gets a stated reason rather than a
500 and a fresh traceback. Genuine database errors are untouched.
"""

import sqlite3

import pytest
from fastapi import HTTPException

from storage import rag_db

UNAVAILABLE = "RAG is unavailable: the sqlite-vec extension could not be loaded."


@pytest.fixture
def reset_unavailable_warning(monkeypatch):
    """Each test starts from an unwarned process that has not yet loaded the library."""
    monkeypatch.setattr(rag_db, "_unavailable_warned", False)
    monkeypatch.setattr(rag_db, "_extension_loaded", False)


def _break_extension_load(monkeypatch, exc = None):
    """Make sqlite_vec.load fail the way a missing vec0 dylib does."""

    class _Stub:
        @staticmethod
        def load(conn):
            raise exc or sqlite3.OperationalError("dlopen(vec0.dylib): no such file")

    monkeypatch.setattr(rag_db, "sqlite_vec", _Stub)
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)


def _client():
    """The RAG router alone behind TestClient, with the subject dependency stubbed."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import rag as rag_routes

    app = FastAPI()
    app.include_router(rag_routes.router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def test_unavailable_is_a_runtime_error_subclass():
    assert issubclass(rag_db.RagExtensionUnavailable, RuntimeError)


def test_extension_load_failure_raises_the_typed_error(
    rag_home, reset_unavailable_warning, monkeypatch
):
    _break_extension_load(monkeypatch)
    with pytest.raises(rag_db.RagExtensionUnavailable):
        rag_db.get_connection()


def test_import_failure_also_raises_the_typed_error(reset_unavailable_warning, monkeypatch):
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", False)
    with pytest.raises(rag_db.RagExtensionUnavailable):
        rag_db.get_connection()


def test_unavailability_warns_only_once(rag_home, reset_unavailable_warning, monkeypatch, caplog):
    _break_extension_load(monkeypatch)
    with caplog.at_level("WARNING", logger = rag_db.__name__):
        for _ in range(5):
            with pytest.raises(rag_db.RagExtensionUnavailable):
                rag_db.get_connection()
    warnings = [
        r for r in caplog.records if "sqlite-vec extension could not be loaded" in r.message
    ]
    assert len(warnings) == 1
    assert "disabled for this session" in warnings[0].getMessage()


def test_rag_available_is_false_when_the_library_will_not_load(
    rag_home, reset_unavailable_warning, monkeypatch
):
    _break_extension_load(monkeypatch)
    assert rag_db.rag_available() is False


def test_rag_available_propagates_real_database_errors(
    rag_home, reset_unavailable_warning, monkeypatch
):
    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(rag_db, "get_connection", _boom)
    with pytest.raises(sqlite3.OperationalError, match = "database is locked"):
        rag_db.rag_available()


def test_list_knowledge_bases_degrades_to_empty(rag_home, reset_unavailable_warning, monkeypatch):
    from routes import rag as rag_routes

    _break_extension_load(monkeypatch)
    out = rag_routes.list_knowledge_bases(subject = "tester")
    assert out["knowledgeBases"] == []
    # The marker is what stops the frontend reading this as a working empty state.
    assert out["ragAvailable"] is False
    assert out["ragUnavailableReason"] == UNAVAILABLE


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda r: r.create_knowledge_base(r.CreateKbRequest(name = "notes"), subject = "tester"),
            id = "create-kb",
        ),
        pytest.param(
            lambda r: r.update_knowledge_base(
                "kb1", r.UpdateKbRequest(name = "renamed"), subject = "tester"
            ),
            id = "update-kb",
        ),
        pytest.param(
            lambda r: r.delete_knowledge_base("kb1", subject = "tester"),
            id = "delete-kb",
        ),
        pytest.param(
            lambda r: r.list_kb_documents("kb1", subject = "tester"),
            id = "list-kb-documents",
        ),
        pytest.param(
            lambda r: r.list_thread_documents("t1", subject = "tester"),
            id = "list-thread-documents",
        ),
        pytest.param(
            lambda r: r.list_project_documents("p1", subject = "tester"),
            id = "list-project-documents",
        ),
        pytest.param(
            lambda r: r.list_all_uploaded_documents(subject = "tester"),
            id = "list-all-documents",
        ),
        pytest.param(
            lambda r: r.delete_document("doc1", subject = "tester"),
            id = "delete-document",
        ),
        pytest.param(
            lambda r: r.search(r.SearchRequest(query = "hello", kb_id = "kb1"), subject = "tester"),
            id = "search",
        ),
        pytest.param(
            lambda r: r.job_status("job1", subject = "tester"),
            id = "job-status",
        ),
        pytest.param(
            lambda r: r.preview_target("doc1", subject = "tester"),
            id = "preview-target",
        ),
        pytest.param(
            lambda r: r.document_file_url("doc1", subject = "tester"),
            id = "file-url",
        ),
    ],
)
def test_every_other_endpoint_answers_503_not_a_traceback(
    rag_home, reset_unavailable_warning, monkeypatch, call
):
    # Create states the reason instead of a 500, and listings degrade so a marker-blind frontend keeps today's surfaces.
    from routes import rag as rag_routes

    _break_extension_load(monkeypatch)
    with pytest.raises(HTTPException) as err:
        call(rag_routes)
    assert err.value.status_code == 503
    assert err.value.detail == UNAVAILABLE


def test_upload_is_refused_before_the_file_is_written(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # The gate runs before _resolve_document_upload so no orphan uploads; over HTTP, as the Form default is truthy.
    from utils.paths import rag_uploads_root

    _break_extension_load(monkeypatch)
    with _client() as client:
        response = client.post(
            "/api/rag/threads/t1/documents",
            files = {"file": ("notes.txt", b"alpha bravo", "text/plain")},
        )
    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
    uploads = rag_uploads_root()
    assert not uploads.exists() or list(uploads.iterdir()) == []


def test_a_saved_upload_is_removed_when_ingestion_cannot_start(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # The window the gate cannot cover: the upload is admitted from a remembered verdict, so start_ingestion must 503.
    from utils.paths import rag_uploads_root

    _break_extension_load(monkeypatch)
    monkeypatch.setattr(rag_db, "_extension_loaded", True)
    with _client() as client:
        response = client.post(
            "/api/rag/threads/t1/documents",
            files = {"file": ("notes.txt", b"alpha bravo", "text/plain")},
        )
    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
    assert list(rag_uploads_root().iterdir()) == []


def test_a_first_request_that_discovers_the_missing_library_still_gets_503(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # Same window on the connection path: _rag_connection() is what reports the 503.
    from routes import rag as rag_routes

    _break_extension_load(monkeypatch)
    monkeypatch.setattr(rag_db, "_extension_loaded", True)
    assert rag_db.rag_available() is True
    with pytest.raises(HTTPException) as err:
        rag_routes.create_knowledge_base(rag_routes.CreateKbRequest(name = "notes"), subject = "tester")
    assert err.value.status_code == 503
    assert err.value.detail == UNAVAILABLE


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "this interpreter's sqlite3 is built without extension loading, so the "
    "healthy path cannot be exercised (python.org macOS builds, some distros)",
)
def test_a_failed_load_is_not_latched_for_the_session(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # Only the positive verdict is remembered: a one-off load failure must not switch RAG off until restart.
    real = rag_db.sqlite_vec
    calls = {"n": 0}

    class _FailsOnce:
        @staticmethod
        def load(conn):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("dlopen(vec0.dylib): no such file")
            real.load(conn)

    monkeypatch.setattr(rag_db, "sqlite_vec", _FailsOnce)
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)
    assert rag_db.rag_available() is False
    assert rag_db.rag_available() is True


def test_the_whole_router_stays_quiet_under_repeated_use(
    rag_home, reset_unavailable_warning, monkeypatch, caplog
):
    # A 503 path that logged per request would reinstate the spam this exists to remove.
    from routes import rag as rag_routes

    _break_extension_load(monkeypatch)
    with caplog.at_level("WARNING", logger = rag_db.__name__):
        for _ in range(5):
            rag_routes.list_knowledge_bases(subject = "tester")
            with pytest.raises(HTTPException):
                rag_routes.create_knowledge_base(
                    rag_routes.CreateKbRequest(name = "notes"), subject = "tester"
                )
    records = [r for r in caplog.records if r.name == rag_db.__name__]
    assert len(records) == 1
    assert "sqlite-vec extension could not be loaded" in records[0].message
    assert not any(r.exc_info for r in records)


def test_startup_reconcile_is_a_no_op_when_rag_cannot_run(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # Raising out of startup makes main.py log "reconcile failed" when there is nothing to reconcile.
    _break_extension_load(monkeypatch)
    assert rag_db.reconcile_orphaned_ingestion_jobs() == 0


def test_over_http_the_poll_is_200_and_create_is_503(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # What the browser sees: a success list carrying the marker, and a 503 whose `detail` parseErrorText surfaces.
    _break_extension_load(monkeypatch)
    with _client() as client:
        listed = client.get("/api/rag/knowledge-bases")
        assert listed.status_code == 200
        assert listed.json() == {
            "knowledgeBases": [],
            "ragAvailable": False,
            "ragUnavailableReason": UNAVAILABLE,
        }

        created = client.post("/api/rag/knowledge-bases", json = {"name": "notes"})
        assert created.status_code == 503
        assert created.json() == {"detail": UNAVAILABLE}


def test_list_knowledge_bases_still_raises_real_database_errors(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # Only a missing extension degrades; a locked or corrupt database must keep surfacing.
    from routes import rag as rag_routes

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(rag_db, "get_connection", _boom)
    with pytest.raises(sqlite3.OperationalError, match = "database is locked"):
        rag_routes.list_knowledge_bases(subject = "tester")


def test_mutating_endpoints_still_raise_real_database_errors(
    rag_home, reset_unavailable_warning, monkeypatch
):
    from routes import rag as rag_routes

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(rag_db, "_extension_loaded", True)
    monkeypatch.setattr(rag_db, "get_connection", _boom)
    with pytest.raises(sqlite3.OperationalError, match = "database is locked"):
        rag_routes.create_knowledge_base(rag_routes.CreateKbRequest(name = "notes"), subject = "tester")


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "this interpreter's sqlite3 is built without extension loading, so the "
    "healthy path cannot be exercised (python.org macOS builds, some distros)",
)
def test_list_knowledge_bases_works_when_the_extension_loads(rag_home, rag_conn):
    from core.rag import store
    from routes import rag as rag_routes

    store.create_kb(rag_conn, name = "notes", description = None, embedding_model = None)
    rag_conn.commit()
    out = rag_routes.list_knowledge_bases(subject = "tester")
    assert [kb["name"] for kb in out["knowledgeBases"]] == ["notes"]
    assert out["ragAvailable"] is True
    assert out["ragUnavailableReason"] is None


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "this interpreter's sqlite3 is built without extension loading, so the "
    "healthy path cannot be exercised (python.org macOS builds, some distros)",
)
def test_mutations_still_work_when_the_extension_loads(rag_home, rag_conn):
    from routes import rag as rag_routes

    created = rag_routes.create_knowledge_base(
        rag_routes.CreateKbRequest(name = "notes"), subject = "tester"
    )
    assert created["name"] == "notes"
    assert rag_routes.update_knowledge_base(
        created["id"], rag_routes.UpdateKbRequest(name = "renamed"), subject = "tester"
    ) == {"ok": True}
    assert rag_routes.delete_knowledge_base(created["id"], subject = "tester") == {"ok": True}
