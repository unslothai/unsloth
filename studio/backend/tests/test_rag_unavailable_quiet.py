# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A missing sqlite-vec binary must be reported once, not on every KB poll.

sqlite_vec imports fine while its native vec0 library is absent from the venv (a
real macOS condition), so RAG_AVAILABLE stays True and every
/api/rag/knowledge-bases poll used to raise, producing a 500 plus two full
tracebacks a few seconds apart for a condition that cannot change mid-session.
rag_db now warns once and raises RagExtensionUnavailable, and the list route
degrades to an empty list. Genuine database errors are untouched.
"""

import sqlite3

import pytest

from storage import rag_db


@pytest.fixture
def reset_unavailable_warning(monkeypatch):
    """Each test starts from an unwarned process."""
    monkeypatch.setattr(rag_db, "_unavailable_warned", False)


def _break_extension_load(monkeypatch, exc = None):
    """Make sqlite_vec.load fail the way a missing vec0 dylib does."""

    class _Stub:
        @staticmethod
        def load(conn):
            raise exc or sqlite3.OperationalError("dlopen(vec0.dylib): no such file")

    monkeypatch.setattr(rag_db, "sqlite_vec", _Stub)
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)


def test_unavailable_is_a_runtime_error_subclass():
    # Callers that already catch RuntimeError keep working.
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


def test_list_knowledge_bases_degrades_to_empty(rag_home, reset_unavailable_warning, monkeypatch):
    from routes import rag as rag_routes
    _break_extension_load(monkeypatch)
    assert rag_routes.list_knowledge_bases(subject = "tester") == {"knowledgeBases": []}


def test_list_knowledge_bases_still_raises_real_database_errors(
    rag_home, reset_unavailable_warning, monkeypatch
):
    # Only "the extension is not there" degrades. A locked or corrupt database is a
    # real failure and must keep surfacing.
    from routes import rag as rag_routes

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(rag_db, "get_connection", _boom)
    with pytest.raises(sqlite3.OperationalError, match = "database is locked"):
        rag_routes.list_knowledge_bases(subject = "tester")


def test_list_knowledge_bases_works_when_the_extension_loads(rag_home, rag_conn):
    # The healthy path is unchanged: a real connection still lists what is stored.
    from core.rag import store
    from routes import rag as rag_routes

    store.create_kb(rag_conn, name = "notes", description = None, embedding_model = None)
    rag_conn.commit()
    out = rag_routes.list_knowledge_bases(subject = "tester")
    assert [kb["name"] for kb in out["knowledgeBases"]] == ["notes"]
