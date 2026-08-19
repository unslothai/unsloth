# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sources-panel metadata and the guarantees the panel's remove action relies on.

The panel sorts by size and offers a bulk remove, so `_doc_view` has to carry
`sizeBytes`, and removal has to stay confined to Unsloth's own upload copies.
"""

import os

# Imported inside each test: `routes.rag` is reached through `routes/__init__`,
# which pulls in the whole app (providers, training, auth). Deferring keeps a
# missing optional dependency from erroring collection for the entire module,
# matching test_rag_preview.py.


def _wait(job_id, timeout = 30.0):
    import time

    from core.rag import ingestion

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = ingestion.get_job_status(job_id)
        if status and status["status"] in ("completed", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError("ingestion did not finish in time")


def test_doc_view_reports_size_for_a_project_source(rag_home, stub_embeddings):
    from core.rag import ingestion, store
    from routes.rag import _doc_view
    from storage import rag_db
    from utils.paths import rag_uploads_root

    # Ingest from the uploads root, as a real upload does: _save_upload copies the
    # browser's bytes there first, and that copy is what stored_path (and so the
    # reported size) refers to.
    uploads = rag_uploads_root()
    uploads.mkdir(parents = True, exist_ok = True)
    body = "alpha bravo charlie " * 50
    path = uploads / "notes.txt"
    path.write_text(body, encoding = "utf-8")
    expected = path.stat().st_size

    _, job_id = ingestion.start_ingestion(
        store.project_scope("P1"), None, None, "notes.txt", str(path), project_id = "P1"
    )
    assert _wait(job_id)["status"] == "completed"

    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, store.project_scope("P1"))
        view = _doc_view(docs[0])
    finally:
        conn.close()

    # Sorting by size needs a real number, not the None the endpoint used to omit.
    assert view["sizeBytes"] == expected


def test_size_is_none_when_the_stored_file_is_gone(tmp_path):
    from routes.rag import _stored_size

    missing = tmp_path / "never-written.txt"
    assert _stored_size(str(missing)) is None
    assert _stored_size(None) is None
    assert _stored_size("") is None


def test_size_reads_the_actual_byte_count(tmp_path):
    from routes.rag import _stored_size

    path = tmp_path / "sized.bin"
    path.write_bytes(b"x" * 1234)
    assert _stored_size(str(path)) == 1234


def test_remove_never_touches_a_file_outside_the_uploads_root(rag_home, tmp_path):
    """A source indexed in place (linked folders, native drops) must survive
    removal from a project: only Unsloth's managed copy is ever deleted."""
    from routes.rag import _remove_stored_upload

    outside = tmp_path / "my-documents" / "thesis.pdf"
    outside.parent.mkdir(parents = True)
    outside.write_bytes(b"important user data")

    _remove_stored_upload(str(outside))

    assert outside.exists(), "removal escaped the uploads root and deleted a user file"
    assert outside.read_bytes() == b"important user data"


def test_remove_deletes_only_unsloths_own_upload_copy(rag_home):
    from routes.rag import _remove_stored_upload
    from utils.paths import rag_uploads_root

    uploads = rag_uploads_root()
    uploads.mkdir(parents = True, exist_ok = True)
    managed = uploads / "copy.txt"
    managed.write_text("managed copy", encoding = "utf-8")

    _remove_stored_upload(str(managed))

    assert not managed.exists()


def test_remove_tolerates_a_path_that_is_already_gone(rag_home):
    from routes.rag import _remove_stored_upload
    from utils.paths import rag_uploads_root

    uploads = rag_uploads_root()
    uploads.mkdir(parents = True, exist_ok = True)
    # Deleting twice is reachable: a retry after a partially-applied bulk remove.
    ghost = uploads / "ghost.txt"
    ghost.write_text("x", encoding = "utf-8")
    _remove_stored_upload(str(ghost))
    _remove_stored_upload(str(ghost))
    assert not ghost.exists()


def test_symlink_into_the_uploads_root_cannot_redirect_the_delete(rag_home, tmp_path):
    """realpath resolution means a link planted in uploads/ still resolves to the
    user's file, which sits outside the root and must therefore be spared."""
    from routes.rag import _remove_stored_upload
    from utils.paths import rag_uploads_root

    uploads = rag_uploads_root()
    uploads.mkdir(parents = True, exist_ok = True)
    target = tmp_path / "outside.txt"
    target.write_text("user data", encoding = "utf-8")
    link = uploads / "link.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        # Unprivileged Windows cannot create symlinks; the realpath guard is
        # exercised by the outside-the-root test above.
        return

    _remove_stored_upload(str(link))

    assert target.exists(), "a symlink redirected the delete onto a user file"
