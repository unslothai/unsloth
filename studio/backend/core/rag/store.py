# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unified SQLite store: relational chunks + FTS5 lexical + sqlite-vec dense.

Module-level functions each take a ``conn`` the caller opens and closes. Inserts
are incremental: ``add_chunks`` appends one document's rows without rebuilding the
scope. Scope ("kb_<id>" / "thread_<id>") is a column on every table and the vec0
partition key.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import uuid
from datetime import datetime, timezone

from storage import rag_db

from . import config


def kb_scope(kb_id: str) -> str:
    return f"kb_{kb_id}"


def thread_scope(thread_id: str) -> str:
    return f"thread_{thread_id}"


def project_scope(project_id: str) -> str:
    return f"project_{project_id}"


CONVERSATION_ARCHIVE_PREFIX = "convarchive_"


def conversation_archive_scope(thread_id: str) -> str:
    """Scope holding the turns a thread's rolling context window has evicted.

    Deliberately NOT ``thread_scope``: with ``config.THREAD_WHOLE_DOC`` on, that scope is
    rendered in full into every request, so archiving turns there would re-inject the
    history and undo the compaction. A separate scope also keeps the archive out of the
    attachments UI and the citation panel.
    """
    return f"{CONVERSATION_ARCHIVE_PREFIX}{thread_id}"


def _scopes(scope) -> list[str]:
    """Search helpers accept one scope or several (e.g. project + thread)."""
    return [scope] if isinstance(scope, str) else list(scope)


def _f32(vector) -> bytes:
    """Pack a vector into float32 bytes for vec0."""
    return struct.pack(f"{len(vector)}f", *(float(x) for x in vector))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_TOKEN = re.compile(r"\w+", re.UNICODE)
# Quotes mark a word being named rather than used; non-greedy and single-line so an unclosed quote spans nothing.
_QUOTED = re.compile(r"\"([^\"\n]+)\"|\u201c([^\u201d\n]+)\u201d|'([^'\n]+)'|`([^`\n]+)`")


def _match_query(query: str) -> str:
    """User text -> safe FTS5 OR-of-quoted-terms query; quoting defuses FTS5
    operators. "" (no tokens) means no lexical results."""
    toks = _TOKEN.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in toks)


# A closed list of function words, so behaviour is identical on every install. no and not are
# deliberately NOT here: they carry the whole difference in "what did I say not to delete?",
# where dropping them leaves only terms BM25 floors at 1e-6.
_ARCHIVE_STOPWORDS = frozenset(
    """
a about all am an and any are as at be been being but by can could did do does doing
for from get give had has have how i if in into is it its just let me my now of
on or please should so tell that the their them then there these they this those to us
was we were what when where which who why will with would you your
""".split()
)

# Identifier-ish tokens are how a person names one specific thing; a digit alone counts, since a
# purely numeric subject has no other shape.
_HAS_DIGIT = re.compile(r"\d", re.UNICODE)
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def _is_identifier(token: str, raw_tokens: frozenset[str]) -> bool:
    """``raw_tokens`` is the query's tokens BEFORE lower-casing, tokenized once.

    Once, and as a set, because the caller runs this per distinct token: re-scanning the
    query text inside the loop made the whole function quadratic in the question's
    length, which a pasted log turns into a multi-second stall on the request that
    compacts the thread (48 KB of pasted text measured at 4.6s, 96 KB at 17.7s, against
    2.3ms for the same text through `_match_query`).

    The capitals rule needs CONTRAST, not just capitals: in a line with no lower case
    anywhere every word satisfies it and the filter stops filtering. The caller passes an
    empty ``raw_tokens`` for such a line, so shape alone decides there.
    """
    if "_" in token:
        return True
    if _HAS_DIGIT.search(token):
        # A bare number needs LENGTH to be a name, else "answer in 2 sentences" filters the archive on "2".
        return bool(_HAS_LETTER.search(token)) or len(token) >= 3
    return len(token) >= 3 and token.upper() in raw_tokens


def conversation_match_queries(query: str) -> list[str]:
    """FTS5 expressions for searching a CONVERSATION ARCHIVE, most selective first.

    Why the archive needs its own query shaping, when `_match_query` is fine everywhere
    else: in a per-thread archive the SUBJECT of the conversation is by construction
    present in many chunks, so BM25 gives it almost no weight, while an incidental word
    from the question appears once and dominates. Measured on an archive of 17 chunks
    about one variable: `zqxvara123` scored 0.16 and `value`, from "what is the current
    value of X", scored 4.755. ORing them lets the filler decide the ranking, and a chunk
    about "a good default value for a retry budget" outranks every chunk that names the
    variable. The subject of a long conversation becomes the least discriminative term in
    its own archive.

    So: first REQUIRE the identifier-like tokens, which restricts the candidates to
    chunks that are actually about the thing asked about; then fall back to an OR over
    the content words. Two expressions rather than one, because a filter that matches
    nothing must not mean "this archive has nothing to say".

    A question made entirely of function words ("what about it?") keeps all its tokens:
    an empty expression would make `search_lexical` return nothing at all, and a query
    that retrieves the wrong turns is still better than a recall that silently vanishes
    on exactly the turns that needed it.

    SEVERAL identifiers are ORed, not ANDed. "What are the current values of A123 and
    B456" is two questions in one envelope, and the turn answering either one names one
    of them: requiring both keeps only the turns that DISCUSS the pair, which are exactly
    the older comparisons, and drops both current assignments. Measured on an archive of
    six comparison turns plus one latest assignment each: the conjunction returned the
    four oldest comparisons and neither value, where the permissive pass returns both.
    The filter's job is to keep every slot on something the question asked about, and one
    identifier out of two is still that; the content-word pass still does the ranking,
    and a chunk naming both still outranks a chunk naming one, because it matches more.
    """
    tokens = list(dict.fromkeys(_TOKEN.findall(query.lower())))
    if not tokens:
        return []
    # Identifier-ish: a token containing a digit (ZQXVARA123, 9134) or an underscore, or one in
    # capitals and long enough not to be an "I" or an "OK". The capitals rule needs CONTRAST: in an
    # all-caps line every word passes it and the filter filters nothing.
    raw_tokens = frozenset() if query == query.upper() else frozenset(_TOKEN.findall(query))
    identifiers = [t for t in tokens if _is_identifier(t, raw_tokens)]
    # A QUOTED word is the subject whatever the stopword list thinks; quoted tokens stay out of
    # identifiers, so this widens only the permissive pass.
    quoted = frozenset(
        token
        for match in _QUOTED.findall(query.lower())
        for token in _TOKEN.findall("".join(match))
    )
    content = [t for t in tokens if t not in _ARCHIVE_STOPWORDS or t in quoted] or tokens
    permissive = " OR ".join(f'"{t}"' for t in content)
    if not identifiers:
        return [permissive]
    focused = " OR ".join(f'"{t}"' for t in identifiers)
    return [focused] if focused == permissive else [focused, permissive]


def lexical_matching_ids(conn: sqlite3.Connection, chunk_ids, expression: str) -> set:
    """Which of ``chunk_ids`` match ``expression``, by the index's own tokenizer.

    Membership, not ranking, and therefore not subject to any top-k window. A ranked pass
    truncated at k answers "is this chunk among the k the index happened to return",
    which is a different question and the wrong one when the scores are tied: FTS5 floors
    the BM25 IDF of a term present in more than half the index at 1e-6, so the identifier
    a whole thread is about orders nothing and the k that come back are arbitrary. Asking
    the index directly, restricted to candidates already in hand, is exact however long
    the thread gets.
    """
    ids = list(dict.fromkeys(chunk_ids))
    if not ids or not expression:
        return set()
    found: set = set()
    # Chunked to stay under SQLITE_MAX_VARIABLE_NUMBER, which is 999 on older builds.
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            f"AND chunk_id IN ({placeholders})",
            [expression, *batch],
        ).fetchall()
        found.update(row[0] for row in rows)
    return found


def create_kb(
    conn: sqlite3.Connection,
    *,
    name: str,
    description: str | None = None,
    embedding_model: str | None = None,
    kb_id: str | None = None,
) -> str:
    kb_id = kb_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO knowledge_bases(id, name, description, embedding_model, created_at) "
        "VALUES(?,?,?,?,?)",
        (kb_id, name, description, embedding_model, _now()),
    )
    conn.commit()
    return kb_id


def list_kbs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def get_kb(conn: sqlite3.Connection, kb_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
    return dict(row) if row else None


def delete_kb(
    conn: sqlite3.Connection,
    kb_id: str,
    *,
    commit: bool = True,
    delete_documents: bool = True,
) -> None:
    """Delete a knowledge base, optionally retaining documents for durable cleanup."""
    try:
        if commit:
            conn.execute("BEGIN IMMEDIATE")
        scope = kb_scope(kb_id)
        if delete_documents:
            doc_ids = [
                r["id"]
                for r in conn.execute("SELECT id FROM documents WHERE scope=?", (scope,)).fetchall()
            ]
            for doc_id in doc_ids:
                delete_document(conn, doc_id, commit = False)
        conn.execute("DELETE FROM knowledge_bases WHERE id=?", (kb_id,))
        if commit:
            conn.commit()
    except Exception:
        if commit:
            conn.rollback()
        raise


def create_document(
    conn: sqlite3.Connection,
    *,
    scope: str,
    filename: str,
    sha256: str,
    kb_id: str | None = None,
    thread_id: str | None = None,
    project_id: str | None = None,
    status: str = "pending",
    stored_path: str | None = None,
    document_id: str | None = None,
    embedding_model: str | None = None,
    linked_folder_id: str | None = None,
    linked_relative_path: str | None = None,
    archive_messages: int | None = None,
    archive_ordinal: int | None = None,
    created_at: str | None = None,
    commit: bool = True,
) -> str:
    """``created_at`` is for a REWRITE of a row that already exists, and nothing else.

    A re-embed deletes the old row and inserts a new one for the same content, so stamping
    it with the current time would say the turn was archived when its vectors were
    rebuilt. That is not a cosmetic difference for an archived turn: an archive written
    before `archive_ordinal` existed is ordered by `created_at` alone, so a rewrite that
    takes a fresh timestamp moves that turn to the end of its own conversation. Omitted,
    this is byte for byte what every other caller has always got.
    """
    document_id = document_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO documents(id, scope, kb_id, thread_id, project_id, filename, sha256, "
        "status, stored_path, created_at, embedding_model, linked_folder_id, "
        "linked_relative_path, archive_messages, archive_ordinal) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            document_id,
            scope,
            kb_id,
            thread_id,
            project_id,
            filename,
            sha256,
            status,
            stored_path,
            created_at or _now(),
            embedding_model,
            linked_folder_id,
            linked_relative_path,
            archive_messages,
            archive_ordinal,
        ),
    )
    if commit:
        conn.commit()
    return document_id


def set_document_status(
    conn: sqlite3.Connection,
    document_id: str,
    status: str,
    *,
    num_chunks: int | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE documents SET status=?, num_chunks=COALESCE(?, num_chunks), error=? WHERE id=?",
        (status, num_chunks, error, document_id),
    )
    conn.commit()


def set_document_embedding_model(
    conn: sqlite3.Connection, document_id: str, embedding_model: str
) -> None:
    """Record which embedder actually produced this document's vectors. Written after
    the encode, because the process can swap backends part way through a job."""
    conn.execute(
        "UPDATE documents SET embedding_model=? WHERE id=?", (embedding_model, document_id)
    )
    conn.commit()


def list_documents(conn: sqlite3.Connection, scope: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, scope, kb_id, thread_id, project_id, filename, sha256, status, error, "
        "num_chunks, created_at, linked_folder_id "
        "FROM documents d WHERE scope=? AND NOT EXISTS "
        "(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        "ORDER BY created_at DESC",
        (scope,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_all_documents(conn: sqlite3.Connection) -> list[dict]:
    """Every uploaded document across all scopes (KBs, threads, projects).

    Archived conversation turns are excluded: nobody uploaded them, so listing them would
    show a chat's own history back as files the user never added.
    """
    rows = conn.execute(
        "SELECT id, scope, kb_id, thread_id, project_id, filename, sha256, status, error, "
        "num_chunks, stored_path, created_at, linked_folder_id "
        "FROM documents d WHERE NOT EXISTS "
        "(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        "AND d.scope NOT LIKE 'convarchive#_%' ESCAPE '#' "
        "ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def next_archive_ordinal(conn: sqlite3.Connection, scope: str) -> int:
    """The next conversation position for an archived turn group in this scope.

    Deliberately not derived from `created_at`: every turn a single compaction evicts is
    written microseconds apart, so wall-clock separates compaction EPOCHS and says
    nothing about order WITHIN one. This counter does, because `archive_turns` allocates
    it in `group_turns` order.
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(archive_ordinal), -1) + 1 AS n FROM documents WHERE scope=?",
        (scope,),
    ).fetchone()
    return int(row["n"]) if row else 0


def get_document(conn: sqlite3.Connection, document_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    return dict(row) if row else None


def get_visible_document(conn: sqlite3.Connection, document_id: str) -> dict | None:
    """Return a document only while its owning scope is available to readers."""
    row = conn.execute(
        "SELECT d.* FROM documents d WHERE d.id=? AND NOT EXISTS "
        "(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope)",
        (document_id,),
    ).fetchone()
    return dict(row) if row else None


def document_by_hash(conn: sqlite3.Connection, scope: str, sha256: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM documents WHERE scope=? AND sha256=? AND status!='failed' "
        "AND linked_folder_id IS NULL "
        "ORDER BY created_at DESC LIMIT 1",
        (scope, sha256),
    ).fetchone()
    return row["id"] if row else None


def documents_by_hash(conn: sqlite3.Connection, scope: str, sha256: str) -> list[dict]:
    """Every live copy of this text in the scope, oldest first.

    The archive can legitimately hold more than one: a user who says the same thing twice
    in one conversation said it twice, and the second time is often the one that matters.
    Ordered so the nth copy lines up with the nth occurrence in the transcript.
    """
    rows = conn.execute(
        "SELECT id, archive_ordinal, embedding_model, created_at FROM documents "
        "WHERE scope=? AND sha256=? AND status!='failed' AND linked_folder_id IS NULL "
        "ORDER BY COALESCE(archive_ordinal, -1), created_at",
        (scope, sha256),
    ).fetchall()
    return [dict(row) for row in rows]


def set_archive_ordinal(conn: sqlite3.Connection, document_id: str, ordinal: int) -> None:
    """Re-stamp one document's position. Used to migrate rows numbered by archive time."""
    conn.execute("UPDATE documents SET archive_ordinal=? WHERE id=?", (int(ordinal), document_id))


def failed_documents_by_hash(conn: sqlite3.Connection, scope: str, sha256: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, stored_path FROM documents WHERE scope=? AND sha256=? AND status='failed' "
        "AND linked_folder_id IS NULL",
        (scope, sha256),
    ).fetchall()
    return [dict(r) for r in rows]


def add_chunks(
    conn: sqlite3.Connection,
    scope: str,
    document_id: str,
    chunks,
    vectors,
    regions = None,
) -> None:
    """Incrementally index one document's chunks into chunks + FTS5 + vec0.
    ``vectors`` parallels ``chunks``; optional ``regions`` (also parallel) holds
    per-chunk PDF highlight rects, stored as JSON."""
    if len(vectors):
        rag_db.ensure_vec(conn, len(vectors[0]))
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{document_id}:{chunk.chunk_index}"
        chunk_regions = regions[i] if regions and i < len(regions) else None
        regions_json = json.dumps(chunk_regions) if chunk_regions else None
        conn.execute(
            "INSERT OR REPLACE INTO chunks("
            "id, document_id, scope, chunk_index, text, page_number, "
            "source_page_index, token_count, kind, pdf_regions_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                chunk_id,
                document_id,
                scope,
                chunk.chunk_index,
                chunk.text,
                chunk.page_number,
                chunk.source_page_index,
                chunk.token_count,
                getattr(chunk, "kind", "text"),
                regions_json,
            ),
        )
        conn.execute(
            "INSERT INTO chunks_fts(text, chunk_id, scope) VALUES(?,?,?)",
            (chunk.text, chunk_id, scope),
        )
        conn.execute(
            "INSERT INTO chunks_vec(scope, chunk_id, embedding) VALUES(?,?,?)",
            (scope, chunk_id, _f32(vector)),
        )
    conn.commit()


def delete_document(
    conn: sqlite3.Connection,
    document_id: str,
    *,
    commit: bool = True,
) -> None:
    """Remove a document and all its chunks (+ fts + vec rows)."""
    ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE document_id=?", (document_id,)
        ).fetchall()
    ]
    has_vec = rag_db.vec_table_exists(conn)
    for chunk_id in ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
        if has_vec:
            conn.execute("DELETE FROM chunks_vec WHERE chunk_id=?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    if commit:
        conn.commit()


def linked_folder_rows_exist(conn: sqlite3.Connection) -> bool:
    """Whether anything here can be hidden by the linked-folder filters.

    One EXISTS per thing they hide, so with all three empty the plain query returns the
    same rows straight out of the FTS index.

    A purged tombstone does not count: every knowledge base delete leaves one for good
    and its scope keeps no documents, so counting it would end the fast path on the first
    delete. Folder-owned documents are counted directly, not via `linked_folders`: a
    crash before `_install_mapping` leaves one that outlives its folder row.
    """
    return bool(
        conn.execute(
            "SELECT EXISTS(SELECT 1 FROM linked_folders) "
            "OR EXISTS(SELECT 1 FROM linked_folder_retired_scopes WHERE purged_at IS NULL) "
            "OR EXISTS(SELECT 1 FROM documents WHERE linked_folder_id IS NOT NULL)"
        ).fetchone()[0]
    )


def search_lexical(
    conn: sqlite3.Connection,
    scope,
    query: str,
    k: int,
    *,
    match_query: str | None = None,
    newest_first: bool = False,
    oldest_first: bool = False,
):
    """BM25 lexical search over one scope or several. Returns
    [(chunk_id, score)], higher = better.

    `match_query` lets a caller supply the FTS5 expression itself; the conversation
    archive shapes its own (see `conversation_match_queries`). Omitted, this is byte for
    byte what every other caller has always got.

    `newest_first` breaks TIES the other way round. FTS5 floors the IDF of a term the
    whole index shares, so every hit on a per-thread archive's own subject scores the
    same, and `ORDER BY s LIMIT k` then returns the k OLDEST rows: past k chunks on that
    subject the newest assignment is unreachable at any k. Ordering is by rowid, which is
    insertion order rather than exact conversation order, so this widens the candidate
    set and does not decide anything; the caller still orders what it gets.
    """
    mq = match_query if match_query is not None else _match_query(query)
    if not mq:
        return []
    scopes = _scopes(scope)
    if not scopes:
        return []
    placeholders = ",".join("?" * len(scopes))
    # One snapshot for the gate and the read: WAL pins it at the transaction's first read, so a scope
    # retired in between cannot land rows in a result the gate decided to run unfiltered.
    own_read_txn = not conn.in_transaction
    # Read-only, but it has to end: an open snapshot blocks WAL checkpointing.
    if own_read_txn:
        conn.execute("BEGIN")
    try:
        # The filtered form runs both subqueries for every matched row BEFORE the LIMIT, and with nothing
        # linked that work is provably wasted (linked_folder_rows_exist).
        if oldest_first:
            # Order by archive ordinal, not rowid, which a re-embed scrambles; NULLs first as oldest, then
            # created_at and chunk id, else on a legacy archive both halves return the same subset.
            sql = (
                f"SELECT chunks_fts.chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                f"JOIN chunks c ON c.id=chunks_fts.chunk_id "
                f"JOIN documents d ON d.id=c.document_id "
                f"WHERE chunks_fts MATCH ? AND chunks_fts.scope IN ({placeholders}) "
                f"ORDER BY s, d.archive_ordinal IS NOT NULL, d.archive_ordinal ASC, "
                f"d.created_at ASC, chunks_fts.chunk_id ASC LIMIT ?"
            )
        elif newest_first:
            # rowid is insertion order and a re-embed reinserts a chunk, so a rowid DESC window missed the newest turn.
            sql = (
                f"SELECT chunks_fts.chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                f"JOIN chunks c ON c.id=chunks_fts.chunk_id "
                f"JOIN documents d ON d.id=c.document_id "
                f"WHERE chunks_fts MATCH ? AND chunks_fts.scope IN ({placeholders}) "
                f"ORDER BY s, d.archive_ordinal IS NULL, d.archive_ordinal DESC, "
                f"d.created_at DESC, chunks_fts.chunk_id DESC LIMIT ?"
            )
        elif linked_folder_rows_exist(conn):
            sql = (
                f"SELECT chunks_fts.chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                f"JOIN chunks c ON c.id=chunks_fts.chunk_id "
                f"JOIN documents d ON d.id=c.document_id "
                f"WHERE chunks_fts MATCH ? AND chunks_fts.scope IN ({placeholders}) "
                f"AND NOT EXISTS "
                f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
                f"AND (d.linked_folder_id IS NULL OR EXISTS "
                f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)) "
                f"ORDER BY s LIMIT ?"
            )
        else:
            sql = (
                f"SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts "
                f"WHERE chunks_fts MATCH ? AND scope IN ({placeholders}) "
                f"ORDER BY s LIMIT ?"
            )
        rows = conn.execute(sql, (mq, *scopes, k)).fetchall()
    finally:
        if own_read_txn:
            conn.commit()
    # bm25() is negative (more negative = better); flip to higher-is-better.
    return [(r["chunk_id"], -r["s"]) for r in rows]


def search_dense(
    conn: sqlite3.Connection,
    scope,
    vector,
    k: int,
    *,
    embedding_model: str | None = None,
):
    """Cosine KNN over vec0 for one scope or several. Returns
    [(chunk_id, 1 - distance)]. vec0 KNN constrains its partition key by
    equality, so multi-scope runs one query per scope and merges by score.
    ``embedding_model`` is the querying embedder's identity (backend plus model, see
    ``embeddings.embedding_identity``); it drops hits from documents indexed by a
    different embedder of the same width, whose vectors live in another space. Rows
    written before identities carried a backend match on the model name alone, and
    NULL-model legacy documents are assumed current, matching the ingestion dedupe
    rule."""
    if not rag_db.vec_table_exists(conn):
        return []
    dim = rag_db.vec_table_dim(conn)
    if dim is not None and dim != len(vector):
        # The stale table cannot answer new-model queries: vec0 errors on the MATCH.
        return []
    # The pre-tag spelling of the same request, kept acceptable so an existing index keeps answering after an upgrade.
    untagged = config.embedding_identity_model(embedding_model) or embedding_model
    # dict.fromkeys collapses a scope named twice: a repeat would multiply that scope's fetch and emit
    # its hits twice into the merge.
    scopes = list(
        dict.fromkeys(
            s
            for s in _scopes(scope)
            if not conn.execute(
                "SELECT 1 FROM linked_folder_retired_scopes WHERE scope=?", (s,)
            ).fetchone()
        )
    )
    # Stale-model hits come from another space and can fill every fetched slot, so widen until k
    # compatible ones survive the filter.
    # Per scope, not across the merge: vec0 constrains its partition key by equality, so each scope has
    # its own stale prefix.
    kept: dict[str, list[tuple[str, float]]] = {}
    fetches = dict.fromkeys(scopes, max(k * 3, k + 10))
    pending = list(scopes)
    while pending:
        widen: list[str] = []
        for s in pending:
            fetch = fetches[s]
            rows = conn.execute(
                "SELECT chunk_id, distance FROM chunks_vec "
                "WHERE scope=? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                (s, _f32(vector), fetch),
            ).fetchall()
            kept[s] = _drop_incompatible(
                conn,
                [(r["chunk_id"], 1.0 - r["distance"]) for r in rows],
                embedding_model,
                untagged,
            )
            if len(kept[s]) < k and len(rows) >= fetch and fetch < _MAX_DENSE_FETCH:
                fetches[s] = min(fetch * 4, _MAX_DENSE_FETCH)
                widen.append(s)
        pending = widen
    out = [hit for s in scopes for hit in kept[s]]
    out.sort(key = lambda t: t[1], reverse = True)
    return out[:k]


# Past this many nearest neighbours the scope is effectively another embedder's, and a re-upload is the answer.
_MAX_DENSE_FETCH = 4096
# One id per bound parameter, kept under the oldest SQLITE_MAX_VARIABLE_NUMBER.
_ID_BATCH = 900


def _drop_incompatible(
    conn: sqlite3.Connection,
    candidates: list[tuple[str, float]],
    embedding_model: str | None,
    untagged: str | None,
) -> list[tuple[str, float]]:
    """Keep the KNN candidates whose document is still live and whose vectors this
    query can be compared against."""
    valid: set[str] = set()
    ids = [cid for cid, _ in candidates]
    for start in range(0, len(ids), _ID_BATCH):
        batch = ids[start : start + _ID_BATCH]
        placeholders = ",".join("?" * len(batch))
        valid.update(
            r["id"]
            for r in conn.execute(
                f"SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id "
                f"WHERE c.id IN ({placeholders}) AND NOT EXISTS "
                f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
                f"AND (d.linked_folder_id IS NULL OR EXISTS "
                f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)) "
                f"AND (? IS NULL OR d.embedding_model IS NULL OR d.embedding_model=? "
                f"OR d.embedding_model=?)",
                (*batch, embedding_model, embedding_model, untagged),
            ).fetchall()
        )
    return [t for t in candidates if t[0] in valid]


def count_untagged_documents(conn: sqlite3.Connection) -> int:
    """Documents whose ``embedding_model`` predates backend tagging.

    Either backend could have written them, because the llama-server fallback never
    recorded that it had taken over, and nothing in the row says which pooling the
    vectors came from. We keep serving them rather than drop a corpus or re-embed one
    behind the user's back, so this exists to say how many are in that state."""
    tags = " ".join(f"AND embedding_model NOT LIKE '{t}:%'" for t in config.EMBEDDING_IDENTITY_TAGS)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM documents WHERE embedding_model IS NOT NULL {tags}"
    ).fetchone()
    return int(row["n"]) if row else 0


def chunks_by_id(conn: sqlite3.Connection, ids) -> dict:
    """Hydrate chunk rows (joined with document filename), keyed by id."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT c.id, c.text, c.document_id, c.chunk_index, c.page_number, "
        f"c.source_page_index, d.filename, d.archive_ordinal, d.created_at "
        f"FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE c.id IN ({placeholders}) AND NOT EXISTS "
        f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        f"AND (d.linked_folder_id IS NULL OR EXISTS "
        f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id))",
        list(ids),
    ).fetchall()
    return {r["id"]: r for r in rows}


def all_chunks_for_scope(conn: sqlite3.Connection, scope) -> list[dict]:
    """Every completed-document chunk for a scope, ordered document-then-index and
    joined with the document filename. Backs whole-document context injection, so
    it does no retrieval or embedding."""
    scopes = _scopes(scope)
    if not scopes:
        return []
    placeholders = ",".join("?" * len(scopes))
    rows = conn.execute(
        f"SELECT c.id, c.text, c.document_id, c.chunk_index, c.page_number, "
        f"c.token_count, d.filename, d.created_at "
        f"FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE c.scope IN ({placeholders}) AND d.status='completed' "
        f"AND NOT EXISTS "
        f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        f"AND (d.linked_folder_id IS NULL OR EXISTS "
        f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)) "
        f"ORDER BY d.created_at, c.document_id, c.chunk_index",
        list(scopes),
    ).fetchall()
    return [dict(r) for r in rows]


def scope_token_estimate(conn: sqlite3.Connection, scope) -> int:
    """Upper-bound token total for a scope's completed chunks without hydrating text.
    Mirrors ``all_chunks_for_scope`` + the ``tool._row_token_count`` fallback (stored
    count, else length/4), so the whole-doc budget can be checked before loading text."""
    scopes = _scopes(scope)
    if not scopes:
        return 0
    placeholders = ",".join("?" * len(scopes))
    row = conn.execute(
        f"SELECT COALESCE(SUM(CASE WHEN c.token_count > 0 THEN c.token_count "
        f"ELSE MAX(1, length(COALESCE(c.text, '')) / 4) END), 0) AS total "
        f"FROM chunks c JOIN documents d ON d.id=c.document_id "
        f"WHERE c.scope IN ({placeholders}) AND d.status='completed' "
        f"AND NOT EXISTS "
        f"(SELECT 1 FROM linked_folder_retired_scopes r WHERE r.scope=d.scope) "
        f"AND (d.linked_folder_id IS NULL OR EXISTS "
        f"(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id))",
        list(scopes),
    ).fetchone()
    return int(row["total"] or 0)
