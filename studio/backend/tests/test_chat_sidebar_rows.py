# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paged sidebar rows: parity with the frontend's groupThreads + applyManualOrder."""

import inspect
import os
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from auth.authentication import get_current_subject
from routes import chat_history
from storage import studio_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    yield tmp_path


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _insert_threads(threads: list[dict]) -> None:
    conn = studio_db.get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO chat_threads (id, title, model_type, model_id, pair_id, project_id, archived,
                                      created_at, updated_at, forked_from_thread_id, modified_at)
            VALUES (:id, :title, :model_type, :model_id, :pair_id, :project_id, :archived,
                    :created_at, :updated_at, :forked_from_thread_id, :modified_at)
            """,
            [
                {
                    "title": t["id"],
                    "model_type": "base",
                    "model_id": None,
                    "pair_id": None,
                    "project_id": None,
                    "archived": 0,
                    "updated_at": None,
                    "forked_from_thread_id": None,
                    "modified_at": None,
                    **t,
                }
                for t in threads
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _insert_project(
    project_id: str,
    *,
    created_at: int,
    updated_at: int,
    name = None,
) -> None:
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_projects (id, name, instructions, archived, created_at, updated_at) "
            "VALUES (?, ?, '', 0, ?, ?)",
            (project_id, name or project_id, created_at, updated_at),
        )
        conn.commit()
    finally:
        conn.close()


def _singles(n: int, *, start: int = 1_000) -> list[dict]:
    return [{"id": f"t{i:04d}", "created_at": start + i, "updated_at": start + i} for i in range(n)]


def _group_threads(threads: list[dict], archived: bool = False) -> list[dict]:
    """Python port of use-chat-sidebar-items.ts groupThreads, fed what GET /threads returns."""
    items: list[dict] = []
    pairs: dict[str, dict] = {}
    for t in threads:
        if bool(t.get("archived")) != archived:
            continue
        activity = t.get("updatedAt") if t.get("updatedAt") is not None else t["createdAt"]
        if t.get("pairId"):
            existing = pairs.get(t["pairId"])
            if existing:
                existing["createdAt"] = max(existing["createdAt"], t["createdAt"])
                existing["updatedAt"] = max(existing["updatedAt"], activity)
                if t.get("modifiedAt"):
                    existing["modifiedAt"] = max(existing.get("modifiedAt") or 0, t["modifiedAt"])
                if t.get("modelId") and t["modelId"] not in existing.get("modelIds", []):
                    existing["modelIds"] = [*existing.get("modelIds", []), t["modelId"]]
                existing["threadIds"].append(t["id"])
                continue
            item = {
                "type": "compare",
                "id": t["pairId"],
                "threadIds": [t["id"]],
                "title": t["title"],
                "createdAt": t["createdAt"],
                "updatedAt": activity,
                **({"modifiedAt": t["modifiedAt"]} if t.get("modifiedAt") else {}),
                **({"modelIds": [t["modelId"]]} if t.get("modelId") else {}),
                "projectId": t.get("projectId"),
            }
            pairs[t["pairId"]] = item
            items.append(item)
        else:
            items.append(
                {
                    "type": "single",
                    "id": t["id"],
                    "threadIds": [t["id"]],
                    "title": t["title"],
                    "createdAt": t["createdAt"],
                    "updatedAt": activity,
                    **({"modifiedAt": t["modifiedAt"]} if t.get("modifiedAt") else {}),
                    **({"modelIds": [t["modelId"]]} if t.get("modelId") else {}),
                    "isFork": bool(t.get("forkedFromThreadId")),
                    "projectId": t.get("projectId"),
                }
            )
    return sorted(items, key = lambda item: -item["updatedAt"])


def _apply_manual_order(items: list[dict], order: list[str]) -> list[dict]:
    if not order:
        return items
    rank = {row_id: index for index, row_id in enumerate(order)}
    return sorted(items, key = lambda item: rank.get(item["id"], -1))


def _page_all(client, body: dict) -> list[dict]:
    rows, cursor, pages = [], None, 0
    while True:
        response = client.post("/api/chat/sidebar/rows", json = {**body, "cursor": cursor})
        assert response.status_code == 200, response.text
        data = response.json()
        assert len(data["rows"]) <= body.get("limit", 20)
        rows += data["rows"]
        pages += 1
        assert data["hasMore"] == (data["nextCursor"] is not None)
        if not data["hasMore"]:
            return rows
        cursor = data["nextCursor"]
        assert pages < 1000


@pytest.mark.parametrize("n", [0, 1, 19, 20, 21, 40, 41])
def test_page_edges(client, n):
    _insert_threads(_singles(n))
    first = client.post("/api/chat/sidebar/rows", json = {}).json()
    assert len(first["rows"]) == min(n, 20)
    assert first["total"] == n
    assert first["hasMore"] == (n > 20)
    rows = _page_all(client, {})
    assert [row["id"] for row in rows] == [f"t{i:04d}" for i in reversed(range(n))]


def test_equal_timestamps_page_exactly_once(client):
    _insert_threads([{"id": f"t{i:03d}", "created_at": 5, "updated_at": 5} for i in range(45)])
    rows = _page_all(client, {"limit": 7})
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == 45
    # groupThreads sorts stably, so a full tie keeps list order, which is insertion order here.
    assert ids == [f"t{i:03d}" for i in range(45)]


def test_full_traversal_matches_get_threads_and_group_threads(client):
    threads = []
    for i in range(70):
        base = {"created_at": 1_000 + i * 10, "updated_at": 1_000 + i * 10 + 3}
        if i % 7 == 0:
            threads.append(
                {
                    **base,
                    "id": f"p{i}a",
                    "pair_id": f"pair{i}",
                    "title": f"old {i}",
                    "model_id": "m1",
                }
            )
            threads.append(
                {
                    **base,
                    "id": f"p{i}b",
                    "pair_id": f"pair{i}",
                    "title": f"new {i}",
                    "model_id": "m2",
                    "updated_at": base["updated_at"] + 5,
                    "modified_at": 77 if i % 2 else None,
                    "project_id": None,
                }
            )
        else:
            threads.append(
                {
                    **base,
                    "id": f"s{i}",
                    "model_id": "m1" if i % 3 else "",
                    "updated_at": None if i % 5 == 0 else base["updated_at"],
                    "forked_from_thread_id": "s1" if i % 4 == 0 else None,
                    "modified_at": 9 if i % 6 == 0 else None,
                    "archived": 1 if i % 11 == 0 else 0,
                }
            )
    _insert_threads(threads)
    listed = client.get("/api/chat/threads?include_archived=true").json()["threads"]
    for archived in (False, True):
        expected = _group_threads(listed, archived = archived)
        rows = _page_all(client, {"archived": archived, "limit": 6})
        # compare rows also carry each pane's own title, by threadIds position
        title_of = {t["id"]: t["title"] for t in listed}
        for row in rows:
            titles = row.pop("threadTitles", None)
            assert titles == (
                [title_of[t] for t in row["threadIds"]] if row["type"] == "compare" else None
            )
        assert rows == expected


def test_pair_is_never_split_and_takes_newest_member_title_and_project(client, db):
    _insert_project("proj", created_at = 1, updated_at = 1)
    threads = _singles(19)
    threads += [
        {"id": "pa", "pair_id": "pair", "title": "A", "created_at": 50, "updated_at": 60},
        {
            "id": "pb",
            "pair_id": "pair",
            "title": "B",
            "project_id": "proj",
            "created_at": 40,
            "updated_at": 9_000,
        },
    ]
    _insert_threads(threads)
    first = client.post("/api/chat/sidebar/rows", json = {"limit": 1}).json()
    assert first["rows"][0]["id"] == "pair"
    assert first["rows"][0]["threadIds"] == ["pb", "pa"]
    assert first["rows"][0]["title"] == "B"
    assert first["rows"][0]["threadTitles"] == ["B", "A"]
    assert first["rows"][0]["projectId"] == "proj"
    assert first["rows"][0]["createdAt"] == 50
    assert first["rows"][0]["type"] == "compare"
    assert "isFork" not in first["rows"][0]
    assert first["total"] == 20


def test_archived_null_reads_as_unarchived(client, db):
    _insert_threads(_singles(3))
    conn = studio_db.get_connection()
    try:
        # The column is NOT NULL today; an older table may not be, so rebuild it without the check.
        conn.execute("ALTER TABLE chat_threads RENAME TO chat_threads_old")
        conn.execute("CREATE TABLE chat_threads AS SELECT * FROM chat_threads_old WHERE 0")
        conn.execute("INSERT INTO chat_threads SELECT * FROM chat_threads_old")
        conn.execute("DROP TABLE chat_threads_old")
        conn.execute("UPDATE chat_threads SET archived = NULL WHERE id = 't0001'")
        conn.commit()
    finally:
        conn.close()
    ids = [row["id"] for row in _page_all(client, {})]
    assert ids == ["t0002", "t0001", "t0000"]
    assert _page_all(client, {"archived": True}) == []


def test_include_exclude_project_and_unfiled(client, db):
    _insert_project("proj", created_at = 1, updated_at = 1)
    threads = _singles(30)
    for t in threads[:10]:
        t["project_id"] = "proj"
    _insert_threads(threads)
    pinned = ["t0003", "t0025", "missing"]
    included = _page_all(client, {"includeIds": pinned})
    assert [row["id"] for row in included] == ["t0025", "t0003"]
    assert _page_all(client, {"includeIds": []}) == []

    recents = _page_all(client, {"excludeIds": ["t0025", "t0029"], "unfiledOnly": True})
    assert [row["id"] for row in recents] == [
        f"t{i:04d}" for i in reversed(range(10, 30)) if i not in (25, 29)
    ]
    folder = client.post(
        "/api/chat/sidebar/rows",
        json = {"projectId": "proj", "excludeIds": ["t0003"], "limit": 4},
    ).json()
    assert [row["id"] for row in folder["rows"]] == ["t0009", "t0008", "t0007", "t0006"]
    assert folder["total"] == 9
    assert all(row["projectId"] == "proj" for row in folder["rows"])


def test_include_ids_match_a_compare_pair_through_a_pane_thread(client):
    _insert_threads(
        [
            {"id": "pane-a", "pair_id": "pair", "created_at": 1},
            {"id": "pane-b", "pair_id": "pair", "created_at": 2},
            {"id": "solo", "created_at": 3},
        ]
    )
    rows = _page_all(client, {"includeIds": ["pane-b", "solo", "pair"]})
    assert [(row["id"], sorted(row["threadIds"])) for row in rows] == [
        ("solo", ["solo"]),
        ("pair", ["pane-a", "pane-b"]),
    ]


def test_manual_sort_matches_apply_manual_order(client):
    _insert_threads(_singles(45))
    listed = client.get("/api/chat/threads").json()["threads"]
    order = ["t0002", "t0040", "missing", "t0010", "t0002", "t0033"]
    expected = _apply_manual_order(_group_threads(listed), order)
    rows = _page_all(client, {"sort": "manual", "manualOrder": order, "limit": 20})
    assert rows == expected
    assert [row["id"] for row in rows[-4:]] == ["t0040", "t0010", "t0002", "t0033"]
    assert _page_all(client, {"sort": "manual"}) == _group_threads(listed)


@pytest.mark.parametrize(
    "cursor",
    [
        "!!!",
        "bm90IGpzb24",  # "not json"
        studio_db.encode_sidebar_cursor({"s": "manual", "o": 3}),  # another sort's cursor
        studio_db.encode_sidebar_cursor({"s": "updated", "u": "x", "i": "t"}),
        studio_db.encode_sidebar_cursor({"s": "updated", "u": 2**63, "c": 0, "r": 0}),
        studio_db.encode_sidebar_cursor({"s": "updated", "u": 0, "c": 0, "r": -(2**63)}),
    ],
)
def test_malformed_cursor_is_400(client, cursor):
    _insert_threads(_singles(3))
    response = client.post("/api/chat/sidebar/rows", json = {"cursor": cursor})
    assert response.status_code == 400
    with pytest.raises(HTTPException) as raised:
        chat_history.list_sidebar_rows(
            chat_history.ChatSidebarRowsRequest(cursor = cursor), current_subject = "t"
        )
    assert raised.value.status_code == 400


def test_id_lists_past_ten_thousand_are_accepted(client):
    # a manual order spanning a long history, or that many filed chats, still pages
    _insert_threads(_singles(30))
    order = [f"t{i:04d}" for i in reversed(range(30))] + [f"gone{i}" for i in range(20_000)]
    exclude = ["t0000"] + [f"gone{i}" for i in range(20_000)]
    response = client.post(
        "/api/chat/sidebar/rows",
        json = {"sort": "manual", "manualOrder": order, "excludeIds": exclude, "limit": 5},
    )
    assert response.status_code == 200
    page = response.json()
    assert [row["id"] for row in page["rows"]] == ["t0029", "t0028", "t0027", "t0026", "t0025"]
    assert page["total"] == 29


@pytest.mark.parametrize("limit", [0, 201])
def test_limit_is_bounded(client, limit):
    assert client.post("/api/chat/sidebar/rows", json = {"limit": limit}).status_code == 422


def test_projects_last_activity(client, db):
    _insert_project("a", created_at = 10, updated_at = 10, name = "beta")
    _insert_project("b", created_at = 20, updated_at = 500, name = "Alpha")
    _insert_project("c", created_at = 30, updated_at = 30, name = "gamma")
    _insert_threads(
        [
            {"id": "x", "project_id": "a", "created_at": 900, "updated_at": 1_000},
            {"id": "y", "project_id": "c", "created_at": 9_999, "archived": 1},
            {"id": "z", "project_id": "c", "created_at": 40, "updated_at": None},
        ]
    )
    data = client.get("/api/chat/projects").json()
    by_id = {p["id"]: p for p in data["projects"]}
    assert {k: v["lastActivityAt"] for k, v in by_id.items()} == {"a": 1_000, "b": 500, "c": 40}
    assert [p["id"] for p in data["projects"]] == ["b", "c", "a"]


def test_thread_rows_lookup(client):
    _insert_threads(
        [
            {"id": "pa", "pair_id": "pair", "created_at": 1},
            {"id": "pb", "pair_id": "pair", "created_at": 2},
            {"id": "s", "created_at": 3, "archived": 1},
        ]
    )
    response = client.get(
        "/api/chat/sidebar/thread-rows?thread_ids=pa&thread_ids=pb&thread_ids=s&thread_ids=gone"
    )
    assert response.json() == {"rowIdByThreadId": {"pa": "pair", "pb": "pair", "s": "s"}}
    too_many = "&".join(f"thread_ids=t{i}" for i in range(501))
    assert client.get(f"/api/chat/sidebar/thread-rows?{too_many}").status_code == 400


def test_sidebar_handlers_stay_sync():
    """Sync handlers run in the threadpool; the async-handler inventory test pins the rest."""
    assert not inspect.iscoroutinefunction(chat_history.list_sidebar_rows)
    assert not inspect.iscoroutinefunction(chat_history.sidebar_thread_rows)
