# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resource factories for route isolation tests. Add a table entry to cover another route."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from auth import storage
from utils.account_context import run_as
from utils.paths import workspace_root

from .seed import MESSAGE_ID, SENTINEL, THREAD_ID, seed_studio_db

PROJECT_ID = "matrix-project"
RUN_ID = "matrix-run"
SERVER_ID = "matrix-server"
EDITED = "matrix-edited"
MESSAGE = {
    "id": MESSAGE_ID, "threadId": THREAD_ID, "role": "user",
    "content": [{"type": "text", "text": SENTINEL}], "createdAt": 1000,
}


@dataclass(frozen = True)
class Factory:
    name: str
    body: dict | None = None
    success: int = 200
    fragment: str | None = None


FACTORIES = {
    "routes.chat_history:GET:/threads/{thread_id}": Factory("chat", fragment = SENTINEL),
    "routes.chat_history:PATCH:/threads/{thread_id}": Factory("chat", {"title": EDITED}, fragment = EDITED),
    "routes.chat_history:GET:/threads/{thread_id}/messages": Factory("chat", fragment = SENTINEL),
    "routes.chat_history:GET:/threads/{thread_id}/messages/{message_id}": Factory("chat", fragment = SENTINEL),
    "routes.chat_history:PUT:/threads/{thread_id}/messages/{message_id}": Factory("chat", MESSAGE, fragment = SENTINEL),
    "routes.chat_history:PUT:/threads/{thread_id}/messages": Factory("chat", {"messages": [MESSAGE]}, fragment = SENTINEL),
    "routes.chat_history:GET:/projects/{project_id}": Factory("project", fragment = SENTINEL),
    "routes.chat_history:PATCH:/projects/{project_id}": Factory("project", {"name": EDITED}, fragment = EDITED),
    "routes.training_history:GET:/runs/{run_id}": Factory("training", fragment = SENTINEL),
    "routes.training_history:PATCH:/runs/{run_id}": Factory("training", {"display_name": EDITED}, fragment = EDITED),
    "routes.auth:DELETE:/api-keys/{key_id}": Factory("api-key"),
    "routes.mcp_servers:PUT:/{server_id}": Factory("mcp", {"display_name": EDITED}, fragment = EDITED),
    "routes.mcp_servers:DELETE:/{server_id}": Factory("mcp", success = 204),
}


def initialize_workspaces(accounts: dict) -> None:
    """Use populated on-disk schemas so this matrix tests authorization, not first-use migration.

    The separate first-use test intentionally exercises production schema initialization.
    """
    for account in accounts.values():
        path = run_as(account, workspace_root) / "studio.db"
        seed_studio_db(path, populated = False)
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript("""
                CREATE TABLE mcp_servers (
                    id TEXT PRIMARY KEY, display_name TEXT NOT NULL, url TEXT NOT NULL,
                    headers_json TEXT, is_enabled INTEGER NOT NULL DEFAULT 1,
                    use_oauth INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)


def seed_resource(factory: Factory, account) -> dict[str, str]:
    root = run_as(account, workspace_root)
    path = root / "studio.db"
    with closing(sqlite3.connect(path)) as conn:
        if factory.name == "chat":
            conn.execute(
                "INSERT INTO chat_threads (id,title,model_type,created_at,updated_at) VALUES (?,?,?,1000,1000)",
                (THREAD_ID, SENTINEL, "base"),
            )
            conn.execute(
                "INSERT INTO chat_messages (id,thread_id,role,content_json,created_at) VALUES (?,?,?, ?,1000)",
                (MESSAGE_ID, THREAD_ID, "user", json.dumps(MESSAGE["content"])),
            )
        elif factory.name == "project":
            conn.execute(
                "INSERT INTO chat_projects (id,name,root_path,created_at,updated_at) VALUES (?,?,?,1000,1000)",
                (PROJECT_ID, SENTINEL, str(root / "projects" / PROJECT_ID)),
            )
        elif factory.name == "training":
            conn.execute(
                "INSERT INTO training_runs (id,status,model_name,dataset_name,config_json,started_at,display_name) "
                "VALUES (?,'completed','local/model','local/dataset','{}','2026-01-01T00:00:00+00:00',?)",
                (RUN_ID, SENTINEL),
            )
        conn.commit()
    params = {
        "thread_id": THREAD_ID, "message_id": MESSAGE_ID, "project_id": PROJECT_ID,
        "run_id": RUN_ID, "server_id": SERVER_ID,
    }
    if factory.name == "api-key":
        _, row = storage.create_api_key(account.username, name = SENTINEL)
        params["key_id"] = str(row["id"])
    if factory.name == "mcp":
        from storage import mcp_servers_db
        run_as(
            account, mcp_servers_db.create_server, SERVER_ID, SENTINEL, "http://8.8.8.8:9/mcp",
            headers_json = None, is_enabled = False, use_oauth = False,
        )
    return params


def snapshot_resource(account) -> tuple:
    """Logical snapshot avoids WAL/checkpoint differences on a read-only rejected request."""
    path: Path = run_as(account, workspace_root) / "studio.db"
    with closing(sqlite3.connect(path)) as conn:
        return tuple(conn.iterdump())
