# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resource factories for route isolation tests. Add a table entry to cover another route."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from auth import storage
from utils.account_context import run_as
from utils.paths import workspace_root

from .factory_base import Factory, call_seeder, format_path, merge, seeder
from .seed import MESSAGE_ID, SENTINEL, THREAD_ID, seed_studio_db

PROJECT_ID = "matrix-project"
RUN_ID = "matrix-run"
SERVER_ID = "matrix-server"
EDITED = "matrix-edited"
MESSAGE = {
    "id": MESSAGE_ID,
    "threadId": THREAD_ID,
    "role": "user",
    "content": [{"type": "text", "text": SENTINEL}],
    "createdAt": 1000,
}

__all__ = ["Factory", "FACTORIES", "SKIPPED", "format_path", "seed_resource", "snapshot_resource"]


def studio_db_path(account) -> Path:
    return run_as(account, workspace_root) / "studio.db"


def studio_connection(account):
    return closing(sqlite3.connect(studio_db_path(account)))


@seeder("chat")
def seed_chat(account) -> dict[str, str]:
    with studio_connection(account) as conn:
        conn.execute(
            "INSERT INTO chat_threads (id,title,model_type,created_at,updated_at) VALUES (?,?,?,1000,1000)",
            (THREAD_ID, SENTINEL, "base"),
        )
        conn.execute(
            "INSERT INTO chat_messages (id,thread_id,role,content_json,created_at) VALUES (?,?,?,?,1000)",
            (MESSAGE_ID, THREAD_ID, "user", json.dumps(MESSAGE["content"])),
        )
        conn.commit()
    return {"thread_id": THREAD_ID, "message_id": MESSAGE_ID}


@seeder("project")
def seed_project(account) -> dict[str, str]:
    root = run_as(account, workspace_root)
    with studio_connection(account) as conn:
        conn.execute(
            "INSERT INTO chat_projects (id,name,root_path,created_at,updated_at) VALUES (?,?,?,1000,1000)",
            (PROJECT_ID, SENTINEL, str(root / "projects" / PROJECT_ID)),
        )
        conn.commit()
    return {"project_id": PROJECT_ID}


@seeder("training")
def seed_training(account) -> dict[str, str]:
    with studio_connection(account) as conn:
        conn.execute(
            "INSERT INTO training_runs (id,status,model_name,dataset_name,config_json,started_at,display_name) "
            "VALUES (?,'completed','local/model','local/dataset','{}','2026-01-01T00:00:00+00:00',?)",
            (RUN_ID, SENTINEL),
        )
        conn.commit()
    return {"run_id": RUN_ID}


@seeder("api-key")
def seed_api_key(account) -> dict[str, str]:
    _, row = storage.create_api_key(account.username, name = SENTINEL)
    return {"key_id": str(row["id"])}


@seeder("mcp")
def seed_mcp(account) -> dict[str, str]:
    from storage import mcp_servers_db
    run_as(
        account,
        mcp_servers_db.create_server,
        SERVER_ID,
        SENTINEL,
        "http://8.8.8.8:9/mcp",
        headers_json = None,
        is_enabled = False,
        use_oauth = False,
    )
    return {"server_id": SERVER_ID}


CORE_FACTORIES = {
    "routes.chat_history:GET:/threads/{thread_id}": Factory("chat", fragment = SENTINEL),
    "routes.chat_history:PATCH:/threads/{thread_id}": Factory(
        "chat", {"title": EDITED}, fragment = EDITED
    ),
    "routes.chat_history:GET:/threads/{thread_id}/messages": Factory("chat", fragment = SENTINEL),
    "routes.chat_history:GET:/threads/{thread_id}/messages/{message_id}": Factory(
        "chat", fragment = SENTINEL
    ),
    "routes.chat_history:PUT:/threads/{thread_id}/messages/{message_id}": Factory(
        "chat", MESSAGE, fragment = SENTINEL
    ),
    "routes.chat_history:PUT:/threads/{thread_id}/messages": Factory(
        "chat", {"messages": [MESSAGE]}, fragment = SENTINEL
    ),
    "routes.chat_history:GET:/projects/{project_id}": Factory("project", fragment = SENTINEL),
    "routes.chat_history:PATCH:/projects/{project_id}": Factory(
        "project", {"name": EDITED}, fragment = EDITED
    ),
    "routes.training_history:GET:/runs/{run_id}": Factory("training", fragment = SENTINEL),
    "routes.training_history:PATCH:/runs/{run_id}": Factory(
        "training", {"display_name": EDITED}, fragment = EDITED
    ),
    "routes.auth:DELETE:/api-keys/{key_id}": Factory("api-key"),
    "routes.mcp_servers:PUT:/{server_id}": Factory(
        "mcp", {"display_name": EDITED}, fragment = EDITED
    ),
    "routes.mcp_servers:DELETE:/{server_id}": Factory("mcp", success = 204),
}


from . import (  # noqa: E402  domain tables import Factory and register their seeders first
    factories_chat,
    factories_media,
    factories_providers,
    factories_rag,
    factories_runs,
    factories_training,
)

DOMAINS = (
    factories_chat,
    factories_media,
    factories_providers,
    factories_rag,
    factories_runs,
    factories_training,
)

FACTORIES = merge(CORE_FACTORIES, *(domain.FACTORIES for domain in DOMAINS))

SKIPPED = merge(*(domain.SKIPPED for domain in DOMAINS))


def initialize_workspaces(accounts: dict) -> None:
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


def seed_resource(
    factory: Factory,
    account,
    actor: str = "right",
) -> dict[str, str]:
    params = dict(call_seeder(factory.name, account, actor))
    params.update(factory.extra_params)
    return params


_VOLATILE = (".db-wal", ".db-shm", ".db-journal")


def snapshot_resource(account) -> tuple:
    """Logical snapshot avoids WAL/checkpoint differences on a read-only rejected request."""
    root: Path = run_as(account, workspace_root)
    entries = []
    for path in sorted(root.rglob("*")) if root.exists() else []:
        if not path.is_file() or path.name.endswith(_VOLATILE):
            continue
        relative = str(path.relative_to(root))
        if path.suffix == ".db":
            with closing(sqlite3.connect(path)) as conn:
                entries.append((relative, tuple(conn.iterdump())))
        else:
            entries.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(entries)
