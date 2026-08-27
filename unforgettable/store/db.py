# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-call SQLite connections. Same pattern as Studio's stores, own file."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

from .schema import ensure_schema

_schema_lock = threading.Lock()
_ready_paths: set[str] = set()


def default_db_path() -> Path:
    override = (os.environ.get("UNFORGETTABLE_HOME") or "").strip()
    if override:
        return Path(override).expanduser() / "memory.db"
    return Path.home() / ".unforgettable" / "memory.db"


def get_connection(db_path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(str(path), timeout = 5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    key = str(path.resolve())
    if key not in _ready_paths:
        with _schema_lock:
            if key not in _ready_paths:
                ensure_schema(conn)
                conn.commit()
                _ready_paths.add(key)
    return conn
