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

from __future__ import annotations

import contextvars
import threading

from unforgettable.loop.runtime import (
    bind_episode,
    current_db_path,
    note_tool_result,
    reset_episode,
)
from unforgettable.store.records import list_records
from unforgettable.tools.handlers import dispatch


def test_copy_context_carries_episode_into_worker(tmp_path):
    db = str(tmp_path / "memory.db")
    seen: dict[str, str | None] = {}

    def worker() -> None:
        seen["db"] = current_db_path()
        dispatch(
            "memory_write",
            {
                "kind": "directive",
                "title": "From worker",
                "body": "context copied",
                "provenance": "human",
            },
        )
        note_tool_result("memory_write", {"title": "From worker"}, "ok")

    tokens, traces = bind_episode(db_path = db, episode_id = "ep-ctx")
    try:
        thread = threading.Thread(target = contextvars.copy_context().run, args = (worker,))
        thread.start()
        thread.join()
        from unforgettable.loop.runtime import current_traces

        after = current_traces()
    finally:
        reset_episode(tokens)
    assert seen["db"] == db
    rows = list_records(db_path = db)
    assert any(row["title"] == "From worker" for row in rows)
    assert any(t.name == "memory_write" for t in after)


def test_raw_thread_does_not_see_episode_db(tmp_path):
    db = str(tmp_path / "memory.db")
    seen: dict[str, str | None] = {}

    def worker() -> None:
        seen["db"] = current_db_path()

    tokens, _ = bind_episode(db_path = db, episode_id = "ep-raw")
    try:
        thread = threading.Thread(target = worker)
        thread.start()
        thread.join()
    finally:
        reset_episode(tokens)
    assert seen["db"] is None
