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

import asyncio
from pathlib import Path

from unforgettable.agents.extractor import EPISODE_TITLE_ID_CHARS, TWIN_NOTE_TITLE
from unforgettable.host import EXTRACT_MAX_TOKENS, GenerateRequest, GenerateResult, ToolTrace
from unforgettable.loop.context import EpisodeRequest
from unforgettable.loop.episode import run
from unforgettable.store.records import (
    get_record,
    insert_record,
    list_records,
    list_rollouts,
)
from unforgettable.store.search import search_records
from unforgettable.throne.policy import Action


class FakeHost:
    def __init__(self, root: Path, results: list[GenerateResult]):
        self.db = root / "memory.db"
        self.world = root / "world"
        self.world.mkdir()
        (self.world / "app.py").write_text("print('world')\n")
        self.sims: dict[str, Path] = {}
        self.removed: list[str] = []
        self.calls: list[str] = []
        self._results = list(results)
        self.last_messages = None

    def memory_db_path(self) -> Path:
        return self.db

    def world_session_id(self, request) -> str:
        return "world"

    def create_sim_session(self, episode_id: str) -> str:
        sid = f"sim-{episode_id}-1"
        path = self.world.parent / sid
        path.mkdir()
        self.sims[sid] = path
        return sid

    def sandbox_path(self, session_id: str) -> Path:
        if session_id == "world":
            return self.world
        return self.sims[session_id]

    def remove_sim_session(self, session_id: str) -> None:
        self.removed.append(session_id)

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        self.calls.append(req.session_id)
        self.last_messages = req.messages
        if not self._results:
            raise AssertionError("unexpected extra generate")
        return self._results.pop(0)

    async def complete(self, messages, *, max_tokens=EXTRACT_MAX_TOKENS) -> str:
        return ""


def _fail_world() -> GenerateResult:
    return GenerateResult(
        text="that command failed",
        tool_traces=[
            ToolTrace("terminal", {"command": "false"}, "exit code 1", "world")
        ],
    )


def _ok(text: str, contact: str) -> GenerateResult:
    return GenerateResult(
        text=text,
        tool_traces=[ToolTrace("terminal", {"command": "true"}, "ok\n", contact)],
    )


def test_episode_fail_sim_retry_writes_error_fix(tmp_path: Path):
    insert_record(
        kind="procedure",
        title="Run the tests",
        body="Use pytest in the project root.",
        provenance="human",
        db_path=tmp_path / "memory.db",
    )
    host = FakeHost(
        tmp_path,
        [_fail_world(), _ok("fixed in sim", "sim"), _ok("works in world", "world")],
    )
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(messages=[{"role": "user", "content": "run the tests"}]),
        )
    )
    assert host.calls[0] == "world"
    assert host.calls[1].startswith("sim-")
    assert host.calls[2] == "world"
    assert Action.ENTER_SIM in outcome.actions
    assert Action.RETRY_WORLD in outcome.actions
    assert outcome.error_fix_id
    fix = get_record(outcome.error_fix_id, db_path=host.db)
    assert fix["kind"] == "error_fix"
    assert fix["provenance"] == "mixed"
    assert fix["status"] == "proposed"
    assert (host.sims[host.calls[1]] / "app.py").read_text() == "print('world')\n"
    assert host.removed == []
    injected = " ".join(
        str(m.get("content")) for m in (host.last_messages or []) if m.get("role") == "system"
    )
    assert "memory_write" in injected
    assert "Run the tests" in " ".join(
        str(m.get("content")) for m in (outcome.state and host.last_messages or [])
    )


def test_episode_sim_ok_world_retry_fail_writes_twin_note(tmp_path: Path):
    host = FakeHost(
        tmp_path,
        [_fail_world(), _ok("fixed in sim", "sim"), _fail_world()],
    )
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(messages=[{"role": "user", "content": "run the tests"}]),
        )
    )
    assert host.calls[0] == "world"
    assert host.calls[1].startswith("sim-")
    assert host.calls[2] == "world"
    assert Action.ENTER_SIM in outcome.actions
    assert Action.RETRY_WORLD in outcome.actions
    assert Action.ESCALATE in outcome.actions
    notes = list_records(kinds=["twin_note"], db_path=host.db)
    assert len(notes) == 1
    note = notes[0]
    assert note["kind"] == "twin_note"
    assert note["status"] == "active"
    assert note["provenance"] == "mixed"
    assert note["title"] == TWIN_NOTE_TITLE
    fixes = list_records(kinds=["error_fix"], db_path=host.db)
    assert len(fixes) == 1
    assert fixes[0]["status"] == "proposed"
    assert fixes[0]["kind"] == "error_fix"
    episodes = list_records(kinds=["episode"], db_path=host.db)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["status"] == "active"
    assert episode["source_episode_id"] == outcome.state.episode_id
    assert episode["title"] == f"Episode {outcome.state.episode_id[:EPISODE_TITLE_ID_CHARS]}"
    grades = {
        (row["contact"], row["outcome"])
        for row in list_rollouts(episode_id=outcome.state.episode_id, db_path=host.db)
    }
    assert grades == {("world", "fail"), ("sim", "pass")}


def test_retrieve_injects_before_generate(tmp_path: Path):
    insert_record(
        kind="claim",
        title="Build uses pytest",
        body="The test runner is pytest.",
        provenance="world",
        db_path=tmp_path / "memory.db",
    )
    host = FakeHost(tmp_path, [_ok("ok", "world")])
    asyncio.run(
        run(
            host,
            EpisodeRequest(messages=[{"role": "user", "content": "how do we run pytest"}]),
        )
    )
    system = host.last_messages[0]["content"]
    assert "Build uses pytest" in system
    assert search_records("pytest", db_path=host.db)
