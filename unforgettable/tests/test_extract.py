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
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from unforgettable.host import EXTRACT_MAX_TOKENS, ToolTrace
from unforgettable.loop.context import EpisodeState
from unforgettable.loop.episode import _extract
from unforgettable.store.records import list_records

TWO_DRAFTS = [
    {
        "kind": "claim",
        "title": "Tests use pytest",
        "body": "The suite is pytest.",
        "provenance": "world",
    },
    {
        "kind": "procedure",
        "title": "Retry after lint",
        "body": "Re-run after fixing lint.",
        "provenance": "human",
    },
]


class ExtractHost:
    def __init__(self, db: Path, text: str):
        self.db = db
        self._text = text
        self.complete_calls = 0

    def memory_db_path(self) -> Path:
        return self.db

    async def complete(
        self,
        messages,
        *,
        max_tokens = EXTRACT_MAX_TOKENS,
    ) -> str:
        self.complete_calls += 1
        return self._text


class NoCompleteHost:
    def __init__(self, db: Path):
        self.db = db

    def memory_db_path(self) -> Path:
        return self.db


def _failed_then_fixed_state() -> EpisodeState:
    state = EpisodeState(
        episode_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        world_session = "world",
    )
    state.traces = [
        ToolTrace("terminal", {"command": "false"}, "exit code 1", "world"),
        ToolTrace("terminal", {"command": "true"}, "ok\n", "world"),
    ]
    state.note_failure("exit code 1", "world")
    state.note_success("ok", "world")
    return state


def _run_extract(host, state: EpisodeState | None = None) -> EpisodeState:
    state = state or _failed_then_fixed_state()
    asyncio.run(
        _extract(
            state,
            str(host.memory_db_path()),
            last_user = "run the tests",
            actions = ["act"],
            host = host,
        )
    )
    return state


def _infer_rows(db_path) -> list[dict]:
    return [row for row in list_records(db_path = db_path) if row["provenance"] == "infer"]


def test_two_well_formed_drafts_are_proposed_infer(tmp_path: Path):
    host = ExtractHost(tmp_path / "memory.db", json.dumps(TWO_DRAFTS))
    _run_extract(host)
    infer = _infer_rows(host.db)
    assert len(infer) == 2
    assert {row["title"] for row in infer} == {"Tests use pytest", "Retry after lint"}
    assert all(row["status"] == "proposed" for row in infer)
    assert all(row["provenance"] == "infer" for row in infer)


def test_malformed_json_writes_no_infer_rows(tmp_path: Path):
    host = ExtractHost(tmp_path / "memory.db", "not-json {{")
    _run_extract(host)
    assert _infer_rows(host.db) == []


def test_model_provenance_is_overwritten_to_infer(tmp_path: Path):
    host = ExtractHost(
        tmp_path / "memory.db",
        json.dumps(
            [
                {
                    "kind": "claim",
                    "title": "World fact",
                    "body": "The model claimed world provenance.",
                    "provenance": "world",
                }
            ]
        ),
    )
    _run_extract(host)
    infer = _infer_rows(host.db)
    assert len(infer) == 1
    assert infer[0]["provenance"] == "infer"
    assert infer[0]["status"] == "proposed"


def test_directive_kind_is_dropped(tmp_path: Path):
    host = ExtractHost(
        tmp_path / "memory.db",
        json.dumps(
            [
                {
                    "kind": "directive",
                    "title": "Always cite ids",
                    "body": "User-only kind.",
                },
                {
                    "kind": "claim",
                    "title": "Keep this",
                    "body": "Allowed kind.",
                },
            ]
        ),
    )
    _run_extract(host)
    infer = _infer_rows(host.db)
    assert len(infer) == 1
    assert infer[0]["kind"] == "claim"
    assert infer[0]["title"] == "Keep this"
    assert not any(row["kind"] == "directive" for row in list_records(db_path = host.db))


def test_naive_from_episode_still_runs(tmp_path: Path):
    host = ExtractHost(tmp_path / "memory.db", json.dumps(TWO_DRAFTS))
    _run_extract(host)
    fixes = list_records(kinds = ["error_fix"], db_path = host.db)
    assert len(fixes) == 1
    assert fixes[0]["title"].startswith("Error then fix")
    assert fixes[0]["status"] == "proposed"
    infer = _infer_rows(host.db)
    assert len(infer) == 2


def test_host_without_complete_skips_llm_path(tmp_path: Path):
    host = NoCompleteHost(tmp_path / "memory.db")
    _run_extract(host)
    assert _infer_rows(host.db) == []
    fixes = list_records(kinds = ["error_fix"], db_path = host.db)
    assert len(fixes) == 1
    episodes = list_records(kinds = ["episode"], db_path = host.db)
    assert len(episodes) == 1


def test_studio_host_complete_is_one_shot_no_tools():
    path = (
        Path(__file__).resolve().parents[2]
        / "studio"
        / "backend"
        / "core"
        / "unforgettable_host.py"
    )
    if not path.is_file():
        pytest.skip("StudioHost not present")
    spec = importlib.util.spec_from_file_location("unforgettable_studio_host_under_test", path)
    if spec is None or spec.loader is None:
        pytest.skip("StudioHost import is heavy")
    backend = str(path.parents[1])
    if backend not in sys.path:
        sys.path.insert(0, backend)
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("StudioHost import is heavy")

    mod._as_chat_messages = lambda messages: messages

    class _Payload:
        def __init__(self):
            self.model = "unforgettable"
            self.stream = True
            self.enable_tools = True
            self.mcp_enabled = True
            self.tools = [{"type": "function", "function": {"name": "terminal"}}]
            self.tool_choice = "auto"
            self.max_tokens = 64
            self.max_completion_tokens = 8192
            self.messages = []
            self.session_id = "world"
            self.thread_id = None

        def model_copy(self, deep = True):
            clone = _Payload()
            clone.model = self.model
            clone.stream = self.stream
            clone.enable_tools = self.enable_tools
            clone.mcp_enabled = self.mcp_enabled
            clone.tools = list(self.tools) if self.tools is not None else None
            clone.tool_choice = self.tool_choice
            clone.max_tokens = self.max_tokens
            clone.max_completion_tokens = self.max_completion_tokens
            clone.messages = list(self.messages)
            clone.session_id = self.session_id
            clone.thread_id = self.thread_id
            return clone

    seen: dict = {}

    async def inner(payload, request, subject):
        seen["stream"] = payload.stream
        seen["enable_tools"] = payload.enable_tools
        seen["mcp_enabled"] = payload.mcp_enabled
        seen["tools"] = payload.tools
        seen["tool_choice"] = payload.tool_choice
        seen["model"] = payload.model
        seen["max_tokens"] = payload.max_tokens
        seen["max_completion_tokens"] = payload.max_completion_tokens
        seen["inner"] = mod.in_inner_generate()
        try:
            from state.tool_policy import get_tool_policy
            seen["tool_policy"] = get_tool_policy()
        except Exception:
            seen["tool_policy"] = None
        return {"choices": [{"message": {"content": "[]"}}]}

    source = _Payload()
    host = mod.StudioHost(
        source,
        request = None,
        current_subject = "u",
        inner = inner,
        inner_model = "qwen-inner",
    )
    text = asyncio.run(host.complete([{"role": "user", "content": "x"}]))
    assert text == "[]"
    assert seen["stream"] is False
    assert seen["enable_tools"] is False
    assert seen["mcp_enabled"] is False
    assert seen["tools"] is None
    assert seen["tool_choice"] == "none"
    assert seen["model"] == "qwen-inner"
    assert seen["max_tokens"] == EXTRACT_MAX_TOKENS
    assert seen["max_completion_tokens"] == EXTRACT_MAX_TOKENS


def test_studio_host_supervise_uses_planner_model():
    path = (
        Path(__file__).resolve().parents[2]
        / "studio"
        / "backend"
        / "core"
        / "unforgettable_host.py"
    )
    if not path.is_file():
        pytest.skip("StudioHost not present")
    spec = importlib.util.spec_from_file_location("unforgettable_studio_host_supervise", path)
    if spec is None or spec.loader is None:
        pytest.skip("StudioHost import is heavy")
    backend = str(path.parents[1])
    if backend not in sys.path:
        sys.path.insert(0, backend)
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("StudioHost import is heavy")

    from unforgettable.host import SUPERVISE_MAX_TOKENS

    mod._as_chat_messages = lambda messages: messages

    class _Payload:
        def __init__(self):
            self.model = "unforgettable"
            self.stream = True
            self.enable_tools = True
            self.mcp_enabled = True
            self.tools = [{"type": "function", "function": {"name": "terminal"}}]
            self.tool_choice = "auto"
            self.max_tokens = 64
            self.max_completion_tokens = 8192
            self.messages = []
            self.session_id = "world"
            self.thread_id = None
            self.planner_model = "large-planner"

        def model_copy(self, deep = True):
            clone = _Payload()
            clone.model = self.model
            clone.stream = self.stream
            clone.enable_tools = self.enable_tools
            clone.mcp_enabled = self.mcp_enabled
            clone.tools = list(self.tools) if self.tools is not None else None
            clone.tool_choice = self.tool_choice
            clone.max_tokens = self.max_tokens
            clone.max_completion_tokens = self.max_completion_tokens
            clone.messages = list(self.messages)
            clone.session_id = self.session_id
            clone.thread_id = self.thread_id
            clone.planner_model = self.planner_model
            return clone

    seen: dict = {}

    async def inner(payload, request, subject):
        seen["model"] = payload.model
        seen["enable_tools"] = payload.enable_tools
        seen["max_tokens"] = payload.max_tokens
        seen["inner"] = mod.in_inner_generate()
        return {"choices": [{"message": {"content": "1. run tests"}}]}

    host = mod.StudioHost(
        _Payload(),
        request = None,
        current_subject = "u",
        inner = inner,
        inner_model = "qwen-inner",
    )
    text = asyncio.run(host.supervise("plan", [{"role": "user", "content": "fix it"}]))
    assert text == "1. run tests"
    assert seen["model"] == "large-planner"
    assert seen["enable_tools"] is False
    assert seen["max_tokens"] == SUPERVISE_MAX_TOKENS
    assert seen["inner"] is True
    assert mod.in_inner_generate() is False
