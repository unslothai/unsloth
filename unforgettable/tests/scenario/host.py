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

"""Scripted Host: real sandboxes, real memory tools, canned inner text."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from unforgettable.host import (
    EXTRACT_MAX_TOKENS,
    RUN_ACTION_NAMES,
    RUN_ACTION_TIMEOUT_SEC,
    SUPERVISE_MAX_TOKENS,
    GenerateRequest,
    GenerateResult,
    ToolTrace,
)
from unforgettable.store.records import list_records
from unforgettable.store.titles import normalize_title
from unforgettable.tools.handlers import dispatch

from .script import Move, Scene

PROJECT_ROOT = Path(__file__).resolve().parent / "project"


def seed_world(dest: Path, *, src: Path = PROJECT_ROOT) -> Path:
    dest.mkdir(parents = True, exist_ok = True)
    shutil.copytree(src, dest, dirs_exist_ok = True)
    return dest


class ScenarioHost:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.db = self.root / "memory.db"
        self.world = self.root / "world"
        seed_world(self.world)
        self.sims: dict[str, Path] = {}
        self.removed: list[str] = []
        self.calls: list[str] = []
        self.confirm_calls = 0
        self.confirm_result = True
        self.generate_messages: list = []
        self.scene_generates: dict[str, list] = {}
        self.last_messages = None
        self.supervise_calls: list = []
        self.snapshots: list[dict[str, Any]] = []
        self.scene_name = ""
        self._moves: list[Move] = []
        self._complete = ""
        self._filter_text = ""
        self._plan_text = ""

    def begin_scene(self, scene: Scene) -> None:
        self.scene_name = scene.name
        self._moves = list(scene.moves)
        self._complete = scene.complete_text
        self._filter_text = scene.filter_text
        self._plan_text = scene.plan_text
        self.confirm_result = scene.confirm_result

    def memory_db_path(self) -> Path:
        return self.db

    def world_session_id(self, request) -> str:
        del request
        return "world"

    def create_sim_session(self, episode_id: str) -> str:
        n = len(self.sims) + 1
        sid = f"sim-{episode_id}-{n}"
        path = self.root / sid
        path.mkdir()
        self.sims[sid] = path
        return sid

    def sandbox_path(self, session_id: str) -> Path:
        if session_id == "world":
            return self.world
        if session_id in self.sims:
            return self.sims[session_id]
        path = self.root / session_id
        path.mkdir(exist_ok = True)
        self.sims[session_id] = path
        return path

    def remove_sim_session(self, session_id: str) -> None:
        self.removed.append(session_id)
        path = self.sims.get(session_id) or (self.root / session_id)
        if path.is_dir() and session_id != "world":
            shutil.rmtree(path, ignore_errors = True)

    def _write_files(self, sandbox: Path, files: dict[str, str]) -> None:
        for rel, body in files.items():
            dest = sandbox / rel
            dest.parent.mkdir(parents = True, exist_ok = True)
            dest.write_text(body)

    def _snapshot(self, session_id: str) -> None:
        world_tax = self.world / "ledger" / "tax.py"
        session_tax = self.sandbox_path(session_id) / "ledger" / "tax.py"
        self.snapshots.append(
            {
                "scene": self.scene_name,
                "session": session_id,
                "is_sim": str(session_id).startswith("sim-"),
                "world_tax": world_tax.read_text() if world_tax.is_file() else "",
                "session_tax": session_tax.read_text() if session_tax.is_file() else "",
            }
        )

    def _dispatch_memory(self, item: dict[str, Any], contact: str) -> ToolTrace:
        tool = str(item.get("_tool") or "memory_write")
        args = {key: value for key, value in item.items() if not str(key).startswith("_")}
        if tool == "memory_supersede" and "id" not in args and item.get("_title"):
            title_key = normalize_title(str(item.get("_title") or ""))
            for rec in list_records(db_path = self.db):
                if rec.get("status") != "active":
                    continue
                if normalize_title(rec.get("title") or "") == title_key:
                    args["id"] = rec["id"]
                    break
        result = dispatch(tool, args)
        return ToolTrace(name = tool, arguments = args, result = str(result), contact = contact)

    def _contact_of(self, session_id: str) -> str:
        return "sim" if str(session_id).startswith("sim-") else "world"

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        self.calls.append(req.session_id)
        self.last_messages = req.messages
        self.generate_messages.append(req.messages)
        self.scene_generates.setdefault(self.scene_name, []).append(req.messages)
        if not self._moves:
            raise AssertionError(
                f"unexpected extra generate in scene {self.scene_name!r} session {req.session_id!r}"
            )
        move = self._moves.pop(0)
        sandbox = self.sandbox_path(req.session_id)
        contact = self._contact_of(req.session_id)
        self._write_files(sandbox, move.files)
        self._snapshot(req.session_id)
        traces: list[ToolTrace] = []
        if move.search:
            traces.append(
                self._dispatch_memory(
                    {"_tool": "memory_search", "query": move.search},
                    contact,
                )
            )
        if move.supersede_title:
            traces.append(
                self._dispatch_memory(
                    {
                        "_tool": "memory_supersede",
                        "_title": move.supersede_title,
                        "body": move.supersede_body,
                        "provenance": move.supersede_provenance,
                    },
                    contact,
                )
            )
        for item in move.memory:
            traces.append(self._dispatch_memory(item, contact))
        if move.terminal:
            result = await self.run_action(
                req.session_id,
                "terminal",
                {"command": move.terminal},
            )
            traces.append(
                ToolTrace(
                    name = "terminal",
                    arguments = {"command": move.terminal},
                    result = result,
                    contact = contact,
                )
            )
        return GenerateResult(text = move.text, tool_traces = traces, finished = move.finished)

    async def complete(
        self,
        messages,
        *,
        max_tokens = EXTRACT_MAX_TOKENS,
    ) -> str:
        del messages, max_tokens
        return self._complete

    async def supervise(
        self,
        purpose: str,
        messages,
        *,
        model = None,
        max_tokens = SUPERVISE_MAX_TOKENS,
    ) -> str:
        self.supervise_calls.append(
            {
                "purpose": purpose,
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        if purpose == "filter":
            return self._filter_text
        if purpose == "plan":
            return self._plan_text
        return ""

    async def run_action(
        self,
        session_id: str,
        name: str,
        arguments: dict,
        *,
        timeout: int | None = None,
        on_chunk = None,
    ) -> str:
        del on_chunk
        if name not in RUN_ACTION_NAMES:
            return f"Error: run_action supports python|terminal only, got {name!r}"
        sandbox = self.sandbox_path(session_id)
        effective = RUN_ACTION_TIMEOUT_SEC if timeout is None else timeout
        args = arguments or {}
        if name == "terminal":
            try:
                completed = subprocess.run(
                    args.get("command") or "",
                    shell = True,
                    cwd = sandbox,
                    capture_output = True,
                    text = True,
                    timeout = effective,
                )
            except subprocess.TimeoutExpired:
                return f"Execution timed out after {effective} seconds."
        else:
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", args.get("code") or ""],
                    cwd = sandbox,
                    capture_output = True,
                    text = True,
                    timeout = effective,
                )
            except subprocess.TimeoutExpired:
                return f"Execution timed out after {effective} seconds."
        text = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode:
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"exit code {completed.returncode}"
        return text

    async def confirm(
        self,
        prompt: str,
        *,
        kind: str = "retry_world",
        on_chunk = None,
        session_id: str | None = None,
    ) -> bool:
        del prompt, kind, on_chunk, session_id
        self.confirm_calls += 1
        return bool(self.confirm_result)


def system_text(messages: Optional[list]) -> str:
    if not messages:
        return ""
    parts = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            parts.append(str(msg.get("content") or ""))
    return "\n".join(parts)


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii = False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding = "utf-8")
