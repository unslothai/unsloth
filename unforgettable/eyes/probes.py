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
import inspect
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from unforgettable.eyes.basic import grade_run_action
from unforgettable.eyes.gate import LogGateEyes
from unforgettable.host import RUN_ACTION_TIMEOUT_SEC
from unforgettable.rims.clone import clone_tree
from unforgettable.rims.detect import first_nonempty_line
from unforgettable.rims.fs_copy import FsCopyPlugin
from unforgettable.store.records import list_records

PROBE_TITLE_PREFIX = "probe:"
MAX_EPISODE_PROBES = 3


def is_probe_title(title: str) -> bool:
    return (title or "").strip().casefold().startswith(PROBE_TITLE_PREFIX)


def list_probes(db_path = None) -> list[dict]:
    rows = []
    for rec in list_records(kinds = ["procedure"], statuses = ["active"], db_path = db_path):
        if not is_probe_title(rec["title"]):
            continue
        rows.append({**rec, "command": first_nonempty_line(rec["body"])})
    return rows


def run_probes(
    *,
    world,
    host = None,
    db_path = None,
    limit = None,
    on_chunk = None,
) -> list[dict]:
    coro = _run_probes(world = world, host = host, db_path = db_path, limit = limit, on_chunk = on_chunk)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return coro


async def _run_probes(
    *,
    world,
    host = None,
    db_path = None,
    limit = None,
    on_chunk = None,
) -> list[dict]:
    rows = list_probes(db_path = db_path)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    world_path = Path(world)
    out: list[dict] = []
    for rec in rows:
        title = rec["title"]
        command = rec.get("command") or ""
        try:
            passed = await _execute_probe(
                world = world_path, command = command, host = host, on_chunk = on_chunk
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            passed = False
        outcome = "pass" if passed else "fail"
        LogGateEyes().note(f"probe: {title} {outcome}", db_path = db_path)
        out.append({**rec, "outcome": outcome})
    return out


async def _execute_probe(*, world: Path, command: str, host, on_chunk) -> bool:
    if host is None:
        return _execute_local(world, command)
    plugin = FsCopyPlugin()
    binding = plugin.spawn_from_world_path(host, Path(world), f"probe-{uuid.uuid4().hex}")
    try:
        result = await plugin.run(
            host,
            binding,
            "terminal",
            {"command": command},
            on_chunk = on_chunk,
        )
        return grade_run_action("terminal", result) is None
    finally:
        plugin.cleanup(host, binding)


def _execute_local(world: Path, command: str) -> bool:
    tmp = tempfile.mkdtemp(prefix = "unforgettable-probe-")
    try:
        clone_tree(world, tmp)
        return grade_run_action("terminal", _local_terminal(tmp, command)) is None
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


def _local_terminal(cwd: Path, command: str) -> str:
    try:
        completed = subprocess.run(
            command,
            shell = True,
            cwd = cwd,
            capture_output = True,
            text = True,
            timeout = RUN_ACTION_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {RUN_ACTION_TIMEOUT_SEC} seconds."
    text = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode:
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"exit code {completed.returncode}"
    return text
