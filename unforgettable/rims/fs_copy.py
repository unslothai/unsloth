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

"""Filesystem-copy twin: today's coding-domain sim, named as a plugin."""

from __future__ import annotations

import inspect
from pathlib import Path

from unforgettable.eyes.basic import grade_run_action
from unforgettable.host import RUN_ACTION_NAMES, OnChunk
from unforgettable.rims.clone import clone_tree
from unforgettable.rims.plugin import (
    FS_COPY_ID,
    HarnessGrade,
    Location,
    TwinBinding,
)

WORLD_DESCRIBE = (
    "You are in the project sandbox at {uri}. python and terminal run here; this is the world."
)
SIM_DESCRIBE = (
    "You are in a disposable filesystem clone of the world tree at {uri}. "
    "World remains at {world_uri}. Writes here do not touch the world. "
    "Same python and terminal tools."
)
SHARE_WORLD = "refusing to share world sandbox as sim: {sid!r}"
NONE_ACTION = "Error: fs.copy run_action supports python|terminal only, got {name!r}"


def _uri(path: Path) -> str:
    return Path(path).resolve().as_uri()


class FsCopyPlugin:
    id = FS_COPY_ID

    def world(self, host, request) -> TwinBinding:
        handle = getattr(request, "world_session_id", None) or host.world_session_id(request)
        path = host.sandbox_path(handle)
        uri = _uri(path)
        return TwinBinding(
            location = Location(
                plugin = self.id,
                contact = "world",
                handle = str(handle),
                uri = uri,
            ),
            describe = WORLD_DESCRIBE.format(uri = uri),
        )

    def spawn_sim(
        self,
        host,
        world: TwinBinding,
        episode_id: str,
        *,
        clone_index: int = 1,
    ) -> TwinBinding:
        del clone_index
        sid = host.create_sim_session(episode_id)
        world_handle = world.location.handle
        try:
            if not sid or sid == world_handle or str(sid).startswith("project-"):
                raise ValueError(SHARE_WORLD.format(sid = sid))
            dest = host.sandbox_path(sid)
            clone_tree(host.sandbox_path(world_handle), dest)
        except Exception:
            if sid and sid != world_handle:
                host.remove_sim_session(sid)
            raise
        uri = _uri(dest)
        return TwinBinding(
            location = Location(
                plugin = self.id,
                contact = "sim",
                handle = sid,
                uri = uri,
                parent_handle = world_handle,
            ),
            describe = SIM_DESCRIBE.format(uri = uri, world_uri = world.location.uri),
        )

    def spawn_from_world_path(self, host, world: Path, episode_id: str) -> TwinBinding:
        sid = host.create_sim_session(episode_id)
        try:
            if not sid or str(sid).startswith("project-"):
                raise ValueError(SHARE_WORLD.format(sid = sid))
            dest = host.sandbox_path(sid)
            clone_tree(world, dest)
        except Exception:
            if sid:
                host.remove_sim_session(sid)
            raise
        uri = _uri(dest)
        return TwinBinding(
            location = Location(
                plugin = self.id,
                contact = "sim",
                handle = sid,
                uri = uri,
                parent_handle = None,
            ),
            describe = SIM_DESCRIBE.format(uri = uri, world_uri = _uri(Path(world))),
        )

    def cleanup(self, host, binding: TwinBinding) -> None:
        handle = binding.location.handle
        if handle and binding.location.contact == "sim":
            host.remove_sim_session(handle)

    async def run(
        self,
        host,
        binding: TwinBinding,
        name: str,
        arguments: dict,
        *,
        timeout: int | None = None,
        on_chunk: OnChunk | None = None,
    ) -> str:
        if name not in RUN_ACTION_NAMES:
            return NONE_ACTION.format(name = name)
        run_fn = getattr(host, "run_action", None)
        if run_fn is None:
            return "Error: host has no run_action"
        result = run_fn(
            binding.location.handle,
            name,
            arguments or {},
            timeout = timeout,
            on_chunk = on_chunk,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    async def grade(
        self,
        host,
        binding: TwinBinding,
        *,
        test_command: str | None,
        on_chunk: OnChunk | None = None,
    ) -> HarnessGrade:
        cmd = (test_command or "").strip()
        if not cmd or getattr(host, "run_action", None) is None:
            return HarnessGrade(ran = False)
        result = await self.run(
            host,
            binding,
            "terminal",
            {"command": cmd},
            on_chunk = on_chunk,
        )
        return HarnessGrade(
            ran = True,
            failure = grade_run_action("terminal", result, contact = "sim"),
        )
