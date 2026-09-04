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

"""Verbal-only twin. No clone, no extra disk. PoC escape hatch."""

from __future__ import annotations

from pathlib import Path

from unforgettable.host import OnChunk
from unforgettable.rims.plugin import (
    NONE_ID,
    HarnessGrade,
    Location,
    TwinBinding,
)

WORLD_DESCRIBE = (
    "You are in the world. There is no cloned twin; python and terminal "
    "still run in the world sandbox if the host provides them."
)
SIM_DESCRIBE = (
    "Rehearse in text only; no cloned environment. Do not assume filesystem "
    "tools write to a twin. World remains at {world_uri}."
)
NONE_ACTION = "Error: none twin has no actions"
SIM_URI = "sim://none"


class NonePlugin:
    id = NONE_ID

    def world(self, host, request) -> TwinBinding:
        handle = getattr(request, "world_session_id", None) or host.world_session_id(request)
        uri = "world://none"
        sandbox = getattr(host, "sandbox_path", None)
        if callable(sandbox):
            try:
                uri = Path(sandbox(handle)).resolve().as_uri()
            except Exception:
                uri = f"world://{handle}"
        return TwinBinding(
            location = Location(
                plugin = self.id,
                contact = "world",
                handle = str(handle),
                uri = uri,
            ),
            describe = WORLD_DESCRIBE,
        )

    def spawn_sim(
        self,
        host,
        world: TwinBinding,
        episode_id: str,
        *,
        clone_index: int = 1,
    ) -> TwinBinding:
        del host
        handle = f"sim-none-{episode_id[:8]}-{clone_index}"
        return TwinBinding(
            location = Location(
                plugin = self.id,
                contact = "sim",
                handle = handle,
                uri = SIM_URI,
                parent_handle = world.location.handle,
            ),
            describe = SIM_DESCRIBE.format(world_uri = world.location.uri),
        )

    def cleanup(self, host, binding: TwinBinding) -> None:
        del host, binding

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
        del host, binding, name, arguments, timeout, on_chunk
        return NONE_ACTION

    async def grade(
        self,
        host,
        binding: TwinBinding,
        *,
        test_command: str | None,
        on_chunk: OnChunk | None = None,
    ) -> HarnessGrade:
        del host, binding, test_command, on_chunk
        return HarnessGrade(ran = False)
