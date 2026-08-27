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

"""Twin plugin registry: location + tools for world and sim rims."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from unforgettable.eyes.protocols import RecognizedFailure
from unforgettable.host import OnChunk
from unforgettable.rims.types import ContactMode

TWIN_ENV = "UNFORGETTABLE_TWIN"
FS_COPY_ID = "fs.copy"
NONE_ID = "none"
DEFAULT_TWIN_PLUGIN = FS_COPY_ID
TWIN_PLUGIN_IDS = frozenset({FS_COPY_ID, NONE_ID})
UNKNOWN_TWIN_PLUGIN = "unknown twin plugin: {value}"


@dataclass(frozen = True)
class Location:
    plugin: str
    contact: ContactMode
    handle: str
    uri: str
    parent_handle: Optional[str] = None


@dataclass(frozen = True)
class TwinBinding:
    location: Location
    describe: str
    tool_specs: list[dict[str, Any]] = field(default_factory = list)


@dataclass(frozen = True)
class HarnessGrade:
    ran: bool
    failure: Optional[RecognizedFailure] = None


class TwinPlugin(Protocol):
    id: str

    def world(self, host, request) -> TwinBinding: ...

    def spawn_sim(
        self,
        host,
        world: TwinBinding,
        episode_id: str,
        *,
        clone_index: int = 1,
    ) -> TwinBinding: ...

    def cleanup(self, host, binding: TwinBinding) -> None: ...

    async def run(
        self,
        host,
        binding: TwinBinding,
        name: str,
        arguments: dict,
        *,
        timeout: int | None = None,
        on_chunk: OnChunk | None = None,
    ) -> str: ...

    async def grade(
        self,
        host,
        binding: TwinBinding,
        *,
        test_command: str | None,
        on_chunk: OnChunk | None = None,
    ) -> HarnessGrade: ...


def coerce_twin_plugin(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        env = (os.environ.get(TWIN_ENV) or "").strip()
        value = env or DEFAULT_TWIN_PLUGIN
    text = str(value).strip().lower()
    if text in {FS_COPY_ID, "fs", "copy", "filesystem"}:
        return FS_COPY_ID
    if text in {NONE_ID, "off", "verbal"}:
        return NONE_ID
    raise ValueError(UNKNOWN_TWIN_PLUGIN.format(value = value))


def get_twin_plugin(name: Any = None) -> TwinPlugin:
    key = coerce_twin_plugin(name)
    if key == NONE_ID:
        from unforgettable.rims.none import NonePlugin
        return NonePlugin()
    from unforgettable.rims.fs_copy import FsCopyPlugin

    return FsCopyPlugin()
