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

"""Host protocol: the only contract a UI / Studio adapter must implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol


OnChunk = Callable[[bytes], Optional[Awaitable[None]]]

# One-shot extract completion cap. Not the user's generate budget.
EXTRACT_MAX_TOKENS = 800
# One-shot supervisor (vote / plan / mine). Smaller than extract.
SUPERVISE_MAX_TOKENS = 400

RUN_ACTION_NAMES = frozenset({"python", "terminal"})
RUN_ACTION_TIMEOUT_SEC = 300
RUN_ACTION_CLIP = 200


@dataclass
class GenerateRequest:
    """Narrow DTO. Not a Studio ChatCompletionRequest."""

    messages: list[dict[str, Any]]
    session_id: str
    thread_id: Optional[str] = None
    stream: bool = True
    extra_tools: list[dict[str, Any]] = field(default_factory = list)
    inner_model: Optional[str] = None
    permission_mode: Optional[str] = None
    on_chunk: Optional[OnChunk] = None
    adapter_path: Optional[str] = None
    gguf_adapter_path: Optional[str] = None


@dataclass
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: str
    contact: str  # world | sim


@dataclass
class GenerateResult:
    text: str
    tool_traces: list[ToolTrace] = field(default_factory = list)
    finished: bool = True


class Host(Protocol):
    def memory_db_path(self) -> Path: ...

    def world_session_id(self, request: Any) -> str: ...

    def create_sim_session(self, episode_id: str) -> str: ...

    def sandbox_path(self, session_id: str) -> Path: ...

    def remove_sim_session(self, session_id: str) -> None: ...

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        """Run one inner-wheel pass. session_id selects the active rim sandbox."""
        ...

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = EXTRACT_MAX_TOKENS,
    ) -> str:
        """One-shot text completion. No tools, no memory inject, no act/sim.
        Used by llm_extract. Must not re-enter episode.run."""
        ...

    async def supervise(
        self,
        purpose: str,
        messages: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tokens: int = SUPERVISE_MAX_TOKENS,
    ) -> str:
        """Optional. One-shot vote/plan/mine/filter/judge. No tools, no episode loop.
        Missing → getattr skip (voter abstains, planner off, filter uses algo,
        judge uses algo)."""
        ...

    async def run_action(
        self,
        session_id: str,
        name: str,
        arguments: dict,
        *,
        timeout: int | None = None,
        on_chunk: OnChunk | None = None,
    ) -> str:
        """May be absent; episode.run uses getattr."""
        ...

    async def confirm(
        self,
        prompt: str,
        *,
        kind: str = "retry_world",
        on_chunk: OnChunk | None = None,
        session_id: str | None = None,
    ) -> bool:
        """May be absent; missing + required → ESCALATE."""
        ...
