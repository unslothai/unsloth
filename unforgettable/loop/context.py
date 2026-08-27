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

from dataclasses import dataclass, field
from typing import Any, Optional

from unforgettable.constants import DEFAULT_NAMESPACE_ID, EVENT_SUMMARY_CHARS
from unforgettable.host import OnChunk, ToolTrace
from unforgettable.rims.types import ContactMode


def _clip_event_summary(summary: str, limit: int = EVENT_SUMMARY_CHARS) -> str:
    text = (summary or "").strip()
    if not text:
        return ""
    line = text.splitlines()[0].strip()
    if len(line) <= limit:
        return line
    return line[:limit].rstrip() + "..."


@dataclass
class EpisodeRequest:
    messages: list[dict[str, Any]]
    world_session_id: Optional[str] = None
    thread_id: Optional[str] = None
    stream: bool = True
    inner_model: Optional[str] = None
    namespace: str = DEFAULT_NAMESPACE_ID
    on_chunk: Optional[OnChunk] = None
    stakes: Optional[str] = None
    test_command: Optional[str] = None
    confirm_retry: Optional[bool] = None
    permission_mode: Optional[str] = None
    max_clones: Optional[int] = None
    max_sim_turns: Optional[int] = None
    skip_standing: bool = False
    adapter_id: Optional[str] = None
    shrink_standing: Optional[bool] = None
    planner: Optional[str] = None
    planner_model: Optional[str] = None
    filter: Optional[str] = None
    filter_model: Optional[str] = None
    judge_model: Optional[str] = None
    user_label: Optional[str] = None
    twin_plugin: Optional[str] = None


@dataclass
class EpisodeState:
    episode_id: str
    world_session: str
    contact: ContactMode = "world"
    sim_session: Optional[str] = None
    clone_count: int = 0
    sim_turns: int = 0
    had_world_failure: bool = False
    had_success_after_failure: bool = False
    traces: list[ToolTrace] = field(default_factory = list)
    trace_events: list[dict[str, Any]] = field(default_factory = list)
    keep_sim: bool = False
    test_command: Optional[str] = None
    created_sims: list[str] = field(default_factory = list)
    last_generate_text: str = ""
    last_fail_summary: str = ""
    last_sim_summary: str = ""
    planner_text: str = ""

    @property
    def active_session(self) -> str:
        if self.contact == "sim" and self.sim_session:
            return self.sim_session
        return self.world_session

    def track_sim(self, session_id: str) -> None:
        if session_id and session_id not in self.created_sims:
            self.created_sims.append(session_id)

    def enter_sim(self, session_id: str) -> None:
        self.track_sim(session_id)
        self.contact = "sim"
        self.sim_session = session_id
        self.clone_count += 1

    def enter_world(self) -> None:
        self.contact = "world"

    def note_failure(self, summary: str, contact: str) -> None:
        summary = _clip_event_summary(summary)
        if contact == "world":
            self.had_world_failure = True
        self.last_fail_summary = summary or self.last_fail_summary
        if contact == "sim":
            self.last_sim_summary = summary or self.last_sim_summary
        self.trace_events.append({"kind": "failure", "summary": summary, "contact": contact})

    def note_success(self, summary: str, contact: str) -> None:
        summary = _clip_event_summary(summary)
        if self.had_world_failure or any(ev.get("kind") == "failure" for ev in self.trace_events):
            self.had_success_after_failure = True
        if contact == "sim":
            self.last_sim_summary = summary or self.last_sim_summary
        self.trace_events.append({"kind": "success", "summary": summary, "contact": contact})


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
                    elif isinstance(part, str):
                        parts.append(part)
                return "\n".join(parts)
    return ""
