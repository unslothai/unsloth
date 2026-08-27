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

"""Contextvars so memory tools and execute_tool can see the active episode."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from unforgettable.constants import DEFAULT_NAMESPACE_ID
from unforgettable.host import ToolTrace

_db_path: ContextVar[Optional[str]] = ContextVar("unforgettable_db_path", default = None)
_episode_id: ContextVar[Optional[str]] = ContextVar("unforgettable_episode_id", default = None)
_namespace: ContextVar[str] = ContextVar("unforgettable_namespace", default = DEFAULT_NAMESPACE_ID)
_contact: ContextVar[str] = ContextVar("unforgettable_contact", default = "world")
_traces: ContextVar[Optional[list[ToolTrace]]] = ContextVar("unforgettable_traces", default = None)
_filter_stripped: ContextVar[tuple] = ContextVar("unforgettable_filter_stripped", default = ())
_user_label: ContextVar[Optional[str]] = ContextVar("unforgettable_user_label", default = None)


def current_db_path() -> Optional[str]:
    return _db_path.get()


def current_episode_id() -> Optional[str]:
    return _episode_id.get()


def current_namespace() -> str:
    return _namespace.get()


def current_contact() -> str:
    return _contact.get()


def current_traces() -> list:
    return list(_traces.get() or [])


def current_filter_stripped() -> tuple:
    return _filter_stripped.get() or ()


def set_filter_stripped(spans: tuple) -> None:
    _filter_stripped.set(tuple(spans or ()))


def current_user_label() -> Optional[str]:
    return _user_label.get()


def set_user_label(label: Optional[str]) -> None:
    _user_label.set(label)


def note_tool_result(name: str, arguments: dict, result: str) -> None:
    traces = _traces.get()
    if traces is None:
        return
    traces.append(
        ToolTrace(
            name = name,
            arguments = dict(arguments or {}),
            result = str(result),
            contact = current_contact(),
        )
    )


def bind_episode(
    *,
    db_path: str,
    episode_id: str,
    namespace: str = DEFAULT_NAMESPACE_ID,
):
    """Set episode locals. Returns the tokens + a fresh trace list."""
    traces: list[ToolTrace] = []
    tokens = (
        _db_path.set(db_path),
        _episode_id.set(episode_id),
        _namespace.set(namespace),
        _traces.set(traces),
        _contact.set("world"),
    )
    return tokens, traces


def set_contact(mode: str) -> None:
    _contact.set(mode)


def reset_episode(tokens) -> None:
    _db_path.reset(tokens[0])
    _episode_id.reset(tokens[1])
    _namespace.reset(tokens[2])
    _traces.reset(tokens[3])
    _contact.reset(tokens[4])
