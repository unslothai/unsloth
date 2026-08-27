# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime patches for Unforgettable on top of unchanged Studio inference tools.

Keeps ``core.inference.tools`` matching upstream: memory/rims dispatch and
episode ``note_tool_result`` wrap ``execute_tool`` here instead of editing
every return in that module.
"""

from __future__ import annotations

import logging
from functools import wraps

_log = logging.getLogger(__name__)

_ALWAYS_SAFE_EXTRA = frozenset({"memory_search", "memory_get"})
_installed = False


def _is_memory_or_rims_tool(name: object) -> bool:
    if not isinstance(name, str):
        return False
    return (
        name.startswith("memory_")
        or name.startswith("memory.")
        or name.startswith("rims_")
        or name.startswith("rims.")
    )


def install() -> None:
    """Wrap Studio ``execute_tool`` and extend the always-safe name set.

    Idempotent. Call from app startup and from tests that do not boot ``main``.
    """
    global _installed
    if _installed:
        return

    from core.inference import tools as tools_mod

    original = tools_mod.execute_tool

    @wraps(original)
    def execute_tool(name, arguments, *args, **kwargs):
        if _is_memory_or_rims_tool(name):
            from unforgettable.tools.handlers import dispatch
            result = dispatch(name, arguments or {})
        else:
            result = original(name, arguments, *args, **kwargs)
        try:
            from unforgettable.loop.runtime import note_tool_result
            note_tool_result(name, arguments or {}, result)
        except Exception:
            _log.exception("unforgettable note_tool_result failed for %s", name)
        return result

    tools_mod.execute_tool = execute_tool
    tools_mod._ALWAYS_SAFE_TOOLS = frozenset(tools_mod._ALWAYS_SAFE_TOOLS) | _ALWAYS_SAFE_EXTRA
    _installed = True
