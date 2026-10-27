# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Names the test that leaves a torch dynamo / inductor config entry deleted, then puts the entry back.

``del torch._dynamo.config.recompile_limit`` (also what ``unittest.mock.patch`` does on exit before it re-creates
the attribute) marks the entry hidden, and every later read in that process raises
``torch._dynamo.config.recompile_limit does not exist``. One leak on an xdist worker used to fail every compile
test that worker ran afterwards, with no hint of where it came from."""

from __future__ import annotations

import sys
import threading
import traceback
from typing import Any

CONFIG_MODULES = (
    "torch._dynamo.config",
    "torch._inductor.config",
    "torch._functorch.config",
)

_LOCK = threading.Lock()
# (module, key) -> (user_override before the delete, stack of the delete)
_DELETES: dict[tuple[str, str], tuple[Any, str]] = {}
_installed = False


def _entries(module_name: str) -> dict[str, Any] | None:
    module = sys.modules.get(module_name)
    # vars(): a ConfigModule answers unknown attributes through __getattr__.
    entries = vars(module).get("_config") if module is not None else None
    return entries if isinstance(entries, dict) else None


def install_delete_recorder() -> bool:
    """Wrap ConfigModule.__delattr__ to remember who deleted what; a no-op until torch is imported."""
    global _installed
    if _installed:
        return True
    config_module = sys.modules.get("torch.utils._config_module")
    cls = getattr(config_module, "ConfigModule", None) if config_module is not None else None
    if cls is None:
        return False
    original = cls.__delattr__

    def __delattr__(self, name: str) -> None:
        entries = vars(self).get("_config")
        entry = entries.get(name) if isinstance(entries, dict) else None
        if entry is not None:
            with _LOCK:
                _DELETES[(self.__name__, name)] = (
                    entry.user_override,
                    "".join(traceback.format_stack(limit = 16)[:-1]),
                )
        return original(self, name)

    cls.__delattr__ = __delattr__
    _installed = True
    return True


def hidden_entries() -> list[tuple[str, str]]:
    """Config entries that are currently deleted (read as ``... does not exist``)."""
    found = []
    for module_name in CONFIG_MODULES:
        entries = _entries(module_name)
        for key, entry in (entries or {}).items():
            if getattr(entry, "hide", False):
                found.append((module_name, key))
    return found


def restore(hidden: list[tuple[str, str]]) -> str:
    """Un-hide ``hidden`` with the value each held before its delete; returns a report naming the deleters."""
    lines = []
    for module_name, key in hidden:
        entry = (_entries(module_name) or {}).get(key)
        if entry is None:
            continue
        with _LOCK:
            prior = _DELETES.pop((module_name, key), None)
        if prior is not None:
            entry.user_override = prior[0]
        entry.hide = False
        module = sys.modules.get(module_name)
        if module is not None:
            vars(module)["_is_dirty"] = True
        lines.append(f"{module_name}.{key} was deleted and never re-created")
        lines.append(
            prior[1] if prior is not None else "  (deleted before the recorder was installed)\n"
        )
    return "\n".join(lines)


def check_after_test() -> None:
    """Raise if the test just run left a config entry deleted; the entries are restored first."""
    install_delete_recorder()
    hidden = hidden_entries()
    if hidden:
        report = restore(hidden)
        raise AssertionError(
            "this test left torch config entries deleted, which breaks every later compile in the process:\n"
            + report
        )
