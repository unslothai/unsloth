# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Names the test that leaves a torch dynamo / inductor config entry deleted, then puts the entry back.

``del torch._dynamo.config.recompile_limit`` (also what ``unittest.mock.patch`` does on exit before it re-creates
the attribute) marks the entry hidden, and every later read in that process raises
``torch._dynamo.config.recompile_limit does not exist``. One leak on an xdist worker used to fail every compile
test that worker ran afterwards, with no hint of where it came from.

pytest 9.2 made that the normal outcome of ``monkeypatch.setattr(torch._dynamo.config, ...)``: for a non-class target
it now records the old value from the instance ``__dict__`` (pytest #10644), where a config entry never lives, so
``undo()`` deletes the entry instead of restoring it. ``install_monkeypatch_fix`` records the real value instead."""

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


def _is_torch_config(target: Any) -> bool:
    config_module = sys.modules.get("torch.utils._config_module")
    if config_module is None:
        return False
    kinds = tuple(
        k
        for k in (
            getattr(config_module, "ConfigModule", None),
            getattr(config_module, "SubConfigProxy", None),
        )
        if k
    )
    return bool(kinds) and isinstance(target, kinds)


def install_monkeypatch_fix() -> None:
    """Make MonkeyPatch.setattr on a torch config entry restore the value it replaced rather than delete the entry."""
    import pytest
    from _pytest import monkeypatch as mp_module

    cls = pytest.MonkeyPatch
    if getattr(cls.setattr, "_unsloth_torch_config_fix", False):
        return
    notset = getattr(mp_module, "NOTSET", None) or getattr(mp_module, "notset")
    original = cls.setattr

    def setattr(
        self,
        target,
        name = notset,
        value = notset,
        raising = True,
    ):
        obj, attr = target, name
        if isinstance(target, str) and value is notset:
            try:
                attr, obj = mp_module.derive_importpath(target, raising)
            except Exception:  # noqa: BLE001 - let the real setattr raise its own error
                obj = None
        real = (
            getattr(obj, attr, notset)
            if isinstance(attr, str) and _is_torch_config(obj)
            else notset
        )
        result = original(self, target, name, value, raising = raising)
        if real is not notset and self._setattr:
            last_obj, last_attr, last_old = self._setattr[-1]
            if last_obj is obj and last_attr == attr and last_old is notset:
                self._setattr[-1] = (last_obj, last_attr, real)
        return result

    setattr._unsloth_torch_config_fix = True
    cls.setattr = setattr


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
