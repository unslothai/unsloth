# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The status field a client reads before declining the resident-model shortcut.

``spec_binary_fallback_can_retry`` needs a different llama-server installed before an
identical /load can repair a binary stand-down. The chat UI cannot see that, so it
reloaded (and prompted to stop running chats) for every re-pick of a model whose drafter
stood down, for a load the backend would have deduplicated.

``_spec_fallback_binary_changed`` publishes the cheap half of that predicate. It is
answered ONLY for the two binary reasons: /api/inference/status is polled from first
paint, and the binary lookup has no business running on every poll of a healthy runtime.

The helper is extracted from the route module's source rather than imported, so the test
costs nothing and does not drag FastAPI in behind it.
"""

from __future__ import annotations

import ast
import sys
import types as _types
from pathlib import Path
from typing import Optional

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# conftest's autouse fixture imports core.inference.llama_cpp, which wants these.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

_ROUTE = Path(__file__).resolve().parent.parent / "routes" / "inference.py"
_NAME = "_spec_fallback_binary_changed"


def _load_helper():
    tree = ast.parse(_ROUTE.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == _NAME:
            namespace: dict = {"Optional": Optional}
            exec(compile(ast.Module([node], []), str(_ROUTE), "exec"), namespace)
            return namespace[_NAME]
    raise AssertionError(f"{_NAME} is gone; the status no longer reports it")


class _Backend:
    def __init__(
        self,
        reason,
        changed = False,
        raises = False,
    ):
        self.spec_fallback_reason = reason
        self._changed = changed
        self._raises = raises
        self.calls = 0

    def _binary_changed_since_launch(self):
        self.calls += 1
        if self._raises:
            raise RuntimeError("binary lookup failed")
        return self._changed


def test_answers_only_for_the_two_binary_reasons():
    helper = _load_helper()
    for reason in (
        None,
        "drafter_not_found",
        "drafter_no_vram",
        "runtime_error",
        "mla_mtp_disabled",
    ):
        backend = _Backend(reason)
        assert helper(backend) is None
        # The point of the gate: no binary lookup on a poll that cannot need one.
        assert backend.calls == 0


def test_reports_whether_the_binary_moved():
    helper = _load_helper()
    for reason in ("binary_no_mtp", "binary_outdated"):
        assert helper(_Backend(reason, changed = False)) is False
        assert helper(_Backend(reason, changed = True)) is True


def test_an_unreadable_binary_is_unknown_not_false():
    # False would tell the client the drafter cannot be repaired and suppress the reload
    # an update was meant to enable; None leaves it with the coarser answer.
    helper = _load_helper()
    assert helper(_Backend("binary_no_mtp", raises = True)) is None
