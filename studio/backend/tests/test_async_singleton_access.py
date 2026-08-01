# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async handlers must not build the inference singleton on the event loop.

Construction runs get_default_models() -> hw.get_device(), so the first caller waits
for the background warm. Inline, that holds the event-loop thread for the whole torch
import, stalling login, liveness and the deadline-bound desktop health probe.

The offload has to stay at the call site, passing the route module's own
`get_inference_backend` to a thread. A helper in orchestrator.py would resolve that
module's global instead, silently bypassing callers that patch
`routes.inference.get_inference_backend`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Every read below pins utf-8: Path.read_text() defaults to the locale encoding (cp1252
# on Windows), which cannot decode routes/inference.py, so these guards would raise
# instead of failing honestly.
_ROUTE_FILES = ("routes/inference.py", "routes/models.py")


def _async_call_sites(rel: str) -> list[str]:
    """Bare get_inference_backend() invocations inside an async def.

    `asyncio.to_thread(get_inference_backend)` passes the function object, an ast.Name
    and never an ast.Call, so only real on-loop invocations are reported.
    """
    tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        for sub in ast.walk(fn):
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
                continue
            if sub.func.id == "get_inference_backend":
                found.append(f"{rel}:{sub.lineno} in async {fn.name}")
    return found


def test_no_async_handler_builds_the_singleton_inline():
    offenders = [s for rel in _ROUTE_FILES for s in _async_call_sites(rel)]
    assert not offenders, "async handlers building the singleton inline:\n  " + "\n  ".join(
        offenders
    )


def test_the_offload_is_actually_present():
    """Guard against the sweep passing because the calls simply vanished.

    Counted off the AST: a literal-string count would report the offload gone the
    moment a formatter wraps one of these calls across lines.
    """
    total = 0
    for rel in _ROUTE_FILES:
        tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
        total += sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and any(isinstance(a, ast.Name) and a.id == "get_inference_backend" for a in node.args)
        )
    assert total >= 14, f"expected the offloaded call sites to survive, found {total}"


def _sync_helpers_that_build_the_singleton(rel: str) -> set[str]:
    """Sync functions in this module that call get_inference_backend() inline."""
    tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
    names = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):  # sync only
            continue
        for sub in ast.walk(fn):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "get_inference_backend"
            ):
                names.add(fn.name)
    return names


def test_no_async_handler_reaches_the_singleton_through_a_sync_helper():
    """The direct sweep is not enough: a sync helper hides the same stall.

    _loaded_satisfies calls get_inference_backend() inline, so an async handler calling
    it on the loop pays the cold build exactly as if it had called the getter itself.
    Walking only ast.AsyncFunctionDef misses that.
    """
    offenders = []
    for rel in _ROUTE_FILES:
        helpers = _sync_helpers_that_build_the_singleton(rel)
        if not helpers:
            continue
        tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for sub in ast.walk(fn):
                # A bare Call to the helper runs it on the loop; passing it to
                # to_thread makes it an ast.Name argument, never a Call.
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id in helpers
                ):
                    offenders.append(f"{rel}:{sub.lineno} async {fn.name} -> {sub.func.id}()")

    # These reach the singleton through a sync helper and are NOT individually
    # offloaded. A known gap, not an endorsement: the warm's hardware stage is the
    # multi-second torch import, so during exactly the window this path exists to fix,
    # these block. Do not grow the set without an offload or a justification here.
    known = {
        "_monitor_active_model",
        "_monitor_context_length",
    }

    # _resolves_to_resident is offloaded at its two singleton-reading call sites. The
    # third, in _openai_catalog_objects, passes llama_only = True, under which the
    # helper never evaluates the getter. This sweep matches on callee name and cannot
    # see that, so exempt by argument rather than blanket-exempting the helper.
    def _is_llama_only(site: str) -> bool:
        rel, rest = site.split(":", 1)
        lineno = int(rest.split(" ", 1)[0])
        tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_resolves_to_resident"
                and node.lineno == lineno
            ):
                return any(
                    kw.arg == "llama_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
        return False

    offenders = [o for o in offenders if not _is_llama_only(o)]
    new = [o for o in offenders if o.rsplit("-> ", 1)[-1].rstrip("()") not in known]
    assert not new, (
        "new async handlers reaching the singleton through a sync helper; "
        "offload at the call site rather than widening the baseline:\n  " + "\n  ".join(new)
    )


def test_the_offload_stays_at_the_call_site():
    """No orchestrator-level async helper: it would bypass patched route globals.

    tests/test_orchestrator_unload_cancel.py patches
    routes.inference.get_inference_backend. An accessor defined in orchestrator.py
    resolves orchestrator's own global, so the patch would not take and the test hangs
    on a load gate that never opens.
    """
    orch = (_BACKEND / "core/inference/orchestrator.py").read_text(encoding = "utf-8")
    assert "async def get_inference_backend_async" not in orch, (
        "an async accessor in orchestrator.py bypasses callers that patch the "
        "route module's get_inference_backend"
    )
