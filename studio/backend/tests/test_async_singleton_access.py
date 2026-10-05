# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async handlers must not build the inference singleton on the event loop.

Construction runs get_default_models() -> hw.get_device(), so the first caller waits
for the background warm. Inline, that holds the event-loop thread for the whole torch
import, stalling login, liveness and the deadline-bound desktop health probe.

The offload has to stay at the call site, passing the route module's own
`get_inference_backend` to a thread. A helper in orchestrator.py would resolve that
module's global instead, bypassing callers that patch `routes.inference.get_inference_backend`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Every read below pins utf-8: Path.read_text() defaults to the locale encoding (cp1252
# on Windows), which cannot decode routes/inference.py, so these guards would raise
# instead of failing honestly.
_ROUTE_FILES = ("routes/inference.py", "routes/models.py")


def _async_call_sites(rel: str) -> list[str]:
    """Bare get_inference_backend() invocations inside an async def.
    `asyncio.to_thread(get_inference_backend)` passes the function object, an ast.Name and
    never an ast.Call, so only real on-loop invocations are reported."""
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
    """Guard against the sweep passing because the calls simply vanished. Counted off the
    AST: a literal-string count would report the offload gone the moment a formatter wraps
    one of these calls across lines."""
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
    # 13, not 14: the status poll's site became a non-constructing peek, which needs no
    # offload at all. Lower the floor only when a site is removed that way, never when
    # one goes back on the loop.
    assert total >= 13, f"expected the offloaded call sites to survive, found {total}"


def _sync_helpers_that_build_the_singleton(rel: str) -> set[str]:
    """Sync functions in this module that call get_inference_backend() inline."""
    tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
    names = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):  # sync only
            continue
        # The peek helper is the module's injection seam: it invokes the getter only
        # when that global has been patched, which is a test double, and otherwise
        # returns orchestrator.peek_inference_backend(). Reading it as a builder would
        # report every caller that deliberately stopped constructing.
        if fn.name == "_peek_inference_backend":
            continue
        for sub in ast.walk(fn):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "get_inference_backend"
            ):
                names.add(fn.name)
    return names


# Workers that call a callable they are handed, on the worker thread. to_thread runs only its
# first argument; a lambda further along is just passed to it, so it counts as off the loop only
# when that worker is known to invoke it there. Pinned below by reading the worker itself.
# Keyed by the qualified name the routes call it by, so an unrelated in_slot elsewhere is not
# trusted: "module.function" -> (its file, the name and position of the parameter it calls).
_WORKERS_THAT_RUN_THEIR_CALLBACK = {
    "model_slots.in_slot": ("core/inference/model_slots.py", "fn", 1),
}


def _executed_calls(lam: ast.Lambda) -> list[ast.Call]:
    # A lambda with a yield in its own body is a generator function: calling it runs nothing.
    if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _nodes_in_scope([lam.body])):
        return []
    return _calls_run_in_scope([lam.body])


def _nodes_in_scope(roots: list[ast.AST]) -> list[ast.AST]:
    found, stack = [], list(roots)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.GeneratorExp):
            # Only the outermost iterable is evaluated when the generator is built.
            stack.append(node.generators[0].iter)
            continue
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _calls_run_in_scope(roots: list[ast.AST]) -> list[ast.Call]:
    """Calls these nodes make when their own scope runs. A lambda, def or generator expression
    nested inside is only created there, and could be returned and run later on the loop, so it
    is not descended."""
    return [n for n in _nodes_in_scope(roots) if isinstance(n, ast.Call)]


def _calls_inside_offloaded_lambdas(fn: ast.AST) -> set[int]:
    """ids of the Call nodes in a lambda that asyncio.to_thread runs on its worker thread: the
    lambda is to_thread's first argument, or a later argument to a worker listed in
    _WORKERS_THAT_RUN_THEIR_CALLBACK. Any other lambda, stored or handed to a worker that might
    return it, is not exempt."""
    inside = set()
    for node in ast.walk(fn):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
            and node.args
        ):
            continue
        worker, *rest = node.args
        lambdas = [worker] if isinstance(worker, ast.Lambda) else []
        worker_name = (
            f"{worker.value.id}.{worker.attr}"
            if isinstance(worker, ast.Attribute) and isinstance(worker.value, ast.Name)
            else None
        )
        if worker_name in _WORKERS_THAT_RUN_THEIR_CALLBACK:
            _file, param, position = _WORKERS_THAT_RUN_THEIR_CALLBACK[worker_name]
            callback = [rest[position]] if len(rest) > position else []
            callback += [kw.value for kw in node.keywords if kw.arg == param]
            lambdas += [a for a in callback if isinstance(a, ast.Lambda)]
        for lam in lambdas:
            inside.update(id(n) for n in _executed_calls(lam))
    return inside


def test_no_async_handler_reaches_the_singleton_through_a_sync_helper():
    """The direct sweep is not enough: a sync helper hides the same stall. _loaded_satisfies
    calls get_inference_backend() inline, so an async handler calling it on the loop pays
    the cold build all the same, and walking only ast.AsyncFunctionDef misses that."""
    offenders = []
    for rel in _ROUTE_FILES:
        helpers = _sync_helpers_that_build_the_singleton(rel)
        if not helpers:
            continue
        tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            off_loop = _calls_inside_offloaded_lambdas(fn)
            for sub in ast.walk(fn):
                # A bare Call to the helper runs it on the loop; passing it to
                # to_thread makes it an ast.Name argument, never a Call. A call inside a
                # lambda handed to to_thread runs on the worker thread too (#11591's
                # get_active_generations does `to_thread(..., lambda: _loaded_satisfies(model))`).
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id in helpers
                    and id(sub) not in off_loop
                ):
                    offenders.append(f"{rel}:{sub.lineno} async {fn.name} -> {sub.func.id}()")

    # Empty on purpose. Both monitor helpers used to sit here as a known gap: they
    # reached the singleton through a sync helper and were not individually offloaded,
    # so they blocked during exactly the window this path exists to fix. Both now peek
    # instead. Do not add a name back without an offload or a justification here.
    known: set[str] = set()

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

    tests/test_orchestrator_unload_cancel.py patches routes.inference.get_inference_backend.
    An accessor defined in orchestrator.py resolves orchestrator's own global, so the patch
    would not take and the test hangs on a load gate that never opens."""
    orch = (_BACKEND / "core/inference/orchestrator.py").read_text(encoding = "utf-8")
    assert "async def get_inference_backend_async" not in orch, (
        "an async accessor in orchestrator.py bypasses callers that patch the "
        "route module's get_inference_backend"
    )


# The read-only surface: these answer "what is loaded" and must never be the reason a
# host imports torch. Each is polled from first paint or fired by a metadata-only
# action, so building the singleton here defeats UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1
# until a genuinely hardware-dependent operation runs.
_READ_ONLY_SITES = (
    ("routes/inference.py", "_monitor_active_model"),
    ("routes/inference.py", "get_status"),
    ("routes/models.py", "delete_finetuned_model"),
)


def test_read_only_endpoints_never_construct_the_singleton():
    """Peek, not build. A peek is a plain global read, so it needs no offload either."""
    offenders = []
    for rel, name in _READ_ONLY_SITES:
        tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
        fn = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
            ),
            None,
        )
        assert fn is not None, f"{rel}:{name} moved; update this guard"
        for sub in ast.walk(fn):
            # Both shapes: a bare call on the loop, and the name handed to to_thread,
            # which still constructs and still imports torch.
            if isinstance(sub, ast.Name) and sub.id == "get_inference_backend":
                offenders.append(f"{rel}:{sub.lineno} {name}")
    assert not offenders, (
        "read-only paths construct the inference singleton, so a status poll or a "
        "metadata-only delete imports torch on a warm-disabled host:\n  " + "\n  ".join(offenders)
    )


def test_a_lambda_handed_to_to_thread_is_off_the_loop_but_an_inline_call_is_not():
    """The exemption is exactly as wide as the offload: the same helper call on the loop, in a
    stored lambda, or in a lambda handed to a worker that may only return it is still reported."""
    fn = ast.parse(
        "async def handler(model):\n"
        "    await asyncio.to_thread(slots.in_slot, None, lambda: helper(model))\n"
        "    await asyncio.to_thread(lambda: helper(model))\n"
        "    stored = lambda: helper(model)\n"
        "    helper(model)\n"
        "    (await asyncio.to_thread(identity, lambda: helper(model)))()\n"
        "    (await asyncio.to_thread(lambda: lambda: helper(model)))()\n"
        "    await asyncio.to_thread(model_slots.in_slot, None, lambda: helper(model))\n"
        "    await asyncio.to_thread(other.in_slot, None, lambda: helper(model))\n"
        "    (await asyncio.to_thread(lambda: (helper(model) for _ in range(1)))).__next__()\n"
        "    await dispatcher.to_thread(lambda: helper(model))\n"
        "    (await asyncio.to_thread(lambda: (yield helper(model)))).__next__()\n"
        "    await asyncio.to_thread(lambda: (x for x in helper(model)))\n"
    ).body[0]
    off_loop = _calls_inside_offloaded_lambdas(fn)
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "helper"
    ]
    assert sorted(n.lineno for n in calls if id(n) in off_loop) == [3, 8, 13]
    assert sorted(n.lineno for n in calls if id(n) not in off_loop) == [
        2,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
    ]


@pytest.mark.parametrize("worker", sorted(_WORKERS_THAT_RUN_THEIR_CALLBACK))
def test_each_listed_worker_still_runs_the_callable_it_is_handed(worker):
    """The exemption trusts these workers to call their callback; read that off their source."""
    rel, param, position = _WORKERS_THAT_RUN_THEIR_CALLBACK[worker]
    tree = ast.parse((_BACKEND / rel).read_text(encoding = "utf-8"))
    module, name = worker.split(".")
    assert Path(rel).stem == module
    # Every scanned route that calls it by that name must have imported that module.
    for route in _ROUTE_FILES:
        source = (_BACKEND / route).read_text(encoding = "utf-8")
        if f"{worker}(" in source or f"{worker}," in source:
            assert (
                f"from core.inference import {module}\n" in source
            ), f"{route} calls {worker} but does not import it from {rel}"
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert [a.arg for a in fn.args.args][position] == param, f"{worker}'s callback moved"
    # A name bound to contextvars.copy_context() inside the worker: its .run is synchronous.
    contexts = {
        target.id
        for n in _nodes_in_scope(fn.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "copy_context"
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "contextvars"
        for target in n.targets
        if isinstance(target, ast.Name)
    }
    # Called directly, or run through contextvars' Context.run: forwarding it anywhere else could
    # return it uncalled, so it does not count.
    runs = [
        n
        for n in _calls_run_in_scope(fn.body)
        if (
            (isinstance(n.func, ast.Name) and n.func.id == param)
            or (
                isinstance(n.func, ast.Attribute)
                and n.func.attr == "run"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in contexts
                and n.args[:1]
                and isinstance(n.args[0], ast.Name)
                and n.args[0].id == param
            )
        )
    ]
    assert runs, f"{worker} no longer runs its {param} argument; drop it from the list"
