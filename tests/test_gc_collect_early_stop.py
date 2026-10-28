"""Repeated full gc.collect() passes stop once a pass finds nothing (and no gc.callbacks are registered).

A pass that finds no unreachable objects frees nothing and runs no finalizer, so the next pass walks the
same heap and finds nothing either; each full pass costs ~0.15 to 0.5 s on an Unsloth heap. CPU only: the
loops are sliced out of the source with ast and run against a fake gc, so nothing heavy is imported."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "unsloth"

# file -> enclosing function of every repeated-pass release loop
LOOPS = {
    "models/rl.py": ["_maybe_prepare_vllm_for_resume"],
    "models/loader_utils.py": ["_offline_quantize_to_fp8"],
    "dataprep/synthetic.py": ["cleanup"],
}


def _is_collect(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gc"
    )


def _function(tree, name):
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def _release_loops(function):
    """`for _ in range(N)` loops in `function` that call gc.collect()."""
    return [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Call)
        and getattr(n.iter.func, "id", None) == "range"
        and any(_is_collect(c) for c in ast.walk(n))
    ]


def _fake_gc(found, callbacks = ()):
    results = iter(found)
    calls = []

    def collect(*args):
        calls.append(args)
        return next(results)

    return SimpleNamespace(collect = collect, callbacks = list(callbacks)), calls


def _run(code, gc, cache_calls):
    clear = lambda *a, **k: cache_calls.append(1)
    exec(code, {"gc": gc, "clean_gpu_cache": clear, "range": range})


def _loop_cases():
    for rel, functions in LOOPS.items():
        tree = ast.parse((ROOT / rel).read_text(encoding = "utf-8"))
        for name in functions:
            loops = _release_loops(_function(tree, name))
            assert loops, f"{rel}:{name} has no gc.collect release loop any more"
            for i, loop in enumerate(loops):
                yield pytest.param(rel, loop, id = f"{rel}:{name}:{i}")


def _passes(loop):
    return loop.iter.args[0].value


@pytest.mark.parametrize("found_first", [0, 5])
@pytest.mark.parametrize("rel,loop", list(_loop_cases()))
def test_release_loop_stops_after_a_pass_that_finds_nothing(rel, loop, found_first):
    code = compile(ast.Module([loop], []), rel, "exec")
    n = _passes(loop)
    if found_first and n < 3:
        pytest.skip("a two-pass loop runs both passes when the first finds something")
    # Later passes would find something only if a quiet pass had changed the heap.
    found = [found_first, 0] + [9] * n if found_first else [0] + [9] * n
    expected = 2 if found_first else 1
    gc, calls = _fake_gc(found)
    cache = []
    _run(code, gc, cache)
    assert len(calls) == expected
    # The cache is still emptied once per collect that ran.
    assert len(cache) == expected


@pytest.mark.parametrize("rel,loop", list(_loop_cases()))
def test_release_loop_runs_every_pass_with_gc_callbacks(rel, loop):
    code = compile(ast.Module([loop], []), rel, "exec")
    n = _passes(loop)
    gc, calls = _fake_gc([0] * n, callbacks = [print])
    _run(code, gc, [])
    assert len(calls) == n


@pytest.mark.parametrize("rel,loop", list(_loop_cases()))
def test_release_loop_runs_every_pass_when_collect_is_replaced(rel, loop):
    # vLLM graph capture swaps gc.collect for `lambda: None`; None is not "found nothing".
    code = compile(ast.Module([loop], []), rel, "exec")
    n = _passes(loop)
    gc, calls = _fake_gc([None] * n)
    _run(code, gc, [])
    assert len(calls) == n


def _save_tree():
    return ast.parse((ROOT / "save.py").read_text(encoding = "utf-8"))


def _collect_garbage():
    fn = _function(_save_tree(), "_collect_garbage")
    namespace = {}
    exec(compile(ast.Module([fn], []), "save.py", "exec"), namespace)
    return namespace["_collect_garbage"], namespace


@pytest.mark.parametrize(
    "found,expected",
    [([5, 0, 7], 2), ([0, 9, 9], 1), ([4, 2, 1], 3), ([None, None, None], 3)],
)
def test_save_collect_garbage_stops_once_a_pass_finds_nothing(found, expected):
    fn, namespace = _collect_garbage()
    namespace["gc"], calls = _fake_gc(found)
    order = []
    fn(before = lambda: order.append("before"), after = lambda: order.append("after"))
    assert len(calls) == expected
    assert order == ["before", "after"] * expected


def test_save_collect_garbage_keeps_every_pass_with_gc_callbacks():
    fn, namespace = _collect_garbage()
    namespace["gc"], calls = _fake_gc([0, 0, 0], callbacks = [print])
    fn()
    assert len(calls) == 3


def test_save_has_no_unbounded_repeated_collect_loops():
    # Every repeated full collection in save.py goes through _collect_garbage.
    tree = _save_tree()
    helper = _function(tree, "_collect_garbage")
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)) or node is helper.body[0]:
            continue
        if node in ast.walk(helper):
            continue
        collects = [c for c in ast.walk(node) if _is_collect(c)]
        if collects:
            offenders.append(node.lineno)
    assert offenders == []


def test_unsloth_save_model_releases_state_dict_with_one_collect():
    # The per-entry release loop allocates nothing; the collect after it finds every cycle.
    fn = _function(_save_tree(), "unsloth_save_model")
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and ast.unparse(n.iter) == "enumerate(state_dict.items())"
    )
    assert not any(_is_collect(c) for c in ast.walk(loop))
    after = fn.body[fn.body.index(loop) + 1 :]
    assert any(_is_collect(c) for stmt in after[:4] for c in ast.walk(stmt))
