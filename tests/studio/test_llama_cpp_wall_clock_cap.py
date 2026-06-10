"""
Tests for the llama-server wall-clock cap (t_max_predict_ms).

The UI always sends max_tokens = context_length, so gating
t_max_predict_ms on `max_tokens is None` makes the safety net dead
code. The fix applies the wall-clock cap unconditionally on all three
streaming payload sites and raises the default to 10 minutes so slow
CPU / macOS / Windows installs are not cut off mid-generation.

Verifies:
  - t_max_predict_ms is assigned unconditionally at the three
    payload-builder sites (not inside an `if max_tokens is None` else
    branch).
  - _DEFAULT_T_MAX_PREDICT_MS is at least 10 minutes (previously
    120_000).
  - The default max_tokens path still applies _DEFAULT_MAX_TOKENS.
  - The three payload variable names (payload x2, stream_payload x1)
    each get both `max_tokens` and `t_max_predict_ms`.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "core"
    / "inference"
    / "llama_cpp.py"
)
SRC = SOURCE_PATH.read_text()
TREE = ast.parse(SRC)


def _is_subscript_assign(stmt: ast.stmt, target_name: str, key: str) -> bool:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    t = stmt.targets[0]
    if not isinstance(t, ast.Subscript):
        return False
    if not (isinstance(t.value, ast.Name) and t.value.id == target_name):
        return False
    slc = t.slice
    return isinstance(slc, ast.Constant) and slc.value == key


def _collect_assignments(tree, target_name, key):
    """Return list of (node, stack_of_enclosing_ifs) for each match."""
    hits = []

    def visit(node, stack):
        if _is_subscript_assign(node, target_name, key):
            hits.append((node, stack))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                for sub in child.body:
                    visit(sub, stack + [(child, "body")])
                for sub in child.orelse:
                    visit(sub, stack + [(child, "orelse")])
            else:
                visit(child, stack)

    visit(tree, [])
    return hits


def test_default_t_max_predict_ms_is_at_least_ten_minutes():
    for node in TREE.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == "_DEFAULT_T_MAX_PREDICT_MS":
                value = node.value
                assert isinstance(value, ast.Constant)
                assert value.value >= 600_000, (
                    f"_DEFAULT_T_MAX_PREDICT_MS must be >= 10 minutes "
                    f"(600_000 ms) to avoid cutting off slow-CPU generations; "
                    f"got {value.value}"
                )
                return
    raise AssertionError("_DEFAULT_T_MAX_PREDICT_MS constant missing")


def test_t_max_predict_ms_set_unconditionally_at_three_sites():
    hits_payload = _collect_assignments(TREE, "payload", "t_max_predict_ms")
    hits_stream = _collect_assignments(TREE, "stream_payload", "t_max_predict_ms")
    total = len(hits_payload) + len(hits_stream)
    assert total == 3, (
        f"expected 3 total t_max_predict_ms assignments "
        f"(payload x2 + stream_payload x1), got {total}"
    )
    for node, stack in hits_payload + hits_stream:
        for parent_if, branch in stack:
            # The assignment must not be gated by a test that checks
            # `max_tokens is None` (which would make it dead code for
            # the UI path where max_tokens is always set).
            test_src = ast.unparse(parent_if.test)
            assert "max_tokens" not in test_src, (
                f"t_max_predict_ms at line {node.lineno} is nested under "
                f"`if {test_src}:` -- it must be applied unconditionally so "
                f"the wall-clock cap is not dead code for callers that set "
                f"max_tokens"
            )


def test_max_tokens_default_cap_still_applied():
    # _DEFAULT_MAX_TOKENS must still kick in when caller passes None.
    # We check the conditional expression `max_tokens if max_tokens is not
    # None else _DEFAULT_MAX_TOKENS` appears at each site.
    matches = 0
    for node in ast.walk(TREE):
        if not isinstance(node, ast.IfExp):
            continue
        src = ast.unparse(node)
        if "max_tokens" in src and "_DEFAULT_MAX_TOKENS" in src:
            matches += 1
    assert matches >= 3, (
        f"expected >=3 `max_tokens if max_tokens is not None else "
        f"_DEFAULT_MAX_TOKENS` expressions; got {matches}"
    )
