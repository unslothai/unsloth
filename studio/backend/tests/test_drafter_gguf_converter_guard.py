# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF export with an EAGLE-3 drafter runs convert_hf_to_gguf.py a second time on the checkpoint's
eagle3/ folder. That run must sit under the same remote-code decision as the first one: with remote code
not approved, HF_MODULES_CACHE points where no module can be imported."""

import ast
import inspect
import os

from core.export import export as export_mod


def _with_guards(fn, callee):
    tree = ast.parse(_dedent(inspect.getsource(fn)))
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        items = {
            getattr(i.context_expr.func, "id", getattr(i.context_expr.func, "attr", None))
            for i in node.items
            if isinstance(i.context_expr, ast.Call)
        }
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and getattr(inner.func, "attr", None) == callee:
                guarded.append(items)
    return guarded


def _dedent(src):
    import textwrap
    return textwrap.dedent(src)


def test_drafter_conversion_runs_under_the_converter_guard():
    guards = _with_guards(export_mod.ExportBackend.export_gguf, "_package_drafter_gguf")
    assert guards and all("_converter_remote_code" in g for g in guards)


def test_converter_guard_blocks_module_imports_when_not_approved(monkeypatch):
    monkeypatch.delenv("HF_MODULES_CACHE", raising = False)
    with export_mod._converter_remote_code(False):
        blocked = os.environ["HF_MODULES_CACHE"]
        assert not os.path.isdir(os.path.dirname(blocked))
    assert "HF_MODULES_CACHE" not in os.environ
