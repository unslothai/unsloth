# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A /load that replaces the resident llama-server saves its slot KV first (#9037).

Slot KV was saved only by the idle unload, a media swap and the /unload route. The reload a
Context Length change makes goes through /load, whose teardown is inside ``load_model``, so a
100k-token chat was prefilled again after every settings change. Pinned on the source like the
other ordering guards in this suite: the save has to sit after the point of no return (a load
that is still refusable must not write chat KV to disk) and before the teardown.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROUTES = Path(__file__).resolve().parents[1] / "routes" / "inference.py"
_SAVE = "await save_chat_kv_before_unload()"
_GUARD = "if replacing and llama_backend.is_loaded:"


def _load_model_impl_source() -> str:
    source = _ROUTES.read_text(encoding = "utf-8")
    start = source.index("async def _load_model_impl(")
    end = source.index("\nasync def ", start + 1)
    return source[start:end]


def test_both_teardowns_in_the_load_route_save_the_slot_kv_first():
    body = _load_model_impl_source()
    assert body.count(_SAVE) == 2, "one save per teardown: the GGUF relaunch and the standard load"
    assert body.count(_GUARD) == 2, "each save is guarded by the same replacement predicate"

    gguf = body.index(_SAVE)
    standard = body.index(_SAVE, gguf + 1)
    # GGUF branch: after the last rejection and the preview handoff, before any teardown.
    assert body.rindex("_set_preview_resident(None)", 0, gguf) > body.rindex(
        "_raise_if_scoped_load_cancelled()", 0, gguf
    )
    assert gguf < body.index("unsloth_backend.unload_model", gguf)
    assert gguf < body.index("await _run_gguf_load_attempt(", gguf)
    # Standard branch: after the preview handoff, before llama-server is torn down.
    assert body.rindex("_set_preview_resident(None)", 0, standard) > gguf
    assert standard < body.index(
        "await _unload_llama_before_standard_load(llama_backend)", standard
    )


def test_the_unload_route_keeps_its_own_save():
    source = _ROUTES.read_text(encoding = "utf-8")
    start = source.index("async def _unload_model_impl(")
    end = source.index("\nasync def ", start + 1)
    assert source[start:end].count(_SAVE) == 1
    assert source.count(_SAVE) == 3


def test_each_save_is_awaited_under_its_guard():
    """AST, not text: the guard has to be the ``if`` that encloses the await."""
    tree = ast.parse(_ROUTES.read_text(encoding = "utf-8"))
    impl = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_load_model_impl"
    )
    guarded = []
    for node in ast.walk(impl):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "replacing and llama_backend.is_loaded":
            continue
        awaited = [
            sub
            for sub in ast.walk(node)
            if isinstance(sub, ast.Await)
            and isinstance(sub.value, ast.Call)
            and getattr(sub.value.func, "id", None) == "save_chat_kv_before_unload"
        ]
        guarded.extend(awaited)
    assert len(guarded) == 2
