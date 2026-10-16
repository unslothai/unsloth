# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO reward rewrite keeps locals() in a name. On CPython <= 3.12 a later locals() call in the
same frame stores that dict in itself, so every rollout tensor of the step outlives the call until
the cyclic GC runs. The name must be gone before any later locals() call."""

import ast
import inspect
import textwrap

import pytest

from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions


def _rewritten():
    trl_grpo = pytest.importorskip("trl.trainer.grpo_trainer")
    source = textwrap.dedent(inspect.getsource(trl_grpo.GRPOTrainer._generate_and_score_completions))
    source = textwrap.indent(source, "    ")
    out = grpo_trainer__generate_and_score_completions("_generate_and_score_completions", source)
    if "_unsloth_reward_locals = locals()" not in out:
        pytest.skip("this TRL has no reward call for the rewrite to anchor on")
    return textwrap.dedent(out)


def test_reward_locals_alias_is_deleted_before_the_next_locals_call():
    tree = ast.parse(_rewritten())
    lines = sorted(
        (node.lineno, kind)
        for node in ast.walk(tree)
        for kind in (
            ("bind",) if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_unsloth_reward_locals" for t in node.targets
            ) else ("del",) if isinstance(node, ast.Delete) and any(
                isinstance(t, ast.Name) and t.id == "_unsloth_reward_locals" for t in node.targets
            ) else ("locals",) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "locals"
            else ()
        )
    )
    bind = next(line for line, kind in lines if kind == "bind")
    deleted = [line for line, kind in lines if kind == "del" and line > bind]
    assert deleted, lines
    # Every locals() call after the binding's own line comes after the del.
    assert not [line for line, kind in lines if kind == "locals" and bind < line < deleted[0]], lines


def test_frame_dict_is_not_kept_alive_after_return():
    import gc
    import weakref

    class Tensor:
        pass

    def step():
        big = Tensor()
        ref = weakref.ref(big)
        _unsloth_reward_locals = locals()
        _unsloth_reward_locals.get("completions_text")
        del _unsloth_reward_locals
        locals().get("tool_mask")
        return ref

    enabled = gc.isenabled()
    gc.disable()
    try:
        assert step()() is None
    finally:
        if enabled:
            gc.enable()
