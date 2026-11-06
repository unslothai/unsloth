# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Restores the GC mode recorded at setup, so a plain TrainingArguments cannot silently disable it."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent / "unsloth" / "models"
_RL = (_ROOT / "rl.py").read_text(encoding = "utf-8")
_RL_REPLACEMENTS = (_ROOT / "rl_replacements.py").read_text(encoding = "utf-8")

# The single-line ternary form used at the trainer call sites:
#   <obj>._unsloth_gradient_checkpointing if hasattr(<obj>, '...') else getattr(<args>, 'gradient_checkpointing', True)
_TERNARY = re.compile(
    r"(?P<model>[\w.]+)\._unsloth_gradient_checkpointing "
    r"if hasattr\((?P=model), '_unsloth_gradient_checkpointing'\) "
    r"else getattr\((?P<args>[\w.]+), 'gradient_checkpointing', True\)"
)

_MISSING = object()


class _Obj:
    """Bare attribute bag; ``_unsloth_gradient_checkpointing`` present only when recorded."""

    def __init__(
        self,
        recorded = _MISSING,
        gradient_checkpointing = _MISSING,
    ):
        if recorded is not _MISSING:
            self._unsloth_gradient_checkpointing = recorded
        if gradient_checkpointing is not _MISSING:
            self.gradient_checkpointing = gradient_checkpointing


class _Self:
    def __init__(
        self,
        model = None,
        args = None,
    ):
        if model is not None:
            self.model = model
        self.args = args


# (recorded on model, args.gradient_checkpointing, expected restored value)
# A recorded mode, including None, wins over args.
_MATRIX = [
    ("unsloth", False, "unsloth"),  # args=False must NOT win
    (True, False, True),
    (False, True, False),
    (None, True, None),  # explicit None is restored, not treated as "unrecorded"
    (_MISSING, True, True),
    (_MISSING, False, False),
]


def _eval_ternary(expr, recorded, args_gc):
    """Eval a restore expression that references either ``model``/``args`` or ``self.model``/``self.args``."""
    model = _Obj(recorded = recorded)
    args = _Obj(gradient_checkpointing = args_gc)
    self = _Self(model = model, args = args)
    return eval(
        expr, {"hasattr": hasattr, "getattr": getattr}, {"model": model, "args": args, "self": self}
    )


def test_ternary_restore_semantics():
    exprs = [m.group(0) for m in _TERNARY.finditer(_RL)]
    exprs += [m.group(0) for m in _TERNARY.finditer(_RL_REPLACEMENTS)]
    assert len(exprs) >= 3, f"expected the 3 trainer-call restore sites, found {len(exprs)}"
    for expr in exprs:
        for recorded, args_gc, expected in _MATRIX:
            got = _eval_ternary(expr, recorded, args_gc)
            assert got == expected and type(got) is type(
                expected
            ), f"{expr!r}: recorded={recorded!r} args={args_gc!r} -> {got!r}, expected {expected!r}"


def _extract_prepare_restore_block():
    """The restore block lives in an exec'd template string, so it is cut out textually, not parsed."""
    lines = _RL.splitlines()
    start = next(
        i for i, l in enumerate(lines) if l.strip() == "_model = getattr(self, 'model', None)"
    )
    # End at the fallback assignment, not a fixed line count, so inserted lines are not truncated.
    end = next(
        i
        for i, l in enumerate(lines)
        if i > start and "use_gc = getattr(self.args, 'gradient_checkpointing', True)" in l
    )
    block = lines[start : end + 1]
    indent = len(block[0]) - len(block[0].lstrip())
    return "\n".join(l[indent:] for l in block)


def test_prepare_for_training_mode_block_semantics():
    block = _extract_prepare_restore_block()
    ast.parse(block)

    for recorded, args_gc, expected in _MATRIX:
        model = _Obj(recorded = recorded)
        args = _Obj(gradient_checkpointing = args_gc)
        ns = {"self": _Self(model = model, args = args), "hasattr": hasattr, "getattr": getattr}
        exec(block, {}, ns)
        got = ns["use_gc"]
        assert (
            got == expected and type(got) is type(expected)
        ), f"prepare block: recorded={recorded!r} args={args_gc!r} -> {got!r}, expected {expected!r}"


def test_prepare_block_tolerates_missing_model():
    # A trainer without a .model attribute must fall back to args rather than raise.
    block = _extract_prepare_restore_block()
    args = _Obj(gradient_checkpointing = True)
    self_no_model = _Self(model = None, args = args)  # _Self leaves .model unset when model is None
    assert not hasattr(self_no_model, "model")
    ns = {"self": self_no_model, "hasattr": hasattr, "getattr": getattr}
    exec(block, {}, ns)
    assert ns["use_gc"] is True


def test_recording_sites_are_real_module_code():
    # Recording must happen in patch_peft_model (loaded adapters) and the pre-wrapped pass-through.
    llama = (_ROOT / "llama.py").read_text(encoding = "utf-8")
    tree = ast.parse(llama)

    def assigns_marker(node):
        return any(
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_unsloth_gradient_checkpointing"
                for t in n.targets
            )
            for n in ast.walk(node)
        )

    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "patch_peft_model" in fns and assigns_marker(
        fns["patch_peft_model"]
    ), "patch_peft_model must record _unsloth_gradient_checkpointing so loaded adapters are covered"
    assert assigns_marker(
        fns["get_peft_model"]
    ), "get_peft_model pass-through must record _unsloth_gradient_checkpointing"
