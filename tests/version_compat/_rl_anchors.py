"""Anchors of unsloth/models/rl.py, read without importing it (that needs a GPU stack)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

RL_PY = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "rl.py"


def _rl_py_constant(name: str) -> str:
    """A module-level string constant of unsloth/models/rl.py."""
    tree = ast.parse(RL_PY.read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"unsloth/models/rl.py no longer defines {name}")


def _reject_aux_loss_opt_in(src: str) -> str:
    """rl.py's substitution, applied exactly as rl.py applies it."""
    return re.sub(
        _rl_py_constant("_GRPO_AUX_LOSS_ENABLED_LINE"),
        lambda m: m.group(0) + "\n" + m.group(1) + _rl_py_constant("_GRPO_AUX_LOSS_REJECT"),
        src,
        count = 1,
        flags = re.MULTILINE,
    )
