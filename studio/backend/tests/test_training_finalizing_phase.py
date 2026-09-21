# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The post-training save must be visible, and must stay non-terminal (#7897).

After the last optimizer step the worker still merges and saves, emitting no step
updates, so /api/train/status reported phase="training" at 100% throughout,
indistinguishable from a hang. The `finalizing` phase names it.

Two invariants matter more than the label:
  1. Reaching total_steps must never imply completion; `completed` still comes
     only from progress.is_completed.
  2. Every phase the route emits must be in TrainingStatus's Literal, or pydantic
     raises ValidationError and the blanket handler turns /status into a 500.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_ROUTES_TRAINING = _BACKEND_DIR / "routes" / "training.py"
_MODELS_TRAINING = _BACKEND_DIR / "models" / "training.py"


def _load_is_finalizing():
    """Exec just the helper: routes/training.py pulls in the whole app otherwise."""
    src = _ROUTES_TRAINING.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_is_finalizing":
            ns: dict = {}
            exec(compile(ast.Module([node], []), str(_ROUTES_TRAINING), "exec"), ns)
            return ns["_is_finalizing"]
    raise AssertionError("routes/training.py does not define _is_finalizing")


def _progress(step = 0, total_steps = 0):
    return SimpleNamespace(step = step, total_steps = total_steps)


# _is_finalizing


@pytest.mark.parametrize(
    "step, total, msg, expected",
    [
        (126, 126, "training in progress...", True),  # the reported symptom
        (127, 126, "training in progress...", True),  # defensive overshoot
        (125, 126, "training in progress...", False),  # steps remain
        (0, 126, "training in progress...", False),
        (0, 0, "training in progress...", False),  # total unknown -> inert
        (5, 0, "training in progress...", False),
        (0, 0, "saving model...", True),  # MLX/embedding say so
        (10, 126, "saving stopped model...", True),
        (10, 126, "merging weights into 16bit", True),
        (10, 126, "ready to train", False),
    ],
)
def test_is_finalizing(step, total, msg, expected):
    assert _load_is_finalizing()(_progress(step, total), msg) is expected


def test_is_finalizing_tolerates_missing_attributes():
    """A progress object may be None or partial early in a run."""
    fn = _load_is_finalizing()
    assert fn(None, "training") is False
    assert fn(SimpleNamespace(), "training") is False
    assert fn(SimpleNamespace(step = None, total_steps = None), "training") is False


# Contract guards


def _phase_literals() -> set[str]:
    src = _MODELS_TRAINING.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "TrainingStatus"):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "phase"
            ):
                sub = stmt.annotation
                # phase: Literal[...] = Field(...)
                while isinstance(sub, ast.Subscript) and not (
                    isinstance(sub.value, ast.Name) and sub.value.id == "Literal"
                ):
                    sub = sub.value
                literal = sub.slice
                elts = literal.elts if isinstance(literal, ast.Tuple) else [literal]
                return {e.value for e in elts if isinstance(e, ast.Constant)}
    raise AssertionError("TrainingStatus.phase Literal not found")


def test_every_emitted_phase_is_in_the_response_literal():
    """A phase missing from the Literal makes /api/train/status 500, not degrade."""
    src = _ROUTES_TRAINING.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    # The phase derivation moved into _build_training_status, so scan both, not just inline.
    fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in {"get_training_status", "_build_training_status"}
    ]
    assert fns, "neither status function found"
    emitted = {
        node.value.value
        for fn in fns
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "phase" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert emitted, "no literal phase assignments found; guard needs updating"
    missing = emitted - _phase_literals()
    assert not missing, f"phases emitted but not declared in TrainingStatus: {sorted(missing)}"


def test_finalizing_is_declared():
    assert "finalizing" in _phase_literals()


def test_completion_still_comes_only_from_is_completed():
    """100% must not be promoted to a terminal state."""
    src = _ROUTES_TRAINING.read_text(encoding = "utf-8")
    # Follow the phase derivation wherever it lives: it moved into _build_training_status.
    fn_src = next(
        seg
        for seg in (
            ast.get_source_segment(src, n)
            for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in {"_build_training_status", "get_training_status"}
        )
        if seg and 'phase = "completed"' in seg
    )
    completed_branch = re.search(r'phase\s*=\s*"completed"', fn_src)
    assert completed_branch, "no completed branch found"
    preceding = fn_src[: completed_branch.start()]
    # The guard immediately governing `completed` must still be is_completed.
    assert (
        "is_completed" in preceding.rsplit("elif", 1)[-1]
    ), "the `completed` phase is no longer gated on progress.is_completed"
    # And `finalizing` must sit inside the is_active branch, never after it.
    assert fn_src.index('phase = "finalizing"') < completed_branch.start()


def test_frontend_phase_union_covers_the_backend_literal():
    """phaseColors/phaseLabelKeys are Record<TrainingPhase, ...>, so a backend
    phase missing from the union is a compile error the frontend never sees."""
    runtime_ts = (
        _BACKEND_DIR.parent / "frontend" / "src" / "features" / "training" / "types" / "runtime.ts"
    )
    if not runtime_ts.is_file():
        pytest.skip("frontend sources not present")
    union = set(re.findall(r'\|\s*"([a-z_]+)"', runtime_ts.read_text(encoding = "utf-8")))
    missing = _phase_literals() - union
    assert not missing, f"TrainingPhase is missing backend phases: {sorted(missing)}"
