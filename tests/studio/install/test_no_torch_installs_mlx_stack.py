# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Chat only (--no-torch) install on Apple Silicon still gets the MLX stack: MLX needs no torch,
and without it the Mac cannot run MLX models."""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STACK = REPO_ROOT / "studio" / "install_python_stack.py"

MLX_STEP_LABEL = "Installing MLX stack"


def _source() -> str:
    return STACK.read_text(encoding = "utf-8")


def _guard_chains_of_calls_mentioning(source: str, needle: str) -> list[list[str]]:
    """Enclosing if-test chains for each call mentioning needle, outermost first."""
    tree = ast.parse(source)
    chains: list[list[str]] = []

    def walk(node, enclosing: list[str]):
        for child in ast.iter_child_nodes(node):
            inner = enclosing
            if isinstance(child, ast.If):
                inner = enclosing + [ast.get_source_segment(source, child.test) or ""]
            if isinstance(child, ast.Call) and any(
                isinstance(arg, ast.Constant) and isinstance(arg.value, str) and needle in arg.value
                for arg in child.args
            ):
                chains.append(enclosing)
            walk(child, inner)

    walk(tree, [])
    return chains


def _guards_of_calls_mentioning(source: str, needle: str) -> list[str]:
    """The outermost `if` test guarding every call whose arguments mention ``needle``."""
    return [chain[0] for chain in _guard_chains_of_calls_mentioning(source, needle) if chain]


def test_the_mlx_install_step_runs_on_no_torch_installs():
    guards = _guards_of_calls_mentioning(_source(), MLX_STEP_LABEL)
    assert guards, f"no call carrying {MLX_STEP_LABEL!r} found in {STACK.name}"
    for guard in guards:
        assert guard == "IS_MAC_ARM", (
            f"the MLX install step is guarded by `{guard}`. Every Apple Silicon install needs "
            f"the stack, a --no-torch one included."
        )


def test_the_re_resolve_and_health_report_share_the_step_gate():
    """A narrower gate leaves a --no-torch Mac with a stale MLX range or an unexplained chat-only."""
    source = _source()
    for needle in ("_report_mlx_stack_health(", "_mlx_vlm_spec_now = "):
        guards = [
            line.strip()
            for line, following in zip(source.splitlines(), source.splitlines()[1:])
            if line.strip().startswith("if ") and needle in following
        ]
        assert guards == ["if IS_MAC_ARM:"], (needle, guards)


def test_the_progress_total_uses_the_same_gate_as_the_step():
    """_TOTAL copies the gate; drift makes the bar count a step that never runs."""
    source = _source()
    step_guards = set(_guards_of_calls_mentioning(source, MLX_STEP_LABEL))
    tree = ast.parse(source)
    lines = source.splitlines()
    total_guards = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Sliced by line: the trailing comment on the += 1 lies past the node's extent.
        segment = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if "base_total += 1" in segment and "MLX stack" in segment:
            total_guards.add(ast.get_source_segment(source, node.test) or "")
    assert total_guards, "no base_total accounting found for the MLX stack step"
    assert total_guards == step_guards, (
        f"the MLX step runs under {step_guards} but is counted under {total_guards}; "
        f"they have to be the same expression"
    )
