# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update` must not hand a --no-torch install the MLX stack back.

The updater clears SKIP_STUDIO_BASE (`unsloth_cli/commands/studio.py`), so without the
`not NO_TORCH` guard a routine update reinstalls MLX into a GGUF-only venv and the next
launch enables Train before the no_torch verdict is reached. Invisible to the backend
tests. Structural, like test_diffusers_pin.py: running the installer needs a Mac, and
what must hold is a property of the gate, not of one run.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STACK = REPO_ROOT / "studio" / "install_python_stack.py"
STUDIO_CLI = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"

MLX_STEP_LABEL = "Installing MLX stack"


def _source() -> str:
    return STACK.read_text(encoding = "utf-8")


def _guard_chains_of_calls_mentioning(source: str, needle: str) -> list[list[str]]:
    """Every enclosing `if` test, outermost first, for each call mentioning ``needle``.

    A chain because the step sits under the platform gate and then under a wheel-floor
    branch; only the outermost is the gate this file is about.
    """
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


def test_the_mlx_install_step_is_gated_on_no_torch():
    guards = _guards_of_calls_mentioning(_source(), MLX_STEP_LABEL)
    assert guards, f"no call carrying {MLX_STEP_LABEL!r} found in {STACK.name}"
    for guard in guards:
        assert "NO_TORCH" in guard and "not NO_TORCH" in guard, (
            f"the MLX install step is guarded by `{guard}`, which does not exclude a "
            f"--no-torch install. The updater clears SKIP_STUDIO_BASE, so without this the "
            f"next `unsloth studio update` reinstalls the stack the user declined."
        )
        assert "IS_MAC_ARM" in guard, (
            f"the MLX install step is guarded by `{guard}`, which no longer scopes to "
            f"Apple Silicon"
        )


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
        # Sliced by line: what names this branch is the trailing comment on the += 1,
        # which a node's extent stops before.
        segment = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        # The += 1 whose comment names the MLX stack, i.e. the accounting twin.
        if "base_total += 1" in segment and "MLX stack" in segment:
            total_guards.add(ast.get_source_segment(source, node.test) or "")
    assert total_guards, "no base_total accounting found for the MLX stack step"
    assert total_guards == step_guards, (
        f"the MLX step runs under {step_guards} but is counted under {total_guards}; "
        f"they have to be the same expression"
    )


def test_the_updater_still_clears_skip_studio_base():
    """The premise: stop clearing this and the test above passes vacuously."""
    assert 'os.environ.pop("SKIP_STUDIO_BASE", None)' in STUDIO_CLI.read_text(encoding = "utf-8")


def test_no_torch_is_resolved_from_the_manifest_not_only_from_the_environment():
    """No UNSLOTH_NO_TORCH is injected, so the manifest tier is what fires the gate on update."""
    source = _source()
    tree = ast.parse(source)
    infer = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_infer_no_torch"
        ),
        None,
    )
    assert infer is not None, "_infer_no_torch is gone; NO_TORCH now comes from somewhere else"
    body = ast.get_source_segment(source, infer) or ""
    assert "recorded_no_torch" in body, (
        "_infer_no_torch no longer consults the install manifest, so an update would run "
        "with NO_TORCH False in a GGUF-only venv and reinstall the MLX stack"
    )
