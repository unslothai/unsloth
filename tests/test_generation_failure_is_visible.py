"""A trainer that fails to build must be reported, not swallowed into a silent trl fallback."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "unsloth" / "models" / "rl.py").read_text(encoding = "utf-8")


def _handler():
    """The except block of `_patch_trl_rl_trainers`, by AST rather than by
    line offsets, which drift."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_trl_rl_trainers":
            return ast.get_source_segment(SRC, node)
    raise AssertionError("_patch_trl_rl_trainers not found")


def test_generation_failure_warns_rather_than_whispers():
    body = _handler()
    assert "logger.warning_once(" in body
    assert "logger.info(" not in body, "info level is how this hid for a day"


def test_the_message_says_what_the_user_loses():
    """'Could not patch X' reads like a no-op. The user needs to know training
    silently changed."""
    body = _handler()
    assert "trl's own trainer" in body


def test_the_message_carries_the_exception():
    body = _handler()
    assert "type(e).__name__" in body and "{e}" in body


def test_it_still_swallows():
    """Raising would break every TRL version that renames a trainer, which is
    the reason the swallow exists. The fix is visibility, not propagation."""
    body = _handler()
    assert "except Exception" in body
    assert "raise" not in body


def test_the_benign_case_never_reaches_the_handler():
    """`_patch_trl_rl_trainers_impl` returns early when the trainer module is
    absent, so a missing trainer cannot trigger the new warning and spam it for
    every trainer this TRL does not ship."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_trl_rl_trainers_impl":
            body = ast.get_source_segment(SRC, node)
            break
    else:
        raise AssertionError("impl not found")
    head = body[: body.index("Patch for vLLM") + 2000] if "Patch for vLLM" in body else body[:2000]
    assert "Could not import trl.trainer" in head
    i = head.index("Could not import trl.trainer")
    assert "return" in head[i : i + 200], "the benign path must return, not raise"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
