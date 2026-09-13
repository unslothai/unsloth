# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""No test may gate hardware on a call this repo's own harnesses spoof.

`tests/_zoo_aggressive_cuda_spoof.py` sets `torch.cuda.is_available` to return
True and leaves it there. It is applied at module import by everything under
`tests/version_compat` and `tests/vllm_compat`. pytest imports every selected
module before running anything and evaluates `@pytest.mark.skipif` at import, so
one spoofing module in a session decides the guard for every module collected
after it, whatever directory it lives in.

A guard written as `skipif(not torch.cuda.is_available())` therefore un-skips on
a CPU-only box as soon as it shares a session with one of those files, and the
test dies inside torch with `RuntimeError: Cannot access accelerator device when
none is available` -- a message about torch, from a cause that is neither torch
nor the test.

CI is only clear of this because the three jobs covering those directories each
`--ignore` the others. Nothing enforced that, and dropping one `--ignore` would
have reopened it silently, so enforce the guard side instead:
`tests/_shared/real_accelerator.py` records the answer before any spoof runs.

Scanned with ast rather than grep so a reformatted decorator, a multi-line
`skipif(...)` or a `# noqa` cannot walk past it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_TESTS_ROOT = Path(__file__).resolve().parents[1]

# The spoof itself, the recorded-answer helper and conftest are what implement the
# real probe. They are the only places allowed to call the raw torch API for it.
_ALLOWED = {
    _TESTS_ROOT / "_zoo_aggressive_cuda_spoof.py",
    _TESTS_ROOT / "_shared" / "real_accelerator.py",
    _TESTS_ROOT / "conftest.py",
}

_SPOOFED_CALLS = {
    ("torch", "cuda", "is_available"),
    ("torch", "xpu", "is_available"),
    ("torch", "accelerator", "is_available"),
}


def _dotted(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return ()


def _skipif_calls(tree: ast.AST):
    """Every `pytest.mark.skipif(...)` decorator call in the module."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if _dotted(decorator.func)[-1:] == ("skipif",):
                yield decorator


def _python_test_files():
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if path in _ALLOWED:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_skip_guard_reads_a_spoofable_accelerator_probe():
    offenders = []
    for path in _python_test_files():
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except SyntaxError:
            # tests/python/ holds fixtures that are deliberately unparseable on this
            # interpreter (the 3.9 floor checks). Not our business.
            continue
        for decorator in _skipif_calls(tree):
            for call in ast.walk(decorator):
                if isinstance(call, ast.Call) and _dotted(call.func) in _SPOOFED_CALLS:
                    offenders.append(
                        f"{path.relative_to(_TESTS_ROOT)}:{call.lineno}: "
                        f"{'.'.join(_dotted(call.func))}()"
                    )
    assert not offenders, (
        "these skip guards read a probe tests/_zoo_aggressive_cuda_spoof.py patches to "
        "True process-wide, so they un-skip on a CPU-only box whenever they share a "
        "pytest session with tests/version_compat or tests/vllm_compat. Use "
        "`from real_accelerator import has_real_accelerator` instead:\n  " + "\n  ".join(offenders)
    )


def test_the_recorded_answer_survives_the_spoof():
    """The property the helper exists for, asserted rather than assumed.

    Primed by conftest before collection, so by the time this runs the aggressive
    spoof may or may not have been applied depending on what else is in the
    session. Apply it here and check the recorded answer does not move.
    """
    import sys

    sys.path.insert(0, str(_TESTS_ROOT))
    from real_accelerator import has_real_accelerator

    before = has_real_accelerator()

    import _zoo_aggressive_cuda_spoof as spoof

    spoof.apply()

    import torch

    assert torch.cuda.is_available() is True, (
        "the spoof no longer patches torch.cuda.is_available, so this test is not "
        "checking anything; re-point it at whatever it patches now"
    )
    assert has_real_accelerator() is before, (
        "has_real_accelerator() moved after the spoof was applied, which is the whole "
        "thing it is supposed to be immune to"
    )


@pytest.mark.parametrize(
    "probe", ["torch.cuda.is_available", "torch.xpu.is_available", "torch.accelerator.is_available"]
)
def test_the_scanner_would_catch_a_regression(probe, tmp_path):
    """The scan is not vacuous: hand it the shape it forbids and it must object."""
    offending = tmp_path / "test_offending.py"
    offending.write_text(
        "import pytest\nimport torch\n\n\n"
        f"@pytest.mark.skipif(not {probe}(), reason = 'needs a GPU')\n"
        "def test_x():\n    pass\n",
        encoding = "utf-8",
    )
    tree = ast.parse(offending.read_text(encoding = "utf-8"))
    found = [
        call
        for decorator in _skipif_calls(tree)
        for call in ast.walk(decorator)
        if isinstance(call, ast.Call) and _dotted(call.func) in _SPOOFED_CALLS
    ]
    assert len(found) == 1
