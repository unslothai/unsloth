# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""No test here may name a dtype kwarg on `from_pretrained`.

Both spellings are traps across the declared transformers window, in opposite directions:

  * `dtype = ...` is rejected at the 4.52.4 floor. `from_pretrained` forwards the unknown
    kwarg into the model constructor and it dies with
    `TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'dtype'`.
    Two loaders here did exactly that, and it went unnoticed because the floor lane could
    not get as far as running them: peft 0.18.0 could not import against the old
    `transformers>=4.51.3` floor, so 45 of these tests had never executed at the floor at
    all.
  * `torch_dtype = ...` spans the whole window but is deprecated from 4.57.6 onward
    (`modeling_utils.py` drops from 117 references at 4.52.4 to 20 deprecated ones at
    4.57.6), so it is a future red rather than a current one.

Load with neither and cast the result, which needs no version probe and no branch. These
models are tiny and CPU-only, so the intermediate costs nothing. A call-site check rather
than a version check, so it fails on every platform and every transformers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
BANNED = ("dtype", "torch_dtype")


def _offending_calls(source: str) -> list[tuple[int, str]]:
    """`(lineno, kwarg)` for every `*.from_pretrained(..., dtype = ...)` in `source`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "from_pretrained"):
            continue
        for keyword in node.keywords:
            if keyword.arg in BANNED:
                found.append((node.lineno, keyword.arg))
    return found


def test_no_version_compat_test_names_a_dtype_kwarg_at_load() -> None:
    offenders = {}
    scanned = 0
    for path in sorted(HERE.rglob("*.py")):
        scanned += 1
        hits = _offending_calls(path.read_text(encoding = "utf-8"))
        if hits:
            offenders[path.name] = hits
    assert scanned >= 5, (
        f"this guard scanned only {scanned} files, so it is not looking where it thinks it "
        f"is. Check the directory it globs before trusting a pass."
    )
    assert not offenders, (
        f"these loaders name a dtype kwarg on from_pretrained: {offenders}. `dtype=` raises "
        f"TypeError at the transformers floor and `torch_dtype=` is deprecated above 4.57.6, "
        f"so neither spelling holds across the declared window. Load without it and cast "
        f"the result, asserting the dtype you got."
    )


@pytest.mark.parametrize(
    "snippet, want",
    [
        ("M.from_pretrained(name, dtype = torch.float32)", [(1, "dtype")]),
        ("M.from_pretrained(name, torch_dtype = torch.float32)", [(1, "torch_dtype")]),
        ("M.from_pretrained(name).to(torch.float32)", []),
        ("M.from_pretrained(name)", []),
        # Not a load: the ban is about what from_pretrained is asked to do, and a dtype
        # kwarg elsewhere is ordinary.
        ("torch.zeros(4, dtype = torch.float32)", []),
    ],
)
def test_the_scanner_sees_both_spellings_and_nothing_else(snippet, want) -> None:
    """Negative control: the test above is a "nothing found" shape, which is also what a
    scanner that has stopped scanning reports."""
    assert _offending_calls(snippet) == want
