# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`UNSLOTH_FORCE_CUSTOM_DTYPE` is parsed by two separately released packages.

Both do `value.split(";", 4)` and assert at least four separators, and they ship on
their own schedules, so changing the layout on one side breaks the other on skew. The
hardening changed how the fields are INTERPRETED, not what they are; this pins the
count, the positions, and the direction of the trust decision.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from unsloth.models._custom_dtype import register_custom_dtype, trusted_custom_dtype


LOADER = pathlib.Path(__import__("unsloth.models.loader", fromlist = ["x"]).__file__).read_text(
    encoding = "utf-8"
)


def _shipped_values() -> list[str]:
    tree = ast.parse(LOADER)
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "register_custom_dtype":
                values.append(ast.literal_eval(node.args[0]))
    return values


@pytest.mark.parametrize("value", _shipped_values())
def test_every_shipped_value_has_five_fields(value):
    """Both readers do split(";", 4) after asserting count(";") >= 4."""
    assert value.count(";") >= 4
    fields = value.split(";", 4)
    assert len(fields) == 5


@pytest.mark.parametrize("value", _shipped_values())
def test_field_positions_are_unchanged(value):
    """checker, dtype, bnb_compute_dtype, custom_datatype, execute_code."""
    checker, dtype, bnb_compute_dtype, _custom, _execute = value.split(";", 4)
    assert checker in ("all", "float16", "torch.float16"), checker
    for field in (dtype, bnb_compute_dtype):
        assert re.fullmatch(r"(None|torch\.\w+)", field.strip()), field


def test_zoo_reader_still_parses_what_unsloth_writes():
    """unsloth_zoo is imported rather than copied, so a layout change there fails."""
    import unsloth_zoo.compiler as zoo_compiler

    source = pathlib.Path(zoo_compiler.__file__).read_text(encoding = "utf-8")
    assert (
        'custom_datatype.count(";") >= 4' in source
    ), "unsloth_zoo changed how it validates UNSLOTH_FORCE_CUSTOM_DTYPE"
    assert (
        'custom_datatype.split(";", 4)' in source
    ), "unsloth_zoo changed the field split for UNSLOTH_FORCE_CUSTOM_DTYPE"

    from unsloth_zoo.utils import _get_dtype

    import torch

    checked = 0
    for value in _shipped_values():
        _, dtype, _, _, _ = value.split(";", 4)
        name = dtype.strip().removeprefix("torch.")
        resolved = _get_dtype(name)
        # The oracle is torch, not the value under test.
        # Deriving `expected` from `resolved` made this assertion tautological for every non-None dtype, so a separately
        # released unsloth_zoo that started resolving `float16` to the wrong thing would still have passed.
        if dtype.strip() == "None":
            expected = None
        else:
            expected = getattr(torch, name)
            assert isinstance(expected, torch.dtype), (name, expected)
            checked += 1
        assert resolved == expected, (name, resolved, expected)
    assert checked, "no non-None dtype was checked, so this proves nothing"


def test_trust_decision_is_on_the_value_we_set(monkeypatch):
    """Identical layout, different provenance. The two halves use different strings on
    purpose, since the registry is never cleared."""
    template = _shipped_values()[0]
    checker, dtype, bnb, custom, execute = template.split(";", 4)

    inherited = ";".join([checker, dtype, bnb, custom, "import os; os.system('x')"])
    monkeypatch.setenv("UNSLOTH_FORCE_CUSTOM_DTYPE", inherited)
    value, trusted = trusted_custom_dtype()
    assert value == inherited  # dtype fields still readable
    assert not trusted  # code fields are not

    ours = ";".join([checker, dtype, bnb, custom, "pass  # only this test sets this"])
    register_custom_dtype(ours)
    value, trusted = trusted_custom_dtype()
    assert value == ours
    assert trusted


def test_unset_variable_leaves_custom_datatype_as_none():
    """`trusted_custom_dtype()` returns "" when unset, and "" is not None: `vision.py`
    gates on `is not None`, so nearly every model would walk itself to `exec("")`."""
    import unsloth.models.vision as vision

    source = pathlib.Path(vision.__file__).read_text(encoding = "utf-8")
    bound = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
            continue
        if value.func.id != "trusted_custom_dtype":
            continue
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                bound.update(e.id for e in target.elts if isinstance(e, ast.Name))

    assert bound, "trusted_custom_dtype() is no longer assigned in vision.py"
    assert "custom_datatype" not in bound, (
        "the raw value must not be bound to `custom_datatype`, which is gated on "
        "`is not None` further down"
    )
    assert "if custom_datatype is not None:" in source, "the consumer this test protects has moved"


def test_unset_variable_reads_as_empty_and_untrusted(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_CUSTOM_DTYPE", raising = False)
    assert trusted_custom_dtype() == ("", False)


def test_no_producer_bypasses_the_registry():
    """A direct os.environ write would set the variable without registering it, so
    its own code fields would then be dropped as untrusted."""
    assert 'os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"]' not in LOADER
    assert "os.environ['UNSLOTH_FORCE_CUSTOM_DTYPE']" not in LOADER
