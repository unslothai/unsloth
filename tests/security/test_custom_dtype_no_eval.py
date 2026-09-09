# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`UNSLOTH_FORCE_CUSTOM_DTYPE` is a five field string, two of whose fields are code.

`vision.py` used to `eval` the dtype fields and `exec` the code fields straight out of
the environment. Two changes are pinned: the dtype fields go through a fixed table, and
the code fields run only when this process set the variable. Defence in depth rather
than a privilege boundary, but the code path is gone either way.
"""

import ast
import pathlib
import re

import pytest
import torch

from unsloth.models._custom_dtype import (
    DTYPE_ALIASES,
    register_custom_dtype,
    resolve_dtype,
    trusted_custom_dtype,
)


LOADER = pathlib.Path(__import__("unsloth.models.loader", fromlist = ["x"]).__file__).read_text(
    encoding = "utf-8"
)


def _shipped_values():
    """Read out of loader.py rather than copied, so the test cannot drift."""
    tree = ast.parse(LOADER)
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Name) and function.id == "register_custom_dtype"):
            continue
        try:
            values.append(ast.literal_eval(node.args[0]))
        except (ValueError, IndexError):
            pytest.fail(f"register_custom_dtype called with a non-literal at line {node.lineno}")
    return values


def test_the_producers_were_all_found():
    values = _shipped_values()
    assert len(values) >= 6, values
    assert LOADER.count('os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"] =') == 0, (
        "a producer still writes the variable directly, so its code fields would be "
        "dropped as untrusted"
    )


@pytest.mark.parametrize("value", _shipped_values())
def test_shipped_dtype_fields_resolve(value):
    """A false rejection here breaks loading one of the six model families."""
    checker, dtype, bnb_compute_dtype, custom_datatype, execute_code = value.split(";", 4)
    assert resolve_dtype(dtype) is None or isinstance(resolve_dtype(dtype), torch.dtype)
    assert resolve_dtype(bnb_compute_dtype) is None or isinstance(
        resolve_dtype(bnb_compute_dtype), torch.dtype
    )


@pytest.mark.parametrize("value", _shipped_values())
def test_shipped_dtype_fields_match_the_old_eval(value):
    """The table must return exactly what `eval(field)` returned."""
    _, dtype, bnb_compute_dtype, _, _ = value.split(";", 4)
    for field in (dtype, bnb_compute_dtype):
        reference = eval(field, {"torch": torch})
        assert resolve_dtype(field) is reference, field


# --- a dtype field is no longer an expression --------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('touch /tmp/pwned')",
        "torch.float16 if __import__('os') else None",
        "open('/etc/passwd').read()",
        "exec('x=1')",
        "torch.cuda.synchronize()",
        "[].__class__",
        "torch.float8_e4m3fn",  # a real dtype, but not one this channel supports
    ],
)
def test_hostile_dtype_field_rejected(payload):
    with pytest.raises(ValueError, match = "unsupported dtype"):
        resolve_dtype(payload)


def test_table_covers_only_dtypes():
    for key, value in DTYPE_ALIASES.items():
        assert value is None or isinstance(value, torch.dtype), key


# --- the code fields are only ours -------------------------------------------


def test_a_value_we_set_is_trusted(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_CUSTOM_DTYPE", raising = False)
    value = _shipped_values()[0]
    register_custom_dtype(value)
    got, trusted = trusted_custom_dtype()
    assert got == value
    assert trusted


def test_an_inherited_value_is_not_trusted(monkeypatch):
    """The case that matters: the variable arrives from outside this process."""
    payload = "all;None;None;pass;import os; os.system('touch /tmp/pwned')"
    monkeypatch.setenv("UNSLOTH_FORCE_CUSTOM_DTYPE", payload)
    got, trusted = trusted_custom_dtype()
    assert got == payload  # dtype fields still readable
    assert not trusted


def test_an_inherited_value_that_mimics_ours_is_still_not_trusted(monkeypatch):
    """Trust is on the exact string we set, so a near miss does not qualify."""
    value = _shipped_values()[0] + " "
    monkeypatch.setenv("UNSLOTH_FORCE_CUSTOM_DTYPE", value)
    _, trusted = trusted_custom_dtype()
    assert not trusted


def test_unset_is_empty(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_CUSTOM_DTYPE", raising = False)
    assert trusted_custom_dtype() == ("", False)


# --- vision.py no longer evaluates the fields --------------------------------


def test_vision_does_not_eval_the_dtype_fields():
    source = pathlib.Path(__import__("unsloth.models.vision", fromlist = ["x"]).__file__).read_text(
        encoding = "utf-8"
    )
    assert "eval(_dtype)" not in source
    assert "eval(_bnb_compute_dtype)" not in source
    # The remaining exec is the code field, and it is gated.
    assert re.search(r"if _code_is_trusted:", source), "the code fields lost their gate"
