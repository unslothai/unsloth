# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An inherited `UNSLOTH_FORCE_CUSTOM_DTYPE` must be harmless to EVERY reader.

`unsloth_zoo==2026.8.15`, which this package's floor resolves to, still `eval`s the
dtype field, and a version floor cannot fix an already resolved install, so the VALUE
is rewritten at import. The old reader is reproduced as the one line that matters,
`eval(field)`. Imported inside each test, for the four-package runner.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ENV_KEY = "UNSLOTH_FORCE_CUSTOM_DTYPE"
_HOSTILE = 'all;__import__("os").system("touch {marker}");None;;'


def _import_with(value, marker = None):
    """Imports the module in a fresh process with `value` inherited, returns the env."""
    environment = dict(os.environ)
    environment[_ENV_KEY] = value.format(marker = marker) if marker else value
    # The child imports unsloth, and unsloth_zoo.get_device_type() raises on a host with no torch accelerator.
    # studio/backend/tests/conftest.py does, with setdefault
    environment.setdefault("UNSLOTH_ALLOW_CPU", "1")
    program = (
        "import unsloth.models._custom_dtype as module, os;"
        f"print(repr(os.environ.get({_ENV_KEY!r})))"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program],
        env = environment,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]
    return finished.stdout.strip().splitlines()[-1]


def test_a_hostile_dtype_field_cannot_reach_an_eval(tmp_path):
    """The field the older zoo reader evaluates is a dtype NAME afterwards, or None."""
    from unsloth.models._custom_dtype import DTYPE_ALIASES

    marker = tmp_path / "pwned"
    seen = _import_with(_HOSTILE, marker = marker)
    assert not marker.exists(), "importing the module ran the inherited code"

    value = seen.strip("'\"")
    field = value.split(";", 4)[1]
    assert field.strip() in DTYPE_ALIASES, field
    assert eval(field) is None  # noqa: S307 - the point of the test
    assert not marker.exists()


def test_the_code_fields_of_an_inherited_value_are_emptied(monkeypatch):
    from unsloth.models._custom_dtype import neutralize_inherited_custom_dtype

    monkeypatch.setenv(_ENV_KEY, "all;torch.float16;torch.float16;custom;import os")
    sanitized = neutralize_inherited_custom_dtype()
    checker, dtype, bnb, custom, execute = sanitized.split(";", 4)
    assert (custom, execute) == ("", "")
    # The dtype fields of a well formed value still apply:
    assert (checker, dtype, bnb) == ("all", "torch.float16", "torch.float16")
    assert sanitized.count(";") == 4


def test_an_unknown_dtype_name_becomes_none(monkeypatch):
    from unsloth.models._custom_dtype import neutralize_inherited_custom_dtype

    monkeypatch.setenv(_ENV_KEY, "all;os.system('x');None;;")
    sanitized = neutralize_inherited_custom_dtype()
    assert sanitized.split(";", 4)[1] == "None"


def test_a_malformed_inherited_value_is_removed(monkeypatch):
    """Both readers assert on the separator count, so it can only crash a run."""
    from unsloth.models._custom_dtype import neutralize_inherited_custom_dtype

    monkeypatch.setenv(_ENV_KEY, "not-even-close")
    assert neutralize_inherited_custom_dtype() == ""
    assert _ENV_KEY not in os.environ


def test_a_value_this_process_registered_is_untouched(monkeypatch):
    from unsloth.models._custom_dtype import (
        neutralize_inherited_custom_dtype,
        register_custom_dtype,
        trusted_custom_dtype,
    )

    monkeypatch.delenv(_ENV_KEY, raising = False)
    ours = "all;torch.float16;torch.float16;pass  # only this test;pass  # only this test"
    register_custom_dtype(ours)
    assert neutralize_inherited_custom_dtype() == ours
    assert trusted_custom_dtype() == (ours, True)


@pytest.mark.parametrize(
    "value",
    [
        "all;torch.bfloat16;None;;",
        "float16;torch.float16;torch.float16;;",
    ],
)
def test_neutralizing_is_idempotent(value, monkeypatch):
    from unsloth.models._custom_dtype import neutralize_inherited_custom_dtype

    monkeypatch.setenv(_ENV_KEY, value)
    once = neutralize_inherited_custom_dtype()
    monkeypatch.setenv(_ENV_KEY, once)
    assert neutralize_inherited_custom_dtype() == once


def test_the_module_neutralizes_on_import():
    """A caller that never touches the function is still protected."""

    import ast
    import pathlib

    import unsloth.models._custom_dtype as module

    source = pathlib.Path(module.__file__).read_text(encoding = "utf-8")
    called = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", "") == "neutralize_inherited_custom_dtype"
    ]
    assert called, "the module no longer neutralizes an inherited value on import"


def test_the_compiler_entry_point_neutralizes_again():
    """A value set AFTER import must not reach the older `unsloth_zoo` reader, so it
    is sanitized again at `unsloth_compile_transformers`."""

    import ast
    import pathlib

    import unsloth.models._utils as module

    source = pathlib.Path(module.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    wrappers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "unsloth_compile_transformers"
    ]
    assert wrappers, "unsloth_compile_transformers is gone"
    called = [
        node
        for node in ast.walk(wrappers[0])
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "neutralize_inherited_custom_dtype"
    ]
    assert called, "the compiler entry point no longer re-neutralizes an inherited value"


def test_a_value_set_after_import_is_still_neutralized(monkeypatch):
    """End to end through the entry point, with the compilation itself disabled."""

    from unsloth.models._utils import unsloth_compile_transformers

    hostile = 'all;__import__("os").environ["UNSLOTH_TEST_MARKER"] = "1";None;;print(1)'
    monkeypatch.setenv(_ENV_KEY, hostile)
    monkeypatch.delenv("UNSLOTH_TEST_MARKER", raising = False)

    # `disable = True` returns before any compilation, which is all this needs:
    unsloth_compile_transformers(
        dtype = None,
        model_name = "unsloth/tiny",
        model_types = ["llama"],
        disable = True,
    )

    sanitized = os.environ[_ENV_KEY]
    assert sanitized.count(";") == 4, sanitized
    checker, dtype, bnb, custom, code = sanitized.split(";", 4)
    assert dtype == "None" and bnb == "None", sanitized
    assert custom == "" and code == "", sanitized
    assert "UNSLOTH_TEST_MARKER" not in os.environ


def test_a_shorthand_alias_is_written_back_canonically(monkeypatch):
    """`eval("fp16")` is a NameError in the older zoo reader, so preserving the field
    verbatim was only safe for the spellings that happen to be evaluable."""
    import torch
    from unsloth.models._custom_dtype import (
        neutralize_inherited_custom_dtype,
        resolve_dtype,
    )

    monkeypatch.setenv(_ENV_KEY, "all;fp16;bf16;;")
    assert neutralize_inherited_custom_dtype() == "all;torch.float16;torch.bfloat16;;"
    # And each canonical spelling is one the legacy reader's `eval` can evaluate.
    _checker, dtype, compute, _code, _execute = os.environ[_ENV_KEY].split(";", 4)
    for field in (dtype, compute):
        assert eval(field, {"torch": torch}) is resolve_dtype(field)


def test_an_empty_dtype_field_stays_empty(monkeypatch):
    """An unset field already reads as None to both readers; it is not rewritten."""
    from unsloth.models._custom_dtype import neutralize_inherited_custom_dtype

    monkeypatch.setenv(_ENV_KEY, "all;;;;")
    assert neutralize_inherited_custom_dtype() == "all;;;;"
