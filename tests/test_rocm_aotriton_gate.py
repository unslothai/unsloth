# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Importing Unsloth opens the ROCm AOTriton SDPA gate, whatever loaded first.

Torch reads TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL late: into a function-local
`static const bool` inside `check_flash_attention_hardware_support` and
`check_mem_efficient_hardware_support`
(`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, unchanged from 2.8
through 2.11 and main), so it is fixed at the first ROCm SDPA capability probe,
not while the C++ extension loads. `import torch` before `import unsloth` is
therefore still in time, and that is the order
`studio/backend/core/training/trainer.py` uses. Skipping the write there leaves
the gate shut and finetuning back on the quadratic MATH path #8819 measured."""

import ast
import functools
import os
import pathlib
import subprocess
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_INIT = _ROOT / "unsloth" / "__init__.py"
_SOURCE = _INIT.read_text(encoding = "utf-8")
_GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


def _gate_statement():
    """The top-level statement that opens the gate, with any block it sits in."""
    for node in ast.parse(_SOURCE).body:
        if _GATE in ast.unparse(node):
            return node
    raise AssertionError(f"no top-level statement in {_INIT} sets {_GATE}")


def _open_gate(modules, environ):
    """Run that statement against a synthetic `sys.modules` / environment. Running
    it rather than grepping its source is what keeps a re-added guard visible."""
    scope = {
        "os": types.SimpleNamespace(environ = environ),
        "sys": types.SimpleNamespace(modules = modules),
    }
    exec(ast.unparse(_gate_statement()), scope)


def _first_torch_import():
    """Where Unsloth itself first imports torch. The variable is only read at the
    first SDPA probe, but staying above every import here keeps it that way even if
    something Unsloth imports probes on the way in."""
    return min(
        (
            node.lineno
            for node in ast.walk(ast.parse(_SOURCE))
            if (
                isinstance(node, ast.Import)
                and any(alias.name.split(".")[0] == "torch" for alias in node.names)
            )
            or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch")
        ),
        default = 10**9,
    )


def test_the_gate_is_opened_above_unsloths_own_torch_import():
    statement = _gate_statement()
    environ = {}
    _open_gate({}, environ)
    assert environ.get(_GATE) == "1"
    assert statement.lineno < _first_torch_import()


def test_an_already_imported_torch_does_not_skip_the_write():
    """The regression a `"torch" not in sys.modules` guard reintroduces. Torch has
    not read the variable yet at that point, so the write still decides the gate."""
    environ = {}
    _open_gate({"torch": types.ModuleType("torch")}, environ)
    assert environ.get(_GATE) == "1"


@pytest.mark.parametrize("value", ["0", "1"])
def test_an_explicit_value_is_never_overwritten(value):
    """`c10::utils::check_env` reads "0" as false and "1" as true, so "0" is the
    opt-out and it has to survive both orderings."""
    for modules in ({}, {"torch": types.ModuleType("torch")}):
        environ = {_GATE: value}
        _open_gate(modules, environ)
        assert environ[_GATE] == value


def _run(code, **env):
    """Run `code` in a fresh interpreter, so no module state leaks between cases."""
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    # Importing Unsloth sets the gate; each case supplies its own starting value.
    clean = {k: v for k, v in os.environ.items() if k != _GATE}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = dict(clean, PYTHONPATH = os.pathsep.join(path), **env),
        timeout = 900,
    )


@functools.cache
def _unsloth_is_importable():
    return _run("import unsloth").returncode == 0


def _needs_unsloth():
    if not _unsloth_is_importable():
        pytest.skip("unsloth is not importable in this environment")


_REPORT = "import os, unsloth\nprint('GATE', os.environ.get({name!r}))".format(name = _GATE)


@pytest.mark.parametrize(
    "prologue",
    ["", "import torch\n"],
    ids = ["unsloth_first", "torch_first"],
)
def test_importing_unsloth_opens_the_gate(prologue):
    """The end-to-end shape of #8819, both ways round."""
    _needs_unsloth()
    out = _run(prologue + _REPORT)
    assert out.returncode == 0, out.stderr
    assert "GATE 1" in out.stdout, out.stdout


def test_an_explicit_opt_out_survives_the_import():
    _needs_unsloth()
    out = _run(_REPORT, **{_GATE: "0"})
    assert out.returncode == 0, out.stderr
    assert "GATE 0" in out.stdout, out.stdout
