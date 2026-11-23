# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Set HF_HUB_ENABLE_HF_TRANSFER only if hf_transfer imports; without it, hub < 1.0 downloads fail."""

import ast
import types
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "dataprep" / "synthetic.py"
_TREE = ast.parse(_SOURCE.read_text(encoding = "utf-8"))


def _nodes():
    probe = next(
        (
            n
            for n in _TREE.body
            if isinstance(n, ast.FunctionDef) and n.name == "_hf_transfer_importable"
        ),
        None,
    )
    offline = next(
        n
        for n in _TREE.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_OFFLINE_VALS" for t in n.targets)
    )
    gate = [
        n
        for n in _TREE.body
        if isinstance(n, ast.If) and "HF_HUB_ENABLE_HF_TRANSFER" in ast.dump(n)
    ]
    assert (
        len(gate) == 1
    ), "expected exactly one module-level if that sets HF_HUB_ENABLE_HF_TRANSFER"
    return probe, offline, gate[0]


def _resolve(environ, find_spec):
    probe, offline, gate = _nodes()
    env = dict(environ)
    namespace = {
        "os": types.SimpleNamespace(environ = env),
        "_importlib_util": types.SimpleNamespace(find_spec = find_spec),
    }
    body = [node for node in (probe, offline, gate) if node is not None]
    exec(compile(ast.Module(body = body, type_ignores = []), str(_SOURCE), "exec"), namespace)
    return env


def _installed(name):
    return object() if name == "hf_transfer" else None


def _missing(name):
    return None


def _stub_without_spec(name):
    raise ValueError(f"{name}.__spec__ is None")


def test_an_installed_hf_transfer_turns_the_flag_on():
    assert _resolve({}, _installed).get("HF_HUB_ENABLE_HF_TRANSFER") == "1"


def test_a_missing_hf_transfer_leaves_the_flag_unset():
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in _resolve({}, _missing)


def test_a_spec_less_stub_reads_as_not_installed():
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in _resolve({}, _stub_without_spec)


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_offline_mode_leaves_the_flag_unset_even_when_installed(var):
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in _resolve({var: "1"}, _installed)


@pytest.mark.parametrize("value", ["0", "1"])
def test_an_explicit_value_is_kept_even_when_installed(value):
    assert (
        _resolve({"HF_HUB_ENABLE_HF_TRANSFER": value}, _installed)["HF_HUB_ENABLE_HF_TRANSFER"]
        == value
    )
