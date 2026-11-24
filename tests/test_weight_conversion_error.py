# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import pathlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "loader_utils.py"
_CONVERSION_MESSAGE = (
    "We encountered some issues during automatic conversion of the weights. For details look at the "
    "`CONVERSION` entries of the above report!"
)
_OOM_ENTRY = (
    'Traceback (most recent call last):\n  File "core_model_loading.py", line 1528, in log_conversion_errors\n'
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 162.00 MiB. GPU 1 has a total capacity of "
    "23.61 GiB of which 2.19 MiB is free.\nCUDA out of memory. Tried to allocate 162.00 MiB. GPU 1 has a total "
    "capacity of 23.61 GiB of which 2.19 MiB is free.\nError: Inkling2LoopDequantizeExperts on tensors destined "
    "for model.language_model.layers.20.mlp.experts.gate_up_proj. Ckpt contains: 2"
)
_KEY = "model.language_model.layers.20.mlp.experts.gate_up_proj"


def _explain():
    # By source, so the test needs no `import unsloth` (which wants a GPU).
    tree = ast.parse(_SOURCE.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "explain_weight_conversion_error"
    )
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(_SOURCE), "exec"), namespace)
    return namespace["explain_weight_conversion_error"]


def _raise_like_transformers(conversion_errors):
    """The shape of log_state_dict_report: the report object is a local of the raising frame."""

    def log_state_dict_report(loading_info):
        if loading_info.conversion_errors:
            raise RuntimeError(_CONVERSION_MESSAGE)

    try:
        log_state_dict_report(SimpleNamespace(conversion_errors = conversion_errors))
    except RuntimeError as error:
        return error
    raise AssertionError("expected a RuntimeError")


def test_out_of_memory_is_raised_as_out_of_memory():
    explained = _explain()(_raise_like_transformers({_KEY: _OOM_ENTRY}))
    assert isinstance(explained, torch.OutOfMemoryError)
    message = str(explained)
    assert _KEY in message
    assert "CUDA out of memory. Tried to allocate 162.00 MiB" in message
    assert "CONVERSION" not in message


def test_other_conversion_failure_names_the_tensor_and_cause():
    entry = "Traceback ...\nValueError: shape mismatch\nError: SomeOp on tensors destined for x. Ckpt contains: 1"
    explained = _explain()(_raise_like_transformers({"x": entry}))
    assert type(explained) is RuntimeError
    assert "`x`" in str(explained) and "shape mismatch" in str(explained)


def test_unrelated_errors_are_left_alone():
    explain = _explain()
    assert explain(RuntimeError("something else")) is None
    assert explain(ValueError(_CONVERSION_MESSAGE)) is None
    assert explain(RuntimeError(_CONVERSION_MESSAGE)) is None
