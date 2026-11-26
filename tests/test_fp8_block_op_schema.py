# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The FP8 block matmul launchers register as opaque torch ops via torch.library.infer_schema. torch 2.6
rejects PEP 585 `list[int]` there (only typing.List[int]), so the op stayed unregistered and Dynamo
traced into the Triton launch. Rebuilds each launcher's signature from source (no Triton or GPU)."""

import ast
import copy
import typing
from pathlib import Path

import pytest
import torch

_FP8 = Path(__file__).resolve().parents[1] / "unsloth" / "kernels" / "fp8.py"
_OPAQUE = {"w8a8_block_fp8_matmul_triton", "torchao_block_matmul"}


def _launchers():
    tree = ast.parse(_FP8.read_text(encoding = "utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _OPAQUE:
            stub = copy.copy(node)
            stub.decorator_list = []
            stub.body = [ast.Expr(ast.Constant(...))]
            module = ast.fix_missing_locations(ast.Module(body = [stub], type_ignores = []))
            namespace = {"torch": torch, "List": typing.List, "Optional": typing.Optional}
            exec(compile(module, str(_FP8), "exec"), namespace)
            found[node.name] = namespace[node.name]
    assert set(found) == _OPAQUE
    return found


@pytest.mark.skipif(
    not hasattr(torch.library, "infer_schema"), reason = "no infer_schema on this torch"
)
@pytest.mark.parametrize("name", sorted(_OPAQUE))
def test_block_matmul_signature_infers_a_schema(name):
    schema = torch.library.infer_schema(_launchers()[name], mutates_args = ())
    assert "[] block_size" in schema
