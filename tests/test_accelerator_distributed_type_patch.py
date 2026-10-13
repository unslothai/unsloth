# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`_utils.py` must patch `Accelerator.distributed_type` with a property, not a bare
function: a function binds as a method and inverts accelerate's `!= NO` device_map
guards on a single device (#10016). It must still read as DEEPSPEED under DeepSpeed, or
accelerate turns on a float16 autocast over a bfloat16 model (#225). `import unsloth` needs
CUDA or XPU, so these run the statements from source in a fresh interpreter, which also
catches a revert.
"""

import ast
import pathlib
import subprocess
import sys
import textwrap

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_UTILS = _ROOT / "unsloth" / "models" / "_utils.py"

accelerate = pytest.importorskip("accelerate")


def _patch_statement() -> ast.Assign:
    """The one assignment to `Accelerator.distributed_type`; a second would be ambiguous."""
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]).endswith("Accelerator.distributed_type")
    ]
    assert len(matches) == 1, f"expected one assignment, found {len(matches)}"
    return matches[0]


def _getter_source() -> str:
    """The function the property wraps, so the fresh interpreter can define it first."""
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    name = ast.unparse(_patch_statement().value.args[0])
    defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(defs) == 1, f"expected one {name}, found {len(defs)}"
    return ast.unparse(defs[0])


def _read_distributed_type(state_type):
    """repr of the patched attribute and its `== NO` / `!= NO` on an Accelerator whose state says `state_type`."""
    script = "\n".join(
        [
            "import types",
            "import accelerate.accelerator",
            "from accelerate.utils.dataclasses import DistributedType",
            _getter_source(),
            ast.unparse(_patch_statement()),
            # Read it off an instance without standing up a real distributed backend.
            "accelerator = accelerate.accelerator.Accelerator.__new__(accelerate.accelerator.Accelerator)",
            *(
                []
                if state_type is None
                else [
                    f"accelerator.state = types.SimpleNamespace(distributed_type = DistributedType.{state_type})"
                ]
            ),
            "value = accelerator.distributed_type",
            "print(repr(value))",
            "print(value == DistributedType.NO, value != DistributedType.NO)",
        ]
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output = True, text = True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-2:]


def test_distributed_type_is_replaced_with_a_property():
    value = _patch_statement().value
    assert isinstance(value, ast.Call) and ast.unparse(value.func) == "property", (
        "Accelerator.distributed_type is a property upstream; a bare function binds "
        "as a method and never compares equal to a DistributedType member"
    )


def test_patched_distributed_type_reads_as_no():
    """Running the real assignment must leave the attribute reading as NO."""
    DistributedType = accelerate.utils.dataclasses.DistributedType
    for state_type in (None, "NO", "MULTI_GPU", "FSDP"):
        value, comparisons = _read_distributed_type(state_type)
        assert value == repr(DistributedType.NO), (state_type, value)
        # The second is what accelerate's device_map guards actually test.
        assert comparisons == "True False", (state_type, comparisons)


def test_patched_distributed_type_keeps_deepspeed():
    """Under DeepSpeed, NO enables native AMP and get_mixed_precision_context_manager's
    DeepSpeed fallthrough is a dtype-less, i.e. float16, autocast."""
    value, comparisons = _read_distributed_type("DEEPSPEED")
    assert value == repr(accelerate.utils.dataclasses.DistributedType.DEEPSPEED), value
    assert comparisons == "False True", comparisons
