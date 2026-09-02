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
guards on a single device (#10016). `import unsloth` needs CUDA or XPU, so these run
the statement from source in a fresh interpreter, which also catches a revert.
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


def test_distributed_type_is_replaced_with_a_property():
    value = _patch_statement().value
    assert isinstance(value, ast.Call) and ast.unparse(value.func) == "property", (
        "Accelerator.distributed_type is a property upstream; a bare function binds "
        "as a method and never compares equal to a DistributedType member"
    )


def test_patched_distributed_type_reads_as_no():
    """Running the real assignment must leave the attribute reading as NO."""
    statement = ast.unparse(_patch_statement())
    script = textwrap.dedent(
        f"""
        import accelerate.accelerator
        from accelerate.utils.dataclasses import DistributedType

        {statement}

        # Read it off an instance without standing up a real distributed backend.
        accelerator = accelerate.accelerator.Accelerator.__new__(
            accelerate.accelerator.Accelerator
        )
        value = accelerator.distributed_type
        print(repr(value))
        print(value == DistributedType.NO, value != DistributedType.NO)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output = True, text = True)
    assert result.returncode == 0, result.stderr
    value, comparisons = result.stdout.strip().splitlines()[-2:]
    assert value == repr(accelerate.utils.dataclasses.DistributedType.NO), value
    # The second is what accelerate's device_map guards actually test.
    assert comparisons == "True False", comparisons
