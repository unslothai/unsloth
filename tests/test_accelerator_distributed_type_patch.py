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
"""The single-device block in `_utils.py` forces accelerate into non-distributed mode.

`Accelerator.distributed_type` is a `@property` upstream, so assigning a plain
function to it binds as a method rather than replacing the value. Accelerate then
compares a bound method against the enum, e.g.

    if (... and self.verify_device_map(obj)
            and self.distributed_type != DistributedType.NO ...):
        raise ValueError("You can't train a model that has been loaded with "
                         "`device_map='auto'` in any distributed mode. ...")

which is always True, so the guard fires on a single device -- exactly the case the
block exists to rule out. Reported as #10016 for a single-GPU 4-bit QLoRA run whose
model legitimately spanned devices.

`import unsloth` needs a CUDA or XPU build (`torch_amp_custom_fwd` is only defined
on those branches), so these run the real assignment against the real accelerate
class in a fresh interpreter instead, taking the statement from the source so a
revert to a bare lambda is caught.
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
    """The one assignment `_utils.py` makes to `Accelerator.distributed_type`."""
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
    """The assignment, run for real, has to leave the attribute reading as NO."""
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
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output = True, text = True
    )
    assert result.returncode == 0, result.stderr
    value, comparisons = result.stdout.strip().splitlines()[-2:]
    assert value == repr(accelerate.utils.dataclasses.DistributedType.NO), value
    # The second is what accelerate's device_map guards actually test.
    assert comparisons == "True False", comparisons
