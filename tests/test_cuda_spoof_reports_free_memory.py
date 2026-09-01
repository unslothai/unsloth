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
"""Every CUDA spoof must report a plausible amount of FREE memory.

`torch.cuda.mem_get_info` returns `(free, total)` and delegates to
`cudart().cudaMemGetInfo`, so a spoof answering zero free describes an exhausted
card. The fused cross entropy then raises rather than chunking, which failed
`test_sft_trains_on_cpu` on a host with four idle GPUs and read as a product bug.

Source-level, because importing either spoof mutates the interpreter's torch.
"""

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent
_SPOOFS = ("conftest.py", "_zoo_aggressive_cuda_spoof.py")


def _memory_tuples(path):
    """Every `(free, total)` literal a memory probe in `path` hands back."""
    found = []
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.targets:
            name = getattr(node.targets[0], "attr", "") or ""
            values = [node.value]
        elif isinstance(node, ast.FunctionDef):
            name = node.name
            values = [n.value for n in ast.walk(node) if isinstance(n, ast.Return) and n.value]
        else:
            continue
        if "memgetinfo" not in name.lower().replace("_", ""):
            continue
        for value in values:
            if isinstance(value, ast.Lambda):
                value = value.body
            if not (isinstance(value, ast.Tuple) and len(value.elts) == 2):
                continue
            try:
                # `literal_eval` cannot fold `60 * 1024**3`, so evaluate with
                found.append(
                    tuple(eval(ast.unparse(e), {"__builtins__": {}}, {}) for e in value.elts)
                )
            except Exception:
                pass
    return found


@pytest.mark.parametrize("filename", _SPOOFS)
def test_a_spoofed_card_is_not_reported_as_full(filename):
    tuples = _memory_tuples(_ROOT / filename)
    assert tuples, f"no mem_get_info tuple found in {filename}; did it move?"
    for free, total in tuples:
        assert free > 0, f"{filename} reports {free} bytes free, i.e. an exhausted card"
        assert free <= total, f"{filename} reports more free ({free}) than total ({total})"
        # Half the free pool is the fused loss's chunk target, capped at 4GB.
        assert free >= 8 * 1024**3, f"{filename} reports only {free} bytes free"
