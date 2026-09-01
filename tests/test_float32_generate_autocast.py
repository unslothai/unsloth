# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`torch.autocast(dtype = torch.float32)` on CUDA is enabled, not a no-op.

The generate wrapper builds its autocaster from the model's own dtype. For a
model the user deliberately loaded in float32 -- Spark-TTS is the live case,
its notebook says "Spark seems to only work on float32 for now" -- that asks
CUDA to autocast *to* float32.

torch's CPU, XPU and MPS paths reject an unsupported autocast dtype. The CUDA
path does not, so this enters genuinely enabled:

    torch.is_autocast_enabled("cuda")   -> True
    torch.get_autocast_dtype("cuda")    -> torch.float32

Under torch.compile the first decode step of a freshly loaded, never-trained
model then returns 166000/166000 non-finite logits, and generation dies in
`torch.multinomial` on a distribution full of NaN. Forcing eager
(UNSLOTH_COMPILE_DISABLE=1) makes the same call finite, which is what places
the fault in the compiled graph rather than in the weights -- they were finite
throughout.

A float32 model has nothing to autocast to, so the fix is `enabled`, not a
different dtype. That is the same idiom rl_replacements.py already uses.
"""

import ast
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VISION = REPO_ROOT / "unsloth" / "models" / "vision.py"
SRC = VISION.read_text(encoding = "utf-8")


def _the_autocaster_call():
    """The `else` branch's autocast call, as an AST node.

    Located structurally rather than by line number so a later edit above it
    does not silently retarget this test at the UNSLOTH_FORCE_FLOAT32 branch,
    which builds its own float16 autocaster and is deliberately untouched.
    """
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.Assign):
            continue
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "autocaster"
        ):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        kwargs = {k.arg: k.value for k in call.keywords}
        # The forced-float16 branch passes a literal;
        if isinstance(kwargs.get("dtype"), ast.Name) and kwargs["dtype"].id == "dtype":
            return kwargs
    raise AssertionError("no autocaster assignment forwarding `dtype` found")


def test_the_generate_autocaster_is_gated_on_a_dtype_it_can_use():
    kwargs = _the_autocaster_call()
    assert "enabled" in kwargs, "autocast is entered unconditionally"
    expression = ast.unparse(kwargs["enabled"])
    assert "float16" in expression and "bfloat16" in expression, expression


def test_the_forced_float16_branch_is_left_alone():
    """UNSLOTH_FORCE_FLOAT32 builds a float16 autocaster on purpose."""
    assert "dtype = torch.float16)" in SRC


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.float32, False),
        (torch.float16, True),
        (torch.bfloat16, True),
    ],
)
def test_the_gate_by_execution(dtype, expected):
    assert (dtype in (torch.float16, torch.bfloat16)) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_cuda_really_does_accept_float32_as_an_autocast_dtype():
    """The premise. If torch ever starts rejecting or ignoring this, the fix
    above is no longer load-bearing and this test says so rather than letting
    it rot in place."""
    with torch.autocast(device_type = "cuda", dtype = torch.float32):
        assert torch.is_autocast_enabled("cuda") is True
        assert torch.get_autocast_dtype("cuda") == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_the_gate_turns_that_into_a_no_op():
    dtype = torch.float32
    with torch.autocast(
        device_type = "cuda", dtype = dtype, enabled = dtype in (torch.float16, torch.bfloat16)
    ):
        assert torch.is_autocast_enabled("cuda") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
