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

"""An explicit bf16 load stays bf16 on an XPU without native bf16, so the notebook idiom
`fp16 = not is_bfloat16_supported()` asks float16 for a bfloat16 model: follow the model."""

import pytest
import torch

from tests.test_float32_no_fp16_autocast import _run


def test_xpu_bf16_model_with_the_fp16_idiom_trains_in_bf16():
    args, env = _run(torch.bfloat16, bf16_supported = False, fp16 = True, device_type = "xpu")
    assert (args.fp16, args.bf16) == (False, True)
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_xpu_bf16_model_with_both_flags_trains_in_bf16():
    args, env = _run(torch.bfloat16, bf16_supported = False, fp16 = True, bf16 = True, device_type = "xpu")
    assert (args.fp16, args.bf16) == (False, True)
    assert env["ACCELERATE_MIXED_PRECISION"] == "bf16"


@pytest.mark.parametrize("device_type", ["cuda", "hip", "mps", None])
def test_other_devices_still_refuse_fp16_on_a_bf16_model(device_type):
    with pytest.raises(TypeError, match = "bfloat16 precision but you want to use float16"):
        _run(torch.bfloat16, bf16_supported = False, fp16 = True, device_type = device_type)


def test_native_bf16_xpu_still_refuses_an_explicit_fp16():
    # With native bf16 the idiom asks bf16, so fp16 here is the caller's own conflicting choice.
    with pytest.raises(TypeError, match = "bfloat16 precision but you want to use float16"):
        _run(torch.bfloat16, bf16_supported = True, fp16 = True, device_type = "xpu")


def test_xpu_fp16_model_with_the_fp16_idiom_unchanged():
    args, env = _run(torch.float16, bf16_supported = False, fp16 = True, device_type = "xpu")
    assert (args.fp16, args.bf16) == (True, False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "fp16"


def test_xpu_forced_float32_still_wins():
    args, env = _run(
        torch.bfloat16, bf16_supported = False, fp16 = True, force_float32 = "1", device_type = "xpu"
    )
    assert (args.fp16, args.bf16) == (False, False)
    assert env["ACCELERATE_MIXED_PRECISION"] == "no"
