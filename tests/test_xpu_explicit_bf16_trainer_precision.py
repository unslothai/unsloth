# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

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
