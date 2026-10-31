# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real FlashOptim on two GPUs: an update of a parameter on a non-current device (device_map
split model) must match the same update run with that device current, bit for bit."""

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason = "requires two visible CUDA GPUs",
    ),
]


@pytest.mark.parametrize(
    "dtype,quantize",
    [
        (torch.float32, False),
        (torch.bfloat16, False),
        (torch.bfloat16, True),
        (torch.float16, True),
    ],
)
@pytest.mark.parametrize("entry", ["step", "step_param"])
def test_adamw_flash_noncurrent_cuda_device(dtype, quantize, entry):
    pytest.importorskip("flashoptim")
    from unsloth.optimizers.flash_adamw import _get_flash_adamw_class

    torch.cuda.set_device(0)
    optimizer_cls = _get_flash_adamw_class()
    p0 = torch.nn.Parameter(torch.ones(33, device = "cuda:0", dtype = dtype))
    p1 = torch.nn.Parameter(torch.ones(33, device = "cuda:1", dtype = dtype))
    q0 = torch.nn.Parameter(p0.detach().clone())
    q1 = torch.nn.Parameter(p1.detach().clone())
    optimizer = optimizer_cls([p0, p1], quantize = quantize, lr = 0.001)
    reference = optimizer_cls([q0, q1], quantize = quantize, lr = 0.001)
    for grad in (0.125, -0.25, 0.0, 0.0625):
        for p in (p0, p1, q0, q1):
            p.grad = torch.full_like(p, grad)
        for q in (q0, q1):
            with torch.cuda.device(q.device):
                reference.step_param(q)
        if entry == "step":
            optimizer.step()
        else:
            for p in (p0, p1):
                optimizer.step_param(p)
        assert torch.cuda.current_device() == 0, "the caller's device must be restored"
        for p, q in ((p0, q0), (p1, q1)):
            torch.cuda.synchronize(p.device)
            torch.testing.assert_close(p, q, rtol = 0, atol = 0)
            a, b = optimizer.state[p], reference.state[q]
            assert a["step"].item() == b["step"].item()
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    a[key].kernel_tensor, b[key].kernel_tensor, rtol = 0, atol = 0
                )
                torch.testing.assert_close(
                    a[key].kernel_scales_or_self, b[key].kernel_scales_or_self, rtol = 0, atol = 0
                )
            if "error_bits" in a:
                torch.testing.assert_close(a["error_bits"], b["error_bits"], rtol = 0, atol = 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adamw_flash_uncompressed_state_dict_round_trip_noncurrent_device(dtype):
    """compress_state_dict=False dequantizes the moments on save and requantizes them on load,
    both with Triton: a parameter on cuda:1 with cuda:0 current must save and resume."""
    pytest.importorskip("flashoptim")
    from unsloth.optimizers.flash_adamw import _get_flash_adamw_class

    torch.cuda.set_device(0)
    optimizer_cls = _get_flash_adamw_class()
    kwargs = dict(quantize = True, compress_state_dict = False, lr = 0.001)
    params = [torch.nn.Parameter(torch.ones(65, device = f"cuda:{i}", dtype = dtype)) for i in (0, 1)]
    optimizer = optimizer_cls(params, **kwargs)
    for grad in (0.125, -0.25):
        for p in params:
            p.grad = torch.full_like(p, grad)
        optimizer.step()
    expected = {}
    for i, p in enumerate(params):
        with torch.cuda.device(p.device):
            expected[i] = {
                k: optimizer.state[p][k].materialize() for k in ("exp_avg", "exp_avg_sq")
            }

    saved = optimizer.state_dict()
    assert torch.cuda.current_device() == 0, "the caller's device must be restored"
    for i, p in enumerate(params):
        for key, value in expected[i].items():
            assert saved["state"][i][key].device == p.device
            torch.testing.assert_close(saved["state"][i][key], value, rtol = 0, atol = 0)

    resumed = optimizer_cls(params, **kwargs)
    resumed.load_state_dict(saved)
    assert torch.cuda.current_device() == 0, "the caller's device must be restored"
    for i, p in enumerate(params):
        for key, value in expected[i].items():
            with torch.cuda.device(p.device):
                loaded = resumed.state[p][key].materialize()
            assert loaded.device == p.device
            torch.testing.assert_close(loaded.float(), value.float(), rtol = 0.02, atol = 1e-6)
        p.grad = torch.full_like(p, 0.0625)
    resumed.step()
