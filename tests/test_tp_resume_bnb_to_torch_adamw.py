# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tensor parallel checkpoint written with bitsandbytes AdamW resumes into torch AdamW (Studio steps sharded
full finetunes with torch AdamW): the moments carry over and the next step runs."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402


@pytest.mark.parametrize("bits", [8, 32])
def test_bitsandbytes_adamw_state_resumes_into_torch_adamw(bits):
    bnb = pytest.importorskip("bitsandbytes")
    from unsloth.models.fsdp2_bnb_optim import _dequantize, bnb_steps_cpu_tensors

    if not bnb_steps_cpu_tensors():
        pytest.skip("this bitsandbytes does not step CPU tensors")

    torch.manual_seed(0)
    n = 256 * 3 + 17
    saved_param = torch.nn.Parameter(torch.randn(n))
    optim = bnb.optim.AdamW8bit if bits == 8 else bnb.optim.AdamW
    saved_opt = optim([saved_param], lr = 1e-2, weight_decay = 0.01, min_8bit_size = 0)
    for _ in range(3):
        saved_param.grad = torch.randn(n)
        saved_opt.step()
    raw = saved_opt.state[saved_param]
    moments = {key: _dequantize(raw, key, torch.device("cpu")) for key in ("state1", "state2")}
    saved = tpn._optimizer_local_state(saved_opt)

    param = torch.nn.Parameter(saved_param.detach().clone())
    opt = torch.optim.AdamW([param], lr = 1e-2, weight_decay = 0.01)
    tpn._load_optimizer_local_state(opt, saved)
    state = opt.state[param]
    assert torch.equal(state["exp_avg"], moments["state1"])
    assert torch.equal(state["exp_avg_sq"], moments["state2"])
    assert float(state["step"]) == 3.0
    assert "optim_bits" not in opt.param_groups[0] and "foreach" in opt.param_groups[0]

    param.grad = torch.randn(n)
    before = param.detach().clone()
    opt.step()
    assert not torch.equal(param.detach(), before)
