# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tensor parallel export replica that does not fit the writer's GPU is gathered to the CPU
(models/tp_replica.py). A merged_16bit export of a 4-bit model then merges each projection with
save.py _merge_lora, whose dequantizer is CUDA only: every CPU-staged 4-bit weight is dequantized and merged
on the GPU it came from, one projection at a time, and the merged weight comes back on the CPU, equal to
the merge of the same weight left on the GPU. Needs a real CUDA GPU (one process, a 1-rank NCCL group)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from real_accelerator import has_real_cuda  # noqa: E402

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA GPU")


def _qlora(nested):
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(256, 192, bias = False)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(0)
    model = Tiny()
    dense = model.proj.weight.data.clone()
    q = bnb.nn.Linear4bit(
        256,
        192,
        bias = False,
        compute_dtype = torch.bfloat16,
        compress_statistics = nested,
        quant_type = "nf4",
    )
    q.weight = bnb.nn.Params4bit(
        dense, requires_grad = False, compress_statistics = nested, quant_type = "nf4"
    )
    model.proj = q
    model = model.cuda()  # quantizes on the GPU
    torch.manual_seed(1)
    return get_peft_model(
        model, LoraConfig(r = 8, lora_alpha = 16, target_modules = ["proj"], init_lora_weights = False)
    )


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("triton_nf4", [True, False], ids = ["triton", "ctypes"])
def test_cpu_staged_4bit_replica_merges_on_its_gpu(nested, triton_nf4, monkeypatch):
    import torch.distributed as dist

    from unsloth import save
    from unsloth.models import tp_native, tp_replica
    from unsloth.kernels import utils as kernel_utils
    from unsloth.save import _merge_lora

    # HMM hosts can dequantize CPU tensors correctly, so assert the device directly.
    real = save.fast_dequantize
    devices = []

    def cuda_only(
        W,
        quant_state = None,
        *args,
        **kwargs,
    ):
        devices.append((W.device.type, quant_state.absmax.device.type))
        assert W.is_cuda and quant_state.absmax.is_cuda, devices[-1]
        return real(W, quant_state, *args, **kwargs)

    monkeypatch.setattr(save, "fast_dequantize", cuda_only)
    if not triton_nf4:
        # Only the bitsandbytes ctypes path fails deterministically without staging (Triton may read via HMM).
        monkeypatch.setattr(kernel_utils, "_USE_NF4_KERNELS", False)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29617")
        dist.init_process_group("nccl", rank = 0, world_size = 1)
    group = dist.new_group([0])
    model = _qlora(nested)
    proj = model.base_model.model.proj
    base = proj.base_layer
    want, _ = _merge_lora(proj, "proj")  # the GPU replica's merge
    want = want.cpu()
    tp_native._shard_params4bit("proj", base, base.weight, tp_native.COLWISE, (192,), 0, 1)
    staged = tp_replica._full_params4bit(base.weight, base, group, torch.device("cpu"), True)
    base._parameters["weight"] = staged
    for factor in (proj.lora_A, proj.lora_B):
        factor["default"].weight.data = factor["default"].weight.data.cpu()
    assert staged.device.type == "cpu" and staged.quant_state.absmax.device.type == "cpu"
    got, _ = _merge_lora(proj, "proj")
    assert got.device.type == "cpu", got.device
    assert got.dtype == want.dtype and torch.equal(got, want)
    # The staged weight itself stays on the CPU.
    assert base.weight.device.type == "cpu"
