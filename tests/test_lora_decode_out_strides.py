# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""LoRA decode returns the caller's strided `out` as is, so torch.matmul on it still folds to one mm."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        reason = "real bnb 4-bit / CUDA decode path: needs a CUDA GPU", allow_module_level = True
    )
peft = pytest.importorskip("peft")
bnb = pytest.importorskip("bitsandbytes")

import unsloth  # noqa: F401  (must precede transformers / peft users)
from unsloth.kernels.utils import fast_linear_forward

DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
N, K = 512, 256


class _Tiny(torch.nn.Module):
    def __init__(self, load_in_4bit):
        super().__init__()
        if load_in_4bit:
            self.q = bnb.nn.Linear4bit(K, N, bias = False, compute_dtype = DTYPE, quant_type = "nf4")
            w = torch.randn(N, K, generator = torch.Generator().manual_seed(0)) * 0.02
            self.q.weight = bnb.nn.Params4bit(w.to(DTYPE), requires_grad = False, quant_type = "nf4")
        else:
            self.q = torch.nn.Linear(K, N, bias = False, dtype = DTYPE)


def _proj(load_in_4bit):
    model = _Tiny(load_in_4bit).cuda()
    model = peft.get_peft_model(
        model, peft.LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q"], lora_dropout = 0.0)
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn_like(p) * 0.05)
    proj = model.base_model.model.q
    proj.eval()
    return proj


@pytest.mark.parametrize("load_in_4bit", [False, True])
@pytest.mark.parametrize("bsz", [1, 2, 8, 33])
def test_lora_decode_returns_out_with_its_strides(load_in_4bit, bsz):
    proj = _proj(load_in_4bit)
    X = torch.randn(bsz, 1, K, device = "cuda", dtype = DTYPE)
    buffer = torch.empty(bsz, 1, N + 64, device = "cuda", dtype = DTYPE)
    out = buffer[:, :, :N]
    with torch.inference_mode():
        y = fast_linear_forward(proj, X, out = out)
        ref = proj(X)
    assert y.data_ptr() == out.data_ptr() and y.shape == out.shape and y.stride() == out.stride()
    # torch.matmul's fold precondition for a 3-D input.
    assert y.stride(0) == y.stride(1) * y.shape[1]
    torch.testing.assert_close(y.float(), ref.float(), rtol = 2e-2, atol = 2e-2)
