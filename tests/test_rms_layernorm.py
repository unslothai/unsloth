"""GPU correctness tests for unsloth.kernels.rms_layernorm's Triton kernel.

Compares Fast_RMS_Layernorm against HF's LlamaRMSNorm (llama-style) and
against Gemma's variant of RMSNorm (weight + 1 scaling, full fp32 compute).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "RMS layernorm Triton kernel requires a CUDA GPU"
)


@pytest.fixture(params = [torch.float16, torch.bfloat16])
def dtype(request):
    return request.param


@pytest.mark.parametrize("dim,seqlen,bsz", [(512, 349, 4), (1024, 128, 2), (2048, 17, 1)])
def test_fast_rms_layernorm_forward_and_backward_match_llama(dim, seqlen, bsz, dtype):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import fast_rms_layernorm

    torch.manual_seed(3407)
    eps = 1e-5
    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    torch.nn.init.uniform_(layernorm.weight)

    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    X_ref = X.clone().requires_grad_(True)
    X_fast = X.clone().requires_grad_(True)

    Y_ref = layernorm(X_ref)
    dY = torch.randn_like(Y_ref)
    Y_ref.backward(dY)

    Y_fast = fast_rms_layernorm(layernorm, X_fast)
    Y_fast.backward(dY)

    assert Y_fast.shape == X.shape
    assert Y_fast.dtype == X.dtype
    torch.testing.assert_close(Y_fast, Y_ref.detach(), atol = 5e-2, rtol = 5e-2)
    torch.testing.assert_close(X_fast.grad, X_ref.grad, atol = 5e-2, rtol = 5e-2)


def test_fast_rms_layernorm_gemma_matches_reference(dtype):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm

    torch.manual_seed(3407)
    dim, seqlen, bsz, eps = 512, 64, 3, 1e-6
    weight = torch.nn.Parameter(torch.rand(dim, device = "cuda"))
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")

    out = Fast_RMS_Layernorm.apply(X, weight, eps, True)

    Xf = X.float()
    var = (Xf * Xf).mean(-1, keepdim = True)
    normed = Xf * torch.rsqrt(var + eps)
    ref = normed * (weight.float() + 1.0)

    assert out.shape == X.shape
    torch.testing.assert_close(out.float(), ref, atol = 5e-2, rtol = 5e-2)


def test_fast_rms_layernorm_preserves_shape_with_leading_dims(dtype):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import fast_rms_layernorm

    torch.manual_seed(0)
    dim = 256
    layernorm = LlamaRMSNorm((dim,), eps = 1e-5).to("cuda")
    X = torch.randn((2, 3, 5, dim), dtype = dtype, device = "cuda")

    out = fast_rms_layernorm(layernorm, X)
    assert out.shape == X.shape


def test_patch_and_unpatch_rms_layernorm_roundtrip():
    import transformers.models.llama.modeling_llama as llama_mod

    from unsloth.kernels.rms_layernorm import (
        Unsloth_LlamaRMSNorm,
        patch_rms_layernorm,
        unpatch_rms_layernorm,
    )

    original = llama_mod.LlamaRMSNorm
    try:
        patch_rms_layernorm()
        assert llama_mod.LlamaRMSNorm is Unsloth_LlamaRMSNorm
    finally:
        unpatch_rms_layernorm()
    assert llama_mod.LlamaRMSNorm is original
