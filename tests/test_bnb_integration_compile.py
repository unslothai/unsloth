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

"""The 4bit paths in unsloth.kernels.utils (fast_dequantize, fast_gemv, matmul_lora and the
fast_lora autograd functions) must match bitsandbytes byte for byte, run on whatever stream is
current, survive CUDA graph capture, and trace under torch.compile without graph breaks.

Every correctness test runs twice: on the NF4 kernels and on the bitsandbytes ctypes fallback
(UNSLOTH_BNB_TRITON=0).
"""

import pytest
import torch

pytest.importorskip("bitsandbytes")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA/HIP device", allow_module_level = True)

import unsloth  # noqa: F401  (sets UNSLOTH_IS_PRESENT before transformers)
import bitsandbytes as bnb
import bitsandbytes.functional as F
from unsloth.kernels import utils as U
from unsloth.kernels.fast_lora import (
    LoRA_W,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)

_NF4_KERNELS_AVAILABLE = U._USE_NF4_KERNELS
DEVICE = "cuda"


@pytest.fixture(params = ["nf4_kernels", "ctypes_fallback"])
def path(request, monkeypatch):
    use = request.param == "nf4_kernels"
    if use and not _NF4_KERNELS_AVAILABLE:
        pytest.skip("NF4 kernels unavailable or disabled")
    monkeypatch.setattr(U, "_USE_NF4_KERNELS", use)
    U._SCRATCH.clear()
    yield request.param
    U._SCRATCH.clear()


@pytest.fixture
def nf4_kernels(monkeypatch):
    if not _NF4_KERNELS_AVAILABLE:
        pytest.skip("NF4 kernels unavailable or disabled")
    monkeypatch.setattr(U, "_USE_NF4_KERNELS", True)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _quantize(shape, dtype, nested = True, seed = 0):
    g = torch.Generator(device = DEVICE).manual_seed(seed)
    W = torch.randn(shape, dtype = dtype, device = DEVICE, generator = g)
    return F.quantize_4bit(W, quant_type = "nf4", compress_statistics = nested)


def _as_list(s):
    # The pre TimDettmers/bitsandbytes#763 layout fast_dequantize still accepts.
    state2 = [s.state2.absmax, s.state2.code, s.state2.blocksize, None, None, None, None]
    return [s.absmax, s.shape, s.dtype, s.blocksize, [s.offset, state2], s.quant_type, s.code]


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(1024, 768), (130, 70), (4096, 1024)])
@pytest.mark.parametrize("global_buffer", [False, True])
def test_dequantize_matches_bitsandbytes(path, dtype, nested, shape, global_buffer):
    q, s = _quantize(shape, dtype, nested)
    ref = F.dequantize_4bit(q, s)
    out = U.fast_dequantize(q, s, use_global_buffer = global_buffer)
    assert torch.equal(_bits(out), _bits(ref))
    if nested:
        out_list = U.fast_dequantize(q, _as_list(s), use_global_buffer = global_buffer)
        assert torch.equal(_bits(out_list), _bits(ref))
    # The backward passes dequantize W.t(); a single packed column must come back transposed.
    out_t = U.fast_dequantize(q.t(), s)
    assert out_t.shape == (shape[1], shape[0])
    assert torch.equal(_bits(out_t.t()), _bits(ref))


def test_explicit_out(path):
    q, s = _quantize((512, 256), torch.bfloat16)
    out = torch.empty((512, 256), dtype = torch.bfloat16, device = DEVICE)
    got = U.fast_dequantize(q, s, out = out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(_bits(out), _bits(F.dequantize_4bit(q, s)))


def test_fp4_state_uses_its_own_codebook(path):
    W = torch.randn(256, 128, dtype = torch.bfloat16, device = DEVICE)
    q, s = F.quantize_4bit(W, quant_type = "fp4", compress_statistics = True)
    assert torch.equal(_bits(U.fast_dequantize(q, s)), _bits(F.dequantize_4bit(q, s)))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemv_matches_reference(path, dtype):
    q, s = _quantize((2048, 1024), dtype)
    X = torch.randn(1, 1, 1024, dtype = dtype, device = DEVICE)
    got = U.fast_gemv(X, q, s)
    ref = X.float() @ F.dequantize_4bit(q, s).float().t()
    assert got.shape == (1, 1, 2048)
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, rel


def test_two_side_streams_get_their_own_scratch(path):
    """Two streams dequantizing different weights into the global buffer at the same time: each
    must read its own weight back, and each stream must own a distinct scratch."""
    (q1, s1), (q2, s2) = _quantize((4096, 4096), torch.bfloat16, seed = 1), _quantize(
        (4096, 4096), torch.bfloat16, seed = 2
    )
    X = torch.randn(64, 4096, dtype = torch.bfloat16, device = DEVICE)
    refs = [X @ F.dequantize_4bit(q, s).t() for q, s in ((q1, s1), (q2, s2))]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outs = [None, None]
    torch.cuda.synchronize()
    for _ in range(10):
        for i, (st, (q, s)) in enumerate(zip(streams, ((q1, s1), (q2, s2)))):
            with torch.cuda.stream(st):
                W = U.fast_dequantize(q, s, use_global_buffer = True)
                outs[i] = X @ W.t()
    torch.cuda.synchronize()
    for got, ref in zip(outs, refs):
        assert torch.equal(got, ref)
    weight_keys = {k for k in U._SCRATCH if k[0] == "weight"}
    assert {k[2] for k in weight_keys} >= {st.cuda_stream for st in streams}
    ptrs = {U._SCRATCH[k].data_ptr() for k in weight_keys}
    assert len(ptrs) == len(weight_keys)


def test_scratch_first_allocated_under_inference_mode_still_trains(path):
    """generate() runs under torch.inference_mode(); the scratch it allocates first must stay
    usable by the training step that follows."""
    q, s = _quantize((512, 256), torch.bfloat16)
    A = (torch.randn(8, 256, dtype = torch.bfloat16, device = DEVICE) * 0.02).requires_grad_()
    B = (torch.randn(512, 8, dtype = torch.bfloat16, device = DEVICE) * 0.02).requires_grad_()
    X = torch.randn(2, 4, 256, dtype = torch.bfloat16, device = DEVICE)
    with torch.inference_mode():
        U.matmul_lora(X, q, s, A.detach(), B.detach(), 2.0)
    assert all(not t.is_inference() for t in U._SCRATCH.values())

    def step():
        x = X.clone().requires_grad_()
        LoRA_W.apply(x, q, s, A, B, 2.0).float().pow(2).mean().backward()
        grads = (x.grad.clone(), A.grad.clone(), B.grad.clone())
        A.grad = B.grad = None
        return grads

    after_generate = step()
    U._SCRATCH.clear()
    fresh = step()
    for a, b in zip(after_generate, fresh):
        assert torch.equal(a, b)


def test_matmul_lora_cuda_graph_capture_and_replay(path):
    q1, s1 = _quantize((1024, 512), torch.bfloat16, seed = 3)
    q2, s2 = _quantize((1024, 512), torch.bfloat16, seed = 4)
    A = torch.randn(8, 512, dtype = torch.bfloat16, device = DEVICE) * 0.02
    B = torch.randn(1024, 8, dtype = torch.bfloat16, device = DEVICE) * 0.02
    static_x = torch.randn(4, 512, dtype = torch.bfloat16, device = DEVICE)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            U.matmul_lora(static_x, q1, s1, A, B, 2.0)
            U.matmul_lora(static_x, q2, s2, A, B, 2.0)
    torch.cuda.current_stream().wait_stream(side)

    g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        static_y1 = U.matmul_lora(static_x, q1, s1, A, B, 2.0)
    with torch.cuda.graph(g2):
        static_y2 = U.matmul_lora(static_x, q2, s2, A, B, 2.0)

    for seed in range(3):
        static_x.copy_(torch.randn(4, 512, dtype = torch.bfloat16, device = DEVICE))
        g1.replay()
        # An eager call in between may grow or reuse the eager scratch; the graphs must not care.
        big_q, big_s = _quantize((4096, 2048), torch.bfloat16, seed = 10 + seed)
        U.fast_dequantize(big_q, big_s, use_global_buffer = True)
        g2.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_y1, U.matmul_lora(static_x, q1, s1, A, B, 2.0))
        assert torch.equal(static_y2, U.matmul_lora(static_x, q2, s2, A, B, 2.0))


# ---------------------------------------------------------------------------------------------
# torch.compile: every entry point traces as one graph and matches eager bit for bit.


def _lora_block(dtype = torch.bfloat16, D = 256, H = 512):
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            L = lambda i, o: bnb.nn.Linear4bit(
                i, o, bias = False, compute_dtype = dtype, quant_type = "nf4"
            )
            self.gate_proj, self.up_proj, self.down_proj = L(D, H), L(D, H), L(H, D)
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (
                L(D, D),
                L(D, 128),
                L(D, 128),
                L(D, D),
            )

    block = Block()
    for module in block.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.weight = bnb.nn.Params4bit(
                (torch.randn(module.out_features, module.in_features) * 0.05).to(dtype),
                requires_grad = False,
                quant_type = "nf4",
                compress_statistics = True,
            )
    targets = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    model = get_peft_model(
        block.to(DEVICE), LoraConfig(r = 8, lora_alpha = 16, target_modules = targets)
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    return model, model.base_model.model


def _fwd_bwd(model, fn, X):
    x = X.clone().requires_grad_()
    out = fn(x)
    outs = out if isinstance(out, tuple) else (out,)
    sum(o.float().pow(2).mean() for o in outs).backward()
    grads = [x.grad] + [p.grad for p in model.parameters() if p.grad is not None]
    result = [o.detach().clone() for o in outs] + [g.clone() for g in grads]
    model.zero_grad(set_to_none = True)
    return result


@pytest.mark.parametrize("which", ["mlp", "qkv", "o"])
def test_fast_lora_compiles_fullgraph(nf4_kernels, which):
    model, block = _lora_block()
    fn = {
        "mlp": lambda x: apply_lora_mlp_swiglu(block, x),
        "qkv": lambda x: apply_lora_qkv(block, x, inplace = False),
        "o": lambda x: apply_lora_o(block, x),
    }[which]
    X = torch.randn(2, 16, 256, dtype = torch.bfloat16, device = DEVICE)
    eager = _fwd_bwd(model, fn, X)
    explained = torch._dynamo.explain(fn)(X.clone().requires_grad_())
    assert explained.graph_break_count == 0, explained.break_reasons
    torch._dynamo.reset()
    compiled = _fwd_bwd(model, torch.compile(fn, fullgraph = True), X)
    assert len(compiled) == len(eager)
    for a, b in zip(eager, compiled):
        assert torch.equal(a, b)


def test_primitives_compile_fullgraph(nf4_kernels):
    q, s = _quantize((512, 256), torch.bfloat16)
    X1 = torch.randn(1, 1, 256, dtype = torch.bfloat16, device = DEVICE)
    X = torch.randn(2, 16, 256, dtype = torch.bfloat16, device = DEVICE)
    A = torch.randn(8, 256, dtype = torch.bfloat16, device = DEVICE) * 0.02
    B = torch.randn(512, 8, dtype = torch.bfloat16, device = DEVICE) * 0.02
    cases = [
        (lambda q: U.fast_dequantize(q, s, use_global_buffer = True) * 1, (q,)),
        (lambda q: U.fast_dequantize(q.t(), s) * 1, (q,)),
        (lambda x: U.fast_gemv(x, q, s) * 1, (X1,)),
        (lambda x: U.matmul_lora(x, q, s, A, B, 2.0) * 1, (X,)),
    ]
    for fn, args in cases:
        explained = torch._dynamo.explain(fn)(*args)
        assert explained.graph_break_count == 0, explained.break_reasons
        torch._dynamo.reset()
        assert torch.equal(torch.compile(fn, fullgraph = True)(*args), fn(*args))
        torch._dynamo.reset()


@pytest.mark.parametrize("bsz", [1, 2])
def test_fast_linear_forward_compiles_fullgraph(nf4_kernels, bsz):
    _, block = _lora_block()
    fn = lambda x: U.fast_linear_forward(block.o_proj, x)
    X = torch.randn(bsz, 1, 256, dtype = torch.bfloat16, device = DEVICE)
    with torch.inference_mode():
        eager = fn(X).clone()
        explained = torch._dynamo.explain(fn)(X)
        assert explained.graph_break_count == 0, explained.break_reasons
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph = True, backend = "aot_eager")(X)
    # aot_eager: inductor may lower the bsz=1 LoRA addmv differently (one bf16 ulp); the 4bit ops
    # themselves are what this checks.
    assert torch.equal(compiled, eager)
