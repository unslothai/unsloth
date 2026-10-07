# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

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
from unsloth.kernels import fast_lora
from unsloth.kernels.fast_lora import (
    LoRA_W,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)

_NF4_KERNELS_AVAILABLE = U._USE_NF4_KERNELS
DEVICE = "cuda"
# Inductor cannot emit bf16 Triton kernels on pre-Ampere GPUs, so use Unsloth's dtype.
CDTYPE = (
    torch.bfloat16
    if torch.version.hip or torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


@pytest.fixture(params = ["nf4_kernels", "ctypes_fallback"])
def path(request, monkeypatch):
    use = request.param == "nf4_kernels"
    if use and not _NF4_KERNELS_AVAILABLE:
        pytest.skip("NF4 kernels unavailable or disabled")
    monkeypatch.setattr(U, "_USE_NF4_KERNELS", use)
    # Exercise the Triton GEMV even where eager decode would keep bitsandbytes' (Triton < 3.7).
    monkeypatch.setattr(U, "_TRITON_GEMV_EAGER", use)
    U._SCRATCH.clear()
    yield request.param
    U._SCRATCH.clear()


@pytest.fixture
def nf4_kernels(monkeypatch):
    if not _NF4_KERNELS_AVAILABLE:
        pytest.skip("NF4 kernels unavailable or disabled")
    monkeypatch.setattr(U, "_USE_NF4_KERNELS", True)
    monkeypatch.setattr(U, "_TRITON_GEMV_EAGER", True)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _quantize(
    shape,
    dtype,
    nested = True,
    seed = 0,
):
    g = torch.Generator(device = DEVICE).manual_seed(seed)
    W = torch.randn(shape, dtype = dtype, device = DEVICE, generator = g)
    return F.quantize_4bit(W, quant_type = "nf4", compress_statistics = nested)


def _as_list(s):
    # The pre TimDettmers/bitsandbytes#763 layout fast_dequantize still accepts.
    state2 = [s.state2.absmax, s.state2.code, s.state2.blocksize, None, None, None, None]
    return [s.absmax, s.shape, s.dtype, s.blocksize, [s.offset, state2], s.quant_type, s.code]


_CONTROL = {}


def _torch_compiles_lora_exactly(backend):
    """Some torch versions (2.7) decompose the in-place LoRA addmm_/addmv_ differently when
    compiled, even on a plain 16bit weight with no 4bit op in sight. Measure that on a control so
    compiled-vs-eager checks stay exact wherever torch itself is exact."""
    if backend not in _CONTROL:
        g = torch.Generator(device = DEVICE).manual_seed(1)
        r = lambda *s: torch.randn(*s, dtype = CDTYPE, device = DEVICE, generator = g)
        W, A, B, X, x = r(512, 256), r(8, 256), r(512, 8), r(32, 256), r(256)

        def control(X, x):
            out = X @ W.t()
            out.addmm_(X @ A.t(), B.t(), alpha = 2.0)
            v = W @ x
            v.addmv_(B, A @ x, alpha = 2.0)
            return out, v

        torch._dynamo.reset()
        compiled = torch.compile(control, fullgraph = True, backend = backend)(X, x)
        torch._dynamo.reset()
        _CONTROL[backend] = all(torch.equal(a, b) for a, b in zip(compiled, control(X, x)))
    return _CONTROL[backend]


_TRACES_PARAMS4BIT = []


@torch.library.custom_op("unsloth_test::passthrough", mutates_args = ())
def _passthrough(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_passthrough.register_fake
def _(x):
    return torch.empty_like(x)


def _dynamo_traces_params4bit():
    """Older Dynamo (torch 2.7) cannot hand a Params4bit weight to any torch.library op (nor call
    some of its methods) without a graph break, whatever the op does. There the fast_lora paths
    can only be checked for correctness under torch.compile, not for a single graph. Probed with
    a throwaway op so a regression in the 4bit ops cannot turn the full-graph checks off."""
    if not _TRACES_PARAMS4BIT:
        lin = bnb.nn.Linear4bit(64, 64, bias = False, quant_type = "nf4").to(DEVICE)
        fn = lambda x: x + _passthrough(lin.weight)[0, 0] + lin.weight.t()[0, 0]
        torch._dynamo.reset()
        breaks = torch._dynamo.explain(fn)(torch.ones(1, device = DEVICE)).graph_break_count
        _TRACES_PARAMS4BIT.append(breaks == 0)
        torch._dynamo.reset()
    return _TRACES_PARAMS4BIT[0]


def _assert_compiled_matches(
    compiled,
    eager,
    backend = "inductor",
    exact = None,
):
    if exact is None:
        exact = _torch_compiles_lora_exactly(backend)
    if exact:
        assert torch.equal(compiled, eager)
        return
    scale = eager.float().abs().max().clamp_min(1e-6)
    err = ((compiled.float() - eager.float()).abs().max() / scale).item()
    assert err < 2e-2, err


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


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float32], ids = str)
def test_float_packed_storage(path, storage):
    """FSDP-QLoRA packs the 4bit weight into float storage (quant_storage); both the dequant and
    the GEMV must read it as bytes, as the ctypes path always did."""
    W = torch.randn(1024, 512, dtype = torch.bfloat16, device = DEVICE)
    q, s = F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True, quant_storage = storage)
    ref = F.dequantize_4bit(q, s)
    assert torch.equal(_bits(U.fast_dequantize(q, s)), _bits(ref))
    X = torch.randn(1, 1, 512, dtype = torch.bfloat16, device = DEVICE)
    got = U.fast_gemv(X, q, s).float()
    want = (X @ ref.t()).float()
    assert ((got - want).abs().max() / want.abs().max()).item() < 1e-2


def test_side_streams_never_share_the_scratch(path):
    """Two streams dequantizing different weights with use_global_buffer at the same time: each
    must read its own weight back. Only the default stream owns a scratch, so a program making a
    new stream per step cannot pin a weight-sized buffer per stream."""
    (q1, s1), (q2, s2) = (
        _quantize((4096, 4096), torch.bfloat16, seed = 1),
        _quantize((4096, 4096), torch.bfloat16, seed = 2),
    )
    X = torch.randn(64, 4096, dtype = torch.bfloat16, device = DEVICE)
    refs = [X @ F.dequantize_4bit(q, s).t() for q, s in ((q1, s1), (q2, s2))]
    outs = [None, None]
    torch.cuda.synchronize()
    for _ in range(10):
        for i, (q, s) in enumerate(((q1, s1), (q2, s2))):
            with torch.cuda.stream(torch.cuda.Stream()):
                W = U.fast_dequantize(q, s, use_global_buffer = True)
                outs[i] = X @ W.t()
    torch.cuda.synchronize()
    for got, ref in zip(outs, refs):
        assert torch.equal(got, ref)
    assert not U._SCRATCH
    U.fast_dequantize(q1, s1, use_global_buffer = True)
    assert {k[0] for k in U._SCRATCH} <= {"weight", "absmax"}
    assert len([k for k in U._SCRATCH if k[0] == "weight"]) == 1


@pytest.mark.parametrize("shape", [(512, 96), (256, 4160), (130, 100)])
def test_gemv_rows_not_starting_a_block_fall_back_to_dequant(path, shape):
    """Both GEMV kernels assume each row starts a quantization block; other shapes must still be
    right (bitsandbytes' own GEMV is wrong for them too)."""
    n, k = shape
    q, s = _quantize(shape, torch.bfloat16)
    X = torch.randn(1, 1, k, dtype = torch.bfloat16, device = DEVICE)
    ref = (X @ F.dequantize_4bit(q, s).t()).float()
    got = U.fast_gemv(X, q, s).float()
    assert got.shape == (1, 1, n)
    assert ((got - ref).abs().max() / ref.abs().max()).item() < 1e-2


def test_gemv_side_stream_and_cuda_graph(path):
    q, s = _quantize((4096, 4096), torch.bfloat16)
    X = torch.randn(1, 1, 4096, dtype = torch.bfloat16, device = DEVICE)
    expected = U.fast_gemv(X, q, s)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        x = X.clone() * 1
        results = [U.fast_gemv(x, q, s) for _ in range(10)]
        for _ in range(2):
            U.fast_gemv(x, q, s)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert all(torch.equal(r, expected) for r in results)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = U.fast_gemv(x, q, s)
    x.copy_(X * 2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, U.fast_gemv(X * 2, q, s))


def test_kernel_failure_falls_back_to_bitsandbytes(nf4_kernels, monkeypatch):
    """A Triton compile failure on an untested GPU switches to the bitsandbytes kernels."""
    q, s = _quantize((512, 256), torch.bfloat16)

    def broken(*args, **kwargs):
        raise RuntimeError("PTX compile failed")

    for name in ("dequantize_nf4", "dequantize_nf4_planned", "gemv_nf4", "gemv_nf4_planned"):
        monkeypatch.setattr(U, name, broken)
    assert torch.equal(_bits(U.fast_dequantize(q, s)), _bits(F.dequantize_4bit(q, s)))
    assert U._USE_NF4_KERNELS is False
    monkeypatch.setattr(U, "_USE_NF4_KERNELS", True)
    X = torch.randn(1, 1, 256, dtype = torch.bfloat16, device = DEVICE)
    assert U.fast_gemv(X, q, s).shape == (1, 1, 512)
    assert U._USE_NF4_KERNELS is False
    monkeypatch.setattr(U, "_USE_NF4_KERNELS", True)
    got = U.fast_dequantize(q.t(), s, use_global_buffer = True).t()
    assert torch.equal(_bits(got), _bits(F.dequantize_4bit(q, s)))
    assert U._USE_NF4_KERNELS is False


def test_out_of_memory_is_not_mistaken_for_a_kernel_failure(nf4_kernels, monkeypatch):
    q, s = _quantize((512, 256), torch.bfloat16)

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(U, "dequantize_nf4", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        U.fast_dequantize(q, s)
    assert U._USE_NF4_KERNELS is True


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


def _lora_block(
    dtype = torch.bfloat16,
    D = 256,
    H = 512,
):
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
    model = get_peft_model(block.to(DEVICE), LoraConfig(r = 8, lora_alpha = 16, target_modules = targets))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    return model, model.base_model.model


_LORA_FNS = {
    "mlp": lambda block: (lambda x: apply_lora_mlp_swiglu(block, x)),
    "qkv": lambda block: (lambda x: apply_lora_qkv(block, x, inplace = False)),
    "o": lambda block: (lambda x: apply_lora_o(block, x)),
}


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
    model, block = _lora_block(dtype = CDTYPE)
    fn = _LORA_FNS[which](block)
    X = torch.randn(2, 16, 256, dtype = CDTYPE, device = DEVICE)
    eager = _fwd_bwd(model, fn, X)
    # Before torch 2.11 the LoRA Functions stay opaque to torch.compile.
    fullgraph = _dynamo_traces_params4bit() and fast_lora.TRACE_LORA_FUNCTIONS
    if fullgraph:
        explained = torch._dynamo.explain(fn)(X.clone().requires_grad_())
        assert explained.graph_break_count == 0, explained.break_reasons
    torch._dynamo.reset()
    compiled = _fwd_bwd(model, torch.compile(fn, fullgraph = fullgraph), X)
    assert len(compiled) == len(eager)
    for a, b in zip(eager, compiled):
        _assert_compiled_matches(b, a, exact = False)


def test_primitives_compile_fullgraph(nf4_kernels):
    q, s = _quantize((512, 256), CDTYPE)
    X1 = torch.randn(1, 1, 256, dtype = CDTYPE, device = DEVICE)
    X = torch.randn(2, 16, 256, dtype = CDTYPE, device = DEVICE)
    A = torch.randn(8, 256, dtype = CDTYPE, device = DEVICE) * 0.02
    B = torch.randn(512, 8, dtype = CDTYPE, device = DEVICE) * 0.02
    cases = [
        (lambda q: U.fast_dequantize(q, s, use_global_buffer = True) * 1, (q,), True),
        (lambda q: U.fast_dequantize(q.t(), s) * 1, (q,), True),
        # Inductor re-emits the GEMV reduction and may round a partial sum 1 ulp differently.
        (lambda x: U.fast_gemv(x, q, s) * 1, (X1,), "close"),
        (lambda x: U.matmul_lora(x, q, s, A, B, 2.0) * 1, (X,), False),
    ]
    for fn, args, always_exact in cases:
        explained = torch._dynamo.explain(fn)(*args)
        assert explained.graph_break_count == 0, explained.break_reasons
        torch._dynamo.reset()
        compiled, eager = torch.compile(fn, fullgraph = True)(*args), fn(*args)
        if always_exact == "close":
            _assert_compiled_matches(compiled, eager, exact = False)
        elif always_exact:
            assert torch.equal(compiled, eager)
        else:
            _assert_compiled_matches(compiled, eager)
        torch._dynamo.reset()


@pytest.mark.parametrize("bsz", [1, 2])
def test_fast_linear_forward_compiles_fullgraph(nf4_kernels, bsz):
    _, block = _lora_block(dtype = CDTYPE)
    fn = lambda x: U.fast_linear_forward(block.o_proj, x)
    X = torch.randn(bsz, 1, 256, dtype = CDTYPE, device = DEVICE)
    with torch.inference_mode():
        eager = fn(X).clone()
        fullgraph = _dynamo_traces_params4bit()
        if fullgraph:
            explained = torch._dynamo.explain(fn)(X)
            assert explained.graph_break_count == 0, explained.break_reasons
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph = fullgraph, backend = "aot_eager")(X)
    # aot_eager: inductor may lower the bsz=1 LoRA addmv one bf16 ulp differently.
    _assert_compiled_matches(compiled, eager, backend = "aot_eager")


def test_eager_gemv_gate_off_takes_bitsandbytes(monkeypatch):
    """Where the Triton GEMV is not the eager choice (HIP before Triton 3.7), eager decode takes
    bitsandbytes' GEMV, on the live stream. On CUDA the Triton GEMV is the eager choice."""
    from unsloth.kernels import nf4_gemv

    if torch.version.hip is None and _NF4_KERNELS_AVAILABLE:
        assert nf4_gemv.triton_gemv_eager()
    if not _NF4_KERNELS_AVAILABLE:
        pytest.skip("NF4 kernels unavailable or disabled")
    monkeypatch.setattr(U, "_USE_NF4_KERNELS", True)
    monkeypatch.setattr(U, "_TRITON_GEMV_EAGER", False)
    q, s = _quantize((512, 256), torch.bfloat16)
    X = torch.randn(1, 1, 256, dtype = torch.bfloat16, device = DEVICE)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = U.fast_gemv(X, q, s)
    stream.synchronize()
    assert torch.equal(out, U._fast_gemv_ctypes(X, q, s))


def test_graph_break_is_not_mistaken_for_a_kernel_failure(nf4_kernels, monkeypatch):
    """A Dynamo exception surfacing through fast_dequantize (e.g. an older Dynamo tracing the
    fallback's try/except) must re-raise, not switch the kernels off for the process."""
    q, s = _quantize((512, 256), torch.bfloat16)

    def unsupported(*args, **kwargs):
        raise torch._dynamo.exc.Unsupported("Data-dependent branching")

    monkeypatch.setattr(U, "dequantize_nf4", unsupported)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        U.fast_dequantize(q, s)
    assert U._USE_NF4_KERNELS is True


# Runs as a module-level script: only that shape reproduced the torch 2.10 stale reads.
_STALE_MEMORY_SCRIPT = """
import sys, torch
sys.path.insert(0, sys.argv[1])
from unsloth.kernels import fast_lora
import test_bnb_integration_compile as t

model, block = t._lora_block(dtype = getattr(torch, sys.argv[3]))
fn = lambda x: getattr(fast_lora, sys.argv[2])(block, x)
X = torch.randn(2, 16, 256, dtype = getattr(torch, sys.argv[3]), device = "cuda")
compiled = torch.compile(fn)
eager = t._fwd_bwd(model, fn, X)
bad = 0
for _ in range(10):
    poison = [torch.full((n,), float("nan"), device = "cuda") for n in (2048, 4096, 8192, 16384, 65536, 131072)]
    del poison
    # The previous step's outputs stay alive until this one returns, as in a training loop.
    out = t._fwd_bwd(model, compiled, X)
    bad += sum((~torch.isfinite(o)).sum().item() for o in out)
print("NONFINITE", bad)
"""


@pytest.mark.parametrize("mlp", ["swiglu", "geglu_exact", "geglu_approx"])
def test_compiled_lora_mlp_backward_never_reads_stale_memory(mlp):
    """torch 2.10 miscompiles the traced LoRA_MLP backward: with freed NaN memory around, LoRA
    gradients came back non-finite. There the LoRA Functions must stay opaque to torch.compile."""
    import os, subprocess, sys

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            _STALE_MEMORY_SCRIPT,
            tests_dir,
            f"apply_lora_mlp_{mlp}",
            str(CDTYPE).split(".")[-1],
        ],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert "NONFINITE 0" in run.stdout, run.stdout[-2000:] + run.stderr[-3000:]
