# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The grouped GEMM dW backward adds straight into an existing param.grad (fp32 add, one rounding)
instead of returning a fresh dW for AccumulateGrad to add. Checked against the returned-gradient
path (UNSLOTH_MOE_DW_ACCUMULATE=0) and an fp64 reference, plus every case that must fall back."""

import os
import re
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

pytest.importorskip("triton", reason = "the grouped GEMM is a Triton kernel")
if not torch.cuda.is_available():
    pytest.skip("grouped GEMM needs a CUDA device", allow_module_level = True)

try:
    from unsloth.kernels.moe.grouped_gemm import interface as gg
    from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
        KernelConfigBackward_dW,
        KernelConfigBackward_dX,
        KernelConfigForward,
    )
except Exception as exc:  # pragma: no cover - depends on the installed stack
    pytest.skip(f"grouped_gemm is unimportable here: {exc}", allow_module_level = True)

DEV = "cuda"
TMA_STORE = [False, True] if gg.supports_tma() else [False]


def _cfgs(bn = 64, bk = 64, tma_store = False):
    fwd = KernelConfigForward(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = bn, BLOCK_SIZE_K = bk)
    dx = KernelConfigBackward_dX(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = bn, BLOCK_SIZE_K = bk)
    dw = KernelConfigBackward_dW(
        BLOCK_SIZE_M = 64, BLOCK_SIZE_N = bn, BLOCK_SIZE_K = bk, use_tma_store = tma_store
    )
    return fwd, dx, dw


def _routing(T, E, topk, empty = ()):
    logits = torch.randn(T, E, device = DEV)
    for e in empty:
        logits[:, e] = -float("inf")
    idx = torch.topk(logits, topk).indices.view(-1)
    m_sizes = torch.bincount(idx, minlength = E).to(torch.int32)
    gather = torch.argsort(idx, stable = True).to(torch.int32)
    return m_sizes, gather


def _gemm(X, W, m_sizes, gather, topk, mode, cfgs):
    fwd, dx, dw = cfgs
    return gg.grouped_gemm(
        X = X, W = W, m_sizes = m_sizes, topk = topk, gather_indices = gather,
        permute_x = mode == "permute_x", permute_y = mode == "permute_y",
        kernel_config_fwd = fwd, kernel_config_bwd_dX = dx, kernel_config_bwd_dW = dw,
        is_first_gemm = mode != "permute_y",
    )  # fmt: skip


def _ref_dW(X, G, m_sizes, gather, topk, mode, E, N, K):
    """fp64 dW = sum over the expert's rows of dY^T X, in the kernel's expert-grouped row order."""
    X, G, gather = X.double(), G.double(), gather.long()
    if mode == "permute_x":
        xs, dys = X[gather // topk], G
    elif mode == "permute_y":
        xs, dys = X, G[gather]
    else:
        xs, dys = X, G
    out = torch.zeros(E, N, K, dtype = torch.float64, device = X.device)
    start = 0
    for e, m in enumerate(m_sizes.tolist()):
        out[e] = dys[start : start + m].T @ xs[start : start + m]
        start += m
    return out


class _Case:
    def __init__(self, E, N, K, T, topk, mode, dtype, empty = (), seed = 0):
        g = torch.Generator(device = DEV).manual_seed(seed)
        torch.manual_seed(seed)
        self.E, self.N, self.K, self.topk, self.mode = E, N, K, topk, mode
        self.m_sizes, self.gather = _routing(T, E, topk, empty)
        rows = T if mode == "permute_x" else T * topk
        self.X = torch.randn(rows, K, device = DEV, generator = g).to(dtype)
        self.G = torch.randn(T * topk, N, device = DEV, generator = g).to(dtype)
        self.ref = _ref_dW(self.X, self.G, self.m_sizes, self.gather, topk, mode, E, N, K)
        # sum |dY| |X|: scales the fp32 accumulation error of the kernel's dot products.
        self.absref = _ref_dW(self.X.abs(), self.G.abs(), self.m_sizes, self.gather, topk, mode, E, N, K)

    def backward(self, W, cfgs, **bw):
        Y = _gemm(self.X, W, self.m_sizes, self.gather, self.topk, self.mode, cfgs)
        Y.backward(self.G, **bw)


def _param(E, N, K, dtype, seed = 1):
    torch.manual_seed(seed)
    return torch.nn.Parameter(torch.randn(E, N, K, device = DEV).to(dtype) * 0.05)


def _run_steps(cases, W, cfgs, inplace, g0, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1" if inplace else "0")
    W.grad = None if g0 is None else g0.clone()
    before = dict(gg.DW_ACCUMULATE_STATS)
    steps = []
    for c in cases:
        prev = None if W.grad is None else W.grad.clone()
        c.backward(W, cfgs)
        steps.append((prev, W.grad.clone(), c.ref, c.absref))
    used = gg.DW_ACCUMULATE_STATS["inplace"] - before["inplace"]
    _run_steps.steps = steps
    return W.grad.clone(), used


@pytest.mark.parametrize("mode", ["permute_x", "permute_y", "none"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("ga", [1, 4])
@pytest.mark.parametrize("grad_start", ["none", "zeros", "random"])
@pytest.mark.parametrize("tma_store", TMA_STORE)
def test_matches_returned_gradient_path_and_is_closer_to_fp64(
    mode, dtype, ga, grad_start, tma_store, monkeypatch
):
    E, N, K, T, topk = 8, 128, 192, 200, 2
    cfgs = _cfgs(tma_store = tma_store)
    cases = [_Case(E, N, K, T, topk, mode, dtype, seed = 10 + i) for i in range(ga)]
    W = _param(E, N, K, dtype)
    g0 = {
        "none": None,
        "zeros": torch.zeros_like(W),
        "random": (torch.randn_like(W.float()) * 3).to(dtype),
    }[grad_start]
    stock, used_stock = _run_steps(cases, W, cfgs, False, g0, monkeypatch)
    ours, used_ours = _run_steps(cases, W, cfgs, True, g0, monkeypatch)
    ours_steps = _run_steps.steps
    assert used_stock == 0
    # The first micro-step from grad None returns dW (AccumulateGrad steals it); every later one adds in place.
    assert used_ours == (ga - 1 if grad_start == "none" else ga)

    if ga == 1 and grad_start in ("none", "zeros"):
        # One kernel rounding either way: 0 + acc rounds exactly like acc.
        assert torch.equal(ours, stock)
        return
    ref = sum(c.ref for c in cases) + (0 if g0 is None else g0.double())
    err_o, err_s = (ours.double() - ref).abs(), (stock.double() - ref).abs()
    if dtype == torch.float32:
        # An fp32 dW adds the finished fp32 sum to the grad: the same single add autograd does.
        assert torch.equal(ours, stock)
        return
    # bf16 / fp16: stock rounds dW then the sum (two roundings), ours rounds the fp32 sum once.
    assert err_o.mean() <= err_s.mean(), (err_o.mean().item(), err_s.mean().item())
    closer = (err_o < err_s).sum().item()
    farther = (err_o > err_s).sum().item()
    assert closer >= farther, (closer, farther)
    # Every in-place micro-step is correctly rounded (up to fp32 accumulation noise): one rounding of
    # the stored grad plus the exact dW. The max error of both paths sits at that rounding floor.
    for prev, after, step_ref, absref in ours_steps:
        if prev is None:
            continue
        r = prev.double() + step_ref
        bound = _half_ulp(r, after, dtype) + _FP32_SLACK * (absref + prev.double().abs())
        assert ((after.double() - r).abs() <= bound).all()


# fp32 accumulation error of the dot products and the fp32 add, relative to sum |terms| (measured <= 2^-24).
_FP32_SLACK = 2.0**-20


def _half_ulp(ref, out, dtype):
    mant = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    mag = torch.maximum(ref.abs(), out.double().abs()).clamp_min(torch.finfo(dtype).tiny)
    _, exp = torch.frexp(mag)
    return torch.ldexp(torch.ones_like(mag), exp - 1 - mant) / 2


@pytest.mark.parametrize("mode", ["permute_x", "permute_y"])
def test_bf16_weight_with_fp32_activations(mode, monkeypatch):
    # Unsloth full finetuning runs the experts on fp32 hidden states with bf16 weights.
    E, N, K, T, topk = 8, 128, 192, 200, 2
    cfgs = _cfgs()
    cases = [_Case(E, N, K, T, topk, mode, torch.float32, seed = 20 + i) for i in range(3)]
    kw = dict(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 64,
              permute_x = mode == "permute_x", permute_y = mode == "permute_y")  # fmt: skip
    # The kernel's own fp32 accumulators: what today's X.dtype buffer holds before autograd's cast.
    accs = [gg.grouped_gemm_dW(c.X, c.G, c.m_sizes, c.gather, topk, **kw) for c in cases]
    assert accs[0].dtype == torch.float32
    W = _param(E, N, K, torch.bfloat16)
    out = {}
    for inplace in (False, True):
        _run_steps(cases, W, cfgs, inplace, None, monkeypatch)
        out[inplace] = _run_steps.steps
    # First micro-step: both round the same fp32 accumulator once (fresh bf16 buffer vs fp32 + cast).
    for inplace in (False, True):
        assert torch.equal(out[inplace][0][1], accs[0].to(torch.bfloat16)), inplace
    assert gg.DW_ACCUMULATE_STATS["inplace"] > 0
    errs = {True: [], False: []}
    for k in (1, 2):
        for inplace in (False, True):
            prev, after, _, absref = out[inplace][k]
            r = prev.double() + accs[k].double()
            err = (after.double() - r).abs()
            errs[inplace].append(err.mean().item())
            if inplace:
                bound = _half_ulp(r, after, torch.bfloat16) + _FP32_SLACK * (absref + prev.double().abs())
                assert (err <= bound).all()
    assert sum(errs[True]) <= sum(errs[False]), errs


@pytest.mark.parametrize("tma_store", TMA_STORE)
def test_empty_experts_keep_their_gradient(tma_store, monkeypatch):
    E, N, K, T, topk = 8, 128, 128, 64, 2
    cfgs = _cfgs(tma_store = tma_store)
    case = _Case(E, N, K, T, topk, "permute_x", torch.bfloat16, empty = (1, 5))
    assert case.m_sizes[1].item() == 0 and case.m_sizes[5].item() == 0
    W = _param(E, N, K, torch.bfloat16)
    g0 = torch.randn_like(W)
    g0[1, 0, 0] = -0.0  # stock 0 + -0 gives +0; untouched means the sign bit survives
    ours, used = _run_steps([case], W, cfgs, True, g0, monkeypatch)
    assert used == 1
    for e in (1, 5):
        assert torch.equal(ours[e], g0[e]) and torch.signbit(ours[1, 0, 0])
    # From grad None the fresh buffer holds zeros for them.
    fresh, _ = _run_steps([case], W, cfgs, True, None, monkeypatch)
    assert fresh[1].abs().sum() == 0 and fresh[5].abs().sum() == 0


@pytest.mark.parametrize("N, K, T", [(96, 80, 37), (40, 72, 1), (130, 48, 301)])
def test_odd_shapes_direct_op(N, K, T):
    # Partial N / K tiles and token counts that are not multiples of BLOCK_SIZE_M.
    E, topk = 5, 2
    for mode in ("permute_x", "permute_y", "none"):
        c = _Case(E, N, K, T, topk, mode, torch.bfloat16, seed = N + K + T)
        g0 = torch.randn(E, N, K, device = DEV).bfloat16()
        out = g0.clone()
        kw = dict(BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 32, BLOCK_SIZE_K = 32,
                  permute_x = mode == "permute_x", permute_y = mode == "permute_y")  # fmt: skip
        gg.grouped_gemm_dW_accumulate(c.X, c.G, c.m_sizes, c.gather, out, topk, **kw)
        fresh = gg.grouped_gemm_dW(c.X, c.G, c.m_sizes, c.gather, topk, **kw)
        ref = c.ref + g0.double()
        err_o = (out.double() - ref).abs()
        err_s = ((g0 + fresh).double() - ref).abs()
        assert err_o.mean() <= err_s.mean(), mode
        bound = _half_ulp(ref, out, torch.bfloat16) + _FP32_SLACK * (c.absref + g0.double().abs())
        assert (err_o <= bound).all(), mode


def _fallback_check(W, cfgs, case, fn):
    g0 = torch.randn_like(W)
    W.grad = g0.clone()
    before = gg.DW_ACCUMULATE_STATS["inplace"]
    out = fn()
    assert gg.DW_ACCUMULATE_STATS["inplace"] == before, "must not write param.grad in place"
    return g0, out


def test_autograd_grad_and_backward_inputs_do_not_touch_param_grad(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1")
    E, N, K, T, topk = 4, 64, 64, 32, 2
    cfgs = _cfgs()
    c = _Case(E, N, K, T, topk, "permute_x", torch.bfloat16)
    W = _param(E, N, K, torch.bfloat16)

    def grad_call():
        Y = _gemm(c.X, W, c.m_sizes, c.gather, topk, c.mode, cfgs)
        return torch.autograd.grad(Y, [W], c.G)[0]

    g0, dW = _fallback_check(W, cfgs, c, grad_call)
    assert torch.equal(W.grad, g0)
    torch.testing.assert_close(dW.double(), c.ref, rtol = 2e-2, atol = 2e-2 * c.ref.abs().max().item())

    X = c.X.clone().requires_grad_(True)

    def inputs_call():
        _gemm(X, W, c.m_sizes, c.gather, topk, c.mode, cfgs).backward(c.G, inputs = [X])

    g0, _ = _fallback_check(W, cfgs, c, inputs_call)
    assert torch.equal(W.grad, g0) and X.grad is not None

    # backward(inputs=[W]) does accumulate into W, so the in-place path applies.
    W.grad = g0.clone()
    before = gg.DW_ACCUMULATE_STATS["inplace"]
    _gemm(X, W, c.m_sizes, c.gather, topk, c.mode, cfgs).backward(c.G, inputs = [W])
    assert gg.DW_ACCUMULATE_STATS["inplace"] == before + 1
    torch.testing.assert_close((W.grad.double() - g0.double()), c.ref, rtol = 3e-2, atol = 3e-2 * c.ref.abs().max().item())


def test_hooks_create_graph_kill_switch_and_non_leaf_fall_back(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1")
    E, N, K, T, topk = 4, 64, 64, 32, 2
    cfgs = _cfgs()
    c = _Case(E, N, K, T, topk, "permute_x", torch.bfloat16)

    W = _param(E, N, K, torch.bfloat16)
    seen = []
    h = W.register_hook(lambda g: seen.append(g is not None))
    _fallback_check(W, cfgs, c, lambda: c.backward(W, cfgs))
    assert seen == [True]
    h.remove()

    W = _param(E, N, K, torch.bfloat16)
    W.register_post_accumulate_grad_hook(lambda p: seen.append("post"))
    _fallback_check(W, cfgs, c, lambda: c.backward(W, cfgs))
    assert seen[-1] == "post"

    W = _param(E, N, K, torch.bfloat16)
    _fallback_check(W, cfgs, c, lambda: c.backward(W, cfgs, create_graph = True))

    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "0")
    _fallback_check(W, cfgs, c, lambda: c.backward(W, cfgs))
    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1")

    # A non-leaf weight (scaled copy): its grad must flow on through autograd.
    W = _param(E, N, K, torch.bfloat16)
    g0, _ = _fallback_check(W, cfgs, c, lambda: c.backward(W * 1.0, cfgs))
    torch.testing.assert_close((W.grad.double() - g0.double()), c.ref, rtol = 3e-2, atol = 3e-2 * c.ref.abs().max().item())

    # Frozen experts (LoRA): no dW at all.
    W = _param(E, N, K, torch.bfloat16).requires_grad_(False)
    X = c.X.clone().requires_grad_(True)
    before = dict(gg.DW_ACCUMULATE_STATS)
    _gemm(X, W, c.m_sizes, c.gather, topk, c.mode, cfgs).backward(c.G)
    assert gg.DW_ACCUMULATE_STATS == before and W.grad is None


def test_retain_graph_tied_weight_and_inf_propagation(monkeypatch):
    E, N, K, T, topk = 4, 64, 128, 48, 2
    cfgs = _cfgs()
    c = _Case(E, N, K, T, topk, "permute_x", torch.bfloat16)
    c2 = _Case(E, N, K, T, topk, "permute_x", torch.bfloat16, seed = 7)
    g0 = torch.randn(E, N, K, device = DEV).bfloat16()
    results = {}
    for inplace in (False, True):
        monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1" if inplace else "0")
        W = _param(E, N, K, torch.bfloat16)
        W.grad = g0.clone()
        Y = _gemm(c.X, W, c.m_sizes, c.gather, topk, c.mode, cfgs)
        Y.backward(c.G, retain_graph = True)
        Y.backward(c.G)
        # Tied: the same weight in two GEMMs of one graph.
        Y1 = _gemm(c.X, W, c.m_sizes, c.gather, topk, c.mode, cfgs)
        Y2 = _gemm(c2.X, W, c2.m_sizes, c2.gather, topk, c2.mode, cfgs)
        ((Y1 * c.G).sum() + (Y2 * c2.G).sum()).backward()
        results[inplace] = W.grad.clone()
    ref = g0.double() + 3 * c.ref + c2.ref
    err_o = (results[True].double() - ref).abs()
    err_s = (results[False].double() - ref).abs()
    assert err_o.mean() <= err_s.mean()

    # A GradScaler overflow in dY must reach the accumulated gradient so found_inf sees it.
    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1")
    W = _param(E, N, K, torch.bfloat16)
    W.grad = g0.clone()
    G = c.G.clone()
    G[3, 5] = float("inf")
    _gemm(c.X, W, c.m_sizes, c.gather, topk, c.mode, cfgs).backward(G)
    assert not torch.isfinite(W.grad).all()


def test_compiled_fullgraph_zero_breaks_and_matches_eager(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1")
    from torch._dynamo.utils import counters

    E, N, K, T, topk = 4, 128, 128, 64, 2
    cfgs = _cfgs()
    c = _Case(E, N, K, T, topk, "permute_x", torch.bfloat16)

    def fn(X, W):
        return _gemm(X, W, c.m_sizes, c.gather, topk, c.mode, cfgs)

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, fullgraph = True)
    grads = []
    for f in (fn, compiled):
        W = _param(E, N, K, torch.bfloat16)
        g0 = torch.randn_like(W)
        W.grad = g0.clone()
        X = c.X.clone().requires_grad_(True)
        for _ in range(2):
            f(X, W).backward(c.G)
        grads.append(W.grad.double() - g0.double())
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    for g in grads:
        torch.testing.assert_close(g, 2 * c.ref, rtol = 3e-2, atol = 3e-2 * c.ref.abs().max().item())

    # The mutating op itself is opaque to the compiler and mutates its dW argument.
    def acc(X, dY, buf):
        gg.grouped_gemm_dW_accumulate(
            X, dY, c.m_sizes, c.gather, buf, topk, BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64,
            BLOCK_SIZE_K = 64, permute_x = True,
        )  # fmt: skip
        return buf

    counters.clear()
    buf_e = torch.randn(E, N, K, device = DEV).bfloat16()
    buf_c = buf_e.clone()
    acc(c.X, c.G, buf_e)
    torch.compile(acc, fullgraph = True)(c.X, c.G, buf_c)
    assert sum(counters["graph_break"].values()) == 0
    assert torch.equal(buf_e, buf_c)
    schema = str(torch.ops.unsloth.grouped_gemm_dW_accumulate.default._schema)
    assert re.search(r"Tensor\(a\d*!\) dW", schema), schema


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(rank, world, port, inplace, bucket_view, out_path):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    os.environ["UNSLOTH_MOE_DW_ACCUMULATE"] = "1" if inplace else "0"
    dist.init_process_group("gloo", rank = rank, world_size = world)
    E, N, K, T, topk = 4, 128, 128, 64, 2
    cfgs = _cfgs()

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.w = torch.nn.Parameter(torch.randn(E, N, K, device = DEV).bfloat16() * 0.05)
            self.head = torch.nn.Linear(N, 1, device = DEV, dtype = torch.bfloat16)

        def forward(self, x, m_sizes, gather):
            return self.head(_gemm(x, self.w, m_sizes, gather, topk, "permute_x", cfgs))

    model = Block()
    ddp = torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view = bucket_view)
    cases = [_Case(E, N, K, T, topk, "permute_x", torch.bfloat16, seed = 100 * rank + i) for i in range(4)]
    for i, case in enumerate(cases):
        loss = ddp(case.X, case.m_sizes, case.gather).float().pow(2).mean()
        if i < 3:
            with ddp.no_sync():
                loss.backward()
        else:
            loss.backward()
    torch.save(
        {"grad": model.w.grad.float().cpu(), "inplace": gg.DW_ACCUMULATE_STATS["inplace"]},
        f"{out_path}.{rank}",
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("bucket_view", [False, True])
def test_ddp_gradient_accumulation_with_bucket_hooks(bucket_view, tmp_path):
    import torch.multiprocessing as mp

    got = {}
    for inplace in (False, True):
        out = str(tmp_path / f"g_{inplace}")
        mp.spawn(_ddp_worker, args = (2, _free_port(), inplace, bucket_view, out), nprocs = 2)
        got[inplace] = [torch.load(f"{out}.{r}") for r in range(2)]
    for r in range(2):
        assert got[False][r]["inplace"] == 0
        assert got[True][r]["inplace"] == 3  # micro-steps 2..4
    # The reducer fired: both ranks hold the same averaged gradient.
    assert torch.equal(got[True][0]["grad"], got[True][1]["grad"])
    a, b = got[True][0]["grad"], got[False][0]["grad"]
    rel = ((a - b).norm() / b.norm()).item()
    assert rel < 1e-2, rel  # a dropped micro-step or a skipped all-reduce is a > 10% error


def _fsdp2_worker(rank, inplace, out_path):
    import torch.distributed as dist

    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import fully_shard

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(_free_port()))
    os.environ["UNSLOTH_MOE_DW_ACCUMULATE"] = "1" if inplace else "0"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank = 0, world_size = 1)
    E, N, K, T, topk = 4, 128, 128, 64, 2
    cfgs = _cfgs()

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.w = torch.nn.Parameter(torch.randn(E, N, K, device = DEV).bfloat16() * 0.05)

        def forward(self, x, m_sizes, gather):
            return _gemm(x, self.w, m_sizes, gather, topk, "permute_x", cfgs)

    model = Block()
    fully_shard(model)
    cases = [_Case(E, N, K, T, topk, "permute_x", torch.bfloat16, seed = i) for i in range(4)]
    for i, case in enumerate(cases):
        model.set_requires_gradient_sync(i == 3)
        model(case.X, case.m_sizes, case.gather).float().pow(2).mean().backward()
    grad = model.w.grad.full_tensor() if hasattr(model.w.grad, "full_tensor") else model.w.grad
    torch.save(
        {"grad": grad.float().cpu(), "inplace": gg.DW_ACCUMULATE_STATS["inplace"]}, out_path
    )
    dist.destroy_process_group()


def test_fsdp2_no_sync_gradient_accumulation(tmp_path):
    import torch.multiprocessing as mp

    if not torch.distributed.is_nccl_available():
        pytest.skip("FSDP2 smoke needs NCCL")
    got = {}
    for inplace in (False, True):
        out = str(tmp_path / f"f_{inplace}.pt")
        mp.spawn(_fsdp2_worker, args = (inplace, out), nprocs = 1)
        got[inplace] = torch.load(out)
    assert got[False]["inplace"] == 0
    a, b = got[True]["grad"], got[False]["grad"]
    rel = ((a - b).norm() / b.norm()).item()
    assert rel < 1e-2, rel
    # Whether FSDP2 keeps the unsharded grad between no-sync micro-steps decides engagement; record it.
    print(f"FSDP2 in-place launches: {got[True]['inplace']}, rel vs stock {rel:.2e}")
