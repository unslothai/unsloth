# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 stochastic rounding on CUDA: the one-pass AdamW kernel and the one-pass round (bitsandbytes)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA"),
]
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


@pytest.fixture
def sr(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    monkeypatch.delenv("UNSLOTH_BF16_SR_FUSED", raising = False)
    path = Path(__file__).resolve().parents[1] / "unsloth/optimizers/bf16_stochastic_rounding.py"
    spec = importlib.util.spec_from_file_location("_test_bf16_sr_fused", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@triton.jit
def _lanes(OUT, N, SEED, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    r0, r1, r2, r3 = tl.randint4x(SEED, i)
    tl.store(OUT + i * 4 + 0, r0, valid)
    tl.store(OUT + i * 4 + 1, r1, valid)
    tl.store(OUT + i * 4 + 2, r2, valid)
    tl.store(OUT + i * 4 + 3, r3, valid)


def _philox(n, seed):
    out = torch.empty(n * 4, dtype = torch.int32, device = "cuda")
    _lanes[((n + 1023) // 1024,)](out, n, seed, BLOCK = 1024)
    return out.view(n, 4).to(torch.int64) & 65535


def _round_with(x, random):
    """x float32 -> bfloat16 rounded up iff random < low 16 bits."""
    bits = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    low, upper = bits & 65535, bits & 0xFFFF0000
    special = (bits & 0x7F800000) == 0x7F800000
    out = torch.where(special, bits, torch.where(random < low, upper + 65536, upper))
    out = torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)
    return out.view(torch.float32).to(torch.bfloat16)


def test_round_kernel_exact_specials_unbiased(sr):
    exact = torch.tensor([1.0, -2.5, 0.0, -0.0, 3.0e38], dtype = torch.bfloat16, device = "cuda")
    out = torch.empty_like(exact)
    sr._round_cuda_(out, exact.float(), 123)
    assert torch.equal(out.view(torch.int16), exact.view(torch.int16))
    special = torch.tensor([float("inf"), float("-inf"), float("nan")], device = "cuda")
    out = torch.empty(3, dtype = torch.bfloat16, device = "cuda")
    sr._round_cuda_(out, special, 1)
    assert out[0] == float("inf") and out[1] == float("-inf") and out[2].isnan()
    for sign in (1, -1):
        x = torch.full((1 << 20,), sign * (1 + 2**-9), device = "cuda")
        out = torch.empty(x.shape, dtype = torch.bfloat16, device = "cuda")
        sr._round_cuda_(out, x, 7)
        assert set(out.float().abs().unique().tolist()) == {1.0, 1 + 2**-7}
        assert abs(out.float().mean().item() - sign * (1 + 2**-9)) < 1e-5
        assert torch.equal(out, _round_with(x, _philox(x.numel(), 7)[:, 0]))


def _op_by_op(
    p,
    g,
    m,
    v,
    step,
    lr,
    wd,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
):
    """torch.optim.AdamW's foreach float32 formula, one rounded op at a time (no contraction)."""
    p, g, m, v = p.float(), g.float(), m.float(), v.float()
    p = p * (1 - lr * wd)
    m = m + (g - m) * (1 - beta1)
    v = v * beta2
    v = v + (g * (1 - beta2)) * g
    denom = torch.sqrt(v) / ((1 - beta2**step) ** 0.5) + eps
    p = p + (m / denom) * ((lr / (1 - beta1**step)) * -1)
    return p, m, v


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_one_pass_step_is_adamw_float32_then_rounded(sr, fused, weight_decay):
    """(Near) bitwise: AdamW's float32 arithmetic op by op, rounded with the same Philox draws.
    Against torch.optim.AdamW itself (whose kernels may contract into FMAs): all but a few elements bitwise."""
    torch.manual_seed(0)
    n = 300_001
    p = torch.nn.Parameter((torch.randn(n, device = "cuda") * 0.05).to(torch.bfloat16))
    opt = torch.optim.AdamW([p], lr = 3e-4, weight_decay = weight_decay, fused = fused)
    assert sr.enable_bf16_stochastic_rounding(opt, seed = 11, verbose = False) == 1
    for step in range(1, 4):
        p.grad = (torch.randn(n, device = "cuda") * 1e-3).to(torch.bfloat16)
        if step > 1:
            m0, v0 = (opt.state[p][k].clone() for k in ("exp_avg", "exp_avg_sq"))
        else:
            m0, v0 = torch.zeros_like(p), torch.zeros_like(p)
        w0, g0 = p.detach().clone(), p.grad.clone()
        opt.step()
        assert opt.__dict__["_unsloth_sr_fused_steps"] == step
        got = [p.detach(), opt.state[p]["exp_avg"], opt.state[p]["exp_avg_sq"]]
        lanes = _philox(n, sr._seed_for(opt, p, 11, step))
        exact = _op_by_op(w0, g0, m0, v0, step, 3e-4, weight_decay)
        for a, x, lane in zip(got, exact, range(3)):
            assert a.dtype == torch.bfloat16
            diff = (
                a.view(torch.int16).int() - _round_with(x, lanes[:, lane]).view(torch.int16).int()
            ).abs()
            # Bitwise but for a rare float32 ulp that lands on a draw's threshold (1 in 300k seen on B200).
            assert diff.max().item() <= 1 and diff.sum().item() <= 3, diff.sum().item()
        q = torch.nn.Parameter(w0.float())
        ref = torch.optim.AdamW([q], lr = 3e-4, weight_decay = weight_decay, foreach = True)
        q.grad = g0.float()
        if step > 1:
            ref.state[q] = {
                "step": torch.tensor(float(step - 1)),
                "exp_avg": m0.float(),
                "exp_avg_sq": v0.float(),
            }
        ref.step()
        torch_ref = [q.detach(), ref.state[q]["exp_avg"], ref.state[q]["exp_avg_sq"]]
        for a, x, lane in zip(got, torch_ref, range(3)):
            same = a.view(torch.int16) == _round_with(x, lanes[:, lane]).view(torch.int16)
            assert (~same).sum().item() <= max(8, n >> 12)
        assert opt.state[p]["step"].device.type == ("cuda" if fused else "cpu")
        assert float(opt.state[p]["step"]) == step


def _train(
    sr,
    mode,
    steps = 30,
    optimizer = "foreach",
    env = None,
    monkeypatch = None,
):
    g = torch.Generator().manual_seed(0)
    weights = [torch.randn(256, 192, generator = g) * 0.05, torch.randn(3000, generator = g) * 0.05]
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    params = [torch.nn.Parameter(w.to(torch.bfloat16).to(dtype).cuda()) for w in weights]
    start = [p.detach().float().clone() for p in params]
    kw = {"fused": True} if optimizer == "fused" else {"foreach": True}
    opt = torch.optim.AdamW(params, lr = 1e-5, weight_decay = 0.0, **kw)
    if mode == "sr":
        assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    for step in range(steps):
        gg = torch.Generator().manual_seed(100 + step)
        for p, w in zip(params, weights):
            p.grad = (torch.randn(w.shape, generator = gg) * 1e-3).to(dtype).cuda()
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none = True)
    delta = torch.cat([(p.detach().float() - s).flatten() for p, s in zip(params, start)]).double()
    return delta, opt


@pytest.mark.parametrize("optimizer", ["foreach", "fused"])
def test_one_pass_keeps_the_float32_update_and_kill_switch(sr, monkeypatch, optimizer):
    ref, _ = _train(sr, "fp32", optimizer = optimizer)
    bf16, _ = _train(sr, "bf16", optimizer = optimizer)
    one_pass, opt = _train(sr, "sr", optimizer = optimizer)
    assert opt.__dict__.get("_unsloth_sr_fused_steps") == 60
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "0")
    two_pass, opt = _train(sr, "sr", optimizer = optimizer)
    assert opt.__dict__.get("_unsloth_sr_fused_steps", 0) == 0
    monkeypatch.delenv("UNSLOTH_BF16_SR_FUSED")
    kept = {
        k: (v @ ref / (ref @ ref)).item()
        for k, v in [("bf16", bf16), ("one", one_pass), ("two", two_pass)]
    }
    assert kept["bf16"] < 0.5, kept
    assert abs(kept["one"] - 1) < 0.05 and abs(kept["two"] - 1) < 0.05, kept
    monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "0")
    p = torch.nn.Parameter(torch.zeros(4, dtype = torch.bfloat16, device = "cuda"))
    assert sr.enable_bf16_stochastic_rounding(torch.optim.AdamW([p]), verbose = False) == 0


@pytest.mark.parametrize("optimizer", ["foreach", "fused"])
def test_one_pass_resume_and_replicas_are_bitwise(sr, optimizer):
    def run(
        steps,
        state = None,
        seed = 3407,
    ):
        g = torch.Generator().manual_seed(0)
        params = [
            torch.nn.Parameter((torch.randn(64, 48, generator = g) * 0.05).to(torch.bfloat16).cuda()),
            torch.nn.Parameter((torch.randn(300, generator = g) * 0.05).to(torch.bfloat16).cuda()),
        ]
        if state is not None:
            for p, w in zip(params, state["weights"]):
                p.data.copy_(w)
        kw = {"fused": True} if optimizer == "fused" else {"foreach": True}
        opt = torch.optim.AdamW(params, lr = 1e-5, weight_decay = 0.01, **kw)
        if state is not None:
            opt.load_state_dict(state["optimizer"])
        sr.enable_bf16_stochastic_rounding(opt, seed = seed, verbose = False)
        start = 0 if state is None else state["done"]
        for step in range(start, start + steps):
            gg = torch.Generator().manual_seed(100 + step)
            for p in params:
                p.grad = (torch.randn(p.shape, generator = gg) * 1e-3).to(torch.bfloat16).cuda()
            opt.step()
            opt.zero_grad(set_to_none = True)
        assert opt.__dict__.get("_unsloth_sr_fused_steps") == 2 * steps
        return params, opt

    whole, opt = run(4)
    replica, _ = run(4)  # a second DDP rank: same seed, same index, same step
    other, _ = run(4, seed = 1)
    half, half_opt = run(2)
    state = {
        "weights": [p.detach().clone() for p in half],
        "optimizer": half_opt.state_dict(),
        "done": 2,
    }
    resumed, resumed_opt = run(2, state)
    for a, b, c, d in zip(whole, resumed, replica, other):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        assert torch.equal(a.view(torch.int16), c.view(torch.int16))
        assert not torch.equal(a.view(torch.int16), d.view(torch.int16))
    for p, q in zip(whole, resumed):
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                opt.state[p][k].view(torch.int16), resumed_opt.state[q][k].view(torch.int16)
            )
            assert opt.state[p][k].dtype == torch.bfloat16


def test_float32_moments_take_the_float32_route_once(sr):
    p = torch.nn.Parameter(torch.ones(1000, dtype = torch.bfloat16, device = "cuda"))
    opt = torch.optim.AdamW([p], lr = 1e-3)
    sr.enable_bf16_stochastic_rounding(opt, verbose = False)
    opt.state[p] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.zeros(1000, device = "cuda"),
        "exp_avg_sq": torch.zeros(1000, device = "cuda"),
    }
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.__dict__.get("_unsloth_sr_fused_steps", 0) == 0
    assert opt.state[p]["exp_avg"].dtype == torch.bfloat16
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.__dict__["_unsloth_sr_fused_steps"] == 1 and float(opt.state[p]["step"]) == 5


@pytest.mark.parametrize("kw", [{"amsgrad": True}, {"maximize": True}])
def test_unsupported_groups_keep_the_float32_route(sr, kw):
    p = torch.nn.Parameter(torch.ones(1000, dtype = torch.bfloat16, device = "cuda"))
    opt = torch.optim.AdamW([p], lr = 1e-3, **kw)
    sr.enable_bf16_stochastic_rounding(opt, verbose = False)
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.__dict__.get("_unsloth_sr_fused_steps", 0) == 0
    assert p.dtype == torch.bfloat16 and p.float().mean().item() != 1.0


def test_adam_l2_decay_keeps_the_float32_route(sr):
    p = torch.nn.Parameter(torch.ones(1000, dtype = torch.bfloat16, device = "cuda"))
    q = torch.nn.Parameter(torch.ones(1000, dtype = torch.bfloat16, device = "cuda"))
    opt = torch.optim.Adam([{"params": [p], "weight_decay": 0.1}, {"params": [q]}], lr = 1e-3)
    sr.enable_bf16_stochastic_rounding(opt, verbose = False)
    p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
    opt.step()
    assert opt.__dict__.get("_unsloth_sr_fused_steps", 0) == 1  # q only


def test_bitsandbytes_rounds_in_one_pass(sr, monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")

    def train(mode):
        g = torch.Generator().manual_seed(0)
        weights = [torch.randn(256, 192, generator = g) * 0.05, torch.randn(3000, generator = g) * 0.05]
        dtype = torch.float32 if mode == "fp32" else torch.bfloat16
        params = [torch.nn.Parameter(w.to(torch.bfloat16).to(dtype).cuda()) for w in weights]
        start = [p.detach().float().clone() for p in params]
        opt = bnb.optim.AdamW8bit(params, lr = 1e-5, weight_decay = 0.0, min_8bit_size = 256)
        if mode != "fp32" and mode != "bf16":
            assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 2
        for step in range(30):
            gg = torch.Generator().manual_seed(100 + step)
            for p, w in zip(params, weights):
                p.grad = (torch.randn(w.shape, generator = gg) * 1e-3).to(dtype).cuda()
            opt.step()
            opt.zero_grad(set_to_none = True)
        delta = torch.cat(
            [(p.detach().float() - s).flatten() for p, s in zip(params, start)]
        ).double()
        return delta, opt, params

    ref, _, _ = train("fp32")
    one, opt, params = train("sr")
    assert opt.__dict__.get("_unsloth_sr_triton_rounds") == 60
    again, _, params2 = train("sr")
    assert all(torch.equal(a, b) for a, b in zip(params, params2))  # seeded per (index, step)
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "0")
    two, opt, _ = train("sr")
    assert opt.__dict__.get("_unsloth_sr_triton_rounds", 0) == 0
    kept = {k: (v @ ref / (ref @ ref)).item() for k, v in [("one", one), ("two", two)]}
    assert abs(kept["one"] - 1) < 0.05 and abs(kept["two"] - 1) < 0.05, kept
