# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The eager DiT LoRA fast path (one input cast, per-step bf16 masters, fused add) and the nf4 dequant cache.

CPU only: bf16 autocast on CPU drives the same code as CUDA autocast. Parity is judged against an fp32
reference: the fast path may not be further from it than PEFT's own bf16 path (both round in bf16)."""

import copy
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

from core.training import diffusion_lora_fast as fl  # noqa: E402

BF16 = torch.bfloat16


class _Block(torch.nn.Module):
    """fp32 LayerNorm output into three LoRA linears, like a DiT attention block under autocast."""

    def __init__(
        self,
        d = 64,
        h = 96,
        bias = True,
    ):
        super().__init__()
        self.q = torch.nn.Linear(d, h, bias = bias)
        self.k = torch.nn.Linear(d, h, bias = False)
        self.o = torch.nn.Linear(h, d, bias = bias)

    def forward(self, x):
        n = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        return x + self.o(torch.nn.functional.gelu(self.q(n)) * self.k(n))


def _lora(
    model,
    r = 8,
    alpha = 16,
    dropout = 0.0,
):
    model = inject_adapter_in_model(
        LoraConfig(
            r = r,
            lora_alpha = alpha,
            lora_dropout = dropout,
            target_modules = ["q", "k", "o"],
            init_lora_weights = "gaussian",
        ),
        model,
    )
    gen = torch.Generator().manual_seed(1)
    for name, p in model.named_parameters():
        is_lora = "lora_" in name
        p.requires_grad_(is_lora)
        if is_lora:
            p.data = p.data.float()
            if "lora_B" in name:
                # PEFT zero-inits B; a nonzero B exercises every gradient path.
                p.data = torch.randn(p.shape, generator = gen) * 0.05
    return model


def _dense_model():
    torch.manual_seed(0)
    return _lora(_Block().to(BF16))


def _run(
    model,
    x,
    autocast = True,
    ckpt = False,
):
    xx = x.clone().requires_grad_(True)
    ctx = torch.autocast("cpu", dtype = BF16) if autocast else torch.autocast("cpu", enabled = False)
    with ctx:
        if ckpt:
            y = torch.utils.checkpoint.checkpoint(model, xx, use_reentrant = False)
        else:
            y = model(xx)
    y.float().pow(2).mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return y.detach().float(), xx.grad.detach().clone(), grads


def _fp32_reference(model, x):
    ref = copy.deepcopy(model).float()
    for p in ref.parameters():
        p.grad = None
    return _run(ref, x.float(), autocast = False)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _assert_not_worse(
    fast,
    peft_out,
    ref,
    slack = 2.0,
    floor = 2e-3,
):
    yf, gxf, gf = fast
    yp, gxp, gp = peft_out
    yr, gxr, gr = ref
    assert set(gf) == set(gp) == set(gr) and gf
    assert _rel(yf, yr) <= max(slack * _rel(yp, yr), floor)
    assert _rel(gxf, gxr) <= max(slack * _rel(gxp, gxr), floor)
    for n in gr:
        assert gf[n].dtype == torch.float32
        assert _rel(gf[n], gr[n]) <= max(slack * _rel(gp[n], gr[n]), floor), n


def test_split_cast_sums_branch_grads_in_the_input_dtype():
    x = torch.randn(4, 8, requires_grad = True)
    a, b = fl._SplitCast.apply(x, BF16)
    assert a.dtype == BF16 and a.data_ptr() == b.data_ptr()  # one cast, two aliases
    w1 = torch.randn(8, 8, dtype = BF16)
    w2 = torch.randn(8, 8, dtype = BF16)
    ((a @ w1).float().sum() + (b @ w2).float().sum()).backward()
    ref = x.detach().clone().requires_grad_(True)
    ((ref.to(BF16) @ w1).float().sum() + (ref.to(BF16) @ w2).float().sum()).backward()
    assert x.grad.dtype == torch.float32 and torch.equal(x.grad, ref.grad)


def test_install_patches_every_plain_lora_linear_and_uninstall_restores():
    model = _dense_model()
    assert fl.install_fast_lora(model) == 3
    assert fl.install_fast_lora(model) == 0  # idempotent
    assert all("_unsloth_orig_forward" in m.__dict__ for m in (model.q, model.k, model.o))
    assert fl.uninstall_fast_lora(model) == 3
    assert not any("forward" in m.__dict__ for m in model.modules())


def test_fast_path_matches_peft_within_bf16_against_fp32_reference():
    x = torch.randn(4, 32, 64)
    model = _dense_model()
    fast_model = copy.deepcopy(model)
    fl.install_fast_lora(fast_model)
    before = dict(fl.STATS)
    fast = _run(fast_model, x)
    assert fl.STATS["calls"] - before["calls"] == 3
    _assert_not_worse(fast, _run(model, x), _fp32_reference(model, x))
    # Output dtype contract: a dense base under autocast returns the autocast dtype, as PEFT does.
    with torch.autocast("cpu", dtype = BF16):
        assert fast_model.q(x).dtype == model.q(x).dtype


class _Bf16Block(torch.nn.Module):
    """A bf16 residual stream (Wan's blocks keep their LoRA inputs in bf16)."""

    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(64, 96)
        self.o = torch.nn.Linear(96, 64)

    def forward(self, x):
        return x + self.o(torch.nn.functional.gelu(self.q(x)))


@pytest.mark.parametrize("alpha", [8, 16])
def test_bit_identical_to_peft_on_bf16_inputs(alpha):
    """bf16 inputs (Wan's first block): forward, input grads and LoRA grads equal PEFT's bit for bit,
    with a LoRA scale of 1 and of 2."""
    torch.manual_seed(0)
    model = _lora(_Bf16Block().to(BF16), r = 8, alpha = alpha)
    fast_model = copy.deepcopy(model)
    assert fl.install_fast_lora(fast_model) == 2
    x = torch.randn(4, 32, 64, dtype = BF16)
    before = fl.STATS["calls"]
    want = _run(model, x)
    got = _run(fast_model, x)
    assert fl.STATS["calls"] - before == 2
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    for n in want[2]:
        assert torch.equal(got[2][n], want[2][n]), n


@pytest.mark.parametrize("seed", [0, 1049])
def test_dense_base_with_fp32_inputs_is_bit_identical_to_peft(seed):
    """Forward and LoRA grads match PEFT bit for bit. The input grad sums the q and k branches
    into the layer norm, so autograd's accumulation order can move a rare element by one fp32
    ulp (seed 1049: 3.6e-12); it is held to that, far below any real gradient error."""
    x = torch.randn(4, 32, 64, generator = torch.Generator().manual_seed(seed))
    model = _dense_model()
    fast_model = copy.deepcopy(model)
    fl.install_fast_lora(fast_model)
    want = _run(model, x)
    got = _run(fast_model, x)
    assert torch.equal(got[0], want[0])
    torch.testing.assert_close(got[1], want[1], rtol = 1e-6, atol = 1e-9)
    for n in want[2]:
        assert torch.equal(got[2][n], want[2][n]), n


def test_without_autocast_the_peft_path_runs_unchanged():
    x = torch.randn(2, 8, 64, dtype = BF16)
    model = _dense_model()
    fast_model = copy.deepcopy(model)
    fl.install_fast_lora(fast_model)
    before = dict(fl.STATS)
    a = _run(model, x, autocast = False)
    b = _run(fast_model, x, autocast = False)
    assert fl.STATS["calls"] == before["calls"]
    assert fl.STATS["fallback"] > before["fallback"]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for n in a[2]:
        assert torch.equal(a[2][n], b[2][n])


def test_masters_cast_once_per_step_and_recast_after_an_update():
    x = torch.randn(2, 16, 64)
    model = _dense_model()
    fl.install_fast_lora(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2)
    c0 = fl.STATS["casts"]
    _run(model, x)
    _run(model, x)  # a second micro-batch: same masters, no recast
    _run(model, x, ckpt = True)  # checkpoint recompute reuses them too
    assert fl.STATS["casts"] - c0 == 3
    opt.step()
    _run(model, x)
    assert fl.STATS["casts"] - c0 == 6
    # The bf16 copy tracks the updated master, with the LoRA scale folded into B.
    a_lp = model.q.__dict__["_unsloth_lp"][1]
    b_lp = model.q.__dict__["_unsloth_lp"][2]
    assert torch.equal(a_lp, model.q.lora_A["default"].weight.detach().to(BF16))
    assert torch.equal(b_lp, model.q.lora_B["default"].weight.detach().to(BF16))


def test_fused_adamw_steps_reach_the_forward():
    """Studio trains with ``AdamW(fused=True)``; each step must change what the next forward computes."""
    x = torch.randn(2, 16, 64)
    model = _dense_model()
    peft_model = copy.deepcopy(model)
    fl.install_fast_lora(model)

    def train(m, fast):
        params = [p for p in m.parameters() if p.requires_grad]
        try:
            opt = torch.optim.AdamW(params, lr = 1e-2, fused = True)
        except (RuntimeError, TypeError):  # no CPU fused kernel on this torch
            pytest.skip("fused AdamW unavailable on CPU here")
        losses = []
        for _ in range(4):
            opt.zero_grad(set_to_none = True)
            with torch.autocast("cpu", dtype = BF16):
                loss = m(x).float().pow(2).mean()
            loss.backward()
            opt.step()
            if fast:
                fl.mark_lora_updated()
            losses.append(loss.item())
        return losses

    want = train(peft_model, False)
    got = train(model, True)
    assert got[-1] != got[0]
    for a, b in zip(got, want):
        assert abs(a - b) <= 2e-2 * abs(b)
    _run(model, x)
    a_lp = model.q.__dict__["_unsloth_lp"][1]
    assert torch.equal(a_lp, model.q.lora_A["default"].weight.detach().to(BF16))


def test_writes_that_skip_the_version_counter_are_seen_after_mark():
    x = torch.randn(2, 8, 64)
    model = _dense_model()
    fl.install_fast_lora(model)
    _run(model, x)
    w = model.q.lora_A["default"].weight
    w.data.add_(1.0)  # like a fused optimizer kernel: the version counter does not move
    fl.mark_lora_updated()
    _run(model, x)
    assert torch.equal(model.q.__dict__["_unsloth_lp"][1], w.detach().to(BF16))


def test_grad_accumulation_and_checkpoint_recompute_match_peft():
    xs = [torch.randn(2, 16, 64) for _ in range(2)]
    model = _dense_model()
    fast_model = copy.deepcopy(model)
    fl.install_fast_lora(fast_model)
    for x in xs:
        peft_out = _run(model, x, ckpt = True)
        fast = _run(fast_model, x, ckpt = True)
    ref_model = copy.deepcopy(model).float()
    for p in ref_model.parameters():
        p.grad = None
    for x in xs:
        ref = _run(ref_model, x.float(), autocast = False)
    _assert_not_worse(fast, peft_out, ref)


def test_dropout_runs_on_the_fast_path_and_changes_the_lora_branch():
    x = torch.randn(2, 16, 64)
    torch.manual_seed(0)
    model = _lora(_Block().to(BF16), dropout = 0.5)
    fl.install_fast_lora(model)
    before = fl.STATS["calls"]
    model.train()
    y1 = _run(model, x)[0]
    model.eval()
    y2 = _run(model, x)[0]
    assert fl.STATS["calls"] - before == 6
    assert not torch.equal(y1, y2)


def test_unsupported_adapters_fall_back_to_peft():
    x = torch.randn(2, 8, 64)
    model = _dense_model()
    fl.install_fast_lora(model)
    before = dict(fl.STATS)
    model.q.enable_adapters(False)  # PEFT's disabled-adapter path must stay PEFT's
    with torch.autocast("cpu", dtype = BF16):
        model(x)
    assert fl.STATS["fallback"] - before["fallback"] == 1
    assert fl.STATS["calls"] - before["calls"] == 2


def test_budget_math():
    gib = 1 << 30
    # 24 GiB card, 20 free, 3 reserved by us, step peak 5: room 18 minus margin 2.4, times 0.85.
    got = fl.dequant_budget_bytes(
        free = 20 * gib, reserved = 3 * gib, peak_alloc = 5 * gib, total = 24 * gib
    )
    assert got == int((18 * gib - int(24 * gib * 0.10)) * 0.85)
    # Small card: the 2 GiB floor on the margin applies.
    got = fl.dequant_budget_bytes(free = 3 * gib, reserved = 1 * gib, peak_alloc = 1 * gib, total = 8 * gib)
    assert got == int((3 * gib - 2 * gib) * 0.85)
    # No room: zero, never negative.
    assert fl.dequant_budget_bytes(free = gib, reserved = gib, peak_alloc = 2 * gib, total = 16 * gib) == 0


def test_kill_switches(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FAST_LORA", "0")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE", "off")
    assert not fl.fast_lora_enabled() and not fl.dequant_cache_enabled()
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FAST_LORA")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE")
    assert fl.fast_lora_enabled() and fl.dequant_cache_enabled()


def test_plan_dequant_cache_override(monkeypatch):
    cache = fl.Nf4DequantCache()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", "1.5")
    fl.plan_dequant_cache(cache, 0, 0)
    assert cache.budget == int(1.5 * (1 << 30))
    monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", "0")
    fl.plan_dequant_cache(cache, 0, 0)
    assert cache.budget == 0


def _fake_cuda(monkeypatch, free, total, reserved, allocated):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", raising = False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: allocated)


def test_plan_dequant_cache_from_step_one(monkeypatch):
    gib = 1 << 30
    # After the optimizer step: 4 GiB allocated (3.5 before it, so 0.5 GiB of new optimizer state).
    _fake_cuda(monkeypatch, 16 * gib, 24 * gib, 6 * gib, 4 * gib)
    cache = fl.Nf4DequantCache()
    pre = int(3.5 * gib)
    fl.plan_dequant_cache(cache, 5 * gib, pre)
    need = 5 * gib + gib // 2
    assert cache.budget == fl.dequant_budget_bytes(16 * gib, 6 * gib, need, 24 * gib)
    # Bucketed: the 1.5 GiB of activations scale to the largest canvas (2x the area here).
    fl.plan_dequant_cache(cache, 5 * gib, pre, act_scale = 2.0)
    assert cache.budget == fl.dequant_budget_bytes(
        16 * gib, 6 * gib, need + int(1.5 * gib), 24 * gib
    )
    assert fl.bucket_activation_scale(None) == 1.0
    assert fl.bucket_activation_scale([(512, 512), (512, 1024)]) == 2.0


def test_optimizer_hooks_recast_every_step_and_plan_once(monkeypatch):
    gib = 1 << 30
    _fake_cuda(monkeypatch, 16 * gib, 24 * gib, 6 * gib, 4 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 5 * gib)
    x = torch.randn(2, 8, 64)
    model = _dense_model()
    fl.install_fast_lora(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
    cache = fl.Nf4DequantCache()
    planned = []
    handles = fl.attach_training_hooks(opt, cache, 1.0, planned.append)
    gen = fl._GEN[0]
    for _ in range(3):
        _run(model, x)
        opt.step()
    assert fl._GEN[0] - gen == 3
    assert len(planned) == 1 and cache.budget > 0
    for h in handles:
        h.remove()
    opt.step()
    assert fl._GEN[0] - gen == 3


# ---- nf4 (bitsandbytes) base ----


def _nf4_model():
    bnb = pytest.importorskip("bitsandbytes")

    def q4(lin):
        m = bnb.nn.Linear4bit(
            lin.in_features,
            lin.out_features,
            bias = lin.bias is not None,
            compute_dtype = BF16,
            quant_type = "nf4",
        )
        m.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad = False, quant_type = "nf4")
        if lin.bias is not None:
            m.bias = torch.nn.Parameter(lin.bias.data.clone().to(BF16), requires_grad = False)
        return m.to("cpu")

    torch.manual_seed(0)
    block = _Block()
    for name in ("q", "k", "o"):
        setattr(block, name, q4(getattr(block, name)))
    if getattr(block.q.weight, "quant_state", None) is None:
        pytest.skip("this bitsandbytes cannot quantize nf4 on CPU")
    return _lora(block)


def _nf4_reference(model, x):
    """fp32 copy of the same (dequantised) nf4 weights: the numbers the bf16 paths approximate."""
    import bitsandbytes.functional as bnbF

    ref = _Block()
    for name in ("q", "k", "o"):
        lora = getattr(model, name)
        base = lora.base_layer
        dense = getattr(ref, name)
        dense.weight.data = bnbF.dequantize_4bit(base.weight.data, base.weight.quant_state).float()
        if base.bias is not None:
            dense.bias.data = base.bias.data.float()
    ref = _lora(ref)
    src = dict(model.named_parameters())
    for n, p in ref.named_parameters():
        if "lora_" in n:
            p.data = src[n].data.clone().float()
    return _run(ref, x, autocast = False)


def test_nf4_fast_path_matches_peft_and_keeps_the_fp32_output_contract():
    x = torch.randn(2, 16, 64)
    model = _nf4_model()
    fast_model = copy.deepcopy(model)
    assert fl.install_fast_lora(fast_model) == 3
    fast = _run(fast_model, x)
    want = _run(model, x)
    _assert_not_worse(fast, want, _nf4_reference(model, x))
    # fp32 inputs into an nf4 base (FLUX.2 klein, Wan after block 0): PEFT's fp32 base + LoRA sum, bit for bit.
    assert torch.equal(fast[0], want[0]) and torch.equal(fast[1], want[1])
    for n in want[2]:
        assert torch.equal(fast[2][n], want[2][n]), n
    with torch.autocast("cpu", dtype = BF16):
        # bnb hands back the caller's dtype (fp32 here); the next op sees the same dtype as on PEFT.
        assert fast_model.q(torch.nn.functional.layer_norm(x, (64,))).dtype == torch.float32


def test_nf4_dequant_cache_is_exact_and_respects_its_budget():
    x = torch.randn(2, 16, 64)
    model = _nf4_model()
    uncached = copy.deepcopy(model)
    fl.install_fast_lora(uncached)
    want = _run(uncached, x)

    one_layer = 64 * 96 * 2  # q: 96 x 64 bf16
    cache = fl.Nf4DequantCache(one_layer)
    cached = copy.deepcopy(model)
    fl.install_fast_lora(cached, cache)
    got = _run(cached, x)
    # Same dequantised weights, same GEMMs: bit-identical to dequantising per call.
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    for n in want[2]:
        assert torch.equal(got[2][n], want[2][n]), n
    st = cache.stats()
    assert st["cached_layers"] == 1 and cache.used == one_layer and cache.full
    hits = cache.hits
    _run(cached, x)
    assert cache.hits > hits  # forward + backward of the cached layer reuse the entry
    assert cache.used <= cache.budget

    cache.suspend()
    assert cache.used == 0 and not cache.entries and cache.budget == one_layer
    _run(cached, x)
    assert not cache.entries  # a preview render while suspended does not refill it
    cache.resume()
    _run(cached, x)
    assert cache.stats()["cached_layers"] == 1
    cache.clear()
    assert cache.used == 0 and not cache.entries and cache.budget == one_layer
    cache.release()
    assert cache.budget == 0
    _run(cached, x)
    assert not cache.entries  # released: nothing cached any more, memory stays flat


def test_nf4_zero_budget_caches_nothing():
    x = torch.randn(2, 8, 64)
    model = _nf4_model()
    cache = fl.Nf4DequantCache(0)
    fl.install_fast_lora(model, cache)
    _run(model, x)
    assert not cache.entries and cache.used == 0 and cache.misses > 0


def test_nf4_linear_outside_lora_reads_the_cache_too():
    x = torch.randn(2, 8, 64)
    model = _nf4_model()
    # An untargeted nf4 linear beside the adapters (FLUX.2 klein's feed-forwards).
    extra = copy.deepcopy(model.q.base_layer)
    holder = torch.nn.ModuleDict({"lora": model, "plain": extra})
    with torch.no_grad():
        want = extra(x).float()
    cache = fl.Nf4DequantCache(1 << 30)
    assert fl.install_fast_lora(holder, cache) == 3
    assert cache.layers == 4  # three LoRA bases plus the plain one
    with torch.no_grad():
        got = extra(x)
    assert got.dtype == x.dtype and torch.equal(got.float(), want)
    assert id(extra) in cache.entries
    fl.uninstall_fast_lora(holder)
    assert "forward" not in extra.__dict__
