# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 stochastic rounding for full finetuning. CPU; bitsandbytes on CUDA only."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

STEPS = 30


@pytest.fixture
def sr(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    path = Path(__file__).resolve().parents[1] / "unsloth/optimizers/bf16_stochastic_rounding.py"
    spec = importlib.util.spec_from_file_location("_test_bf16_sr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rounding_exact_specials_and_unbiased(sr):
    exact = torch.tensor([1.0, -2.5, 0.0, -0.0, 3.0e38], dtype = torch.bfloat16)
    out = torch.empty_like(exact)
    sr.stochastic_round_to_bf16_(out, exact.float())
    assert torch.equal(out.view(torch.int16), exact.view(torch.int16))
    special = torch.tensor([float("inf"), float("-inf"), float("nan")])
    out = torch.empty(3, dtype = torch.bfloat16)
    sr.stochastic_round_to_bf16_(out, special)
    assert out[0] == float("inf") and out[1] == float("-inf") and out[2].isnan()
    for sign in (1, -1):
        x = torch.full((200000,), sign * (1 + 2**-9))
        out = torch.empty(x.shape, dtype = torch.bfloat16)
        sr.stochastic_round_to_bf16_(out, x, torch.Generator().manual_seed(0))
        assert set(out.float().abs().unique().tolist()) == {1.0, 1 + 2**-7}
        assert abs(out.float().mean().item() - sign * (1 + 2**-9)) < 3e-5


def _weights():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(64, 48, generator = g) * 0.05, torch.randn(300, generator = g) * 0.05]


def _grads(step):
    g = torch.Generator().manual_seed(100 + step)
    return [torch.randn(64, 48, generator = g) * 1e-3, torch.randn(300, generator = g) * 1e-3]


def _train(
    sr,
    mode,
    lr = 1e-5,
    **kw,
):
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    params = [torch.nn.Parameter(w.to(torch.bfloat16).to(dtype)) for w in _weights()]
    start = [p.detach().float().clone() for p in params]
    opt = torch.optim.AdamW(params, lr = lr, weight_decay = 0.0, **kw)
    if mode == "sr":
        assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    for step in range(STEPS):
        for p, g in zip(params, _grads(step)):
            p.grad = g.to(dtype)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none = True)
    assert all(p.dtype == dtype for p in params)
    return torch.cat([(p.detach().float() - s).flatten() for p, s in zip(params, start)])


@pytest.mark.parametrize("foreach", [False, True])
def test_torch_adamw_keeps_the_float32_update(sr, foreach):
    ref = _train(sr, "fp32", foreach = foreach).double()
    kept = {
        m: (_train(sr, m, foreach = foreach).double() @ ref / (ref @ ref)).item()
        for m in ("bf16", "sr")
    }
    assert kept["bf16"] < 0.5, kept
    assert abs(kept["sr"] - 1) < 0.1, kept


def test_resume_with_an_initially_inactive_weight(sr):
    def setup(weights):
        params = [torch.nn.Parameter(weight.detach().clone()) for weight in weights]
        optimizer = torch.optim.AdamW(params, lr = 0.001, weight_decay = 0.0)
        sr.enable_bf16_stochastic_rounding(optimizer, verbose = False)
        return params, optimizer

    params, optimizer = setup([torch.ones(1024, dtype = torch.bfloat16) for _ in range(2)])
    params[1].sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none = True)

    resumed, restored = setup(params)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for weights, opt in ((params, optimizer), (resumed, restored)):
        (weights[0].sum() + weights[1].sum()).backward()
        opt.step()
    for actual, expected in zip(resumed, params):
        assert torch.equal(actual, expected)


def test_idempotent_kill_switch_and_scope(sr, monkeypatch):
    p = torch.nn.Parameter(torch.zeros(4, dtype = torch.bfloat16))
    q = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.AdamW([p, q])
    assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 1
    step = opt.step
    assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 1
    assert opt.step is step
    monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "0")
    assert sr.enable_bf16_stochastic_rounding(torch.optim.AdamW([p]), verbose = False) == 0
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING")
    assert sr.enable_bf16_stochastic_rounding(torch.optim.SGD([p], lr = 0.1), verbose = False) == 0
    assert sr.enable_bf16_stochastic_rounding(torch.optim.AdamW([q]), verbose = False) == 0


def test_bf16_state_from_a_checkpoint_is_upcast(sr):
    p = torch.nn.Parameter(torch.ones(8, dtype = torch.bfloat16))
    opt = torch.optim.AdamW([p], lr = 1e-3)
    p.grad = torch.ones_like(p)
    opt.step()
    saved = opt.state_dict()
    fresh = torch.optim.AdamW([p], lr = 1e-3)
    fresh.load_state_dict(saved)
    sr.enable_bf16_stochastic_rounding(fresh, verbose = False)
    p.grad = torch.ones_like(p)
    fresh.step()
    assert fresh.state[p]["exp_avg"].dtype == torch.bfloat16 and p.dtype == torch.bfloat16
    assert fresh.state[p]["exp_avg_sq"].dtype == torch.bfloat16
    assert p.grad.dtype == torch.bfloat16


def test_trainer_hook_enables_unsloth_models_only(sr):
    class Trainer:
        def create_scheduler(
            self,
            num_training_steps,
            optimizer = None,
        ):
            self.lr_scheduler = "built"
            return self.lr_scheduler

    sr.patch_trainer_bf16_stochastic_rounding(Trainer)
    patched = Trainer.create_scheduler
    sr.patch_trainer_bf16_stochastic_rounding(Trainer)
    assert Trainer.create_scheduler is patched

    class Args:
        seed = 1

    def run(full):
        model = torch.nn.Linear(4, 4).to(torch.bfloat16)
        if full is not None:
            model._unsloth_full_finetuning = full
        t = Trainer()
        t.model, t.args, t.lr_scheduler = model, Args(), None
        t.optimizer = torch.optim.AdamW(model.parameters())
        t.create_scheduler(10)
        return getattr(t.optimizer, "_unsloth_bf16_stochastic_rounding", 0)

    # Full finetuning and LoRA loads (stamped False) both cover their trained bfloat16 weights; a model Unsloth
    # did not load is left alone.
    assert run(True) == 2 and run(False) == 2
    assert run(None) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA")
@pytest.mark.parametrize("name", ["AdamW8bit", "PagedAdamW8bit"])
def test_bitsandbytes_keeps_the_float32_update(sr, name):
    bnb = pytest.importorskip("bitsandbytes")
    cls = getattr(bnb.optim, name)

    def train(mode):
        dtype = torch.float32 if mode == "fp32" else torch.bfloat16
        params = [torch.nn.Parameter(w.cuda().to(torch.bfloat16).to(dtype)) for w in _weights()]
        start = [p.detach().float().clone() for p in params]
        opt = cls(params, lr = 1e-5, weight_decay = 0.0, min_8bit_size = 256)
        if mode == "sr":
            assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 2
        for step in range(STEPS):
            for p, g in zip(params, _grads(step)):
                p.grad = g.cuda().to(dtype)
            opt.step()
            opt.zero_grad(set_to_none = True)
        assert all(p.dtype == dtype for p in params)
        return torch.cat(
            [(p.detach().float() - s).flatten() for p, s in zip(params, start)]
        ).double()

    ref = train("fp32")
    kept = {m: (train(m) @ ref / (ref @ ref)).item() for m in ("bf16", "sr")}
    assert kept["bf16"] < 0.5, kept
    assert abs(kept["sr"] - 1) < 0.1, kept


def test_noncontiguous_weights_are_left_to_the_optimizer(sr):
    p = torch.nn.Parameter(torch.randn(4, 4, dtype = torch.bfloat16).T)
    q = torch.nn.Parameter(torch.randn(4, 4, dtype = torch.bfloat16))
    opt = torch.optim.AdamW([p, q], lr = 1e-3)
    assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 1
    p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
    opt.step()
    assert p.dtype == q.dtype == torch.bfloat16


def test_scalar_weight_keeps_a_float32_step_counter(sr):
    p = torch.nn.Parameter(torch.tensor(0.5, dtype = torch.bfloat16))
    opt = torch.optim.AdamW([p], lr = 1e-3)
    sr.enable_bf16_stochastic_rounding(opt, verbose = False)
    for _ in range(300):
        p.grad = torch.ones_like(p)
        opt.step()
    step = opt.state[p]["step"]
    assert step.dtype == torch.float32 and int(step) == 300
    assert opt.state[p]["exp_avg"].dtype == torch.bfloat16


def test_resume_draws_what_an_uninterrupted_run_draws(sr):
    def run(steps, state = None):
        params = [torch.nn.Parameter(w.to(torch.bfloat16)) for w in _weights()]
        if state is not None:
            for p, w in zip(params, state["weights"]):
                p.data.copy_(w)
        opt = torch.optim.AdamW(params, lr = 1e-5, weight_decay = 0.0)
        if state is not None:
            opt.load_state_dict(state["optimizer"])
        sr.enable_bf16_stochastic_rounding(opt, verbose = False)
        start = 0 if state is None else state["done"]
        for step in range(start, start + steps):
            for p, g in zip(params, _grads(step)):
                p.grad = g.to(torch.bfloat16)
            opt.step()
            opt.zero_grad(set_to_none = True)
        return params, opt

    whole, _ = run(4)
    half, opt = run(2)
    state = {
        "weights": [p.detach().clone() for p in half],
        "optimizer": opt.state_dict(),
        "done": 2,
    }
    resumed, _ = run(2, state)
    for a, b in zip(whole, resumed):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def _inactive_first_resume(sr, opt_fn, device, cut):
    """3 weights; step 0 only the last has a gradient, step 1 the last two, then all. Run 4 steps straight,
    or `cut` steps, save weights + optimizer state, and continue in a fresh optimizer: same bits."""
    active = lambda step: range(max(0, 2 - step), 3)

    def run(steps, state = None):
        g = torch.Generator().manual_seed(0)
        params = [
            torch.nn.Parameter((torch.randn(n, generator = g) * 0.05).to(torch.bfloat16).to(device))
            for n in (4096, 300, 1024)
        ]
        if state is not None:
            for p, w in zip(params, state["weights"]):
                p.data.copy_(w)
        opt = opt_fn(params)
        assert sr.enable_bf16_stochastic_rounding(opt, verbose = False) == 3
        if state is not None:  # after enabling, as Trainer resumes
            opt.load_state_dict(state["optimizer"])
        start = 0 if state is None else state["done"]
        for step in range(start, start + steps):
            gg = torch.Generator().manual_seed(100 + step)
            for i in active(step):
                params[i].grad = (
                    (torch.randn(params[i].shape, generator = gg) * 1e-3)
                    .to(torch.bfloat16)
                    .to(device)
                )
            opt.step()
            opt.zero_grad(set_to_none = True)
        return params, opt

    whole, opt = run(4)
    half, half_opt = run(cut)
    state = {
        "weights": [p.detach().clone() for p in half],
        "optimizer": half_opt.state_dict(),
        "done": cut,
    }
    resumed, resumed_opt = run(4 - cut, state)
    for i, (a, b) in enumerate(zip(whole, resumed)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (i, int((a != b).sum()))
    return whole, opt, resumed, resumed_opt


@pytest.mark.parametrize("cut", [1, 2])
def test_resume_after_an_inactive_weight_draws_what_an_uninterrupted_run_draws(sr, cut):
    whole, opt, resumed, resumed_opt = _inactive_first_resume(
        sr, lambda params: torch.optim.AdamW(params, lr = 1e-3, weight_decay = 0.0), "cpu", cut
    )
    for p, q in zip(whole, resumed):
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                opt.state[p][k].view(torch.int16), resumed_opt.state[q][k].view(torch.int16)
            )


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA")
@pytest.mark.parametrize("cut", [1, 2])
@pytest.mark.parametrize(
    "route", ["foreach", "fused", "bnb_one_kernel", "bnb_float32_copy", "float32_copy"]
)
def test_resume_after_an_inactive_weight_is_bitwise_on_cuda(sr, monkeypatch, route, cut):
    monkeypatch.delenv("UNSLOTH_BF16_SR_FUSED", raising = False)
    monkeypatch.delenv("UNSLOTH_BF16_SR_FUSED_BNB", raising = False)
    if route == "float32_copy":
        monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "0")
    if route == "bnb_float32_copy":
        monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED_BNB", "0")
    if route.startswith("bnb"):
        bnb = pytest.importorskip("bitsandbytes")
        # 4096 elements: 8-bit state; 300 and 1024: 32-bit state, so both kernels resume
        opt_fn = lambda params: bnb.optim.AdamW8bit(params, lr = 1e-3, weight_decay = 0.01)
    else:
        kw = {"fused": True} if route == "fused" else {"foreach": route == "foreach"}
        opt_fn = lambda params: torch.optim.AdamW(params, lr = 1e-3, weight_decay = 0.01, **kw)
    whole, opt, _, resumed_opt = _inactive_first_resume(sr, opt_fn, "cuda", cut)
    counters = {
        k: opt.__dict__.get(k, 0) + resumed_opt.__dict__.get(k, 0)
        for k in ("_unsloth_sr_fused_steps", "_unsloth_sr_bnb_fused_steps")
    }
    expected = {
        "foreach": "_unsloth_sr_fused_steps",
        "fused": "_unsloth_sr_fused_steps",
        "bnb_one_kernel": "_unsloth_sr_bnb_fused_steps",
    }.get(route)
    for key, count in counters.items():
        assert (count > 0) == (key == expected), (route, counters)
