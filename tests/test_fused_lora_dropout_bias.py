# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""lora_dropout > 0 and lora_bias = True run inside the fused LoRA kernels with PEFT's semantics.

PEFT computes base(X) + lora_B(lora_A(dropout(X))) * scaling: only the LoRA branch sees the dropped
out input, every projection draws its own mask, and lora_B may carry a bias. The fused kernels used to
hand these configs back to PEFT. They now draw the same masks from the same RNG stream in the same
order, so outputs and gradients match PEFT run from the same seed, and RNG state afterwards is equal.
"""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import fast_lora

H, I = 64, 96
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEVICES = ["cpu"] + (["cuda"] if has_real_cuda() else [])


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(H, H, bias = False)
        self.k_proj = torch.nn.Linear(H, H, bias = False)
        self.v_proj = torch.nn.Linear(H, H, bias = False)
        self.o_proj = torch.nn.Linear(H, H, bias = False)
        self.gate_proj = torch.nn.Linear(H, I, bias = False)
        self.up_proj = torch.nn.Linear(H, I, bias = False)
        self.down_proj = torch.nn.Linear(I, H, bias = False)
        self.act_fn = torch.nn.SiLU()


def _peft_block(
    device,
    dtype,
    dropout,
    lora_bias = False,
):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(3407)
    cfg = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = dropout,
        target_modules = TARGETS,
        init_lora_weights = False,
        lora_bias = lora_bias,
    )
    model = get_peft_model(_Block(), cfg).to(device, dtype)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    if lora_bias:
        with torch.no_grad():
            for proj in TARGETS:
                model.base_model.model.get_submodule(proj).lora_B["default"].bias.normal_()
    model.train()
    return model, model.base_model.model


PATHS = {
    "qkv": (
        lambda b: (lambda X: fast_lora.apply_lora_qkv(b, X)),
        lambda b: (lambda X: (b.q_proj(X), b.k_proj(X), b.v_proj(X))),
    ),
    "o": (
        lambda b: (lambda X: fast_lora.apply_lora_o(b, X)),
        lambda b: b.o_proj,
    ),
    "mlp": (
        lambda b: (lambda X: fast_lora.apply_lora_mlp_swiglu(b, X)),
        lambda b: (lambda X: b.down_proj(b.act_fn(b.gate_proj(X)) * b.up_proj(X))),
    ),
}


def _skip_triton_on_cpu(device, path):
    if device == "cpu" and path == "mlp":
        pytest.skip("the SwiGLU kernels are Triton")


def _rng_state(device):
    return torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()


def _run(fn, model, X0, seed, device):
    torch.manual_seed(seed)
    X = X0.detach().clone().requires_grad_()
    out = fn(X * 1)  # a non-leaf, since the fused backward may write dX into its input buffer
    outs = out if isinstance(out, tuple) else (out,)
    gen = torch.Generator(device = "cpu").manual_seed(seed + 1)
    loss = sum((o.float() * torch.randn(o.shape, generator = gen).to(device)).sum() for o in outs)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    return [o.detach() for o in outs], X.grad.clone(), grads, _rng_state(device)


class _Spy:
    def __init__(self, monkeypatch):
        self.calls = []
        real = fast_lora._apply

        def spy(function, *args):
            self.calls.append((function.__name__, args))
            return real(function, *args)

        monkeypatch.setattr(fast_lora, "_apply", spy)


class _FixedDropout(torch.nn.Module):
    def __init__(self, mask, p):
        super().__init__()
        self.mask, self.scale = mask, 1.0 / (1.0 - p)

    def forward(self, x):
        return x * self.mask.to(x.dtype) * self.scale


def _fp64_oracle(model, path, X0, seed, device, masks, p):
    import copy

    ref_model = copy.deepcopy(model).double()
    ref_block = ref_model.base_model.model
    for name, mask in masks.items():
        ref_block.get_submodule(name).lora_dropout["default"] = _FixedDropout(mask, p)
    return _run(PATHS[path][1](ref_block), ref_model, X0.double(), seed, device)


class _Recorder(torch.nn.Module):
    # nn.Dropout under autograd is native_dropout on CUDA and bernoulli_ noise on CPU; recording the mask
    # as `output != 0` would also catch inputs that are exactly zero.
    def __init__(self, dropout, masks, name):
        super().__init__()
        self.dropout, self.masks, self.name = dropout, masks, name

    def forward(self, x):
        if x.device.type == "cpu":
            out = self.dropout(x)
            self.masks[self.name] = (out != 0) | (
                x == 0
            )  # zeros never matter on the reference path
            return out
        out, self.masks[self.name] = torch.native_dropout(x, self.dropout.p, True)
        return out


def _record_masks(block):
    masks, restore = {}, []
    for name in TARGETS:
        holder = block.get_submodule(name).lora_dropout
        dropout = holder["default"]
        if isinstance(dropout, torch.nn.Dropout):
            holder["default"] = _Recorder(dropout, masks, name)
            restore.append((holder, dropout))
    return masks, restore


def _rel_err(x, ref):
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("dropout,lora_bias", [(0.3, False), (0.0, True), (0.3, True)])
def test_fused_matches_peft(monkeypatch, device, dtype, path, dropout, lora_bias):
    """Fused vs PEFT from the same seed, each scored against an fp64 PEFT run on PEFT's masks."""
    _skip_triton_on_cpu(device, path)
    if dtype == torch.float16 and device == "cpu":
        pytest.skip("fp16 matmuls on CPU")
    model, block = _peft_block(device, dtype, dropout, lora_bias)
    fused, ref = (make(block) for make in PATHS[path])
    X0 = torch.randn(2, 7, H, device = device, dtype = dtype)
    spy = _Spy(monkeypatch)

    f_out, f_dX, f_grads, f_rng = _run(fused, model, X0, 11, device)
    assert len(spy.calls) == 1, "the fused kernel did not run"
    masks, restore = _record_masks(block)
    r_out, r_dX, r_grads, r_rng = _run(ref, model, X0, 11, device)
    for holder, module in restore:
        holder["default"] = module
    assert len(spy.calls) == 1
    assert torch.equal(f_rng, r_rng), "different number of RNG draws than PEFT"
    o_out, o_dX, o_grads, _ = _fp64_oracle(model, path, X0, 11, device, masks, dropout)

    assert f_grads.keys() == r_grads.keys()
    if lora_bias:
        assert any(n.endswith("lora_B.default.bias") for n in f_grads)
    pairs = list(zip(f_out, r_out, o_out)) + [(f_dX, r_dX, o_dX)]
    pairs += [(f_grads[n], r_grads[n], o_grads[n]) for n in r_grads]
    # Same masks, so fused is within a few roundings of PEFT, and no further from fp64 than PEFT is
    # (with one dtype step of slack: the two paths round in a different order).
    eps = torch.finfo(dtype).eps
    for f, r, o in pairs:
        e_fused, e_peft = _rel_err(f, o), _rel_err(r, o)
        assert e_fused <= 1.5 * e_peft + 2 * eps, (e_fused, e_peft)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", list(PATHS))
def test_masks_are_bitwise_peft_masks(monkeypatch, device, path):
    _skip_triton_on_cpu(device, path)
    # p = 0.5 with fp32: a single different mask element would move outputs by O(1).
    model, block = _peft_block(device, torch.float32, 0.5)
    fused, ref = (make(block) for make in PATHS[path])
    X0 = torch.randn(3, 5, H, device = device)
    for seed in range(3):
        f_out = _run(fused, model, X0, seed, device)[0]
        r_out = _run(ref, model, X0, seed, device)[0]
        for f, r in zip(f_out, r_out):
            torch.testing.assert_close(f, r, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_eval_mode_skips_dropout_like_peft(monkeypatch, device):
    model, block = _peft_block(device, torch.float32, 0.3)
    model.eval()
    spy = _Spy(monkeypatch)
    X = torch.randn(2, 5, H, device = device)
    with torch.no_grad():
        got = fast_lora.apply_lora_qkv(block, X.clone())
        expected = (block.q_proj(X), block.k_proj(X), block.v_proj(X))
    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e)
    name, args = spy.calls[0]
    assert name == "LoRA_QKV" and args[-4:] == (None, None, None, None)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("path", list(PATHS))
def test_checkpoint_recompute_replays_the_same_masks(device, path):
    from torch.utils.checkpoint import checkpoint

    _skip_triton_on_cpu(device, path)
    model, block = _peft_block(device, torch.float32, 0.3)
    fused = PATHS[path][0](block)
    X0 = torch.randn(2, 5, H, device = device)
    plain = _run(fused, model, X0, 5, device)
    ckpt = _run(
        lambda X: checkpoint(fused, X, use_reentrant = True, preserve_rng_state = True),
        model,
        X0,
        5,
        device,
    )
    torch.testing.assert_close(plain[1], ckpt[1], rtol = 0, atol = 0)
    for n in plain[2]:
        torch.testing.assert_close(plain[2][n], ckpt[2][n], rtol = 0, atol = 0)


@pytest.mark.parametrize(
    "dropout,lora_bias,flag", [(0.3, False, "FUSED_LORA_DROPOUT"), (0.0, True, "FUSED_LORA_BIAS")]
)
def test_kill_switch_restores_the_peft_fallback(monkeypatch, dropout, lora_bias, flag):
    model, block = _peft_block("cpu", torch.float32, dropout, lora_bias)
    monkeypatch.setattr(fast_lora, flag, False)
    spy = _Spy(monkeypatch)
    X = torch.randn(2, 5, H)
    fast_lora.apply_lora_qkv(block, X)
    fast_lora.apply_lora_o(block, X)
    if has_real_cuda():
        model, block = _peft_block("cuda", torch.float32, dropout, lora_bias)
        fast_lora.apply_lora_mlp_swiglu(block, X.cuda())
    assert spy.calls == []


def test_unknown_dropout_module_falls_back(monkeypatch):
    model, block = _peft_block("cpu", torch.float32, 0.3)
    block.o_proj.lora_dropout["default"] = torch.nn.AlphaDropout(0.3)
    spy = _Spy(monkeypatch)
    fast_lora.apply_lora_o(block, torch.randn(2, 5, H))
    assert spy.calls == []


def test_zero_dropout_no_bias_call_is_unchanged(monkeypatch):
    # The pre-existing fused call shape: no dropout / bias arguments reach LoRA_W.
    model, block = _peft_block("cpu", torch.float32, 0.0)
    spy = _Spy(monkeypatch)
    fast_lora.apply_lora_o(block, torch.randn(2, 5, H))
    assert [len(args) for _, args in spy.calls] == [6]


@pytest.mark.parametrize("dropout,lora_bias", [(0.0, False), (0.3, False), (0.0, True)])
def test_active_adapter_absent_from_projection_falls_back(monkeypatch, dropout, lora_bias):
    # Adapters with different target modules: PEFT runs the base layer where the active one is missing.
    from peft import LoraConfig

    model, block = _peft_block("cpu", torch.float32, dropout, lora_bias)
    model.add_adapter(
        "other",
        LoraConfig(
            r = 4, lora_alpha = 8, lora_dropout = dropout, target_modules = ["q_proj"], lora_bias = lora_bias
        ),
    )
    model.set_adapter("other")
    model.train()
    spy = _Spy(monkeypatch)
    X = torch.randn(2, 5, H)
    torch.testing.assert_close(fast_lora.apply_lora_o(block, X), block.o_proj.base_layer(X))
    fast_lora.apply_lora_qkv(block, X)
    assert spy.calls == []


@pytest.mark.parametrize("lora_bias", [False, True])
def test_routing_resolves_each_active_adapter_once(monkeypatch, lora_bias):
    # The routing runs on every fused LoRA call: the default config (no dropout, no bias) never reaches the
    # general adapter resolution, the rest resolve each projection's adapter once, and the lora_bias scan
    # is skipped unless UNSLOTH_FUSED_LORA_BIAS=0.
    model, block = _peft_block("cpu", torch.float32, 0.0, lora_bias)
    projs = (block.q_proj, block.k_proj, block.v_proj)
    real = fast_lora._active_lora_adapter
    seen = []
    monkeypatch.setattr(
        fast_lora, "_active_lora_adapter", lambda proj: seen.append(proj) or real(proj)
    )
    monkeypatch.setattr(
        fast_lora,
        "_has_active_lora_bias",
        lambda proj: pytest.fail("lora_bias scanned with fused bias on"),
    )
    dropout_p, biases = fast_lora._lora_dropout_and_bias(projs, torch.randn(2, 5, H))
    assert dropout_p is None
    assert len(seen) == (len(projs) if lora_bias else 0)
    assert all((b is not None) == lora_bias for b in biases)


@pytest.mark.parametrize(
    "state", ["train", "eval", "disabled", "merged", "absent", "list_adapter", "dropout", "bias"]
)
def test_plain_lora_shortcut_matches_general_routing(monkeypatch, state):
    # The default-config shortcut must give the general routing's answer in every adapter state.
    from peft import LoraConfig

    model, block = _peft_block(
        "cpu", torch.float32, 0.3 if state == "dropout" else 0.0, state == "bias"
    )
    if state == "eval":
        model.eval()
    elif state == "disabled":
        model.base_model.disable_adapter_layers()
    elif state == "merged":
        block.q_proj.merge()
    elif state == "absent":
        model.add_adapter("other", LoraConfig(r = 4, lora_alpha = 8, target_modules = ["o_proj"]))
        model.set_adapter("other")
    elif state == "list_adapter":
        for proj in (block.q_proj, block.k_proj, block.v_proj):
            proj._active_adapter = ["default"]
    projs = (block.q_proj, block.k_proj, block.v_proj)
    X = torch.randn(2, 5, H)
    plain = [fast_lora._is_plain_lora(proj) for proj in projs]
    fast = fast_lora._lora_dropout_and_bias(projs, X)
    monkeypatch.setattr(fast_lora, "_is_plain_lora", lambda proj: False)
    general = fast_lora._lora_dropout_and_bias(projs, X)
    assert fast == general
    assert all(plain) == (state in ("train", "eval", "list_adapter"))
