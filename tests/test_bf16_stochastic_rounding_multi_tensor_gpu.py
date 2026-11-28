# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 stochastic rounding on CUDA: the multi-tensor AdamW launch (one per device and gradient dtype) gives
the same bits as one _adamw_kernel launch per weight, and UNSLOTH_BF16_SR_MULTI_TENSOR=0 restores the latter."""

import importlib.util
from pathlib import Path

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA"),
]
pytest.importorskip("triton")

# Sizes: tiny, one block plus a tail, past one 16384-element chunk, a 2-D weight, and an empty one.
SHAPES = [(3,), (1025,), (16384 * 2 + 5,), (96, 257), (0,), (64, 64)]
G_DTYPES = [
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
    torch.bfloat16,
    torch.bfloat16,
    torch.float16,
]


@pytest.fixture
def sr(monkeypatch):
    for name in (
        "UNSLOTH_BF16_STOCHASTIC_ROUNDING",
        "UNSLOTH_BF16_SR_FUSED",
        "UNSLOTH_BF16_SR_MULTI_TENSOR",
    ):
        monkeypatch.delenv(name, raising = False)
    path = Path(__file__).resolve().parents[1] / "unsloth/optimizers/bf16_stochastic_rounding.py"
    spec = importlib.util.spec_from_file_location("_test_bf16_sr_multi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _active(step, index):
    """Weight 0 first gets a gradient at step 2 and weight 3 skips step 3: steps differ inside one launch."""
    return not (index == 0 and step < 2) and not (index == 3 and step == 3)


def _run(
    sr,
    monkeypatch,
    multi,
    steps = 6,
    state = None,
    start = 0,
    groups = False,
):
    monkeypatch.setenv("UNSLOTH_BF16_SR_MULTI_TENSOR", "1" if multi else "0")
    g = torch.Generator().manual_seed(0)
    params = [
        torch.nn.Parameter((torch.randn(shape, generator = g) * 0.05).to(torch.bfloat16).cuda())
        for shape in SHAPES
    ]
    for p, dtype in zip(params, G_DTYPES):
        if dtype != p.dtype and hasattr(p, "grad_dtype"):
            p.grad_dtype = None  # torch >= 2.10 refuses another gradient dtype unless asked
    if state is not None:
        for p, w in zip(params, state["weights"]):
            p.data.copy_(w)
    if groups:
        param_groups = [
            {"params": params[:3], "weight_decay": 0.01},
            {"params": params[3:], "weight_decay": 0.0, "lr": 3e-4},
        ]
    else:
        param_groups = [{"params": params}]
    opt = torch.optim.AdamW(param_groups, lr = 1e-3, weight_decay = 0.01)
    assert sr.enable_bf16_stochastic_rounding(opt, seed = 7, verbose = False) == len(params)
    if state is not None:
        opt.load_state_dict(state["optimizer"])
    for step in range(start, start + steps):
        gen = torch.Generator().manual_seed(100 + step)
        for i, (p, dtype) in enumerate(zip(params, G_DTYPES)):
            grad = torch.randn(p.shape, generator = gen) * 1e-3
            if _active(step, i):
                p.grad = grad.to(dtype).cuda()
        opt.step()
        opt.zero_grad(set_to_none = True)
    return params, opt


def _same(a_params, a_opt, b_params, b_opt):
    for i, (a, b) in enumerate(zip(a_params, b_params)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (i, int((a != b).sum()))
        sa, sb = a_opt.state[a], b_opt.state[b]
        assert float(sa["step"]) == float(sb["step"]), i
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(sa[key].view(torch.int16), sb[key].view(torch.int16)), (i, key)


@pytest.mark.parametrize("groups", [False, True])
def test_multi_tensor_is_bitwise_one_launch_per_weight(sr, monkeypatch, groups):
    multi, multi_opt = _run(sr, monkeypatch, True, groups = groups)
    single, single_opt = _run(sr, monkeypatch, False, groups = groups)
    if not sr._multi_ok(multi[0].device):
        pytest.skip("multi-tensor kernel unavailable on this device")
    _same(multi, multi_opt, single, single_opt)
    # Engaged on one side only, and every weight still counted as a fused step.
    assert multi_opt.__dict__.get("_unsloth_sr_multi_tensor_launches", 0) > 0
    assert single_opt.__dict__.get("_unsloth_sr_multi_tensor_launches", 0) == 0
    assert (
        multi_opt.__dict__["_unsloth_sr_fused_steps"]
        == single_opt.__dict__["_unsloth_sr_fused_steps"]
    )


def test_multi_tensor_buckets_by_gradient_dtype(sr, monkeypatch):
    _, opt = _run(sr, monkeypatch, True, steps = 1)
    if not sr._multi_ok(torch.device("cuda")):
        pytest.skip("multi-tensor kernel unavailable on this device")
    # Step 0: weight 0 has no gradient; bf16 grads (weights 1, 3) share one launch, the fp32 and fp16 grads are
    # alone in their buckets and the empty weight needs no launch, so they take one launch per weight.
    assert opt.__dict__["_unsloth_sr_multi_tensor_launches"] == 1


def test_multi_tensor_resume_is_bitwise(sr, monkeypatch):
    whole, whole_opt = _run(sr, monkeypatch, True, steps = 5)
    half, half_opt = _run(sr, monkeypatch, True, steps = 2)
    state = {"weights": [p.detach().clone() for p in half], "optimizer": half_opt.state_dict()}
    resumed, resumed_opt = _run(sr, monkeypatch, True, steps = 3, state = state, start = 2)
    _same(whole, whole_opt, resumed, resumed_opt)


def test_multi_tensor_launch_failure_falls_back_bitwise(sr, monkeypatch, capsys):
    single, single_opt = _run(sr, monkeypatch, False)
    assert sr._multi_ok(single[0].device)

    class Broken:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("simulated compile failure")

            return launch

    sr._adamw_multi_kernel = Broken()
    multi, multi_opt = _run(sr, monkeypatch, True)
    _same(multi, multi_opt, single, single_opt)
    assert multi_opt.__dict__.get("_unsloth_sr_multi_tensor_launches", 0) == 0
    assert sr._MULTI_OK[str(multi[0].device)] is False
    assert "multi-tensor bfloat16 stochastic rounding failed" in capsys.readouterr().out


def test_multi_tensor_probe_compares_bits(sr):
    assert sr._multi_ok(torch.device("cuda")) is True
