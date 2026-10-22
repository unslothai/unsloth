# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched multi-LoRA training: kernels against an fp64 reference, and per-adapter isolation."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import multi_lora as ml

REPO = pathlib.Path(__file__).resolve().parents[1]


def _case(
    U = 3,
    M = 37,
    K = 48,
    N = 40,
    R = 8,
    seed = 0,
    dtype = torch.float64,
    device = "cpu",
):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(M, K, generator = g, dtype = torch.float64)
    W = torch.randn(N, K, generator = g, dtype = torch.float64) / 7
    A = torch.randn(U, R, K, generator = g, dtype = torch.float64)
    B = torch.randn(U, N, R, generator = g, dtype = torch.float64)
    ids = torch.randint(-1, U, (M,), generator = g)
    ids[ids == U - 1] = 0  # the last adapter gets no tokens
    scales = torch.linspace(0.5, 2.0, U, dtype = torch.float64)
    cast = lambda t: t.to(device = device, dtype = dtype)
    return cast(X), cast(W), cast(A), cast(B), scales.to(device), ids.to(device)


def _dense(X, W, A, B, scales, ids):
    rows = []
    for t in range(X.shape[0]):
        y = X[t] @ W.t()
        u = int(ids[t])
        if u >= 0:
            y = y + scales[u] * (X[t] @ A[u].t()) @ B[u].t()
        rows.append(y)
    return torch.stack(rows)


def test_batch_groups_tokens_by_adapter_in_original_order():
    ids = torch.tensor([2, 0, -1, 0, 2, 1, -1, 0])
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, 4)
    assert batch.seg_start.tolist() == [0, 3, 4, 6, 6]
    assert batch.sorted_idx.tolist()[:6] == [1, 3, 7, 5, 0, 4]
    assert sorted(batch.sorted_idx.tolist()[6:]) == [2, 6]
    per_row = ml.MultiLoraBatch.from_adapter_ids(torch.tensor([1, 0]), 2, seq_len = 3)
    assert per_row.seg_start.tolist() == [0, 3, 6]
    assert per_row.sorted_idx.tolist() == [3, 4, 5, 0, 1, 2]
    with pytest.raises(ValueError):
        ml.MultiLoraBatch.from_adapter_ids(torch.tensor([0, 4]), 4)


def test_reference_delta_matches_the_per_token_definition():
    X, W, A, B, scales, ids = _case()
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, A.shape[0])
    Y = X @ W.t() + ml.multi_lora_delta(X, A, B, scales, batch)
    torch.testing.assert_close(Y, _dense(X, W, A, B, scales, ids), rtol = 1e-12, atol = 1e-12)


def _tiny_model(U, ranks = 4):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 24), torch.nn.Tanh(), torch.nn.Linear(24, 8))
    model = model.double()
    ml.inject_multi_lora(model, ["0", "2"], U, r = ranks, lora_alpha = 8)
    for _, layer in ml._layers(model):
        layer.double()
        torch.nn.init.normal_(layer.lora_B)
        if not isinstance(ranks, int):
            for u, k in enumerate(layer.ranks):
                layer.lora_B.data[u, :, k:] = 0
    return model


def test_each_adapter_gets_only_its_own_gradient():
    model = _tiny_model(3)
    x = torch.randn(10, 16, dtype = torch.float64)
    ids = torch.tensor([0, 1, 0, 2, 1, 0, -1, 2, 1, 0])
    ml.set_multi_lora_batch(model, ids)
    out = model(x)
    # Loss on adapter 1's rows only: adapters 0 and 2 must see exactly zero.
    out[ids == 1].pow(2).sum().backward()
    for _, layer in ml._layers(model):
        for p in (layer.lora_A, layer.lora_B):
            assert p.grad[0].abs().max() == 0 and p.grad[2].abs().max() == 0
            assert p.grad[1].abs().max() > 0
    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)


def test_batched_training_matches_each_adapter_trained_alone():
    U, steps = 2, 4
    x = torch.randn(8, 16, dtype = torch.float64)
    target = torch.randn(8, 8, dtype = torch.float64)
    ids = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1])

    def train(model, rows, row_ids):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
        ml.set_multi_lora_batch(model, row_ids)
        for _ in range(steps):
            opt.zero_grad()
            err = (model(x[rows]) - target[rows]).pow(2).mean(1)
            loss = sum(err[row_ids == u].mean() for u in range(U) if (row_ids == u).any())
            loss.backward()
            ml.clip_grad_norm_per_adapter_(model, 0.5)
            opt.step()
        return model

    batched = train(_tiny_model(U), slice(None), ids)
    for u in range(U):
        solo = train(_tiny_model(U), ids == u, torch.full((int((ids == u).sum()),), u))
        for (_, a), (_, b) in zip(ml._layers(batched), ml._layers(solo)):
            torch.testing.assert_close(a.lora_A[u], b.lora_A[u], rtol = 1e-10, atol = 1e-12)
            torch.testing.assert_close(a.lora_B[u], b.lora_B[u], rtol = 1e-10, atol = 1e-12)


def test_padded_ranks_stay_zero_and_export_at_their_own_rank():
    model = _tiny_model(2, ranks = [2, 4])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr = 1e-2, weight_decay = 0.1
    )
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 0, 1]))
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(4, 16, dtype = torch.float64)).sum().backward()
        opt.step()
    layer = ml._layers(model)[0][1]
    assert layer.lora_A[0, 2:].abs().max() == 0 and layer.lora_B[0, :, 2:].abs().max() == 0
    state = ml.adapter_state_dict(model, 0)
    assert state["base_model.model.0.lora_A.weight"].shape == (2, 16)
    assert state["base_model.model.0.lora_B.weight"].shape == (24, 2)
    assert layer.scales.tolist() == [4.0, 2.0]


def test_per_adapter_loss_is_each_adapters_own_mean():
    logits = torch.randn(3, 5, 7)
    labels = torch.randint(0, 7, (3, 5))
    labels[1, 3:] = -100
    total, losses = ml.per_adapter_loss(logits, labels, torch.tensor([1, 0, 1]), 2)
    lone = lambda r: torch.nn.functional.cross_entropy(
        logits[r, :-1].reshape(-1, 7), labels[r, 1:].reshape(-1), ignore_index = -100
    )
    torch.testing.assert_close(losses, torch.stack([lone([1]), lone([0, 2])]))
    torch.testing.assert_close(total, losses.sum())


def test_targets_match_qualified_names_like_peft():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(4, 4)), torch.nn.Linear(4, 4))
    assert ml.inject_multi_lora(model, ["0.0"], 2) == ["0.0"]
    assert isinstance(model[0][0], ml.MultiLoraLinear) and not isinstance(
        model[1], ml.MultiLoraLinear
    )


def test_layers_refuse_a_batch_that_does_not_match_the_tokens():
    model = _tiny_model(2)
    with pytest.raises(ValueError):
        model(torch.randn(4, 16, dtype = torch.float64))
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 0]))
    with pytest.raises(ValueError):
        model(torch.randn(4, 16, dtype = torch.float64))


_INTERPRET = textwrap.dedent(
    """
    import importlib.util, os, sys, types
    import torch
    root = sys.argv[1]
    pkg = types.ModuleType("mlk"); pkg.__path__ = []; sys.modules["mlk"] = pkg
    utils = types.ModuleType("mlk.utils")
    utils._base_quant_state = lambda layer, W: None
    utils.fast_dequantize = lambda W, q = None, dtype = None, **k: W
    utils.matmul_lora = lambda X, W, Wq, A, B, s, out = None: X @ W.t()
    utils.torch_amp_custom_fwd = utils.torch_amp_custom_bwd = lambda f: f
    sys.modules["mlk.utils"] = utils
    spec = importlib.util.spec_from_file_location("mlk.multi_lora", os.path.join(root, "unsloth/kernels/multi_lora.py"))
    ml = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ml; spec.loader.exec_module(ml)
    torch.manual_seed(0)
    U, M, K, N, R = 3, 37, 48, 40, 8
    X = torch.randn(M, K, requires_grad = True)
    W = torch.randn(N, K) / 7
    A = torch.randn(U, R, K, requires_grad = True)
    B = torch.randn(U, N, R, requires_grad = True)
    scales = torch.tensor([0.5, 2.0, 1.0])
    ids = torch.randint(-1, U, (M,)); ids[ids == 2] = 0
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, U)
    if len(sys.argv) > 2:  # negative control: a wrong segment table must be caught
        batch.seg_start[1] += 1
    Y = ml.MultiLoRA_W.apply(X, W, None, A, B, scales, batch.sorted_idx, batch.seg_start)
    g = torch.randn_like(Y)
    got = (Y,) + torch.autograd.grad(Y, (X, A, B), g)
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, U)
    X64, A64, B64 = [t.detach().double().requires_grad_() for t in (X, A, B)]
    Y64 = X64 @ W.double().t() + ml.multi_lora_delta(X64, A64, B64, scales.double(), batch)
    want = (Y64,) + torch.autograd.grad(Y64, (X64, A64, B64), g.double())
    for name, a, b in zip(("Y", "dX", "dA", "dB"), got, want):
        err = (a.double() - b).abs().max().item()
        assert err <= 1e-4 * max(b.abs().max().item(), 1), (name, err)
    assert got[2][2].abs().max() == 0 and got[3][2].abs().max() == 0
    print("KERNELS OK")
    """
)


@pytest.mark.parametrize("negative_control", [False, True])
def test_triton_kernels_match_fp64_in_the_interpreter(tmp_path, negative_control):
    pytest.importorskip("triton")
    script = tmp_path / "interpret.py"
    script.write_text(_INTERPRET, encoding = "utf-8")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    argv = [sys.executable, str(script), str(REPO)] + (["break"] if negative_control else [])
    proc = subprocess.run(argv, env = env, capture_output = True, text = True, timeout = 600)
    if negative_control:
        assert proc.returncode != 0 and "AssertionError" in proc.stderr, proc.stdout + proc.stderr
    else:
        assert proc.returncode == 0 and "KERNELS OK" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(3, 37, 48, 40, 8), (4, 1000, 512, 768, 16), (2, 257, 4096, 1024, 64)]
)
def test_triton_kernels_match_fp64_on_gpu(dtype, shape):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16")
    U, M, K, N, R = shape
    X, W, A, B, scales, ids = _case(U, M, K, N, R, dtype = dtype, device = "cuda")
    X.requires_grad_()
    A.requires_grad_()
    B.requires_grad_()
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, U)
    Y = ml.multi_lora_linear(X, W, None, None, A, B, scales.float(), batch)
    g = torch.randn_like(Y)
    got = (Y,) + torch.autograd.grad(Y, (X, A, B), g)
    X64, A64, B64 = [t.detach().double().requires_grad_() for t in (X, A, B)]
    Y64 = X64 @ W.double().t() + ml.multi_lora_delta(X64, A64, B64, scales, batch)
    want = (Y64,) + torch.autograd.grad(Y64, (X64, A64, B64), g.double())
    tol = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 6e-2}[dtype]
    for name, a, b in zip(("Y", "dX", "dA", "dB"), got, want):
        rel = (a.double() - b).norm() / b.norm().clamp_min(1e-12)
        assert rel <= tol, (name, rel.item())
    assert got[2][U - 1].abs().max() == 0 and got[3][U - 1].abs().max() == 0
