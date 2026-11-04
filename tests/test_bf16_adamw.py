# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from real_accelerator import has_real_cuda


@pytest.fixture(scope = "module")
def api():
    path = Path(__file__).resolve().parents[1] / "unsloth/optimizers/bf16_adamw.py"
    spec = importlib.util.spec_from_file_location("_test_bf16_adamw", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_preserves_identity_ties_and_frozen_parameters(api):
    model = torch.nn.Module()
    model.first = torch.nn.Linear(4, 4)
    model.second = torch.nn.Linear(4, 4, bias = False)
    model.second.weight = model.first.weight
    model.frozen = torch.nn.Parameter(torch.arange(4.0), requires_grad = False)
    model.register_buffer("statistics", torch.ones(4))
    before = {name: id(p) for name, p in model.named_parameters()}
    frozen = model.frozen.detach().clone()
    api.prepare_bf16_training(model)
    assert {name: id(p) for name, p in model.named_parameters()} == before
    assert model.first.weight is model.second.weight
    assert model.first.weight.dtype == model.first.bias.dtype == torch.bfloat16
    assert model.first.weight.requires_grad and model.first.bias.requires_grad
    assert model.frozen.dtype == torch.float32 and not model.frozen.requires_grad
    assert torch.equal(model.frozen, frozen)
    assert model.statistics.dtype == torch.float32
    api.prepare_bf16_training(model)
    assert {name: id(p) for name, p in model.named_parameters()} == before


@pytest.mark.parametrize("invalid", ["dtype", "gradient"])
def test_prepare_preflight_does_not_partially_cast(api, invalid):
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.ones(4))
    model.last = torch.nn.Parameter(
        torch.ones(4, dtype = torch.float64 if invalid == "dtype" else torch.float32)
    )
    if invalid == "gradient":
        model.last.grad = torch.ones_like(model.last)
    before = [(id(p), p.dtype, p.detach().clone()) for p in model.parameters()]
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        api.prepare_bf16_training(model)
    for param, (identity, dtype, value) in zip(model.parameters(), before):
        assert id(param) == identity and param.dtype == dtype
        assert torch.equal(param, value)


def test_moments_match_independent_fp32_adamw(api):
    p = torch.nn.Parameter(torch.linspace(-2, 2, 65, dtype = torch.bfloat16))
    reference = torch.nn.Parameter(p.detach().float())
    kwargs = dict(lr = 0.003, betas = (0.7, 0.95), eps = 1e-6, weight_decay = 0.03)
    optimizer = api.AdamWBF16([p], seed = 57, **kwargs)
    control = torch.optim.AdamW([reference], foreach = False, fused = False, **kwargs)
    for step in range(1, 5):
        gradient = (torch.arange(65).float().sin() * step / 7).bfloat16()
        p.grad = gradient.clone()
        reference.grad = gradient.float()
        optimizer.step()
        control.step()
        for key in ("exp_avg", "exp_avg_sq"):
            assert optimizer.state[p][key].dtype == torch.float32
            torch.testing.assert_close(
                optimizer.state[p][key], control.state[reference][key], atol = 2e-8, rtol = 2e-6
            )
        assert int(optimizer.state[p]["step"]) == step


def test_sub_ulp_update_moves_correct_fraction_and_preserves_rng(api):
    p = torch.nn.Parameter(torch.ones(65536, dtype = torch.bfloat16))
    optimizer = api.AdamWBF16([p], lr = 2**-10, betas = (0.0, 0.0), eps = 0.0, weight_decay = 0.0, seed = 71)
    p.grad = torch.ones_like(p)
    global_rng = torch.random.get_rng_state().clone()
    optimizer.step()
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    lower = torch.tensor(1 - 2**-8, dtype = torch.bfloat16)
    assert torch.all((p == 1) | (p == lower))
    moved = (p == lower).sum().item()
    assert abs(moved - p.numel() / 4) < 6 * (p.numel() * 0.25 * 0.75) ** 0.5
    assert abs(p.float().mean().item() - (1 - 2**-10)) < 3e-5


def test_checkpoint_restores_fp32_moments_and_next_random_update(api):
    p = torch.nn.Parameter(torch.linspace(0.5, 1.5, 4096, dtype = torch.bfloat16))
    optimizer = api.AdamWBF16([p], lr = 0.0009, weight_decay = 0.01, seed = 81)
    for step in range(3):
        p.grad = (torch.arange(p.numel()).float().cos() + step / 10).bfloat16()
        optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    restored_p = torch.nn.Parameter(p.detach().clone())
    restored = api.AdamWBF16([restored_p], lr = 0.2, seed = 999)
    restored.load_state_dict(state)
    for key in ("exp_avg", "exp_avg_sq"):
        assert restored.state[restored_p][key].dtype == torch.float32
        assert torch.equal(restored.state[restored_p][key], optimizer.state[p][key])
    for step in range(3):
        gradient = (torch.arange(p.numel()).float().sin() + step / 20).bfloat16()
        p.grad = gradient.clone()
        restored_p.grad = gradient.clone()
        optimizer.step()
        restored.step()
        assert torch.equal(p, restored_p)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                torch.as_tensor(optimizer.state[p][key]),
                torch.as_tensor(restored.state[restored_p][key]),
            )


def test_none_gradient_is_skipped_and_groups_keep_own_steps(api):
    p = torch.nn.Parameter(torch.ones(64, dtype = torch.bfloat16))
    q = torch.nn.Parameter(torch.full((64,), 2.0, dtype = torch.bfloat16))
    optimizer = api.AdamWBF16([p], lr = 0.01, weight_decay = 0.1)
    optimizer.add_param_group({"params": [q], "lr": 0.0})
    p.grad = torch.ones_like(p)
    optimizer.step()
    if optimizer.state.get(q):
        assert int(optimizer.state[q]["step"]) == 0
        assert torch.count_nonzero(optimizer.state[q]["exp_avg"]) == 0
        assert torch.count_nonzero(optimizer.state[q]["exp_avg_sq"]) == 0
    assert torch.all(q == 2)
    p_before = p.detach().clone()
    p.grad = None
    q.grad = torch.ones_like(q)
    optimizer.step()
    assert torch.equal(p, p_before)
    assert int(optimizer.state[p]["step"]) == 1
    assert int(optimizer.state[q]["step"]) == 1
    assert torch.all(q == 2)
    optimizer.param_groups[1]["lr"] = 0.03
    optimizer.step()
    assert torch.all(q < 2)
    assert int(optimizer.state[q]["step"]) == 2


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def test_persistent_storage_has_no_master_weight(api):
    p = torch.nn.Parameter(torch.ones(4096, dtype = torch.bfloat16))
    optimizer = api.AdamWBF16([p])
    p.sum().backward()
    optimizer.step()
    storage = p.numel() * p.element_size() + p.grad.numel() * p.grad.element_size()
    storage += _tensor_bytes(optimizer.state_dict())
    assert p.grad.dtype == torch.bfloat16
    assert storage <= 12 * p.numel() + 8192
    for key, value in optimizer.state[p].items():
        if isinstance(value, torch.Tensor) and value.numel() == p.numel():
            assert key in ("exp_avg", "exp_avg_sq")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.complex64])
def test_unsupported_parameter_dtype_is_rejected(api, dtype):
    p = torch.nn.Parameter(torch.ones(8, dtype = dtype))
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        optimizer = api.AdamWBF16([p])
        p.grad = torch.ones_like(p)
        optimizer.step()


def test_sparse_gradient_is_rejected_without_updating_parameter(api):
    p = torch.nn.Parameter(torch.ones(8, dtype = torch.bfloat16))
    optimizer = api.AdamWBF16([p])
    p.grad = torch.sparse_coo_tensor(
        [[1, 3]], torch.ones(2, dtype = p.dtype), (8,), check_invariants = True
    )
    before = p.detach().clone()
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        optimizer.step()
    assert torch.equal(p, before)


def _cuda_pair(api):
    initial = torch.linspace(0.5, 1.5, 4096, device = "cuda", dtype = torch.bfloat16)
    params = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
    optimizers = [api.AdamWBF16([p], lr = 0.0009, seed = 73) for p in params]
    for p in params:
        p.grad = torch.ones_like(p)
    return params, optimizers


def _assert_optimizer_equal(params, optimizers):
    assert torch.equal(*params)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(optimizers[0].state[params[0]][key], optimizers[1].state[params[1]][key])


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires real CUDA")
def test_compiled_optimizer_matches_eager_without_graph_breaks(api):
    from torch._dynamo.utils import counters

    params, optimizers = _cuda_pair(api)
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(optimizers[1].step, fullgraph = True)
    try:
        for index in range(4):
            gradient = (torch.arange(4096, device = "cuda").float().sin() + index / 10).bfloat16()
            for p in params:
                p.grad.copy_(gradient)
            optimizers[0].step()
            compiled()
            _assert_optimizer_equal(params, optimizers)
        assert int(optimizers[1].state[params[1]]["step"]) == 4
        assert sum(counters["graph_break"].values()) == 0
        assert counters["stats"]["unique_graphs"] > 0
    finally:
        torch._dynamo.reset()


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires real CUDA")
def test_cuda_graph_replay_advances_stochastic_updates(api):
    params, optimizers = _cuda_pair(api)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for optimizer in optimizers:
            optimizer.step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        optimizers[1].step()
    for _ in range(5):
        optimizers[0].step()
        graph.replay()
        _assert_optimizer_equal(params, optimizers)
    assert int(optimizers[1].state[params[1]]["step"]) == 6
