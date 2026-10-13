# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO's manual DDP gradient average copies each bucket back with one fused copy.

``_unsloth_average_gradients`` is lifted from ``unsloth/models/rl.py`` with ``ast`` so the test stays
CPU-only; collectives are stubbed (a 2-rank all_reduce of identical grads is a SUM of two copies).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_average_gradients"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([node], []), str(SOURCE), "exec"), namespace)
    return namespace["_unsloth_average_gradients"]


@pytest.fixture
def two_identical_ranks(monkeypatch):
    def all_reduce(
        tensor,
        op = None,
        group = None,
    ):
        if op is None or op == dist.ReduceOp.SUM:
            tensor.mul_(2)

    monkeypatch.setattr(dist, "get_world_size", lambda group = None: 2)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)


def _model(shapes, dtype = torch.float32):
    torch.manual_seed(0)
    module = torch.nn.Module()
    module.params = torch.nn.ParameterList(
        torch.nn.Parameter(torch.zeros(*s, dtype = dtype)) for s in shapes
    )
    for p in module.params:
        p.grad = torch.randn_like(p)
    return SimpleNamespace(process_group = None, module = module)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bucket_copy_back_is_fused_and_exact(two_identical_ranks, monkeypatch, dtype):
    average = _load()
    # LoRA-like: many small A / B tensors that share a bucket, one non-contiguous gradient.
    ddp = _model([(4, 16), (16, 4), (4, 32), (32, 4), (8,)], dtype)
    ddp.module.params[1].grad = torch.randn(4, 16, dtype = dtype).t()
    before = [p.grad.clone() for p in ddp.module.params]
    calls = []
    real = torch._foreach_copy_

    def counting(dst, src, *args, **kwargs):
        calls.append(len(dst))
        return real(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch, "_foreach_copy_", counting)
    average(ddp)
    assert calls == [len(before)]
    for p, g in zip(ddp.module.params, before):
        # SUM of two identical ranks then div by 2 gives the gradient back bit for bit.
        assert torch.equal(p.grad, g)


def test_oversized_gradient_still_reduced_in_place(two_identical_ranks):
    average = _load()
    ddp = _model([(64, 64), (3,)])
    before = [p.grad.clone() for p in ddp.module.params]
    average(ddp, bucket_bytes = 1)
    for p, g in zip(ddp.module.params, before):
        assert torch.equal(p.grad, g)


@pytest.mark.parametrize("mode", ["missing", "raises"])
def test_per_tensor_fallback_when_foreach_copy_unavailable(two_identical_ranks, monkeypatch, mode):
    average = _load()
    ddp = _model([(4, 16), (16, 4), (8,)])
    ddp.module.params[1].grad = torch.randn(4, 16).t()
    before = [p.grad.clone() for p in ddp.module.params]
    if mode == "missing":
        monkeypatch.delattr(torch, "_foreach_copy_")
    else:

        def broken(dst, src, *args, **kwargs):
            dst[0].zero_()  # a partial write before failing must not leak through
            raise RuntimeError("unsupported")

        monkeypatch.setattr(torch, "_foreach_copy_", broken)
    average(ddp)
    for p, g in zip(ddp.module.params, before):
        assert torch.equal(p.grad, g)
