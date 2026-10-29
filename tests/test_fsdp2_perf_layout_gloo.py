# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 fit layout (unsloth/models/fsdp2_perf.py), two gloo ranks on CPU.

When the model fits: frozen weights stay plain replicated tensors (never DTensor, never gathered),
trainable weights are sharded DTensors, and loss + reduced gradients match an unsharded copy. Both the
single root unit (small trainable set) and per-block units are checked. The planner falls back to the
plain sharded layout when the model does not fit or CPU offload is asked for.
"""

from __future__ import annotations

import os

import pytest

# Spawned gloo ranks re-import this module: keep them on the CPU (deterministic on GPU hosts; CUDA
# reads the mask at its lazy first use, after this).
if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
pytest.importorskip("torch.distributed.fsdp")

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)

import importlib.util  # noqa: E402

# Load the module by path: it imports nothing from unsloth, and the spawned ranks skip unsloth's import.
_spec = importlib.util.spec_from_file_location(
    "unsloth_fsdp2_perf_under_test",
    os.path.join(os.path.dirname(__file__), "..", "unsloth", "models", "fsdp2_perf.py"),
)
fsdp2_perf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsdp2_perf)


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


D = 16


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(D, D, bias = False)
        self.A = torch.nn.Linear(D, 4, bias = False)
        self.B = torch.nn.Linear(4, D, bias = False)
        self.base.weight.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = Adapter()
        self.o = Adapter()

    def forward(self, x):
        return x + torch.tanh(self.o(self.q(x)))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, D)
        self.layers = torch.nn.ModuleList(Block() for _ in range(3))
        self.head = torch.nn.Linear(D, 32, bias = False)
        self.embed.weight.requires_grad_(False)
        self.head.weight.requires_grad_(False)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return torch.nn.functional.cross_entropy(self.head(x).flatten(0, 1), ids.flatten())


def _worker(rank, world, port, units_mode, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), UNSLOTH_FSDP2_TRAINABLE_UNITS = units_mode
    )
    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
        from torch.distributed.tensor import DTensor

        fsdp2_perf._device_bytes = lambda device = None: 1 << 40  # pretend a big GPU: the model fits
        torch.manual_seed(0)
        ref = Toy()
        torch.manual_seed(0)
        model = Toy()
        ids = torch.randint(0, 32, (2, 8), generator = torch.Generator().manual_seed(1))
        ref_loss = ref(ids)
        ref_loss.backward()

        layout = fsdp2_perf.plan_fsdp2_layout(model, param_dtype = torch.float32)
        wrapped = fsdp2_perf.shard_model_layout(
            model, list(model.layers), MixedPrecisionPolicy(), layout, mesh = None
        )
        loss = model(ids)
        loss.backward()
        res = {
            "layout": layout.as_dict(),
            "n_wrapped": len(wrapped),
            "loss": float(loss),
            "ref_loss": float(ref_loss),
            "frozen_plain": all(
                not isinstance(p, DTensor) for p in model.parameters() if not p.requires_grad
            ),
            "trainable_dtensor": all(
                isinstance(p, DTensor) for p in model.parameters() if p.requires_grad
            ),
            "frozen_no_grad": all(
                p.grad is None for p in model.parameters() if not p.requires_grad
            ),
        }
        ref_grads = {n: p.grad for n, p in ref.named_parameters() if p.requires_grad}
        diffs = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                diffs.append(float((g - ref_grads[n]).abs().max()))
        res["max_grad_diff"] = max(diffs)
        if rank == 0:
            out.put(res)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("units_mode", ["auto", "blocks"])
def test_fit_layout_matches_unsharded(units_mode):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(
        _worker, args = (2, _free_port(), units_mode, q), nprocs = 2, join = True, start_method = "spawn"
    )
    res = q.get(timeout = 60)
    assert res["layout"]["replicate_frozen"] is True
    assert res["layout"]["reshard_after_forward"] is False
    assert res["layout"]["root_only"] is (units_mode == "auto")
    assert res["n_wrapped"] == (0 if units_mode == "auto" else 3)
    assert res["frozen_plain"] and res["trainable_dtensor"] and res["frozen_no_grad"]
    assert abs(res["loss"] - res["ref_loss"]) < 1e-6
    assert res["max_grad_diff"] < 1e-6


def test_planner_falls_back_when_not_fitting(monkeypatch):
    model = Toy()
    monkeypatch.setattr(fsdp2_perf, "_device_bytes", lambda device = None: 1024)
    layout = fsdp2_perf.plan_fsdp2_layout(model)
    assert not layout.fits and not layout.replicate_frozen and layout.reshard_after_forward is True
    assert not layout.root_only


def test_planner_cpu_offload_and_overrides(monkeypatch):
    model = Toy()
    monkeypatch.setattr(fsdp2_perf, "_device_bytes", lambda device = None: 1 << 40)
    assert not fsdp2_perf.plan_fsdp2_layout(model, cpu_offload = True).replicate_frozen
    assert (
        fsdp2_perf.plan_fsdp2_layout(model, reshard_after_forward = True).reshard_after_forward
        is True
    )
    monkeypatch.setenv("UNSLOTH_FSDP2_REPLICATE_FROZEN", "0")
    assert not fsdp2_perf.plan_fsdp2_layout(model).replicate_frozen
    monkeypatch.setenv("UNSLOTH_FSDP2_REPLICATE_FROZEN", "auto")
    monkeypatch.setenv("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD", "1")
    assert fsdp2_perf.plan_fsdp2_layout(model).reshard_after_forward is True
    # Full fine-tuning (nothing frozen): nothing to replicate.
    for p in model.parameters():
        p.requires_grad_(True)
    monkeypatch.delenv("UNSLOTH_FSDP2_RESHARD_AFTER_FORWARD")
    layout = fsdp2_perf.plan_fsdp2_layout(model)
    assert not layout.replicate_frozen and layout.reshard_after_forward is False
