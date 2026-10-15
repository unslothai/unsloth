# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""UNSLOTH_OFFLOAD_OPTIMIZER_STATE: loaded from its file so the CPU cases need no GPU unsloth import."""

import importlib.util
import io
import os
import types

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "optimizers", "state_offload.py")


def _module():
    spec = importlib.util.spec_from_file_location("_unsloth_state_offload", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_trainer(optimizer):
    trainer = types.SimpleNamespace(optimizer = optimizer, is_deepspeed_enabled = False)
    trainer.calls = 0

    def create_scheduler(num_training_steps, optimizer = None):
        trainer.calls += 1
        return "scheduler"

    trainer.create_scheduler = create_scheduler
    return trainer


def test_env_off_is_a_no_op(monkeypatch):
    monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", raising = False)
    module = _module()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))])
    trainer = _fake_trainer(optimizer)
    original = trainer.create_scheduler
    assert module.install_optimizer_state_offload(trainer) is False
    assert trainer.create_scheduler is original
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "0")
    assert module.install_optimizer_state_offload(trainer) is False
    assert trainer.create_scheduler is original


def test_env_on_wraps_create_scheduler(monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "1")
    module = _module()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))])
    trainer = _fake_trainer(optimizer)
    assert module.install_optimizer_state_offload(trainer) is True
    wrapped = trainer.create_scheduler
    assert module.install_optimizer_state_offload(trainer) is True
    assert trainer.create_scheduler is wrapped
    assert trainer.create_scheduler(10) == "scheduler"
    assert trainer.calls == 1
    # CPU parameters: nothing to offload, optimizer untouched.
    assert getattr(optimizer, "_unsloth_state_offloader", None) is None
    assert "load_state_dict" not in vars(optimizer)


def test_deepspeed_is_skipped(monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "1")
    module = _module()
    trainer = _fake_trainer(None)
    trainer.is_deepspeed_enabled = True
    original = trainer.create_scheduler
    assert module.install_optimizer_state_offload(trainer) is False
    assert trainer.create_scheduler is original


def _train(optimizer_factory, offload, steps = 6, seed = 0, chunk_bytes = None):
    module = _module()
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 8192), torch.nn.GELU(), torch.nn.Linear(8192, 64)
    ).cuda()
    optimizer = optimizer_factory(model.parameters())
    # Bound before offloading, as accelerate keeps it and swaps it back in on every fp16 step.
    step_fn = optimizer.step
    if offload:
        assert module.offload_optimizer_state(optimizer, chunk_bytes = chunk_bytes) is not None
    x = torch.randn(32, 64, device = "cuda", generator = None)
    losses = []
    for step in range(steps):
        loss = (model(x + step) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_fn()
        optimizer.zero_grad(set_to_none = True)
        losses.append(loss.item())
    return model, optimizer, losses


def _state_devices(optimizer):
    return {
        v.device.type
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor) and v.numel() > 1
    }


_FACTORIES = {
    "adamw": lambda p: torch.optim.AdamW(p, lr = 1e-3, weight_decay = 0.01),
    "adamw_foreach": lambda p: torch.optim.AdamW(p, lr = 1e-3, foreach = True),
    "adamw_fused": lambda p: torch.optim.AdamW(p, lr = 1e-3, fused = True),
}


def _bnb_factory():
    bnb = pytest.importorskip("bitsandbytes")
    return lambda p: bnb.optim.AdamW8bit(p, lr = 1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("name", ["adamw", "adamw_foreach", "adamw_fused", "adamw_8bit"])
@pytest.mark.parametrize("chunk_bytes", [None, 1])
def test_offload_matches_on_device_steps_and_resumes(name, chunk_bytes):
    factory = _bnb_factory() if name == "adamw_8bit" else _FACTORIES[name]
    base_model, base_opt, base_losses = _train(factory, offload = False)
    head_model, head_opt, head_losses = _train(factory, offload = True, chunk_bytes = chunk_bytes)
    assert head_losses == base_losses
    for a, b in zip(base_model.parameters(), head_model.parameters()):
        assert torch.equal(a, b)
    assert _state_devices(base_opt) == {"cuda"}
    assert _state_devices(head_opt) == {"cpu"}
    assert head_opt._unsloth_state_offloader.steps == 6

    # Checkpoint written while offloaded, loaded the way HF Trainer does (map_location = "cpu").
    buffer = io.BytesIO()
    torch.save(head_opt.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location = "cpu", weights_only = False)
    resumed_model, resumed_opt, _ = _train(factory, offload = True, steps = 0, chunk_bytes = chunk_bytes)
    resumed_model.load_state_dict(head_model.state_dict())
    resumed_opt.load_state_dict(saved)
    assert _state_devices(resumed_opt) == {"cpu"}

    x = torch.randn(32, 64, device = "cuda")
    for model, optimizer in ((base_model, base_opt), (resumed_model, resumed_opt)):
        (model(x) ** 2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
    for a, b in zip(base_model.parameters(), resumed_model.parameters()):
        assert torch.equal(a, b)
    assert _state_devices(resumed_opt) == {"cpu"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_chunked_step_bounds_device_state_and_keeps_grads():
    module = _module()
    params = [torch.nn.Parameter(torch.randn(1 << 20, device = "cuda")) for _ in range(4)]
    optimizer = torch.optim.AdamW(params, lr = 1e-3)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 9 << 20)
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    for p in params:
        p.grad = torch.ones_like(p)
    grads = [p.grad for p in params]
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    peak = torch.cuda.max_memory_allocated() - base
    assert all(p.grad is g for p, g in zip(params, grads))
    # One parameter's state (2 x 4 MiB) on device at a time, not all four (32 MiB).
    assert peak < 16 << 20
    assert _state_devices(optimizer) == {"cpu"}
    assert offloader.steps == 2
