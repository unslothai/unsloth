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
import sys
import types

import pytest
import torch
from real_accelerator import has_real_cuda

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


def _train(
    optimizer_factory,
    offload,
    steps = 6,
    seed = 0,
    chunk_bytes = None,
):
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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
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


class _GroupCountingSGD(torch.optim.SGD):
    # Like schedulefree: group-level bookkeeping advances on every step call.
    def step(self, closure = None):
        for group in self.param_groups:
            group["k"] = group.get("k", 0) + 1
        return super().step(closure)


def test_group_global_optimizer_steps_once_per_step():
    module = _module()
    params = [torch.nn.Parameter(torch.randn(16)) for _ in range(3)]
    optimizer = _GroupCountingSGD(params, lr = 0.1)
    offloader = module._StateOffloader(optimizer, chunk_bytes = 1)
    optimizer.register_step_pre_hook(offloader._pre_step)
    optimizer.register_step_post_hook(offloader._post_step)
    expected = [p.detach() - 0.1 for p in params]
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    assert optimizer.param_groups[0]["k"] == 1
    for p, e in zip(params, expected):
        assert torch.allclose(p, e)


def test_chunk_safety_by_step_owner():
    module = _module()
    p = [torch.nn.Parameter(torch.randn(2))]
    assert module._chunk_safe(torch.optim.AdamW(p))
    assert not module._chunk_safe(_GroupCountingSGD(p, lr = 0.1))


def test_unstepped_chunk_size_counts_amsgrad_moment():
    module = _module()
    n = 1000
    params = [torch.nn.Parameter(torch.randn(n)) for _ in range(2)]
    # 8 bytes per fp32 parameter would fit both in 20n; amsgrad holds 12n each.
    plain = module._StateOffloader(torch.optim.AdamW(params), chunk_bytes = 20 * n)
    assert len(plain._chunks(params)) == 1
    amsgrad = module._StateOffloader(torch.optim.AdamW(params, amsgrad = True), chunk_bytes = 20 * n)
    assert len(amsgrad._chunks(params)) == 2


class _Projector:
    __slots__ = ("ortho_matrix", "rank")

    def __init__(self, ortho_matrix):
        self.ortho_matrix = ortho_matrix
        self.rank = 2


def test_nested_state_tensors_are_sized():
    module = _module()
    param = torch.nn.Parameter(torch.randn(4))
    optimizer = torch.optim.AdamW([param])
    optimizer.state[param]["projector"] = _Projector(torch.zeros(1000))
    offloader = module._StateOffloader(optimizer, chunk_bytes = 1)
    assert [name for name, _ in module._tensor_attrs(optimizer.state[param]["projector"])] == [
        "ortho_matrix"
    ]
    other = torch.nn.Parameter(torch.randn(4))
    assert len(offloader._chunks([param, other])) == 2


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_nested_state_object_round_trip():
    module = _module()
    param = torch.nn.Parameter(torch.randn(64, device = "cuda"))
    optimizer = torch.optim.AdamW([param], lr = 1e-3)
    param.grad = torch.ones_like(param)
    optimizer.step()
    ortho = torch.randn(64, 8, device = "cuda")
    projector = _Projector(ortho.clone())
    optimizer.state[param]["projector"] = projector
    offloader = module.offload_optimizer_state(optimizer)
    assert projector.ortho_matrix.device.type == "cpu"
    assert offloader.offloaded_bytes >= ortho.numel() * 4
    offloader.restore()
    assert projector.ortho_matrix.device.type == "cuda"
    assert torch.equal(projector.ortho_matrix, ortho)
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert projector.ortho_matrix.device.type == "cpu"


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_layerwise_optimizer_dict_is_offloaded():
    module = _module()
    LayerWiseDummyOptimizer = pytest.importorskip(
        "transformers.trainer_pt_utils"
    ).LayerWiseDummyOptimizer
    params = [torch.nn.Parameter(torch.randn(256, device = "cuda")) for _ in range(2)]
    optimizer_dict = {p: torch.optim.AdamW([p], lr = 1e-3) for p in params}
    dummy = LayerWiseDummyOptimizer(optimizer_dict = optimizer_dict, lr = 1e-3)
    offloaders = module.offload_optimizer_state(dummy)
    assert offloaders is not None and len(offloaders) == 2
    for p in params:
        p.grad = torch.ones_like(p)
        optimizer_dict[p].step()
    for inner in optimizer_dict.values():
        assert _state_devices(inner) == {"cpu"}


def _hooked(module, optimizer, chunk_bytes):
    offloader = module._StateOffloader(optimizer, chunk_bytes = chunk_bytes)
    optimizer.register_step_pre_hook(offloader._pre_step)
    optimizer.register_step_post_hook(offloader._post_step)
    return offloader


def test_chunked_step_synchronizes_chunk_copies(monkeypatch):
    module = _module()
    params = [torch.nn.Parameter(torch.randn(16)) for _ in range(3)]
    offloader = _hooked(module, torch.optim.AdamW(params), chunk_bytes = 1)
    offload = offloader.offload
    # CPU tensors are never copied, so stand in for the device a chunk copy was issued on.
    monkeypatch.setattr(
        offloader, "offload", lambda params = None: offload(params) | ({"dev"} if params else set())
    )
    synced = []
    monkeypatch.setattr(module, "_synchronize", synced.append)
    for p in params:
        p.grad = torch.ones_like(p)
    offloader.optimizer.step()
    assert synced == ["dev"]


def test_unstepped_chunk_size_counts_centered_rmsprop_momentum():
    module = _module()
    n = 1000
    params = [torch.nn.Parameter(torch.randn(n)) for _ in range(2)]
    plain = module._StateOffloader(torch.optim.RMSprop(params, centered = True), chunk_bytes = 20 * n)
    assert len(plain._chunks(params)) == 1
    both = torch.optim.RMSprop(params, centered = True, momentum = 0.9)
    assert len(module._StateOffloader(both, chunk_bytes = 20 * n)._chunks(params)) == 2


def test_single_chunk_restores_only_active_state(monkeypatch):
    module = _module()
    active, idle = torch.nn.Parameter(torch.randn(16)), torch.nn.Parameter(torch.randn(16))
    offloader = _hooked(module, torch.optim.AdamW([active, idle]), chunk_bytes = None)
    restored = []
    monkeypatch.setattr(offloader, "restore", lambda params = None: restored.append(params))
    active.grad = torch.ones_like(active)
    offloader.optimizer.step()
    assert restored == [[active]]


def test_chunked_step_runs_external_hooks_once():
    module = _module()
    params = [torch.nn.Parameter(torch.randn(16)) for _ in range(3)]
    optimizer = torch.optim.AdamW(params, lr = 0.1)
    _hooked(module, optimizer, chunk_bytes = 1)
    calls = []
    optimizer.register_step_pre_hook(lambda *a: calls.append("pre"))
    optimizer.register_step_post_hook(lambda *a: calls.append("post"))
    reference = torch.optim.AdamW([torch.nn.Parameter(p.detach().clone()) for p in params], lr = 0.1)
    for p, q in zip(params, reference.param_groups[0]["params"]):
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
    optimizer.step()
    reference.step()
    assert calls == ["pre", "post"]
    for p, q in zip(params, reference.param_groups[0]["params"]):
        assert torch.equal(p, q)


def test_list_state_tensors_are_sized():
    module = _module()
    params = [torch.nn.Parameter(torch.randn(4)) for _ in range(2)]
    optimizer = torch.optim.AdamW(params)
    optimizer.state[params[0]]["old_dirs"] = [torch.zeros(1000)]
    assert len(module._StateOffloader(optimizer, chunk_bytes = 100)._chunks(params)) == 2


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_lbfgs_history_is_offloaded():
    module = _module()
    torch.manual_seed(0)
    target = torch.randn(64, device = "cuda")

    def run(offload):
        p = torch.nn.Parameter(torch.zeros(64, device = "cuda"))
        optimizer = torch.optim.LBFGS([p], lr = 0.5, history_size = 4, max_iter = 3)
        if offload:
            module.offload_optimizer_state(optimizer)

        def closure():
            optimizer.zero_grad()
            loss = ((p - target) ** 2).sum()
            loss.backward()
            return loss

        for _ in range(3):
            optimizer.step(closure)
        return p.detach(), optimizer.state[p]

    base, _ = run(False)
    head, state = run(True)
    assert torch.allclose(base, head)
    assert state["old_dirs"] and all(
        t.device.type == "cpu" for t in state["old_dirs"] + state["ro"]
    )


def test_chunked_step_runs_after_later_pre_hooks():
    module = _module()
    params = [torch.nn.Parameter(torch.randn(16)) for _ in range(3)]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]

    def halve_and_fill(optimizer, args, kwargs):
        # Clips existing gradients and creates the missing one, as a later library hook might.
        for group in optimizer.param_groups:
            group["lr"] = 0.05
            for p in group["params"]:
                p.grad = torch.full_like(p, 0.5) if p.grad is None else p.grad * 0.5

    optimizer = torch.optim.AdamW(params, lr = 0.1)
    offloader = _hooked(module, optimizer, chunk_bytes = 1)
    optimizer.register_step_pre_hook(halve_and_fill)
    plain = torch.optim.AdamW(reference, lr = 0.1)
    plain.register_step_pre_hook(halve_and_fill)
    for p, q in zip(params[:2], reference[:2]):
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
    optimizer.step()
    plain.step()
    for p, q in zip(params, reference):
        assert torch.equal(p, q)
        assert torch.equal(optimizer.state[p]["exp_avg"], plain.state[q]["exp_avg"])
    assert offloader.hidden is None and offloader.steps == 1


def test_positional_none_closure_still_chunks():
    # accelerate calls optimizer.step(closure) with closure = None.
    module = _module()
    params = [torch.nn.Parameter(torch.randn(16)) for _ in range(3)]
    reference = torch.optim.AdamW([torch.nn.Parameter(p.detach().clone()) for p in params], lr = 0.1)
    optimizer = torch.optim.AdamW(params, lr = 0.1)
    chunked = []
    offloader = _hooked(module, optimizer, chunk_bytes = 1)
    restore = offloader.restore
    offloader.restore = lambda params = None: (chunked.append(params), restore(params))
    for p, q in zip(params, reference.param_groups[0]["params"]):
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
    assert optimizer.step(None) is None
    reference.step()
    assert chunked == [[p] for p in params]
    for p, q in zip(params, reference.param_groups[0]["params"]):
        assert torch.equal(p, q)


def _big_params(n = 4, numel = 1 << 20):
    return [torch.nn.Parameter(torch.randn(numel, device = "cuda")) for _ in range(n)]


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize("fused", [False, True])
def test_load_state_dict_keeps_state_on_host(fused):
    module = _module()
    params = _big_params()
    source = torch.optim.AdamW(params, lr = 1e-3, fused = fused)
    for p in params:
        p.grad = torch.ones_like(p)
    source.step()
    saved = {
        "state": {
            k: {n: t.cpu().clone() for n, t in v.items()}
            for k, v in source.state_dict()["state"].items()
        },
        "param_groups": source.state_dict()["param_groups"],
    }
    clones = [torch.nn.Parameter(p.detach().clone()) for p in params]
    optimizer = torch.optim.AdamW(clones, lr = 1e-3, fused = fused)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 9 << 20)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.load_state_dict(saved)
    torch.cuda.synchronize()
    # All four parameters' moments would be 32 MiB on device.
    assert torch.cuda.max_memory_allocated() - base < 1 << 20
    assert _state_devices(optimizer) == {"cpu"}
    # Fused steps keep "step" on the parameter's device, the others on the host.
    step_device = offloader.offloaded.get((clones[0], "step"))
    assert (step_device.type if step_device is not None else "cpu") == ("cuda" if fused else "cpu")
    for p, q in zip(params, clones):
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
    source.step()
    optimizer.step()
    for p, q in zip(params, clones):
        assert torch.equal(p, q)
    assert _state_devices(optimizer) == {"cpu"}


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_resumed_bitsandbytes_first_step_stays_chunked():
    bnb = pytest.importorskip("bitsandbytes")
    module = _module()
    params = _big_params()
    source = bnb.optim.AdamW8bit(params, lr = 1e-3)
    for p in params:
        p.grad = torch.ones_like(p)
    source.step()
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location = "cpu", weights_only = False)
    clones = [torch.nn.Parameter(p.detach().clone()) for p in params]
    optimizer = bnb.optim.AdamW8bit(clones, lr = 1e-3)
    module.offload_optimizer_state(optimizer, chunk_bytes = 3 << 20)
    optimizer.load_state_dict(saved)
    for p, q in zip(params, clones):
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    peak = torch.cuda.max_memory_allocated() - base
    source.step()
    # One parameter's 8-bit moments (about 2 MiB) at a time, not all four (about 8 MiB).
    assert peak < 5 << 20
    for p, q in zip(params, clones):
        assert torch.equal(p, q)
    assert _state_devices(optimizer) == {"cpu"}


def test_unstepped_chunk_size_counts_the_ademamix_double_moment():
    module = _module()
    n = 1000
    params = [torch.nn.Parameter(torch.randn(n)) for _ in range(2)]

    class AdEMAMix(torch.optim.AdamW):
        pass

    plain = module._StateOffloader(torch.optim.AdamW(params), chunk_bytes = 20 * n)
    assert len(plain._chunks(params)) == 1
    ademamix = module._StateOffloader(AdEMAMix(params), chunk_bytes = 20 * n)
    assert len(ademamix._chunks(params)) == 2


def test_mps_synchronize_takes_no_device(monkeypatch):
    module = _module()
    calls = []
    monkeypatch.setattr(
        torch, "mps", types.SimpleNamespace(synchronize = lambda *a: calls.append(a)), raising = False
    )
    module._synchronize(torch.device("mps"))
    assert calls == [()]


def _q_galore(monkeypatch):
    # Loaded by path under its real module name (so _chunk_safe sees unsloth.optimizers.*),
    # without importing the full unsloth package.
    pytest.importorskip("bitsandbytes")
    for name, path in (
        ("unsloth", [os.path.join(_ROOT, "unsloth")]),
        ("unsloth.optimizers", [os.path.join(_ROOT, "unsloth", "optimizers")]),
    ):
        package = types.ModuleType(name)
        package.__path__ = path
        monkeypatch.setitem(sys.modules, name, package)
    device_type = types.ModuleType("unsloth_zoo.device_type")
    device_type.device_synchronize = torch.cuda.synchronize
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.device_type", device_type)
    for name in ("q_galore_projector", "q_galore_adamw"):
        full = "unsloth.optimizers." + name
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(_ROOT, "unsloth", "optimizers", name + ".py")
        )
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, loaded)
        spec.loader.exec_module(loaded)
    return loaded


def test_weight_quant_q_galore_steps_unchunked():
    module = _module()
    p = [torch.nn.Parameter(torch.randn(2))]

    class QGaLoreAdamW8bit(torch.optim.AdamW):
        pass

    QGaLoreAdamW8bit.__module__ = "unsloth.optimizers.q_galore_adamw"
    assert module._chunk_safe(QGaLoreAdamW8bit([{"params": p, "weight_quant": False}]))
    assert not module._chunk_safe(QGaLoreAdamW8bit([{"params": p, "weight_quant": True}]))


def test_weight_quant_q_galore_restores_only_active_state(monkeypatch):
    module = _module()

    class QGaLoreAdamW8bit(torch.optim.AdamW):
        pass

    QGaLoreAdamW8bit.__module__ = "unsloth.optimizers.q_galore_adamw"
    active, idle = torch.nn.Parameter(torch.randn(16)), torch.nn.Parameter(torch.randn(16))
    optimizer = QGaLoreAdamW8bit([{"params": [active, idle], "weight_quant": True}])
    offloader = _hooked(module, optimizer, chunk_bytes = None)
    restored = []
    monkeypatch.setattr(offloader, "restore", lambda params = None: restored.append(params))
    active.grad = torch.ones_like(active)
    optimizer.step()
    # A closure still restores everything, as does a class that may touch gradless parameters.
    optimizer.step(lambda: None)
    class Custom(torch.optim.SGD):
        def step(self, closure = None):
            return None

    other = Custom([{"params": [active, idle], "weight_quant": True}], lr = 0.1)
    _hooked(module, other, chunk_bytes = None).restore = lambda params = None: restored.append(params)
    other.step()
    assert restored == [[active], None, None]


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_q_galore_weight_quant_inactive_state_stays_on_host(monkeypatch):
    module = _module()
    q_galore = _q_galore(monkeypatch)
    torch.manual_seed(17)
    params = [torch.nn.Parameter(torch.randn(256, 256, device = "cuda")) for _ in range(3)]
    groups = [{"params": params, "rank": 16, "weight_quant": True, "stochastic_round": False}]
    optimizer = q_galore.QGaLoreAdamW8bit(groups, lr = 1e-3)
    q_galore.QGaLoreAdamW8bit.init_weight_quantization(torch.nn.ParameterList(params), groups)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 256 << 10)
    for step in range(2):
        for p in params:
            if step:
                q_galore._materialize_weight(p)
            p.grad = torch.full_like(p, 0.01)
        optimizer.step()
    # Only params[0] is active; the others' state must not visit the device during the step.
    seen = []
    restore = offloader.restore

    def _spy(params = None):
        restore(params)
        seen.append(_state_devices_of(optimizer))

    monkeypatch.setattr(offloader, "restore", _spy)
    q_galore._materialize_weight(params[0])
    params[0].grad = torch.full_like(params[0], 0.01)
    for p in params[1:]:
        p.grad = None
    optimizer.step()
    assert seen and all(devices[1:] == [{"cpu"}, {"cpu"}] for devices in seen)
    assert seen[0][0] == {"cuda"}
    assert _state_devices(optimizer) == {"cpu"}


def _state_devices_of(optimizer):
    groups = [p for group in optimizer.param_groups for p in group["params"]]
    return [
        {
            v.device.type
            for v in optimizer.state.get(p, {}).values()
            if isinstance(v, torch.Tensor) and v.numel() > 1
        }
        for p in groups
    ]


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize("weight_quant", [False, True])
def test_q_galore_multi_chunk_offload(monkeypatch, weight_quant):
    module = _module()
    q_galore = _q_galore(monkeypatch)

    def run(offload):
        torch.manual_seed(17)
        params = [torch.nn.Parameter(torch.randn(256, 256, device = "cuda")) for _ in range(2)]
        groups = [
            {
                "params": params,
                "rank": 16,
                "weight_quant": weight_quant,
                "stochastic_round": False,
            }
        ]
        optimizer = q_galore.QGaLoreAdamW8bit(groups, lr = 1e-3)
        if weight_quant:
            q_galore.QGaLoreAdamW8bit.init_weight_quantization(
                torch.nn.ParameterList(params), groups
            )
        if offload:
            module.offload_optimizer_state(optimizer, chunk_bytes = 256 << 10)
        for step in range(2):
            for p in params:
                if weight_quant and step:
                    q_galore._materialize_weight(p)
                p.grad = torch.full_like(p, 0.01 * (step + 1))
            optimizer.step()
        return params, optimizer

    base, _ = run(False)
    params, optimizer = run(True)
    for p, q in zip(params, base):
        if weight_quant:
            # p itself is a one element placeholder; the INT8 copy holds the weight.
            assert torch.equal(p._q_data, q._q_data)
            assert torch.equal(p._q_scales, q._q_scales)
        else:
            assert torch.equal(p, q)
        assert torch.equal(p.grad, q.grad)
    assert _state_devices(optimizer) == {"cpu"}
