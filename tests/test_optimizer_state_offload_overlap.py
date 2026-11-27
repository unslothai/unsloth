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

"""Overlapped chunk copies of UNSLOTH_OFFLOAD_OPTIMIZER_STATE: same updates as the serial chunks."""

import importlib.util
import io
import os

import pytest
import torch
from real_accelerator import has_real_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "optimizers", "state_offload.py")

needs_cuda = pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
# ROCm keeps the serial copies: same results, nothing overlapped.
_ENGAGES = not getattr(torch.version, "hip", None)


def _n(count):
    return count if _ENGAGES else 0


def _module():
    spec = importlib.util.spec_from_file_location("_unsloth_state_offload_overlap", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ReplacingMomentum(torch.optim.Optimizer):
    # Rebinds its state tensor every step instead of updating it in place.
    def __init__(
        self,
        params,
        lr = 1e-2,
    ):
        super().__init__(params, dict(lr = lr))

    @torch.no_grad()
    def step(self, closure = None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                state["m"] = state["m"] * 0.9 + p.grad
                p.add_(state["m"], alpha = -group["lr"])


# Chunk safety is decided by the class that defines step.
_ReplacingMomentum.__module__ = "torch.optim._unsloth_test"


def _factory(name):
    if name == "adamw":
        return lambda p: torch.optim.AdamW(p, lr = 1e-3, weight_decay = 0.01)
    if name == "adamw_foreach":
        return lambda p: torch.optim.AdamW(p, lr = 1e-3, foreach = True)
    if name == "adamw_fused":
        return lambda p: torch.optim.AdamW(p, lr = 1e-3, fused = True)
    if name == "adam_amsgrad":
        return lambda p: torch.optim.Adam(p, lr = 1e-3, amsgrad = True)
    if name == "replacing":
        return lambda p: _ReplacingMomentum(p)
    bnb = pytest.importorskip("bitsandbytes")
    if name == "adamw_8bit":
        return lambda p: bnb.optim.AdamW8bit(p, lr = 1e-3, min_8bit_size = 1024)
    if name == "adamw_32bit_bnb":
        return lambda p: bnb.optim.AdamW(p, lr = 1e-3, optim_bits = 32)
    raise KeyError(name)


def _train(
    monkeypatch,
    name,
    overlap,
    dtype = torch.float32,
    steps = 5,
    chunk_bytes = 1,
    pin_memory = True,
):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1" if overlap else "0")
    module = _module()
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Embedding(512, 256),
        torch.nn.Linear(256, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 256),
        torch.nn.Linear(256, 512),
    ).to("cuda", dtype)
    optimizer = _factory(name)(model.parameters())
    offloader = module.offload_optimizer_state(
        optimizer, chunk_bytes = chunk_bytes, pin_memory = pin_memory
    )
    assert offloader is not None
    tokens = torch.randint(0, 512, (8, 32), device = "cuda")
    losses = []
    for step in range(steps):
        logits = model((tokens + step) % 512).float()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 512), tokens.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
        losses.append(loss.item())
    return model, optimizer, offloader, losses


def _host_state(optimizer):
    out = []
    for state in optimizer.state.values():
        for key, value in sorted(state.items(), key = lambda kv: kv[0]):
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu", key
                out.append(value.clone())
            else:
                out.append(value)
    return out


def _assert_same(a, b):
    model_a, opt_a, _, losses_a = a
    model_b, opt_b, _, losses_b = b
    assert losses_a == losses_b
    for x, y in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(x, y)
    sa, sb = _host_state(opt_a), _host_state(opt_b)
    assert len(sa) == len(sb)
    for x, y in zip(sa, sb):
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and torch.equal(x, y)
        else:
            assert x == y


@needs_cuda
@pytest.mark.parametrize(
    "name",
    ["adamw", "adamw_foreach", "adamw_fused", "adam_amsgrad", "adamw_8bit", "adamw_32bit_bnb"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_bytes", [1, 600 << 10])
def test_overlap_matches_serial_chunks(monkeypatch, name, dtype, chunk_bytes):
    serial = _train(monkeypatch, name, overlap = False, dtype = dtype, chunk_bytes = chunk_bytes)
    overlap = _train(monkeypatch, name, overlap = True, dtype = dtype, chunk_bytes = chunk_bytes)
    _assert_same(serial, overlap)
    assert serial[2].overlapped_steps == 0
    # Step 1 creates the state on the device through the serial path; every later step overlaps.
    assert overlap[2].overlapped_steps == _n(4)
    assert overlap[2].steps == 5


@needs_cuda
def test_state_the_step_replaces_is_still_written_back(monkeypatch):
    serial = _train(monkeypatch, "replacing", overlap = False)
    overlap = _train(monkeypatch, "replacing", overlap = True)
    _assert_same(serial, overlap)
    assert overlap[2].overlapped_steps == _n(4)


@needs_cuda
def test_overlap_staging_is_not_larger_than_a_serial_chunk(monkeypatch):
    peaks = {}
    for overlap in (False, True):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1" if overlap else "0")
        module = _module()
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(1 << 20, device = "cuda")) for _ in range(8)]
        optimizer = torch.optim.AdamW(params, lr = 1e-3, fused = True)
        offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 17 << 20)
        for _ in range(2):
            for p in params:
                p.grad = torch.ones_like(p)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            optimizer.step()
            torch.cuda.synchronize()
            peaks[overlap] = torch.cuda.max_memory_allocated() - base
        assert offloader.overlapped_steps == (_n(1) if overlap else 0)
        del optimizer, params, offloader
    # Serial: two parameters (2 x 8 MiB of moments) per chunk. The ring holds no more than that.
    assert peaks[True] <= peaks[False]


@needs_cuda
def test_resume_from_checkpoint_then_overlap(monkeypatch):
    model, optimizer, offloader, _ = _train(monkeypatch, "adamw_fused", overlap = True)
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location = "cpu", weights_only = False)
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    def resume(overlap):
        m, o, off, _ = _train(monkeypatch, "adamw_fused", overlap = overlap, steps = 0)
        m.load_state_dict(reference)
        o.load_state_dict(saved)
        x = torch.randint(0, 512, (8, 32), device = "cuda")
        for _ in range(2):
            torch.nn.functional.cross_entropy(m(x).view(-1, 512), x.view(-1)).backward()
            o.step()
            o.zero_grad(set_to_none = True)
        return m, o, off

    m0, o0, off0 = resume(False)
    m1, o1, off1 = resume(True)
    for a, b in zip(m0.parameters(), m1.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(_host_state(o0), _host_state(o1)):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    # Loaded state is already host resident, so even the first resumed step overlaps.
    assert off1.overlapped_steps == _n(2) and off0.overlapped_steps == 0


@needs_cuda
def test_pageable_host_memory_stays_serial(monkeypatch):
    serial = _train(monkeypatch, "adamw_fused", overlap = False, pin_memory = False)
    overlap = _train(monkeypatch, "adamw_fused", overlap = True, pin_memory = False)
    _assert_same(serial, overlap)
    assert overlap[2].overlapped_steps == 0


@needs_cuda
def test_rocm_stays_serial(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    _, _, offloader, _ = _train(monkeypatch, "adamw_fused", overlap = True)
    assert offloader.overlapped_steps == 0


@needs_cuda
def test_error_inside_an_overlapped_step_leaves_state_on_host(monkeypatch):
    model, optimizer, offloader, _ = _train(monkeypatch, "adamw_fused", overlap = True, steps = 2)
    calls = {"n": 0}
    step = type(optimizer).step

    def failing(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return step(self, *args, **kwargs)

    monkeypatch.setattr(type(optimizer), "step", failing)
    x = torch.randint(0, 512, (8, 32), device = "cuda")
    torch.nn.functional.cross_entropy(model(x).view(-1, 512), x.view(-1)).backward()
    grads = [p.grad for p in model.parameters()]
    with pytest.raises(RuntimeError, match = "boom"):
        offloader._step_chunks(optimizer, (), {})
    monkeypatch.setattr(type(optimizer), "step", step)
    assert all(p.grad is g for p, g in zip(model.parameters(), grads))
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                assert value.device.type == "cpu"
    assert not offloader.inside


def test_cpu_parameters_are_not_offloaded(monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1")
    module = _module()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4))])
    assert module.offload_optimizer_state(optimizer) is None


def test_overlap_env_parsing(monkeypatch):
    module = _module()
    monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", raising = False)
    assert module._overlap_enabled()
    for off in ("0", "false", "No", "off"):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", off)
        assert not module._overlap_enabled()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1")
    assert module._overlap_enabled()


def _large_run(
    monkeypatch,
    overlap,
    delay_cycles = 20_000_000,
):
    # Large copies plus a step that first stalls its stream: a missing wait between copy in, step
    # and copy out would read or write moments mid-copy and change the result.
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1" if overlap else "0")
    module = _module()
    generator = torch.Generator(device = "cuda").manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(1 << 24, device = "cuda", generator = generator))
        for _ in range(6)
    ]
    optimizer = torch.optim.AdamW(params, lr = 1e-2, fused = True)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 300 << 20)
    step = type(optimizer).step

    def slow_step(self, *args, **kwargs):
        torch.cuda._sleep(delay_cycles)
        return step(self, *args, **kwargs)

    monkeypatch.setattr(type(optimizer), "step", slow_step)
    try:
        for i in range(4):
            for j, p in enumerate(params):
                p.grad = torch.randn(p.shape, device = "cuda", generator = generator) * (i + j + 1)
            optimizer.step()
        torch.cuda.synchronize()
    finally:
        monkeypatch.setattr(type(optimizer), "step", step)
    return params, optimizer, offloader


@needs_cuda
def test_large_overlapped_copies_with_a_slow_step_match_serial(monkeypatch):
    p0, o0, off0 = _large_run(monkeypatch, overlap = False)
    p1, o1, off1 = _large_run(monkeypatch, overlap = True)
    assert off0.overlapped_steps == 0 and off1.overlapped_steps == _n(3)
    for a, b in zip(p0, p1):
        assert torch.equal(a, b)
    for a, b in zip(_host_state(o0), _host_state(o1)):
        assert torch.equal(a, b)


def _sr_module():
    path = os.path.join(_ROOT, "unsloth", "optimizers", "bf16_stochastic_rounding.py")
    spec = importlib.util.spec_from_file_location("_unsloth_sr_overlap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sr_run(
    monkeypatch,
    overlap,
    fused_route,
    steps = 4,
    order = None,
):
    # bfloat16 full finetuning: stochastic rounding steps one weight per call under state offload.
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP", "1" if overlap else "0")
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "1" if fused_route else "0")
    module, sr = _module(), _sr_module()
    generator = torch.Generator(device = "cuda").manual_seed(0)
    shapes = [(1024, 512), (64,), (2048, 256), (512, 512), (3000, 7)]
    params = [
        torch.nn.Parameter(
            (torch.randn(s, device = "cuda", generator = generator) * 0.05).to(torch.bfloat16)
        )
        for s in shapes
    ]
    optimizer = torch.optim.AdamW(
        [{"params": params[:3], "weight_decay": 0.01}, {"params": params[3:], "weight_decay": 0.0}],
        lr = 1e-3,
    )
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 1 << 20)
    assert sr.enable_bf16_stochastic_rounding(optimizer, verbose = False) == len(params)
    for _ in range(steps):
        for p in params:
            p.grad = (torch.randn(p.shape, device = "cuda", generator = generator) * 1e-3).to(
                torch.bfloat16
            )
        optimizer.step()
    torch.cuda.synchronize()
    return params, optimizer, offloader


@needs_cuda
@pytest.mark.parametrize("fused_route", [False, True])
def test_stochastic_rounding_per_weight_steps_overlap_and_match(monkeypatch, fused_route):
    p0, o0, off0 = _sr_run(monkeypatch, overlap = False, fused_route = fused_route)
    p1, o1, off1 = _sr_run(monkeypatch, overlap = True, fused_route = fused_route)
    for a, b in zip(p0, p1):
        assert torch.equal(a, b)
    s0, s1 = _host_state(o0), _host_state(o1)
    assert len(s0) == len(s1)
    for a, b in zip(s0, s1):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    for state in o1.state.values():
        assert state["exp_avg"].dtype == torch.bfloat16
    # One sequence per parameter group on each step after the first.
    assert off0.overlapped_steps == 0
    assert off1.overlapped_steps == _n(2 * 3)
    assert off1.sequence is None


@needs_cuda
def test_unexpected_call_inside_a_sequence_falls_back(monkeypatch):
    params, optimizer, offloader, _ = _train(monkeypatch, "adamw_fused", overlap = True, steps = 2)
    weights = [p for p in params.parameters()]
    for p in weights:
        p.grad = torch.ones_like(p)
    reference = {p: {k: v.clone() for k, v in optimizer.state[p].items()} for p in weights}
    groups = optimizer.param_groups
    with offloader.pipelined([[weights[0]], [weights[1]], [weights[2]]]):
        assert (offloader.sequence is not None) == _ENGAGES
        for p in (
            weights[0],
            weights[2],
        ):  # weights[1] skipped: the sequence must stop, not misuse it
            optimizer.param_groups = [dict(groups[0], params = [p])]
            optimizer.step()
        assert offloader.sequence is None
    optimizer.param_groups = groups
    assert offloader.overlapped_steps == _n(1)  # only the plain step 2 of _train
    for p in weights:
        for key, value in optimizer.state[p].items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
    for p in (weights[1],) + tuple(weights[3:]):
        for key, value in optimizer.state[p].items():
            assert torch.equal(value, reference[p][key]), key


@needs_cuda
def test_a_chunked_step_inside_a_sequence_ends_it_first(monkeypatch):
    def run(overlap):
        model, optimizer, offloader, _ = _train(
            monkeypatch, "adamw_fused", overlap = overlap, steps = 2
        )
        weights = list(model.parameters())
        generator = torch.Generator(device = "cuda").manual_seed(1)
        for p in weights:
            p.grad = torch.randn(p.shape, device = "cuda", generator = generator)
        with offloader.pipelined([[weights[0]], [weights[1]], [weights[2]]]):
            optimizer.step()  # every weight, several chunks: not the sequence's first call
            assert offloader.sequence is None
        torch.cuda.synchronize()
        return weights, optimizer

    w0, o0 = run(False)
    w1, o1 = run(True)
    for a, b in zip(w0, w1):
        assert torch.equal(a, b)
    for a, b in zip(_host_state(o0), _host_state(o1)):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
