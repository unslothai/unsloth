# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT: part of the state stays on the GPU, same updates as streaming it."""

import importlib.util
import io
import os

import pytest
import torch
from real_accelerator import has_real_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "optimizers", "state_offload.py")
_SR_PATH = os.path.join(_ROOT, "unsloth", "optimizers", "bf16_stochastic_rounding.py")

needs_cuda = pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resident_env_parsing(monkeypatch):
    module = _load(_PATH, "_unsloth_state_offload_resident_env")
    for value, expected in (
        (None, None),
        ("0", None),
        ("off", None),
        ("1", "auto"),
        ("auto", "auto"),
        ("0.5", 1 << 29),
        ("2", 2 << 30),
        ("-1", None),
        ("junk", None),
    ):
        if value is None:
            monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", value)
        assert module._resident_setting() == expected, value


def _all_state(optimizer):
    out = []
    for state in optimizer.state.values():
        for key, value in sorted(state.items(), key = lambda kv: kv[0]):
            out.append(value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value)
    return out


def _model(dtype):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Embedding(4096, 256),  # the largest weight: 1M elements
        torch.nn.Linear(256, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 256),
        torch.nn.Linear(256, 512),
    ).to("cuda", dtype)


def _train(
    monkeypatch,
    resident,
    name = "adamw_fused",
    dtype = torch.float32,
    steps = 6,
    offload = True,
    model = None,
    optimizer = None,
):
    if resident is None:
        monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", resident)
    module = _load(_PATH, "_unsloth_state_offload_resident")
    model = model if model is not None else _model(dtype)
    if optimizer is None:
        if name == "adamw_fused":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr = 1e-3, weight_decay = 0.01, fused = True
            )
        elif name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.01)
        else:
            bnb = pytest.importorskip("bitsandbytes")
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr = 1e-3, min_8bit_size = 1024)
    offloader = (
        module.offload_optimizer_state(optimizer, chunk_bytes = 600 << 10) if offload else None
    )
    tokens = torch.randint(
        0, 4096, (8, 32), device = "cuda", generator = torch.Generator("cuda").manual_seed(1)
    )
    losses = []
    for step in range(steps):
        logits = model((tokens + step) % 4096).float()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 512), (tokens % 512).view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
        losses.append(loss.item())
    torch.cuda.synchronize()
    return model, optimizer, offloader, losses


def _assert_same(a, b):
    assert a[3] == b[3]
    for x, y in zip(a[0].parameters(), b[0].parameters()):
        assert torch.equal(x, y)
    sa, sb = _all_state(a[1]), _all_state(b[1])
    assert len(sa) == len(sb)
    for x, y in zip(sa, sb):
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y


def _devices(optimizer, offloader):
    on_gpu, on_host = set(), set()
    for p, state in optimizer.state.items():
        kinds = {
            v.device.type for k, v in state.items() if isinstance(v, torch.Tensor) and v.numel() > 1
        }
        (on_gpu if kinds == {"cuda"} else on_host).add(p)
    return on_gpu, on_host


@needs_cuda
@pytest.mark.parametrize("name", ["adamw_fused", "adamw", "adamw_8bit"])
def test_everything_resident_matches_streaming_and_no_offload(monkeypatch, name):
    streamed = _train(monkeypatch, None, name = name)
    resident = _train(monkeypatch, "auto", name = name)
    plain = _train(monkeypatch, None, name = name, offload = False)
    _assert_same(streamed, resident)
    _assert_same(plain, resident)
    off = resident[2]
    on_gpu, on_host = _devices(resident[1], off)
    assert on_gpu and not on_host
    assert off.resident and off.resident_bytes > 0 and off.offloaded_bytes == 0
    assert streamed[2].resident_bytes == 0 and not streamed[2].resident


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_budget_keeps_the_largest_weights_and_streams_the_rest(monkeypatch, dtype):
    streamed = _train(monkeypatch, None, dtype = dtype)
    # The embedding's two moments (1M elements each) and half a MiB: the 256 x 1024 Linear no longer fits.
    size = torch.empty((), dtype = dtype).element_size()
    budget = _train(monkeypatch, str((2 * size + 0.5) / 1024), dtype = dtype)
    _assert_same(streamed, budget)
    model, optimizer, off, _ = budget
    embedding = model[0].weight
    on_gpu, on_host = _devices(optimizer, off)
    assert embedding in on_gpu and embedding in off.resident
    assert on_host, "the rest is still streamed"
    assert model[1].weight in on_host
    # The streamed rest still overlaps its copies.
    assert off.overlapped_steps > 0
    assert off.offloaded_bytes > 0 and off.resident_bytes > 0


@needs_cuda
def test_resume_keeps_resident_weights_on_the_gpu(monkeypatch):
    model, optimizer, off, _ = _train(monkeypatch, "auto")
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location = "cpu", weights_only = False)
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    def resume(resident):
        m, o, offloader, _ = _train(monkeypatch, resident, steps = 0)
        m.load_state_dict(reference)
        o.load_state_dict(saved)
        x = torch.randint(0, 4096, (8, 32), device = "cuda")
        for _ in range(3):
            torch.nn.functional.cross_entropy(
                m(x).float().view(-1, 512), (x % 512).view(-1)
            ).backward()
            o.step()
            o.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        return m, o, offloader

    m0, o0, off0 = resume(None)
    m1, o1, off1 = resume("auto")
    for a, b in zip(m0.parameters(), m1.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(_all_state(o0), _all_state(o1)):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    assert off1.resident and off1.offloaded_bytes == 0
    # A second load keeps the decision: the loaded state goes straight back to the GPU.
    o1.load_state_dict(saved)
    on_gpu, on_host = _devices(o1, off1)
    assert on_gpu and not on_host


@needs_cuda
@pytest.mark.parametrize("fused_route", [False, True])
def test_stochastic_rounding_sequence_with_resident_weights(monkeypatch, fused_route):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "1" if fused_route else "0")

    def run(resident):
        if resident is None:
            monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", resident)
        module = _load(_PATH, "_unsloth_state_offload_resident_sr")
        sr = _load(_SR_PATH, "_unsloth_sr_resident")
        generator = torch.Generator(device = "cuda").manual_seed(0)
        shapes = [(2048, 1024), (64,), (2048, 256), (512, 512), (3000, 7)]
        params = [
            torch.nn.Parameter(
                (torch.randn(s, device = "cuda", generator = generator) * 0.05).to(torch.bfloat16)
            )
            for s in shapes
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": params[:3], "weight_decay": 0.01},
                {"params": params[3:], "weight_decay": 0.0},
            ],
            lr = 1e-3,
        )
        offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 1 << 20)
        assert sr.enable_bf16_stochastic_rounding(optimizer, verbose = False) == len(params)
        for _ in range(5):
            for p in params:
                p.grad = (torch.randn(p.shape, device = "cuda", generator = generator) * 1e-3).to(
                    torch.bfloat16
                )
            optimizer.step()
        torch.cuda.synchronize()
        return params, optimizer, offloader

    p0, o0, off0 = run(None)
    # 8.5 MiB: the (2048, 1024) weight's two bf16 moments (8 MiB), then only the tiny ones fit.
    p1, o1, off1 = run(str(8.5 / 1024))
    for a, b in zip(p0, p1):
        assert torch.equal(a, b)
    for a, b in zip(_all_state(o0), _all_state(o1)):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    assert p1[0] in off1.resident and p1[2] not in off1.resident and p1[3] not in off1.resident
    assert o1.state[p1[0]]["exp_avg"].device.type == "cuda"
    assert o1.state[p1[2]]["exp_avg"].device.type == "cpu"
    assert off1.overlapped_steps > 0 and off1.sequence is None


def test_segment_env_parsing(monkeypatch):
    module = _load(_PATH, "_unsloth_state_offload_segment_env")
    for value, expected in (
        (None, 64 << 20),
        ("0", 0),
        ("256", 256 << 20),
        ("0.5", 1 << 19),
        ("x", 0),
    ):
        if value is None:
            monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_SEGMENT_MB", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_SEGMENT_MB", value)
        assert module._segment_bytes() == expected, value


@needs_cuda
@pytest.mark.parametrize("fused_route", [False, True])
def test_stochastic_rounding_segments_match_whole_weights(monkeypatch, fused_route):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", raising = False)
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "1" if fused_route else "0")

    def run(segment_mb):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_SEGMENT_MB", segment_mb)
        module = _load(_PATH, "_unsloth_state_offload_segment")
        sr = _load(_SR_PATH, "_unsloth_sr_segment")
        generator = torch.Generator(device = "cuda").manual_seed(0)
        # 2M elements (8 MiB of bf16 moments) and a size that is not a multiple of the range.
        shapes = [(2048, 1024), (64,), (1000, 777), (500, 500)]
        params = [
            torch.nn.Parameter(
                (torch.randn(s, device = "cuda", generator = generator) * 0.05).to(torch.bfloat16)
            )
            for s in shapes
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": params[:3], "weight_decay": 0.01},
                {"params": params[3:], "weight_decay": 0.0},
            ],
            lr = 1e-3,
            fused = True,
        )
        offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 1 << 20)
        assert sr.enable_bf16_stochastic_rounding(optimizer, verbose = False) == len(params)
        peaks = []
        for _ in range(5):
            for p in params:
                p.grad = (torch.randn(p.shape, device = "cuda", generator = generator) * 1e-3).to(
                    torch.bfloat16
                )
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            optimizer.step()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - base)
        return params, optimizer, offloader, peaks

    p0, o0, off0, peaks0 = run("0")
    p1, o1, off1, peaks1 = run("1")
    for a, b in zip(p0, p1):
        assert torch.equal(a, b)
    s0, s1 = _all_state(o0), _all_state(o1)
    assert len(s0) == len(s1)
    for a, b in zip(s0, s1):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    for state in o1.state.values():
        assert state["exp_avg"].device.type == "cpu" and state["exp_avg"].shape != (0,)
    segmented = o1.__dict__.get("_unsloth_sr_segmented_steps", 0)
    if fused_route and not getattr(torch.version, "hip", None):
        # Two weights over 1 MiB of state, on each of the 4 overlapped steps.
        assert segmented == 2 * 4
        # The ring holds four 1 MiB ranges instead of the whole 8 MiB weight.
        assert max(peaks1[1:]) <= max(peaks0[1:]) - (3 << 20)
    else:
        assert segmented == 0
    assert off1.sequence is None


@needs_cuda
def test_resident_weights_return_their_pinned_host_memory(monkeypatch):
    psutil = pytest.importorskip("psutil")
    module = _load(_PATH, "_unsloth_state_offload_resident_rss")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", "auto")
    params = [torch.nn.Parameter(torch.randn(16 << 20, device = "cuda")) for _ in range(4)]
    optimizer = torch.optim.AdamW(params, lr = 1e-3, fused = True)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 600 << 20)
    process = psutil.Process()
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    torch.cuda.synchronize()
    held = process.memory_info().rss
    assert offloader.offloaded_bytes >= 4 * 2 * (64 << 20)
    optimizer.step()  # the decision: everything fits on the GPU
    torch.cuda.synchronize()
    assert offloader.offloaded_bytes == 0 and len(offloader.resident) == 4
    # 512 MiB of pinned moments were freed; most of it leaves the process.
    assert process.memory_info().rss < held - (256 << 20)


@needs_cuda
def test_error_inside_a_segment_leaves_whole_state_on_host(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", raising = False)
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "1")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_SEGMENT_MB", "1")
    module = _load(_PATH, "_unsloth_state_offload_segment_error")
    sr = _load(_SR_PATH, "_unsloth_sr_segment_error")
    p = torch.nn.Parameter((torch.randn(2048, 1024, device = "cuda") * 0.05).to(torch.bfloat16))
    # A second weight makes the group a per-weight sequence (a single weight steps on its own).
    q = torch.nn.Parameter(torch.zeros(64, device = "cuda", dtype = torch.bfloat16))
    optimizer = torch.optim.AdamW([p, q], lr = 1e-3, fused = True)
    offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 1 << 20)
    assert sr.enable_bf16_stochastic_rounding(optimizer, verbose = False) == 2
    for _ in range(2):
        p.grad = torch.full_like(p, 1e-3)
        q.grad = torch.full_like(q, 1e-3)
        optimizer.step()
    torch.cuda.synchronize()
    before = {k: v.clone() for k, v in optimizer.state[p].items()}
    calls = []
    original = sr._fused_adamw_segment

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(sr, "_fused_adamw_segment", failing)
    p.grad = torch.full_like(p, 1e-3)
    q.grad = torch.full_like(q, 1e-3)
    with pytest.raises(RuntimeError, match = "boom"):
        optimizer.step()
    torch.cuda.synchronize()
    assert offloader.sequence is None and len(calls) == 3
    for key, value in optimizer.state[p].items():
        assert value.device.type == "cpu" and value.shape == before[key].shape, key
    # The two ranges stepped before the error are written back; the rest is unchanged.
    moved = (optimizer.state[p]["exp_avg"] != before["exp_avg"]).view(-1)
    assert moved[: 2 * 262144].any() and not moved[2 * 262144 :].any()


@needs_cuda
def test_sequence_ending_inside_a_ranged_weight_still_matches(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", raising = False)
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "1")

    def run(cut):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_SEGMENT_MB", "1")
        module = _load(_PATH, "_unsloth_state_offload_segment_cut")
        sr = _load(_SR_PATH, "_unsloth_sr_segment_cut")
        generator = torch.Generator(device = "cuda").manual_seed(0)
        params = [
            torch.nn.Parameter(
                (torch.randn(s, device = "cuda", generator = generator) * 0.05).to(torch.bfloat16)
            )
            for s in ((2048, 1024), (64,))
        ]
        optimizer = torch.optim.AdamW(params, lr = 1e-3, fused = True)
        offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 1 << 20)
        assert sr.enable_bf16_stochastic_rounding(optimizer, verbose = False) == 2
        calls = []
        original = sr._fused_adamw_segment

        def spy(*args, **kwargs):
            calls.append(1)
            result = original(*args, **kwargs)
            if cut and len(calls) == 10 and offloader.sequence is not None:
                # The sequence ends between two ranges of one weight (as an unexpected call ends it); the next
                # ranges then step on the whole moments, restored the serial way.
                offloader._sequence_end(offloader.sequence)
            return result

        monkeypatch.setattr(sr, "_fused_adamw_segment", spy)
        for _ in range(3):
            for p in params:
                p.grad = (torch.randn(p.shape, device = "cuda", generator = generator) * 1e-3).to(
                    torch.bfloat16
                )
            optimizer.step()
        torch.cuda.synchronize()
        return params, optimizer, len(calls)

    p0, o0, n0 = run(False)
    p1, o1, n1 = run(True)
    assert n0 == n1 == 16  # 8 ranges on each of steps 2 and 3
    for a, b in zip(p0, p1):
        assert torch.equal(a, b)
    for a, b in zip(_all_state(o0), _all_state(o1)):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    for state in o1.state.values():
        for value in state.values():
            assert value.device.type == "cpu"


@needs_cuda
def test_ranged_weight_whose_gradient_a_pre_hook_drops_is_skipped(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_RESIDENT", raising = False)
    monkeypatch.setenv("UNSLOTH_BF16_SR_FUSED", "1")

    def run(segment_mb):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_OPTIMIZER_STATE_SEGMENT_MB", segment_mb)
        module = _load(_PATH, "_unsloth_state_offload_segment_drop")
        sr = _load(_SR_PATH, "_unsloth_sr_segment_drop")
        generator = torch.Generator(device = "cuda").manual_seed(0)
        params = [
            torch.nn.Parameter(
                (torch.randn(s, device = "cuda", generator = generator) * 0.05).to(torch.bfloat16)
            )
            for s in ((2048, 1024), (64,), (512, 300))
        ]
        optimizer = torch.optim.AdamW(params, lr = 1e-3, fused = True)
        module.offload_optimizer_state(optimizer, chunk_bytes = 1 << 20)
        drop = {"on": False}

        def idle(opt, args, kwargs):
            # As the idle-expert hook does: a weight the step should leave alone loses its gradient.
            if drop["on"]:
                params[0].grad = None

        optimizer.register_step_pre_hook(idle)
        assert sr.enable_bf16_stochastic_rounding(optimizer, verbose = False) == 3
        for step in range(4):
            drop["on"] = step == 2
            for p in params:
                p.grad = (torch.randn(p.shape, device = "cuda", generator = generator) * 1e-3).to(
                    torch.bfloat16
                )
            optimizer.step()
        torch.cuda.synchronize()
        return params, optimizer, optimizer.__dict__.get("_unsloth_sr_segmented_steps", 0)

    p0, o0, n0 = run("0")
    p1, o1, n1 = run("1")
    assert n0 == 0 and n1 == 2  # steps 2 and 4 range the large weight; step 3 drops it
    for a, b in zip(p0, p1):
        assert torch.equal(a, b)
    for a, b in zip(_all_state(o0), _all_state(o1)):
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
