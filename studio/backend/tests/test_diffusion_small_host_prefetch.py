# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host route speed fixes: the streamed text encoder's prefetching copy and the fused int8 dequant.

The prefetch order / budget / fallback logic runs on CPU with fake groups; the end-to-end check (diffusers leaf-level
group offload + layerwise casting, the small-host encoder setup) needs CUDA and is skipped without it.
"""

from __future__ import annotations

import threading
import time
import types

import pytest
import torch

import core.inference.diffusion_small_host as sh


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(sh.ENCODER_PREFETCH_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN", raising = False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_int8_dequant_is_one_pass_and_bit_identical(dtype):
    from torch.utils._python_dispatch import TorchDispatchMode

    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 512).to(torch.bfloat16)
    # same per-row absmax as quantize_int8_weight_ (this layer is below its size threshold)
    w = lin.weight.detach().float()
    scale = w.abs().amax(dim = 1, keepdim = True).clamp_min(1e-12) / 127.0
    q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
    layer = sh.int8_linear_class()(q, scale.to(dtype), lin.bias.detach().to(dtype), 256, 512)
    x = torch.randn(4, 256, dtype = dtype)

    class Ops(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(
            self,
            func,
            types_,
            args = (),
            kwargs = None,
        ):
            self.ops.append(str(func))
            return func(*args, **(kwargs or {}))

    with Ops() as rec:
        out = layer(x)
    passes = [o for o in rec.ops if "_to_copy" in o or "mul" in o]
    assert passes == ["aten.mul.Tensor"], rec.ops
    want = torch.nn.functional.linear(x, q.to(dtype) * scale.to(dtype), lin.bias.detach().to(dtype))
    assert torch.equal(out, want)


def test_int8_dequant_other_input_dtype_still_casts():
    q = torch.randint(-127, 128, (8, 4), dtype = torch.int8)
    scale = torch.rand(8, 1, dtype = torch.float16)
    layer = sh.int8_linear_class()(q, scale, None, 4, 8)
    x = torch.randn(2, 4, dtype = torch.float32)
    out = layer(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.nn.functional.linear(x, q.float() * scale.float()))


class _FakeGroup:
    def __init__(self, name: str, nbytes: int):
        self.name = name
        self.nbytes = nbytes


class _FakePrefetcher(sh._EncoderPrefetcher):
    """The real ordering / budget / fallback logic with the CUDA copy replaced by a recorded fake."""

    def __init__(
        self,
        groups,
        *,
        fail_at = None,
        delay = 0.0,
    ):
        super().__init__(module = None, groups = groups, device = "cpu")
        self.copies: list = []
        self.attached: list = []
        self.fail_at = fail_at
        self.delay = delay
        self.peak_inflight = 0
        self.lock = threading.Lock()

    def _copy_group(self, group):
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            if (
                self.fail_at is not None
                and group.name == self.fail_at
                and threading.current_thread() is not threading.main_thread()
            ):
                raise RuntimeError("copy failed")
            self.copies.append((group.name, threading.current_thread() is threading.main_thread()))
        with self.cond:
            self.peak_inflight = max(self.peak_inflight, self.inflight + group.nbytes)
        return [group.name], None, group.nbytes

    def _attach(self, group, moved, done):
        self.attached.append(moved[0])

    def _order_after_compute(self):
        return None


def _forward(pf, groups):
    pf.begin()
    try:
        for g in groups:
            pf.onload(g)
    finally:
        pf.end()


def test_prefetch_records_the_order_then_copies_ahead_on_a_worker():
    groups = [_FakeGroup(f"g{i}", 10) for i in range(12)]
    pf = _FakePrefetcher(groups)
    _forward(pf, groups)
    # first forward has no recorded order, so every group copies on the calling thread
    assert (pf.stats["prefetched"], pf.stats["sync"], pf.stats["passes"]) == (0, 12, 1)
    assert all(main for _n, main in pf.copies)
    pf.copies.clear()
    _forward(pf, groups)
    assert pf.stats["prefetched"] == 12 and pf.stats["sync"] == 12
    assert [n for n, _ in pf.copies] == [g.name for g in groups]
    assert not any(main for _n, main in pf.copies)
    assert pf.attached[-12:] == [g.name for g in groups]
    assert pf.worker is None and pf.inflight == 0 and not pf.ready


def test_prefetch_stays_within_the_byte_budget():
    groups = [_FakeGroup(f"g{i}", sh._PREFETCH_MIN_BYTES // 4) for i in range(16)]
    pf = _FakePrefetcher(groups)
    _forward(pf, groups)

    pf.begin()
    try:
        for g in groups:
            time.sleep(0.01)
            pf.onload(g)
    finally:
        pf.end()
    assert pf.stats["prefetched"] == 16
    assert pf.peak_inflight <= pf.budget + sh._PREFETCH_MIN_BYTES // 4


def test_prefetch_one_oversized_group_still_progresses():
    groups = [_FakeGroup("big", sh._PREFETCH_MAX_BYTES * 4), _FakeGroup("small", 1)]
    pf = _FakePrefetcher(groups)
    _forward(pf, groups)
    _forward(pf, groups)
    assert pf.stats["prefetched"] == 2


def test_prefetch_off_order_falls_back_and_rerecords():
    groups = [_FakeGroup(f"g{i}", 10) for i in range(6)]
    pf = _FakePrefetcher(groups)
    _forward(pf, groups)
    swapped = groups[:2] + [groups[3], groups[2]] + groups[4:]
    _forward(pf, swapped)
    # g3 is off the recorded order, so it and everything after copy synchronously
    assert pf.stats["prefetched"] == 2
    assert pf.attached[-6:] == [g.name for g in swapped]
    assert pf.order == [id(g) for g in swapped]
    _forward(pf, swapped)
    assert pf.stats["prefetched"] == 2 + 6


def test_prefetch_worker_error_falls_back_to_synchronous_copies():
    groups = [_FakeGroup(f"g{i}", 10) for i in range(5)]
    pf = _FakePrefetcher(groups, fail_at = "g2")
    _forward(pf, groups)
    _forward(pf, groups)
    assert pf.attached[-5:] == [g.name for g in groups]
    assert pf.stats["prefetched"] == 2 and pf.stats["sync"] == 5 + 3


def test_prefetch_exception_in_forward_stops_the_worker():
    groups = [_FakeGroup(f"g{i}", 10) for i in range(5)]
    pf = _FakePrefetcher(groups, delay = 0.01)
    _forward(pf, groups)
    pf.begin()
    pf.onload(groups[0])
    pf.end()
    assert pf.worker is None and not pf.ready and pf.inflight == 0


def test_install_is_off_on_cpu_and_with_the_kill_switch(monkeypatch):
    m = torch.nn.Linear(4, 4)
    assert sh.install_encoder_prefetch(m, "cpu") == 0
    monkeypatch.setenv(sh.ENCODER_PREFETCH_ENV, "0")
    assert sh.encoder_prefetch_disabled()
    assert sh.install_encoder_prefetch(m, "cuda") == 0


def test_group_offload_installs_prefetch_only_on_small_host_pipes(monkeypatch):
    pytest.importorskip("diffusers")
    import core.inference.diffusion_memory as dm

    calls: list = []
    monkeypatch.setattr(
        sh,
        "install_encoder_prefetch",
        lambda module, device, logger = None: calls.append(module) or 1,
    )

    def _pipe(small_host: bool):
        te = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        tr = torch.nn.Sequential(torch.nn.Linear(8, 8))
        pipe = types.SimpleNamespace(
            transformer = tr, text_encoder = te, components = {"transformer": tr, "text_encoder": te}
        )
        if small_host:
            pipe._unsloth_small_host = {
                "components": {"text_encoder": "memory-mapped, layerwise cast (1 MiB)"}
            }
        return pipe, te

    pipe, te = _pipe(False)
    assert dm._apply_group_offload(pipe, "cpu", None, stream_text_encoders = True)
    assert calls == []
    pipe, te = _pipe(True)
    assert dm._apply_group_offload(pipe, "cpu", None, stream_text_encoders = True)
    assert calls == [te]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_prefetched_encoder_is_bit_identical_to_diffusers_stream(tmp_path):
    from diffusers.hooks import apply_group_offloading

    def build(path = None):
        torch.manual_seed(0)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(256)
                self.wi = torch.nn.Linear(256, 1024, bias = False)
                self.wo = torch.nn.Linear(1024, 256, bias = False)

            def forward(self, x):
                return x + self.wo(torch.relu(self.wi(self.norm(x))))

        class Enc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 256)
                self.blocks = torch.nn.ModuleList([Block() for _ in range(6)])
                self.final = torch.nn.LayerNorm(256)

            def forward(self, ids):
                h = self.embed(ids)
                for b in self.blocks:
                    h = b(h)
                return self.final(h)

        enc = Enc().to(torch.bfloat16).eval()
        sh.prepare_streamed_encoder_(enc, torch.float16)
        if path is not None:
            from safetensors.torch import load_file, save_file

            save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, str(path))
            loaded = load_file(str(path))
            for name, p in enc.named_parameters():
                p.data = loaded[name]
        apply_group_offloading(
            enc,
            onload_device = torch.device("cuda"),
            offload_device = torch.device("cpu"),
            offload_type = "leaf_level",
            use_stream = True,
            non_blocking = True,
            record_stream = True,
            low_cpu_mem_usage = True,
        )
        return enc

    ref_enc = build()
    enc = build(tmp_path / "enc.safetensors")

    n = sh.install_encoder_prefetch(enc, "cuda")
    assert n > 0
    pf = getattr(enc, sh.ENCODER_PREFETCH_ATTR)
    with torch.no_grad():
        for i in range(4):
            ids = torch.randint(0, 1000, (1, 16 + i), device = "cuda")
            want = ref_enc(ids)
            got = enc(ids)
            torch.cuda.synchronize()
            assert torch.equal(got, want)
    assert pf.error is None
    assert pf.stats["prefetched"] == 3 * len(pf.order) and pf.stats["sync"] == len(pf.order)
    assert enc.blocks[0].wi.weight.device.type == "cpu"
    assert enc.blocks[0].wi.weight.dtype == torch.bfloat16


def _mapped_streamed(enc, path):
    """The small-host setup: weights view a safetensors file mapping, diffusers leaf-level stream offload."""
    pytest.importorskip("diffusers")
    from diffusers.hooks import apply_group_offloading
    from safetensors.torch import load_file, save_file

    save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, str(path))
    loaded = load_file(str(path))
    for name, p in enc.named_parameters():
        p.data = loaded[name]
    apply_group_offloading(
        enc,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "leaf_level",
        use_stream = True,
        non_blocking = True,
        record_stream = True,
        low_cpu_mem_usage = True,
    )
    return enc


def _small_linear_stack():
    torch.manual_seed(0)
    return (
        torch.nn.Sequential(*[torch.nn.Linear(64, 64) for _ in range(3)]).to(torch.bfloat16).eval()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_pin_opt_out_keeps_diffusers_onload(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN", "0")
    enc = _mapped_streamed(_small_linear_stack(), tmp_path / "enc.safetensors")
    assert sh.install_encoder_prefetch(enc, "cuda") == 0
    assert getattr(enc, sh.ENCODER_PREFETCH_ATTR, None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_failed_ring_pin_keeps_diffusers_onload(tmp_path, monkeypatch):
    enc = _mapped_streamed(_small_linear_stack(), tmp_path / "enc.safetensors")
    real_empty = torch.empty

    def empty(*a, **k):
        if k.get("pin_memory"):
            raise RuntimeError("CUDA error: out of memory")
        return real_empty(*a, **k)

    monkeypatch.setattr(torch, "empty", empty)
    assert sh.install_encoder_prefetch(enc, "cuda") == 0
    monkeypatch.setattr(torch, "empty", real_empty)
    assert getattr(enc, sh.ENCODER_PREFETCH_ATTR, None) is None
    ref = _small_linear_stack().cuda()
    x = torch.randn(2, 64, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        assert torch.equal(enc(x), ref(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_prefetch_runs_under_inference_mode(tmp_path):
    # Studio renders under inference_mode; the worker thread is outside it and must still fill the ring
    def build():
        torch.manual_seed(0)
        return (
            torch.nn.Sequential(*[torch.nn.Linear(512, 512) for _ in range(6)])
            .to(torch.bfloat16)
            .eval()
        )

    ref = build().cuda()
    enc = _mapped_streamed(build(), tmp_path / "enc.safetensors")
    assert sh.install_encoder_prefetch(enc, "cuda") > 0
    pf = getattr(enc, sh.ENCODER_PREFETCH_ATTR)
    with torch.inference_mode():
        for i in range(4):
            x = torch.randn(2 + i, 512, dtype = torch.bfloat16, device = "cuda")
            assert torch.equal(enc(x), ref(x))
    assert pf.error is None
    assert pf.stats["prefetched"] == 3 * len(pf.order)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("drop", ["exception", "prefix"])
def test_dropped_prefetch_never_lands_in_a_reused_block(tmp_path, drop):
    # copied-ahead groups never onloaded are freed while their copy may still be queued
    H = 4096

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.ModuleList([torch.nn.Linear(H, H, bias = False) for _ in range(2)])
            self.tail = torch.nn.ModuleList([torch.nn.Linear(H, H, bias = False) for _ in range(6)])

        def forward(
            self,
            x,
            tail = True,
            boom = False,
        ):
            for m in self.head:
                x = m(x)
            if boom:
                raise RuntimeError("boom")
            for m in self.tail if tail else ():
                x = m(x)
            return x

    enc = Enc().to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in enc.parameters():
            p.fill_(1.0)
    enc = _mapped_streamed(enc, tmp_path / "enc.safetensors")
    assert sh.install_encoder_prefetch(enc, "cuda") > 0
    x = torch.zeros(1, H, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        enc(x)
        for _ in range(10):
            try:
                enc(x, tail = False, boom = drop == "exception")
            except RuntimeError:
                pass
            outs = [torch.empty(H, H, dtype = torch.bfloat16, device = "cuda") for _ in range(6)]
            for o in outs:
                o.fill_(-2.0)
            torch.cuda.synchronize()
            assert all(bool((o == -2.0).all()) for o in outs)
            del outs
            enc(x)
            torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_released_resident_groups_are_fenced():
    # resident groups go back to diffusers' onload_, whose copy runs on diffusers' own stream
    pytest.importorskip("diffusers")
    import copy

    import core.inference.diffusion_memory as dm

    H, F = 4096, 16384

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(H)
            self.wi = torch.nn.Linear(H, F, bias = False)
            self.wo = torch.nn.Linear(F, H, bias = False)

        def forward(self, x):
            return x + self.wo(torch.relu(self.wi(self.norm(x))))

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(1000, H)
            self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])
            self.final = torch.nn.LayerNorm(H)

        def forward(self, ids):
            h = self.embed(ids)
            for b in self.blocks:
                h = b(h)
            return self.final(h)

    torch.manual_seed(0)
    enc = Enc().half().eval()
    for p in enc.parameters():
        p.data.normal_(0, 0.02)
    ref = copy.deepcopy(enc).cuda()
    tr = torch.nn.Sequential(torch.nn.Linear(8, 8))
    pipe = types.SimpleNamespace(
        transformer = tr, text_encoder = enc, components = {"transformer": tr, "text_encoder": enc}
    )
    pipe._unsloth_small_host = {"components": {"text_encoder": "memory-mapped"}}
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, resident_text_encoder_mib = 140
    )
    pf = getattr(enc, sh.ENCODER_PREFETCH_ATTR)
    with torch.no_grad():
        for _ in range(2):
            enc(torch.randint(0, 1000, (4, 512), device = "cuda"))
        assert dm.release_resident_groups(pipe, 100, None) is not None
        for _ in range(10):
            ids = torch.randint(0, 1000, (4, 512), device = "cuda")
            want = ref(ids)
            torch.cuda.synchronize()
            got = enc(ids)
            torch.cuda.synchronize()
            assert torch.equal(got, want)
    assert pf.error is None and pf.stats["prefetched"] > 0
