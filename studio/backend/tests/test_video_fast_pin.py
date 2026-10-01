# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The video loads' fast background pin: registered host memory filled on worker threads, same bytes and results."""

import gc
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as mem  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


class _Pipe:
    pass


def test_fast_pins_are_requested_per_pipe_and_have_a_kill_switch(monkeypatch):
    pipe = _Pipe()
    assert not getattr(pipe, mem._FAST_PIN_REQUEST_ATTR, False)
    mem.request_fast_pins(pipe)
    assert getattr(pipe, mem._FAST_PIN_REQUEST_ATTR) is True
    for off in ("0", "off", "false", "no"):
        monkeypatch.setenv(mem.FAST_PIN_ENV, off)
        assert mem._fast_pin_supported() is False
    monkeypatch.delenv(mem.FAST_PIN_ENV)
    monkeypatch.setattr(mem.sys, "platform", "win32")
    assert mem._fast_pin_supported() is False


def test_a_pipe_without_the_request_keeps_the_allocator_pin(monkeypatch):
    # Image pipelines never ask, so their pinner is unchanged even where registration is supported.
    monkeypatch.setattr(mem, "_fast_pin_supported", lambda: True)
    monkeypatch.setattr(mem, "_offload_groups", lambda module: [types.SimpleNamespace()])
    plain, fast = _Pipe(), _Pipe()
    mem.request_fast_pins(fast)
    assert mem._defer_pinning(plain, object(), "cuda", None)
    assert mem._defer_pinning(fast, object(), "cuda", None)
    assert getattr(plain, mem._PENDING_PINS_ATTR)[0].fast is False
    assert getattr(fast, mem._PENDING_PINS_ATTR)[0].fast is True


@needs_cuda
@pytest.mark.parametrize(
    "src",
    [
        lambda: torch.randn(37, 41).to(torch.bfloat16),
        lambda: torch.arange(5000, dtype = torch.int64),
        lambda: torch.randn(3, 1),
        lambda: torch.randn(8, 8)[:, ::2],
    ],
    ids = ["bf16", "int64", "tiny", "strided"],
)
def test_registered_copy_is_pinned_and_byte_identical(src):
    src = src()
    out = mem.registered_host_copy(src)
    assert out.is_pinned() and out.device.type == "cpu"
    assert out.dtype == src.dtype and out.shape == src.shape and torch.equal(out, src)
    assert out.data_ptr() % 4096 == 0
    up = out.to("cuda", non_blocking = True)
    torch.cuda.synchronize()
    assert torch.equal(up.cpu(), src)


@needs_cuda
def test_the_last_view_releasing_its_buffer_unregisters_it():
    calls = []
    real = torch.cuda.cudart()
    proxy = types.SimpleNamespace(
        cudaHostRegister = real.cudaHostRegister,
        cudaHostUnregister = lambda ptr: (calls.append(ptr), real.cudaHostUnregister(ptr))[1],
    )
    holder = mem._RegisteredHostBuffer(1 << 20)
    holder._cudart = proxy
    holder.register()
    import numpy as np

    view = torch.from_numpy(np.asarray(holder))
    ptr = holder.ptr
    del holder
    gc.collect()
    assert calls == []  # the tensor still holds the buffer
    del view
    gc.collect()
    assert calls == [ptr]


@needs_cuda
def test_a_fast_deferred_pin_renders_identically_and_ends_with_every_group_registered():
    pytest.importorskip("diffusers")
    from diffusers.hooks import apply_group_offloading

    def _net():
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(256, 384), torch.nn.GELU(), torch.nn.Linear(384, 256)
                )
                for _ in range(6)
            ]
        )
        return net.to(torch.bfloat16)

    def _offload(net, pinned):
        apply_group_offloading(
            net,
            onload_device = torch.device("cuda"),
            offload_device = torch.device("cpu"),
            offload_type = "leaf_level",
            use_stream = True,
            low_cpu_mem_usage = not pinned,
        )
        return net

    mem.install_group_pin_wait()
    x = torch.randn(8, 256, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        ref = _offload(_net(), pinned = True)(x)
    pipe, net = _Pipe(), _offload(_net(), pinned = False)
    mem.request_fast_pins(pipe)
    before = {id(p): p.detach().clone() for p in net.parameters()}
    assert mem._defer_pinning(pipe, net, torch.device("cuda"), None)
    assert getattr(pipe, mem._PENDING_PINS_ATTR)[0].fast is mem._fast_pin_supported()
    assert mem.start_background_pins(pipe) == 1
    with torch.no_grad():
        out = net(x)
        again = net(x)
    torch.cuda.synchronize()
    mem.finish_background_pins(pipe)
    assert torch.equal(out, ref) and torch.equal(again, ref)
    groups = mem._offload_groups(net)
    assert groups and all(t.is_pinned() for g in groups for t in g.cpu_param_dict.values())
    for p in net.parameters():
        assert torch.equal(p.detach(), before[id(p)])
