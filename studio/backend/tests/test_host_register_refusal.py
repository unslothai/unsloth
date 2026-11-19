# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A refused cudaHostRegister must not leave its error pending for the thread's next kernel launch."""

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as mem  # noqa: E402
import core.inference.host_pinning as hp  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def test_consume_never_raises_without_a_cuda_runtime(monkeypatch):
    import ctypes

    def _no_runtime(*_a, **_k):
        raise OSError("no libcudart")

    monkeypatch.setattr(hp, "_get_last_error", None)
    monkeypatch.setattr(ctypes, "CDLL", _no_runtime)
    hp.consume_cuda_error()
    assert hp._get_last_error is None


def test_consume_calls_cuda_get_last_error_once_resolved(monkeypatch):
    calls = []
    monkeypatch.setattr(hp, "_get_last_error", lambda: calls.append(1) or 0)
    hp.consume_cuda_error()
    assert calls == [1]


@pytest.mark.parametrize(
    "hip, platform, want",
    [
        (None, "linux", "cudaGetLastError"),
        ("6.4", "linux", "hipGetLastError"),
        (None, "win32", "cudaGetLastError"),
        ("6.4", "win32", "hipGetLastError"),
    ],
)
def test_the_resolver_asks_the_runtime_torch_runs_on(monkeypatch, tmp_path, hip, platform, want):
    import ctypes
    import sys

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / ("amdhip64_6.dll" if hip else "cudart64_12.dll")).write_bytes(b"")
    opened = []

    class _Lib:
        def __init__(self, path):
            opened.append(path)

        def __getattr__(self, name):
            return name

    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(ctypes, "CDLL", _Lib)
    assert hp._resolve_get_last_error() == want
    if platform == "win32":
        assert len(opened) == 1 and str(lib) in opened[0]


def test_a_refused_registration_consumes_the_pending_error(monkeypatch):
    calls = []
    holder = object.__new__(mem._RegisteredHostBuffer)
    holder._cudart = type("R", (), {"cudaHostRegister": staticmethod(lambda *_a: 712)})()
    holder.ptr, holder._size, holder._registered = 4096, 4096, False
    monkeypatch.setattr(mem, "consume_cuda_error", lambda: calls.append("consumed"))
    with pytest.raises(RuntimeError, match = "cudaHostRegister failed"):
        holder.register()
    assert calls == ["consumed"] and holder._registered is False


def _launch() -> None:
    x = torch.ones(16, device = "cuda")
    (x * 2).sum().item()


@needs_cuda
def test_the_next_kernel_after_a_refused_registration_runs():
    holder = mem._RegisteredHostBuffer(1 << 20)
    holder.register()
    # registering the same range again is refused (cudaErrorHostMemoryAlreadyRegistered)
    with pytest.raises(RuntimeError, match = "cudaHostRegister failed"):
        holder.register()
    _launch()  # raised the stale "part or all of the requested memory range is already mapped" before
    del holder


@needs_cuda
def test_the_training_stream_registration_consumes_its_refusal_too():
    from core.training import diffusion_te_stream as ts

    buf = torch.empty(1 << 20, dtype = torch.uint8)
    stream = torch.cuda.current_stream()
    first = ts._Registered(buf, buf.numel(), stream)
    with pytest.raises(RuntimeError, match = "cudaHostRegister failed"):
        ts._Registered(buf, buf.numel(), stream)
    _launch()
    first.release()
