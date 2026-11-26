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


@pytest.mark.parametrize(
    "hip, cuda, want", [(None, "13.0", "libcudart.so.13"), ("6.4.1", None, "libamdhip64.so.6")]
)
def test_the_resolver_opens_the_runtime_torch_links_not_another_mapped_one(
    monkeypatch, hip, cuda, want
):
    import ctypes
    import io
    import sys

    # Another package mapped an older runtime first; the global namespace would resolve to it.
    maps = "".join(
        f"7f00-7f01 r-xp 0 08:01 1 /env/lib/{name}\n"
        for name in ("libcudart.so.12", "libamdhip64.so.5", want, want, "libc.so.6")
    )
    opened = []

    class _Lib:
        def __init__(self, path):
            opened.append(path)
            self.path = path

        def __getattr__(self, name):
            # torch's extension does not reach the symbol here (a static runtime): fall back to the maps
            if self.path == torch._C.__file__:
                raise AttributeError(name)
            return name

    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.version, "cuda", cuda, raising = False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hp, "open", lambda *_a, **_k: io.StringIO(maps), raising = False)
    monkeypatch.setattr(ctypes, "CDLL", _Lib)
    assert hp._resolve_get_last_error() == ("hipGetLastError" if hip else "cudaGetLastError")
    assert opened == [torch._C.__file__, f"/env/lib/{want}"]


def test_the_resolver_looks_up_through_torch_own_extension_first(monkeypatch):
    import ctypes
    import sys

    opened = []

    class _Lib:
        def __init__(self, path):
            opened.append(path)

        def __getattr__(self, name):
            return name

    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", _Lib)
    assert hp._resolve_get_last_error() == "cudaGetLastError"
    assert opened == [torch._C.__file__]


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


def _fake_holder(unregister_rc: int):
    calls = []
    holder = object.__new__(mem._RegisteredHostBuffer)
    holder._cudart = type(
        "R",
        (),
        {"cudaHostUnregister": staticmethod(lambda ptr: calls.append(ptr) or unregister_rc)},
    )()
    holder.ptr, holder._size, holder._registered, holder._device = 4096, 4096, True, 0
    holder._anchor = None
    holder._map = type("M", (), {"close": lambda self: None})()
    return holder, calls


@pytest.mark.parametrize("rc, consumed", [(713, ["consumed"]), (0, [])])
def test_a_failed_unregister_consumes_the_pending_error(monkeypatch, rc, consumed):
    seen = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a, **_k: None)
    monkeypatch.setattr(mem, "consume_cuda_error", lambda: seen.append("consumed"))
    holder, calls = _fake_holder(rc)
    holder.__del__()
    assert calls == [4096] and seen == consumed and holder._registered is False


@pytest.mark.parametrize("rc, consumed", [(713, ["consumed"]), (0, [])])
def test_the_training_stream_consumes_a_failed_unregister(monkeypatch, rc, consumed):
    import types

    from core.training import diffusion_te_stream as ts

    seen = []
    monkeypatch.setattr(hp, "consume_cuda_error", lambda: seen.append("consumed"))
    monkeypatch.setattr(
        torch.cuda, "cudart", lambda: types.SimpleNamespace(cudaHostUnregister = lambda ptr: rc)
    )
    reg = object.__new__(ts._Registered)
    reg.ptr, reg.stream = 4096, types.SimpleNamespace(synchronize = lambda: None)
    reg.release()
    assert seen == consumed and reg.ptr is None


@pytest.mark.parametrize(
    "platform, registers", [("linux", True), ("win32", False), ("darwin", False)]
)
def test_h3_residency_registers_host_copies_only_on_linux(monkeypatch, platform, registers):
    import sys

    import core.inference.video_minimax_h3_residency as res

    registered, pinned = [], []
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(mem, "registered_host_copy", lambda src: registered.append(src) or src)
    monkeypatch.setattr(
        torch.Tensor, "pin_memory", lambda self, *a, **k: pinned.append(self) or self.clone()
    )
    src = torch.ones(3000, dtype = torch.uint8)
    assert torch.equal(res._pin_host_tensor(src), src)
    assert (len(registered), len(pinned)) == ((1, 0) if registers else (0, 1))
    # the compaction prices what the pin takes: page-exact when registered, else the allocator's power of two
    assert res._pin_cost(torch.ones(9000, dtype = torch.uint8)) == (12288 if registers else 16384)


@needs_cuda
def test_the_next_kernel_after_a_failed_unregister_runs():
    holder = mem._RegisteredHostBuffer(1 << 20)
    holder.register()
    # unregister behind the holder's back: its own unregister then fails (cudaErrorHostMemoryNotRegistered)
    assert int(holder._cudart.cudaHostUnregister(holder.ptr)) == 0
    holder.__del__()
    _launch()  # raised the stale "pointer does not correspond to a registered memory region" before


@needs_cuda
def test_the_training_stream_next_kernel_after_a_failed_unregister_runs():
    from core.training import diffusion_te_stream as ts

    buf = torch.empty(1 << 20, dtype = torch.uint8)
    reg = ts._Registered(buf, buf.numel(), torch.cuda.current_stream())
    assert int(torch.cuda.cudart().cudaHostUnregister(reg.ptr)) == 0
    reg.release()
    _launch()


def _allocator_pin_on_cpu(monkeypatch, pinned_storages: set):
    """torch's semantics without a GPU: pin_memory() returns an already-pinned tensor itself, else a pinned copy."""
    real_empty_like = torch.empty_like

    def pin_memory(self, *_a, **_k):
        if self.untyped_storage().data_ptr() in pinned_storages:
            return self
        out = self.clone()
        pinned_storages.add(out.untyped_storage().data_ptr())
        return out

    def empty_like(
        src,
        *a,
        pin_memory = False,
        **k,
    ):
        out = real_empty_like(src, *a, **k)
        if pin_memory:
            pinned_storages.add(out.untyped_storage().data_ptr())
        return out

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    monkeypatch.setattr(torch, "empty_like", empty_like)


def test_off_linux_a_pinned_slab_view_gets_its_own_pinned_copy(monkeypatch):
    import sys

    import core.inference.video_minimax_h3_residency as res

    pinned: set = set()
    _allocator_pin_on_cpu(monkeypatch, pinned)
    monkeypatch.setattr(sys, "platform", "win32")
    slab = torch.zeros(1 << 20, dtype = torch.float32)
    pinned.add(slab.untyped_storage().data_ptr())
    view = slab[:256]
    view.copy_(torch.randn(256))
    out = res._pin_host_tensor(view)
    slab_ptr, out_ptr = slab.untyped_storage().data_ptr(), out.untyped_storage().data_ptr()
    assert out_ptr != slab_ptr
    assert out.untyped_storage().nbytes() == view.numel() * view.element_size()
    assert out_ptr in pinned and bool(torch.equal(out, view))


@pytest.mark.parametrize("priced", ["1", "0"])
def test_off_linux_compaction_moves_weights_off_the_slab_it_reports_freed(monkeypatch, priced):
    import sys
    import types

    import core.inference.video_minimax_h3_residency as res
    from core.inference import diffusion_small_host

    pinned: set = set()
    _allocator_pin_on_cpu(monkeypatch, pinned)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("UNSLOTH_H3_COMPACT_PRICED", priced)
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (53000, None))
    slab = torch.zeros(16 << 20, dtype = torch.float32)  # 64 MiB pinned slab
    pinned.add(slab.untyped_storage().data_ptr())
    host = slab[:256]
    weight = torch.empty(256)  # the device weight stands apart from its host copy
    group = types.SimpleNamespace(cpu_param_dict = {weight: host})
    freed = res.compact_host_slabs([group])
    assert freed == slab.untyped_storage().nbytes()
    moved = group.cpu_param_dict[weight]
    # the slab is freed only when no surviving copy still points into it
    slab_ptr, moved_ptr = slab.untyped_storage().data_ptr(), moved.untyped_storage().data_ptr()
    assert moved_ptr != slab_ptr
    assert bool(torch.equal(moved, host))


def test_off_linux_a_rebuild_asks_the_conditioner_for_the_allocator_rounded_bytes(monkeypatch):
    import sys
    import types

    import core.inference.video_minimax_h3_residency as res
    from core.inference import diffusion_small_host

    keep = res.H3_REBUILD_KEEP_MIB + 100
    mib = 1 << 20
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diffusion_small_host, "encoder_resident_keep_mib", lambda total: keep)
    # 6 MiB above the reserve: the 5 MiB payload fits, its 8 MiB pinned block does not
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (53000, keep + 6))
    weight = torch.nn.Parameter(torch.zeros(5 * mib, dtype = torch.uint8), requires_grad = False)
    group = types.SimpleNamespace(parameters = [weight], cpu_param_dict = {})
    setattr(group, res._RELEASED_ATTR, 5 * mib)
    asked = []
    assert res.rebuild_host_copy(group, host_yield = asked.append) == 5 * mib
    assert asked == [2 * mib]
    copy_ptr = group.cpu_param_dict[weight].untyped_storage().data_ptr()
    assert copy_ptr != weight.untyped_storage().data_ptr()


def test_on_linux_a_rebuild_still_asks_for_the_payload(monkeypatch):
    import sys
    import types

    import core.inference.video_minimax_h3_residency as res
    from core.inference import diffusion_small_host

    keep = res.H3_REBUILD_KEEP_MIB + 100
    mib = 1 << 20
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diffusion_small_host, "encoder_resident_keep_mib", lambda total: keep)
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (53000, keep + 6))
    weight = torch.nn.Parameter(torch.zeros(5 * mib, dtype = torch.uint8), requires_grad = False)
    group = types.SimpleNamespace(parameters = [weight], cpu_param_dict = {})
    setattr(group, res._RELEASED_ATTR, 5 * mib)
    asked = []
    res.rebuild_host_copy(group, host_yield = asked.append)
    assert asked == []
