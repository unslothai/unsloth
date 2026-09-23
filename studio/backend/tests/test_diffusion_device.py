# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic, CPU-only tests for the diffusion device/dtype resolver.

`torch` is stubbed via a fake module so no GPU/torch is needed, and
`utils.hardware` is either stubbed (studio-layer path) or forced to fail
(torch-probe fallback path). Both paths are asserted.
"""

from __future__ import annotations

import sys
import types
from typing import Optional

import pytest

from core.inference import diffusion_device as dd


# ── Fakes ─────────────────────────────────────────────────────────────


class _FakeDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDtype) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:  # str(dtype) -> "torch.bfloat16"
        return f"torch.{self.name}"


BF16 = _FakeDtype("bfloat16")
FP16 = _FakeDtype("float16")
FP32 = _FakeDtype("float32")


class _FiniteResult:
    def __init__(self, finite: bool) -> None:
        self._finite = finite

    def all(self) -> "_FiniteResult":
        return self

    def item(self) -> bool:
        return self._finite


class _FakeTensor:
    def __init__(self, finite: bool = True) -> None:
        self._finite = finite

    def __add__(self, other: object) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self


def _make_torch(
    *,
    cuda_available: bool = False,
    capability = (8, 0),
    capability_raises: bool = False,
    bf16_supported: bool = False,
    hip = None,
    mps_available: bool = False,
    mps_probe: str = "pass",  # "pass" | "raise" | "nonfinite"
    xpu_available = None,  # None -> no xpu attr; True/False -> present
    xpu_bf16: bool = False,
    device_count: int = 1,
    # Free VRAM per physical index, for the multi-card pick. Absent -> 0.
    free_vram_by_index: Optional[dict] = None,
    # Records every torch.cuda.set_device() this fake receives, so a test can assert the pin.
    set_device_calls: Optional[list] = None,
) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.bfloat16 = BF16
    torch.float16 = FP16
    torch.float32 = FP32
    torch.version = types.SimpleNamespace(hip = hip)
    free_vram_by_index = free_vram_by_index or {}

    def _set_device(index):
        if set_device_calls is not None:
            set_device_calls.append(index)

    # Same optional-device signature as torch's own, so a probe of a SELECTED card is answered rather than raising.
    def _get_cap(device = None):
        if capability_raises:
            raise RuntimeError("no capability")
        return capability

    torch.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        get_device_capability = _get_cap,
        is_bf16_supported = lambda: bf16_supported,
        device_count = lambda: device_count,
        mem_get_info = lambda index = None: (free_vram_by_index.get(index, 0), 0),
        set_device = _set_device,
    )

    mps_ns = types.SimpleNamespace(is_available = lambda: mps_available)
    torch.backends = types.SimpleNamespace(mps = mps_ns)

    def _ones(*_a, **_k):
        if mps_probe == "raise":
            raise RuntimeError("bf16 unsupported on this MPS")
        return _FakeTensor(finite = (mps_probe == "pass"))

    torch.ones = _ones
    torch.isfinite = lambda t: _FiniteResult(getattr(t, "_finite", True))

    if xpu_available is not None:
        torch.xpu = types.SimpleNamespace(
            is_available = lambda: xpu_available,
            is_bf16_supported = lambda: xpu_bf16,
        )
    return torch


def _install(
    monkeypatch,
    torch,
    *,
    studio_device = None,
    is_rocm = False,
    hardware_fails = False,
):
    """Install the fake torch and either a fake or failing `utils.hardware`."""
    monkeypatch.setitem(sys.modules, "torch", torch)
    if hardware_fails:
        # Force `from utils.hardware import ...` to raise, exercising the torch-probe fallback.
        monkeypatch.setitem(sys.modules, "utils.hardware", None)
        return

    class _DT:
        CUDA = "cuda"
        XPU = "xpu"
        MLX = "mlx"
        CPU = "cpu"

    fake_uh = types.ModuleType("utils.hardware")
    fake_uh.DeviceType = _DT
    fake_uh.get_device = lambda: studio_device
    fake_uh.hardware = types.SimpleNamespace(IS_ROCM = is_rocm)
    monkeypatch.setitem(sys.modules, "utils.hardware", fake_uh)


# ── Unsloth-layer path ─────────────────────────────────────────────────


def test_cuda_ampere_bf16(monkeypatch):
    torch = _make_torch(cuda_available = True, capability = (8, 0))
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.dtype, t.backend, t.vendor) == ("cuda", BF16, "cuda", "nvidia")
    assert (
        t.supports_model_cpu_offload
        and t.supports_default_torch_compile
        and t.supports_pinned_transfer
    )


def test_cuda_pre_ampere_fp16(monkeypatch):
    torch = _make_torch(cuda_available = True, capability = (7, 5), bf16_supported = True)
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    # is_bf16_supported() is True (emulated) but capability < 8, so fp16.
    assert t.dtype == FP16 and t.backend == "cuda"


def test_cuda_capability_raises_falls_back_fp16(monkeypatch):
    torch = _make_torch(cuda_available = True, capability_raises = True)
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    assert t.dtype == FP16 and t.device == "cuda"


def test_cuda_studio_says_cuda_but_unavailable_is_cpu(monkeypatch):
    torch = _make_torch(cuda_available = False)
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "cpu" and t.dtype == FP32


def test_rocm_target(monkeypatch):
    torch = _make_torch(cuda_available = True, bf16_supported = True)
    _install(monkeypatch, torch, studio_device = "cuda", is_rocm = True)
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor) == ("cuda", "rocm", "amd")
    assert t.dtype == BF16
    assert t.supports_default_torch_compile is False  # ROCm disables default compile


def test_rocm_without_bf16_uses_fp16(monkeypatch):
    torch = _make_torch(cuda_available = True, bf16_supported = False)
    _install(monkeypatch, torch, studio_device = "cuda", is_rocm = True)
    t = dd.resolve_diffusion_device_target()
    assert t.dtype == FP16 and t.backend == "rocm"


def test_xpu_bf16(monkeypatch):
    torch = _make_torch(xpu_available = True, xpu_bf16 = True)
    _install(monkeypatch, torch, studio_device = "xpu")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor, t.dtype) == ("xpu", "xpu", "intel", BF16)
    assert (
        t.supports_model_cpu_offload
        and not t.supports_default_torch_compile
        and not t.supports_pinned_transfer
    )


def test_xpu_without_bf16_fp16(monkeypatch):
    torch = _make_torch(xpu_available = True, xpu_bf16 = False)
    _install(monkeypatch, torch, studio_device = "xpu")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "xpu" and t.dtype == FP16


def test_mps_probe_pass_bf16(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "pass")
    _install(monkeypatch, torch, studio_device = "mlx")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor, t.dtype) == ("mps", "mps", "apple", BF16)
    assert not t.supports_model_cpu_offload


def test_mps_probe_raises_uses_fp32_not_fp16(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "raise")
    _install(monkeypatch, torch, studio_device = "mlx")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == FP32  # strict: never silent fp16


def test_mps_probe_nonfinite_uses_fp32(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "nonfinite")
    _install(monkeypatch, torch, studio_device = "mlx")
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == FP32


def test_studio_cpu_on_apple_prefers_mps(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "pass")
    _install(monkeypatch, torch, studio_device = "cpu")  # Unsloth reports CPU (no mlx pkg)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == BF16


def test_cpu_when_nothing_available(monkeypatch):
    torch = _make_torch(mps_available = False)
    _install(monkeypatch, torch, studio_device = "cpu")
    t = dd.resolve_diffusion_device_target()
    assert (t.device, t.backend, t.vendor, t.dtype) == ("cpu", "cpu", None, FP32)
    assert not any(
        (t.supports_model_cpu_offload, t.supports_default_torch_compile, t.supports_pinned_transfer)
    )


# ── torch-probe fallback path (utils.hardware import fails) ────────────


def test_fallback_cuda(monkeypatch):
    torch = _make_torch(cuda_available = True, capability = (9, 0))
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "cuda" and t.dtype == BF16 and t.backend == "cuda"


def test_fallback_rocm_via_torch_hip(monkeypatch):
    torch = _make_torch(cuda_available = True, bf16_supported = True, hip = "6.2")
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.backend == "rocm" and t.vendor == "amd"


def test_fallback_xpu(monkeypatch):
    torch = _make_torch(cuda_available = False, xpu_available = True, xpu_bf16 = True)
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "xpu" and t.dtype == BF16


def test_fallback_mps(monkeypatch):
    torch = _make_torch(cuda_available = False, mps_available = True, mps_probe = "pass")
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "mps" and t.dtype == BF16


def test_fallback_cpu(monkeypatch):
    torch = _make_torch(cuda_available = False, mps_available = False)
    _install(monkeypatch, torch, hardware_fails = True)
    t = dd.resolve_diffusion_device_target()
    assert t.device == "cpu" and t.dtype == FP32


# ── from-torch-device reconstruction + public dict ────────────────────


def test_from_torch_device_cuda(monkeypatch):
    torch = _make_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    t = dd.diffusion_device_target_from_torch_device("cuda:0", FP32)
    assert (t.device, t.backend, t.vendor, t.dtype) == ("cuda", "cuda", "nvidia", FP32)
    assert t.is_cuda_torch_device


def test_from_torch_device_mps_and_cpu(monkeypatch):
    torch = _make_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    mps = dd.diffusion_device_target_from_torch_device("mps", FP16)
    assert mps.device == "mps" and not mps.supports_model_cpu_offload
    cpu = dd.diffusion_device_target_from_torch_device("cpu", FP32)
    assert cpu.device == "cpu" and cpu.vendor is None


@pytest.mark.parametrize(
    "dtype,expected", [(BF16, "bfloat16"), (FP16, "float16"), (FP32, "float32")]
)
def test_public_dict_dtype_string(dtype, expected):
    t = dd.DiffusionDeviceTarget(
        device = "cuda",
        dtype = dtype,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        supports_pinned_transfer = True,
    )
    d = t.as_public_dict()
    assert d["dtype"] == expected and "torch." not in d["dtype"]


# -- float64 capability + the RoPE demotion it drives -------------------------------------------


def test_only_mps_lacks_float64(monkeypatch):
    torch = _make_torch(mps_available = True, mps_probe = "pass")
    _install(monkeypatch, torch)
    assert dd.resolve_diffusion_device_target().supports_float64 is False
    for device in ("cuda", "xpu", "cpu"):
        assert dd.diffusion_device_target_from_torch_device(device, FP32).supports_float64 is True
    assert dd.diffusion_device_target_from_torch_device("mps", FP32).supports_float64 is False


class _RopeModule:
    def __init__(self, double_precision = True):
        self.double_precision = double_precision


class _Component:
    def __init__(self, *mods):
        self._mods = mods

    def modules(self):
        return iter(self._mods)


class _Pipe:
    def __init__(self, **components):
        self.components = components


def _mps_target():
    return dd.diffusion_device_target_from_torch_device("mps", FP32)


def _cuda_target():
    return dd.diffusion_device_target_from_torch_device("cuda", FP32)


def test_force_float32_rope_demotes_every_component_on_mps():
    # Two components, several modules each: the connectors and the transformer both carry RoPE,
    # so demoting only the first one found would still crash inside the denoise loop.
    conn, dit_a, dit_b = _RopeModule(), _RopeModule(), _RopeModule()
    pipe = _Pipe(connectors = _Component(conn), transformer = _Component(dit_a, dit_b))
    assert dd.force_float32_rope(pipe, _mps_target()) == 3
    assert not any(m.double_precision for m in (conn, dit_a, dit_b))


def test_force_float32_rope_leaves_float64_devices_untouched():
    rope = _RopeModule()
    pipe = _Pipe(transformer = _Component(rope))
    assert dd.force_float32_rope(pipe, _cuda_target()) == 0
    assert rope.double_precision is True


def test_force_float32_rope_skips_modules_without_the_flag():
    already_off = _RopeModule(double_precision = False)
    plain = object()
    pipe = _Pipe(vae = _Component(already_off, plain))
    assert dd.force_float32_rope(pipe, _mps_target()) == 0


def test_force_float32_rope_tolerates_non_module_components():
    # Pipelines carry schedulers and tokenizers with no .modules(); they must not abort the walk.
    rope = _RopeModule()
    pipe = _Pipe(scheduler = object(), tokenizer = None, transformer = _Component(rope))
    assert dd.force_float32_rope(pipe, _mps_target()) == 1
    assert rope.double_precision is False


def test_the_video_loader_demotes_rope():
    # The tests above prove the helper works, not that anything calls it: deleting the call site
    # leaves every one of them green while LTX-2 goes back to raising on Metal. Where in the
    # loader is not asserted -- the flag is read when a pipeline first builds its frequency
    # tables, after load_pipeline returns -- but reaching it unconditionally is, since the helper
    # already no-ops on a float64 device and a guard here could only ever get the polarity wrong.
    #
    # Asserted as "reached with no condition above it" rather than by rejecting `if`: a guard can
    # equally be written `target.supports_float64 and force_float32_rope(...)` or as a ternary,
    # and naming the shapes only rejects the ones already thought of.
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "core/inference/video.py").read_text(
        encoding = "utf-8"
    )
    loader = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "load_pipeline"
    )

    def _is_rope_call(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "force_float32_rope"
        )

    # Everything a condition could skip, whatever syntax expresses it.
    conditional = {
        id(inner)
        for node in ast.walk(loader)
        if isinstance(node, (ast.If, ast.IfExp, ast.BoolOp))
        for inner in ast.walk(node)
    }
    assert any(
        isinstance(n, ast.Expr) and _is_rope_call(n.value) and id(n) not in conditional
        for n in ast.walk(loader)
    ), (
        "load_pipeline does not reach force_float32_rope unconditionally, so the demotion is "
        "either gone or behind a guard -- and a guard here can only be wrong, since the helper "
        "already no-ops wherever float64 works"
    )


# ── Pressure-gated decoder sync ───────────────────────────────────────


def _target(device: str) -> dd.DiffusionDeviceTarget:
    return dd.DiffusionDeviceTarget(
        device = device,
        dtype = FP32,
        backend = device,
        vendor = None,
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )


class _FakeDecoder:
    """A VAE decoder module, following nn.Module: a hook returning non-None replaces the output."""

    def __init__(self) -> None:
        self.hooks: list = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

    def decode(self, calls: int) -> list:
        outputs = []
        for index in range(calls):
            out = f"out{index}"
            for hook in self.hooks:
                replacement = hook(self, (), out)
                if replacement is not None:
                    out = replacement
            outputs.append(out)
        return outputs


def _pipe_with(decoder) -> types.SimpleNamespace:
    return types.SimpleNamespace(vae = types.SimpleNamespace(decoder = decoder))


def _mps_torch(used = 0, recommended = 100) -> types.ModuleType:
    """torch whose mps backend counts synchronize() calls over a settable memory reading."""
    torch = types.ModuleType("torch")
    torch.syncs = 0
    torch.used = used

    def _bump():
        torch.syncs += 1

    torch.mps = types.SimpleNamespace(
        synchronize = _bump,
        recommended_max_memory = lambda: recommended,
        driver_allocated_memory = lambda: torch.used,
    )
    return torch


@pytest.mark.parametrize("device", ["cuda", "xpu", "cpu"])
def test_decoder_sync_is_metal_only(monkeypatch, device):
    monkeypatch.setitem(sys.modules, "torch", _mps_torch())
    decoder = _FakeDecoder()
    assert dd.install_decoder_sync(_pipe_with(decoder), _target(device)) is False
    assert decoder.hooks == []


def test_decoder_sync_idle_while_memory_is_plentiful(monkeypatch):
    # The whole point of the gate: a decode that fits pays nothing at all.
    torch = _mps_torch(recommended = 100, used = 10)
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    assert dd.install_decoder_sync(_pipe_with(decoder), _target("mps")) is True
    decoder.decode(5)
    assert torch.syncs == 0


def test_decoder_sync_runs_once_per_decoder_call_above_the_threshold(monkeypatch):
    torch = _mps_torch(recommended = 100, used = 10)
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    dd.install_decoder_sync(_pipe_with(decoder), _target("mps"))
    decoder.decode(2)
    assert torch.syncs == 0
    # The growth this bounds is per decoder call, so every call above the threshold syncs.
    torch.used = 100 * dd.DECODE_SYNC_FRACTION
    decoder.decode(3)
    assert torch.syncs == 3
    # ...and it stands down again once the allocator has given the memory back.
    torch.used = 10
    decoder.decode(4)
    assert torch.syncs == 3


def test_decoder_sync_threshold_scales_with_the_device(monkeypatch):
    # Pins the policy AND that the budget is a fraction of this device's working set rather than a
    # fixed byte count -- a decode is only "running out" relative to the machine it runs on.
    torch = _mps_torch(recommended = 200, used = 169)
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    dd.install_decoder_sync(_pipe_with(decoder), _target("mps"))
    decoder.decode(1)
    assert torch.syncs == 0
    torch.used = 170
    decoder.decode(1)
    assert torch.syncs == 1
    assert dd.DECODE_SYNC_FRACTION == 0.85


def test_decoder_sync_preserves_the_decoder_output(monkeypatch):
    # An nn.Module forward hook that returns non-None REPLACES the output; this one must not.
    torch = _mps_torch(recommended = 100, used = 100)
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    dd.install_decoder_sync(_pipe_with(decoder), _target("mps"))
    assert decoder.decode(2) == ["out0", "out1"]
    assert torch.syncs == 2


@pytest.mark.parametrize("pipe", [types.SimpleNamespace(), _pipe_with(None), _pipe_with(object())])
def test_decoder_sync_no_op_without_a_hookable_decoder(monkeypatch, pipe):
    monkeypatch.setitem(sys.modules, "torch", _mps_torch())
    assert dd.install_decoder_sync(pipe, _target("mps")) is False


def _mps_torch_without_recommended(used = 0) -> types.ModuleType:
    """torch 2.4's mps surface: driver_allocated_memory and synchronize, no working-set reading.

    Verified against torch/mps/__init__.py at v2.4.0 (absent) and v2.5.0 (present), and against
    an installed torch 2.4.1.
    """
    torch = _mps_torch(used = used)
    del torch.mps.recommended_max_memory
    return torch


def test_decoder_sync_survives_a_torch_without_the_memory_reading(monkeypatch):
    # install.sh keeps an existing venv's torch (>=2.4), and reading a 2.5 API there raised
    # AttributeError from inside the video load -- after the download, with no OOM to explain it.
    torch = _mps_torch_without_recommended()
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    assert dd.install_decoder_sync(_pipe_with(decoder), _target("mps")) is True
    # No budget to compare against, so it must not silently decide the decode is fine: an
    # unbounded Wan decode is what grew past 148 GiB.
    assert decoder.decode(3) == ["out0", "out1", "out2"]
    assert torch.syncs == 3


def test_decoder_sync_survives_a_working_set_reading_that_raises(monkeypatch):
    torch = _mps_torch()

    def _boom():
        raise RuntimeError("MPS backend is not available")

    torch.mps.recommended_max_memory = _boom
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    assert dd.install_decoder_sync(_pipe_with(decoder), _target("mps")) is True
    decoder.decode(2)
    assert torch.syncs == 2


def test_decoder_sync_survives_a_gauge_that_raises_mid_decode(monkeypatch):
    torch = _mps_torch(recommended = 100, used = 10)

    def _boom():
        raise RuntimeError("driver reading unavailable")

    torch.mps.driver_allocated_memory = _boom
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    dd.install_decoder_sync(_pipe_with(decoder), _target("mps"))
    # The decode survives, and an unreadable gauge takes the safe side rather than skipping.
    assert decoder.decode(2) == ["out0", "out1"]
    assert torch.syncs == 2


def test_decoder_sync_survives_a_synchronize_that_raises(monkeypatch):
    # The no-budget fallback synchronises every call, so a torch whose mps surface is degraded
    # enough to hide recommended_max_memory would then raise on every decoder call. The bound is
    # an optimisation; losing the generation to it is not a trade worth making.
    torch = _mps_torch_without_recommended()

    def _boom():
        raise RuntimeError("Torch not compiled with MPS enabled")

    torch.mps.synchronize = _boom
    monkeypatch.setitem(sys.modules, "torch", torch)
    decoder = _FakeDecoder()
    assert dd.install_decoder_sync(_pipe_with(decoder), _target("mps")) is True
    assert decoder.decode(2) == ["out0", "out1"]


# ── GPU selection ─────────────────────────────────────────────────────


def _mask(
    monkeypatch,
    visible,
    *,
    physical_count = None,
):
    """Stub the hardware layer's parent-visible view, the mask `gpu_ids` is expressed against."""
    import utils.hardware.hardware as hw

    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": list(visible), "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        hw, "get_physical_gpu_count", lambda: physical_count or (max(visible) + 1 if visible else 0)
    )


def test_no_selection_leaves_the_target_on_the_default_device(monkeypatch):
    # The automatic pick must stay byte-for-byte what it was: no index, nothing pinned.
    calls: list = []
    torch = _make_torch(cuda_available = True, capability = (8, 0), set_device_calls = calls)
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.resolve_diffusion_device_target()
    assert t.ordinal is None
    assert t.torch_device == "cuda"
    dd.apply_diffusion_device_ordinal(t)
    assert calls == []


def test_a_single_card_pick_is_honoured_exactly(monkeypatch):
    calls: list = []
    torch = _make_torch(
        cuda_available = True, capability = (8, 0), device_count = 2, set_device_calls = calls
    )
    _install(monkeypatch, torch, studio_device = "cuda")
    _mask(monkeypatch, [0, 1])
    t = dd.resolve_diffusion_device_target(ordinal = dd.resolve_selected_cuda_ordinal([1]))
    assert t.ordinal == 1
    # The device string stays BARE: is_cuda / memory / speed / attention all compare it by value.
    assert t.device == "cuda"
    assert t.is_cuda_torch_device is True
    assert t.torch_device == "cuda:1"
    dd.apply_diffusion_device_ordinal(t)
    assert calls == [1]


def test_physical_ids_are_translated_through_the_visibility_mask(monkeypatch):
    # CUDA_VISIBLE_DEVICES=4,5: physical 4 and 5 are the valid picks and torch sees 0 and 1.
    # Validating against torch.cuda.device_count() would reject both.
    torch = _make_torch(cuda_available = True, capability = (8, 0), device_count = 2)
    _install(monkeypatch, torch, studio_device = "cuda")
    _mask(monkeypatch, [4, 5], physical_count = 8)
    assert dd.resolve_selected_cuda_ordinal([4]) == 0
    assert dd.resolve_selected_cuda_ordinal([5]) == 1
    with pytest.raises(ValueError):
        dd.resolve_selected_cuda_ordinal([0])

    # A REORDERED mask: physical 1 is torch ordinal 0, so the order matters, not just membership.
    _mask(monkeypatch, [1, 0], physical_count = 2)
    assert dd.resolve_selected_cuda_ordinal([1]) == 0
    assert dd.resolve_selected_cuda_ordinal([0]) == 1


def test_several_cards_resolve_to_the_one_with_the_most_free_vram(monkeypatch):
    # The mixed box this exists for: ordinal 0 is the SMALL card, so taking the first id lands on the GPU that cannot hold the checkpoint.
    torch = _make_torch(
        cuda_available = True,
        capability = (8, 0),
        device_count = 2,
        free_vram_by_index = {0: 6 * 1024**3, 1: 15 * 1024**3},
    )
    _install(monkeypatch, torch, studio_device = "cuda")
    _mask(monkeypatch, [0, 1])
    assert dd.resolve_selected_cuda_ordinal([0, 1]) == 1

    # Equal cards take the lowest ordinal, so the same selection always resolves the same way.
    torch.cuda.mem_get_info = lambda index = None: (8 * 1024**3, 0)
    assert dd.resolve_selected_cuda_ordinal([0, 1]) == 0


def test_free_vram_is_read_on_the_torch_ordinal_not_the_physical_id(monkeypatch):
    # Under a mask the two differ, and querying the physical id would rank the wrong cards.
    seen: list = []
    torch = _make_torch(cuda_available = True, capability = (8, 0), device_count = 2)
    torch.cuda.mem_get_info = lambda index = None: (seen.append(index), 1 << 30)[1:]
    _install(monkeypatch, torch, studio_device = "cuda")
    _mask(monkeypatch, [4, 5], physical_count = 8)
    dd.resolve_selected_cuda_ordinal([4, 5])
    assert seen == [0, 1]


def test_an_unreadable_card_sorts_last_rather_than_failing_the_load(monkeypatch):
    torch = _make_torch(
        cuda_available = True,
        capability = (8, 0),
        device_count = 3,
        free_vram_by_index = {2: 4 * 1024**3},
    )
    _install(monkeypatch, torch, studio_device = "cuda")
    _mask(monkeypatch, [0, 1, 2])
    assert dd.resolve_selected_cuda_ordinal([0, 2]) == 2
    # Nothing readable at all: a stable answer, not an exception.
    assert dd.resolve_selected_cuda_ordinal([0, 1]) == 0


def test_an_index_this_host_does_not_have_is_refused(monkeypatch):
    torch = _make_torch(cuda_available = True, capability = (8, 0), device_count = 2)
    _install(monkeypatch, torch, studio_device = "cuda")
    _mask(monkeypatch, [0, 1])
    with pytest.raises(ValueError):
        dd.resolve_selected_cuda_ordinal([5])
    with pytest.raises(ValueError):
        dd.resolve_selected_cuda_ordinal([-1])
    # Empty and None are "automatic", never a refusal.
    assert dd.resolve_selected_cuda_ordinal([]) is None
    assert dd.resolve_selected_cuda_ordinal(None) is None


def test_the_capability_probe_asks_about_the_selected_card(monkeypatch):
    # Ordinal 0 is pre-Ampere and ordinal 1 is not, so an index-less probe picks the wrong dtype.
    torch = _make_torch(cuda_available = True, device_count = 2)
    seen: list = []
    _NOTHING = object()

    def _cap(device = _NOTHING):
        seen.append(device)
        return (7, 5) if device in (_NOTHING, 0) else (8, 9)

    torch.cuda.get_device_capability = _cap
    _install(monkeypatch, torch, studio_device = "cuda")
    assert dd.resolve_diffusion_device_target(ordinal = 1).dtype == BF16
    assert seen == [1]

    # No selection probes with NO argument: a stub or older build that takes none would
    # otherwise raise into the fp16 fallback.
    seen.clear()
    assert dd.resolve_diffusion_device_target().dtype == FP16
    assert seen == [_NOTHING]


def test_an_indexed_override_string_keeps_its_card(monkeypatch):
    # _pick_device_and_dtype hands back the indexed string; rebuilding must not drop to ordinal 0.
    torch = _make_torch(cuda_available = True, capability = (8, 0))
    _install(monkeypatch, torch, studio_device = "cuda")
    t = dd.diffusion_device_target_from_torch_device("cuda:1", BF16)
    assert (t.device, t.ordinal, t.torch_device) == ("cuda", 1, "cuda:1")
    assert dd.diffusion_device_target_from_torch_device("cuda", BF16).ordinal is None


def test_a_selection_is_ignored_where_physical_indices_mean_nothing(monkeypatch):
    # MPS has one device and no applicator for an index; the pick must not become a refusal.
    torch = _make_torch(mps_available = True)
    _install(monkeypatch, torch, studio_device = "mlx")
    assert dd.resolve_diffusion_device_target(ordinal = 1).ordinal is None


def test_the_device_scope_restores_the_previous_card(monkeypatch):
    # Route preflights run on a pooled executor, so a pin left set there is inherited by the
    # NEXT request on that thread, including an automatic one.
    calls: list = []
    torch = _make_torch(cuda_available = True, device_count = 2, set_device_calls = calls)

    class _Scope:
        def __init__(self, index):
            calls.append(("enter", index))

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            calls.append(("exit", None))
            return False

    torch.cuda.device = _Scope
    monkeypatch.setitem(sys.modules, "torch", torch)
    with dd.diffusion_device_scope(1):
        pass
    assert calls == [("enter", 1), ("exit", None)]

    # No selection is a plain no-op, so the automatic path never touches the current device.
    calls.clear()
    with dd.diffusion_device_scope(None):
        pass
    assert calls == []


def test_the_rocm_bf16_probe_asks_about_the_selected_card(monkeypatch):
    # is_bf16_supported() takes no device argument, so asking about the selected card means
    # making it current; otherwise a bf16-capable pick behind an older default goes to fp32.
    torch = _make_torch(cuda_available = True, hip = "6.0", device_count = 2)
    scoped: list = []

    class _Scope:
        def __init__(self, index):
            scoped.append(index)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    torch.cuda.device = _Scope
    torch.cuda.is_bf16_supported = lambda: bool(scoped and scoped[-1] == 1)
    _install(monkeypatch, torch, studio_device = "cuda", is_rocm = True)
    assert dd.resolve_diffusion_device_target(ordinal = 1).dtype == BF16
    assert scoped == [1]


def test_the_device_scope_lets_the_body_exception_through(monkeypatch):
    # Catching around the yield made contextlib raise "generator didn't stop after throw()",
    # replacing a precision refusal with an error the route maps to the wrong status.
    torch = _make_torch(cuda_available = True, device_count = 2)

    class _Scope:
        def __init__(self, index):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    torch.cuda.device = _Scope
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match = "the real refusal"):
        with dd.diffusion_device_scope(1):
            raise RuntimeError("the real refusal")


def test_the_device_scope_still_runs_the_body_on_an_unusable_index(monkeypatch):
    # Entering may fail on a stale index; the probe then runs unpinned rather than not at all.
    torch = _make_torch(cuda_available = True, device_count = 2)

    def _boom(_index):
        raise RuntimeError("invalid device ordinal")

    torch.cuda.device = _boom
    monkeypatch.setitem(sys.modules, "torch", torch)
    ran = []
    with dd.diffusion_device_scope(9):
        ran.append(True)
    assert ran == [True]


def test_the_placed_ordinal_records_the_card_an_automatic_load_used(monkeypatch):
    # /images/generate runs on a pooled worker, so a pinned load leaves that thread on its card
    # for good and a later automatic load has no ordinal to re-pin with. The card it landed on is
    # recorded separately and puts the worker back.
    current = [3]
    torch = _make_torch(cuda_available = True, device_count = 4)
    torch.cuda.current_device = lambda: current[0]
    torch.cuda.set_device = lambda index: current.__setitem__(0, index)
    monkeypatch.setitem(sys.modules, "torch", torch)
    _install(monkeypatch, torch, studio_device = "cuda")

    automatic = dd.resolve_diffusion_device_target()
    assert automatic.ordinal is None  # the target itself stays un-indexed
    assert dd.placed_cuda_ordinal(automatic) == 3  # but the card is known

    selected = dd.resolve_diffusion_device_target(ordinal = 1)
    assert dd.placed_cuda_ordinal(selected) == 1  # a selection needs no observation

    # Nothing to record off CUDA: there is no thread-local device to put back.
    cpu_torch = _make_torch(cuda_available = False)
    monkeypatch.setitem(sys.modules, "torch", cpu_torch)
    _install(monkeypatch, cpu_torch, studio_device = "cpu")
    assert dd.placed_cuda_ordinal(dd.resolve_diffusion_device_target()) is None


def test_pinning_an_automatic_load_puts_a_shared_worker_back(monkeypatch):
    current = [0]
    torch = _make_torch(cuda_available = True, device_count = 4)
    torch.cuda.current_device = lambda: current[0]
    torch.cuda.set_device = lambda index: current.__setitem__(0, index)
    monkeypatch.setitem(sys.modules, "torch", torch)
    _install(monkeypatch, torch, studio_device = "cuda")

    # A pinned load runs here first and leaves the thread on its card.
    dd.apply_diffusion_device_ordinal(dd.resolve_diffusion_device_target(ordinal = 2))
    assert current == [2]
    # The next model loaded automatically; its weights are on 0, so the worker goes back to 0.
    dd.pin_cuda_ordinal(0)
    assert current == [0]
    # And a None never moves anything.
    dd.pin_cuda_ordinal(None)
    assert current == [0]


def test_a_multi_card_pick_declines_to_rank_when_ranking_is_barred(monkeypatch):
    # The plan routes must not open a CUDA context while a trainer holds the cards; validating
    # and translating the ids costs none, so a bad pick is still refused at the plan.
    torch = _make_torch(cuda_available = True, device_count = 4, free_vram_by_index = {0: 1, 1: 2})
    probed: list = []
    torch.cuda.mem_get_info = lambda index = None: (probed.append(index), (1, 2))[1]
    monkeypatch.setitem(sys.modules, "torch", torch)
    _install(monkeypatch, torch, studio_device = "cuda")
    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 4)

    assert dd.resolve_selected_cuda_ordinal([2], allow_ranking = False) == 2
    assert probed == []  # no free-VRAM probe, so no CUDA context
    assert dd.resolve_selected_cuda_ordinal([0, 1], allow_ranking = False) is None
    assert probed == []
    with pytest.raises(ValueError):
        dd.resolve_selected_cuda_ordinal([9], allow_ranking = False)
